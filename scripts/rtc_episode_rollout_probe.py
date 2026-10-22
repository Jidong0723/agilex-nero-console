"""Fixed-observation, continuous-plan RTC counterfactual. No robot interfaces."""
import argparse
import json
from pathlib import Path
import sys
import time
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from supervisor.openpi_client import OpenPIClient


def absolute(actions,state):
    actions=np.asarray(actions,float)
    if np.any(actions[:,6] < -.005) or np.any(actions[:,6] > 1.05):
        raise ValueError('Gripper prediction outside console boundary tolerance')
    return np.c_[actions[:,:3]+state[:3],
                 (Rotation.from_rotvec(actions[:,3:6])*Rotation.from_rotvec(state[3:6])).as_quat(),np.clip(actions[:,6],0,1)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episode',type=Path,default=ROOT.parent/'dataset/failed_episodes/episode_000010')
    ap.add_argument('--run',type=Path,default=ROOT/'runtime/logs/inference_runs/20261021T123838-09d7b497')
    ap.add_argument('--output',type=Path,default=ROOT/'runtime/rtc_episode10_optimization_20261021')
    ap.add_argument('--profiles',default='ones20,linear20,exp20')
    ap.add_argument('--steps',type=int,default=144)
    ap.add_argument('--seeds',default='0')
    ap.add_argument('--inference-delay',type=int,default=6)
    ap.add_argument('--execute-horizon',type=int,default=6)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    events=[json.loads(line) for line in (args.run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    requests=[event for event in events if event['event']=='rtc_request']
    prompt=events[0]['metadata']['config']['model']['prompt']
    episode=json.loads((args.episode/'episode.json').read_text(encoding='utf-8'))
    start=episode['collection_monotonic_ns']['start']
    end=episode['collection_monotonic_ns']['end']
    delay,horizon=args.inference_delay,args.execute_horizon
    if not 0 <= delay <= horizon < 20 or not 1 <= horizon or delay+horizon>20:
        raise ValueError('Invalid RTC delay/horizon')
    if args.steps < 2*horizon or args.steps % horizon or args.steps*.05 > (end-start)/1e9:
        raise ValueError('Replay steps must be a multiple of execute_horizon inside the recorded episode duration')
    frames=[json.loads(line) for line in (args.episode/'raw/camera_frames.jsonl').read_text(encoding='utf-8').splitlines()]
    observations=[];manifest=[]
    for step in range(0,args.steps,horizon):
        stamp=start+step*50_000_000
        req=min(requests,key=lambda r:abs(r['context']['observation_perf_ns']-stamp))
        obs={'observation/state':np.asarray(req['observation_state'],np.float32),'prompt':prompt}
        for source,key in [('external','observation/image'),('wrist','observation/wrist_image')]:
            frame=min((f for f in frames if f['source']==source),key=lambda f:abs(f['capture_monotonic_ns']-stamp))
            obs[key]=cv2.resize(cv2.cvtColor(cv2.imread(str(args.episode/frame['image'])),cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_AREA)
        observations.append(obs)
        manifest.append(dict(step=step,request_id=req['context']['request_id'],state_time_error_ms=(req['context']['observation_perf_ns']-stamp)/1e6))
    profiles={}
    for name in args.profiles.split(','):
        match=re.fullmatch(r'(ones|linear|exp)(\d+)(?:r(\d+))?',name)
        if not match:raise ValueError('Invalid profile: '+name)
        schedule,steps=match.group(1),int(match.group(2))
        profiles[name]=dict(num_steps=steps,prefix_attention_schedule=schedule,guidance_cap=1000.)
        if match.group(3):profiles[name]['terminal_refinement_steps']=int(match.group(3))
    client=OpenPIClient('127.0.0.1',18762,90)
    if not client.metadata.get('rtc_diagnostic_sampler_overrides'):
        raise RuntimeError('Cloud diagnostic overrides are not loaded')
    report=dict(metadata=client.metadata,hardware_commands=0,observation_manifest=manifest,
                timing=dict(inference_delay=delay,execute_horizon=horizon),rows=[])
    def infer(payload):
        nonlocal client
        for attempt in range(20):
            try:return client.infer(payload)
            except RuntimeError as exc:
                if 'busy' not in str(exc).lower() or attempt==19:raise
                client.close();time.sleep(.11+.013*attempt);client=OpenPIClient('127.0.0.1',18762,90)
    try:
        for seed in map(int,args.seeds.split(',')):
            initial=absolute(infer(dict(observations[0],rtc_diagnostic_noise_seed=seed))['actions'],observations[0]['observation/state'])
            for name,settings in profiles.items():
                plan=initial.copy();executed=[];chunks=[];seam_errors=[];latencies=[];tail_steps=[]
                for cycle,obs in enumerate(observations):
                    boundary=cycle*horizon
                    active=plan if cycle==0 else plan[horizon:]
                    # Same official old/new indexing for every profile: execute
                    # frozen D old rows and new rows D:E, then shift by E.
                    origin=1_000_000_000;stamp=origin+boundary*50_000_000
                    context=dict(version='nero.rtc.v2',phase='replan',run_epoch=1,request_id=cycle+1,
                        observation_tick=boundary,plan_revision=cycle,freeze_until=boundary+delay,
                        observation_perf_ns=stamp,timeline_origin_perf_ns=origin,
                        freeze_until_perf_ns=stamp+delay*50_000_000,action_row0_offset_s=0,period_s=.05,
                        inference_delay=delay,execute_horizon=horizon,prefix_attention_horizon=20-horizon,
                        absolute_prefix=active.tolist())
                    reply=infer(dict(obs,rtc_context=context,rtc_diagnostic_noise_seed=seed*1000+cycle,
                                     rtc_diagnostic_sampler=settings))
                    plan=absolute(reply['actions'],obs['observation/state'])
                    executed.extend(active[:delay,:3].tolist()+plan[delay:horizon,:3].tolist())
                    seam_errors.append(float(np.linalg.norm(plan[delay,:3]-active[delay,:3])*1000))
                    latencies.append(reply['policy_timing']['infer_ms'])
                    attention_end=20-horizon
                    tail_steps.append(float(np.linalg.norm(plan[attention_end,:3]-plan[attention_end-1,:3])*1000))
                    chunks.append(dict(boundary=boundary,old=active.tolist(),new=plan.tolist(),
                                       diagnostics=reply['rtc'],server_ms=latencies[-1]))
                xyz=np.asarray(executed);velocity=np.diff(xyz,axis=0)/.05
                acceleration=np.diff(velocity,axis=0)/.05
                jerk=np.diff(acceleration,axis=0)/.05
                step_mm=np.linalg.norm(np.diff(xyz,axis=0),axis=1)*1000
                metric=dict(target_step_p95_mm=float(np.percentile(step_mm,95)),target_step_max_mm=float(step_mm.max()),
                    acceleration_p95_m_s2=float(np.percentile(np.linalg.norm(acceleration,axis=1),95)),
                    jerk_p95_m_s3=float(np.percentile(np.linalg.norm(jerk,axis=1),95)),
                    seam_error_median_mm=float(np.median(seam_errors)),seam_error_max_mm=float(max(seam_errors)),
                    free_boundary_step_max_mm=float(max(tail_steps)),
                    compute_median_ms=float(np.median(latencies[1:])))
                report['rows'].append(dict(profile=name,seed=seed,metrics=metric,trajectory=xyz.tolist(),chunks=chunks))
                (args.output/'rollouts.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(json.dumps(dict(profile=name,seed=seed,**metric)),flush=True)
    finally:client.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True)
    for row in report['rows']:
        if row['seed']!=int(args.seeds.split(',')[0]):continue
        xyz=np.asarray(row['trajectory']);xyz-=xyz[0]
        for k in range(3):axes[k].plot(xyz[:,k]*1000,label=row['profile'])
        axes[3].plot(np.arange(1,len(xyz)),np.linalg.norm(np.diff(xyz,axis=0),axis=1)*1000,label=row['profile'])
    for k,axis in enumerate(axes):
        axis.set_ylabel('XYZ'[k]+' progress (mm)' if k<3 else 'Target step (mm)');axis.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('50 ms step; fixed recorded observations, not a closed-loop robot simulation')
    fig.tight_layout();fig.savefig(args.output/'rollouts.png',dpi=160)


if __name__=='__main__':main()
