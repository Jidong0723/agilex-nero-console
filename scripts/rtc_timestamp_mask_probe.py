"""Paired real-timestamp mask ablation; no robot imports or commands."""
import json
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supervisor.openpi_client import OpenPIClient
from supervisor.inference_rtc import Target, sample_rows, PERIOD_NS, orientation_distance


def main():
    out=ROOT/'runtime/rtc_episode16_20261021';data=np.load(out/'samples/observations.npz')
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-sweep',action='store_true')
    parser.add_argument('--smaller-delay',type=int);args=parser.parse_args()
    profiles={name:{'prefix_mask_mode':name} for name in ['legacy_compressed','nominal','timestamp_frozen']}
    if args.terminal_sweep:
        profiles={'timestamp_frozen':{'prefix_mask_mode':'timestamp_frozen'},
                  'terminal_1e4':{'prefix_mask_mode':'timestamp_frozen','terminal_noise_time':.0001,'guidance_cap':10000.},
                  'terminal_1e5':{'prefix_mask_mode':'timestamp_frozen','terminal_noise_time':.00001,'guidance_cap':100000.}}
    filename='terminal_sweep.json' if args.terminal_sweep else 'paired_masks.json'
    if args.smaller_delay:
        profiles={'timestamp_frozen':{'prefix_mask_mode':'timestamp_frozen'}}
        filename=f'timing_{args.smaller_delay}.json'
    client=OpenPIClient('127.0.0.1',18762,90)
    report=dict(metadata=client.metadata,hardware_commands=0,rows=[],
                limitation='Recorded observation/state/prefix/timestamps with nearest saved JPEG and fixed noise; no closed-loop robot simulation')
    events=[json.loads(l) for l in (ROOT/'runtime/logs/inference_runs/20261021T151523-814e408a/events.jsonl').read_text(encoding='utf-8').splitlines()]
    splices={e['observation_perf_ns']:e for e in events if e['event']=='rtc_splice'}
    dispatch={e['tick']:e for e in events if e['event']=='rtc_dispatch'}
    def infer(payload):
        nonlocal client
        for attempt in range(20):
            try:return client.infer(payload)
            except RuntimeError as exc:
                if 'busy' not in str(exc).lower() or attempt==19:raise
                client.close();time.sleep(.11+.013*attempt);client=OpenPIClient('127.0.0.1',18762,90)
    try:
        for rid in (6,9,11,14,32,34):
            state=data[f'{rid}_state'];context=json.loads(str(data[f'{rid}_context']))
            if args.smaller_delay:
                delay=args.smaller_delay
                context.update(inference_delay=delay,execute_horizon=delay,prefix_attention_horizon=20-delay)
                context['freeze_until']=(context['observation_perf_ns']+delay*PERIOD_NS-context['timeline_origin_perf_ns']+PERIOD_NS-1)//PERIOD_NS
                context['freeze_until_perf_ns']=context['timeline_origin_perf_ns']+context['freeze_until']*PERIOD_NS
            obs={'observation/state':state,'observation/image':data[f'{rid}_image'],
                 'observation/wrist_image':data[f'{rid}_wrist_image'],'prompt':str(data['prompt'])}
            cut=splices[context['observation_perf_ns']]['replacement_tick']
            if args.smaller_delay:cut=context['freeze_until']
            row=(context['timeline_origin_perf_ns']+cut*PERIOD_NS-context['observation_perf_ns'])/PERIOD_NS
            reference=[Target(tuple(r[:3]),tuple(r[3:7]),r[7]) for r in context['absolute_prefix']]
            old=sample_rows(reference,context['observation_perf_ns'],context['timeline_origin_perf_ns']+cut*PERIOD_NS)
            for seed in (0,1,2):
                for mode,settings in profiles.items():
                    try:
                        reply=infer(dict(obs,rtc_context=context,rtc_diagnostic_noise_seed=seed,
                                         rtc_diagnostic_sampler=settings))
                    except RuntimeError as exc:
                        report['rows'].append(dict(request_id=rid,seed=seed,mode=mode,error=str(exc),continuity_allowed=False))
                        (out/filename).write_text(json.dumps(report,indent=2),encoding='utf-8')
                        continue
                    actions=np.asarray(reply['actions']);quats=(Rotation.from_rotvec(actions[:,3:6])*Rotation.from_rotvec(state[3:6])).as_quat()
                    rows=[Target(tuple(p),tuple(q),float(g)) for p,q,g in zip(actions[:,:3]+state[:3],quats,actions[:,6])]
                    sampled=[sample_rows(rows,context['observation_perf_ns'],context['timeline_origin_perf_ns']+tick*PERIOD_NS) for tick in range(cut,cut+8)]
                    gap=float(np.linalg.norm(np.subtract(sampled[0].position,old.position))*1000)
                    angle=float(np.degrees(orientation_distance(sampled[0].quaternion,old.quaternion)))
                    before=dispatch[cut-1]['target_tcp']['position_m']
                    step=np.linalg.norm(np.diff(np.asarray([before,*[t.position for t in sampled]]),axis=0),axis=1)*1000
                    diagnostic=reply['rtc']
                    item=dict(request_id=rid,seed=seed,mode=mode,source_row=row,reference_length=len(reference),
                        same_time_position_mm=gap,same_time_orientation_deg=angle,
                        continuity_allowed=gap<=2 and angle<=.5,
                        first_dispatch_step_mm=float(step[0]),max_8step_mm=float(step.max()),
                        server_timing=reply['policy_timing'],config=diagnostic['sampler_config'],
                        diagnostics=diagnostic,predicted_positions=(actions[:,:3]+state[:3]).tolist(),
                        reference=[list(t.position) for t in reference])
                    report['rows'].append(item)
                    (out/filename).write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('completed request',rid,flush=True)
    finally:client.close()
    for mode in profiles:
        selected=[r for r in report['rows'] if r['mode']==mode]
        gap=[r['same_time_position_mm'] for r in selected if 'same_time_position_mm' in r]
        if not gap:print(mode,'all responses rejected',flush=True);continue
        print(mode,json.dumps(dict(median_mm=float(np.median(gap)),max_mm=float(max(gap)),
            continuity_pass=sum(r['continuity_allowed'] for r in selected),count=len(selected))),flush=True)


if __name__=='__main__':main()
