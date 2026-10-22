"""Reconstruct offline observations for logged RTC replans; no robot APIs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
EP=ROOT.parent/'dataset/failed_episodes/episode_000003'
OUT=ROOT/'runtime/rtc_seam_probe_20261021'
OUT.mkdir(parents=True,exist_ok=True)
events=[json.loads(l) for l in (ROOT/'runtime/logs/inference_runs/20261020T231807-ab89cf09/events.jsonl').read_text(encoding='utf-8').splitlines()]
frames=[json.loads(l) for l in (EP/'raw/camera_frames.jsonl').read_text(encoding='utf-8').splitlines()]
prompt=events[0]['metadata']['config']['model']['prompt']
requests={r['context']['request_id']:r for r in events if r['event']=='rtc_request'}
payload={};manifest=[]
for rid in (3,6,8,10,12,13,20,21):
    r=requests[rid];c=r['context'];images={};info=[]
    for source,stamp in zip(('external','wrist'),r['observation_times']['image_perf_ns']):
        frame=min((f for f in frames if f['source']==source),key=lambda f:abs(f['capture_monotonic_ns']-stamp))
        # SharedCameraResource uses OpenCV INTER_AREA; store native RGB here
        # so the remote probe can use the identical resize implementation.
        images[source]=np.asarray(Image.open(EP/frame['image']).convert('RGB'))
        info.append(dict(source=source,file=frame['image'],image_time_error_ms=(frame['capture_monotonic_ns']-stamp)/1e6))
    prior=next(x for x in events if x['event']=='rtc_splice' and x['observation_perf_ns']==c['observation_perf_ns'])
    cut=prior['freeze_until_perf_ns']
    before_tick=c['freeze_until']-1
    previous=next(x for x in events if x['event']=='rtc_dispatch' and x['tick']==before_tick)
    payload.update({f'{rid}_state':np.asarray(r['observation_state'],np.float32),
        f'{rid}_external':images['external'],f'{rid}_wrist':images['wrist'],
        f'{rid}_prefix':np.asarray(c['absolute_prefix'],np.float64),
        f'{rid}_previous':np.asarray([*previous['target_tcp']['position_m'],*previous['target_tcp']['orientation_xyzw'],previous['gripper_ratio']]),
        f'{rid}_logged_actions':np.asarray(next(x for x in events if x['event']=='rtc_response' and x['context']['request_id']==rid)['actions']),
        f'{rid}_cut_row':np.asarray((cut-c['observation_perf_ns'])/50_000_000)})
    manifest.append(dict(request_id=rid,context=c,images=info))
payload['request_ids']=np.asarray([r['request_id'] for r in manifest]);payload['prompt']=np.asarray(prompt)
np.savez_compressed(OUT/'samples.npz',**payload)
(OUT/'samples_manifest.json').write_text(json.dumps(dict(samples=manifest,prompt=prompt,
    limitation='Exact logged state/prefix/timing; nearest saved native JPEG images, not original inference RGB; original noise not recorded.'),indent=2),encoding='utf-8')
print(OUT)
print([(r['request_id'],[round(x['image_time_error_ms'],3) for x in r['images']]) for r in manifest])

# Additional observation/prefix pairs for D=E=6. Their executed rows 6..11
# remain inside the known overlap (about 14 rows), rather than crossing its end.
# These use nearest recorded feedback/images and are counterfactual offline
# observation pairs, not a replay of a D6/E6 hardware run.
import math
robots=[json.loads(l) for l in (EP/'raw/robot_states.jsonl').read_text(encoding='utf-8').splitlines()]
timing={};timing_manifest=[]
responses={r['context']['request_id']:r for r in events if r['event']=='rtc_response'}
for rid in (6,8,10,12,13,20,21):
    old=responses[rid-1]; oldstamp=old['context']['observation_perf_ns']; ideal=oldstamp+6*50_000_000
    robot=min((r for r in robots if r['feedback_fresh_perf_counter_ns']<=ideal),key=lambda r:ideal-r['feedback_fresh_perf_counter_ns'])
    stamp=robot['feedback_fresh_perf_counter_ns'];offset=(stamp-oldstamp)/50_000_000
    oldrows=np.asarray(old['absolute_targets'],float)
    prefix=[]
    for i in range(20):
        z=offset+i; j=int(np.floor(z));alpha=z-j
        if j<0 or j+1>=20:break
        v=oldrows[j]*(1-alpha)+oldrows[j+1]*alpha
        v[3:7]/=np.linalg.norm(v[3:7]);prefix.append(v)
    q=np.asarray(robot['measured_tcp_pose']['orientation_xyzw'],float)
    if q[3]<0:q=-q
    n=np.linalg.norm(q[:3]);rv=q[:3]*(2*math.atan2(n,q[3])/n) if n>1e-12 else np.zeros(3)
    width=float(robot['gripper_opening_ratio'])*.085+.01
    state=np.r_[robot['measured_tcp_pose']['position_m'],rv,width/.095].astype(np.float32)
    for source in ('external','wrist'):
        frame=min((f for f in frames if f['source']==source),key=lambda f:abs(f['capture_monotonic_ns']-stamp))
        timing[f'{rid}_{source}']=np.asarray(Image.open(EP/frame['image']).convert('RGB'))
    prefix=np.asarray(prefix)
    timing.update({f'{rid}_state':state,f'{rid}_prefix':prefix,
        f'{rid}_previous':prefix[5],f'{rid}_cut_row':np.asarray(6.),
        f'{rid}_logged_actions':np.asarray(responses[rid]['actions'])})
    timing_manifest.append(dict(request_id=rid,old_request_id=rid-1,old_observation_perf_ns=oldstamp,
        reconstructed_observation_perf_ns=stamp,offset=offset,prefix_rows=len(prefix),feedback_earliness_ms=(ideal-stamp)/1e6))
timing['request_ids']=np.asarray([r['request_id'] for r in timing_manifest]);timing['prompt']=np.asarray(prompt)
np.savez_compressed(OUT/'timing6_samples.npz',**timing)
(OUT/'timing6_manifest.json').write_text(json.dumps(timing_manifest,indent=2),encoding='utf-8')
print('D6/E6 offline pairs',[(r['request_id'],r['prefix_rows'],round(r['feedback_earliness_ms'],2)) for r in timing_manifest])
