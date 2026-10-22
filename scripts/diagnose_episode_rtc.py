"""Offline RTC seam replay against the recorded production timeline."""
import json
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.inference_rtc import Timeline, Target, PERIOD_NS, sample_rows

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=Path, default=ROOT.parent/'dataset/failed_episodes/episode_000003')
parser.add_argument('--run', type=Path, default=ROOT/'runtime/logs/inference_runs/20261020T231807-ab89cf09')
args = parser.parse_args()
EP, RUN = args.episode, args.run
episode_start=json.loads((EP/'episode.json').read_text(encoding='utf-8'))['collection_monotonic_ns']['start']
OUT = EP / 'analysis/rtc_diagnosis'
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def stats(values):
    v = np.asarray(values, float)
    return dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v,95)), maximum=float(np.max(v))) if len(v) else {}

def angle(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.degrees(2*np.arccos(np.clip(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),0,1))))

events=list(load(RUN/'events.jsonl'))
responses={r['context']['request_id']:r for r in events if r['event']=='rtc_response'}
requests={r['context']['request_id']:r for r in events if r['event']=='rtc_request'}
origin=next(iter(responses.values()))['context']['timeline_origin_perf_ns']
timeline=Timeline(7,origin)
active_context=None
labels={}
seams=[]
prefix_errors=[]
dispatch_errors=[]
sampled=[]
for event in events:
    typ=event['event']
    if typ=='rtc_bootstrap_alignment':
        timeline.seed([Target(tuple(v[:3]),tuple(v[3:7]),v[7]) for v in event['scheduled_targets']],
                      event['response_ready_perf_ns'],Target(tuple(event['scheduled_targets'][0][:3]),tuple(event['scheduled_targets'][0][3:7]),event['scheduled_targets'][0][7]))
        labels={tick:1 for tick in timeline.targets}
    elif typ=='rtc_request' and event['context']['phase']=='replan':
        c=event['context']; known={**timeline.history,**timeline.targets}
        for i,row in enumerate(c['absolute_prefix']):
            tick,rem=divmod(c['observation_perf_ns']+i*PERIOD_NS-origin,PERIOD_NS)
            if tick in known and (not rem or tick+1 in known):
                from supervisor.inference_rtc import interpolate_target
                expected=known[tick] if not rem else interpolate_target(known[tick],known[tick+1],rem/PERIOD_NS)
                prefix_errors.append(np.linalg.norm(np.subtract(expected.position,row[:3]))*1000)
    elif typ=='rtc_response' and event['context']['phase']=='replan':
        active_context=event
    elif typ=='rtc_splice':
        response=active_context; c=response['context']; rid=c['request_id']
        targets=[Target(tuple(v[:3]),tuple(v[3:7]),v[7]) for v in response['absolute_targets']]
        old=dict(timeline.targets)
        cut=event.get('replacement_tick',max(c['freeze_until'],timeline.next_tick,timeline.committed_until))
        last=(c['observation_perf_ns']+19*PERIOD_NS-origin)//PERIOD_NS
        before=timeline.history.get(cut-1) or old.get(cut-1)
        new=sample_rows(targets,c['observation_perf_ns'],origin+cut*PERIOD_NS)
        reference=old.get(cut)
        if before and reference:
            wrong_shift=[]
            for shift in (-8,-1,0,1,8):
                tick=cut+shift
                if tick in old:
                    wrong_shift.append({'old_tick_offset':shift,'gap_mm':float(np.linalg.norm(np.subtract(new.position,old[tick].position))*1000)})
            seams.append(dict(request_id=rid,cut_tick=cut,step_episode=(origin+cut*PERIOD_NS-episode_start)/PERIOD_NS,
                source_row=(origin+cut*PERIOD_NS-c['observation_perf_ns'])/PERIOD_NS,
                jump_mm=float(np.linalg.norm(np.subtract(new.position,before.position))*1000),
                old_continuation_mm=float(np.linalg.norm(np.subtract(reference.position,before.position))*1000),
                same_time_error_mm=float(np.linalg.norm(np.subtract(new.position,reference.position))*1000),
                angle_jump_deg=angle(new.quaternion,before.quaternion),
                same_time_angle_error_deg=angle(new.quaternion,reference.quaternion),
                old_before=list(before.position),old_same_time=list(reference.position),new=list(new.position),
                reference_shift_control=wrong_shift,
                old_tail={str(t):list(v.position) for t,v in old.items()},new_rows=[list(v.position) for v in targets],
                observation_step=(c['observation_perf_ns']-origin)/PERIOD_NS,
                response_ms=response['elapsed_ms'],
                observation_to_response_ms=(response['perf_ns']-c['observation_perf_ns'])/1e6))
        timeline.targets={t:v for t,v in old.items() if timeline.next_tick<=t<cut}
        timeline.targets.update({t:sample_rows(targets,c['observation_perf_ns'],origin+t*PERIOD_NS) for t in range(cut,last+1)})
        labels={t:v for t,v in labels.items() if t<cut}
        labels.update({t:rid for t in range(cut,last+1)})
        timeline.revision+=1
    elif typ=='rtc_dispatch':
        tick=event['tick']; pos=np.asarray(event['target_tcp']['position_m'])
        expected=timeline.targets.get(tick)
        if expected:
            dispatch_errors.append(np.linalg.norm(pos-expected.position)*1000)
            timeline.next_tick=tick
            timeline.complete(tick)
        sampled.append(dict(tick=tick,pos=pos.tolist(),q=event['target_tcp']['orientation_xyzw'],rid=labels.get(tick),time=event['started_perf_s']))

steps=np.asarray([s['tick'] for s in sampled]); xyz=np.asarray([s['pos'] for s in sampled]); q=np.asarray([s['q'] for s in sampled]);
jumps=np.linalg.norm(np.diff(xyz,axis=0),axis=1)*1000
switch=np.asarray([a['rid']!=b['rid'] for a,b in zip(sampled,sampled[1:])])
normal_reports=[]
for run_id in ('20261019T214755-860c58d6','20261019T213602-d7252248'):
    normal=[r for r in load(ROOT/f'runtime/logs/inference_runs/{run_id}/events.jsonl') if r['event']=='tcp_request']
    if len(normal)<2: continue
    values=np.asarray([r['target_tcp']['position_m'] for r in normal]); d=np.linalg.norm(np.diff(values,axis=0),axis=1)*1000
    boundary=np.asarray([a['chunk_id']!=b['chunk_id'] for a,b in zip(normal,normal[1:])])
    timing=np.diff([r['perf_ns']/1e9 for r in normal])*1000
    # Movement-only subset avoids diluting statistics with long task dwells.
    moving=d>.5
    normal_reports.append(dict(run=run_id,targets=len(normal),all_mm=stats(d),seam_mm=stats(d[boundary]),inside_mm=stats(d[~boundary]),
        moving_mm=stats(d[moving]),interval_ms=stats(timing),above_10mm=int(sum(d>10)),above_30mm=int(sum(d>30))))
raw=list(load(EP/'raw/robot_states.jsonl'))
tracking=[np.linalg.norm(np.subtract(r['target_tcp_pose']['position_m'],r['measured_tcp_pose']['position_m']))*1000 for r in raw if r.get('target_tcp_pose') and r.get('measured_tcp_pose')]
report=dict(run=RUN.name,episode=str(EP),events=dict(Counter(r['event'] for r in events)),prefix_replay_error_mm=stats(prefix_errors),
    dispatch_replay_error_mm=stats(dispatch_errors),rtc_target_step_mm=stats(jumps),rtc_actual_seam_mm=stats(jumps[switch]),
    rtc_inside_chunk_mm=stats(jumps[~switch]),rtc_seam_same_time_error_mm=stats([s['same_time_error_mm'] for s in seams]),
    rtc_target_angle_step_deg=stats([angle(a,b) for a,b in zip(q,q[1:])]),
    rtc_dispatch_interval_ms=stats(np.diff([s['time'] for s in sampled])*1000),tracking_error_mm=stats(tracking),
    normal_historical_controls=normal_reports,top_seams=sorted(seams,key=lambda s:s['jump_mm'],reverse=True)[:10],
    fault=[r for r in events if r['event']=='rtc_fault'],fault_dispatch=[r for r in events if r['event']=='rtc_dispatch_timing'][-1])
report['seam_squared_step_fraction']=float(sum(jumps[switch]**2)/sum(jumps**2))
report['large_step_counts']={'over_20mm_total':int(sum(jumps>20)), 'over_20mm_at_seams':int(sum(jumps[switch]>20))}
replans=[r for r in responses.values() if r['context']['phase']=='replan']
row_errors=[np.linalg.norm(np.asarray(r['absolute_targets'])[:len(r['context']['absolute_prefix']),:3]-np.asarray(r['context']['absolute_prefix'])[:,:3],axis=1)*1000 for r in replans]
report['same_time_prefix_error_median_by_row_mm']=[float(np.median([e[i] for e in row_errors if len(e)>i])) for i in range(min(map(len,row_errors)))]
delay = replans[0]['context'].get('inference_delay',7)
report['rtc_hard_prefix_rmse_mm']=stats([np.sqrt(np.mean(e[:delay]**2)) for e in row_errors])
report['rtc_response_ms']=stats([r['elapsed_ms'] for r in replans])
report['production_fast_snapshot_has_limits']=False
report['control_limit_note']='osc_fast_state omits limits; RTC validate_targets only checks speed when osc.limits entries exist.'
(OUT/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
base=xyz[0]
fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
rawsteps=np.asarray([(r['sample_target_monotonic_ns']-origin)/PERIOD_NS for r in raw])
measured=np.asarray([r['measured_tcp_pose']['position_m'] for r in raw])
for k,ax in enumerate(axes):
    ax.plot(rawsteps,(measured-base)[:,k],color='#334155',lw=1,label='Measured TCP')
    ax.plot(steps,(xyz-base)[:,k],color='#d97706',lw=1.2,marker='.',ms=2,label='Actually dispatched RTC targets')
    for n,s in enumerate(seams):
        ax.axvline(s['cut_tick'],color='#7c3aed',ls=':',alpha=.5,lw=.8,label='Actual old/new splice' if n==0 else None)
        tail=s['old_tail']; ts=np.array([int(t) for t in tail]); vals=np.array(list(tail.values()))
        visible=ts>=s['cut_tick']-1
        ax.plot(ts[visible],(vals-base)[visible,k],color='gray',ls='--',alpha=.4,lw=.8,label='Discarded old continuation' if n==0 else None)
    ax.set_ylabel('XYZ'[k]+' progress (m)');ax.grid(alpha=.2)
episode_start=json.loads((EP/'episode.json').read_text(encoding='utf-8'))['collection_monotonic_ns']['start']
axes[-1].set_xlabel(f'RTC global step (50 ms); episode starts at step {(episode_start-origin)/PERIOD_NS:.1f}')
axes[0].legend(ncol=2,fontsize=9)
fig.suptitle(f'{EP.name}: actual commands and true RTC splices (production-log replay)')
fig.tight_layout();fig.savefig(OUT/'actual_splices.png',dpi=160);plt.close(fig)
worst=sorted(seams,key=lambda s:s['jump_mm'],reverse=True)[:3]
fig,axes=plt.subplots(3,3,figsize=(15,10))
for col,s in enumerate(worst):
    ts=np.asarray([int(t) for t in s['old_tail']]); vals=np.array(list(s['old_tail'].values())); nstep=s['observation_step']+np.arange(20); new=np.array(s['new_rows'])
    for k in range(3):
        ax=axes[k,col];ax.plot(ts,vals[:,k],'.--',color='gray',label='Old same-time plan')
        ax.plot(nstep,new[:,k],'.-',color='#dc2626',label='RTC-guided new chunk')
        valid=(steps>=s['cut_tick']-8)&(steps<=s['cut_tick']+8)
        ax.plot(steps[valid],xyz[valid,k],'.-',color='#2563eb',label='Actual dispatch')
        ax.axvline(s['cut_tick'],color='#7c3aed',ls=':');ax.set_xlim(s['cut_tick']-8,s['cut_tick']+8);ax.grid(alpha=.2)
        ax.set_ylabel('XYZ'[k]+' (m)');ax.set_xlabel('RTC step')
    axes[0,col].set_title(f"request {s['request_id']}, tick {s['cut_tick']}\nstep jump {s['jump_mm']:.1f} mm / same-time mismatch {s['same_time_error_mm']:.1f} mm")
axes[0,0].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'worst_seams.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(13,8))
axes[0].plot(steps[1:],jumps,'.-',color='#64748b',lw=.8,label='Every actual target step')
axes[0].scatter(steps[1:][switch],jumps[switch],color='#dc2626',s=28,label='Old/new chunk switch',zorder=3)
axes[0].set(xlabel='RTC step (50 ms)',ylabel='Position change (mm)',title='Actual target steps and chunk switches')
axes[0].legend();axes[0].grid(alpha=.2)
rr=np.arange(len(report['same_time_prefix_error_median_by_row_mm']))
sampler=(replans[0].get('rtc_diagnostics') or {}).get('sampler_config') or {}
schedule=sampler.get('prefix_attention_schedule','exp')
end=replans[0]['context'].get('prefix_attention_horizon',len(rr))
weights=np.clip((delay-1-rr)/(end-delay+1)+1,0,1)
if schedule=='ones': weights=np.ones_like(weights)
elif schedule=='exp': weights=weights*np.expm1(weights)/(np.e-1)
elif schedule=='zeros': weights=(rr<delay).astype(float)
axes[1].plot(rr,report['same_time_prefix_error_median_by_row_mm'],'o-',color='#dc2626',label=f'Median new vs reference Y ({len(replans)} replans)')
if schedule!='ones':axes[1].axvspan(delay-.5,len(rr)-.5,color='gold',alpha=.18,label='Soft guidance')
axes[1].axvline(delay-.5,color='#7c3aed',ls=':',label=f'Configured frozen rows 0..{delay-1}')
axes[1].set(xlabel='New chunk row (0 = observation time)',ylabel='Same-time position error (mm)')
secondary=axes[1].twinx();secondary.plot(rr,weights,'s--',color='#2563eb',label=f'{schedule} mask weight');secondary.set_ylabel('Mask weight');secondary.set_ylim(-.05,1.08)
handles,labels_=axes[1].get_legend_handles_labels();h2,l2=secondary.get_legend_handles_labels();axes[1].legend(handles+h2,labels_+l2,fontsize=8,loc='upper left')
axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'seam_and_mask.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in report.items() if k!='top_seams'},ensure_ascii=False,indent=2))
print('TOP SEAMS',[(s['request_id'],s['cut_tick'],round(s['jump_mm'],2),round(s['same_time_error_mm'],2),round(s['old_continuation_mm'],2),s['source_row']) for s in worst])
