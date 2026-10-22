"""Summarize fixed-noise official RTC ablations on saved episode observations."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runtime/rtc_seam_probe_20261021'
files=('bf16','fp32compute','fp32extended','fp32schedules','bf16schedules')
data=np.load(OUT/'samples.npz')
groups=defaultdict(list);reports={}
for name in files:
    report=json.loads((OUT/f'{name}.json').read_text(encoding='utf-8'));reports[name]=report
    for row in report['rows']:
        if row['request_id']==3:continue
        rid=row['request_id'];cut=float(data[f'{rid}_cut_row'])
        xyz=np.asarray(row['actions'])[:,:3]+data[f'{rid}_state'][:3]
        grid=cut+np.arange(8)
        emitted=np.asarray([xyz[int(t)]*(1-t%1)+xyz[int(t)+1]*(t%1) for t in grid])
        changes=np.linalg.norm(np.diff(np.vstack([data[f'{rid}_previous'][:3],emitted]),axis=0),axis=1)*1000
        row['executed_max_step_mm']=float(np.max(changes))
        schedule='exp' if row['mode']=='rtc' else row['mode']
        key=(report['compute'],schedule,row['steps'],row['cap'])
        groups[key].append(row)
summary=[]
for (compute,schedule,steps,cap),rows in groups.items():
    row=dict(compute=compute,schedule=schedule,steps=steps,cap=cap,n=len(rows))
    for field in ('hard_rmse_mm','same_time_seam_error_mm','actual_seam_jump_mm','executed_max_step_mm','elapsed_ms'):
        values=[r[field] for r in rows]
        row[field]=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(np.max(values)))
    summary.append(row)
(OUT/'comparison.json').write_text(json.dumps(dict(excluded_request=3,summary=summary,
    derivatives={name:report['derivatives'] for name,report in reports.items()},
    finished={name:report.get('finished',False) for name,report in reports.items()}),indent=2),encoding='utf-8')

selection=[('bfloat16','exp',10,5.),('float32','exp',10,5.),('float32','exp',100,100.),
           ('float32','ones',100,100.),('bfloat16','ones',20,20.),('bfloat16','ones',30,30.),('bfloat16','ones',50,50.)]
fig,axes=plt.subplots(2,1,figsize=(14,10))
colors=['#dc2626','#f97316','#a855f7','#0891b2','#64748b','#16a34a','#2563eb']
for key,color in zip(selection,colors):
    values=groups.get(key,[])
    sample=next((r for r in values if r['request_id']==13 and r['seed']==11649),None)
    if sample is None:continue
    label=f'{key[0]} / {key[1]} / steps={key[2]}, cap={int(key[3])}'
    axes[0].plot(np.arange(len(sample['row_error_mm'])),sample['row_error_mm'],'o-',color=color,label=label,lw=1.3,ms=3)
axes[0].axvline(6.5,color='#7c3aed',ls=':',label='Configured splice starts at row 7')
axes[0].set(xlabel='New chunk row (0 = observation time)',ylabel='Position error vs same-time old reference (mm)',
    title='Hardest recorded seam, request 13: same observation, reference and noise')
axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,ncol=2)
for key,rows in groups.items():
    if key[1] not in ('exp','ones'):continue
    lat=np.median([r['elapsed_ms'] for r in rows]);err=np.max([r['executed_max_step_mm'] for r in rows])
    color='#2563eb' if key[1]=='ones' else '#dc2626'
    axes[1].scatter(lat,err,color=color,marker='o' if key[0]=='bfloat16' else '^',s=35)
    if key[1]=='ones' or (key[2],key[3]) in ((10,5.),(100,100.)):
        axes[1].annotate(f'{key[0]} {key[1]} {key[2]}/{int(key[3])}',(lat,err),xytext=(4,4),textcoords='offset points',fontsize=7)
axes[1].axvline(350,color='gray',ls='--',label='Current D=7 budget (network/preparation also consume it)')
axes[1].set(xlabel='Median offline inference time on RTX 4090 (ms)',ylabel='Worst target step across entire executed window (mm)',
    title='Accuracy vs runtime; dots = BF16, triangles = FP32 compute; blue = ones, red = exp')
axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)

data=np.load(OUT/'samples.npz');rid=13;state=data[f'{rid}_state'];ref=data[f'{rid}_prefix'];prev=data[f'{rid}_previous']
fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
for k,ax in enumerate(axes):
    ax.plot(np.arange(len(ref)),ref[:,k],'k.--',lw=1.3,label='Old same-time reference Y')
    ax.scatter([float(data[f'{rid}_cut_row'])-1],[prev[k]],color='black',marker='x',s=55,label='Last old target actually sent')
    for key,color in zip(selection,colors):
        sample=next((r for r in groups.get(key,[]) if r['request_id']==rid and r['seed']==11649),None)
        if sample is None:continue
        absolute=np.asarray(sample['actions'])[:,:3]+state[:3]
        ax.plot(np.arange(20),absolute[:,k],color=color,lw=1.1,label=f'{key[0]} {key[1]} {key[2]}/{int(key[3])}')
    ax.axvline(float(data[f'{rid}_cut_row']),color='#7c3aed',ls=':');ax.set_ylabel('XYZ'[k]+' (m)');ax.set_xlim(0,19);ax.grid(alpha=.2)
axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('New chunk row (50 ms)')
fig.suptitle('Request 13: full chunk comparison, identical reconstructed RGB/state/reference/noise')
fig.tight_layout();fig.savefig(OUT/'request13_trajectories.png',dpi=170);plt.close(fig)
for s in summary:
    print(s['compute'],s['schedule'],s['steps'],s['cap'],s['n'],
          'seam median/max',round(s['same_time_seam_error_mm']['median'],3),round(s['same_time_seam_error_mm']['maximum'],3),
          'whole-window max',round(s['executed_max_step_mm']['maximum'],3),
          'ms median/p95',round(s['elapsed_ms']['median'],1),round(s['elapsed_ms']['p95'],1))
