"""Full execution-window acceptance summary; keep sampler experiments explicit."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runtime/rtc_seam_probe_20261021'
sys.path.insert(0,str(ROOT))
data=np.load(OUT/'timing6_samples.npz')
uniform=json.loads((OUT/'timing6compact.json').read_text())
refined=json.loads((OUT/'refined_production10.json').read_text())
refined20=json.loads((OUT/'refined_production.json').read_text())
series=[('Uniform exp / 10 steps / cap5',[r for r in uniform['rows'] if r['mode']=='exp' and r['steps']==10]),
        ('Uniform ones / 40 steps / cap40',[r for r in uniform['rows'] if r['mode']=='ones' and r['steps']==40]),
        ('Refined ones / 10 steps / cap1000',refined['rows']),
        ('Refined ones / 20 steps / cap1000',refined20['rows'])]
summary=[]
for name,rows in series:
    entry=dict(name=name,n=len(rows))
    for field in ('hard_rmse_mm','same_time_seam_error_mm','executed_max_step_mm'):
        values=[r[field] for r in rows];entry[field]=dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))
    steady=[r['elapsed_ms'] for r in rows if r['elapsed_ms']<1000]
    entry['steady_infer_ms']=dict(median=float(np.median(steady)),p95=float(np.percentile(steady,95)),maximum=float(max(steady)))
    entry['cold_compile_records']=sum(r['elapsed_ms']>=1000 for r in rows)
    # Measure orientation discontinuities across the entire executable window.
    from supervisor.inference_rtc import Target,interpolate_target,orientation_distance
    from supervisor.pi05_adapter import _rotvec_quaternion,_quat_product
    angles=[];grippers=[]
    for row in rows:
        rid=row['request_id'];state=data[f'{rid}_state'];actions=np.asarray(row['actions'])
        previous=data[f'{rid}_previous'];base=_rotvec_quaternion(state[3:6].tolist())
        targets=[Target(tuple(state[:3]+a[:3]),tuple(_quat_product(_rotvec_quaternion(a[3:6].tolist()),base)),float(a[6])) for a in actions]
        emitted=[Target(tuple(previous[:3]),tuple(previous[3:7]),float(previous[7])),*targets[6:12]]
        angles.append(max(np.degrees(orientation_distance(a.quaternion,b.quaternion)) for a,b in zip(emitted,emitted[1:])))
        grippers.extend(actions[:,6].tolist())
    entry['maximum_executed_orientation_step_deg']=float(max(angles));entry['predicted_gripper_range']=[float(min(grippers)),float(max(grippers))]
    summary.append(entry)
wire=json.loads((OUT/'wire_validation10.json').read_text())
(OUT/'solution_summary.json').write_text(json.dumps(dict(summary=summary,wire_latency_ms=wire['latency_ms'],
    qualification='Offline observation pairs reconstructed from a failed episode; no physical closed-loop validation.'),indent=2),encoding='utf-8')
colors=['#dc2626','#a855f7','#2563eb','#0891b2'];rid=13;ref=data[f'{rid}_prefix'];state=data[f'{rid}_state']
fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
for k,ax in enumerate(axes):
    ax.axvspan(5.5,11.5,color='#dbeafe',alpha=.55,label='Executed rows 6..11')
    ax.plot(np.arange(len(ref)),ref[:,k],'k.--',lw=1.1,label='Old same-time reference Y')
    for (name,rows),color in zip(series,colors):
        row=next(r for r in rows if r['request_id']==rid and r['seed']==11649)
        xyz=np.asarray(row['actions'])[:,:3]+state[:3]
        ax.plot(np.arange(20),xyz[:,k],ls='--',color=color,alpha=.6,lw=.9)
        executed=np.vstack([data[f'{rid}_previous'][:3],xyz[6:12]])
        ax.plot(np.arange(5,12),executed[:,k],color=color,lw=1.7,marker='.',ms=3,label=name)
    ax.axvline(13.5,color='gray',ls=':',label='Guidance overlap ends after row13')
    ax.set_ylabel('XYZ'[k]+' (m)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('New chunk step (50 ms)')
fig.suptitle('D=E=6: check the entire executed window, not only its first seam\nSolid = executed region (plus previous row); dashed = nonexecuted predictions')
fig.tight_layout();fig.savefig(OUT/'solution_trajectories.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(14,5))
labels=['Uniform\nexp 10/5','Uniform\nones 40/40','Refined\nones 10/1000','Refined\nones 20/1000']
axes[0].bar(np.arange(4),[s['same_time_seam_error_mm']['maximum'] for s in summary],color=colors)
axes[0].set_xticks(np.arange(4),labels);axes[0].set_ylabel('Worst same-time seam error (mm)')
axes[1].bar(np.arange(4),[s['executed_max_step_mm']['maximum'] for s in summary],color=colors)
axes[1].set_xticks(np.arange(4),labels);axes[1].set_ylabel('Worst target step in full execution window (mm)')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('7 reconstructed observation/prefix pairs x 3 fixed noise seeds; no robot motion')
fig.tight_layout();fig.savefig(OUT/'solution_metrics.png',dpi=170);plt.close(fig)
print(json.dumps(summary,indent=2))
