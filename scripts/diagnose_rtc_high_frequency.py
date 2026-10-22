"""Recorded target/feedback roughness, with no robot or inference connection."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();out=args.episode/'analysis/rtc_diagnosis';out.mkdir(parents=True,exist_ok=True)
    events=[json.loads(l) for l in (args.run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    raw=[json.loads(l) for l in (args.episode/'raw/robot_states.jsonl').read_text(encoding='utf-8').splitlines()]
    dispatch=[e for e in events if e['event']=='rtc_dispatch']
    ticks=np.asarray([e['tick'] for e in dispatch]);targets=np.asarray([e['target_tcp']['position_m'] for e in dispatch])
    origin=next(e['context']['timeline_origin_perf_ns'] for e in events if e['event']=='rtc_request')
    t=ticks*.05;measured_time=np.asarray([(r['feedback_monotonic_ns']-origin)/1e9 for r in raw])
    positions=np.asarray([r['measured_tcp_pose']['position_m'] for r in raw])
    valid=(t>=measured_time[0]+.5)&(t<=measured_time[-1]-.5)
    t=t[valid];targets=targets[valid];ticks=ticks[valid]
    measured=np.stack([np.interp(t,measured_time,positions[:,k]) for k in range(3)],axis=1)
    highpass=signal.butter(3,2,fs=20,btype='highpass',output='sos')
    hp_target=signal.sosfiltfilt(highpass,targets,axis=0)*1000
    hp_measured=signal.sosfiltfilt(highpass,measured,axis=0)*1000
    freq,pt=signal.welch(targets,fs=20,nperseg=min(128,len(t)),axis=0)
    _,pm=signal.welch(measured,fs=20,nperseg=min(128,len(t)),axis=0)
    residual=hp_measured-hp_target
    acceleration=np.diff(targets,n=2,axis=0)/.05**2
    splices=[e['replacement_tick'] for e in events if e['event']=='rtc_splice' and 'replacement_tick' in e]
    near_splice=np.asarray([any(abs(tick-s)<=1 for s in splices) for tick in ticks[2:]])
    energy=np.sum(acceleration**2,axis=1)
    def rms(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
    report=dict(highpass_cutoff_hz=2,band_rms_mm=dict(target=rms(hp_target),measured=rms(hp_measured),
        unshifted_tracking_residual=rms(residual)),axis_highpass_correlation=[float(np.corrcoef(hp_target[:,k],hp_measured[:,k])[0,1]) for k in range(3)],
        spectrum_peak_above_2hz=float(freq[freq>=2][np.argmax(pt[freq>=2].sum(axis=1))]),
        target_acceleration_energy_near_splices=float(energy[near_splice].sum()/energy.sum()),
        near_splice_sample_fraction=float(near_splice.mean()),
        raw_feedback_interval_ms=dict(median=float(np.median(np.diff(measured_time))*1000),max=float(np.max(np.diff(measured_time))*1000)),
        position_20hz_limit='Common-grid comparison only: 20 Hz target grid, no conclusion above 10 Hz; raw feedback is interpolated. Lag estimate is descriptive, not a controller delay measurement.',
        gripper_commands=sum(e['event']=='gripper_request' for e in events))
    # Estimate lag on the same band before interpreting negative zero-lag
    # correlation as an independent hardware oscillation.
    lags=np.arange(0,.201,.005);correlations=[]
    for lag in lags:
        shifted=np.stack([np.interp(t+lag,measured_time,positions[:,k]) for k in range(3)],axis=1)
        hp=signal.sosfiltfilt(highpass,shifted,axis=0)*1000
        correlations.append(float(np.sum(hp_target*hp)/np.sqrt(np.sum(hp_target**2)*np.sum(hp**2))))
    best=int(np.argmax(correlations));report['estimated_highpass_tracking_lag_ms']=float(lags[best]*1000)
    report['lag_compensated_highpass_correlation']=correlations[best]
    shifted=np.stack([np.interp(t+lags[best],measured_time,positions[:,k]) for k in range(3)],axis=1)
    hp_shifted=signal.sosfiltfilt(highpass,shifted,axis=0)*1000
    episode=json.loads((args.episode/'episode.json').read_text(encoding='utf-8'))
    begin,end=[episode['collection_monotonic_ns'][key] for key in ('start','end')]
    selected=[e for e in dispatch if begin<=e['perf_ns']<=end]
    pos=np.asarray([e['target_tcp']['position_m'] for e in selected])
    step=np.linalg.norm(np.diff(pos,axis=0),axis=1)*1000
    report['episode_target_steps']=dict(n=len(selected),median_mm=float(np.median(step)),p95_mm=float(np.percentile(step,95)),max_mm=float(step.max()))
    report['episode_tracking_error_mm']={}
    error=np.asarray([np.linalg.norm(np.subtract(r['target_tcp_pose']['position_m'],r['measured_tcp_pose']['position_m']))*1000 for r in raw if r.get('target_tcp_pose') and r.get('measured_tcp_pose')])
    report['episode_tracking_error_mm']=dict(median=float(np.median(error)),p95=float(np.percentile(error,95)),max=float(error.max()))
    report['episode_duration_s']=(end-begin)/1e9
    (out/'high_frequency.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(4,1,figsize=(13,11))
    for k in range(3):
        axes[k].plot(t,hp_target[:,k],label='Target >2 Hz',color='#d97706')
        axes[k].plot(t,hp_measured[:,k],label='Measured >2 Hz (unshifted)',color='#94a3b8',alpha=.4)
        axes[k].plot(t,hp_shifted[:,k],label=f'Measured, lag-compensated {lags[best]*1000:.0f} ms',color='#334155',alpha=.8)
        for s in splices:axes[k].axvline(s*.05,ls=':',lw=.7,color='#7c3aed',alpha=.4)
        axes[k].set_ylabel('XYZ'[k]+' residual (mm)');axes[k].set_xlim(t[0],t[-1]);axes[k].grid(alpha=.2)
    axes[0].legend(ncol=2);axes[2].set_xlabel('Seconds from original observation epoch; purple = actual replacement')
    axes[3].semilogy(freq,pt.sum(axis=1)*1e6,label='Target PSD')
    axes[3].semilogy(freq,pm.sum(axis=1)*1e6,label='Measured PSD')
    axes[3].axvline(1/.3,ls=':',color='#7c3aed',label='D=E=6 refresh: 3.33 Hz')
    axes[3].set(xlabel='Frequency (Hz)',ylabel='Position PSD (mm²/Hz)',xlim=(0,10));axes[3].legend();axes[3].grid(alpha=.2)
    fig.suptitle(args.episode.name+': target already contains small repeated corrections')
    fig.tight_layout();fig.savefig(out/'high_frequency.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
