"""Read-only real feedback monitor with concurrent UI status load; no POSTs."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
import urllib.request

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seconds',type=float,default=60.)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    base='http://127.0.0.1:8765'
    def read(path):
        with urllib.request.urlopen(base+path,timeout=5) as response:
            return json.load(response)['data']
    stop=threading.Event(); load={'requests':0,'errors':0}
    def ui_load():
        while not stop.is_set():
            try:
                read('/api/status');load['requests']+=1
            except Exception:load['errors']+=1
            stop.wait(.1)
    state=read('/api/pi05/state')
    if state.get('execution_enabled') or state.get('priming'):
        raise RuntimeError('Read-only verification requires inactive model control')
    rows=[]; errors=[]; started=time.perf_counter()
    thread=threading.Thread(target=ui_load,daemon=True);thread.start()
    try:
        while time.perf_counter()-started < args.seconds:
            try:
                state=read('/api/pi05/state')
                if state.get('execution_enabled') or state.get('priming'):
                    raise RuntimeError('Control started during read-only validation')
                now=time.perf_counter_ns();feedback=state.get('measured_tcp_feedback') or {}
                stamp=feedback.get('sample_perf_ns')
                rows.append(dict(time_s=(now/1e9-started),sample_perf_ns=stamp,
                    sample_age_at_receive_ms=(now-stamp)/1e6 if isinstance(stamp,int) else None,
                    feedback=feedback,preview=(state.get('preview') or {}).get('state')))
            except Exception as exc:
                errors.append(str(exc))
                if 'Control started' in str(exc):raise
            stop.wait(.025)
    finally:
        stop.set();thread.join(6.)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        def stats(values):
            return dict(zip(('median','p95','maximum'),map(float,np.percentile(values,[50,95,100])))) if values else None
        ages=[r['sample_age_at_receive_ms'] for r in rows if r['sample_age_at_receive_ms'] is not None]
        rpc=[r['feedback']['sensor_ipc']['roundtrip_ms'] for r in rows if r['feedback'].get('sensor_ipc')]
        report=dict(duration_s=time.perf_counter()-started,hardware_targets_sent=0,samples=len(rows),
            sample_age_ms=stats(ages),sensor_rpc_ms=stats(rpc),
            over_100ms=sum(age>100 for age in ages),missing=sum(r['sample_perf_ns'] is None for r in rows),
            errors=errors,concurrent_ui_load=load,rows=rows)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
