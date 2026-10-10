"""Switch only the approved 11649 backend on port8001; proxy8000 unchanged."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/root/autodl-tmp/nero_11649_rtc_20261006')
CHECKPOINT='/root/autodl-tmp/11649'
PYTHON='/root/openpi/.venv/bin/python'
SCRIPT=str(ROOT/'serve_nero_rtc.py')

def matches():
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        except (OSError,UnicodeError):continue
        original=('scripts/serve_policy.py' in args and CHECKPOINT in args and '8001' in args)
        rtc=(SCRIPT in args and CHECKPOINT in args and '8001' in args)
        if original or rtc:result.append((int(proc.name),args))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['status','start-rtc','start-image-proxy','restart-image-proxy','restore-original'])
    args=parser.parse_args();processes=matches()
    if args.mode=='status':print(json.dumps(processes));return
    if args.mode in ('start-image-proxy','restart-image-proxy','restore-original'):
        candidates=[]
        original='/root/autodl-tmp/nero_deploy/nero_compat_proxy.py'
        image=str(ROOT/'rtc_image_proxy.py')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:cmd=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            except (OSError,UnicodeError):continue
            if original in cmd or image in cmd:candidates.append((int(proc.name),cmd))
        if len(candidates)>1:raise RuntimeError('Ambiguous inference proxies')
        for pid,cmd in candidates:
            if args.mode=='start-image-proxy' and image in cmd:
                print(json.dumps({'image_proxy_already_running':pid}));return
            if args.mode=='restore-original' and original in cmd:continue
            os.kill(pid,signal.SIGTERM)
            deadline=time.monotonic()+5
            while (Path('/proc')/str(pid)).exists() and time.monotonic()<deadline:time.sleep(.1)
            if (Path('/proc')/str(pid)).exists():raise RuntimeError('Proxy did not stop; no duplicate spawn')
        if args.mode in ('start-image-proxy','restart-image-proxy'):
            with (ROOT/'image_proxy.log').open('ab') as log:
                child=subprocess.Popen([PYTHON,'-u',image],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(json.dumps({'image_proxy_pid':child.pid,'transport':'jpeg95_rgb_v1','lossless':False,'no_robot':True}));return
    if len(processes)>1:raise RuntimeError('Multiple matching inference backends; refusing ambiguous transition')
    for pid,cmd in processes:
        if args.mode=='start-rtc' and SCRIPT in cmd:
            print(json.dumps({'already_running':pid}));return
        (ROOT/'previous_backend.json').write_text(json.dumps({'pid':pid,'args':cmd},indent=2))
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+15
        while any(p==pid for p,_ in matches()) and time.monotonic()<deadline:time.sleep(.1)
        if any(p==pid for p,_ in matches()):raise RuntimeError('Backend did not stop; no forced kill or duplicate launch')
    if args.mode=='restore-original':
        subprocess.run([PYTHON,'/root/autodl-tmp/nero_11649_console_20261006/manage_server.py','start'],check=True)
        return
    env=dict(os.environ,XLA_PYTHON_CLIENT_MEM_FRACTION='0.85',PYTHONUNBUFFERED='1')
    cmd=[PYTHON,'-u',SCRIPT,'--checkpoint',CHECKPOINT,'--config','pi05_nero_chunk7_full','--port','8001','--audit-dir',str(ROOT/'audit')]
    with (ROOT/'rtc_server.log').open('ab') as log:
        child=subprocess.Popen(cmd,cwd='/root/openpi',env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':child.pid,'args':cmd,'proxy_unchanged':True,'no_robot':True}
    (ROOT/'transition.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))

if __name__=='__main__':main()
