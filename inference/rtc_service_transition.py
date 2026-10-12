"""Exact-package inference-process transition, never a robot/control operation."""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/nero_h10_9999_20261005')
RTC=ROOT/'rtc_20261006'


def package_processes():
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
            cwd=(path/'cwd').resolve()
            args=[x.decode() for x in args if x]
        except (OSError,UnicodeError):continue
        if cwd!=ROOT:continue
        original='serve_nero.py' in args and '--checkpoint' in args
        new=str(RTC/'serve_nero_rtc.py') in args and '--checkpoint' in args
        if original or new:matches.append((int(path.name),args))
    return matches


def stop_exact():
    matches=package_processes()
    if len(matches)>1:raise RuntimeError('multiple package inference processes; do not stop blindly')
    for pid,args in matches:
        print(json.dumps({'stopping_inference_pid':pid,'args':args}),flush=True)
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+15
        while any(p==pid for p,_ in package_processes()) and time.monotonic()<deadline:time.sleep(.1)
        if any(p==pid for p,_ in package_processes()):raise RuntimeError('inference did not exit after SIGTERM; no force kill')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['start-rtc','restore-original','status'])
    args=parser.parse_args()
    if args.mode=='status':print(json.dumps(package_processes()));return
    stop_exact()
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('0.0.0.0',8000))
    audit=RTC/'audit';audit.mkdir(exist_ok=True)
    script=RTC/'start_rtc_server.sh' if args.mode=='start-rtc' else ROOT/'start_server.sh'
    log=audit/('rtc_server.log' if args.mode=='start-rtc' else 'restored_original.log')
    with log.open('ab') as output:
        process=subprocess.Popen(['bash',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,
                                  stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    record={'mode':args.mode,'pid':process.pid,'log':str(log),'no_robot':True}
    (audit/'service_transition.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
