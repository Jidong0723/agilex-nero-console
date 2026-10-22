"""No-motion SSH/WebSocket roundtrip to a dedicated offline diagnostic model."""
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import sys
import threading
import time
import numpy as np
import paramiko

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from supervisor.openpi_client import OpenPIClient


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=20);args=parser.parse_args()
    suffix='10' if args.steps==10 else ''
    output=ROOT/'runtime/rtc_seam_probe_20261021'
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('connect.nmb1.seetacloud.com',port=12112,username='root',
                   password=os.environ.pop('NERO_DIAG_SSH_PASSWORD'),timeout=15,
                   allow_agent=False,look_for_keys=False,compress=True)
    transport=client.get_transport();transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    remote='/root/autodl-tmp/nero_rtc_seam_probe_20261021'
    sftp=client.open_sftp()
    sftp.get(remote+'/wire_observations.npz',str(output/'wire_observations.npz'))
    sftp.get(remote+'/refined_production'+suffix+'.json',str(output/('refined_production'+suffix+'.json')))
    sftp.close()
    data=np.load(output/'wire_observations.npz')
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel=transport.open_channel('direct-tcpip',('127.0.0.1',8005),self.request.getpeername(),timeout=10)
            self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            try:
                while transport.is_active():
                    ready,_,_=select.select([self.request,channel],[],[],10)
                    for source in ready:
                        block=source.recv(65536)
                        if not block:return
                        (channel if source is self.request else self.request).sendall(block)
            finally:channel.close()
    class Forward(socketserver.ThreadingTCPServer):
        allow_reuse_address=True;daemon_threads=True
    report=dict(hardware_commands=0,rows=[])
    with Forward(('127.0.0.1',0),Handler) as forward:
        worker=threading.Thread(target=forward.serve_forever,daemon=True);worker.start()
        policy=OpenPIClient('127.0.0.1',forward.server_address[1],30)
        report['metadata']=policy.metadata
        assert policy.metadata['ready'] is True
        assert policy.metadata['rtc_integration_grid']=='terminal_refined_v1'
        for repeat in range(3):
            for rid in data['request_ids']:
                rid=int(rid)
                obs={'observation/state':data[f'{rid}_state'],'observation/image':data[f'{rid}_image'],
                     'observation/wrist_image':data[f'{rid}_wrist_image'],'prompt':str(data['prompt'])}
                context=json.loads(str(data[f'{rid}_context']));context['request_id']=repeat*100+rid
                start=time.perf_counter();reply=policy.infer(dict(obs,rtc_context=context));elapsed=(time.perf_counter()-start)*1000
                actions=np.asarray(reply['actions']);assert actions.shape==(20,7) and np.isfinite(actions).all()
                assert reply['rtc']['request_id']==context['request_id']
                report['rows'].append(dict(request_id=rid,repeat=repeat,elapsed_ms=elapsed,server_ms=reply['policy_timing']['infer_ms']))
        # Confirm ordinary requests remain available on the shared connection.
        ordinary=policy.infer(obs);assert np.asarray(ordinary['actions']).shape==(20,7)
        policy.close();forward.shutdown()
    client.close()
    report['latency_ms']={name:float(value) for name,value in zip(('median','p95','maximum'),np.percentile([r['elapsed_ms'] for r in report['rows']],[50,95,100]))}
    report['finished']=True
    (output/('wire_validation'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['latency_ms']))

if __name__=='__main__':main()
