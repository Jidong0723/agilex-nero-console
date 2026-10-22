"""Verify masked-prefix pruning and real MessagePack WebSocket dispatch offline."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from openpi_client import msgpack_numpy
from serve import RtcPolicy, Server


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();data=np.load(args.samples)
    import jax
    jax.config.update('jax_default_matmul_precision','highest')
    policy=RtcPolicy(args.checkpoint,compute_dtype='float32')
    report=dict(precision='float32 compute / bf16 parameters',equivalence=[])
    for rid in (12,13,20):
        obs={'observation/state':data[f'{rid}_state'],'prompt':str(data['prompt']),
             'observation/image':cv2.resize(data[f'{rid}_external'],(224,224),interpolation=cv2.INTER_AREA),
             'observation/wrist_image':cv2.resize(data[f'{rid}_wrist'],(224,224),interpolation=cv2.INTER_AREA)}
        context=dict(version='nero.rtc.v2',phase='replan',run_epoch=1,request_id=rid,observation_tick=8,plan_revision=1,
            freeze_until=15,observation_perf_ns=1_400_000_000,timeline_origin_perf_ns=1_000_000_000,freeze_until_perf_ns=1_750_000_000,
            action_row0_offset_s=0,period_s=.05,inference_delay=7,execute_horizon=8,prefix_attention_horizon=12,
            absolute_prefix=data[f'{rid}_prefix'].tolist())
        noise=np.random.default_rng(11649).normal(size=(20,32)).astype(np.float32)
        for steps,cap,schedule in ((10,0.,'exp'),(10,5.,'exp'),(40,40.,'ones')):
            policy.steps,policy.cap,policy.prefix_attention_schedule=steps,cap,schedule
            policy.compact_prefix=False;full=policy.infer(dict(obs,rtc_context=context),noise=noise)['actions']
            policy.compact_prefix=True;compact=policy.infer(dict(obs,rtc_context=context),noise=noise)['actions']
            position=float(np.max(np.linalg.norm(full[:,:3]-compact[:,:3],axis=1))*1000)
            maximum=float(np.max(np.abs(full-compact)))
            row=dict(request_id=rid,steps=steps,cap=cap,schedule=schedule,max_position_difference_mm=position,max_all_components_difference=maximum)
            report['equivalence'].append(row);print(row,flush=True)
            if position>.01 or maximum>1e-4:raise AssertionError('Masked-prefix compaction changed valid outputs beyond FP32 numerical tolerance')
        # Explicitly verify that ordinary inference does not use the compact flag.
        policy.compact_prefix=False;normal=policy.infer(obs,noise=noise)['actions']
        policy.compact_prefix=True;normal2=policy.infer(obs,noise=noise)['actions']
        if not np.array_equal(normal,normal2):raise AssertionError('Ordinary inference changed')
    policy.steps,policy.cap,policy.prefix_attention_schedule=40,40.,'ones'
    policy.metadata.update(num_steps=40,guidance_cap=40.,prefix_attention_schedule='ones',compact_prefix=True)
    policy.warmup()
    async def wire():
        owner=Server(policy,'127.0.0.1',0)
        packer=msgpack_numpy.Packer()
        results=[]
        async with serve(owner.handler,'127.0.0.1',0,compression=None,max_size=4*1024*1024) as server:
            port=server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}',compression=None,max_size=4*1024*1024) as socket:
                metadata=msgpack_numpy.unpackb(await socket.recv())
                for payload in (obs,dict(obs,rtc_context=context),obs):
                    start=time.perf_counter();await socket.send(packer.pack(payload));reply=msgpack_numpy.unpackb(await socket.recv())
                    actions=np.asarray(reply['actions']);assert actions.shape==(20,7) and np.isfinite(actions).all()
                    results.append(dict(rtc='rtc_context' in payload,shape=list(actions.shape),elapsed_ms=(time.perf_counter()-start)*1000,sampler=(reply.get('rtc') or {}).get('sampler')))
        return dict(metadata=metadata,requests=results)
    report['wire']=asyncio.run(wire());report['finished']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('FINISHED',flush=True)

if __name__=='__main__':main()
