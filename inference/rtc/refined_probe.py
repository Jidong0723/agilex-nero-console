"""Validate the opt-in production sampler against real counterfactual episode pairs."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from serve import RtcPolicy, Server
from codec import encode_prefix
from seam_probe import measure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--port',type=int,default=8005);ap.add_argument('--steps',type=int,default=20)
    args=ap.parse_args();data=np.load(args.samples)
    policy=RtcPolicy(args.checkpoint,steps=args.steps,cap=1000.,normal_steps=10,compact_prefix=True,
                     prefix_attention_schedule='ones',terminal_noise_time=.001)
    policy.warmup()
    report=dict(metadata=policy.metadata,rows=[]);wire={}
    for rid in data['request_ids']:
        rid=int(rid);state=data[f'{rid}_state'];reference=data[f'{rid}_prefix']
        obs={'observation/state':state,'prompt':str(data['prompt']),
             'observation/image':cv2.resize(data[f'{rid}_external'],(224,224),interpolation=cv2.INTER_AREA),
             'observation/wrist_image':cv2.resize(data[f'{rid}_wrist'],(224,224),interpolation=cv2.INTER_AREA)}
        context=dict(version='nero.rtc.v2',phase='replan',run_epoch=1,request_id=rid,observation_tick=8,plan_revision=1,
                     freeze_until=14,observation_perf_ns=1_400_000_000,timeline_origin_perf_ns=1_000_000_000,
                     freeze_until_perf_ns=1_700_000_000,action_row0_offset_s=0,period_s=.05,inference_delay=6,
                     execute_horizon=6,prefix_attention_horizon=14,absolute_prefix=reference.tolist())
        wire.update({f'{rid}_state':state,f'{rid}_image':obs['observation/image'],f'{rid}_wrist_image':obs['observation/wrist_image'],
                     f'{rid}_context':np.asarray(json.dumps(context))})
        for seed in (11649,11650,11651):
            noise=np.random.default_rng(seed).normal(size=(20,32)).astype(np.float32)
            start=time.perf_counter();result=policy.infer(dict(obs,rtc_context=context),noise=noise)
            report['rows'].append(dict(request_id=rid,seed=seed,elapsed_ms=(time.perf_counter()-start)*1000,
                **measure(result['actions'],encode_prefix(reference,state),state,data[f'{rid}_previous'],6.,6,6)))
        print('DONE',rid,flush=True)
    wire['request_ids']=data['request_ids'];wire['prompt']=data['prompt']
    np.savez_compressed(args.output.with_name('wire_observations.npz'),**wire)
    report['finished']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('WIRE_READY',args.port,flush=True)
    # Dedicated diagnostic endpoint expires automatically. It has no robot APIs.
    import asyncio
    async def run():
        try:await asyncio.wait_for(Server(policy,'127.0.0.1',args.port).run(),timeout=180.)
        except asyncio.TimeoutError:pass
    asyncio.run(run())

if __name__=='__main__':main()
