"""Fixed-noise, real-run RTC comparison. Does not connect to robot or server."""
import argparse
import hashlib
import json
import time
import types
from pathlib import Path

import cv2
import einops
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils

from codec import encode_prefix, pack_normalized_prefix
from sampling import prefix_weights
from serve import RtcPolicy, compact_inputs


def check_denoiser(self, observation, target, weights, noise, t):
    observation=model_module.preprocess_observation(None,observation,train=False)
    tokens,mask,ar=self.embed_prefix(observation)
    _,cache=self.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
    def clean(z):
        suffix,sm,sa,cond=self.embed_suffix(observation,z,jnp.broadcast_to(t,z.shape[0]))
        attn=jnp.concatenate([einops.repeat(mask,'b p -> b s p',s=suffix.shape[1]),make_attn_mask(sm,sa)],axis=-1)
        positions=jnp.sum(mask,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
        (_,out),_=self.PaliGemma.llm([None,suffix],mask=attn,positions=positions,kv_cache=cache,adarms_cond=[None,cond])
        return z-t*self.action_out_proj(out[:,-self.action_horizon:])
    value,pb=jax.vjp(clean,noise)
    residual=(target-value)*weights
    correction=pb(residual)[0]
    direction=correction/jnp.maximum(jnp.linalg.norm(correction),1e-12)
    tangent=jax.jvp(clean,(noise,),(direction,))[1]
    analytic=jnp.sum(correction*direction)
    jvp_inner=jnp.sum(tangent*residual)
    def scalar(z):return jnp.sum(clean(z)*residual)
    def diff(_,eps):
        finite=(scalar(noise+eps*direction)-scalar(noise-eps*direction))/(2*eps)
        return None,finite
    eps=jnp.asarray([1.,.1,.01,.001],dtype=jnp.float32)
    _,fd=jax.lax.scan(diff,None,eps)
    return dict(t=t,vjp=analytic,jvp=jvp_inner,eps=eps,finite_difference=fd,
                adjoint_relative_error=jnp.abs(analytic-jvp_inner)/jnp.maximum(jnp.abs(analytic),1e-12),
                correction_l2=jnp.linalg.norm(correction),residual_l2=jnp.linalg.norm(residual))


def measure(actions,desired,state,previous,cut_row,delay=7,execute=8):
    actual=np.asarray(actions,dtype=float)
    n=len(desired)
    error=np.linalg.norm(actual[:n,:3]-desired[:n,:3],axis=1)*1000
    cut=int(np.floor(cut_row));alpha=cut_row-cut
    position=actual[cut,:3]*(1-alpha)+actual[cut+1,:3]*alpha+state[:3]
    ref=desired[cut,:3]*(1-alpha)+desired[cut+1,:3]*alpha+state[:3]
    grid=cut_row+np.arange(execute)
    emitted=np.asarray([actual[int(t),:3]*(1-t%1)+actual[int(t)+1,:3]*(t%1)+state[:3] for t in grid])
    changes=np.linalg.norm(np.diff(np.vstack([previous[:3],emitted]),axis=0),axis=1)*1000
    return dict(hard_rmse_mm=float(np.sqrt(np.mean(error[:delay]**2))),soft_rmse_mm=float(np.sqrt(np.mean(error[delay:]**2))),
                same_time_seam_error_mm=float(np.linalg.norm(position-ref)*1000),
                actual_seam_jump_mm=float(np.linalg.norm(position-previous[:3])*1000),
                executed_max_step_mm=float(np.max(changes)),executed_step_mm=changes.tolist(),
                row_error_mm=error.tolist(),actions=actual.tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--compute',default='bfloat16');ap.add_argument('--parameters',default='bfloat16')
    ap.add_argument('--extended',action='store_true')
    ap.add_argument('--schedules',action='store_true')
    ap.add_argument('--timing6',action='store_true')
    ap.add_argument('--compact',action='store_true')
    args=ap.parse_args()
    if args.compute=='float32':jax.config.update('jax_default_matmul_precision','highest')
    policy=RtcPolicy(args.checkpoint,compute_dtype=args.compute,parameter_dtype=args.parameters,compact_prefix=args.compact)
    data=np.load(args.samples)
    report=dict(compute=args.compute,parameters=args.parameters,metadata=policy.metadata,
                compact=args.compact,
                samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),rows=[],derivatives=[])
    original_rng=policy.policy._rng
    derivative=nnx_utils.module_jit(types.MethodType(check_denoiser,policy.model))
    def save():
        temporary=args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temporary.replace(args.output)
    for rid in data['request_ids']:
        rid=int(rid);state=data[f'{rid}_state'];reference=data[f'{rid}_prefix']
        obs={'observation/state':state,'prompt':str(data['prompt']),
             'observation/image':cv2.resize(data[f'{rid}_external'],(224,224),interpolation=cv2.INTER_AREA),
             'observation/wrist_image':cv2.resize(data[f'{rid}_wrist'],(224,224),interpolation=cv2.INTER_AREA)}
        desired=encode_prefix(reference,state)
        context=dict(version='nero.rtc.v2',phase='replan',run_epoch=1,request_id=rid,observation_tick=8,plan_revision=1,
                     freeze_until=15,observation_perf_ns=1_400_000_000,timeline_origin_perf_ns=1_000_000_000,
                     freeze_until_perf_ns=1_750_000_000,action_row0_offset_s=0,period_s=.05,inference_delay=7,
                     execute_horizon=8,prefix_attention_horizon=12,absolute_prefix=reference.tolist())
        previous=data[f'{rid}_previous'];cut_row=float(data[f'{rid}_cut_row'])
        if args.timing6:
            context.update(inference_delay=6,execute_horizon=6,prefix_attention_horizon=14,freeze_until=14,
                           freeze_until_perf_ns=1_700_000_000)
        for seed in (11649,11650,11651):
            noise=np.random.default_rng(seed).normal(size=(20,32)).astype(np.float32)
            plain=None
            variants=(('normal',0.,10),('rtc_zero',0.,10),('rtc',5.,10),('rtc',20.,10),('rtc',100.,10),('rtc',5.,100))
            if args.extended:
                variants=(('rtc',20.,20),('rtc',50.,50),('rtc',100.,100),('rtc',200.,100))
            if args.schedules:
                variants=(('linear',5.,10),('linear',20.,20),('linear',100.,100),
                          ('ones',5.,10),('ones',20.,20),('ones',100.,100),('zeros',5.,10))
                if args.compute=='bfloat16':
                    variants=(('ones',20.,20),('ones',25.,25),('ones',30.,30),
                              ('ones',40.,40),('ones',50.,50),('ones',100.,100))
            if args.timing6:
                variants=(('exp',5.,10),('exp',20.,20),('linear',30.,30),('ones',20.,20),('ones',30.,30),('ones',40.,40))
            for name,cap,steps in variants:
                policy.policy._rng=original_rng;policy.cap=cap;policy.steps=steps
                start=time.perf_counter()
                if args.schedules or args.timing6:
                    inputs=policy.policy._input_transform(obs)
                    if args.compact:inputs=compact_inputs(inputs)
                    inputs=jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
                    packed=pack_normalized_prefix(policy.normalize({'actions':desired})['actions'])
                    weights=np.zeros((1,20,32),np.float32)
                    weights[0,:,:7]=np.asarray(prefix_weights(context['inference_delay'],min(len(reference),context['prefix_attention_horizon']),schedule=name))[:,None]
                    sampled=policy.rtc_sample(original_rng,model_module.Observation.from_dict(inputs),jnp.asarray(packed),
                                              jnp.asarray(weights),num_steps=steps,cap=cap,noise=jnp.asarray(noise).reshape(1,20,32),compact_prefix=args.compact)
                    jax.block_until_ready(sampled)
                    result=policy.policy._output_transform({'state':np.asarray(inputs['state'][0]),'actions':np.asarray(sampled[0])})
                else:
                    result=policy.infer(obs if name=='normal' else dict(obs,rtc_context=context),noise=noise)
                actions=np.asarray(result['actions'])
                row=dict(request_id=rid,seed=seed,mode=name,cap=cap,steps=steps,
                         inference_delay=context['inference_delay'],execute_horizon=context['execute_horizon'],
                         elapsed_ms=(time.perf_counter()-start)*1000,
                         **measure(actions,desired,state,previous,cut_row,context['inference_delay'],context['execute_horizon']))
                if name=='normal':plain=actions
                elif name=='rtc_zero':row['zero_vs_normal_max_abs']=float(np.max(np.abs(actions-plain)))
                report['rows'].append(row);save()
            print(json.dumps(dict(request_id=rid,seed=seed,done=True)),flush=True)
        if rid==13 and not args.extended and not args.schedules and not args.timing6:
            inputs=policy.policy._input_transform(obs)
            inputs=jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
            observation=model_module.Observation.from_dict(inputs)
            packed=pack_normalized_prefix(policy.normalize({'actions':desired})['actions'])
            weights=np.zeros((1,20,32),np.float32);weights[0,:,:7]=np.asarray(prefix_weights(7,12))[:,None]
            for t in (1.,.5,.1):
                result=derivative(observation,jnp.asarray(packed),jnp.asarray(weights),jnp.asarray(noise).reshape(1,20,32),jnp.asarray(t,np.float32))
                report['derivatives'].append({k:np.asarray(v).tolist() for k,v in jax.device_get(result).items()});save()
    report['finished']=True;save()
    print('FINISHED',str(args.output),flush=True)

if __name__=='__main__':main()
