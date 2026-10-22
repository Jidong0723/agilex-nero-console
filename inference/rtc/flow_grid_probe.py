"""Experimental ODE grid; official VJP field unchanged, not upstream uniform integration."""
import argparse
import json
import types
import time
from pathlib import Path
import cv2
import einops
import numpy as np
import jax
import jax.numpy as jnp
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from serve import RtcPolicy, compact_inputs
from sampling import _guided_velocity_terms, prefix_weights
from codec import encode_prefix,pack_normalized_prefix
from seam_probe import measure


def grid_sample(self,rng,observation,prefix,weights,noise,*,num_steps,cap,terminal_time):
    observation=model_module.preprocess_observation(None,observation,train=False,
        image_keys=tuple(k for k in observation.images if k!='right_wrist_0_rgb'))
    tokens,mask,ar=self.embed_prefix(observation)
    _,cache=self.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
    times=jnp.concatenate([jnp.linspace(1.,terminal_time,num_steps),jnp.zeros(1)])
    def step(x, pair):
        t,dt=pair
        def velocity(z):
            suffix,sm,sa,cond=self.embed_suffix(observation,z,jnp.broadcast_to(t,z.shape[0]))
            attention=jnp.concatenate([einops.repeat(mask,'b p -> b s p',s=suffix.shape[1]),make_attn_mask(sm,sa)],axis=-1)
            positions=jnp.sum(mask,axis=-1)[:,None]+jnp.cumsum(sm,axis=-1)-1
            (_,out),_=self.PaliGemma.llm([None,suffix],mask=attention,positions=positions,kv_cache=cache,adarms_cond=[None,cond])
            return self.action_out_proj(out[:,-self.action_horizon:])
        guided,clean,correction,strength=_guided_velocity_terms(velocity,x,t,prefix,weights,cap)
        nxt=x-dt*guided
        return nxt,dict(t=t,dt=dt,strength=strength,clean=clean,
                        correction_l2=jnp.linalg.norm(correction))
    return jax.lax.scan(step,noise,(times[:-1],times[:-1]-times[1:]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();data=np.load(args.samples);policy=RtcPolicy(args.checkpoint,compact_prefix=True)
    sampler=nnx_utils.module_jit(types.MethodType(grid_sample,policy.model),static_argnames=('num_steps','cap','terminal_time'))
    report=dict(experimental=True,note='Official VJP vector field with a nonuniform time grid; no action projection.',rows=[],traces=[])
    for rid in data['request_ids']:
        rid=int(rid);state=data[f'{rid}_state'];reference=data[f'{rid}_prefix'];desired=encode_prefix(reference,state)
        obs={'observation/state':state,'prompt':str(data['prompt']),
             'observation/image':cv2.resize(data[f'{rid}_external'],(224,224),interpolation=cv2.INTER_AREA),
             'observation/wrist_image':cv2.resize(data[f'{rid}_wrist'],(224,224),interpolation=cv2.INTER_AREA)}
        inputs=compact_inputs(policy.policy._input_transform(obs));inputs=jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
        observation=model_module.Observation.from_dict(inputs)
        packed=pack_normalized_prefix(policy.normalize({'actions':desired})['actions'])
        weights=np.zeros((1,20,32),np.float32);weights[0,:,:7]=np.asarray(prefix_weights(6,14,schedule='ones'))[:,None]
        for seed in (11649,11650,11651):
            noise=np.random.default_rng(seed).normal(size=(1,20,32)).astype(np.float32)
            for steps,terminal,cap in ((20,.02,50.),(20,.01,100.),(20,.005,200.),(20,.001,1000.),(10,.001,1000.)):
                start=time.perf_counter()
                actions,trace=sampler(policy.policy._rng,observation,jnp.asarray(packed),jnp.asarray(weights),jnp.asarray(noise),num_steps=steps,terminal_time=terminal,cap=cap)
                jax.block_until_ready(actions)
                result=policy.policy._output_transform({'state':np.asarray(inputs['state'][0]),'actions':np.asarray(actions[0])})
                row=dict(request_id=rid,seed=seed,steps=steps,terminal_time=terminal,cap=cap,
                    elapsed_ms=(time.perf_counter()-start)*1000,
                    **measure(result['actions'],desired,state,data[f'{rid}_previous'],6.,6,6))
                report['rows'].append(row)
                if rid==13 and seed==11649:
                    tr={k:np.asarray(v).tolist() for k,v in jax.device_get(trace).items() if k!='clean'}
                    report['traces'].append(dict(steps=steps,terminal_time=terminal,cap=cap,**tr))
                tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
            print(json.dumps(dict(request_id=rid,seed=seed)),flush=True)
    report['finished']=True;args.output.write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',flush=True)

if __name__=='__main__':main()
