"""Isolated OpenPI RTC server; reads the original checkpoint, owns no robot.

Run with the original package's venv and PYTHONPATH. Ordinary observations are
still supported. RTC is negotiated in metadata, never silently faked by a client.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from nero_rtc_sampling import RTCSampler
from nero_rtc_contract import PROTOCOL, unscale_targets, weights_for

class RTCPolicy:
    def __init__(self, policy, checkpoint):
        self.policy = policy
        if policy._is_pytorch_model or not getattr(policy._model, 'pi05', False):
            raise ValueError('This adapter requires the existing JAX pi0.5 model')
        self.sampler = nnx_utils.module_jit(RTCSampler(policy._model).sample, static_argnames=('num_steps',))
        self.metadata = dict(policy.metadata, nero_rtc={'protocol': PROTOCOL, 'valid_action_dim': 7,
                                                     'action_horizon': 10, 'mode': 'inference_guidance'})
        self.metadata['checkpoint_dir'] = str(Path(checkpoint).resolve())
        paths = list((Path(checkpoint)/'assets').rglob('norm_stats.json'))
        if len(paths) != 1:
            raise ValueError('RTC requires one unambiguous checkpoint normalization file')
        self.norm_path = paths[0]
        self.metadata['norm_stats_sha256'] = hashlib.sha256(self.norm_path.read_bytes()).hexdigest()

    def infer(self, request, *, noise=None):
        request = dict(request)
        # Current 11649 model is state7. Keep legacy console state8 compatible
        # exactly as the existing proxy: discard only the redundant -gripper.
        if self.metadata.get('state_dim') == 7:
            state = np.asarray(request['observation/state'], dtype=np.float32)
            if state.shape == (8,):
                request['observation/state'] = state[:7].copy()
        context = request.pop('rtc_context', None)
        if context is None:
            result = self.policy.infer(request, noise=noise)
            result['rtc'] = {'protocol': PROTOCOL, 'applied': False, 'reason': 'no_prior'}
            return result
        if context.get('protocol') != PROTOCOL:
            raise ValueError('unsupported RTC protocol')
        horizon = self.policy._model.action_horizon
        count = int(context['overlap_steps']); delay = int(context['delay_steps'])
        if horizon != 10 or not 1 <= count <= horizon:
            raise ValueError('NERO RTC requires H10 and a valid overlap')
        target = np.asarray(context['absolute_targets'], dtype=np.float32)
        if target.shape != (count, 7):
            raise ValueError('RTC prior count mismatch')
        target = unscale_targets(request['observation/state'], target, float(context['action_delta_scale']))
        # Padding is finite but NOT constrained. Reuse exact existing input
        # transforms: absolute -> chunk start -> quantile norm -> 32D padding.
        padded = np.repeat(target[-1:], horizon, axis=0); padded[:count] = target
        with_actions = dict(request, actions=padded)
        inputs = self.policy._input_transform(copy.deepcopy(with_actions))
        prior = np.asarray(inputs.pop('actions'), dtype=np.float32)
        if prior.shape != (horizon, self.policy._model.action_dim) or not np.isfinite(prior).all():
            raise ValueError('invalid transformed RTC prior')
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_api.Observation.from_dict(inputs)
        self.policy._rng, rng = jax.random.split(self.policy._rng)
        noise = jax.random.normal(rng, (1, horizon, self.policy._model.action_dim)) if noise is None else jnp.asarray(noise).reshape(1,horizon,self.policy._model.action_dim)
        weights = weights_for(delay, count, horizon)
        started = time.perf_counter()
        sampled = self.sampler(rng, observation, jnp.asarray(prior)[None, ...],
                               jnp.asarray(weights), jnp.asarray(10., dtype=jnp.float32), noise)
        sampled = np.asarray(sampled[0])  # Synchronize before reporting latency.
        if not np.isfinite(sampled).all():
            raise ValueError('nonfinite RTC output')
        result = self.policy._output_transform({'state': np.asarray(inputs['state'][0]), 'actions': sampled})
        result['policy_timing'] = {'infer_ms': (time.perf_counter()-started)*1000}
        result['rtc'] = {'protocol': PROTOCOL, 'applied': True,
                         'reference_chunk_id': context['reference_chunk_id'],
                         'overlap_steps': count, 'delay_steps': delay, 'weights': weights.tolist()}
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--audit-dir', required=True)
    args = parser.parse_args()
    if args.config:
        from openpi.training.config import get_config
        config = get_config(args.config)
    else:
        from nero_config import make_config
        config = make_config()
    policy = RTCPolicy(policy_config.create_trained_policy(config, args.checkpoint), args.checkpoint)
    # Both compiled paths warm up BEFORE opening the WebSocket. No robot/UI
    # data, no CAN: prevents cold JIT results expiring during a first real run.
    obs = {'observation/image': np.zeros((224,224,3), np.uint8),
           'observation/wrist_image': np.zeros((224,224,3), np.uint8),
           'observation/state': np.array([-.3,0,.3,0,0,0,.5,-.5], np.float32),
           'prompt': 'Pick up the yellow cube and move it to the P1 area.'}
    if policy.metadata.get('state_dim') == 7:
        obs['observation/state'] = obs['observation/state'][:7].copy()
    initial = policy.infer(obs)
    state = obs['observation/state']
    rows = initial['actions'][:5]
    abs_rows = np.concatenate([state[:3]+rows[:,:3],
                               (Rotation.from_rotvec(rows[:,3:6])*Rotation.from_rotvec(state[3:6])).as_rotvec(),
                               rows[:,6:7]], axis=1)
    obs['rtc_context'] = {'protocol': PROTOCOL, 'reference_chunk_id': 0, 'absolute_targets': abs_rows,
                          'overlap_steps': 5, 'delay_steps': 3, 'action_delta_scale': 1.}
    cold = policy.infer(obs)
    warm = [policy.infer(obs) for _ in range(3)]
    # Same observation/noise comparison: verify guidance direction and check
    # the zero-guidance sampler against the original pi0.5 implementation.
    seed_noise = np.random.default_rng(20261006).normal(size=(1,10,32)).astype(np.float32)
    ordinary_obs = {k:v for k,v in obs.items() if k != 'rtc_context'}
    plain = policy.infer(ordinary_obs, noise=seed_noise)
    guided = policy.infer(obs, noise=seed_noise)
    padded = np.repeat(abs_rows[-1:],10,axis=0);padded[:5]=abs_rows
    transformed = policy.policy._input_transform(dict(ordinary_obs, actions=padded))
    normalized_prior = np.asarray(transformed.pop('actions'),dtype=np.float32)
    batch = jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed)
    zero = policy.sampler(jax.random.key(0), model_api.Observation.from_dict(batch),
                           jnp.asarray(normalized_prior)[None,...],jnp.zeros(10),
                           jnp.asarray(10.,dtype=jnp.float32),jnp.asarray(seed_noise))
    zero = policy.policy._output_transform({'state':np.asarray(batch['state'][0]),'actions':np.asarray(zero[0])})
    # Compare in normalized 7D action space, not raw mixed metre/radian units.
    stats = json.loads(policy.norm_path.read_text())['norm_stats']['actions']
    span = np.asarray(stats['q99'])-np.asarray(stats['q01'])+1e-6
    desired = initial['actions'][:5]
    weight = weights_for(3,5,10)[:5,None]
    errors = {name:float(np.sum(((value['actions'][:5]-desired)/span)**2*weight))
              for name,value in [('plain',plain),('guided',guided)]}
    parity_error = float(np.max(np.abs(zero['actions']-plain['actions'])))
    report = {'no_robot': True, 'metadata': policy.metadata,
              'ordinary_shape': list(initial['actions'].shape), 'rtc_shape': list(cold['actions'].shape),
              'finite': bool(all(np.isfinite(x['actions']).all() for x in [initial,cold,*warm])),
              'warm_infer_ms': [x['policy_timing']['infer_ms'] for x in warm],
              'rtc': warm[-1]['rtc'], 'same_noise_prefix_error': errors,
              'zero_guidance_max_physical_error': parity_error}
    audit = Path(args.audit_dir); audit.mkdir(parents=True, exist_ok=True)
    (audit/'rtc_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['finite'] or report['rtc_shape'] != [10,7]:
        raise RuntimeError('RTC preflight failed')
    if parity_error>1e-4 or errors['guided']>=errors['plain']:
        raise RuntimeError('RTC parity/prefix guidance did not pass; do not open server')
    from jpeg95_teacher_audit import audit as jpeg_audit
    jpeg_audit(policy, audit/'jpeg95_teacher_audit.json')
    if not args.preflight_only:
        websocket_policy_server.WebsocketPolicyServer(policy, host='0.0.0.0', port=args.port,
                                                       metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
