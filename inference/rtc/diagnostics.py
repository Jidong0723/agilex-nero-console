"""Physical, same-time RTC prefix residuals; no GPU or robot dependencies."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation


@contextmanager
def diagnostic_sampler_override(policy, values):
    """Offline-only, per-request override inside RtcPolicy's existing lock."""
    allowed = {'num_steps':'steps', 'guidance_cap':'cap',
               'prefix_attention_schedule':'prefix_attention_schedule',
               'terminal_refinement_steps':'terminal_refinement_steps', 'prefix_mask_mode':'prefix_mask_mode',
               'terminal_noise_time':'terminal_noise_time'}
    if not isinstance(values, dict) or not values or set(values)-set(allowed):
        raise ValueError('Invalid diagnostic sampler fields')
    if 'num_steps' in values and (type(values['num_steps']) is not int or not 1 <= values['num_steps'] <= 100):
        raise ValueError('Invalid diagnostic steps')
    if 'guidance_cap' in values and (type(values['guidance_cap']) not in (float,int) or not 0 < values['guidance_cap'] <= 100000):
        raise ValueError('Invalid diagnostic cap')
    if 'prefix_attention_schedule' in values and values['prefix_attention_schedule'] not in {'ones','linear','exp','zeros'}:
        raise ValueError('Invalid diagnostic schedule')
    if values.get('prefix_mask_mode','timestamp_frozen') not in {'legacy_compressed','nominal','timestamp_frozen'}:
        raise ValueError('Invalid diagnostic mask mode')
    refinement = values.get('terminal_refinement_steps', getattr(policy, 'terminal_refinement_steps', 1))
    if type(refinement) is not int or not 1 <= refinement < values.get('num_steps', policy.steps):
        raise ValueError('Invalid diagnostic terminal refinement steps')
    terminal=values.get('terminal_noise_time',policy.terminal_noise_time)
    if type(terminal) not in (int,float) or not 0 <= terminal < 1:
        raise ValueError('Invalid diagnostic terminal noise time')
    if refinement > 1 and not 0 < terminal < .02:
        raise ValueError('Invalid diagnostic terminal noise time')
    prior = {name:getattr(policy,name,'timestamp_frozen' if name=='prefix_mask_mode' else 1)
             for name in (*allowed.values(),'requires_guided_execution')}
    try:
        for key,value in values.items():setattr(policy,allowed[key],value)
        policy.requires_guided_execution = bool(policy.terminal_noise_time and policy.prefix_attention_schedule in {'ones', 'linear'})
        yield
    finally:
        for name,value in prior.items():setattr(policy,name,value)


def prefix_diagnostics(actions, state, reference, weights, *, delay, config):
    actions = np.asarray(actions, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if actions.shape != (20, 7) or state.shape != (7,) or reference.ndim != 2 or reference.shape[1] != 8:
        raise ValueError('Invalid RTC diagnostic shapes')
    length = min(len(reference), len(weights), len(actions))
    if not all(np.isfinite(value).all() for value in (actions, state, reference, weights)):
        raise ValueError('Non-finite RTC diagnostic input')
    predicted_position = state[:3] + actions[:length, :3]
    predicted_rotation = Rotation.from_rotvec(actions[:length, 3:6]) * Rotation.from_rotvec(state[3:6])
    reference_rotation = Rotation.from_quat(reference[:length, 3:7])
    position_delta = predicted_position - reference[:length, :3]
    rotation_delta = (predicted_rotation * reference_rotation.inv()).as_rotvec()
    gripper_delta = actions[:length, 6] - reference[:length, 7]
    rows = [dict(row=i, target_offset_s=i*.05, guidance_weight=float(weights[i]),
                 frozen_by_delay=i < delay,
                 region='hard' if weights[i] >= 1-1e-6 else ('soft' if weights[i] > 0 else 'unguided'),
                 position_delta_m=position_delta[i].tolist(),
                 position_error_m=float(np.linalg.norm(position_delta[i])),
                 orientation_delta_rotvec_rad=rotation_delta[i].tolist(),
                 orientation_error_rad=float(np.linalg.norm(rotation_delta[i])),
                 gripper_delta=float(gripper_delta[i])) for i in range(length)]
    hard = [row['position_error_m'] for row in rows if row['region'] == 'hard']
    return dict(schema_version='nero.rtc.diagnostics.v1', sampler_config=config,
                prefix_residuals=rows,
                hard_prefix_max_position_error_m=max(hard) if hard else None,
                hard_prefix_all_within_1mm=all(value <= .001 for value in hard) if hard else None,
                reference_sha256=hashlib.sha256(reference.astype('<f8').tobytes()).hexdigest())
