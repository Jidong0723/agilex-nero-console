"""Physical state7/action7 and absolute RTC prefix conversion (no OpenPI dependency)."""
from __future__ import annotations

import numpy as np

VERSION = "nero.rtc.v2"
HORIZON = 20
PERIOD = 0.05
PERIOD_NS = 50_000_000
TIME_CONTRACT = "observation_time_plus_row_index_times_period"


def pack_normalized_prefix(normalized):
    """Physical prefix length is independent of its guidance overlap length."""
    values = np.asarray(normalized, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) > HORIZON or not np.isfinite(values).all():
        raise ValueError('Normalized RTC prefix must be finite [L,7], L<=20')
    padded = np.zeros((1, HORIZON, 32), np.float32)
    padded[0, :len(values), :7] = values
    return padded


def encode_prefix(targets, state):
    """Encode absolute position/xyzw/gripper rows against ONE observation."""
    from scipy.spatial.transform import Rotation
    state = np.asarray(state, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("RTC requires finite state7")
    if targets.size == 0:
        return np.empty((0, 7), np.float32)
    if targets.ndim != 2 or targets.shape[1] != 8 or len(targets) > HORIZON or not np.isfinite(targets).all():
        raise ValueError("RTC absolute prefix must be finite [L,8], L<=20")
    norms = np.linalg.norm(targets[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1) > 1e-3) or np.any((targets[:, 7] < 0) | (targets[:, 7] > 1)):
        raise ValueError("RTC prefix requires unit xyzw and gripper in [0,1]")
    result = np.empty((len(targets), 7), np.float32)
    result[:, :3] = targets[:, :3] - state[:3]
    result[:, 3:6] = (Rotation.from_quat(targets[:, 3:7]) * Rotation.from_rotvec(state[3:6]).inv()).as_rotvec()
    result[:, 6] = targets[:, 7]
    return result


def validate_context(context):
    if not isinstance(context, dict) or context.get("version") != VERSION:
        raise ValueError("Unsupported rtc_context version")
    for name in ("run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until",
                 "observation_perf_ns", "timeline_origin_perf_ns", "freeze_until_perf_ns"):
        value = context.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"RTC {name} must be a non-negative integer")
    if context.get("period_s") != PERIOD:
        raise ValueError("RTC requires a 50 ms period")
    if context.get('action_row0_offset_s') != 0:
        raise ValueError('RTC action row 0 must correspond to observation time')
    origin, observation = context['timeline_origin_perf_ns'], context['observation_perf_ns']
    if (observation-origin)//PERIOD_NS != context['observation_tick']:
        raise ValueError('RTC observation tick does not match its timestamp')
    if origin+context['freeze_until']*PERIOD_NS != context['freeze_until_perf_ns']:
        raise ValueError('RTC frozen boundary does not match its timestamp')
    targets = context.get("absolute_prefix")
    if not isinstance(targets, (list, np.ndarray)) or len(targets) > HORIZON:
        raise ValueError("Invalid absolute prefix")
    delay = (context['freeze_until_perf_ns']-observation+PERIOD_NS-1)//PERIOD_NS
    if 'execute_horizon' in context:
        d, e = context.get('inference_delay'), context.get('execute_horizon')
        if (isinstance(d, bool) or not isinstance(d, int) or isinstance(e, bool) or not isinstance(e, int)
                or not 0 <= d <= e < HORIZON or e < 1 or d+e > HORIZON):
            raise ValueError('Invalid RTC inference_delay / execute_horizon')
        attention_end = HORIZON if context.get('phase') == 'bootstrap_hold' else HORIZON-e
        if context.get('prefix_attention_horizon') != attention_end:
            raise ValueError('RTC prefix_attention_horizon must equal H for stationary bootstrap, otherwise H - execute_horizon')
        if delay < d:
            raise ValueError('RTC frozen boundary must cover inference_delay')
    bootstrap = context.get("phase") == "bootstrap"
    if bootstrap:
        if len(targets) or delay:
            raise ValueError("Bootstrap must have an empty prefix")
    elif context.get("phase") not in ("replan", "bootstrap_hold") or not 0 <= delay <= len(targets):
        raise ValueError("RTC requires replaceable future rows after the frozen prefix")
    if context.get('phase') == 'bootstrap_hold':
        array = np.asarray(targets, dtype=float)
        if len(targets) != HORIZON or context['observation_tick'] != 0 or not np.allclose(array,array[0],rtol=0,atol=1e-9):
            raise ValueError('HOLD bootstrap must describe one stationary H20 plan at observation tick zero')
    if 'execute_horizon' in context:
        return context['inference_delay'], min(len(targets), context['prefix_attention_horizon'])
    return delay, len(targets)


def echo_context(context):
    result = {key: context[key] for key in ("version", "run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until", "phase",
                                        "observation_perf_ns", "timeline_origin_perf_ns", "freeze_until_perf_ns", "action_row0_offset_s")}
    result.update({key: context[key] for key in ('inference_delay','execute_horizon','prefix_attention_horizon') if key in context})
    return result
