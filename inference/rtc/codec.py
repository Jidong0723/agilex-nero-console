"""Physical state7/action7 and absolute RTC prefix conversion (no OpenPI dependency)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = "nero.rtc.v1"
HORIZON = 20
PERIOD = 0.05


def encode_prefix(targets, state):
    """Encode absolute position/xyzw/gripper rows against ONE observation."""
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
    for name in ("run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until"):
        value = context.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"RTC {name} must be a non-negative integer")
    if context.get("period_s") != PERIOD:
        raise ValueError("RTC requires a 50 ms period")
    targets = context.get("absolute_prefix")
    if not isinstance(targets, (list, np.ndarray)) or len(targets) > HORIZON:
        raise ValueError("Invalid absolute prefix")
    delay = context["freeze_until"] - context["observation_tick"]
    bootstrap = context.get("phase") == "bootstrap"
    if bootstrap:
        if len(targets) or delay:
            raise ValueError("Bootstrap must have an empty prefix")
    elif context.get("phase") != "replan" or not 1 <= delay < len(targets):
        raise ValueError("RTC requires replaceable future rows after the frozen prefix")
    return delay, len(targets)


def echo_context(context):
    return {key: context[key] for key in ("version", "run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until", "phase")}
