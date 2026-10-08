"""Transactional MIT reference; only the CAN sender (or shadow sender) commits.

Snapshots are observations, never a second trajectory or an acknowledgement
to be integrated by OSC. Failed/partial batches must not commit a candidate.
"""
from __future__ import annotations

import math
import threading


def vector(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError(f"{name} requires seven values")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be finite")
    return result


class ImpedanceReference:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None

    def snapshot(self):
        with self._lock:
            return None if self._state is None else {
                k: list(v) if isinstance(v, list) else v for k, v in self._state.items()
            }

    def prepare(self, command, velocity, *, started_perf_ns, epoch, generation,
                max_speed, max_acceleration):
        measured = vector(command["feedback_q_rad"], "measured position")
        desired = vector(velocity, "reference velocity")
        state = self.snapshot()
        action = command.get("reference_action", "track")
        if action not in {"track", "hold"}:
            raise ValueError("invalid MIT reference action")
        token = (int(epoch), int(generation), action)
        reset = (state is None or state["epoch"] != epoch
                 or (action == "hold" and tuple(state["token"]) != token)
                 or (action == "track" and state["action"] == "hold"))
        dt = 0.0 if reset else (started_perf_ns - state["started_perf_ns"]) / 1e9
        if not reset and not 0 < dt <= .2:
            raise RuntimeError(f"invalid MIT control interval: {dt}")
        if reset or action == "hold":
            q = measured if reset else state["position_rad"]
            v = [0.0] * 7
        else:
            speed, acc = float(max_speed), float(max_acceleration)
            if not math.isfinite(speed) or not math.isfinite(acc) or speed <= 0 or acc <= 0:
                raise ValueError("MIT reference needs finite positive motion limits")
            v = [max(prior - acc * dt, min(prior + acc * dt, max(-speed, min(speed, d))))
                 for d, prior in zip(desired, state["velocity_rad_s"])]
            # Measured-state safety clipping takes priority over acceleration
            # shaping, including nonzero reduced-speed commands.
            mask = command.get("reference_gate_mask", [False] * 7)
            if len(mask) != 7:
                raise ValueError("invalid MIT reference gate mask")
            v = [d if gated else x for gated, d, x in zip(mask, desired, v)]
            q = [p + x * dt for p, x in zip(state["position_rad"], v)]
        limits = command.get("reference_limits")
        if not limits:
            raise ValueError("MIT reference requires joint limits")
        lower, upper = vector(limits["lower_rad"], "lower"), vector(limits["upper_rad"], "upper")
        if any(not lo <= p <= hi for p, lo, hi in zip(q, lower, upper)):
            raise RuntimeError("MIT reference outside joint limits")
        candidate = {"position_rad": list(q), "velocity_rad_s": v,
            "started_perf_ns": started_perf_ns, "epoch": int(epoch),
            "token": list(token), "action": action, "dt_s": dt,
            "limited": any(abs(a - b) > 1e-9 for a, b in zip(v, desired))}
        return candidate

    def commit(self, candidate):
        with self._lock:
            self._state = {k: list(v) if isinstance(v, list) else v for k, v in candidate.items()}


def torque_diagnostics(command, q_des):
    """Host estimate only; neither a total-torque limiter nor a load sensor."""
    q = vector(command["feedback_q_rad"], "position")
    qd = vector(command.get("feedback_velocity_rad_s", [0.0] * 7), "velocity")
    desired_velocity = vector(command.get("dq_des_rad_s", [0.0] * 7), "desired velocity")
    spring = [k * (r - p) for k, r, p in zip(command["kp"], q_des, q)]
    damping = [k * (desired - actual) for k, desired, actual in zip(command["kd"], desired_velocity, qd)]
    return {"spring_nm": spring, "damping_nm": damping,
        "feedforward_nm": list(command["tau_ff_nm"]),
        "estimated_total_nm": [a + b + c for a, b, c in zip(spring, damping, command["tau_ff_nm"])],
        "total_torque_protection_confirmed": False}
