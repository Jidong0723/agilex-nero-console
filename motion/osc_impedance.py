"""Optional OSC output adapter. Pink and all trajectory gates remain upstream.

The LX tracking gains, zero desired velocity, bare-flange gravity (no gripper
mass), and torque shaping are retained. There is no separate control thread:
commands go through the existing epoch-guarded, latest-only CAN mailbox.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from motion.osc_impedance_dynamics import (
    GravityFeedforwardManager, GravityProcessWorkerClient, GravityReadonlyProcess, _vector,
)


class ImpedanceOutput:
    def __init__(self, root: Path, osc_config: dict, lower: list, upper: list, *, worker=None) -> None:
        self.config = json.loads((root / "config" / "osc_impedance.json").read_text(encoding="utf-8-sig"))
        self.gravity_config = self.config["gravity_compensation"]
        self.lower, self.upper = list(lower), list(upper)
        self.config["joint_limits"] = {"lower_rad": self.lower, "upper_rad": self.upper}
        self.kp, self.kd = _vector(self.config["kp"], "kp"), _vector(self.config["kd"], "kd")
        if any(not 0 <= x <= 500 for x in self.kp) or any(not 0 <= x <= 5 for x in self.kd):
            raise ValueError("invalid impedance gains")
        solver = osc_config["solver"]
        self.worker = worker or GravityProcessWorkerClient(GravityReadonlyProcess(
            root / solver["python"], root / "motion" / "osc_impedance_gravity_server.py",
            root / solver["urdf"], osc_config.get("joint_conventions"),
            {"bare_flange": {"mass_kg": 0, "enabled": True, "validated": True}},
        ))
        self.manager = GravityFeedforwardManager(self.gravity_config)
        self.last_command = None
        self.last_error = None
        # LX first_frame_gravity=fresh initializes alpha at its target, not zero.
        self.scale = min(float(self.gravity_config["gravity_scale"]), float(self.gravity_config["alpha_target_max"]))

    def start(self, q: list) -> dict:
        self.worker.start()
        return self.entry(q)

    def entry(self, q: list) -> dict:
        self.worker.request_gravity(q, 0, "bare_flange")
        deadline = time.monotonic() + float(self.config["preheat_timeout_s"])
        try:
            while time.monotonic() < deadline:
                result = self.worker.latest_result()
                if (result and result.get("ok")
                        and (time.monotonic_ns() - result["computed_monotonic_ns"]) / 1e9 < 0.06
                        and max(abs(a - b) for a, b in zip(q, result["q_actual_rad"])) < 0.02):
                    # LX hardware starts with dt=0: fresh model, zero slewed torque.
                    return self.build(q, q, 0.0, 0, 0)
                time.sleep(0.005)
            raise RuntimeError(f"gravity preheat timed out: {self.worker.latest_result()}")
        except Exception:
            self.close()
            raise

    def build(self, q_des: list, q_actual: list, dt: float, sample_id: int, epoch: int, *, feedback_monotonic_ns=None) -> dict:
        q_des, q_actual = _vector(q_des, "q_des"), _vector(q_actual, "q_actual")
        if any(not lo <= q <= hi for q, lo, hi in zip(q_des, self.lower, self.upper)):
            raise ValueError("MIT target outside OSC limits")
        self.worker.request_gravity(q_actual, sample_id, "bare_flange")
        result = self.worker.latest_result()
        ff = self.manager.compute(q_actual, gravity_result=result, dt_s=dt,
            enabled=True, arm_token=True, scale_override=self.scale,
            allow_hold_last_valid=bool(self.gravity_config["allow_hold_last_valid"]))
        if ff.state == "INVALID":
            self.last_error = f"invalid impedance gravity: {ff.limit_reason}"
            raise RuntimeError(self.last_error)
        command = {"q_des_rad": q_des, "dq_des_rad_s": [0.0] * 7,
            "kp": self.kp, "kd": self.kd, "tau_ff_nm": list(ff.tau_ff_sent_nm),
            "tau_ff_source": "pinocchio_gravity", "gravity": ff.as_dict(),
            "gravity_computed_monotonic_ns": result["computed_monotonic_ns"],
            "gravity_max_age_s": float(self.gravity_config["hold_last_valid_max_age_s"]),
            "gravity_model_tau_nm": list(ff.tau_ff_target_nm),
            "feedback_q_rad": q_actual, "sample_id": sample_id, "motion_epoch": epoch,
            "feedback_monotonic_ns": feedback_monotonic_ns or time.monotonic_ns(),
            "created_monotonic_ns": time.monotonic_ns()}
        self.last_command = command
        self.last_error = None
        return command

    def diagnostics(self) -> dict:
        return {"loaded": True, "gravity_worker": self.worker.health(),
            "command": self.last_command, "error": self.last_error,
            "tool_assumption": "bare flange; gripper and payload mass ignored"}

    def close(self) -> None:
        self.worker.close()


class ShadowImpedancePlant:
    """MIT PD surrogate, not a hardware model or a CPV position follower.

    Feedforward metadata supplies the nominal calibrated gravity load. This
    isolates output/gain/torque behavior; it cannot validate real-arm dynamics.
    """
    def __init__(self, q, output, config):
        self.q, self.qd = list(q), [0.0] * 7
        self.output, self.config = output, config
        self.command = None
        self.count = 0

    def dispatch(self, q, now):
        self.command = dict(self.output.last_command)
        self.command["q_des_rad"] = list(q)
        self.count += 1

    def advance(self, dt, now):
        if self.command:
            import math
            c = self.command
            steps = max(1, math.ceil(dt / 0.002))
            h = dt / steps
            for _ in range(steps):
                for i in range(7):
                    torque = (c["kp"][i] * (c["q_des_rad"][i] - self.q[i])
                        - (c["kd"][i] + 0.5) * self.qd[i]
                        + c["tau_ff_nm"][i] - c["gravity_model_tau_nm"][i])
                    acceleration = max(-self.config["max_joint_acceleration_rad_s2"],
                        min(self.config["max_joint_acceleration_rad_s2"], torque / 0.25))
                    speed = self.config["max_joint_speed_rad_s"]
                    self.qd[i] = max(-speed, min(speed, self.qd[i] + acceleration * h))
                    self.q[i] = max(self.output.lower[i], min(self.output.upper[i], self.q[i] + self.qd[i] * h))
        return list(self.q), list(self.qd), 0.0

    def diagnostics(self):
        return {"enabled": True, "output_mode": "impedance", "dispatch_count": self.count,
            "model": "MIT PD surrogate; calibrated gravity command metadata"}
