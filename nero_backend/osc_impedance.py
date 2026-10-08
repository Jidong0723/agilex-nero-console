"""MIT hardware boundary, used exclusively by HardwareTxOwner's CAN thread."""
from __future__ import annotations

import math
import time


class ImpedanceHardware:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.active = False
        self.auto_mode = None
        self.config = {}
        self.last_entry = None

    @staticmethod
    def vector(value, name):
        if not isinstance(value, (list, tuple)) or len(value) != 7:
            raise ValueError(f"{name} must contain seven values")
        result = [float(x) for x in value]
        if not all(math.isfinite(x) for x in result):
            raise ValueError(f"{name} must be finite")
        return result

    def validate(self, command):
        values = {key: self.vector(command.get(key), key)
            for key in ("q_des_rad", "dq_des_rad_s", "kp", "kd", "tau_ff_nm")}
        limits = self.config.get("joint_limits") or {}
        lo = limits.get("lower_rad", [-12.5] * 7)
        hi = limits.get("upper_rad", [12.5] * 7)
        if any(not l <= q <= h for q, l, h in zip(values["q_des_rad"], lo, hi)):
            raise ValueError("MIT position outside hardware limits")
        # Reject rather than let the SDK silently clamp the reference velocity.
        # This is a protocol bound, not an interaction-safety limit.
        if any(abs(v) > 45 for v in values["dq_des_rad_s"]):
            raise ValueError("MIT desired velocity outside firmware range")
        if any(not 0 <= x <= 500 for x in values["kp"]) or any(not 0 <= x <= 5 for x in values["kd"]):
            raise ValueError("MIT gains outside firmware range")
        caps = [24, 24, 16, 16, 8, 8, 8] if str(self.backend.sdk_config.get("firmware")) == "default" else [16] * 7
        if any(abs(t) > cap for t, cap in zip(values["tau_ff_nm"], caps)):
            raise ValueError("MIT feedforward outside firmware range")
        gravity = command.get("gravity") or {}
        timestamp = int(command.get("gravity_computed_monotonic_ns") or 0)
        age = (time.monotonic_ns() - timestamp) / 1e9
        if (command.get("tau_ff_source") != "pinocchio_gravity"
                or gravity.get("gravity_state") not in {"FRESH", "HOLD_LAST_VALID"}
                or not gravity.get("model_revision") or not 0 <= age <= 0.1):
            raise RuntimeError("MIT gravity invalid or expired before CAN dispatch")
        feedback_age = (time.monotonic_ns() - int(command.get("feedback_monotonic_ns") or 0)) / 1e9
        if not 0 <= feedback_age <= 0.15 or (time.monotonic_ns() - int(command.get("created_monotonic_ns") or 0)) / 1e9 > 0.15:
            raise RuntimeError("MIT command feedback expired before CAN dispatch")
        return values

    def _wait_mode(self, baseline, *, entering):
        b = self.backend
        expected = 0x04 if str(b.sdk_config.get("firmware")) == "default" else 0x06
        deadline = time.monotonic() + float(self.config.get("enter_timeout_s" if entering else "exit_timeout_s", 1.0))
        while time.monotonic() < deadline:
            sample = b.arm_status_snapshot()
            if int(sample.get("revision") or 0) > baseline and sample.get("ctrl_mode") == 1:
                matches = sample.get("mode_feedback") == expected
                if matches if entering else sample.get("mode_feedback") == 0x05:
                    return sample
            time.sleep(0.005)
        raise RuntimeError(f"MIT {'entry' if entering else 'exit'} mode confirmation timed out: {b.arm_status_snapshot()}")

    def enter(self, entry_factory, config, entry_guard=None):
        b = self.backend
        self.config = dict(config)
        if self.active or b._cpv_stream_started:
            raise RuntimeError("cannot enter MIT while a continuous output is active")
        if b.robot is None or not b._arm_status_observer_installed:
            raise RuntimeError("MIT needs connected robot and fresh Arm Status observer")
        enabled, status = b._enable_all_with_retry(timeout=float(b.motion_config.get("enable_timeout_s", 8.0)))
        if not enabled:
            raise RuntimeError(f"MIT drives not enabled: {status}")
        q = b._read_stable_follower_joints(timeout=0.8)
        feedback_ns = time.monotonic_ns()
        q = self.vector(q, "MIT measured entry position")
        command = entry_factory(q)
        command["feedback_monotonic_ns"] = min(feedback_ns, int(command.get("feedback_monotonic_ns") or feedback_ns))
        values = self.validate(command)
        measured = self.vector(command.get("feedback_q_rad"), "MIT first-frame measured feedback")
        if max(abs(a - c) for a, c in zip(q, measured)) > 0.02:
            raise ValueError("MIT first-frame feedback no longer matches settled hardware")
        if max(abs(a - c) for a, c in zip(measured, values["q_des_rad"])) > 1e-9:
            raise ValueError("MIT first frame must anchor measured position")
        self.auto_mode = b.robot.get_auto_set_motion_mode_enabled()
        baseline = int(b.arm_status_snapshot().get("revision") or 0)
        with b._command_lock:
            b.robot.set_auto_set_motion_mode_enabled(False)
            # Mark possible hardware ownership BEFORE mode request, so failures
            # and partial seven-joint batches always take the official exit.
            self.active = True
            try:
                if entry_guard is not None and not entry_guard():
                    raise PermissionError("MIT entry authority revoked")
                b.robot.set_motion_mode(b.robot.OPTIONS.MOTION_MODE.MIT)
                first = self._send(values, entry_guard, command)
                confirmed = self._wait_mode(baseline, entering=True)
                if entry_guard is not None and not entry_guard():
                    raise PermissionError("MIT entry authority revoked during mode confirmation")
            except Exception:
                self.exit("MIT entry failed")
                raise
        b._control_mode = "OSC_MIT"
        self.last_entry = {"confirmed": True, "baseline_revision": baseline,
            "first_frame": command, "first_frame_result": first, "mode_feedback": confirmed}
        return {"ok": True, "impedance_mode_entry": self.last_entry}

    def _send(self, values, guard=None, command=None):
        times = []
        for index in range(7):
            if guard is not None and not guard():
                error = PermissionError("MIT batch authority revoked")
                error.partial_batch = bool(times)
                raise error
            if command is not None:
                self.validate(command)
            self.backend.robot.move_mit(joint_index=index + 1,
                p_des=values["q_des_rad"][index], v_des=values["dq_des_rad_s"][index],
                kp=values["kp"][index], kd=values["kd"][index], t_ff=values["tau_ff_nm"][index])
            times.append(time.monotonic_ns())
        return {"ok": True, "joint_sent_monotonic_ns": times,
            "finished_monotonic_ns": times[-1], "batch_skew_ms": (times[-1] - times[0]) / 1e6,
            "tau_ff_nm": values["tau_ff_nm"], "output_mode": "impedance"}

    def send(self, command, guard=None):
        b = self.backend
        if not self.active or b._control_mode != "OSC_MIT" or b._cpv_stream_started:
            raise RuntimeError("MIT stream is not exclusively confirmed")
        values = self.validate(command)
        with b._command_lock:
            return self._send(values, guard, command)

    def exit(self, reason, hold_factory=None):
        b = self.backend
        if not self.active:
            return {"ok": True, "already_exited": True}
        with b._command_lock:
            try:
                hold_error = None
                if hold_factory is not None:
                    try:
                        command = hold_factory()
                        values = self.validate(command)
                        measured = self.vector(command.get("feedback_q_rad"), "MIT exit measured anchor")
                        if max(abs(a - c) for a, c in zip(measured, values["q_des_rad"])) > 1e-9:
                            raise ValueError("MIT exit first frame must clear old spring reference")
                        self._send(values, command=command)
                    except Exception as exc:
                        hold_error = exc
                # Follower selects the linkage role, NOT the motion mode. The
                # vendor call preserves MIT and can disable feedback push.
                # Explicitly leave MIT before any CPV position frame; never
                # switch to P/J, whose old targets could be resurrected.
                cpv = getattr(b.robot.OPTIONS.MOTION_MODE, "CPV", None)
                if cpv is None or not callable(getattr(b.robot, "move_cpv_pos", None)):
                    raise RuntimeError("MIT exit requires SDK CPV position hold support")
                baseline = int(b.arm_status_snapshot().get("revision") or 0)
                b.robot.set_motion_mode(cpv)
                confirmed = self._wait_mode(baseline, entering=False)
                q = b._read_stable_follower_joints(timeout=float(self.config.get("exit_timeout_s", 1)))
                q = self.vector(q, "MIT exit feedback")
                # Fresh measured hold, not an old CPV or MIT reference. All
                # writes still occur inside the unique Transport Owner thread.
                for index, position in enumerate(q, 1):
                    b.robot.move_cpv_pos(joint_index=index, pos=position)
                q = b._read_stable_follower_joints(timeout=float(self.config.get("exit_timeout_s", 1)))
                self.vector(q, "MIT exit settled feedback")
                feedback = b.read_cached_osc_feedback() if hasattr(b, "read_cached_osc_feedback") else None
                if feedback is not None:
                    velocities = self.vector(feedback.get("joint_velocity_rad_s"), "MIT exit measured velocity")
                    if max(abs(v) for v in velocities) > .005:
                        raise RuntimeError("MIT exit joints not settled")
                if hold_error is not None:
                    raise RuntimeError(f"MIT exit hold frame unconfirmed: {hold_error}") from hold_error
                b.robot.set_auto_set_motion_mode_enabled(self.auto_mode)
                self.active = False
                b._control_mode, b._last_control_reason = "HOLD", reason
                return {"ok": True, "confirmed": True, "mode_feedback": confirmed, "joint_angles_rad": q}
            except Exception:
                b._control_mode = "FAULT"
                # Keep active=True: CPV may NOT be primed after unconfirmed exit.
                raise
