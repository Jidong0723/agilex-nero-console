from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any


def finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


class Solver:
    """Pinocchio/Pink differential IK worker.

    Trajectory integration is intentionally owned by the control-service
    thread. This process only turns one current robot estimate into a
    constrained joint-velocity proposal.
    """

    def __init__(self, urdf: Path, tcp_offset: list[float], period_s: float = 0.02) -> None:
        import pinocchio as pin
        import pink
        import numpy as np

        self.pin = pin
        self.pink = pink
        self.np = np
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.hard_lower = np.asarray(self.model.lowerPositionLimit[:7], dtype=float).copy()
        self.hard_upper = np.asarray(self.model.upperPositionLimit[:7], dtype=float).copy()
        joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [self.model.getJointId(name) for name in joint_names]
        if any(item == 0 for item in self.joint_ids):
            raise RuntimeError(f"NERO URDF is missing one of {joint_names}")
        self.frame_name = "osc_tcp_candidate"
        tcp_placement = pin.SE3(pin.Quaternion.Identity(), np.asarray(tcp_offset, dtype=float))
        self.frame_id = self.model.addFrame(
            pin.Frame(self.frame_name, self.joint_ids[-1], tcp_placement, pin.FrameType.OP_FRAME)
        )
        self.data = self.model.createData()
        from pink.tasks import DampingTask, FrameTask, PostureTask
        from pink.limits import AccelerationLimit, ConfigurationLimit, VelocityLimit
        self.FrameTask = FrameTask
        self.PostureTask = PostureTask
        self.DampingTask = DampingTask
        self.AccelerationLimit = AccelerationLimit
        self.ConfigurationLimit = ConfigurationLimit
        self.VelocityLimit = VelocityLimit
        self.period_s = float(period_s)
        if not 0.005 <= self.period_s <= 0.1:
            raise ValueError("period_s must be between 0.005 and 0.1")

    def _q(self, q: list[float]):
        qv = self.np.zeros(self.model.nq)
        qv[:7] = q
        return qv

    def fk(self, q: list[float]) -> dict[str, Any]:
        qv = self._q(q)
        self.pin.forwardKinematics(self.model, self.data, qv)
        self.pin.updateFramePlacements(self.model, self.data)
        tcp_pose = self.data.oMf[self.frame_id]
        jac = self.pin.computeFrameJacobian(
            self.model, self.data, qv, self.frame_id, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return {
            "position_m": tcp_pose.translation.tolist(),
            "rotation": tcp_pose.rotation.tolist(),
            "jacobian": jac.tolist(),
            "link_positions_m": [[0.0, 0.0, 0.0]] + [self.data.oMi[joint_id].translation.tolist() for joint_id in self.joint_ids],
        }

    def solve(self, request: dict[str, Any]) -> dict[str, Any]:
        np = self.np
        pin = self.pin
        started = time.perf_counter()
        q_feedback = finite_vector(request.get("joint_angles_rad"), 7, "joint_angles_rad")
        measured_joint_angles = finite_vector(
            request.get("measured_joint_angles_rad", q_feedback), 7, "measured_joint_angles_rad"
        )
        target_position = finite_vector(request.get("target_position_m"), 3, "target_position_m")
        target_quaternion = finite_vector(request.get("target_orientation_xyzw"), 4, "target_orientation_xyzw")
        quaternion_norm = float(np.linalg.norm(np.asarray(target_quaternion, dtype=float)))
        if not 0.9 <= quaternion_norm <= 1.1:
            raise ValueError("target_orientation_xyzw must have a norm between 0.9 and 1.1")
        target_quaternion = [value / quaternion_norm for value in target_quaternion]
        last_sent = finite_vector(request.get("last_sent_joint_velocity_rad_s", [0.0] * 7), 7, "last_sent_joint_velocity_rad_s")
        velocity_limits = finite_vector(request.get("joint_speed_limit_rad_s"), 7, "joint_speed_limit_rad_s")
        acceleration_limits = finite_vector(request.get("joint_acceleration_limit_rad_s2"), 7, "joint_acceleration_limit_rad_s2")
        soft_lower = finite_vector(request.get("soft_lower_rad"), 7, "soft_lower_rad")
        soft_upper = finite_vector(request.get("soft_upper_rad"), 7, "soft_upper_rad")
        dt = float(request.get("dt_s", self.period_s))
        if not math.isfinite(dt) or not 0.001 <= dt <= 0.2:
            raise ValueError("dt_s must be finite and between 0.001 and 0.2")
        if any(lower >= upper for lower, upper in zip(soft_lower, soft_upper)):
            raise ValueError("soft joint limits are invalid")
        if any(
            lower < hard_lower - 1e-8 or upper > hard_upper + 1e-8
            for lower, upper, hard_lower, hard_upper in zip(soft_lower, soft_upper, self.hard_lower, self.hard_upper)
        ):
            raise ValueError("soft joint limits must stay within URDF limits")

        # Pink's configuration and braking-distance constraints use the
        # software soft range. A real NERO can report a sample a few
        # milliradians beyond that range while its controller is still
        # healthy (typically at a hard-limit calibration seam). Reject a
        # large excursion, but include a small current-state-only extension
        # so ConfigurationLimit can evaluate the measured configuration.
        # The extension is not a permission to move outward: the final
        # supervisor gate still uses the immutable hardware/URDF limits.
        feedback_limit_tolerance = float(request.get("feedback_limit_tolerance_rad", 0.05))
        if not math.isfinite(feedback_limit_tolerance) or feedback_limit_tolerance < 0.0:
            raise ValueError("feedback_limit_tolerance_rad must be finite and non-negative")
        lower_excursion = np.maximum(0.0, self.hard_lower - np.asarray(q_feedback, dtype=float))
        upper_excursion = np.maximum(0.0, np.asarray(q_feedback, dtype=float) - self.hard_upper)
        if float(np.max(np.maximum(lower_excursion, upper_excursion))) > feedback_limit_tolerance:
            raise ValueError("joint feedback is outside the configured model limits")
        solver_lower = np.minimum(np.asarray(soft_lower, dtype=float), np.asarray(q_feedback, dtype=float) - 1e-6)
        solver_upper = np.maximum(np.asarray(soft_upper, dtype=float), np.asarray(q_feedback, dtype=float) + 1e-6)
        self.model.lowerPositionLimit[:7] = solver_lower
        self.model.upperPositionLimit[:7] = solver_upper
        self.model.velocityLimit[:7] = velocity_limits
        qv = self._q(q_feedback)
        pin.forwardKinematics(self.model, self.data, qv)
        pin.updateFramePlacements(self.model, self.data)
        jac = pin.computeFrameJacobian(self.model, self.data, qv, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, :7]
        singular_values = np.linalg.svd(jac, compute_uv=False)
        condition = float(singular_values[0] / max(singular_values[-1], 1e-9))
        if condition > float(request.get("condition_limit", 2500.0)):
            return {"ok": False, "error": "singularity condition limit exceeded", "condition_number": condition}

        current_tcp = self.data.oMf[self.frame_id]
        target_tcp = pin.SE3(
            pin.Quaternion(
                float(target_quaternion[3]), float(target_quaternion[0]),
                float(target_quaternion[1]), float(target_quaternion[2]),
            ).toRotationMatrix(),
            np.asarray(target_position, dtype=float),
        )
        command_position = finite_vector(request.get("command_target_position_m", target_position), 3, "command_target_position_m")
        command_quaternion = finite_vector(request.get("command_target_orientation_xyzw", target_quaternion), 4, "command_target_orientation_xyzw")
        command_norm = math.sqrt(sum(item * item for item in command_quaternion))
        if command_norm <= 1e-12:
            raise ValueError("command_target_orientation_xyzw must be non-zero")
        command_quaternion = [item / command_norm for item in command_quaternion]
        command_tcp = pin.SE3(
            pin.Quaternion(
                float(command_quaternion[3]), float(command_quaternion[0]),
                float(command_quaternion[1]), float(command_quaternion[2]),
            ).toRotationMatrix(),
            np.asarray(command_position, dtype=float),
        )
        configuration = self.pink.Configuration(self.model, self.data, qv)
        # Publish errors against the caller's absolute target, not the bounded
        # one-cycle prediction used internally by the IK task.
        rotation_error = float(np.linalg.norm(pin.log3(current_tcp.rotation.T @ command_tcp.rotation)))
        position_error = float(np.linalg.norm(command_tcp.translation - current_tcp.translation))
        position_cost = float(request.get("frame_position_cost", 10.0))
        orientation_cost = float(request.get("frame_orientation_cost", 1.0))
        damping_cost = float(request.get("damping_cost", 0.05))
        frame_gain = float(request.get("frame_gain", 0.5))
        frame_lm_damping = float(request.get("frame_lm_damping", 0.0))
        # Frame gain is an IK feedback gain, not a joint-speed limit.  The
        # OSC output path independently enforces joint velocity and the
        # 5 rad/s² acceleration limit, so permit a bounded gain above one for
        # critically faster Cartesian error correction.
        if not all(math.isfinite(value) and value > 0.0 for value in (position_cost, orientation_cost, damping_cost)) or not math.isfinite(frame_gain) or not 0.0 < frame_gain <= 5.0 or not math.isfinite(frame_lm_damping) or frame_lm_damping < 0.0:
            raise ValueError("Pink task costs or gain are invalid")
        frame_task = self.FrameTask(
            self.frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=frame_gain,
            lm_damping=frame_lm_damping,
        )
        frame_task.set_target(target_tcp)
        posture_reference = finite_vector(request.get("posture_reference_rad", q_feedback), 7, "posture_reference_rad")
        posture_cost = float(request.get("posture_cost", 0.005))
        posture_task = self.PostureTask(cost=posture_cost)
        posture_task.set_target(self._q(posture_reference))
        damping_task = self.DampingTask(cost=damping_cost)
        # Joint centering is a soft, limit-local objective rather than a
        # permanent pull toward the geometric midpoint. Inside the central
        # band its cost is exactly zero, so it cannot create hold-stage null
        # motion merely because the session posture is not centered.
        center_deadband = float(request.get("joint_center_deadband", 0.70))
        center_cost = float(request.get("joint_center_cost", 0.0))
        if not 0.0 <= center_deadband < 1.0 or not math.isfinite(center_cost) or center_cost < 0.0:
            raise ValueError("joint-center parameters are invalid")
        midpoint = 0.5 * (solver_lower + solver_upper)
        half_range = np.maximum(1e-6, 0.5 * (solver_upper - solver_lower))
        normalized_offset = np.abs((np.asarray(q_feedback, dtype=float) - midpoint) / half_range)
        center_activation = np.square(
            np.clip((normalized_offset - center_deadband) / max(1e-6, 1.0 - center_deadband), 0.0, 1.0)
        )
        center_task = None
        if center_cost > 0.0 and bool(np.any(center_activation > 0.0)):
            center_task = self.PostureTask(cost=center_cost * center_activation)
            center_task.set_target(self._q(midpoint.tolist()))
        acceleration_limit = self.AccelerationLimit(self.model, np.asarray(acceleration_limits, dtype=float))
        # This is deliberately the final CAN velocity from the previous cycle,
        # never the raw Pink proposal.
        acceleration_limit.set_last_integration(np.asarray(last_sent, dtype=float), dt)
        limits = [
            self.ConfigurationLimit(self.model),
            self.VelocityLimit(self.model),
            acceleration_limit,
        ]
        tasks = [frame_task, posture_task, damping_task]
        if center_task is not None:
            tasks.append(center_task)
        dq = np.asarray(self.pink.solve_ik(configuration, tasks, dt, solver="quadprog", limits=limits), dtype=float)[:7]
        tcp_twist = np.asarray(jac, dtype=float) @ dq
        jacobian_rank = int(np.linalg.matrix_rank(np.asarray(jac, dtype=float)))
        nullspace_velocity = dq - np.linalg.pinv(np.asarray(jac, dtype=float)) @ tcp_twist
        posture_error = np.asarray(q_feedback, dtype=float) - np.asarray(posture_reference, dtype=float)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": True,
            "pink_joint_velocity_rad_s": dq.tolist(),
            "tcp": self.fk(q_feedback),
            # This FK uses the exact CAN-feedback joint vector attached to the
            # control sample.  It is telemetry only: Pink still solves from
            # q_feedback, the delay-compensated control estimate.
            "measured_tcp": self.fk(measured_joint_angles),
            "condition_number": condition,
            "jacobian_rank": jacobian_rank,
            "joint_velocity_norm": float(np.linalg.norm(dq)),
            "tcp_twist_norm": float(np.linalg.norm(tcp_twist)),
            "nullspace_velocity_norm": float(np.linalg.norm(nullspace_velocity)),
            "posture_error_norm": float(np.linalg.norm(posture_error)),
            "damping_cost": damping_cost,
            "frame_gain": frame_gain,
            "frame_lm_damping": frame_lm_damping,
            "position_cost": position_cost,
            "posture_cost": posture_cost,
            "joint_center_cost": center_cost,
            "joint_center_deadband": center_deadband,
            "joint_center_activation": center_activation.tolist(),
            "joint_acceleration_limit_rad_s2": acceleration_limits,
            "solver": "pinocchio+pink",
            "target_pose": {
                "position_m": command_position,
                "orientation_xyzw": command_quaternion,
            },
            "solver_target_pose": {"position_m": target_position, "orientation_xyzw": target_quaternion},
            "position_error_m": position_error,
            "orientation_error_rad": rotation_error,
            "posture_reference_rad": posture_reference,
            "orientation_mode": "fixed_weight",
            "orientation_cost": orientation_cost,
            "feedback_limit_adjusted": bool(np.any(solver_lower < np.asarray(soft_lower)) or np.any(solver_upper > np.asarray(soft_upper))),
            "timing_ms": {"pink": elapsed_ms},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--tcp-offset", default="0.175,0,-0.0235")
    parser.add_argument("--period-s", type=float, default=0.02)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        solver = Solver(args.urdf, [float(x) for x in args.tcp_offset.split(",")], args.period_s)
        if args.self_test:
            print(json.dumps({"ok": True, "solver": "pinocchio+pink", "nq": solver.model.nq}, ensure_ascii=False), flush=True)
            return 0
        print(json.dumps({"ready": True, "solver": "pinocchio+pink", "period_s": args.period_s}, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 2
    incoming: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
    closed = threading.Event()

    def receive() -> None:
        for line in sys.stdin:
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                # Queue.Queue.empty() is not a synchronization primitive on
                # Windows.  Drain with get_nowait(), then retry the put so a
                # producer/consumer race cannot silently discard every new
                # request while the solver is working on the previous one.
                try:
                    incoming.get_nowait()
                except queue.Empty:
                    pass
                try:
                    incoming.put_nowait(payload)
                except queue.Full:
                    try:
                        incoming.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        incoming.put_nowait(payload)
                    except queue.Full:
                        pass
            except Exception:
                continue
        closed.set()

    threading.Thread(target=receive, name="osc-solver-input", daemon=True).start()
    latest: dict[str, Any] | None = None
    while True:
        # Process the newest state as soon as it arrives. A fixed-rate solver
        # tick adds up to one full control period of q-state age, which makes
        # a velocity proposal computed from q[k] act on q[k+1].
        while True:
            try:
                latest = incoming.get(timeout=args.period_s)
                while True:
                    try:
                        latest = incoming.get_nowait()
                    except queue.Empty:
                        break
                break
            except queue.Empty:
                if closed.is_set():
                    return 0
                continue
        if latest is None:
            if closed.is_set():
                break
            continue
        started_ns = time.monotonic_ns()
        try:
            if latest.get("kind") == "fk":
                response = {"ok": True, "kind": "fk", "tcp": solver.fk(finite_vector(latest.get("joint_angles_rad"), 7, "joint_angles_rad"))}
            else:
                response = solver.solve(dict(latest))
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        response.update({
            "input_sequence": latest.get("sequence"),
            "solver_request_id": latest.get("solver_request_id"),
            "control_sample_id": latest.get("control_sample_id"),
            "target_generation": latest.get("target_generation"),
            "joint_state_rad": latest.get("joint_angles_rad"),
            "joint_state_monotonic_ns": latest.get("joint_state_monotonic_ns"),
            "motion_epoch": latest.get("motion_epoch"),
            "osc_pink_requested_monotonic_ns": latest.get("osc_pink_requested_monotonic_ns"),
            "osc_pink_written_monotonic_ns": latest.get("osc_pink_written_monotonic_ns"),
            "solver_monotonic_ns": started_ns,
            "solver_finished_monotonic_ns": time.monotonic_ns(),
        })
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
        if closed.is_set():
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
