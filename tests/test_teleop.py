from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from motion.teleop import JointConvention, JointLimitAuthority, SafetySupervisor, ShadowCpvPlant, TeleopController
from supervisor.authority import (
    ArmWriter, CommandRevoked, ControlSupervisor, HardwareTxOwner, ServoMode,
    ServoWriteRevoked,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeSolver:
    """Immediate newest-result solver used to test the service-side stream."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.process = None
        self.python = Path("fake-solver")
        self.last: dict[str, Any] | None = None

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def discard_before_epoch(self, epoch: int) -> None:
        return None

    def submit(self, payload: dict[str, Any]) -> None:
        self.calls.append(dict(payload))
        q = list(payload["joint_angles_rad"])
        moving = list(payload.get("target_position_m", [0.0, 0.0, 0.0])) != [0.0, 0.0, 0.0]
        self.last = {
            "ok": True,
            "input_sequence": payload.get("sequence"),
            "motion_epoch": payload.get("motion_epoch"),
            "solver_monotonic_ns": time.monotonic_ns(),
            "pink_joint_velocity_rad_s": [0.05 if moving else 0.0] * 7,
            "tcp": {"position_m": [0.0, 0.3, 0.2], "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        }

    def fk(self, joints: list[float]) -> dict[str, Any]:
        del joints
        return {"position_m": [0.0, 0.3, 0.2], "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}

    def poll(self, epoch: int | None = None, anchor_id: int | None = None, reference_revision: int | None = None) -> dict[str, Any] | None:
        del anchor_id, reference_revision
        return dict(self.last) if self.last else None


class FailingStartSolver(FakeSolver):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def start(self) -> None:
        raise RuntimeError("solver boot failed")

    def close(self) -> None:
        self.closed = True


class FakeRobot:
    def __init__(self) -> None:
        self.velocities: list[list[float]] = []
        self.positions: list[list[float]] = []
        self.controller: TeleopController | None = None
        self.stream_active = False

    def read_state(self, include_motor_states: bool = False):
        motors = [{"velocity": 0.0} for _ in range(7)] if include_motor_states else []
        return SimpleNamespace(joint_angles_rad=[0.0] * 7, motor_states=motors)

    def send_cpv_position(self, joints: list[float]) -> dict[str, Any]:
        self.positions.append(list(joints))
        if len(self.positions) >= 4 and self.controller is not None:
            self.controller.stop_event.set()
        now = time.monotonic_ns()
        return {"motion_mode": "CPV_POSITION", "joint_target_rad": list(joints), "batch_duration_ms": 0.1, "batch_skew_ms": 0.02, "finished_monotonic_ns": now}

    def continuous_stream_active(self) -> bool:
        return self.stream_active

    def read_cpv_parameter(self, joint: int, name: str) -> float:
        return 2.0

    def get_control_state(self):
        return {"mode": "HOLD"}


class FakeBroker:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.writer = "SERVO"
        self.epoch = 0
        self.servo_tracking = False

    def _require_operational_control(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"control": {"robot": {"joint_angles_rad": [0.0] * 7}}}

    def prepare_teleop_hardware(self) -> dict[str, Any]:
        self.epoch += 1
        self.writer = "SERVO"
        return {"arm_writer": "SERVO", "servo_mode": "HOLDING", "control_epoch": self.epoch}

    def read_teleop_joint_limits(self) -> dict[str, Any]:
        return {"joints": [{"joint_index": index + 1, "angle_velocity": {"min_angle_limit": -2.0, "max_angle_limit": 2.0, "max_joint_spd": 0.45}, "acceleration": {"max_joint_acc": 2.0}} for index in range(7)]}

    def read_teleop_feedback(self) -> dict[str, Any]:
        return {"joint_angles_rad": [0.0] * 7, "joint_velocity_rad_s": [0.0] * 7, "timestamp_monotonic_ns": time.monotonic_ns()}

    def read_teleop_cpv_parameters(self) -> dict[str, Any]:
        return {"status": "available", "values": {"acc": [2.0] * 7}}

    def teleop_stream_active(self) -> bool:
        return self.robot.continuous_stream_active()

    def grant_teleop_tracking(self, session_id: str, epoch: int) -> bool:
        del session_id
        self.servo_tracking = self.writer == "SERVO" and epoch == self.epoch
        return self.servo_tracking

    def mark_teleop_stopping(self, session_id: str, epoch: int, reason: str) -> bool:
        del session_id, reason
        return self.writer == "SERVO" and epoch == self.epoch

    def latch_teleop_hold(self, reason: str) -> dict[str, Any]:
        return {"arm_writer": self.writer, "servo_mode": "HOLDING", "reason": reason}

    def servo_can_write(self, session_id: str, epoch: int) -> bool:
        del session_id
        return self.writer == "SERVO" and epoch == self.epoch

    def send_servo_position(self, joints: list[float], session_id: str, epoch: int) -> dict[str, Any]:
        if not self.servo_can_write(session_id, epoch):
            raise ServoWriteRevoked("servo writer revoked")
        return self.robot.send_cpv_position(joints)

    def trigger_safety_fault(self, reason: str) -> dict[str, Any]:
        self.writer = "SAFETY"
        return {"ok": True, "reason": reason}


class SafetyGateTests(unittest.TestCase):
    def test_velocity_clipping_is_permitted_and_reported(self) -> None:
        supervisor = SafetySupervisor({})
        limits = {
            "effective_lower_rad": [-1.0] * 7,
            "effective_upper_rad": [1.0] * 7,
            "controller_speed_rad_s": [0.45] * 7,
            "controller_acceleration_rad_s2": [2.0] * 7,
        }
        supervisor.configure(limits, [0.45] * 7, [2.0] * 7)

        safe, accepted, reason = supervisor.final_gate(
            [0.94] + [0.0] * 6,
            [0.0] * 7,
            [0.45] + [0.0] * 6,
            delay_s=0.1,
            feedback_hard_stale=False,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "velocity clipped by final safety gate")
        self.assertGreaterEqual(safe[0], 0.0)
        self.assertLess(safe[0], 0.45)

    def test_feedback_staleness_remains_a_hard_rejection(self) -> None:
        supervisor = SafetySupervisor({})
        limits = {
            "effective_lower_rad": [-1.0] * 7,
            "effective_upper_rad": [1.0] * 7,
            "controller_speed_rad_s": [0.45] * 7,
            "controller_acceleration_rad_s2": [2.0] * 7,
        }
        supervisor.configure(limits, [0.45] * 7, [2.0] * 7)

        safe, accepted, reason = supervisor.final_gate(
            [0.0] * 7, [0.0] * 7, [0.1] * 7, delay_s=0.1, feedback_hard_stale=True
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "feedback hard stale")
        self.assertEqual(safe, [0.0] * 7)

    def test_small_feedback_limit_excursion_clips_outward_motion_without_faulting(self) -> None:
        supervisor = SafetySupervisor({"feedback_limit_tolerance_rad": 0.05})
        limits = {
            "effective_lower_rad": [-1.0] * 7,
            "effective_upper_rad": [1.0] * 7,
            "controller_speed_rad_s": [0.45] * 7,
            "controller_acceleration_rad_s2": [2.0] * 7,
        }
        supervisor.configure(limits, [0.45] * 7, [2.0] * 7)

        safe, accepted, reason = supervisor.final_gate(
            [1.02] + [0.0] * 6,
            [0.0] * 7,
            [0.1] + [0.0] * 6,
            delay_s=0.1,
            feedback_hard_stale=False,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "velocity clipped by final safety gate")
        self.assertEqual(safe[0], 0.0)


class TeleopPositionDispatchTests(unittest.TestCase):
    def test_hardware_loop_can_send_direct_pink_joint_positions(self) -> None:
        config = {
            "solver": {"dt_s": 0.02, "direct_pink_cpv_position": True, "ruckig_max_acceleration": 2.0, "ruckig_max_jerk": 20.0, "urdf": str(ROOT / "vendor/nero_description/nero_description.urdf")},
            "runtime": {"control_hz": 50, "max_control_hz": 100},
            "limits": {"joint_speed_rad_s": 0.45, "input_filter_alpha": 1.0, "deadman_timeout_s": 1.0, "feedback_soft_stale_s": 1.0, "feedback_hard_stale_s": 2.0, "solver_stale_s": 1.0, "max_stale_velocity_repeats": 3},
        }
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), config)
            solver = FakeSolver()
            controller.solver = solver  # type: ignore[assignment]
            broker.robot.controller = controller
            authority = {"status": "shadow", "effective_lower_rad": controller.authority.hard_lower, "effective_upper_rad": controller.authority.hard_upper, "controller_speed_rad_s": [0.45] * 7, "controller_acceleration_rad_s2": [2.0] * 7}
            controller.authority.effective = authority
            controller.supervisor.configure(authority, [0.45] * 7, [2.0] * 7)
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.posture_reference = [0.0] * 7
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "hardware", "execution_mode": "hardware", "sequence": 1}
            controller.trajectory_state = "RUNNING"
            controller.clutch_active = True
            controller.intent = {"sequence": 1, "host_monotonic_ns": time.monotonic_ns(), "reference_pose": {"position_m": [0.1, 0.3, 0.2], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}, "anchor_id": 1, "reference_revision": 1}
            controller._feedback = {"joints": [0.0] * 7, "velocities": [0.0] * 7, "monotonic_ns": time.monotonic_ns()}
            controller._loop()
        self.assertGreaterEqual(len(broker.robot.positions), 4)
        self.assertEqual(broker.robot.velocities, [])
        self.assertTrue(all(len(target) == 7 for target in broker.robot.positions))
        self.assertGreater(max(abs(value) for value in broker.robot.positions[-1]), 0.0)
        self.assertEqual(controller.status()["last_result"]["reason"], "CPV joint-position batch sent")
        self.assertEqual(controller.status()["last_result"]["ruckig"]["mode"], "bypassed")


class ShadowCpvPlantTests(unittest.TestCase):
    def test_delays_feedback_and_respects_acceleration(self) -> None:
        plant = ShadowCpvPlant({"dispatch_delay_s": 0.02, "feedback_delay_s": 0.04,
                                "position_time_constant_s": 0.01, "max_joint_speed_rad_s": 2.0,
                                "max_joint_acceleration_rad_s2": 5.0}, [0.0] * 7)
        plant.dispatch([1.0] * 7, 0.0)
        q0, qd0, _ = plant.advance(0.02, 0.0)
        self.assertEqual(q0, [0.0] * 7)
        self.assertEqual(qd0, [0.0] * 7)
        plant.advance(0.02, 0.02)
        q, qd, _ = plant.advance(0.02, 0.06)
        self.assertGreater(q[0], 0.0)
        self.assertLessEqual(qd[0], 5.0 * 0.02 + 1e-9)
        self.assertEqual(plant.diagnostics()["output_count"], 1)


@unittest.skip("Superseded by pose-clutch coverage in tests.test_pose_teleop")
class TeleopVelocityStreamTests(unittest.TestCase):
    @staticmethod
    def config() -> dict[str, Any]:
        return {
            "solver": {"dt_s": 0.02, "ruckig_max_acceleration": 2.0, "ruckig_max_jerk": 20.0, "urdf": str(ROOT / "vendor/nero_description/nero_description.urdf")},
            "runtime": {"control_hz": 50, "max_control_hz": 100},
            "limits": {
                "angular_speed_rad_s": 0.15,
                "joint_speed_rad_s": 0.45,
                "input_filter_alpha": 1.0,
                "deadman_timeout_s": 1.0,
                "feedback_soft_stale_s": 1.0,
                "feedback_hard_stale_s": 2.0,
                "solver_stale_s": 1.0,
                "max_stale_velocity_repeats": 3,
            },
        }

    def test_solver_start_failure_does_not_leave_active_session(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            solver = FailingStartSolver()
            controller.solver = solver  # type: ignore[assignment]
            with self.assertRaises(RuntimeError):
                controller.start_session("shadow")
            status = controller.status()
        self.assertTrue(solver.closed)
        self.assertEqual(status["session"]["state"], "IDLE")
        self.assertIn("teleop start failed", status["last_result"]["reason"])

    def test_requested_hardware_mode_requires_stopping_active_shadow_session(self) -> None:
        """Mode changes never hide an implicit session replacement."""
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            controller.session = {"state": "ACTIVE", "session_id": "shadow", "client_id": "anonymous", "mode": "shadow", "sequence": 4}
            controller._heartbeat_monotonic_ns = time.monotonic_ns()
            with self.assertRaises(RuntimeError):
                controller.start_session("hardware", confirm_hardware=True)

    def test_shadow_session_starts_without_hardware_feedback(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            result = controller.start_session("shadow", client_id="browser-a")
            try:
                self.assertEqual(result["session"]["state"], "ACTIVE")
                self.assertEqual(result["session"]["client_id"], "browser-a")
                self.assertTrue(result["input_enabled"])
                self.assertNotIn("servo_mode", result)
            finally:
                controller.stop_event.set()
                controller.thread.join(timeout=1.0)
                controller.solver.close()

    def test_hardware_session_starts_without_confirmation_flag(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            result = controller.start_session("hardware", confirm_hardware=False, client_id="browser-a")
            try:
                self.assertEqual(result["session"]["state"], "ACTIVE")
                self.assertEqual(result["session"]["mode"], "hardware")
                self.assertTrue(result["input_enabled"])
                self.assertNotIn("servo_mode", result)
                self.assertTrue(controller.trajectory_state == "RUNNING")
            finally:
                controller.stop_event.set()
                if controller.thread:
                    controller.thread.join(timeout=1.0)
                if controller._feedback_thread:
                    controller._feedback_stop.set()
                    controller._feedback_thread.join(timeout=1.0)
                controller.solver.close()

    def test_hardware_stream_uses_only_cpv_positions(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            solver = FakeSolver()
            controller.solver = solver  # type: ignore[assignment]
            broker.robot.controller = controller
            authority = {
                "status": "shadow", "effective_lower_rad": controller.authority.hard_lower,
                "effective_upper_rad": controller.authority.hard_upper,
                "controller_speed_rad_s": [0.45] * 7,
                "controller_acceleration_rad_s2": [2.0] * 7,
            }
            controller.authority.effective = authority
            controller.supervisor.configure(authority, [0.45] * 7, [2.0] * 7)
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.posture_reference = [0.0] * 7
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "hardware", "sequence": 1}
            controller.intent = {
                "sequence": 1,
                "host_monotonic_ns": time.monotonic_ns(),
                "tcp_velocity": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                "speed_scale": 1.0,
                "deadman": True,
            }
            controller._feedback = {"joints": [0.0] * 7, "velocities": [0.0] * 7, "monotonic_ns": time.monotonic_ns()}
            controller._loop()
        self.assertGreaterEqual(len(broker.robot.positions), 4)
        self.assertGreaterEqual(len(solver.calls), 4)
        self.assertTrue(all(len(value) == 7 for value in broker.robot.positions))
        self.assertEqual(controller.cpv_send_count, len(broker.robot.positions))

    def test_zero_target_does_not_activate_unstarted_controller(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            solver = FakeSolver()
            controller.solver = solver  # type: ignore[assignment]
            authority = {
                "status": "shadow", "effective_lower_rad": controller.authority.hard_lower,
                "effective_upper_rad": controller.authority.hard_upper,
                "controller_speed_rad_s": [0.45] * 7,
                "controller_acceleration_rad_s2": [2.0] * 7,
            }
            controller.authority.effective = authority
            controller.supervisor.configure(authority, [0.45] * 7, [2.0] * 7)
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.posture_reference = [0.0] * 7
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "hardware", "sequence": 1}
            controller.intent = {
                "sequence": 1,
                "host_monotonic_ns": time.monotonic_ns(),
                "tcp_velocity": [0.0] * 6,
                "speed_scale": 1.0,
                "deadman": True,
            }
            controller._feedback = {"joints": [0.0] * 7, "velocities": [0.0] * 7, "monotonic_ns": time.monotonic_ns()}
            timer = threading.Timer(0.06, controller.stop_event.set)
            timer.start()
            controller._loop()
            timer.cancel()
        self.assertFalse(controller.input_enabled)
        self.assertEqual(controller.trajectory_state, "HOLD_READY")
        self.assertFalse(broker.servo_tracking)

    def test_deadman_release_discards_old_solver_velocity(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            authority = {
                "status": "shadow", "effective_lower_rad": controller.authority.hard_lower,
                "effective_upper_rad": controller.authority.hard_upper,
                "controller_speed_rad_s": [0.45] * 7,
                "controller_acceleration_rad_s2": [2.0] * 7,
            }
            controller.authority.effective = authority
            controller.supervisor.configure(authority, [0.45] * 7, [2.0] * 7)
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "shadow", "sequence": 1}
            controller.intent = {
                "sequence": 1, "host_monotonic_ns": time.monotonic_ns(),
                "tcp_velocity": [0.2, 0, 0, 0, 0, 0], "speed_scale": 1.0, "deadman": True,
            }

            class NoResultSolver(FakeSolver):
                def submit(self, payload: dict[str, Any]) -> None:
                    self.calls.append(dict(payload))

                def poll(self, epoch=None, anchor_id=None, reference_revision=None):
                    del anchor_id, reference_revision
                    return None

            controller.solver = NoResultSolver()  # type: ignore[assignment]
            controller.shadow_joints = [0.0] * 7
            timer = threading.Timer(0.13, controller.stop_event.set)
            timer.start()
            controller._loop()
            timer.cancel()
        self.assertEqual(controller.trajectory_state, "HOLD_READY")
        self.assertEqual(controller.last_sent_velocity, [0.0] * 7)

    def test_deadman_release_keeps_input_permission_for_next_resume(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.session = {
                "state": "ACTIVE", "session_id": "test",
                "client_id": "browser-a", "mode": "shadow", "sequence": 1,
            }
            controller.input_enabled = True
            controller._invalidate_motion("deadman released")
        self.assertTrue(controller.input_enabled)
        self.assertEqual(controller.trajectory_state, "BRAKING")

    def test_soft_stale_feedback_derates_without_braking(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            config = self.config()
            config["limits"]["feedback_soft_stale_s"] = 0.05
            config["limits"]["feedback_hard_stale_s"] = 0.5
            controller = TeleopController(broker, Path(temp), config)
            controller.solver = FakeSolver()  # type: ignore[assignment]
            broker.robot.controller = controller
            authority = {
                "status": "shadow", "effective_lower_rad": controller.authority.hard_lower,
                "effective_upper_rad": controller.authority.hard_upper,
                "controller_speed_rad_s": [0.45] * 7,
                "controller_acceleration_rad_s2": [2.0] * 7,
            }
            controller.authority.effective = authority
            controller.supervisor.configure(authority, [0.45] * 7, [2.0] * 7)
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.posture_reference = [0.0] * 7
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "hardware", "sequence": 1}
            controller.trajectory_state = "RUNNING"
            controller.input_enabled = True
            controller.intent = {"sequence": 1, "host_monotonic_ns": time.monotonic_ns(), "tcp_velocity": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0], "speed_scale": 1.0, "deadman": True}
            controller._feedback = {"joints": [0.0] * 7, "velocities": [0.0] * 7, "monotonic_ns": time.monotonic_ns() - 250_000_000}
            timer = threading.Timer(0.06, controller.stop_event.set)
            timer.start()
            controller._loop()
            timer.cancel()
        self.assertEqual(controller.trajectory_state, "RUNNING")
        self.assertLess(controller.last_result["supervisor"]["feedback_velocity_scale"], 1.0)

    def test_intent_heartbeat_response_does_not_embed_full_status(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.shadow_joints = [0.0] * 7
            controller.session = {"state": "ACTIVE", "session_id": "test", "client_id": "anonymous", "mode": "shadow", "sequence": 2}
            response = controller.submit_intent({
                "sequence": 3,
                "tcp_velocity": [0.0] * 6,
                "speed_scale": 1.0,
                "deadman": True,
            })
        self.assertEqual(response["accepted_sequence"], 3)
        self.assertNotIn("last_result", response)
        self.assertNotIn("diagnostics", response)

    def test_hold_ready_resumes_shadow_on_fresh_deadman(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller._initialize_ruckig([0.1] * 7, 0.02)
            controller.shadow_joints = [0.1] * 7
            controller.session = {
                "state": "ACTIVE", "session_id": "shadow-session",
                "client_id": "browser-a", "mode": "shadow", "sequence": 4,
            }
            controller.input_enabled = False
            controller.trajectory_state = "HOLD_READY"
            controller.feedback_sync_pending = True
            controller.needs_resync = True
            response = controller.submit_intent({
                "client_id": "browser-a", "session_id": "shadow-session",
                "sequence": 5, "tcp_velocity": [0.0] * 6,
                "speed_scale": 1.0, "deadman": True,
            })
            status = controller.status()
        self.assertTrue(response["accepted"])
        self.assertEqual(status["diagnostics"]["trajectory_state"], "RUNNING")
        self.assertTrue(status["input_enabled"])
        self.assertFalse(status["diagnostics"]["needs_resync"])
        self.assertEqual(controller.trajectory["position_rad"], [0.1] * 7)

    def test_hold_ready_resumes_hardware_only_with_fresh_servo_authority(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.session = {
                "state": "ACTIVE", "session_id": "hardware-session",
                "client_id": "browser-a", "mode": "hardware", "sequence": 7,
            }
            controller.input_enabled = False
            controller.trajectory_state = "HOLD_READY"
            controller.feedback_sync_pending = True
            controller.needs_resync = True
            controller.motion_epoch = 0
            controller._feedback = {
                "joints": [0.2] * 7, "velocities": [0.0] * 7,
                "monotonic_ns": time.monotonic_ns(),
            }
            response = controller.submit_intent({
                "client_id": "browser-a", "session_id": "hardware-session",
                "sequence": 8, "tcp_velocity": [0.0] * 6,
                "speed_scale": 1.0, "deadman": True,
            })
            status = controller.status()
        self.assertTrue(response["accepted"])
        self.assertTrue(broker.servo_tracking)
        self.assertEqual(status["diagnostics"]["trajectory_state"], "RUNNING")
        self.assertTrue(status["input_enabled"])
        self.assertEqual(controller.trajectory["position_rad"], [0.2] * 7)

    def test_hold_ready_does_not_resume_with_stale_hardware_feedback(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.session = {
                "state": "ACTIVE", "session_id": "hardware-session",
                "client_id": "browser-a", "mode": "hardware", "sequence": 1,
            }
            controller.input_enabled = False
            controller.trajectory_state = "HOLD_READY"
            controller._feedback = {
                "joints": [0.0] * 7, "velocities": [0.0] * 7,
                "monotonic_ns": time.monotonic_ns() - 2_000_000_000,
            }
            with self.assertRaisesRegex(PermissionError, "stale"):
                controller.submit_intent({
                    "client_id": "browser-a", "session_id": "hardware-session",
                    "sequence": 2, "tcp_velocity": [0.0] * 6,
                    "speed_scale": 1.0, "deadman": True,
                })
        self.assertEqual(controller.trajectory_state, "HOLD_READY")
        self.assertFalse(controller.input_enabled)

    def test_hard_stale_feedback_while_hold_ready_is_not_a_fault(self) -> None:
        """Idle feedback is reported, while the next deadman input rechecks it."""
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.session = {
                "state": "ACTIVE", "session_id": "hardware-session",
                "client_id": "browser-a", "mode": "hardware", "sequence": 1,
            }
            controller.trajectory_state = "HOLD_READY"
            controller._feedback = {
                "joints": [0.0] * 7, "velocities": [0.0] * 7,
                "monotonic_ns": time.monotonic_ns() - 1_000_000_000,
            }
            timer = threading.Timer(0.04, controller.stop_event.set)
            timer.start()
            controller._loop()
            timer.cancel()
        self.assertEqual(controller.trajectory_state, "HOLD_READY")

    def test_intent_is_rejected_when_session_belongs_to_another_client(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.session = {
                "state": "ACTIVE", "session_id": "test",
                "client_id": "browser-a",
                "mode": "shadow",
                "sequence": 2,
            }
            with self.assertRaises(PermissionError):
                controller.submit_intent({
                    "client_id": "browser-b",
                    "sequence": 3,
                    "tcp_velocity": [0.0] * 6,
                    "speed_scale": 1.0,
                    "deadman": True,
                })

    def test_faulted_hardware_session_rejects_nonzero_intent(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.session = {
                "state": "ACTIVE", "session_id": "test",
                "client_id": "browser-a", "mode": "hardware", "sequence": 2,
            }
            controller.input_enabled = True
            controller.trajectory_state = "FAULT"
            controller.trajectory_brake_reason = "feedback hard stale"
            with self.assertRaisesRegex(PermissionError, "FAULT"):
                controller.submit_intent({
                    "client_id": "browser-a", "session_id": "test", "sequence": 3,
                    "tcp_velocity": [0.2, 0, 0, 0, 0, 0],
                    "speed_scale": 1.0, "deadman": True,
                })

    def test_inactive_hardware_session_does_not_reenter_cpv_to_send_zero(self) -> None:
        """A repeated handoff must not reinitialise CPV after HOLD is active."""
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            controller.session = {
                "state": "STOPPING", "session_id": "stale-hardware",
                "mode": "hardware",
                "sequence": 7,
            }
            result = controller.stop_session("duplicate console handoff")
        self.assertEqual(broker.robot.velocities, [])
        self.assertFalse(result["handoff"]["zero_velocity_sent"])

    def test_direct_freedrive_handoff_revokes_session_without_braking_or_zero_stream(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            controller.session = {
                "state": "ACTIVE", "session_id": "direct-freedrive",
                "client_id": "browser-a", "mode": "hardware", "sequence": 7,
            }
            controller.motion_epoch = 7
            controller.intent = {
                "sequence": 7, "tcp_velocity": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                "deadman": True,
            }
            result = controller.abandon_session_without_braking(8, "direct FREEDRIVE")

        self.assertIsNone(controller.session)
        self.assertIsNone(controller.intent)
        self.assertEqual(controller.motion_epoch, 8)
        self.assertEqual(controller.trajectory_state, "HOLD_READY")
        self.assertFalse(result["direct_handoff"]["braking_requested"])
        self.assertFalse(result["direct_handoff"]["zero_velocity_sent"])
        self.assertEqual(broker.robot.velocities, [])

    def test_revoked_servo_write_freezes_teleop_without_p0_safety_zero(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config())
            controller.solver = FakeSolver()  # type: ignore[assignment]
            controller.start_session("hardware", client_id="browser-a")
            broker.writer = "NONE"
            time.sleep(0.08)
            controller.stop_event.set()
            if controller.thread:
                controller.thread.join(timeout=1.0)
            if controller._feedback_thread:
                controller._feedback_stop.set()
                controller._feedback_thread.join(timeout=1.0)

        self.assertEqual(broker.writer, "NONE")
        self.assertEqual(controller.trajectory_state, "HOLD_READY")
        self.assertNotEqual(controller.trajectory_state, "FAULT")


class TransportOwnerEpochTests(unittest.TestCase):
    def test_epoch_promotion_drops_queued_old_p1_and_p2_but_runs_current_freedrive(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.entered = threading.Event()
                self.release = threading.Event()
                self.calls: list[str] = []

            def block(self) -> None:
                self.entered.set()
                self.release.wait(1.0)

            def cpv(self) -> None:
                self.calls.append("cpv")

            def gripper(self) -> None:
                self.calls.append("gripper")

            def freedrive(self) -> None:
                self.calls.append("freedrive")

        backend = Backend()
        owner = HardwareTxOwner(backend)
        blocker = threading.Thread(target=lambda: owner.call("p1", "block"))
        queued_errors: list[Exception] = []

        def queue_stale_cpv() -> None:
            try:
                owner.call("p1", "cpv", command_epoch=0, category="servo_velocity")
            except Exception as exc:
                queued_errors.append(exc)

        def queue_stale_gripper() -> None:
            try:
                owner.call("p2", "gripper", command_epoch=0, category="gripper")
            except Exception as exc:
                queued_errors.append(exc)

        stale = threading.Thread(target=queue_stale_cpv)
        stale_p2 = threading.Thread(target=queue_stale_gripper)
        promoted: list[int] = []
        try:
            blocker.start()
            self.assertTrue(backend.entered.wait(0.5))
            stale.start()
            stale_p2.start()
            time.sleep(0.03)
            promoter = threading.Thread(
                target=lambda: promoted.append(
                    owner.advance_epoch(1, exclusive_category="freedrive_transition")
                )
            )
            promoter.start()
            time.sleep(0.03)
            backend.release.set()
            blocker.join(timeout=1.0)
            stale.join(timeout=1.0)
            stale_p2.join(timeout=1.0)
            promoter.join(timeout=1.0)
            self.assertEqual(promoted, [1])
            self.assertEqual(backend.calls, [])
            self.assertEqual(len(queued_errors), 2)
            self.assertTrue(any(isinstance(error, ServoWriteRevoked) for error in queued_errors))
            self.assertTrue(any(type(error) is CommandRevoked for error in queued_errors))
            with self.assertRaises(CommandRevoked):
                owner.call("p2", "gripper", command_epoch=1, category="gripper")
            owner.call("p2", "freedrive", command_epoch=1, category="freedrive_transition")
            self.assertEqual(backend.calls, ["freedrive"])
            owner.complete_epoch_transition(1, "freedrive_transition")
            owner.call("p2", "gripper", command_epoch=1, category="gripper")
            self.assertEqual(backend.calls, ["freedrive", "gripper"])
        finally:
            backend.release.set()
            owner.close()


class JointLimitAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.urdf = ROOT / "vendor/nero_description/nero_description.urdf"

    def _authority(self, sign: float = 1.0) -> JointLimitAuthority:
        conventions = [JointConvention(index + 1, f"joint{index + 1}", sign if index == 0 else 1.0, 0.0, 1.0) for index in range(7)]
        return JointLimitAuthority(self.urdf, {"joint_conventions": [item.__dict__ for item in conventions]})

    @staticmethod
    def _controller_limits(lower: float = -2.0, upper: float = 2.0) -> dict[str, Any]:
        return {"joints": [{"joint_index": index + 1, "angle_velocity": {"min_angle_limit": lower, "max_angle_limit": upper, "max_joint_spd": 0.45}, "acceleration": {"max_joint_acc": 2.0}} for index in range(7)]}

    def test_controller_limits_are_converted_before_intersection(self) -> None:
        authority = self._authority(sign=-1.0)
        effective = authority.initialize(self._controller_limits(lower=-1.0, upper=2.0))
        # sign=-1 reverses the controller interval before canonical URDF intersection.
        self.assertAlmostEqual(effective["controller_raw"][0]["converted_lower_rad"], -2.0)
        self.assertAlmostEqual(effective["controller_raw"][0]["converted_upper_rad"], 1.0)
        self.assertLessEqual(effective["effective_lower_rad"][0], effective["effective_upper_rad"][0])

    def test_empty_limit_intersection_is_rejected(self) -> None:
        authority = self._authority()
        with self.assertRaisesRegex(ValueError, "empty"):
            authority.initialize(self._controller_limits(lower=20.0, upper=21.0))

    def test_supervisor_only_restricts_outward_velocity(self) -> None:
        authority = self._authority()
        limits = authority.initialize(self._controller_limits())
        supervisor = SafetySupervisor({"fixed_model_margin_rad": [0.01] * 7, "controller_stop_reserve_rad": [0.01] * 7, "controller_response_delay_s": 0.04})
        supervisor.configure(limits, [0.45] * 7, [2.0] * 7)
        near_upper = list(limits["effective_upper_rad"])
        near_upper[0] -= 0.021
        outward, _ = supervisor.limit_velocity(near_upper, [0.0] * 7, [0.3] + [0.0] * 6, 0.1)
        inward, _ = supervisor.limit_velocity(near_upper, [0.0] * 7, [-0.3] + [0.0] * 6, 0.1)
        self.assertLess(abs(outward[0]), 0.3)
        self.assertAlmostEqual(inward[0], -0.3)


class ControlAuthorityTests(unittest.TestCase):
    def test_servo_permission_requires_current_epoch_and_writer(self) -> None:
        authority = ControlSupervisor()
        state = authority.transition(ArmWriter.SERVO, ServoMode.HOLDING, "prepared", advance_epoch=True)
        self.assertTrue(authority.allows_servo("session", state.epoch))
        authority.transition(ArmWriter.MODE_TRANSITION, ServoMode.SUSPENDED, "leader transition", advance_epoch=True)
        self.assertFalse(authority.allows_servo("session", state.epoch))

    def test_safety_holding_retains_writer_authority(self) -> None:
        authority = ControlSupervisor()
        authority.transition(ArmWriter.SAFETY, ServoMode.STOPPING, "hard stale", advance_epoch=True)
        state = authority.transition(ArmWriter.SAFETY, ServoMode.HOLDING, "fault hold")
        self.assertEqual(state.writer, ArmWriter.SAFETY)
        self.assertEqual(state.servo_mode, ServoMode.HOLDING)


if __name__ == "__main__":
    unittest.main()
