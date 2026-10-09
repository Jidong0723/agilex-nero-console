from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from motion.osc import OscRuntime, _OperationalSpaceServo


ROOT = Path(__file__).resolve().parents[1]
POSE = {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


class DirectCpvTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.config = json.loads((ROOT / "config/osc.json").read_text(encoding="utf-8"))
        self.config["solver"]["urdf"] = str(ROOT / "vendor/nero_description/nero_description.urdf")
        self.config["solver"]["joint_acceleration_limit_rad_s2"] = 2.0
        self.config["state_estimator"]["enabled"] = False
        self.config["shadow_transport"]["enabled"] = False
        self.q = [0.1] * 7
        self.feedback_age = 0.0
        self.raw_velocity = [0.5] * 7
        self.fresh_solver = True
        self.hardware = Mock()
        self.hardware.servo_can_write.return_value = True
        self.hardware.servo_transport_diagnostics.return_value = {}
        self.hardware.publish_servo_position.return_value = {"mailbox_revision": 1}
        self.hardware.wait_for_servo_result.return_value = {"status": "sent"}
        with patch("supervisor.logging.AsyncJsonlTraceLogger"), patch.object(
            _OperationalSpaceServo, "_enable_high_resolution_timer"
        ):
            self.servo = _OperationalSpaceServo(
                self.hardware, Path(folder.name), self.config, Mock()
            )
        self.servo.solver = SimpleNamespace(solve_current=self.solve_current)
        authority = {
            "effective_lower_rad": [-1.0] * 7,
            "effective_upper_rad": [1.0] * 7,
            "controller_speed_rad_s": [1.0] * 7,
            "controller_acceleration_rad_s2": [2.0] * 7,
        }
        self.servo.supervisor.configure(authority, [1.0] * 7, [2.0] * 7)
        self.servo._initialize_trajectory(self.q)
        self.servo.shadow_joints = list(self.q)
        self.servo.session = {
            "state": "ACTIVE", "session_id": "test", "execution_mode": "hardware", "sequence": 1,
        }
        self.servo.command = {
            "sequence": 1, "host_monotonic_ns": time.monotonic_ns(),
            "target_pose": dict(POSE), "target_generation": 1,
        }
        self.servo.trajectory_state = "RUNNING"
        self.servo._feedback_snapshot = self.feedback
        # Stop after one complete cycle without depending on scheduler timing.
        set_result = self.servo._set_result

        def stop_after_result(*args, **kwargs):
            set_result(*args, **kwargs)
            self.servo.stop_event.set()

        self.servo._set_result = stop_after_result

    def feedback(self):
        return {
            "joints": list(self.q), "velocities": [0.0] * 7,
            "monotonic_ns": time.monotonic_ns() - int(self.feedback_age * 1e9),
        }

    def solve_current(self, request, timeout):
        if not self.fresh_solver:
            return None
        return {
            "ok": True, "pink_joint_velocity_rad_s": list(self.raw_velocity),
            "joint_state_monotonic_ns": request["joint_state_monotonic_ns"],
            "tcp": dict(POSE), "measured_tcp": dict(POSE),
            "target_generation": request["target_generation"],
            "position_error_m": 0.1, "orientation_error_rad": 0.1,
        }

    def step(self):
        self.servo.stop_event.clear()
        self.servo._loop()
        self.assertEqual(len(self.servo._cycle_trace), self.servo.output_count)
        return self.servo.last_output

    def test_hardware_integrates_acceleration_limited_velocity_once(self):
        result = self.step()
        self.assertTrue(self.servo.last_result["ok"], self.servo.last_result)
        for q, velocity in zip(result["final_joint_target_rad"], result["final_joint_velocity_rad_s"]):
            self.assertAlmostEqual(velocity, 0.04)
            self.assertAlmostEqual(q, 0.1008)
        published = self.hardware.publish_servo_position.call_args.args[0]
        self.assertEqual(published["joint_target_rad"], result["final_joint_target_rad"])
        self.assertEqual(published["max_joint_acceleration_rad_s2"], 2.0)

    def test_shipped_acceleration_reaches_solver_dispatch_and_profile_sync(self):
        shipped = json.loads((ROOT / "config/osc.json").read_text(encoding="utf-8"))
        acceleration = shipped["solver"]["joint_acceleration_limit_rad_s2"]
        self.assertEqual(acceleration, 10.0)
        self.assertEqual(shipped["hardware_limits"]["acceleration_rad_s2"], [10.0] * 7)
        self.servo.config["solver"]["joint_acceleration_limit_rad_s2"] = acceleration
        authority = self.servo.authority.initialize_fixed()
        self.servo.supervisor.configure(authority, [1.0] * 7, [acceleration] * 7)
        runtime = object.__new__(OscRuntime)
        runtime._servo = self.servo
        self.assertEqual(runtime.cpv_limits()[1], 10.0)
        solve = self.servo.solver.solve_current
        requests = []
        def capture(request, timeout):
            requests.append(request)
            return solve(request, timeout)
        self.servo.solver.solve_current = capture
        result = self.step()
        self.assertEqual(requests[0]["joint_acceleration_limit_rad_s2"], [10.0] * 7)
        published = self.hardware.publish_servo_position.call_args.args[0]
        self.assertEqual(published["max_joint_acceleration_rad_s2"], 10.0)
        for q, velocity in zip(result["final_joint_target_rad"], result["final_joint_velocity_rad_s"]):
            self.assertAlmostEqual(velocity, .2)
            self.assertAlmostEqual(q, .104)
        self.servo.session["execution_mode"] = "shadow"
        self.servo._initialize_trajectory(self.q)
        self.servo.shadow_joints = list(self.q)
        self.hardware.publish_servo_position.reset_mock()
        shadow_result = self.step()
        self.assertEqual(shadow_result["final_joint_velocity_rad_s"], result["final_joint_velocity_rad_s"])
        self.assertEqual(shadow_result["final_joint_target_rad"], result["final_joint_target_rad"])
        self.hardware.publish_servo_position.assert_not_called()

    def test_shadow_matches_hardware_command(self):
        hardware_result = dict(self.step())
        self.servo.session["execution_mode"] = "shadow"
        self.servo._initialize_trajectory(self.q)
        self.servo.shadow_joints = list(self.q)
        self.hardware.publish_servo_position.reset_mock()
        shadow_result = self.step()
        self.assertEqual(shadow_result["final_joint_target_rad"], hardware_result["final_joint_target_rad"])
        self.assertEqual(shadow_result["final_joint_velocity_rad_s"], hardware_result["final_joint_velocity_rad_s"])
        self.hardware.publish_servo_position.assert_not_called()

    def test_missing_solver_brakes_with_existing_acceleration_limit(self):
        self.fresh_solver = False
        self.servo.last_sent_velocity = [0.1] * 7
        self.servo.trajectory["velocity_rad_s"] = [0.1] * 7
        result = self.step()
        for velocity in result["final_joint_velocity_rad_s"]:
            self.assertAlmostEqual(velocity, 0.06)
        self.assertIn("acceleration-limited braking", self.servo.last_result["supervisor"]["reason"])

    def test_final_gate_clips_outward_velocity_after_acceleration_clamp(self):
        self.q[0] = 1.02
        self.servo.last_sent_velocity = [0.5] + [0.0] * 6
        self.servo.trajectory["velocity_rad_s"] = list(self.servo.last_sent_velocity)
        result = self.step()
        self.assertTrue(self.servo.last_result["ok"], self.servo.last_result)
        self.assertTrue(self.servo.last_result["gate_limited"])
        self.assertEqual(result["final_joint_velocity_rad_s"][0], 0.0)
        self.assertEqual(result["final_joint_target_rad"][0], self.q[0])

    def test_hard_stale_feedback_publishes_zero_and_faults(self):
        self.feedback_age = 0.2
        result = self.step()
        self.assertEqual(self.servo.trajectory_state, "FAULT")
        self.assertEqual(result["final_joint_velocity_rad_s"], [0.0] * 7)
        self.assertEqual(result["final_joint_target_rad"], self.q)
        self.assertFalse(self.hardware.publish_servo_position.call_args.args[0]["gate_ok"])

    def test_soft_stale_feedback_derates_without_fault(self):
        self.feedback_age = 0.1
        self.servo.last_sent_velocity = [0.5] * 7
        self.servo.trajectory["velocity_rad_s"] = [0.5] * 7
        self.step()
        self.assertEqual(self.servo.trajectory_state, "RUNNING")
        self.assertTrue(self.servo.last_result["ok"])
        self.assertLess(self.servo.last_result["supervisor"]["feedback_velocity_scale"], 1.0)
        self.assertLess(self.servo.last_sent_velocity[0], 0.5)

    def test_revoked_authority_freezes_without_publication(self):
        self.hardware.servo_can_write.return_value = False
        self.step()
        self.hardware.publish_servo_position.assert_not_called()
        self.assertEqual(self.servo.trajectory_state, "HOLD_READY")
        self.assertEqual(self.servo.trajectory["position_rad"], self.q)

    def test_braking_waits_for_final_publication_before_hold(self):
        self.servo.trajectory_state = "BRAKING"
        self.servo.command = None
        self.servo.last_sent_velocity = [0.1] * 7
        self.servo.trajectory["velocity_rad_s"] = [0.1] * 7
        for _ in range(5):
            self.step()
            if self.servo.trajectory_state == "HOLD_READY":
                break
        self.assertEqual(self.servo.trajectory_state, "HOLD_READY")
        self.assertEqual(self.servo.last_sent_velocity, [0.0] * 7)
        self.hardware.wait_for_servo_result.assert_called_once()
        self.hardware.latch_osc_hold.assert_called_once_with("osc braking settled")
        methods = [call[0] for call in self.hardware.mock_calls]
        self.assertLess(methods.index("wait_for_servo_result"), methods.index("latch_osc_hold"))


if __name__ == "__main__":
    unittest.main()
