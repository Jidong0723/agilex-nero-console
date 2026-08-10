from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from motion.teleop import KinematicsClient, TeleopController
from tests.test_teleop import FakeBroker, FakeSolver


class PoseTeleopTests(unittest.TestCase):
    @staticmethod
    def config(mapping_verified: bool = False) -> dict:
        return {
            "solver": {"dt_s": 0.02, "ruckig_max_acceleration": 2.0, "ruckig_max_jerk": 20.0, "urdf": str(Path(__file__).resolve().parents[1] / "vendor/nero_description/nero_description.urdf")},
            "runtime": {"control_hz": 50, "max_control_hz": 100},
            "limits": {"joint_speed_rad_s": 0.45, "deadman_timeout_s": 1.0, "feedback_soft_stale_s": 1.0, "feedback_hard_stale_s": 2.0, "solver_stale_s": 1.0, "max_stale_velocity_repeats": 3, "workspace_min_m": [-0.45, -0.45, -0.01], "workspace_max_m": [0.45, 0.60, 0.70], "min_tcp_z_m": -0.01},
            "pose_input": {"mapping_verified": mapping_verified, "position_axis_map": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "orientation_axis_map": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        }

    def _controller(self) -> tuple[TeleopController, FakeSolver]:
        broker = FakeBroker()
        controller = TeleopController(broker, Path(tempfile.mkdtemp()), self.config())
        solver = FakeSolver()
        controller.solver = solver  # type: ignore[assignment]
        controller.start_session("shadow", client_id="browser-a")
        return controller, solver

    def _close(self, controller: TeleopController) -> None:
        controller.stop_event.set()
        if controller.thread:
            controller.thread.join(timeout=1.0)

    def test_clutch_pose_updates_reference_without_velocity_payload(self) -> None:
        controller, _solver = self._controller()
        try:
            started = controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 1, "event": "clutch_begin", "pose_scale": 1.0})
            anchor = started["anchor_id"]
            result = controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 2, "event": "pose", "anchor_id": anchor, "pose_scale": 1.0, "relative_pose": {"position_m": [0.02, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}})
            self.assertTrue(result["accepted"])
            self.assertAlmostEqual(controller.reference_pose["position_m"][0], 0.02)
            self.assertTrue(controller.clutch_active)
        finally:
            self._close(controller)

    def test_repeated_pose_keeps_reference_revision_and_new_pose_increments_it(self) -> None:
        controller, _solver = self._controller()
        try:
            started = controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 1, "event": "clutch_begin", "pose_scale": 1.0})
            anchor = started["anchor_id"]
            pose = {"position_m": [0.02, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 2, "event": "pose", "anchor_id": anchor, "pose_scale": 1.0, "relative_pose": pose})
            first_revision = controller.reference_revision
            controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 3, "event": "pose", "anchor_id": anchor, "pose_scale": 1.0, "relative_pose": pose})
            self.assertEqual(controller.reference_revision, first_revision)
            changed = dict(pose)
            changed["position_m"] = [0.03, 0.0, 0.0]
            controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 4, "event": "pose", "anchor_id": anchor, "pose_scale": 1.0, "relative_pose": changed})
            self.assertEqual(controller.reference_revision, first_revision + 1)
        finally:
            self._close(controller)

    def test_ruckig_state_is_reused_between_cycles(self) -> None:
        controller, _solver = self._controller()
        try:
            controller._initialize_ruckig([0.0] * 7, 0.02)
            ruckig_id = id(controller.ruckig_otg)
            controller._advance_ruckig([0.1] * 7, 0.02)
            controller._advance_ruckig([0.0] * 7, 0.02)
            self.assertEqual(id(controller.ruckig_otg), ruckig_id)
            self.assertEqual(controller.ruckig_period_s, 0.02)
        finally:
            self._close(controller)

    def test_active_osc_session_reports_an_expired_heartbeat_lease(self) -> None:
        controller, _solver = self._controller()
        try:
            controller.config["session_timeout_s"] = 0.01
            controller._heartbeat_monotonic_ns = time.monotonic_ns() - 1_000_000_000
            self.assertTrue(controller.heartbeat_expired())
            controller._heartbeat_monotonic_ns = time.monotonic_ns()
            self.assertFalse(controller.heartbeat_expired())
        finally:
            self._close(controller)

    def test_absolute_osc_target_resumes_without_clutch_state(self) -> None:
        controller, solver = self._controller()
        try:
            controller._initialize_ruckig([0.0] * 7, 0.02)
            controller.shadow_joints = [0.0] * 7
            controller.posture_reference = [0.0] * 7
            controller.session = {
                "state": "ACTIVE", "session_id": "osc-session", "client_id": "osc-client",
                "mode": "shadow", "execution_mode": "shadow", "sequence": 0,
            }
            controller.trajectory_state = "HOLD_READY"
            result = controller.submit_absolute_target({
                "session_id": "osc-session", "client_id": "osc-client", "sequence": 1,
                "payload": {"target_pose": {"position_m": [0.0, 0.3, 0.2], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
            }, mode="track_tcp")
            self.assertTrue(result["accepted"])
            self.assertEqual(controller.trajectory_state, "RUNNING")
            self.assertTrue(controller.absolute_target_active)
            self.assertFalse(controller.clutch_active)
            self.assertTrue(controller.intent["persistent"])
            self.assertEqual(controller.intent["reference_pose"]["position_m"], [0.0, 0.3, 0.2])
            self.assertEqual(solver.calls, [])
        finally:
            self._close(controller)

    def test_solver_result_from_previous_pose_revision_is_usable_within_anchor(self) -> None:
        client = KinematicsClient(Path(__file__).resolve().parents[1], self.config())
        client._solver_request_id = 3
        client.responses.put({"ok": True, "motion_epoch": 4, "anchor_id": 9, "reference_revision": 2, "solver_request_id": 2})
        result = client.poll(4, anchor_id=9, reference_revision=3)
        self.assertIsNotNone(result)
        client.responses.put({"ok": True, "motion_epoch": 4, "anchor_id": 8, "reference_revision": 3, "solver_request_id": 3})
        self.assertIsNone(client.poll(4, anchor_id=9, reference_revision=4))

    def test_release_invalidates_anchor_and_brakes(self) -> None:
        controller, _solver = self._controller()
        try:
            start = controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 1, "event": "clutch_begin", "pose_scale": 1.0})
            controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 2, "event": "clutch_release", "anchor_id": start["anchor_id"], "pose_scale": 1.0})
            self.assertFalse(controller.clutch_active)
            self.assertEqual(controller.trajectory_state, "BRAKING")
            with self.assertRaises(PermissionError):
                controller.submit_intent({"session_id": controller.session["session_id"], "client_id": "browser-a", "sequence": 3, "event": "pose", "anchor_id": start["anchor_id"], "pose_scale": 1.0, "relative_pose": {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}})
        finally:
            self._close(controller)

    def test_unverified_pico_mapping_blocks_hardware_mode(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config(mapping_verified=False))
            controller.solver = FakeSolver()  # type: ignore[assignment]
            with self.assertRaises(PermissionError):
                controller.start_session("pico_hardware", client_id="browser-a")

    def test_verified_pico_mapping_allows_session_start(self) -> None:
        broker = FakeBroker()
        with tempfile.TemporaryDirectory() as temp:
            controller = TeleopController(broker, Path(temp), self.config(mapping_verified=True))
            controller.solver = FakeSolver()  # type: ignore[assignment]
            result = controller.start_session("pico_hardware", client_id="pico-app")
            try:
                self.assertEqual(result["session"]["mode"], "pico_hardware")
                self.assertEqual(result["session"]["client_id"], "pico-app")
                self.assertTrue(result["input_enabled"])
            finally:
                self._close(controller)

    def test_pico_shadow_is_allowed_before_mapping_verification(self) -> None:
        controller = TeleopController(FakeBroker(), Path(tempfile.mkdtemp()), self.config(mapping_verified=False))
        controller.solver = FakeSolver()  # type: ignore[assignment]
        result = controller.start_session("pico_shadow", client_id="pico-app")
        try:
            self.assertEqual(result["session"]["execution_mode"], "shadow")
            self.assertEqual(result["session"]["input_source"], "pico")
            self.assertEqual(result["session"]["mode"], "pico_shadow")
        finally:
            self._close(controller)


if __name__ == "__main__":
    unittest.main()
