from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from supervisor.control import (
    LeaseError,
    LeaseManager,
    OperationalSpaceController,
    absolute_pose_from_sdk_rpy,
)
RobotControlBroker = OperationalSpaceController
from shared.schemas import ExecutedAction, GripperState, now_iso


class FakeRobot:
    def __init__(self, log_dir: Path) -> None:
        self.config = {
            "control_service": {"lease_ttl_s": 5.0, "log_dir": str(log_dir)},
        }
        self.mode = "HOLD"
        self.freedrive_backend = None
        self.reason = "ready"
        self.preempted = threading.Event()
        self.epoch = 0
        self.hold_calls = 0
        self.follower_hold_calls = 0
        self.freedrive_calls = 0
        self.cpv_stop_calls = 0
        self.hold_should_fail = False
        self.continuous_stream_active = False
        self.events: list[str] = []
        self.execute_calls = 0
        self.estop_calls = 0
        self.cpv_parameter_reads: list[tuple[int, str]] = []
        self.gripper_hold = {
            "supported": False, "active": False, "mode": None,
            "target_width_m": None, "force_n": None, "alarm": None,
        }

    def connect(self) -> dict[str, Any]:
        return {"ok": True}

    def disconnect(self) -> dict[str, Any]:
        return {"ok": True}

    def preempt_epoch(self) -> int:
        return self.epoch

    def request_preempt(self, reason: str) -> None:
        self.epoch += 1
        self.reason = reason
        self.events.append("preempt")
        self.preempted.set()

    def hold_follower_without_position_target(self, reason: str) -> ExecutedAction:
        self.follower_hold_calls += 1
        self.events.append("follower_hold")
        if self.hold_should_fail:
            return ExecutedAction(now_iso(), {"type": "follower_hold"}, None, False, True, "simulated hold failure")
        self.mode = "HOLD"
        self.continuous_stream_active = False
        self.freedrive_backend = None
        return ExecutedAction(
            now_iso(), {"type": "follower_hold"},
            None, True, True, reason,
            result={"position_target_sent": False, "cpv_zero_confirmed": True},
        )

    def enter_freedrive(
        self, reason: str, recover_emergency: bool = False, preserve_gripper: bool = False
    ) -> ExecutedAction:
        self.freedrive_calls += 1
        self.events.append("freedrive")
        self.mode = "FREEDRIVE"
        self.freedrive_backend = "leader"
        return ExecutedAction(now_iso(), {"type": "freedrive"}, {"type": "freedrive"}, True, True, reason)

    def stop_cpv_for_mode_transition(self, reason: str) -> dict[str, Any]:
        self.cpv_stop_calls += 1
        self.events.append("cpv_stop")
        self.continuous_stream_active = False
        return {
            "ok": True,
            "cpv_was_active": True,
            "cpv_zero_confirmed": True,
            "post_cpv_dwell_s": 0.0,
            "position_target_sent": False,
        }

    def e_stop(self, reason: str) -> ExecutedAction:
        self.estop_calls += 1
        self.mode = "EMERGENCY_DAMPING"
        return ExecutedAction(now_iso(), {"type": "emergency_damping"}, {"type": "emergency_damping"}, True, True, reason)

    def get_control_state(self) -> dict[str, Any]:
        robot = None if getattr(self, "feedback_unavailable", False) else {
            "joint_angles_rad": [0.0] * 7,
            "tcp_pose": [0.0] * 6,
            "arm_status": {"arm_status": 0},
            "joint_enable_status": [True] * 7,
        }
        if getattr(self, "leader_only_feedback", False):
            robot = {
                "joint_angles_rad": [0.0] * 7,
                "tcp_pose": [0.0] * 6,
                "arm_status": None,
                "joint_enable_status": [False] * 7,
            }
        return {
            "mode": self.mode,
            "continuous_stream_active": self.continuous_stream_active,
            "freedrive_backend": self.freedrive_backend,
            "reason": self.reason,
            "connected": True,
            "robot": robot,
            "leader_feedback_age_s": 0.0 if getattr(self, "leader_only_feedback", False) else None,
            "leader_feedback_hz": 220.0 if getattr(self, "leader_only_feedback", False) else None,
        }

    def read_cpv_parameter(self, joint_index: int, name: str) -> float:
        self.cpv_parameter_reads.append((joint_index, name))
        return float(joint_index)

    def get_observation(self, include_motor_states: bool = False) -> Any:
        raise NotImplementedError

    def read_gripper(self) -> GripperState:
        return GripperState(now_iso(), 0.02, 1.0, "width", {
            "foc_status": {"driver_enable_status": True}
        })

    def get_gripper_hold_state(self) -> dict[str, Any]:
        return dict(self.gripper_hold)

    def monitor_gripper_hold(self) -> None:
        return None

    def command_gripper(
        self, mode: str, width_m: float | None, force_n: float, preserve_on_freedrive: bool
    ) -> dict[str, Any]:
        return {"ok": True, "mode": mode, "gripper_hold": dict(self.gripper_hold)}

    def clear_gripper_hold(self) -> dict[str, Any]:
        self.gripper_hold.update({
            "active": False, "mode": None, "target_width_m": None,
            "force_n": None, "alarm": None,
        })
        return dict(self.gripper_hold)

    def release_gripper_zero_force(self) -> dict[str, Any]:
        hold = self.clear_gripper_hold()
        return {"ok": True, "zero_force": True, "gripper_hold": hold}


class LeaseManagerTests(unittest.TestCase):
    def test_only_one_motion_owner(self) -> None:
        leases = LeaseManager()
        first = leases.acquire("script-a")
        with self.assertRaises(LeaseError):
            leases.acquire("script-b")
        leases.release(first.token)
        self.assertEqual(leases.acquire("script-b").owner, "script-b")

    def test_wrong_owner_cannot_release(self) -> None:
        leases = LeaseManager()
        leases.acquire("script-a")
        with self.assertRaises(LeaseError):
            leases.release("wrong-token")


class BrokerPreemptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fake = FakeRobot(Path(self.temp.name))
        self.hardware = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]
        self.hardware.start()

    def tearDown(self) -> None:
        self.hardware.close()
        self.temp.cleanup()

    def test_async_operator_job_is_single_flight_and_queryable(self) -> None:
        first = self.hardware.submit_action_job({"kind": "hold", "reason": "test"})
        second = self.hardware.submit_action_job({"kind": "hold", "reason": "duplicate"})
        self.assertEqual(first["action_id"], second["action_id"])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = self.hardware.action_status(first["action_id"])
            if state["status"] not in {"queued", "running", "already_running"}:
                break
            time.sleep(0.01)
        self.assertEqual(self.hardware.action_status(first["action_id"])["status"], "completed")

    def test_cpv_parameter_snapshot_is_read_only_and_serialized(self) -> None:
        snapshot = self.hardware.read_osc_cpv_parameters()
        self.assertEqual(snapshot["status"], "available")
        self.assertEqual(len(self.fake.cpv_parameter_reads), 42)
        self.assertEqual(snapshot["values"]["acc"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(self.fake.execute_calls, 0)

    def test_start_remains_running_when_usb_can_connect_fails(self) -> None:
        self.hardware.close()
        self.fake.connect = lambda: (_ for _ in ()).throw(RuntimeError("USB-CAN channel 0 unavailable"))  # type: ignore[method-assign]
        self.fake.get_control_state = lambda: (_ for _ in ()).throw(RuntimeError("USB-CAN channel 0 unavailable"))  # type: ignore[method-assign]
        self.hardware = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]
        self.hardware._status_monitor_interval_s = 60.0

        initial = self.hardware.start()

        self.assertFalse(initial["connected"])
        self.assertTrue(self.hardware.health()["running"])
        status = self.hardware.status()
        self.assertEqual(status["control"]["mode"], "DISCONNECTED")
        self.assertFalse(status["feedback_ready"]["ok"])

    def test_start_times_out_stuck_hardware_and_keeps_shadow_backend_ready(self) -> None:
        self.hardware.close()
        blocked = threading.Event()

        def stuck_connect() -> dict[str, Any]:
            blocked.wait(5.0)
            return {"ok": True}

        self.fake.config["control_service"]["startup_connect_timeout_s"] = 0.05
        self.fake.connect = stuck_connect  # type: ignore[method-assign]
        self.hardware = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]

        started = time.monotonic()
        initial = self.hardware.start()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertFalse(initial["connected"])
        self.assertIn("did not finish", initial["error"])
        self.assertTrue(self.hardware.health()["running"])
        self.assertFalse(self.hardware.health()["hardware_transport_available"])
        self.assertEqual(self.hardware.status()["control"]["mode"], "DISCONNECTED")
        self.assertEqual(self.hardware._osc.status()["session"]["state"], "IDLE")
        blocked.set()

    def test_freedrive_brakes_then_waits_for_cpv_stop_before_leader_transition(self) -> None:
        original_handoff = self.hardware.handoff_to_console
        try:
            self.hardware.handoff_to_console = lambda _reason: self.fail("FREEDRIVE must not use console handoff")  # type: ignore[method-assign]
            result = self.hardware.freedrive("direct Leader test")
        finally:
            self.hardware.handoff_to_console = original_handoff  # type: ignore[method-assign]

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.freedrive_calls, 1)
        self.assertEqual(self.fake.cpv_stop_calls, 1)
        self.assertEqual(self.fake.follower_hold_calls, 0)
        self.assertEqual(self.fake.hold_calls, 0)

        self.assertEqual(self.fake.events[-3:], ["preempt", "cpv_stop", "freedrive"])
        self.assertIn("osc_brake", result)
        self.assertTrue(result["cpv_stop"]["cpv_zero_confirmed"])
        self.assertEqual(result["cpv_stop"]["post_cpv_dwell_s"], 0.0)
        authority = self.hardware.status()["broker"]
        self.assertEqual(authority["arm_writer"], "NONE")
        self.assertEqual(authority["servo_mode"], "SUSPENDED")
        self.assertFalse(self.hardware.servo_can_write("old-session", 0))

    def test_hold_is_allowed_from_fresh_leader_feedback_without_follower_arm_status(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"
        self.fake.leader_only_feedback = True
        self.hardware._refresh_status_snapshot()

        readiness = self.hardware.status()["feedback_ready"]
        self.assertTrue(readiness["ok"])
        self.assertIn("Leader feedback", readiness["reason"])

        job = self.hardware.submit_action_job({"kind": "hold", "reason": "exit leader"})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = self.hardware.action_status(job["action_id"])
            if state["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)

        self.assertEqual(self.hardware.action_status(job["action_id"])["status"], "completed")
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)

    def test_operator_gripper_command_does_not_create_leader_hold(self) -> None:
        result = self.hardware.command_gripper("grip", 0.0, 1.0, True)
        self.assertTrue(result["ok"])
        self.assertFalse(self.hardware.status()["gripper_hold"]["active"])

    def test_operator_gripper_preserves_active_osc_session(self) -> None:
        original_status = self.hardware._osc.status
        original_stop = self.hardware._osc.stop_session
        try:
            self.hardware._osc.status = lambda: {"session": {"state": "ACTIVE"}}  # type: ignore[method-assign]
            self.hardware._osc.stop_session = lambda _reason: self.fail("gripper must not stop an active OSC session")  # type: ignore[method-assign]
            result = self.hardware.command_gripper("grip", 0.0, 1.0, True)
        finally:
            self.hardware._osc.status = original_status  # type: ignore[method-assign]
            self.hardware._osc.stop_session = original_stop  # type: ignore[method-assign]
        self.assertTrue(result["ok"])
        self.assertTrue(result["osc_preserved"])

    def test_failed_operator_action_closes_observed_lifecycle(self) -> None:
        events: list[dict[str, Any]] = []
        self.hardware.add_action_observer(lambda **event: events.append(event))

        def fail(_: str) -> ExecutedAction:
            raise RuntimeError("simulated hold failure")

        original = self.fake.hold_follower_without_position_target
        try:
            self.fake.mode = "OSC_CPV"
            self.fake.continuous_stream_active = True
            self.fake.hold_follower_without_position_target = fail  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "simulated hold failure"):
                self.hardware.hold("test failure")
        finally:
            self.fake.hold_follower_without_position_target = original  # type: ignore[method-assign]

        self.assertEqual([event["lifecycle"] for event in events], ["requested", "failed"])
        self.assertEqual(events[0]["action_id"], events[1]["action_id"])

    def test_console_handoff_revokes_ownership_then_issues_official_follower_hold(self) -> None:
        lease = self.hardware.acquire("stale-model-owner")
        result = self.hardware.handoff_to_console("test console handoff")

        self.assertTrue(result["ok"])
        self.assertEqual(result["revoked_lease"]["owner"], lease["owner"])
        self.assertIsNone(self.hardware.status()["lease"])
        self.assertIn("preempt", self.fake.events)
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)

    def test_console_handoff_does_not_call_hold_when_osc_worker_wont_stop(self) -> None:
        original = self.hardware._osc.stop_session
        try:
            self.hardware._osc.stop_session = lambda reason: {  # type: ignore[method-assign]
                "handoff": {"servo_stopped": False, "feedback_stopped": True}
            }
            result = self.hardware.handoff_to_console("blocked osc worker")
        finally:
            self.hardware._osc.stop_session = original  # type: ignore[method-assign]

        self.assertFalse(result["ok"])
        self.assertEqual(result["handoff"]["stage"], "osc_stop")
        self.assertEqual(self.fake.hold_calls, 0)

    def test_console_handoff_uses_follower_hold_without_position_target(self) -> None:
        self.fake.mode = "OSC_CPV"
        self.fake.continuous_stream_active = True

        result = self.hardware.handoff_to_console("test CPV handoff")

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertFalse(result["hold"]["result"]["position_target_sent"])
        self.assertEqual(self.hardware.status()["broker"]["command_stream"], "NONE")
        self.assertEqual(self.fake.events[-3:], ["preempt", "cpv_stop", "follower_hold"])

        cleared = self.hardware.clear_gripper_hold()
        self.assertFalse(cleared["active"])
        self.assertFalse(self.hardware.status()["gripper_hold"]["active"])

    def test_operator_gripper_zero_force_clears_hold(self) -> None:
        self.hardware.command_gripper("grip", 0.0, 1.0, True)

        result = self.hardware.release_gripper_zero_force()

        self.assertTrue(result["ok"])
        self.assertTrue(result["zero_force"])
        self.assertFalse(self.hardware.status()["gripper_hold"]["active"])

    def test_operator_gripper_exits_freedrive_before_command(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"

        result = self.hardware.command_gripper("grip", 0.0, 1.0, True)

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.mode, "HOLD")
        self.assertTrue(result["arm_hold"]["ok"])

    def test_electronic_emergency_is_disabled_by_default(self) -> None:
        self.hardware.acquire("script")
        with self.assertRaisesRegex(RuntimeError, "disabled by the non-disabling control policy"):
            self.hardware.emergency_damping("danger")
        self.assertEqual(self.fake.estop_calls, 0)
        self.assertIsNotNone(self.hardware.status()["lease"])

    def test_expired_lease_triggers_hold(self) -> None:
        self.hardware.acquire("abandoned-script")
        assert self.hardware.leases._lease is not None
        self.hardware.leases._lease.expires_monotonic = time.monotonic() - 0.01
        deadline = time.monotonic() + 1.0
        while self.fake.follower_hold_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.estop_calls, 0)

    def test_fault_mode_blocks_non_emergency_operator_commands(self) -> None:
        self.fake.mode = "FAULT"
        self.hardware._refresh_status_snapshot()

        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.hardware.hold("must not reach robot")
        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.hardware.freedrive("must not reach robot")
        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.hardware.command_gripper("open", None, 1.0, False)

        self.assertEqual(self.fake.hold_calls, 0)

class HardwareFeedbackPoseTests(unittest.TestCase):
    def test_sdk_rpy_feedback_has_public_xyzw_pose(self) -> None:
        pose = absolute_pose_from_sdk_rpy([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertEqual(pose, {
            "position_m": [0.1, -0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        })


@unittest.skip("legacy osc facade was removed; OSC absolute-target coverage lives in test_osc_boundary")
class OscFacadeTests(unittest.TestCase):
    def test_osc_is_the_public_controller_and_hides_clutch_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = OperationalSpaceController({}, robot=FakeRobot(Path(directory)))  # type: ignore[arg-type]
            try:
                self.assertIsInstance(controller, OperationalSpaceController)

                class FakeOscServo:
                    def submit_absolute_target(self, body, *, mode):
                        return {"accepted": True, "mode": mode, "target_pose": body["payload"]["target_pose"]}

                    def status(self):
                        return {
                            "session": {"state": "ACTIVE", "id": "osc-1", "input_source": "legacy-adapter"},
                            "clutch_active": True, "anchor_id": 9, "relative_pose": {"position_m": [1, 2, 3]},
                            "target_pose": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
                            "execution_sample": {
                                "sample_id": 41,
                                "target_generation": 7,
                                "target_tcp": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
                                "estimated_tcp_pose": {"position_m": [0.02, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
                                "measured_tcp_pose": {"position_m": [0.01, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
                                "measured_joint_state_rad": [0.05] * 7,
                                "feedback_age_s": 0.012,
                            },
                            "last_result": {"solver": {"tcp": {
                                "position_m": [0.0, 0.2, 0.3],
                                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            }, "condition_number": 12.5, "nullspace_velocity_norm": 0.02}},
                            "last_output": {"status": "limited", "final_joint_target_rad": [0.1] * 7, "final_joint_velocity_rad_s": [0.2] * 7, "sequence": 3, "epoch": 4},
                            "solver": {}, "workspace": {}, "diagnostics": {"loop_count": 8, "output_count": 7, "cpv_dispatch_count": 0, "timing": {}},
                        }

                    def target_pose(self):
                        return {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}

                    def accepting_targets(self):
                        return True

                    def stop_session(self, reason):
                        return {"handoff": {"reason": reason}}

                    def heartbeat(self, client_id, session_id):
                        self.heartbeat_call = (client_id, session_id)
                        return self.status()

                controller._osc = FakeOscServo()  # type: ignore[assignment]
                controller.osc = controller._osc
                controller.status = lambda: {  # type: ignore[method-assign]
                    "control": {"robot": {"joint_angles_rad": [0.0] * 7}},
                }
                state = controller.osc_state()
                self.assertNotIn("clutch_active", state)
                self.assertNotIn("anchor_id", state)
                self.assertNotIn("input_source", state["session"])
                self.assertEqual(state["transport"]["hardware_feedback"]["joint_angles_rad"], [0.0] * 7)
                self.assertNotIn("robot", state)
                self.assertEqual(state["execution"]["estimated_tcp_pose"]["position_m"], [0.02, 0.2, 0.3])
                self.assertEqual(state["execution"]["measured_tcp_pose"]["position_m"], [0.01, 0.2, 0.3])
                self.assertEqual(state["execution"]["measured_joint_state_rad"], [0.05] * 7)
                self.assertEqual(state["execution"]["sample_id"], 41)
                self.assertEqual(state["execution"]["observed_source"], "simulated_cpv_feedback")
                self.assertEqual(state["execution"]["output_count"], 7)
                self.assertEqual(state["command"]["output_status"], "limited")
                self.assertAlmostEqual(state["diagnostics"]["tcp_error"]["position_norm_m"], 0.09)
                self.assertAlmostEqual(state["diagnostics"]["measured_tracking_error"]["position_norm_m"], 0.09)
                self.assertAlmostEqual(state["diagnostics"]["estimated_tracking_error"]["position_norm_m"], 0.08)
                self.assertAlmostEqual(state["diagnostics"]["tcp_error"]["orientation_angle_rad"], 0.0)
                self.assertEqual(state["diagnostics"]["pink"]["condition_number"], 12.5)
                self.assertEqual(state["transport"]["participation"], "shadow_simulated")
                self.assertIsNone(state["transport"]["last_cpv_dispatch"])
                result = controller.osc_command({
                    "session_id": "osc-1", "client_id": "client", "sequence": 1, "type": "track_tcp",
                    "payload": {"target_pose": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}},
                })
                self.assertTrue(result["ok"])
                self.assertEqual(result["result"]["mode"], "track_tcp")
                compact = controller.osc_command({
                    "session_id": "osc-1", "client_id": "client", "sequence": 2, "type": "track_tcp",
                    "acknowledgement_only": True,
                    "payload": {"target_pose": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}},
                })
                self.assertTrue(compact["ok"])
                self.assertNotIn("state", compact)
                with self.assertRaisesRegex(ValueError, "OSC command type"):
                    controller.osc_command({
                        "session_id": "osc-1", "client_id": "client", "sequence": 3,
                        "type": "joint_target", "payload": {"joint_angles_rad": [0.0] * 7},
                    })
                heartbeat = controller.osc_heartbeat("client", "osc-1")
                self.assertTrue(heartbeat["ok"])
                self.assertEqual(controller._osc.heartbeat_call, ("client", "osc-1"))
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
