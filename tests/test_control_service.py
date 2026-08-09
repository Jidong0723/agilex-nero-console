from __future__ import annotations

import tempfile
import math
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from supervisor.control import (
    LeaseError,
    LeaseManager,
    RobotControlBroker,
    candidate_tool_pose_from_flange,
)
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
        self.gripper_hold = {
            "supported": False, "active": False, "mode": None,
            "target_width_m": None, "force_n": None, "alarm": None,
        }
        self.can_feedback_recovery = {
            "status": "not_needed", "enabled": True, "frame_sent": False,
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

    def execute_action(
        self, action: dict[str, Any], timeout: float | None = None, expected_preempt_epoch: int | None = None
    ) -> ExecutedAction:
        self.execute_calls += 1
        if expected_preempt_epoch != self.epoch:
            return ExecutedAction(now_iso(), action, None, False, False, "PREEMPTED_BY_OPERATOR")
        started = time.monotonic()
        while action.get("block") and time.monotonic() - started < 1.0:
            if self.preempted.wait(0.01):
                return ExecutedAction(now_iso(), action, action, False, True, "PREEMPTED_BY_OPERATOR")
        return ExecutedAction(now_iso(), action, action, True, True)

    def hold_position(self, reason: str, recover_stale_leader: bool = False) -> ExecutedAction:
        self.hold_calls += 1
        self.events.append("hold")
        if self.hold_should_fail:
            return ExecutedAction(now_iso(), {"type": "hold"}, {"type": "hold"}, False, True, "simulated hold failure")
        self.mode = "HOLD"
        self.freedrive_backend = None
        return ExecutedAction(now_iso(), {"type": "hold"}, {"type": "hold"}, True, True, reason)

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
            "flange_pose": [0.0] * 6,
            "arm_status": {"arm_status": 0},
            "joint_enable_status": [True] * 7,
        }
        if getattr(self, "leader_only_feedback", False):
            robot = {
                "joint_angles_rad": [0.0] * 7,
                "flange_pose": [0.0] * 6,
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
            "can_feedback_recovery": dict(self.can_feedback_recovery),
            "leader_feedback_age_s": 0.0 if getattr(self, "leader_only_feedback", False) else None,
            "leader_feedback_hz": 220.0 if getattr(self, "leader_only_feedback", False) else None,
        }

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


class CandidateToolPoseTests(unittest.TestCase):
    def test_zero_rotation_applies_candidate_offset(self) -> None:
        pose = candidate_tool_pose_from_flange([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertIsNotNone(pose)
        assert pose is not None
        for actual, expected in zip(pose["position_m"], [0.275, 0.2, 0.2765]):
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertFalse(pose["verified"])

    def test_yaw_rotates_offset_in_base_frame(self) -> None:
        pose = candidate_tool_pose_from_flange([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2])
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose["position_m"][0], 0.0, places=7)
        self.assertAlmostEqual(pose["position_m"][1], 0.175, places=7)
        self.assertAlmostEqual(pose["position_m"][2], -0.0235, places=7)

    def test_invalid_flange_pose_returns_none(self) -> None:
        for value in (None, [0.0] * 5, [0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]):
            self.assertIsNone(candidate_tool_pose_from_flange(value))


class BrokerPreemptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fake = FakeRobot(Path(self.temp.name))
        self.broker = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]
        self.broker.start()

    def tearDown(self) -> None:
        self.broker.close()
        self.temp.cleanup()

    def test_async_operator_job_is_single_flight_and_queryable(self) -> None:
        first = self.broker.submit_action_job({"kind": "hold", "reason": "test"})
        second = self.broker.submit_action_job({"kind": "hold", "reason": "duplicate"})
        self.assertEqual(first["action_id"], second["action_id"])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = self.broker.action_status(first["action_id"])
            if state["status"] not in {"queued", "running", "already_running"}:
                break
            time.sleep(0.01)
        self.assertEqual(self.broker.action_status(first["action_id"])["status"], "completed")

    def test_start_remains_running_when_usb_can_connect_fails(self) -> None:
        self.broker.close()
        self.fake.connect = lambda: (_ for _ in ()).throw(RuntimeError("USB-CAN channel 0 unavailable"))  # type: ignore[method-assign]
        self.fake.get_control_state = lambda: (_ for _ in ()).throw(RuntimeError("USB-CAN channel 0 unavailable"))  # type: ignore[method-assign]
        self.broker = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]
        self.broker._status_monitor_interval_s = 60.0

        initial = self.broker.start()

        self.assertFalse(initial["connected"])
        self.assertTrue(self.broker.health()["running"])
        status = self.broker.status()
        self.assertEqual(status["control"]["mode"], "DISCONNECTED")
        self.assertFalse(status["feedback_ready"]["ok"])

    def test_start_times_out_stuck_hardware_and_keeps_shadow_backend_ready(self) -> None:
        self.broker.close()
        blocked = threading.Event()

        def stuck_connect() -> dict[str, Any]:
            blocked.wait(5.0)
            return {"ok": True}

        self.fake.config["control_service"]["startup_connect_timeout_s"] = 0.05
        self.fake.connect = stuck_connect  # type: ignore[method-assign]
        self.broker = RobotControlBroker({}, robot=self.fake)  # type: ignore[arg-type]

        started = time.monotonic()
        initial = self.broker.start()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertFalse(initial["connected"])
        self.assertIn("did not finish", initial["error"])
        self.assertTrue(self.broker.health()["running"])
        self.assertFalse(self.broker.health()["hardware_transport_available"])
        self.assertEqual(self.broker.status()["control"]["mode"], "DISCONNECTED")
        self.assertEqual(self.broker.teleop_status()["session"]["state"], "IDLE")
        blocked.set()

    def test_operator_hold_preempts_active_script_and_revokes_lease(self) -> None:
        lease = self.broker.acquire("long-script")
        output: dict[str, Any] = {}

        def run_action() -> None:
            output.update(self.broker.execute(lease["token"], {"type": "joint_target", "block": True}))

        thread = threading.Thread(target=run_action)
        thread.start()
        time.sleep(0.03)
        result = self.broker.hold("operator pressed hold")
        thread.join(1.0)
        self.assertTrue(result["ok"])
        self.assertIn("PREEMPTED_BY_OPERATOR", output["reason"])
        self.assertIsNone(self.broker.status()["lease"])
        self.assertEqual(self.fake.mode, "HOLD")
        self.assertEqual(self.fake.estop_calls, 0)

    def test_operator_freedrive_preempts_active_script_and_revokes_lease(self) -> None:
        lease = self.broker.acquire("long-script")
        output: dict[str, Any] = {}

        def run_action() -> None:
            output.update(self.broker.execute(lease["token"], {"type": "joint_target", "block": True}))

        thread = threading.Thread(target=run_action)
        thread.start()
        time.sleep(0.03)
        result = self.broker.freedrive("operator requested manual takeover")
        thread.join(1.0)
        self.assertTrue(result["ok"])
        self.assertIn("PREEMPTED_BY_OPERATOR", output["reason"])
        self.assertIsNone(self.broker.status()["lease"])
        self.assertEqual(self.fake.mode, "FREEDRIVE")

    def test_freedrive_brakes_then_waits_for_cpv_stop_before_leader_transition(self) -> None:
        original_handoff = self.broker.handoff_to_console
        try:
            self.broker.handoff_to_console = lambda _reason: self.fail("FREEDRIVE must not use console handoff")  # type: ignore[method-assign]
            result = self.broker.freedrive("direct Leader test")
        finally:
            self.broker.handoff_to_console = original_handoff  # type: ignore[method-assign]

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.freedrive_calls, 1)
        self.assertEqual(self.fake.cpv_stop_calls, 1)
        self.assertEqual(self.fake.follower_hold_calls, 0)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.events[-3:], ["preempt", "cpv_stop", "freedrive"])
        self.assertIn("teleop_brake", result)
        self.assertTrue(result["cpv_stop"]["cpv_zero_confirmed"])
        self.assertEqual(result["cpv_stop"]["post_cpv_dwell_s"], 0.0)
        authority = self.broker.status()["broker"]
        self.assertEqual(authority["arm_writer"], "NONE")
        self.assertEqual(authority["servo_mode"], "SUSPENDED")
        self.assertFalse(self.broker.servo_can_write("old-session", 0))

    def test_freedrive_rejects_missing_startup_feedback_before_a_job_or_enable_attempt(self) -> None:
        self.fake.feedback_unavailable = True
        self.fake.can_feedback_recovery = {
            "status": "failed", "enabled": True, "frame_sent": False,
            "reason": "no known NERO feedback IDs observed",
        }
        self.broker._refresh_status_snapshot()

        with self.assertRaisesRegex(RuntimeError, "fresh CAN feedback"):
            self.broker.submit_action_job({"kind": "freedrive"})

        self.assertEqual(self.fake.freedrive_calls, 0)
        self.assertEqual(self.broker.health()["job_count"], 0)

    def test_fresh_follower_feedback_overrides_stale_startup_recovery_failure(self) -> None:
        self.fake.can_feedback_recovery = {
            "status": "failed", "enabled": True, "frame_sent": False,
            "reason": "startup probe did not observe known IDs",
        }
        self.broker._refresh_status_snapshot()

        readiness = self.broker.status()["feedback_ready"]

        self.assertTrue(readiness["ok"])
        self.assertIn("follower feedback", readiness["reason"])

    def test_hold_is_allowed_from_fresh_leader_feedback_without_follower_arm_status(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"
        self.fake.leader_only_feedback = True
        self.fake.can_feedback_recovery = {
            "status": "failed", "enabled": True, "frame_sent": False,
            "reason": "no known NERO feedback IDs observed",
        }
        self.broker._refresh_status_snapshot()

        readiness = self.broker.status()["feedback_ready"]
        self.assertTrue(readiness["ok"])
        self.assertIn("Leader feedback", readiness["reason"])

        job = self.broker.submit_action_job({"kind": "hold", "reason": "exit leader"})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = self.broker.action_status(job["action_id"])
            if state["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)

        self.assertEqual(self.broker.action_status(job["action_id"])["status"], "completed")
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)

    def test_status_reports_active_action_owner(self) -> None:
        lease = self.broker.acquire("model-policy")
        output: dict[str, Any] = {}

        def run_action() -> None:
            output.update(self.broker.execute(lease["token"], {"type": "joint_target", "block": True}))

        thread = threading.Thread(target=run_action)
        thread.start()
        deadline = time.monotonic() + 0.5
        active = None
        while active is None and time.monotonic() < deadline:
            active = self.broker.status()["active_action"]
            time.sleep(0.01)
        self.assertIsNotNone(active)
        self.assertEqual(active["owner"], "model-policy")
        self.fake.request_preempt("test cleanup")
        thread.join(1.0)

    def test_status_exposes_can_feedback_recovery(self) -> None:
        status = self.broker.status()
        self.assertEqual(status["can_feedback_recovery"]["status"], "not_needed")
        self.assertFalse(status["can_feedback_recovery"]["frame_sent"])

    def test_operator_gripper_command_does_not_create_leader_hold(self) -> None:
        result = self.broker.command_gripper("grip", 0.0, 1.0, True)
        self.assertTrue(result["ok"])
        self.assertFalse(self.broker.status()["gripper_hold"]["active"])

    def test_operator_gripper_preserves_active_teleop_session(self) -> None:
        original_status = self.broker.teleop.status
        original_stop = self.broker.teleop.stop_session
        try:
            self.broker.teleop.status = lambda: {"session": {"state": "ACTIVE"}}  # type: ignore[method-assign]
            self.broker.teleop.stop_session = lambda _reason: self.fail("gripper must not stop an active teleop session")  # type: ignore[method-assign]
            result = self.broker.command_gripper("grip", 0.0, 1.0, True)
        finally:
            self.broker.teleop.status = original_status  # type: ignore[method-assign]
            self.broker.teleop.stop_session = original_stop  # type: ignore[method-assign]
        self.assertTrue(result["ok"])
        self.assertTrue(result["teleop_preserved"])

    def test_failed_operator_action_closes_observed_lifecycle(self) -> None:
        events: list[dict[str, Any]] = []
        self.broker.add_action_observer(lambda **event: events.append(event))

        def fail(_: str) -> ExecutedAction:
            raise RuntimeError("simulated hold failure")

        original = self.fake.hold_follower_without_position_target
        try:
            self.fake.mode = "TELEOP_CPV_VELOCITY"
            self.fake.continuous_stream_active = True
            self.fake.hold_follower_without_position_target = fail  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "simulated hold failure"):
                self.broker.hold("test failure")
        finally:
            self.fake.hold_follower_without_position_target = original  # type: ignore[method-assign]

        self.assertEqual([event["lifecycle"] for event in events], ["requested", "failed"])
        self.assertEqual(events[0]["action_id"], events[1]["action_id"])

    def test_console_handoff_revokes_ownership_then_issues_official_follower_hold(self) -> None:
        lease = self.broker.acquire("stale-model-owner")
        result = self.broker.handoff_to_console("test console handoff")

        self.assertTrue(result["ok"])
        self.assertEqual(result["revoked_lease"]["owner"], lease["owner"])
        self.assertIsNone(self.broker.status()["lease"])
        self.assertIn("preempt", self.fake.events)
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)

    def test_console_handoff_does_not_call_hold_when_teleop_worker_wont_stop(self) -> None:
        original = self.broker.teleop.stop_session
        try:
            self.broker.teleop.stop_session = lambda reason: {  # type: ignore[method-assign]
                "handoff": {"servo_stopped": False, "feedback_stopped": True}
            }
            result = self.broker.handoff_to_console("blocked teleop worker")
        finally:
            self.broker.teleop.stop_session = original  # type: ignore[method-assign]

        self.assertFalse(result["ok"])
        self.assertEqual(result["handoff"]["stage"], "teleop_stop")
        self.assertEqual(self.fake.hold_calls, 0)

    def test_console_handoff_uses_follower_hold_without_position_target(self) -> None:
        self.fake.mode = "TELEOP_CPV_VELOCITY"
        self.fake.continuous_stream_active = True

        result = self.broker.handoff_to_console("test CPV handoff")

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertFalse(result["hold"]["result"]["position_target_sent"])
        self.assertEqual(self.broker.status()["broker"]["command_stream"], "NONE")
        self.assertEqual(self.fake.events[-3:], ["preempt", "cpv_stop", "follower_hold"])

        cleared = self.broker.clear_gripper_hold()
        self.assertFalse(cleared["active"])
        self.assertFalse(self.broker.status()["gripper_hold"]["active"])

    def test_operator_gripper_zero_force_clears_hold(self) -> None:
        self.broker.command_gripper("grip", 0.0, 1.0, True)

        result = self.broker.release_gripper_zero_force()

        self.assertTrue(result["ok"])
        self.assertTrue(result["zero_force"])
        self.assertFalse(self.broker.status()["gripper_hold"]["active"])

    def test_operator_gripper_holds_arm_when_preempting_motion(self) -> None:
        lease = self.broker.acquire("moving-model")
        output: dict[str, Any] = {}

        def run_action() -> None:
            output.update(self.broker.execute(
                lease["token"], {"type": "joint_target", "block": True}
            ))

        thread = threading.Thread(target=run_action)
        thread.start()
        time.sleep(0.03)
        result = self.broker.command_gripper("grip", 0.0, 1.0, True)
        thread.join(1.0)

        self.assertTrue(result["ok"])
        self.assertTrue(result["arm_hold"]["ok"])
        self.assertIn("PREEMPTED_BY_OPERATOR", output["reason"])
        self.assertIsNone(self.broker.status()["lease"])

    def test_operator_gripper_exits_freedrive_before_command(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"

        result = self.broker.command_gripper("grip", 0.0, 1.0, True)

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.mode, "HOLD")
        self.assertTrue(result["arm_hold"]["ok"])

    def test_explicit_motion_exits_freedrive_before_dispatch(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"
        lease = self.broker.acquire("explicit-motion")

        result = self.broker.execute(
            lease["token"],
            {"type": "joint_target", "joint_angles_rad": [0.0] * 7},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.mode, "HOLD")

    def test_freedrive_exit_failure_does_not_dispatch_motion(self) -> None:
        self.fake.mode = "FREEDRIVE"
        self.fake.freedrive_backend = "leader"
        self.fake.hold_should_fail = True
        lease = self.broker.acquire("explicit-motion")

        result = self.broker.execute(
            lease["token"],
            {"type": "joint_target", "joint_angles_rad": [0.0] * 7},
        )

        self.assertFalse(result["ok"])
        self.assertEqual(self.fake.execute_calls, 0)

    def test_electronic_emergency_is_disabled_by_default(self) -> None:
        self.broker.acquire("script")
        with self.assertRaisesRegex(RuntimeError, "disabled by the non-disabling control policy"):
            self.broker.emergency_damping("danger")
        self.assertEqual(self.fake.estop_calls, 0)
        self.assertIsNotNone(self.broker.status()["lease"])

    def test_expired_lease_triggers_hold(self) -> None:
        self.broker.acquire("abandoned-script")
        assert self.broker.leases._lease is not None
        self.broker.leases._lease.expires_monotonic = time.monotonic() - 0.01
        deadline = time.monotonic() + 1.0
        while self.fake.follower_hold_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(self.fake.follower_hold_calls, 1)
        self.assertEqual(self.fake.hold_calls, 0)
        self.assertEqual(self.fake.estop_calls, 0)

    def test_enabled_emergency_preempts_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRobot(Path(directory))
            fake.config["safety"] = {"allow_electronic_emergency_stop": True}
            broker = RobotControlBroker({}, robot=fake)  # type: ignore[arg-type]
            broker.start()
            try:
                lease = broker.acquire("running-model")
                output: dict[str, Any] = {}

                def run_action() -> None:
                    output.update(broker.execute(
                        lease["token"], {"type": "joint_target", "block": True}
                    ))

                thread = threading.Thread(target=run_action)
                thread.start()
                time.sleep(0.03)
                result = broker.emergency_damping("operator emergency")
                thread.join(1.0)
                self.assertTrue(result["ok"])
                self.assertEqual(fake.estop_calls, 1)
                self.assertIn("PREEMPTED_BY_OPERATOR", output["reason"])
                self.assertIsNone(broker.status()["lease"])
            finally:
                broker.close()

    def test_fault_mode_blocks_non_emergency_operator_commands(self) -> None:
        self.fake.mode = "FAULT"
        self.broker._refresh_status_snapshot()

        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.broker.hold("must not reach robot")
        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.broker.freedrive("must not reach robot")
        with self.assertRaisesRegex(RuntimeError, "restore fresh CAN feedback"):
            self.broker.command_gripper("open", None, 1.0, False)

        self.assertEqual(self.fake.hold_calls, 0)


if __name__ == "__main__":
    unittest.main()
