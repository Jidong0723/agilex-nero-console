from __future__ import annotations

import unittest
from types import SimpleNamespace

from nero_backend.robot import NeroRobot


class FakeGripper:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.width = 0.015
        self.force = 0.0
        self.enabled = True
        self.fail_commands = False
        self.requires_reset = False
        self.hide_feedback_when_disabled = False
        self.teaching_friction = 5
        self.commands: list[tuple[float, float]] = []

    def get_gripper_status(self):
        if self.hide_feedback_when_disabled and not self.enabled:
            return None
        foc = SimpleNamespace(
            voltage_too_low=False,
            motor_overheating=False,
            driver_overcurrent=False,
            driver_overheating=False,
            sensor_status=False,
            driver_error_status=False,
            driver_enable_status=self.enabled,
            homing_status=False,
        )
        return SimpleNamespace(msg=SimpleNamespace(
            value=self.width, force=self.force, foc_status=foc, mode="width"
        ))

    def move_gripper_m(self, value, force):
        self.events.append("gripper")
        self.commands.append((float(value), float(force)))
        self.force = float(force)
        self.enabled = not self.fail_commands and not self.requires_reset
        if value > 0.0:
            self.width = float(value)

    def disable_gripper(self):
        self.events.append("gripper-disable")
        self.enabled = False
        self.requires_reset = True
        self.force = 0.0
        return True

    def reset_gripper(self):
        self.events.append("gripper-reset")
        self.enabled = False
        return True

    def _send_msg(self, message):
        self.events.append("gripper-enable-clear")
        if getattr(message, "status_code", None) == 0x03:
            self.requires_reset = False
            self.enabled = not self.fail_commands
            if not self.fail_commands:
                self.width = float(message.value) / 1e6
                self.force = float(message.force) / 1e3

    def get_gripper_teaching_pendant_param(self, timeout=1.0, min_interval=1.0):
        return SimpleNamespace(msg=SimpleNamespace(
            teaching_range_per=100,
            max_range_config=0.1,
            teaching_friction=self.teaching_friction,
        ))

    def set_gripper_teaching_pendant_param(
        self, teaching_range_per=100, max_range_config=0.0,
        teaching_friction=1, timeout=1.0
    ):
        self.teaching_friction = int(teaching_friction)
        return True


class FakeSdkRobot:
    OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(J="j", P="p", L="l", CPV="cpv"))

    def __init__(self) -> None:
        self.leader_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.follower_values = [-0.5] * 7
        self.leader_timestamp = 0
        self.mode = "leader"
        self.targets: list[list[float]] = []
        self.reset_calls = 0
        self.arm_status_value = 0
        self.enable_status = [True] * 7
        self.enable_timeouts: list[float] = []
        self.enable_joint_indices: list[int] = []
        self.disconnect_calls = 0
        self.disable_calls = 0
        self.events: list[str] = []
        self.motion_mode_calls = 0
        self.motion_modes: list[str] = []
        self.cartesian_calls: list[tuple[str, list[float]]] = []
        self.cpv_position_calls: list[tuple[int, float]] = []
        self.cpv_profile = {"acc": [1.0] * 7, "dcc": [1.0] * 7, "cv": [0.1] * 7}
        self.motor_velocity = [0.01 * index for index in range(1, 8)]
        self.gripper: FakeGripper | None = None
        self.gripper_fail_after_leader = False
        self.drag_teach = False

    def get_leader_joint_angles(self):
        self.leader_timestamp += 1
        return SimpleNamespace(msg=list(self.leader_values), timestamp=self.leader_timestamp, hz=100.0)

    def get_joint_angles(self):
        return SimpleNamespace(msg=list(self.follower_values), timestamp=1, hz=100.0)

    def get_motor_states(self, joint_index):
        return SimpleNamespace(msg=SimpleNamespace(velocity=self.motor_velocity[int(joint_index) - 1]))

    def get_driver_states(self, joint_index):
        return SimpleNamespace(msg=SimpleNamespace())

    def get_arm_status(self):
        return SimpleNamespace(msg=SimpleNamespace(
            ctrl_mode=2 if self.drag_teach else 1,
            arm_status=self.arm_status_value,
            mode_feedback=1,
            teach_status=1 if self.drag_teach else 0,
            motion_status=0, trajectory_num=0, err_status={},
        ))

    def get_flange_pose(self):
        return SimpleNamespace(msg=[0.0, 0.2, 0.3, 0.0, 0.0, 0.0])

    def get_tcp_pose(self):
        return SimpleNamespace(msg=[0.0, 0.2, 0.3, 0.0, 0.0, 0.0])

    def fk(self, joints):
        return [float(joints[0]), 0.2, 0.3, 0.0, 0.0, 0.0]

    def get_joints_enable_status_list(self):
        return list(self.enable_status)

    def set_motion_mode(self, value):
        self.motion_mode_calls += 1
        self.motion_modes.append(value)
        self.events.append(f"motion:{value}")
        return None

    def set_cpv_acc(self, joint_index, acc, timeout=1.0):
        self.cpv_profile["acc"][int(joint_index) - 1] = float(acc); return True

    def set_cpv_dcc(self, joint_index, dcc, timeout=1.0):
        self.cpv_profile["dcc"][int(joint_index) - 1] = float(dcc); return True

    def set_cpv_cv(self, joint_index, cv, timeout=1.0):
        self.cpv_profile["cv"][int(joint_index) - 1] = float(cv); return True

    def get_cpv_acc(self, joint_index, timeout=1.0, min_interval=0.0):
        return self.cpv_profile["acc"][int(joint_index) - 1]

    def get_cpv_dcc(self, joint_index, timeout=1.0, min_interval=0.0):
        return self.cpv_profile["dcc"][int(joint_index) - 1]

    def get_cpv_cv(self, joint_index, timeout=1.0, min_interval=0.0):
        return self.cpv_profile["cv"][int(joint_index) - 1]

    def move_p(self, target):
        self.cartesian_calls.append(("point", list(target)))

    def move_l(self, target):
        self.cartesian_calls.append(("linear", list(target)))

    def move_j(self, target):
        self.events.append("move_j")
        self.targets.append(list(target))
        if self.mode == "follower":
            self.follower_values = list(target)

    def move_cpv_pos(self, joint_index, pos):
        self.events.append(f"cpv-pos:{int(joint_index)}:{float(pos):.3f}")
        self.cpv_position_calls.append((int(joint_index), float(pos)))

    def set_follower_mode(self):
        self.mode = "follower"
        self.follower_values = list(self.leader_values)
        self.events.append("follower")

    def set_leader_mode(self):
        self.mode = "leader"
        self.events.append("leader")
        if self.gripper is not None and self.gripper_fail_after_leader:
            self.gripper.fail_commands = True
            self.gripper.enabled = False

    def _send_msg(self, message):
        teach_command = getattr(message, "grag_teach_ctrl", 0)
        if teach_command == 0x01:
            self.drag_teach = True
            self.mode = "drag_teach"
            self.events.append("drag-start")
            if self.gripper is not None and self.gripper_fail_after_leader:
                self.gripper.fail_commands = True
                self.gripper.enabled = False
        elif teach_command == 0x02:
            self.drag_teach = False
            self.mode = "follower"
            self.events.append("drag-stop")

    def enable(self, joint_index=255, timeout=1.5):
        self.events.append("enable")
        self.enable_timeouts.append(timeout)
        self.enable_joint_indices.append(int(joint_index))
        if joint_index == 255 and timeout >= 5.0:
            self.enable_status = [True] * 7
        elif joint_index != 255:
            self.enable_status[joint_index - 1] = True
        return True

    def reset(self):
        self.reset_calls += 1
        self.arm_status_value = 0
        self.events.append("reset")

    def disconnect(self):
        self.disconnect_calls += 1

    def disable(self, joint_index=255, timeout=1.5):
        self.disable_calls += 1
        self.enable_status = [False] * 7
        return True


class RobotModeTests(unittest.TestCase):
    def make_robot(self) -> tuple[NeroRobot, FakeSdkRobot]:
        robot = NeroRobot({
            "motion": {"gripper_verify_settle_s": 0.0, "gripper_hold_verify_s": 0.0},
            "safety": {},
            "logging": {},
        })
        sdk = FakeSdkRobot()
        gripper = FakeGripper(sdk.events)
        sdk.gripper = gripper
        robot.robot = sdk
        robot.gripper = gripper
        robot._control_mode = "FREEDRIVE"
        robot._freedrive_backend = "leader"
        return robot, sdk

    def test_configure_cpv_profile_acknowledges_and_reads_back_all_joints(self) -> None:
        robot, sdk = self.make_robot()
        result = robot.configure_cpv_profile(2.0, 5.0, 5.0)
        self.assertTrue(result["ok"])
        self.assertEqual(sdk.cpv_profile["cv"], [2.0] * 7)
        self.assertEqual(sdk.cpv_profile["acc"], [5.0] * 7)
        self.assertEqual(sdk.cpv_profile["dcc"], [5.0] * 7)
        self.assertEqual(len(result["joints"]), 7)

    def test_freedrive_state_uses_live_leader_angles(self) -> None:
        robot, sdk = self.make_robot()
        first = robot.read_state().joint_angles_rad
        sdk.leader_values[0] = 0.9
        second = robot.read_state().joint_angles_rad
        self.assertEqual(first, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.assertEqual(second[0], 0.9)

    def test_cpv_position_sends_joint_targets_in_joint_order(self) -> None:
        robot, sdk = self.make_robot()
        result = robot.send_cpv_position([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        self.assertTrue(result["ok"])
        self.assertEqual(result["motion_mode"], "CPV_POSITION")
        self.assertEqual(sdk.cpv_position_calls, [
            (1, 0.1), (2, -0.2), (3, 0.3), (4, -0.4),
            (5, 0.5), (6, -0.6), (7, 0.7),
        ])
        self.assertEqual(result["joint_target_rad"], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        self.assertNotIn("enable", sdk.events)

    def test_follower_hold_exits_cpv_without_sending_j_target(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot._freedrive_backend = None
        sdk.follower_values = [0.11, -0.21, 0.31, -0.41, 0.51, -0.61, 0.71]
        sdk.motor_velocity = [0.0] * 7
        robot.send_cpv_position([0.1] * 7)
        sdk.events.clear()

        result = robot.hold_follower_without_position_target("exit continuous teleop")

        self.assertTrue(result.ok, result.result)
        self.assertTrue(result.result["cpv_stream_stopped"])
        self.assertTrue(result.result["cpv_zero_confirmed"])
        self.assertFalse(result.result["position_target_sent"])
        hold_events = [event for event in sdk.events if event.startswith("cpv-pos:")]
        self.assertEqual(hold_events, [f"cpv-pos:{index}:{value:.3f}" for index, value in enumerate(sdk.follower_values, start=1)])
        self.assertNotIn("follower", sdk.events)
        self.assertNotIn("motion:j", sdk.events)
        self.assertNotIn("move_j", sdk.events)
        self.assertNotIn("enable", sdk.events)
        self.assertEqual(sdk.targets, [])
        self.assertEqual(result.result["follower_mode_command"], "skipped_already_follower_cpv")

    def test_follower_hold_sends_official_follower_only_when_returning_from_leader(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "FREEDRIVE"
        robot._freedrive_backend = "leader"
        sdk.events.clear()

        result = robot.hold_follower_without_position_target("exit official freedrive")

        self.assertTrue(result.ok, result.result)
        self.assertIn("follower", sdk.events)
        self.assertEqual(result.result["follower_mode_command"], "sent_returning_from_leader_or_drag")

    def test_enable_helper_targets_only_feedback_disabled_joint(self) -> None:
        robot, sdk = self.make_robot()
        sdk.enable_status[3] = False

        enabled, status = robot._enable_all_with_retry(timeout=0.2, poll_interval=0.0)

        self.assertTrue(enabled)
        self.assertEqual(status, [True] * 7)
        self.assertEqual(sdk.enable_joint_indices, [4])

    def test_enable_helper_never_retries_a_joint_when_feedback_is_late(self) -> None:
        robot, sdk = self.make_robot()
        sdk.enable_status[3] = False

        def delayed_enable(joint_index=255, timeout=1.5):
            sdk.events.append("enable")
            sdk.enable_joint_indices.append(int(joint_index))
            # Simulate a controller which accepts the request but has not yet
            # published the enabled bit back to the SDK.
            return True

        sdk.enable = delayed_enable  # type: ignore[method-assign]
        enabled, status = robot._enable_all_with_retry(timeout=0.03, poll_interval=0.001)

        self.assertFalse(enabled)
        self.assertEqual(status[3], False)
        self.assertEqual(sdk.enable_joint_indices, [4])

    def test_teleop_feedback_uses_only_joint_and_motor_cache(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        feedback = robot.read_teleop_feedback()
        self.assertEqual(feedback["joint_angles_rad"], sdk.follower_values)
        self.assertEqual(feedback["joint_velocity_rad_s"], sdk.motor_velocity)
        self.assertIsInstance(feedback["timestamp_monotonic_ns"], int)

    def test_freedrive_state_uses_sdk_fk_for_live_flange_pose(self) -> None:
        robot, sdk = self.make_robot()
        sdk.leader_values[0] = 0.9
        state = robot.read_state()
        self.assertEqual(state.flange_pose[0], 0.9)
        self.assertEqual(state.raw["flange_pose_source"], "sdk_fk_leader")

    def test_emergency_recovery_requires_explicit_confirmation(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "EMERGENCY_DAMPING"
        robot._emergency_latched = True
        sdk.arm_status_value = 1
        rejected = robot.enter_freedrive("test", recover_emergency=False)
        self.assertFalse(rejected.ok)
        self.assertEqual(sdk.reset_calls, 0)
        robot._reconnect_after_emergency_reset = lambda: None
        recovered = robot.enter_freedrive("test", recover_emergency=True)
        self.assertTrue(recovered.ok, recovered.result)
        self.assertEqual(sdk.reset_calls, 2)
        self.assertEqual(sdk.disable_calls, 0)
        self.assertIn("leader", sdk.events)
        self.assertNotIn("enable", sdk.events)
        self.assertEqual(robot.get_control_state()["mode"], "FREEDRIVE")

    def test_emergency_recovery_allows_delayed_joint7_enable(self) -> None:
        robot, sdk = self.make_robot()
        robot.motion_config["enable_timeout_s"] = 8.0
        robot.motion_config["emergency_reset_settle_s"] = 0.0
        robot._control_mode = "EMERGENCY_DAMPING"
        robot._emergency_latched = True
        sdk.arm_status_value = 1
        sdk.enable_status = [True, True, True, True, True, True, False]
        robot._reconnect_after_emergency_reset = lambda: None

        result = robot.enter_freedrive("test delayed J7", recover_emergency=True)

        self.assertTrue(result.ok, result.result)
        self.assertTrue(all(sdk.enable_status))
        self.assertGreaterEqual(max(sdk.enable_timeouts), 5.0)

    def test_post_reset_feedback_rejects_emergency_status(self) -> None:
        robot, sdk = self.make_robot()
        sdk.arm_status_value = 1
        with self.assertRaisesRegex(RuntimeError, "post-reset non-emergency feedback"):
            robot._wait_post_reset_feedback(timeout=0.05)

    def test_recovery_status_is_reported_after_failure(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "EMERGENCY_DAMPING"
        robot._emergency_latched = True
        robot._reconnect_after_emergency_reset = lambda: None
        robot._wait_post_reset_feedback = lambda timeout: (_ for _ in ()).throw(
            RuntimeError("post-reset non-emergency feedback was not confirmed")
        )
        result = robot.enter_freedrive("test recovery status", recover_emergency=True)
        self.assertFalse(result.ok)
        self.assertEqual(robot.get_control_state()["freedrive_recovery"]["state"], "failed")

    def test_gripper_grip_sends_command_without_arm_motion_mode_or_leader_hold(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot._freedrive_backend = None

        result = robot.command_gripper("grip", force_n=1.0)

        self.assertTrue(result["ok"], result)
        self.assertEqual(sdk.gripper.commands[-1], (0.0, 1.0))
        self.assertEqual(sdk.motion_mode_calls, 0)
        self.assertFalse(robot.get_gripper_hold_state()["active"])
        self.assertFalse(robot.get_gripper_hold_state()["supported"])

    def test_gripper_commands_are_rejected_during_drag_teach(self) -> None:
        robot, sdk = self.make_robot()
        robot._freedrive_backend = "drag_teach"
        sdk.drag_teach = True

        with self.assertRaisesRegex(RuntimeError, "unavailable during NERO drag teaching"):
            robot.command_gripper("position", width_m=0.01, force_n=1.0)

    def test_freedrive_ignores_legacy_preserve_request_and_uses_leader(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot.command_gripper("grip", force_n=1.0)
        sdk.events.clear()

        result = robot.enter_freedrive("test grip preservation", preserve_gripper=True)

        self.assertTrue(result.ok, result.result)
        self.assertIn("leader", sdk.events)
        self.assertEqual(result.result["gripper_policy"], "unavailable_in_official_leader_mode")
        self.assertTrue(result.result["legacy_preserve_request_ignored"])
        self.assertEqual(robot.get_control_state()["mode"], "FREEDRIVE")

    def test_gripper_command_never_creates_restore_target(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot._freedrive_backend = None
        robot.command_gripper("grip", force_n=1.0)

        self.assertFalse(robot.get_gripper_hold_state()["active"])

    def test_freedrive_does_not_read_or_recover_unhealthy_gripper(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot.command_gripper("grip", force_n=1.0)
        sdk.gripper.enabled = False
        sdk.events.clear()

        result = robot.enter_freedrive("test unhealthy gripper", preserve_gripper=True)

        self.assertTrue(result.ok, result.result)
        self.assertIn("leader", sdk.events)
        self.assertNotIn("gripper", sdk.events)
        self.assertNotIn("gripper-enable-clear", sdk.events)
        self.assertFalse(sdk.gripper.enabled)
        self.assertEqual(robot.get_control_state()["mode"], "FREEDRIVE")

    def test_freedrive_without_preservation_uses_leader_mode(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        sdk.events.clear()

        result = robot.enter_freedrive("test drag teach", preserve_gripper=False)

        self.assertTrue(result.ok, result.result)
        self.assertEqual(robot.get_control_state()["mode"], "FREEDRIVE")
        self.assertEqual(robot._freedrive_backend, "leader")
        self.assertIn("leader", sdk.events)
        self.assertNotIn("enable", sdk.events)
        self.assertEqual(result.result["leader_enable_action"], "skipped_already_enabled")

    def test_leader_has_no_gripper_hold_monitor(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot._freedrive_backend = None
        grip = robot.command_gripper("grip", force_n=1.0)
        self.assertTrue(grip["ok"], grip)
        robot._control_mode = "FREEDRIVE"
        robot._freedrive_backend = "leader"
        sdk.events.clear()

        result = robot.monitor_gripper_hold()

        self.assertIsNone(result)
        self.assertNotIn("gripper", sdk.events)
        self.assertNotIn("gripper-enable-clear", sdk.events)

    def test_gripper_rejects_invalid_parameters(self) -> None:
        robot, _ = self.make_robot()
        with self.assertRaises(ValueError):
            robot.command_gripper("position", width_m=float("nan"), force_n=1.0)
        with self.assertRaises(ValueError):
            robot.command_gripper("position", width_m=0.02, force_n=3.1)
        with self.assertRaises(ValueError):
            robot.command_gripper("unsupported", width_m=0.02, force_n=1.0)

    def test_gripper_zero_force_clears_hold_and_disables_drive(self) -> None:
        robot, sdk = self.make_robot()
        robot.command_gripper("grip", force_n=1.0)

        result = robot.release_gripper_zero_force()

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["zero_force"])
        self.assertFalse(sdk.gripper.enabled)
        self.assertFalse(robot.get_gripper_hold_state()["active"])

    def test_gripper_teaching_friction_round_trip_in_hold(self) -> None:
        robot, sdk = self.make_robot()
        robot._control_mode = "HOLD"
        robot._freedrive_backend = None

        result = robot.set_gripper_teaching_friction(1)

        self.assertTrue(result["ok"], result)
        self.assertEqual(sdk.gripper.teaching_friction, 1)
        self.assertEqual(result["after"]["params"]["teaching_friction"], 1)

    def test_gripper_command_recovers_from_zero_force(self) -> None:
        robot, sdk = self.make_robot()
        zero_force = robot.release_gripper_zero_force()
        self.assertTrue(zero_force["ok"], zero_force)

        result = robot.command_gripper("grip", force_n=1.0)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["recovered_from_disabled"])
        self.assertTrue(sdk.gripper.enabled)
        self.assertEqual(sdk.gripper.events[-1], "gripper-enable-clear")
        self.assertFalse(robot.get_gripper_hold_state()["active"])

    def test_gripper_command_recovers_when_disabled_feedback_is_absent(self) -> None:
        robot, sdk = self.make_robot()
        zero_force = robot.release_gripper_zero_force()
        self.assertTrue(zero_force["ok"], zero_force)
        sdk.gripper.hide_feedback_when_disabled = True

        result = robot.command_gripper("grip", force_n=1.0)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["recovered_from_disabled"])
        self.assertTrue(sdk.gripper.enabled)
        self.assertIn("gripper-enable-clear", sdk.gripper.events)


if __name__ == "__main__":
    unittest.main()
