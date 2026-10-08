from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from motion.osc import OscRuntime, _OperationalSpaceServo
from motion.osc_impedance import ImpedanceOutput
from motion.osc_impedance_dynamics import GravityFeedforwardManager, NeroJointMapping
from motion.osc_output import OutputSelection
from nero_backend.osc_impedance import ImpedanceHardware
from nero_backend.robot import NeroRobot
from supervisor.authority import ArmWriter, ControlSupervisor, HardwareTxOwner, ServoMode
from supervisor.control import LeaseManager, OperationalSpaceController
from scripts.nero_control_server import ControlRequestHandler
from tests import test_osc_direct_cpv as cpv_tests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/osc.json").read_text(encoding="utf-8"))
Q = [0.1] * 7


def gravity(q=Q, age=0, torque=2.0):
    return {"ok": True, "q_actual_rad": list(q), "tau_gravity_urdf_nm": [torque] * 7,
        "tau_gravity_sdk_nm": [torque] * 7, "model_revision": "test-urdf",
        "sample_id": 1, "computed_monotonic_ns": time.monotonic_ns() - int(age * 1e9)}


def reference_command(q):
    return {"q_des_rad": list(q), "feedback_q_rad": list(q),
        "reference_limits": {"lower_rad": [-1] * 7, "upper_rad": [1] * 7},
        "kp": [3.5] * 7, "kd": [.3] * 7, "tau_ff_nm": [0] * 7}


class Worker:
    def __init__(self):
        self.result = gravity()
        self.closed = False

    def start(self): pass
    def request_gravity(self, *args): pass
    def latest_result(self): return self.result
    def health(self): return {"thread_alive": not self.closed}
    def close(self): self.closed = True


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.worker = Worker()
        self.output = ImpedanceOutput(ROOT, CONFIG, [-1] * 7, [1] * 7, worker=self.worker)
        self.addCleanup(self.output.close)

    def test_lx_first_frame_has_measured_anchor_fresh_model_and_zero_slewed_torque(self):
        command = self.output.start(Q)
        self.assertEqual(command["q_des_rad"], Q)
        self.assertEqual(command["tau_ff_nm"], [0] * 7)
        self.assertEqual(command["gravity"]["gravity_state"], "FRESH")
        self.assertEqual(command["gravity"]["gravity_scale"], 1.3)
        self.assertEqual(command["kp"], [3.5] * 7)
        self.assertEqual(command["kd"], [0.3] * 7)

    def test_position_passes_through_but_mit_velocity_is_zero(self):
        target = [0.101] * 7
        command = self.output.build(target, Q, .02, 2, 3)
        self.assertEqual(command["q_des_rad"], target)
        self.assertEqual(command["dq_des_rad_s"], [0] * 7)
        self.assertEqual(command["tau_ff_nm"], [2] * 7)
        self.assertEqual(command["motion_epoch"], 3)

    def test_torque_limits_and_slew(self):
        self.worker.result = gravity(torque=50)
        a = self.output.build(Q, Q, .02, 1, 0)
        b = self.output.build(Q, Q, .2, 2, 0)
        self.assertEqual(a["tau_ff_nm"], [2] * 7)
        self.assertEqual(b["tau_ff_nm"], [16] * 7)
        self.assertIn("torque_limit", b["gravity"]["limit_reason"])

    def test_recent_cached_gravity_is_allowed_with_diagnostics(self):
        self.worker.result = gravity(age=.075)
        self.assertEqual(self.output.build(Q, Q, .02, 1, 0)["gravity"]["gravity_state"], "HOLD_LAST_VALID")

    def test_invalid_gravity_is_not_silently_replaced_by_zero(self):
        for result in (None, {"ok": False}, gravity(age=.2), gravity(q=[.2] * 7), gravity(torque=float("nan")), gravity(age=-1)):
            with self.subTest(result=result):
                self.worker.result = result
                with self.assertRaisesRegex(RuntimeError, "invalid impedance gravity"):
                    self.output.build(Q, Q, .02, 1, 0)

    def test_bad_targets_are_rejected(self):
        for target in ([2] * 7, [float("nan")] * 7, [0] * 6):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.output.build(target, Q, .02, 1, 0)

    def test_joint_torque_sign_mapping(self):
        mapping = NeroJointMapping([{"sdk_index": i + 1, "sign": -1 if i == 1 else 1,
            "zero_offset_rad": .1, "torque_sign": -1 if i == 1 else 1} for i in range(7)])
        self.assertAlmostEqual(mapping.to_urdf_q([.2] * 7)[1], -.1)
        self.assertEqual(mapping.to_sdk_tau([2] * 7)[1], -2)

    def test_real_isolated_pinocchio(self):
        output = ImpedanceOutput(ROOT, CONFIG, [-3.14] * 7, [3.14] * 7)
        try:
            command = output.start(CONFIG["shadow_initial_joints_rad"])
            self.assertEqual(command["gravity"]["gravity_state"], "FRESH")
            self.assertTrue(output.worker.health()["process_alive"])
            self.assertNotIn("pinocchio", sys.modules)
        finally:
            output.close()
        self.assertFalse(output.worker.health()["process_alive"])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = copy.deepcopy(CONFIG)
        config["solver"]["urdf"] = str(ROOT / config["solver"]["urdf"])
        with patch("supervisor.logging.AsyncJsonlTraceLogger"), patch.object(_OperationalSpaceServo, "_enable_high_resolution_timer"):
            self.servo = _OperationalSpaceServo(Mock(), Path(self.folder.name), config, Mock())
        self.servo.solver.close = Mock()
        self.runtime = object.__new__(OscRuntime)
        self.runtime._servo = self.servo
        self.exit = Mock(return_value={"ok": True})

    def test_default_and_atomic_persistence_without_motion(self):
        self.assertEqual(self.runtime.output_status()["output_mode"], "cpv")
        self.runtime.select_output_mode("impedance", "test", self.exit)
        self.exit.assert_not_called()
        self.assertEqual(OutputSelection(Path(self.folder.name)).mode, "impedance")
        self.runtime.select_output_mode("cpv", "test", self.exit)
        self.assertEqual(OutputSelection(Path(self.folder.name)).mode, "cpv")
        self.servo.hardware.prepare_osc_hardware.assert_not_called()
        self.servo.hardware.prepare_osc_impedance.assert_not_called()

    def test_corrupt_config_falls_back_to_cpv_with_reason(self):
        selection = self.servo.output_selection
        selection.path.parent.mkdir(exist_ok=True)
        selection.path.write_text('{"mode":"invalid"}', encoding="utf-8")
        restored = OutputSelection(Path(self.folder.name))
        self.assertEqual(restored.mode, "cpv")
        self.assertIn("invalid", restored.error)

    def test_running_braking_fault_and_starting_reject_switch(self):
        self.servo.session = {"state": "ACTIVE", "client_id": "test", "execution_mode": "shadow"}
        for state in ("RUNNING", "BRAKING", "FAULT"):
            self.servo.trajectory_state = state
            with self.subTest(state=state), self.assertRaises(PermissionError):
                self.runtime.select_output_mode("impedance", "test", self.exit)
        self.servo.trajectory_state = "HOLD_READY"
        self.servo.session["state"] = "STARTING"
        with self.assertRaises(PermissionError):
            self.runtime.select_output_mode("impedance", "test", self.exit)
        self.assertEqual(self.servo.output_selection.mode, "cpv")

    def test_unsettled_and_other_client_reject_switch(self):
        self.servo.session = {"state": "ACTIVE", "client_id": "test", "execution_mode": "shadow"}
        self.servo._initialize_trajectory(Q)
        self.servo.trajectory["velocity_rad_s"] = [.1] * 7
        with self.assertRaises(PermissionError):
            self.runtime.select_output_mode("impedance", "test", self.exit)
        self.servo._initialize_trajectory(Q)
        with self.assertRaisesRegex(PermissionError, "another client"):
            self.runtime.select_output_mode("impedance", "other", self.exit)

    def test_held_session_ends_and_old_targets_are_not_restored(self):
        self.servo.session = {"state": "ACTIVE", "client_id": "test", "execution_mode": "shadow"}
        self.servo._initialize_trajectory(Q)
        self.servo.command = {"old": True}
        self.runtime.select_output_mode("impedance", "test", self.exit)
        self.assertIsNone(self.servo.session)
        self.assertIsNone(self.servo.command)
        self.assertFalse(self.servo._accepting_targets)

    def test_same_mode_does_not_end_session_or_revoke_targets(self):
        self.servo.session = {"state": "ACTIVE", "client_id": "test", "execution_mode": "shadow"}
        self.servo._initialize_trajectory(Q)
        self.servo.command = {"old": True}
        self.servo._accepting_targets = True
        self.runtime.select_output_mode("cpv", "test", self.exit)
        self.assertIsNotNone(self.servo.session)
        self.assertEqual(self.servo.command, {"old": True})
        self.assertTrue(self.servo._accepting_targets)
        self.exit.assert_not_called()

    def test_selection_has_a_newer_sequence_than_the_stopped_snapshot(self):
        stopped_sequences = []
        original = self.servo.stop_session
        def stop(reason):
            result = original(reason)
            stopped_sequences.append(self.servo.state_sequence)
            return result
        self.servo.stop_session = stop
        self.runtime.select_output_mode("impedance", "test", self.exit)
        self.assertGreater(self.servo.state_sequence, stopped_sequences[-1])

    def test_shadow_idle_remembers_execution_context_without_a_session(self):
        self.servo.last_execution_mode = "shadow"
        self.assertIsNone(self.servo.session)
        self.assertEqual(self.runtime.output_status()["output_switch"]["execution_mode"], "shadow")

    def test_failed_hardware_exit_does_not_save_or_restore_targets(self):
        self.servo.session = {"state": "ACTIVE", "client_id": "test", "execution_mode": "hardware"}
        self.servo._initialize_trajectory(Q)
        self.servo.command = {"old": True}
        self.exit.return_value = {"ok": False}
        with self.assertRaisesRegex(RuntimeError, "hardware exit failed"):
            self.runtime.select_output_mode("impedance", "test", self.exit)
        self.assertEqual(self.servo.output_selection.mode, "cpv")
        self.assertFalse(self.servo.output_selection.path.exists())
        self.assertIsNone(self.servo.command)

    def test_failed_atomic_replace_keeps_old_selection(self):
        selection = self.servo.output_selection
        selection.save("cpv")
        with patch("motion.osc_output.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                selection.save("impedance")
        self.assertEqual(selection.mode, "cpv")
        self.assertEqual(OutputSelection(Path(self.folder.name)).mode, "cpv")

    def test_default_cpv_does_not_load_impedance_runtime(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from motion.osc import OscRuntime; assert 'motion.osc_impedance' not in sys.modules; assert 'motion.osc_impedance_dynamics' not in sys.modules; assert 'pinocchio' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.frames = []
        self.revision = 1
        self.mode_feedback = 0x05
        self.confirm = True
        self.sdk = SimpleNamespace(
            OPTIONS=SimpleNamespace(MOTION_MODE=SimpleNamespace(MIT="mit")),
            get_auto_set_motion_mode_enabled=lambda: True,
            set_auto_set_motion_mode_enabled=Mock(),
            set_motion_mode=Mock(side_effect=lambda mode: self.frames.append(("mode", mode))),
            set_follower_mode=Mock(side_effect=self.follower), move_mit=self.move)
        self.backend = SimpleNamespace(robot=self.sdk, _command_lock=threading.RLock(),
            _cpv_stream_started=False, _control_mode="HOLD", _arm_status_observer_installed=True,
            config={}, sdk_config={"firmware": "v120"}, motion_config={},
            _enable_all_with_retry=lambda **kw: (True, {}),
            _read_stable_follower_joints=lambda **kw: list(Q),
            arm_status_snapshot=lambda: {"revision": self.revision, "ctrl_mode": 1, "mode_feedback": self.mode_feedback})
        self.hardware = ImpedanceHardware(self.backend)
        self.output = ImpedanceOutput(ROOT, CONFIG, [-1] * 7, [1] * 7, worker=Worker())
        self.command = self.output.start(Q)

    def move(self, **kw):
        self.frames.append(("mit", kw))
        if kw["joint_index"] == 7 and self.confirm:
            self.revision += 1
            self.mode_feedback = 0x06

    def follower(self):
        self.frames.append(("follower", None))
        self.revision += 1
        self.mode_feedback = 0x05

    def enter(self):
        return self.hardware.enter(lambda q: self.output.entry(q), {**self.output.config, "enter_timeout_s": .01})

    def test_mode_request_first_frame_then_new_confirmation(self):
        result = self.enter()
        self.assertTrue(result["ok"])
        self.assertEqual(self.frames[0], ("mode", "mit"))
        self.assertEqual(len([x for x in self.frames if x[0] == "mit"]), 7)
        self.assertTrue(all(x[1]["p_des"] == .1 for x in self.frames if x[0] == "mit"))
        self.assertGreater(result["impedance_mode_entry"]["mode_feedback"]["revision"], 1)
        self.assertTrue(self.hardware.active)

    def test_mit_hardware_preserves_each_joint_desired_velocity(self):
        self.enter()
        self.assertTrue(all(frame["v_des"] == 0 for kind, frame in self.frames if kind == "mit"))
        command = self.output.build(Q, Q, .02, 1, 0)
        velocities = [.04, -.03, .02, 0, -.01, .05, -.06]
        command["dq_des_rad_s"] = velocities
        before = len(self.frames)
        self.hardware.send(command)
        frames = [frame for kind, frame in self.frames[before:] if kind == "mit"]
        self.assertEqual([frame["v_des"] for frame in frames], velocities)
        self.assertEqual([frame["p_des"] for frame in frames], Q)

    def test_nonfinite_desired_velocity_rejected_before_any_frame(self):
        self.enter()
        for invalid in (float("nan"), float("inf")):
            command = self.output.build(Q, Q, .02, 1, 0)
            command["dq_des_rad_s"][2] = invalid
            before = len(self.frames)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                self.hardware.send(command)
            self.assertEqual(len(self.frames), before)

    def test_out_of_protocol_velocity_rejected_not_silently_clamped(self):
        self.enter()
        command = self.output.build(Q, Q, .02, 1, 0)
        command["dq_des_rad_s"][0] = 46
        before = len(self.frames)
        with self.assertRaisesRegex(ValueError, "velocity outside"):
            self.hardware.send(command)
        self.assertEqual(len(self.frames), before)

    def test_mode_timeout_performs_official_exit(self):
        self.confirm = False
        with self.assertRaisesRegex(RuntimeError, "confirmation timed out"):
            self.enter()
        self.assertFalse(self.hardware.active)
        self.assertEqual(self.frames[-1][0], "follower")
        self.assertEqual(self.backend._control_mode, "HOLD")

    def test_exit_failure_blocks_cpv(self):
        self.enter()
        self.sdk.set_follower_mode.side_effect = RuntimeError("CAN failure")
        with self.assertRaises(RuntimeError):
            self.hardware.exit("test")
        self.assertTrue(self.hardware.active)
        self.assertEqual(self.backend._control_mode, "FAULT")
        fake = SimpleNamespace(impedance_stream_active=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "MIT exit"):
            NeroRobot._enter_cpv_stream_if_needed(fake)

    def test_exit_sends_measured_anchor_before_follower_and_mode_confirmation(self):
        self.enter()
        hold = self.output.build(Q, Q, .02, 1, 0)
        before = len(self.frames)
        result = self.hardware.exit("operator HOLD", lambda: hold)
        self.assertTrue(result["confirmed"])
        self.assertEqual([kind for kind, _ in self.frames[before:]], ["mit"] * 7 + ["follower"])
        self.assertTrue(all(frame["p_des"] == .1 for kind, frame in self.frames[before:] if kind == "mit"))

    def test_unconfirmed_hold_frame_keeps_exit_faulted_even_if_follower_replies(self):
        self.enter()
        hold = self.output.build([.2] * 7, Q, .02, 1, 0)
        with self.assertRaisesRegex(RuntimeError, "hold frame unconfirmed"):
            self.hardware.exit("HOLD", lambda: hold)
        self.assertTrue(self.hardware.active)
        self.assertEqual(self.backend._control_mode, "FAULT")

    def test_non_mit_feedback_alone_does_not_prove_actual_stop(self):
        self.enter()
        self.backend.read_cached_osc_feedback = lambda: {"joint_velocity_rad_s": [.1] * 7}
        with self.assertRaisesRegex(RuntimeError, "not settled"):
            self.hardware.exit("HOLD")
        self.assertTrue(self.hardware.active)

    def test_queued_gravity_and_feedback_expiry_rejected(self):
        self.enter()
        for key, seconds in (("gravity_computed_monotonic_ns", .2), ("created_monotonic_ns", .2), ("feedback_monotonic_ns", .2)):
            command = copy.deepcopy(self.command)
            command[key] = time.monotonic_ns() - int(seconds * 1e9)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "expired"):
                self.hardware.send(command)

    def test_partial_batch_checks_revocation_per_joint(self):
        self.enter()
        count = len(self.frames)
        with self.assertRaises(PermissionError):
            self.hardware.send(self.command, guard=lambda: len(self.frames) - count < 2)
        self.assertEqual(len(self.frames) - count, 2)

    def test_cpv_active_rejects_mit_entry(self):
        self.backend._cpv_stream_started = True
        with self.assertRaises(RuntimeError):
            self.enter()
        self.assertEqual(self.frames, [])

    def test_stop_uses_mit_exit_not_cpv(self):
        fake = SimpleNamespace(robot=self.sdk, impedance_stream_active=lambda: True,
            exit_impedance_mode=Mock(return_value={"ok": True}), _stop_cpv_stream=Mock())
        self.assertTrue(NeroRobot.stop_cpv_for_mode_transition(fake)["ok"])
        fake.exit_impedance_mode.assert_called_once()
        fake._stop_cpv_stream.assert_not_called()


class MailboxTests(unittest.TestCase):
    def test_mode_boundary_discards_previous_output_anchor(self):
        backend = SimpleNamespace(send_cpv_position=Mock(return_value={}), send_impedance_command=Mock(return_value={}))
        owner = HardwareTxOwner(backend)
        try:
            for mode, target in (("cpv", [0] * 7), ("impedance", [.5] * 7), ("cpv", [.2] * 7)):
                revision = owner.publish_cpv({"joint_target_rad": target, "joint_velocity_rad_s": [0] * 7,
                    "output_mode": mode, "impedance_command": reference_command(target),
                    "epoch": owner.epoch(), "max_joint_speed_rad_s": .01,
                    "max_joint_acceleration_rad_s2": .01})["mailbox_revision"]
                result = owner.wait_cpv_result(revision, 1)
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["joint_target_rad"], target)
        finally:
            owner.close()

    def test_impedance_and_cpv_share_one_sender_and_epoch(self):
        backend = SimpleNamespace(send_cpv_position=Mock(return_value={}), send_impedance_command=Mock(return_value={}))
        owner = HardwareTxOwner(backend)
        try:
            command = {"joint_target_rad": Q, "joint_velocity_rad_s": [0] * 7,
                "output_mode": "impedance", "impedance_command": reference_command(Q),
                "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5,
                "epoch": owner.epoch(), "target_generation": 1}
            revision = owner.publish_cpv(command)["mailbox_revision"]
            self.assertEqual(owner.wait_cpv_result(revision, 1)["status"], "sent")
            self.assertEqual(backend.send_impedance_command.call_args.args[0]["q_des_rad"], Q)
            backend.send_cpv_position.assert_not_called()
            command["epoch"] -= 1
            revision = owner.publish_cpv(command)["mailbox_revision"]
            self.assertEqual(owner.wait_cpv_result(revision, 1)["status"], "revoked")
            self.assertEqual(backend.send_impedance_command.call_count, 1)
        finally:
            owner.close()

    def test_impedance_old_target_generation_revoked(self):
        backend = SimpleNamespace(send_impedance_command=Mock())
        owner = HardwareTxOwner(backend)
        try:
            owner.revoke_cpv_before_generation(owner.epoch(), 2, "HOLD")
            revision = owner.publish_cpv({"joint_target_rad": Q, "output_mode": "impedance",
                "impedance_command": {}, "epoch": owner.epoch(), "target_generation": 1})["mailbox_revision"]
            self.assertEqual(owner.wait_cpv_result(revision, 1)["status"], "revoked")
            backend.send_impedance_command.assert_not_called()
        finally:
            owner.close()


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.controller = object.__new__(OperationalSpaceController)
        self.controller.supervisor = ControlSupervisor()
        self.controller.supervisor.transition(ArmWriter.SERVO, ServoMode.HOLDING, "test")
        self.controller._osc = Mock()
        self.controller._osc.output_status.return_value = {"output_switch": {"allowed": True, "reason": None}}
        self.controller._osc.status.return_value = {"session": None}
        self.controller._handoff_lock = threading.RLock()
        self.controller._jobs_lock = threading.RLock()
        self.controller._jobs = {}
        self.controller._active_action = None
        self.controller.leases = LeaseManager()
        self.control = {"connected": True, "mode": "HOLD", "robot": {
            "joint_angles_rad": Q, "tcp_pose": [0] * 6, "arm_status": {"arm_status": 0}}}
        self.controller.status = Mock(side_effect=lambda: {"control": self.control})
        self.controller.osc_state = Mock(return_value={"output_mode": "impedance"})
        self.controller._osc_cached_feedback = Mock(return_value={
            "age_s": .01, "joint_angles_rad": Q, "joint_velocity_rad_s": [0] * 7})

    def test_backend_rejects_motion_fault_transition_and_active_action(self):
        for writer, servo_mode in ((ArmWriter.SERVO, ServoMode.TRACKING),
                (ArmWriter.SERVO, ServoMode.STOPPING), (ArmWriter.SAFETY, ServoMode.HOLDING),
                (ArmWriter.MODE_TRANSITION, ServoMode.SUSPENDED)):
            self.controller.supervisor.transition(writer, servo_mode, "test")
            with self.subTest(writer=writer, mode=servo_mode), self.assertRaises(PermissionError):
                self.controller.osc_output_mode("impedance", "test")
        self.controller.supervisor.transition(ArmWriter.SERVO, ServoMode.HOLDING, "test")
        self.controller._active_action = {"id": "moving-action"}
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")
        self.controller._osc.select_output_mode.assert_not_called()

    def test_idle_hardware_requires_fresh_finite_and_settled_feedback(self):
        self.control["connected"] = True
        for feedback in ({"age_s": .3}, {"age_s": None},
                {"age_s": .01, "joint_angles_rad": Q, "joint_velocity_rad_s": [.1] * 7},
                {"age_s": .01, "joint_angles_rad": Q, "joint_velocity_rad_s": [float("nan")] * 7}):
            self.controller._osc_cached_feedback.return_value = feedback
            with self.subTest(feedback=feedback), self.assertRaises(PermissionError):
                self.controller.osc_output_mode("impedance", "test")

    def test_active_lease_rejects_selection(self):
        self.controller.leases.acquire("external")
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")

    def test_idle_does_not_allow_disconnected_freedrive_or_unconfirmed_hold(self):
        for connected, mode in ((False, "HOLD"), (True, "FREEDRIVE"),
                (True, "OSC_CPV"), (True, "OSC_MIT"), (True, "TRANSITIONING")):
            self.control.update(connected=connected, mode=mode)
            with self.subTest(connected=connected, mode=mode), self.assertRaises(PermissionError):
                self.controller.osc_output_mode("impedance", "test")
        self.control.update(connected=True, mode="HOLD")
        self.controller.supervisor.transition(ArmWriter.NONE, ServoMode.SUSPENDED, "hold not confirmed")
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")

    def test_shadow_idle_selection_does_not_require_hardware(self):
        self.control.update(connected=False, mode="DISCONNECTED")
        self.controller._osc.output_status.return_value["output_switch"]["execution_mode"] = "shadow"
        self.controller.osc_output_mode("impedance", "test")
        self.controller._osc.select_output_mode.assert_called_once()

    def test_switch_guard_rechecks_under_transition_lock(self):
        def select(*args, switch_guard, **kw):
            self.control["mode"] = "FREEDRIVE"
            switch_guard()
        self.controller._osc.select_output_mode.side_effect = select
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")

    def test_hold_with_arm_error_or_emergency_latch_cannot_switch(self):
        self.control["robot"]["arm_status"] = {"arm_status": 0, "err_status": {"joint_1": True}}
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")
        self.control["robot"]["arm_status"] = {"arm_status": 0}
        self.control["emergency_latched"] = True
        with self.assertRaises(PermissionError):
            self.controller.osc_output_mode("impedance", "test")
        self.controller._osc.select_output_mode.assert_not_called()

    def test_mit_preparation_ignores_aggregate_age_not_real_feedback_age(self):
        self.controller._status_lock = threading.Lock()
        self.controller._status_cache = (time.monotonic() - 2, {"control": self.control})
        self.assertFalse(self.controller._cached_feedback_readiness()["ok"])
        self.controller.robot = SimpleNamespace(get_control_state=lambda: self.control,
            call=Mock(return_value={"impedance_mode_entry": {"confirmed": True}}))
        self.controller._set_authority = Mock(return_value={"control_epoch": 1})
        self.controller.prepare_osc_impedance(Mock(), {})
        self.controller.robot.call.assert_called_once()

    def test_mit_preparation_rejects_real_stale_feedback_and_hardware_faults(self):
        self.controller._require_operational_control = Mock()
        self.controller.robot = SimpleNamespace(get_control_state=lambda: self.control, call=Mock())
        for age in (.2, float("nan"), -1, None):
            self.controller._osc_cached_feedback.return_value["age_s"] = age
            with self.subTest(age=age), self.assertRaisesRegex(RuntimeError, "fresh seven-joint"):
                self.controller.prepare_osc_impedance(Mock(), {})
        self.controller._osc_cached_feedback.return_value["age_s"] = .01
        for status in ({"arm_status": 1}, {"arm_status": 0, "err_status": {"joint_1": True}}):
            self.control["robot"]["arm_status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "fault-free"):
                self.controller.prepare_osc_impedance(Mock(), {})
        self.control["robot"]["arm_status"] = {"arm_status": 0}
        self.control["emergency_latched"] = True
        with self.assertRaisesRegex(RuntimeError, "fault-free"):
            self.controller.prepare_osc_impedance(Mock(), {})
        self.controller.robot.call.assert_not_called()

    def test_hardware_first_frame_uses_fresh_rx_pose_and_current_epoch(self):
        measured = [.101] * 7
        timestamp = time.monotonic_ns() - 10_000_000
        self.controller._osc.rx_snapshot.return_value = {"age_s": .01, "joints": measured,
            "velocities": [0] * 7, "fresh_received_at_monotonic_ns": timestamp}
        self.controller._require_operational_control = Mock()
        def authority(writer, mode, reason, **kw):
            return self.controller.supervisor.transition(writer, mode, reason,
                advance_epoch=kw.get("advance_epoch", False)).public()
        self.controller._set_authority = authority
        commands = []
        def dispatch(priority, method, factory, config, **kw):
            self.assertEqual(method, "prepare_impedance")
            self.assertTrue(kw["execute_guard"]())
            commands.append(factory(Q))
            return {"impedance_mode_entry": {"confirmed": True}}
        self.controller.robot = SimpleNamespace(get_control_state=lambda: self.control, call=dispatch)
        self.controller.prepare_osc_impedance(lambda q: {"q_des_rad": q}, {})
        self.assertEqual(commands[0]["q_des_rad"], measured)
        self.assertEqual(commands[0]["feedback_monotonic_ns"], timestamp)
        self.assertEqual(commands[0]["motion_epoch"], self.controller.supervisor.snapshot().epoch)

    def test_first_frame_rejects_stale_rx_even_if_sdk_position_is_stable(self):
        self.controller._osc.rx_snapshot.return_value = {"age_s": .2, "joints": Q, "velocities": [0] * 7}
        self.controller._require_operational_control = Mock()
        self.controller._set_authority = Mock(return_value={"control_epoch": 1})
        self.controller.robot = SimpleNamespace(get_control_state=lambda: self.control,
            call=lambda priority, method, factory, config, **kw: factory(Q))
        builder = Mock()
        with self.assertRaisesRegex(RuntimeError, "fresh, settled"):
            self.controller.prepare_osc_impedance(builder, {})
        builder.assert_not_called()

    def test_http_route_forwards_identity_and_returns_clear_rejection(self):
        controller = self.controller
        class Handler(ControlRequestHandler):
            runtime = SimpleNamespace(require_broker=lambda: controller)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            body = json.dumps({"mode": "impedance", "client_id": "browser-test"}).encode()
            with urlopen(Request(base + "/api/osc/output-mode", data=body,
                    headers={"Content-Type": "application/json"}), timeout=2) as response:
                result = json.load(response)
                self.assertTrue(result["data"]["ok"])
                self.assertEqual(result["data"]["state"]["output_mode"], "impedance")
            self.assertEqual(controller._osc.select_output_mode.call_args.args[:2], ("impedance", "browser-test"))
            with urlopen(base + "/", timeout=2) as response:
                page = response.read().decode("utf-8")
                self.assertIn('id="output-cpv"', page)
                self.assertIn('id="output-impedance"', page)
            controller.supervisor.transition(ArmWriter.SERVO, ServoMode.TRACKING, "test")
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/api/osc/output-mode", data=body,
                    headers={"Content-Type": "application/json"}), timeout=2)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(1)


class ServoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = cpv_tests.DirectCpvTests("test_hardware_integrates_acceleration_limited_velocity_once")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.servo = self.fixture.servo
        self.servo.solver.close = Mock()
        self.servo.solver.process = None
        self.servo.solver.python = Path("fake")
        self.worker = Worker()
        self.output = ImpedanceOutput(ROOT, CONFIG, [-1] * 7, [1] * 7, worker=self.worker)
        self.servo.impedance_output = self.output
        self.addCleanup(self.output.close)

    def test_mit_publishes_velocity_without_integrating_a_second_reference(self):
        self.fixture.step()
        command = self.fixture.hardware.publish_servo_position.call_args.args[0]
        self.assertEqual(command["output_mode"], "impedance")
        for value in command["joint_target_rad"]:
            self.assertAlmostEqual(value, .1)
        self.assertEqual(command["joint_velocity_rad_s"], [.5] * 7)
        self.assertEqual(command["impedance_command"]["q_des_rad"], command["joint_target_rad"])
        self.assertEqual(command["impedance_command"]["dq_des_rad_s"], [0] * 7)

    def test_hard_stale_feedback_faults_instead_of_emitting_mit(self):
        self.fixture.feedback_age = .2
        self.fixture.step()
        self.fixture.hardware.publish_servo_position.assert_not_called()
        self.fixture.hardware.trigger_safety_fault.assert_called_once()
        self.assertEqual(self.servo.trajectory_state, "FAULT")

    def test_hold_refreshes_only_the_anchored_position_and_gravity(self):
        self.servo.trajectory_state = "HOLD_READY"
        self.fixture.step()
        command = self.fixture.hardware.publish_servo_position.call_args.args[0]
        self.assertEqual(command["joint_target_rad"], Q)
        self.assertEqual(command["joint_velocity_rad_s"], [0] * 7)
        self.assertEqual(command["impedance_command"]["dq_des_rad_s"], [0] * 7)

    def test_hold_telemetry_error_is_recomputed_not_a_frozen_tracking_sample(self):
        self.servo.trajectory_state = "HOLD_READY"
        self.servo.execution_sample = {"target_tcp": cpv_tests.POSE, "position_error_m": .5}
        self.servo.solver.fk = lambda q: {"position_m": [.101, .2, .3],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        self.fixture.step()
        self.assertAlmostEqual(self.servo.execution_sample["position_error_m"], .001)

    def test_reference_is_read_only_and_pink_solves_from_it_not_actual_position(self):
        reference = {"epoch": self.servo.motion_epoch, "action": "track",
            "position_rad": [.2] * 7, "velocity_rad_s": [.1] * 7}
        self.fixture.hardware.servo_transport_diagnostics.return_value = {"impedance_reference": reference}
        solver = Mock(side_effect=self.fixture.solve_current)
        self.servo.solver.solve_current = solver
        self.fixture.step()
        request = solver.call_args.args[0]
        self.assertEqual(request["joint_angles_rad"], [.2] * 7)
        self.assertEqual(request["measured_joint_angles_rad"], Q)
        self.assertTrue(request["report_measured_error"])
        self.assertEqual(reference["position_rad"], [.2] * 7)
        self.assertEqual(self.servo.trajectory["position_rad"], [.2] * 7)

    def test_moving_measured_joints_do_not_report_hold_ready(self):
        self.servo.trajectory_state = "HOLD_READY"
        original = self.servo._feedback_snapshot
        self.servo._feedback_snapshot = lambda: {**original(), "velocities": [.1] * 7}
        self.fixture.step()
        self.assertEqual(self.servo.trajectory_state, "BRAKING")
        command = self.fixture.hardware.publish_servo_position.call_args.args[0]
        self.assertEqual(command["impedance_command"]["reference_action"], "hold")
        self.fixture.hardware.latch_osc_hold.assert_not_called()

    def test_terminal_hold_factory_clears_old_reference_and_rejects_stale_feedback(self):
        original = self.servo._feedback_snapshot
        self.servo._feedback_snapshot = lambda: {**original(), "age_s": self.fixture.feedback_age}
        self.servo.trajectory["position_rad"] = [.4] * 7
        self.assertEqual(self.servo.impedance_hold_command()["q_des_rad"], Q)
        self.fixture.feedback_age = .2
        with self.assertRaisesRegex(RuntimeError, "fresh feedback"):
            self.servo.impedance_hold_command()

    def test_expired_gravity_in_hold_is_a_terminal_fault(self):
        self.servo.trajectory_state = "HOLD_READY"
        self.worker.result = gravity(age=.3)
        self.fixture.step()
        self.fixture.hardware.publish_servo_position.assert_not_called()
        self.fixture.hardware.trigger_safety_fault.assert_called_once()
        self.assertEqual(self.servo.trajectory_state, "FAULT")

    def test_failed_hold_send_is_not_ignored(self):
        self.servo.trajectory_state = "HOLD_READY"
        self.fixture.hardware.servo_transport_diagnostics.return_value = {"last_result": {"status": "failed"}}
        self.fixture.step()
        self.fixture.hardware.trigger_safety_fault.assert_called_once()
        self.assertEqual(self.servo.trajectory_state, "FAULT")

    def test_shadow_impedance_fault_never_calls_hardware(self):
        self.servo.session["execution_mode"] = "shadow"
        self.worker.result = gravity(age=.3)
        self.fixture.step()
        self.fixture.hardware.trigger_safety_fault.assert_not_called()
        self.assertEqual(self.servo.trajectory_state, "FAULT")

    def test_session_stop_exits_mit_and_closes_gravity_worker(self):
        self.servo.stop_session("operator HOLD")
        self.fixture.hardware.exit_osc_impedance.assert_called_once_with("operator HOLD")
        self.assertTrue(self.worker.closed)
        self.assertIsNone(self.servo.impedance_output)
        self.assertIsNone(self.servo.command)

    def test_exit_failure_still_closes_worker_and_never_restores_old_targets(self):
        self.fixture.hardware.exit_osc_impedance.side_effect = RuntimeError("exit unconfirmed")
        with self.assertRaisesRegex(RuntimeError, "exit unconfirmed"):
            self.servo.stop_session("operator HOLD")
        self.assertTrue(self.worker.closed)
        self.assertIsNone(self.servo.command)
        self.assertIsNone(self.servo.impedance_output)
        self.assertFalse(self.servo._accepting_targets)


if __name__ == "__main__":
    unittest.main()
