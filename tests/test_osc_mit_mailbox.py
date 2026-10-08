"""MIT mailbox scheduling and stop barriers, without a CAN connection.

Virtual milliseconds make 50 Hz input and delayed control/TX schedules
reproducible. A separate event-driven test overlaps real Python threads.
The actual OSC, reference maintainer and per-joint hardware checks are used.
"""
from __future__ import annotations

import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nero_backend.osc_impedance import ImpedanceHardware
from supervisor.authority import ArmWriter, ControlSupervisor, HardwareTxOwner, ServoMode
from tests import test_osc_direct_cpv as cpv_tests
from tests.test_osc_impedance import CONFIG, ROOT, Worker, gravity
from motion.osc_impedance import ImpedanceOutput


class MitMailboxSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = cpv_tests.DirectCpvTests("test_hardware_integrates_acceleration_limited_velocity_once")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.servo = self.fixture.servo
        self.servo.solver.discard_before_epoch = Mock()
        self.servo.solver.close = Mock()
        self.servo.session["client_id"] = "test"
        self.servo._accepting_targets = True
        self.servo._target_pose = copy.deepcopy(cpv_tests.POSE)
        self.servo.target_generation = self.servo.command["target_generation"]
        self.output = ImpedanceOutput(ROOT, CONFIG, [-1] * 7, [1] * 7, worker=Worker())
        self.servo.impedance_output = self.output
        self.addCleanup(self.output.close)
        self.fixture.raw_velocity = [.05] * 7
        self.authority = ControlSupervisor()
        self.authority.transition(ArmWriter.SERVO, ServoMode.TRACKING, "test session")
        self.frames = []
        self.frame_hook = lambda index: None
        backend = SimpleNamespace(robot=SimpleNamespace(move_mit=self.move),
            sdk_config={"firmware": "v120"}, _command_lock=threading.RLock(),
            _control_mode="OSC_MIT", _cpv_stream_started=False,
            send_cpv_position=Mock(side_effect=AssertionError("CPV/MIT mixed send")))
        self.hardware = ImpedanceHardware(backend)
        self.hardware.active = True
        self.hardware.config = {"joint_limits": {"lower_rad": [-1] * 7, "upper_rad": [1] * 7}}
        backend.send_impedance_command = lambda command, execute_guard: self.hardware.send(command, execute_guard)
        self.owner = HardwareTxOwner(backend)
        self.owner.close()  # Test controls dispatch scheduling, not a second CAN writer.
        self.fixture.hardware.servo_transport_diagnostics.side_effect = self.owner.cpv_diagnostics
        self.fixture.hardware.revoke_servo_targets.side_effect = self.owner.revoke_cpv_before_generation
        self.sequence = 1

    def move(self, **frame):
        self.frames.append(frame)
        self.frame_hook(frame["joint_index"])

    def input(self):
        self.sequence += 1
        target = copy.deepcopy(cpv_tests.POSE)
        target["position_m"][0] += self.sequence * .00001
        return self.servo.submit_absolute_target({"client_id": "test", "session_id": "test",
            "sequence": self.sequence, "payload": {"target_pose": target}}, mode="track_tcp")

    def ready(self):
        # Fresh RX/gravity sample at each virtual control cycle; expiry is
        # exercised separately, not left to the host scheduler's test timing.
        self.output.worker.result = gravity(q=self.fixture.q)
        self.fixture.step()
        return copy.deepcopy(self.fixture.hardware.publish_servo_position.call_args.args[0])

    def offer(self, command, epoch=0):
        return self.owner.publish_cpv({**command, "epoch": epoch},
            execute_guard=lambda: self.authority.allows_servo("test", epoch))["mailbox_revision"]

    def send(self, milliseconds):
        entry = self.owner._take_cpv()
        if entry is not None:
            with patch("supervisor.authority.time.perf_counter_ns", return_value=1_000_000_000 + milliseconds * 1_000_000):
                self.owner._dispatch_cpv(entry)
        return entry

    def test_50hz_input_with_solver_and_send_delay_never_starves_ready_samples(self):
        for solve_delay, send_period, phase in ((5, 11, 7), (35, 37, 13), (65, 71, 19)):
            with self.subTest(solve_delay=solve_delay, send_period=send_period, phase=phase):
                # Independent fixture/reference per schedule, no ack rollback.
                case = MitMailboxSchedulingTests()
                case.setUp()
                try:
                    completions, sends = [], []
                    newest_ready = None
                    for ms in range(1201):
                        if ms % 20 == 0:
                            case.input()  # 50 Hz, independent of solve/TX phase.
                        if ms % 40 == phase:
                            completions.append((ms + solve_delay, case.ready()))
                        while completions and completions[0][0] <= ms:
                            _, command = completions.pop(0)
                            newest_ready = case.offer(command)
                        if ms % send_period == 3:
                            entry = case.send(ms)
                            if entry:
                                sends.append(ms)
                                self.assertEqual(entry["mailbox_revision"], newest_ready)
                                self.assertEqual(case.owner.cpv_diagnostics()["last_result"]["status"], "sent")
                    diagnostic = case.owner.cpv_diagnostics()
                    self.assertGreaterEqual(len(sends), 12)
                    self.assertLessEqual(max(b - a for a, b in zip(sends, sends[1:])), 2 * max(40, send_period))
                    self.assertLessEqual(max(b - a for a, b in zip(sends, sends[1:])), 200)
                    self.assertEqual(diagnostic["revoked_count"], 0)
                    self.assertEqual(diagnostic["failed_count"], 0)
                    self.assertEqual(diagnostic["generation_barriers"], {})
                    case.fixture.hardware.revoke_servo_targets.assert_not_called()
                finally:
                    case.doCleanups()

    def test_b_received_preserves_a_until_b_ready_then_overwrites_it(self):
        a = self.ready()
        revision_a = self.offer(a)
        for _ in range(3):
            self.input()
        self.assertEqual(self.owner.cpv_diagnostics()["pending"]["mailbox_revision"], revision_a)
        self.assertEqual(self.owner.wait_cpv_result(revision_a, .001)["status"], "timeout")
        b = self.ready()
        revision_b = self.offer(b)
        self.assertEqual(self.owner.wait_cpv_result(revision_a, .01)["status"], "superseded")
        self.assertEqual(self.send(20)["mailbox_revision"], revision_b)
        self.assertEqual(self.owner.wait_cpv_result(revision_b, .01)["status"], "sent")

    def test_input_and_ready_replacement_do_not_interrupt_an_inflight_batch(self):
        self.offer(self.ready())
        paused, release = threading.Event(), threading.Event()
        def hook(index):
            if index == 2:
                paused.set()
                if not release.wait(1):
                    raise AssertionError("test sender was not released")
        self.frame_hook = hook
        sender = threading.Thread(target=self.send, args=(0,))
        sender.start()
        try:
            self.assertTrue(paused.wait(1))
            self.input()
            self.input()
            next_revision = self.offer(self.ready())
        finally:
            release.set()
            sender.join(1)
        self.assertFalse(sender.is_alive())
        self.assertEqual([frame["joint_index"] for frame in self.frames], list(range(1, 8)))
        self.assertEqual(self.owner.cpv_diagnostics()["last_result"]["status"], "sent")
        self.frame_hook = lambda index: None
        self.send(20)
        self.assertEqual(self.owner.wait_cpv_result(next_revision, .01)["status"], "sent")
        self.assertEqual(self.owner.cpv_diagnostics()["revoked_count"], 0)

    def test_hold_revokes_pending_and_late_old_results_without_resurrection(self):
        old = self.ready()
        revision = self.offer(old)
        self.servo.request_hardware_hold("operator HOLD")
        self.assertEqual(self.owner.wait_cpv_result(revision, .01)["status"], "revoked")
        for ms in (20, 40, 60):
            late = self.offer(old)
            self.send(ms)
            self.assertEqual(self.owner.wait_cpv_result(late, .01)["status"], "revoked")
        self.assertEqual(self.frames, [])
        self.assertIsNone(self.owner.cpv_diagnostics()["impedance_reference"])

    def test_stop_and_ownership_boundaries_interrupt_each_joint_and_old_session(self):
        for reason in ("HOLD", "stop", "fault", "emergency", "session end", "authority handoff", "mode switch"):
            with self.subTest(reason=reason):
                case = MitMailboxSchedulingTests()
                case.setUp()
                try:
                    old = case.ready()
                    case.offer(old)
                    def boundary(index):
                        if index == 2:
                            if reason in {"HOLD", "stop"}:
                                case.servo.request_hardware_hold(reason)
                            else:
                                case.authority.transition(ArmWriter.SAFETY, ServoMode.SUSPENDED, reason, advance_epoch=True)
                                case.owner.advance_epoch(1)
                    case.frame_hook = boundary
                    case.send(0)
                    self.assertEqual(len(case.frames), 2)
                    self.assertEqual(case.owner.cpv_diagnostics()["last_result"]["status"], "failed")
                    self.assertIsNone(case.owner.cpv_diagnostics()["impedance_reference"])
                    late = case.offer(old)
                    case.send(20)
                    self.assertIn(case.owner.wait_cpv_result(late, .01)["status"], {"revoked", "failed"})
                    self.assertEqual(len(case.frames), 2)
                    case.owner.backend.send_cpv_position.assert_not_called()
                finally:
                    case.doCleanups()

    def test_stale_gravity_feedback_and_interval_remain_rejected(self):
        for field, age in (("created_monotonic_ns", .2), ("feedback_monotonic_ns", .2), ("gravity_computed_monotonic_ns", .2)):
            with self.subTest(field=field):
                case = MitMailboxSchedulingTests()
                case.setUp()
                try:
                    command = case.ready()
                    command["impedance_command"][field] = time.monotonic_ns() - int(age * 1e9)
                    revision = case.offer(command)
                    case.send(0)
                    self.assertEqual(case.owner.wait_cpv_result(revision, .01)["status"], "failed")
                    self.assertEqual(case.frames, [])
                finally:
                    case.doCleanups()
        self.offer(self.ready())
        self.send(0)
        before = len(self.frames)
        revision = self.offer(self.ready())
        self.send(201)
        result = self.owner.wait_cpv_result(revision, .01)
        self.assertEqual(result["status"], "failed")
        self.assertIn("invalid MIT control interval", result["error"])
        self.assertEqual(len(self.frames), before)

    def test_new_session_anchors_fresh_feedback_and_rejects_old_session_commands(self):
        old = self.ready()
        self.offer(old)
        self.send(0)
        self.authority.transition(ArmWriter.SERVO, ServoMode.TRACKING, "new session", advance_epoch=True)
        self.owner.advance_epoch(1)
        self.servo.motion_epoch = 1
        self.servo.session["session_id"] = "new-session"
        self.fixture.q = [.15] * 7
        self.output.worker.result = gravity(q=self.fixture.q)
        with self.assertRaises(PermissionError):
            self.input()  # Same client, but old session ID must still fail.
        before = len(self.frames)
        stale = self.offer(old, epoch=0)
        self.send(20)
        self.assertEqual(self.owner.wait_cpv_result(stale, .01)["status"], "revoked")
        self.assertEqual(len(self.frames), before)
        fresh = self.offer(self.ready(), epoch=1)
        self.send(40)
        self.assertEqual(self.owner.wait_cpv_result(fresh, .01)["status"], "sent")
        frames = self.frames[before:]
        self.assertEqual(len(frames), 7)
        self.assertEqual([frame["p_des"] for frame in frames], self.fixture.q)
        self.assertEqual([frame["v_des"] for frame in frames], [0] * 7)
        self.owner.backend.send_cpv_position.assert_not_called()
