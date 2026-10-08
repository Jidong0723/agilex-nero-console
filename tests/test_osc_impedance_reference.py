from __future__ import annotations

import copy
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from motion.osc_impedance_reference import ImpedanceReference, torque_diagnostics
from motion.osc_impedance import ImpedanceOutput, ShadowImpedancePlant
from supervisor.authority import HardwareTxOwner
from tests.test_osc_impedance import reference_command, Worker, ROOT, CONFIG, Q


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ref = ImpedanceReference()
        self.command = reference_command(Q)
        self.ns = 1_000_000_000

    def step(self, velocity=.5, *, commit=True, generation=1, epoch=1, dt=.02):
        self.ns += int(dt * 1e9)
        candidate = self.ref.prepare(self.command, [velocity] * 7,
            started_perf_ns=self.ns, epoch=epoch, generation=generation,
            max_speed=1, max_acceleration=5)
        if commit:
            self.ref.commit(candidate)
        return candidate

    def test_blocked_measured_position_does_not_reset_spring_deflection(self):
        self.step()
        for _ in range(15):
            self.step()
        self.assertGreater(self.ref.snapshot()["position_rad"][0] - Q[0], .1)
        self.assertEqual(self.command["feedback_q_rad"], Q)

    def test_failed_candidate_and_mutated_snapshot_cannot_change_reference(self):
        before = self.step()
        self.step(commit=False)
        view = self.ref.snapshot()
        view["position_rad"][0] = -1
        self.assertEqual(self.ref.snapshot(), before)

    def test_target_generation_change_does_not_reanchor_tracking(self):
        self.step()
        previous = self.step()
        changed = self.step(generation=2)
        self.assertGreater(changed["position_rad"][0], previous["position_rad"][0])

    def test_hold_clears_old_spring_then_remains_anchored(self):
        self.step()
        self.step()
        self.command["reference_action"] = "hold"
        hold = self.step(generation=2)
        self.assertEqual(hold["position_rad"], Q)
        self.assertEqual(hold["velocity_rad_s"], [0] * 7)
        self.command["feedback_q_rad"] = [.2] * 7
        self.assertEqual(self.step(generation=2)["position_rad"], Q)
        self.command["reference_action"] = "track"
        self.assertEqual(self.step(generation=3)["position_rad"], [.2] * 7)

    def test_epoch_change_reanchors_not_old_pose(self):
        self.step()
        self.step()
        self.command["feedback_q_rad"] = [.3] * 7
        self.assertEqual(self.step(epoch=2)["position_rad"], [.3] * 7)

    def test_reverse_is_acceleration_bounded_without_ack_rollback(self):
        self.step()
        previous = self.step()
        reverse = self.step(-.5)
        self.assertLessEqual(abs(reverse["velocity_rad_s"][0] - previous["velocity_rad_s"][0]), .10000001)
        self.assertGreaterEqual(reverse["position_rad"][0], previous["position_rad"][0])
        self.assertLess(self.step(-.5)["position_rad"][0], reverse["position_rad"][0])

    def test_invalid_interval_is_not_clamped_or_caught_up(self):
        self.step()
        before = self.ref.snapshot()
        for dt in (0, -.01, .3):
            with self.subTest(dt=dt), self.assertRaisesRegex(RuntimeError, "interval"):
                self.step(dt=dt, commit=False)
            self.assertEqual(self.ref.snapshot(), before)

    def test_long_valid_interval_uses_actual_time_not_fixed_period(self):
        self.step()
        result = self.step(dt=.04)
        self.assertAlmostEqual(result["velocity_rad_s"][0], .2)
        self.assertAlmostEqual(result["position_rad"][0], .108)

    def test_joint_safety_gate_zero_cannot_be_undone_by_acceleration_shape(self):
        self.step()
        self.step()
        self.command["reference_gate_mask"] = [True] * 7
        self.assertEqual(self.step(0)["velocity_rad_s"], [0] * 7)

    def test_nonzero_safety_gate_cap_cannot_be_undone(self):
        self.step()
        for _ in range(5):
            self.step()
        self.command["reference_gate_mask"] = [True] * 7
        self.assertEqual(self.step(.05)["velocity_rad_s"], [.05] * 7)

    def test_total_estimate_is_distinct_from_feedforward(self):
        self.command["feedback_velocity_rad_s"] = [.2] * 7
        self.command["tau_ff_nm"] = [1] * 7
        result = torque_diagnostics(self.command, [.2] * 7)
        self.assertAlmostEqual(result["estimated_total_nm"][0], 1.29)
        self.assertFalse(result["total_torque_protection_confirmed"])

    def test_damping_estimate_uses_final_desired_velocity(self):
        self.command["feedback_velocity_rad_s"] = [.02] * 7
        self.command["dq_des_rad_s"] = [.04] * 7
        result = torque_diagnostics(self.command, Q)
        self.assertAlmostEqual(result["damping_nm"][0], .3 * (.04 - .02))


class SenderReferenceTests(unittest.TestCase):
    def test_same_batch_position_and_velocity_use_final_limited_reference(self):
        backend = SimpleNamespace(send_impedance_command=Mock(return_value={"ok": True}),
            send_cpv_position=Mock())
        owner = HardwareTxOwner(backend)
        owner.close()
        # Anchor, acceleration cap, safety cap, reversal, HOLD, reanchor.
        proposals = [(.1, "track", False), (.1, "track", False),
            (.01, "track", True), (-.1, "track", False),
            (.1, "hold", False), (.1, "track", False)]
        with patch("supervisor.authority.time.perf_counter_ns",
                side_effect=[1_000_000_000 + i * 20_000_000 for i in range(len(proposals))]):
            previous = None
            for revision, (raw, action, gated) in enumerate(proposals, 1):
                command = reference_command(Q)
                command.update(reference_action=action, reference_gate_mask=[gated] * 7,
                    q_des_rad=[.9] * 7, dq_des_rad_s=[9.0] * 7)
                owner._dispatch_cpv({"output_mode": "impedance", "joint_target_rad": [.9] * 7,
                    "joint_velocity_rad_s": [raw] * 7, "impedance_command": command,
                    "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 2,
                    "mailbox_revision": revision, "epoch": 0, "target_generation": revision,
                    "published_monotonic_ns": time.monotonic_ns()})
                sent = backend.send_impedance_command.call_args.args[0]
                state = owner.cpv_diagnostics()["impedance_reference"]
                self.assertEqual(sent["q_des_rad"], state["position_rad"])
                self.assertEqual(sent["dq_des_rad_s"], state["velocity_rad_s"])
                if state["dt_s"]:
                    for q, prior, v in zip(sent["q_des_rad"], previous, sent["dq_des_rad_s"]):
                        self.assertAlmostEqual(q - prior, v * state["dt_s"])
                else:
                    self.assertEqual(sent["q_des_rad"], Q)
                    self.assertEqual(sent["dq_des_rad_s"], [0] * 7)
                if revision == 2:
                    self.assertAlmostEqual(sent["dq_des_rad_s"][0], .04)
                    self.assertAlmostEqual(sent["q_des_rad"][0] - Q[0], .0008)
                if gated:
                    self.assertEqual(sent["dq_des_rad_s"], [.01] * 7)
                previous = sent["q_des_rad"]
        self.assertEqual(owner.cpv_diagnostics()["failed_count"], 0)
        backend.send_cpv_position.assert_not_called()

    def test_normal_target_refresh_finishes_started_batch_but_hold_interrupts_it(self):
        for interrupt in (False, True):
            guards = []
            def send(command, execute_guard):
                guards.append(execute_guard())
                owner.revoke_cpv_before_generation(0, 2, "target refresh" if not interrupt else "HOLD",
                    interrupt_inflight=interrupt)
                guards.append(execute_guard())
                if not guards[-1]:
                    error = PermissionError("partial batch")
                    error.partial_batch = True
                    raise error
                return {"ok": True}
            backend = SimpleNamespace(send_impedance_command=send)
            owner = HardwareTxOwner(backend)
            try:
                revision = owner.publish_cpv({"output_mode": "impedance", "joint_target_rad": Q,
                    "joint_velocity_rad_s": [.5] * 7, "impedance_command": reference_command(Q),
                    "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5,
                    "target_generation": 1})["mailbox_revision"]
                result = owner.wait_cpv_result(revision, 1)
                self.assertEqual(guards, [True, not interrupt])
                self.assertEqual(result["status"], "failed" if interrupt else "sent")
                self.assertEqual(owner.cpv_diagnostics()["impedance_reference"] is None, interrupt)
                stale = owner.publish_cpv({"output_mode": "impedance", "joint_target_rad": Q,
                    "target_generation": 1, "impedance_command": reference_command(Q)})["mailbox_revision"]
                self.assertEqual(owner.wait_cpv_result(stale, 1)["status"], "revoked")
            finally:
                owner.close()

    def test_partial_send_failure_does_not_commit_or_fallback_to_cpv(self):
        backend = SimpleNamespace(send_impedance_command=Mock(side_effect=RuntimeError("partial CAN batch")),
            send_cpv_position=Mock())
        owner = HardwareTxOwner(backend)
        try:
            revision = owner.publish_cpv({"output_mode": "impedance", "joint_target_rad": Q,
                "joint_velocity_rad_s": [.5] * 7, "impedance_command": reference_command(Q),
                "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5})["mailbox_revision"]
            self.assertEqual(owner.wait_cpv_result(revision, 1)["status"], "failed")
            self.assertIsNone(owner.cpv_diagnostics()["impedance_reference"])
            revision = owner.publish_cpv({"output_mode": "impedance", "joint_target_rad": Q,
                "joint_velocity_rad_s": [.5] * 7, "impedance_command": reference_command(Q),
                "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5})["mailbox_revision"]
            self.assertEqual(owner.wait_cpv_result(revision, 1)["status"], "failed")
            self.assertEqual(backend.send_impedance_command.call_count, 1)
            backend.send_cpv_position.assert_not_called()
        finally:
            owner.close()

    def test_mit_uses_start_intervals_even_when_log_clock_is_quantized(self):
        backend = SimpleNamespace(send_impedance_command=Mock(return_value={"ok": True}))
        owner = HardwareTxOwner(backend)
        owner.close()
        with patch("supervisor.authority.time.perf_counter_ns", side_effect=[1_000_000_000, 1_020_000_000, 1_040_000_000]), \
                patch("supervisor.authority.time.monotonic_ns", return_value=99):
            for revision in range(1, 4):
                owner._dispatch_cpv({"output_mode": "impedance", "joint_target_rad": Q,
                    "joint_velocity_rad_s": [.5] * 7, "impedance_command": reference_command(Q),
                    "max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5,
                    "mailbox_revision": revision, "epoch": 0, "target_generation": 1,
                    "published_monotonic_ns": 99})
        state = owner.cpv_diagnostics()["impedance_reference"]
        self.assertAlmostEqual(state["position_rad"][0], .106)
        self.assertAlmostEqual(state["dt_s"], .02)
        self.assertEqual(owner.cpv_diagnostics()["failed_count"], 0)


class SimplePlantTests(unittest.TestCase):
    def setUp(self):
        self.output = ImpedanceOutput(ROOT, CONFIG, [-1] * 7, [1] * 7, worker=Worker())
        self.addCleanup(self.output.close)
        self.config = {"max_joint_speed_rad_s": 1, "max_joint_acceleration_rad_s2": 5}

    def command(self, action="track", generation=1):
        command = self.output.build(Q, Q, .02, 1, 1)
        command.update(reference_velocity_rad_s=[.5] * 7, reference_action=action, target_generation=generation)
        return command

    def test_physical_gravity_does_not_use_clipped_feedforward_target(self):
        plant = ShadowImpedancePlant(Q, self.output, self.config)
        command = self.command()
        command["tau_ff_nm"] = [2] * 7
        command["gravity_model_tau_nm"] = [100] * 7
        plant.dispatch(Q, 0)
        q, qd, _ = plant.advance(.02, 0)
        self.assertEqual(q, Q)
        self.assertEqual(qd, [0] * 7)

    def test_shadow_sends_consistent_final_reference_and_velocity_damping(self):
        plant = ShadowImpedancePlant(Q, self.output, self.config)
        with patch("motion.osc_impedance.time.perf_counter_ns", side_effect=[1_000_000_000, 1_020_000_000]):
            self.command()["tau_ff_nm"] = [2] * 7
            plant.dispatch(Q, 0)
            plant.advance(.02, 0)
            previous = plant.command["q_des_rad"][:]
            self.command()["tau_ff_nm"] = [2] * 7
            plant.dispatch(Q, .02)
            plant.advance(.002, .02)
        self.assertEqual(plant.command["dq_des_rad_s"], [.1] * 7)
        for q, prior, v in zip(plant.command["q_des_rad"], previous, plant.command["dq_des_rad_s"]):
            self.assertAlmostEqual(q - prior, v * .02)
        # One substep, zero physical/FF mismatch and initial measured velocity:
        # nonzero desired velocity contributes the kd * dq_d term.
        expected = (3.5 * .002 + .3 * .1) / .25 * .002
        self.assertAlmostEqual(plant.qd[0], expected)

    def test_static_friction_allows_equilibrium_without_fault(self):
        plant = ShadowImpedancePlant(Q, self.output, {**self.config, "impedance_coulomb_friction_nm": 1})
        self.command()
        plant.dispatch(Q, 0)
        self.assertEqual(plant.advance(.02, 0)[1], [0] * 7)

    def test_constraint_release_moves_continuously_not_by_reference_backlog(self):
        plant = ShadowImpedancePlant(Q, self.output, {**self.config, "impedance_coulomb_friction_nm": 1})
        c = self.command()
        c["q_des_rad"], c["tau_ff_nm"] = [.3] * 7, [2] * 7
        plant.command = copy.deepcopy(c)
        for _ in range(30):
            self.assertEqual(plant.advance(.02, 0)[0], Q)
        plant.config["impedance_coulomb_friction_nm"] = 0
        q, qd, _ = plant.advance(.02, 0)
        self.assertGreater(q[0], Q[0])
        self.assertLess(q[0] - Q[0], .02 * self.config["max_joint_speed_rad_s"])
        self.assertEqual(plant.command["q_des_rad"], [.3] * 7)

    def test_delay_is_latest_only_and_hold_revokes_pending_old_target(self):
        plant = ShadowImpedancePlant(Q, self.output, {**self.config, "impedance_send_delay_s": .04})
        self.command()
        plant.dispatch(Q, 0)
        plant.advance(.02, .02)
        self.assertEqual(plant.count, 0)
        self.command("hold", 2)
        plant.dispatch(Q, .02)
        plant.advance(.02, .04)
        self.assertEqual(plant.count, 0)
        plant.advance(.02, .061)
        self.assertEqual(plant.count, 1)
        self.assertEqual(plant.reference.snapshot()["action"], "hold")

    def test_shadow_target_generation_change_again_revokes_delayed_batch(self):
        plant = ShadowImpedancePlant(Q, self.output, {**self.config, "impedance_send_delay_s": .04})
        with patch("motion.osc_impedance.time.perf_counter_ns", side_effect=[
                1_000_000_000 + i * 20_000_000 for i in range(20)]):
            for i in range(20):
                self.command(generation=i + 1)
                plant.dispatch(Q, i * .02)
                plant.advance(.02, i * .02)
        self.assertEqual(plant.count, 0)
        self.assertIsNone(plant.reference.snapshot())

    def test_shadow_rejects_expired_gravity_before_reference_commit(self):
        plant = ShadowImpedancePlant(Q, self.output, self.config)
        command = self.command()
        command["gravity_computed_monotonic_ns"] = time.monotonic_ns() - 200_000_000
        plant.dispatch(Q, 0)
        with self.assertRaisesRegex(RuntimeError, "gravity.*expired"):
            plant.advance(.02, 0)
        self.assertIsNone(plant.reference.snapshot())
