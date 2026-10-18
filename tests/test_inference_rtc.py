"""RTC runs against fake cameras/OSC only; these tests never own hardware."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from supervisor.inference_config import prepare_update, project
from supervisor.inference_rtc import Timeline, Target, compatibility, VERSION, ObservationAssembler
from supervisor.pi05_adapter import Pi05InputAdapter
from tests.test_pi05_adapter import _Broker, _InitiallyUnavailablePolicy

ROOT = Path(__file__).parents[1]
METADATA = {"state_dim": 7, "valid_action_dim": 7, "action_dim": 32, "action_horizon": 20,
            "rtc_version": VERSION, "period_s": .05, "ready": True,
            "action_contract": "base_relative_observation_start_rotvec_gripper_absolute"}


def targets(value):
    return [Target((value+i*.0001, 0, .3), (0, 0, 0, 1), .5) for i in range(20)]


class TimelineTests(unittest.TestCase):
    def test_full_queue_splice_preserves_frozen_and_committed_prefix(self):
        timeline = Timeline(1)
        old = targets(0)
        timeline.seed(old)
        context = timeline.context(7, 7, 1)
        for _ in range(12):
            tick, _ = timeline.commit_next()
            timeline.complete(tick)
        with timeline.lock:
            merged = timeline.merged(context, targets(.1))
        self.assertEqual(merged[12:14] if isinstance(merged, list) else [merged[12], merged[13]], old[12:14])
        self.assertEqual(merged[14], targets(.1)[7])
        self.assertEqual(sorted(merged), list(range(12, 27)))

    def test_obsolete_plan_and_queue_gap_rejected(self):
        timeline = Timeline(1)
        timeline.seed(targets(0))
        context = timeline.context(7, 7, 1)
        timeline.revision += 1
        with self.assertRaisesRegex(RuntimeError, "Obsolete"):
            timeline.merged(context, targets(.1))
        with self.assertRaisesRegex(RuntimeError, "runway"):
            timeline.context(15, 7, 2)

    def test_ten_minutes_virtual_time_with_150_to_300ms_jitter(self):
        timeline = Timeline(2)
        timeline.seed(targets(0))
        pending = None
        next_request = 7
        request_id = 0
        accepted = 0
        dispatches = []
        for tick in range(12000):
            if pending and tick == pending[0]:
                _, context, new_targets = pending
                with timeline.lock:
                    before = dict(timeline.targets)
                    merged = timeline.merged(context, new_targets)
                    for index in range(timeline.next_tick, context["freeze_until"]):
                        self.assertEqual(merged[index], before[index])
                    timeline.targets = merged
                    timeline.revision += 1
                pending = None
                accepted += 1
            if tick == next_request:
                request_id += 1
                context = timeline.context(tick, 7, request_id)
                latency_ticks = (3, 6, 5, 4)[request_id % 4]
                pending = (tick+latency_ticks, context, targets(request_id*.001))
                next_request += 7
            index, _ = timeline.commit_next()
            dispatches.append(index)
            timeline.complete(index)
        self.assertEqual(dispatches, list(range(12000)))
        self.assertGreater(accepted, 1700)

    def test_contract_and_config_explicit_no_fallback(self):
        config = project(json.loads((ROOT/"config/pi05.json").read_text()))
        # Operator settings may already be RTC/H20; isolate this migration
        # test from the live console's persisted selection.
        config = prepare_update(config, {"execution": {"mode": "normal", "expected_chunk_steps": 10, "replan_steps": 5}})
        with self.assertRaises(ValueError):
            prepare_update(config, {"execution": {"mode": "rtc"}})
        updated = prepare_update(config, {"execution": {"mode": "rtc", "expected_chunk_steps": 20, "replan_steps": 5}})
        compatibility(METADATA, updated["execution"])
        for key, value in (("rtc_version", None), ("ready", False), ("state_dim", 8), ("action_horizon", 10), ("period_s", .02)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                compatibility(dict(METADATA, **{key: value}), updated["execution"])


class Camera:
    def __init__(self):
        self.lock = threading.RLock()
        self.cameras = object()
        self.frames = {key: np.zeros((224, 224, 3), np.uint8) for key in ("external", "wrist")}
        self.frame_times_ns = {}
        self.frame_version = 0
        self.config = {}
        self.dataset_frame_history = {}
        self.dataset_frame_sizes = {}
        self.last_error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        while not self.done.is_set():
            with self.lock:
                self.frame_times_ns = {k: time.perf_counter_ns() for k in self.frames}
                self.frame_version += 1
            self.done.wait(.01)

    def snapshot(self):
        return {"ready": True, "frame_version": self.frame_version}

    def close(self):
        self.done.set()
        self.thread.join(1)


class Policy:
    metadata = METADATA

    def __init__(self, latency=.15, corrupt=None):
        self.latency, self.corrupt = latency, corrupt
        self.requests = []
        self.closed = threading.Event()

    def infer(self, obs):
        self.requests.append(copy.deepcopy(obs))
        if self.closed.wait(self.latency):
            raise RuntimeError("Closed in-flight inference")
        context = obs["rtc_context"]
        echo = {k: v for k, v in context.items() if k != "absolute_prefix"}
        echo["sampler"] = "bootstrap_normal" if context["phase"] == "bootstrap" else "rtc_vjp"
        if self.corrupt:
            echo[self.corrupt] = -1
        return {"actions": np.asarray([[.001, 0, 0, 0, 0, 0, .5]]*20, np.float32), "rtc": echo}

    def is_alive(self):
        return not self.closed.is_set()

    def close(self):
        self.closed.set()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name)/"config/pi05.json"
        path.parent.mkdir()
        config = project(json.loads((ROOT/"config/pi05.json").read_text()))
        config = prepare_update(config, {"execution": {"mode": "rtc", "expected_chunk_steps": 20, "replan_steps": 5}})
        path.write_text(json.dumps(config))
        self.camera = Camera()
        self.broker = _Broker()
        with patch("supervisor.pi05_adapter.OpenPIClient", _InitiallyUnavailablePolicy), patch.object(Pi05InputAdapter, "_local_forward_listening", return_value=True):
            self.a = Pi05InputAdapter(self.broker, path, self.camera)
        self.a._connection_stop.set()
        self.a._connection_wakeup.set()
        self.a._connection_thread.join(2)
        self.policy = Policy()
        self.a._policy = self.policy
        self.a.state.update(model_state="CONNECTED", protocol_compatibility=self.a._metadata_status(METADATA))

    def tearDown(self):
        self.a.close()
        self.camera.close()
        self.tmp.cleanup()

    def wait_for(self, predicate, timeout=3):
        end = time.perf_counter()+timeout
        while not predicate() and time.perf_counter() < end:
            time.sleep(.005)
        self.assertTrue(predicate(), self.a.state)

    def test_async_replanning_cadence_and_stop_no_future_dispatch(self):
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state["executed_steps"] >= 20)
        self.assertGreaterEqual(len(self.policy.requests), 3)
        self.a.stop("test stop")
        count = len(self.broker.commands)
        time.sleep(.15)
        self.assertEqual(len(self.broker.commands), count)
        self.assertEqual(self.broker.commands[-1]["type"], "hold")
        self.assertTrue(self.a.state["rtc"]["hold_confirmed"])
        track = [c for c in self.broker.commands if c["type"] == "track_tcp"]
        spacing = np.diff([c["started_at"] for c in track])
        self.assertGreater(float(spacing.min()), .025)
        self.assertLess(float(np.percentile(spacing, 95)), .075)
        self.assertFalse(any(c["type"] == "gripper" for c in self.broker.commands))
        self.assertTrue(all(r["observation/state"].shape == (7,) for r in self.policy.requests))

    def test_mismatched_identity_never_sends_target(self):
        self.policy.corrupt = "request_id"
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("identity mismatch", self.a.state["last_error"])
        self.assertFalse(any(c["type"] == "track_tcp" for c in self.broker.commands))

    def test_control_timeout_holds_and_requires_manual_restart(self):
        self.broker.track_delay_s = .08
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("exceeded its slot", self.a.state["last_error"])
        self.assertEqual([c["type"] for c in self.broker.commands], ["track_tcp", "hold"])
        self.assertTrue(self.a.state["rtc"]["manual_restart_required"])

    def test_stop_waits_for_inflight_control_then_holds(self):
        entered, release = threading.Event(), threading.Event()
        original = self.broker.track_tcp
        def blocked(*args):
            entered.set()
            release.wait(2)
            return original(*args)
        self.broker.track_tcp = blocked
        self.a.start("session-1", "client-1")
        self.assertTrue(entered.wait(2))
        stopped = threading.Event()
        stop = threading.Thread(target=lambda: (self.a.stop("blocked test"), stopped.set()))
        stop.start()
        time.sleep(.05)
        self.assertFalse(stopped.is_set())
        self.assertFalse(self.a.state["execution_enabled"])
        release.set()
        stop.join(2)
        self.assertTrue(stopped.is_set())
        self.assertEqual([c["type"] for c in self.broker.commands], ["track_tcp", "hold"])

    def test_settings_rejected_during_bootstrap_and_stale_images_fail(self):
        self.a.start("session-1", "client-1")
        with self.assertRaises(RuntimeError):
            self.a.update_config({"model": {"prompt": "new task"}})
        self.camera.close()
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("camera feedback", self.a.state["last_error"])

    def test_measured_pose_required_not_command_target(self):
        with self.a.lock:
            self.a.state.update(session_id="session-1", client_id="client-1")
            self.a._osc_snapshot = self.broker.state()
            self.a._osc_snapshot["execution"] = {}
            self.a._log_feedback_snapshot = {"cache_published_perf_ns": time.perf_counter_ns()}
        with self.assertRaisesRegex(RuntimeError, "measured TCP"):
            ObservationAssembler(self.a).read()

    def test_shadow_hold_pose_reused_only_with_stationary_plant_evidence(self):
        with self.a.lock:
            self.a.state.update(session_id="session-1", client_id="client-1")
            snapshot = self.broker.state()
            snapshot["execution"].update(sample_monotonic_ns=time.monotonic_ns()-2_000_000_000,
                                         measured_joint_state_rad=[0.0]*7)
            snapshot["diagnostics"] = {"trajectory_state": "HOLD_READY", "shadow_transport": {
                "enabled": True, "applied_joint_state_rad": [0.0]*7,
                "applied_joint_velocity_rad_s": [0.0]*7}}
            self.a._osc_snapshot = snapshot
            self.a._log_feedback_snapshot = {"cache_published_perf_ns": time.perf_counter_ns()}
            assembler = ObservationAssembler(self.a)
            assembler.read()
            snapshot["diagnostics"]["shadow_transport"]["applied_joint_state_rad"][0] = .01
            with self.assertRaisesRegex(RuntimeError, "skew"):
                assembler.read()
            snapshot["diagnostics"]["shadow_transport"]["applied_joint_state_rad"][0] = 0
            snapshot["diagnostics"]["shadow_transport"]["applied_joint_velocity_rad_s"][0] = .01
            with self.assertRaisesRegex(RuntimeError, "skew"):
                assembler.read()
            snapshot["session"]["execution_mode"] = "hardware"
            with self.assertRaisesRegex(RuntimeError, "timestamp missing or stale"):
                assembler.read()
            health = self.a.state['rtc_observation_health']
            self.assertGreater(health['sample_age_ms'], 100.)
            self.assertLess(health['cache_age_ms'], 100.)
            self.assertEqual(health['source'], 'osc_execution')

    def test_tcp_clock_conversion_with_windows_epoch_offset(self):
        with self.a.lock:
            self.a.state.update(session_id="session-1", client_id="client-1")
            snapshot = self.broker.state()
            snapshot["execution"]["sample_monotonic_ns"] = 1_800_000_000
            self.a._osc_snapshot = snapshot
            self.a._log_feedback_snapshot = {"cache_published_perf_ns": 2_000_000_000}
            with self.camera.lock:
                self.camera.frame_times_ns = {k: 1_980_000_000 for k in self.camera.frames}
            with patch("supervisor.inference_rtc.time.perf_counter_ns", return_value=2_000_000_000), \
                    patch("supervisor.inference_rtc.time.monotonic_ns", return_value=1_800_000_000):
                ObservationAssembler(self.a).read()
                snapshot["execution"]["sample_monotonic_ns"] -= 200_000_000
                with self.assertRaisesRegex(RuntimeError, "skew"):
                    ObservationAssembler(self.a).read()

    def test_training_provenance_is_informational_not_a_start_gate(self):
        self.broker.session["execution_mode"] = "hardware"
        result = self.a.start("session-1", "client-1")
        self.assertEqual(result['execution_mode'], 'hardware')
        self.assertTrue(result['priming'])
        self.assertIn('不阻止', result['training_notice'])
        self.assertEqual(self.broker.commands, [])

    def test_slow_bootstrap_rejected_before_any_target(self):
        self.policy.latency = .46
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("bootstrap latency", self.a.state["last_error"])
        self.assertFalse(any(c["type"] == "track_tcp" for c in self.broker.commands))

    def test_failure_can_be_manually_restarted_with_fresh_bootstrap(self):
        self.policy.corrupt = "request_id"
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a._rtc_run is not None and self.a._rtc_run.finished.is_set())
        self.assertIsNone(self.a._capture)
        self.assertTrue(self.a._rtc_run.hold_confirmed)
        self.policy = Policy()
        self.a._policy = self.policy
        self.a.state["protocol_compatibility"] = self.a._metadata_status(METADATA)
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state["executed_steps"] >= 2)
        self.assertEqual(self.policy.requests[0]["rtc_context"]["phase"], "bootstrap")

    def test_tcp_success_gripper_failure_is_partial_dispatch_then_hold(self):
        self.broker.session["execution_mode"] = "hardware"
        self.policy.metadata = dict(METADATA, training_contract_verified=True)
        original_state = self.broker.state
        def state():
            result = original_state()
            result["execution"].update(sample_monotonic_ns=time.monotonic_ns(), feedback_age_s=0)
            return result
        self.broker.state = state
        self.broker.inference_gripper_feedback = lambda: {"width_m": .02,
            "raw": {"value": {"timestamp": time.perf_counter_ns()}},
            "status": {"foc_status": {"driver_enable_status": True}}}
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("gripper was not sent", self.a.state["last_error"])
        self.assertEqual([c["type"] for c in self.broker.commands], ["track_tcp", "gripper", "hold"])

    def test_unconfirmed_hold_blocks_restart_until_stop_retried(self):
        original_hold = self.broker.hold
        self.broker.hold = lambda *args: {"ok": False}
        self.policy.corrupt = "request_id"
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("state") == "HOLD_UNCONFIRMED")
        with self.assertRaisesRegex(RuntimeError, "HOLD is unconfirmed"):
            self.a.start("session-1", "client-1")
        self.broker.hold = original_hold
        self.a.stop("retry HOLD")
        self.assertTrue(self.a._rtc_run.hold_confirmed)

    def test_session_change_does_not_hold_other_control_source(self):
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state["executed_steps"] >= 2)
        self.broker.session.update(id="other-session", client_id="other-client")
        self.wait_for(lambda: self.a._rtc_run.authority_released)
        self.assertFalse(self.a._rtc_run.hold_confirmed)
        self.assertFalse(any(c["type"] == "hold" for c in self.broker.commands))


if __name__ == "__main__":
    unittest.main()
