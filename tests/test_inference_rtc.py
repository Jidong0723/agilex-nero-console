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
from supervisor.inference_rtc import Timeline, Target, compatibility, VERSION, ObservationAssembler, TIME_CONTRACT, PERIOD_NS, bootstrap_bridge
from supervisor.pi05_adapter import Pi05InputAdapter
from tests.test_pi05_adapter import _Broker, _InitiallyUnavailablePolicy

ROOT = Path(__file__).parents[1]
METADATA = {"state_dim": 7, "valid_action_dim": 7, "action_dim": 32, "action_horizon": 20,
            "rtc_bootstrap_mode": "hold_prefix_v1",
            "rtc_bootstrap_attention": "full_hold_v2",
            "rtc_schedule": "fixed_delay_execute_horizon_v1",
            "rtc_version": VERSION, "period_s": .05, "ready": True,
            "action_time_contract": TIME_CONTRACT, "action_row0_offset_s": 0,
            "action_contract": "base_relative_observation_start_rotvec_gripper_absolute"}


def targets(value):
    return [Target((value+i*.0001, 0, .3), (0, 0, 0, 1), .5) for i in range(20)]


class TimelineTests(unittest.TestCase):
    def test_refined_profile_requires_full_execution_inside_overlap(self):
        metadata=dict(METADATA,rtc_requires_execution_within_prefix=True)
        execution={'expected_chunk_steps':20,'inference_delay':7,'execute_horizon':8}
        with self.assertRaisesRegex(RuntimeError,r'D\+2E'):
            compatibility(metadata,execution)
        compatibility(metadata,dict(execution,inference_delay=6,execute_horizon=6))
        compatibility(METADATA,execution)  # Legacy official grid remains selectable.
        exp=dict(METADATA,prefix_attention_schedule='exp',rtc_requires_execution_within_prefix=False,
                 rtc_late_response_policy='guided_same_time_continuity_v1')
        compatibility(exp,dict(execution,inference_delay=8,execute_horizon=8))
        with self.assertRaises(ValueError):
            compatibility(exp,dict(execution,inference_delay=11,execute_horizon=11))

    def test_startup_bridge_joins_future_row_without_retiming_or_rewriting_model(self):
        hold = Target((0,0,.3),(0,0,0,1),.5)
        rows = [Target((.006,0,.3),(0,0,0,1),.5)]*20
        bridge,end = bootstrap_bridge(rows,7,hold,{})
        self.assertEqual(bridge[7],hold)
        self.assertEqual(bridge[end],rows[end])
        self.assertEqual(rows[7].position[0],.006)
        self.assertEqual(bridge[end+1:],rows[end+1:])
        timeline = Timeline(1,1_000_000_000)
        timeline.seed(bridge,1_324_000_000,hold)
        context=timeline.context(7,7,1,observation_ns=1_350_000_000,
                                 now_ns=1_350_000_000,minimum_freeze_tick=end+1)
        self.assertGreater(context['freeze_until'],end)
        self.assertEqual(context['absolute_prefix'][0],hold.absolute_row())
        self.assertEqual(timeline.origin_ns+end*PERIOD_NS,1_000_000_000+end*50_000_000)

    def test_bridge_refuses_large_or_fast_targets(self):
        hold=Target((0,0,.3),(0,0,0,1),.5)
        with self.assertRaisesRegex(RuntimeError,'cannot safely join HOLD'):
            bootstrap_bridge([Target((.03,0,.3),(0,0,0,1),.5)]*20,7,hold,{})
        with self.assertRaisesRegex(RuntimeError,'cannot safely join HOLD'):
            bootstrap_bridge([Target((.006,0,.3),(0,0,0,1),.5)]*20,7,hold,{'linear_speed_m_s':.001})

    def test_bridge_matches_moving_tail_without_zero_speed_seam(self):
        hold = Target((0,0,.3),(0,0,0,1),.5)
        # The old zero-terminal-speed quintic cannot join this 20 mm/s tail
        # within 0.3 m/s^2; a continuation-aware fit can, at the same times.
        rows = [Target((.006+.001*(i-15),0,.3),(0,0,0,1),.5) for i in range(20)]
        original = list(rows)
        bridge,end = bootstrap_bridge(rows,9,hold,{})
        self.assertLessEqual(end,17)
        self.assertEqual(bridge[9],hold)
        self.assertEqual(bridge[end:],original[end:])
        self.assertEqual(rows,original)
        velocities = np.diff(np.asarray([hold.position,*[t.position for t in bridge[9:]]]),axis=0)/.05
        self.assertLessEqual(float(np.linalg.norm(velocities,axis=1).max()),.05+1e-6)
        self.assertLessEqual(float(np.linalg.norm(np.diff(velocities,axis=0)/.05,axis=1).max()),.3+1e-6)

    def test_cloud_accepts_only_stationary_full_hold_bootstrap(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('hold_bootstrap_codec',ROOT/'inference/rtc/codec.py')
        codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
        hold=Target((0,0,.3),(0,0,0,1),.5)
        context={'version':VERSION,'phase':'bootstrap_hold','run_epoch':1,'request_id':1,
                 'observation_tick':0,'plan_revision':1,'freeze_until':7,
                 'observation_perf_ns':1_000_000_000,'timeline_origin_perf_ns':1_000_000_000,
                 'freeze_until_perf_ns':1_350_000_000,'action_row0_offset_s':0,
                 'period_s':.05,'absolute_prefix':[hold.absolute_row() for _ in range(20)]}
        self.assertEqual(codec.validate_context(context),(7,20))
        moving=copy.deepcopy(context);moving['absolute_prefix'][-1][0]=.01
        for bad in (moving,dict(context,absolute_prefix=context['absolute_prefix'][:10])):
            with self.assertRaisesRegex(ValueError,'stationary H20'):
                codec.validate_context(bad)

    def test_bootstrap_skips_expired_rows_without_rebasing_response_time(self):
        origin = 1_000_000_000
        for latency_ms in (150, 199, 300):
            timeline = Timeline(1, origin)
            rows = targets(0)
            hold = Target((0, 0, .3), (0, 0, 0, 1), .5)
            timeline.seed(rows, origin+latency_ms*1_000_000, hold)
            first = latency_ms//50+1
            self.assertEqual(timeline.next_tick, first)
            self.assertEqual(timeline.targets[first], rows[first])
            self.assertNotIn(0, timeline.targets)
            self.assertEqual(timeline.history[0], hold)
            self.assertEqual(origin+first*PERIOD_NS, origin+(latency_ms//50+1)*50_000_000)
        with self.assertRaisesRegex(RuntimeError, 'no future'):
            Timeline(1, origin).seed(rows, origin+950_000_000, hold)

    def test_off_grid_observation_prefix_and_new_rows_use_actual_timestamps(self):
        timeline = Timeline(1)
        old = targets(0)
        timeline.seed(old)
        for _ in range(9):
            tick, _ = timeline.commit_next()
            timeline.complete(tick)
        context = timeline.context(7, 7, 1, observation_ns=375_000_000)
        self.assertEqual(context['observation_tick'], 7)
        self.assertEqual(context['observation_perf_ns'], 375_000_000)
        self.assertAlmostEqual(context['absolute_prefix'][0][0], .00075)
        # Response arrives at 801ms: preserve every old target before tick17,
        # including unconsumed expired slots; executor must HOLD, not catch up.
        merged = timeline.merged(context, targets(.1), now_ns=801_000_000)
        self.assertEqual(merged[16], old[16])
        self.assertAlmostEqual(merged[17].position[0], .10095)  # (850-375)/50 = row9.5
        self.assertEqual(max(merged), 26)  # last source row is at1325ms, no extrapolation

    def test_local_and_cloud_contract_match_off_grid_frozen_prefix(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('rtc_codec_test', ROOT/'inference/rtc/codec.py')
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        timeline = Timeline(1)
        timeline.seed(targets(0))
        context = timeline.context(7, 7, 1, observation_ns=375_000_000, now_ns=400_000_000)
        delay, length = codec.validate_context(context)
        self.assertEqual(delay, 8)  # freeze750ms; row7 is725ms, row8 is775ms
        self.assertEqual(length, len(context['absolute_prefix']))
        self.assertEqual(codec.echo_context(context)['observation_perf_ns'], 375_000_000)
        for bad in (dict(context, version='nero.rtc.v1'), dict(context, action_row0_offset_s=.05),
                    dict(context, freeze_until_perf_ns=1), dict(context, observation_perf_ns=500_000_000)):
            with self.assertRaises(ValueError):
                codec.validate_context(bad)

    def test_delayed_response_cannot_replace_committed_command_or_rebase_expired_rows(self):
        timeline = Timeline(1)
        timeline.seed(targets(0))
        context = timeline.context(7, 7, 1)
        for _ in range(15):
            tick, _ = timeline.commit_next()
            timeline.complete(tick)
        tick, committed = timeline.commit_next()  # tick15 in synchronous output call
        merged = timeline.merged(context, targets(.1), now_ns=760_000_000)
        self.assertEqual(merged[15], committed)
        self.assertEqual(merged[16], targets(.1)[9])  # still row9 at800ms, never row0
        with self.assertRaisesRegex(RuntimeError, 'no usable future'):
            timeline.merged(context, targets(.1), now_ns=1_300_000_000)

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

    def test_virtual_off_grid_observations_with_150_to_300ms_responses(self):
        timeline = Timeline(2)
        timeline.seed(targets(0))
        pending = None
        accepted = 0
        for tick in range(700):
            if pending and tick == pending[0]:
                _, context, new = pending
                before = dict(timeline.targets)
                timeline.targets = timeline.merged(context, new, now_ns=tick*PERIOD_NS)
                timeline.revision += 1
                for old_tick in range(timeline.next_tick, context['freeze_until']):
                    self.assertEqual(timeline.targets[old_tick], before[old_tick])
                accepted += 1
                pending = None
            if tick and tick % 7 == 0:
                context = timeline.context(tick-1, 7, tick, observation_ns=tick*PERIOD_NS-25_000_000,
                                           now_ns=tick*PERIOD_NS)
                pending = (tick+(3, 6, 5, 4)[accepted % 4], context, targets(tick*.001))
            dispatch, _ = timeline.commit_next()
            self.assertEqual(dispatch, tick)
            timeline.complete(dispatch)
        self.assertGreater(accepted, 90)

    def test_contract_and_config_explicit_no_fallback(self):
        config = project(json.loads((ROOT/"config/pi05.json").read_text()))
        # Operator settings may already be RTC/H20; isolate this migration
        # test from the live console's persisted selection.
        config = prepare_update(config, {"execution": {"mode": "normal", "expected_chunk_steps": 10, "replan_steps": 5}})
        with self.assertRaises(ValueError):
            prepare_update(config, {"execution": {"mode": "rtc"}})
        updated = prepare_update(config, {"execution": {"mode": "rtc", "expected_chunk_steps": 20, "replan_steps": 5}})
        compatibility(METADATA, updated["execution"])
        for key, value in (("rtc_version", None), ("ready", False), ("state_dim", 8), ("action_horizon", 10), ("period_s", .02),
                           ('action_row0_offset_s', .05), ('action_time_contract', None)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                compatibility(dict(METADATA, **{key: value}), updated["execution"])


class WireTimeContractTests(unittest.TestCase):
    def test_normal_protocol_and_rtc_timestamps_share_one_fake_websocket(self):
        import importlib.util
        import msgpack
        from websockets.sync.server import serve
        from supervisor.openpi_client import OpenPIClient, _pack_array, _unpack_array
        spec = importlib.util.spec_from_file_location('rtc_wire_codec', ROOT/'inference/rtc/codec.py')
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        received, errors = [], []
        def handler(socket):
            try:
                socket.send(msgpack.packb(METADATA))
                for _ in range(2):
                    request = msgpack.unpackb(socket.recv(), object_hook=_unpack_array)
                    received.append(request)
                    response = {'actions': np.zeros((20, 7), np.float32)}
                    if 'rtc_context' in request:
                        codec.validate_context(request['rtc_context'])
                        response['rtc'] = dict(codec.echo_context(request['rtc_context']), sampler='rtc_vjp')
                    socket.send(msgpack.packb(response, default=_pack_array))
            except Exception as exc:
                errors.append(str(exc))
        with serve(handler, '127.0.0.1', 0, compression=None) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = OpenPIClient('127.0.0.1', server.socket.getsockname()[1], 2)
            try:
                compatibility(client.metadata, {'expected_chunk_steps': 20})
                observation = {'observation/image': np.zeros((224, 224, 3), np.uint8),
                               'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
                               'observation/state': np.zeros(7, np.float32), 'prompt': 'test'}
                self.assertEqual(client.infer(observation)['actions'].shape, (20, 7))
                timeline = Timeline(1, 10_000_000_000)
                timeline.seed(targets(0))
                context = timeline.context(7, 7, 1, observation_ns=10_375_000_000)
                response = client.infer(dict(observation, rtc_context=context))
                self.assertEqual(response['rtc'], dict(codec.echo_context(context), sampler='rtc_vjp'))
            finally:
                client.close()
                server.shutdown()
                thread.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(set(received[0]), set(observation))
        self.assertEqual(received[1]['observation/state'].shape, (7,))


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
        echo["sampler"] = "bootstrap_hold_vjp" if context["phase"] == "bootstrap_hold" else "rtc_vjp"
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
        config = prepare_update(config, {"execution": {"mode": "rtc", "expected_chunk_steps": 20, "replan_steps": 5,
            "inference_delay": 6, "execute_horizon": 8}})
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
        self.assertFalse(self.a.snapshot()['recovery']['stop_required'])
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
        capture=self.a._completed_capture
        events=[json.loads(line) for line in (capture.path/'events.jsonl').read_text().splitlines()]
        splices=[r for r in events if r['event']=='rtc_splice']
        self.assertTrue(splices)
        for splice in splices:
            self.assertEqual(splice['next_request_perf_ns'],
                splice['observation_perf_ns']+splice['execute_horizon']*PERIOD_NS)
        contexts=[r['rtc_context'] for r in self.policy.requests if r['rtc_context']['phase'] != 'bootstrap_hold']
        self.assertGreaterEqual(len(contexts),2)
        for context in contexts:
            self.assertEqual((context['observation_perf_ns']-context['timeline_origin_perf_ns'])%PERIOD_NS,0)
        for before,after in zip(contexts,contexts[1:]):
            self.assertEqual(after['observation_perf_ns']-before['observation_perf_ns'],8*PERIOD_NS)

    def test_mismatched_identity_never_sends_target(self):
        self.policy.corrupt = "request_id"
        self.a.start("session-1", "client-1")
        self.wait_for(lambda: self.a.state.get("rtc", {}).get("hold_confirmed"))
        self.assertIn("identity mismatch", self.a.state["last_error"])
        self.assertFalse(any(c["type"] == "track_tcp" for c in self.broker.commands))

    def test_aligned_run_with_fake_hardware_joint_and_sdk_gripper_histories(self):
        from supervisor.inference_feedback import MeasuredTCPFeed
        a=self.a
        class Source:
            def pose_for_joints(self,q):
                return {'position_m':[float(q[0]),0.,.3],'orientation_xyzw':[0.,0.,0.,1.]}
            def refresh(self,snapshot,*,tcp_only=False):
                snapshot['inference_observation']={'pose':self.pose_for_joints([0.]*7),
                    'joint_position_rad':[0.]*7,'perf_ns':time.perf_counter_ns()-1_000_000,
                    'feedback_age_s':.001,'source':'fake_measured_joint_fk','timing':{}}
                return snapshot
            def close(self):pass
        self.broker.session['execution_mode']='hardware'
        self.broker.inference_gripper_feedback=lambda:{'width_m':.0475,
            'raw':{'value':{'timestamp':time.time()}}}
        self.broker.gripper=lambda *args:{'ok':True,'result':{'ok':True,'sent':True,'feedback_pending':True}}
        a._observation_source=Source()
        a._tcp_feed=MeasuredTCPFeed(a._observation_source,a._feedback_context,a._osc_state_stop)
        a._tcp_feed.thread.start()
        a.start('session-1','client-1')
        self.wait_for(lambda:a.state['executed_steps']>=25)
        a.stop('fake hardware alignment test')
        contexts=[r['rtc_context'] for r in self.policy.requests if r['rtc_context']['phase']!='bootstrap_hold']
        self.assertGreaterEqual(len(contexts),3)
        for before,after in zip(contexts,contexts[1:]):
            self.assertEqual(after['observation_perf_ns']-before['observation_perf_ns'],8*PERIOD_NS)
        events=[json.loads(line) for line in (a._completed_capture.path/'events.jsonl').read_text().splitlines()]
        aligned=[r for r in events if r['event']=='rtc_observation_alignment']
        self.assertGreaterEqual(len(aligned),2)
        for row in aligned:
            for name in ('state_alignment','gripper_alignment'):
                left,right=row[name]['source_perf_ns']
                self.assertLessEqual(left,row['observation_perf_ns'])
                self.assertGreaterEqual(right,row['observation_perf_ns'])

    def test_bootstrap_future_target_too_far_from_hold_never_moves(self):
        original = self.policy.infer
        def jumping(obs):
            result = original(obs)
            result['actions'][:, 0] = .03
            return result
        self.policy.infer = jumping
        self.a.start('session-1', 'client-1')
        self.wait_for(lambda: self.a.state.get('rtc', {}).get('hold_confirmed'))
        self.assertIn('cannot safely join HOLD', self.a.state['last_error'])
        self.assertFalse(any(c['type'] == 'track_tcp' for c in self.broker.commands))

    def test_moving_bootstrap_starts_with_hold_bridge_and_replans_actual_prefix(self):
        original=self.policy.infer
        def moving(obs):
            result=original(obs)
            result['actions'][:,0]=.006
            return result
        self.policy.infer=moving
        self.a.start('session-1','client-1')
        self.wait_for(lambda:self.a.state['executed_steps']>=10)
        self.assertEqual(self.a.state['rtc']['bootstrap_strategy'],'bounded_bridge')
        track=[c for c in self.broker.commands if c['type']=='track_tcp']
        self.assertGreater(len(track),0)
        self.assertGreater(len(self.policy.requests),1)
        self.assertTrue(self.policy.requests[1]['rtc_context']['absolute_prefix'])

    def test_observation_epoch_is_sample_timestamp_not_cache_read_time(self):
        with self.a.lock:
            self.a.state.update(session_id='session-1', client_id='client-1')
            snapshot = self.broker.state()
            stamp = time.perf_counter_ns()-20_000_000
            snapshot['inference_observation'] = {
                'pose': snapshot['execution']['measured_tcp_pose'], 'perf_ns': stamp,
                'feedback_age_s': .02, 'gripper_width_m': .0475}
            self.a._osc_snapshot = snapshot
            self.a._log_feedback_snapshot = {'cache_published_perf_ns': time.perf_counter_ns()}
            _, _, osc, epoch = ObservationAssembler(self.a).read()
        self.assertEqual(epoch, stamp)
        self.assertEqual(osc['rtc_observation_times']['observation_perf_ns'], stamp)
        self.assertGreater(osc['rtc_observation_times']['read_perf_ns'], stamp)

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
        self.assertIn("bootstrap inference_delay exceeded", self.a.state["last_error"])
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
        self.assertEqual(self.policy.requests[0]["rtc_context"]["phase"], "bootstrap_hold")
        self.assertEqual(len(self.policy.requests[0]['rtc_context']['absolute_prefix']),20)

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
            "raw": {"value": {"timestamp": time.time()}},
            "status": {"foc_status": {"driver_enable_status": True}}}
        # This old fixture has no joint RX/FK producer. Isolate its intended
        # partial-send failure; real aligned feedback is tested separately.
        with patch.object(ObservationAssembler, 'read_aligned',
                          lambda assembler, anchor, cancel: assembler.read()):
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
