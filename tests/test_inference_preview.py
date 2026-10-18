"""Idle predictions must never acquire control or become executable results."""
import copy
import threading
import time
import unittest

import numpy as np

from supervisor.inference_preview import preview_once, ObservationSource
from tests import test_inference_rtc as rtc


class PredictionPolicy(rtc.Policy):
    def infer(self, observation):
        if 'rtc_context' in observation:
            return super().infer(observation)
        self.requests.append(copy.deepcopy(observation))
        return {'actions': np.asarray([[.001,0,0,0,0,0,.5]]*20, np.float32)}


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.f = rtc.RuntimeTests()
        self.f.setUp()
        self.a = self.f.a
        self.policy = PredictionPolicy(.03)
        self.a._policy = self.policy
        self.a._osc_snapshot = self.f.broker.state()

    def tearDown(self):
        self.f.tearDown()

    def test_rtc_idle_preview_is_ordinary_and_never_sends_commands(self):
        self.f.broker.session['state'] = 'IDLE'
        preview_once(self.a)
        preview_once(self.a)
        self.assertEqual(len(self.policy.requests), 2)
        self.assertNotIn('rtc_context', self.policy.requests[0])
        self.assertEqual(self.policy.requests[0]['observation/state'].shape, (7,))
        self.assertEqual(self.a.state['preview']['sequence'], 2)
        self.assertEqual(self.a.state['last_response_steps'], 20)
        self.assertEqual(self.a.state['chunk_source'], 'preview')
        self.assertEqual(self.f.broker.commands, [])

    def test_missing_measured_gripper_never_uses_placeholder(self):
        self.a._osc_state_stop.set()
        self.a._osc_state_thread.join(1)
        self.a._osc_snapshot['gripper']['width_m'] = None
        preview_once(self.a)
        self.assertEqual(self.policy.requests, [])
        self.assertIn('夹爪', self.a.state['preview']['error'])
        self.assertEqual(self.f.broker.commands, [])

    def test_preview_failure_preserves_control_fault_and_last_chunk(self):
        preview_once(self.a)
        previous = copy.deepcopy(self.a.state['action_chunk'])
        self.a.state['last_error'] = 'persistent control fault'
        self.policy.infer = lambda _: {'actions': [[float('nan')]*7]*20}
        preview_once(self.a)
        self.assertEqual(self.a.state['preview']['state'], 'ERROR')
        self.assertEqual(self.a.state['last_error'], 'persistent control fault')
        self.assertEqual(self.a.state['action_chunk'], previous)
        self.assertTrue(self.a.state['preview']['stale'])
        self.assertEqual(self.f.broker.commands, [])

    def test_late_preview_response_discarded_after_revision_change(self):
        entered, release = threading.Event(), threading.Event()
        original = self.policy.infer
        def delayed(obs):
            entered.set()
            release.wait(2)
            return original(obs)
        self.policy.infer = delayed
        thread = threading.Thread(target=preview_once, args=(self.a,))
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.a.lock:
            self.a._run_revision += 1
        release.set()
        thread.join(2)
        self.assertIsNone(self.a.state['action_chunk'])
        self.assertEqual(self.f.broker.commands, [])

    def test_stop_rtc_control_resumes_preview_without_restarting_motion(self):
        self.a.start('session-1','client-1')
        self.f.wait_for(lambda: self.a.state['executed_steps'] >= 2)
        self.a.stop('operator stop')
        self.policy = PredictionPolicy()
        self.a._policy = self.policy
        self.f.wait_for(lambda: self.a.state['preview']['sequence'] >= 2)
        count = len(self.f.broker.commands)
        time.sleep(.15)
        self.assertEqual(len(self.f.broker.commands), count)
        self.assertFalse(self.a.state['execution_enabled'])
        self.assertEqual(self.a.state['chunk_source'], 'preview')

    def test_readonly_observation_rejects_stopped_gripper_timestamp(self):
        from types import SimpleNamespace
        broker=SimpleNamespace(sensor_sample=lambda now, wait: {
            'joint_position_rad':[0.]*7,'feedback_monotonic_ns':now,'latest_feedback_age_s':0.},
            inference_gripper_feedback=lambda:{'width_m':.02,'raw':{'value':{'timestamp':1}}})
        source=ObservationSource(SimpleNamespace(osc=broker))
        source.last_joints=(0.,)*7
        source.last_pose={'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]}
        observed = source.refresh({'session':{'state':'IDLE'}})['inference_observation']
        self.assertEqual(observed['timing']['sample_perf_ns'], observed['perf_ns'])
        self.assertTrue(observed['timing']['fk_reused'])
        self.assertGreaterEqual(observed['timing']['refresh_ms'], observed['timing']['fk_ms'])
        source.gripper_seen_ns-=200_000_000
        with self.assertRaisesRegex(RuntimeError,'150 ms'):
            source.refresh({'session':{'state':'IDLE'}})

    def test_normal_stop_after_prior_rtc_run_still_confirms_hold(self):
        from types import SimpleNamespace
        self.a.config['execution']['mode']='normal'
        finished=threading.Event();finished.set()
        self.a._rtc_run=SimpleNamespace(stop_confirmed=True,finished=finished)
        self.a.state.update(execution_enabled=True,session_id='session-1',client_id='client-1')
        self.a.stop('normal stop after RTC')
        self.assertEqual(self.f.broker.commands[-1]['type'],'hold')

    def test_finished_run_with_unconfirmed_hold_does_not_resume_preview(self):
        from types import SimpleNamespace
        finished=threading.Event();finished.set()
        self.a._rtc_run=SimpleNamespace(stop_confirmed=False,finished=finished)
        preview_once(self.a)
        self.assertEqual(self.policy.requests,[])
        self.a._rtc_run=None


if __name__ == '__main__':
    unittest.main()
