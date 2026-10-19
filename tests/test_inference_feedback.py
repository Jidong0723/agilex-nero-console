import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace

from supervisor.inference_feedback import MeasuredTCPFeed
from supervisor.inference_preview import ObservationSource
from supervisor.pi05_adapter import Pi05InputAdapter
from tests.test_pi05_adapter import _Broker, _InitiallyUnavailablePolicy


class FeedbackTests(unittest.TestCase):
    def test_tcp_only_read_never_queries_gripper(self):
        def forbidden():raise AssertionError('TCP producer must not query gripper')
        broker=SimpleNamespace(sensor_sample=lambda q,w:{'joint_position_rad':[0.]*7,
            'feedback_monotonic_ns':time.perf_counter_ns(),'latest_feedback_age_s':0.},
            inference_gripper_feedback=forbidden)
        source=ObservationSource(SimpleNamespace(osc=broker))
        source.last_joints=(0.,)*7;source.last_pose={'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]}
        result=source.refresh({'session':{}},tcp_only=True)['inference_observation']
        self.assertNotIn('gripper_width_m',result)
        self.assertLess(time.perf_counter_ns()-result['perf_ns'],100_000_000)

    def test_cross_session_cache_cannot_be_used_and_timestamp_never_refreshed(self):
        stop=threading.Event();ctx={'session':{'id':'old','client_id':'c','execution_mode':'hardware'}}
        stamp=time.perf_counter_ns()-80_000_000
        feed=MeasuredTCPFeed(None,lambda:ctx,stop)
        feed.value={'identity':('old','c','hardware'),'observation':{'perf_ns':stamp},'published_perf_ns':stamp+1}
        same=feed.attach(copy.deepcopy(ctx));self.assertEqual(same['inference_observation']['perf_ns'],stamp)
        self.assertGreaterEqual(feed.status()['sample_age_ms'],80.)
        self.assertEqual(feed.status()['sample_perf_ns'],stamp)
        other=feed.attach({'session':dict(ctx['session'],id='new'),'inference_observation':{'perf_ns':0}})
        self.assertNotIn('inference_observation',other)

    def test_slow_gripper_and_state_do_not_block_live_tcp_cache(self):
        release=threading.Event();blocked=threading.Event();slow=threading.Event()
        class Broker(_Broker):
            def __init__(self):super().__init__();self.session['execution_mode']='hardware'
            def sensor_sample(self,q,w):return {}
            def inference_gripper_feedback(self):
                if slow.is_set():blocked.set();release.wait(2)
                return {'width_m':.02,'raw':{'value':{'timestamp':time.time()}}}
        class Source:
            def __init__(self,a):self.calls=0
            def refresh(self,snapshot,*,tcp_only=False):
                assert tcp_only
                self.calls+=1
                now=time.perf_counter_ns()
                snapshot['inference_observation']={'pose':{'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]},
                    'perf_ns':now,'feedback_age_s':0.,'source':'measured_joint_fk','timing':{}}
                return snapshot
            def close(self):pass
        broker=Broker()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config/pi05.json';p.parent.mkdir()
            p.write_text((Path(__file__).parents[1]/'config/pi05.json').read_text())
            with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy),patch('supervisor.pi05_adapter.ObservationSource',Source):
                a=Pi05InputAdapter(broker,p)
            try:
                deadline=time.perf_counter()+2
                while time.perf_counter()<deadline:
                    snap=a._inference_snapshot()
                    if snap.get('inference_observation') and snap.get('session',{}).get('id')=='session-1':break
                    time.sleep(.005)
                slow.set();self.assertTrue(blocked.wait(1))
                calls=a._observation_source.calls
                time.sleep(.16)
                measured=a._inference_snapshot()['inference_observation']
                self.assertGreaterEqual(a._observation_source.calls-calls,4)
                self.assertLess(time.perf_counter_ns()-measured['perf_ns'],100_000_000)
                self.assertEqual(broker.commands,[])
                snapshot=a._inference_snapshot()
                snapshot['gripper']={'width_m':.02}
                observation=a._observation(snapshot,np.zeros((224,224,3),np.uint8),np.zeros((224,224,3),np.uint8))
                self.assertEqual(observation['observation/state'].shape,(7,))
            finally:release.set();a.close()
            self.assertFalse(a._tcp_feed.thread.is_alive())


if __name__=='__main__':unittest.main()
