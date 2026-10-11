import json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from supervisor.pi05_adapter import Pi05InputAdapter
from test_pi05_adapter import _Broker,_Camera,_InitiallyUnavailablePolicy

class SlowDiagnosticsCamera(_Camera):
    def __init__(self):self.block=False;self.entered=threading.Event()
    def snapshot(self):
        if self.block:self.entered.set();time.sleep(.35)
        return {'ready':True,'frame_version':1}

class CountingBroker(_Broker):
    def __init__(self):super().__init__();self.polls=[]
    def state(self):
        self.polls.append(time.perf_counter());r=super().state()
        r['execution']['feedback_revision']=len(self.polls)
        r['execution']['control_sample_id']=len(self.polls)
        return r

class SchedulingTests(unittest.TestCase):
    def test_active_camera_status_reads_metadata_not_driver_properties(self):
        from types import SimpleNamespace
        a=Pi05InputAdapter.__new__(Pi05InputAdapter)
        a._camera_status_snapshot={'sources':{'external':{'capture':{'backend':'opencv'}}}}
        now=time.perf_counter_ns()
        def forbidden():raise AssertionError('Active UI must not query driver properties')
        a.camera_resource=SimpleNamespace(lock=threading.RLock(),cameras=object(),frame_times_ns={'external':now,'wrist':now},
            frame_version=5,config={},dataset_frame_history={'external':[(now-50_000_000,None),(now,None)],'wrist':[]},
            dataset_frame_sizes={'external':(640,480)},last_error=None,snapshot=forbidden)
        result=a._camera_ui_snapshot(True)
        self.assertEqual(result['frame_version'],5)
        self.assertEqual(result['sources']['external']['capture_hz'],20)
        self.assertEqual(result['sources']['external']['capture']['backend'],'opencv')
        self.assertEqual(result['capture_properties_source'],'last_idle_diagnostics')

    def make(self,tmp,horizon=10):
        p=Path(tmp)/'config/pi05.json';p.parent.mkdir()
        c=json.loads((Path(__file__).parents[1]/'config/pi05.json').read_text())
        c['execution'].update(expected_chunk_steps=horizon,max_chunk_steps=horizon,replan_steps=horizon//2,arrival_wait_s=0,period_s=.05,osc_state_poll_s=.05)
        p.write_text(json.dumps(c));camera=SlowDiagnosticsCamera();broker=CountingBroker()
        with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy),patch.object(Pi05InputAdapter,'_local_forward_listening',return_value=True):
            a=Pi05InputAdapter(broker,p,camera)
        a._connection_stop.set();a._connection_wakeup.set();a._connection_thread.join(2)
        return a,camera,broker

    def test_slow_ui_diagnostics_do_not_block_feedback_or_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,camera,broker=self.make(tmp)
            class Policy:
                def infer(self,obs):return {'actions':np.array([[.001*i,0,0,0,0,0,.5] for i in range(10)])}
                def close(self):pass
            a._policy=Policy();a.start('session-1','client-1')
            camera.block=True;finished=threading.Event()
            def ui():
                while not finished.is_set():a.snapshot()
            thread=threading.Thread(target=ui);thread.start()
            try:
                self.assertTrue(camera.entered.wait(1))
                start=len(broker.polls);time.sleep(.75)
                self.assertGreaterEqual(len(broker.polls)-start,12)
                dispatched=[x for x in broker.commands if x['type']=='track_tcp']
                self.assertGreaterEqual(len(dispatched),12)
                # Zero arrival budget plus instantaneous fake inference can
                # put adjacent chunks together; check spacing within prefixes.
                dt=np.array([b['started_at']-a['started_at'] for i,(a,b) in enumerate(zip(dispatched,dispatched[1:])) if i%5!=4])
                self.assertGreaterEqual(float(dt.min()),.045)
                self.assertLess(float(np.percentile(dt,95)),.12)
                self.assertFalse(any(x['type']=='gripper' for x in broker.commands))
            finally:finished.set();thread.join(2);camera.block=False;a.close()

    def test_snapshot_does_not_launch_port_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,camera,broker=self.make(tmp)
            try:
                a._last_connection_probe=0
                with patch.object(a,'_local_forward_listening',side_effect=AssertionError('No UI subprocess')):
                    a.snapshot()
            finally:a.close()

    def test_both_horizons_keep_prefix_and_log_dispatch_timing(self):
        for horizon in (10,20):
            with self.subTest(horizon=horizon),tempfile.TemporaryDirectory() as tmp:
                a,camera,broker=self.make(tmp,horizon)
                class Policy:
                    def infer(self,obs):return {'actions':np.array([[.001*i,0,0,0,0,0,.5] for i in range(horizon)])}
                    def close(self):pass
                a._policy=Policy();a.start('session-1','client-1')
                try:
                    end=time.perf_counter()+2
                    while a.state['executed_steps']<horizon//2 and time.perf_counter()<end:time.sleep(.005)
                    a.stop('test')
                    log=a._completed_capture
                    rows=[json.loads(s) for s in (log.path/'events.jsonl').read_text().splitlines()]
                    chunk=next(r for r in rows if r['event']=='model_chunk')
                    req=[r for r in rows if r['event']=='tcp_request' and r['chunk_id']==chunk['chunk_id']]
                    self.assertEqual([r['step'] for r in req],list(range(horizon//2)))
                    self.assertTrue(all(r['dispatch_preparation_ms']>=0 for r in req))
                    self.assertLess(req[0]['since_inference_return_ms'],100)
                    self.assertTrue(any(r['event']=='feedback_cache_poll' for r in rows))
                finally:a.close()

if __name__=='__main__':unittest.main()
