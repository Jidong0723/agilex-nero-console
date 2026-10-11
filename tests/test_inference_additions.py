import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from supervisor.inference_recording import RunCapture, await_target, distance
from supervisor.pi05_adapter import Pi05InputAdapter
from test_pi05_adapter import _Broker, _Camera, _InitiallyUnavailablePolicy

POSE={'position_m':[0.,0.,.3],'orientation_xyzw':[0.,0.,0.,1.]}


class InferenceAdditionsTests(unittest.TestCase):
    def cached(self, sample=2, sequence=20, generation=3):
        return {'session':{'state':'ACTIVE','id':'s','client_id':'c'},
                'command':{'sequence':sequence,'target_generation':3,'target_tcp':copy.deepcopy(POSE)},
                'execution':{'control_sample_id':sample,'target_generation':generation,'measured_tcp_pose':copy.deepcopy(POSE)}}

    def wait(self, state, previous=1, seconds=.025, event=None):
        return await_target(lambda:state,POSE,('s','c'),20,previous,time.perf_counter()+seconds,event or threading.Event())

    def test_arrival_and_quaternion_sign(self):
        state=self.cached(); state['execution']['measured_tcp_pose']['orientation_xyzw']=[0,0,0,-1]
        self.assertEqual(distance(POSE,state['execution']['measured_tcp_pose']),(0,0))
        self.assertEqual(self.wait(state)['outcome'],'arrived')

    def test_old_missing_and_other_target_feedback_timeout(self):
        for state in [self.cached(sample=1),self.cached(sample=None),self.cached(sequence=19),self.cached(generation=2),{}]:
            result=self.wait(state)
            self.assertEqual(result['outcome'],'cancelled' if not state else 'timeout')
            self.assertLess(result['blocking_ms'],60)

    def test_tolerance_and_cancel(self):
        state=self.cached(); state['execution']['measured_tcp_pose']['position_m'][0]=.004
        self.assertEqual(self.wait(state)['outcome'],'timeout')
        state['execution']['measured_tcp_pose']['position_m'][0]=.002
        self.assertEqual(self.wait(state)['outcome'],'arrived')
        event=threading.Event(); event.set()
        self.assertEqual(self.wait(state,event=event)['outcome'],'cancelled')
        self.assertEqual(self.wait(state,seconds=-.1)['outcome'],'timeout')

    def test_log_samples_order_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=RunCapture(tmp,{'mode':'shadow'},self.cached)
            with self.assertRaises(RuntimeError): log.zip_bytes()
            log.emit('model_chunk',actions=[[.1]*7]*20)
            time.sleep(.075); log.finish('operator stopped')
            _,data=log.zip_bytes()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                rows=[json.loads(s) for s in archive.read('events.jsonl').decode().splitlines()]
            self.assertEqual(rows[0]['event'],'run_start'); self.assertEqual(rows[-1]['event'],'run_end')
            timestamps=[r['perf_ns'] for r in rows]; self.assertEqual(timestamps,sorted(timestamps))
            samples=[r for r in rows if r['event']=='feedback_sample']
            self.assertGreaterEqual(len(samples),3)
            self.assertTrue(samples[0]['fresh_source_sample'])
            self.assertTrue(any(not r['fresh_source_sample'] for r in samples[1:]))
            self.assertFalse(samples[0]['reference_is_can_transmission'])
            self.assertFalse(samples[0]['can_record_available'])
            self.assertEqual(len(next(r for r in rows if r['event']=='model_chunk')['actions']),20)

    def test_write_failure_is_visible_and_refuses_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=RunCapture(tmp,{},self.cached)
            log.emit('unsupported',value=object())
            time.sleep(.03); log.finish('test')
            self.assertIsNotNone(log.status()['write_error'])
            with self.assertRaises(RuntimeError): log.zip_bytes()

    def test_export_http_route_uses_no_broker(self):
        from types import SimpleNamespace
        from urllib.request import urlopen
        import sys
        sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
        with patch('nero_console.runtime.control_python',return_value=Path(sys.executable)):
            from scripts.nero_control_server import ControlRequestHandler, ExclusiveThreadingHTTPServer
        with tempfile.TemporaryDirectory() as tmp:
            log=RunCapture(tmp,{},self.cached); log.finish('stopped')
            class Handler(ControlRequestHandler):
                runtime=SimpleNamespace(require_adapters=lambda:SimpleNamespace(pi05=SimpleNamespace(export_inference_log=log.zip_bytes)))
                def log_message(inner,*args): pass
                @property
                def broker(inner): raise AssertionError('Export must not query OSC')
            server=ExclusiveThreadingHTTPServer(('127.0.0.1',0),Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            try:
                with urlopen(f'http://127.0.0.1:{server.server_address[1]}/api/pi05/log/export') as response:
                    self.assertEqual(response.headers['Content-Type'],'application/zip')
                    with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                        self.assertIn('events.jsonl',archive.namelist())
            finally: server.shutdown(); server.server_close(); thread.join(1)

    def create(self,tmp):
        path=Path(tmp)/'config/pi05.json'; path.parent.mkdir()
        source=Path(__file__).parents[1]/'config/pi05.json'; path.write_text(source.read_text())
        with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy):
            adapter=Pi05InputAdapter(_Broker(),path)
        adapter._connection_stop.set(); adapter._connection_wakeup.set(); adapter._connection_thread.join(2)
        return adapter

    def wait_for(self, condition,seconds=2):
        end=time.perf_counter()+seconds
        while not condition() and time.perf_counter()<end: time.sleep(.002)
        self.assertTrue(condition())

    def test_profiles_prefix_and_zero_wait(self):
        for name,horizon,prefix,port in [('local5090',10,5,8000),('autodl',10,5,8001)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                adapter=self.create(tmp)
                try:
                    adapter.update_config({'inference_profile':name})
                    self.assertEqual(adapter.config['model']['port'],port)
                    self.assertEqual(adapter.config['execution']['expected_chunk_steps'],horizon)
                    self.assertEqual(adapter.config['execution']['replan_steps'],prefix)
                    adapter.config['execution'].update(period_s=.003,arrival_wait_s=0)
                    class Policy:
                        def infer(inner,obs): return {'actions':np.array([[.001*i,0,0,0,0,0,.5] for i in range(horizon)])}
                        def close(inner): pass
                    adapter._policy=Policy(); adapter.cameras=_Camera(); adapter.start('session-1','client-1')
                    self.wait_for(lambda:adapter.state['executed_steps']>=prefix)
                    adapter.stop('operator stopped')
                    _,data=adapter.export_inference_log()
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        rows=[json.loads(s) for s in archive.read('events.jsonl').decode().splitlines()]
                    chunk=next(r for r in rows if r['event']=='model_chunk')
                    self.assertEqual(len(chunk['actions']),horizon)
                    steps=[r for r in rows if r['event']=='tcp_return' and r['chunk_id']==chunk['chunk_id']]
                    self.assertEqual([r['step'] for r in steps],list(range(prefix)))
                    self.assertTrue(any(r.get('outcome')=='disabled' for r in rows))
                finally: adapter.close()

    def test_horizon_mismatch_does_not_send_any_tcp(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.create(tmp)
            class WrongPolicy:
                def infer(inner,obs): return {'actions':np.array([[.001,0,0,0,0,0,.5]]*20)}
                def close(inner): pass
            try:
                adapter.update_config({'inference_profile':'local5090'})
                adapter._policy=WrongPolicy(); adapter.cameras=_Camera(); adapter.start('session-1','client-1')
                self.wait_for(lambda:adapter.state['state']=='ERROR')
                self.assertFalse(any(c['type']=='track_tcp' for c in adapter.osc.commands))
                self.assertIn('exactly 10',adapter.state['last_error'])
                self.wait_for(lambda:adapter._completed_capture is not None)
                self.assertTrue(adapter._completed_capture.status()['export_ready'])
            finally: adapter.close()

    def test_stop_discards_inflight_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.create(tmp); entered=threading.Event(); release=threading.Event()
            class SlowPolicy:
                def infer(inner,obs):
                    entered.set(); release.wait(1)
                    return {'actions':np.array([[.001,0,0,0,0,0,.5]]*10)}
                def close(inner): release.set()
            try:
                adapter._policy=SlowPolicy(); adapter.cameras=_Camera(); adapter.start('session-1','client-1')
                self.assertTrue(entered.wait(1)); adapter.stop('stopped during inference'); release.set()
                time.sleep(.04)
                self.assertFalse(any(c['type']=='track_tcp' for c in adapter.osc.commands))
                self.assertTrue(adapter._completed_capture.status()['export_ready'])
            finally: release.set(); adapter.close()

    def test_switch_closes_old_client_and_preserves_other_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.create(tmp)
            class Client:
                closed=False
                def close(inner): inner.closed=True
            client=Client(); adapter._policy=client
            old=copy.deepcopy(adapter.config)
            try:
                adapter.update_config({'inference_profile':'local5090','policy_endpoint':{'host':'localhost','port':8000}})
                self.assertTrue(client.closed); self.assertIsNone(adapter._policy)
                self.assertEqual(adapter.config['cameras'],old['cameras'])
                self.assertEqual(adapter.config['gripper'],old['gripper'])
                self.assertFalse(adapter.state['execution_enabled'])
                with self.assertRaises(ValueError): adapter.update_config({'execution':{'arrival_wait_s':float('nan')}})
            finally: adapter.close()

    def test_original_gripper_time_consumes_not_extends_arrival_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.create(tmp)
            class DelayedBroker(_Broker):
                def __init__(inner):
                    super().__init__()
                    inner.session['execution_mode']='hardware'
                def gripper(inner,*args):
                    time.sleep(.18)
                    super().gripper(*args)
                    return {'ok':True,'result':{'sent':True}}
            class Policy:
                def infer(inner,obs): return {'actions':np.array([[.001,0,0,0,0,0,.1+.08*i] for i in range(10)])}
                def close(inner): pass
            try:
                adapter.update_config({'inference_profile':'autodl'})
                adapter.osc=DelayedBroker(); adapter._policy=Policy(); adapter.cameras=_Camera()
                adapter.config['execution'].update(period_s=.001,arrival_wait_s=.15,startup_hold_s=0)
                adapter.start('session-1','client-1')
                self.wait_for(lambda:adapter.state.get('arrival_result') is not None,seconds=3)
                adapter.stop('complete')
                result=adapter.state['arrival_result']
                self.assertEqual(result['outcome'],'timeout')
                self.assertLess(result['blocking_ms'],15)
                self.assertGreaterEqual(result['since_tcp_return_ms'],170)
            finally: adapter.close()

if __name__=='__main__': unittest.main()
