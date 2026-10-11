import copy
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile
from supervisor.inference_extensions import InferenceJournal,connection_config,pose_error,tcp_delta,wait_for_arrival

POSE={'position_m':[0.,0.,.3],'orientation_xyzw':[0.,0.,0.,1.]}
class ExtensionTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parents[1]/'config/pi05.json').read_text(encoding='utf-8'))
    def test_connection_changes_no_action_or_camera_fields(self):
        initial=self.config();old=copy.deepcopy(initial)
        cloud=connection_config(initial,{'profile':'autodl'})
        local=connection_config(cloud,{'profile':'local5090'})
        self.assertEqual(cloud['model']['port'],8001);self.assertEqual(local['model']['port'],8000)
        for key in ('execution','cameras','gripper'):self.assertEqual(old[key],local[key])
        self.assertEqual(old,initial)
    def test_invalid_connection_rejected(self):
        for request in ({'profile':'unknown'},{'profile':'local5090','port':0},{'profile':'local5090','port':True}):
            with self.assertRaises(ValueError):connection_config(self.config(),request)
    def fake(self,profile,wait,error):
        class Adapter:
            lock=threading.RLock();stop_event=threading.Event()
            state={'execution_enabled':True}
            def __init__(self):self.events=[]
            def _record_inference(self,kind,**fields):self.events.append((kind,fields))
        a=Adapter();a.config=self.config();a.config['execution']['arrival_wait_s']=wait
        a.config['inference_profiles']['active']=profile
        measured=copy.deepcopy(POSE);measured['position_m'][0]=error
        a._osc_snapshot={'session':{'id':'s','client_id':'c','state':'ACTIVE'},
            'execution':{'measured_tcp_pose':measured,'target_generation':1},
            'command':{'target_tcp':copy.deepcopy(POSE),'target_generation':1}}
        return a
    def test_wait_same_for_cloud_and_local(self):
        for profile in ('autodl','local5090'):
            a=self.fake(profile,.01,0);self.assertIsNotNone(wait_for_arrival(a,POSE,'s','c'))
            self.assertEqual(a.events[-1][0],'tcp_arrival_reached')
    def test_timeout_preserves_measured_pose(self):
        a=self.fake('local5090',.01,.02);s=wait_for_arrival(a,POSE,'s','c')
        self.assertEqual(s['execution']['measured_tcp_pose']['position_m'][0],.02)
        self.assertEqual(a.events[-1][0],'tcp_arrival_wait_timeout')
    def test_zero_wait_and_operator_stop(self):
        a=self.fake('autodl',0,.02);wait_for_arrival(a,POSE,'s','c')
        self.assertEqual(a.events[-1][0],'tcp_arrival_wait_bypassed')
        a.state={'execution_enabled':False};self.assertIsNone(wait_for_arrival(a,POSE,'s','c'))
    def test_old_generation_is_not_arrival(self):
        a=self.fake('local5090',.01,0);a._osc_snapshot['execution']['target_generation']=0
        wait_for_arrival(a,POSE,'s','c');self.assertEqual(a.events[-1][0],'tcp_arrival_wait_timeout')
    def test_pose_and_delta_use_base_axes(self):
        after=copy.deepcopy(POSE);after['position_m'][0]=.01
        after['orientation_xyzw']=[0,0,math.sin(.1),math.cos(.1)]
        delta=tcp_delta(POSE,after)
        self.assertAlmostEqual(delta['translation_base_m'][0],.01)
        self.assertAlmostEqual(delta['rotvec_base_rad'][2],.2)
        self.assertEqual(pose_error(None,POSE),(math.inf,math.inf))
    def test_completed_log_export_contains_all_model_commands(self):
        with tempfile.TemporaryDirectory() as root:
            log=InferenceJournal(root,{'profile':'local5090'})
            log.append('model_action_chunk',action_chunk=[[0]*7]*10)
            log.append('tcp_sample',measured_tcp_pose=POSE)
            with self.assertRaises(RuntimeError):log.export()
            log.finish('operator stopped');name,data=log.export()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                records=[json.loads(s) for s in z.read('run.jsonl').splitlines()]
            self.assertEqual([r['record_type'] for r in records],['run_start','model_action_chunk','tcp_sample','run_end'])
            self.assertTrue(name.endswith('.zip'));self.assertIsNone(log.error)
    def test_active_connection_switch_never_sends_commands(self):
        from supervisor.pi05_adapter import Pi05InputAdapter
        a=object.__new__(Pi05InputAdapter);a._profile_lock=threading.Lock();a.lock=threading.RLock()
        a.state={'execution_enabled':True};a.config=self.config()
        with self.assertRaises(RuntimeError):a._change_inference_profile({'profile':'autodl'})

if __name__=='__main__':unittest.main()
