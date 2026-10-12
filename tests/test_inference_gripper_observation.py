import copy,json,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from nero_console.application.adapter_runtime import OscClient
from supervisor.pi05_adapter import Pi05InputAdapter
from shared.schemas import GripperState

class GripperObservationTests(unittest.TestCase):
    def feedback(self,width=.02,marker=2):
        return {'width_m':width,'timestamp':'outer timestamp changes every read',
                'status':{'foc_status':{'driver_enable_status':True}},
                'raw':{'ok':True,'value':{'timestamp':marker}}}
    def adapter(self):
        a=Pi05InputAdapter.__new__(Pi05InputAdapter)
        a.lock=threading.RLock();a.state={'session_id':'s','client_id':'c','execution_enabled':True}
        a._gripper_observation_cache=None;a._gripper_observation_identity=None
        a.config={'gripper':{'min_width_m':0.,'max_width_m':.095},'model':{'prompt':'original task'}}
        a.events=[];a._event=lambda kind,**fields:a.events.append((kind,fields))
        a.calls=0;a.feedback=self.feedback()
        def read():a.calls+=1;return copy.deepcopy(a.feedback)
        a.osc=SimpleNamespace(inference_gripper_feedback=read)
        return a
    def snapshot(self,mode='hardware',session='s'):
        return {'session':{'state':'ACTIVE','id':session,'client_id':'c','execution_mode':mode},
                'execution':{'measured_tcp_pose':{'position_m':[.1,.2,.3],'orientation_xyzw':[0,0,0,1]}},
                'gripper':self.feedback(.094319,1)}

    def test_current_cache_replaces_old_observation_and_preserves_7d_contract(self):
        a=self.adapter();source=self.snapshot();updated=a._refresh_gripper_observation(source)
        self.assertEqual(source['gripper']['width_m'],.094319)
        self.assertEqual(updated['gripper']['width_m'],.02)
        image=np.zeros((224,224,3),np.uint8)
        state=a._observation(updated,image,image)['observation/state']
        self.assertEqual(state.shape,(7,));self.assertAlmostEqual(float(state[6]),.02/.095,places=6)
        self.assertTrue(updated['gripper_observation']['new_sdk_feedback'])

    def test_same_outer_timestamp_does_not_count_as_new_sdk_data(self):
        a=self.adapter();a._refresh_gripper_observation(self.snapshot())
        a.feedback['timestamp']='new outer time';a.feedback['width_m']=.08
        updated=a._refresh_gripper_observation(self.snapshot())
        self.assertEqual(updated['gripper']['width_m'],.02)
        self.assertFalse(updated['gripper_observation']['new_sdk_feedback'])
        a.feedback=self.feedback(.03,3)
        self.assertEqual(a._refresh_gripper_observation(self.snapshot())['gripper']['width_m'],.03)

    def test_regressed_missing_and_invalid_feedback_are_visible_not_faked(self):
        a=self.adapter();a._refresh_gripper_observation(self.snapshot())
        for bad in (self.feedback(.08,1),self.feedback(float('nan'),3),self.feedback(.03,None)):
            a.feedback=bad;updated=a._refresh_gripper_observation(self.snapshot())
            self.assertEqual(updated['gripper']['width_m'],.02)
            self.assertIn('error',updated['gripper_observation'])

    def test_shadow_stop_and_other_session_make_no_reader_call(self):
        a=self.adapter()
        a._refresh_gripper_observation(self.snapshot(mode='shadow'))
        a._refresh_gripper_observation(self.snapshot(session='other'))
        a.state['execution_enabled']=False;a._refresh_gripper_observation(self.snapshot())
        self.assertEqual(a.calls,0);self.assertIsNone(a._gripper_observation_cache)

    def test_new_session_cannot_inherit_previous_feedback(self):
        a=self.adapter();a._refresh_gripper_observation(self.snapshot())
        a.state['session_id']='new';a.feedback=self.feedback(.04,1)
        updated=a._refresh_gripper_observation(self.snapshot(session='new'))
        self.assertEqual(updated['gripper']['width_m'],.04)

    def test_client_reuses_existing_readonly_backend_getter(self):
        calls=[]
        def getter():calls.append('read');return GripperState(timestamp='outer',width_m=.023,raw={'value':{'timestamp':5}})
        broker=SimpleNamespace(_dataset_read_gripper=getter)
        result=OscClient(broker).inference_gripper_feedback()
        self.assertEqual(calls,['read']);self.assertEqual(result['width_m'],.023)
        self.assertEqual(result['raw']['value']['timestamp'],5)

    def test_existing_rpc_transports_gripper_state_without_a_control_command(self):
        from multiprocessing import Pipe
        from unittest.mock import patch
        with patch('nero_console.runtime.control_python',return_value=Path(sys.executable)):
            from scripts.nero_control_server import BackendProcessProxy
        client,server=Pipe();requests=[]
        def worker():
            request=server.recv();requests.append(request['method'])
            server.send({'kind':'result','id':request['id'],
                'result':GripperState(timestamp='outer',width_m=.025,raw={'value':{'timestamp':8}})})
        thread=threading.Thread(target=worker);thread.start()
        try:
            proxy=BackendProcessProxy(SimpleNamespace(is_alive=lambda:True),client)
            result=OscClient(proxy).inference_gripper_feedback()
            self.assertEqual(requests,['_dataset_read_gripper'])
            self.assertEqual(result['width_m'],.025)
            self.assertEqual(result['raw']['value']['timestamp'],8)
        finally:thread.join(1);client.close();server.close()

    def test_fresh_feedback_reaches_confirmation_and_logging(self):
        a=self.adapter();a._run_revision=1
        a._pending_gripper={'sequence':8,'target_width_m':.02,'last_sdk_marker':1,'run_revision':1,'session_id':'s','client_id':'c'}
        updated=a._refresh_gripper_observation(self.snapshot())
        a._observe_inference_gripper(updated)
        self.assertEqual(a.state['gripper_confirmation']['actual_width_m'],.02)
        self.assertTrue(a.state['gripper_confirmation']['health_confirmed'])
        self.assertIsNone(a.state['gripper_confirmation']['target_reached'])
        a._publish_logging_feedback(updated)
        frame=a._logging_feedback()
        self.assertEqual(frame['gripper']['raw']['value']['timestamp'],2)
        self.assertTrue(frame['gripper_observation']['new_sdk_feedback'])

if __name__=='__main__':unittest.main()
