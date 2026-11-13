"""Hardware-free regressions for PICO dispatch and confirmed HOLD resume."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from nero_backend.robot import NeroRobot
from supervisor.pico_adapter import PicoInputAdapter
module=types.SimpleNamespace(PicoInputAdapter=PicoInputAdapter)


def robot(cpv=True,healthy=True):
    value=NeroRobot(dict(sdk=dict(task_tcp_offset_from_flange_m=[.175,0,-.0235]),motion={},safety={}))
    value.robot=object();value._cpv_stream_started=cpv;value._control_mode='OSC_CPV'
    calls=[]
    def send(**kwargs):calls.append(kwargs);return True
    value.gripper=types.SimpleNamespace(move_gripper_m=send)
    value.read_gripper=lambda:types.SimpleNamespace(width_m=.05,to_dict=lambda:dict(width_m=.05))
    value._gripper_health=lambda *args,**kwargs:dict(ok=healthy,driver_enabled=healthy,faults=[] if healthy else ['driver_disabled'])
    value.get_gripper_hold_state=lambda:{}
    return value,calls


def adapter(mode,backend):
    class Port:
        def __init__(self):self.calls=[]
        def state(self):return dict(session=dict(id='session',client_id='client',state='ACTIVE',execution_mode=mode))
        def gripper(self,s,c,n,payload):
            self.calls.append(dict(payload))
            result=backend.command_gripper(**payload) if mode=='hardware' else dict(ok=True)
            return dict(ok=result['ok'],result=result)
    port=Port();value=module.PicoInputAdapter(port,{})
    value._session_id='session';value._client_id='client';value._execution_mode=mode
    return value,port


class PicoQuickGripperTests(unittest.TestCase):
    def test_hardware_routes_to_existing_fast_path_without_sleep(self):
        backend,calls=robot();value,port=adapter('hardware',backend)
        with patch('nero_backend.robot.time.sleep',side_effect=AssertionError('Blocking sleep')):
            result=value.gripper(.25)
        self.assertTrue(result['ok']);self.assertEqual(port.calls[0]['mode'],'position_inference')
        self.assertEqual(len(calls),1);self.assertAlmostEqual(calls[0]['value'],.07125)

    def test_shadow_keeps_original_mode_and_no_hardware_call(self):
        backend,calls=robot();value,port=adapter('shadow',backend)
        self.assertTrue(value.gripper(.25)['ok']);self.assertEqual(port.calls[0]['mode'],'position');self.assertEqual(calls,[])

    def test_receive_only_standby_never_sends(self):
        backend,calls=robot();value,port=adapter('hardware',backend);value._control_enabled=False
        value.gripper(.5);self.assertEqual(port.calls,[]);self.assertEqual(calls,[])

    def test_no_cpv_does_not_fall_back_to_blocking_recovery(self):
        backend,calls=robot(cpv=False);value,_=adapter('hardware',backend)
        with patch('nero_backend.robot.time.sleep',side_effect=AssertionError('Blocking recovery')):
            self.assertFalse(value.gripper(.5)['ok'])
        self.assertEqual(calls,[])

    def test_disabled_gripper_remains_rejected(self):
        backend,calls=robot(healthy=False);value,_=adapter('hardware',backend)
        with patch('nero_backend.robot.time.sleep',side_effect=AssertionError('Blocking recovery')):
            self.assertFalse(value.gripper(.5)['ok'])
        self.assertEqual(calls,[])

    def test_manual_position_retains_existing_verification(self):
        backend,calls=robot()
        with patch('nero_backend.robot.time.sleep') as sleep:
            result=backend.command_gripper('position',.05,1.)
        self.assertTrue(result['ok']);sleep.assert_called_once_with(.25);self.assertEqual(len(calls),1)

    def test_existing_width_and_force_validation_retained(self):
        backend,calls=robot()
        for width,force in [(float('nan'),1.),(.05,float('inf')),(-1.,1.),(.05,100.)]:
            with self.assertRaises(ValueError):backend.command_gripper('position_inference',width,force)
        self.assertEqual(calls,[])

from nero_console.application.adapter_runtime import OscClient
class Broker:
    def __init__(self,state): self.stage=state; self.calls=[]; self.session='current'
    def osc_fast_state(self): return {'diagnostics':{'trajectory_state':self.stage}}
    def osc_track_tcp_nonblocking(self,*args): self.calls.append(('mailbox',args));return {'ok':True}
    def osc_command(self,body):
        self.calls.append(('resume',body))
        if body['session_id']!=self.session: raise PermissionError('stale session')
        if self.stage!='HOLD_READY':return {'ok':False,'result':{'accepted':False,'recoverable':True}}
        self.stage='TRACKING';return {'ok':True,'result':{'accepted':True}}
class PicoHoldResumeTests(unittest.TestCase):
    def send(self,stage,source='pico',session='current'):
        b=Broker(stage);c=OscClient(b,source);r=c.track_tcp(session,'owner',1,{'position_m':[0,0,0],'orientation_xyzw':[0,0,0,1]});return b,c,r
    def test_hold_uses_guarded_resume(self):
        b,c,r=self.send('HOLD_READY');self.assertEqual(b.calls[0][0],'resume');self.assertTrue(r['ok'])
    def test_braking_does_not_send(self):
        b,c,r=self.send('BRAKING');self.assertFalse(r['ok']);self.assertEqual(b.calls[0][0],'resume')
    def test_continuous_fast_path(self):self.assertEqual(self.send('TRACKING')[0].calls[0][0],'mailbox')
    def test_model_unchanged(self):self.assertEqual(self.send('HOLD_READY','pi05')[0].calls[0][0],'mailbox')
    def test_stale_session_rejected(self):
        with self.assertRaises(PermissionError):self.send('HOLD_READY',session='old')
    def test_repeat_release_regrip(self):
        b,c,r=self.send('HOLD_READY')
        for _ in range(4):
            c.track_tcp('current','owner',2,{})
            b.stage='HOLD_READY';c.track_tcp('current','owner',3,{})
        self.assertEqual([x[0] for x in b.calls],['resume']+['mailbox','resume']*4)
    def test_other_sources_unchanged(self):self.assertEqual(self.send('HOLD_READY','external')[0].calls[0][0],'mailbox')
if __name__=='__main__':unittest.main()
