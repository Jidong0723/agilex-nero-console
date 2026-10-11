import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from test_robot_modes import RobotModeTests
from test_pi05_adapter import _Broker, _InitiallyUnavailablePolicy
from supervisor.pi05_adapter import Pi05InputAdapter


class InferenceGripperTests(unittest.TestCase):
    def robot(self):
        robot,sdk=RobotModeTests().make_robot()
        robot._control_mode='HOLD'; robot._freedrive_backend=None
        robot._cpv_stream_started=True
        robot.motion_config['gripper_verify_settle_s']=.25
        return robot,sdk

    def test_fast_sends_once_and_never_sleeps_or_spawns(self):
        robot,sdk=self.robot()
        with patch('nero_backend.robot.time.sleep',side_effect=AssertionError('sender must not sleep')),patch('nero_backend.robot.threading.Thread') as spawn:
            result=robot.command_gripper('position_inference',.04,.8)
        self.assertTrue(result['ok']); self.assertTrue(result['sent'])
        self.assertTrue(result['feedback_pending']); self.assertFalse(result['feedback_confirmed'])
        self.assertIsNone(result['target_reached']); self.assertEqual(len(robot.gripper.commands),1)
        spawn.assert_not_called()

    def test_ordinary_modes_keep_original_wait(self):
        for mode,width in [('position',.04),('open',None),('grip',0.)]:
            robot,_=self.robot()
            with patch('nero_backend.robot.time.sleep') as sleep:
                result=robot.command_gripper(mode,width,.8)
            self.assertTrue(result['ok']); sleep.assert_called_once_with(.25)

    def test_invalid_values_health_and_cpv_are_not_bypassed(self):
        robot,_=self.robot()
        for width,force in [(float('nan'),.8),(.04,float('inf')),(-1,.8),(.2,.8),(.04,4)]:
            with self.assertRaises(ValueError): robot.command_gripper('position_inference',width,force)
        robot.gripper.enabled=False
        with patch('nero_backend.robot.time.sleep',side_effect=AssertionError('no recovery wait')):
            result=robot.command_gripper('position_inference',.04,.8)
        self.assertFalse(result['ok']); self.assertFalse(result['sent']); self.assertEqual(robot.gripper.commands,[])
        robot.gripper.enabled=True; robot._cpv_stream_started=False
        self.assertFalse(robot.command_gripper('position_inference',.04,.8)['ok'])

    def test_sdk_failure_returns_failure(self):
        robot,_=self.robot()
        robot.gripper.move_gripper_m=lambda **kwargs: False
        self.assertFalse(robot.command_gripper('position_inference',.04,.8)['sent'])
        def fail(**kwargs): raise RuntimeError('fake bus failure')
        robot.gripper.move_gripper_m=fail
        result=robot.command_gripper('position_inference',.04,.8)
        self.assertFalse(result['ok']); self.assertIn('fake bus failure',result['reason'])

    def test_serial_cpv_with_fast_gripper_has_no_250ms_gap(self):
        robot,sdk=self.robot(); times=[]; sender_ids=[]
        original=sdk.move_cpv_pos
        def send(**kwargs):
            sender_ids.append(threading.get_ident())
            return original(**kwargs)
        sdk.move_cpv_pos=send
        grip=robot.gripper.move_gripper_m
        def send_grip(**kwargs):
            sender_ids.append(threading.get_ident()); return grip(**kwargs)
        robot.gripper.move_gripper_m=send_grip
        for i in range(6):
            times.append(time.perf_counter())
            robot.send_cpv_position([.01*i]*7)
            robot.command_gripper('position_inference',.01+.005*i,.8)
            time.sleep(.02)
        self.assertLess(max(b-a for a,b in zip(times,times[1:])),.2)
        self.assertEqual(len(set(sender_ids)),1)

    def adapter(self,tmp):
        path=Path(tmp)/'config/pi05.json'; path.parent.mkdir()
        path.write_text((Path(__file__).parents[1]/'config/pi05.json').read_text(encoding='utf-8'),encoding='utf-8')
        broker=_Broker(); broker.session['execution_mode']='hardware'
        with patch('supervisor.pi05_adapter.OpenPIClient',_InitiallyUnavailablePolicy): adapter=Pi05InputAdapter(broker,path)
        adapter._connection_stop.set(); adapter._connection_wakeup.set(); adapter._connection_thread.join(2)
        adapter._osc_state_stop.set(); adapter._osc_state_thread.join(2)
        adapter.state.update(execution_enabled=True,execution_mode='hardware',session_id='session-1',client_id='client-1')
        adapter.events=[]; adapter._event=lambda event,**fields:adapter.events.append((event,fields))
        return adapter

    def feedback(self,marker,width=.04,enabled=True):
        return {'timestamp':'fresh-wrapper-even-when-sdk-is-old','width_m':width,
                'raw':{'value':{'timestamp':marker}},'status':{'foc_status':{'driver_enable_status':enabled}}}

    def send(self,adapter,value=.5):
        def request(s,c,sequence,payload):
            adapter.osc.commands.append({'type':'gripper','sequence':sequence,'payload':payload})
            return {'ok':True,'result':{'sent':True,'before':self.feedback(100)}}
        adapter.osc.gripper=request
        adapter._send_gripper_if_needed('session-1','client-1',value)

    def test_async_marker_width_health_and_replace_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.adapter(tmp)
            try:
                self.send(adapter)
                self.assertEqual(adapter.osc.commands[-1]['payload']['mode'],'position_inference')
                frame=adapter.osc.state(); frame['gripper']=self.feedback(100)
                adapter._observe_inference_gripper(frame)
                self.assertEqual(adapter.state['gripper_confirmation']['status'],'pending_feedback')
                frame['gripper']=self.feedback(101,.025)
                adapter._observe_inference_gripper(frame)
                status=adapter.state['gripper_confirmation']
                self.assertEqual(status['status'],'feedback_received'); self.assertTrue(status['health_confirmed'])
                self.assertAlmostEqual(status['target_error_m'],abs(.025-.0475)); self.assertIsNone(status['target_reached'])
                old_sequence=status['sequence']; self.send(adapter,.7)
                self.assertGreater(adapter._pending_gripper['sequence'],old_sequence)
                adapter.stop('cancel'); self.assertIsNone(adapter._pending_gripper)
            finally: adapter.close()

    def test_new_feedback_fault_uses_existing_failure_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.adapter(tmp)
            try:
                self.send(adapter)
                frame=adapter.osc.state(); frame['gripper']=self.feedback(101,enabled=False)
                adapter._observe_inference_gripper(frame)
                self.assertEqual(adapter.state['state'],'ERROR')
                self.assertFalse(adapter.state['execution_enabled'])
                self.assertTrue(any(c['type']=='hold' for c in adapter.osc.commands))
            finally: adapter.close()

    def test_shadow_sends_nothing_and_changed_session_cancels(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=self.adapter(tmp)
            try:
                adapter.state['execution_mode']='shadow'
                adapter.osc.gripper=lambda *args: self.fail('shadow must not send')
                adapter._send_gripper_if_needed('session-1','client-1',.5)
                self.assertEqual(adapter.state['gripper_confirmation']['status'],'simulated')
                adapter.state['execution_mode']='hardware'; self.send(adapter,.7)
                frame=adapter.osc.state(); frame['session']['id']='other'
                adapter._observe_inference_gripper(frame)
                self.assertIsNone(adapter._pending_gripper)
            finally: adapter.close()

if __name__=='__main__': unittest.main()
