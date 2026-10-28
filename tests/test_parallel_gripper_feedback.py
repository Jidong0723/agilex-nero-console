"""Fake SDK/cache/IPC only: no CAN device, commands or OSC service."""
import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace

from scripts.nero_control_server import BackendProcessProxy, _backend_fast_requests, INFERENCE_SENSOR_READ_METHODS
from supervisor.inference_gripper_feedback import GripperFeedbackCache, GripperFeedbackFeed
from supervisor.pi05_adapter import Pi05InputAdapter


def sdk_sample(stamp=None,width=.02):
    return {'width_m':width,'mode':'width','force_n':0.,
            'raw':{'value':{'timestamp':time.time() if stamp is None else stamp,'hz':200.,'msg_type':'ArmMsgFeedbackGripper'}}}


class ParallelFeedbackTests(unittest.TestCase):
    def context(self):
        return {'session':{'id':'s','client_id':'c','execution_mode':'hardware'}}

    def test_read_channel_runs_while_command_and_state_locks_are_held(self):
        general,unused=multiprocessing.Pipe();fast,unused_fast=multiprocessing.Pipe()
        parent,child=multiprocessing.Pipe();stop=threading.Event()
        controller=SimpleNamespace(_dataset_read_gripper=lambda:sdk_sample(),
            osc_inference_sensor_sample=lambda:{'joint_position_rad':[0.]*7})
        worker=threading.Thread(target=_backend_fast_requests,
            args=(child,controller,stop,INFERENCE_SENSOR_READ_METHODS),daemon=True)
        worker.start()
        proxy=BackendProcessProxy(SimpleNamespace(is_alive=lambda:True),general,
            fast_connection=fast,sensor_connection=parent)
        ctx=self.context(); cache=GripperFeedbackCache()
        feed=GripperFeedbackFeed(lambda:proxy.call('_dataset_read_gripper'),lambda:ctx,stop,cache)
        try:
            with proxy._lock,proxy._fast_lock:
                feed.thread.start()
                # Simulate a 200 ms normal command/UI RPC stall. Read-only
                # sampling must continue while both those locks are occupied.
                time.sleep(.21)
                with cache.history.lock: count=len(cache.history.samples)
                self.assertGreaterEqual(count,8)
                attached=feed.attach(self.context())
                self.assertNotIn('error',attached['gripper_observation'])
                joint=proxy.call('osc_inference_sensor_sample')
                self.assertEqual(joint['joint_position_rad'],[0.]*7)
            # New channel must not admit a hardware writer.
            for method in ('osc_command','command_gripper','osc_track_tcp_nonblocking'):
                parent.send({'id':999,'method':method})
                self.assertTrue(parent.poll(1))
                self.assertEqual(parent.recv()['error_type'],'PermissionError')
        finally:
            stop.set();feed.thread.join(1);parent.close();worker.join(1)
            for c in (child,general,unused,fast,unused_fast):c.close()

    def test_repeated_sdk_cache_cannot_refresh_age_or_create_frames(self):
        stop=threading.Event();ctx=self.context();cache=GripperFeedbackCache();events=[]
        stamp=time.time()
        feed=GripperFeedbackFeed(lambda:sdk_sample(stamp),lambda:ctx,stop,cache,
            lambda *args,**kwargs:events.append((args,kwargs)))
        feed.thread.start()
        try:
            time.sleep(.19)
            self.assertIn('error',feed.attach(self.context())['gripper_observation'])
            with cache.history.lock:self.assertEqual(len(cache.history.samples),1)
            self.assertEqual(len(events),1)
            self.assertFalse(events[0][1]['physical_opening_verified'])
        finally:stop.set();feed.thread.join(1)

    def test_shadow_makes_no_sdk_call_and_session_switch_rejects_old_snapshot(self):
        stop=threading.Event();ctx=self.context();cache=GripperFeedbackCache();calls=[]
        def reader():calls.append(1);return sdk_sample()
        feed=GripperFeedbackFeed(reader,lambda:ctx,stop,cache)
        feed.thread.start()
        try:
            time.sleep(.04)
            self.assertIsNotNone(feed.attach(self.context())['gripper'])
            other=self.context();other['session']['id']='other'
            self.assertIsNone(feed.attach(other)['gripper'])
            ctx['session']['execution_mode']='shadow'
            time.sleep(.025); count=len(calls)
            time.sleep(.04);self.assertEqual(len(calls),count)
            explicit=dict(ctx,gripper={'width_m':.03})
            self.assertEqual(feed.attach(explicit)['gripper']['width_m'],.03)
        finally:stop.set();feed.thread.join(1)
        count=len(calls);time.sleep(.02);self.assertEqual(len(calls),count)

    def test_health_is_not_reported_arrival_or_physical_measurement(self):
        events=[]
        a=SimpleNamespace(lock=threading.RLock(),state={'execution_enabled':True},_run_revision=1,
            _gripper_sdk_marker=Pi05InputAdapter._gripper_sdk_marker,
            _event=lambda *args,**kwargs:events.append(kwargs))
        a._pending_gripper={'sequence':1,'target_width_m':.08,'sent_perf_ns':time.perf_counter_ns(),
            'session_id':'s','client_id':'c','run_revision':1,'last_sdk_marker':1.}
        def observe(stamp,width):
            g=sdk_sample(stamp,width);g['status']={'foc_status':{'driver_enable_status':True}}
            snapshot=self.context();snapshot['session']['state']='ACTIVE';snapshot['gripper']=g
            Pi05InputAdapter._observe_inference_gripper(a,snapshot)
        observe(2.,.02)
        self.assertTrue(a.state['gripper_confirmation']['health_confirmed'])
        self.assertFalse(a.state['gripper_confirmation']['reported_target_reached'])
        observe(3.,.079);observe(3.,.079)
        self.assertFalse(a.state['gripper_confirmation']['reported_target_reached'])
        observe(4.,.079)
        self.assertTrue(a.state['gripper_confirmation']['reported_target_reached'])
        self.assertFalse(a.state['gripper_confirmation']['physical_opening_verified'])
        self.assertIsNone(a.state['gripper_confirmation']['target_reached'])
        self.assertTrue(a.state['execution_enabled'])  # No new arrival gate.


if __name__=='__main__':unittest.main()
