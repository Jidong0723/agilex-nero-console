import copy
import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace

from scripts.nero_control_server import BackendProcessProxy, _backend_fast_requests
from supervisor.control import OperationalSpaceController
from supervisor.inference_preview import ObservationSource


class InferenceSensorTests(unittest.TestCase):
    def test_raw_reader_preserves_producer_timestamp_without_decoration(self):
        raw = {'revision':31, 'joints':[.1]*7, 'velocities':[0.]*7,
               'alignment_monotonic_ns':12345, 'latest_feedback_age_s':.006}
        controller = OperationalSpaceController.__new__(OperationalSpaceController)
        controller._osc = SimpleNamespace(rx_sample_nearest=lambda now,wait:copy.deepcopy(raw))
        def forbidden(*args):
            raise AssertionError('Raw inference read must not decorate OSC/gripper state')
        controller._decorate_osc_sensor_samples = forbidden
        controller._dataset_read_gripper = forbidden
        reply = controller.osc_inference_sensor_sample()
        self.assertEqual(reply['feedback_monotonic_ns'], 12345)
        self.assertEqual(reply['joint_position_rad'], [.1]*7)
        self.assertNotIn('pink', reply)
        self.assertEqual(reply['latest_feedback_age_s'], .006)

    def test_sensor_channel_works_while_both_other_rpc_locks_are_held(self):
        general_parent, general_child = multiprocessing.Pipe()
        fast_parent, fast_child = multiprocessing.Pipe()
        parent, child = multiprocessing.Pipe(); stopped = threading.Event()
        controller = SimpleNamespace(osc_inference_sensor_sample=lambda:{'feedback_monotonic_ns':12345})
        worker = threading.Thread(target=_backend_fast_requests,
            args=(child,controller,stopped,{'osc_inference_sensor_sample'}),daemon=True)
        worker.start()
        proxy = BackendProcessProxy(SimpleNamespace(is_alive=lambda:True), general_parent,
                                    fast_connection=fast_parent, sensor_connection=parent)
        try:
            with proxy._lock, proxy._fast_lock:
                result = proxy.call('osc_inference_sensor_sample')
                self.assertEqual(result['feedback_monotonic_ns'],12345)
                self.assertEqual(result['sensor_ipc']['path'],'dedicated_sensor_pipe')
            parent.send({'id':2,'method':'osc_track_tcp_nonblocking'})
            self.assertTrue(parent.poll(1.))
            self.assertEqual(parent.recv()['error_type'],'PermissionError')
        finally:
            stopped.set(); parent.close(); worker.join(1.); child.close()
            for connection in (general_parent,general_child,fast_parent,fast_child):connection.close()

    def test_model_uses_new_reader_and_keeps_100ms_stale_rejection(self):
        stamp = time.perf_counter_ns()
        def forbidden(*args):raise AssertionError('Do not use decorated dataset RPC')
        def raw():
            return {'joint_position_rad':[0.]*7, 'feedback_monotonic_ns':stamp,
                    'latest_feedback_age_s':0.,'sensor_path':'raw_inference_feedback'}
        source = ObservationSource(SimpleNamespace(osc=SimpleNamespace(sensor_sample=forbidden,
                                      inference_sensor_sample=raw)))
        source.last_joints=(0.,)*7
        source.last_pose={'position_m':[0,0,.3],'orientation_xyzw':[0,0,0,1]}
        measured=source.refresh({'session':{}},tcp_only=True)['inference_observation']
        self.assertEqual(measured['perf_ns'],stamp)
        self.assertEqual(measured['timing']['sensor_path'],'raw_inference_feedback')
        stamp -= 200_000_000
        with self.assertRaisesRegex(RuntimeError,'age_ms'):
            source.refresh({'session':{}},tcp_only=True)


if __name__=='__main__':unittest.main()
