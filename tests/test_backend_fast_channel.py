import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace

from scripts.nero_control_server import BackendProcessProxy, _backend_fast_requests


class FastChannelTests(unittest.TestCase):
    def test_target_and_cache_do_not_wait_for_blocked_ui_rpc(self):
        parent, child = multiprocessing.Pipe()
        fast_parent, fast_child = multiprocessing.Pipe()
        stopped = threading.Event(); entered = threading.Event(); release = threading.Event()
        controller = SimpleNamespace(osc_track_tcp_nonblocking=lambda *a: {'ok':True, 'accepted':True},
                                     osc_fast_state=lambda: {'session': {'state':'ACTIVE'}})
        worker = threading.Thread(target=_backend_fast_requests, args=(fast_child, controller, stopped), daemon=True)
        worker.start()
        def slow_ui():
            request = child.recv(); entered.set(); release.wait(2.)
            child.send({'kind':'result','id':request['id'],'result':{'ui':True}})
        ui = threading.Thread(target=slow_ui, daemon=True); ui.start()
        proxy = BackendProcessProxy(SimpleNamespace(is_alive=lambda:True), parent, fast_connection=fast_parent)
        reader = threading.Thread(target=lambda: proxy.call('osc_state'), daemon=True); reader.start()
        try:
            self.assertTrue(entered.wait(1.))
            result = proxy.call('osc_track_tcp_nonblocking', 's', 'c', 1, {})
            self.assertTrue(result['accepted'])
            self.assertEqual(result['track_tcp_timing']['ipc_path'], 'dedicated_fast_pipe')
            self.assertTrue(reader.is_alive(), 'UI must still be blocked when target is accepted')
            self.assertEqual(proxy.call('osc_fast_state')['session']['state'], 'ACTIVE')
        finally:
            release.set(); reader.join(2.); stopped.set()
            fast_parent.close(); parent.close(); child.close()
            worker.join(1.)
            fast_child.close()

    def test_fast_receiver_rejects_unrelated_hardware_calls(self):
        parent, child = multiprocessing.Pipe(); stopped = threading.Event()
        thread = threading.Thread(target=_backend_fast_requests, args=(child, object(), stopped), daemon=True)
        thread.start()
        try:
            parent.send({'id':1,'method':'initialize_robot'})
            self.assertTrue(parent.poll(1.))
            reply = parent.recv()
            self.assertEqual(reply['error_type'], 'PermissionError')
        finally:
            stopped.set(); parent.close(); thread.join(1.); child.close()


if __name__ == '__main__':
    unittest.main()
