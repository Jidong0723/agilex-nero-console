from __future__ import annotations

import threading
import unittest
from collections import deque
from types import SimpleNamespace

from can import Message
from pyAgxArm.protocols.can_protocol.comms.can_comm import CanCommImpl


class CanCommTxOwnerTests(unittest.TestCase):
    def make_comm(self) -> CanCommImpl:
        comm = CanCommImpl.__new__(CanCommImpl)
        comm.recv_bus = SimpleNamespace(shutdown=lambda: None)
        comm.send_bus = SimpleNamespace(send=lambda _msg, _timeout: None, state="ACTIVE")
        comm._send_diagnostics_lock = threading.Lock()
        comm._active_send_diagnostic = None
        comm._slow_send_events = deque(maxlen=16)
        comm._slow_send_threshold_s = 0.05
        comm._channel = "0"
        comm._interface = "agx_cando"
        comm._is_connected = True
        comm._is_stopped = False
        comm.last_error = None
        comm._tx_owner_thread_id = None
        return comm

    def test_rejects_send_outside_bound_tx_owner_thread(self) -> None:
        comm = self.make_comm()
        comm.bind_tx_owner_thread(threading.get_ident() + 1)
        with self.assertRaisesRegex(RuntimeError, "outside HardwareTxOwner"):
            comm.send(Message(arbitration_id=0x187, is_extended_id=False, data=b"\x01"))

    def test_closes_shared_bus_once(self) -> None:
        calls = []
        shared = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
        comm = self.make_comm()
        comm.recv_bus = shared
        comm.send_bus = shared
        comm.close()
        self.assertEqual(calls, ["shutdown"])


if __name__ == "__main__":
    unittest.main()
