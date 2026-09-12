from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from can import Message
from agx_cando.bus import AgxCandoBus, CandoFrame, _VOID_P


class AgxCandoConcurrencyTests(unittest.TestCase):
    def test_official_bus_has_no_native_rx_tx_lock(self) -> None:
        bus = AgxCandoBus.__new__(AgxCandoBus)
        self.assertFalse(hasattr(bus, "_native_lock"))

    def test_tx_runs_while_rx_dll_read_is_waiting(self) -> None:
        read_started = threading.Event()
        release_read = threading.Event()
        send_called = threading.Event()

        def frame_read(_handle, _frame, timeout_ms):
            if timeout_ms == 10:
                read_started.set()
                release_read.wait(timeout=0.5)
            return False

        def frame_send(_handle, _frame):
            send_called.set()
            return True

        bus = AgxCandoBus.__new__(AgxCandoBus)
        bus._api = SimpleNamespace(dll=SimpleNamespace(
            cando_frame_read=frame_read,
            cando_frame_send=frame_send,
        ))
        bus._dev_handle = _VOID_P(1)
        bus._shutdown_flag = threading.Event()
        bus._queue_cond = threading.Condition()
        bus._queue = __import__("collections").deque()
        bus._receive_own_messages = False
        bus._loopback = False

        reader = threading.Thread(target=bus._recv_loop, daemon=True)
        reader.start()
        self.assertTrue(read_started.wait(timeout=0.2))
        bus.send(Message(arbitration_id=0x187, is_extended_id=False, data=b"\x01"))
        self.assertTrue(send_called.wait(timeout=0.1))
        release_read.set()
        bus._shutdown_flag.set()
        reader.join(timeout=0.2)


if __name__ == "__main__":
    unittest.main()
