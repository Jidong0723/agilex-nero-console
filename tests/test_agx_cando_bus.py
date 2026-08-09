from __future__ import annotations

import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from can import BusABC, CanInitializationError
from agx_cando.bus import AgxCandoBus, _VOID_P


class AgxCandoHandleLifecycleTests(unittest.TestCase):
    def test_failed_open_frees_fresh_instances_without_close(self) -> None:
        dll = mock.MagicMock()
        dll.cando_list_malloc.side_effect = lambda pointer: setattr(pointer._obj, "value", 10) or True
        dll.cando_list_scan.return_value = True
        dll.cando_list_num.side_effect = lambda _handle, pointer: setattr(pointer._obj, "value", 1) or True
        handles = iter((101, 102, 103))
        dll.cando_malloc.side_effect = lambda _list, _index, pointer: setattr(pointer._obj, "value", next(handles)) or True
        dll.cando_open.return_value = False
        dll.cando_free.return_value = True
        dll.cando_list_free.return_value = True

        bus = AgxCandoBus.__new__(AgxCandoBus)
        bus._api = SimpleNamespace(dll=dll)
        bus._channel_index = 0
        bus._bitrate = 1_000_000
        bus._loopback = False
        bus._receive_own_messages = False
        bus._list_handle = _VOID_P()
        bus._dev_handle = _VOID_P()
        bus._native_lock = threading.RLock()
        bus._device_opened = False
        bus._device_started = False

        with self.assertRaisesRegex(CanInitializationError, "cando_open failed"):
            bus._open_device()

        self.assertEqual(dll.cando_malloc.call_count, 3)
        self.assertEqual(dll.cando_free.call_count, 3)
        dll.cando_stop.assert_not_called()
        dll.cando_close.assert_not_called()

    def test_shutdown_is_idempotent_for_native_handles(self) -> None:
        dll = mock.MagicMock()
        bus = AgxCandoBus.__new__(AgxCandoBus)
        bus._api = SimpleNamespace(dll=dll)
        bus._native_lock = threading.RLock()
        bus._shutdown_lock = threading.Lock()
        bus._shutdown_flag = threading.Event()
        bus._queue_cond = threading.Condition()
        bus._rx_thread = None
        bus._closed = False
        bus._device_opened = True
        bus._device_started = True
        bus._dev_handle = _VOID_P(123)
        bus._list_handle = _VOID_P(456)

        with mock.patch.object(BusABC, "shutdown") as base_shutdown:
            bus.shutdown()
            bus.shutdown()

        dll.cando_stop.assert_called_once()
        dll.cando_close.assert_called_once()
        dll.cando_free.assert_called_once()
        dll.cando_list_free.assert_called_once()
        base_shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
