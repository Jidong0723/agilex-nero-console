import ctypes
import collections
import threading
import time
from typing import Optional

from can import BusABC, CanInitializationError, CanOperationError, Message

from .dll import load_cando_dll

CANDO_MODE_NORMAL = 0
CANDO_MODE_LISTEN_ONLY = 1 << 0
CANDO_MODE_LOOP_BACK = 1 << 1
CANDO_MODE_ONE_SHOT = 1 << 3
CANDO_MODE_NO_ECHO_BACK = 1 << 8

CANDO_ID_EXTENDED = 0x80000000
CANDO_ID_RTR = 0x40000000
CANDO_ID_ERR = 0x20000000
CANDO_ID_MASK = 0x1FFFFFFF

_BOOL = ctypes.c_bool
_U8 = ctypes.c_uint8
_U32 = ctypes.c_uint32
_VOID_P = ctypes.c_void_p


class CandoFrame(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("echo_id", _U32),
        ("can_id", _U32),
        ("can_dlc", _U8),
        ("channel", _U8),
        ("flags", _U8),
        ("reserved", _U8),
        ("data", _U8 * 8),
        ("timestamp_us", _U32),
    ]


class CandoBitTiming(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("prop_seg", _U32),
        ("phase_seg1", _U32),
        ("phase_seg2", _U32),
        ("sjw", _U32),
        ("brp", _U32),
    ]


BITRATE_TABLE = {
    10000: (1, 12, 2, 1, 300),
    20000: (1, 12, 2, 1, 150),
    50000: (1, 12, 2, 1, 60),
    83333: (1, 12, 2, 1, 36),
    100000: (1, 12, 2, 1, 30),
    125000: (1, 12, 2, 1, 24),
    250000: (1, 12, 2, 1, 12),
    500000: (1, 12, 2, 1, 6),
    800000: (1, 11, 2, 1, 4),
    1000000: (1, 12, 2, 1, 3),
}


class _DeviceInfo(object):
    def __init__(self, serial, fw_version, hw_version):
        self.serial = serial
        self.fw_version = fw_version
        self.hw_version = hw_version


class _CandoApi(object):
    def __init__(self):
        self.dll = load_cando_dll()
        self._bind_functions()

    def _bind_functions(self):
        dll = self.dll
        dll.cando_list_malloc.argtypes = [ctypes.POINTER(_VOID_P)]
        dll.cando_list_malloc.restype = _BOOL
        dll.cando_list_free.argtypes = [_VOID_P]
        dll.cando_list_free.restype = _BOOL
        dll.cando_list_scan.argtypes = [_VOID_P]
        dll.cando_list_scan.restype = _BOOL
        dll.cando_list_num.argtypes = [_VOID_P, ctypes.POINTER(_U8)]
        dll.cando_list_num.restype = _BOOL

        dll.cando_malloc.argtypes = [_VOID_P, _U8, ctypes.POINTER(_VOID_P)]
        dll.cando_malloc.restype = _BOOL
        dll.cando_free.argtypes = [_VOID_P]
        dll.cando_free.restype = _BOOL
        dll.cando_open.argtypes = [_VOID_P]
        dll.cando_open.restype = _BOOL
        dll.cando_close.argtypes = [_VOID_P]
        dll.cando_close.restype = _BOOL
        dll.cando_get_serial_number_str.argtypes = [_VOID_P]
        dll.cando_get_serial_number_str.restype = ctypes.c_wchar_p
        dll.cando_get_dev_info.argtypes = [_VOID_P, ctypes.POINTER(_U32), ctypes.POINTER(_U32)]
        dll.cando_get_dev_info.restype = _BOOL
        dll.cando_set_timing.argtypes = [_VOID_P, ctypes.POINTER(CandoBitTiming)]
        dll.cando_set_timing.restype = _BOOL
        dll.cando_start.argtypes = [_VOID_P, _U32]
        dll.cando_start.restype = _BOOL
        dll.cando_stop.argtypes = [_VOID_P]
        dll.cando_stop.restype = _BOOL
        dll.cando_frame_send.argtypes = [_VOID_P, ctypes.POINTER(CandoFrame)]
        dll.cando_frame_send.restype = _BOOL
        dll.cando_frame_read.argtypes = [_VOID_P, ctypes.POINTER(CandoFrame), _U32]
        dll.cando_frame_read.restype = _BOOL

    def scan(self):
        list_handle = _VOID_P()
        if not self.dll.cando_list_malloc(ctypes.byref(list_handle)):
            raise CanInitializationError("cando_list_malloc failed")
        try:
            if not self.dll.cando_list_scan(list_handle):
                raise CanInitializationError("cando_list_scan failed")
            count = _U8()
            if not self.dll.cando_list_num(list_handle, ctypes.byref(count)):
                raise CanInitializationError("cando_list_num failed")
            return int(count.value)
        finally:
            self.dll.cando_list_free(list_handle)

    def describe(self, index):
        list_handle = _VOID_P()
        dev_handle = _VOID_P()
        opened = False
        if not self.dll.cando_list_malloc(ctypes.byref(list_handle)):
            raise CanInitializationError("cando_list_malloc failed")
        try:
            if not self.dll.cando_list_scan(list_handle):
                raise CanInitializationError("cando_list_scan failed")
            if not self.dll.cando_malloc(list_handle, index, ctypes.byref(dev_handle)):
                raise CanInitializationError("cando_malloc failed for channel %s" % index)
            try:
                if not self.dll.cando_open(dev_handle):
                    raise CanInitializationError("cando_open failed for channel %s" % index)
                opened = True
                fw = _U32()
                hw = _U32()
                serial = self.dll.cando_get_serial_number_str(dev_handle) or ""
                if not self.dll.cando_get_dev_info(dev_handle, ctypes.byref(fw), ctypes.byref(hw)):
                    raise CanInitializationError("cando_get_dev_info failed")
                return _DeviceInfo(serial=serial, fw_version=int(fw.value), hw_version=int(hw.value))
            finally:
                if dev_handle:
                    if opened:
                        self.dll.cando_close(dev_handle)
                    self.dll.cando_free(dev_handle)
        finally:
            self.dll.cando_list_free(list_handle)


class AgxCandoBus(BusABC):
    def __init__(
        self,
        channel=0,
        can_filters=None,
        bitrate=1000000,
        local_loopback=True,
        receive_own_messages=True,
        **kwargs
    ):
        super(AgxCandoBus, self).__init__(channel=channel, can_filters=can_filters, **kwargs)
        self._api = _CandoApi()
        self._channel_index = int(channel or 0)
        self._bitrate = bitrate
        self._receive_own_messages = receive_own_messages
        self._loopback = local_loopback
        self._queue = collections.deque()
        self._queue_cond = threading.Condition()
        self._rx_thread = None  # type: Optional[threading.Thread]
        self._shutdown_flag = threading.Event()
        self._list_handle = _VOID_P()
        self._dev_handle = _VOID_P()
        self._filters = can_filters

        self._open_device()
        self.channel_info = "agx_cando channel=%s" % self._channel_index

    def _open_device(self):
        dll = self._api.dll
        if self._bitrate not in BITRATE_TABLE:
            raise CanInitializationError("Unsupported bitrate: %s" % self._bitrate)

        if not dll.cando_list_malloc(ctypes.byref(self._list_handle)):
            raise CanInitializationError("cando_list_malloc failed")
        try:
            if not dll.cando_list_scan(self._list_handle):
                raise CanInitializationError("cando_list_scan failed")

            count = _U8()
            if not dll.cando_list_num(self._list_handle, ctypes.byref(count)):
                raise CanInitializationError("cando_list_num failed")
            if int(count.value) <= self._channel_index:
                raise CanInitializationError(
                    "Channel %s out of range, detected %s device(s)"
                    % (self._channel_index, int(count.value))
                )

            if not dll.cando_malloc(self._list_handle, self._channel_index, ctypes.byref(self._dev_handle)):
                raise CanInitializationError("cando_malloc failed for channel %s" % self._channel_index)
            opened = False
            for _ in range(20):
                if dll.cando_open(self._dev_handle):
                    opened = True
                    break
                time.sleep(0.1)
            if not opened:
                raise CanInitializationError("cando_open failed for channel %s" % self._channel_index)

            timing = CandoBitTiming(*BITRATE_TABLE[self._bitrate])
            if not dll.cando_set_timing(self._dev_handle, ctypes.byref(timing)):
                raise CanInitializationError("cando_set_timing failed")

            mode = CANDO_MODE_LOOP_BACK if self._loopback else CANDO_MODE_NORMAL
            if not self._receive_own_messages and not self._loopback:
                mode |= CANDO_MODE_NO_ECHO_BACK
            if not dll.cando_start(self._dev_handle, mode):
                raise CanInitializationError("cando_start failed")
        except Exception:
            self._close_handles()
            raise

        self._rx_thread = threading.Thread(target=self._recv_loop)
        self._rx_thread.daemon = True
        self._rx_thread.start()

    def _close_handles(self):
        dll = self._api.dll
        if self._dev_handle:
            try:
                dll.cando_stop(self._dev_handle)
            except Exception:
                pass
            try:
                dll.cando_close(self._dev_handle)
            except Exception:
                pass
            try:
                dll.cando_free(self._dev_handle)
            except Exception:
                pass
        if self._list_handle:
            try:
                dll.cando_list_free(self._list_handle)
            except Exception:
                pass
        self._dev_handle = _VOID_P()
        self._list_handle = _VOID_P()

    def _recv_loop(self):
        dll = self._api.dll
        while not self._shutdown_flag.is_set():
            first = CandoFrame()
            if not dll.cando_frame_read(self._dev_handle, ctypes.byref(first), 10):
                continue
            self._enqueue_frame(first)

            # Drain any frames already queued in the device without another blocking wait.
            while not self._shutdown_flag.is_set():
                frame = CandoFrame()
                received = dll.cando_frame_read(self._dev_handle, ctypes.byref(frame), 0)
                if not received:
                    break
                self._enqueue_frame(frame)

    def _enqueue_frame(self, frame):
        record = (
            int(frame.can_id),
            int(frame.can_dlc),
            bytes(frame.data[: frame.can_dlc]),
            int(frame.timestamp_us),
        )
        with self._queue_cond:
            self._queue.append(record)
            self._queue_cond.notify()

    def send(self, msg, timeout=None):
        del timeout
        if msg.dlc > 8:
            raise CanOperationError("Classic CAN payload length must be <= 8")
        frame = CandoFrame()
        frame.echo_id = 0 if (self._receive_own_messages or self._loopback) else 0xFFFFFFFF
        frame.can_id = int(msg.arbitration_id) & CANDO_ID_MASK
        if msg.is_extended_id:
            frame.can_id |= CANDO_ID_EXTENDED
        if msg.is_remote_frame:
            frame.can_id |= CANDO_ID_RTR
        if msg.is_error_frame:
            frame.can_id |= CANDO_ID_ERR
        frame.can_dlc = int(msg.dlc)
        frame.channel = 0
        for idx, value in enumerate(bytes(msg.data)):
            frame.data[idx] = value
        if not self._api.dll.cando_frame_send(self._dev_handle, ctypes.byref(frame)):
            raise CanOperationError("cando_frame_send failed")

    def _recv_internal(self, timeout):
        deadline = None
        if timeout is not None:
            deadline = time.time() + timeout

        with self._queue_cond:
            while not self._queue:
                if timeout == 0:
                    return None, False
                if deadline is None:
                    self._queue_cond.wait()
                else:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None, False
                    self._queue_cond.wait(remaining)
                if self._shutdown_flag.is_set() and not self._queue:
                    return None, False
            can_id, dlc, data, timestamp_us = self._queue.popleft()

        msg = Message(
            arbitration_id=int(can_id & CANDO_ID_MASK),
            is_extended_id=bool(can_id & CANDO_ID_EXTENDED),
            is_remote_frame=bool(can_id & CANDO_ID_RTR),
            is_error_frame=bool(can_id & CANDO_ID_ERR),
            dlc=dlc,
            data=data,
            channel=self._channel_index,
            timestamp=time.time(),
        )
        return msg, False

    def set_filters(self, filters=None):
        self._filters = filters

    def shutdown(self):
        if getattr(self, "_shutdown_flag", None) is None:
            return
        self._shutdown_flag.set()
        with self._queue_cond:
            self._queue_cond.notify_all()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=0.2)
        self._close_handles()
        super(AgxCandoBus, self).shutdown()

    @staticmethod
    def _detect_available_configs():
        api = _CandoApi()
        count = api.scan()
        return [{"interface": "agx_cando", "channel": i} for i in range(count)]
