#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import errno
import can
import threading
import time
from collections import deque
from can.message import Message
from platform import system

from .core.can_comm_base import CanCommBase
from .can_sys_utils import CanSystemInfoBase, LinuxSocketCanSystemInfo

_SUPPORTED_PLATFORMS = {"Linux", "Windows", "Darwin"}


def create_can_comm_config(
    *,
    channel: str = "can0",
    interface: str = "socketcan",
    bitrate: int = 1_000_000,
    enable_check_can: bool = True,
    auto_connect: bool = True,
    timeout: float = 1.0,
    receive_own_messages: bool = False,
    local_loopback: bool = False,
):
    return {
        "channel": channel,
        "interface": interface,
        "bitrate": bitrate,
        "enable_check_can": enable_check_can,
        "auto_connect": auto_connect,
        "timeout": timeout,
        "receive_own_messages": receive_own_messages,
        "local_loopback": local_loopback,
    }


class CanComm:
    """
    Platform selector for python-can based communication.
    """

    def __new__(cls, config: dict, comm_type: str = "can"):
        platform_system = system()
        if platform_system not in _SUPPORTED_PLATFORMS:
            supported_text = ", ".join(sorted(_SUPPORTED_PLATFORMS))
            raise RuntimeError(
                "Unsupported platform: %s. " % platform_system +
                "Supported platforms: %s." % supported_text
            )
        return CanCommImpl(config, comm_type)


class CanCommImpl(CanCommBase):
    def __init__(self, config: dict, comm_type: str = "can") -> None:
        super().__init__()
        self.recv_bus = None
        self.send_bus = None
        self.sysinfo: CanSystemInfoBase = None
        self.last_error = None
        self._config = config.copy()
        self._type = comm_type
        self._channel = self._config["channel"]
        self._interface = (
            self._config["interface"]
            if "interface" in self._config
            else self._config.get("bustype", "socketcan")
        )
        self._bitrate = self._config.get("bitrate", 1000000)
        self._enable_check_can = self._config.get("enable_check_can", False)
        self._auto_connect = self._config.get("auto_connect", False)
        self._timeout = self._config.get("timeout", 1.0)
        self._receive_own_messages = self._config.get("receive_own_messages", False)
        self._local_loopback = self._config.get("local_loopback", False)
        # Keep transport diagnostics bounded and silent during normal 50 Hz
        # operation.  The control service consumes these only when a CPV
        # send is slow or times out.
        self._slow_send_threshold_s = float(self._config.get("slow_send_threshold_s", 0.05))
        self._send_diagnostics_lock = threading.Lock()
        self._active_send_diagnostic = None
        self._slow_send_events = deque(maxlen=16)
        self._tx_owner_thread_id: int | None = None
        self._is_connected = False
        self._is_stopped = False
        if self._interface == "socketcan":
            self.sysinfo = LinuxSocketCanSystemInfo
        if self._enable_check_can and self.sysinfo is not None:
            self.check_can()
        if self._auto_connect:
            self.connect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _classify_can_error(self, exc: Exception):
        """
        Classify recoverable CAN exceptions only.

        Returns
        -------
        - errno.ENETDOWN: interface is down (recoverable warning path).
        - errno.ENOBUFS: tx queue/buffer full (recoverable warning path).
        - None: unknown/hard-disconnect/unclassified error (caller should raise).

        Notes
        -----
        - Classification priority:
          1) CanOperationError.error_code
          2) OSError.errno in exception chain
          3) conservative text fallback for known socketcan messages
        - Only known recoverable categories are classified here.
          Hard disconnects (e.g. ENODEV/no such device) are intentionally not
          downgraded and will be raised by caller.
        """
        def _map_known_errno(eno):
            if eno in (errno.ENETDOWN, errno.ENOBUFS):
                return eno
            return None

        candidates = [exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)]

        # 1) Prefer structured python-can error code.
        for err in candidates:
            if err is None:
                continue
            if isinstance(err, can.CanOperationError):
                mapped = _map_known_errno(getattr(err, "error_code", None))
                if mapped is not None:
                    return mapped

        # 2) Fall back to OS errno from exception chain.
        for err in candidates:
            if err is None:
                continue
            if isinstance(err, OSError):
                mapped = _map_known_errno(getattr(err, "errno", None))
                if mapped is not None:
                    return mapped

        # 3) Conservative text fallback.
        for err in candidates:
            if err is None:
                continue
            err_text = str(err).lower()
            if (
                "no buffer space available" in err_text or
                "transmit buffer full" in err_text
            ):
                return errno.ENOBUFS
            if "network is down" in err_text:
                return errno.ENETDOWN
        return None

    def connect(self, **kwargs):
        if self.recv_bus is not None and self.send_bus is not None:
            return

        common_kwargs = dict(
            channel=self._channel,
            interface=self._interface,
            bitrate=self._bitrate,
            receive_own_messages=self._receive_own_messages,
            local_loopback=self._local_loopback,
        )

        try:
            self.recv_bus = can.interface.Bus(**common_kwargs)
            self.send_bus = self.recv_bus
            self._is_connected = True
            self._is_stopped = False
            self.last_error = None
        except Exception as exc:
            self.last_error = exc
            self.close()
            raise can.CanInitializationError(
                "Failed to open CAN bus "
                "(interface='%s', channel='%s', bitrate=%s)."
                % (self._interface, self._channel, self._bitrate)
            )

    def close(self):
        # ``send_bus`` and ``recv_bus`` intentionally share one official
        # CANDO bus.  Upstream shutdown is not specified as idempotent, so do
        # not close that same native handle twice.
        buses = []
        for bus in (self.recv_bus, self.send_bus):
            if bus is not None and all(bus is not existing for existing in buses):
                buses.append(bus)
        for bus in buses:
            try:
                bus.shutdown()
            except Exception:
                pass

        self.recv_bus = None
        self.send_bus = None
        self._is_connected = False
        self._is_stopped = True

    def bind_tx_owner_thread(self, thread_id: int) -> None:
        """Permit CAN sends only from the HardwareTxOwner worker thread."""
        self._tx_owner_thread_id = int(thread_id)

    @staticmethod
    def _message_diagnostic(msg: Message) -> dict:
        return {
            "arbitration_id": int(getattr(msg, "arbitration_id", 0)),
            "is_extended_id": bool(getattr(msg, "is_extended_id", False)),
            "is_remote_frame": bool(getattr(msg, "is_remote_frame", False)),
            "dlc": int(getattr(msg, "dlc", len(getattr(msg, "data", b"")))),
            "data_hex": bytes(getattr(msg, "data", b"")).hex(" "),
        }

    def send_diagnostics(self, *, consume_slow: bool = False) -> dict:
        """Return bounded state for diagnosing a stalled CAN transmission."""
        with self._send_diagnostics_lock:
            active = dict(self._active_send_diagnostic) if self._active_send_diagnostic else None
            slow = [dict(item) for item in self._slow_send_events]
            if consume_slow:
                self._slow_send_events.clear()
        bus = self.send_bus
        return {
            "channel": self._channel,
            "interface": self._interface,
            "connected": bool(self._is_connected and bus is not None),
            "stopped": bool(self._is_stopped),
            "last_error": None if self.last_error is None else f"{type(self.last_error).__name__}: {self.last_error}",
            "bus_state": None if bus is None else str(getattr(bus, "state", None)),
            "active_send": active,
            "slow_send_events": slow,
            "slow_send_threshold_ms": self._slow_send_threshold_s * 1000.0,
        }

    def send(self, msg: Message, timeout=None):
        if self.send_bus is None:
            self.close()
            raise RuntimeError("CAN bus is not connected.")
        expected_thread = self._tx_owner_thread_id
        actual_thread = threading.get_ident()
        if expected_thread is not None and actual_thread != expected_thread:
            raise RuntimeError(
                "CAN TX rejected outside HardwareTxOwner "
                f"(expected_thread={expected_thread}, actual_thread={actual_thread})"
            )

        started_ns = time.monotonic_ns()
        diagnostic = {
            **self._message_diagnostic(msg),
            "started_monotonic_ns": started_ns,
            "timeout": timeout,
        }
        with self._send_diagnostics_lock:
            self._active_send_diagnostic = diagnostic
        try:
            self.send_bus.send(msg, timeout)
            self.last_error = None
        except Exception as exc:
            self.last_error = exc
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["force_record"] = True
            err_kind = self._classify_can_error(exc)
            if err_kind in (errno.ENOBUFS, errno.ENETDOWN):
                diagnostic["classified_error"] = err_kind
                return
            self.close()
            raise self.last_error
        finally:
            finished_ns = time.monotonic_ns()
            diagnostic.setdefault("finished_monotonic_ns", finished_ns)
            diagnostic.setdefault("duration_ms", (finished_ns - started_ns) / 1e6)
            with self._send_diagnostics_lock:
                self._active_send_diagnostic = None
                if diagnostic.get("force_record") or diagnostic["duration_ms"] >= self._slow_send_threshold_s * 1000.0:
                    self._slow_send_events.append(dict(diagnostic))

    def recv(self):
        if self.recv_bus is None:
            self.close()
            raise RuntimeError("CAN bus is not connected.")

        try:
            msg = self.recv_bus.recv(self._timeout)
            if msg is not None:
                if not msg.is_error_frame:
                    self.last_error = None
                    self._trigger_callback(msg)
                    return msg
        except Exception as exc:
            self.last_error = exc
            err_kind = self._classify_can_error(exc)
            if err_kind in (errno.ENOBUFS, errno.ENETDOWN):
                return
            self.close()
            raise self.last_error

    def check_can(self):
        self.last_error = None
        if not self.sysinfo.is_exists(self._channel):
            self.last_error = ValueError("Device '%s' does not exist." % self._channel)
            raise self.last_error
        if not self.sysinfo.is_up(self._channel):
            print("[WARN] Device is DOWN.")
        actual_bitrate = self.sysinfo.get_bitrate(self._channel)
        if (
            self._bitrate is not None
            and actual_bitrate is not None
            and actual_bitrate != self._bitrate
        ):
            print(
                "[WARN] CAN port %s bitrate is %s bps, expected %s bps."
                % (self._channel, actual_bitrate, self._bitrate)
            )
