"""MessagePack/OpenPI WebSocket transport shared by all inference presets."""
from __future__ import annotations

import threading
import socket
import time
from typing import Any
from .inference_config import endpoint_uri

def _pack_array(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _unpack_array(value: dict[bytes, Any]) -> Any:
    import numpy as np
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class OpenPIClient:
    """Small copy of OpenPI's MessagePack/WebSocket protocol client."""
    def __init__(self, host: str, port: int, timeout_s: float, connection_event=None) -> None:
        import msgpack
        from websockets.sync.client import connect, ClientConnection
        from websockets.uri import parse_uri
        uri = endpoint_uri(host, port)
        parsed = parse_uri(uri)
        emit = connection_event or (lambda stage, detail=None: None)
        deadline = time.monotonic() + timeout_s
        raw = None
        tcp_ready = False
        try:
            # Respect websockets' environment proxy selection when supported.
            try:
                from websockets.uri import get_proxy
                proxy = get_proxy(parsed)
            except ImportError:
                proxy = None
            options = {}
            if proxy is None:
                raw = socket.create_connection((parsed.host, parsed.port), timeout=timeout_s)
                tcp_ready = True
                emit("tcp_connected")
                raw.settimeout(None)
                options["sock"] = raw
            else:
                # The library owns proxy negotiation and TLS. Its connection
                # factory is reached only after the transport is established.
                class ObservedConnection(ClientConnection):
                    def __init__(inner, *args, **kwargs):
                        nonlocal tcp_ready
                        tcp_ready = True
                        emit("tcp_connected", "通过系统代理建立传输通道")
                        super().__init__(*args, **kwargs)
                options["create_connection"] = ObservedConnection
            self.socket = connect(uri, compression=None, max_size=None,
                                  open_timeout=max(.001, deadline-time.monotonic()), close_timeout=2, **options)
            emit("websocket_connected")
            handshake = self.socket.recv(timeout=timeout_s)
            if not isinstance(handshake, bytes):
                raise RuntimeError(f"OpenPI handshake error: {handshake}")
            self.metadata = msgpack.unpackb(handshake, object_hook=_unpack_array)
            if not isinstance(self.metadata, dict):
                raise RuntimeError("OpenPI metadata must be an object")
            emit("metadata_received")
        except Exception as exc:
            emit("protocol_error" if tcp_ready else "tcp_error", f"{type(exc).__name__}: {exc}")
            if hasattr(self, "socket"):
                self.socket.close()
            elif raw is not None:
                raw.close()
            raise
        # Match OpenPI's ``msgpack_numpy.Packer`` rather than ``packb``.  The
        # streaming packer preserves the binary ndarray marker keys required
        # by the server's object hook; packb makes valid RGB arrays arrive as
        # zero-dimensional values with this OpenPI runtime.
        self._packer = msgpack.Packer(default=_pack_array)
        self._io_lock = threading.Lock()
        self.last_health_error: str | None = None
        self.timeout_s = timeout_s

    def infer(self, observation: dict[str, Any]) -> Any:
        import msgpack
        # ``websockets.sync`` permits only one receive operation per socket.
        # Keep inference and the idle health probe on the same serialized path.
        with self._io_lock:
            self.socket.send(self._packer.pack(observation))
            reply = self.socket.recv(timeout=self.timeout_s)
        if isinstance(reply, str):
            raise RuntimeError(f"OpenPI server error: {reply}")
        return msgpack.unpackb(reply, object_hook=_unpack_array)

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception:
            pass

    def is_alive(self) -> bool:
        """Return whether the WebSocket is still open, with a real ping."""
        state = str(getattr(self.socket, "state", "")).upper()
        if state and state not in {"OPEN", "1"}:
            self.last_health_error = f"socket state {state}"
            return False
        # An inference is already a stronger liveness check than ping.  Do not
        # race it just because the status refresh happened at the same moment.
        if not self._io_lock.acquire(blocking=False):
            return True
        try:
            pong = self.socket.ping()
            alive = bool(pong.wait(timeout=1.0))
            self.last_health_error = None if alive else "ping timed out"
            return alive
        except Exception as exc:
            self.last_health_error = f"{type(exc).__name__}: {exc}"
            return False
        finally:
            self._io_lock.release()
