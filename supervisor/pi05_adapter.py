"""AutoDL OpenPI inference adapter: Nero observations to OSC TCP commands.

This module deliberately owns no robot object, transport, or servo loop.  It
is an input adapter running beside the HTTP service; its only output path is
``OperationalSpaceController.osc_command``.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any
from .camera_resource import SharedCameraResource
from .inference_extensions import InferenceJournal, connection_config, wait_for_arrival, wait_deadline


LOGGER = logging.getLogger(__name__)


def _finite(values: Any, length: int, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result


def _quat_product(a: list[float], b: list[float]) -> list[float]:
    """Hamilton product for xyzw quaternions (kept separate for clarity)."""
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return [w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2,
            w1*w2-x1*x2-y1*y2-z1*z2]


def _rotvec_quaternion(vector: list[float]) -> list[float]:
    angle = math.sqrt(sum(value * value for value in vector))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    scale = math.sin(angle / 2.0) / angle
    return [vector[0] * scale, vector[1] * scale, vector[2] * scale, math.cos(angle / 2.0)]


def _quaternion_rotvec(quaternion: list[float]) -> list[float]:
    x, y, z, w = _finite(quaternion, 4, "TCP orientation")
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError("TCP orientation cannot be zero")
    x, y, z, w = x / norm, y / norm, z / norm, max(-1.0, min(1.0, w / norm))
    angle = 2.0 * math.acos(w)
    sine = math.sqrt(max(0.0, 1.0 - w*w))
    return [0.0, 0.0, 0.0] if sine < 1e-12 else [x / sine * angle, y / sine * angle, z / sine * angle]


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
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        import msgpack
        from websockets.sync.client import connect
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.socket = connect(uri, compression=None, max_size=None, open_timeout=timeout_s, close_timeout=2)
        self.socket.recv(timeout=timeout_s)  # policy metadata handshake
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


class _LegacyCameraPair:
    """Latest-frame dual camera reader with the AutoDL 224x224 RGB contract."""
    def __init__(self, config: dict[str, Any]) -> None:
        import cv2
        self.cv2 = cv2
        self.read_lock = threading.Lock()
        self.width, self.height = int(config["model_width"]), int(config["model_height"])
        self.captures = []
        for key in ("external", "wrist"):
            item = config[key]
            capture = cv2.VideoCapture(int(item["index"]), cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(item["width"]))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(item["height"]))
            if not capture.isOpened():
                self.close()
                raise RuntimeError(f"cannot open {key} camera index {item['index']}")
            self.captures.append(capture)

    def read(self) -> tuple[Any, Any]:
        frames = []
        with self.read_lock:
            for capture in self.captures:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("camera frame capture failed")
                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]; scale = min(self.width / w, self.height / h)
                resized = self.cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
                canvas = __import__("numpy").zeros((self.height, self.width, 3), dtype=__import__("numpy").uint8)
                y, x = (self.height - resized.shape[0]) // 2, (self.width - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                frames.append(canvas)
        return frames[0], frames[1]

    def close(self) -> None:
        for capture in getattr(self, "captures", []):
            capture.release()


class Pi05InputAdapter:
    def __init__(self, osc: Any, config_path: Path, camera_resource: SharedCameraResource | None = None) -> None:
        self.osc, self.config_path = osc, Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.lock = threading.RLock(); self.stop_event = threading.Event(); self.worker: threading.Thread | None = None
        self._connection_stop = threading.Event(); self._policy: OpenPIClient | None = None
        self._connection_wakeup = threading.Event()
        self._policy_create_lock = threading.Lock()
        self._policy_generation = 0
        self._policy_retry_at = 0.0
        self._connection_thread = threading.Thread(target=self._connection_loop, name="nero-pi05-websocket", daemon=True)
        self._osc_state_stop = threading.Event()
        self._osc_state_thread = threading.Thread(target=self._osc_state_loop, name="nero-pi05-osc-state", daemon=True)
        self.camera_resource = camera_resource
        # Compatibility seam for direct adapter tests; production always uses
        # the HTTP-process shared camera resource.
        self.cameras: Any | None = None if camera_resource else None
        self._last_connection_probe = 0.0
        self._connections: dict[str, Any] = {}
        self._osc_snapshot: dict[str, Any] = {}
        self._last_gripper_target: str | None = None
        self.state: dict[str, Any] = self._new_state()
        self._journal = None
        self._completed_journal = None
        self._profile_lock = threading.Lock()
        self._profile_switching = False
        self._connection_thread.start()
        self._osc_state_thread.start()

    def _new_state(self) -> dict[str, Any]:
        return {"adapter": "pi05", "state": "IDLE", "camera_state": "IDLE", "model_state": "UNKNOWN",
                "session_id": None, "client_id": None, "execution_mode": None, "prompt": self.config["model"]["prompt"],
                "last_error": None, "websocket_error": None, "inference_ms": None, "action_chunk": None, "absolute_tcp_chunk": None,
                "inference_base_tcp": None, "action_chunk_length": 0, "chunk_sequence": 0,
                "executed_steps": 0, "sequence": 0, "execution_enabled": False,
                "osc_error": None,
                "camera_wait_reason": None,
                "priming": False, "last_rejection": None,
                # The policy was trained to emit a closed [0, 1] gripper
                # ratio.  Keep the last boundary decision visible so an
                # operator can distinguish floating-point round-off from a
                # genuinely invalid policy output.
                "gripper_ratio_diagnostic": None,
                "decoded_first_target": None,
                "websocket_diagnostics": {"connect_attempts": 0, "connect_successes": 0,
                                          "disconnects": 0, "ping_failures": 0,
                                          "inference_failures": 0, "last_event": "等待 AutoDL Policy WebSocket"},
                "updated_at": time.time()}

    def _record_websocket_event(self, event: str, detail: str | None = None, *, log: bool = True) -> None:
        """Retain compact, operator-visible evidence for connection flapping."""
        counters = {"connect_attempt": "connect_attempts", "connect_success": "connect_successes",
                    "disconnect": "disconnects", "ping_failure": "ping_failures",
                    "inference_failure": "inference_failures"}
        text = event if not detail else f"{event}: {detail}"
        with self.lock:
            diagnostics = self.state.setdefault("websocket_diagnostics", {})
            counter = counters.get(event)
            if counter:
                diagnostics[counter] = int(diagnostics.get(counter, 0)) + 1
            diagnostics["last_event"] = text
            diagnostics["updated_at"] = time.time()
        if log:
            LOGGER.info("AutoDL Policy WebSocket %s", text)

    def _record_inference_input(self, external: Any, wrist: Any, observation: dict[str, Any]) -> None:
        def describe(value: Any) -> str:
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            return f"shape={tuple(shape) if shape is not None else None}, dtype={dtype}"
        with self.lock:
            diagnostics = self.state.setdefault("websocket_diagnostics", {})
            diagnostics["last_input"] = {"external_rgb": describe(external), "wrist_rgb": describe(wrist),
                                         "state": describe(observation.get("observation/state"))}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_connections_locked()
            result = copy.deepcopy(self.state); result["config"] = copy.deepcopy(self.config)
            shared = self.camera_resource.snapshot() if self.camera_resource else {"ready": self.cameras is not None, "frame_version": 0}
            result["camera_ready"] = shared["ready"]; result["frame_version"] = shared["frame_version"]
            result["camera"] = shared
            result["run_log"] = (self._journal or self._completed_journal).summary() if (self._journal or self._completed_journal) else None
            result["log_export_ready"] = bool(self._completed_journal and self._completed_journal.closed and not self._completed_journal.error)
            result["connections"] = copy.deepcopy(self._connections)
            return result

    def _next_sequence(self) -> int:
        """Allocate one strictly increasing sequence across TCP, gripper and HOLD.

        Two commands in one Action Chunk can be issued in the same millisecond.
        OSC rejects duplicate sequences, so retain the last emitted value rather
        than recalculating an independent timestamp at every call site.
        """
        with self.lock:
            previous = int(self.state.get("sequence", 0))
            sequence = max(previous + 1, int(time.time() * 1000))
            self.state["sequence"] = sequence
            return sequence

    def frame_jpeg(self, source: str) -> bytes | None:
        return self.camera_resource.frame_jpeg(source) if self.camera_resource else None

    def _preview_loop(self) -> None:
        while False:
            try:
                if self.cameras is None: return
                external, wrist = self.cameras.read()
                with self.lock: self.frames = {"external": external, "wrist": wrist}; self.frame_version += 1
                self.preview_stop.wait(.1)
            except Exception as exc:
                with self.lock: self.state.update({"camera_state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                return

    @staticmethod
    def _local_forward_listening(port: int) -> bool:
        """Check only for a local SSH listener, without touching the tunnel.

        A TCP connect probe would reach OpenPI through the tunnel and close
        before sending a WebSocket Upgrade request, which the server reports
        as an invalid handshake.  Process-local listener inspection keeps the
        SSH card independent from WebSocket state and is side-effect free.
        """
        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["netstat.exe", "-ano", "-p", "tcp"],
                    capture_output=True, text=True, timeout=.75,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for line in result.stdout.splitlines():
                    fields = line.split()
                    if len(fields) >= 4 and fields[0].upper() == "TCP" and fields[3].upper() == "LISTENING":
                        if fields[1].rsplit(":", 1)[-1] == str(port):
                            return True
                return False
            result = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=.75)
            return any(line.rsplit(":", 1)[-1].split()[0] == str(port) for line in result.stdout.splitlines()[1:] if ":" in line)
        except (OSError, subprocess.SubprocessError):
            return False

    def _connection_loop(self) -> None:
        """Keep a lightweight policy WebSocket handshake independent of inference."""
        while not self._connection_stop.is_set():
            with self.lock:
                policy = self._policy
                model = dict(self.config["model"])
                worker_active = self.worker is not None and self.worker.is_alive()
                retry_at = self._policy_retry_at
            # The synchronous WebSocket client serializes send/recv/ping.
            # Never issue a health ping concurrently with inference on the
            # same socket; inference failures are the active-session liveness
            # signal.  Idle sessions may still use ping to publish status.
            connected = policy is not None and (worker_active or policy.is_alive())
            if policy is not None and not connected:
                self._record_websocket_event("ping_failure", getattr(policy, "last_health_error", None))
                self._invalidate_policy(policy, "OpenPI WebSocket disconnected")
            if not connected and time.monotonic() >= retry_at:
                with self._policy_create_lock:
                    with self.lock:
                        if self._policy is not None:
                            policy = self._policy
                            continue_connect = False
                        else:
                            continue_connect = True
                    if continue_connect:
                        try:
                            self._record_websocket_event("connect_attempt")
                            policy = OpenPIClient(str(model["host"]), int(model["port"]), float(model["request_timeout_s"]))
                            with self.lock:
                                if self._connection_stop.is_set() or (model["host"], model["port"]) != (self.config["model"]["host"], self.config["model"]["port"]):
                                    policy.close()
                                else:
                                    self._policy = policy
                                    self._policy_generation += 1
                                    self.state.update({"model_state": "CONNECTED", "websocket_error": None, "updated_at": time.time()})
                                    self._record_websocket_event("connect_success")
                                    if self.camera_resource and self.camera_resource.snapshot().get("ready"):
                                        self._ensure_worker()
                        except Exception as exc:
                            self._record_websocket_event("connect_failure", f"{type(exc).__name__}: {exc}")
                            with self.lock:
                                self.state.update({"model_state": "DISCONNECTED", "websocket_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
            self._connection_wakeup.wait(1.0)
            self._connection_wakeup.clear()

    def _invalidate_policy(self, policy: OpenPIClient | None, reason: str) -> None:
        with self.lock:
            if policy is not None and self._policy is not policy:
                return
            self._policy = None
            self._policy_retry_at = time.monotonic() + 2.0
            self.state.update({"model_state": "DISCONNECTED", "websocket_error": reason, "updated_at": time.time()})
            self._record_websocket_event("disconnect", reason)
        if policy is not None:
            policy.close()

    def _osc_state_loop(self) -> None:
        """Refresh OSC feedback independently from inference and UI state reads."""
        period_s = max(0.02, float((self.config.get("execution") or {}).get("osc_state_poll_s", 0.05)))
        while not self._osc_state_stop.is_set():
            try:
                snapshot = self.osc.state()
                with self.lock:
                    self._osc_snapshot = copy.deepcopy(snapshot)
                    self.state.update({"osc_error": None, "updated_at": time.time()})
            except Exception as exc:
                with self.lock:
                    self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
            self._osc_state_stop.wait(period_s)

    def _policy_client(self) -> OpenPIClient:
        """Wait for the single connection-owner thread to publish a client.

        Connection creation must stay in ``_connection_loop``.  Creating a
        fallback client here would race that loop and produce multiple OpenPI
        sockets, which appear on the server as clients that vanish during
        handshake.  It would also erase the observable disconnected interval
        after a failed inference.
        """
        deadline = time.monotonic() + float(self.config["model"]["request_timeout_s"])
        while time.monotonic() < deadline and not self.stop_event.is_set():
            with self.lock:
                policy = self._policy
            if policy is not None:
                return policy
            self._connection_wakeup.set()
            self.stop_event.wait(.05)
        raise RuntimeError("AutoDL Policy WebSocket is not connected")

    def _ensure_worker(self) -> None:
        with self.lock:
            if self._profile_switching:
                return
            if self.worker and self.worker.is_alive():
                return
            self.stop_event = threading.Event()
            self.state.update({"state": "RUNNING", "last_error": None, "updated_at": time.time()})
            self.worker = threading.Thread(target=self._run, name="nero-pi05-input-adapter", daemon=True)
            self.worker.start()

    def cameras_ready(self) -> None:
        """Start observation/inference once the shared camera resource is live."""
        self._ensure_worker()

    def _refresh_connections_locked(self) -> None:
        if time.monotonic() - self._last_connection_probe < 1.0: return
        self._last_connection_probe = time.monotonic()
        connection = self.config.get("connection") or {}
        host = str(connection.get("policy_host", self.config["model"]["host"]))
        port = int(connection.get("policy_port", self.config["model"]["port"]))
        # Do not probe a WebSocket endpoint with a raw TCP connect.  That
        # opens a socket and closes it without sending an HTTP Upgrade request,
        # which OpenPI correctly logs as an invalid handshake (EOF while
        # reading the request line).  The real WebSocket worker is the source
        # of truth for policy connectivity.
        is_local = (self.config.get("inference_profiles") or {}).get("active") == "local5090"
        local_forward_listening = False if is_local else self._local_forward_listening(port)
        policy_connected = self.state.get("model_state") == "CONNECTED"
        # Do not synchronously query OSC while serving pi05 state. A stalled
        # control backend must not block inference telemetry or Action Chunks.
        transport = (self._osc_snapshot.get("transport") or {}) if self._osc_snapshot else {}
        osc_error = self.state.get("osc_error")
        self._connections = {
            "control": {"state": "bad" if osc_error else ("ok" if transport.get("connected") else "warn"), "label": "NERO control service", "endpoint": "127.0.0.1:8765", "message": str(osc_error) if osc_error else "OSC control channel available"},
            "ssh_forward": {"state": "ok" if local_forward_listening else "bad", "label": "SSH 本地转发", "endpoint": f"127.0.0.1:{port}", "message": "本地转发端口已监听" if local_forward_listening else "本地转发端口未监听"},
            "policy": {"state": "ok" if policy_connected else "bad", "label": "AutoDL Policy WebSocket", "endpoint": "OpenPI policy server", "message": "AutoDL Policy WebSocket 已连接" if policy_connected else (str(self.state.get("websocket_error")) if self.state.get("websocket_error") else "AutoDL Policy WebSocket 未连接")},
        }
        if is_local:
            status = "ok" if policy_connected else "bad"
            endpoint = f"{host}:{port}"
            self._connections["ssh_forward"] = {"state":status,"label":"5090本地直连","endpoint":endpoint,"message":"无需SSH转发"}
            self._connections["policy"] = {"state":status,"label":"5090 Policy WebSocket","endpoint":endpoint,"message":"已连接" if policy_connected else "等待本地模型服务"}

    def camera_devices(self) -> list[dict[str, Any]]:
        if self.camera_resource: return self.camera_resource.devices()
        with self.lock:
            if self._camera_devices is not None: return copy.deepcopy(self._camera_devices)
        try:
            import cv2
            devices = []
            try:
                from cv2_enumerate_cameras import enumerate_cameras
                devices = [{"index": int(item.index), "name": str(item.name), "backend": int(item.backend)} for item in enumerate_cameras(cv2.CAP_DSHOW)]
            except Exception:
                for index in range(10):
                    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
                    opened = capture.isOpened(); capture.release()
                    if opened: devices.append({"index": index, "name": f"OpenCV camera {index}", "backend": int(getattr(cv2, "CAP_DSHOW", 0))})
        except Exception as exc:
            raise RuntimeError(f"camera enumeration failed: {exc}") from exc
        with self.lock: self._camera_devices = devices; return copy.deepcopy(devices)

    def update_config(self, value: dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict) and "inference_profile" in value:
            self._change_inference_profile(value["inference_profile"])
        with self.lock:
            model = value.get("model") if isinstance(value, dict) else None
            cameras = value.get("cameras") if isinstance(value, dict) else None
            execution = value.get("execution") if isinstance(value, dict) else None
            if isinstance(execution, dict) and "arrival_wait_s" in execution:
                raw = execution["arrival_wait_s"]
                wait = float(raw)
                if isinstance(raw, bool) or not math.isfinite(wait) or not 0 <= wait <= 5:
                    raise ValueError("Maximum arrival wait must be within 0–5 seconds")
                self.config["execution"]["arrival_wait_s"] = wait
            if isinstance(model, dict) and "prompt" in model:
                prompt = str(model["prompt"]).strip()
                if not 1 <= len(prompt) <= 500: raise ValueError("prompt must contain 1-500 characters")
                self.config["model"]["prompt"] = prompt; self.state["prompt"] = prompt
            if isinstance(execution, dict) and "action_delta_scale" in execution:
                scale = float(execution["action_delta_scale"])
                if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
                    raise ValueError("AutoDL action delta scale must be within [0, 1]")
                self.config.setdefault("execution", {})["action_delta_scale"] = scale
            if isinstance(cameras, dict):
                if self.state["state"] == "RUNNING": raise RuntimeError("stop AutoDL cloud inference before changing cameras")
                if self.camera_resource:
                    self.camera_resource.update_config(cameras)
                    self.config["cameras"] = copy.deepcopy(self.camera_resource.config)
                    self._persist_config_locked()
                    self.state["updated_at"] = time.time(); return self.snapshot()
                for key in ("external", "wrist"):
                    item = cameras.get(key)
                    if not isinstance(item, dict): continue
                    index = int(item.get("index", self.config["cameras"][key]["index"]))
                    if not 0 <= index <= 32: raise ValueError(f"{key} camera index must be 0-32")
                    self.config["cameras"][key]["index"] = index
                if self.config["cameras"]["external"]["index"] == self.config["cameras"]["wrist"]["index"]:
                    raise ValueError("external and wrist cameras must be different")
            self._persist_config_locked()
            self.state["updated_at"] = time.time()
            return self.snapshot()

    def _persist_config_locked(self) -> None:
        """Atomically retain operator-facing prompt and test settings across restart."""
        temporary = self.config_path.with_suffix(f"{self.config_path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(self.config, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, self.config_path)

    def _change_inference_profile(self, request) -> None:
        with self._profile_lock:
            with self.lock:
                if self.state.get("execution_enabled") or self.state.get("priming"):
                    raise RuntimeError("Stop inference output before switching model connections")
                previous = self.config
                updated = connection_config(previous, request)
                if updated == previous:
                    return
                self._profile_switching = True
                self.stop_event.set()
                worker, policy = self.worker, self._policy
                self._policy = None
                self.config = updated
                self.state.update(model_state="DISCONNECTED", websocket_error=None)
                self._policy_retry_at = 0
            try:
                if policy:
                    policy.close()
                if worker and worker is not threading.current_thread():
                    worker.join(timeout=3)
                    if worker.is_alive():
                        raise RuntimeError("Previous inference call is still ending; retry connection switch")
                with self.lock:
                    self._persist_config_locked()
                    self._last_connection_probe = 0
            except Exception:
                with self.lock:
                    self.config = previous
                raise
            finally:
                with self.lock:
                    self._profile_switching = False
                self._connection_wakeup.set()
            if self.camera_resource and self.camera_resource.snapshot()["ready"]:
                self._ensure_worker()

    def _record_inference(self, kind, **fields) -> None:
        with self.lock:
            journal = self._journal
        if journal:
            journal.append(kind, **fields)

    def _finish_inference_journal(self, reason) -> None:
        with self.lock:
            journal, self._journal = self._journal, None
        if journal:
            journal.finish(reason)
            with self.lock:
                self._completed_journal = journal

    def export_last_run_log(self):
        with self.lock:
            journal = self._completed_journal
        if journal is None:
            raise RuntimeError("No completed inference run; start and stop inference first")
        return journal.export()

    def activate_cameras(self) -> dict[str, Any]:
        if self.camera_resource:
            self.camera_resource.activate()
            with self.lock: self.state.update({"camera_state": "READY", "last_error": None, "updated_at": time.time()})
            self._ensure_worker()
            return self.snapshot()
        with self.lock:
            if self.state["state"] == "RUNNING": raise RuntimeError("cannot change cameras while AutoDL cloud inference is running")
            self.preview_stop.set()
            if self.preview_thread and self.preview_thread is not threading.current_thread(): self.preview_thread.join(timeout=.5)
            if self.cameras: self.cameras.close()
            self.cameras = CameraPair(self.config["cameras"])
            self.frames = {}; self.frame_version = 0; self.preview_stop = threading.Event()
            self.preview_thread = threading.Thread(target=self._preview_loop, name="nero-pi05-camera-preview", daemon=True); self.preview_thread.start()
            self.state.update({"camera_state": "READY", "last_error": None, "updated_at": time.time()})
            self._ensure_worker()
            return self.snapshot()

    def start(self, session_id: str, client_id: str) -> dict[str, Any]:
        with self.lock:
            if (self.camera_resource is not None and not self.camera_resource.snapshot()["ready"]) or (self.camera_resource is None and self.cameras is None): raise RuntimeError("activate the external and wrist cameras first")
            try:
                osc = self.osc.state()
            except Exception as exc:
                self.state["osc_error"] = f"{type(exc).__name__}: {exc}"
                self.state["updated_at"] = time.time()
                raise
            self._osc_snapshot = copy.deepcopy(osc)
            session = osc.get("session") or {}
            if session.get("state") != "ACTIVE" or session.get("id") != session_id or session.get("client_id") != client_id:
                raise PermissionError("AutoDL cloud inference requires the caller's active OSC session")
            self._last_gripper_target = None
            self._finish_inference_journal("new inference run")
            self._journal = InferenceJournal(self.config_path.parent.parent / "runtime/logs/autodl_runs",
                {"session_id":session_id,"client_id":client_id,"execution_mode":session.get("execution_mode"),
                 "prompt":self.config["model"]["prompt"],"config":copy.deepcopy(self.config)})
            self._journal.start_sampling(self.osc.state)
            needs_priming = session.get("execution_mode") == "hardware"
            self.state.update({"state": "PRIMING" if needs_priming else "RUNNING",
                               "execution_enabled": not needs_priming, "priming": needs_priming,
                               "session_id": session_id, "client_id": client_id,
                               "execution_mode": session.get("execution_mode"), "last_error": None, "updated_at": time.time()})
            self.state["osc_error"] = None
            self._ensure_worker()
            self._connection_wakeup.set()
            if needs_priming:
                threading.Thread(target=self._prime_control_worker, args=(session_id, client_id), name="nero-pi05-osc-prime", daemon=True).start()
            return self.snapshot()

    def _prime_control_worker(self, session_id: str, client_id: str) -> None:
        """Run hardware priming independently from policy inference."""
        try:
            self._prime_startup_hold(session_id, client_id)
        except Exception as exc:
            with self.lock:
                self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})

    def stop(self, reason: str = "AutoDL cloud inference stopped") -> dict[str, Any]:
        with self.lock:
            self.state.update({"state": "RUNNING" if self.state.get("state") == "PRIMING" else self.state.get("state"),
                               "execution_enabled": False, "priming": False, "updated_at": time.time()})
        self._finish_inference_journal(reason)
        return self.snapshot()

    def _prime_startup_hold(self, session_id: str, client_id: str) -> bool:
        """Send one zero-motion target before releasing AutoDL actions."""
        osc = self.osc.state()
        execution = osc.get("execution") or {}
        command = osc.get("command") or {}
        target = execution.get("measured_tcp_pose") or command.get("target_tcp")
        if not isinstance(target, dict):
            raise RuntimeError("AutoDL startup hold requires a current measured TCP pose")
        position = target.get("position_m")
        orientation = target.get("orientation_xyzw")
        if not isinstance(position, list) or not isinstance(orientation, list):
            raise RuntimeError("AutoDL startup hold received an invalid TCP pose")
        sequence = self._next_sequence()
        result = self.osc.track_tcp(session_id, client_id, sequence, {
            "position_m": list(position), "orientation_xyzw": list(orientation),
        })
        if not result.get("ok"):
            raise RuntimeError(f"AutoDL startup hold rejected: {result}")
        hold_s = max(0.0, float(self.config["execution"].get("startup_hold_s", 0.3)))
        if self.stop_event.wait(hold_s):
            return False
        with self.lock:
            if not self.state.get("priming"):
                return False
            self.state.update({"state": "RUNNING", "execution_enabled": True,
                               "priming": False, "updated_at": time.time()})
        return True

    @staticmethod
    def _feedback_pose(osc: dict[str, Any]) -> dict[str, Any] | None:
        """Pose represented by the state frame captured for one inference."""
        pose = (osc.get("execution") or {}).get("measured_tcp_pose")
        if not isinstance(pose, dict):
            pose = (osc.get("command") or {}).get("target_tcp")
        return copy.deepcopy(pose) if isinstance(pose, dict) else None

    def _observation(self, osc: dict[str, Any], external: Any, wrist: Any) -> dict[str, Any]:
        import numpy as np
        pose = self._feedback_pose(osc)
        # Before an OSC session is opened there may be no published target
        # pose yet. Inference is still useful in preview mode; execution will
        # only be enabled after a session has supplied a real measured pose.
        if not isinstance(pose, dict):
            pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        position = _finite(pose.get("position_m"), 3, "TCP position")
        # The AutoDL training state is expressed directly in the NERO base
        # frame.  Do not apply LIBERO axis flips or arbitrary normalisation.
        rotvec = _quaternion_rotvec(_finite(pose.get("orientation_xyzw"), 4, "TCP orientation"))
        gripper = (osc.get("gripper") or {}).get("width_m", 0.0)
        minimum = float(self.config["gripper"]["min_width_m"])
        maximum = float(self.config["gripper"]["max_width_m"])
        width = max(minimum, min(maximum, float(gripper)))
        ratio = (width - minimum) / (maximum - minimum)
        state = np.asarray([*position, *rotvec, ratio, -ratio], dtype=np.float32)
        return {"observation/image": external, "observation/wrist_image": wrist, "observation/state": state,
                "prompt": self.config["model"]["prompt"]}

    def _normalise_gripper_ratio(self, ratio: float, action_index: int | None = None) -> float:
        """Accept only harmless endpoint round-off, preserving hard safety rejects.

        Network inference can return float32 values infinitesimally outside a
        closed training range (for example ``1.0000001``).  Treating that as a
        fatal action chunk is needlessly disruptive, but broadly clipping a
        policy output would conceal a real semantic error.  The configurable
        tolerance is therefore deliberately small: values outside it still
        reject the complete chunk and request HOLD.
        """
        tolerance = float(self.config["gripper"].get("ratio_boundary_tolerance", 0.001))
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise RuntimeError("AutoDL gripper ratio boundary tolerance must be finite and non-negative")
        if 0.0 <= ratio <= 1.0:
            return ratio
        row = "unknown" if action_index is None else str(action_index)
        if -tolerance <= ratio <= 1.0 + tolerance:
            command_ratio = min(1.0, max(0.0, ratio))
            diagnostic = {"status": "boundary_normalized", "action_index": action_index,
                          "raw_ratio": ratio, "command_ratio": command_ratio,
                          "tolerance": tolerance, "updated_at": time.time()}
            with self.lock:
                self.state["gripper_ratio_diagnostic"] = diagnostic
            # This is a recoverable model-output boundary condition.  The
            # state diagnostic is deliberately user-visible; keep the process
            # log at debug level so a sustained endpoint command does not
            # drown out actual safety events.
            LOGGER.debug("AutoDL gripper ratio boundary-normalized at row %s: %.9g -> %.9g (tolerance %.9g)",
                         row, ratio, command_ratio, tolerance)
            return command_ratio
        diagnostic = {"status": "rejected", "action_index": action_index,
                      "raw_ratio": ratio, "tolerance": tolerance, "updated_at": time.time()}
        with self.lock:
            self.state["gripper_ratio_diagnostic"] = diagnostic
        raise RuntimeError(
            f"AutoDL gripper ratio at row {row} is {ratio:.9g}; expected [0, 1] "
            f"(only endpoint round-off within ±{tolerance:.9g} is normalized)"
        )

    def _target_from_action(self, action: Any, base: dict[str, Any], delta_scale: float | None = None,
                            action_index: int | None = None) -> tuple[dict[str, Any], float]:
        values = _finite(action, 7, "AutoDL Nero action")
        gripper_ratio = self._normalise_gripper_ratio(values[6], action_index)
        scale = float(self.config["execution"].get("action_delta_scale", 1.0) if delta_scale is None else delta_scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise RuntimeError("AutoDL action delta scale must be within [0, 1]")
        # Only the policy's relative TCP translation and rotation are scaled.
        # Its final component is an absolute gripper-opening ratio, not a
        # delta, so leaving it untouched preserves trained gripper semantics.
        delta, rotvec = [value * scale for value in values[:3]], [value * scale for value in values[3:6]]
        position = [float(base["position_m"][i]) + delta[i] for i in range(3)]
        orientation = _quat_product(_rotvec_quaternion(rotvec), _finite(base["orientation_xyzw"], 4, "TCP orientation"))
        norm = math.sqrt(sum(value * value for value in orientation)); orientation = [value / norm for value in orientation]
        return {"position_m": position, "orientation_xyzw": orientation}, gripper_ratio

    @staticmethod
    def _validate_execution_targets(decoded: list[tuple[dict[str, Any], float]], limit: int,
                                   osc: dict[str, Any]) -> None:
        """Reject a whole chunk before dispatch when its executable rows leave the OSC workspace.

        OSC remains authoritative for IK, lease and live safety checks.  This
        local pass deliberately mirrors only the published Cartesian envelope,
        so a later OSC rejection is still fail-closed rather than treated as a
        partially valid chunk.
        """
        workspace = osc.get("workspace") or {}
        lower = workspace.get("min_xyz_m")
        upper = workspace.get("max_xyz_m")
        min_tcp_z = workspace.get("min_tcp_z_m")
        if lower is None or upper is None:
            # Some early OSC snapshots have no workspace publication.  They
            # cannot be locally preflighted, but every row is still finite and
            # the authoritative OSC check below will reject the complete
            # chunk on its first refusal.
            return
        lower = _finite(lower, 3, "OSC workspace minimum")
        upper = _finite(upper, 3, "OSC workspace maximum")
        floor = float(min_tcp_z) if min_tcp_z is not None else lower[2]
        if not math.isfinite(floor) or any(low > high for low, high in zip(lower, upper)):
            raise RuntimeError("OSC published an invalid workspace")
        for index, (target, _gripper) in enumerate(decoded[:limit]):
            position = _finite(target.get("position_m"), 3, f"AutoDL target {index} position")
            if any(value < low or value > high for value, low, high in zip(position, lower, upper)) or position[2] < floor:
                raise RuntimeError(f"AutoDL target {index} is outside the OSC workspace")

    def _send_gripper_if_needed(self, session_id: str, client_id: str, action_value: float) -> None:
        """Send the trained continuous gripper opening target when it changes."""
        minimum, maximum = float(self.config["gripper"]["min_width_m"]), float(self.config["gripper"]["max_width_m"])
        target_width = minimum + float(action_value) * (maximum - minimum)
        tolerance = float(self.config["gripper"].get("dedupe_tolerance_ratio", 0.01)) * (maximum - minimum)
        if isinstance(self._last_gripper_target, (int, float)) and abs(float(self._last_gripper_target) - target_width) <= tolerance:
            return
        sequence = self._next_sequence()
        self._record_inference("gripper_command_requested",sequence=sequence,ratio=action_value,width_m=target_width)
        result = self.osc.gripper(session_id, client_id, sequence, {
            "inference_stream": True,
            "mode": "position", "width_m": target_width,
            "force_n": self.config["gripper"]["force_n"],
        })
        if not result.get("ok"):
            raise RuntimeError(f"AutoDL gripper command rejected: {result}")
        self._last_gripper_target = target_width
        self._record_inference("gripper_dispatch",sequence=sequence,ratio=action_value,width_m=target_width,
                               result={"ok":result.get("ok")})

    def _reject_chunk(self, session_id: str | None, client_id: str | None, reason: str) -> None:
        """Fail closed: no later row from a bad AutoDL chunk may execute."""
        sequence = self._next_sequence()
        with self.lock:
            self.state.update({"state": "ERROR", "execution_enabled": False,
                               "last_error": reason, "last_rejection": reason,
                               "updated_at": time.time()})
        if session_id and client_id:
            try:
                self.osc.hold(str(session_id), str(client_id), sequence, reason)
            except Exception:
                pass
        self.stop_event.set()

    def _run(self) -> None:
        policy = None
        try:
            while not self.stop_event.is_set():
                if policy is None:
                    try:
                        policy = self._policy_client()
                    except Exception as exc:
                        # Keep the worker alive while the independent
                        # connection loop retries.  A transient WebSocket
                        # outage must not permanently stop Action Chunk
                        # publication.
                        with self.lock:
                            self.state.update({"state": "RUNNING", "websocket_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                        self.stop_event.wait(.1)
                        continue
                observation_started = time.perf_counter_ns()
                with self.lock:
                    osc = copy.deepcopy(self._osc_snapshot)
                    session = dict(osc.get("session") or {})
                    session_id, client_id = self.state["session_id"], self.state["client_id"]
                    execution_enabled = bool(self.state.get("execution_enabled"))
                active_session = session.get("state") == "ACTIVE" and session.get("id") == session_id and session.get("client_id") == client_id
                if active_session:
                    try:
                        self.osc.heartbeat(str(client_id), str(session_id))
                    except Exception as exc:
                        with self.lock: self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                try:
                    external, wrist = self.camera_resource.read() if self.camera_resource else (self.cameras.read() if self.cameras else (_ for _ in ()).throw(RuntimeError("cameras are unavailable")))
                except Exception as exc:
                    # A USB camera can momentarily skip a frame while its
                    # shared preview thread is still recovering.  Keep the
                    # existing Action Chunk visible and wait for the next
                    # complete RGB pair instead of terminating AutoDL
                    # inference with a misleading ERROR state.
                    with self.lock:
                        self.state.update({"camera_state": "WAITING", "last_error": None,
                                           "camera_wait_reason": f"waiting for complete RGB pair: {type(exc).__name__}: {exc}",
                                           "updated_at": time.time()})
                    self.stop_event.wait(.1)
                    continue
                with self.lock:
                    if self.state.get("camera_state") == "WAITING":
                        self.state.update({"camera_state": "READY", "camera_wait_reason": None, "updated_at": time.time()})
                observation = self._observation(osc, external, wrist)
                self._record_inference_input(external, wrist, observation)
                try:
                    inference_started = time.perf_counter()
                    response = policy.infer(observation)
                    inference_ms = (time.perf_counter() - inference_started) * 1000.0
                except Exception as exc:
                    self._record_websocket_event("inference_failure", f"{type(exc).__name__}: {exc}")
                    self._invalidate_policy(policy, f"OpenPI WebSocket inference failed: {type(exc).__name__}: {exc}")
                    policy = None
                    self.stop_event.wait(.05)
                    continue
                base = self._feedback_pose(osc)
                self._record_inference("model_action_chunk",chunk_id=int(self.state.get("chunk_sequence",0))+1,
                    observation_started_perf_counter_ns=observation_started,inference_ms=inference_ms,
                    observation_state=observation["observation/state"].tolist(),prompt=observation["prompt"],
                    inference_base_tcp=base,action_chunk=(response.get("actions").tolist() if hasattr(response.get("actions"),"tolist") else response.get("actions")) if isinstance(response,dict) else response)
                if base is None and (not execution_enabled or not active_session):
                    # Preview inference may continue after the operator stops
                    # forwarding or closes the OSC session.  Without a live
                    # TCP reference there is nothing safe to decode or send,
                    # but this is an expected idle condition—not a rejected
                    # AutoDL chunk and not a reason to issue HOLD again.
                    self.stop_event.wait(float(self.config["execution"]["period_s"]))
                    continue
                try:
                    actions = response.get("actions") if isinstance(response, dict) else None
                    if actions is None: raise RuntimeError("AutoDL response does not contain actions")
                    rows = actions.tolist() if hasattr(actions, "tolist") else actions
                    if not isinstance(rows, list) or not rows: raise RuntimeError("AutoDL action chunk is empty")
                    expected_steps = int(self.config["execution"].get("expected_chunk_steps", 10))
                    if len(rows) != expected_steps:
                        raise RuntimeError(f"AutoDL action chunk must contain exactly {expected_steps} steps, got {len(rows)}")
                    if base is None:
                        raise RuntimeError("OSC measured TCP feedback is unavailable for this AutoDL chunk")
                    with self.lock:
                        delta_scale = float(self.config["execution"].get("action_delta_scale", 1.0))
                    decoded = [self._target_from_action(action, base, delta_scale, index)
                               for index, action in enumerate(rows)]
                    absolute_chunk = [target for target, _ratio in decoded]
                    limit = min(len(rows), int(self.config["execution"]["replan_steps"]), int(self.config["execution"]["max_chunk_steps"]))
                    self._validate_execution_targets(decoded, limit, osc)
                except Exception as exc:
                    self._reject_chunk(session_id, client_id, f"AutoDL action chunk rejected: {type(exc).__name__}: {exc}")
                    break
                with self.lock:
                    self.state.update({"action_chunk": rows, "absolute_tcp_chunk": absolute_chunk,
                                       "inference_base_tcp": copy.deepcopy(base), "action_chunk_length": len(rows),
                                       "decoded_first_target": copy.deepcopy(absolute_chunk[0]),
                                       "last_rejection": None,
                                       "chunk_sequence": int(self.state.get("chunk_sequence", 0)) + 1,
                                       "inference_ms": inference_ms, "updated_at": time.time()})
                with self.lock:
                    priming = bool(self.state.get("priming"))
                if priming and active_session and session.get("execution_mode") == "hardware":
                    # Priming is a control concern. Keep inference and Action
                    # Chunk publication alive while the separate control path
                    # is unavailable or waiting for a hardware handoff.
                    self.stop_event.wait(float(self.config["execution"]["period_s"]))
                    continue
                if not execution_enabled or not active_session:
                    self.stop_event.wait(float(self.config["execution"]["period_s"]))
                    continue
                period_s = float(self.config["execution"]["period_s"])
                last_sent_target = None
                last_dispatch_completed_at: float | None = None
                for index, ((target, gripper), _action) in enumerate(zip(decoded[:limit], rows[:limit])):
                    if self.stop_event.is_set(): break
                    # The first validated target is sent immediately.  Every
                    # following target is paced from the completion of the
                    # previous successful dispatch, rather than from the
                    # observation time.  Inference or an OSC call may take
                    # longer than one period; in that case we stretch the
                    # chunk instead of trying to catch up with a burst.
                    if last_dispatch_completed_at is not None:
                        if wait_deadline(self.stop_event, last_dispatch_completed_at + period_s):
                            break
                    with self.lock:
                        execution_enabled = bool(self.state.get("execution_enabled"))
                    if not execution_enabled:
                        # The operator may stop AutoDL forwarding while this
                        # chunk is between rows.  Do not let a deliberately
                        # revoked session turn that normal stop into an OSC
                        # rejection on the UI.
                        break
                    sequence = self._next_sequence()
                    try:
                        self._record_inference("tcp_command_requested",chunk_id=int(self.state["chunk_sequence"]),
                            action_index=index,sequence=sequence,target_tcp=target,gripper_ratio=gripper)
                        result = self.osc.track_tcp(session_id, client_id, sequence, target)
                        self._record_inference("tcp_dispatch",chunk_id=int(self.state["chunk_sequence"]),
                            action_index=index,sequence=sequence,target_tcp=target,gripper_ratio=gripper,
                            result={"ok":result.get("ok")})
                        if not result.get("ok"): raise RuntimeError(f"OSC target rejected: {result}")
                        last_sent_target = target
                        self._send_gripper_if_needed(session_id, client_id, gripper)
                    except Exception as exc:
                        with self.lock:
                            execution_enabled = bool(self.state.get("execution_enabled"))
                        if not execution_enabled:
                            break
                        self._reject_chunk(session_id, client_id, f"AutoDL chunk execution rejected: {type(exc).__name__}: {exc}")
                        break
                    last_dispatch_completed_at = time.perf_counter()
                    with self.lock: self.state["executed_steps"] += 1; self.state["updated_at"] = time.time()
                if last_sent_target is not None and not self.stop_event.is_set():
                    wait_for_arrival(self,last_sent_target,session_id,client_id)
        except Exception as exc:
            with self.lock: self.state.update({"state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        finally:
            self._finish_inference_journal("inference worker ended")
            with self.lock:
                session_id, client_id = self.state.get("session_id"), self.state.get("client_id")
                if self.stop_event.is_set() and self.state.get("state") != "ERROR":
                    self.state["state"] = "IDLE"
                self.state["updated_at"] = time.time()
            if session_id and client_id and not self.stop_event.is_set():
                try:
                    self.osc.hold(str(session_id), str(client_id), self._next_sequence(), "AutoDL cloud inference failed")
                except Exception:
                    pass

    def close(self) -> None:
        self.stop("service closing")
        self._connection_stop.set()
        self._osc_state_stop.set()
        if self._connection_thread is not threading.current_thread():
            self._connection_thread.join(timeout=1.0)
        if self._osc_state_thread is not threading.current_thread():
            self._osc_state_thread.join(timeout=1.0)
        with self.lock:
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        with self.lock:
            policy = self._policy
            self._policy = None
            self.state.update({"model_state": "DISCONNECTED", "websocket_error": "service closing", "updated_at": time.time()})
        if policy:
            try: policy.close()
            except Exception: pass
        if self.camera_resource is None and self.cameras: self.cameras.close()
