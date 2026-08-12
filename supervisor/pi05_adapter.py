"""π0.5 input adapter: OpenPI observations to OSC absolute TCP commands.

This module deliberately owns no robot object, transport, or servo loop.  It
is an input adapter running beside the HTTP service; its only output path is
``OperationalSpaceController.osc_command``.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any
from .camera_resource import SharedCameraResource


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
        from websockets.sync.client import connect
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.socket = connect(uri, compression=None, max_size=None, open_timeout=timeout_s, close_timeout=2)
        self.socket.recv(timeout=timeout_s)  # policy metadata handshake
        self.timeout_s = timeout_s

    def infer(self, observation: dict[str, Any]) -> Any:
        import msgpack
        self.socket.send(msgpack.packb(observation, default=_pack_array))
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
            return False
        try:
            pong = self.socket.ping()
            return bool(pong.wait(timeout=1.0))
        except Exception:
            return False


class _LegacyCameraPair:
    """Latest-frame dual camera reader with the π0.5 224x224 RGB contract."""
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
        self._connection_thread.start()
        self._osc_state_thread.start()

    def _new_state(self) -> dict[str, Any]:
        return {"adapter": "pi05", "state": "IDLE", "camera_state": "IDLE", "model_state": "UNKNOWN",
                "session_id": None, "client_id": None, "execution_mode": None, "prompt": self.config["model"]["prompt"],
                "last_error": None, "websocket_error": None, "inference_ms": None, "action_chunk": None, "absolute_tcp_chunk": None,
                "inference_base_tcp": None, "action_chunk_length": 0, "chunk_sequence": 0,
                "executed_steps": 0, "sequence": 0, "execution_enabled": False,
                "osc_error": None,
                "priming": False, "updated_at": time.time()}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_connections_locked()
            result = copy.deepcopy(self.state); result["config"] = copy.deepcopy(self.config)
            shared = self.camera_resource.snapshot() if self.camera_resource else {"ready": self.cameras is not None, "frame_version": 0}
            result["camera_ready"] = shared["ready"]; result["frame_version"] = shared["frame_version"]
            result["camera"] = shared
            result["connections"] = copy.deepcopy(self._connections)
            return result

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
                            policy = OpenPIClient(str(model["host"]), int(model["port"]), float(model["request_timeout_s"]))
                            with self.lock:
                                if self._connection_stop.is_set():
                                    policy.close()
                                else:
                                    self._policy = policy
                                    self._policy_generation += 1
                                    self.state.update({"model_state": "CONNECTED", "websocket_error": None, "updated_at": time.time()})
                                    if self.camera_resource and self.camera_resource.snapshot().get("ready"):
                                        self._ensure_worker()
                        except Exception as exc:
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
        fallback client here races that loop and produces multiple OpenPI
        sockets, which appear on the server as clients that vanish during
        handshake.
        """
        deadline = time.monotonic() + float(self.config["model"]["request_timeout_s"])
        while time.monotonic() < deadline and not self.stop_event.is_set():
            with self.lock:
                policy = self._policy
            if policy is not None:
                return policy
            with self._policy_create_lock:
                with self.lock:
                    policy = self._policy
                if policy is not None:
                    return policy
                try:
                    model = self.config["model"]
                    policy = OpenPIClient(str(model["host"]), int(model["port"]), float(model["request_timeout_s"]))
                    with self.lock:
                        self._policy = policy
                        self._policy_generation += 1
                        self.state.update({"model_state": "CONNECTED", "websocket_error": None, "updated_at": time.time()})
                    return policy
                except Exception as exc:
                    with self.lock:
                        self.state.update({"model_state": "DISCONNECTED", "websocket_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
            time.sleep(.05)
        raise RuntimeError("π0.5 WebSocket is not connected")

    def _ensure_worker(self) -> None:
        with self.lock:
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
        local_forward_listening = self._local_forward_listening(port)
        policy_connected = self.state.get("model_state") == "CONNECTED"
        # Do not synchronously query OSC while serving pi05 state. A stalled
        # control backend must not block inference telemetry or Action Chunks.
        transport = (self._osc_snapshot.get("transport") or {}) if self._osc_snapshot else {}
        osc_error = self.state.get("osc_error")
        self._connections = {
            "control": {"state": "bad" if osc_error else ("ok" if transport.get("connected") else "warn"), "label": "NERO control service", "endpoint": "127.0.0.1:8765", "message": str(osc_error) if osc_error else "OSC control channel available"},
            "ssh_forward": {"state": "ok" if local_forward_listening else "bad", "label": "SSH 本地转发", "endpoint": f"127.0.0.1:{port}", "message": "本地转发端口已监听" if local_forward_listening else "本地转发端口未监听"},
            "policy": {"state": "ok" if policy_connected else "bad", "label": "π0.5 WebSocket", "endpoint": "OpenPI policy server", "message": "OpenPI WebSocket 已连接" if policy_connected else (str(self.state.get("websocket_error")) if self.state.get("websocket_error") else "OpenPI WebSocket 未连接")},
        }

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
        with self.lock:
            model = value.get("model") if isinstance(value, dict) else None
            cameras = value.get("cameras") if isinstance(value, dict) else None
            if isinstance(model, dict) and "prompt" in model:
                prompt = str(model["prompt"]).strip()
                if not 1 <= len(prompt) <= 500: raise ValueError("prompt must contain 1-500 characters")
                self.config["model"]["prompt"] = prompt; self.state["prompt"] = prompt
            if isinstance(cameras, dict):
                if self.state["state"] == "RUNNING": raise RuntimeError("stop π0.5 inference before changing cameras")
                if self.camera_resource:
                    self.camera_resource.update_config(cameras)
                    self.config["cameras"] = copy.deepcopy(self.camera_resource.config)
                    self.state["updated_at"] = time.time(); return self.snapshot()
                for key in ("external", "wrist"):
                    item = cameras.get(key)
                    if not isinstance(item, dict): continue
                    index = int(item.get("index", self.config["cameras"][key]["index"]))
                    if not 0 <= index <= 32: raise ValueError(f"{key} camera index must be 0-32")
                    self.config["cameras"][key]["index"] = index
                if self.config["cameras"]["external"]["index"] == self.config["cameras"]["wrist"]["index"]:
                    raise ValueError("external and wrist cameras must be different")
            self.state["updated_at"] = time.time()
            return self.snapshot()

    def activate_cameras(self) -> dict[str, Any]:
        if self.camera_resource:
            self.camera_resource.activate()
            with self.lock: self.state.update({"camera_state": "READY", "last_error": None, "updated_at": time.time()})
            self._ensure_worker()
            return self.snapshot()
        with self.lock:
            if self.state["state"] == "RUNNING": raise RuntimeError("cannot change cameras while π0.5 is running")
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
                raise PermissionError("π0.5 requires the caller's active OSC session")
            self._last_gripper_target = None
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

    def stop(self, reason: str = "π0.5 adapter stopped") -> dict[str, Any]:
        with self.lock:
            self.state.update({"state": "RUNNING" if self.state.get("state") == "PRIMING" else self.state.get("state"),
                               "execution_enabled": False, "priming": False, "updated_at": time.time()})
            return self.snapshot()

    def _prime_startup_hold(self, session_id: str, client_id: str) -> bool:
        """Send one zero-motion target before releasing π0.5 actions."""
        osc = self.osc.state()
        execution = osc.get("execution") or {}
        command = osc.get("command") or {}
        target = execution.get("measured_tcp_pose") or command.get("target_tcp")
        if not isinstance(target, dict):
            raise RuntimeError("π0.5 startup hold requires a current measured TCP pose")
        position = target.get("position_m")
        orientation = target.get("orientation_xyzw")
        if not isinstance(position, list) or not isinstance(orientation, list):
            raise RuntimeError("π0.5 startup hold received an invalid TCP pose")
        with self.lock:
            self.state["sequence"] += 1
            sequence = max(self.state["sequence"], int(time.time() * 1000))
        result = self.osc.track_tcp(session_id, client_id, sequence, {
            "position_m": list(position), "orientation_xyzw": list(orientation),
        })
        if not result.get("ok"):
            raise RuntimeError(f"π0.5 startup hold rejected: {result}")
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
        # π0.5's state receives its rotation-vector representation.  The current
        # system's OSC pose is xyzw; retaining zero rotvec is unsafe, so the policy
        # currently receives the calibrated translational state plus neutral rotation
        # until a policy-specific quaternion->rotvec calibration is configured.
        rotvec = _quaternion_rotvec(_finite(pose.get("orientation_xyzw"), 4, "TCP orientation"))
        signs_r = _finite(self.config["frames"]["libero_to_nero"]["rotation_axis_sign"], 3, "rotation signs")
        gripper = (osc.get("gripper") or {}).get("width_m", 0.0); width = max(0.0, min(0.095, float(gripper)))
        state = np.asarray([-position[0], position[1], position[2], *(rotvec[i] * signs_r[i] for i in range(3)), width / 0.095 * .04, -width / 0.095 * .04], dtype=np.float32)
        return {"observation/image": external, "observation/wrist_image": wrist, "observation/state": state,
                "prompt": self.config["model"]["prompt"]}

    def _target_from_action(self, action: Any, base: dict[str, Any]) -> tuple[dict[str, Any], float]:
        values = _finite(action, 7, "π0.5 action")
        if any(abs(value) > 1.05 for value in values): raise RuntimeError("π0.5 action is outside [-1, 1]")
        values = [max(-1., min(1., value)) for value in values]
        model, frame = self.config["model"], self.config["frames"]["libero_to_nero"]
        rate = float(model["rate_scale"]); ps = _finite(model["position_scale_m"], 3, "position scale"); rs = _finite(model["rotation_scale_rad"], 3, "rotation scale")
        signs_t = _finite(frame["translation_axis_sign"], 3, "translation signs"); signs_r = _finite(frame["rotation_axis_sign"], 3, "rotation signs")
        delta = [values[i] * ps[i] * rate * signs_t[i] for i in range(3)]
        rotvec = [values[i + 3] * rs[i] * rate * signs_r[i] for i in range(3)]
        position = [float(base["position_m"][i]) + delta[i] for i in range(3)]
        orientation = _quat_product(_finite(base["orientation_xyzw"], 4, "TCP orientation"), _rotvec_quaternion(rotvec))
        norm = math.sqrt(sum(value * value for value in orientation)); orientation = [value / norm for value in orientation]
        return {"position_m": position, "orientation_xyzw": orientation}, values[6]

    def _send_gripper_if_needed(self, session_id: str, client_id: str, action_value: float) -> None:
        """Send a gripper edge only when feedback says a transition is needed."""
        threshold = float(self.config["gripper"]["switch_threshold"])
        if abs(float(action_value)) < threshold:
            return
        desired = "open" if float(action_value) <= -threshold else "closed"
        with self.lock:
            current = (self._osc_snapshot.get("gripper") or {}).copy()
        width = current.get("width_m")
        openness = None
        if isinstance(width, (int, float)):
            openness = max(0.0, min(1.0, float(width) / 0.095))
        # Match the reference adapter's 80%/20% hysteresis. Once a target has
        # been sent, suppress duplicate SDK transactions until the opposite
        # gripper edge is requested.
        if desired == "open" and openness is not None and openness >= 0.8:
            self._last_gripper_target = desired
            return
        if desired == "closed" and openness is not None and openness <= 0.2:
            self._last_gripper_target = desired
            return
        if self._last_gripper_target == desired:
            return
        with self.lock:
            self.state["sequence"] += 1
            sequence = max(self.state["sequence"], int(time.time() * 1000))
        result = self.osc.gripper(session_id, client_id, sequence, {
            "mode": "open" if desired == "open" else "position",
            "width_m": self.config["gripper"]["open_width_m"] if desired == "open" else self.config["gripper"]["closed_width_m"],
            "force_n": self.config["gripper"]["force_n"],
        })
        if not result.get("ok"):
            raise RuntimeError(f"π0.5 gripper command rejected: {result}")
        self._last_gripper_target = desired

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
                started = time.monotonic()
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
                external, wrist = self.camera_resource.read() if self.camera_resource else (self.cameras.read() if self.cameras else (_ for _ in ()).throw(RuntimeError("cameras are unavailable")))
                try:
                    response = policy.infer(self._observation(osc, external, wrist))
                except Exception as exc:
                    self._invalidate_policy(policy, f"OpenPI WebSocket inference failed: {type(exc).__name__}: {exc}")
                    policy = None
                    self.stop_event.wait(.05)
                    continue
                actions = response.get("actions") if isinstance(response, dict) else None
                if actions is None: raise RuntimeError("OpenPI response does not contain actions")
                rows = actions.tolist() if hasattr(actions, "tolist") else actions
                if not isinstance(rows, list) or not rows: raise RuntimeError("OpenPI Action Chunk is empty")
                expected_steps = int(self.config["execution"].get("expected_chunk_steps", 10))
                if len(rows) != expected_steps:
                    raise RuntimeError(f"OpenPI Action Chunk must contain exactly {expected_steps} steps, got {len(rows)}")
                limit = min(len(rows), int(self.config["execution"]["replan_steps"]), int(self.config["execution"]["max_chunk_steps"]))
                base = self._feedback_pose(osc)
                absolute_chunk = [self._target_from_action(action, base)[0] for action in rows] if base is not None else None
                with self.lock:
                    self.state.update({"action_chunk": rows, "absolute_tcp_chunk": absolute_chunk,
                                       "inference_base_tcp": copy.deepcopy(base), "action_chunk_length": len(rows),
                                       "chunk_sequence": int(self.state.get("chunk_sequence", 0)) + 1,
                                       "inference_ms": (time.monotonic()-started)*1000., "updated_at": time.time()})
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
                if base is None or absolute_chunk is None:
                    with self.lock:
                        self.state.update({"osc_error": "OSC measured TCP feedback is unavailable for this Action Chunk", "updated_at": time.time()})
                    self.stop_event.wait(float(self.config["execution"]["period_s"]))
                    continue
                period_s = float(self.config["execution"]["period_s"])
                next_step_at = started + period_s
                for index, (action, target) in enumerate(zip(rows[:limit], absolute_chunk[:limit])):
                    if self.stop_event.is_set(): break
                    gripper = _finite(action, 7, "π0.5 action")[6]
                    with self.lock: self.state["sequence"] += 1; sequence = max(self.state["sequence"], int(time.time()*1000))
                    try:
                        result = self.osc.track_tcp(session_id, client_id, sequence, target)
                        if not result.get("ok"): raise RuntimeError(f"OSC target rejected: {result}")
                    except Exception as exc:
                        with self.lock: self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                    try:
                        self._send_gripper_if_needed(session_id, client_id, gripper)
                    except Exception as exc:
                        with self.lock: self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                    with self.lock: self.state["executed_steps"] += 1; self.state["updated_at"] = time.time()
                    # period_s is the complete observation-to-command cadence;
                    # policy inference time must not be added on top of it.
                    if self.stop_event.wait(max(0.0, next_step_at - time.monotonic())): break
                    next_step_at += period_s
        except Exception as exc:
            with self.lock: self.state.update({"state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        finally:
            with self.lock:
                session_id, client_id = self.state.get("session_id"), self.state.get("client_id")
                self.state["state"] = "IDLE" if self.stop_event.is_set() else self.state["state"]
                self.state["updated_at"] = time.time()
            if session_id and client_id and not self.stop_event.is_set():
                try:
                    self.state["sequence"] += 1
                    self.osc.hold(str(session_id), str(client_id), max(int(self.state["sequence"]), int(time.time() * 1000)), "π0.5 adapter failed")
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
