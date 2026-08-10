"""π0.5 input adapter: OpenPI observations to OSC absolute TCP commands.

This module deliberately owns no robot object, transport, or servo loop.  It
is an input adapter running beside the HTTP service; its only output path is
``OperationalSpaceController.osc_command``.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import socket
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
        self.socket.close()


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
    def __init__(self, broker: Any, config_path: Path, camera_resource: SharedCameraResource | None = None) -> None:
        self.broker, self.config_path = broker, Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.lock = threading.RLock(); self.stop_event = threading.Event(); self.worker: threading.Thread | None = None
        self.camera_resource = camera_resource
        # Compatibility seam for direct adapter tests; production always uses
        # the broker-owned shared camera resource.
        self.cameras: Any | None = None if camera_resource else None
        self._last_connection_probe = 0.0
        self._connections: dict[str, Any] = {}
        self.state: dict[str, Any] = self._new_state()

    def _new_state(self) -> dict[str, Any]:
        return {"adapter": "pi05", "state": "IDLE", "camera_state": "IDLE", "model_state": "UNKNOWN",
                "session_id": None, "client_id": None, "execution_mode": None, "prompt": self.config["model"]["prompt"],
                "last_error": None, "inference_ms": None, "action_chunk": None, "action_chunk_length": 0,
                "executed_steps": 0, "sequence": 0, "updated_at": time.time()}

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
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=.25): return True
        except OSError: return False

    def _refresh_connections_locked(self) -> None:
        if time.monotonic() - self._last_connection_probe < 1.0: return
        self._last_connection_probe = time.monotonic()
        connection = self.config.get("connection") or {}
        host = str(connection.get("policy_host", self.config["model"]["host"]))
        port = int(connection.get("policy_port", self.config["model"]["port"]))
        port_ok = self._port_open(host, port)
        osc = self.broker.osc_state(); transport = osc.get("transport") or {}
        self._connections = {
            "control": {"state": "ok" if transport.get("connected") else "warn", "label": "NERO 控制服务", "endpoint": "127.0.0.1:8765", "message": "OSC 状态可读"},
            "ssh_forward": {"state": "ok" if port_ok else "bad", "label": "SSH 本地转发", "endpoint": f"{host}:{port}", "message": "端口已连通" if port_ok else "未连接"},
            "policy": {"state": "ok" if self.state.get("model_state") == "CONNECTED" else ("warn" if port_ok else "bad"), "label": "π0.5 WebSocket", "endpoint": "OpenPI policy server", "message": "OpenPI 已连接" if self.state.get("model_state") == "CONNECTED" else ("等待推理握手" if port_ok else "OpenPI WebSocket 不可达")},
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
            if self.state["state"] == "RUNNING": raise RuntimeError("stop π0.5 inference before changing configuration")
            model = value.get("model") if isinstance(value, dict) else None
            cameras = value.get("cameras") if isinstance(value, dict) else None
            if isinstance(model, dict) and "prompt" in model:
                prompt = str(model["prompt"]).strip()
                if not 1 <= len(prompt) <= 500: raise ValueError("prompt must contain 1-500 characters")
                self.config["model"]["prompt"] = prompt; self.state["prompt"] = prompt
            if isinstance(cameras, dict):
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
            with self.lock: self.state.update({"state": "IDLE", "camera_state": "READY", "model_state": "UNKNOWN", "last_error": None, "updated_at": time.time()})
            return self.snapshot()
        with self.lock:
            if self.state["state"] == "RUNNING": raise RuntimeError("cannot change cameras while π0.5 is running")
            self.preview_stop.set()
            if self.preview_thread and self.preview_thread is not threading.current_thread(): self.preview_thread.join(timeout=.5)
            if self.cameras: self.cameras.close()
            self.cameras = CameraPair(self.config["cameras"])
            self.frames = {}; self.frame_version = 0; self.preview_stop = threading.Event()
            self.preview_thread = threading.Thread(target=self._preview_loop, name="nero-pi05-camera-preview", daemon=True); self.preview_thread.start()
            self.state.update({"state": "IDLE", "camera_state": "READY", "model_state": "UNKNOWN", "last_error": None, "updated_at": time.time()})
            return self.snapshot()

    def start(self, session_id: str, client_id: str) -> dict[str, Any]:
        with self.lock:
            if self.worker and self.worker.is_alive(): raise RuntimeError("π0.5 inference is already running")
            if (self.camera_resource is not None and not self.camera_resource.snapshot()["ready"]) or (self.camera_resource is None and self.cameras is None): raise RuntimeError("activate the external and wrist cameras first")
            osc = self.broker.osc_state(); session = osc.get("session") or {}
            if session.get("state") != "ACTIVE" or session.get("id") != session_id or session.get("client_id") != client_id:
                raise PermissionError("π0.5 requires the caller's active OSC session")
            self.stop_event = threading.Event()
            self.state.update({"state": "RUNNING", "session_id": session_id, "client_id": client_id,
                               "execution_mode": session.get("execution_mode"), "last_error": None, "updated_at": time.time()})
            self.worker = threading.Thread(target=self._run, name="nero-pi05-input-adapter", daemon=True); self.worker.start()
            return self.snapshot()

    def stop(self, reason: str = "π0.5 adapter stopped") -> dict[str, Any]:
        self.stop_event.set()
        with self.lock:
            if self.state["state"] == "RUNNING": self.state["state"] = "STOPPING"
        worker = self.worker
        if worker and worker is not threading.current_thread(): worker.join(timeout=2.0)
        with self.lock:
            self.state.update({"state": "IDLE", "updated_at": time.time()})
            return self.snapshot()

    def _observation(self, osc: dict[str, Any], external: Any, wrist: Any) -> dict[str, Any]:
        import numpy as np
        pose = (osc.get("execution") or {}).get("measured_tcp_pose") or (osc.get("command") or {}).get("target_tcp")
        if not isinstance(pose, dict): raise RuntimeError("OSC did not publish a current TCP pose")
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

    def _run(self) -> None:
        policy = None
        try:
            model = self.config["model"]; policy = OpenPIClient(str(model["host"]), int(model["port"]), float(model["request_timeout_s"]))
            with self.lock: self.state["model_state"] = "CONNECTED"
            while not self.stop_event.is_set():
                started = time.monotonic(); osc = self.broker.osc_state(); session = osc.get("session") or {}
                with self.lock: session_id, client_id = self.state["session_id"], self.state["client_id"]
                if session.get("state") != "ACTIVE" or session.get("id") != session_id: raise RuntimeError("OSC session ended")
                external, wrist = self.camera_resource.read() if self.camera_resource else (self.cameras.read() if self.cameras else (_ for _ in ()).throw(RuntimeError("cameras are unavailable")))
                response = policy.infer(self._observation(osc, external, wrist)); actions = response.get("actions") if isinstance(response, dict) else None
                if actions is None: raise RuntimeError("OpenPI response does not contain actions")
                rows = actions.tolist() if hasattr(actions, "tolist") else actions
                if not isinstance(rows, list) or not rows: raise RuntimeError("OpenPI Action Chunk is empty")
                limit = min(len(rows), int(self.config["execution"]["replan_steps"]), int(self.config["execution"]["max_chunk_steps"]))
                with self.lock:
                    self.state.update({"action_chunk": rows, "action_chunk_length": len(rows), "inference_ms": (time.monotonic()-started)*1000., "updated_at": time.time()})
                base = (osc.get("command") or {}).get("target_tcp") or (osc.get("execution") or {}).get("measured_tcp_pose")
                period_s = float(self.config["execution"]["period_s"])
                next_step_at = started + period_s
                for index, action in enumerate(rows[:limit]):
                    if self.stop_event.is_set(): break
                    target, gripper = self._target_from_action(action, base)
                    with self.lock: self.state["sequence"] += 1; sequence = max(self.state["sequence"], int(time.time()*1000))
                    result = self.broker.osc_command({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "track_tcp", "acknowledgement_only": True, "payload": {"target_pose": target}})
                    if not result.get("ok"): raise RuntimeError(f"OSC rejected π0.5 target: {result}")
                    base = target
                    if abs(gripper) >= float(self.config["gripper"]["switch_threshold"]):
                        # LIBERO/Panda uses -1=open and +1=closed.  The NERO
                        # width convention is the inverse, exactly as in the
                        # migrated π0.5 bridge.
                        opening = gripper <= -float(self.config["gripper"]["switch_threshold"])
                        self.state["sequence"] += 1
                        self.broker.osc_command({"session_id": session_id, "client_id": client_id, "sequence": max(self.state["sequence"], int(time.time()*1000)+1), "type": "gripper", "payload": {"mode": "open" if opening else "position", "width_m": self.config["gripper"]["open_width_m"] if opening else self.config["gripper"]["closed_width_m"], "force_n": self.config["gripper"]["force_n"]}})
                    with self.lock: self.state["executed_steps"] += 1; self.state["updated_at"] = time.time()
                    # period_s is the complete observation-to-command cadence;
                    # policy inference time must not be added on top of it.
                    if self.stop_event.wait(max(0.0, next_step_at - time.monotonic())): break
                    next_step_at += period_s
        except Exception as exc:
            with self.lock: self.state.update({"state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        finally:
            if policy:
                try: policy.close()
                except Exception: pass

    def close(self) -> None:
        self.stop("service closing")
        if self.camera_resource is None and self.cameras: self.cameras.close()
