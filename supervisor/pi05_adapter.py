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
from pathlib import Path
import socket
import threading
import time
from typing import Any
from .camera_resource import SharedCameraResource
from .inference_recording import RunCapture, CompletedRun, latest_completed_run, await_target
from .openpi_client import OpenPIClient, _pack_array, _unpack_array
from .inference_config import project, prepare_update, save_config, endpoint_uri
from .inference_contract import metadata_compatibility, make_observation, action_rows
from .inference_connection import local_listener, loopback
from .inference_rtc import run_adapter as run_rtc_adapter, compatibility as rtc_compatibility
from .inference_preview import ObservationSource, preview_once
from .inference_feedback import MeasuredTCPFeed
from .inference_gripper_feedback import GripperFeedbackCache, GripperFeedbackFeed


LOGGER = logging.getLogger(__name__)


class ObservationNotReady(RuntimeError):
    """A measured field is absent; wait instead of inventing an output input."""


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
        self.config = project(json.loads(self.config_path.read_text(encoding="utf-8")))
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
        self._forward_listener = False
        self._listener_checked_at = 0.0
        self._tcp_evidence: dict[str, Any] = {}
        self._connection_stage = "pending"
        self._camera_status_snapshot: dict[str, Any] | None = None
        self._connections: dict[str, Any] = {}
        self._osc_snapshot: dict[str, Any] = {}
        # Logging has an independent immutable publication. Its reader never
        # waits for inference/UI state locks and never requests hardware data.
        self._log_feedback_snapshot: dict[str, Any] = {}
        self._last_gripper_target: str | None = None
        self._connection_revision = 0
        self._run_revision = 0
        self._changing_connection = False
        self._switch_lock = threading.Lock()
        self._row_lock = threading.RLock()
        self._wait_cancel = threading.Event()
        self._capture: RunCapture | None = None
        self._completed_capture: RunCapture | CompletedRun | None = latest_completed_run(
            self.config_path.parent.parent/'runtime/logs/inference_runs')
        self._pending_gripper: dict[str, Any] | None = None
        # Gripper SDK calls are deliberately kept out of the 50 ms TCP
        # dispatch path.  A slow CAN/SDK acknowledgement must not consume the
        # next TCP slot; at most one not-yet-sent target is retained and newer
        # targets replace it.
        self._gripper_condition = threading.Condition()
        self._gripper_stop = threading.Event()
        self._gripper_job: dict[str, Any] | None = None
        self._gripper_inflight_target: float | None = None
        self._gripper_worker = threading.Thread(target=self._gripper_loop,
                                                 name="nero-pi05-gripper",
                                                 daemon=True)
        self._rtc_run = None
        self._closing = False
        self._preview_busy = False
        self._observation_source = ObservationSource(self)
        self._tcp_feed = (MeasuredTCPFeed(self._observation_source, self._feedback_context, self._osc_state_stop)
                          if callable(getattr(self.osc, 'sensor_sample', None)) else None)
        self._gripper_observation_identity = None
        self._gripper_observation_cache: dict[str, Any] | None = None
        self._measured_gripper_cache = GripperFeedbackCache()
        self._gripper_feed = (GripperFeedbackFeed(self.osc.inference_gripper_feedback,
            self._feedback_context, self._osc_state_stop, self._measured_gripper_cache, self._event)
            if callable(getattr(self.osc, 'inference_gripper_feedback', None)) else None)
        self.state: dict[str, Any] = self._new_state()
        self._gripper_worker.start()
        self._connection_thread.start()
        if self._gripper_feed:
            self._gripper_feed.thread.start()
        self._osc_state_thread.start()
        if self._tcp_feed:
            self._tcp_feed.thread.start()

    def _new_state(self) -> dict[str, Any]:
        return {"preview": {"state": "WAITING", "error": None, "updated_at": None, "sequence": 0, "stale": True},
                "chunk_source": None, "chunk_updated_at": None,
                "adapter": "pi05", "state": "IDLE", "camera_state": "IDLE", "model_state": "UNKNOWN",
                "session_id": None, "client_id": None, "execution_mode": None, "prompt": self.config["model"]["prompt"],
                "last_error": None, "websocket_error": None, "inference_ms": None, "action_chunk": None, "absolute_tcp_chunk": None,
                "inference_base_tcp": None, "action_chunk_length": 0, "chunk_sequence": 0,
                "executed_steps": 0, "sequence": 0, "execution_enabled": False,
                "osc_error": None,
                "protocol_compatibility": {"status": "pending", "message": "待连接验证"},
                "last_response_steps": None,
                "camera_wait_reason": None,
                "priming": False, "last_rejection": None,
                # The policy was trained to emit a closed [0, 1] gripper
                # ratio.  Keep the last boundary decision visible so an
                # operator can distinguish floating-point round-off from a
                # genuinely invalid policy output.
                "gripper_ratio_diagnostic": None,
                "gripper_transport": {"state": "idle", "error": None, "updated_at": None},
                "decoded_first_target": None,
                "websocket_diagnostics": {"connect_attempts": 0, "connect_successes": 0,
                                          "disconnects": 0, "ping_failures": 0,
                                          "inference_failures": 0, "last_event": "等待 Model Policy WebSocket"},
                "updated_at": time.time()}

    def _metadata_status(self, metadata):
        status = metadata_compatibility(metadata, self.config["execution"]["expected_chunk_steps"])
        if self.config["execution"].get("mode") == "rtc":
            try:
                rtc_compatibility(metadata, self.config["execution"])
                return {"status": "compatible", "message": "RTC 能力已确认；启动时验证实际 H20 响应"}
            except Exception as exc:
                from .inference_rtc import VERSION
                if metadata.get('rtc_version') != VERSION:
                    return {'status': 'incompatible', 'message':
                            f"云端 RTC 协议不兼容：当前 {metadata.get('rtc_version') or '未声明'}，需要 {VERSION}；请更新云端推理服务",
                            'detail': str(exc)}
                return {"status": "incompatible", "message": str(exc)}
        return status

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
            LOGGER.info("Model Policy WebSocket %s", text)

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
        # Camera diagnostics can wait for capture locks/driver calls. Never
        # hold the inference state lock across that external operation.
        with self.lock:
            active = bool(self.state.get('execution_enabled') or self.state.get('priming'))
        shared = self._camera_ui_snapshot(active)
        with self.lock:
            self._refresh_connections_locked()
            result = copy.deepcopy(self.state); result["config"] = copy.deepcopy(self.config)
            result["camera_ready"] = shared["ready"]; result["frame_version"] = shared["frame_version"]
            result["camera"] = shared
            result["connections"] = copy.deepcopy(self._connections)
            result['measured_tcp_feedback'] = self._tcp_feed.status() if self._tcp_feed else None
            capture = self._capture or self._completed_capture
            result['inference_log'] = capture.status() if capture else None
            result['log_export_ready'] = bool(self._completed_capture and self._completed_capture.status()['export_ready'])
            result['policy_metadata'] = copy.deepcopy(getattr(self._policy,'metadata',{}))
            result['connection_switching'] = self._changing_connection
            preview = result['preview']
            preview['stale'] = bool(preview.get('stale') or not preview.get('updated_at') or time.time()-preview['updated_at'] > 1)
            result['chunk_stale'] = not result.get('chunk_updated_at') or time.time()-result['chunk_updated_at'] > 1
            if self.config['execution'].get('mode') == 'rtc' and result['model_state'] == 'CONNECTED':
                contract = self._metadata_status(result['policy_metadata'])
                if contract['status'] == 'incompatible':
                    result['protocol_compatibility'] = contract
            basic = (None if result['camera_ready'] and result['model_state'] == 'CONNECTED'
                     and result['protocol_compatibility']['status'] == 'compatible'
                     and result.get('last_response_steps') == self.config['execution']['expected_chunk_steps']
                     and not result['chunk_stale'] and not result.get('observation_wait_reason')
                     and result['preview'].get('state') != 'ERROR'
                     else (result['protocol_compatibility']['message'] if result['protocol_compatibility']['status'] == 'incompatible' else None)
                     or result.get('observation_wait_reason') or result['preview'].get('error') or '等待连接、有效观测和实际动作响应')
            control_busy = bool(result['execution_enabled'] or result['priming'] or self._changing_connection
                                or result.get('rtc', {}).get('state') in {'STOPPING','HOLD_PENDING','HOLD_UNCONFIRMED'})
            rtc_run = self._rtc_run
            control_active = bool(result['execution_enabled'] or result['priming'])
            old_run_pending = bool(rtc_run and not rtc_run.finished.is_set() and not control_active)
            stop = result.get('control_stop') or {}
            stop_unconfirmed = bool((rtc_run and not rtc_run.stop_confirmed and not control_active) or
                                    stop.get('state') in {'HOLD_PENDING', 'HOLD_UNCONFIRMED'})
            if stop_unconfirmed:
                basic = '停止尚未确认，请点击“重试停止”；只读预览可继续'
            elif old_run_pending:
                basic = '等待旧控制请求退出；不会自动恢复控制'
            elif result['preview'].get('control_valid') is False:
                basic = result['preview'].get('warning') or '当前预测不适合执行，预览继续'
            result['recovery'] = {'stop_required': stop_unconfirmed,
                                  'old_run_pending': old_run_pending,
                                  'can_stop': bool(stop_unconfirmed or result.get('last_error') or control_busy),
                                  'can_restart': not basic and not control_busy}
            training = bool(result['policy_metadata'].get('training_contract_verified'))
            rtc = self.config['execution'].get('mode') == 'rtc'
            result['control_readiness'] = {
                'shadow': {'allowed': not basic and not control_busy, 'reason': basic or ('控制正在运行或停止确认中' if control_busy else None)},
                'hardware': {'allowed': not basic and not control_busy,
                             'reason': basic or ('控制正在运行或停止确认中' if control_busy else None)}}
            result['training_notice'] = '第0行=t_obs已由用户确认；完整训练产物溯源尚未核验，此信息不阻止手动启动控制' if rtc and not training else None
            return result

    def _camera_ui_snapshot(self, active: bool) -> dict[str, Any]:
        resource = self.camera_resource
        if resource is None:
            return {"ready": self.cameras is not None, "frame_version": 0}
        if (not active and self._camera_status_snapshot is None) or not hasattr(resource,'frame_times_ns'):
            # Idle diagnostics retain the original full driver properties.
            result = resource.snapshot()
            self._camera_status_snapshot = copy.deepcopy(result)
            return result
        # During output, use existing frame/cache metadata only. The full
        # camera snapshot queries OpenCV properties while holding its cache
        # lock, which also blocks observation.read(); never do that here.
        result = copy.deepcopy(self._camera_status_snapshot or {})
        with resource.lock:
            ready = resource.cameras is not None
            frames = dict(resource.frame_times_ns)
            versions = resource.frame_version
            config = copy.deepcopy(resource.config)
            histories = {k:[x[0] for x in resource.dataset_frame_history.get(k,[])] for k in ('external','wrist')}
            sizes = dict(resource.dataset_frame_sizes)
            last_error = resource.last_error
        now = time.perf_counter_ns()
        result.update(ready=ready,state='READY' if ready else 'IDLE',frame_version=versions,config=config,last_error=last_error,
                      capture_properties_source='last_idle_diagnostics')
        sources = result.setdefault('sources',{})
        for key in ('external','wrist'):
            entry = sources.setdefault(key,{})
            times = histories[key]
            entry.update(available=ready and key in frames,frame_available=key in frames,
                timestamp_monotonic_ns=frames.get(key),
                last_frame_age_ms=(now-frames[key])/1e6 if key in frames else None,
                dataset_size=list(sizes[key]) if key in sizes else None,
                capture_hz=(len(times)-1)*1e9/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None)
        return result

    def _cached_feedback(self) -> dict[str, Any]:
        # Read only the already-published cache: no additional OSC/CAN requests.
        with self.lock:
            source = self._osc_snapshot
            selected = {k:source.get(k) for k in ('session','execution','command','gripper')}
            selected['last_cpv_dispatch'] = (source.get('transport') or {}).get('last_cpv_dispatch')
        return copy.deepcopy(selected)

    def _publish_logging_feedback(self, source: dict[str, Any]) -> None:
        selected = {k:source.get(k) for k in ('session','execution','command','gripper','gripper_observation')}
        selected['last_cpv_dispatch'] = (source.get('transport') or {}).get('last_cpv_dispatch')
        selected['cache_published_perf_ns'] = time.perf_counter_ns()
        self._log_feedback_snapshot = copy.deepcopy(selected)

    def _logging_feedback(self) -> dict[str, Any]:
        # Publication is replacement-only; keep one stable reference throughout
        # the copy. No adapter lock, RPC, CAN call or target mutation here.
        snapshot = self._log_feedback_snapshot
        result = copy.deepcopy(snapshot)
        published = snapshot.get('cache_published_perf_ns')
        result['logging_cache_age_ms'] = (time.perf_counter_ns()-published)/1e6 if published is not None else None
        return result

    def _event(self, event: str, **fields: Any) -> None:
        capture = self._capture
        if capture: capture.emit(event,**fields)

    def _finish_capture(self, reason: str) -> None:
        with self.lock:
            capture,self._capture = self._capture,None
        if capture:
            capture.finish(reason)
            with self.lock: self._completed_capture = capture

    def export_inference_log(self) -> tuple[str, bytes]:
        if self._completed_capture is None: raise RuntimeError('No completed inference run')
        return self._completed_capture.zip_bytes()

    def _change_connection(self, name: str, endpoint: Any = None) -> None:
        request = {"inference_profile": name}
        if endpoint is not None:
            request["policy_endpoint"] = endpoint
        self.update_config(request)

    def _apply_connection_config(self, candidate: dict[str, Any]) -> None:
        """Publish a fully validated saved preset, invalidating preview results."""
        with self.lock:
            if self.state.get("execution_enabled") or self.state.get("priming") or self._capture is not None:
                raise RuntimeError("请先停止推理输出，再修改连接或 H/K")
            if self._rtc_run and not self._rtc_run.stop_confirmed:
                raise RuntimeError("RTC HOLD is unconfirmed; confirm stop before changing configuration")
            save_config(self.config_path, candidate)
            self._changing_connection = True
            self._connection_revision += 1
            self._run_revision += 1
            previous, self._policy = self._policy, None
            self.stop_event.set()
            old_worker = self.worker
            self.config = candidate
            self._forward_listener = None
            self._listener_checked_at = 0.0
            self._tcp_evidence = {}
            self._connection_stage = "pending"
            self._last_connection_probe = 0.0
            self._connections = {}
            self.state.update(state="IDLE", model_state="DISCONNECTED", websocket_error=None,
                              protocol_compatibility={"status": "pending", "message": "待连接验证"},
                              action_chunk=None, absolute_tcp_chunk=None, action_chunk_length=0,
                              last_response_steps=None, last_error=None, last_rejection=None,
                              prompt=candidate["model"]["prompt"], updated_at=time.time())
        try:
            if previous:
                previous.close()
            if old_worker and old_worker is not threading.current_thread():
                old_worker.join(timeout=3)
            with self.lock:
                self._policy_retry_at = 0
                self._last_connection_probe = 0
        finally:
            with self.lock:
                self._changing_connection = False
            self._connection_wakeup.set()
        if not old_worker or not old_worker.is_alive():
            if self.camera_resource and self.camera_resource.snapshot().get("ready"):
                self._ensure_worker()
            elif self.cameras is not None:
                self._ensure_worker()

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
    def _local_forward_listening(port: int, host: str = "127.0.0.1") -> bool | None:
        return local_listener(host, port)

    def _connection_event(self, revision: int, stage: str, detail: str | None = None) -> None:
        with self.lock:
            if revision != self._connection_revision or self._changing_connection:
                return
            self._connection_stage = stage
            if stage in ("tcp_connected", "tcp_error"):
                self._tcp_evidence = {"available": stage == "tcp_connected", "checked_at": time.time(), "detail": detail}
            self._last_connection_probe = 0.0

    def _connection_loop(self) -> None:
        """Keep a lightweight policy WebSocket handshake independent of inference."""
        while not self._connection_stop.is_set():
            with self.lock:
                policy = self._policy
                model = dict(self.config["model"])
                worker_active = self.worker is not None and self.worker.is_alive()
                retry_at = self._policy_retry_at
                revision = self._connection_revision
                changing = self._changing_connection
            if changing:
                self._connection_wakeup.wait(.05); self._connection_wakeup.clear()
                continue
            # UI status consumes this cached listener result. netstat must not
            # execute under the lock shared with target dispatch/feedback.
            listening = self._local_forward_listening(int(model['port']), str(model['host']))
            with self.lock:
                if revision == self._connection_revision:
                    self._forward_listener = listening
                    self._listener_checked_at = time.time()
            # The synchronous WebSocket client serializes send/recv/ping.
            # Never issue a health ping concurrently with inference on the
            # same socket; inference failures are the active-session liveness
            # signal.  Idle sessions may still use ping to publish status.
            connected = policy is not None and (worker_active or policy.is_alive())
            if connected:
                with self.lock:
                    if revision == self._connection_revision and policy is self._policy:
                        # Existing live transport evidence, not a second probe.
                        self._tcp_evidence = {"available": True, "checked_at": time.time(), "detail": "现有 TCP 通道保持连接"}
            if connected and not worker_active:
                with self.lock:
                    resume_preview = self.state.get("state") == "IDLE" and not self._changing_connection
                if resume_preview and ((self.camera_resource and self.camera_resource.snapshot().get("ready")) or self.cameras is not None):
                    self._ensure_worker()
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
                            self._connection_event(revision, "connecting")
                            policy = OpenPIClient(str(model["host"]), int(model["port"]), float(model["request_timeout_s"]),
                                                  lambda stage, detail=None: self._connection_event(revision, stage, detail))
                            with self.lock:
                                if self._connection_stop.is_set() or self._changing_connection or revision != self._connection_revision:
                                    policy.close()
                                else:
                                    self.state["protocol_compatibility"] = self._metadata_status(getattr(policy, "metadata", {}))
                                    self._policy = policy
                                    self._policy_generation += 1
                                    self.state.update({"model_state": "CONNECTED", "websocket_error": None, "updated_at": time.time()})
                                    self._connection_stage = "metadata_received"
                                    self._record_websocket_event("connect_success")
                            if self.camera_resource and self.camera_resource.snapshot().get("ready"):
                                self._ensure_worker()
                        except Exception as exc:
                            with self.lock:
                                if revision != self._connection_revision: continue
                            self._record_websocket_event("connect_failure", f"{type(exc).__name__}: {exc}")
                            with self.lock:
                                self.state.update({"model_state": "DISCONNECTED", "websocket_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                                self._connection_stage = "failed"
            self._connection_wakeup.wait(1.0)
            self._connection_wakeup.clear()

    def _invalidate_policy(self, policy: OpenPIClient | None, reason: str) -> None:
        with self.lock:
            if policy is not None and self._policy is not policy:
                return
            self._policy = None
            self._connection_stage = "failed"
            self._tcp_evidence = {}
            self._last_connection_probe = 0.0
            self.state["protocol_compatibility"] = {"status": "pending", "message": "待重连验证"}
            self._policy_retry_at = time.monotonic() + 2.0
            self.state.update({"model_state": "DISCONNECTED", "websocket_error": reason, "updated_at": time.time()})
            self._record_websocket_event("disconnect", reason)
        if policy is not None:
            policy.close()

    def _feedback_context(self):
        with self.lock:
            source = self._osc_snapshot
            context = {'session': copy.deepcopy(source.get('session') or {})}
            if context['session'].get('execution_mode') == 'shadow':
                context['execution'] = copy.deepcopy(source.get('execution') or {})
                context['diagnostics'] = {'shadow_transport': copy.deepcopy(
                    (source.get('diagnostics') or {}).get('shadow_transport') or {})}
            return context

    def _inference_snapshot(self):
        with self.lock:
            snapshot = copy.deepcopy(self._osc_snapshot)
        return self._tcp_feed.attach(snapshot) if self._tcp_feed else snapshot

    def _osc_state_loop(self) -> None:
        """Refresh OSC feedback independently from inference and UI state reads."""
        period_s = max(0.02, float((self.config.get("execution") or {}).get("osc_state_poll_s", 0.05)))
        deadline = time.perf_counter()
        previous_poll_ns = None
        while not self._osc_state_stop.is_set():
            poll_ns = time.perf_counter_ns()
            try:
                # Feedback refresh only needs the session and the latest
                # measured execution sample. Full osc.state() includes
                # solver/transport diagnostics and can block target ingress.
                fast_state = getattr(self.osc, "fast_state", None)
                snapshot = fast_state() if callable(fast_state) else self.osc.state()
                returned_ns = time.perf_counter_ns()
                try:
                    if self._tcp_feed:
                        snapshot = self._tcp_feed.attach(snapshot)
                    else:
                        snapshot = self._observation_source.refresh(snapshot)
                    with self.lock:
                        self.state['observation_wait_reason'] = None
                except Exception as observation_error:
                    with self.lock:
                        self.state['observation_wait_reason'] = str(observation_error)
                        prior = self._osc_snapshot
                        identity = lambda s: tuple((s.get('session') or {}).get(k) for k in ('id','client_id','execution_mode'))
                        measured = prior.get('inference_observation') or {}
                        stamp = measured.get('perf_ns')
                        if (identity(prior) == identity(snapshot) and isinstance(stamp,int)
                                and 0 <= time.perf_counter_ns()-stamp <= 100_000_000):
                            # Retain only the timestamped real measurement.
                            # Never refresh its age or replace it with HOLD's
                            # frozen execution sample after one failed poll.
                            snapshot['inference_observation'] = copy.deepcopy(measured)
                    snapshot['inference_observation_error'] = str(observation_error)
                snapshot = self._refresh_gripper_observation(snapshot)
                publish_ns = time.perf_counter_ns()
                if not self._tcp_feed:
                    snapshot['inference_cache_perf_ns'] = publish_ns
                observed = snapshot.get('inference_observation') or {}
                observation_timing = dict(observed.get('timing') or {})
                if observation_timing:
                    observation_timing.update(published_perf_ns=publish_ns,
                        sample_age_at_publish_ms=(publish_ns-observed['perf_ns'])/1e6)
                    observed['timing'] = observation_timing
                self._publish_logging_feedback(snapshot)
                published = copy.deepcopy(snapshot)
                lock_started_ns = time.perf_counter_ns()
                with self.lock:
                    lock_acquired_ns = time.perf_counter_ns()
                    self._osc_snapshot = published
                    self.state.update({"osc_error": None, "updated_at": time.time()})
                    self.state['gripper_observation'] = copy.deepcopy(snapshot.get('gripper_observation'))
                self._observe_inference_gripper(snapshot)
                self._event('feedback_cache_poll',rpc_ms=(returned_ns-poll_ns)/1e6,
                    publish_lock_wait_ms=(lock_acquired_ns-lock_started_ns)/1e6,
                    poll_started_perf_ns=poll_ns,poll_returned_perf_ns=returned_ns,
                    observation_timing=observation_timing,
                    observation_error=self.state.get('observation_wait_reason'),
                    poll_interval_ms=(poll_ns-previous_poll_ns)/1e6 if previous_poll_ns is not None else None)
            except Exception as exc:
                with self.lock:
                    self.state.update({"osc_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
            previous_poll_ns = poll_ns
            deadline += period_s
            now = time.perf_counter()
            if deadline < now:
                deadline = now
            self._osc_state_stop.wait(max(0.,deadline-time.perf_counter()))

    def _refresh_gripper_observation(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        session = snapshot.get('session') or {}
        if session.get('execution_mode') == 'shadow':
            return snapshot
        feed = getattr(self, '_gripper_feed', None)
        if feed is not None:
            # A pure cache read: no SDK/RPC call on the UI/state polling thread.
            return feed.attach(snapshot)
        reader = getattr(self.osc,'inference_gripper_feedback',None)
        if not callable(reader): return snapshot
        with self.lock:
            if not hasattr(self, '_measured_gripper_cache'):
                self._measured_gripper_cache = GripperFeedbackCache()
            cache = self._measured_gripper_cache
        sample, diagnostic = cache.read(reader, session.get('id'))
        result = dict(snapshot)
        if sample is not None:
            result['gripper'] = sample
        result['gripper_observation'] = diagnostic
        return result

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
        raise RuntimeError("Model Policy WebSocket is not connected")

    def _ensure_worker(self) -> None:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            self.stop_event = threading.Event()
            self.state.update({"state": "RUNNING", "updated_at": time.time()})
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
        policy_connected = self.state.get("model_state") == "CONNECTED"
        now = time.time()
        profile = self.config["inference_profiles"]["active"]
        local = loopback(host)
        mode = "AutoDL 本机转发入口" if profile == "autodl" and local else ("5090 本地直连" if profile == "local5090" else "AutoDL 远端直连")
        listener_fresh = now - self._listener_checked_at < 10
        tcp = self._tcp_evidence
        tcp_fresh = bool(tcp) and now - tcp["checked_at"] < 10
        entry = {"label": "服务入口", "subtitle": mode, "endpoint": f"{host}:{port}",
                 "state": "neutral", "basis": "pending", "checked_at": None, "message": "待检测服务入口"}
        if local and listener_fresh and self._forward_listener is True:
            entry.update(state="ok", basis="local_listener", checked_at=self._listener_checked_at,
                         message="本机转发端口已监听；远端模型连接见右侧" if profile == "autodl" else "配置地址与端口已有服务监听")
        elif tcp_fresh:
            entry.update(state="ok" if tcp["available"] else "bad", basis="tcp_connection", checked_at=tcp["checked_at"],
                         message=(tcp.get("detail") or "配置地址与端口 TCP 可达") if tcp["available"] else (tcp.get("detail") or "服务入口不可达"))
        elif local and listener_fresh:
            entry.update(state="bad" if self._forward_listener is False else "neutral", basis="local_listener",
                         checked_at=self._listener_checked_at,
                         message="未发现匹配地址的监听；仍会尝试实际连接" if self._forward_listener is False else "监听检测未知；等待实际连接结果")
        policy_state = "ok" if policy_connected else ("warn" if self._connection_stage in ("connecting", "tcp_connected", "websocket_connected") or self._changing_connection else ("bad" if self.state.get("websocket_error") else "neutral"))
        # Do not synchronously query OSC while serving pi05 state. A stalled
        # control backend must not block inference telemetry or Action Chunks.
        transport = (self._osc_snapshot.get("transport") or {}) if self._osc_snapshot else {}
        osc_error = self.state.get("osc_error")
        self._connections = {
            "control": {"state": "bad" if osc_error else ("ok" if transport.get("connected") else "warn"), "label": "NERO control service", "endpoint": "127.0.0.1:8765", "message": str(osc_error) if osc_error else "OSC control channel available"},
            "service_entry": entry,
            "ssh_forward": copy.deepcopy(entry),
            "policy": {"state": policy_state, "label": "模型 WebSocket", "subtitle": "推理协议连接", "stage": self._connection_stage,
                       "endpoint": endpoint_uri(host, port),
                       "message": "握手成功，已收到 OpenPI metadata" if policy_connected else ("正在切换连接" if self._changing_connection else ("正在连接推理服务" if policy_state == "warn" else (str(self.state.get("websocket_error")) if self.state.get("websocket_error") else "等待连接推理服务")))},
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
        with self._switch_lock:
            with self.lock:
                candidate = prepare_update(self.config, value)
                execution = value.get("execution") or {}
                rtc_active = self.config["execution"].get("mode") == "rtc" and (self.state.get("execution_enabled") or self.state.get("priming"))
                if rtc_active:
                    raise RuntimeError("Stop RTC output before changing model settings")
                switch = ("inference_profile" in value or "policy_endpoint" in value or "model" in value
                          or "expected_chunk_steps" in execution or "replan_steps" in execution
                          or "mode" in execution or "inference_delay" in execution or "execute_horizon" in execution)
                cameras = value.get("cameras")
                if cameras is not None:
                    if not isinstance(cameras, dict):
                        raise ValueError("Cameras must be an object")
                    if self.state.get("execution_enabled") or self.state.get("priming"):
                        raise RuntimeError("Stop inference output before changing cameras")
                    for key in ("external", "wrist"):
                        if key in cameras:
                            item = cameras[key]
                            index = item.get("index") if isinstance(item, dict) else None
                            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 32:
                                raise ValueError("Camera index must be 0-32")
                            candidate["cameras"][key]["index"] = index
                    if candidate["cameras"]["external"]["index"] == candidate["cameras"]["wrist"]["index"]:
                        raise ValueError("external and wrist cameras must be different")
                if cameras is not None and switch:
                    raise ValueError("Update camera selection separately from connection settings")
            if switch:
                self._apply_connection_config(candidate)
            else:
                with self.lock:
                    save_config(self.config_path, candidate)
                    if cameras is not None and self.camera_resource:
                        self.camera_resource.update_config(cameras)
                    self.config = candidate
                    self.state["prompt"] = candidate["model"]["prompt"]
                    self.state["updated_at"] = time.time()
        return self.snapshot()

    def _persist_config_locked(self) -> None:
        save_config(self.config_path, self.config)

    def activate_cameras(self) -> dict[str, Any]:
        if self.camera_resource:
            self.camera_resource.activate()
            with self.lock: self.state.update({"camera_state": "READY", "last_error": None, "updated_at": time.time()})
            self._ensure_worker()
            return self.snapshot()
        with self.lock:
            if self.state["state"] == "RUNNING": raise RuntimeError("cannot change cameras while Model inference is running")
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
        if (self.state.get('control_stop') or {}).get('state') in {'HOLD_PENDING', 'HOLD_UNCONFIRMED'}:
            raise RuntimeError('Previous control HOLD is unconfirmed; retry stop first')
        previous_rtc = self._rtc_run
        if previous_rtc and (not previous_rtc.finished.is_set() or not previous_rtc.stop_confirmed):
            raise RuntimeError("Previous RTC run is still exiting or HOLD is unconfirmed")
        previous_worker = self.worker
        if self.stop_event.is_set() and previous_worker and previous_worker.is_alive():
            previous_worker.join(timeout=1)
            if previous_worker.is_alive(): raise RuntimeError('Previous inference is still exiting')
        with self.lock:
            if self._changing_connection: raise RuntimeError('Policy connection is switching')
            metadata_status = self._metadata_status(getattr(self._policy, "metadata", {}))
            if metadata_status["status"] == "incompatible":
                self.state["protocol_compatibility"] = metadata_status
                raise RuntimeError(metadata_status["message"])
            if self.state.get("protocol_compatibility", {}).get("status") == "incompatible":
                raise RuntimeError(self.state["protocol_compatibility"]["message"])
            if self._capture is not None: raise RuntimeError('Inference output already started')
            startup_revision = self._run_revision
        if not self._camera_ui_snapshot(True)['ready']:
            raise RuntimeError("activate the external and wrist cameras first")
        try:
            osc = self.osc.state()
        except Exception as exc:
            with self.lock:
                self.state["osc_error"] = f"{type(exc).__name__}: {exc}"
                self.state["updated_at"] = time.time()
            raise
        published = copy.deepcopy(osc)
        with self.lock:
            if self._changing_connection or startup_revision != self._run_revision:
                raise RuntimeError('Inference startup cancelled or connection changed')
            if (self.state.get('control_stop') or {}).get('state') in {'HOLD_PENDING', 'HOLD_UNCONFIRMED'}:
                raise RuntimeError('Previous control HOLD is unconfirmed; retry stop first')
            if self._capture is not None: raise RuntimeError('Inference output already started')
            self._osc_snapshot = published
            self._publish_logging_feedback(osc)
            session = osc.get("session") or {}
            if session.get("state") != "ACTIVE" or session.get("id") != session_id or session.get("client_id") != client_id:
                raise PermissionError("Model inference requires the caller's active OSC session")
            rtc = self.config["execution"].get("mode") == "rtc"
            self._cancel_queued_gripper('new control run')
            self._last_gripper_target = None
            self._run_revision += 1
            self._wait_cancel = threading.Event()
            self.state['log_error'] = None
            try:
                self._capture = RunCapture(self.config_path.parent.parent/'runtime/logs/inference_runs',
                    {'session':session,'config':copy.deepcopy(self.config)},self._logging_feedback)
            except Exception as exc: self.state['log_error'] = f'{type(exc).__name__}: {exc}'
            needs_priming = rtc or session.get("execution_mode") == "hardware"
            if self.state.get('last_error'):
                self.state['last_control_error'] = self.state['last_error']
            self.state.update({"state": "PRIMING" if needs_priming else "RUNNING",
                               "execution_enabled": not needs_priming, "priming": needs_priming,
                               "session_id": session_id, "client_id": client_id,
                               "execution_mode": session.get("execution_mode"), "last_error": None, "updated_at": time.time()})
            self.state["osc_error"] = None
            self.state['preview'].update(state='PAUSED', stale=True)
            if self._preview_busy and self._policy is not None:
                self._invalidate_policy(self._policy, 'Cancel preview; acquire fresh control observation')
            self._ensure_worker()
            self._connection_wakeup.set()
            if needs_priming and not rtc:
                threading.Thread(target=self._prime_control_worker, args=(session_id, client_id, self._run_revision), name="nero-pi05-osc-prime", daemon=True).start()
        return self.snapshot()

    def _prime_control_worker(self, session_id: str, client_id: str, revision: int) -> None:
        """Run hardware priming independently from policy inference."""
        try:
            deadline = time.monotonic() + float(self.config["model"]["request_timeout_s"])
            while True:
                with self.lock:
                    if not self.state.get("priming") or revision != self._run_revision:
                        return
                    compatibility = self.state.get("protocol_compatibility", {})
                if compatibility.get("status") == "compatible":
                    break
                if compatibility.get("status") == "incompatible" or time.monotonic() >= deadline:
                    raise RuntimeError(compatibility.get("message", "Model response validation timed out"))
                if self.stop_event.wait(.02):
                    return
            self._prime_startup_hold(session_id, client_id, revision)
        except Exception as exc:
            with self.lock:
                if revision != self._run_revision:
                    return
                self.state.update(state="ERROR", priming=False, execution_enabled=False,
                                  last_error=f"{type(exc).__name__}: {exc}",
                                  osc_error=f"{type(exc).__name__}: {exc}", updated_at=time.time())
            self._finish_capture("inference startup validation failed")

    def stop(self, reason: str = "Model inference stopped") -> dict[str, Any]:
        with self.lock:
            was_active = bool(self.state.get('execution_enabled') or self.state.get('priming'))
            session_id, client_id = self.state.get('session_id'), self.state.get('client_id')
            self._run_revision += 1
            self._wait_cancel.set()
            self._cancel_pending_gripper(reason)
            self.state.update({"state": "RUNNING" if self.state.get("state") == "PRIMING" else self.state.get("state"),
                               "execution_enabled": False, "priming": False, "updated_at": time.time()})
            rtc_run = self._rtc_run
        self._cancel_queued_gripper(reason)
        if rtc_run and not rtc_run.stop_confirmed:
            with self.lock:
                self.state.setdefault("rtc", {}).update(state="STOPPING", hold_confirmed=False)
            rtc_run.cancel.set()
            self._invalidate_policy(rtc_run.policy, "RTC stopped; discard in-flight response")
            try:
                rtc_run.stop(reason)
                with self.lock:
                    self.state.setdefault("rtc", {}).update(state="RELEASED" if rtc_run.authority_released else "HOLD",
                        hold_confirmed=rtc_run.hold_confirmed, authority_released=rtc_run.authority_released, manual_restart_required=True)
            except Exception as exc:
                with self.lock:
                    self.state.setdefault("rtc", {}).update(state="HOLD_UNCONFIRMED", hold_confirmed=False)
                    self.state["last_error"] = f"RTC stop unconfirmed: {exc}"
                raise
        retry_hold = (self.state.get('control_stop') or {}).get('state') in {'HOLD_PENDING', 'HOLD_UNCONFIRMED'}
        if (was_active or retry_hold) and (rtc_run is None or self.config['execution'].get('mode') != 'rtc'):
            self._confirm_nonrtc_hold(session_id, client_id, reason)
        with self._row_lock: self._finish_capture(reason)
        with self.lock:
            if self.state.get('last_error'):
                self.state['last_control_error'] = self.state['last_error']
            self.state.update(state='RUNNING', last_error=None, last_rejection=None)
        if not self._closing and not self._connection_stop.is_set():
            self._ensure_worker()
        return self.snapshot()

    def _confirm_nonrtc_hold(self, session_id, client_id, reason):
        with self.lock:
            self.state['control_stop'] = {'state': 'HOLD_PENDING', 'reason': reason}
        try:
            with self._row_lock:
                current = self.osc.state().get('session') or {}
                owned = (current.get('id'), current.get('client_id')) == (session_id, client_id) and session_id is not None
                if owned:
                    result = self.osc.hold(session_id, client_id, self._next_sequence(), reason)
                    if not result.get('ok'):
                        raise RuntimeError('Control HOLD not confirmed')
            with self.lock:
                self.state['control_stop'] = {'state': 'HOLD' if owned else 'RELEASED', 'reason': reason,
                                              'hold_confirmed': owned, 'authority_released': not owned}
        except Exception as exc:
            with self.lock:
                self.state['control_stop'] = {'state': 'HOLD_UNCONFIRMED', 'reason': str(exc), 'hold_confirmed': False}
            raise

    def _prime_startup_hold(self, session_id: str, client_id: str, revision: int | None = None) -> bool:
        """Send one zero-motion target before releasing Model actions."""
        osc = self.osc.state()
        with self.lock:
            if not self.state.get("priming") or (revision is not None and revision != self._run_revision):
                return False
        execution = osc.get("execution") or {}
        command = osc.get("command") or {}
        target = execution.get("measured_tcp_pose") or command.get("target_tcp")
        if not isinstance(target, dict):
            raise RuntimeError("Model startup hold requires a current measured TCP pose")
        position = target.get("position_m")
        orientation = target.get("orientation_xyzw")
        if not isinstance(position, list) or not isinstance(orientation, list):
            raise RuntimeError("Model startup hold received an invalid TCP pose")
        sequence = self._next_sequence()
        result = self.osc.track_tcp(session_id, client_id, sequence, {
            "position_m": list(position), "orientation_xyzw": list(orientation),
        })
        if not result.get("ok"):
            raise RuntimeError(f"Model startup hold rejected: {result}")
        hold_s = max(0.0, float(self.config["execution"].get("startup_hold_s", 0.3)))
        if self.stop_event.wait(hold_s):
            return False
        with self.lock:
            if not self.state.get("priming") or (revision is not None and revision != self._run_revision):
                return False
            self.state.update({"state": "RUNNING", "execution_enabled": True,
                               "priming": False, "updated_at": time.time()})
        return True

    @staticmethod
    def _feedback_pose(osc: dict[str, Any]) -> dict[str, Any] | None:
        """Pose represented by the state frame captured for one inference."""
        observed = osc.get('inference_observation')
        pose = observed['pose'] if observed else (osc.get("execution") or {}).get("measured_tcp_pose")
        if not isinstance(pose, dict):
            pose = (osc.get("command") or {}).get("target_tcp")
        return copy.deepcopy(pose) if isinstance(pose, dict) else None

    def _observation(self, osc: dict[str, Any], external: Any, wrist: Any) -> dict[str, Any]:
        observed = osc.get('inference_observation')
        if observed and not 0 <= time.perf_counter_ns()-observed['perf_ns'] <= 100_000_000:
            raise ObservationNotReady('waiting for fresh measured TCP')
        pose = self._feedback_pose(osc)
        # Before an OSC session is opened there may be no published target
        # pose yet. Inference is still useful in preview mode; execution will
        # only be enabled after a session has supplied a real measured pose.
        if not isinstance(pose, dict):
            raise ObservationNotReady('waiting for measured TCP pose')
        position = _finite(pose.get("position_m"), 3, "TCP position")
        # The AutoDL training state is expressed directly in the NERO base
        # frame.  Do not apply LIBERO axis flips or arbitrary normalisation.
        rotvec = _quaternion_rotvec(_finite(pose.get("orientation_xyzw"), 4, "TCP orientation"))
        gripper = (observed or {}).get('gripper_width_m', (osc.get("gripper") or {}).get("width_m"))
        if gripper is None:
            raise ObservationNotReady('waiting for measured gripper opening')
        minimum = float(self.config["gripper"]["min_width_m"])
        maximum = float(self.config["gripper"]["max_width_m"])
        width = max(minimum, min(maximum, float(gripper)))
        ratio = (width - minimum) / (maximum - minimum)
        return make_observation(position, rotvec, ratio, external, wrist, self.config["model"]["prompt"])

    def _normalise_gripper_ratio(self, ratio: float, action_index: int | None = None) -> float:
        """Project finite model predictions onto the physical command domain.

        Raw predictions remain in the model-chunk log. Only gripper commands
        are saturated; TCP, OSC and hardware checks are not changed.
        """
        if not math.isfinite(ratio):
            raise RuntimeError(f'Model gripper ratio at row {action_index} must be finite')
        if 0.0 <= ratio <= 1.0:
            return ratio
        command_ratio = min(1.0, max(0.0, ratio))
        diagnostic = {"status": "boundary_normalized", "mode": "finite_saturation",
                      "action_index": action_index, "raw_ratio": ratio,
                      "command_ratio": command_ratio, "command_min": 0.0,
                      "command_max": 1.0, "updated_at": time.time()}
        with self.lock:
            self.state["gripper_ratio_diagnostic"] = diagnostic
        self._event('gripper_ratio_saturated', **diagnostic)
        return command_ratio

    def _target_from_action(self, action: Any, base: dict[str, Any],
                            action_index: int | None = None, *, preview_only: bool = False) -> tuple[dict[str, Any], float]:
        values = _finite(action, 7, "AutoDL Nero action")
        gripper_ratio = min(1., max(0., values[6])) if preview_only else self._normalise_gripper_ratio(values[6], action_index)
        delta, rotvec = values[:3], values[3:6]
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
            position = _finite(target.get("position_m"), 3, f"Model target {index} position")
            if any(value < low or value > high for value, low, high in zip(position, lower, upper)) or position[2] < floor:
                raise RuntimeError(f"Model target {index} is outside the OSC workspace")

    def _send_gripper_if_needed(self, session_id: str, client_id: str, action_value: float) -> None:
        """Queue the trained gripper target without blocking TCP dispatch.

        The physical command is submitted by ``_gripper_loop``.  This method
        is called while the RTC row lock is held, so it must remain a bounded
        bookkeeping operation and never wait for the SDK/CAN response.
        """
        minimum, maximum = float(self.config["gripper"]["min_width_m"]), float(self.config["gripper"]["max_width_m"])
        target_width = minimum + float(action_value) * (maximum - minimum)
        tolerance = float(self.config["gripper"].get("dedupe_tolerance_ratio", 0.01)) * (maximum - minimum)
        with self.lock:
            shadow = self.state.get('execution_mode') == 'shadow'
            revision = self._run_revision
            last_target = self._last_gripper_target
        if isinstance(last_target, (int, float)) and abs(float(last_target) - target_width) <= tolerance:
            return

        if shadow:
            # Shadow mode intentionally preserves its immediate, simulated
            # confirmation and never starts a physical SDK call.
            sequence = self._next_sequence()
            started_ns = time.perf_counter_ns()
            self._event('gripper_request', sequence=sequence, width_m=target_width,
                        opening_ratio=action_value, mode='simulated')
            confirmation = {'sequence':sequence,'target_width_m':target_width,'sent_perf_ns':started_ns,
                'status':'simulated','health_confirmed':False,'target_reached':None,
                'session_id':session_id,'client_id':client_id,'run_revision':revision,
                'last_sdk_marker':None}
            with self.lock:
                if revision != self._run_revision or not self.state.get('execution_enabled'):
                    return
                self._last_gripper_target = target_width
                self.state['gripper_confirmation'] = copy.deepcopy(confirmation)
            return

        job = {'session_id': session_id, 'client_id': client_id,
               'action_value': float(action_value), 'target_width_m': target_width,
               'revision': revision, 'queued_perf_ns': time.perf_counter_ns()}
        with self._gripper_condition:
            queued = self._gripper_job
            inflight = self._gripper_inflight_target
            if ((queued and abs(float(queued['target_width_m']) - target_width) <= tolerance)
                    or (inflight is not None and abs(inflight - target_width) <= tolerance)):
                return
            self._gripper_job = job
            self._gripper_condition.notify()
        if queued:
            self._event('gripper_coalesced', old_width_m=queued['target_width_m'],
                        new_width_m=target_width, reason='newer target replaced unsent command')
        with self.lock:
            transport = self.state.setdefault('gripper_transport', {})
            transport.update({'state': 'queued', 'error': None, 'updated_at': time.time(),
                              'target_width_m': target_width})

    def _cancel_queued_gripper(self, reason: str) -> None:
        """Drop an unsent command; an SDK call already in progress is stale."""
        with self._gripper_condition:
            job = self._gripper_job
            self._gripper_job = None
            self._gripper_condition.notify()
        if job:
            self._event('gripper_queue_cancelled', width_m=job['target_width_m'], reason=reason)

    def _gripper_loop(self) -> None:
        """Submit gripper commands independently of the TCP/RTC scheduler."""
        while not self._gripper_stop.is_set():
            with self._gripper_condition:
                while self._gripper_job is None and not self._gripper_stop.is_set():
                    self._gripper_condition.wait(.1)
                if self._gripper_stop.is_set():
                    return
                job = self._gripper_job
                self._gripper_job = None
                self._gripper_inflight_target = float(job['target_width_m']) if job else None
            if job is not None:
                self._dispatch_gripper(job)
            with self._gripper_condition:
                self._gripper_inflight_target = None

    def _dispatch_gripper(self, job: dict[str, Any]) -> None:
        session_id, client_id = job['session_id'], job['client_id']
        with self.lock:
            live = (job['revision'] == self._run_revision and
                    bool(self.state.get('execution_enabled')) and
                    self.state.get('session_id') == session_id and
                    self.state.get('client_id') == client_id)
        if not live:
            self._event('gripper_queue_cancelled', width_m=job['target_width_m'], reason='run changed before send')
            return
        sequence = self._next_sequence()
        started_ns = time.perf_counter_ns()
        self._event('gripper_request', sequence=sequence, width_m=job['target_width_m'],
                    opening_ratio=job['action_value'], mode='position_inference')
        try:
            result = self.osc.gripper(session_id, client_id, sequence, {
                "mode": "position_inference", "width_m": job['target_width_m'],
                "force_n": self.config["gripper"]["force_n"],
            })
        except Exception as exc:
            result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        duration_ms = (time.perf_counter_ns() - started_ns) / 1e6
        self._event('gripper_return', sequence=sequence, width_m=job['target_width_m'],
                    result=result, started_perf_ns=started_ns, duration_ms=duration_ms)
        ok = bool(result.get('ok'))
        receipt = result.get('result') or {}
        if ok and not receipt.get('sent'):
            ok = False
        with self.lock:
            still_live = (job['revision'] == self._run_revision and
                           bool(self.state.get('execution_enabled')))
            transport = self.state.setdefault('gripper_transport', {})
            transport.update({'state': 'sent' if ok else 'rejected',
                              'error': None if ok else ('command rejected' if result.get('ok') else result.get('error')),
                              'updated_at': time.time(), 'duration_ms': duration_ms,
                              'target_width_m': job['target_width_m']})
        if not still_live:
            return
        if not ok:
            reason = f"Model gripper command rejected: {result}" if not result.get('ok') \
                     else f"Model gripper was not sent: {result}"
            # RTC owns its own HOLD/release state machine.  Route an
            # asynchronous submission failure through it instead of the
            # ordinary chunk rejection path, otherwise the TCP run could be
            # marked ERROR while the RTC card incorrectly remains RUNNING.
            rtc_run = self._rtc_run
            if (self.config['execution'].get('mode') == 'rtc' and rtc_run is not None
                    and rtc_run.epoch == job['revision']):
                with self.lock:
                    self.state['last_rejection'] = reason
                rtc_run.fail(reason)
                return
            self._reject_chunk(session_id, client_id, reason,
                               expected_revision=(self._connection_revision, job['revision']))
            return
        with self.lock:
            if job['revision'] != self._run_revision or not self.state.get('execution_enabled'):
                return
            self._cancel_pending_gripper('replaced by new target')
            self._last_gripper_target = job['target_width_m']
            confirmation = {'sequence':sequence,'target_width_m':job['target_width_m'],
                'sent_perf_ns':started_ns,'status':'pending_feedback','health_confirmed':False,
                'target_reached':None,'session_id':session_id,'client_id':client_id,
                'run_revision':job['revision'],
                'reported_in_tolerance_samples':0, 'reported_target_reached':False,
                'physical_opening_verified':False, 'arrival_basis':'sdk_reported_width',
                'last_sdk_marker':self._gripper_sdk_marker(receipt.get('before') or {})}
            self.state['gripper_confirmation'] = copy.deepcopy(confirmation)
            self._pending_gripper = confirmation

    @staticmethod
    def _gripper_sdk_marker(gripper: dict[str, Any]) -> Any:
        # The outer gripper timestamp is generated on every read. Only the
        # timestamp on the SDK's received CAN message proves a new sample.
        raw = gripper.get('raw') or {}
        value = raw.get('value') if isinstance(raw,dict) else None
        marker = value.get('timestamp') if isinstance(value,dict) else None
        return marker if isinstance(marker,(int,float)) and math.isfinite(marker) and marker>0 else None

    def _cancel_pending_gripper(self, reason: str) -> None:
        pending,self._pending_gripper = self._pending_gripper,None
        if pending:
            self._event('gripper_feedback_cancelled',sequence=pending['sequence'],reason=reason)
            self.state['gripper_confirmation'] = {**pending,'status':'cancelled','reason':reason}

    def _observe_inference_gripper(self, snapshot: dict[str, Any]) -> None:
        fault = None; revision = None
        with self.lock:
            pending = self._pending_gripper
            if not pending: return
            session = snapshot.get('session') or {}
            if (not self.state.get('execution_enabled') or pending['run_revision'] != self._run_revision
                or session.get('state') != 'ACTIVE' or session.get('id') != pending['session_id']
                or session.get('client_id') != pending['client_id']):
                self._cancel_pending_gripper('output or session changed'); return
            gripper = snapshot.get('gripper') or {}
            marker = self._gripper_sdk_marker(gripper)
            previous = pending.get('last_sdk_marker')
            if marker is None or (previous is not None and marker<=previous): return
            status = gripper.get('status') or {}
            foc = status.get('foc_status') if isinstance(status,dict) else None
            if not isinstance(foc,dict): return
            faults = [k for k in ('voltage_too_low','motor_overheating','driver_overcurrent','driver_overheating','driver_error_status') if foc.get(k)]
            if not foc.get('driver_enable_status'): faults.append('driver_disabled')
            width = gripper.get('width_m')
            width_error = abs(float(width)-pending['target_width_m']) if isinstance(width,(int,float)) and math.isfinite(width) else None
            count = pending.get('reported_in_tolerance_samples',0)+1 if width_error is not None and width_error <= .003 else 0
            pending.update(last_sdk_marker=marker,status='feedback_fault' if faults else 'feedback_received',
                health_confirmed=not faults,actual_width_m=width,target_error_m=width_error,
                sdk_reported_width_m=width, reported_in_tolerance_samples=count,
                reported_target_reached=count>=2, physical_opening_verified=False,
                feedback_after_command_ms=(time.perf_counter_ns()-pending['sent_perf_ns'])/1e6,
                # Width error is diagnostic, not a new reached/grasp gate.
                target_reached=None,faults=faults,feedback_observed_perf_ns=time.perf_counter_ns())
            self.state['gripper_confirmation'] = copy.deepcopy(pending)
            self._event('gripper_feedback',**pending)
            if faults:
                fault = 'Model gripper feedback fault: '+', '.join(faults)
                revision = pending['run_revision']
                self.state['execution_enabled'] = False
                self._pending_gripper = None
        if fault:
            with self._row_lock:
                with self.lock:
                    if revision != self._run_revision: return
                self._reject_chunk(pending['session_id'],pending['client_id'],fault)

    def _reject_chunk(self, session_id: str | None, client_id: str | None, reason: str,
                      expected_revision: tuple[int, int] | None = None,
                      protocol_error: str | None = None) -> None:
        """Fail closed: no later row from a bad Model chunk may execute."""
        with self.lock:
            if expected_revision is not None and expected_revision != (self._connection_revision, self._run_revision):
                return
            self._event('execution_error',reason=reason)
            sequence = self._next_sequence()
            if protocol_error is not None:
                self.state["protocol_compatibility"] = {"status": "incompatible", "message": protocol_error}
            self.state.update({"state": "ERROR", "execution_enabled": False,
                               "last_error": reason, "last_rejection": reason,
                               "updated_at": time.time()})
            self._cancel_pending_gripper(reason)
            self.state['priming'] = False
            self._run_revision += 1
            self._wait_cancel.set()
        if session_id and client_id:
            try:
                self._confirm_nonrtc_hold(session_id, client_id, reason)
            except Exception as exc:
                with self.lock:
                    self.state['stop_error'] = str(exc)
        self._finish_capture(reason)

    def _run(self) -> None:
        if self.config["execution"].get("mode") == "rtc":
            run_rtc_adapter(self)
            return
        # Control failures end one run, not the read-only prediction worker.
        lifetime = self.stop_event
        while not lifetime.is_set():
            self._run_normal()
            lifetime.wait(.05)

    def _run_normal(self) -> None:
        policy = None
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    active = bool(self.state.get('execution_enabled') or self.state.get('priming'))
                if not active:
                    preview_once(self)
                    policy = None
                    continue
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
                with self.lock:
                    osc_source = self._osc_snapshot
                    session_id, client_id = self.state["session_id"], self.state["client_id"]
                    execution_enabled = bool(self.state.get("execution_enabled"))
                    priming_requested = bool(self.state.get('priming'))
                    revision,run_revision = self._connection_revision,self._run_revision
                    execution_config = dict(self.config["execution"])
                    capture = self._capture
                osc = copy.deepcopy(osc_source)
                session = dict(osc.get("session") or {})
                observation_ns = time.perf_counter_ns()
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
                        self.state.update({"camera_state": "WAITING",
                                           "camera_wait_reason": f"waiting for complete RGB pair: {type(exc).__name__}: {exc}",
                                           "updated_at": time.time()})
                    self.stop_event.wait(.1)
                    continue
                with self.lock:
                    if self.state.get("camera_state") == "WAITING":
                        self.state.update({"camera_state": "READY", "camera_wait_reason": None, "updated_at": time.time()})
                placeholder = False
                try:
                    observation = self._observation(osc, external, wrist)
                except ObservationNotReady as exc:
                    with self.lock:
                        changed = self.state.get('observation_wait_reason') != str(exc)
                        self.state.update(observation_wait_reason=str(exc), observation_preview_placeholder=False)
                    if changed: self._event('observation_wait',reason=str(exc))
                    self.stop_event.wait(.05)
                    continue
                with self.lock:
                    self.state.update(observation_wait_reason=None, observation_preview_placeholder=placeholder)
                self._record_inference_input(external, wrist, observation)
                try:
                    inference_ns = time.perf_counter_ns()
                    inference_started = time.monotonic()
                    compatibility = metadata_compatibility(getattr(policy, "metadata", {}), execution_config["expected_chunk_steps"])
                    if compatibility["status"] == "incompatible":
                        self._reject_chunk(session_id, client_id, compatibility["message"],
                                           (revision, run_revision), compatibility["message"])
                        break
                    response = policy.infer(observation)
                    inference_finished_ns = time.perf_counter_ns()
                    inference_ms = (inference_finished_ns-inference_ns)/1e6
                except Exception as exc:
                    self._record_websocket_event("inference_failure", f"{type(exc).__name__}: {exc}")
                    self._event('inference_error',error=f'{type(exc).__name__}: {exc}')
                    self._invalidate_policy(policy, f"OpenPI WebSocket inference failed: {type(exc).__name__}: {exc}")
                    policy = None
                    self.stop_event.wait(.05)
                    continue
                with self.lock:
                    obsolete = revision != self._connection_revision or run_revision != self._run_revision or self._changing_connection
                if obsolete:
                    policy = None
                    continue
                base = self._feedback_pose(osc)
                try:
                    actions = response.get("actions") if isinstance(response, dict) else None
                    with self.lock:
                        if (revision, run_revision) != (self._connection_revision, self._run_revision):
                            continue
                        self.state["last_response_steps"] = len(actions) if hasattr(actions, "__len__") else None
                    expected_steps = execution_config["expected_chunk_steps"]
                    rows = action_rows(response, expected_steps)
                    if base is None and (not execution_enabled or not active_session):
                        # A first preview can validate the protocol before an
                        # OSC session publishes TCP; never decode or send it.
                        with self.lock:
                            if (revision, run_revision) != (self._connection_revision, self._run_revision):
                                continue
                            self.state["protocol_compatibility"] = {"status": "compatible", "message": "7D 输入／动作响应验证通过"}
                            self.state.update(action_chunk=rows, action_chunk_length=len(rows), inference_ms=inference_ms)
                        self.stop_event.wait(float(self.config["execution"]["period_s"]))
                        continue
                    chunk_id = int(self.state.get('chunk_sequence',0))+1
                    if capture:
                        capture.emit('model_chunk',chunk_id=chunk_id,actions=rows,base_tcp=base,
                            observation_state=observation['observation/state'],prompt=observation.get('prompt'),
                            observation_started_perf_ns=observation_ns,inference_started_perf_ns=inference_ns,
                            inference_finished_perf_ns=inference_finished_ns,
                            observation_preparation_ms=(inference_ns-observation_ns)/1e6,inference_ms=inference_ms)
                    if base is None:
                        raise RuntimeError("OSC measured TCP feedback is unavailable for this Model chunk")
                    decoded = [self._target_from_action(action, base, action_index=index)
                               for index, action in enumerate(rows)]
                    absolute_chunk = [target for target, _ratio in decoded]
                    limit = execution_config["replan_steps"]
                    self._validate_execution_targets(decoded, limit, osc)
                except Exception as exc:
                    self._reject_chunk(session_id, client_id, f"Model action chunk rejected: {type(exc).__name__}: {exc}",
                                       (revision, run_revision), str(exc))
                    break
                with self.lock:
                    if (revision, run_revision) != (self._connection_revision, self._run_revision):
                        continue
                    self.state["protocol_compatibility"] = {"status": "compatible", "message": "7D 输入／动作响应验证通过"}
                    self.state.update({"action_chunk": rows, "absolute_tcp_chunk": absolute_chunk,
                                       "chunk_source": "control", "chunk_updated_at": time.time(),
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
                if placeholder or not execution_enabled or not active_session:
                    self.stop_event.wait(float(self.config["execution"]["period_s"]))
                    continue
                period_s = float(self.config["execution"]["period_s"])
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
                        wait_s = max(0.0, last_dispatch_completed_at + period_s - time.monotonic())
                        if self.stop_event.wait(wait_s):
                            break
                    dispatch_preparation_ns = time.perf_counter_ns()
                    with self.lock:
                        execution_enabled = bool(self.state.get("execution_enabled"))
                        execution_enabled = execution_enabled and run_revision == self._run_revision and revision == self._connection_revision
                    if not execution_enabled:
                        # The operator may stop AutoDL forwarding while this
                        # chunk is between rows.  Do not let a deliberately
                        # revoked session turn that normal stop into an OSC
                        # rejection on the UI.
                        break
                    sequence = self._next_sequence()
                    try:
                        with self._row_lock:
                            with self.lock:
                                if not self.state.get('execution_enabled') or run_revision != self._run_revision: break
                            previous_id = (self._cached_feedback().get('execution') or {}).get('control_sample_id')
                            request_ns = time.perf_counter_ns()
                            self._event('tcp_request',chunk_id=chunk_id,step=index,sequence=sequence,target_tcp=target,gripper_ratio=gripper,
                                dispatch_preparation_ms=(request_ns-dispatch_preparation_ns)/1e6,
                                since_inference_return_ms=(request_ns-inference_finished_ns)/1e6)
                            result = self.osc.track_tcp(session_id, client_id, sequence, target)
                            tcp_return_ns = time.perf_counter_ns()
                            self._event('tcp_return',chunk_id=chunk_id,step=index,sequence=sequence,target_tcp=target,result=result,
                                started_perf_ns=request_ns,duration_ms=(tcp_return_ns-request_ns)/1e6)
                            if not result.get("ok"): raise RuntimeError(f"OSC target rejected: {result}")
                            with self.lock: enabled = self.state.get('execution_enabled') and run_revision == self._run_revision
                            if enabled: self._send_gripper_if_needed(session_id, client_id, gripper)
                    except Exception as exc:
                        with self.lock:
                            execution_enabled = bool(self.state.get("execution_enabled"))
                        if not execution_enabled:
                            break
                        self._reject_chunk(session_id, client_id, f"Model chunk execution rejected: {type(exc).__name__}: {exc}")
                        break
                    last_dispatch_completed_at = time.monotonic()
                    with self.lock: self.state["executed_steps"] += 1; self.state["updated_at"] = time.time()
                    if index == limit-1:
                        seconds = float(self.config['execution'].get('arrival_wait_s',0))
                        self._event('arrival_start',chunk_id=chunk_id,target_tcp=target,sequence=sequence,
                            budget_started_perf_ns=tcp_return_ns,maximum_wait_s=seconds)
                        arrival = await_target(self._cached_feedback,target,(session_id,client_id),sequence,previous_id,
                            tcp_return_ns/1e9+seconds,self._wait_cancel) if seconds>0 else {'outcome':'disabled','blocking_ms':0}
                        arrival['since_tcp_return_ms'] = (time.perf_counter_ns()-tcp_return_ns)/1e6
                        self._event('arrival_end',chunk_id=chunk_id,**arrival)
                        with self.lock: self.state['arrival_result'] = arrival
        except Exception as exc:
            self._event('worker_error',error=f'{type(exc).__name__}: {exc}')
            with self.lock: self.state.update({"state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
        finally:
            self._finish_capture('inference worker exited')
            with self.lock:
                session_id, client_id = self.state.get("session_id"), self.state.get("client_id")
                needs_hold = bool(self.state.get('execution_enabled') or self.state.get('priming'))
                if needs_hold:
                    self.state['control_stop'] = {'state': 'HOLD_PENDING', 'reason': 'Model inference failed'}
                self.state.update(execution_enabled=False, priming=False)
                if self.stop_event.is_set() and self.state.get("state") != "ERROR":
                    self.state["state"] = "IDLE"
                self.state["updated_at"] = time.time()
            if needs_hold and session_id and client_id and not self.stop_event.is_set():
                try:
                    self._confirm_nonrtc_hold(session_id, client_id, "Model inference failed")
                except Exception as exc:
                    with self.lock:
                        self.state['stop_error'] = str(exc)

    def close(self) -> None:
        self._closing = True
        self.stop_event.set()
        try:
            self.stop("service closing")
        except Exception as exc:
            # Retain the failure, but still terminate all input producers.
            with self.lock:
                self.state['stop_error'] = str(exc)
        self._connection_stop.set()
        self._osc_state_stop.set()
        self._gripper_stop.set()
        with self._gripper_condition:
            self._gripper_condition.notify_all()
        if self._connection_thread is not threading.current_thread():
            self._connection_thread.join(timeout=1.0)
        if self._osc_state_thread is not threading.current_thread():
            self._osc_state_thread.join(timeout=1.0)
        if self._tcp_feed and self._tcp_feed.thread is not threading.current_thread():
            self._tcp_feed.thread.join(timeout=2.0)
        if self._gripper_feed and self._gripper_feed.thread is not threading.current_thread():
            self._gripper_feed.thread.join(timeout=2.0)
        with self.lock:
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        if self._gripper_worker is not threading.current_thread():
            self._gripper_worker.join(timeout=2.0)
        with self.lock:
            policy = self._policy
            self._policy = None
            self.state.update({"model_state": "DISCONNECTED", "websocket_error": "service closing", "updated_at": time.time()})
        if policy:
            try: policy.close()
            except Exception: pass
        if self.camera_resource is None and self.cameras: self.cameras.close()
        self._observation_source.close()
