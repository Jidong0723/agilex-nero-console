"""Shared dual-RGB camera resource for input adapters.

The cameras are a Console resource, not a policy-specific implementation
detail.  Adapters may consume the same latest-frame pair but never own a
capture device themselves.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any


class RealSenseRgbCapture:
    """Open a RealSense RGB sensor through librealsense, not DirectShow."""
    def __init__(self) -> None:
        import pyrealsense2 as rs
        self.rs = rs
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("no RealSense device found by librealsense")
        device = devices[0]
        self.serial = device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, Any | None]:
        try:
            frames = self.pipeline.wait_for_frames(1500)
            color = frames.get_color_frame()
            if not color:
                return False, None
            return True, __import__("numpy").asanyarray(color.get_data()).copy()
        except RuntimeError:
            return False, None

    def release(self) -> None:
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


class CameraPair:
    def __init__(self, config: dict[str, Any]) -> None:
        import cv2
        self.cv2 = cv2; self.read_lock = threading.Lock()
        self.width, self.height = int(config["model_width"]), int(config["model_height"])
        self.captures: dict[int, Any] = {}
        self.sources: dict[str, Any | None] = {}
        for key in ("external", "wrist"):
            item = config[key]
            index = int(item["index"])
            if index in self.captures:
                self.sources[key] = self.captures[index]
                continue
            capture = self._open_capture(index, item)
            # Keep the driver-negotiated format. The RealSense RGB DirectShow
            # endpoint on this host returns black frames after a forced 640×480
            # mode change; frames are resized below for the model input.
            if capture.isOpened():
                self.captures[index] = capture
                self.sources[key] = capture
            else:
                capture.release()
                self.sources[key] = None
        if not self.captures:
            raise RuntimeError("cannot open any configured camera")
        self.read_pool = ThreadPoolExecutor(max_workers=len(self.captures), thread_name_prefix="nero-camera-read")
        self.pending_reads: dict[int, Future[Any]] = {
            index: self.read_pool.submit(self._read_one, index, capture)
            for index, capture in self.captures.items()
        }

    def _open_capture(self, index: int, settings: dict[str, Any]) -> Any:
        """Prefer librealsense for the RGB endpoint, with OpenCV fallback."""
        name = ""
        try:
            from cv2_enumerate_cameras import enumerate_cameras
            item = next((item for item in enumerate_cameras(self.cv2.CAP_DSHOW) if int(item.index) == index), None)
            name = str(getattr(item, "name", ""))
        except Exception:
            pass
        if "realsense" in name.lower() and "rgb" in name.lower():
            try:
                return RealSenseRgbCapture()
            except (ImportError, RuntimeError):
                # Keeping the OpenCV fallback makes non-RealSense setups and
                # installations without librealsense continue to work.
                pass
        backend = self.cv2.CAP_DSHOW if hasattr(self.cv2, "CAP_DSHOW") else 0
        capture = self.cv2.VideoCapture(index, backend)
        if capture.isOpened() and "realsense" not in name.lower():
            # Two 640x480 uncompressed USB streams exceeded the shared bus
            # budget on the NERO host and fell to roughly 12 Hz. Dabai DC1
            # supports MJPEG; requesting it explicitly restores sufficient
            # headroom for synchronized 15 Hz dataset sampling.
            capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, int(settings.get("width", 640)))
            capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, int(settings.get("height", 480)))
            # DirectShow reapplies its default subtype when resolution is
            # changed, so FOURCC must be set after width and height.
            capture.set(self.cv2.CAP_PROP_FOURCC, self.cv2.VideoWriter_fourcc(*"MJPG"))
            # These two cameras expose only YUY2 through DirectShow on this
            # host. Requesting 30 Hz oversubscribes their shared USB path;
            # 20 Hz leaves headroom for a stable 15 Hz synchronized dataset.
            capture.set(self.cv2.CAP_PROP_FPS, float(settings.get("fps", 20.0)))
            # Dabai DC1's automatic exposure extends a frame to about 80 ms
            # in the collection room, limiting even one camera to 12.5 Hz.
            # Pinning exposure and gain keeps exposure below the 50 ms frame
            # period and also makes image statistics repeatable across runs.
            if settings.get("exposure") is not None:
                capture.set(self.cv2.CAP_PROP_EXPOSURE, float(settings["exposure"]))
            if settings.get("gain") is not None:
                capture.set(self.cv2.CAP_PROP_GAIN, float(settings["gain"]))
            if hasattr(self.cv2, "CAP_PROP_BUFFERSIZE"):
                capture.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def available(self, source: str) -> bool:
        return self.sources.get(source) is not None

    def diagnostics(self, source: str) -> dict[str, Any]:
        capture = self.sources.get(source)
        if capture is None or isinstance(capture, RealSenseRgbCapture):
            return {"backend": "librealsense" if capture is not None else None}
        fourcc = int(round(capture.get(self.cv2.CAP_PROP_FOURCC)))
        return {
            "backend": "opencv",
            "negotiated_width": int(round(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH))),
            "negotiated_height": int(round(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT))),
            "negotiated_fps": float(capture.get(self.cv2.CAP_PROP_FPS)),
            "negotiated_fourcc": "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)),
            "exposure": float(capture.get(self.cv2.CAP_PROP_EXPOSURE)),
            "gain": float(capture.get(self.cv2.CAP_PROP_GAIN)),
        }

    def _read_one(self, index: int, capture: Any) -> tuple[int, Any | None]:
        ok, frame = capture.read()
        # GetTickCount64-backed monotonic_ns is only 15.625 ms on this host.
        # QPC-backed perf_counter_ns preserves independent camera timing.
        received_ns = time.perf_counter_ns()
        if not ok or frame is None:
            return index, None
        if float(sum(self.cv2.mean(frame)[:3])) / 3.0 <= 2.0:
            return index, None
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]; scale = min(self.width / w, self.height / h)
        resized = self.cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
        canvas = __import__("numpy").zeros((self.height, self.width, 3), dtype=__import__("numpy").uint8)
        y, x = (self.height - resized.shape[0]) // 2, (self.width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return index, (canvas, rgb, received_ns)

    def read_with_raw(self) -> dict[str, tuple[Any, Any, int]]:
        """Return whichever independently paced camera reads completed next."""
        with self.read_lock:
            done, _ = wait(list(self.pending_reads.values()), timeout=1.5, return_when=FIRST_COMPLETED)
            frames: dict[int, Any] = {}
            for index, future in list(self.pending_reads.items()):
                if future not in done:
                    continue
                try:
                    _, frames[index] = future.result()
                except Exception:
                    frames[index] = None
                self.pending_reads[index] = self.read_pool.submit(self._read_one, index, self.captures[index])
        result: dict[str, tuple[Any, Any, int]] = {}
        for source in ("external", "wrist"):
            frame = frames.get(self._source_index(source))
            if frame is not None:
                result[source] = frame
        return result

    def read(self) -> tuple[Any | None, Any | None]:
        frames = self.read_with_raw()
        return tuple(frames.get(source, (None, None))[0] for source in ("external", "wrist"))

    def _source_index(self, source: str) -> int | None:
        capture = self.sources.get(source)
        return next((index for index, item in self.captures.items() if item is capture), None)

    def close(self) -> None:
        with self.read_lock:
            for capture in getattr(self, "captures", {}).values(): capture.release()
            for future in getattr(self, "pending_reads", {}).values(): future.cancel()
        self.read_pool.shutdown(wait=True, cancel_futures=True)


class SharedCameraResource:
    """One capture owner, with latest frames safe for multiple adapters."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config); self.lock = threading.RLock(); self.frame_condition = threading.Condition(self.lock); self.cameras: CameraPair | None = None
        self.preview_stop = threading.Event(); self.preview_thread: threading.Thread | None = None
        self.frames: dict[str, Any] = {}; self.frame_times: dict[str, float] = {}; self.frame_times_ns: dict[str, int] = {}; self.frame_history: dict[str, list[tuple[int, Any]]] = {"external": [], "wrist": []}
        self.dataset_frame_history: dict[str, list[tuple[int, Any]]] = {"external": [], "wrist": []}; self.dataset_frame_sizes: dict[str, tuple[int, int]] = {}
        self.dataset_source_history: dict[str, list[dict[str, Any]]] = {"external": [], "wrist": []}
        self.dataset_source_sequence: dict[str, int] = {"external": 0, "wrist": 0}
        self.dataset_pair_history: list[dict[str, Any]] = []; self.dataset_pair_sequence = 0
        self.dataset_pair_candidates: dict[str, list[tuple[int, Any]]] = {"external": [], "wrist": []}
        self.dataset_pair_unmatched_drops: dict[str, int] = {"external": 0, "wrist": 0}
        self.dataset_pair_sync_limit_ns = int(float(config.get("pair_sync_limit_s", 0.020)) * 1e9)
        self.frame_version = 0; self.last_error: str | None = None
        self._devices: list[dict[str, Any]] | None = None
        self._devices_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.perf_counter()
            sources = {
                key: {
                    "available": self.cameras is not None and self.cameras.available(key),
                    "frame_available": key in self.frames,
                    "last_frame_age_ms": round((now - self.frame_times[key]) * 1000, 1) if key in self.frame_times else None,
                    "timestamp_monotonic_ns": self.frame_times_ns.get(key),
                    "preview_size": [int(self.config.get("model_width", 224)), int(self.config.get("model_height", 224))],
                    "dataset_size": list(self.dataset_frame_sizes[key]) if key in self.dataset_frame_sizes else None,
                    "capture_hz": self._capture_hz(self.dataset_frame_history.get(key, [])),
                    "capture": self.cameras.diagnostics(key) if self.cameras is not None else {},
                }
                for key in ("external", "wrist")
            }
            pair_times = [
                (int(item["frames"]["external"][0]) + int(item["frames"]["wrist"][0])) // 2
                for item in self.dataset_pair_history
            ]
            pair_skews = [
                abs(int(item["frames"]["external"][0]) - int(item["frames"]["wrist"][0])) / 1e9
                for item in self.dataset_pair_history
            ]
            return {"ready": self.cameras is not None, "state": "READY" if self.cameras else ("ERROR" if self.last_error else "IDLE"),
                    "config": copy.deepcopy(self.config), "sources": sources, "frame_version": self.frame_version,
                    "dataset_pair_sequence": self.dataset_pair_sequence,
                    "dataset_source_sequences": dict(self.dataset_source_sequence),
                    "dataset_pair_hz": self._capture_hz([(stamp, None) for stamp in pair_times]),
                    "dataset_pair_skew_s": {"maximum": max(pair_skews, default=None),
                                             "mean": sum(pair_skews) / len(pair_skews) if pair_skews else None,
                                             "limit": self.dataset_pair_sync_limit_ns / 1e9},
                    "dataset_pair_unmatched_drops": dict(self.dataset_pair_unmatched_drops),
                    "last_error": self.last_error}

    @staticmethod
    def _capture_hz(history: list[tuple[int, Any]]) -> float | None:
        if len(history) < 3:
            return None
        elapsed_s = (history[-1][0] - history[0][0]) / 1e9
        return round((len(history) - 1) / elapsed_s, 3) if elapsed_s > 0.0 else None

    @staticmethod
    def _windows_camera_devices() -> list[dict[str, Any]]:
        """List present Windows camera devices when OpenCV is unavailable.

        OpenCV is still required to open a capture stream, but this fallback
        keeps the selector useful and makes a missing runtime dependency
        diagnosable instead of presenting an empty device list.
        """
        if os.name != "nt":
            return []
        command = (
            "Get-PnpDevice -PresentOnly -Class Camera | "
            "Select-Object FriendlyName,InstanceId | ConvertTo-Json -Compress"
        )
        try:
            shell = shutil.which("pwsh") or shutil.which("powershell")
            if not shell:
                return []
            result = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=8,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            rows = json.loads(result.stdout)
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                return []
            return [
                {
                    "index": index,
                    "name": str(row.get("FriendlyName") or f"Windows camera {index}"),
                    "backend": "windows-pnp",
                    "instance_id": str(row.get("InstanceId") or ""),
                }
                for index, row in enumerate(rows)
                if isinstance(row, dict)
            ]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return []

    def devices(self) -> list[dict[str, Any]]:
        with self.lock:
            if self._devices is not None and time.monotonic() - self._devices_at < 2.0:
                return copy.deepcopy(self._devices)
        try:
            import cv2
            devices: dict[int, dict[str, Any]] = {}
            try:
                from cv2_enumerate_cameras import enumerate_cameras
                for item in enumerate_cameras(cv2.CAP_DSHOW):
                    devices[int(item.index)] = {"index": int(item.index), "name": str(item.name), "backend": int(item.backend)}
            except Exception:
                pass
            # The enumeration package is not installed in every service
            # environment. Probe all usable OpenCV backends as a fallback and
            # merge the results instead of stopping after the first backend.
            backends = [getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), getattr(cv2, "CAP_ANY", 0)]
            for backend in dict.fromkeys(backends):
                for index in range(10):
                    if index in devices: continue
                    capture = cv2.VideoCapture(index, backend)
                    opened = capture.isOpened(); capture.release()
                    if opened:
                        devices[index] = {"index": index, "name": f"OpenCV camera {index}", "backend": int(backend)}
            devices = [devices[index] for index in sorted(devices)]
        except ImportError:
            devices = self._windows_camera_devices()
            if not devices:
                raise RuntimeError("camera enumeration requires OpenCV; no Windows camera devices were found")
        except Exception as exc:
            raise RuntimeError(f"camera enumeration failed: {exc}") from exc
        with self.lock:
            self._devices = devices; self._devices_at = time.monotonic()
            return copy.deepcopy(devices)

    def update_config(self, cameras: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            for key in ("external", "wrist"):
                item = cameras.get(key) if isinstance(cameras, dict) else None
                if not isinstance(item, dict): continue
                index = int(item.get("index", self.config[key]["index"]))
                if not 0 <= index <= 32: raise ValueError(f"{key} camera index must be 0-32")
                self.config[key]["index"] = index
            return self.snapshot()

    def _preview_loop(self) -> None:
        while not self.preview_stop.is_set():
            try:
                if self.cameras is None: return
                captured = self.cameras.read_with_raw()
                with self.lock:
                    current = {key: pair[0] for key, pair in captured.items()}
                    if not current:
                        # USB cameras, especially RealSense, can have a short
                        # warm-up interval or transient empty read. Keep the
                        # worker alive and retain any previously captured JPEG
                        # instead of turning a momentary gap into a black UI.
                        self.last_error = None if self.frames else "waiting for a camera frame"
                        self.preview_stop.wait(.05)
                        continue
                    self.last_error = None
                    # A failed source must not erase a last-known-good frame
                    # from the other source (or from a shared capture).
                    self.frames.update(current)
                    source_times_ns = {key: int(captured[key][2]) for key in current}
                    self.frame_times.update({key: source_times_ns[key] / 1e9 for key in current})
                    self.frame_times_ns.update(source_times_ns)
                    for key, frame in current.items():
                        captured_ns = source_times_ns[key]
                        history = self.frame_history.setdefault(key, [])
                        history.append((captured_ns, frame.copy()))
                        del history[:-40]
                        raw = captured[key][1]
                        raw_history = self.dataset_frame_history.setdefault(key, [])
                        raw_history.append((captured_ns, raw.copy()))
                        # 12 frames is enough for the recorder's 100 ms
                        # alignment window while bounding native-frame memory.
                        del raw_history[:-12]
                        self.dataset_source_sequence[key] += 1
                        source_history = self.dataset_source_history.setdefault(key, [])
                        source_history.append({"sequence": self.dataset_source_sequence[key],
                                               "timestamp": captured_ns, "frame": raw_history[-1][1]})
                        # This is the authoritative recorder queue. It is
                        # independent per camera: phase drift must never make
                        # one valid native frame discard another.
                        del source_history[:-64]
                        self.dataset_frame_sizes[key] = (int(raw.shape[1]), int(raw.shape[0]))
                        candidates = self.dataset_pair_candidates.setdefault(key, [])
                        candidates.append((captured_ns, raw_history[-1][1]))
                        del candidates[:-64]
                    while all(self.dataset_pair_candidates.get(key) for key in ("external", "wrist")):
                        external_ns = int(self.dataset_pair_candidates["external"][0][0])
                        wrist_ns = int(self.dataset_pair_candidates["wrist"][0][0])
                        if abs(external_ns - wrist_ns) > self.dataset_pair_sync_limit_ns:
                            older = "external" if external_ns < wrist_ns else "wrist"
                            self.dataset_pair_candidates[older].pop(0)
                            self.dataset_pair_unmatched_drops[older] += 1
                            continue
                        pair_frames = {key: self.dataset_pair_candidates[key].pop(0) for key in ("external", "wrist")}
                        self.dataset_pair_sequence += 1
                        self.dataset_pair_history.append({"sequence": self.dataset_pair_sequence, "frames": pair_frames})
                        # The recorder drains this queue continuously. Keep a
                        # bounded recovery window without retaining minutes of
                        # uncompressed RGB in memory.
                        del self.dataset_pair_history[:-64]
                    self.frame_version += 1
                    self.frame_condition.notify_all()
                # Successful blocking camera reads already pace this loop.
                # An additional fixed sleep reduced 15 Hz devices to roughly
                # 12-13 Hz and caused avoidable duplicate-frame rejection.
            except Exception as exc:
                # Keep previewing after a recoverable camera/backend hiccup.
                with self.lock: self.last_error = f"{type(exc).__name__}: {exc}"
                self.preview_stop.wait(.1)

    def activate(self) -> dict[str, Any]:
        with self.lock:
            self.preview_stop.set()
            if self.preview_thread and self.preview_thread is not threading.current_thread(): self.preview_thread.join(timeout=.5)
            if self.cameras: self.cameras.close()
            self.cameras = CameraPair(self.config); self.frames = {}; self.frame_times = {}; self.frame_times_ns = {}; self.frame_history = {"external": [], "wrist": []}; self.dataset_frame_history = {"external": [], "wrist": []}; self.dataset_frame_sizes = {}; self.dataset_source_history = {"external": [], "wrist": []}; self.dataset_source_sequence = {"external": 0, "wrist": 0}; self.dataset_pair_history = []; self.dataset_pair_sequence = 0; self.dataset_pair_candidates = {"external": [], "wrist": []}; self.dataset_pair_unmatched_drops = {"external": 0, "wrist": 0}; self.frame_version = 0; self.last_error = None; self.preview_stop = threading.Event()
            self.preview_thread = threading.Thread(target=self._preview_loop, name="nero-shared-camera-preview", daemon=True); self.preview_thread.start()
            return self.snapshot()

    def deactivate(self) -> dict[str, Any]:
        """Release both camera handles and stop preview work immediately."""
        with self.lock:
            self.preview_stop.set()
            preview = self.preview_thread
        if preview and preview is not threading.current_thread(): preview.join(timeout=1.0)
        with self.lock:
            if self.cameras: self.cameras.close()
            self.cameras = None; self.preview_thread = None; self.frames = {}; self.frame_times = {}; self.frame_times_ns = {}; self.frame_history = {"external": [], "wrist": []}; self.dataset_frame_history = {"external": [], "wrist": []}; self.dataset_frame_sizes = {}; self.dataset_source_history = {"external": [], "wrist": []}; self.dataset_source_sequence = {"external": 0, "wrist": 0}; self.dataset_pair_history = []; self.dataset_pair_sequence = 0; self.dataset_pair_candidates = {"external": [], "wrist": []}; self.dataset_pair_unmatched_drops = {"external": 0, "wrist": 0}; self.frame_version = 0; self.last_error = None
            return self.snapshot()

    def read(self) -> tuple[Any, Any]:
        with self.lock: cameras = self.cameras
        if cameras is None: raise RuntimeError("activate the external and wrist cameras first")
        return cameras.read()

    def frame_jpeg(self, source: str, target_monotonic_ns: int | None = None) -> bytes | None:
        with self.lock:
            frame = self.frames.get(source)
            if target_monotonic_ns is not None:
                history = self.frame_history.get(source, [])
                if history:
                    _, frame = min(history, key=lambda item: abs(item[0] - int(target_monotonic_ns)))
        if frame is None: return None
        import cv2
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
        return encoded.tobytes() if ok else None

    def frame_timestamp(self, source: str, target_monotonic_ns: int | None = None) -> int | None:
        with self.lock:
            history = self.frame_history.get(source, [])
            if not history: return None
            if target_monotonic_ns is None: return self.frame_times_ns.get(source)
            return min(history, key=lambda item: abs(item[0] - int(target_monotonic_ns)))[0]

    def dataset_frame(self, source: str, target_monotonic_ns: int | None = None) -> tuple[int, Any] | None:
        """Return a copy of the nearest native RGB frame for DatasetRecorder."""
        with self.lock:
            history = self.dataset_frame_history.get(source, [])
            if not history:
                return None
            if target_monotonic_ns is None:
                stamp, frame = history[-1]
            else:
                stamp, frame = min(history, key=lambda item: abs(item[0] - int(target_monotonic_ns)))
            return stamp, frame.copy()

    def dataset_frame_pair(self, target_monotonic_ns: int, after: dict[str, int], timeout_s: float) -> dict[str, tuple[int, Any]] | None:
        """Wait briefly for one new native frame from each camera.

        This resolves harmless phase differences between the 15 Hz recorder
        and the faster camera streams without duplicating a stale frame.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self.frame_condition:
            while True:
                selected: dict[str, tuple[int, Any]] = {}
                for source in ("external", "wrist"):
                    candidates = [
                        item for item in self.dataset_frame_history.get(source, [])
                        if item[0] > int(after.get(source, 0))
                    ]
                    if candidates:
                        stamp, frame = min(candidates, key=lambda item: abs(item[0] - int(target_monotonic_ns)))
                        selected[source] = (stamp, frame.copy())
                if len(selected) == 2:
                    return selected
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.frame_condition.wait(remaining)

    def dataset_frame_pairs_after(self, after_sequence: int, timeout_s: float = 0.0,
                                  max_items: int = 8) -> list[dict[str, Any]]:
        """Drain every produced native RGB pair after a producer sequence."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self.frame_condition:
            while True:
                available = [item for item in self.dataset_pair_history if int(item["sequence"]) > int(after_sequence)]
                if available:
                    result = []
                    for item in available[:max(1, int(max_items))]:
                        result.append({
                            "sequence": int(item["sequence"]),
                            "frames": {source: (int(stamp), frame.copy()) for source, (stamp, frame) in item["frames"].items()},
                        })
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return []
                self.frame_condition.wait(remaining)

    def dataset_frames_after(self, source: str, after_sequence: int, timeout_s: float = 0.0,
                             max_items: int = 8) -> list[dict[str, Any]]:
        """Drain native frames from one camera without collection-time pairing."""
        if source not in ("external", "wrist"):
            raise ValueError(f"unknown camera source: {source}")
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self.frame_condition:
            while True:
                available = [item for item in self.dataset_source_history.get(source, [])
                             if int(item["sequence"]) > int(after_sequence)]
                if available:
                    return [{"sequence": int(item["sequence"]),
                             "timestamp": int(item["timestamp"]),
                             "frame": item["frame"].copy()}
                            for item in available[:max(1, int(max_items))]]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return []
                self.frame_condition.wait(remaining)

    def close(self) -> None:
        self.deactivate()
