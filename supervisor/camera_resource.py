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
            capture = self._open_capture(index)
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

    def _open_capture(self, index: int) -> Any:
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
        return self.cv2.VideoCapture(index, self.cv2.CAP_DSHOW if hasattr(self.cv2, "CAP_DSHOW") else 0)

    def available(self, source: str) -> bool:
        return self.sources.get(source) is not None

    def read(self) -> tuple[Any | None, Any | None]:
        frames: dict[Any, Any | None] = {}
        with self.read_lock:
            for index, capture in self.captures.items():
                ok, frame = capture.read()
                if not ok or frame is None:
                    frames[index] = None
                    continue
                # Some DirectShow RealSense drivers intermittently deliver an
                # all-black placeholder frame between valid RGB frames. Do not
                # publish it: replacing the last image with that placeholder
                # is perceived as a flashing black preview in the console.
                if float(self.cv2.mean(frame)[0] + self.cv2.mean(frame)[1] + self.cv2.mean(frame)[2]) / 3.0 <= 2.0:
                    frames[index] = None
                    continue
                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]; scale = min(self.width / w, self.height / h)
                resized = self.cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
                canvas = __import__("numpy").zeros((self.height, self.width, 3), dtype=__import__("numpy").uint8)
                y, x = (self.height - resized.shape[0]) // 2, (self.width - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                frames[index] = canvas
        return tuple(frames.get(self._source_index(source)) for source in ("external", "wrist"))

    def _source_index(self, source: str) -> int | None:
        capture = self.sources.get(source)
        return next((index for index, item in self.captures.items() if item is capture), None)

    def close(self) -> None:
        for capture in getattr(self, "captures", {}).values(): capture.release()


class SharedCameraResource:
    """One capture owner, with latest frames safe for multiple adapters."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config); self.lock = threading.RLock(); self.cameras: CameraPair | None = None
        self.preview_stop = threading.Event(); self.preview_thread: threading.Thread | None = None
        self.frames: dict[str, Any] = {}; self.frame_times: dict[str, float] = {}; self.frame_times_ns: dict[str, int] = {}; self.frame_history: dict[str, list[tuple[int, Any]]] = {"external": [], "wrist": []}; self.frame_version = 0; self.last_error: str | None = None
        self._devices: list[dict[str, Any]] | None = None
        self._devices_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            sources = {
                key: {
                    "available": self.cameras is not None and self.cameras.available(key),
                    "frame_available": key in self.frames,
                    "last_frame_age_ms": round((now - self.frame_times[key]) * 1000, 1) if key in self.frame_times else None,
                    "timestamp_monotonic_ns": self.frame_times_ns.get(key),
                }
                for key in ("external", "wrist")
            }
            return {"ready": self.cameras is not None, "state": "READY" if self.cameras else ("ERROR" if self.last_error else "IDLE"),
                    "config": copy.deepcopy(self.config), "sources": sources, "frame_version": self.frame_version, "last_error": self.last_error}

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
                external, wrist = self.cameras.read()
                with self.lock:
                    captured_ns = time.monotonic_ns()
                    current = {key: frame for key, frame in (("external", external), ("wrist", wrist)) if frame is not None}
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
                    self.frame_times.update({key: captured_ns / 1e9 for key in current})
                    self.frame_times_ns.update({key: captured_ns for key in current})
                    for key, frame in current.items():
                        history = self.frame_history.setdefault(key, [])
                        history.append((captured_ns, frame.copy()))
                        del history[:-40]
                    self.frame_version += 1
                self.preview_stop.wait(.05)
            except Exception as exc:
                # Keep previewing after a recoverable camera/backend hiccup.
                with self.lock: self.last_error = f"{type(exc).__name__}: {exc}"
                self.preview_stop.wait(.1)

    def activate(self) -> dict[str, Any]:
        with self.lock:
            self.preview_stop.set()
            if self.preview_thread and self.preview_thread is not threading.current_thread(): self.preview_thread.join(timeout=.5)
            if self.cameras: self.cameras.close()
            self.cameras = CameraPair(self.config); self.frames = {}; self.frame_times = {}; self.frame_times_ns = {}; self.frame_history = {"external": [], "wrist": []}; self.frame_version = 0; self.last_error = None; self.preview_stop = threading.Event()
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
            self.cameras = None; self.preview_thread = None; self.frames = {}; self.frame_times = {}; self.frame_times_ns = {}; self.frame_history = {"external": [], "wrist": []}; self.frame_version = 0; self.last_error = None
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

    def close(self) -> None:
        self.deactivate()
