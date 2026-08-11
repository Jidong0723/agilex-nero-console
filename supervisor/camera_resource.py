"""Shared dual-RGB camera resource for input adapters.

The cameras are a Console resource, not a policy-specific implementation
detail.  Adapters may consume the same latest-frame pair but never own a
capture device themselves.
"""
from __future__ import annotations

import copy
import threading
import time
from typing import Any


class CameraPair:
    def __init__(self, config: dict[str, Any]) -> None:
        import cv2
        self.cv2 = cv2; self.read_lock = threading.Lock()
        self.width, self.height = int(config["model_width"]), int(config["model_height"])
        self.captures = []
        for key in ("external", "wrist"):
            item = config[key]
            capture = cv2.VideoCapture(int(item["index"]), cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(item["width"])); capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(item["height"]))
            if not capture.isOpened():
                self.close(); raise RuntimeError(f"cannot open {key} camera index {item['index']}")
            self.captures.append(capture)

    def read(self) -> tuple[Any, Any]:
        frames = []
        with self.read_lock:
            for capture in self.captures:
                ok, frame = capture.read()
                if not ok or frame is None: raise RuntimeError("camera frame capture failed")
                rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]; scale = min(self.width / w, self.height / h)
                resized = self.cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
                canvas = __import__("numpy").zeros((self.height, self.width, 3), dtype=__import__("numpy").uint8)
                y, x = (self.height - resized.shape[0]) // 2, (self.width - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized; frames.append(canvas)
        return frames[0], frames[1]

    def close(self) -> None:
        for capture in getattr(self, "captures", []): capture.release()


class SharedCameraResource:
    """One capture owner, with latest frames safe for multiple adapters."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config); self.lock = threading.RLock(); self.cameras: CameraPair | None = None
        self.preview_stop = threading.Event(); self.preview_thread: threading.Thread | None = None
        self.frames: dict[str, Any] = {}; self.frame_version = 0; self.last_error: str | None = None
        self._devices: list[dict[str, Any]] | None = None
        self._devices_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"ready": self.cameras is not None, "state": "READY" if self.cameras else ("ERROR" if self.last_error else "IDLE"),
                    "config": copy.deepcopy(self.config), "frame_version": self.frame_version, "last_error": self.last_error}

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
        except Exception as exc: raise RuntimeError(f"camera enumeration failed: {exc}") from exc
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
            if self.config["external"]["index"] == self.config["wrist"]["index"]: raise ValueError("external and wrist cameras must be different")
            return self.snapshot()

    def _preview_loop(self) -> None:
        while not self.preview_stop.is_set():
            try:
                if self.cameras is None: return
                external, wrist = self.cameras.read()
                with self.lock: self.frames = {"external": external, "wrist": wrist}; self.frame_version += 1
                self.preview_stop.wait(.1)
            except Exception as exc:
                with self.lock: self.last_error = f"{type(exc).__name__}: {exc}"
                return

    def activate(self) -> dict[str, Any]:
        with self.lock:
            self.preview_stop.set()
            if self.preview_thread and self.preview_thread is not threading.current_thread(): self.preview_thread.join(timeout=.5)
            if self.cameras: self.cameras.close()
            self.cameras = CameraPair(self.config); self.frames = {}; self.frame_version = 0; self.last_error = None; self.preview_stop = threading.Event()
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
            self.cameras = None; self.preview_thread = None; self.frames = {}; self.frame_version = 0; self.last_error = None
            return self.snapshot()

    def read(self) -> tuple[Any, Any]:
        with self.lock: cameras = self.cameras
        if cameras is None: raise RuntimeError("activate the external and wrist cameras first")
        return cameras.read()

    def frame_jpeg(self, source: str) -> bytes | None:
        with self.lock: frame = self.frames.get(source)
        if frame is None: return None
        import cv2
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
        return encoded.tobytes() if ok else None

    def close(self) -> None:
        self.deactivate()
