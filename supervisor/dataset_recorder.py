"""Timestamp-aligned, hardware-truth demonstration recorder."""
from __future__ import annotations

import copy
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_list(value: Any, length: int | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        return None
    try:
        values = [float(item) for item in value]
        return values if all(item == item and abs(item) != float("inf") for item in values) else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value == value and abs(value) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _monotonic_ns(value: Any) -> int | None:
    return value if not isinstance(value, bool) and isinstance(value, int) and value > 0 else None


class DatasetRecorder:
    """Record synchronized observations without owning cameras or controls."""

    _CAMERAS = {"external": ("front", "外部 RGB"), "wrist": ("wrist", "腕部 RGB")}
    _MAX_GAP_NS = 100_000_000
    _JPEG_QUALITY = 84
    _MAX_PENDING = 8

    def __init__(self, osc: Any, cameras: Any, pico: Any, dataset_root: Path, sample_hz: float = 20.0) -> None:
        self.osc, self.cameras, self.pico, self.root = osc, cameras, pico, Path(dataset_root)
        self.sample_hz = float(sample_hz) if sample_hz > 0 else 20.0
        self.sample_period = 1.0 / self.sample_hz
        self.lock = threading.RLock()
        self.active: dict[str, Any] | None = None
        self.last_episode: dict[str, Any] | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _ensure_root(self) -> None:
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text("# NERO demonstration dataset\n\nNew episodes use synchronized nero-demonstration.v2 frames; legacy v1 remains readable.\n", encoding="utf-8")

    def _next_episode_index(self) -> int:
        values = []
        for path in (self.root / "episodes").glob("episode_*"):
            try:
                values.append(int(path.name.split("_")[-1]))
            except ValueError:
                pass
        return max(values, default=0) + 1

    def _assert_episode_path(self, path: Path) -> Path:
        root, target = (self.root / "episodes").resolve(), path.resolve()
        if root not in target.parents:
            raise RuntimeError("refusing to delete outside dataset root")
        return target

    @staticmethod
    def _source(value: Any) -> tuple[str, str]:
        source = str(value or "web").strip().lower()
        return {"pico": ("teleoperation", "pico_4_ultra"), "web": ("teleoperation", "web_joystick"), "pi05": ("policy", "pi05"), "spacemouse": ("teleoperation", "spacemouse"), "keyboard": ("teleoperation", "keyboard")}.get(source, ("teleoperation", source or "unknown"))

    def _camera_sources(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        config, reported = state.get("config") or {}, state.get("sources") or {}
        result = {}
        for source, (folder, label) in self._CAMERAS.items():
            item, status = config.get(source) or {}, reported.get(source) or {}
            preview = status.get("preview_size") or [config.get("model_width"), config.get("model_height")]
            saved = status.get("dataset_size") or [item.get("width"), item.get("height")]
            result[source] = {"source": source, "folder": folder, "label": label, "index": item.get("index"), "width": item.get("width"), "height": item.get("height"), "preview_width": preview[0] if len(preview) > 1 else None, "preview_height": preview[1] if len(preview) > 1 else None, "saved_width": saved[0] if len(saved) > 1 else None, "saved_height": saved[1] if len(saved) > 1 else None, "available": bool(status.get("available")), "frame_available": bool(status.get("frame_available")), "captured_frames": 0, "dropped_frames": 0, "duplicate_or_stale_frames": 0, "encoding_failures": 0, "write_failures": 0, "bytes_written": 0}
        return result

    def _public(self, active: dict[str, Any], recording: bool) -> dict[str, Any]:
        result = {key: copy.deepcopy(value) for key, value in active.items() if not key.startswith("_")}
        elapsed = max(0.0, time.monotonic() - active["_started_monotonic"]) if recording else float(result.get("duration_s", 0.0))
        result.update(recording=recording, duration_s=round(elapsed, 3), sample_hz=self.sample_hz, effective_hz=round(result.get("frame_count", 0) / elapsed, 3) if elapsed else 0.0, pending_writes=len(active.get("_pending", {})))
        return result

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._public(self.active, True) if self.active else {"recording": False, "dataset_root": str(self.root), "last_episode": copy.deepcopy(self.last_episode)}

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("已有 Episode 正在采集")
            task, description = str(body.get("task", "")).strip(), str(body.get("description", "")).strip()
            if not task:
                raise ValueError("任务名称不能为空")
            requested_source = str(body.get("control_source") or "web").strip().lower()
            category, configured_device = self._source(requested_source)
            pico_connected = bool((self.pico.snapshot() if self.pico is not None else {}).get("connected"))
            operator_device = configured_device if requested_source != "pico" or pico_connected else None
            operator_connection = ("connected" if pico_connected else "not_connected") if requested_source == "pico" else "not_applicable"
            cameras = self._camera_sources(self.cameras.snapshot() or {}) if requested_source in {"pico", "pi05"} else {}
            self._ensure_root()
            index = self._next_episode_index()
            directory = self.root / "episodes" / f"episode_{index:06d}"
            directory.mkdir(parents=True)
            image_paths = {details["source"]: f"images/{details['folder']}" for details in cameras.values()}
            metadata = {"schema": "nero-demonstration.v2", "episode_index": index, "task": task, "description": description, "status": "recording", "started_at": _utc(), "sample_hz": self.sample_hz, "control_source": category, "requested_control_source": requested_source, "operator_device": operator_device, "operator_connection": operator_connection, "alignment": {"clock": "monotonic_ns", "state": "interpolation_or_nearest", "image": "nearest_unique_native_frame", "max_gap_ms": 100}, "camera_snapshot": copy.deepcopy(cameras), "files": {"frames": "frames.jsonl", "images": image_paths}, "data_contents": {"observation": "primary training data: measured hardware state and native camera images", "control_context": "optional provenance only; not required as a visual-model training target", "diagnostics": "per-source monotonic timestamps and rejection reasons"}}
            (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.active = {"episode_index": index, "episode_dir": str(directory), "task": task, "description": description, "status": "recording", "started_at": metadata["started_at"], "control_source": category, "requested_control_source": requested_source, "operator_device": operator_device, "operator_connection": operator_connection, "execution_mode": None, "frame_count": 0, "uncommanded_frame_count": 0, "commanded_frame_count": 0, "dropped_frames": 0, "rejected_samples": 0, "rejection_reasons": {}, "sample_errors": 0, "last_error": None, "camera_sources": cameras, "data_contents": metadata["data_contents"], "bytes_written": 0, "backpressure_drops": 0, "encoding_failures": 0, "write_failures": 0, "_metadata": metadata, "_frames": (directory / "frames.jsonl").open("w", encoding="utf-8"), "_started_monotonic": time.monotonic(), "_started_ns": time.monotonic_ns(), "_started_utc": datetime.now(timezone.utc), "_states": [], "_actions": [], "_last_saved_camera_ns": {}, "_pending": {}, "_next_sequence": 0, "_next_commit_sequence": 0, "_executor": ThreadPoolExecutor(max_workers=2, thread_name_prefix="nero-jpeg")}
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._loop, name="nero-dataset-recorder", daemon=True)
            self.thread.start()
            return self._public(self.active, True)

    @classmethod
    def _nearest_or_interp(cls, samples: list[tuple[int, list[float]]], target: int) -> tuple[list[float] | None, int | None]:
        if not samples:
            return None, None
        before, after = [sample for sample in samples if sample[0] <= target], [sample for sample in samples if sample[0] >= target]
        if before and after:
            a, b = before[-1], after[0]
            if target - a[0] > cls._MAX_GAP_NS or b[0] - target > cls._MAX_GAP_NS:
                return None, None
            if a[0] == b[0]:
                return list(a[1]), a[0]
            ratio = (target - a[0]) / (b[0] - a[0])
            return [x + (y - x) * ratio for x, y in zip(a[1], b[1])], target
        nearest = min(samples, key=lambda sample: abs(sample[0] - target))
        return (list(nearest[1]), nearest[0]) if abs(nearest[0] - target) <= cls._MAX_GAP_NS else (None, None)

    def _reject(self, active: dict[str, Any], reason: str) -> None:
        active["dropped_frames"] += 1
        active["rejected_samples"] += 1
        active["rejection_reasons"][reason] = int(active["rejection_reasons"].get(reason, 0)) + 1

    @classmethod
    def _encode_native_frames(cls, frames: dict[str, Any]) -> dict[str, bytes]:
        import cv2
        payloads = {}
        for source, frame in frames.items():
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, cls._JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for {source}")
            payloads[source] = encoded.tobytes()
        return payloads

    def _make_row(self, active: dict[str, Any], target: int, measured: list[float], measured_ns: int, osc: dict[str, Any], pico: dict[str, Any], raw_frames: dict[str, Any], image_times: dict[str, int], missing_sources: list[str]) -> dict[str, Any]:
        execution, transport, command, diagnostics = osc.get("execution") or {}, osc.get("transport") or {}, osc.get("command") or {}, osc.get("diagnostics") or {}
        targets = _finite_list(command.get("final_joint_target_rad"), 7)
        sent_ns = _monotonic_ns((diagnostics.get("timing") or {}).get("joint_sent_monotonic_ns") or command.get("sent_monotonic_ns"))
        action_ns, action_joints = None, None
        if targets is not None and sent_ns is not None and abs(sent_ns - target) <= self._MAX_GAP_NS:
            active["_actions"].append((sent_ns, targets)); del active["_actions"][:-40]
            action_ns, action_joints = min(active["_actions"], key=lambda item: abs(item[0] - target))
            if abs(action_ns - target) > self._MAX_GAP_NS:
                action_ns, action_joints = None, None
        timestamp = (active["_started_utc"] + timedelta(seconds=(target - active["_started_ns"]) / 1e9)).isoformat()
        control_context = None if action_joints is None else {"joint_target_rad": action_joints, "tcp_target": command.get("target_tcp"), "gripper_width_target_m": command.get("gripper_target_width_m") or (osc.get("active_action") or {}).get("width_m")}
        return {"episode_index": active["episode_index"], "frame_index": active["_next_sequence"], "timestamp": timestamp, "observation": {"images": {}, "missing_image_sources": missing_sources, "state": {"joint_positions_rad": measured, "gripper_width_measured_m": _number((osc.get("gripper") or {}).get("width_m")), "tcp_pose": (transport.get("hardware_feedback") or {}).get("tcp_pose") or execution.get("measured_tcp_pose")}}, "control_context": control_context, "diagnostics": {"control_command_recorded": control_context is not None, "timestamps_ns": {"robot_feedback": measured_ns, "control_command": action_ns, "pico": (pico.get("diagnostics") or {}).get("input_received_monotonic_ns"), "front_camera": image_times.get("external"), "wrist_camera": image_times.get("wrist")}, "alignment_target_monotonic_ns": target}, "_raw_frames": raw_frames}

    def _sample(self) -> None:
        active, target, osc = self.active, time.monotonic_ns(), self.osc.state() or {}
        pico = self.pico.snapshot() if self.pico is not None else {}
        transport, feedback = osc.get("transport") or {}, (osc.get("transport") or {}).get("hardware_feedback") or {}
        joints = _finite_list(feedback.get("joint_angles_rad"), 7)
        if joints is None:
            self._reject(active, "feedback_joints_unavailable"); return
        state_ns = _monotonic_ns(feedback.get("received_monotonic_ns") or (transport.get("feedback_mailbox") or {}).get("received_monotonic_ns") or (osc.get("diagnostics") or {}).get("timing", {}).get("feedback_received_monotonic_ns"))
        if state_ns is None:
            self._reject(active, "feedback_timestamp_invalid"); return
        active["_states"].append((state_ns, joints)); del active["_states"][:-20]
        measured, measured_ns = self._nearest_or_interp(active["_states"], target)
        if measured is None or measured_ns is None:
            self._reject(active, "feedback_stale"); return
        if len(active["_pending"]) >= self._MAX_PENDING:
            active["backpressure_drops"] += 1; self._reject(active, "jpeg_queue_full"); return
        raw_frames, image_times, missing_sources = {}, {}, []
        for source, details in active["camera_sources"].items():
            selected = self.cameras.dataset_frame(source, target)
            if selected is None:
                details["dropped_frames"] += 1; missing_sources.append(source); continue
            stamp, frame = selected; stamp = _monotonic_ns(stamp)
            if stamp is None or abs(stamp - target) > self._MAX_GAP_NS:
                details["dropped_frames"] += 1; missing_sources.append(source); continue
            if stamp <= active["_last_saved_camera_ns"].get(source, 0):
                details["duplicate_or_stale_frames"] += 1; missing_sources.append(source); continue
            if getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
                details["dropped_frames"] += 1; missing_sources.append(source); continue
            raw_frames[source], image_times[source] = frame, stamp
            details["saved_width"], details["saved_height"] = int(frame.shape[1]), int(frame.shape[0])
        if active["camera_sources"] and not raw_frames:
            self._reject(active, "camera_frames_duplicate_or_stale" if missing_sources else "camera_frames_unavailable"); return
        row = self._make_row(active, target, measured, measured_ns, osc, pico, raw_frames, image_times, missing_sources)
        sequence = active["_next_sequence"]; active["_next_sequence"] += 1
        active["_pending"][sequence] = (row, image_times, active["_executor"].submit(self._encode_native_frames, raw_frames))

    def _commit_ready(self, wait: bool = False) -> None:
        active = self.active
        while active and active["_next_commit_sequence"] in active["_pending"]:
            sequence = active["_next_commit_sequence"]
            row, image_times, future = active["_pending"][sequence]
            if not future.done() and not wait:
                return
            try:
                payloads = future.result()
                directory = Path(active["episode_dir"])
                for source, payload in payloads.items():
                    details = active["camera_sources"][source]
                    rel = f"images/{details['folder']}/{sequence:06d}.jpg"; path = directory / rel
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
                    row["observation"]["images"]["front" if source == "external" else source] = rel
                    details["captured_frames"] += 1; details["bytes_written"] += len(payload); active["bytes_written"] += len(payload); active["_last_saved_camera_ns"][source] = image_times[source]
                row.pop("_raw_frames", None)
                active["_frames"].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"); active["_frames"].flush(); active["frame_count"] += 1
                if row["control_context"] is None: active["uncommanded_frame_count"] += 1
                else: active["commanded_frame_count"] += 1
            except OSError as exc:
                active["write_failures"] += 1; active["last_error"] = f"{type(exc).__name__}: {exc}"; self._reject(active, "image_write_failed")
            except Exception as exc:
                active["encoding_failures"] += 1; active["last_error"] = f"{type(exc).__name__}: {exc}"; self._reject(active, "jpeg_encoding_failed")
            finally:
                active["_pending"].pop(sequence, None); active["_next_commit_sequence"] += 1

    def _loop(self) -> None:
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    self._commit_ready(); self._sample(); self._commit_ready()
            except Exception as exc:
                with self.lock:
                    if self.active:
                        self.active["sample_errors"] += 1; self.active["last_error"] = f"{type(exc).__name__}: {exc}"
            deadline += self.sample_period
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def stop(self, status: str = "completed", reason: str = "") -> dict[str, Any]:
        with self.lock:
            if not self.active:
                return self.state()
            active, thread = self.active, self.thread
            active["status"], active["reason"], active["duration_s"] = "completed", reason, time.monotonic() - active["_started_monotonic"]
            self.stop_event.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self.lock:
            active = self.active; self._commit_ready(wait=True); active["_executor"].shutdown(wait=True); active.pop("_frames").close()
            metadata, directory = active.pop("_metadata"), Path(active["episode_dir"]); active.pop("_executor", None)
            result = self._public(active, False); delete = status in {"failed", "discarded"}
            if delete:
                shutil.rmtree(self._assert_episode_path(directory)); result.update(deleted=True, status="deleted", reason=reason or "operator discarded episode")
            else:
                metadata.update({key: value for key, value in result.items() if key not in {"recording", "episode_dir", "pending_writes"}}); metadata["status"] = "completed"
                (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.last_episode = copy.deepcopy(result); self.active = None; self.thread = None
            return result

    def close(self) -> None:
        self.stop("discarded", "control service shutdown")

    def episodes(self) -> dict[str, Any]:
        self._ensure_root(); items = []
        for path in sorted((self.root / "episodes").glob("episode_*/metadata.json")):
            try: items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError): items.append({"episode_dir": str(path.parent), "status": "invalid"})
        return {"dataset_root": str(self.root), "episodes": items[-100:]}
