"""Best-effort, read-only recorder for teleoperation demonstration episodes."""
from __future__ import annotations

import copy
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_list(value: Any, length: int | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or (length is not None and len(value) != length):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


class DatasetRecorder:
    """Threaded recorder. It never opens cameras or sends robot commands."""

    _CAMERAS = {"external": ("front", "外部 RGB"), "wrist": ("wrist", "腕部 RGB")}

    def __init__(self, osc: Any, cameras: Any, pico: Any, dataset_root: Path, sample_hz: float = 20.0) -> None:
        self.osc = osc
        self.cameras = cameras
        self.pico = pico
        self.root = Path(dataset_root)
        self.sample_hz = max(1.0, float(sample_hz))
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
            readme.write_text(
                "# NERO demonstration dataset\n\n"
                "Each episode contains `metadata.json`, `frames.jsonl`, and JPEG image directories when camera frames are available.\n"
                "The JSONL schema is intentionally close to LeRobot and can be converted after validation.\n",
                encoding="utf-8",
            )

    def _next_episode_index(self) -> int:
        indexes: list[int] = []
        for item in (self.root / "episodes").glob("episode_*"):
            try:
                indexes.append(int(item.name.split("_")[-1]))
            except ValueError:
                pass
        return max(indexes, default=0) + 1

    def _assert_episode_path(self, episode_dir: Path) -> Path:
        root = (self.root / "episodes").resolve()
        target = episode_dir.resolve()
        if root not in target.parents:
            raise RuntimeError("refusing to delete an episode outside the dataset root")
        return target

    @staticmethod
    def _control_source(value: Any) -> str:
        source = str(value or "web").strip().lower()
        return source if source in {"web", "pi05", "pico"} else "web"

    def _camera_sources(self, camera_state: dict[str, Any], control_source: str) -> dict[str, dict[str, Any]]:
        # Web control has no camera selector. PICO and pi0.5 share the
        # resource configured by their own panels; recording only snapshots it.
        if control_source not in {"pi05", "pico"}:
            return {}
        config = camera_state.get("config") if isinstance(camera_state.get("config"), dict) else {}
        reported = camera_state.get("sources") if isinstance(camera_state.get("sources"), dict) else {}
        result: dict[str, dict[str, Any]] = {}
        for source, (folder, label) in self._CAMERAS.items():
            item = config.get(source) if isinstance(config.get(source), dict) else {}
            status = reported.get(source) if isinstance(reported.get(source), dict) else {}
            result[source] = {
                "source": source, "folder": folder, "label": label,
                "index": item.get("index"), "width": item.get("width"), "height": item.get("height"),
                "available": bool(status.get("available", camera_state.get("ready", False))),
                "frame_available": bool(status.get("frame_available", camera_state.get("ready", False))),
                "last_frame_age_ms": status.get("last_frame_age_ms"),
                "captured_frames": 0, "dropped_frames": 0, "bytes_written": 0,
                "last_frame_at": None,
            }
        return result

    @staticmethod
    def _contents(camera_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
        images = [{"source": value["source"], "label": value["label"], "format": "JPEG",
                   "directory": f"images/{value['folder']}", "available": value["available"]}
                  for value in camera_sources.values()]
        return {
            "metadata": {"path": "metadata.json", "format": "UTF-8 JSON"},
            "frames": {"path": "frames.jsonl", "format": "UTF-8 JSON Lines, one 20 Hz sample per line"},
            "timestamps": "ISO 8601 timestamp plus episode_index and frame_index",
            "observation": "7 joint positions (rad), TCP pose, gripper width (m); unavailable values are null",
            "action": "joint target (rad), TCP target and gripper width (m)",
            "execution": "execution mode and selected control source",
            "pico_input": "position (m), quaternion (xyzw), Grip and Trigger; null when unavailable",
            "images": images,
        }

    def _public(self, active: dict[str, Any], recording: bool) -> dict[str, Any]:
        result = {key: copy.deepcopy(value) for key, value in active.items() if not key.startswith("_")}
        elapsed = max(0.0, time.monotonic() - float(active.get("_started_monotonic", time.monotonic()))) if recording else float(result.get("duration_s", 0.0))
        if recording:
            now = datetime.now(timezone.utc)
            for source in result.get("camera_sources", {}).values():
                stamp = source.get("last_frame_at")
                if stamp:
                    try:
                        source["last_frame_age_ms"] = round(max(0.0, (now - datetime.fromisoformat(stamp)).total_seconds()) * 1000, 1)
                    except (TypeError, ValueError):
                        pass
        result["recording"] = recording
        result["duration_s"] = round(elapsed, 3)
        result["effective_hz"] = round(float(result.get("frame_count", 0)) / elapsed, 3) if elapsed else 0.0
        result["sample_hz"] = self.sample_hz
        return result

    def state(self) -> dict[str, Any]:
        with self.lock:
            if self.active:
                return self._public(self.active, True)
            return {"recording": False, "dataset_root": str(self.root), "last_episode": copy.deepcopy(self.last_episode)}

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("已有 Episode 正在采集")
            task = str(body.get("task", "")).strip()
            description = str(body.get("description", "")).strip()
            if not task:
                raise ValueError("任务名称不能为空")
            control_source = self._control_source(body.get("control_source"))
            camera_state = self.cameras.snapshot() or {}
            camera_sources = self._camera_sources(camera_state, control_source)
            self._ensure_root()
            index = self._next_episode_index()
            episode_dir = self.root / "episodes" / f"episode_{index:06d}"
            episode_dir.mkdir(parents=True)
            metadata = {
                "schema": "nero-demonstration.v1", "episode_index": index, "task": task,
                "description": description, "status": "recording", "started_at": _utc(),
                "sample_hz": self.sample_hz, "control_source": control_source,
                "camera_snapshot": copy.deepcopy(camera_sources), "data_contents": self._contents(camera_sources),
                "files": {"frames": "frames.jsonl", "images": {}},
            }
            (episode_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            frames = (episode_dir / "frames.jsonl").open("w", encoding="utf-8")
            warnings = []
            if not camera_sources:
                warnings.append("当前控制源未选择相机；本 Episode 仅记录状态、动作和控制输入。")
            else:
                unavailable = [value["label"] for value in camera_sources.values() if not value["available"]]
                if unavailable:
                    warnings.append(f"相机当前不可用：{'、'.join(unavailable)}；其余数据仍会记录。")
            self.active = {
                "episode_index": index, "episode_dir": str(episode_dir), "task": task, "description": description,
                "status": "recording", "started_at": metadata["started_at"], "control_source": control_source,
                "execution_mode": None, "frame_count": 0, "dropped_frames": 0, "bytes_written": 0,
                "sample_errors": 0, "last_error": None, "warnings": warnings, "camera_sources": camera_sources,
                "data_contents": metadata["data_contents"], "_frames": frames, "_metadata": metadata,
                "_started_monotonic": time.monotonic(),
            }
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._loop, name=f"nero-dataset-episode-{index:06d}", daemon=True)
            self.thread.start()
            return self._public(self.active, True)

    def _sample(self, frame_index: int) -> dict[str, Any]:
        osc = self.osc.state() or {}
        execution, transport = osc.get("execution") or {}, osc.get("transport") or {}
        feedback = transport.get("hardware_feedback") or {}
        joints = (_finite_list(feedback.get("joint_angles_rad"), 7) or _finite_list(execution.get("measured_joint_state_rad"), 7)
                  or _finite_list(execution.get("observed_joint_state_rad"), 7) or _finite_list(execution.get("commanded_joint_state_rad"), 7))
        gripper = (osc.get("gripper") or {}).get("width_m")
        command, pico = osc.get("command") or {}, (self.pico.snapshot() if self.pico is not None else {})
        target_tcp = command.get("target_tcp") or pico.get("last_target_pose")
        target_joints = _finite_list(command.get("final_joint_target_rad"), 7)
        with self.lock:
            active = self.active
            if not active:
                return {"frame_index": frame_index, "timestamp": _utc(), "error": "recorder stopped"}
            sources = copy.deepcopy(active["camera_sources"])
            episode_dir = Path(active["episode_dir"])
        paths: dict[str, str | None] = {}
        for source, details in sources.items():
            payload = self.cameras.frame_jpeg(source)
            folder = str(details["folder"])
            if payload:
                relative = f"images/{folder}/{frame_index:06d}.jpg"
                path = episode_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                paths[source] = relative
                with self.lock:
                    if self.active:
                        stats = self.active["camera_sources"][source]
                        stats["captured_frames"] += 1; stats["bytes_written"] += len(payload); stats["last_frame_at"] = _utc()
                        self.active["bytes_written"] += len(payload)
                        self.active["_metadata"]["files"]["images"][source] = f"images/{folder}"
            else:
                paths[source] = None
                with self.lock:
                    if self.active:
                        self.active["dropped_frames"] += 1
                        self.active["camera_sources"][source]["dropped_frames"] += 1
        execution_mode = (osc.get("session") or {}).get("execution_mode")
        with self.lock:
            if self.active:
                self.active["execution_mode"] = execution_mode
                for details in self.active["camera_sources"].values():
                    if details.get("last_frame_at"):
                        details["last_frame_age_ms"] = 0.0
        return {
            "episode_index": active["episode_index"], "frame_index": frame_index, "timestamp": _utc(),
            "observation": {"images": paths, "state": {"joint_positions_rad": joints, "gripper_width_m": gripper,
                            "tcp_pose": feedback.get("tcp_pose") or execution.get("measured_tcp_pose")}},
            "action": {"joint_target_rad": target_joints, "tcp_target": target_tcp, "gripper_width_m": gripper},
            "input": {"pico_position_m": pico.get("input_position_m"), "pico_orientation_xyzw": pico.get("input_orientation_xyzw"),
                      "grip": pico.get("input_clutch"), "trigger": pico.get("input_trigger_value")},
            "execution_mode": execution_mode, "control_source": active["control_source"],
        }

    def _loop(self) -> None:
        frame_index = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                row = self._sample(frame_index)
                encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                with self.lock:
                    if self.active:
                        self.active["_frames"].write(encoded); self.active["_frames"].flush()
                        self.active["frame_count"] = frame_index + 1; self.active["bytes_written"] += len(encoded.encode("utf-8"))
                frame_index += 1
            except Exception as exc:
                with self.lock:
                    if self.active:
                        self.active["sample_errors"] += 1; self.active["last_error"] = f"{type(exc).__name__}: {exc}"
            self.stop_event.wait(max(0.0, self.sample_period - (time.monotonic() - started)))

    def stop(self, status: str = "completed", reason: str = "") -> dict[str, Any]:
        with self.lock:
            if not self.active:
                return self.state()
            active, thread = self.active, self.thread
            delete_episode = status in {"failed", "discarded"}
            active["status"] = "completed"; active["reason"] = reason; active["ended_at"] = _utc()
            active["duration_s"] = max(0.0, time.monotonic() - float(active["_started_monotonic"]))
            self.stop_event.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self.lock:
            active = self.active
            if not active:
                return {"recording": False}
            active.pop("_frames").close()
            metadata, episode_dir = active.pop("_metadata"), Path(active["episode_dir"])
            result = self._public(active, False)
            if delete_episode:
                shutil.rmtree(self._assert_episode_path(episode_dir), ignore_errors=False)
                result.update({"deleted": True, "status": "deleted", "reason": reason or "operator discarded episode"})
            else:
                metadata.update({key: value for key, value in result.items() if key not in {"recording", "episode_dir"}})
                metadata["status"] = "completed"
                episode_dir.joinpath("metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.last_episode = copy.deepcopy(result)
            self.active = None; self.thread = None
            return result

    def close(self) -> None:
        self.stop("discarded", "control service shutdown")

    def episodes(self) -> dict[str, Any]:
        self._ensure_root()
        items = []
        for path in sorted((self.root / "episodes").glob("episode_*/metadata.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                items.append({"episode_dir": str(path.parent), "status": "invalid"})
        return {"dataset_root": str(self.root), "episodes": items[-100:]}
