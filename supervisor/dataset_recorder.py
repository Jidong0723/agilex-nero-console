"""Episode recorder for teleoperation demonstrations.

The recorder writes only under the workspace-level ``dataset`` directory.
Each episode contains JPEG observations, JSONL state/action samples, and a
metadata manifest that can later be converted to LeRobot format.
"""
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
    if not isinstance(value, (list, tuple)):
        return None
    if length is not None and len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result


class DatasetRecorder:
    """Threaded, best-effort recorder that never sends robot commands."""

    def __init__(self, osc: Any, cameras: Any, pico: Any, dataset_root: Path, sample_hz: float = 20.0) -> None:
        self.osc = osc
        self.cameras = cameras
        self.pico = pico
        self.root = Path(dataset_root)
        self.sample_period = 1.0 / max(1.0, float(sample_hz))
        self.lock = threading.RLock()
        self.active: dict[str, Any] | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _ensure_root(self) -> None:
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                "# NERO demonstration dataset\n\n"
                "Each episode contains `metadata.json`, `frames.jsonl`, and `images/front`/`images/wrist`.\n"
                "The JSONL schema is intentionally close to LeRobot and can be converted after validation.\n",
                encoding="utf-8",
            )

    def _next_episode_index(self) -> int:
        episodes = self.root / "episodes"
        indexes = []
        for item in episodes.glob("episode_*"):
            try:
                indexes.append(int(item.name.split("_")[-1]))
            except ValueError:
                continue
        return max(indexes, default=0) + 1

    def _assert_episode_path(self, episode_dir: Path) -> Path:
        """Allow destructive cleanup only inside this recorder's episode root."""
        root = (self.root / "episodes").resolve()
        target = episode_dir.resolve()
        if root not in target.parents:
            raise RuntimeError("refusing to delete an episode outside the dataset root")
        return target

    def state(self) -> dict[str, Any]:
        with self.lock:
            if not self.active:
                return {"recording": False, "dataset_root": str(self.root), "last_episode": None}
            result = {key: copy.deepcopy(value) for key, value in self.active.items()
                      if not key.startswith("_")}
            result["recording"] = True
            return result

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("已有 Episode 正在采集")
            camera_state = self.cameras.snapshot()
            if not camera_state.get("ready"):
                raise RuntimeError("请先打开两台相机，再开始采集")
            task = str(body.get("task", "")).strip()
            description = str(body.get("description", "")).strip()
            if not task:
                raise ValueError("任务名称不能为空")
            self._ensure_root()
            index = self._next_episode_index()
            episode_dir = self.root / "episodes" / f"episode_{index:06d}"
            (episode_dir / "images" / "front").mkdir(parents=True)
            (episode_dir / "images" / "wrist").mkdir(parents=True)
            metadata = {
                "schema": "nero-demonstration.v1",
                "episode_index": index,
                "task": task,
                "description": description,
                "status": "recording",
                "started_at": _utc(),
                "sample_hz": 1.0 / self.sample_period,
                "camera_config": camera_state.get("config"),
                "files": {"frames": "frames.jsonl", "front": "images/front", "wrist": "images/wrist"},
            }
            (episode_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            frames = (episode_dir / "frames.jsonl").open("w", encoding="utf-8")
            self.active = {"episode_index": index, "episode_dir": str(episode_dir), "task": task,
                           "description": description, "status": "recording", "started_at": metadata["started_at"],
                           "frame_count": 0, "dropped_frames": 0, "_frames": frames, "_metadata": metadata}
            self.stop_event = threading.Event()
            self.thread = threading.Thread(target=self._loop, name=f"nero-dataset-episode-{index:06d}", daemon=True)
            self.thread.start()
            return self.state()

    def _sample(self, frame_index: int) -> dict[str, Any]:
        osc = self.osc.state() or {}
        execution = osc.get("execution") or {}
        transport = osc.get("transport") or {}
        feedback = transport.get("hardware_feedback") or {}
        joints = (_finite_list(feedback.get("joint_angles_rad"), 7) or
                  _finite_list(execution.get("measured_joint_state_rad"), 7) or
                  _finite_list(execution.get("observed_joint_state_rad"), 7) or
                  _finite_list(execution.get("commanded_joint_state_rad"), 7))
        gripper = (osc.get("gripper") or {}).get("width_m")
        command = osc.get("command") or {}
        pico = self.pico.snapshot() if self.pico is not None else {}
        target_tcp = command.get("target_tcp") or pico.get("last_target_pose")
        target_joints = _finite_list(command.get("final_joint_target_rad"), 7)
        external = self.cameras.frame_jpeg("external")
        wrist = self.cameras.frame_jpeg("wrist")
        with self.lock:
            active = self.active
            if not active:
                return {"frame_index": frame_index, "timestamp": _utc(), "error": "recorder stopped"}
            episode_dir = Path(active["episode_dir"])
            paths = {}
            for name, payload in (("front", external), ("wrist", wrist)):
                if payload:
                    path = episode_dir / "images" / name / f"{frame_index:06d}.jpg"
                    path.write_bytes(payload)
                    paths[name] = str(path.relative_to(episode_dir)).replace("\\", "/")
                else:
                    active["dropped_frames"] += 1
            return {
                "episode_index": active["episode_index"], "frame_index": frame_index, "timestamp": _utc(),
                "observation": {"images": paths, "state": {"joint_positions_rad": joints, "gripper_width_m": gripper,
                              "tcp_pose": feedback.get("tcp_pose") or execution.get("measured_tcp_pose")}},
                "action": {"joint_target_rad": target_joints, "tcp_target": target_tcp,
                            "gripper_width_m": gripper},
                "input": {"pico_position_m": pico.get("input_position_m"),
                           "pico_orientation_xyzw": pico.get("input_orientation_xyzw"),
                           "grip": pico.get("input_clutch"), "trigger": pico.get("input_trigger_value")},
                "execution_mode": (osc.get("session") or {}).get("execution_mode"),
            }

    def _loop(self) -> None:
        frame_index = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                row = self._sample(frame_index)
                with self.lock:
                    if self.active:
                        self.active["_frames"].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        self.active["_frames"].flush()
                        self.active["frame_count"] = frame_index + 1
                frame_index += 1
            except Exception as exc:
                with self.lock:
                    if self.active:
                        self.active["last_error"] = f"{type(exc).__name__}: {exc}"
            self.stop_event.wait(max(0.0, self.sample_period - (time.monotonic() - started)))

    def stop(self, status: str = "completed", reason: str = "") -> dict[str, Any]:
        with self.lock:
            if not self.active:
                return self.state()
            active = self.active
            delete_episode = status in {"failed", "discarded"}
            active["status"] = "completed"
            active["reason"] = reason
            active["ended_at"] = _utc()
            thread = self.thread
            self.stop_event.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self.lock:
            active = self.active
            if not active:
                return {"recording": False}
            frames = active.pop("_frames")
            frames.close()
            metadata = active.pop("_metadata")
            episode_dir = Path(active["episode_dir"])
            if delete_episode:
                shutil.rmtree(self._assert_episode_path(episode_dir), ignore_errors=False)
                result = {key: value for key, value in active.items() if key != "episode_dir"}
                result.update({"recording": False, "deleted": True, "episode_dir": str(episode_dir),
                               "status": "deleted", "reason": reason or "operator discarded episode"})
                self.active = None
                self.thread = None
                return result
            metadata.update({key: value for key, value in active.items() if key not in {"episode_dir"}})
            metadata["ended_at"] = active.get("ended_at")
            metadata["duration_s"] = max(0.0, (datetime.fromisoformat(metadata["ended_at"]) - datetime.fromisoformat(metadata["started_at"])).total_seconds())
            Path(active["episode_dir"]).joinpath("metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result = {key: value for key, value in active.items() if key != "episode_dir"}
            result["recording"] = False
            result["episode_dir"] = active["episode_dir"]
            self.active = None
            self.thread = None
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

