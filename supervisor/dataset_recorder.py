"""RGB-only, dual-rate, hardware-truth demonstration recorder."""
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
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return values if all(value == value and abs(value) != float("inf") for value in values) else None


def _number(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value == value and abs(value) != float("inf") else None
    except (TypeError, ValueError):
        return None


class DatasetRecorder:
    """Record RGB cameras at 20 Hz and robot state/action at 15 Hz.

    Camera and robot streams have separate files and timestamps, but share one
    monotonic clock origin. A frame is never reused: a missing or duplicate
    source timestamp makes the episode abnormal instead of silently filling it.
    """

    _CAMERAS = {"external": ("front", "外部 RGB"), "wrist": ("wrist", "腕部 RGB")}
    _CAMERA_HZ = 20.0
    _STATE_HZ = 15.0
    _MAX_GAP_NS = 100_000_000
    _CAMERA_WAIT_S = 0.25

    def __init__(self, osc: Any, cameras: Any, pico: Any, dataset_root: Path, sample_hz: float = 20.0) -> None:
        self.osc, self.cameras, self.pico = osc, cameras, pico
        self.root = Path(dataset_root)
        self.sample_hz = self._CAMERA_HZ
        self.lock = threading.RLock()
        self.active: dict[str, Any] | None = None
        self.last_episode: dict[str, Any] | None = None
        self.stop_event = threading.Event()
        self.camera_thread: threading.Thread | None = None
        self.state_thread: threading.Thread | None = None

    def _ensure_root(self) -> None:
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                "# NERO demonstration dataset\n\n"
                "New episodes use RGB-only nero-demonstration.v3 with separate 20 Hz camera and 15 Hz robot-state streams.\n",
                encoding="utf-8",
            )

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
        return {
            "pico": ("teleoperation", "pico_4_ultra"),
            "web": ("teleoperation", "web_joystick"),
            "pi05": ("policy", "pi05"),
            "spacemouse": ("teleoperation", "spacemouse"),
            "keyboard": ("teleoperation", "keyboard"),
        }.get(source, ("teleoperation", source or "unknown"))

    def _camera_sources(self, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        config, reported = state.get("config") or {}, state.get("sources") or {}
        result = {}
        for source, (folder, label) in self._CAMERAS.items():
            item, status = config.get(source) or {}, reported.get(source) or {}
            result[source] = {
                "source": source, "folder": folder, "label": label,
                "index": item.get("index"), "width": item.get("width"), "height": item.get("height"),
                "available": bool(status.get("available")), "frame_available": bool(status.get("frame_available")),
                "last_frame_age_ms": status.get("last_frame_age_ms"),
                "captured_frames": 0, "written_frames": 0, "dropped_frames": 0,
                "duplicate_frames": 0, "bytes_written": 0,
            }
        return result

    def _public(self, active: dict[str, Any], recording: bool) -> dict[str, Any]:
        result = {key: copy.deepcopy(value) for key, value in active.items() if not key.startswith("_")}
        elapsed = max(0.0, time.monotonic() - active["_started_monotonic"]) if recording else float(result.get("duration_s", 0.0))
        result.update(
            recording=recording, duration_s=round(elapsed, 3), camera_hz=self._CAMERA_HZ,
            robot_state_hz=self._STATE_HZ,
            effective_camera_hz=round(result.get("camera_frame_count", 0) / elapsed, 3) if elapsed else 0.0,
            effective_robot_state_hz=round(result.get("robot_state_count", 0) / elapsed, 3) if elapsed else 0.0,
        )
        return result

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._public(self.active, bool(self.active and not self.active.get("fatal_error"))) if self.active else {
                "recording": False, "dataset_root": str(self.root), "last_episode": copy.deepcopy(self.last_episode)
            }

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("已有 Episode 正在采集")
            task, description = str(body.get("task", "")).strip(), str(body.get("description", "")).strip()
            if not task:
                raise ValueError("任务名称不能为空")
            category, device = self._source(body.get("control_source"))
            cameras = self._camera_sources(self.cameras.snapshot() or {})
            self._ensure_root()
            index = self._next_episode_index()
            directory = self.root / "episodes" / f"episode_{index:06d}"
            directory.mkdir(parents=True)
            origin_ns = time.monotonic_ns()
            metadata = {
                "schema": "nero-demonstration.v3-rgb", "episode_index": index, "task": task, "description": description,
                "status": "recording", "started_at": _utc(), "camera_rate_hz": self._CAMERA_HZ,
                "robot_state_rate_hz": self._STATE_HZ, "control_source": category, "operator_device": device,
                "timeline_origin_monotonic_ns": origin_ns,
                "alignment": {"clock": "time.monotonic_ns", "camera": "independent_20hz", "robot_state": "independent_15hz", "policy": "block_and_alarm_no_reuse", "max_gap_ms": self._MAX_GAP_NS / 1_000_000},
                "camera_snapshot": copy.deepcopy(cameras),
                "files": {"camera_frames": "camera_frames.jsonl", "robot_states": "robot_states.jsonl", "collection_events": "collection_events.jsonl", "images": {source: f"images/{item['folder']}" for source, item in cameras.items()}},
                "data_contents": {"observation": "measured hardware feedback only", "action": "actual accepted robot target command", "images": "RGB JPEG only"},
            }
            (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.active = {
                "episode_index": index, "episode_dir": str(directory), "task": task, "description": description,
                "status": "recording", "started_at": metadata["started_at"], "control_source": category, "operator_device": device,
                "camera_sources": cameras, "camera_frame_count": 0, "robot_state_count": 0,
                "camera_dropped_frames": 0, "robot_state_dropped": 0, "state_invalid_events": 0,
                "waiting_for_action": True, "action_available": False,
                "valid_robot_state_count": 0, "event_count": 0, "error_counts": {},
                "duplicate_frames": 0, "timestamp_errors": 0,
                "write_failures": 0, "blocking_events": 0, "fatal_error": None, "bytes_written": 0,
                "state_last_error": None,
                "_metadata": metadata, "_camera_file": (directory / "camera_frames.jsonl").open("w", encoding="utf-8"),
                "_state_file": (directory / "robot_states.jsonl").open("w", encoding="utf-8"),
                "_events_file": (directory / "collection_events.jsonl").open("w", encoding="utf-8"),
                "_started_monotonic": time.monotonic(), "_origin_ns": origin_ns,
                # Never consume a preview frame captured before the operator
                # pressed Start.  Both streams begin after this shared origin.
                "_camera_last_ns": {"external": origin_ns, "wrist": origin_ns},
                "_state_last_ns": origin_ns, "_camera_next_index": 0, "_camera_gap_open": False,
            }
            self.stop_event = threading.Event()
            for source, item in cameras.items():
                if not item["available"] or not item["frame_available"] or item.get("last_frame_age_ms") is None or float(item["last_frame_age_ms"]) > 500:
                    self._record_event(self.active, "camera_unavailable_at_start", f"{item['label']} 在开始采集时没有可用新帧", "等待相机恢复；不停止 Episode", {"source": source})
            self.camera_thread = threading.Thread(target=self._camera_loop, name="nero-camera-recorder", daemon=True)
            self.state_thread = threading.Thread(target=self._state_loop, name="nero-state-recorder", daemon=True)
            self.camera_thread.start(); self.state_thread.start()
            return self._public(self.active, True)

    def _fail(self, message: str) -> None:
        with self.lock:
            if self.active:
                self._record_event(self.active, "recorder_error", message, "跳过当前问题点并继续采集")

    def _record_event(self, active: dict[str, Any], code: str, message: str, disposition: str, details: dict[str, Any] | None = None) -> None:
        """Persist a visible non-fatal collection issue without stopping an episode."""
        event = {
            "event_index": int(active.get("event_count", 0)),
            "event_timestamp_monotonic_ns": time.monotonic_ns(),
            "timestamp_since_origin_ns": time.monotonic_ns() - int(active["_origin_ns"]),
            "code": str(code), "message": str(message), "disposition": str(disposition),
            "details": details or {},
        }
        active["event_count"] = int(active.get("event_count", 0)) + 1
        counts = active.setdefault("error_counts", {}); counts[code] = int(counts.get(code, 0)) + 1
        try:
            self._write_json(active["_events_file"], event)
        except OSError:
            active["event_write_failures"] = int(active.get("event_write_failures", 0)) + 1

    @staticmethod
    def _write_json(handle: Any, row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()

    def _new_camera_frame(self, source: str, last_ns: int) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + self._CAMERA_WAIT_S
        while time.monotonic() < deadline and not self.stop_event.is_set():
            stamp = self.cameras.frame_timestamp(source)
            if stamp is not None and int(stamp) > last_ns:
                payload = self.cameras.frame_jpeg(source, int(stamp))
                if payload:
                    return int(stamp), payload
            time.sleep(0.005)
        return None

    def _new_camera_pair(self, last_ns: dict[str, int]) -> dict[str, tuple[int, bytes]] | None:
        deadline = time.monotonic() + self._CAMERA_WAIT_S
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if hasattr(self.cameras, "frame_jpegs"):
                packet = self.cameras.frame_jpegs()
                if packet and all(source in packet and packet[source][0] > last_ns[source] for source in self._CAMERAS):
                    return packet
            else:
                packet = {}
                for source in self._CAMERAS:
                    frame = self._new_camera_frame(source, last_ns[source])
                    if frame is None:
                        packet = {}
                        break
                    packet[source] = frame
                if packet:
                    return packet
            time.sleep(0.005)
        return None

    def _save_camera(self, active: dict[str, Any], source: str, stamp: int, payload: bytes, sequence: int) -> None:
        previous = active["_camera_last_ns"][source]
        if stamp <= previous:
            active["duplicate_frames"] += 1; active["camera_sources"][source]["duplicate_frames"] += 1; active["timestamp_errors"] += 1
            raise RuntimeError(f"{source} RGB timestamp is duplicate or not monotonic")
        directory = Path(active["episode_dir"]); folder = active["camera_sources"][source]["folder"]
        path = directory / "images" / folder / f"{sequence:06d}.jpg"; path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(payload); size = path.stat().st_size
        except OSError as exc:
            active["write_failures"] += 1; raise RuntimeError(f"{source} RGB 写入失败：{exc}") from exc
        self._write_json(active["_camera_file"], {"camera": source, "frame_index": sequence, "camera_timestamp_monotonic_ns": stamp, "timestamp_since_origin_ns": stamp - active["_origin_ns"], "rgb": str(path.relative_to(directory)).replace("\\", "/"), "rgb_width": active["camera_sources"][source]["width"], "rgb_height": active["camera_sources"][source]["height"]})
        active["_camera_last_ns"][source] = stamp
        details = active["camera_sources"][source]; details["captured_frames"] += 1; details["written_frames"] += 1; details["bytes_written"] += size
        active["bytes_written"] += size

    def _camera_loop(self) -> None:
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            deadline += 1.0 / self._CAMERA_HZ
            with self.lock: active = self.active
            if not active: return
            sequence = active["_camera_next_index"]
            try:
                packet = self._new_camera_pair(active["_camera_last_ns"])
            except Exception as exc:
                active["blocking_events"] += 1
                self._record_event(active, "camera_read_error", f"读取 RGB 相机时发生异常：{exc}", "跳过当前问题点，不使用旧帧，继续尝试读取")
                self.stop_event.wait(0.05)
                continue
            if packet is None:
                active["camera_dropped_frames"] += 1; active["blocking_events"] += 1
                for source in self._CAMERAS:
                    active["camera_sources"][source]["dropped_frames"] += 1
                if not active.get("_camera_gap_open"):
                    active["_camera_gap_open"] = True
                    self._record_event(active, "camera_gap_started", f"两路 RGB 在 {self._CAMERA_WAIT_S:.2f}s 内未同时提供新帧", "跳过当前缺帧点，继续等待并采集后续新帧", {"wait_s": self._CAMERA_WAIT_S})
                self.stop_event.wait(0.05)
                continue
            if active.get("_camera_gap_open"):
                active["_camera_gap_open"] = False
                self._record_event(active, "camera_gap_recovered", "两路 RGB 已恢复同时提供新帧", "缺帧期间不补旧帧；从恢复后的新帧继续写入")
            for source in self._CAMERAS:
                frame = packet[source]
                try: self._save_camera(active, source, frame[0], frame[1], sequence)
                except Exception as exc:
                    self._record_event(active, "camera_write_error", f"{source} RGB 写入失败：{exc}", "跳过当前问题点，不使用旧帧补写，继续采集")
                    continue
            active["camera_frame_count"] += 1; active["_camera_next_index"] += 1
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def _state_loop(self) -> None:
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            deadline += 1.0 / self._STATE_HZ
            with self.lock: active = self.active
            if not active: return
            try:
                osc = self.osc.state() or {}
            except Exception as exc:
                active["robot_state_dropped"] += 1; active["state_invalid_events"] += 1
                active["state_last_error"] = f"读取机器人状态时发生异常：{exc}"
                self._record_event(active, "robot_state_read_error", active["state_last_error"], "跳过当前状态点，继续按 15Hz 重试")
                self.stop_event.wait(max(0.0, deadline - time.monotonic())); continue
            transport = osc.get("transport") or {}; feedback = transport.get("hardware_feedback") or {}
            joints = _finite_list(feedback.get("joint_angles_rad"), 7)
            state_stamp = feedback.get("received_monotonic_ns") or (transport.get("feedback_mailbox") or {}).get("received_monotonic_ns")
            state_stamp = int(state_stamp) if state_stamp is not None else None
            command = osc.get("command") or {}; targets = _finite_list(command.get("final_joint_target_rad"), 7)
            sent_stamp = (osc.get("diagnostics") or {}).get("timing", {}).get("joint_sent_monotonic_ns") or command.get("sent_monotonic_ns")
            sent_stamp = int(sent_stamp) if sent_stamp is not None else None
            # A state tick without measured feedback or an accepted action is
            # invalid, but must not stop the camera stream.  The old code used
            # the episode-wide stop_event here, so one missing action at the
            # first 15 Hz tick left an 80-second episode with only its first
            # camera frame.  A real measured observation is still written with
            # valid=false when its action is missing, while the training
            # reader can filter it out. No action is ever fabricated.
            if joints is None or state_stamp is None or state_stamp <= active["_state_last_ns"]:
                active["robot_state_dropped"] += 1; active["state_invalid_events"] += 1
                active["state_last_error"] = "机器人真实反馈缺失或时间戳重复；未写入伪造 observation"
                active["waiting_for_action"] = True; active["action_available"] = False
                self._record_event(active, "robot_state_invalid", active["state_last_error"], "跳过当前状态点，继续按 15Hz 等待下一条真实反馈")
                self.stop_event.wait(max(0.0, deadline - time.monotonic())); continue
            if abs(state_stamp - time.monotonic_ns()) > self._MAX_GAP_NS * 3:
                active["timestamp_errors"] += 1; active["state_invalid_events"] += 1
                active["state_last_error"] = "机器人状态时间戳不在当前 monotonic 时间轴上"
                active["waiting_for_action"] = True; active["action_available"] = False
                self._record_event(active, "robot_state_timestamp_invalid", active["state_last_error"], "跳过当前状态点，继续按 15Hz 等待下一条真实反馈")
                self.stop_event.wait(max(0.0, deadline - time.monotonic())); continue
            state_valid = targets is not None and sent_stamp is not None
            if state_valid:
                active["waiting_for_action"] = False; active["action_available"] = True
            else:
                active["state_invalid_events"] += 1
                active["state_last_error"] = "未取得实际发送的机器人 action；保留真实 observation，action 标记为无效"
                active["waiting_for_action"] = True; active["action_available"] = False
                self._record_event(active, "action_missing", active["state_last_error"], "保留真实 observation；action=null，训练时过滤该状态点")
            row = {
                "state_index": active["robot_state_count"], "valid": state_valid, "validity": "valid" if state_valid else "invalid_missing_action",
                "robot_state_timestamp_monotonic_ns": state_stamp, "action_timestamp_monotonic_ns": sent_stamp,
                "state_timestamp_since_origin_ns": state_stamp - active["_origin_ns"], "action_timestamp_since_origin_ns": sent_stamp - active["_origin_ns"] if sent_stamp is not None else None,
                "observation": {"joint_positions_rad": joints, "joint_velocities_rad_s": _finite_list(feedback.get("joint_velocities_rad_s")), "tcp_pose": feedback.get("tcp_pose"), "gripper_width_measured_m": _number(feedback.get("gripper_width_m")), "robot_status": feedback.get("robot_status"), "feedback_sequence": feedback.get("sequence")},
                "action": {"joint_target_rad": targets, "tcp_target": command.get("target_tcp"), "gripper_width_target_m": _number(command.get("gripper_target_width_m"))} if state_valid else None,
            }
            try:
                self._write_json(active["_state_file"], row); active["robot_state_count"] += 1; active["_state_last_ns"] = state_stamp
                if state_valid: active["valid_robot_state_count"] += 1
            except OSError as exc:
                active["write_failures"] += 1
                self._record_event(active, "robot_state_write_error", f"robot_states.jsonl 写入失败：{exc}", "跳过当前状态点，继续采集")
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def stop(self, status: str = "completed", reason: str = "") -> dict[str, Any]:
        with self.lock:
            if not self.active: return self.state()
            active = self.active; self.stop_event.set(); threads = [self.camera_thread, self.state_thread]
        for thread in threads:
            if thread and thread is not threading.current_thread(): thread.join(timeout=5)
        with self.lock:
            active = self.active; active["reason"] = reason; active["duration_s"] = time.monotonic() - active["_started_monotonic"]
            if active.get("fatal_error"):
                active["status"] = "abnormal"
            elif status == "completed":
                active["status"] = "completed"
            active["_camera_file"].close(); active["_state_file"].close(); active["_events_file"].close(); directory = Path(active["episode_dir"]); metadata = active.pop("_metadata"); result = self._public(active, False)
            if status in {"failed", "discarded"}:
                shutil.rmtree(self._assert_episode_path(directory)); result.update(deleted=True, status="deleted", reason=reason or "operator discarded episode")
            else:
                metadata.update({key: value for key, value in result.items() if key not in {"recording", "pending_writes"}}); metadata["status"] = result["status"]
                metadata["statistics"] = {key: value for key, value in result.items() if key.endswith("count") or "dropped" in key or "duplicate" in key or key in {"bytes_written", "timestamp_errors", "fatal_error", "event_count", "error_counts", "event_write_failures"}}
                metadata["error_summary"] = result.get("error_counts", {})
                metadata["collection_policy"] = {
                    "operator_controls_episode_lifetime": True,
                    "missing_frames_or_action_never_stop_episode": True,
                    "invalid_observations_are_skipped": True,
                    "real_observation_without_action_is_saved_valid_false": True,
                    "no_old_frame_reuse": True,
                }
                (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.last_episode = copy.deepcopy(result); self.active = None; self.camera_thread = None; self.state_thread = None; return result

    def close(self) -> None: self.stop("discarded", "control service shutdown")

    def episodes(self) -> dict[str, Any]:
        self._ensure_root(); items = []
        for path in sorted((self.root / "episodes").glob("episode_*/metadata.json")):
            try: items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError): items.append({"episode_dir": str(path.parent), "status": "invalid"})
        return {"dataset_root": str(self.root), "episodes": items[-100:]}
