"""Native-rate, read-only recorder for future NERO TCP-VLA processing.

Collection deliberately performs no FK, resampling, action reconstruction,
video encoding, Rot6D conversion, or H16 construction.  Those transformations
must consume the timestamped raw streams after collection.
"""
from __future__ import annotations

import copy
import bisect
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supervisor.dataset_recorder import _finite_list, _json_safe, _monotonic_ns, _number, _pose_components


SCHEMA_VERSION = "nero.tcp-vla.episode.v3"
PHYSICAL_GATES = ("contact", "stable_grasp", "lift", "transport", "lower", "release", "settle")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class TcpVlaDatasetRecorder:
    """Persist 20 Hz dual RGB and 50 Hz measured robot state independently."""

    def __init__(self, osc: Any, cameras: Any, pico: Any, dataset_root: Path, *,
                 fk_client: Any | None = None,
                 sample_hz: float = 15.0, camera_sync_limit_s: float = 0.020,
                 raw_camera_hz: float = 20.0, raw_robot_state_hz: float = 50.0,
                 raw_jpeg_quality: int = 95,
                 feedback_age_limit_s: float = 0.020,
                 training_camera_alignment_limit_s: float = 0.030,
                 training_robot_bracket_limit_s: float = 0.035,
                 gripper_closed_width_m: float = 0.010,
                 gripper_open_width_m: float = 0.095) -> None:
        self.osc, self.cameras, self.pico = osc, cameras, pico
        # Retained only as an injected dependency for API compatibility.  It
        # is intentionally never started or called during raw collection.
        self.fk_client = fk_client
        self.root = Path(dataset_root)
        self.training_view_hz = float(sample_hz)
        self.camera_sync_limit_s = float(camera_sync_limit_s)
        self.feedback_age_limit_s = float(feedback_age_limit_s)
        self.training_camera_alignment_limit_s = float(training_camera_alignment_limit_s)
        self.training_robot_bracket_limit_s = float(training_robot_bracket_limit_s)
        self.raw_camera_hz = float(raw_camera_hz)
        self.raw_robot_state_hz = float(raw_robot_state_hz)
        if min(self.training_view_hz, self.raw_camera_hz, self.raw_robot_state_hz) <= 0.0:
            raise ValueError("dataset rates must be positive")
        self.raw_camera_period = 1.0 / self.raw_camera_hz
        self.raw_robot_state_period = 1.0 / self.raw_robot_state_hz
        self.raw_jpeg_quality = max(70, min(100, int(raw_jpeg_quality)))
        self.gripper_closed_width_m = float(gripper_closed_width_m)
        self.gripper_open_width_m = float(gripper_open_width_m)
        if self.gripper_open_width_m <= self.gripper_closed_width_m:
            raise ValueError("gripper open width must exceed closed width")
        self.lock = threading.RLock()
        self.active: dict[str, Any] | None = None
        self.last_episode: dict[str, Any] | None = None
        self.stop_event = threading.Event()
        self.raw_camera_thread: threading.Thread | None = None
        self.raw_robot_thread: threading.Thread | None = None

    def _ensure_root(self) -> None:
        for name in ("episodes", "failed_episodes", "staging"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _monotonic_to_perf_offset_ns() -> int:
        """Measure the local Windows clock offset without touching control."""
        offsets: list[int] = []
        for _ in range(31):
            before = time.monotonic_ns()
            perf = time.perf_counter_ns()
            after = time.monotonic_ns()
            offsets.append(perf - ((before + after) // 2))
        offsets.sort()
        return offsets[len(offsets) // 2]

    @staticmethod
    def _feedback_age(sample: dict[str, Any], offset_ns: int) -> tuple[float | None, int | None]:
        observed_ns = _monotonic_ns(sample.get("feedback_monotonic_ns"))
        fresh_monotonic_ns = _monotonic_ns(sample.get("sdk_fresh_monotonic_ns"))
        if observed_ns is None:
            return None, None
        if fresh_monotonic_ns is None:
            return None, None
        fresh_perf_ns = fresh_monotonic_ns + int(offset_ns)
        return max(0.0, (observed_ns - fresh_perf_ns) / 1e9), fresh_perf_ns

    def _next_index(self) -> int:
        values: list[int] = []
        for category in ("episodes", "failed_episodes"):
            for path in (self.root / category).glob("episode_*"):
                try:
                    values.append(int(path.name.split("_")[-1]))
                except ValueError:
                    pass
        return max(values, default=-1) + 1

    @staticmethod
    def _camera_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        config, sources = state.get("config") or {}, state.get("sources") or {}
        result: dict[str, Any] = {}
        for source in ("external", "wrist"):
            item, status = config.get(source) or {}, sources.get(source) or {}
            result[source] = {
                "index": item.get("index"), "width": item.get("width"), "height": item.get("height"),
                "available": bool(status.get("available")), "frame_available": bool(status.get("frame_available")),
                "dataset_size": status.get("dataset_size"), "capture_hz": status.get("capture_hz"),
            }
        return result

    def _public(self, active: dict[str, Any], recording: bool) -> dict[str, Any]:
        result = {key: copy.deepcopy(value) for key, value in active.items() if not key.startswith("_")}
        elapsed = max(0.0, time.monotonic() - active["_started_monotonic"]) if recording else float(result.get("duration_s", 0.0))
        camera_rate = result.get("raw_camera_frames", 0) / elapsed if elapsed else 0.0
        robot_rate = result.get("raw_robot_states", 0) / elapsed if elapsed else 0.0
        result.update(
            recording=recording, duration_s=round(elapsed, 3),
            frame_count=int(result.get("raw_camera_frames", 0)), effective_hz=round(camera_rate, 3),
            raw_camera_hz=self.raw_camera_hz, raw_robot_state_hz=self.raw_robot_state_hz,
            raw_camera_effective_hz=round(camera_rate, 3), raw_robot_state_effective_hz=round(robot_rate, 3),
            sample_hz=None, training_view_hz=self.training_view_hz,
        )
        result["raw_robot_drops"] = active["_raw_robot_drops"]
        result["raw_robot_prestart_ignored"] = active["_raw_robot_prestart_ignored"]
        if recording and elapsed > 1.0 and not active["raw_robot_states"] and not result.get("last_error"):
            result["last_error"] = ("机械臂状态尚未写入：请检查反馈流；采集前时间戳被忽略 "
                                    f"{active['_raw_robot_prestart_ignored']} 次")
        return result

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._public(self.active, True) if self.active else {
                "recording": False, "dataset_root": str(self.root), "last_episode": copy.deepcopy(self.last_episode),
                "schema": SCHEMA_VERSION, "dataset_stage": "native_rate_raw_collection",
                "instruction_mode": "manual_per_episode",
            }

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("已有父 Episode 正在采集")
            task = str(body.get("task") or "").strip()
            prompt = str(body.get("description") or "").strip()
            if not task:
                raise ValueError("任务名称不能为空")
            if not prompt:
                raise ValueError("自然语言任务指令不能为空")
            source = str(body.get("control_source") or "pico").strip().lower()
            pico_state = self.pico.snapshot() if self.pico is not None else {}
            if source == "pico" and not bool(pico_state.get("connected")):
                raise RuntimeError("PICO 未连接，禁止开始正式父 Episode")
            camera_state = self.cameras.snapshot() or {}
            camera_snapshot = self._camera_snapshot(camera_state)
            unavailable = [name for name, item in camera_snapshot.items() if not item["available"] or not item["frame_available"]]
            if unavailable:
                raise RuntimeError(f"双 RGB 门禁失败，缺少可用画面: {', '.join(unavailable)}")
            for name, item in camera_snapshot.items():
                if (int(item.get("width") or 0), int(item.get("height") or 0)) != (640, 480):
                    raise RuntimeError(f"{name} 原始RGB要求640×480")
                capture_hz = _number(item.get("capture_hz"))
                if capture_hz is None:
                    raise RuntimeError(f"{name} 相机频率尚未稳定，请等待相机预览稳定后再开始采集")
                if capture_hz < self.raw_camera_hz * 0.90:
                    raise RuntimeError(
                        f"{name} 相机实际仅 {capture_hz:.2f} Hz，低于原始采集门禁 "
                        f"{self.raw_camera_hz * 0.90:.2f} Hz"
                    )
            osc = self.osc.state() or {}
            session = osc.get("session") or {}
            if session.get("state") != "ACTIVE":
                raise RuntimeError("OSC 控制会话未激活，禁止开始正式父 Episode")
            ruckig = (osc.get("diagnostics") or {}).get("ruckig")
            if isinstance(ruckig, dict) and ruckig.get("enabled") is False:
                raise RuntimeError(f"Ruckig 当前被旁路，不能开始采集: {ruckig.get('reason') or 'unknown reason'}")
            self._ensure_root()
            index = self._next_index()
            clock_offset_ns = self._monotonic_to_perf_offset_ns()
            started_ns = time.perf_counter_ns()
            drain_reader = getattr(self.osc, "sensor_samples_after", None)
            sample_reader = getattr(self.osc, "sensor_sample", None)
            if session.get("execution_mode") != "shadow" and callable(drain_reader):
                if not callable(sample_reader):
                    raise RuntimeError("OSC反馈生产队列缺少当前revision读取接口")
                anchor_sample = sample_reader(started_ns, 0.05)
                anchor_revision = (anchor_sample or {}).get("feedback_revision")
                if not isinstance(anchor_revision, int) or anchor_revision <= 0:
                    raise RuntimeError("无法固定采集开始时的OSC反馈revision")
                missing_feedback: list[str] = []
                if _finite_list((anchor_sample or {}).get("joint_position_rad"), 7) is None:
                    missing_feedback.append("7个关节角")
                if _finite_list((anchor_sample or {}).get("joint_velocity_rad_s"), 7) is None:
                    missing_feedback.append("7个关节速度")
                if self._ratio((anchor_sample or {}).get("gripper_width_m")) is None:
                    missing_feedback.append("夹爪开度")
                if _monotonic_ns((anchor_sample or {}).get("feedback_monotonic_ns")) is None:
                    missing_feedback.append("反馈时间戳")
                if _pose_components((anchor_sample or {}).get("measured_tcp_pose")) is None:
                    missing_feedback.append("实测TCP位姿")
                feedback_age, _ = self._feedback_age(anchor_sample or {}, clock_offset_ns)
                if feedback_age is None:
                    anchor_feedback_ns = _monotonic_ns((anchor_sample or {}).get("feedback_monotonic_ns"))
                    feedback_age = (max(0.0, (time.perf_counter_ns() - anchor_feedback_ns) / 1e9)
                                    if anchor_feedback_ns is not None else None)
                if feedback_age is not None and feedback_age > max(0.05, self.feedback_age_limit_s * 2.5):
                    missing_feedback.append(f"新鲜反馈（当前延迟{feedback_age:.3f}s）")
                if missing_feedback:
                    raise RuntimeError("机械臂原始数据门禁失败，缺少: " + "、".join(missing_feedback))
            else:
                anchor_revision = int((osc.get("execution") or {}).get("feedback_revision")
                                      or ((osc.get("transport") or {}).get("hardware_feedback") or {}).get("rx_revision") or 0)
            staging = self.root / "staging" / f"episode_{index:06d}_{uuid.uuid4().hex}"
            for camera in ("external", "wrist"):
                (staging / "raw" / "images" / camera).mkdir(parents=True, exist_ok=True)
            processing_interface = self._snapshot_processing_interface(staging)
            # Refresh the producer cursor immediately before starting. Frames
            # already present in the preview queue belong to pre-episode time.
            camera_state = self.cameras.snapshot() or camera_state
            self.active = {
                "schema": SCHEMA_VERSION, "dataset_stage": "native_rate_raw_collection",
                "episode_index": index, "episode_dir": str(staging), "status": "recording", "started_at": _utc(),
                "control_source": source, "operator_device": "pico_4_ultra" if source == "pico" else source,
                "operator_connection": "connected" if source == "pico" else "not_applicable",
                "task": task, "prompt": prompt,
                "raw_camera_frames": 0, "raw_camera_frames_by_source": {"external": 0, "wrist": 0},
                "raw_robot_states": 0, "bytes_written": 0,
                "sample_errors": 0, "last_error": None, "camera_sources": camera_snapshot,
                "training_view_generated": False, "horizon_windows_built": False,
                "processing_interface": processing_interface,
                "quality": {"training_eligible": False, "raw_valid": None, "reasons": ["recording_in_progress"]},
                "rejected_samples": 0, "rejection_reasons": {}, "backpressure_drops": 0,
                "_staging": staging, "_started_monotonic": time.monotonic(), "_started_ns": started_ns,
                "_hardware_writes": session.get("execution_mode") != "shadow",
                "_shadow": session.get("execution_mode") == "shadow",
                "_raw_camera_manifest": (staging / "raw" / "camera_frames.jsonl").open("x", encoding="utf-8", buffering=1),
                "_raw_robot_manifest": (staging / "raw" / "robot_states.jsonl").open("x", encoding="utf-8", buffering=1),
                "_raw_camera_last_ns": {}, "_raw_camera_sequence": 0, "_raw_camera_next_commit": 0,
                "_raw_camera_source_sequence": {"external": 0, "wrist": 0},
                "_raw_camera_producer_sequences": {key: int((camera_state.get("dataset_source_sequences") or {}).get(key, 0))
                                                   for key in ("external", "wrist")},
                "_raw_camera_source_gaps": {"external": 0, "wrist": 0},
                "_raw_camera_pending": {}, "_raw_camera_drops": 0,
                "_raw_camera_drops_by_source": {"external": 0, "wrist": 0}, "_raw_robot_drops": 0,
                "_raw_camera_prestart_ignored": {"external": 0, "wrist": 0}, "_raw_robot_prestart_ignored": 0,
                "_raw_feedback_revisions_seen": set(), "_raw_robot_revision_gaps": 0,
                "_monotonic_to_perf_offset_ns": clock_offset_ns, "_raw_feedback_ages": [],
                "_raw_camera_times": {"external": [], "wrist": []}, "_raw_robot_times": [],
                "_raw_robot_last_revision": anchor_revision,
                "_raw_executor": ThreadPoolExecutor(max_workers=2, thread_name_prefix="nero-raw-jpeg"),
            }
            self.stop_event = threading.Event()
            self.raw_camera_thread = threading.Thread(target=self._raw_camera_loop, name="nero-raw-camera-20hz", daemon=True)
            self.raw_robot_thread = threading.Thread(target=self._raw_robot_loop, name="nero-raw-robot-50hz", daemon=True)
            self.raw_camera_thread.start(); self.raw_robot_thread.start()
            return self._public(self.active, True)

    def _ratio(self, width_m: Any) -> float | None:
        width = _number(width_m)
        if width is None:
            return None
        return max(0.0, min(1.0, (width - self.gripper_closed_width_m) / (self.gripper_open_width_m - self.gripper_closed_width_m)))

    @staticmethod
    def _snapshot_processing_interface(staging: Path) -> dict[str, str]:
        """Keep exact model/config inputs needed by a later deterministic build."""
        project_root = Path(__file__).resolve().parents[1]
        interface_dir = staging / "raw" / "interface"
        interface_dir.mkdir(parents=True, exist_ok=True)
        sources = {
            "urdf": project_root / "vendor" / "nero_description" / "nero_description.urdf",
            "osc_config": project_root / "config" / "osc.json",
            "runtime_config": project_root / "config" / "runtime.json",
            "camera_config": project_root / "config" / "pi05.json",
        }
        result: dict[str, str] = {}
        for name, source in sources.items():
            if not source.is_file():
                continue
            destination = interface_dir / source.name
            shutil.copy2(source, destination)
            result[name] = str(destination.relative_to(staging)).replace("\\", "/")
        contract = {
            "schema_version": "nero.tcp-vla.raw-interface.v1",
            "clock": "monotonic_ns",
            "camera_stream": {"manifest": "raw/camera_frames.jsonl", "color": "RGB uint8", "shape": [480, 640, 3],
                              "sources": ["external", "wrist"], "pairing": "independent; align by timestamp in postprocessing"},
            "robot_stream": {"manifest": "raw/robot_states.jsonl", "joint_order": [f"joint{i}" for i in range(1, 8)],
                             "joint_unit": "rad", "velocity_unit": "rad/s", "tcp_frame": "robot_base",
                             "feedback_age": "SDK freshness converted from monotonic_ns to perf_counter_ns at collection start"},
            "future_training_view": {"rate_hz": 15.0, "pose_rotation": "Rot6D",
                                     "action": ["delta_x_m", "delta_y_m", "delta_z_m", "rotvec_x_rad", "rotvec_y_rad", "rotvec_z_rad", "absolute_gripper"],
                                     "horizon": 16, "replan_every": 8},
        }
        contract_path = interface_dir / "contract.json"
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["contract"] = str(contract_path.relative_to(staging)).replace("\\", "/")
        return result

    def _encode_raw_frame(self, staging: Path, source: str, source_index: int, frame: Any) -> dict[str, Any]:
        import cv2
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, payload = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.raw_jpeg_quality])
        if not ok:
            raise RuntimeError(f"{source} JPEG encoding failed")
        relative = Path("raw") / "images" / source / f"{source_index:06d}.jpg"
        data = payload.tobytes()
        (staging / relative).write_bytes(data)
        return {"path": str(relative).replace("\\", "/"), "bytes": len(data)}

    def _commit_raw_camera(self, active: dict[str, Any], wait: bool = False) -> None:
        pending: dict[int, tuple[dict[str, Any], Future[Any]]] = active["_raw_camera_pending"]
        while active["_raw_camera_next_commit"] in pending:
            sequence = active["_raw_camera_next_commit"]
            row, future = pending[sequence]
            if not future.done() and not wait:
                return
            try:
                encoded = future.result()
                row["image"] = encoded["path"]
                active["_raw_camera_manifest"].write(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                source = row["source"]
                active["raw_camera_frames_by_source"][source] += 1
                active["raw_camera_frames"] = min(active["raw_camera_frames_by_source"].values())
                active["bytes_written"] += encoded["bytes"]
            except Exception as exc:
                source = row.get("source")
                active["_raw_camera_drops"] += 1
                if source in active["_raw_camera_drops_by_source"]:
                    active["_raw_camera_drops_by_source"][source] += 1
                active["sample_errors"] += 1
                active["last_error"] = f"raw camera write: {type(exc).__name__}: {exc}"
            finally:
                pending.pop(sequence, None); active["_raw_camera_next_commit"] += 1

    def _queue_raw_camera_frame(self, active: dict[str, Any], source: str, stamp: int, frame: Any,
                                producer_sequence: int | None = None) -> None:
        target_ns = time.perf_counter_ns()
        if stamp < active["_started_ns"]:
            active["_raw_camera_prestart_ignored"][source] += 1
            return
        if source not in ("external", "wrist") or stamp <= active["_raw_camera_last_ns"].get(source, 0):
            active["_raw_camera_drops"] += 1
            if source in active["_raw_camera_drops_by_source"]: active["_raw_camera_drops_by_source"][source] += 1
            return
        if getattr(frame, "shape", None) != (480, 640, 3):
            active["_raw_camera_drops"] += 1; active["_raw_camera_drops_by_source"][source] += 1; return
        if len(active["_raw_camera_pending"]) >= 16:
            active["_raw_camera_drops"] += 1; active["_raw_camera_drops_by_source"][source] += 1
            active["backpressure_drops"] += 1; return
        sequence = active["_raw_camera_sequence"]
        source_index = active["_raw_camera_source_sequence"][source]
        active["_raw_camera_sequence"] += 1; active["_raw_camera_source_sequence"][source] += 1
        active["_raw_camera_last_ns"][source] = stamp
        row = {"schema_version": "nero.tcp-vla.raw-camera-frame.v2", "frame_index": sequence,
               "source": source, "source_frame_index": source_index,
               "producer_frame_id": producer_sequence, "capture_target_monotonic_ns": target_ns,
               "capture_monotonic_ns": stamp, "image": None}
        active["_raw_camera_pending"][sequence] = (
            row, active["_raw_executor"].submit(self._encode_raw_frame, active["_staging"], source, source_index, frame)
        )
        active["_raw_camera_times"][source].append(stamp)

    def _raw_camera_sample(self, active: dict[str, Any]) -> None:
        """Compatibility fallback for camera providers without a producer queue."""
        target_ns = time.perf_counter_ns()
        for source in ("external", "wrist"):
            selected = self.cameras.dataset_frame(source, target_ns)
            if selected is None:
                active["_raw_camera_drops"] += 1; active["_raw_camera_drops_by_source"][source] += 1; continue
            stamp, frame = selected; stamp = _monotonic_ns(stamp)
            if stamp is None:
                active["_raw_camera_drops"] += 1; active["_raw_camera_drops_by_source"][source] += 1; continue
            self._queue_raw_camera_frame(active, source, stamp, frame)

    def _raw_camera_loop(self) -> None:
        drain_reader = getattr(self.cameras, "dataset_frames_after", None)
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            active = self.active
            if active is not None:
                try:
                    self._commit_raw_camera(active)
                    if callable(drain_reader):
                        for source_index, source in enumerate(("external", "wrist")):
                            rows = drain_reader(source, active["_raw_camera_producer_sequences"][source],
                                                0.05 if source_index == 0 else 0.0, 8)
                            for row in rows:
                                producer_sequence = int(row["sequence"])
                                expected = active["_raw_camera_producer_sequences"][source] + 1
                                if producer_sequence > expected:
                                    active["_raw_camera_source_gaps"][source] += producer_sequence - expected
                                active["_raw_camera_producer_sequences"][source] = producer_sequence
                                self._queue_raw_camera_frame(active, source, int(row["timestamp"]), row["frame"],
                                                             producer_sequence)
                    else:
                        self._raw_camera_sample(active)
                    self._commit_raw_camera(active)
                except Exception as exc:
                    active["_raw_camera_drops"] += 1
                    active["last_error"] = f"raw camera: {type(exc).__name__}: {exc}"
            if not callable(drain_reader):
                deadline += self.raw_camera_period
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def _raw_robot_sample(self, active: dict[str, Any], supplied_sample: dict[str, Any] | None = None) -> None:
        target_ns = time.perf_counter_ns()
        sample = supplied_sample
        reader = getattr(self.osc, "sensor_sample", None)
        if sample is None and not active["_shadow"] and callable(reader):
            sample = reader(target_ns, 0.0)
        if isinstance(sample, dict):
            joints = _finite_list(sample.get("joint_position_rad"), 7)
            velocities = _finite_list(sample.get("joint_velocity_rad_s"), 7)
            feedback_ns = _monotonic_ns(sample.get("feedback_monotonic_ns"))
            revision = sample.get("feedback_revision")
            gripper = self._ratio(sample.get("gripper_width_m"))
            target_gripper = self._ratio(sample.get("gripper_target_width_m"))
            measured_tcp_pose, target_tcp_pose = sample.get("measured_tcp_pose"), sample.get("target_tcp_pose")
            identifiers = {key: sample.get(key) for key in ("control_sample_id", "target_generation", "motion_epoch")}
            extras = {key: sample.get(key) for key in ("sdk_fresh_monotonic_ns", "sdk_joint_timestamp", "joint_feedback_hz", "alignment_error_s", "latest_feedback_age_s",
                                                     "gripper_observed_monotonic_ns", "gripper_feedback_source", "gripper_feedback_error")}
        else:
            osc = self.osc.state() or {}
            execution, transport = osc.get("execution") or {}, osc.get("transport") or {}
            feedback, mailbox = transport.get("hardware_feedback") or {}, transport.get("feedback_mailbox") or {}
            joints = _finite_list(execution.get("measured_joint_state_rad") if active["_shadow"] else feedback.get("joint_angles_rad"), 7)
            velocities = _finite_list(execution.get("measured_joint_velocity_rad_s") if active["_shadow"] else feedback.get("joint_velocity_rad_s"), 7)
            # Compatibility providers expose the safety clock, which on this
            # Windows host is GetTickCount64-backed and only advances every
            # 15.625 ms.  Timestamp the fallback observation with QPC so the
            # raw stream remains strictly ordered and shares the camera clock.
            source_feedback_ns = _monotonic_ns(mailbox.get("received_monotonic_ns"))
            feedback_ns = target_ns
            revision = execution.get("feedback_revision") if active["_shadow"] else feedback.get("rx_revision")
            gripper = self._ratio((osc.get("gripper") or {}).get("width_m"))
            command = osc.get("command") or {}
            target_gripper = self._ratio(command.get("gripper_target_width_m"))
            if target_gripper is None:
                target_gripper = self._ratio((osc.get("active_action") or {}).get("width_m"))
            measured_tcp_pose = execution.get("measured_tcp_pose") or feedback.get("tcp_pose")
            target_tcp_pose = command.get("target_tcp") or command.get("target_pose")
            identifiers = {"control_sample_id": execution.get("control_sample_id", execution.get("sample_id")),
                           "target_generation": execution.get("target_generation", command.get("target_generation")),
                           "motion_epoch": execution.get("motion_epoch", command.get("epoch"))}
            extras = {"source_feedback_monotonic_ns": source_feedback_ns}
        missing = [name for name, value in (("joint_position_rad", joints), ("joint_velocity_rad_s", velocities),
                                            ("feedback_monotonic_ns", feedback_ns), ("gripper_width_m", gripper))
                   if value is None]
        if missing:
            active["_raw_robot_drops"] += 1
            active["rejected_samples"] += 1
            for name in missing:
                key = f"robot_missing:{name}"
                active["rejection_reasons"][key] = active["rejection_reasons"].get(key, 0) + 1
            active["last_error"] = "机械臂反馈缺少有效字段: " + ", ".join(missing)
            return
        if feedback_ns < active["_started_ns"]:
            # A producer-drain implementation must never turn retained
            # pre-episode history into samples for the new episode.
            active["_raw_robot_prestart_ignored"] += 1
            return
        if isinstance(sample, dict):
            feedback_age_s, fresh_perf_ns = self._feedback_age(sample, active["_monotonic_to_perf_offset_ns"])
        else:
            feedback_age_s, fresh_perf_ns = 0.0, feedback_ns
        if feedback_age_s is None:
            feedback_age_s = max(0.0, (target_ns - feedback_ns) / 1e9)
        extras["latest_feedback_age_s"] = feedback_age_s
        extras["feedback_fresh_perf_counter_ns"] = fresh_perf_ns
        extras["feedback_recording_delay_s"] = max(0.0, (target_ns - feedback_ns) / 1e9)
        row = {
            "schema_version": "nero.tcp-vla.raw-robot.v1", "sample_index": active["raw_robot_states"],
            "sample_target_monotonic_ns": target_ns, "feedback_monotonic_ns": feedback_ns,
            "feedback_revision": revision, "joint_position_rad": joints, "joint_velocity_rad_s": velocities,
            "gripper_opening_ratio": gripper,
            "target_gripper_opening_ratio": gripper if target_gripper is None else target_gripper,
            "measured_tcp_pose": measured_tcp_pose,
            "target_tcp_pose": target_tcp_pose, "prompt": active["prompt"], "identifiers": identifiers, **extras,
        }
        active["_raw_robot_manifest"].write(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        active["raw_robot_states"] += 1
        active["_raw_robot_times"].append(feedback_ns)
        active["_raw_feedback_ages"].append(feedback_age_s)
        if isinstance(revision, int):
            active["_raw_feedback_revisions_seen"].add(revision)

    def _raw_robot_loop(self) -> None:
        drain_reader = getattr(self.osc, "sensor_samples_after", None)
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            active = self.active
            if active is not None:
                try:
                    if not active["_shadow"] and callable(drain_reader):
                        rows = drain_reader(active["_raw_robot_last_revision"], 0.05, 128)
                        for row in rows:
                            revision = int(row.get("feedback_revision") or 0)
                            expected = active["_raw_robot_last_revision"] + 1
                            if revision > expected:
                                active["_raw_robot_revision_gaps"] += revision - expected
                            active["_raw_robot_last_revision"] = max(active["_raw_robot_last_revision"], revision)
                            self._raw_robot_sample(active, row)
                    else:
                        self._raw_robot_sample(active)
                except Exception as exc:
                    active["_raw_robot_drops"] += 1
                    active["last_error"] = f"raw robot: {type(exc).__name__}: {exc}"
            if active is None or active["_shadow"] or not callable(drain_reader):
                deadline += self.raw_robot_state_period
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def _reconstruction_audit(self, camera_times: dict[str, list[int]], robot_times: list[int]) -> dict[str, Any]:
        result = {"grid_points": 0, "valid_grid_points": 0, "coverage": 0.0,
                  "longest_contiguous_points": 0, "possible_h16_windows": 0}
        if any(len(camera_times.get(source, [])) < 2 for source in ("external", "wrist")) or len(robot_times) < 2:
            return result
        target = max(camera_times["external"][0], camera_times["wrist"][0], robot_times[0])
        end = min(camera_times["external"][-1], camera_times["wrist"][-1], robot_times[-1])
        period_ns = round(1e9 / self.training_view_hz)
        mask: list[bool] = []
        while target <= end:
            camera_errors = []
            for source in ("external", "wrist"):
                times = camera_times[source]
                camera_after = bisect.bisect_left(times, target)
                candidates = [item for item in (camera_after - 1, camera_after) if 0 <= item < len(times)]
                camera_errors.append(min((abs(times[item] - target) for item in candidates), default=10**18))
            robot_after = bisect.bisect_left(robot_times, target)
            robot_ok = 0 < robot_after < len(robot_times)
            robot_error = (max(target - robot_times[robot_after - 1], robot_times[robot_after] - target)
                           if robot_ok else 10**18)
            mask.append(
                max(camera_errors) <= round(self.training_camera_alignment_limit_s * 1e9)
                and robot_error <= round(self.training_robot_bracket_limit_s * 1e9)
            )
            target += period_ns
        runs: list[int] = []; current = 0
        for valid in mask:
            if valid:
                current += 1
            elif current:
                runs.append(current); current = 0
        if current:
            runs.append(current)
        return {"grid_points": len(mask), "valid_grid_points": sum(mask),
                "coverage": sum(mask) / len(mask) if mask else 0.0,
                "longest_contiguous_points": max(runs, default=0),
                "possible_h16_windows": sum(max(0, run - 16) for run in runs)}

    def stop(self, status: str = "completed", reason: str = "") -> dict[str, Any]:
        with self.lock:
            if not self.active:
                return self.state()
            active = self.active; self.stop_event.set()
        for thread in (self.raw_camera_thread, self.raw_robot_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=3)
        stopped_monotonic = time.monotonic()
        stopped_ns = time.perf_counter_ns()
        with self.lock:
            active = self.active
            self._commit_raw_camera(active, wait=True)
            active["_raw_executor"].shutdown(wait=True)
            active["_raw_camera_manifest"].flush(); active["_raw_camera_manifest"].close()
            active["_raw_robot_manifest"].flush(); active["_raw_robot_manifest"].close()
            duration = max(0.0, stopped_monotonic - active["_started_monotonic"])
            active["duration_s"] = duration
            if sum(active["raw_camera_frames_by_source"].values()) == 0 and active["raw_robot_states"] == 0:
                shutil.rmtree(active["_staging"])
                active["episode_dir"] = None; active["status"] = "discarded_empty"
                active["quality"].update(training_eligible=False, raw_valid=False, reasons=["no_raw_camera_or_robot_samples"])
                result = self._public(active, False); result["discarded"] = True
                self.last_episode = copy.deepcopy(result); self.active = None
                self.raw_camera_thread = None; self.raw_robot_thread = None
                return result
            camera_span_s = {source: ((times[-1] - times[0]) / 1e9 if len(times) > 1 else 0.0)
                             for source, times in active["_raw_camera_times"].items()}
            robot_span_s = ((active["_raw_robot_times"][-1] - active["_raw_robot_times"][0]) / 1e9
                            if len(active["_raw_robot_times"]) > 1 else 0.0)
            camera_rates = {source: ((len(times) - 1) / camera_span_s[source]
                                     if camera_span_s[source] > 0.0 else 0.0)
                            for source, times in active["_raw_camera_times"].items()}
            camera_rate = min(camera_rates.values())
            robot_rate = ((len(active["_raw_robot_times"]) - 1) / robot_span_s
                          if robot_span_s > 0.0 else 0.0)
            camera_timestamp_nonmonotonic = {
                source: sum(right <= left for left, right in zip(times, times[1:]))
                for source, times in active["_raw_camera_times"].items()
            }
            robot_timestamp_nonmonotonic = sum(
                right <= left for left, right in zip(active["_raw_robot_times"], active["_raw_robot_times"][1:])
            )
            reconstruction = self._reconstruction_audit(
                {source: sorted(set(times)) for source, times in active["_raw_camera_times"].items()},
                sorted(set(active["_raw_robot_times"]))
            )
            reasons: list[str] = []
            feedback_age_gate_s = max(0.05, self.feedback_age_limit_s)
            feedback_age_above_limit = sum(age > feedback_age_gate_s for age in active["_raw_feedback_ages"])
            if any(count == 0 for count in active["raw_camera_frames_by_source"].values()) or active["raw_robot_states"] == 0:
                reasons.append("missing_required_raw_stream")
            if active["_raw_robot_drops"]:
                reasons.append("raw_robot_samples_rejected")
            if duration >= 2.0 and camera_rate < self.raw_camera_hz * 0.90:
                reasons.append("raw_camera_rate_below_90_percent")
            if duration >= 2.0 and robot_rate < self.raw_robot_state_hz * 0.90:
                reasons.append("raw_robot_state_rate_below_90_percent")
            if active["sample_errors"]:
                reasons.append("raw_stream_errors_present")
            if any(active["_raw_camera_source_gaps"].values()):
                reasons.append("raw_camera_source_gaps_present")
            if active["_raw_robot_revision_gaps"]:
                reasons.append("raw_robot_revision_gaps_present")
            if feedback_age_above_limit:
                reasons.append("raw_feedback_age_above_limit")
            if any(camera_timestamp_nonmonotonic.values()):
                reasons.append("raw_camera_timestamps_not_strictly_increasing")
            if robot_timestamp_nonmonotonic:
                reasons.append("raw_robot_timestamps_not_strictly_increasing")
            if duration >= 2.0 and reconstruction["coverage"] < 0.95:
                reasons.append("15hz_reconstruction_coverage_below_95_percent")
            if duration >= 2.0 and reconstruction["grid_points"] >= 17 and reconstruction["possible_h16_windows"] < 1:
                reasons.append("no_contiguous_h16_window_reconstructable")
            accepted = status == "completed" and not reasons
            active["quality"].update(training_eligible=False, raw_valid=accepted, reasons=reasons)
            metadata = {
                "schema_version": SCHEMA_VERSION, "dataset_stage": "native_rate_raw_collection",
                "parent_episode_index": active["episode_index"], "accepted": accepted,
                "failure": None if accepted else (reason or "; ".join(reasons) or "operator marked episode failed"),
                "instruction_mode": "manual_per_episode", "task": active["task"], "prompt": active["prompt"],
                "segment_order": [active["task"]], "segments": [],
                "gates": {name: False for name in PHYSICAL_GATES}, "gate_source": "not_evaluated_during_raw_collection",
                "files": {"raw_camera": "raw/camera_frames.jsonl", "raw_robot_state": "raw/robot_states.jsonl"},
                "videos": {}, "depth_recorded": False, "raw_collection": True,
                "training_view_generated": False, "horizon_windows_built": False,
                "recording_hz": None, "training_view_hz": self.training_view_hz,
                "sensor_frames": None, "control_steps": None,
                "action_alignment": "deferred until 15 Hz training-view build", "duration_s": duration,
                "collection_monotonic_ns": {"start": active["_started_ns"], "end": stopped_ns},
                "raw_rates_hz": {"camera_requested": self.raw_camera_hz, "robot_state_requested": self.raw_robot_state_hz,
                                 "camera_effective": camera_rate, "camera_effective_by_source": camera_rates,
                                 "robot_state_effective": robot_rate},
                "raw_streams": {"camera_frames": active["raw_camera_frames"], "robot_states": active["raw_robot_states"],
                                "camera_frames_by_source": active["raw_camera_frames_by_source"],
                                "camera_total_frames": sum(active["raw_camera_frames_by_source"].values()),
                                "camera_drops": active["_raw_camera_drops"], "robot_state_drops": active["_raw_robot_drops"],
                                "camera_drops_by_source": active["_raw_camera_drops_by_source"],
                                "camera_prestart_ignored": active["_raw_camera_prestart_ignored"],
                                "camera_source_gaps": active["_raw_camera_source_gaps"],
                                "camera_pairing": "deferred_to_postprocessing",
                                "robot_revision_gaps": active["_raw_robot_revision_gaps"],
                                "robot_prestart_ignored": active["_raw_robot_prestart_ignored"],
                                "camera_timestamp_nonmonotonic": camera_timestamp_nonmonotonic,
                                "robot_timestamp_nonmonotonic": robot_timestamp_nonmonotonic,
                                "unique_feedback_revisions": len(active["_raw_feedback_revisions_seen"]),
                                "feedback_age_clock": "perf_counter_ns",
                                "monotonic_to_perf_offset_ns": active["_monotonic_to_perf_offset_ns"],
                                "feedback_age_s": {
                                    "maximum": max(active["_raw_feedback_ages"], default=None),
                                    "mean": (sum(active["_raw_feedback_ages"]) / len(active["_raw_feedback_ages"])
                                             if active["_raw_feedback_ages"] else None),
                                    "limit": feedback_age_gate_s,
                                    "above_limit": feedback_age_above_limit,
                                }},
                "reconstruction_15hz": reconstruction,
                "training_alignment_limits_s": {
                    "camera_nearest": self.training_camera_alignment_limit_s,
                    "robot_bracket": self.training_robot_bracket_limit_s,
                },
                "derived_contract": {"rate_hz": self.training_view_hz,
                                     "state": "timestamp-aligned joints/gripper plus pinned Pinocchio FK and Rot6D",
                                     "action": "adjacent measured TCP relative SE(3) plus absolute gripper",
                                     "horizon": 16, "replan_every": 8},
                "sample_errors": active["sample_errors"], "last_error": active["last_error"],
                "rejection_reasons": active["rejection_reasons"],
                "finalized_unix_s": time.time(), "hardware_writes": bool(active["_hardware_writes"]),
                "camera_snapshot": active["camera_sources"],
                "processing_interface": active["processing_interface"],
                "gripper_calibration_m": {"closed": self.gripper_closed_width_m, "open": self.gripper_open_width_m},
                "quality": active["quality"],
            }
            (active["_staging"] / "episode.json").write_text(
                json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            category = "episodes" if accepted else "failed_episodes"
            destination = self.root / category / f"episode_{active['episode_index']:06d}"
            if destination.exists():
                raise FileExistsError(f"拒绝覆盖既有 Episode: {destination}")
            active["_staging"].replace(destination)
            active["episode_dir"], active["status"] = str(destination), "completed" if accepted else "failed"
            result = self._public(active, False)
            self.last_episode = copy.deepcopy(result); self.active = None
            self.raw_camera_thread = None; self.raw_robot_thread = None
            return result

    def close(self) -> None:
        self.stop("failed", "control service shutdown")

    def episodes(self) -> dict[str, Any]:
        self._ensure_root(); items = []
        for category in ("episodes", "failed_episodes"):
            for path in sorted((self.root / category).glob("episode_*/episode.json")):
                try:
                    item = json.loads(path.read_text(encoding="utf-8")); item["category"] = category; items.append(item)
                except (OSError, json.JSONDecodeError):
                    items.append({"episode_dir": str(path.parent), "status": "invalid", "category": category})
        return {"dataset_root": str(self.root), "episodes": items[-100:]}
