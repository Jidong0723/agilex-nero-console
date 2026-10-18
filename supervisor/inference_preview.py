"""Read-only observations and idle predictions for the model input source."""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np

from .inference_contract import action_rows, metadata_compatibility


class ObservationSource:
    def __init__(self, adapter):
        self.a = adapter
        self.fk = None
        self.last_joints = None
        self.last_pose = None
        self.gripper_marker = None
        self.gripper_seen_ns = 0

    def refresh(self, snapshot):
        """Read cached feedback; never acquire a lease or send a command."""
        reader = getattr(self.a.osc, "sensor_sample", None)
        if not callable(reader):
            return snapshot  # Direct fake clients retain their explicit samples.
        now = time.perf_counter_ns()
        shadow = (snapshot.get("session") or {}).get("execution_mode") == "shadow"
        if shadow:
            execution = snapshot.get("execution") or {}
            plant = (snapshot.get("diagnostics") or {}).get("shadow_transport") or {}
            joints = plant.get("applied_joint_state_rad") or execution.get("measured_joint_state_rad")
            if not plant.get("enabled"):
                raise RuntimeError("影子反馈不可用")
            stamp, age, source = now, 0., "shadow_joint_fk"
        else:
            sample = reader(now, 0.)
            if not sample:
                raise RuntimeError("等待实测关节反馈")
            joints = sample.get("joint_position_rad")
            stamp = sample.get("feedback_monotonic_ns")
            age = float(sample.get("latest_feedback_age_s", math.inf))
            if not isinstance(stamp, int) or not 0 <= now-stamp <= 100_000_000 or age > .1:
                raise RuntimeError("实测关节反馈过期")
            source = "measured_joint_fk"
        values = np.asarray(joints, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise RuntimeError("实测关节反馈无效")
        key = tuple(values)
        if key != self.last_joints:
            from motion.osc import KinematicsClient, pose_from_tcp
            if self.fk is None:
                root = self.a.config_path.parent.parent
                config = json.loads((root/"config/osc.json").read_text(encoding="utf-8"))
                self.fk = KinematicsClient(root, config)
            # Start outside fk()/submit()'s client lock: the reader needs that
            # lock before publishing its startup message.
            self.fk.start()
            self.last_pose = pose_from_tcp(self.fk.fk(list(key)))
            self.last_joints = key
        gripper = copy.deepcopy(snapshot.get("gripper") or {})
        if not shadow:
            feedback = self.a.osc.inference_gripper_feedback()
            gripper.update(width_m=feedback.get("width_m"), raw=feedback.get("raw"))
            marker = ((feedback.get('raw') or {}).get('value') or {}).get('timestamp')
            if marker is None:
                raise RuntimeError('等待带时间戳的实测夹爪反馈')
            if marker != self.gripper_marker:
                self.gripper_marker, self.gripper_seen_ns = marker, time.perf_counter_ns()
            if time.perf_counter_ns()-self.gripper_seen_ns > 150_000_000:
                raise RuntimeError('实测夹爪反馈已停更超过150 ms')
        width = gripper.get("width_m")
        if width is None or not math.isfinite(float(width)):
            raise RuntimeError("等待实测夹爪开度")
        snapshot["inference_observation"] = {"pose": copy.deepcopy(self.last_pose), "perf_ns": stamp,
            "feedback_age_s": age, "source": source, "gripper_width_m": float(width)}
        snapshot["gripper"] = gripper
        return snapshot

    def close(self):
        if self.fk:
            self.fk.close()


def preview_once(a):
    """One serial ordinary sampling request. It has no control-facing calls."""
    with a.lock:
        if a.state.get("execution_enabled") or a.state.get("priming") or a._changing_connection:
            return
        if a._rtc_run and not a._rtc_run.stop_confirmed:
            return
        revision = (a._connection_revision, a._run_revision)
        osc = copy.deepcopy(a._osc_snapshot)
        config = copy.deepcopy(a.config)
        a._preview_busy = True
        a.state["preview"].update(state="WAITING", error=None)
    policy = None
    response = None
    try:
        resource = a.camera_resource
        if resource is not None:
            with resource.lock:
                frames = [resource.frames.get(k) for k in ("external", "wrist")]
                stamps = [resource.frame_times_ns.get(k, 0) for k in ("external", "wrist")]
            now = time.perf_counter_ns()
            if any(f is None for f in frames) or any(not 0 <= now-t <= 150_000_000 for t in stamps):
                raise RuntimeError("等待新鲜双路相机图像（150 ms以内）")
            if abs(stamps[0]-stamps[1]) > 75_000_000:
                raise RuntimeError("双路图像时间差超过75 ms")
        elif a.cameras:
            frames = a.cameras.read()
        else:
            raise RuntimeError("请先打开双路相机")
        measured = osc.get("inference_observation")
        if measured:
            if not 0 <= time.perf_counter_ns()-measured["perf_ns"] <= 100_000_000:
                raise RuntimeError("等待新鲜实测TCP")
            osc["execution"] = dict(osc.get("execution") or {}, measured_tcp_pose=measured["pose"])
            osc["gripper"] = dict(osc.get("gripper") or {}, width_m=measured["gripper_width_m"])
        elif callable(getattr(a.osc, "sensor_sample", None)):
            raise RuntimeError(a.state.get("observation_wait_reason") or "等待实测TCP与夹爪")
        pose = (osc.get("execution") or {}).get("measured_tcp_pose")
        if not pose or (osc.get("gripper") or {}).get("width_m") is None:
            raise RuntimeError("等待实测TCP与夹爪；预览不使用占位值")
        observation = a._observation(osc, *frames)
        policy = a._policy_client()
        compatibility = metadata_compatibility(getattr(policy, 'metadata', {}), config["execution"]["expected_chunk_steps"])
        if compatibility["status"] == "incompatible":
            raise RuntimeError(compatibility["message"])
        with a.lock:
            if revision != (a._connection_revision, a._run_revision):
                return
            a.state["preview"].update(state="INFERENCING")
        began = time.perf_counter()
        response = policy.infer(observation)
        rows = action_rows(response, config["execution"]["expected_chunk_steps"])
        decoded = [a._target_from_action(row, pose, action_index=i)
                   for i, row in enumerate(rows)]
        ended = time.perf_counter()
        with a.lock:
            if revision != (a._connection_revision, a._run_revision) or a.state.get("priming") or a.state.get("execution_enabled"):
                return
            preview = a.state["preview"]
            previous = preview.get("last_completed_perf_s")
            preview.update(state="READY", error=None, updated_at=time.time(), last_completed_perf_s=ended,
                hz=1/(ended-previous) if previous and ended>previous else None,
                source="ordinary_h20_preview" if config["execution"].get("mode") == "rtc" else "ordinary_preview",
                sequence=preview.get("sequence", 0)+1, stale=False)
            a.state.update(action_chunk=rows, absolute_tcp_chunk=[x[0] for x in decoded],
                inference_base_tcp=copy.deepcopy(pose), decoded_first_target=decoded[0][0],
                action_chunk_length=len(rows), last_response_steps=len(rows), chunk_sequence=a.state["chunk_sequence"]+1,
                inference_ms=(ended-began)*1000, chunk_source="preview", chunk_updated_at=time.time(),
                protocol_compatibility={"status": "compatible", "message": "7D／H实际动作响应验证通过"})
    except Exception as exc:
        with a.lock:
            current = revision == (a._connection_revision, a._run_revision)
            if current:
                a.state["preview"].update(state="WAITING" if policy is None else "ERROR", error=str(exc), stale=True)
        if current and policy is not None and response is None:
            a._record_websocket_event('inference_failure', str(exc))
            a._invalidate_policy(policy, f"Preview connection failed: {exc}")
    finally:
        with a.lock:
            a._preview_busy = False
        a.stop_event.wait(.05)
