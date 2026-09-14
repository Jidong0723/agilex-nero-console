"""Read-only validator for a raw NERO TCP-VLA episode.

This validator checks the native camera and measured robot streams. It does
not create 15 Hz samples, TCP action labels, or H16 windows.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervisor.dataset_recorder import _pose_components, _rotvec_between
from supervisor.tcp_vla_dataset_recorder import SCHEMA_VERSION


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    result = {
        "path": str(path),
        "opened": opened,
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0,
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) if opened else 0,
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else 0,
        "fps": float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
        "bytes": path.stat().st_size if path.is_file() else 0,
    }
    ok, _ = capture.read() if opened else (False, None)
    result["first_frame_decodable"] = bool(ok)
    capture.release()
    return result


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    metadata_path = episode_dir / "episode.json"
    steps_path = episode_dir / "steps.jsonl"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"episode_dir": str(episode_dir), "structural_valid": False, "training_ready": False,
                "errors": [f"episode_json_invalid: {exc}"], "warnings": [], "stats": {}}

    if metadata.get("schema_version") not in {SCHEMA_VERSION, "nero.tcp-vla.episode.v2"}:
        errors.append("schema_version_mismatch")
    if metadata.get("raw_collection") is not True:
        errors.append("raw_collection_flag_missing")
    if metadata.get("horizon_windows_built") is not False:
        errors.append("raw_episode_must_not_contain_horizon_windows")
    if not isinstance(metadata.get("prompt"), str) or not metadata["prompt"].strip():
        errors.append("prompt_missing")

    if metadata.get("dataset_stage") == "native_rate_raw_collection":
        camera_rows: list[dict[str, Any]] = []
        robot_rows: list[dict[str, Any]] = []
        camera_path = episode_dir / str((metadata.get("files") or {}).get("raw_camera", "raw/camera_frames.jsonl"))
        robot_path = episode_dir / str((metadata.get("files") or {}).get("raw_robot_state", "raw/robot_states.jsonl"))
        try:
            camera_rows = [json.loads(line) for line in camera_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"raw_camera_manifest_invalid:{exc}")
        try:
            robot_rows = [json.loads(line) for line in robot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"raw_robot_manifest_invalid:{exc}")
        if not camera_rows:
            errors.append("no_raw_camera_frames")
        if not robot_rows:
            errors.append("no_raw_robot_states")
        independent_cameras = any(row.get("source") in ("external", "wrist") for row in camera_rows)
        camera_rows_by_source: dict[str, list[dict[str, Any]]] = {"external": [], "wrist": []}
        for index, row in enumerate(camera_rows):
            if row.get("frame_index") != index:
                errors.append(f"raw_camera:{index}:frame_index_not_sequential")
            if independent_cameras:
                source = row.get("source")
                if source not in camera_rows_by_source:
                    errors.append(f"raw_camera:{index}:source_invalid"); continue
                source_index = len(camera_rows_by_source[source])
                if row.get("source_frame_index") != source_index:
                    errors.append(f"raw_camera:{index}:source_frame_index_not_sequential")
                stamp, relative = row.get("capture_monotonic_ns"), row.get("image")
                if not isinstance(stamp, int) or stamp <= 0:
                    errors.append(f"raw_camera:{index}:capture_timestamp_invalid")
                camera_rows_by_source[source].append(row)
                path = episode_dir / str(relative or "")
                if not relative or not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"raw_camera:{index}:{source}_image_missing")
            else:
                images = row.get("images") or {}
                for source in ("external", "wrist"):
                    key = f"{source}_monotonic_ns"
                    if not isinstance(row.get(key), int) or row[key] <= 0:
                        errors.append(f"raw_camera:{index}:{key}_invalid")
                    relative = images.get(source)
                    path = episode_dir / str(relative or "")
                    if not relative or not path.is_file() or path.stat().st_size == 0:
                        errors.append(f"raw_camera:{index}:{source}_image_missing")
                    camera_rows_by_source[source].append({"source_frame_index": index,
                                                          "capture_monotonic_ns": row.get(key),
                                                          "producer_frame_id": row.get("producer_frame_id")})
        for index, row in enumerate(robot_rows):
            if row.get("sample_index") != index:
                errors.append(f"raw_robot:{index}:sample_index_not_sequential")
            if _finite_vector(row.get("joint_position_rad"), 7) is None:
                errors.append(f"raw_robot:{index}:joint_position_invalid")
            if _finite_vector(row.get("joint_velocity_rad_s"), 7) is None:
                errors.append(f"raw_robot:{index}:joint_velocity_invalid")
            gripper = row.get("gripper_opening_ratio")
            if not isinstance(gripper, (int, float)) or isinstance(gripper, bool) or not math.isfinite(float(gripper)) or not 0.0 <= float(gripper) <= 1.0:
                errors.append(f"raw_robot:{index}:gripper_invalid")
            target_gripper = row.get("target_gripper_opening_ratio")
            if not isinstance(target_gripper, (int, float)) or isinstance(target_gripper, bool) or not math.isfinite(float(target_gripper)) or not 0.0 <= float(target_gripper) <= 1.0:
                errors.append(f"raw_robot:{index}:target_gripper_invalid")
            if not isinstance(row.get("feedback_monotonic_ns"), int) or row["feedback_monotonic_ns"] <= 0:
                errors.append(f"raw_robot:{index}:feedback_timestamp_invalid")
            if row.get("prompt") != metadata.get("prompt"):
                errors.append(f"raw_robot:{index}:prompt_mismatch")
            if _pose_components(row.get("measured_tcp_pose")) is None:
                errors.append(f"raw_robot:{index}:measured_tcp_pose_invalid")
        stream_meta = metadata.get("raw_streams") or {}
        if stream_meta.get("feedback_age_clock") == "perf_counter_ns":
            for index, row in enumerate(robot_rows):
                age = row.get("latest_feedback_age_s")
                delay = row.get("feedback_recording_delay_s")
                fresh_ns = row.get("feedback_fresh_perf_counter_ns")
                if not isinstance(age, (int, float)) or isinstance(age, bool) or not math.isfinite(float(age)) or age < 0:
                    errors.append(f"raw_robot:{index}:feedback_age_invalid")
                if not isinstance(delay, (int, float)) or isinstance(delay, bool) or not math.isfinite(float(delay)) or delay < 0:
                    errors.append(f"raw_robot:{index}:feedback_recording_delay_invalid")
                if not isinstance(fresh_ns, int) or fresh_ns <= 0:
                    errors.append(f"raw_robot:{index}:feedback_fresh_timestamp_invalid")
        camera_counts = {source: len(rows) for source, rows in camera_rows_by_source.items()}
        if independent_cameras:
            if stream_meta.get("camera_frames_by_source") != camera_counts:
                errors.append("raw_camera_count_mismatch")
            if stream_meta.get("camera_total_frames") != len(camera_rows):
                errors.append("raw_camera_total_count_mismatch")
        elif stream_meta.get("camera_frames") != len(camera_rows):
            errors.append("raw_camera_count_mismatch")
        if stream_meta.get("robot_states") != len(robot_rows):
            errors.append("raw_robot_count_mismatch")
        if metadata.get("training_view_generated") is not False:
            errors.append("raw_episode_training_view_flag_invalid")
        interface = metadata.get("processing_interface") or {}
        for name in ("urdf", "osc_config", "runtime_config", "camera_config", "contract"):
            relative = interface.get(name)
            if not relative or not (episode_dir / relative).is_file():
                errors.append(f"processing_interface_missing:{name}")
        for forbidden in ("steps.jsonl", "observations.jsonl"):
            if (episode_dir / forbidden).exists():
                errors.append(f"derived_file_present_in_raw_episode:{forbidden}")
        camera_times = {source: [int(row["capture_monotonic_ns"]) for row in rows
                                 if isinstance(row.get("capture_monotonic_ns"), int)]
                        for source, rows in camera_rows_by_source.items()}
        robot_times = [int(row["feedback_monotonic_ns"]) for row in robot_rows if isinstance(row.get("feedback_monotonic_ns"), int)]
        collection_bounds = metadata.get("collection_monotonic_ns") or {}
        collection_start = collection_bounds.get("start")
        collection_end = collection_bounds.get("end")
        if isinstance(collection_start, int):
            if any(stamp < collection_start for times in camera_times.values() for stamp in times):
                errors.append("raw_camera_contains_prestart_samples")
            if any(stamp < collection_start for stamp in robot_times):
                errors.append("raw_robot_contains_prestart_samples")
        if isinstance(collection_end, int):
            if any(stamp > collection_end for times in camera_times.values() for stamp in times):
                errors.append("raw_camera_contains_poststop_samples")
            if any(stamp > collection_end for stamp in robot_times):
                errors.append("raw_robot_contains_poststop_samples")
        revisions = [row.get("feedback_revision") for row in robot_rows]
        producer_ids = {source: [row.get("producer_frame_id") for row in rows]
                        for source, rows in camera_rows_by_source.items()}
        camera_skews = [
            abs(int(row["external_monotonic_ns"]) - int(row["wrist_monotonic_ns"])) / 1e9
            for row in camera_rows
            if isinstance(row.get("external_monotonic_ns"), int) and isinstance(row.get("wrist_monotonic_ns"), int)
        ]
        for source, times in camera_times.items():
            if len(times) > 1 and any(right <= left for left, right in zip(times, times[1:])):
                errors.append(f"raw_camera_{source}_timestamps_not_strictly_increasing")
        if len(robot_times) > 1 and any(right <= left for left, right in zip(robot_times, robot_times[1:])):
            errors.append("raw_robot_timestamps_not_strictly_increasing")
        # Independently started writer threads can differ by a few cycles, but
        # a multi-second lead means retained producer history leaked into the
        # episode (the failure observed in episode_000012).
        camera_start = max((times[0] for times in camera_times.values() if times), default=None)
        if camera_start is not None and robot_times:
            if robot_times[0] < camera_start - 250_000_000:
                errors.append("raw_robot_stream_contains_pre_camera_history")
            if camera_start < robot_times[0] - 250_000_000:
                errors.append("raw_camera_stream_contains_pre_robot_history")
        if len(revisions) > 1 and any(not isinstance(left, int) or not isinstance(right, int) or right != left + 1
                                      for left, right in zip(revisions, revisions[1:])):
            errors.append("raw_feedback_revisions_not_contiguous")
        for source, ids in producer_ids.items():
            present_producer_ids = [item for item in ids if isinstance(item, int)]
            if present_producer_ids and any(right != left + 1 for left, right in zip(present_producer_ids, present_producer_ids[1:])):
                errors.append(f"raw_camera_{source}_producer_ids_not_contiguous")
        camera_sync_limit_s = float(stream_meta.get("camera_sync_limit_s") or 0.020)
        if not independent_cameras and any(skew > camera_sync_limit_s for skew in camera_skews):
            errors.append("raw_camera_pair_skew_above_limit")
        duration = float(metadata.get("duration_s") or 0.0)
        reconstruction = {"grid_points": 0, "valid_grid_points": 0, "coverage": None,
                          "longest_contiguous_points": 0, "possible_h16_windows": 0}
        if duration >= 2.0 and all(len(times) >= 2 for times in camera_times.values()) and len(robot_times) >= 2:
            target = max(camera_times["external"][0], camera_times["wrist"][0], robot_times[0])
            end = min(camera_times["external"][-1], camera_times["wrist"][-1], robot_times[-1])
            period_ns = round(1e9 / float(metadata.get("training_view_hz") or 15.0))
            valid_mask: list[bool] = []
            limits = metadata.get("training_alignment_limits_s") or {}
            camera_limit_ns = round(float(limits.get("camera_nearest", 0.030)) * 1e9)
            robot_limit_ns = round(float(limits.get("robot_bracket", 0.035)) * 1e9)
            while target <= end:
                camera_errors = []
                for times in camera_times.values():
                    camera_after = bisect.bisect_left(times, target)
                    camera_candidates = [item for item in (camera_after - 1, camera_after) if 0 <= item < len(times)]
                    camera_errors.append(min((abs(times[item] - target) for item in camera_candidates), default=10**18))
                robot_after = bisect.bisect_left(robot_times, target)
                robot_ok = 0 < robot_after < len(robot_times)
                robot_error = (max(target - robot_times[robot_after - 1], robot_times[robot_after] - target)
                               if robot_ok else 10**18)
                valid_mask.append(max(camera_errors) <= camera_limit_ns and robot_error <= robot_limit_ns)
                target += period_ns
            runs: list[int] = []; current = 0
            for valid in valid_mask:
                if valid:
                    current += 1
                elif current:
                    runs.append(current); current = 0
            if current:
                runs.append(current)
            reconstruction = {"grid_points": len(valid_mask), "valid_grid_points": sum(valid_mask),
                              "coverage": sum(valid_mask) / len(valid_mask) if valid_mask else 0.0,
                              "longest_contiguous_points": max(runs, default=0),
                              "possible_h16_windows": sum(max(0, run - 16) for run in runs)}
            if reconstruction["coverage"] < 0.95:
                errors.append("15hz_reconstruction_coverage_below_95_percent")
            if len(valid_mask) >= 17 and reconstruction["possible_h16_windows"] < 1:
                errors.append("no_contiguous_h16_window_reconstructable")
        structural_valid = not errors
        raw_valid = structural_valid and metadata.get("accepted") is True and (metadata.get("quality") or {}).get("raw_valid") is True
        return {
            "episode_dir": str(episode_dir), "schema_version": metadata.get("schema_version"),
            "dataset_stage": metadata.get("dataset_stage"), "structural_valid": structural_valid,
            "raw_valid": raw_valid, "training_ready": False, "errors": errors, "warnings": warnings,
            "stats": {"raw_camera_frames": min(camera_counts.values()),
                      "raw_camera_total_frames": len(camera_rows), "raw_camera_frames_by_source": camera_counts,
                      "raw_robot_states": len(robot_rows),
                      "raw_rates_hz": metadata.get("raw_rates_hz"), "training_view_generated": False,
                      "camera_skew_s": {"maximum": max(camera_skews, default=None),
                                        "mean": (sum(camera_skews) / len(camera_skews)) if camera_skews else None,
                                        "limit": camera_sync_limit_s},
                      "reconstruction_15hz": reconstruction},
        }

    rows: list[dict[str, Any]] = []
    try:
        with steps_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    errors.append(f"blank_jsonl_line:{line_number}")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"step_json_invalid:{line_number}:{exc}")
                    continue
                rows.append(row)
    except OSError as exc:
        errors.append(f"steps_jsonl_unreadable:{exc}")

    if not rows:
        errors.append("no_action_steps")
    if metadata.get("control_steps") != len(rows):
        errors.append("metadata_control_step_count_mismatch")

    observations: list[dict[str, Any]] = []
    observation_path = episode_dir / str((metadata.get("files") or {}).get("observations", "observations.jsonl"))
    if observation_path.is_file():
        try:
            observations = [json.loads(line) for line in observation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"observations_jsonl_invalid:{exc}")
    else:
        warnings.append("observations_jsonl_missing_legacy_episode")
    if observations and len(observations) != metadata.get("sensor_frames"):
        errors.append("observation_count_mismatch")
    observation_by_index = {
        item.get("sensor_frame_index"): item.get("observation", {})
        for item in observations
        if isinstance(item.get("sensor_frame_index"), int)
    }

    terminal = metadata.get("terminal_observation")
    terminal_pose = _pose_components((terminal or {}).get("observation", {}).get("tcp_pose"))
    if terminal_pose is None:
        errors.append("terminal_observation_missing_or_invalid")

    sensor_indices: list[int] = []
    delta_times: list[float] = []
    for index, row in enumerate(rows):
        prefix = f"step:{index}"
        if row.get("control_step") != index:
            errors.append(f"{prefix}:control_step_not_sequential")
        if row.get("prompt") != metadata.get("prompt"):
            errors.append(f"{prefix}:prompt_mismatch")
        observation = row.get("observation") or {}
        joints = _finite_vector(observation.get("joint_position_rad"), 7)
        velocities = _finite_vector(observation.get("joint_velocity_rad_s"), 7)
        pose = _pose_components(observation.get("tcp_pose"))
        gripper = observation.get("gripper_opening_ratio")
        if joints is None:
            errors.append(f"{prefix}:joint_position_invalid")
        if velocities is None:
            warnings.append(f"{prefix}:joint_velocity_invalid")
        if pose is None:
            errors.append(f"{prefix}:tcp_pose_invalid")
        if not isinstance(gripper, (int, float)) or isinstance(gripper, bool) or not math.isfinite(float(gripper)) or not 0.0 <= float(gripper) <= 1.0:
            errors.append(f"{prefix}:gripper_opening_invalid")
        frame_index = observation.get("sensor_frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            errors.append(f"{prefix}:sensor_frame_index_invalid")
        else:
            sensor_indices.append(frame_index)

        action_data = row.get("action") or {}
        action = _finite_vector(action_data.get("tcp_delta_base"), 7)
        delta_time = action_data.get("delta_time_s")
        if action is None:
            errors.append(f"{prefix}:tcp_action_invalid")
            continue
        if any(abs(value) > 0.01 + 1e-9 for value in action[:3]):
            errors.append(f"{prefix}:translation_action_out_of_bounds")
        if math.sqrt(sum(value * value for value in action[3:6])) > 0.1 + 1e-9:
            errors.append(f"{prefix}:rotation_action_out_of_bounds")
        if not 0.0 <= action[6] <= 1.0:
            errors.append(f"{prefix}:gripper_action_out_of_bounds")
        if not isinstance(delta_time, (int, float)) or isinstance(delta_time, bool) or not math.isfinite(float(delta_time)) or float(delta_time) <= 0.0:
            errors.append(f"{prefix}:delta_time_invalid")
        else:
            delta_times.append(float(delta_time))

        expected_next_index = action_data.get("next_sensor_frame_index")
        if not isinstance(expected_next_index, int) or isinstance(expected_next_index, bool):
            expected_next_index = frame_index + 1 if isinstance(frame_index, int) else None
        if expected_next_index in observation_by_index:
            following_observation = observation_by_index[expected_next_index]
            following_index = expected_next_index
        elif index + 1 < len(rows):
            following_observation = rows[index + 1].get("observation", {})
            following_index = following_observation.get("sensor_frame_index")
        else:
            following_observation = (terminal or {}).get("observation", {})
            following_index = following_observation.get("sensor_frame_index")
        next_pose = _pose_components(following_observation.get("tcp_pose"))
        if following_index != expected_next_index:
            warnings.append(f"{prefix}:successor_observation_not_archived")
        elif pose is not None and next_pose is not None:
            expected = [next_pose[0][axis] - pose[0][axis] for axis in range(3)]
            expected += _rotvec_between(pose[1], next_pose[1])
            if any(abs(expected[axis] - action[axis]) > 1e-7 for axis in range(6)):
                errors.append(f"{prefix}:tcp_action_reconstruction_mismatch")

    if sensor_indices and sensor_indices != sorted(set(sensor_indices)):
        errors.append("sensor_frame_indices_not_strictly_increasing")

    expected_fps = float(metadata.get("recording_hz") or 15.0)
    videos: dict[str, dict[str, Any]] = {}
    for source in ("external", "wrist"):
        relative = (metadata.get("videos") or {}).get(source, f"videos/observation.images.{source}.mp4")
        info = _video_info(episode_dir / relative)
        videos[source] = info
        if not info["opened"] or not info["first_frame_decodable"]:
            errors.append(f"{source}_video_not_decodable")
        if (info["width"], info["height"]) != (640, 480):
            errors.append(f"{source}_video_resolution_mismatch")
        if abs(info["fps"] - expected_fps) > 0.25:
            errors.append(f"{source}_video_fps_mismatch")
        if info["frame_count"] != metadata.get("sensor_frames"):
            errors.append(f"{source}_video_frame_count_mismatch")

    if videos.get("external", {}).get("frame_count") != videos.get("wrist", {}).get("frame_count"):
        errors.append("dual_video_frame_count_mismatch")
    if metadata.get("sensor_frames") != len(rows) + int((metadata.get("quality") or {}).get("unlabeled_sensor_frames", 0)):
        errors.append("sensor_action_alignment_count_mismatch")

    structural_valid = not errors
    quality = metadata.get("quality") or {}
    training_ready = structural_valid and metadata.get("accepted") is True and quality.get("training_eligible") is True
    return {
        "episode_dir": str(episode_dir),
        "schema_version": metadata.get("schema_version"),
        "structural_valid": structural_valid,
        "training_ready": training_ready,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "action_steps": len(rows),
            "sensor_frames": metadata.get("sensor_frames"),
            "mean_action_dt_s": sum(delta_times) / len(delta_times) if delta_times else None,
            "videos": videos,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one raw NERO TCP-VLA episode without building H16 windows.")
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    report = validate_episode(args.episode_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["structural_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
