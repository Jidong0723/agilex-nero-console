"""Build a 15 Hz TCP-VLA training view from an immutable raw episode.

This command is intentionally separate from collection.  It never changes the
raw manifests or images and refuses to overwrite an existing derived view.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion.osc import KinematicsClient, pose_from_tcp
from supervisor.dataset_recorder import _pose_components, _rotvec_between


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _camera_time(row: dict[str, Any]) -> int:
    return (int(row["external_monotonic_ns"]) + int(row["wrist_monotonic_ns"])) // 2


def _camera_streams(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize legacy paired rows and v3 independent rows to two streams."""
    result: dict[str, list[dict[str, Any]]] = {"external": [], "wrist": []}
    for row in rows:
        source = row.get("source")
        if source in result:
            result[source].append(row)
            continue
        images = row.get("images") or {}
        for source in result:
            result[source].append({"source": source, "source_frame_index": row.get("frame_index"),
                                   "capture_monotonic_ns": row.get(f"{source}_monotonic_ns"),
                                   "image": images.get(source)})
    for source in result:
        result[source].sort(key=lambda item: int(item["capture_monotonic_ns"]))
    return result


def _quat_to_rot6d(quaternion: list[float]) -> list[float]:
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [matrix[0][0], matrix[1][0], matrix[2][0], matrix[0][1], matrix[1][1], matrix[2][1]]


def _nearest(rows: list[dict[str, Any]], times: list[int], target: int) -> tuple[dict[str, Any], int]:
    index = bisect.bisect_left(times, target)
    candidates = [item for item in (index - 1, index) if 0 <= item < len(rows)]
    selected = min(candidates, key=lambda item: abs(times[item] - target))
    return rows[selected], abs(times[selected] - target)


def _interpolate_robot(rows: list[dict[str, Any]], times: list[int], target: int) -> dict[str, Any] | None:
    after = bisect.bisect_left(times, target)
    if after == 0 or after >= len(rows):
        return None
    before = after - 1
    left, right = rows[before], rows[after]
    span = times[after] - times[before]
    if span <= 0:
        return None
    alpha = (target - times[before]) / span
    blend = lambda key: [(1.0 - alpha) * float(a) + alpha * float(b) for a, b in zip(left[key], right[key])]
    return {
        "joint_position_rad": blend("joint_position_rad"),
        "joint_velocity_rad_s": blend("joint_velocity_rad_s"),
        "gripper_opening_ratio": (1.0 - alpha) * float(left["gripper_opening_ratio"]) + alpha * float(right["gripper_opening_ratio"]),
        "target_gripper_opening_ratio": (1.0 - alpha) * float(left.get("target_gripper_opening_ratio", left["gripper_opening_ratio"]))
                                         + alpha * float(right.get("target_gripper_opening_ratio", right["gripper_opening_ratio"])),
        "left_sample_index": left["sample_index"], "right_sample_index": right["sample_index"],
        "left_time_ns": times[before], "right_time_ns": times[after],
        "nearest_raw": left if alpha < 0.5 else right,
    }


def build_training_view(episode_dir: Path, output_dir: Path,
                        fk: Callable[[list[float]], dict[str, Any]], *,
                        rate_hz: float = 15.0, camera_limit_s: float = 0.030,
                        robot_limit_s: float = 0.035) -> dict[str, Any]:
    episode_dir, output_dir = episode_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite derived view: {output_dir}")
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_stage") != "native_rate_raw_collection":
        raise ValueError("episode is not a native-rate raw collection")
    cameras = _camera_streams(_read_jsonl(episode_dir / metadata["files"]["raw_camera"]))
    robots = _read_jsonl(episode_dir / metadata["files"]["raw_robot_state"])
    bounds = metadata.get("collection_monotonic_ns") or {}
    lower, upper = bounds.get("start"), bounds.get("end")
    if isinstance(lower, int):
        cameras = {source: [row for row in rows if int(row["capture_monotonic_ns"]) >= lower]
                   for source, rows in cameras.items()}
        robots = [row for row in robots if int(row["feedback_monotonic_ns"]) >= lower]
    if isinstance(upper, int):
        cameras = {source: [row for row in rows if int(row["capture_monotonic_ns"]) <= upper]
                   for source, rows in cameras.items()}
        robots = [row for row in robots if int(row["feedback_monotonic_ns"]) <= upper]
    camera_times = {source: [int(row["capture_monotonic_ns"]) for row in rows]
                    for source, rows in cameras.items()}
    robot_times = [int(row["feedback_monotonic_ns"]) for row in robots]
    if any(times != sorted(times) for times in camera_times.values()) or robot_times != sorted(robot_times):
        raise ValueError("raw timestamps are not monotonic")
    if any(not cameras[source] for source in ("external", "wrist")) or not robots:
        raise ValueError("raw episode is missing a required camera or robot stream")
    start = max(camera_times["external"][0], camera_times["wrist"][0], robot_times[0])
    end = min(camera_times["external"][-1], camera_times["wrist"][-1], robot_times[-1])
    period_ns = round(1e9 / float(rate_hz))
    camera_limit_ns, robot_limit_ns = round(camera_limit_s * 1e9), round(robot_limit_s * 1e9)
    aligned: list[dict[str, Any]] = []
    rejected = 0
    target = start
    while target <= end:
        selected_cameras = {source: _nearest(cameras[source], camera_times[source], target)
                            for source in ("external", "wrist")}
        camera_error = max(item[1] for item in selected_cameras.values())
        robot = _interpolate_robot(robots, robot_times, target)
        if camera_error > camera_limit_ns or robot is None:
            rejected += 1; target += period_ns; continue
        if max(target - robot["left_time_ns"], robot["right_time_ns"] - target) > robot_limit_ns:
            rejected += 1; target += period_ns; continue
        pose = _pose_components(pose_from_tcp(fk(robot["joint_position_rad"])))
        if pose is None:
            raise ValueError("Pinocchio FK returned an invalid TCP pose")
        nearest_raw = robot.pop("nearest_raw")
        aligned.append({
            "sensor_frame_index": len(aligned), "target_monotonic_ns": target,
            "prompt": metadata["prompt"],
            "images": {source: selected_cameras[source][0]["image"] for source in ("external", "wrist")},
            "observation": {
                "joint_position_rad": robot["joint_position_rad"],
                "joint_velocity_rad_s": robot["joint_velocity_rad_s"],
                "gripper_opening_ratio": robot["gripper_opening_ratio"],
                "target_gripper_opening_ratio": robot["target_gripper_opening_ratio"],
                "tcp_pose": {"position_m": pose[0], "orientation_xyzw": pose[1], "orientation_rot6d": _quat_to_rot6d(pose[1])},
                "recorded_tcp_pose": nearest_raw.get("measured_tcp_pose"),
                "target_tcp_pose": nearest_raw.get("target_tcp_pose"),
            },
            "alignment": {"raw_camera_frame_index": {source: selected_cameras[source][0]["source_frame_index"]
                                                       for source in ("external", "wrist")},
                          "camera_error_s": {source: selected_cameras[source][1] / 1e9
                                             for source in ("external", "wrist")},
                          "robot_left_sample_index": robot["left_sample_index"], "robot_right_sample_index": robot["right_sample_index"]},
        })
        target += period_ns
    if len(aligned) < 2:
        raise ValueError("not enough aligned samples to reconstruct TCP actions")
    observations = []
    steps = []
    for index, row in enumerate(aligned):
        observations.append(row)
        if index + 1 >= len(aligned):
            continue
        following = aligned[index + 1]
        if following["target_monotonic_ns"] - row["target_monotonic_ns"] != period_ns:
            continue
        pose = _pose_components(row["observation"]["tcp_pose"])
        next_pose = _pose_components(following["observation"]["tcp_pose"])
        action = [next_pose[0][axis] - pose[0][axis] for axis in range(3)]
        action += _rotvec_between(pose[1], next_pose[1])
        action += [float(row["observation"]["target_gripper_opening_ratio"])]
        steps.append({**row, "control_step": len(steps),
                      "action": {"tcp_delta_base": action, "delta_time_s": (following["target_monotonic_ns"] - row["target_monotonic_ns"]) / 1e9,
                                 "next_sensor_frame_index": following["sensor_frame_index"]}})
    windows = []
    for start_index in range(0, max(0, len(steps) - 15)):
        chunk = steps[start_index:start_index + 16]
        contiguous = all(
            chunk[item]["action"]["next_sensor_frame_index"] == chunk[item + 1]["sensor_frame_index"]
            for item in range(15)
        )
        if contiguous:
            windows.append({"start_control_step": start_index,
                            "actions": [item["action"]["tcp_delta_base"] for item in chunk]})
    output_dir.mkdir(parents=True, exist_ok=False)
    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        (output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    write_jsonl("observations.jsonl", observations); write_jsonl("steps.jsonl", steps); write_jsonl("h16.jsonl", windows)
    result = {"schema_version": "nero.tcp-vla.training-view.v1", "source_episode": str(episode_dir),
              "rate_hz": rate_hz, "sensor_frames": len(observations), "control_steps": len(steps),
              "h16_windows": len(windows), "rejected_grid_points": rejected,
              "alignment_limits_s": {"camera_nearest": camera_limit_s, "robot_bracket": robot_limit_s},
              "action_alignment": "observation[t] -> FK(interpolated_joint_state[t+1])"}
    (output_dir / "view.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, but never overwrite, a 15 Hz training view from raw NERO data.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episode = args.episode_dir.resolve()
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    interface = metadata.get("processing_interface") or {}
    osc_config = json.loads((episode / interface["osc_config"]).read_text(encoding="utf-8-sig"))
    client = KinematicsClient(PROJECT_ROOT, osc_config)
    client.urdf = episode / interface["urdf"]
    output = args.output or episode / "derived" / "15hz_v1"
    try:
        client.start()
        result = build_training_view(episode, output, client.fk)
    finally:
        client.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
