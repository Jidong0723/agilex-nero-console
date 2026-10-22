"""No-motion cloud RTC continuity probe.

It uses synthetic camera pixels and a fixed state, never opens OSC nor sends a
robot command. It measures the raw, official RTC sampler: no prefix rewrite,
Cartesian slew limiter, or post-sampling action projection is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervisor.openpi_client import OpenPIClient

PERIOD_NS = 50_000_000
HORIZON = 20


def absolute(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    result = np.empty((len(actions), 8), dtype=np.float64)
    result[:, :3] = state[:3] + actions[:, :3]
    # This probe deliberately uses a zero base rotation, so the relative
    # rotvec is already the world quaternion.  Keep the diagnostic runnable
    # in the slim local control environment, which does not depend on SciPy.
    result[:, 3:7] = np.asarray([rotvec_to_quat(row[3:6]) for row in actions])
    result[:, 7] = actions[:, 6]
    return result


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    axis = np.asarray(rotvec, dtype=np.float64) / angle
    return np.append(axis * np.sin(angle / 2), np.cos(angle / 2))


def quaternion_delta_angle(first: np.ndarray, second: np.ndarray) -> float:
    # For unit quaternions, the absolute dot product gives the smaller
    # relative angle without allocating a rotation-library object.
    return float(2 * np.arccos(np.clip(abs(float(np.dot(first, second))), -1.0, 1.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18762)
    parser.add_argument("--cycles", type=int, default=40)
    parser.add_argument("--inference-delay", type=int, default=7)
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runtime/rtc_cloud_continuity_probe.json"))
    args = parser.parse_args()
    if not 0 <= args.inference_delay <= args.execute_horizon < HORIZON:
        parser.error("require 0 <= inference_delay <= execute_horizon < 20")
    prefix_steps = HORIZON - args.execute_horizon
    rng = np.random.default_rng(11649)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    state = np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    observation = {"observation/image": image, "observation/wrist_image": image.copy(),
                   "observation/state": state, "prompt": "Pick up the yellow cube."}
    report: dict[str, object] = {"cycles": [], "metadata": None, "passed": False}
    client = OpenPIClient(args.host, args.port, 30)
    try:
        report["metadata"] = client.metadata
        initial = np.asarray(client.infer(observation)["actions"], dtype=np.float32)
        # The prefix is a physical contract: feed only values a hardware-side
        # validator could legally retain between replans.
        initial[:, 6] = np.clip(initial[:, 6], 0.0, 1.0)
        prefix = absolute(initial, state)[args.execute_horizon:]
        origin = 1_000_000_000
        for request_id in range(1, args.cycles + 1):
            observation_ns = origin + request_id * args.execute_horizon * PERIOD_NS
            context = {
                "version": "nero.rtc.v2", "run_epoch": 11649, "request_id": request_id,
                "observation_tick": request_id * args.execute_horizon, "plan_revision": request_id,
                "freeze_until": request_id * args.execute_horizon + args.inference_delay, "phase": "replan",
                "observation_perf_ns": observation_ns, "timeline_origin_perf_ns": origin,
                "freeze_until_perf_ns": observation_ns + args.inference_delay * PERIOD_NS,
                "action_row0_offset_s": 0, "period_s": .05, "inference_delay": args.inference_delay,
                "execute_horizon": args.execute_horizon, "prefix_attention_horizon": prefix_steps,
                "absolute_prefix": prefix.tolist(),
            }
            began = time.perf_counter()
            response = client.infer(dict(observation, rtc_context=context))
            elapsed_ms = (time.perf_counter() - began) * 1000
            actions = np.asarray(response["actions"], dtype=np.float32)
            targets = absolute(actions, state)
            seam_m = float(np.max(np.linalg.norm(targets[:prefix_steps, :3] - prefix[:, :3], axis=1)))
            first_new_m = float(np.linalg.norm(targets[prefix_steps, :3] - prefix[-1, :3]))
            rotations = np.vstack((prefix[-1, 3:7], targets[prefix_steps:, 3:7]))
            tail_rotation_rad = max(quaternion_delta_angle(a, b) for a, b in zip(rotations, rotations[1:]))
            row = {"request": request_id, "elapsed_ms": round(elapsed_ms, 3),
                   "prefix_seam_mm": seam_m * 1000, "first_new_step_mm": first_new_m * 1000,
                   "tail_max_rotation_rad": tail_rotation_rad,
                   "gripper_min": float(np.min(targets[:, 7])), "gripper_max": float(np.max(targets[:, 7])),
                   "stabilization": response.get("rtc", {}).get("stabilization")}
            report["cycles"].append(row)
            # This represents the control-side physical contract, which is
            # constrained to a valid gripper ratio before the next request.
            # It deliberately does not alter the raw action measurements.
            prefix = targets[args.execute_horizon:]
            prefix[:, 7] = np.clip(prefix[:, 7], 0.0, 1.0)
        rows = report["cycles"]
        report["summary"] = {
            "max_prefix_seam_mm": max(row["prefix_seam_mm"] for row in rows),
            "max_first_new_step_mm": max(row["first_new_step_mm"] for row in rows),
            "max_tail_rotation_rad": max(row["tail_max_rotation_rad"] for row in rows),
            "min_gripper": min(row["gripper_min"] for row in rows),
            "max_gripper": max(row["gripper_max"] for row in rows),
        }
        summary = report["summary"]
        # Completion records that the official sampler ran through the whole
        # sequence. The reported seam statistics are evidence, not a hidden
        # replacement continuity policy.
        report["passed"] = len(rows) == args.cycles
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("summary", {"error": report.get("error")}), ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
