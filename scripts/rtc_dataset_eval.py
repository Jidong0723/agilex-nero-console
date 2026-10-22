"""Offline dataset replay for the official, unprojected RTC sampler.

This script only reads LeRobot parquet/video data and calls the local model
WebSocket.  It never opens OSC or sends a robot command.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
from websockets.sync.client import connect


H = 20
PERIOD_NS = 50_000_000


def pack_array(value):
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class Client:
    def __init__(self, host, port, timeout=60):
        self.socket = connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=timeout)
        handshake = self.socket.recv(timeout=timeout)
        self.metadata = msgpack.unpackb(handshake, object_hook=unpack_array)
        self.packer = msgpack.Packer(default=pack_array)
        self.timeout = timeout

    def infer(self, request):
        self.socket.send(self.packer.pack(request))
        reply = self.socket.recv(timeout=self.timeout)
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return msgpack.unpackb(reply, object_hook=unpack_array)

    def close(self):
        self.socket.close()


def absolute_from_action(action, state):
    base = Rotation.from_rotvec(state[3:6])
    rows = np.asarray(action, dtype=np.float64).copy()
    result = np.empty((len(rows), 8), dtype=np.float64)
    result[:, :3] = state[:3] + rows[:, :3]
    result[:, 3:7] = np.asarray([(Rotation.from_rotvec(row[3:6]) * base).as_quat() for row in rows])
    result[:, 7] = rows[:, 6]
    return result


def absolute_from_dataset(actions):
    actions = np.asarray(actions, dtype=np.float64)
    result = np.empty((len(actions), 8), dtype=np.float64)
    result[:, :3] = actions[:, :3]
    result[:, 3:7] = np.asarray([Rotation.from_rotvec(row[3:6]).as_quat() for row in actions])
    result[:, 7] = actions[:, 6]
    return result


def angle_error(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def read_episode(root, episode, task_names, samples_per_episode, rng):
    import pyarrow.parquet as pq

    parquet = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(parquet)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    task_index = int(table["task_index"][0].as_py())
    length = len(state)
    candidates = np.arange(0, max(0, length - H + 1), dtype=np.int64)
    if not len(candidates):
        return []
    picks = np.sort(rng.choice(candidates, size=min(samples_per_episode, len(candidates)), replace=False))
    paths = {
        "external": root / "videos" / "chunk-000" / "observation.images.base_0_rgb" / f"episode_{episode:06d}.mp4",
        "wrist": root / "videos" / "chunk-000" / "observation.images.left_wrist_0_rgb" / f"episode_{episode:06d}.mp4",
    }
    captures = {key: cv2.VideoCapture(str(path)) for key, path in paths.items()}
    records = []
    try:
        for t in picks:
            images = {}
            for key, cap in captures.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame[t]))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"video decode failed episode={episode} frame={frame[t]} camera={key}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images[key] = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.uint8)
            records.append({"episode": episode, "frame": int(frame[t]), "task_index": task_index,
                            "task": task_names[task_index], "state": state[t],
                            "future_abs": absolute_from_dataset(action[t:t + H]), "images": images})
    finally:
        for cap in captures.values():
            cap.release()
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--episodes-per-task", type=int, default=4)
    p.add_argument("--samples-per-episode", type=int, default=4)
    p.add_argument("--inference-delay", type=int, default=7)
    p.add_argument("--execute-horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=11649)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot-count", type=int, default=6)
    args = p.parse_args()
    if not 0 <= args.inference_delay <= args.execute_horizon < H or args.inference_delay + args.execute_horizon > H:
        p.error("RTC requires 0 <= D <= E and D+E <= H20")
    root = args.dataset.resolve()
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tasks[int(row["task_index"])] = row["task"]
    import pyarrow.parquet as pq
    episodes = []
    for path in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episodes.append((int(table["episode_index"][0].as_py()), int(table["task_index"][0].as_py())))
    rng = np.random.default_rng(args.seed)
    selected = []
    for task_index in sorted(tasks):
        task_episodes = [episode for episode, task in episodes if task == task_index]
        selected.extend(sorted(rng.choice(task_episodes, size=min(args.episodes_per_task, len(task_episodes)), replace=False).tolist()))
    records = [record for episode in selected for record in read_episode(root, episode, tasks, args.samples_per_episode, rng)]
    client = Client(args.host, args.port)
    rows = []
    try:
        for index, record in enumerate(records):
            state = record["state"]
            observation = {"observation/image": record["images"]["external"],
                           "observation/wrist_image": record["images"]["wrist"],
                           "observation/state": state, "prompt": record["task"]}
            started = time.perf_counter()
            normal = client.infer(observation)
            normal_ms = (time.perf_counter() - started) * 1000
            normal_abs = absolute_from_action(normal["actions"], state)
            prefix = record["future_abs"][:H - args.execute_horizon]
            origin = 1_000_000_000
            tick = int(record["frame"])
            obs_ns = origin + tick * PERIOD_NS
            context = {"version": "nero.rtc.v2", "run_epoch": 1, "request_id": index + 1,
                       "observation_tick": tick, "plan_revision": index + 1,
                       "freeze_until": tick + args.inference_delay, "phase": "replan",
                       "observation_perf_ns": obs_ns, "timeline_origin_perf_ns": origin,
                       "freeze_until_perf_ns": obs_ns + args.inference_delay * PERIOD_NS,
                       "action_row0_offset_s": 0, "period_s": .05,
                       "inference_delay": args.inference_delay, "execute_horizon": args.execute_horizon,
                       "prefix_attention_horizon": H - args.execute_horizon,
                       "absolute_prefix": prefix.tolist()}
            started = time.perf_counter()
            rtc = client.infer(dict(observation, rtc_context=context))
            rtc_ms = (time.perf_counter() - started) * 1000
            rtc_abs = absolute_from_action(rtc["actions"], state)
            target = record["future_abs"]
            prefix_error = np.linalg.norm(rtc_abs[:len(prefix), :3] - prefix[:, :3], axis=1)
            normal_error = np.linalg.norm(normal_abs[:, :3] - target[:, :3], axis=1)
            rtc_error = np.linalg.norm(rtc_abs[:, :3] - target[:, :3], axis=1)
            seam = np.linalg.norm(rtc_abs[args.inference_delay, :3] - prefix[args.inference_delay - 1, :3]) if args.inference_delay else 0
            normal_rot = angle_error(normal_abs[:, 3:7], target[:, 3:7])
            rtc_rot = angle_error(rtc_abs[:, 3:7], target[:, 3:7])
            rows.append({"episode": record["episode"], "frame": record["frame"], "task": record["task"],
                         "normal_ms": normal_ms, "rtc_ms": rtc_ms,
                         "normal_position_rmse_mm": float(np.sqrt(np.mean(normal_error ** 2)) * 1000),
                         "rtc_position_rmse_mm": float(np.sqrt(np.mean(rtc_error ** 2)) * 1000),
                         "rtc_prefix_rmse_mm": float(np.sqrt(np.mean(prefix_error ** 2)) * 1000),
                         "rtc_prefix_max_mm": float(np.max(prefix_error) * 1000), "rtc_seam_mm": float(seam * 1000),
                         "normal_orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(normal_rot ** 2)))),
                         "rtc_orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rtc_rot ** 2)))),
                         "normal_gripper_rmse": float(np.sqrt(np.mean((normal_abs[:, 7] - target[:, 7]) ** 2))),
                         "rtc_gripper_rmse": float(np.sqrt(np.mean((rtc_abs[:, 7] - target[:, 7]) ** 2))),
                         "rtc_sampler": rtc.get("rtc", {}).get("sampler"),
                         "trajectory": {"target_xyz": target[:, :3].tolist(),
                                         "normal_xyz": normal_abs[:, :3].tolist(),
                                         "rtc_xyz": rtc_abs[:, :3].tolist()}})
    finally:
        client.close()
    summary = {"samples": len(rows), "episodes": selected, "D": args.inference_delay, "E": args.execute_horizon,
               "metadata": client.metadata, "dataset": str(root),
               "metrics": {key: {"mean": float(np.mean([row[key] for row in rows])),
                                  "p95": float(np.percentile([row[key] for row in rows], 95)),
                                  "max": float(np.max([row[key] for row in rows]))}
                           for key in ("normal_ms", "rtc_ms", "normal_position_rmse_mm", "rtc_position_rmse_mm",
                                       "rtc_prefix_rmse_mm", "rtc_prefix_max_mm", "rtc_seam_mm",
                                       "normal_orientation_rmse_deg", "rtc_orientation_rmse_deg",
                                       "normal_gripper_rmse", "rtc_gripper_rmse")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.with_suffix(".json")).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        names = ["normal_position_rmse_mm", "rtc_position_rmse_mm", "rtc_prefix_rmse_mm", "rtc_seam_mm"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, name in zip(axes.flat, names):
            values = [row[name] for row in rows]
            axis.plot(values, ".-", linewidth=.8)
            axis.set_title(name); axis.set_xlabel("sample"); axis.set_ylabel("value")
            axis.grid(True, alpha=.25)
        fig.suptitle(f"NERO dataset RTC replay: D={args.inference_delay}, E={args.execute_horizon}, n={len(rows)}")
        fig.tight_layout(); fig.savefig(args.output.with_suffix(".png"), dpi=150); plt.close(fig)
        ranked = sorted(rows, key=lambda row: row["rtc_seam_mm"], reverse=True)[:args.plot_count]
        for plot_index, row in enumerate(ranked, 1):
            trajectory = row["trajectory"]
            target_xyz = np.asarray(trajectory["target_xyz"])
            normal_xyz = np.asarray(trajectory["normal_xyz"])
            rtc_xyz = np.asarray(trajectory["rtc_xyz"])
            steps = np.arange(H)
            figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            for axis, component, name in zip(axes, range(3), ("x", "y", "z")):
                axis.plot(steps, target_xyz[:, component], "k.-", label="dataset target")
                axis.plot(steps, normal_xyz[:, component], "b.--", alpha=.75, label="normal H20")
                axis.plot(steps, rtc_xyz[:, component], "r.-", label="official RTC")
                axis.axvline(args.inference_delay - .5, color="darkorange", linestyle="--", linewidth=1.5,
                             label="RTC splice" if component == 0 else None)
                axis.axvspan(-.5, args.inference_delay - .5, color="gray", alpha=.08)
                axis.set_ylabel(f"{name} (m)"); axis.grid(True, alpha=.25)
            axes[-1].set_xlabel("step (0 = observation time)")
            axes[0].legend(loc="best", ncol=4, fontsize=8)
            figure.suptitle(f"RTC seam #{plot_index}: episode {row['episode']}, frame {row['frame']}, seam={row['rtc_seam_mm']:.2f} mm")
            figure.tight_layout()
            figure.savefig(args.output.with_name(f"{args.output.stem}_seam_{plot_index:02d}_ep{row['episode']:03d}_f{row['frame']:05d}.png"), dpi=160)
            plt.close(figure)
    except Exception as exc:
        summary["plot_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
