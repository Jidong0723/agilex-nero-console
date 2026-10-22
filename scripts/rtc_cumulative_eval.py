"""Long-horizon offline comparison of synchronous and official RTC stitching.

This replays one real LeRobot episode without OSC or robot commands.  It calls
the model at fixed replanning boundaries and plots cumulative TCP XYZ progress
for at least 100 steps.  The plot marks each replan/splice boundary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np

from rtc_dataset_eval import H, PERIOD_NS, Client, absolute_from_action, unpack_array


def load_episode(root: Path, episode: int, task_names: dict[int, str], start: int, steps: int):
    import pyarrow.parquet as pq

    path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    task_index = int(table["task_index"][0].as_py())
    if start < 0 or start + steps > len(state):
        raise ValueError(f"episode {episode} has {len(state)} rows; need {start}+{steps}")
    paths = {
        "external": root / "videos" / "chunk-000" / "observation.images.base_0_rgb" / f"episode_{episode:06d}.mp4",
        "wrist": root / "videos" / "chunk-000" / "observation.images.left_wrist_0_rgb" / f"episode_{episode:06d}.mp4",
    }
    caps = {key: cv2.VideoCapture(str(path)) for key, path in paths.items()}
    observations = []
    targets = []
    try:
        for offset in range(steps):
            t = start + offset
            images = {}
            for key, cap in caps.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame[t]))
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"video decode failed episode={episode} frame={frame[t]} camera={key}")
                images[key] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA).astype(np.uint8)
            observations.append({"observation/image": images["external"],
                                 "observation/wrist_image": images["wrist"],
                                 "observation/state": state[t],
                                 "prompt": task_names[task_index],
                                 "frame": int(frame[t])})
            targets.append(state[t, :3].astype(np.float64))
    finally:
        for cap in caps.values():
            cap.release()
    return observations, np.asarray(targets), task_names[task_index]


def rtc_context(old_plan, state, tick, request_id, origin, delay, execute):
    prefix = np.asarray(old_plan[:H - execute], dtype=np.float64).copy()
    # The dataset/model may emit tiny or larger gripper overshoot.  The RTC
    # wire contract is physical gripper ratio [0, 1]; keep this normalization
    # local to the diagnostic request and do not alter XYZ/orientation curves.
    prefix[:, 7] = np.clip(prefix[:, 7], 0.0, 1.0)
    return {"version": "nero.rtc.v2", "run_epoch": 1, "request_id": request_id,
            "observation_tick": int(tick), "plan_revision": request_id,
            "freeze_until": int(tick + delay), "phase": "replan",
            "observation_perf_ns": int(origin + tick * PERIOD_NS),
            "timeline_origin_perf_ns": int(origin),
            "freeze_until_perf_ns": int(origin + (tick + delay) * PERIOD_NS),
            "action_row0_offset_s": 0, "period_s": .05,
            "inference_delay": delay, "execute_horizon": execute,
            "prefix_attention_horizon": H - execute,
            "absolute_prefix": prefix.tolist()}


def infer_chunk(client, observation, state, context=None):
    request = dict(observation)
    if context is not None:
        request["rtc_context"] = context
    started = time.perf_counter()
    reply = client.infer(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return absolute_from_action(reply["actions"], state), elapsed_ms, reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--inference-delay", type=int, default=7)
    p.add_argument("--execute-horizon", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.steps < 100:
        p.error("--steps must be at least 100")
    if not 0 <= args.inference_delay <= args.execute_horizon < H:
        p.error("RTC requires 0 <= inference-delay <= execute-horizon < 20")

    root = args.dataset.resolve()
    tasks = {int(row["task_index"]): row["task"] for row in
             (json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines())}
    if args.episode is None:
        import pyarrow.parquet as pq
        candidates = []
        for path in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
            table = pq.read_table(path, columns=["episode_index"])
            episode = int(table["episode_index"][0].as_py())
            if len(table) >= args.start + args.steps:
                candidates.append(episode)
        if not candidates:
            raise RuntimeError("no episode is long enough for the requested cumulative window")
        args.episode = candidates[0]
    observations, target_xyz, task = load_episode(root, args.episode, tasks, args.start, args.steps)
    client = Client(args.host, args.port)
    normal = np.zeros((args.steps, 3), dtype=np.float64)
    rtc = np.zeros((args.steps, 3), dtype=np.float64)
    normal_latency = []
    rtc_latency = []
    boundaries = list(range(0, args.steps, args.execute_horizon))
    rtc_splices = [b + args.inference_delay for b in boundaries[1:]]
    chunk_records = []
    origin = 1_000_000_000
    old_plan = None
    try:
        for boundary in boundaries:
            end = min(boundary + args.execute_horizon, args.steps)
            obs = observations[boundary]
            state = obs["observation/state"]
            normal_chunk, normal_ms, _ = infer_chunk(client, obs, state)
            normal_latency.append(normal_ms)
            # Ordinary synchronous counterfactual: replan from the observation
            # at every boundary and execute rows 0..K-1 directly.  Each model
            # row is expressed relative to that observation; for a continuous
            # rollout, re-anchor its relative XYZ at the previously executed
            # blue endpoint instead of the dataset TCP at this boundary.
            # RTC freeze/splice/mask logic must not be applied to this curve.
            normal_anchor = state[:3] if boundary == 0 else normal[boundary - 1]
            normal_delta = normal_chunk[:end - boundary, :3] - state[:3]
            normal[boundary:end] = normal_anchor + normal_delta
            # Official eval-flow index movement: at each replan, the previous
            # chunk is advanced by E rows before its remaining future is used.
            # Reusing old_plan[0:D] would replay already executed actions.
            active_old = None if old_plan is None else old_plan[args.execute_horizon:].copy()
            if old_plan is None:
                rtc_chunk, rtc_ms, _ = infer_chunk(client, obs, state)
            else:
                context = rtc_context(active_old, state, obs["frame"], boundary + 1, origin,
                                       args.inference_delay, args.execute_horizon)
                rtc_chunk, rtc_ms, _ = infer_chunk(client, obs, state, context)
            rtc_latency.append(rtc_ms)
            chunk_records.append({"boundary": boundary,
                                 "old_chunk_xyz": None if active_old is None else active_old[:, :3].tolist(),
                                 "raw_new_chunk_xyz": normal_chunk[:, :3].tolist(),
                                 "guided_new_chunk_xyz": rtc_chunk[:, :3].tolist()})
            if old_plan is None:
                rtc[boundary:end] = rtc_chunk[:end - boundary, :3]
            else:
                # Official eval-flow splice: retain the old chunk for D steps,
                # then use the new chunk's [D:E] future segment.
                keep = min(args.inference_delay, end - boundary)
                if keep:
                    rtc[boundary:boundary + keep] = active_old[:keep, :3]
                future_start = boundary + keep
                if future_start < end:
                    rtc[future_start:end] = rtc_chunk[args.inference_delay:args.inference_delay + (end - future_start), :3]
            # The new chunk becomes the active timeline and is advanced by E
            # rows at the next boundary, matching official index movement.
            old_plan = rtc_chunk
    finally:
        client.close()

    initial = target_xyz[0]
    result = {"episode": args.episode, "start": args.start, "steps": args.steps, "task": task,
              "inference_delay": args.inference_delay, "execute_horizon": args.execute_horizon,
              "boundaries": boundaries, "rtc_splices": rtc_splices,
              "latency_ms": {"normal_mean": float(np.mean(normal_latency)), "rtc_mean": float(np.mean(rtc_latency)),
                             "normal_max": float(np.max(normal_latency)), "rtc_max": float(np.max(rtc_latency))},
              "trajectory": {"target_cumulative_xyz": (target_xyz - initial).tolist(),
                             "synchronous_cumulative_xyz": (normal - initial).tolist(),
                             "rtc_cumulative_xyz": (rtc - initial).tolist()},
              "chunks": chunk_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    steps = np.arange(args.steps)
    colors = {"target": "black", "synchronous": "#2463eb", "rtc": "#e11d48"}
    for component, (axis, name) in enumerate(zip(axes, ("x", "y", "z"))):
        axis.plot(steps, target_xyz[:, component] - initial[component], "k-", linewidth=1.7, label="dataset target")
        axis.plot(steps, normal[:, component] - initial[component], color=colors["synchronous"], linewidth=1.2, label="Synchronous H20 (replan every 8)")
        axis.plot(steps, rtc[:, component] - initial[component], color=colors["rtc"], linewidth=1.2, label="Official RTC")
        for boundary in boundaries[1:]:
            # Official exp mask: rows < D have weight 1, rows D:E_guided
            # have strictly intermediate weights, and rows >= H-E have 0.
            axis.axvspan(boundary + args.inference_delay - .5,
                         boundary + (H - args.execute_horizon) - .5,
                         color="gold", alpha=.18,
                         label="soft mask (0<w<1)" if component == 0 and boundary == boundaries[1] else None)
            axis.axvline(boundary - .5, color="darkorange", linestyle="--", linewidth=.8, alpha=.65,
                         label="replan boundary" if component == 0 and boundary == boundaries[1] else None)
        for splice in rtc_splices:
            axis.axvline(splice - .5, color="purple", linestyle=":", linewidth=.9, alpha=.7,
                         label="RTC splice (D=7)" if component == 0 and splice == rtc_splices[0] else None)
        for chunk in chunk_records[1:]:
            # This is the actual reference Y supplied to IIGDM, placed at the
            # same global row indices as the new guided chunk.  Its first D
            # rows are both hard-masked and executed from the old plan.
            reference = np.asarray(chunk["old_chunk_xyz"], dtype=np.float64)
            reference_steps = np.arange(chunk["boundary"], chunk["boundary"] + len(reference))
            hard = (reference_steps < args.steps) & (np.arange(len(reference)) < args.inference_delay)
            if np.any(hard):
                axis.plot(reference_steps[hard], reference[hard, component] - initial[component],
                          color="#0f766e", linestyle=":", linewidth=1.1, alpha=.9,
                          label="reference Y hard-mask rows" if component == 0 and chunk is chunk_records[1] else None)
            # New RTC chunks sample a guided prefix (rows 0..D-1), but that
            # prefix is not commanded: the corresponding old-plan rows remain
            # authoritative until the splice at b + D.
            guided_prefix = np.asarray(chunk["guided_new_chunk_xyz"], dtype=np.float64)[:args.inference_delay]
            prefix_steps = np.arange(chunk["boundary"], chunk["boundary"] + len(guided_prefix))
            visible = prefix_steps < args.steps
            if np.any(visible):
                axis.plot(prefix_steps[visible], guided_prefix[visible, component] - initial[component],
                          color="#fb7185", linestyle="--", linewidth=.85, alpha=.7,
                          label="new chunk guided prefix (not executed)" if component == 0 and chunk is chunk_records[1] else None)
            # After the old plan has been shifted by E and its D-row execution
            # prefix is consumed, its remaining rows are never commanded.
            tail = reference[args.inference_delay:]
            tail_steps = np.arange(chunk["boundary"] + args.inference_delay,
                                   chunk["boundary"] + args.inference_delay + len(tail))
            visible = tail_steps < args.steps
            if np.any(visible):
                axis.plot(tail_steps[visible], tail[visible, component] - initial[component], color="dimgray",
                          linestyle="--", linewidth=.75, alpha=.45,
                          label="reference Y soft-mask tail (not executed)" if component == 0 and chunk is chunk_records[1] else None)
        axis.set_ylabel(f"cumulative {name} (m)")
        axis.grid(True, alpha=.25)
    axes[-1].set_xlabel("step (50 ms; step 0 = observation start)")
    axes[0].legend(loc="best", ncol=4, fontsize=9)
    fig.suptitle(f"Cumulative TCP progress: episode {args.episode}, {args.steps} steps, D={args.inference_delay}, E={args.execute_horizon}")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    # Per-cycle chunk-space plots: old plan, raw un-guided response, and the
    # RTC-guided response.  Chunk coordinates are intentionally kept separate
    # from the cumulative execution timeline to avoid misleading overlap.
    chunk_dir = args.output.parent / f"{args.output.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranked_chunks = sorted(chunk_records[1:], key=lambda row: max(
        np.linalg.norm(np.asarray(row["raw_new_chunk_xyz"]) - np.asarray(row["guided_new_chunk_xyz"]), axis=1)), reverse=True)[:6]
    for plot_index, row in enumerate(ranked_chunks, 1):
        old_xyz = np.asarray(row["old_chunk_xyz"])
        raw_xyz = np.asarray(row["raw_new_chunk_xyz"])
        guided_xyz = np.asarray(row["guided_new_chunk_xyz"])
        chunk_steps = np.arange(len(raw_xyz))
        figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for component, (axis, name) in enumerate(zip(axes, ("x", "y", "z"))):
            axis.plot(np.arange(len(old_xyz)), old_xyz[:, component], "k.-", label="old chunk remaining")
            axis.plot(chunk_steps, raw_xyz[:, component], "b.--", label="raw new chunk")
            axis.plot(chunk_steps, guided_xyz[:, component], "r.-", label="RTC-guided new chunk")
            axis.axvspan(-.5, args.inference_delay - .5, color="lightgreen", alpha=.12,
                         label="execution freeze" if component == 0 else None)
            axis.axvspan(-.5, H - args.execute_horizon - .5, color="mediumpurple", alpha=.08,
                         label=f"guided overlap (H-E={H - args.execute_horizon})" if component == 0 else None)
            axis.axvline(args.inference_delay - .5, color="purple", linestyle=":", linewidth=1.2,
                         label="RTC splice" if component == 0 else None)
            axis.set_ylabel(f"{name} (m)"); axis.grid(True, alpha=.25)
        axes[-1].set_xlabel("chunk row (0 = observation time)")
        axes[0].legend(loc="best", ncol=5, fontsize=8)
        figure.suptitle(f"RTC chunk comparison: cycle boundary={row['boundary']}, D={args.inference_delay}, E={args.execute_horizon}")
        figure.tight_layout()
        figure.savefig(chunk_dir / f"chunk_{plot_index:02d}_boundary_{row['boundary']:03d}.png", dpi=160)
        plt.close(figure)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
