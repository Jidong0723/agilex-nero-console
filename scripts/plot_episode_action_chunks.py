"""Plot saved model-input target TCP progress for one raw episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pose_xyz(row, key):
    pose = row.get(key) or {}
    value = pose.get("position_m")
    if not isinstance(value, list) or len(value) != 3:
        return [np.nan] * 3
    return [float(x) for x in value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.episode / "raw" / "robot_states.jsonl"
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("robot_states.jsonl is empty")

    step = np.asarray([int(row["sample_index"]) for row in rows])
    measured = np.asarray([pose_xyz(row, "measured_tcp_pose") for row in rows], dtype=float)
    target = np.asarray([pose_xyz(row, "target_tcp_pose") for row in rows], dtype=float)
    generation = np.asarray([
        int((row.get("identifiers") or {}).get("target_generation", -1)) for row in rows
    ])
    valid_target = np.isfinite(target).all(axis=1)
    valid_measured = np.isfinite(measured).all(axis=1)
    if not valid_target.any():
        raise SystemExit("episode has no target_tcp_pose samples")
    origin = target[np.flatnonzero(valid_target)[0]]
    target_progress = target - origin
    measured_progress = measured - origin
    changes = np.flatnonzero(valid_target[1:] & valid_target[:-1] & (generation[1:] != generation[:-1])) + 1

    if args.output is None:
        output = args.episode / "analysis" / "action_chunks_cumulative.png"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    colors = {"target": "#d97706", "measured": "#334155"}
    names = ("x", "y", "z")
    for component, (axis, name) in enumerate(zip(axes, names)):
        axis.plot(step[valid_measured], measured_progress[valid_measured, component],
                  color=colors["measured"], linewidth=1.1, label="measured TCP" if component == 0 else None)
        axis.plot(step[valid_target], target_progress[valid_target, component],
                  color=colors["target"], linewidth=1.25, label="input-source target TCP" if component == 0 else None)
        for index, boundary in enumerate(changes):
            axis.axvline(step[boundary], color="#7c3aed", linestyle=":", linewidth=.55, alpha=.3,
                         label="target_generation change" if component == 0 and index == 0 else None)
        axis.set_ylabel(f"{name} progress (m)")
        axis.grid(True, alpha=.25)
    axes[-1].set_xlabel("raw robot-state step (nominal 50 Hz)")
    axes[0].legend(loc="best", ncol=3, fontsize=9)
    duration = (rows[-1]["sample_target_monotonic_ns"] - rows[0]["sample_target_monotonic_ns"]) / 1e9
    generations = sorted(set(generation[valid_target].tolist()))
    fig.suptitle(
        f"Episode {args.episode.name}: input-source action target TCP cumulative progress\n"
        f"{len(rows)} raw steps / {duration:.2f} s · {len(generations)} target generations · "
        f"{len(changes)} boundaries"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)

    summary = {
        "episode": str(args.episode),
        "output": str(output),
        "raw_steps": int(len(rows)),
        "duration_s": duration,
        "target_samples": int(valid_target.sum()),
        "measured_samples": int(valid_measured.sum()),
        "target_generations": int(len(generations)),
        "target_generation_boundaries": int(len(changes)),
        "target_origin_position_m": origin.tolist(),
        "target_generation_first": int(generations[0]),
        "target_generation_last": int(generations[-1]),
        "target_samples_missing": int((~valid_target).sum()),
    }
    (output.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
