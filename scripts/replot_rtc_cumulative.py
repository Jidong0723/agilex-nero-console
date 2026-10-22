"""Replot a saved RTC evaluation with the direct synchronous baseline.

The saved evaluation already contains each plain model response.  This utility
reconstructs the ordinary baseline by executing rows 0..K-1 at every boundary,
without rerunning the model or applying RTC splice logic to that curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rtc_cumulative_eval import H, load_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = json.loads(args.input.read_text(encoding="utf-8"))
    tasks = {
        int(row["task_index"]): row["task"]
        for row in (
            json.loads(line)
            for line in (args.dataset / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    _, target_xyz, _ = load_episode(
        args.dataset, saved["episode"], tasks, saved["start"], saved["steps"]
    )
    steps = int(saved["steps"])
    boundaries = saved["boundaries"]
    execute = int(saved["execute_horizon"])
    delay = int(saved["inference_delay"])
    initial = target_xyz[0]

    normal = np.zeros((steps, 3), dtype=np.float64)
    for row, boundary in zip(saved["chunks"], boundaries):
        raw = np.asarray(row["raw_new_chunk_xyz"], dtype=np.float64)
        end = min(boundary + execute, steps)
        # `raw` is decoded against the recorded dataset observation.  Preserve
        # its relative displacement but anchor every new ordinary chunk at the
        # prior executed blue endpoint, yielding a continuous counterfactual
        # rollout rather than snapping back to the black dataset trace.
        anchor = target_xyz[boundary] if boundary == 0 else normal[boundary - 1]
        normal[boundary:end] = anchor + (raw[: end - boundary, :3] - target_xyz[boundary])

    output_json = dict(saved)
    output_json["trajectory"] = dict(saved["trajectory"])
    output_json["trajectory"]["synchronous_cumulative_xyz"] = (normal - initial).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(output_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    target = target_xyz - initial
    rtc = np.asarray(saved["trajectory"]["rtc_cumulative_xyz"], dtype=np.float64)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = np.arange(steps)
    colors = {"target": "black", "synchronous": "#2463eb", "rtc": "#e11d48"}
    chunks = saved["chunks"]
    splices = saved["rtc_splices"]
    for component, (axis, name) in enumerate(zip(axes, ("x", "y", "z"))):
        axis.plot(x, target[:, component], "k-", linewidth=1.7, label="dataset target")
        axis.plot(x, normal[:, component] - initial[component], color=colors["synchronous"],
                  linewidth=1.2, label="Synchronous H20 (replan every 8)")
        axis.plot(x, rtc[:, component], color=colors["rtc"], linewidth=1.2, label="Official RTC")
        for boundary in boundaries[1:]:
            axis.axvspan(boundary + delay - .5, boundary + (H - execute) - .5,
                         color="gold", alpha=.18,
                         label="soft mask (0<w<1)" if component == 0 and boundary == boundaries[1] else None)
            axis.axvline(boundary - .5, color="darkorange", linestyle="--", linewidth=.8, alpha=.65,
                         label="replan boundary" if component == 0 and boundary == boundaries[1] else None)
        for splice in splices:
            axis.axvline(splice - .5, color="purple", linestyle=":", linewidth=.9, alpha=.7,
                         label=f"RTC splice (D={delay})" if component == 0 and splice == splices[0] else None)
        for old in chunks[1:]:
            # Full reference Y passed to the sampler.  Rows 0..D-1 are the
            # hard-mask rows and share the same time indices as the dashed
            # guided prefix immediately below.
            reference = np.asarray(old["old_chunk_xyz"], dtype=np.float64)
            reference_x = np.arange(old["boundary"], old["boundary"] + len(reference))
            hard = (reference_x < steps) & (np.arange(len(reference)) < delay)
            if np.any(hard):
                axis.plot(reference_x[hard], reference[hard, component] - initial[component],
                          color="#0f766e", linestyle=":", linewidth=1.1, alpha=.9,
                          label="reference Y hard-mask rows" if component == 0 and old is chunks[1] else None)
            # Guided rows 0..D-1 are generated to condition the new chunk,
            # but old-plan rows are executed during those time slots instead.
            prefix = np.asarray(old["guided_new_chunk_xyz"], dtype=np.float64)[:delay]
            prefix_x = np.arange(old["boundary"], old["boundary"] + len(prefix))
            visible = prefix_x < steps
            if np.any(visible):
                axis.plot(prefix_x[visible], prefix[visible, component] - initial[component],
                          color="#fb7185", linestyle="--", linewidth=.85, alpha=.7,
                          label="new chunk guided prefix (not executed)" if component == 0 and old is chunks[1] else None)
            # The old plan has already been shifted by E; its rows from D on
            # are its remaining future that will be superseded by the new plan.
            tail = reference[delay:]
            tail_x = np.arange(old["boundary"] + delay,
                               old["boundary"] + delay + len(tail))
            visible = tail_x < steps
            if np.any(visible):
                axis.plot(tail_x[visible], tail[visible, component] - initial[component],
                          color="dimgray", linestyle="--", linewidth=.75, alpha=.45,
                          label="reference Y soft-mask tail (not executed)" if component == 0 and old is chunks[1] else None)
        axis.set_ylabel(f"cumulative {name} (m)")
        axis.grid(True, alpha=.25)
    axes[-1].set_xlabel("step (50 ms; step 0 = observation start)")
    axes[0].legend(loc="best", ncol=4, fontsize=9)
    figure.suptitle(f"Cumulative TCP progress: episode {saved['episode']}, {steps} steps, "
                    f"D={delay}, E={execute}")
    figure.tight_layout()
    figure.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
