"""Safely verify PICO translation/rotation mapping without commanding a robot.

This is deliberately an adapter-level diagnostic: it uses the exact runtime
mapping, anchor logic and target construction, but a local OSC recorder instead
of the control service.  It prints the four boundaries that matter in a field
test: PICO input delta -> mapped NERO-base delta -> submitted TCP target -> TCP
orientation delta.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Support the documented direct invocation from the repository root as well as
# ``python -m tests.pico_axis_diagnostic``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervisor.pico_adapter import PicoInputAdapter


IDENTITY_Q = [0.0, 0.0, 0.0, 1.0]
TCP_ANCHOR = {"position_m": [0.10, 0.20, 0.30], "orientation_xyzw": IDENTITY_Q}


class RecordingOsc:
    """Minimal safe OSC replacement: record targets but never contact hardware."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self._state = {
            "session": {"state": "ACTIVE", "id": "pico-axis-diagnostic",
                        "client_id": "diagnostic", "execution_mode": "shadow"},
            "command": {"target_tcp": TCP_ANCHOR},
            "execution": {"measured_tcp_pose": TCP_ANCHOR},
        }

    def state(self) -> dict[str, Any]:
        return self._state

    def track_tcp(self, session_id: str, client_id: str, sequence: int,
                  target_pose: dict[str, Any]) -> dict[str, Any]:
        self.commands.append({"session_id": session_id, "client_id": client_id,
                              "sequence": sequence, "target_pose": target_pose})
        return {"ok": True, "result": {"accepted": True, "accepted_sequence": sequence}}

    def hold(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def gripper(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def heartbeat(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def close(left: list[float], right: list[float], tolerance: float = 1e-9) -> bool:
    return all(math.isclose(actual, expected, abs_tol=tolerance) for actual, expected in zip(left, right))


def run(config_path: Path, distance_m: float) -> dict[str, Any]:
    runtime = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config = runtime.get("pico_adapter")
    if not isinstance(config, dict):
        raise ValueError(f"pico_adapter is missing from {config_path}")
    osc = RecordingOsc()
    adapter = PicoInputAdapter(osc, config)
    adapter.begin_connection("pico-axis-diagnostic", "diagnostic")
    adapter.connected()
    origin = [0.0, 0.0, 0.0]
    adapter.anchor_begin({"position_m": origin, "orientation_xyzw": IDENTITY_Q})

    axes: list[dict[str, Any]] = []
    for axis_index, label in enumerate(("X", "Y", "Z"), start=0):
        position = [0.0, 0.0, 0.0]
        position[axis_index] = distance_m
        adapter.pose({"position_m": position, "orientation_xyzw": IDENTITY_Q,
                      "tracking_valid": True, "_pico_sequence": axis_index + 1})
        target = osc.commands[-1]["target_pose"]
        snapshot = adapter.snapshot()
        target_delta = subtract(target["position_m"], TCP_ANCHOR["position_m"])
        mapped_delta = snapshot["mapped_translation_from_anchor_m"]
        axes.append({
            "pico_input_delta_m": position,
            "mapped_nero_base_delta_m": mapped_delta,
            "tcp_target_delta_in_nero_base_m": target_delta,
            "tcp_orientation_delta_axis_angle_degrees": snapshot["target_rotation_from_anchor_degrees"],
            "pass": close(mapped_delta, target_delta) and close(
                snapshot["target_rotation_from_anchor_degrees"], [0.0, 0.0, 0.0]),
            "axis": f"PICO +{label}",
        })

    rotation_only: list[dict[str, Any]] = []
    for axis_index, label in enumerate(("X", "Y", "Z"), start=0):
        angle = math.radians(30.0)
        orientation = [0.0, 0.0, 0.0, math.cos(angle / 2.0)]
        orientation[axis_index] = math.sin(angle / 2.0)
        adapter.pose({"position_m": origin, "orientation_xyzw": orientation,
                      "tracking_valid": True, "_pico_sequence": axis_index + 4})
        target = osc.commands[-1]["target_pose"]
        target_delta = subtract(target["position_m"], TCP_ANCHOR["position_m"])
        rotation_only.append({
            "axis": f"PICO local +{label} 30 deg",
            "tcp_target_delta_in_nero_base_m": target_delta,
            "tcp_orientation_delta_axis_angle_degrees": adapter.snapshot()["target_rotation_from_anchor_degrees"],
            "pass": close(target_delta, [0.0, 0.0, 0.0]),
        })
    return {
        "safe": True,
        "note": "No OSC service, CAN bus, USB gateway, or robot command was used.",
        "config_path": str(config_path),
        "distance_m": distance_m,
        "position_axis_map": config["position_axis_map"],
        "translation": axes,
        "rotation_only": rotation_only,
        "pass": all(row["pass"] for row in axes + rotation_only),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/runtime.json"))
    parser.add_argument("--distance-m", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.distance_m <= 1.0:
        parser.error("--distance-m must be in (0, 1]")
    result = run(args.config, args.distance_m)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
