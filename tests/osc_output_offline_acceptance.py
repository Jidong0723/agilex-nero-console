"""Real Pink + isolated Pinocchio, CPV -> MIT PD surrogate -> CPV.

Never constructs a hardware backend, connects CAN, or calls the running
service. Selection persistence is redirected to a temporary directory.
The torque surrogate does NOT constitute real-arm dynamics validation.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion.osc import OscRuntime
from motion.osc_output import OutputSelection


def run():
    config = json.loads((ROOT / "config/osc.json").read_text(encoding="utf-8-sig"))
    runtime_config = json.loads((ROOT / "config/runtime.json").read_text(encoding="utf-8-sig"))
    config["tcp"] = {"offset_from_link7_m": runtime_config["sdk"]["task_tcp_offset_from_flange_m"], "verified": True}
    hardware = Mock()
    hardware.trigger_safety_fault.side_effect = AssertionError("offline run may not touch hardware")
    runtime = OscRuntime(hardware, Mock(), ROOT, config)
    phases = []
    with tempfile.TemporaryDirectory() as temporary:
        runtime._servo.output_selection = OutputSelection(Path(temporary))
        try:
            for mode in ("cpv", "impedance", "cpv"):
                runtime.select_output_mode(mode, "offline-acceptance", lambda: (_ for _ in ()).throw(AssertionError("hardware exit in shadow")))
                started = runtime.start_session(client_id="offline-acceptance", execution_mode="shadow")
                session_id = started["session"]["id"]
                anchor = copy.deepcopy(started["execution_sample"]["measured_tcp_pose"])
                sequence = 0
                phase = {"mode": mode, "anchor": anchor, "legs": []}
                # Let the LX zero-first-frame feedforward finish its slew ramp.
                time.sleep(.4)
                if runtime._servo.shadow_plant is not None and mode == "impedance":
                    pose = runtime._servo._current_tcp_pose(runtime._servo.shadow_plant.q)
                    phase["surrogate_startup_drift_m"] = math.dist(pose["position_m"], anchor["position_m"])
                for offset in (.002, 0.0):
                    target = copy.deepcopy(anchor)
                    target["position_m"][0] += offset
                    sequence += 1
                    runtime.submit_absolute_target({"client_id": "offline-acceptance", "session_id": session_id,
                        "sequence": sequence, "payload": {"target_pose": target}}, mode="track_tcp")
                    until = time.monotonic() + (12.0 if mode == "impedance" else 4.0)
                    errors = []
                    while time.monotonic() < until:
                        runtime.heartbeat("offline-acceptance", session_id)
                        status = runtime.status()
                        if status["diagnostics"]["trajectory_state"] == "FAULT":
                            raise RuntimeError(status["last_result"])
                        sample = status.get("execution_sample") or {}
                        if sample.get("target_generation") == status["target_generation"]:
                            errors.append(float(sample.get("position_error_m", 1)))
                        time.sleep(.05)
                    if not errors:
                        raise AssertionError("no coherent Pink execution samples")
                    if errors[-1] > float(config["osc"]["arrival_position_tolerance_m"]):
                        raise AssertionError(f"{mode} did not converge: {errors[-1]} m")
                    phase["legs"].append({"target_x_offset_m": offset,
                        "initial_error_m": errors[0], "final_error_m": errors[-1],
                        "best_error_m": min(errors), "samples": len(errors)})
                runtime.request_shadow_hold("offline manual HOLD")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and runtime.status()["diagnostics"]["trajectory_state"] != "HOLD_READY":
                    time.sleep(.02)
                status = runtime.status()
                phase["hold_state"] = status["diagnostics"]["trajectory_state"]
                phase["output_count"] = status["diagnostics"]["output_count"]
                phase["impedance"] = status["impedance"]
                phase["last_output"] = status["last_output"]
                if phase["hold_state"] != "HOLD_READY" or phase["output_count"] <= 0:
                    raise AssertionError(phase)
                if mode == "cpv" and status["impedance"]["loaded"]:
                    raise AssertionError("CPV started impedance runtime")
                phases.append(phase)
                print(json.dumps(phase, ensure_ascii=False), flush=True)
            if hardware.mock_calls:
                raise AssertionError(f"shadow hardware calls: {hardware.mock_calls}")
            return {"ok": True, "physical_validation": False, "phases": phases}
        finally:
            runtime.close()


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
