"""Read-only SDK telemetry boundary, outside the OSC control loop."""
from __future__ import annotations

import time
from typing import Any


class TelemetryReader:
    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def status_sample(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
        started = time.monotonic()
        control = self._robot.get_control_state()
        robot_state = control.get("robot") or {}
        try:
            gripper = self._robot.read_gripper().to_dict()
        except Exception as exc:
            gripper = {"raw": {"ok": False, "error": f"{type(exc).__name__}: {exc}"}}
        return control, robot_state, gripper, (time.monotonic() - started) * 1000.0

    def observation(self, include_motor_states: bool = False) -> dict[str, Any]:
        return self._robot.get_observation(include_motor_states=include_motor_states).to_dict()
