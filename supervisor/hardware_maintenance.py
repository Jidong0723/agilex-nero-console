"""Explicit hardware-maintenance operations; never used by the OSC servo loop."""
from __future__ import annotations

import time
from typing import Any


class HardwareMaintenance:
    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def read_cpv_parameters(self) -> dict[str, Any]:
        names = ("acc", "dcc", "cv", "pp", "kp", "ki")
        started_ns = time.monotonic_ns()
        values = {name: [self._robot.call("p2", "read_cpv_parameter", joint, name,
                                          category="maintenance_cpv_read") for joint in range(1, 8)]
                  for name in names}
        missing = [f"{name}:J{joint + 1}" for name, row in values.items()
                   for joint, value in enumerate(row) if value is None]
        return {"status": "available" if not missing else "partial", "values": values,
                "missing": missing, "read_started_monotonic_ns": started_ns,
                "read_finished_monotonic_ns": time.monotonic_ns()}
