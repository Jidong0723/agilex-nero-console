"""Shared interpretation helpers for NERO feedback status.

Command validation lives in the OperationalSpaceController.  This module does
not expose a second action API or any legacy speed-percentage configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SafetyConfig:
    """Static gripper limits retained outside OSC TCP command validation."""

    min_gripper_width_m: float = 0.0
    max_gripper_width_m: float = 0.10
    max_gripper_force_n: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SafetyConfig":
        data = data or {}
        return cls(
            min_gripper_width_m=float(data.get("min_gripper_width_m", cls.min_gripper_width_m)),
            max_gripper_width_m=float(data.get("max_gripper_width_m", cls.max_gripper_width_m)),
            max_gripper_force_n=float(data.get("max_gripper_force_n", cls.max_gripper_force_n)),
        )


class SafetyLayer:
    """Compatibility holder for non-motion gripper limits only."""

    def __init__(self, config: SafetyConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, SafetyConfig) else SafetyConfig.from_dict(config)


def enum_is_zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except Exception:
        text = str(value)
        return text.endswith("(0x0)") or text == "0"


def arm_status_has_error(status_msg: Any) -> bool:
    if not isinstance(status_msg, dict):
        return True
    if not enum_is_zero(status_msg.get("arm_status")):
        return True
    err = status_msg.get("err_status")
    return isinstance(err, dict) and any(bool(value) for value in err.values())


def arm_status_is_no_solution_only(status_msg: Any) -> bool:
    """Return true only for a stale Cartesian IK failure with no hardware bits."""
    if not isinstance(status_msg, dict):
        return False
    try:
        status = int(status_msg.get("arm_status"))
    except Exception:
        return False
    err = status_msg.get("err_status")
    return status == 0x02 and not (isinstance(err, dict) and any(bool(value) for value in err.values()))
