from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shared.schemas import Action, RobotState


class SafetyViolation(ValueError):
    pass


@dataclass
class SafetyConfig:
    min_flange_z_m: float = -0.01
    workspace_min_xyz_m: tuple[float, float, float] = (-0.45, -0.45, -0.01)
    workspace_max_xyz_m: tuple[float, float, float] = (0.45, 0.60, 0.70)
    min_speed_percent: int = 1
    max_speed_percent: int = 5
    min_gripper_width_m: float = 0.0
    max_gripper_width_m: float = 0.10
    max_gripper_force_n: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SafetyConfig":
        if not data:
            return cls()
        kwargs = {
            "min_flange_z_m": data.get("min_flange_z_m", cls.min_flange_z_m),
            "min_speed_percent": data.get("min_speed_percent", cls.min_speed_percent),
            "max_speed_percent": data.get("max_speed_percent", cls.max_speed_percent),
            "min_gripper_width_m": data.get("min_gripper_width_m", cls.min_gripper_width_m),
            "max_gripper_width_m": data.get("max_gripper_width_m", cls.max_gripper_width_m),
            "max_gripper_force_n": data.get("max_gripper_force_n", cls.max_gripper_force_n),
        }
        if "workspace_min_xyz_m" in data:
            kwargs["workspace_min_xyz_m"] = tuple(data["workspace_min_xyz_m"])
        if "workspace_max_xyz_m" in data:
            kwargs["workspace_max_xyz_m"] = tuple(data["workspace_max_xyz_m"])
        return cls(**kwargs)


def finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise SafetyViolation(f"{name} must be numeric.")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise SafetyViolation(f"{name} must be finite.")
    return value


def finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SafetyViolation(f"{name} must be a list of length {length}.")
    return [finite_number(item, f"{name}[{idx}]") for idx, item in enumerate(value)]


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
    """Return true only for a stale Cartesian IK failure with no hardware error bits."""
    if not isinstance(status_msg, dict):
        return False
    try:
        status = int(status_msg.get("arm_status"))
    except Exception:
        return False
    err = status_msg.get("err_status")
    return status == 0x02 and not (isinstance(err, dict) and any(bool(value) for value in err.values()))


class SafetyLayer:
    def __init__(self, config: SafetyConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, SafetyConfig) else SafetyConfig.from_dict(config)

    def check_state(self, state: RobotState | None) -> None:
        if state is None:
            return
        if arm_status_has_error(state.arm_status):
            raise SafetyViolation(f"Unsafe arm status: {state.arm_status}")
        pose = state.flange_pose
        if pose and len(pose) >= 3 and float(pose[2]) < self.config.min_flange_z_m:
            raise SafetyViolation(f"Flange z below safety threshold: {pose[2]} < {self.config.min_flange_z_m}")

    def validate(self, action: dict[str, Any] | Action, state: RobotState | None = None) -> dict[str, Any]:
        if isinstance(action, Action):
            data = action.to_dict()
        else:
            data = dict(action)
        action_type = data.get("type")
        if not isinstance(action_type, str):
            raise SafetyViolation("Action type must be a string.")
        # NO_SOLUTION describes the previous Cartesian request. Permit only a
        # newly validated Cartesian pose to recover it; all hardware errors and
        # every other action remain blocked.
        no_solution_recovery = (
            action_type == "cartesian_pose"
            and state is not None
            and arm_status_is_no_solution_only(state.arm_status)
        )
        if not no_solution_recovery:
            self.check_state(state)
        elif state.flange_pose and float(state.flange_pose[2]) < self.config.min_flange_z_m:
            raise SafetyViolation(
                f"Flange z below safety threshold: {state.flange_pose[2]} < {self.config.min_flange_z_m}"
            )
        if action_type == "joint_target":
            return self._validate_joint_target(data)
        if action_type == "cartesian_delta":
            return self._validate_cartesian_delta(data)
        if action_type == "cartesian_pose":
            return self._validate_cartesian_pose(data, state)
        if action_type == "gripper":
            return self._validate_gripper(data)
        if action_type == "stop":
            return {"type": "stop", "reason": str(data.get("reason", "operator/model stop"))}
        raise SafetyViolation(f"Unsupported action type: {action_type}")

    def _validate_speed(self, data: dict[str, Any]) -> int:
        speed = int(finite_number(data.get("speed_percent", 3), "speed_percent"))
        if not self.config.min_speed_percent <= speed <= self.config.max_speed_percent:
            raise SafetyViolation(f"speed_percent {speed} outside [{self.config.min_speed_percent}, {self.config.max_speed_percent}]")
        return speed

    def _validate_joint_target(self, data: dict[str, Any]) -> dict[str, Any]:
        joints = finite_vector(data.get("joint_angles_rad"), 7, "joint_angles_rad")
        return {"type": "joint_target", "joint_angles_rad": joints, "speed_percent": self._validate_speed(data)}

    def _validate_cartesian_delta(self, data: dict[str, Any]) -> dict[str, Any]:
        delta_xyz = finite_vector(data.get("delta_xyz_m"), 3, "delta_xyz_m")
        delta_rpy = finite_vector(data.get("delta_rpy_rad", [0.0, 0.0, 0.0]), 3, "delta_rpy_rad")
        return {
            "type": "cartesian_delta",
            "delta_xyz_m": delta_xyz,
            "delta_rpy_rad": delta_rpy,
            "speed_percent": self._validate_speed(data),
        }

    def _validate_cartesian_pose(self, data: dict[str, Any], state: RobotState | None) -> dict[str, Any]:
        pose = finite_vector(data.get("pose"), 6, "pose")
        motion_mode = str(data.get("motion_mode", "point")).lower()
        if motion_mode not in {"point", "linear"}:
            raise SafetyViolation("cartesian pose motion_mode must be 'point' or 'linear'.")
        self._check_workspace(pose[:3])
        return {
            "type": "cartesian_pose", "pose": pose,
            "speed_percent": self._validate_speed(data), "motion_mode": motion_mode,
        }

    def _validate_gripper(self, data: dict[str, Any]) -> dict[str, Any]:
        width = finite_number(data.get("width_m"), "width_m")
        force = finite_number(data.get("force_n", 1.0), "force_n")
        if not self.config.min_gripper_width_m <= width <= self.config.max_gripper_width_m:
            raise SafetyViolation(f"gripper width {width} outside configured range.")
        if not 0.0 <= force <= self.config.max_gripper_force_n:
            raise SafetyViolation(f"gripper force {force} outside configured range.")
        return {"type": "gripper", "width_m": width, "force_n": force}

    def _check_workspace(self, xyz: list[float]) -> None:
        min_xyz = self.config.workspace_min_xyz_m
        max_xyz = self.config.workspace_max_xyz_m
        for idx, value in enumerate(xyz):
            if not min_xyz[idx] <= value <= max_xyz[idx]:
                raise SafetyViolation(f"pose xyz[{idx}]={value} outside workspace [{min_xyz[idx]}, {max_xyz[idx]}].")
