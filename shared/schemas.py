from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return repr(value)
        return value
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item, depth + 1) for key, item in value.items()}
    if is_dataclass(value):
        return jsonable(asdict(value), depth + 1)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return jsonable(value.to_dict(), depth + 1)
        except Exception:
            pass
    field_names = getattr(value, "_fields_", None)
    if isinstance(field_names, (list, tuple)):
        return {str(key): jsonable(getattr(value, key), depth + 1) for key in field_names if hasattr(value, key)}
    if hasattr(value, "__dict__"):
        data: dict[str, Any] = {}
        for key, item in vars(value).items():
            if key.startswith("_"):
                continue
            data[key] = jsonable(item, depth + 1)
        for key in (
            "ctrl_mode",
            "arm_status",
            "mode_feedback",
            "teach_status",
            "motion_status",
            "trajectory_num",
            "err_status",
            "status_code",
            "msg_type",
            "timestamp",
            "hz",
            "msg",
        ):
            if key not in data and hasattr(value, key):
                try:
                    data[key] = jsonable(getattr(value, key), depth + 1)
                except Exception:
                    pass
        return data
    return repr(value)


def unwrap_msg(record: dict[str, Any] | None) -> Any:
    if not isinstance(record, dict) or not record.get("ok"):
        return None
    value = record.get("value")
    if isinstance(value, dict) and "msg" in value:
        return value["msg"]
    return value


def as_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        pose_keys = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")
        if all(key in value for key in pose_keys):
            return [value[key] for key in pose_keys]
        joint_keys = [f"joint_{idx}" for idx in range(1, 8)]
        if all(key in value for key in joint_keys):
            return [value[key] for key in joint_keys]
        for key in ("value", "joint_angles", "angles", "data", "msg"):
            item = value.get(key)
            if isinstance(item, list):
                return item
    return None


@dataclass
class RobotState:
    timestamp: str
    joint_angles_rad: list[float] | None = None
    flange_pose: list[float] | None = None
    tcp_pose: list[float] | None = None
    arm_status: dict[str, Any] | None = None
    joint_enable_status: list[bool] | None = None
    motor_states: list[dict[str, Any]] = field(default_factory=list)
    driver_states: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class GripperState:
    timestamp: str
    width_m: float | None = None
    force_n: float | None = None
    mode: Any | None = None
    status: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class Action:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        payload = dict(data)
        action_type = str(payload.pop("type", ""))
        return cls(type=action_type, payload=payload)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **jsonable(self.payload)}


@dataclass
class ExecutedAction:
    timestamp: str
    requested_action: dict[str, Any]
    safe_action: dict[str, Any] | None
    ok: bool
    sent_to_robot: bool
    reason: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)
