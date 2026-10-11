"""Validated software-only tuning snapshots shared by HTTP and the servo."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import MappingProxyType


# key: (section, label, lower, upper, default, display scale, unit)
FIELDS = {
    "frame_gain": ("solver", "跟踪增益", .3, .8, .8, 1, ""),
    "frame_position_cost": ("solver", "位置权重", 2., 10., 4., 1, ""),
    "frame_orientation_cost": ("solver", "姿态权重", .5, 5., 1., 1, ""),
    "damping_cost": ("solver", "速度正则", .0001, .1, .03, 1, ""),
    "frame_lm_damping": ("solver", "LM 阻尼", .1, 5., 1., 1, ""),
    "posture_cost": ("solver", "姿势保持权重", .0001, .01, .0001, 1, ""),
    "joint_center_cost": ("solver", "关节居中权重", 0., .003, .0003, 1, ""),
    "joint_center_deadband": ("solver", "关节居中死区", .6, .8, .8, 1, ""),
    "arrival_position_tolerance_m": ("osc", "到位位置容差", .0001, .005, .0005, 1000, "mm"),
    "arrival_orientation_tolerance_rad": ("osc", "到位姿态容差", math.radians(.05), math.radians(2), math.radians(.25), 180 / math.pi, "°"),
    "arrival_joint_velocity_tolerance_rad_s": ("osc", "到位关节速度容差", .001, .02, .005, 1, "rad/s"),
    "arrival_dwell_s": ("osc", "到位保持时间", .05, 2., .25, 1000, "ms"),
}


class OscTuning:
    def __init__(self, config: dict, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.apply_lock = threading.Lock()
        values = {key: config.get(spec[0], {}).get(key, spec[4]) for key, spec in FIELDS.items()}
        self._snapshot = (0, MappingProxyType(self.validate(values)))
        self._saved = dict(self._snapshot[1])

    @staticmethod
    def validate(values: dict) -> dict:
        if not isinstance(values, dict) or set(values) != set(FIELDS):
            raise ValueError("OSC 参数必须包含完整的 12 项软件参数，且不得包含其他参数")
        checked = {}
        for key, spec in FIELDS.items():
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} 必须是数值")
            value = float(value)
            if not math.isfinite(value) or not spec[2] <= value <= spec[3]:
                raise ValueError(f"{key} 超出允许范围 [{spec[2]}, {spec[3]}]")
            checked[key] = value
        return checked

    def snapshot(self):
        with self.lock:
            return self._snapshot

    def public(self):
        with self.lock:
            return {"revision": self._snapshot[0], "running": dict(self._snapshot[1]),
                    "startup": dict(self._saved), "fields": {
                        key: {"label": spec[1], "min": spec[2], "max": spec[3],
                              "scale": spec[5], "unit": spec[6]}
                        for key, spec in FIELDS.items()}}

    def apply(self, values: dict, save: bool = False, expected_revision: int | None = None):
        checked = self.validate(values)
        if not isinstance(save, bool):
            raise ValueError("save_startup 必须是布尔值")
        with self.apply_lock:
            if expected_revision is not None and expected_revision != self._snapshot[0]:
                raise ValueError("OSC 参数已被其他请求更新，请重新载入运行值")
            if save:
                # Save succeeds before publication. A disk error leaves the
                # entire running snapshot unchanged.
                config = json.loads(self.path.read_text(encoding="utf-8-sig"))
                for key, value in checked.items():
                    config.setdefault(FIELDS[key][0], {})[key] = value
                temporary = self.path.with_suffix(".json.tuning.tmp")
                temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(self.path)
            with self.lock:
                if save:
                    self._saved = dict(checked)
                self._snapshot = (self._snapshot[0] + 1, MappingProxyType(checked))
            return self.public()
