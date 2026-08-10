"""Deterministic local runtime discovery for the two supported Python environments."""
from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def control_python(root: Path | None = None) -> Path:
    base = root or project_root()
    executable = base / ".venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    if not executable.is_file():
        raise RuntimeError(f"control Python is missing: {executable}; run setup.ps1")
    return executable.resolve()


def kinematics_python(root: Path | None = None) -> Path:
    base = root or project_root()
    executable = base / ".conda" / "nero-kinematics" / ("python.exe" if platform.system() == "Windows" else "bin/python")
    if not executable.is_file():
        raise RuntimeError(f"kinematics Python is missing: {executable}; run setup-kinematics.ps1")
    return executable.resolve()


def assert_control_interpreter(root: Path | None = None) -> Path:
    expected = control_python(root)
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise RuntimeError(f"control service must run with project .venv Python ({expected}), not {actual}")
    return expected


def _module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"available": spec is not None, "origin": str(spec.origin) if spec and spec.origin else None}


def doctor_report(root: Path | None = None) -> dict[str, Any]:
    base = (root or project_root()).resolve()
    report: dict[str, Any] = {
        "project_root": str(base),
        "current_python": str(Path(sys.executable).resolve()),
        "control_python": str(control_python(base)),
        "kinematics_python": str(kinematics_python(base)),
        "control_modules": {name: _module_status(name) for name in ("numpy", "ruckig", "pyAgxArm", "agx_cando", "can")},
        "kinematics_config": str(base / "environment-kinematics.yml"),
        "teleop_config": str(base / "config" / "teleop.json"),
    }
    try:
        import can
        report["can"] = {"agx_cando": can.detect_available_configs(["agx_cando"])}
    except Exception as exc:
        report["can"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def doctor_json(root: Path | None = None) -> str:
    return json.dumps(doctor_report(root), ensure_ascii=False, indent=2)

