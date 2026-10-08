from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JOINT_COUNT = 7
V120_TORQUE_LIMIT_NM = [16.0] * JOINT_COUNT


def _vector(value: Any, name: str, length: int = JOINT_COUNT) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


class NeroJointMapping:
    """Single SDK/URDF position and torque convention boundary."""

    def __init__(self, conventions: Any = None) -> None:
        items = conventions if isinstance(conventions, list) else []
        if not items:
            items = [{"sdk_index": i + 1, "urdf_joint": f"joint{i + 1}", "sign": 1.0,
                      "zero_offset_rad": 0.0, "torque_sign": 1.0} for i in range(7)]
        if len(items) != 7:
            raise ValueError("joint_conventions must contain seven entries")
        self._entries = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("joint convention must be an object")
            index = int(item.get("sdk_index", 0))
            if index not in range(1, 8) or index in seen:
                raise ValueError("sdk_index must be a unique value from 1 to 7")
            seen.add(index)
            sign = float(item.get("sign", 1.0))
            torque_sign = float(item.get("torque_sign", 1.0))
            offset = float(item.get("zero_offset_rad", 0.0))
            if sign not in (-1.0, 1.0) or torque_sign not in (-1.0, 1.0) or not math.isfinite(offset):
                raise ValueError("joint convention signs must be +/-1 and offset finite")
            self._entries.append((index - 1, str(item.get("urdf_joint", f"joint{index}")), sign, offset, torque_sign))
        self._entries.sort(key=lambda entry: entry[0])

    def to_urdf_q(self, q_sdk: Any) -> list[float]:
        values = _vector(q_sdk, "q_sdk")
        result = [0.0] * 7
        for sdk_index, _, sign, offset, _ in self._entries:
            result[sdk_index] = sign * (values[sdk_index] - offset)
        return result

    def to_sdk_q(self, q_urdf: Any) -> list[float]:
        values = _vector(q_urdf, "q_urdf")
        result = [0.0] * 7
        for sdk_index, _, sign, offset, _ in self._entries:
            result[sdk_index] = sign * values[sdk_index] + offset
        return result

    def to_sdk_tau(self, tau_urdf: Any) -> list[float]:
        values = _vector(tau_urdf, "tau_urdf")
        result = [0.0] * 7
        for sdk_index, _, _, _, torque_sign in self._entries:
            result[sdk_index] = torque_sign * values[sdk_index]
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        return [{"sdk_index": i + 1, "urdf_joint": name, "sign": sign,
                 "zero_offset_rad": offset, "torque_sign": torque_sign}
                for i, name, sign, offset, torque_sign in self._entries]


class GravityModel:
    """Lazy Pinocchio gravity model. Payload is attached at the fixed tool frame."""

    def __init__(self, urdf: str | Path, conventions: Any = None, tool_profiles: dict[str, Any] | None = None,
                 model_revision: str = "official-nero-2dc30fca68cbf4e04d1d0bc15c123d026380ece7") -> None:
        self.urdf = Path(urdf)
        self.mapping = NeroJointMapping(conventions)
        self.tool_profiles = copy.deepcopy(tool_profiles or {})
        self.model_revision = model_revision
        self._pin = None
        self._model = None
        self._data = None
        self._base_model = None
        self._loaded_profile = None

    def _load(self, tool_profile: str) -> None:
        import pinocchio as pin
        self._pin = pin
        urdf_argument = str(self.urdf)
        try:
            urdf_argument = os.path.relpath(self.urdf, Path.cwd())
        except ValueError:
            pass
        self._base_model = pin.buildModelFromUrdf(urdf_argument)
        if self._base_model.nq != 7 or self._base_model.nv != 7:
            raise ValueError("NERO gravity model must be 7-DOF")
        self._base_model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=float)
        profile = self.tool_profiles.get(tool_profile, {"mass_kg": 0.0, "com_xyz_m": [0.0] * 3,
                                                          "inertia_kg_m2": [0.0] * 6, "enabled": True,
                                                          "validated": True})
        if not isinstance(profile, dict) or not profile.get("enabled", False) or not profile.get("validated", False):
            raise ValueError(f"tool profile {tool_profile!r} is not enabled and validated")
        mass = float(profile.get("mass_kg", 0.0))
        com = _vector(profile.get("com_xyz_m", [0.0] * 3), "com_xyz_m", 3)
        inertia_values = _vector(profile.get("inertia_kg_m2", [0.0] * 6), "inertia_kg_m2", 6)
        if mass < 0.0:
            raise ValueError("tool mass must be non-negative")
        self._model = self._base_model
        if mass > 0.0:
            # Inertia constructor is stable across supported Pinocchio releases.
            mat = np.zeros((3, 3), dtype=float)
            mat[0, 0], mat[0, 1], mat[0, 2] = inertia_values[0], inertia_values[1], inertia_values[2]
            mat[1, 0], mat[1, 1], mat[1, 2] = inertia_values[1], inertia_values[3], inertia_values[4]
            mat[2, 0], mat[2, 1], mat[2, 2] = inertia_values[2], inertia_values[4], inertia_values[5]
            inertia = pin.Inertia(mass, np.asarray(com, dtype=float), mat)
            self._model = self._base_model.copy()
            joint_id = self._model.getJointId("joint7")
            # The payload is mounted at the official fixed end-effector
            # frame, not at the joint-7 origin.  The URDF places that frame at
            # (0.033, 0, -0.0235) m relative to joint7; using Identity here
            # silently changes the gravity moment arm and was especially
            # misleading for J2/J4 on the real arm.
            frame_id = self._model.getFrameId("end_effector")
            if frame_id <= 0:
                raise ValueError("official URDF end_effector frame is missing")
            tool_placement = self._model.frames[frame_id].placement
            self._model.appendBodyToJoint(joint_id, inertia, tool_placement)
        self._data = self._model.createData()
        self._loaded_profile = tool_profile

    def compute_gravity(self, q_actual_rad: Any, *, tool_profile: str = "bare_flange", sample_id: int = 0) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            q_sdk = _vector(q_actual_rad, "q_actual_rad")
            if self._model is None or self._loaded_profile != tool_profile:
                self._load(tool_profile)
            q = np.asarray(self.mapping.to_urdf_q(q_sdk), dtype=float)
            tau = self._pin.computeGeneralizedGravity(self._model, self._data, q)
            values = [float(item) for item in tau]
            if len(values) != 7 or not all(math.isfinite(item) for item in values):
                raise ValueError("Pinocchio returned non-finite gravity torque")
            mapped = self.mapping.to_sdk_tau(values)
            return {"ok": True, "q_actual_rad": q_sdk, "tau_gravity_urdf_nm": values,
                    "tau_gravity_sdk_nm": mapped, "sample_id": int(sample_id),
                    "tool_profile": tool_profile, "model_revision": self.model_revision,
                    "computed_monotonic_ns": time.monotonic_ns(),
                    "compute_time_ms": (time.perf_counter() - started) * 1000.0}
        except Exception as exc:
            return {"ok": False, "q_actual_rad": list(q_actual_rad) if isinstance(q_actual_rad, list) else None,
                    "sample_id": int(sample_id), "tool_profile": tool_profile,
                    "model_revision": self.model_revision, "compute_time_ms": (time.perf_counter() - started) * 1000.0,
                    "error": f"{type(exc).__name__}: {exc}"}

    def self_test(self, samples: int = 32) -> dict[str, Any]:
        import numpy as np
        if self._model is None:
            self._load("bare_flange")
        for index in range(1, 8):
            if self._model.getJointId(f"joint{index}") <= 0:
                return {"ok": False, "error": f"joint{index} is missing", "nq": self._model.nq, "nv": self._model.nv}
        if not all(math.isfinite(float(value)) for value in self._model.lowerPositionLimit[:7]) or not all(math.isfinite(float(value)) for value in self._model.upperPositionLimit[:7]):
            return {"ok": False, "error": "model joint limits are not finite", "nq": self._model.nq, "nv": self._model.nv}
        maximum = 0.0
        lower = self._model.lowerPositionLimit[:7]
        upper = self._model.upperPositionLimit[:7]
        for index in range(samples):
            q = np.asarray(lower) + (np.asarray(upper) - np.asarray(lower)) * ((index + 1) / (samples + 1))
            data = self._model.createData()
            gravity = self._pin.computeGeneralizedGravity(self._model, data, q)
            rnea = self._pin.rnea(self._model, data, q, np.zeros(7), np.zeros(7))
            maximum = max(maximum, float(np.max(np.abs(gravity - rnea))))
        return {"ok": maximum < 1e-6, "nq": self._model.nq, "nv": self._model.nv,
                "max_gravity_rnea_error_nm": maximum, "model_revision": self.model_revision}


@dataclass(frozen=True)
class GravityFeedforward:
    tau_g_raw_nm: tuple[float, ...]
    tau_g_mapped_nm: tuple[float, ...]
    tau_ff_target_nm: tuple[float, ...]
    tau_ff_sent_nm: tuple[float, ...]
    gravity_scale: float
    limit_reason: str | None
    slew_limited: bool
    model_age_s: float | None
    source: str = "pinocchio_gravity"
    model_revision: str | None = None
    sample_id: int | None = None
    state: str = "INVALID"

    def as_dict(self) -> dict[str, Any]:
        return {"tau_ff": list(self.tau_ff_sent_nm), "gravity_scale": self.gravity_scale,
                "gravity_state": self.state, "limit_reason": self.limit_reason,
                "model_age_s": self.model_age_s, "model_revision": self.model_revision,
                "sample_id": self.sample_id}


class GravityFeedforwardManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config or {})
        self._previous = [0.0] * 7

    def compute(self, q_actual_rad: Any, *, gravity_result: dict[str, Any] | None, dt_s: float,
                enabled: bool, arm_token: bool = False, scale_override: float | None = None,
                allow_hold_last_valid: bool = False) -> GravityFeedforward:
        zero = [0.0] * 7
        cfg = self.config
        configured_scale = float(cfg.get("gravity_scale", 0.0)) if scale_override is None else float(scale_override)
        scale = min(configured_scale, float(cfg.get("alpha_target_max", 1.0))) if enabled and arm_token else 0.0
        reasons: list[str] = []
        age = None
        raw = zero
        mapped = zero
        revision = None
        sample_id = None
        state = "INVALID"
        if scale > 0.0:
            if not isinstance(gravity_result, dict) or not gravity_result.get("ok"):
                reasons.append("gravity_result_unavailable")
            else:
                timestamp = gravity_result.get("computed_monotonic_ns")
                age = (time.monotonic_ns() - int(timestamp)) / 1e9 if timestamp is not None else None
                strict_age = float(cfg.get("max_result_age_s", 0.06))
                hold_age = float(cfg.get("hold_last_valid_max_age_s", 0.10))
                if age is None or age < 0 or age > (hold_age if allow_hold_last_valid else strict_age):
                    reasons.append("gravity_result_stale")
                else:
                    result_q = gravity_result.get("q_actual_rad")
                    if not isinstance(result_q, (list, tuple)) or len(result_q) != 7:
                        reasons.append("gravity_feedback_missing")
                    else:
                        try:
                            result_q = _vector(result_q, "gravity_feedback")
                            mismatch = max(abs(float(a) - float(b)) for a, b in zip(result_q, q_actual_rad))
                            strict_q_error = float(cfg.get("max_q_feedback_model_error_rad", 0.02))
                            hold_q_error = float(cfg.get("hold_last_valid_max_q_error_rad", 0.05))
                            if not math.isfinite(mismatch) or mismatch > (hold_q_error if allow_hold_last_valid else strict_q_error):
                                raise ValueError("gravity feedback model mismatch")
                            raw = _vector(gravity_result.get("tau_gravity_urdf_nm"), "tau_gravity_urdf_nm")
                            mapped = _vector(gravity_result.get("tau_gravity_sdk_nm"), "tau_gravity_sdk_nm")
                            revision = gravity_result.get("model_revision")
                            sample_id = gravity_result.get("sample_id")
                            state = "HOLD_LAST_VALID" if (
                                allow_hold_last_valid and (age or 0.0) > strict_age or mismatch > strict_q_error
                            ) else "FRESH"
                        except (TypeError, ValueError):
                            reasons.append("gravity_feedback_model_mismatch")
        target_unlimited = [value * scale for value in mapped] if not reasons else zero
        limits = _vector(cfg.get("torque_limit_nm", V120_TORQUE_LIMIT_NM), "torque_limit_nm")
        target = [max(-limit, min(limit, value)) for value, limit in zip(target_unlimited, limits)]
        if any(abs(a - b) > 1e-12 for a, b in zip(target_unlimited, target)):
            reasons.append("torque_limit")
        # A torque limit is an observable, usable result, not a stale/fallback
        # condition.  Keep the configured scale in diagnostics when clipping
        # occurs so acceptance can distinguish nominal scale from sent torque.
        fallback_reasons = {
            "gravity_result_unavailable", "gravity_result_stale",
            "gravity_feedback_missing", "gravity_feedback_model_mismatch",
        }
        usable_scale = scale if not any(reason in fallback_reasons for reason in reasons) else 0.0
        dt = max(0.0, min(float(dt_s), 0.2))
        slew = _vector(cfg.get("torque_slew_nm_s", [0.5] * 7), "torque_slew_nm_s")
        sent = [max(previous - rate * dt, min(previous + rate * dt, value)) for previous, rate, value in zip(self._previous, slew, target)]
        limited = any(abs(a - b) > 1e-12 for a, b in zip(sent, target))
        if limited:
            reasons.append("slew_limited")
        self._previous = sent
        # Result validity is independent from output shaping.  A fresh or
        # held gravity result may still be torque/slew limited; that must be
        # diagnosed as a usable state with a limit reason, not as INVALID.
        if any(reason in fallback_reasons for reason in reasons):
            state = "INVALID"
        return GravityFeedforward(tuple(raw), tuple(mapped), tuple(target), tuple(sent), usable_scale,
                                  ";".join(reasons) if reasons else None, limited, age, "pinocchio_gravity", revision, sample_id,
                                  state=state)

    def reset(self) -> None:
        self._previous = [0.0] * 7


class GravityReadonlyProcess:
    """Synchronous JSONL client for the dedicated Pinocchio interpreter."""

    def __init__(self, python: str | Path, script: str | Path, urdf: str | Path,
                 conventions: Any = None, tool_profiles: dict[str, Any] | None = None) -> None:
        self.python = Path(python)
        self.script = Path(script)
        self.urdf = Path(urdf)
        self.conventions = copy.deepcopy(conventions if isinstance(conventions, list) else [])
        self.tool_profiles = copy.deepcopy(tool_profiles or {})
        self._lock = threading.RLock()
        self._process = None
        self._next_id = 1

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [str(self.python), "-u", str(self.script), "--urdf", str(self.urdf),
             "--joint-conventions-json", json.dumps(self.conventions, separators=(",", ":")),
             "--tool-profiles-json", json.dumps(self.tool_profiles, separators=(",", ":"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if self._process.stdout is None:
            raise RuntimeError("gravity worker stdout is unavailable")
        self._responses = queue.Queue()
        responses = self._responses
        process = self._process
        def read_lines():
            for line in process.stdout:
                responses.put(line)
            responses.put("")
        threading.Thread(target=read_lines, name="osc-gravity-jsonl", daemon=True).start()
        ready = self._readline()
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise RuntimeError("gravity worker did not become ready")

    def compute(self, q_actual_rad: list[float], *, tool_profile: str, sample_id: int) -> dict[str, Any]:
        with self._lock:
            self._start()
            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("gravity worker is unavailable")
            request = {"kind": "gravity", "q_actual_rad": q_actual_rad,
                       "tool_profile": tool_profile, "sample_id": sample_id}
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._readline()
            if not line:
                self.close()
                raise RuntimeError("gravity worker exited before response")
            result = json.loads(line)
            if not isinstance(result, dict):
                raise RuntimeError("gravity worker returned an invalid response")
            return result

    def _readline(self) -> str:
        try:
            return self._responses.get(timeout=2.0)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("gravity process response timed out") from exc

    def is_alive(self) -> bool:
        with self._lock:
            return bool(self._process is not None and self._process.poll() is None)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except Exception:
                    pass
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class GravityProcessWorkerClient:
    """Latest-only gravity worker backed by the Pinocchio interpreter.

    The control-service interpreter deliberately does not need Pinocchio.  A
    single background thread owns the JSONL process, so the 50 Hz servo loop
    only publishes a request and reads a cached result.
    """

    def __init__(self, process: "GravityReadonlyProcess") -> None:
        self.process = process
        self._lock = threading.RLock()
        self._request: tuple[list[float], int, str] | None = None
        self._latest: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None

    def start(self) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return {"ok": True, "running": True}
        self._stop.clear()
        with self._lock:
            self._last_error = None
        self._thread = threading.Thread(target=self._run, name="nero-gravity-process-worker", daemon=True)
        self._thread.start()
        return {"ok": True, "running": True}

    def request_gravity(self, q_actual_rad: Any, sample_id: int, tool_profile: str) -> None:
        values = _vector(q_actual_rad, "q_actual_rad")
        with self._lock:
            self._request = (values, int(sample_id), str(tool_profile))
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            with self._lock:
                request = self._request
                self._request = None
            if request is None:
                continue
            try:
                result = self.process.compute(request[0], sample_id=request[1], tool_profile=request[2])
            except Exception as exc:
                result = {"ok": False, "error": f"gravity process failed: {type(exc).__name__}: {exc}",
                          "sample_id": request[1]}
            else:
                # Age is measured in the control-service process.  Re-stamp
                # on receipt so clocks/IPC latency cannot make a valid result
                # appear stale merely because it was produced by a child.
                result["computed_monotonic_ns"] = time.monotonic_ns()
            with self._lock:
                self._latest = result
                self._last_error = None if result.get("ok") else str(result.get("error", "gravity computation failed"))

    def latest_result(self) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._latest)

    def reset_session(self) -> None:
        """Discard requests/results belonging to the previous control session."""
        with self._lock:
            self._request = None
            self._latest = None
            self._last_error = None

    def health(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                "thread_alive": bool(thread and thread.is_alive()),
                "process_alive": self.process.is_alive(),
                "last_error": self._last_error,
            }

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.process.close()
