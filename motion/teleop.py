from __future__ import annotations

import ctypes
import json
import math
import queue
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any



class KinematicsUnavailable(RuntimeError):
    pass


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _unit_quaternion(value: Any, name: str = "orientation_xyzw") -> list[float]:
    quaternion = _finite_vector(value, 4, name)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not 0.9 <= norm <= 1.1:
        raise ValueError(f"{name} must have a norm between 0.9 and 1.1")
    return [item / norm for item in quaternion]


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    x1, y1, z1, w1 = left; x2, y2, z2, w2 = right
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _quat_conjugate(value: list[float]) -> list[float]:
    return [-value[0], -value[1], -value[2], value[3]]


def _quat_angle(value: list[float]) -> float:
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(value[3]))))


def _quat_scale(value: list[float], scale: float) -> list[float]:
    """Slerp identity->value, retaining a unit quaternion."""
    angle = _quat_angle(value)
    if angle < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    sign = 1.0 if value[3] >= 0.0 else -1.0
    axis_scale = math.sin(angle * max(0.0, min(1.0, scale)) / 2.0) / math.sin(angle / 2.0)
    return _unit_quaternion([value[0] * sign * axis_scale, value[1] * sign * axis_scale, value[2] * sign * axis_scale, math.cos(angle * max(0.0, min(1.0, scale)) / 2.0)])


def _quat_to_matrix(value: list[float]) -> list[list[float]]:
    x, y, z, w = _unit_quaternion(value)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matrix_to_quat(matrix: list[list[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2; return _unit_quaternion([(matrix[2][1] - matrix[1][2]) / s, (matrix[0][2] - matrix[2][0]) / s, (matrix[1][0] - matrix[0][1]) / s, 0.25 * s])
    index = max(range(3), key=lambda item: matrix[item][item])
    if index == 0:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2; return _unit_quaternion([0.25 * s, (matrix[0][1] + matrix[1][0]) / s, (matrix[0][2] + matrix[2][0]) / s, (matrix[2][1] - matrix[1][2]) / s])
    if index == 1:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2; return _unit_quaternion([(matrix[0][1] + matrix[1][0]) / s, 0.25 * s, (matrix[1][2] + matrix[2][1]) / s, (matrix[0][2] - matrix[2][0]) / s])
    s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2; return _unit_quaternion([(matrix[0][2] + matrix[2][0]) / s, (matrix[1][2] + matrix[2][1]) / s, 0.25 * s, (matrix[1][0] - matrix[0][1]) / s])


def _matrix3(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    result = [_finite_vector(row, 3, name) for row in value]
    # Axis maps must be orthonormal; rotations require a proper rotation.
    for row in result:
        if abs(sum(item * item for item in row) - 1.0) > 1e-6:
            raise ValueError(f"{name} must be orthonormal")
    for first in range(3):
        for second in range(first + 1, 3):
            if abs(sum(result[first][i] * result[second][i] for i in range(3))) > 1e-6:
                raise ValueError(f"{name} must be orthonormal")
    return result


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(3)) for row in matrix]


def _mat_mul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)] for row in range(3)]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def normalize_session_selection(
    mode: str | None = None,
    execution_mode: str | None = None,
    input_source: str | None = None,
) -> tuple[str, str, str]:
    """Return execution target, input source, and a legacy combo name."""
    legacy = str(mode or "").strip().lower()
    if execution_mode is None and input_source is None:
        legacy_map = {
            "shadow": ("shadow", "joystick"),
            "hardware": ("hardware", "joystick"),
            "joystick_hardware": ("hardware", "joystick"),
            "pico_hardware": ("hardware", "pico"),
            "pico_shadow": ("shadow", "pico"),
        }
        if legacy not in legacy_map:
            raise ValueError("mode must be shadow, hardware, joystick_hardware, pico_hardware, or pico_shadow")
        execution_mode, input_source = legacy_map[legacy]
    else:
        execution_mode = str(execution_mode or "").strip().lower()
        input_source = str(input_source or "").strip().lower()
        if execution_mode not in {"shadow", "hardware"}:
            raise ValueError("execution_mode must be shadow or hardware")
        if input_source not in {"joystick", "pico"}:
            raise ValueError("input_source must be joystick or pico")
        derived = {
            ("shadow", "joystick"): "shadow",
            ("hardware", "joystick"): "joystick_hardware",
            ("hardware", "pico"): "pico_hardware",
            ("shadow", "pico"): "pico_shadow",
        }[(execution_mode, input_source)]
        if legacy and legacy != derived:
            raise ValueError("mode conflicts with execution_mode/input_source")
        legacy = derived
    return str(execution_mode), str(input_source), legacy


class KinematicsClient:
    """Latest-sample JSONL bridge for Pinocchio/Pink only."""

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        solver = config.get("solver", {})
        self.root = project_root
        python = Path(str(solver.get("python", "")))
        self.python = python if python.is_absolute() else project_root / python
        self.script = project_root / str(solver.get("script", "motion/teleop_kinematics_server.py"))
        self.urdf = project_root / str(solver.get("urdf", "vendor/nero_description/nero_description.urdf"))
        self.offset = config.get("tcp", {}).get("offset_from_link7_m", [0.175, 0.0, -0.0235])
        self.period_s = float(solver.get("dt_s", 0.02))
        self.startup_timeout_s = float(solver.get("startup_timeout_s", 12.0))
        self.response_timeout_s = max(0.5, float(solver.get("timeout_s", 1.0)))
        self.process: subprocess.Popen[str] | None = None
        # submit() may need to restart the solver after the watchdog quarantines
        # a stuck child.  start() clears stale queues through this same lock, so
        # it must be re-entrant during that recovery path.
        self.lock = threading.RLock()
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.fk_responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.ready: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.minimum_epoch = 0
        self._solver_request_id = 0
        self._last_response_request_id = 0
        self._last_response_seen_request_id = 0
        self._pending_solver_response: dict[str, Any] | None = None
        self._motion_write_busy = False
        self._debug = {
            "submitted": 0,
            "accepted": 0,
            "discarded": 0,
            "last_submit": None,
            "last_response": None,
            "last_discard": None,
            "reader_alive": False,
            "reader_error": None,
        }

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.discard_before_epoch(0)
        if not self.python.is_file() or not self.script.is_file() or not self.urdf.is_file():
            raise KinematicsUnavailable("Pinocchio/Pink solver executable, script, or URDF is missing")
        self.process = subprocess.Popen(
            [str(self.python), "-u", str(self.script), "--urdf", str(self.urdf), "--tcp-offset", ",".join(map(str, self.offset)), "--period-s", str(self.period_s)],
            cwd=str(self.root), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read_stdout, name="teleop-pink-reader", daemon=True).start()
        try:
            ready = self.ready.get(timeout=max(1.0, self.startup_timeout_s))
        except queue.Empty as exc:
            self.close()
            raise KinematicsUnavailable("Pinocchio/Pink solver did not become ready") from exc
        if not ready.get("ready"):
            self.close()
            raise KinematicsUnavailable(str(ready.get("error", ready)))

    def close(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        # A new teleop session must never inherit responses from the previous
        # anchor/epoch.  This is especially important after the hard-reset
        # recovery path, where the controller worker may survive while its
        # Pink child still has an older request in flight.
        with self.lock:
            self._solver_request_id = 0
            self._last_response_request_id = 0
            self._last_response_seen_request_id = 0
            self._pending_solver_response = None
            self._motion_write_busy = False
            self._debug = {
                "submitted": 0,
                "accepted": 0,
                "discarded": 0,
                "last_submit": None,
                "last_response": None,
                "last_discard": None,
                "reader_alive": False,
                "reader_error": None,
            }
            for channel in (self.responses, self.ready, self.fk_responses):
                while True:
                    try:
                        channel.get_nowait()
                    except queue.Empty:
                        break

    def discard_before_epoch(self, epoch: int) -> None:
        with self.lock:
            self.minimum_epoch = max(self.minimum_epoch, int(epoch))
            for channel in (self.responses, self.ready, self.fk_responses):
                while True:
                    try:
                        channel.get_nowait()
                    except queue.Empty:
                        break

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        with self.lock:
            self._debug["reader_alive"] = True
        try:
            for line in process.stdout:
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    # Both {ready: true} and {ready: false, error: ...} are
                    # startup terminal messages. Queue failures immediately.
                    if "ready" in payload:
                        try:
                            self.ready.put_nowait(payload)
                        except queue.Full:
                            pass
                        continue
                    if payload.get("kind") == "fk":
                        try:
                            self.fk_responses.put_nowait(payload)
                        except queue.Full:
                            pass
                        continue
                    # Treat absent/None/non-numeric epochs as stale instead
                    # of allowing one malformed solver frame to kill the
                    # reader thread and fill the child stdout pipe.
                    try:
                        # Shadow sessions intentionally use motion_epoch=0.
                        # Do not treat that valid epoch as a false-y missing
                        # value, otherwise every Pink response is discarded
                        # before poll() can consume it.
                        raw_epoch = payload.get("motion_epoch", -1)
                        response_epoch = -1 if raw_epoch is None else int(raw_epoch)
                    except (TypeError, ValueError):
                        with self.lock:
                            self._debug["reader_error"] = "invalid motion_epoch in solver response"
                        continue
                    if response_epoch < self.minimum_epoch:
                        continue
                    try:
                        self.responses.put_nowait(payload)
                    except queue.Full:
                        try:
                            self.responses.get_nowait()
                            self.responses.put_nowait(payload)
                        except queue.Empty:
                            pass
                    try:
                        response_request_id = int(payload.get("solver_request_id", 0) or 0)
                        if response_request_id:
                            with self.lock:
                                self._last_response_seen_request_id = max(self._last_response_seen_request_id, response_request_id)
                    except (TypeError, ValueError):
                        pass
                except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
                    with self.lock:
                        self._debug["reader_error"] = f"{type(exc).__name__}: {exc}"
                    continue
        except (OSError, ValueError) as exc:
            with self.lock:
                self._debug["reader_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self._debug["reader_alive"] = False

    def submit(self, payload: dict[str, Any]) -> None:
        is_motion = payload.get("kind") != "fk"
        with self.lock:
            self.start()
            if not self.process or not self.process.stdin:
                raise KinematicsUnavailable("Pinocchio/Pink solver is unavailable")
            if is_motion:
                self._solver_request_id += 1
                payload = dict(payload)
                payload["solver_request_id"] = self._solver_request_id
            process = self.process
            stdin = process.stdin
            request_id = payload.get("solver_request_id")
            self._debug["submitted"] = int(self._debug["submitted"]) + 1
            self._debug["last_submit"] = {
                "solver_request_id": request_id,
                "motion_epoch": payload.get("motion_epoch"),
                "anchor_id": payload.get("anchor_id"),
                "reference_revision": payload.get("reference_revision"),
                "sequence": payload.get("sequence"),
            }
        # Do not hold the client lock across a Windows pipe flush.  Motion
        # writes are also isolated in one daemon writer: a child that stops
        # consuming stdin cannot block the 50 Hz robot-control thread.
        started = time.monotonic()
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        if is_motion and request_id is not None:
            with self.lock:
                if self._motion_write_busy:
                    return
                self._motion_write_busy = True
            threading.Thread(
                target=self._write_motion_request,
                args=(process, stdin, line, int(request_id), started),
                name="teleop-pink-writer",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._watch_motion_request,
                args=(process, int(request_id), started),
                name="teleop-pink-watchdog",
                daemon=True,
            ).start()
            return
        stdin.write(line)
        stdin.flush()

    def _write_motion_request(self, process: subprocess.Popen[str], stdin: Any, line: str, request_id: int, started: float) -> None:
        try:
            stdin.write(line)
            stdin.flush()
        except (OSError, ValueError):
            with self.lock:
                self._debug["reader_error"] = f"solver request {request_id} write failed"
        finally:
            with self.lock:
                self._motion_write_busy = False

    def _watch_motion_request(self, process: subprocess.Popen[str], request_id: int, started: float) -> None:
        # The first Pink/quadprog request after process startup can include
        # one-time allocator and solver initialization.  Quarantining it at
        # the 50 Hz period turns a valid cold start into KinematicsUnavailable
        # and freezes the shadow/real projection before the first pose update.
        timeout_s = self.response_timeout_s
        while time.monotonic() - started < timeout_s:
            if process.poll() is not None:
                return
            with self.lock:
                if self.process is not process or self._last_response_seen_request_id >= request_id:
                    return
            time.sleep(0.02)
        with self.lock:
            if self.process is not process or self._last_response_seen_request_id >= request_id:
                return
            self._debug["reader_error"] = f"solver request {request_id} timed out after {timeout_s:.2f}s"
            self.process = None
            self._motion_write_busy = False
        try:
            process.terminate()
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def poll(self, epoch: int, anchor_id: int | None = None, reference_revision: int | None = None) -> dict[str, Any] | None:
        # A continuously moving joystick necessarily advances the reference
        # revision faster than Pink can finish every request.  Within one
        # clutch anchor, the newest completed result may therefore be from an
        # earlier revision; the anchor is the hard boundary that prevents a
        # result from a previous clutch segment from being reused.
        latest = self._pending_solver_response
        self._pending_solver_response = None
        if latest is not None:
            if (
                int(latest.get("motion_epoch", -1)) == int(epoch)
                and (anchor_id is None or int(latest.get("anchor_id", -1)) == int(anchor_id))
                and (reference_revision is None or int(latest.get("reference_revision", -1)) <= int(reference_revision))
            ):
                self._last_response_request_id = int(latest.get("solver_request_id", 0) or 0)
                return latest
            latest = None
        while True:
            try:
                candidate = self.responses.get_nowait()
            except queue.Empty:
                break
            request_id = int(candidate.get("solver_request_id", 0) or 0)
            self._debug["last_response"] = {
                "solver_request_id": request_id,
                "ok": candidate.get("ok"),
                "error": candidate.get("error"),
                "motion_epoch": candidate.get("motion_epoch"),
                "anchor_id": candidate.get("anchor_id"),
                "reference_revision": candidate.get("reference_revision"),
            }
            if (
                int(candidate.get("motion_epoch", -1)) == int(epoch)
                and request_id <= self._solver_request_id
                and request_id > self._last_response_request_id
                and (anchor_id is None or int(candidate.get("anchor_id", -1)) == int(anchor_id))
                and (reference_revision is None or int(candidate.get("reference_revision", -1)) <= int(reference_revision))
            ):
                latest = candidate
                self._last_response_request_id = request_id
                self._debug["accepted"] = int(self._debug["accepted"]) + 1
            else:
                self._debug["discarded"] = int(self._debug["discarded"]) + 1
                self._debug["last_discard"] = dict(self._debug["last_response"])
        return latest

    def debug_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                **self._debug,
                "request_id": self._solver_request_id,
                "last_response_request_id": self._last_response_request_id,
                "minimum_epoch": self.minimum_epoch,
                "running": bool(self.process and self.process.poll() is None),
            }

    def fk(self, joints: list[float]) -> dict[str, Any]:
        self.submit({"kind": "fk", "joint_angles_rad": _finite_vector(joints, 7, "joint_angles_rad")})
        try:
            payload = self.fk_responses.get(timeout=1.0)
        except queue.Empty as exc:
            raise KinematicsUnavailable("Pinocchio/Pink FK request timed out") from exc
        if not payload.get("ok"):
            raise KinematicsUnavailable(str(payload.get("error", "Pinocchio/Pink FK request failed")))
        return dict(payload["tcp"])


@dataclass(frozen=True)
class JointConvention:
    sdk_index: int
    urdf_joint: str
    sign: float
    zero_offset_rad: float
    unit_scale: float

    def position_to_urdf(self, value: float) -> float:
        return self.sign * self.unit_scale * float(value) + self.zero_offset_rad

    def velocity_to_urdf(self, value: float) -> float:
        return abs(self.unit_scale) * self.sign * float(value)

    def acceleration_to_urdf(self, value: float) -> float:
        return abs(self.unit_scale) * float(value)


class JointLimitAuthority:
    """Converts controller limits into canonical URDF coordinates."""

    def __init__(self, urdf: Path, config: dict[str, Any]) -> None:
        self.urdf = urdf
        self.config = config
        self.hard_lower, self.hard_upper = self._read_urdf_limits()
        conventions = config.get("joint_conventions") or []
        if not conventions:
            conventions = [{"sdk_index": idx, "urdf_joint": f"joint{idx}", "sign": 1.0, "zero_offset_rad": 0.0, "unit_scale": 1.0} for idx in range(1, 8)]
        self.conventions = [JointConvention(**item) for item in conventions]
        if [item.sdk_index for item in self.conventions] != list(range(1, 8)):
            raise ValueError("joint_conventions must declare SDK indices 1..7 in order")
        if [item.urdf_joint for item in self.conventions] != [f"joint{idx}" for idx in range(1, 8)]:
            raise ValueError("joint_conventions must map to NERO joint1..joint7 in order")
        if any(not math.isfinite(item.sign) or abs(abs(item.sign) - 1.0) > 1e-9 or not math.isfinite(item.unit_scale) or item.unit_scale <= 0 for item in self.conventions):
            raise ValueError("joint_conventions contain an invalid sign or unit_scale")
        self.effective: dict[str, Any] | None = None

    def _read_urdf_limits(self) -> tuple[list[float], list[float]]:
        root = ET.parse(self.urdf).getroot()
        found: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name and limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
                found[name] = (float(limit.get("lower")), float(limit.get("upper")))
        names = [f"joint{idx}" for idx in range(1, 8)]
        if any(name not in found for name in names):
            raise ValueError("URDF has incomplete NERO joint limits")
        return [found[name][0] for name in names], [found[name][1] for name in names]

    @staticmethod
    def _number(row: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        raise ValueError(f"controller limit field missing: {keys}")

    def initialize(self, controller: dict[str, Any]) -> dict[str, Any]:
        rows = controller.get("joints") if isinstance(controller, dict) else None
        if not isinstance(rows, list) or len(rows) != 7:
            raise ValueError("controller did not return seven joint limit rows")
        raw_rows: list[dict[str, Any]] = []
        effective_lower: list[float] = []
        effective_upper: list[float] = []
        speed: list[float] = []
        acceleration: list[float] = []
        for index, (row, convention) in enumerate(zip(rows, self.conventions), start=1):
            if int(row.get("joint_index", -1)) != convention.sdk_index:
                raise ValueError(f"controller joint order mismatch at J{index}")
            angle = row.get("angle_velocity")
            acc = row.get("acceleration")
            if not isinstance(angle, dict) or not isinstance(acc, dict):
                raise ValueError(f"controller limit row J{index} is malformed")
            raw_lower = self._number(angle, "min_angle_limit")
            raw_upper = self._number(angle, "max_angle_limit")
            converted = sorted([convention.position_to_urdf(raw_lower), convention.position_to_urdf(raw_upper)])
            controller_speed = abs(convention.velocity_to_urdf(self._number(angle, "max_joint_spd")))
            controller_acc = abs(convention.acceleration_to_urdf(self._number(acc, "max_joint_acc")))
            if converted[0] >= converted[1] or controller_speed <= 0 or controller_acc <= 0:
                raise ValueError(f"controller limits are invalid for J{index}")
            lower = max(self.hard_lower[index - 1], converted[0])
            upper = min(self.hard_upper[index - 1], converted[1])
            if lower >= upper:
                raise ValueError(f"URDF/controller limit intersection is empty for J{index}")
            effective_lower.append(lower)
            effective_upper.append(upper)
            speed.append(controller_speed)
            acceleration.append(controller_acc)
            raw_rows.append({"sdk": angle, "acceleration": acc, "converted_lower_rad": converted[0], "converted_upper_rad": converted[1]})
        self.effective = {
            "status": "ready", "urdf_lower_rad": self.hard_lower, "urdf_upper_rad": self.hard_upper,
            "effective_lower_rad": effective_lower, "effective_upper_rad": effective_upper,
            "controller_speed_rad_s": speed, "controller_acceleration_rad_s2": acceleration,
            "conventions": [item.__dict__ for item in self.conventions], "controller_raw": raw_rows,
        }
        return dict(self.effective)


class SafetySupervisor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.limit_data: dict[str, Any] | None = None
        self.delay_samples: deque[float] = deque(maxlen=256)
        # The controller response is not measured at every servo cycle. Keep a
        # conservative configured floor in the braking budget until it is
        # characterized from hardware logs.
        self.controller_response_floor_s = max(0.0, float(config.get("controller_response_delay_s", 0.04)))
        self.feedback_limit_tolerance_rad = max(0.0, float(config.get("feedback_limit_tolerance_rad", 0.0)))

    def configure(self, limits: dict[str, Any], requested_speed: list[float], requested_acceleration: list[float]) -> None:
        self.limit_data = dict(limits)
        self.limit_data["speed_rad_s"] = [min(a, b) for a, b in zip(limits["controller_speed_rad_s"], requested_speed)]
        self.limit_data["acceleration_rad_s2"] = [min(a, b) for a, b in zip(limits["controller_acceleration_rad_s2"], requested_acceleration)]
        fixed_model = _finite_vector(self.config.get("fixed_model_margin_rad", [0.03, 0.03, 0.03, 0.025, 0.03, 0.02, 0.02]), 7, "fixed_model_margin_rad")
        stop_reserve = _finite_vector(self.config.get("controller_stop_reserve_rad", [0.02, 0.02, 0.02, 0.02, 0.02, 0.015, 0.015]), 7, "controller_stop_reserve_rad")
        self.limit_data["fixed_margin_rad"] = [a + b for a, b in zip(fixed_model, stop_reserve)]
        self.limit_data["soft_lower_rad"] = [a + b for a, b in zip(limits["effective_lower_rad"], self.limit_data["fixed_margin_rad"])]
        self.limit_data["soft_upper_rad"] = [a - b for a, b in zip(limits["effective_upper_rad"], self.limit_data["fixed_margin_rad"])]
        if any(lower >= upper for lower, upper in zip(self.limit_data["soft_lower_rad"], self.limit_data["soft_upper_rad"])):
            raise ValueError("fixed per-joint margins consume an effective joint range")

    def observe_delay(self, feedback_age: float, cycle_s: float, solver_age: float | None, batch_skew_s: float, response_s: float) -> float:
        response = max(self.controller_response_floor_s, max(0.0, response_s))
        total = max(0.0, feedback_age) + max(0.0, cycle_s) + max(0.0, solver_age or 0.0) + max(0.0, batch_skew_s) + response
        self.delay_samples.append(total)
        ordered = sorted(self.delay_samples)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.99) - 1))]

    def _require(self) -> dict[str, Any]:
        if self.limit_data is None:
            raise RuntimeError("joint limit authority is not initialized")
        return self.limit_data

    def limit_velocity(self, q: list[float], qd: list[float], request: list[float], delay_s: float) -> tuple[list[float], dict[str, Any]]:
        data = self._require()
        safe = list(request)
        rows = []
        for index in range(7):
            lower, upper = data["effective_lower_rad"][index], data["effective_upper_rad"][index]
            acceleration = max(1e-6, data["acceleration_rad_s2"][index])
            direction = 1.0 if request[index] > 0 else -1.0 if request[index] < 0 else 0.0
            distance = (upper - q[index]) if direction > 0 else (q[index] - lower) if direction < 0 else math.inf
            projected_motion = abs(qd[index]) * delay_s
            usable = max(0.0, distance - data["fixed_margin_rad"][index] - projected_motion)
            outward_max = math.sqrt(max(0.0, 2.0 * acceleration * usable))
            if direction != 0.0 and abs(request[index]) > outward_max:
                safe[index] = math.copysign(outward_max, request[index])
            rows.append({"joint": index + 1, "distance_to_limit_rad": distance, "dynamic_braking_distance_rad": projected_motion + (request[index] ** 2) / (2.0 * acceleration), "allowed_outward_rad_s": outward_max, "requested_rad_s": request[index], "safe_rad_s": safe[index]})
        return safe, {"joints": rows, "delay_budget_s": delay_s}

    def final_gate(self, q: list[float], qd: list[float], velocity: list[float], delay_s: float, feedback_hard_stale: bool) -> tuple[list[float], bool, str | None]:
        if feedback_hard_stale:
            return [0.0] * 7, False, "feedback hard stale"
        safe, report = self.limit_velocity(q, qd, velocity, delay_s)
        data = self._require()
        # A complete outward clamp to zero is still a safety limitation and
        # must be observable as such, not reported as an unconstrained pass.
        limited = any(abs(a - b) > 1e-6 for a, b in zip(velocity, safe))
        for row in report["joints"]:
            if row["distance_to_limit_rad"] < 0.0:
                excursion = -float(row["distance_to_limit_rad"])
                requested = float(row["requested_rad_s"])
                dispatched = float(row["safe_rad_s"])
                # A small calibration-seam excursion beyond the
                # URDF/controller intersection is feedback tolerance only:
                # stationary or inward motion may continue, but it never
                # authorizes outward motion or a larger excursion.
                outward = (requested > 1e-9 and dispatched > 1e-9) or (requested < -1e-9 and dispatched < -1e-9)
                if excursion > self.feedback_limit_tolerance_rad or outward:
                    return [0.0] * 7, False, f"J{row['joint']} exceeds effective position limit"
        # A dynamic braking limit is a normal constrained-tracking outcome:
        # execute the clipped velocity and report it, rather than faulting the
        # whole OSC session.  Feedback loss and an already-out-of-range joint
        # remain hard rejections above.
        return safe, True, None if not limited else "velocity clipped by final safety gate"


class OperationalSpaceServo:
    """Fixed-rate operational-space servo: Pink, Ruckig and safety sync.

    This is the sole Pink/Ruckig/CPV loop owned by OperationalSpaceController.
    Legacy teleoperation is only an input adapter to this servo.
    """

    def __init__(self, broker: Any, project_root: Path, config: dict[str, Any]) -> None:
        try:
            import ruckig
        except ImportError as exc:
            raise RuntimeError("teleop requires ruckig in the control-service environment") from exc
        self.ruckig = ruckig
        self.broker, self.root, self.config = broker, project_root, config
        self.limits, self.runtime = config.get("limits", {}), config.get("runtime", {})
        pose_input = config.get("pose_input", {})
        self.position_axis_map = _matrix3(pose_input.get("position_axis_map", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "position_axis_map")
        self.orientation_axis_map = _matrix3(pose_input.get("orientation_axis_map", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "orientation_axis_map")
        self.pose_input = pose_input
        self.solver = KinematicsClient(project_root, config)
        self.authority = JointLimitAuthority(self.solver.urdf, config)
        self.supervisor = SafetySupervisor(config.get("safety_supervisor", {}))
        self.lock = threading.RLock()
        # Serializes session replacement without holding the state lock while
        # the old servo thread performs its normal braking shutdown.
        self._session_transition_lock = threading.Lock()
        self.session: dict[str, Any] | None = None
        self.state_sequence = 0
        self.last_error: str | None = None
        self.input_enabled = False
        self._heartbeat_monotonic_ns = 0
        self.intent: dict[str, Any] | None = None
        self.anchor_id = 0
        self.reference_revision = 0
        self.clutch_active = False
        self.absolute_target_active = False
        self.absolute_target_mode: str | None = None
        self.input_source: str | None = None
        self.tcp_anchor: dict[str, list[float]] | None = None
        self.reference_pose: dict[str, list[float]] | None = None
        self.relative_pose: dict[str, list[float]] = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        self.motion_epoch = 0
        self.filtered_velocity = [0.0] * 6
        self.last_sent_velocity = [0.0] * 7
        self.trajectory: dict[str, list[float]] | None = None
        self.ruckig_input = None
        self.ruckig_output = None
        self.ruckig_otg = None
        self.ruckig_period_s: float | None = None
        self.posture_reference: list[float] | None = None
        self.shadow_joints: list[float] | None = None
        self.last_solver_result: dict[str, Any] | None = None
        self._solver_reuse_count = 0
        self.last_result: dict[str, Any] = {"ok": False, "reason": "not started"}
        self.trajectory_state, self.trajectory_brake_reason = "HOLD_READY", None
        self.feedback_sync_pending = True
        self.needs_resync = True
        self._feedback: dict[str, Any] | None = None
        self._feedback_lock = threading.RLock()
        self._feedback_stop = threading.Event()
        self._feedback_thread: threading.Thread | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cpv_send_count, self.output_count, self.loop_count = 0, 0, 0
        self.last_output: dict[str, Any] = {
            "status": "held",
            "final_joint_target_rad": None,
            "final_joint_velocity_rad_s": [0.0] * 7,
            "sequence": 0,
            "epoch": 0,
        }
        self._send_times: deque[float] = deque(maxlen=256)
        self._batch_history: deque[dict[str, Any]] = deque(maxlen=256)
        self._timing: dict[str, Any] = {}
        self._enable_high_resolution_timer()

    @staticmethod
    def _enable_high_resolution_timer() -> None:
        """Keep the Windows scheduler from rounding 20 ms waits to 31 ms."""
        if not hasattr(ctypes, "windll"):
            return
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except (AttributeError, OSError):
            pass

    def _bump_state(self) -> int:
        self.state_sequence += 1
        return self.state_sequence

    def _session_active(self) -> bool:
        return bool(self.session and self.session.get("state") == "ACTIVE")

    def _session_view(self) -> dict[str, Any]:
        session = self.session or {}
        return {
            "state": session.get("state", "IDLE"),
            "id": session.get("session_id"),
            "client_id": session.get("client_id"),
            "mode": session.get("mode"),
            "execution_mode": session.get("execution_mode", "shadow" if session.get("mode") == "shadow" else "hardware" if session.get("mode") else None),
            "input_source": session.get("input_source", "pico" if session.get("mode") == "pico_hardware" else "joystick" if session.get("mode") else None),
            "sequence": session.get("sequence", 0),
            "last_input_age_s": session.get("last_input_age_s"),
        }

    @staticmethod
    def _pose_from_tcp(tcp: dict[str, Any]) -> dict[str, list[float]]:
        position = _finite_vector(tcp.get("position_m"), 3, "tcp.position_m")
        rotation = _matrix3(tcp.get("rotation"), "tcp.rotation")
        return {"position_m": position, "orientation_xyzw": _matrix_to_quat(rotation)}

    def _current_tcp_pose(self, joints: list[float]) -> dict[str, list[float]]:
        return self._pose_from_tcp(self.solver.fk(joints))

    def _pose_in_workspace(self, pose: dict[str, list[float]]) -> bool:
        position = pose["position_m"]
        lower = _finite_vector(self.limits.get("workspace_min_m", [-0.45, -0.45, -0.01]), 3, "workspace_min_m")
        upper = _finite_vector(self.limits.get("workspace_max_m", [0.45, 0.60, 0.70]), 3, "workspace_max_m")
        anchor_position = (self.tcp_anchor or {}).get("position_m")
        for index, (item, low, high) in enumerate(zip(position, lower, upper)):
            if low <= item <= high:
                continue
            # The measured flange and the Pink TCP can legitimately differ by
            # the configured tool offset.  If the current model anchor is
            # already just outside the nominal box, allow that exact hold
            # coordinate so clutch_begin/first pose cannot deadman-brake the
            # arm.  Any subsequent excursion in that axis remains rejected.
            if not anchor_position or abs(float(item) - float(anchor_position[index])) > 1e-6:
                return False
        min_flange_z = float(self.limits.get("min_flange_z_m", lower[2]))
        if position[2] < min_flange_z and (not anchor_position or abs(float(position[2]) - float(anchor_position[2])) > 1e-6):
            return False
        return True

    def _reference_from_relative(self, relative: dict[str, list[float]], scale: float) -> dict[str, list[float]]:
        if self.tcp_anchor is None:
            raise RuntimeError("teleop clutch anchor is unavailable")
        position = _finite_vector(relative.get("position_m"), 3, "relative_pose.position_m")
        orientation = _unit_quaternion(relative.get("orientation_xyzw"), "relative_pose.orientation_xyzw")
        max_translation = float(self.pose_input.get("max_relative_translation_m", 0.25))
        max_rotation = float(self.pose_input.get("max_relative_rotation_rad", 1.0472))
        if math.sqrt(sum(item * item for item in position)) > max_translation:
            raise ValueError("relative pose translation exceeds configured limit")
        if _quat_angle(orientation) > max_rotation:
            raise ValueError("relative pose rotation exceeds configured limit")
        translation_gain = float(self.pose_input.get("translation_gain", 1.0)) * scale
        rotation_gain = float(self.pose_input.get("rotation_gain", 1.0)) * scale
        mapped_position = _mat_vec(self.position_axis_map, position)
        mapped_position = [item * translation_gain for item in mapped_position]
        mapped_rotation = _mat_mul(_mat_mul(self.orientation_axis_map, _quat_to_matrix(orientation)), _transpose(self.orientation_axis_map))
        mapped_orientation = _quat_scale(_matrix_to_quat(mapped_rotation), rotation_gain)
        anchor_position = self.tcp_anchor["position_m"]
        return {
            "position_m": [anchor_position[index] + mapped_position[index] for index in range(3)],
            "orientation_xyzw": _unit_quaternion(_quat_multiply(self.tcp_anchor["orientation_xyzw"], mapped_orientation)),
        }

    def _set_anchor_locked(self, joints: list[float], source: str) -> int:
        self.tcp_anchor = self._current_tcp_pose(joints)
        self.reference_pose = dict(self.tcp_anchor)
        self.relative_pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        self.anchor_id += 1
        self.reference_revision += 1
        self.clutch_active = True
        self.input_source = source
        return self.anchor_id

    def _invalidate_motion(self, reason: str) -> None:
        # This is a P1 braking request, not an ownership transfer. The active
        # epoch remains valid long enough for Ruckig to transmit the braking
        # curve. Ownership transfers use freeze_for_authority_change().
        session = dict(self.session or {})
        if session.get("execution_mode", "shadow" if session.get("mode") == "shadow" else "hardware") != "shadow":
            # This method is also called while ``self.lock`` is held by the
            # servo loop and the HTTP intent handler.  Authority transitions
            # acquire supervisor/transport locks and can be called back by
            # status/control paths that need the teleop lock.  Never wait for
            # that lock cycle from inside the input critical section: a stale
            # joystick packet must brake, not make the next packet time out.
            threading.Thread(
                target=self.broker.mark_teleop_stopping,
                args=(str(session.get("session_id", "unknown")), self.motion_epoch, reason),
                name="teleop-stop-authority",
                daemon=True,
            ).start()
        self.intent = None
        self.clutch_active = False
        self.absolute_target_active = False
        self.absolute_target_mode = None
        self.anchor_id += 1
        self.reference_revision += 1
        self.tcp_anchor = None
        self.reference_pose = None
        self.relative_pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        self.filtered_velocity = [0.0] * 6
        self.trajectory_state = "BRAKING"
        self.trajectory_brake_reason = reason

    def freeze_for_authority_change(self, epoch: int, reason: str) -> None:
        """Discard old asynchronous work without advancing the trajectory."""
        with self.lock:
            self.motion_epoch = int(epoch)
            self.solver.discard_before_epoch(self.motion_epoch)
            if self.session:
                self.session["motion_epoch"] = self.motion_epoch
            self.intent = None
            self.absolute_target_active = False
            self.absolute_target_mode = None
            self.filtered_velocity = [0.0] * 6
            self.input_enabled = False
            self.feedback_sync_pending = True
            self.needs_resync = True
            self.trajectory_state = "HOLD_READY"
            self.trajectory_brake_reason = reason

    def _initialize_ruckig(self, q: list[float], period: float) -> None:
        period = max(0.001, min(0.1, float(period)))
        self.ruckig_otg = self.ruckig.Ruckig(7, period)
        self.ruckig_input = self.ruckig.InputParameter(7)
        self.ruckig_output = self.ruckig.OutputParameter(7)
        self.ruckig_period_s = period
        self.trajectory = {"position_rad": list(q), "velocity_rad_s": [0.0] * 7, "acceleration_rad_s2": [0.0] * 7}
        self.last_sent_velocity = [0.0] * 7

    def _hardware_preflight(self) -> dict[str, Any]:
        raw = self.broker.read_teleop_joint_limits()
        return self.authority.initialize(raw)

    def start_session(
        self,
        mode: str | None = None,
        confirm_hardware: bool = False,
        client_id: str = "anonymous",
        execution_mode: str | None = None,
        input_source: str | None = None,
    ) -> dict[str, Any]:
        execution_mode, input_source, mode = normalize_session_selection(mode, execution_mode, input_source)
        client_id = str(client_id).strip() or "anonymous"
        if execution_mode == "hardware" and input_source == "pico" and not bool(self.pose_input.get("mapping_verified", False)):
            raise PermissionError("PICO pose mapping is not verified")
        # Hardware mode is selected explicitly in the mode control. The
        # legacy confirm_hardware argument remains accepted for API
        # compatibility, but no longer blocks session startup.
        with self._session_transition_lock:
            session_id = uuid.uuid4().hex
            with self.lock:
                active = self._session_active()
                active_mode = str((self.session or {}).get("mode", "")) if active else None
                active_client = str((self.session or {}).get("client_id", "anonymous"))
                heartbeat_age = (time.monotonic_ns() - self._heartbeat_monotonic_ns) / 1e9 if self._heartbeat_monotonic_ns else float("inf")
            if active and heartbeat_age > float(self.config.get("session_timeout_s", 5.0)):
                self.stop_session("teleop session heartbeat expired")
                active_mode = None
            if active_mode == mode and active_client != client_id:
                raise PermissionError("teleop session is owned by another browser client")
            if active_mode == mode:
                return self.status()
            if active_mode:
                raise RuntimeError("an active teleop session must be stopped before changing mode")
            self.last_error = None
            self.session = {"state": "STARTING", "session_id": None, "client_id": client_id, "mode": mode, "sequence": 0}
            self._bump_state()
            try:
                if execution_mode == "shadow":
                    authority = {"status": "shadow", "effective_lower_rad": self.authority.hard_lower, "effective_upper_rad": self.authority.hard_upper, "controller_speed_rad_s": [float(self.limits.get("joint_speed_rad_s", 0.45))] * 7, "controller_acceleration_rad_s2": [float(self.config.get("solver", {}).get("ruckig_max_acceleration", 2.0))] * 7}
                    self.authority.effective = authority
                else:
                    self.broker._require_operational_control()
                    authority = self._hardware_preflight()
                self.supervisor.configure(authority, [float(self.limits.get("joint_speed_rad_s", 0.45))] * 7, [float(self.config.get("solver", {}).get("ruckig_max_acceleration", 2.0))] * 7)
                # Always create a fresh Pink bridge for a fresh session.  A
                # previous hard reset can leave a solver child alive with a
                # stale anchor; merely changing motion_epoch cannot make that
                # response safe for the new clutch.
                self.solver.close()
                self.solver.start()
                if execution_mode != "shadow":
                    broker_authority = self.broker.prepare_teleop_hardware()
                    # Start feedback first and wait for a sample produced by
                    # this session. Do not initialize Ruckig from the
                    # supervisor snapshot while the feedback worker is still
                    # warming up.
                    with self._feedback_lock:
                        self._feedback = None
                    self._start_feedback_worker()
                    feedback = self._wait_for_feedback(float(self.config.get("feedback_start_timeout_s", 2.0)))
                    joints = list(feedback["joints"])
                else:
                    joints = _finite_vector(self.config.get("shadow_initial_joints_rad", [0.0] * 7), 7, "shadow_initial_joints_rad")
            except Exception as exc:
                self.solver.close()
                self._feedback_stop.set()
                feedback_thread = self._feedback_thread
                if feedback_thread and feedback_thread is not threading.current_thread():
                    feedback_thread.join(timeout=1.0)
                self._feedback_thread = None
                with self.lock:
                    self.session = None
                    self.intent = None
                    self.input_enabled = False
                    self.last_error = f"teleop start failed: {type(exc).__name__}: {exc}"
                    self.last_result = {"ok": False, "reason": self.last_error, "timestamp": time.time(), "robot_commands_sent": False}
                    self._bump_state()
                raise
            period = 1.0 / float(self.runtime.get("control_hz", 50))
            if execution_mode != "shadow" and not self.broker.grant_teleop_tracking(session_id, int(broker_authority["control_epoch"])):
                self.solver.close()
                self._feedback_stop.set()
                feedback_thread = self._feedback_thread
                if feedback_thread and feedback_thread is not threading.current_thread():
                    feedback_thread.join(timeout=1.0)
                self._feedback_thread = None
                with self.lock:
                    self.session = None
                    self.intent = None
                    self.input_enabled = False
                    self.last_error = "teleop tracking authority could not be granted"
                    self.last_result = {"ok": False, "reason": self.last_error, "timestamp": time.time(), "robot_commands_sent": False}
                    self._bump_state()
                raise RuntimeError("teleop tracking authority could not be granted")
            with self.lock:
                self._initialize_ruckig([float(x) for x in joints], period)
                self.posture_reference = [float(x) for x in joints]
                self.shadow_joints = list(self.posture_reference)
                self.last_solver_result = None
                self._solver_reuse_count = 0
                self.motion_epoch = int(broker_authority["control_epoch"]) if execution_mode != "shadow" else self.motion_epoch
                self.solver.discard_before_epoch(self.motion_epoch)
                self.session = {"state": "ACTIVE", "session_id": session_id, "client_id": client_id, "mode": mode, "execution_mode": execution_mode, "input_source": input_source, "started_at": time.time(), "sequence": 0, "last_input_age_s": None, "motion_epoch": self.motion_epoch}
                self.intent = None
                self.anchor_id += 1
                self.clutch_active = False
                self.input_source = input_source
                self.tcp_anchor = self._current_tcp_pose([float(x) for x in joints])
                self.reference_pose = dict(self.tcp_anchor)
                self.relative_pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
                self.input_enabled = True
                self._heartbeat_monotonic_ns = time.monotonic_ns()
                self.trajectory_state, self.trajectory_brake_reason, self.feedback_sync_pending = "HOLD_READY", None, True
                self.needs_resync = True
                self.stop_event.clear()
                self._send_times.clear(); self._batch_history.clear(); self.cpv_send_count = 0; self.output_count = 0; self.loop_count = 0
                self.last_output = {"status": "held", "final_joint_target_rad": list(joints), "final_joint_velocity_rad_s": [0.0] * 7, "sequence": 0, "epoch": self.motion_epoch}
                self.last_result = {"ok": True, "reason": "session started", "robot_commands_sent": False}
                self.thread = threading.Thread(target=self._loop, name="nero-teleop-loop", daemon=True)
                self.thread.start()
                self._bump_state()
            return self.status()

    def stop_session(self, reason: str = "teleop stopped") -> dict[str, Any]:
        """Invalidate inputs and stop workers before a controller handoff.

        This method deliberately does not select J mode or issue move_j.  The
        broker's console handoff owns that second half of the transition.
        """
        with self.lock:
            active = self._session_active()
            if active:
                self._invalidate_motion(reason)
        deadline = time.monotonic() + 1.5
        while active and time.monotonic() < deadline:
            with self.lock:
                if self.trajectory_state in {"HOLD_READY", "FAULT"}:
                    break
            time.sleep(0.02)
        with self.lock:
            mode = (self.session or {}).get("mode")
            # A remembered hardware session is not proof that CPV still owns
            # the controller.  Calling send_cpv_velocity() after HOLD would
            # initialise CPV again, creating an unnecessary mode transition.
            cpv_stream_active = bool(
            (self.session or {}).get("execution_mode", "shadow" if mode == "shadow" else "hardware") != "shadow" and self.broker.teleop_stream_active()
            )
            if self.session:
                self.session.update({"state": "STOPPING", "stopped_reason": reason})
            self.input_enabled = False
            self._bump_state()
            self.stop_event.set(); self._feedback_stop.set(); self.intent = None
        servo_thread, feedback_thread = self.thread, self._feedback_thread
        if servo_thread and servo_thread is not threading.current_thread(): servo_thread.join(timeout=1.0)
        if feedback_thread and feedback_thread is not threading.current_thread(): feedback_thread.join(timeout=1.0)
        servo_stopped = not servo_thread or not servo_thread.is_alive()
        feedback_stopped = not feedback_thread or not feedback_thread.is_alive()
        if servo_stopped:
            self.thread = None
        if feedback_stopped:
            self._feedback_thread = None
        # The broker owns the terminal zero/hold transition.  Sending another
        # CPV sample here could reopen CPV after the mode handoff.
        if servo_stopped and feedback_stopped:
            self.solver.close()
        with self.lock:
            self.session = None
            self.last_error = None
            self.input_enabled = False
            self._bump_state()
        result = self.status()
        result["handoff"] = {
            "reason": reason,
            "active_session": active,
            "servo_stopped": servo_stopped,
            "feedback_stopped": feedback_stopped,
            "zero_velocity_sent": cpv_stream_active,
        }
        return result

    def request_shadow_hold(self, reason: str = "OSC shadow HOLD requested") -> dict[str, Any]:
        """Brake a shadow session to HOLD_READY without touching hardware.

        The session remains active so a deadman-style input adapter can resume
        from a fresh absolute target when its button is pressed again.
        """
        with self.lock:
            session = dict(self.session or {})
            execution_mode = session.get("execution_mode", "shadow" if session.get("mode") == "shadow" else "hardware")
            if not self._session_active() or execution_mode != "shadow":
                raise RuntimeError("shadow HOLD requires an active shadow OSC session")
            self._invalidate_motion(reason)
            self._set_result(True, reason, robot_commands_sent=False)
            self._bump_state()
            return {"ok": True, "accepted": True, "reason": reason,
                    "robot_commands_sent": False, "session_id": session.get("session_id")}

    def abandon_session_without_braking(self, epoch: int, reason: str) -> dict[str, Any]:
        """End a session after ownership revocation without sending a stop stream.

        The caller must revoke SERVO authority before calling this method. A
        CPV batch already executing in the transport owner may finish, but no
        later Pink result or intent can write after the new epoch is installed.
        """
        with self.lock:
            active = self._session_active()
            self.motion_epoch = int(epoch)
            self.solver.discard_before_epoch(self.motion_epoch)
            if self.session:
                self.session.update({"state": "STOPPING", "stopped_reason": reason, "motion_epoch": self.motion_epoch})
            self.intent = None
            self.filtered_velocity = [0.0] * 6
            self.input_enabled = False
            self.feedback_sync_pending = True
            self.needs_resync = True
            self.trajectory_state = "HOLD_READY"
            self.trajectory_brake_reason = reason
            self.stop_event.set()
            self._feedback_stop.set()
            self._bump_state()
            servo_thread, feedback_thread = self.thread, self._feedback_thread

        if servo_thread and servo_thread is not threading.current_thread():
            servo_thread.join(timeout=1.0)
        if feedback_thread and feedback_thread is not threading.current_thread():
            feedback_thread.join(timeout=1.0)
        servo_stopped = not servo_thread or not servo_thread.is_alive()
        feedback_stopped = not feedback_thread or not feedback_thread.is_alive()
        if servo_stopped:
            self.thread = None
        if feedback_stopped:
            self._feedback_thread = None
        if servo_stopped and feedback_stopped:
            self.solver.close()

        with self.lock:
            self.session = None
            self.last_error = None
            self._bump_state()
        result = self.status()
        result["direct_handoff"] = {
            "reason": reason,
            "active_session": active,
            "servo_stopped": servo_stopped,
            "feedback_stopped": feedback_stopped,
            "threads_stopped": servo_stopped and feedback_stopped,
            "braking_requested": False,
            "zero_velocity_sent": False,
        }
        return result

    def recenter(self) -> dict[str, Any]:
        with self.lock:
            if not self._session_active():
                raise RuntimeError("teleop session is not active")
            if self.clutch_active:
                raise PermissionError("release the clutch before recentering")
            q = list(self.shadow_joints or []) if self.session.get("execution_mode", self.session.get("mode")) == "shadow" else list((self._feedback_snapshot() or {}).get("joints") or [])
            if len(q) != 7:
                raise RuntimeError("teleop recenter requires seven-joint feedback")
            self.tcp_anchor = self._current_tcp_pose(q)
            self.reference_pose = dict(self.tcp_anchor)
            self.relative_pose = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            self.anchor_id += 1
            self.reference_revision += 1
            self._bump_state()
        return self.status()

    def submit_absolute_target(self, body: dict[str, Any], *, mode: str) -> dict[str, Any]:
        """Accept a base-frame TCP target without any clutch semantics."""
        if mode not in {"track_tcp", "move_tcp"}:
            raise ValueError("OSC target mode must be track_tcp or move_tcp")
        with self.lock:
            if not self._session_active():
                raise RuntimeError("OSC session is not active")
            client_id = str(body.get("client_id", "anonymous")).strip() or "anonymous"
            if client_id != str(self.session.get("client_id", "anonymous")):
                raise PermissionError("OSC command rejected: session belongs to another client")
            sequence = int(body.get("sequence", -1))
            if sequence <= int(self.session.get("sequence", 0)):
                return {"accepted": False, "reason": "stale sequence", "accepted_sequence": self.session["sequence"], "session_id": self.session["session_id"]}
            if str(body.get("session_id", self.session["session_id"])) != str(self.session["session_id"]):
                raise PermissionError("OSC command rejected: session id mismatch")
            if self.trajectory_state == "FAULT":
                raise PermissionError(f"OSC command rejected: FAULT ({self.trajectory_brake_reason or 'trajectory fault'})")
            payload = dict(body.get("payload") or body)
            target = dict(payload.get("target_pose") or {})
            reference = {
                "position_m": _finite_vector(target.get("position_m"), 3, "target_pose.position_m"),
                "orientation_xyzw": _unit_quaternion(target.get("orientation_xyzw"), "target_pose.orientation_xyzw"),
            }
            if not self._pose_in_workspace(reference):
                raise PermissionError("OSC target is outside the configured workspace")
            if self.trajectory_state == "HOLD_READY":
                resumed, reason = self._resume_from_hold_locked()
                if not resumed:
                    raise PermissionError(f"OSC command rejected: cannot resume from HOLD_READY ({reason})")
            self.session["sequence"] = sequence
            self.reference_pose = reference
            self.reference_revision += 1
            self.intent = {
                "sequence": sequence,
                "host_monotonic_ns": time.monotonic_ns(),
                "reference_revision": self.reference_revision,
                "reference_pose": dict(reference),
                "persistent": True,
                "osc_mode": mode,
            }
            self.absolute_target_active = True
            self.absolute_target_mode = mode
            self._bump_state()
            return {
                "accepted": True,
                "accepted_sequence": sequence,
                "session_id": self.session["session_id"],
                "target_pose": reference,
                "mode": mode,
            }

    def submit_intent(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self._session_active():
                raise RuntimeError("teleop session is not active")
            trusted_pico = bool(body.pop("_trusted_pico", False))
            if self.session.get("input_source", "pico" if self.session.get("mode") == "pico_hardware" else "joystick") == "pico" and not trusted_pico:
                raise PermissionError("PICO mode only accepts the paired WebSocket input")
            client_id = str(body.get("client_id", "anonymous")).strip() or "anonymous"
            if client_id != str(self.session.get("client_id", "anonymous")):
                raise PermissionError("teleop intent rejected: session belongs to another browser client")
            sequence = int(body.get("sequence", -1))
            if sequence <= int(self.session.get("sequence", 0)):
                return {"accepted": False, "reason": "stale sequence", "accepted_sequence": self.session["sequence"], "session_id": self.session["session_id"]}
            if "tcp_velocity" in body:
                raise ValueError("tcp_velocity is deprecated; submit a clutch pose intent")
            event = str(body.get("event", "pose"))
            scale = float(body.get("pose_scale", body.get("speed_scale", 1.0)))
            if not 0.05 <= scale <= 1.0: raise ValueError("pose_scale must be within [0.05, 1]")
            if str(body.get("session_id", self.session["session_id"])) != str(self.session["session_id"]):
                raise PermissionError("teleop intent rejected: session id mismatch")
            if self.trajectory_state == "FAULT":
                raise PermissionError(f"teleop input rejected: FAULT ({self.trajectory_brake_reason or 'trajectory fault'})")
            if event == "clutch_begin":
                if self.trajectory_state != "HOLD_READY":
                    raise PermissionError("teleop clutch can begin only from HOLD_READY")
                resumed, reason = self._resume_from_hold_locked()
                if not resumed:
                    raise PermissionError(f"teleop input rejected: cannot resume from HOLD_READY ({reason})")
                q = list(self.shadow_joints or []) if self.session.get("execution_mode", self.session.get("mode")) == "shadow" else list((self._feedback_snapshot() or {}).get("joints") or [])
                if len(q) != 7:
                    raise PermissionError("teleop clutch cannot anchor without seven-joint feedback")
                anchor = self._set_anchor_locked(q, "pico" if trusted_pico else "joystick")
                self.session["sequence"] = sequence
                self.intent = {"sequence": sequence, "host_monotonic_ns": time.monotonic_ns(), "anchor_id": anchor, "reference_revision": self.reference_revision, "reference_pose": dict(self.reference_pose or {}), "pose_scale": scale}
                self._bump_state()
                return {"accepted": True, "event": event, "anchor_id": anchor, "accepted_sequence": sequence, "session_id": self.session["session_id"]}
            if event == "clutch_release":
                if int(body.get("anchor_id", -1)) != self.anchor_id:
                    raise PermissionError("teleop clutch release rejected: anchor id mismatch")
                self.session["sequence"] = sequence
                self._invalidate_motion("clutch released")
                self._bump_state()
                return {"accepted": True, "event": event, "accepted_sequence": sequence, "session_id": self.session["session_id"]}
            if event != "pose":
                raise ValueError("teleop event must be clutch_begin, pose, or clutch_release")
            if not self.clutch_active or self.trajectory_state != "RUNNING":
                raise PermissionError("teleop pose rejected: clutch is not active")
            if int(body.get("anchor_id", -1)) != self.anchor_id:
                raise PermissionError("teleop pose rejected: anchor id mismatch")
            relative = dict(body.get("relative_pose") or {})
            if trusted_pico and body.get("tracking_valid") is False:
                self._invalidate_motion("PICO tracking is invalid")
                raise PermissionError("teleop pose rejected: PICO tracking is invalid")
            previous = self.relative_pose
            position_candidate = _finite_vector(relative.get("position_m"), 3, "relative_pose.position_m")
            orientation_candidate = _unit_quaternion(relative.get("orientation_xyzw"), "relative_pose.orientation_xyzw")
            packet_translation = math.sqrt(sum((position_candidate[index] - previous["position_m"][index]) ** 2 for index in range(3)))
            packet_rotation = _quat_angle(_quat_multiply(_quat_conjugate(previous["orientation_xyzw"]), orientation_candidate))
            if packet_translation > float(self.pose_input.get("max_packet_translation_m", 0.05)) or packet_rotation > float(self.pose_input.get("max_packet_rotation_rad", 0.35)):
                self._invalidate_motion("pose input discontinuity")
                raise PermissionError("teleop pose rejected: pose input discontinuity")
            reference = self._reference_from_relative(relative, scale)
            if not self._pose_in_workspace(reference):
                self._invalidate_motion("reference pose is outside the configured workspace")
                raise PermissionError("teleop pose rejected: reference pose is outside the configured workspace")
            old_reference = self.reference_pose or {}
            old_position = old_reference.get("position_m") or []
            old_orientation = old_reference.get("orientation_xyzw") or []
            reference_changed = (
                len(old_position) != 3
                or len(old_orientation) != 4
                or max(abs(float(a) - float(b)) for a, b in zip(reference["position_m"], old_position)) > 1e-9
                or _quat_angle(_quat_multiply(_quat_conjugate(_unit_quaternion(old_orientation)), reference["orientation_xyzw"])) > 1e-8
            )
            self.session["sequence"] = sequence
            self.relative_pose = {"position_m": position_candidate, "orientation_xyzw": orientation_candidate}
            self.reference_pose = reference
            if reference_changed:
                self.reference_revision += 1
            self.intent = {"sequence": sequence, "host_monotonic_ns": time.monotonic_ns(), "anchor_id": self.anchor_id, "reference_revision": self.reference_revision, "reference_pose": dict(reference), "pose_scale": scale}
            self._bump_state()
            return {"accepted": True, "accepted_sequence": sequence, "session_id": self.session["session_id"], "accepted_monotonic_ns": self.intent["host_monotonic_ns"]}

    def submit_pico_intent(self, body: dict[str, Any]) -> dict[str, Any]:
        """Trusted ingress used only by the paired PICO WebSocket gateway."""
        with self.lock:
            if not self._session_active() or self.session.get("input_source", "pico" if self.session.get("mode") == "pico_hardware" else "joystick") != "pico":
                raise PermissionError("PICO input rejected: no active PICO session")
            payload = dict(body)
            payload["_trusted_pico"] = True
            payload["client_id"] = str(self.session.get("client_id", "anonymous"))
            payload["session_id"] = str(self.session.get("session_id", ""))
        return self.submit_intent(payload)

    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        with self.lock:
            if not self._session_active() or str(self.session.get("client_id")) != str(client_id) or str(self.session.get("session_id")) != str(session_id):
                raise PermissionError("teleop heartbeat rejected")
            self._heartbeat_monotonic_ns = time.monotonic_ns()
            return self.status()

    def heartbeat_expired(self) -> bool:
        """Whether the active OSC owner lease has expired.

        The control broker's watchdog owns the resulting hardware handoff;
        keeping this method read-only avoids a servo-thread/transport lock
        inversion while still making browser loss a server-enforced boundary.
        """
        with self.lock:
            if not self._session_active() or not self._heartbeat_monotonic_ns:
                return False
            timeout_s = float(self.config.get("session_timeout_s", 5.0))
            return (time.monotonic_ns() - self._heartbeat_monotonic_ns) / 1e9 > timeout_s

    def _start_feedback_worker(self) -> None:
        self._feedback_stop.clear()
        def run() -> None:
            period = 1.0 / max(50.0, float(self.runtime.get("control_hz", 50)))
            next_tick = time.monotonic()
            while not self._feedback_stop.is_set():
                try:
                    row = self.broker.read_teleop_feedback()
                    q = list(row.get("joint_angles_rad") or [])
                    qd = list(row.get("joint_velocity_rad_s") or [])
                    if len(q) == 7 and len(qd) == 7 and all(item is not None for item in qd):
                        received_monotonic_ns = time.monotonic_ns()
                        with self._feedback_lock:
                            self._feedback = {"joints": [float(x) for x in q], "velocities": [float(x) for x in qd], "monotonic_ns": received_monotonic_ns}
                except Exception:
                    pass
                next_tick += period
                self._feedback_stop.wait(max(0.0, next_tick - time.monotonic()))
        self._feedback_thread = threading.Thread(target=run, name="nero-teleop-feedback", daemon=True); self._feedback_thread.start()

    def _feedback_snapshot(self) -> dict[str, Any] | None:
        with self._feedback_lock: return dict(self._feedback) if self._feedback else None

    def _wait_for_feedback(self, timeout_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            row = self._feedback_snapshot()
            if row:
                joints = row.get("joints") or []
                velocities = row.get("velocities") or []
                if len(joints) == 7 and len(velocities) == 7 and all(
                    math.isfinite(float(value)) for value in [*joints, *velocities]
                ):
                    return row
            time.sleep(0.01)
        raise RuntimeError("timed out waiting for first valid teleop joint feedback")

    def _set_result(self, ok: bool, reason: str, **extra: Any) -> None:
        with self.lock:
            # Keep the last valid FK result while braking or holding.  A stop
            # cycle intentionally has no Pink result, but the visualization
            # must not blink away just because motion input became zero.
            if extra.get("solver") is None and isinstance(self.last_result.get("solver"), dict):
                extra["solver"] = self.last_result["solver"]
            self.last_result = {"ok": ok, "reason": reason, "timestamp": time.time(), **extra}

    def status(self) -> dict[str, Any]:
        with self.lock:
            debug_fn = getattr(self.solver, "debug_status", None)
            debug = debug_fn() if callable(debug_fn) else {}
            return {"state_sequence": self.state_sequence, "session": self._session_view(), "intent": dict(self.intent) if self.intent else None, "input_enabled": self.input_enabled, "input_source": self.input_source, "clutch_active": self.clutch_active, "anchor_id": self.anchor_id, "reference_revision": self.reference_revision, "tcp_anchor": dict(self.tcp_anchor) if self.tcp_anchor else None, "relative_pose": dict(self.relative_pose), "reference_pose": dict(self.reference_pose) if self.reference_pose else None, "last_error": self.last_error, "last_result": dict(self.last_result), "last_output": dict(self.last_output), "pose_mapping_verified": bool(self.pose_input.get("mapping_verified", False)), "solver": {"running": bool(self.solver.process and self.solver.process.poll() is None), "python": str(self.solver.python), "tcp_verified": False, "debug": debug}, "workspace": {"min_xyz_m": list(self.limits.get("workspace_min_m", [-0.45, -0.15, -0.02])), "max_xyz_m": list(self.limits.get("workspace_max_m", [0.45, 0.60, 0.70])), "min_flange_z_m": float(self.limits.get("min_flange_z_m", -0.02))}, "diagnostics": {"trajectory_state": self.trajectory_state, "trajectory_brake_reason": self.trajectory_brake_reason, "motion_epoch": self.motion_epoch, "needs_resync": self.needs_resync, "last_sent_velocity_rad_s": list(self.last_sent_velocity), "trajectory": dict(self.trajectory) if self.trajectory else None, "limit_authority": dict(self.authority.effective) if self.authority.effective else None, "supervisor": dict(self.supervisor.limit_data) if self.supervisor.limit_data else None, "timing": dict(self._timing), "loop_count": self.loop_count, "output_count": self.output_count, "cpv_dispatch_count": self.cpv_send_count, "recent_cpv_batches": list(self._batch_history)[-10:]}}

    def kinematics(self) -> dict[str, Any]:
        return {"schema_version": "nero.teleop.v1", "tcp_verified": False, "tcp_offset_m": self.config.get("tcp", {}).get("offset_from_link7_m"), "last_result": dict(self.last_result), "shadow_default": True}

    def _sync_if_settled(self, feedback_q: list[float]) -> bool:
        if self.trajectory_state != "HOLD_READY": return False
        self._initialize_ruckig(feedback_q, 1.0 / float(self.runtime.get("control_hz", 50)))
        self.posture_reference = list(feedback_q)
        self.feedback_sync_pending = False
        self.needs_resync = False
        return True

    def _resume_from_hold_locked(self) -> tuple[bool, str]:
        """Resume only after a fresh, current-client deadman intent.

        This method is called with ``self.lock`` held.  HOLD_READY never
        advances the old trajectory; every recovery starts from a fresh
        measured (or shadow) state.
        """
        session = self.session or {}
        if self.trajectory_state != "HOLD_READY":
            return False, f"trajectory is {self.trajectory_state}"
        if not self._session_active():
            return False, "session is not active"
        if session.get("execution_mode", "shadow" if session.get("mode") == "shadow" else "hardware") != "shadow":
            feedback = self._feedback_snapshot()
            if not feedback:
                return False, "hardware feedback unavailable"
            age = max(0.0, (time.monotonic_ns() - int(feedback.get("monotonic_ns", 0))) / 1e9)
            if age > float(self.limits.get("feedback_soft_stale_s", 0.06)):
                return False, f"hardware feedback is stale ({age * 1000.0:.0f} ms)"
            q = list(feedback.get("joints") or [])
            qd = list(feedback.get("velocities") or [])
            if len(q) != 7 or len(qd) != 7 or not all(math.isfinite(float(x)) for x in [*q, *qd]):
                return False, "hardware feedback is invalid"
            if not self.broker.servo_can_write(str(session.get("session_id")), self.motion_epoch):
                return False, "SERVO write authority is unavailable"
            if not self.broker.grant_teleop_tracking(str(session.get("session_id")), self.motion_epoch):
                return False, "SERVO tracking authority could not be restored"
        else:
            q = list(self.shadow_joints or (self.trajectory or {}).get("position_rad") or [])
            if len(q) != 7 or not all(math.isfinite(float(x)) for x in q):
                return False, "shadow state is invalid"
        self._sync_if_settled([float(x) for x in q])
        self.input_enabled = True
        self.trajectory_state = "RUNNING"
        self.trajectory_brake_reason = None
        return True, "resumed from HOLD_READY"

    def _advance_ruckig(self, target_velocity: list[float], period: float) -> tuple[list[float], dict[str, Any]]:
        if self.trajectory is None or self.ruckig_input is None or self.ruckig_output is None or self.ruckig_otg is None:
            raise RuntimeError("Ruckig state is not initialized")
        period = max(0.001, min(0.1, float(period)))
        # Ruckig's cycle time is fixed for the lifetime of the trajectory.
        # Keep the same instance and pass its output state forward; rebuilding
        # it from wall-clock jitter destroys the downstream dynamic state.
        data = self.supervisor._require()
        inp = self.ruckig_input
        inp.current_position = self.trajectory["position_rad"]
        inp.current_velocity = self.trajectory["velocity_rad_s"]
        inp.current_acceleration = self.trajectory["acceleration_rad_s2"]
        inp.target_velocity = list(target_velocity)
        inp.target_acceleration = [0.0] * 7
        inp.max_velocity = list(data["speed_rad_s"])
        inp.max_acceleration = list(data["acceleration_rad_s2"])
        inp.max_jerk = [float(self.config.get("solver", {}).get("ruckig_max_jerk", 20.0))] * 7
        inp.control_interface = self.ruckig.ControlInterface.Velocity
        started = time.perf_counter()
        code = self.ruckig_otg.update(inp, self.ruckig_output)
        elapsed = (time.perf_counter() - started) * 1000.0
        if code < 0: raise RuntimeError(f"Ruckig rejected velocity trajectory: {code}")
        velocity = list(self.ruckig_output.new_velocity)
        self.trajectory = {"position_rad": list(self.ruckig_output.new_position), "velocity_rad_s": velocity, "acceleration_rad_s2": list(self.ruckig_output.new_acceleration)}
        self.ruckig_output.pass_to_input(inp)
        return velocity, {"ruckig_ms": elapsed, "cycle_s": self.ruckig_period_s, "measured_period_s": period, "result_code": int(code), "trajectory": dict(self.trajectory)}

    def _loop(self) -> None:
        hz = float(self.runtime.get("control_hz", 50)); period = 1.0 / hz
        # perf_counter has sub-millisecond resolution on Windows, unlike the
        # coarse scheduler-facing monotonic clock on this host.
        clock = time.perf_counter
        next_tick, previous_tick = clock(), None
        while not self.stop_event.is_set():
            delay = next_tick - clock()
            if delay > 0:
                # Windows wait handles can round an event wait to 31 ms. Use
                # short sleeps while there is time left, then a short
                # high-resolution spin so a 20 ms Ruckig cycle is not turned
                # into a scheduler-dependent 31 ms cycle.
                while delay > 0 and not self.stop_event.is_set():
                    if delay > 0.005:
                        time.sleep(0.001)
                    else:
                        pass
                    delay = next_tick - clock()
                continue
            tick = clock()
            # Do not catch up missed ticks in a burst. A burst produces
            # artificial 1 ms control intervals while Pink/Ruckig are still
            # configured around the nominal 20 ms servo period.
            next_tick = tick + period
            actual_dt = period if previous_tick is None else max(0.001, tick - previous_tick); previous_tick = tick
            try:
                with self.lock:
                    session = dict(self.session) if self.session else None; intent = dict(self.intent) if self.intent else None; self.loop_count += 1
                if not session or session.get("state") != "ACTIVE": continue
                shadow = session.get("execution_mode", "shadow" if session.get("mode") == "shadow" else "hardware") == "shadow"
                feedback = None if shadow else self._feedback_snapshot()
                if shadow:
                    q = list(self.shadow_joints or (self.trajectory or {}).get("position_rad") or [])
                    qd = list((self.trajectory or {}).get("velocity_rad_s") or [0.0] * 7)
                    feedback_age = 0.0
                else:
                    q = list((feedback or {}).get("joints") or []); qd = list((feedback or {}).get("velocities") or [])
                    feedback_age = float("inf") if not feedback else max(0.0, (time.monotonic_ns() - int(feedback["monotonic_ns"])) / 1e9)
                if len(q) != 7 or len(qd) != 7:
                    self._fault_zero("seven-joint feedback unavailable", shadow); continue
                soft_stale = feedback_age > float(self.limits.get("feedback_soft_stale_s", 0.06))
                hard_stale = feedback_age > float(self.limits.get("feedback_hard_stale_s", 0.15))
                age = float("inf") if not intent else max(0.0, (time.monotonic_ns() - int(intent["host_monotonic_ns"])) / 1e9)
                # OSC absolute targets are persistent setpoints. Clutch is a
                # legacy input-adapter concern and is never required by OSC.
                persistent_target = bool((intent or {}).get("persistent"))
                deadman = bool(self.clutch_active or self.absolute_target_active)
                valid_input = bool(intent) and (persistent_target or age <= float(self.limits.get("deadman_timeout_s", 0.25)))
                with self.lock:
                    if self.session: self.session["last_input_age_s"] = age if math.isfinite(age) else None
                    explicit_nonzero = bool(self.clutch_active and intent)
                    state_before_input_check = self.trajectory_state
                    heartbeat_age = (
                        max(0.0, (time.monotonic_ns() - self._heartbeat_monotonic_ns) / 1e9)
                        if self._heartbeat_monotonic_ns else None
                    )
                    # A normally started OSC session always installs its
                    # lease timestamp.  Keep hand-built legacy/test sessions
                    # without one backward compatible rather than treating
                    # their missing timestamp as an immediate timeout.
                    heartbeat_expired = heartbeat_age is not None and heartbeat_age > float(self.config.get("session_timeout_s", 5.0))
                    trajectory = self.trajectory or {}
                    brake_is_still_moving = (
                        state_before_input_check == "BRAKING"
                        and (
                            max((abs(float(value)) for value in trajectory.get("velocity_rad_s", [])), default=0.0)
                            > float(self.config.get("solver", {}).get("hold_velocity_epsilon_rad_s", 0.005))
                            or max((abs(float(value)) for value in trajectory.get("acceleration_rad_s2", [])), default=0.0)
                            > float(self.config.get("solver", {}).get("hold_acceleration_epsilon_rad_s2", 0.02))
                        )
                    )
                    if self.trajectory_state == "RUNNING" and heartbeat_expired:
                        # A persistent OSC target is never permission to run
                        # unattended.  The browser heartbeat is the owner
                        # lease; immediately brake it when that lease expires.
                        self._invalidate_motion("teleop session heartbeat expired")
                    elif self.trajectory_state == "RUNNING" and intent and (not valid_input or not deadman):
                        self._invalidate_motion("input timed out" if not valid_input else "clutch released")
                    # A stationary session has no active motion to protect.
                    # Keep an old sample visible while HOLD_READY; the next
                    # nonzero input still performs its own freshness check.
                    motion_or_brake_active = (
                        brake_is_still_moving
                        or (state_before_input_check == "RUNNING" and explicit_nonzero)
                    )
                    if hard_stale and motion_or_brake_active:
                        self.trajectory_state, self.trajectory_brake_reason = "FAULT", "feedback hard stale"
                        self.input_enabled = False
                    state = self.trajectory_state; epoch = self.motion_epoch
                # HOLD_READY is a frozen state.  It must not integrate or
                # resynchronise until submit_intent receives a fresh deadman
                # input from the owning client.
                if state == "HOLD_READY":
                    with self.lock:
                        self._timing = {
                            "actual_dt_s": actual_dt,
                            "feedback_age_s": feedback_age,
                            "feedback_soft_stale": soft_stale,
                            "feedback_hard_stale": hard_stale,
                            "solver_age_s": None,
                            "batch_skew_ms": None,
                            "delay_budget_s": None,
                            "motion_epoch": epoch,
                            "gate_ok": True,
                        }
                    reason = (
                        "HOLD_READY; feedback temporarily stale; waiting for fresh deadman"
                        if hard_stale else "HOLD_READY; waiting for fresh deadman"
                    )
                    self._set_result(False, reason, robot_commands_sent=False)
                    continue
                target_pose = dict((intent or {}).get("reference_pose") or self.reference_pose or {})
                if state != "RUNNING" or not target_pose:
                    target_pose = self._current_tcp_pose(q)
                # A live P1 thread without SERVO authority is deliberately
                # frozen. It must not integrate old targets and later chase
                # them when a mode transition returns control to teleop.
                if not shadow and not self.broker.servo_can_write(str(session.get("session_id")), epoch):
                    with self.lock:
                        self.feedback_sync_pending = True
                        self.trajectory_state = "HOLD_READY"
                    self._set_result(False, "servo authority unavailable; trajectory frozen", robot_commands_sent=False)
                    continue
                data = self.supervisor._require()
                anchor_id = int((intent or {}).get("anchor_id", self.anchor_id))
                reference_revision = int((intent or {}).get("reference_revision", self.reference_revision))
                self.solver.submit({"sequence": int((intent or {}).get("sequence", session.get("sequence", 0))), "motion_epoch": epoch, "anchor_id": anchor_id, "reference_revision": reference_revision, "joint_angles_rad": q, "joint_state_monotonic_ns": time.monotonic_ns(), "target_position_m": target_pose["position_m"], "target_orientation_xyzw": target_pose["orientation_xyzw"], "last_sent_joint_velocity_rad_s": self.last_sent_velocity, "joint_speed_limit_rad_s": data["speed_rad_s"], "joint_acceleration_limit_rad_s2": data["acceleration_rad_s2"], "soft_lower_rad": data["soft_lower_rad"], "soft_upper_rad": data["soft_upper_rad"], "posture_reference_rad": self.posture_reference or q, "posture_cost": float(self.config.get("solver", {}).get("posture_cost", 0.005)), "damping_cost": float(self.config.get("solver", {}).get("damping_cost", 0.05)), "frame_position_cost": float(self.config.get("solver", {}).get("frame_position_cost", 10.0)), "frame_orientation_cost": float(self.config.get("solver", {}).get("frame_orientation_cost", 1.0)), "frame_gain": float(self.config.get("solver", {}).get("frame_gain", 0.5)), "feedback_limit_tolerance_rad": float(self.config.get("solver", {}).get("feedback_limit_tolerance_rad", 0.05)), "dt_s": period})
                pink = self.solver.poll(epoch, anchor_id, reference_revision)
                if pink and pink.get("ok"):
                    self._solver_reuse_count = 0
                elif self.last_solver_result and self.last_solver_result.get("ok"):
                    last_revision = int(self.last_solver_result.get("reference_revision", -1))
                    last_anchor = int(self.last_solver_result.get("anchor_id", -1))
                    last_age = max(0.0, (time.monotonic_ns() - int(self.last_solver_result.get("joint_state_monotonic_ns") or self.last_solver_result.get("solver_monotonic_ns", time.monotonic_ns()))) / 1e9)
                    reuse_limit = min(2, max(0, int(self.limits.get("max_stale_velocity_repeats", 2))))
                    if last_anchor == anchor_id and last_revision <= reference_revision and last_age <= float(self.limits.get("solver_stale_s", 0.10)) and self._solver_reuse_count < reuse_limit:
                        pink = self.last_solver_result
                        self._solver_reuse_count += 1
                    else:
                        pink = None
                solver_age = None if not pink else max(0.0, (time.monotonic_ns() - int(pink.get("joint_state_monotonic_ns") or pink.get("solver_monotonic_ns", time.monotonic_ns()))) / 1e9)
                if pink and pink.get("ok"):
                    self.last_solver_result = pink
                if state == "RUNNING" and pink and pink.get("ok"):
                    raw_dq = _finite_vector(pink.get("pink_joint_velocity_rad_s"), 7, "pink_joint_velocity_rad_s")
                    direct_pink_cpv = bool(self.config.get("solver", {}).get("direct_pink_cpv_position", False))
                    if not direct_pink_cpv:
                        # Pink's position-task IK can chatter near workspace
                        # boundaries even when the Cartesian target is fixed.
                        # Smooth its velocity proposal before Ruckig so solver
                        # sign changes do not become visible joint oscillation.
                        filter_alpha = float(self.limits.get("input_filter_alpha", 1.0))
                        if not math.isfinite(filter_alpha):
                            filter_alpha = 1.0
                        filter_alpha = max(0.05, min(1.0, filter_alpha))
                        raw_dq = [
                            filter_alpha * value + (1.0 - filter_alpha) * previous
                            for value, previous in zip(raw_dq, self.last_sent_velocity)
                        ]
                    delay_budget = self.supervisor.observe_delay(feedback_age, actual_dt, solver_age, float(self._timing.get("batch_skew_ms", 0.0) or 0.0) / 1000.0, float(self._timing.get("response_s", 0.0) or 0.0))
                    target_velocity, supervisor_report = self.supervisor.limit_velocity(q, qd, raw_dq, delay_budget)
                    soft_stale_scale = 1.0
                    if soft_stale and not hard_stale:
                        soft_limit = float(self.limits.get("feedback_soft_stale_s", 0.06))
                        hard_limit = float(self.limits.get("feedback_hard_stale_s", 0.15))
                        span = max(1e-6, hard_limit - soft_limit)
                        soft_stale_scale = max(0.2, 1.0 - 0.8 * (feedback_age - soft_limit) / span)
                        target_velocity = [value * soft_stale_scale for value in target_velocity]
                    supervisor_report["feedback_soft_stale"] = soft_stale
                    supervisor_report["feedback_velocity_scale"] = soft_stale_scale
                    self.last_solver_result = pink
                else:
                    target_velocity, supervisor_report, pink = [0.0] * 7, {"reason": "direct Ruckig braking"}, None
                    delay_budget = self.supervisor.observe_delay(feedback_age, actual_dt, None, float(self._timing.get("batch_skew_ms", 0.0) or 0.0) / 1000.0, float(self._timing.get("response_s", 0.0) or 0.0))
                direct_pink_cpv = bool(self.config.get("solver", {}).get("direct_pink_cpv_position", False))
                if direct_pink_cpv:
                    # Diagnostic mode: Pink's unfiltered differential-IK
                    # output is integrated exactly once into the CPV joint
                    # position.  Ruckig is intentionally not advanced.
                    planned = list(target_velocity)
                    ruckig_info = {"enabled": False, "mode": "bypassed", "reason": "direct Pink-to-CPV diagnostic mode", "measured_period_s": actual_dt}
                else:
                    planned, ruckig_info = self._advance_ruckig(target_velocity, actual_dt)
                final_velocity, gate_ok, gate_reason = self.supervisor.final_gate(
                    q, qd, planned, delay_budget,
                    hard_stale and motion_or_brake_active,
                )
                gate_limited = gate_reason == "velocity clipped by final safety gate"
                if not gate_ok:
                    with self.lock: self.trajectory_state, self.trajectory_brake_reason = "FAULT", gate_reason or "final safety gate rejected velocity"
                    final_velocity = [0.0] * 7
                settled = max(abs(x) for x in self.trajectory["velocity_rad_s"]) <= float(self.config.get("solver", {}).get("hold_velocity_epsilon_rad_s", 0.005)) and max(abs(x) for x in self.trajectory["acceleration_rad_s2"]) <= float(self.config.get("solver", {}).get("hold_acceleration_epsilon_rad_s2", 0.02))
                if shadow:
                    # Shadow feedback must model the velocity that would
                    # actually leave the final safety gate.  Using Ruckig's
                    # planned position here lets a clipped/braking command
                    # move the virtual robot along a different trajectory
                    # than the CPV command, which creates a false feedback
                    # error and can sustain a limit cycle around T_ref.
                    previous_velocity = list(self.trajectory["velocity_rad_s"])
                    self.shadow_joints = [
                        float(current) + float(command) * actual_dt
                        for current, command in zip(q, final_velocity)
                    ]
                    self.trajectory["position_rad"] = list(self.shadow_joints)
                    self.trajectory["velocity_rad_s"] = list(final_velocity)
                    self.trajectory["acceleration_rad_s2"] = [
                        (float(command) - float(previous)) / max(0.001, actual_dt)
                        for command, previous in zip(final_velocity, previous_velocity)
                    ]
                    result_reason = "shadow velocity servo"
                    if gate_limited:
                        result_reason += " with final safety gate velocity limit"
                    with self.lock:
                        self.output_count += 1
                        self.last_output = {"status": "limited" if gate_limited else "accepted", "final_joint_target_rad": list(self.shadow_joints), "final_joint_velocity_rad_s": list(final_velocity), "sequence": int((intent or {}).get("sequence", session.get("sequence", 0))), "epoch": epoch}
                    self._set_result(True, result_reason, robot_commands_sent=False, solver=pink if pink and pink.get("ok") else None, ruckig=ruckig_info, supervisor=supervisor_report, gate_reason=gate_reason, gate_limited=gate_limited, final_joint_target_rad=list(self.shadow_joints), final_joint_velocity_rad_s=list(final_velocity))
                    batch = None
                else:
                    # In direct diagnostic mode, and whenever the final gate
                    # changes Ruckig's proposal, build CPV from the measured
                    # joints. This keeps the sole output a joint position.
                    position_target = list(self.trajectory["position_rad"])
                    if direct_pink_cpv or not gate_ok or any(abs(actual - planned_value) > 1e-9 for actual, planned_value in zip(final_velocity, planned)):
                        previous_velocity = list(self.trajectory["velocity_rad_s"])
                        position_target = [
                            float(current) + float(command) * actual_dt
                            for current, command in zip(q, final_velocity)
                        ]
                        self.trajectory["position_rad"] = list(position_target)
                        self.trajectory["velocity_rad_s"] = list(final_velocity)
                        self.trajectory["acceleration_rad_s2"] = [
                            (float(command) - float(previous)) / max(0.001, actual_dt)
                            for command, previous in zip(final_velocity, previous_velocity)
                        ]
                    batch = self.broker.send_servo_position(position_target, str(session.get("session_id")), epoch)
                    with self.lock:
                        self.cpv_send_count += 1; self.output_count += 1; self._send_times.append(time.monotonic()); self._batch_history.append({"motion_epoch": epoch, "joint_target_rad": list(position_target), "joint_velocity_rad_s": list(final_velocity), **batch})
                        self.last_output = {"status": "limited" if gate_limited else "accepted", "final_joint_target_rad": list(position_target), "final_joint_velocity_rad_s": list(final_velocity), "sequence": int((intent or {}).get("sequence", session.get("sequence", 0))), "epoch": epoch}
                    result_reason = "CPV joint-position batch sent"
                    if soft_stale and not hard_stale:
                        result_reason = "CPV joint-position batch sent with soft-stale derating"
                    elif hard_stale:
                        result_reason = "CPV measured-position hold sent after hard stale feedback"
                    elif gate_limited:
                        result_reason = "CPV joint-position batch sent with final safety gate velocity limit"
                    self._set_result(gate_ok, result_reason, robot_commands_sent=True, solver=pink if pink and pink.get("ok") else None, ruckig=ruckig_info, supervisor=supervisor_report, gate_reason=gate_reason, gate_limited=gate_limited, final_joint_target_rad=list(position_target), final_joint_velocity_rad_s=list(final_velocity))
                self.last_sent_velocity = list(final_velocity)
                with self.lock:
                    if (
                        self.absolute_target_mode == "move_tcp"
                        and pink and pink.get("ok")
                        and float(pink.get("position_error_m", float("inf"))) <= float(self.config.get("osc", {}).get("move_position_tolerance_m", 0.004))
                        and float(pink.get("orientation_error_rad", float("inf"))) <= float(self.config.get("osc", {}).get("move_orientation_tolerance_rad", 0.05))
                    ):
                        self._invalidate_motion("OSC move_tcp target reached")
                    if self.trajectory_state == "BRAKING" and settled:
                        self.trajectory_state, self.trajectory_brake_reason, self.feedback_sync_pending = "HOLD_READY", None, True
                        self.needs_resync = True
                        if not shadow:
                            self.broker.latch_teleop_hold("teleop braking settled")
                    self._timing = {"actual_dt_s": actual_dt, "feedback_age_s": feedback_age, "feedback_soft_stale": soft_stale, "feedback_hard_stale": hard_stale, "solver_age_s": solver_age, "batch_skew_ms": (batch or {}).get("batch_skew_ms"), "delay_budget_s": delay_budget, "motion_epoch": epoch, "gate_ok": gate_ok, "gate_limited": gate_limited, "gate_reason": gate_reason}
            except PermissionError as exc:
                # Authority revocation is an expected ownership handoff, not a
                # robot fault.  In particular, it must never promote a stale
                # P1 batch into P0's explicit CPV-zero safety path.
                with self.lock:
                    self.feedback_sync_pending = True
                    self.needs_resync = True
                    if self.trajectory_state != "FAULT":
                        self.trajectory_state = "HOLD_READY"
                        self.trajectory_brake_reason = "servo write authority revoked"
                self._set_result(False, str(exc), robot_commands_sent=False)
            except Exception as exc:
                self._fault_zero(f"teleop loop: {type(exc).__name__}: {exc}", shadow=False)

    def _fault_zero(self, reason: str, shadow: bool) -> None:
        with self.lock: self.trajectory_state, self.trajectory_brake_reason = "FAULT", reason
        if not shadow:
            self.broker.trigger_safety_fault(reason)
        self.last_sent_velocity = [0.0] * 7
        self._set_result(False, reason, robot_commands_sent=not shadow)


# Backwards-compatible name for the legacy clutch input adapter and tests.
TeleopController = OperationalSpaceServo
