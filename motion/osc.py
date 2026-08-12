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
from typing import Any, Protocol



class KinematicsUnavailable(RuntimeError):
    pass


class OscHardwarePort(Protocol):
    """Minimal hardware/authority surface available to the private servo."""

    def require_operational_control(self) -> None: ...
    def prepare_osc_hardware(self) -> dict[str, Any]: ...
    def osc_stream_active(self) -> bool: ...
    def grant_osc_tracking(self, session_id: str, epoch: int) -> bool: ...
    def mark_osc_stopping(self, session_id: str, epoch: int, reason: str) -> bool: ...
    def servo_can_write(self, session_id: str, epoch: int) -> bool: ...
    def send_servo_position(self, joints: list[float], session_id: str, epoch: int) -> dict[str, Any]: ...
    def latch_osc_hold(self, reason: str) -> dict[str, Any]: ...
    def trigger_safety_fault(self, reason: str) -> dict[str, Any]: ...


class _OscRxPort(Protocol):
    """Private, receive-only path to SDK CAN caches.

    Implementations must not enqueue a TX transaction or issue a CAN request.
    """

    def read_cached_feedback(self) -> dict[str, Any]: ...


class _OscFeedbackReceiver:
    """Continuously sample the SDK RX cache without touching TX arbitration."""

    def __init__(self, source: _OscRxPort, control_hz: float) -> None:
        self._source = source
        self._period_s = 1.0 / max(50.0, float(control_hz))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample: dict[str, Any] | None = None
        self._revision = 0
        self._last_sdk_timestamp: Any = None
        self._last_fresh_received_ns: int | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="nero-osc-rx", daemon=True)
        self._thread.start()

    def close(self) -> bool:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        stopped = not thread or not thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._sample is None:
                return None
            sample = dict(self._sample)
            sample["joints"] = list(sample["joints"])
            sample["velocities"] = list(sample["velocities"])
            sample["last_error"] = self._last_error
            sample["running"] = bool(self._thread and self._thread.is_alive())
            sample["age_s"] = max(0.0, (time.monotonic_ns() - int(sample["fresh_received_at_monotonic_ns"])) / 1e9)
            return sample

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def wait_for_revision_after(self, revision: int, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.001, float(timeout_s))
        while time.monotonic() < deadline:
            sample = self.snapshot()
            if sample is not None and int(sample.get("revision", 0)) > int(revision):
                return sample
            time.sleep(0.005)
        raise RuntimeError("timed out waiting for fresh OSC RX feedback")

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                requested_ns = time.monotonic_ns()
                row = self._source.read_cached_feedback()
                received_ns = int(row.get("received_at_monotonic_ns") or time.monotonic_ns())
                joints = list(row.get("joint_angles_rad") or [])
                velocities = list(row.get("joint_velocity_rad_s") or [])
                if len(joints) == 7 and len(velocities) == 7 and all(value is not None for value in velocities):
                    sdk_timestamp = row.get("sdk_joint_timestamp")
                    advanced = sdk_timestamp is not None and sdk_timestamp != self._last_sdk_timestamp
                    if advanced:
                        self._last_sdk_timestamp = sdk_timestamp
                        self._last_fresh_received_ns = received_ns
                    if self._last_fresh_received_ns is None:
                        self._last_fresh_received_ns = received_ns
                    with self._lock:
                        self._revision += 1
                        self._sample = {
                            "revision": self._revision,
                            "joints": [float(value) for value in joints],
                            "velocities": [float(value) for value in velocities],
                            "sdk_joint_timestamp": sdk_timestamp,
                            "joint_feedback_hz": row.get("joint_feedback_hz"),
                            "sdk_timestamp_advanced": advanced,
                            "fresh_received_at_monotonic_ns": self._last_fresh_received_ns,
                            "monotonic_ns": self._last_fresh_received_ns,
                            "requested_monotonic_ns": requested_ns,
                            "received_monotonic_ns": received_ns,
                            "read_duration_s": max(0.0, (received_ns - requested_ns) / 1e9),
                        }
                        self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            next_tick += self._period_s
            self._stop.wait(max(0.0, next_tick - time.monotonic()))


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


def _quat_scale(value: list[float], scale: float, max_scale: float = 1.0) -> list[float]:
    """Slerp identity->value, retaining a unit quaternion."""
    angle = _quat_angle(value)
    if angle < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    sign = 1.0 if value[3] >= 0.0 else -1.0
    bounded_scale = max(0.0, min(float(max_scale), scale))
    return _unit_quaternion([value[0] * sign * math.sin(angle * bounded_scale / 2.0) / max(1e-12, math.sin(angle / 2.0)), value[1] * sign * math.sin(angle * bounded_scale / 2.0) / max(1e-12, math.sin(angle / 2.0)), value[2] * sign * math.sin(angle * bounded_scale / 2.0) / max(1e-12, math.sin(angle / 2.0)), math.cos(angle * bounded_scale / 2.0)])


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


def pose_from_tcp(tcp: dict[str, Any]) -> dict[str, list[float]]:
    """Convert Pink's base-frame TCP dictionary to the public pose format."""
    position = _finite_vector(tcp.get("position_m"), 3, "tcp.position_m")
    rotation = _matrix3(tcp.get("rotation"), "tcp.rotation")
    return {"position_m": position, "orientation_xyzw": _matrix_to_quat(rotation)}


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


class KinematicsClient:
    """Latest-sample JSONL bridge for Pinocchio/Pink only."""

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        solver = config.get("solver", {})
        self.root = project_root
        python = Path(str(solver.get("python", "")))
        self.python = python if python.is_absolute() else project_root / python
        self.script = project_root / str(solver.get("script", "motion/osc_kinematics_server.py"))
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
        self.motion_requests: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        self.fk_responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.ready: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._response_condition = threading.Condition(self.lock)
        self._latest_motion_response: dict[str, Any] | None = None
        self._writer_process: subprocess.Popen[str] | None = None
        self.minimum_epoch = 0
        self._solver_request_id = 0
        self._last_response_request_id = 0
        self._last_response_seen_request_id = 0
        self._pending_solver_response: dict[str, Any] | None = None
        self._debug = {
            "offered": 0,
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
        threading.Thread(target=self._read_stdout, name="osc-pink-reader", daemon=True).start()
        try:
            ready = self.ready.get(timeout=max(1.0, self.startup_timeout_s))
        except queue.Empty as exc:
            self.close()
            raise KinematicsUnavailable("Pinocchio/Pink solver did not become ready") from exc
        if not ready.get("ready"):
            self.close()
            raise KinematicsUnavailable(str(ready.get("error", ready)))
        self._writer_process = self.process
        threading.Thread(
            target=self._motion_writer,
            args=(self.process,),
            name="osc-pink-writer",
            daemon=True,
        ).start()

    def close(self) -> None:
        process, self.process = self.process, None
        self._writer_process = None
        try:
            self.motion_requests.put_nowait(None)
        except queue.Full:
            try:
                self.motion_requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self.motion_requests.put_nowait(None)
            except queue.Full:
                pass
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        # A new osc session must never inherit responses from the previous
        # anchor/epoch.  This is especially important after the hard-reset
        # recovery path, where the controller worker may survive while its
        # Pink child still has an older request in flight.
        with self.lock:
            self._solver_request_id = 0
            self._last_response_request_id = 0
            self._last_response_seen_request_id = 0
            self._pending_solver_response = None
            self._latest_motion_response = None
            self._debug = {
                "offered": 0,
                "submitted": 0,
                "accepted": 0,
                "discarded": 0,
                "last_submit": None,
                "last_response": None,
                "last_discard": None,
                "reader_alive": False,
                "reader_error": None,
            }
            for channel in (self.responses, self.ready, self.fk_responses, self.motion_requests):
                while True:
                    try:
                        channel.get_nowait()
                    except queue.Empty:
                        break

    def discard_before_epoch(self, epoch: int) -> None:
        with self.lock:
            self.minimum_epoch = max(self.minimum_epoch, int(epoch))
            self._latest_motion_response = None
            for channel in (self.responses, self.ready, self.fk_responses, self.motion_requests):
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
                    with self._response_condition:
                        self._latest_motion_response = payload
                        self._response_condition.notify_all()
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
            process = self.process
            stdin = process.stdin
            if is_motion:
                self._debug["offered"] = int(self._debug["offered"]) + 1
        # Motion is a one-slot latest-value mailbox. The persistent writer is
        # the only thread allowed to assign request IDs or touch solver stdin,
        # so an overwritten request never masquerades as submitted work.
        if is_motion:
            request = dict(payload)
            try:
                self.motion_requests.put_nowait(request)
            except queue.Full:
                try:
                    self.motion_requests.get_nowait()
                except queue.Empty:
                    pass
                self.motion_requests.put_nowait(request)
            return
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        stdin.write(line)
        stdin.flush()

    def _motion_writer(self, process: subprocess.Popen[str]) -> None:
        stdin = process.stdin
        if stdin is None:
            return
        while self.process is process and process.poll() is None:
            try:
                payload = self.motion_requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None or self.process is not process:
                return
            with self.lock:
                self._solver_request_id += 1
                request_id = self._solver_request_id
                payload = dict(payload)
                payload["solver_request_id"] = request_id
                payload["osc_pink_written_monotonic_ns"] = time.monotonic_ns()
                self._debug["submitted"] = int(self._debug["submitted"]) + 1
                self._debug["last_submit"] = {
                    "solver_request_id": request_id,
                    "control_sample_id": payload.get("control_sample_id"),
                    "motion_epoch": payload.get("motion_epoch"),
                    "target_generation": payload.get("target_generation"),
                    "sequence": payload.get("sequence"),
                }
            try:
                stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                stdin.flush()
            except (OSError, ValueError) as exc:
                with self.lock:
                    self._debug["reader_error"] = f"solver request {request_id} write failed: {exc}"
                return

    def solve_current(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        """Return only the response for this exact control sample and target."""
        payload = dict(payload)
        payload["osc_pink_requested_monotonic_ns"] = time.monotonic_ns()
        expected_sample = int(payload.get("control_sample_id", -1))
        expected_epoch = int(payload.get("motion_epoch", -1))
        expected_generation = int(payload.get("target_generation", -1))
        self.submit(payload)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._response_condition:
            while True:
                candidate = self._latest_motion_response
                if candidate is not None:
                    matches = (
                        int(candidate.get("control_sample_id", -2)) == expected_sample
                        and int(candidate.get("motion_epoch", -2)) == expected_epoch
                        and int(candidate.get("target_generation", -2)) == expected_generation
                    )
                    if matches:
                        self._latest_motion_response = None
                        self._debug["accepted"] = int(self._debug["accepted"]) + 1
                        self._last_response_request_id = int(candidate.get("solver_request_id", 0) or 0)
                        return candidate
                    if int(candidate.get("control_sample_id", -2)) <= expected_sample:
                        self._latest_motion_response = None
                        self._debug["discarded"] = int(self._debug["discarded"]) + 1
                        self._debug["last_discard"] = {
                            "solver_request_id": candidate.get("solver_request_id"),
                            "control_sample_id": candidate.get("control_sample_id"),
                            "motion_epoch": candidate.get("motion_epoch"),
                            "target_generation": candidate.get("target_generation"),
                        }
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._response_condition.wait(timeout=min(remaining, 0.001))

    def poll(self, epoch: int, target_generation: int | None = None, max_target_lag: int = 1) -> dict[str, Any] | None:
        # A continuously updated target normally receives its Pink result one
        # control tick later.  Permit that bounded lag, but reject an older
        # result so a hold can never continue integrating a trajectory-era
        # velocity batch.
        max_target_lag = max(0, int(max_target_lag))
        latest = self._pending_solver_response
        self._pending_solver_response = None
        if latest is not None:
            if (
                int(latest.get("motion_epoch", -1)) == int(epoch)
                and (target_generation is None or int(latest.get("target_generation", -1)) >= int(target_generation) - max_target_lag)
                and (target_generation is None or int(latest.get("target_generation", -1)) <= int(target_generation))
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
                "target_generation": candidate.get("target_generation"),
            }
            if (
                int(candidate.get("motion_epoch", -1)) == int(epoch)
                and request_id <= self._solver_request_id
                and request_id > self._last_response_request_id
                and (target_generation is None or int(candidate.get("target_generation", -1)) >= int(target_generation) - max_target_lag)
                and (target_generation is None or int(candidate.get("target_generation", -1)) <= int(target_generation))
            ):
                latest = candidate
                self._last_response_request_id = request_id
                self._debug["accepted"] = int(self._debug["accepted"]) + 1
            else:
                self._debug["discarded"] = int(self._debug["discarded"]) + 1
                self._debug["last_discard"] = dict(self._debug["last_response"])
        return latest

    def poll_until(
        self,
        epoch: int,
        target_generation: int | None = None,
        max_target_lag: int = 0,
        timeout_s: float = 0.0,
    ) -> dict[str, Any] | None:
        """Wait briefly for a fresh Pink response without extending a servo tick.

        A target sent at 50 Hz must not be driven by a several-cycle-old IK
        result: that creates an avoidable Cartesian lag and, once the target
        stops changing, a visible joint-space catch-up.  The solver normally
        completes in a small fraction of the 20 ms period, so wait only for a
        bounded budget and let the caller safely hold if it is unavailable.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            result = self.poll(epoch, target_generation, max_target_lag)
            if result is not None or time.monotonic() >= deadline:
                return result
            time.sleep(0.0005)

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

    def initialize_fixed(self) -> dict[str, Any]:
        """Load the commissioned NERO envelope; never query indexed CAN limits."""
        fixed = self.config.get("hardware_limits") or {
            "lower_rad": [-2.705260340591211, -1.7453292519943295, -2.7576202181510405, -1.0122909661567112, -2.7576202181510405, -0.7330382858376184, -1.5707963267948966],
            "upper_rad": [2.705260340591211, 1.7453292519943295, 2.7576202181510405, 2.1467549799530254, 2.7576202181510405, 0.9599310885968813, 1.5707963267948966],
            "speed_rad_s": [3.14, 3.14, 3.14, 3.14, 3.92, 3.92, 3.92],
            "acceleration_rad_s2": [5.0] * 7,
        }
        lower = _finite_vector(fixed.get("lower_rad"), 7, "hardware_limits.lower_rad")
        upper = _finite_vector(fixed.get("upper_rad"), 7, "hardware_limits.upper_rad")
        speed = _finite_vector(fixed.get("speed_rad_s"), 7, "hardware_limits.speed_rad_s")
        acceleration = _finite_vector(fixed.get("acceleration_rad_s2"), 7, "hardware_limits.acceleration_rad_s2")
        effective_lower = [max(urdf, configured) for urdf, configured in zip(self.hard_lower, lower)]
        effective_upper = [min(urdf, configured) for urdf, configured in zip(self.hard_upper, upper)]
        if any(a >= b for a, b in zip(effective_lower, effective_upper)):
            raise ValueError("configured hardware limits do not intersect the URDF")
        self.effective = {
            "status": "ready", "source": "configured NERO hardware envelope",
            "urdf_lower_rad": self.hard_lower, "urdf_upper_rad": self.hard_upper,
            "effective_lower_rad": effective_lower, "effective_upper_rad": effective_upper,
            "controller_speed_rad_s": speed, "controller_acceleration_rad_s2": acceleration,
        }
        return dict(self.effective)

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

class ShadowCpvPlant:
    """Deterministic CPV/feedback surrogate used only by shadow OSC sessions.

    Shadow used to promote a requested position to feedback in the same cycle.
    This makes shadow exercise an actuator queue, bounded acceleration and
    delayed feedback without accessing CAN.
    """

    def __init__(self, config: dict[str, Any], initial_q: list[float]) -> None:
        self.config = config
        self.q = list(initial_q)
        self.qd = [0.0] * 7
        self.target = list(initial_q)
        self._dispatches: deque[tuple[float, list[float]]] = deque()
        self._feedback: deque[tuple[float, list[float], list[float]]] = deque()
        self._last_feedback = (list(initial_q), [0.0] * 7, 0.0)
        self.output_count = 0
        self._jitter_index = 0

    def _jitter(self, key: str) -> float:
        magnitude = max(0.0, float(self.config.get(key, 0.0)))
        self._jitter_index += 1
        return magnitude if self._jitter_index % 2 else -magnitude

    def dispatch(self, target: list[float], now_s: float) -> None:
        delay = max(0.0, float(self.config.get("dispatch_delay_s", 0.0)) + self._jitter("dispatch_jitter_s"))
        self._dispatches.append((now_s + delay, list(target)))
        self.output_count += 1

    def advance(self, dt_s: float, now_s: float) -> tuple[list[float], list[float], float]:
        while self._dispatches and self._dispatches[0][0] <= now_s:
            _, self.target = self._dispatches.popleft()
        tau = max(0.001, float(self.config.get("position_time_constant_s", 0.06)))
        max_speed = max(0.01, float(self.config.get("max_joint_speed_rad_s", 1.5)))
        max_acc = max(0.01, float(self.config.get("max_joint_acceleration_rad_s2", 5.0)))
        desired = [max(-max_speed, min(max_speed, (target - value) / tau)) for target, value in zip(self.target, self.q)]
        max_delta = max_acc * max(0.001, dt_s)
        self.qd = [max(old - max_delta, min(old + max_delta, want)) for old, want in zip(self.qd, desired)]
        self.q = [value + velocity * dt_s for value, velocity in zip(self.q, self.qd)]
        feedback_delay = max(0.0, float(self.config.get("feedback_delay_s", 0.03)) + self._jitter("feedback_jitter_s"))
        self._feedback.append((now_s + feedback_delay, list(self.q), list(self.qd)))
        while self._feedback and self._feedback[0][0] <= now_s:
            _, q, qd = self._feedback.popleft()
            self._last_feedback = (q, qd, feedback_delay)
        q, qd, age = self._last_feedback
        return list(q), list(qd), float(age)

    def diagnostics(self) -> dict[str, Any]:
        return {"enabled": True, "applied_joint_state_rad": list(self.q),
                "applied_joint_velocity_rad_s": list(self.qd), "active_target_rad": list(self.target),
                "queued_dispatches": len(self._dispatches), "queued_feedback": len(self._feedback),
                "output_count": self.output_count, "config": dict(self.config)}


class _OperationalSpaceServo:
    """Fixed-rate operational-space servo: Pink, Ruckig and safety sync.

    This is the sole Pink/Ruckig/CPV loop inside the OSC runtime.
    """

    def __init__(self, hardware: OscHardwarePort, project_root: Path, config: dict[str, Any], feedback_receiver: _OscFeedbackReceiver) -> None:
        try:
            import ruckig
        except ImportError as exc:
            raise RuntimeError("OSC requires ruckig in the control-service environment") from exc
        self.ruckig = ruckig
        self.hardware, self.root, self.config = hardware, project_root, config
        self.limits, self.runtime = config.get("limits", {}), config.get("runtime", {})
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
        self._accepting_targets = False
        self._heartbeat_monotonic_ns = 0
        self.command: dict[str, Any] | None = None
        self.target_generation = 0
        self._target_pose: dict[str, list[float]] | None = None
        self.motion_epoch = 0
        self.last_sent_velocity = [0.0] * 7
        self.trajectory: dict[str, list[float]] | None = None
        self.ruckig_input = None
        self.ruckig_output = None
        self.ruckig_otg = None
        self.ruckig_period_s: float | None = None
        self.posture_reference: list[float] | None = None
        self.shadow_joints: list[float] | None = None
        self.shadow_plant: ShadowCpvPlant | None = None
        self.last_solver_result: dict[str, Any] | None = None
        self._solver_reuse_count = 0
        self.control_sample_id = 0
        self.execution_sample: dict[str, Any] | None = None
        self._target_changed_monotonic_ns = 0
        self._arrival_since_monotonic_ns = 0
        self._arrival_reached = False
        self._last_dispatch_monotonic_ns = 0
        self.last_result: dict[str, Any] = {"ok": False, "reason": "not started"}
        self.trajectory_state, self.trajectory_brake_reason = "HOLD_READY", None
        self.feedback_sync_pending = True
        self.needs_resync = True
        self._feedback_receiver = feedback_receiver
        self._control_q_estimate: list[float] | None = None
        self._control_qd_estimate: list[float] = [0.0] * 7
        self._estimator_snapshot: dict[str, Any] = {}
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
        self._cpv_parameters: dict[str, Any] = {"status": "not_read"}
        self._timing: dict[str, Any] = {}
        diagnostics = config.get("diagnostics", {})
        self._cycle_trace: deque[dict[str, Any]] = deque(maxlen=max(64, int(diagnostics.get("cycle_trace_capacity", 512))))
        self._cycle_period_s = 1.0 / max(1.0, float(self.runtime.get("control_hz", 50)))
        self._enable_high_resolution_timer()

    @staticmethod
    def _percentile(values: list[float], level: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        return values[min(len(values) - 1, max(0, math.ceil(len(values) * level) - 1))]

    def _trace_public(self) -> dict[str, Any]:
        rows = list(self._cycle_trace)
        period = [float(row["actual_dt_s"]) * 1000 for row in rows if isinstance(row.get("actual_dt_s"), (int, float))]
        duration = [float(row["cycle_duration_ms"]) for row in rows if isinstance(row.get("cycle_duration_ms"), (int, float))]
        tail = max(1, int(self.config.get("diagnostics", {}).get("cycle_trace_tail", 32)))
        return {"target_hz": 1.0 / self._cycle_period_s, "target_period_ms": self._cycle_period_s * 1000,
                "sample_count": len(rows), "deadline_miss_count": sum(float(row.get("deadline_overrun_ms") or 0.0) > 0 for row in rows),
                "actual_period_p50_ms": self._percentile(period, .5), "actual_period_p95_ms": self._percentile(period, .95),
                "cycle_duration_p50_ms": self._percentile(duration, .5), "cycle_duration_p95_ms": self._percentile(duration, .95),
                "recent_cycles": rows[-tail:]}

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
            "execution_mode": session.get("execution_mode"),
            "sequence": session.get("sequence", 0),
            "last_input_age_s": session.get("last_input_age_s"),
        }

    def _current_tcp_pose(self, joints: list[float]) -> dict[str, list[float]]:
        return pose_from_tcp(self.solver.fk(joints))

    def _pose_in_workspace(self, pose: dict[str, list[float]]) -> bool:
        position = pose["position_m"]
        lower = _finite_vector(self.limits.get("workspace_min_m", [-0.45, -0.45, -0.01]), 3, "workspace_min_m")
        upper = _finite_vector(self.limits.get("workspace_max_m", [0.45, 0.60, 0.70]), 3, "workspace_max_m")
        for index, (item, low, high) in enumerate(zip(position, lower, upper)):
            if not low <= item <= high:
                return False
        min_tcp_z = float(self.limits.get("min_tcp_z_m", lower[2]))
        if position[2] < min_tcp_z:
            return False
        return True

    def _invalidate_motion(self, reason: str) -> None:
        # This is a P1 braking request, not an ownership transfer. The active
        # epoch remains valid long enough for Ruckig to transmit the braking
        # curve. Ownership transfers use freeze_for_authority_change().
        session = dict(self.session or {})
        if session.get("execution_mode") != "shadow":
            # This method is also called while ``self.lock`` is held by the
            # servo loop and the HTTP intent handler.  Authority transitions
            # acquire supervisor/transport locks and can be called back by
            # status/control paths that need the osc lock.  Never wait for
            # that lock cycle from inside the input critical section: a stale
            # joystick packet must brake, not make the next packet time out.
            threading.Thread(
                target=self.hardware.mark_osc_stopping,
                args=(str(session.get("session_id", "unknown")), self.motion_epoch, reason),
                name="osc-stop-authority",
                daemon=True,
            ).start()
        self.command = None
        self.target_generation += 1
        self._target_pose = None
        self.trajectory_state = "BRAKING"
        self.trajectory_brake_reason = reason

    def freeze_for_authority_change(self, epoch: int, reason: str) -> None:
        """Discard old asynchronous work without advancing the trajectory."""
        with self.lock:
            self.motion_epoch = int(epoch)
            self.solver.discard_before_epoch(self.motion_epoch)
            if self.session:
                self.session["motion_epoch"] = self.motion_epoch
            self.command = None
            self.target_generation += 1
            self._accepting_targets = False
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
        return self.authority.initialize_fixed()

    def start_session(
        self,
        client_id: str = "anonymous",
        execution_mode: str = "shadow",
    ) -> dict[str, Any]:
        execution_mode = str(execution_mode).strip().lower()
        if execution_mode not in {"shadow", "hardware"}:
            raise ValueError("execution_mode must be shadow or hardware")
        client_id = str(client_id).strip() or "anonymous"
        with self._session_transition_lock:
            session_id = uuid.uuid4().hex
            with self.lock:
                active = self._session_active()
                active_client = str((self.session or {}).get("client_id", "anonymous"))
                heartbeat_age = (time.monotonic_ns() - self._heartbeat_monotonic_ns) / 1e9 if self._heartbeat_monotonic_ns else float("inf")
            if active and heartbeat_age > float(self.config.get("session_timeout_s", 5.0)):
                self.stop_session("OSC session heartbeat expired")
                active = False
            if active and active_client != client_id:
                raise PermissionError("OSC session is owned by another client")
            if active:
                return self.status()
            self.last_error = None
            self.session = {"state": "STARTING", "session_id": None, "client_id": client_id, "execution_mode": execution_mode, "sequence": 0}
            self._bump_state()
            try:
                if execution_mode == "shadow":
                    authority = {"status": "shadow", "effective_lower_rad": self.authority.hard_lower, "effective_upper_rad": self.authority.hard_upper, "controller_speed_rad_s": [float(self.limits.get("joint_speed_rad_s", 1.5))] * 7, "controller_acceleration_rad_s2": [float(self.config.get("solver", {}).get("ruckig_max_acceleration", 5.0))] * 7}
                    self.authority.effective = authority
                else:
                    self.hardware.require_operational_control()
                    authority = self._hardware_preflight()
                self.supervisor.configure(authority, [float(self.limits.get("joint_speed_rad_s", 1.5))] * 7, [float(self.config.get("solver", {}).get("ruckig_max_acceleration", 5.0))] * 7)
                # Always create a fresh Pink bridge for a fresh session.  A
                # previous hard reset can leave a solver child alive with a
                # stale anchor; merely changing motion_epoch cannot make that
                # response safe for the new clutch.
                self.solver.close()
                self.solver.start()
                if execution_mode != "shadow":
                    feedback_revision = self._feedback_receiver.revision()
                    hardware_authority = self.hardware.prepare_osc_hardware()
                    # RX runs for the whole OSC runtime. Require a sample
                    # published after hardware preparation before arming CPV.
                    feedback = self._feedback_receiver.wait_for_revision_after(
                        feedback_revision,
                        float(self.config.get("feedback_start_timeout_s", 2.0)),
                    )
                    joints = list(feedback["joints"])
                    self._cpv_parameters = {"status": "not_part_of_osc_runtime"}
                else:
                    joints = _finite_vector(self.config.get("shadow_initial_joints_rad", [0.0] * 7), 7, "shadow_initial_joints_rad")
            except Exception as exc:
                self.solver.close()
                with self.lock:
                    self.session = None
                    self.command = None
                    self._accepting_targets = False
                    self.last_error = f"osc start failed: {type(exc).__name__}: {exc}"
                    self.last_result = {"ok": False, "reason": self.last_error, "timestamp": time.time(), "robot_commands_sent": False}
                    self._bump_state()
                raise
            period = 1.0 / float(self.runtime.get("control_hz", 50))
            if execution_mode != "shadow" and not self.hardware.grant_osc_tracking(session_id, int(hardware_authority["control_epoch"])):
                self.solver.close()
                with self.lock:
                    self.session = None
                    self.command = None
                    self._accepting_targets = False
                    self.last_error = "osc tracking authority could not be granted"
                    self.last_result = {"ok": False, "reason": self.last_error, "timestamp": time.time(), "robot_commands_sent": False}
                    self._bump_state()
                raise RuntimeError("osc tracking authority could not be granted")
            with self.lock:
                self._initialize_ruckig([float(x) for x in joints], period)
                self.posture_reference = [float(x) for x in joints]
                self.shadow_joints = list(self.posture_reference)
                shadow_config = dict(self.config.get("shadow_transport") or {})
                self.shadow_plant = (
                    ShadowCpvPlant(shadow_config, self.posture_reference)
                    if execution_mode == "shadow" and bool(shadow_config.get("enabled", True)) else None
                )
                self.last_solver_result = None
                self._solver_reuse_count = 0
                self.control_sample_id = 0
                self.execution_sample = None
                self._target_changed_monotonic_ns = time.monotonic_ns()
                self._arrival_since_monotonic_ns = 0
                self._arrival_reached = False
                self._last_dispatch_monotonic_ns = 0
                self._control_q_estimate = list(joints)
                self._control_qd_estimate = [0.0] * 7
                self._estimator_snapshot = {"estimated_joint_state_rad": list(joints), "estimated_joint_velocity_rad_s": [0.0] * 7,
                                            "measurement_error_rad": [0.0] * 7, "correction_rad": [0.0] * 7}
                self.motion_epoch = int(hardware_authority["control_epoch"]) if execution_mode != "shadow" else self.motion_epoch
                self.solver.discard_before_epoch(self.motion_epoch)
                self.session = {"state": "ACTIVE", "session_id": session_id, "client_id": client_id, "execution_mode": execution_mode, "started_at": time.time(), "sequence": 0, "last_input_age_s": None, "motion_epoch": self.motion_epoch}
                self.command = None
                self.target_generation += 1
                self._target_pose = self._current_tcp_pose([float(x) for x in joints])
                self._accepting_targets = True
                self._heartbeat_monotonic_ns = time.monotonic_ns()
                self.trajectory_state, self.trajectory_brake_reason, self.feedback_sync_pending = "HOLD_READY", None, True
                self.needs_resync = True
                self.stop_event.clear()
                self._send_times.clear(); self._batch_history.clear(); self.cpv_send_count = 0; self.output_count = 0; self.loop_count = 0
                self._cycle_trace.clear()
                self.last_output = {"status": "held", "final_joint_target_rad": list(joints), "final_joint_velocity_rad_s": [0.0] * 7, "sequence": 0, "epoch": self.motion_epoch}
                self.last_result = {"ok": True, "reason": "session started", "robot_commands_sent": False}
                self.thread = threading.Thread(target=self._loop, name="nero-osc-loop", daemon=True)
                self.thread.start()
                self._bump_state()
            return self.status()

    def stop_session(self, reason: str = "osc stopped") -> dict[str, Any]:
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
            # A remembered hardware session is not proof that CPV still owns
            # the controller.  Sending a CPV sample after HOLD would initialise
            # CPV again, creating an unnecessary mode transition.
            cpv_stream_active = bool(
            (self.session or {}).get("execution_mode") != "shadow" and self.hardware.osc_stream_active()
            )
            if self.session:
                self.session.update({"state": "STOPPING", "stopped_reason": reason})
            self._accepting_targets = False
            self._bump_state()
            self.stop_event.set(); self.command = None
        servo_thread = self.thread
        if servo_thread and servo_thread is not threading.current_thread(): servo_thread.join(timeout=1.0)
        servo_stopped = not servo_thread or not servo_thread.is_alive()
        if servo_stopped:
            self.thread = None
        # The broker owns the terminal zero/hold transition.  Sending another
        # CPV sample here could reopen CPV after the mode handoff.
        if servo_stopped:
            self.solver.close()
        with self.lock:
            self.session = None
            self.last_error = None
            self._accepting_targets = False
            self._bump_state()
        result = self.status()
        result["handoff"] = {
            "reason": reason,
            "active_session": active,
            "servo_stopped": servo_stopped,
            "feedback_stopped": False,
            "rx_running": bool(self._feedback_receiver.snapshot() and self._feedback_receiver.snapshot().get("running")),
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
            execution_mode = session.get("execution_mode")
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
            self.command = None
            self._accepting_targets = False
            self.feedback_sync_pending = True
            self.needs_resync = True
            self.trajectory_state = "HOLD_READY"
            self.trajectory_brake_reason = reason
            self.stop_event.set()
            self._bump_state()
            servo_thread = self.thread

        if servo_thread and servo_thread is not threading.current_thread():
            servo_thread.join(timeout=1.0)
        servo_stopped = not servo_thread or not servo_thread.is_alive()
        if servo_stopped:
            self.thread = None
        if servo_stopped:
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
            "feedback_stopped": False,
            "rx_running": bool(self._feedback_receiver.snapshot() and self._feedback_receiver.snapshot().get("running")),
            "threads_stopped": servo_stopped,
            "braking_requested": False,
            "zero_velocity_sent": False,
        }
        return result

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
                # A continuous input source can legitimately overrun its
                # local integration range. This is a command-level safety
                # rejection, not an OSC fault and not a reason to revoke the
                # session. Return the last safe absolute target so adapters
                # can re-anchor instead of endlessly resending the same pose.
                fallback_q = (
                    list(self.shadow_joints or [])
                    if self.session.get("execution_mode") == "shadow"
                    else list((self._feedback_snapshot() or {}).get("joints") or [])
                )
                safe_target = dict(self._target_pose or self._current_tcp_pose(fallback_q))
                return {
                    "accepted": False,
                    "reason": "OSC target is outside the configured workspace",
                    "accepted_sequence": int(self.session.get("sequence", 0)),
                    "session_id": self.session["session_id"],
                    "safe_target_pose": safe_target,
                    "recoverable": True,
                }
            if self.trajectory_state == "HOLD_READY":
                resumed, reason = self._resume_from_hold_locked()
                if not resumed:
                    raise PermissionError(f"OSC command rejected: cannot resume from HOLD_READY ({reason})")
            previous_reference = self._target_pose or {}
            previous_position = previous_reference.get("position_m") or []
            previous_orientation = previous_reference.get("orientation_xyzw") or []
            same_reference = (
                len(previous_position) == 3
                and len(previous_orientation) == 4
                and max(abs(float(current) - float(previous)) for current, previous in zip(reference["position_m"], previous_position)) <= 1e-9
                and abs(sum(float(current) * float(previous) for current, previous in zip(reference["orientation_xyzw"], previous_orientation))) >= 1.0 - 1e-12
            )
            self.session["sequence"] = sequence
            now_ns = time.monotonic_ns()
            self._target_pose = reference
            if not same_reference:
                self.target_generation += 1
                self._target_changed_monotonic_ns = now_ns
                self._arrival_since_monotonic_ns = 0
                self._arrival_reached = False
            self.command = {
                "sequence": sequence,
                "host_monotonic_ns": time.monotonic_ns(),
                "target_generation": self.target_generation,
                "target_pose": dict(reference),
                "osc_mode": mode,
            }
            self._bump_state()
            return {
                "accepted": True,
                "accepted_sequence": sequence,
                "session_id": self.session["session_id"],
                "target_pose": reference,
                "mode": mode,
            }

    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        with self.lock:
            if not self._session_active() or str(self.session.get("client_id")) != str(client_id) or str(self.session.get("session_id")) != str(session_id):
                raise PermissionError("osc heartbeat rejected")
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

    def _feedback_snapshot(self) -> dict[str, Any] | None:
        return self._feedback_receiver.snapshot()

    def _estimate_hardware_state(self, feedback: dict[str, Any], actual_dt: float) -> tuple[list[float], list[float], float]:
        """One continuous predictor/corrector; never threshold-reset Pink state."""
        now_ns = time.monotonic_ns()
        conservative_ns = int(feedback.get("fresh_received_at_monotonic_ns") or feedback.get("monotonic_ns") or now_ns)
        feedback_age = max(0.0, (now_ns - conservative_ns) / 1e9)
        measured_q = [float(value) for value in feedback.get("joints") or []]
        measured_qd = [float(value) for value in feedback.get("velocities") or []]
        if len(measured_q) != 7 or len(measured_qd) != 7:
            return measured_q, measured_qd, feedback_age
        if self._control_q_estimate is None:
            self._control_q_estimate = list(measured_q)
        predicted = [
            float(position) + float(velocity) * actual_dt
            for position, velocity in zip(self._control_q_estimate, self.last_sent_velocity)
        ]
        horizon = min(feedback_age, float(self.config.get("state_estimator", {}).get("max_prediction_s", 0.15)))
        compensated = [position + velocity * horizon for position, velocity in zip(measured_q, measured_qd)]
        tau = max(0.02, float(self.config.get("state_estimator", {}).get("correction_time_constant_s", 0.20)))
        alpha = 1.0 - math.exp(-actual_dt / tau)
        max_rate = max(0.01, float(self.config.get("state_estimator", {}).get("max_correction_rad_s", 0.50)))
        max_step = max_rate * actual_dt
        correction = [max(-max_step, min(max_step, alpha * (measurement - estimate))) for measurement, estimate in zip(compensated, predicted)]
        self._control_q_estimate = [estimate + delta for estimate, delta in zip(predicted, correction)]
        self._control_qd_estimate = [float(value) + delta / max(0.001, actual_dt) for value, delta in zip(self.last_sent_velocity, correction)]
        self._estimator_snapshot = {
            "measured_joint_state_rad": list(measured_q),
            "measured_joint_velocity_rad_s": list(measured_qd),
            "estimated_joint_state_rad": list(self._control_q_estimate),
            "estimated_joint_velocity_rad_s": list(self._control_qd_estimate),
            "measurement_error_rad": [measurement - estimate for measurement, estimate in zip(compensated, predicted)],
            "correction_rad": list(correction),
            "correction_time_constant_s": tau,
            "max_correction_rad_s": max_rate,
            "feedback_age_s": feedback_age,
        }
        return list(self._control_q_estimate), list(self._control_qd_estimate), feedback_age

    def _wait_for_feedback(self, timeout_s: float = 2.0) -> dict[str, Any]:
        return self._feedback_receiver.wait_for_revision_after(
            self._feedback_receiver.revision() - 1, timeout_s
        )

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
            return {"state_sequence": self.state_sequence, "session": self._session_view(), "command": dict(self.command) if self.command else None, "target_generation": self.target_generation, "last_error": self.last_error, "last_result": dict(self.last_result), "last_output": dict(self.last_output), "execution_sample": dict(self.execution_sample) if self.execution_sample else None, "arrival": {"reached": self._arrival_reached, "stable_since_monotonic_ns": self._arrival_since_monotonic_ns or None, "target_generation": self.target_generation}, "solver": {"running": bool(self.solver.process and self.solver.process.poll() is None), "python": str(self.solver.python), "tcp_verified": True, "debug": debug}, "workspace": {"min_xyz_m": list(self.limits.get("workspace_min_m", [-0.45, -0.15, -0.02])), "max_xyz_m": list(self.limits.get("workspace_max_m", [0.45, 0.60, 0.70])), "min_tcp_z_m": float(self.limits.get("min_tcp_z_m", -0.02))}, "diagnostics": {"trajectory_state": self.trajectory_state, "trajectory_brake_reason": self.trajectory_brake_reason, "motion_epoch": self.motion_epoch, "needs_resync": self.needs_resync, "last_sent_velocity_rad_s": list(self.last_sent_velocity), "trajectory": dict(self.trajectory) if self.trajectory else None, "state_estimator": dict(self._estimator_snapshot), "shadow_transport": self.shadow_plant.diagnostics() if self.shadow_plant else {"enabled": False}, "cpv_parameters": dict(self._cpv_parameters), "limit_authority": dict(self.authority.effective) if self.authority.effective else None, "supervisor": dict(self.supervisor.limit_data) if self.supervisor.limit_data else None, "timing": dict(self._timing), "cycle_trace": self._trace_public(), "loop_count": self.loop_count, "output_count": self.output_count, "cpv_dispatch_count": self.cpv_send_count, "recent_cpv_batches": list(self._batch_history)[-10:]}}

    def _public_target_pose(self) -> dict[str, list[float]] | None:
        with self.lock:
            return dict(self._target_pose) if self._target_pose else None

    def _is_accepting_targets(self) -> bool:
        with self.lock:
            return self._accepting_targets

    def kinematics(self) -> dict[str, Any]:
        return {"schema_version": "nero.osc.kinematics.v1", "tcp_verified": True, "tcp_offset_m": self.config.get("tcp", {}).get("offset_from_link7_m"), "last_result": dict(self.last_result), "shadow_default": True}

    def _sync_if_settled(self, feedback_q: list[float]) -> bool:
        if self.trajectory_state != "HOLD_READY": return False
        self._initialize_ruckig(feedback_q, 1.0 / float(self.runtime.get("control_hz", 50)))
        # The posture objective is fixed for the complete OSC session. HOLD
        # and resume must not manufacture null-space stability by re-anchoring.
        if self.posture_reference is None:
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
            if not self.hardware.servo_can_write(str(session.get("session_id")), self.motion_epoch):
                return False, "SERVO write authority is unavailable"
            if not self.hardware.grant_osc_tracking(str(session.get("session_id")), self.motion_epoch):
                return False, "SERVO tracking authority could not be restored"
        else:
            q = list(self.shadow_joints or (self.trajectory or {}).get("position_rad") or [])
            if len(q) != 7 or not all(math.isfinite(float(x)) for x in q):
                return False, "shadow state is invalid"
        self._sync_if_settled([float(x) for x in q])
        self._accepting_targets = True
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
                    session = dict(self.session) if self.session else None; command = dict(self.command) if self.command else None; self.loop_count += 1
                if not session or session.get("state") != "ACTIVE": continue
                shadow = session.get("execution_mode") == "shadow"
                feedback = None if shadow else self._feedback_snapshot()
                if shadow:
                    if self.shadow_plant is not None:
                        q, qd, feedback_age = self.shadow_plant.advance(actual_dt, time.monotonic())
                    else:
                        q = list(self.shadow_joints or (self.trajectory or {}).get("position_rad") or [])
                        qd = list((self.trajectory or {}).get("velocity_rad_s") or [0.0] * 7)
                        feedback_age = 0.0
                else:
                    if feedback:
                        q, qd, feedback_age = self._estimate_hardware_state(feedback, actual_dt)
                    else:
                        q, qd, feedback_age = [], [], float("inf")
                if len(q) != 7 or len(qd) != 7:
                    self._fault_zero("seven-joint feedback unavailable", shadow); continue
                measured_q = list(q) if shadow else list((feedback or {}).get("joints") or [])
                if len(measured_q) != 7:
                    self._fault_zero("seven-joint measured feedback unavailable", shadow); continue
                soft_stale = feedback_age > float(self.limits.get("feedback_soft_stale_s", 0.06))
                hard_stale = feedback_age > float(self.limits.get("feedback_hard_stale_s", 0.15))
                age = float("inf") if not command else max(0.0, (time.monotonic_ns() - int(command["host_monotonic_ns"])) / 1e9)
                with self.lock:
                    if self.session: self.session["last_input_age_s"] = age if math.isfinite(age) else None
                    command_active = bool(command)
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
                        self._invalidate_motion("OSC session heartbeat expired")
                    # A stationary session has no active motion to protect.
                    # Keep an old sample visible while HOLD_READY; the next
                    # nonzero input still performs its own freshness check.
                    motion_or_brake_active = (
                        brake_is_still_moving
                        or (state_before_input_check == "RUNNING" and command_active)
                    )
                    if hard_stale and motion_or_brake_active:
                        self.trajectory_state, self.trajectory_brake_reason = "FAULT", "feedback hard stale"
                        self._accepting_targets = False
                    state = self.trajectory_state; epoch = self.motion_epoch
                # HOLD_READY is a frozen state until a fresh absolute OSC target arrives.
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
                        "HOLD_READY; feedback temporarily stale; waiting for fresh target"
                        if hard_stale else "HOLD_READY; waiting for fresh target"
                    )
                    self._set_result(False, reason, robot_commands_sent=False)
                    continue
                target_pose = dict((command or {}).get("target_pose") or self._target_pose or {})
                if state != "RUNNING" or not target_pose:
                    # Braking must not issue a second asynchronous FK request
                    # while session shutdown is closing the Pink bridge. The
                    # latest coherent execution sample is exactly the pose
                    # associated with q; fall back to a harmless identity
                    # target only before the first running sample.
                    target_pose = dict((self.execution_sample or {}).get("measured_tcp_pose") or {
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    })
                # A live P1 thread without SERVO authority is deliberately
                # frozen. It must not integrate old targets and later chase
                # them when a mode transition returns control to osc.
                if not shadow and not self.hardware.servo_can_write(str(session.get("session_id")), epoch):
                    with self.lock:
                        self.feedback_sync_pending = True
                        self.trajectory_state = "HOLD_READY"
                    self._set_result(False, "servo authority unavailable; trajectory frozen", robot_commands_sent=False)
                    continue
                data = self.supervisor._require()
                cycle_started_perf_ns = time.perf_counter_ns()
                target_generation = int((command or {}).get("target_generation", self.target_generation))
                self.control_sample_id += 1
                sample_id = self.control_sample_id
                now_ns = time.monotonic_ns()
                dispatch_dt = actual_dt
                if not shadow and self._last_dispatch_monotonic_ns:
                    dispatch_dt = max(0.001, min(0.2, (now_ns - self._last_dispatch_monotonic_ns) / 1e9))
                # OSC accepts an absolute TCP setpoint.  Never extrapolate a
                # jittery, paused, or discontinuous adapter stream beyond
                # that setpoint; Pink receives exactly what the adapter sent.
                request = {"sequence": int((command or {}).get("sequence", session.get("sequence", 0))), "control_sample_id": sample_id, "target_generation": target_generation, "motion_epoch": epoch, "joint_angles_rad": q, "measured_joint_angles_rad": measured_q, "joint_state_monotonic_ns": now_ns, "target_position_m": target_pose["position_m"], "target_orientation_xyzw": target_pose["orientation_xyzw"], "command_target_position_m": target_pose["position_m"], "command_target_orientation_xyzw": target_pose["orientation_xyzw"], "last_sent_joint_velocity_rad_s": self.last_sent_velocity, "joint_speed_limit_rad_s": data["speed_rad_s"], "joint_acceleration_limit_rad_s2": data["acceleration_rad_s2"], "soft_lower_rad": data["soft_lower_rad"], "soft_upper_rad": data["soft_upper_rad"], "posture_reference_rad": self.posture_reference or q, "posture_cost": float(self.config.get("solver", {}).get("posture_cost", 0.005)), "damping_cost": float(self.config.get("solver", {}).get("damping_cost", 0.05)), "frame_position_cost": float(self.config.get("solver", {}).get("frame_position_cost", 10.0)), "frame_orientation_cost": float(self.config.get("solver", {}).get("frame_orientation_cost", 1.0)), "frame_gain": float(self.config.get("solver", {}).get("frame_gain", 0.5)), "frame_lm_damping": float(self.config.get("solver", {}).get("frame_lm_damping", 0.0)), "joint_center_cost": float(self.config.get("solver", {}).get("joint_center_cost", 0.0)), "joint_center_deadband": float(self.config.get("solver", {}).get("joint_center_deadband", 0.70)), "feedback_limit_tolerance_rad": float(self.config.get("solver", {}).get("feedback_limit_tolerance_rad", 0.05)), "dt_s": dispatch_dt}
                solve_current = getattr(self.solver, "solve_current", None)
                if callable(solve_current):
                    pink = solve_current(request, float(self.config.get("solver", {}).get("synchronous_response_budget_s", 0.008)))
                else:
                    self.solver.submit(request)
                    try:
                        pink = self.solver.poll(epoch, target_generation, 0)
                    except TypeError:
                        pink = self.solver.poll(epoch, target_generation)
                if pink and pink.get("ok"):
                    self._solver_reuse_count = 0
                solver_age = None if not pink else max(0.0, (time.monotonic_ns() - int(pink.get("joint_state_monotonic_ns") or pink.get("solver_monotonic_ns", time.monotonic_ns()))) / 1e9)
                if pink and pink.get("ok"):
                    self.last_solver_result = pink
                    try:
                        estimated_tcp = pose_from_tcp(dict(pink.get("tcp") or {}))
                    except (TypeError, ValueError):
                        estimated_tcp = None
                    try:
                        measured_tcp = pose_from_tcp(dict(pink.get("measured_tcp") or {}))
                    except (TypeError, ValueError):
                        measured_tcp = None
                    if measured_tcp is None and shadow:
                        measured_tcp = estimated_tcp
                    solver_finished_ns = int(pink.get("solver_finished_monotonic_ns") or time.monotonic_ns())
                    with self.lock:
                        self.execution_sample = {
                            "sample_id": sample_id,
                            "target_generation": target_generation,
                            "sample_monotonic_ns": int(pink.get("joint_state_monotonic_ns") or now_ns),
                            "solver_finished_monotonic_ns": solver_finished_ns,
                            "joint_state_rad": list(q),
                            "joint_velocity_rad_s": list(qd),
                            "measured_joint_state_rad": list(measured_q),
                            "measured_joint_velocity_rad_s": list(qd) if shadow else list((feedback or {}).get("velocities") or []),
                            "estimated_joint_state_rad": list(q),
                            "estimated_joint_velocity_rad_s": list(qd),
                            "estimated_tcp_pose": estimated_tcp,
                            "measured_tcp_pose": measured_tcp,
                            "target_tcp": dict(target_pose),
                            "position_error_m": float(pink.get("position_error_m", float("inf"))),
                            "orientation_error_rad": float(pink.get("orientation_error_rad", float("inf"))),
                            "feedback_age_s": feedback_age,
                            "estimated_feedback_delay_s": feedback_age,
                            "solver_latency_s": max(0.0, (solver_finished_ns - int(pink.get("joint_state_monotonic_ns") or now_ns)) / 1e9),
                            "dispatch_interval_s": dispatch_dt,
                        }
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
                    max_delta = [float(value) * dispatch_dt for value in data["acceleration_rad_s2"]]
                    target_velocity = [
                        previous - math.copysign(min(abs(previous), delta), previous)
                        if abs(previous) > 0.0 else 0.0
                        for previous, delta in zip(self.last_sent_velocity, max_delta)
                    ]
                    supervisor_report, pink = {"reason": "fresh Pink result unavailable; acceleration-limited braking"}, None
                    delay_budget = self.supervisor.observe_delay(feedback_age, actual_dt, None, float(self._timing.get("batch_skew_ms", 0.0) or 0.0) / 1000.0, float(self._timing.get("response_s", 0.0) or 0.0))
                # Pink owns normal acceleration shaping, but all downstream
                # scaling is checked again against the actual dispatch period.
                acceleration_limited = []
                for requested, previous, acceleration in zip(target_velocity, self.last_sent_velocity, data["acceleration_rad_s2"]):
                    delta = float(acceleration) * dispatch_dt
                    acceleration_limited.append(max(previous - delta, min(previous + delta, requested)))
                target_velocity = acceleration_limited
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
                    # Build the identical CPV position target that hardware
                    # receives.  ShadowPlant applies it asynchronously; it is
                    # never promoted immediately to measured feedback.
                    previous_velocity = list(self.trajectory["velocity_rad_s"])
                    position_target = [
                        float(current) + float(command) * actual_dt
                        for current, command in zip(q, final_velocity)
                    ]
                    self.shadow_joints = list(position_target)
                    if self.shadow_plant is not None:
                        self.shadow_plant.dispatch(position_target, time.monotonic())
                    self.trajectory["position_rad"] = list(position_target)
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
                        self.last_output = {"status": "limited" if gate_limited else "accepted", "final_joint_target_rad": list(position_target), "final_joint_velocity_rad_s": list(final_velocity), "sequence": int((command or {}).get("sequence", session.get("sequence", 0))), "epoch": epoch}
                        self._last_dispatch_monotonic_ns = time.monotonic_ns()
                    self._set_result(True, result_reason, robot_commands_sent=False, solver=pink if pink and pink.get("ok") else None, ruckig=ruckig_info, supervisor=supervisor_report, gate_reason=gate_reason, gate_limited=gate_limited, final_joint_target_rad=list(position_target), final_joint_velocity_rad_s=list(final_velocity))
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
                    batch = self.hardware.send_servo_position(position_target, str(session.get("session_id")), epoch)
                    dispatch_finished_ns = int((batch or {}).get("finished_monotonic_ns") or time.monotonic_ns())
                    with self.lock:
                        self.cpv_send_count += 1; self.output_count += 1; self._send_times.append(time.monotonic()); self._batch_history.append({"control_sample_id": sample_id, "motion_epoch": epoch, "joint_target_rad": list(position_target), "joint_velocity_rad_s": list(final_velocity), **batch})
                        self.last_output = {"status": "limited" if gate_limited else "accepted", "final_joint_target_rad": list(position_target), "final_joint_velocity_rad_s": list(final_velocity), "sequence": int((command or {}).get("sequence", session.get("sequence", 0))), "epoch": epoch}
                        self._last_dispatch_monotonic_ns = dispatch_finished_ns
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
                    arrival_position = float(self.config.get("osc", {}).get("arrival_position_tolerance_m", 0.0005))
                    arrival_orientation = float(self.config.get("osc", {}).get("arrival_orientation_tolerance_rad", math.radians(0.25)))
                    arrival_velocity = float(self.config.get("osc", {}).get("arrival_joint_velocity_tolerance_rad_s", 0.005))
                    arrival_dwell_s = float(self.config.get("osc", {}).get("arrival_dwell_s", 0.25))
                    # A late solver reply is deliberately rejected for output,
                    # but it must not break an already-stable arrival dwell.
                    # Use only an execution sample that belongs to the current
                    # target generation; a sample from an old target can never
                    # satisfy this condition.
                    arrival_sample = pink if pink and pink.get("ok") else self.execution_sample
                    arrival_matches_target = bool(
                        arrival_sample
                        and int(arrival_sample.get("target_generation", -1)) == target_generation
                    )
                    arrival_ok = bool(
                        arrival_matches_target
                        and float(arrival_sample.get("position_error_m", float("inf"))) <= arrival_position
                        and float(arrival_sample.get("orientation_error_rad", float("inf"))) <= arrival_orientation
                        and max((abs(value) for value in final_velocity), default=0.0) <= arrival_velocity
                    )
                    arrival_now_ns = time.monotonic_ns()
                    if arrival_ok:
                        if not self._arrival_since_monotonic_ns:
                            self._arrival_since_monotonic_ns = arrival_now_ns
                        self._arrival_reached = (
                            arrival_now_ns - self._arrival_since_monotonic_ns
                        ) / 1e9 >= arrival_dwell_s
                    else:
                        self._arrival_since_monotonic_ns = 0
                        self._arrival_reached = False
                    if (
                        str((command or {}).get("osc_mode")) == "move_tcp"
                        and self._arrival_reached
                    ):
                        self._invalidate_motion("OSC move_tcp target reached")
                    if self.trajectory_state == "BRAKING" and settled:
                        self.trajectory_state, self.trajectory_brake_reason, self.feedback_sync_pending = "HOLD_READY", None, True
                        self.needs_resync = True
                        if not shadow:
                            self.hardware.latch_osc_hold("osc braking settled")
                    actuator_response = None
                    if not shadow:
                        tolerance = float(self.config.get("diagnostics", {}).get("response_joint_tolerance_rad", 0.02))
                        with self.lock:
                            prior_batches = list(self._batch_history)[:-1]
                        for dispatched in reversed(prior_batches):
                            target = dispatched.get("joint_target_rad") or []
                            finished = dispatched.get("finished_monotonic_ns")
                            if len(target) != 7 or not isinstance(finished, int):
                                continue
                            error = max(abs(float(goal) - float(actual)) for goal, actual in zip(target, measured_q))
                            if error <= tolerance:
                                actuator_response = {"control_sample_id": dispatched.get("control_sample_id"), "latency_ms": (time.monotonic_ns() - finished) / 1e6, "max_joint_error_rad": error, "tolerance_rad": tolerance}
                                break
                    cycle_finished_ns = time.monotonic_ns()
                    cycle_finished_perf_ns = time.perf_counter_ns()
                    cycle_duration_ms = (cycle_finished_perf_ns - cycle_started_perf_ns) / 1e6
                    self._timing = {"actual_dt_s": actual_dt, "dispatch_interval_s": dispatch_dt, "feedback_age_s": feedback_age, "estimated_feedback_delay_s": feedback_age, "feedback_read_duration_s": None if shadow else float((feedback or {}).get("read_duration_s", 0.0)), "feedback_requested_monotonic_ns": None if shadow else (feedback or {}).get("requested_monotonic_ns"), "feedback_received_monotonic_ns": None if shadow else (feedback or {}).get("received_monotonic_ns"), "feedback_sdk_timestamp": None if shadow else (feedback or {}).get("sdk_joint_timestamp"), "feedback_sdk_hz": None if shadow else (feedback or {}).get("joint_feedback_hz"), "feedback_soft_stale": soft_stale, "feedback_hard_stale": hard_stale, "solver_age_s": solver_age, "pink_requested_monotonic_ns": None if not pink else pink.get("osc_pink_requested_monotonic_ns"), "pink_written_monotonic_ns": None if not pink else pink.get("osc_pink_written_monotonic_ns"), "pink_started_monotonic_ns": None if not pink else pink.get("solver_monotonic_ns"), "pink_finished_monotonic_ns": None if not pink else pink.get("solver_finished_monotonic_ns"), "solver_latency_s": None if not pink else max(0.0, (int(pink.get("solver_finished_monotonic_ns") or cycle_finished_ns) - int(pink.get("joint_state_monotonic_ns") or cycle_finished_ns)) / 1e9), "batch_duration_ms": (batch or {}).get("batch_duration_ms"), "batch_skew_ms": (batch or {}).get("batch_skew_ms"), "queue_delay_ms": (batch or {}).get("queue_delay_ms"), "transport_duration_ms": (batch or {}).get("transport_duration_ms"), "joint_sent_monotonic_ns": (batch or {}).get("joint_sent_monotonic_ns"), "actuator_feedback_response": actuator_response, "delay_budget_s": delay_budget, "motion_epoch": epoch, "control_sample_id": sample_id, "target_generation": target_generation, "gate_ok": gate_ok, "gate_limited": gate_limited, "gate_reason": gate_reason, "cycle_duration_ms": cycle_duration_ms, "deadline_overrun_ms": max(0.0, cycle_duration_ms - period * 1000)}
                    self._cycle_trace.append(dict(self._timing))
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
                self._fault_zero(f"osc loop: {type(exc).__name__}: {exc}", shadow=False)

    def _fault_zero(self, reason: str, shadow: bool) -> None:
        with self.lock: self.trajectory_state, self.trajectory_brake_reason = "FAULT", reason
        if not shadow:
            self.hardware.trigger_safety_fault(reason)
        self.last_sent_velocity = [0.0] * 7
        self._set_result(False, reason, robot_commands_sent=not shadow)


class OscRuntime:
    """Internal OSC facade; the Servo implementation never escapes this type."""

    def __init__(self, hardware: OscHardwarePort, rx: _OscRxPort, project_root: Path, config: dict[str, Any]) -> None:
        self._receiver = _OscFeedbackReceiver(rx, float((config.get("runtime") or {}).get("control_hz", 50)))
        self._servo = _OperationalSpaceServo(hardware, project_root, config, self._receiver)

    def start(self) -> None: self._receiver.start()

    def close(self) -> bool:
        self._servo.stop_session("OSC runtime shutdown")
        return self._receiver.close()

    def rx_snapshot(self) -> dict[str, Any] | None: return self._receiver.snapshot()

    def wait_for_rx_after(self, revision: int, timeout_s: float) -> dict[str, Any]:
        return self._receiver.wait_for_revision_after(revision, timeout_s)

    def status(self) -> dict[str, Any]: return self._servo.status()
    def target_pose(self) -> dict[str, list[float]] | None: return self._servo._public_target_pose()
    def accepting_targets(self) -> bool: return self._servo._is_accepting_targets()
    def kinematics(self) -> dict[str, Any]: return self._servo.kinematics()
    def start_session(self, *, client_id: str, execution_mode: str) -> dict[str, Any]: return self._servo.start_session(client_id, execution_mode)
    def stop_session(self, reason: str) -> dict[str, Any]: return self._servo.stop_session(reason)
    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]: return self._servo.heartbeat(client_id, session_id)
    def heartbeat_expired(self) -> bool: return self._servo.heartbeat_expired()
    def submit_absolute_target(self, body: dict[str, Any], *, mode: str) -> dict[str, Any]: return self._servo.submit_absolute_target(body, mode=mode)
    def request_shadow_hold(self, reason: str) -> dict[str, Any]: return self._servo.request_shadow_hold(reason)
    def freeze_for_authority_change(self, epoch: int, reason: str) -> None: self._servo.freeze_for_authority_change(epoch, reason)
    def abandon_session_without_braking(self, epoch: int, reason: str) -> dict[str, Any]: return self._servo.abandon_session_without_braking(epoch, reason)

    def cpv_limits(self) -> tuple[float, float]:
        return (
            float(self._servo.limits.get("joint_speed_rad_s", 1.5)),
            float(self._servo.config.get("solver", {}).get("ruckig_max_acceleration", 5.0)),
        )

    def apply_hardware_calibration(self, calibration: dict[str, Any], feedback: dict[str, Any], cpv_parameters: dict[str, Any]) -> dict[str, Any]:
        shadow = self._servo.config.setdefault("shadow_transport", {})
        estimator = self._servo.config.setdefault("state_estimator", {})
        shadow["feedback_delay_s"] = calibration["feedback_delay_s"]
        shadow["feedback_jitter_s"] = calibration["feedback_jitter_s"]
        estimator["max_prediction_s"] = max(0.02, min(0.15, calibration["feedback_delay_s"] + float(feedback["inter_request_p95_s"])))
        self._servo.config["hardware_readonly_calibration"] = {
            **calibration,
            "feedback_read_duration_p50_s": feedback["read_duration_p50_s"],
            "feedback_read_duration_p95_s": feedback["read_duration_p95_s"],
            "feedback_inter_request_p95_s": feedback["inter_request_p95_s"],
            "cpv_parameters": dict(cpv_parameters),
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return {
            "shadow_transport": {
                "feedback_delay_s": shadow["feedback_delay_s"],
                "feedback_jitter_s": shadow["feedback_jitter_s"],
                "max_joint_speed_rad_s": shadow.get("max_joint_speed_rad_s"),
                "max_joint_acceleration_rad_s2": shadow.get("max_joint_acceleration_rad_s2"),
            },
            "state_estimator": {"max_prediction_s": estimator["max_prediction_s"]},
        }

    def configuration_json(self) -> str:
        return json.dumps(self._servo.config, ensure_ascii=False, indent=2) + "\n"
