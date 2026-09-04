"""PICO 4 Ultra input adapter.

This module intentionally has no robot, CAN, Pink, or Ruckig dependency.  It
owns headset-local concepts (pairing, controller anchors and buttons) and its
only robot-facing calls are standard OSC commands.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any


def _vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} finite values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _normalise(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm < 1e-9:
        raise ValueError("controller orientation cannot be zero")
    return [value / norm for value in q]


def _inverse(q: list[float]) -> list[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def _map(matrix: list[list[float]], value: list[float], gain: float) -> list[float]:
    return [gain * sum(float(row[index]) * value[index] for index in range(3)) for row in matrix]


# Physical-arm verified PICO-to-NERO frame mappings.  Position and
# orientation intentionally use separate matrices; they are not interchangeable
# defaults even though both are orthonormal axis maps.
RECOMMENDED_POSITION_FRAME_MAP = [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
RECOMMENDED_ORIENTATION_FRAME_MAP = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
# Compatibility alias for callers that historically used the position map.
DEFAULT_FRAME_MAP = RECOMMENDED_POSITION_FRAME_MAP


def _frame_map(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 proper rotation matrix")
    rows = tuple(tuple(float(item) for item in row) for row in value)
    if any(len(row) != 3 or not all(math.isfinite(item) for item in row) for row in rows):
        raise ValueError(f"{name} must be a 3x3 proper rotation matrix")
    for row in rows:
        if abs(sum(item * item for item in row) - 1.0) > 1e-5:
            raise ValueError(f"{name} rows must be unit vectors")
    for first in range(3):
        for second in range(first + 1, 3):
            if abs(sum(rows[first][i] * rows[second][i] for i in range(3))) > 1e-5:
                raise ValueError(f"{name} rows must be orthogonal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    # Unity/PICO and ROS/NERO may use opposite handedness. A coordinate-frame
    # conversion can therefore be either a proper rotation (det=+1) or a
    # handedness conversion (det=-1); both preserve lengths and orthogonality.
    # Reject scaling/shearing, but allow either frame chirality.
    if abs(abs(determinant) - 1.0) > 1e-5:
        raise ValueError(f"{name} must have determinant +1 or -1")
    return rows


def _rotation_vector(q: list[float]) -> list[float]:
    """Convert a unit quaternion to its shortest rotation vector."""
    q = _normalise(q)
    if q[3] < 0.0:
        q = [-value for value in q]
    w = max(-1.0, min(1.0, q[3]))
    angle = 2.0 * math.acos(w)
    sine = math.sqrt(max(0.0, 1.0 - w * w))
    if sine < 1e-8:
        return [2.0 * q[0], 2.0 * q[1], 2.0 * q[2]]
    scale = angle / sine
    return [q[index] * scale for index in range(3)]


def _quaternion_from_rotation_vector(vector: list[float]) -> list[float]:
    angle = math.sqrt(sum(value * value for value in vector))
    if angle < 1e-8:
        return _normalise([0.5 * vector[0], 0.5 * vector[1], 0.5 * vector[2], 1.0])
    scale = math.sin(angle / 2.0) / angle
    return _normalise([vector[0] * scale, vector[1] * scale, vector[2] * scale, math.cos(angle / 2.0)])


def _attenuate_relative_rotation(relative: list[float], gain: float) -> list[float]:
    """Scale the shortest rotation from the Grip anchor without changing it."""
    return _quaternion_from_rotation_vector([gain * value for value in _rotation_vector(relative)])


def _quaternion_to_matrix(q: list[float]) -> list[list[float]]:
    x, y, z, w = _normalise(q)
    return [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]]


def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def _matrix_transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[col][row] for col in range(3)] for row in range(3)]


def _matrix_to_quaternion(matrix: list[list[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w, x = 0.25 * scale, (matrix[2][1] - matrix[1][2]) / scale
        y, z = (matrix[0][2] - matrix[2][0]) / scale, (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(max(1e-12, 1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])) * 2.0
        w, x = (matrix[2][1] - matrix[1][2]) / scale, 0.25 * scale
        y, z = (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(max(1e-12, 1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])) * 2.0
        w, x = (matrix[0][2] - matrix[2][0]) / scale, (matrix[0][1] + matrix[1][0]) / scale
        y, z = 0.25 * scale, (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(max(1e-12, 1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])) * 2.0
        w, x = (matrix[1][0] - matrix[0][1]) / scale, (matrix[0][2] + matrix[2][0]) / scale
        y, z = (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale
    return _normalise([x, y, z, w])


def _map_absolute_orientation(matrix: tuple[tuple[float, ...], ...], q: list[float]) -> list[float]:
    frame = [list(row) for row in matrix]
    mapped = _matrix_multiply(_matrix_multiply(frame, _quaternion_to_matrix(q)), _matrix_transpose(frame))
    return _matrix_to_quaternion(mapped)


def _rotation_degrees(q: list[float]) -> list[float]:
    return [value * 180.0 / math.pi for value in _rotation_vector(q)]


class PicoInputAdapter:
    """Convert raw headset input to absolute base-frame OSC targets."""

    MIN_GAIN = 0.25
    MAX_GAIN = 1.0

    def __init__(self, osc: Any, config: dict[str, Any], trace_logger: Any | None = None) -> None:
        supplied_config = dict(config or {})
        has_persisted_mapping = "position_axis_map" in supplied_config and "orientation_axis_map" in supplied_config
        defaults = {"mapping_verified": False, "translation_gain": 1.0, "rotation_gain": 1.0,
                    "position_axis_map": RECOMMENDED_POSITION_FRAME_MAP,
                    "orientation_axis_map": RECOMMENDED_ORIENTATION_FRAME_MAP,
                    "gripper_open_width_m": 0.095, "gripper_force_n": 1.0}
        defaults.update(supplied_config)
        self.osc, self.config = osc, defaults
        self.trace_logger = trace_logger
        # Cache immutable hot-path settings.  Runtime configuration is loaded
        # once when the adapter is created, so looking these up through the
        # config dictionary on every pose sample only adds avoidable work.
        self._translation_gain = float(defaults.get("translation_gain", 1.0))
        self._rotation_gain = float(defaults.get("rotation_gain", 1.0))
        if not self.MIN_GAIN <= self._translation_gain <= self.MAX_GAIN or not self.MIN_GAIN <= self._rotation_gain <= self.MAX_GAIN:
            raise ValueError("PICO translation_gain and rotation_gain must be between 0.25 and 1.00")
        self._position_axis_map = _frame_map(defaults["position_axis_map"], "position_axis_map")
        self._orientation_axis_map = _frame_map(defaults["orientation_axis_map"], "orientation_axis_map")
        self._recommended_position_axis_map = _frame_map(RECOMMENDED_POSITION_FRAME_MAP, "recommended_position_axis_map")
        self._recommended_orientation_axis_map = _frame_map(RECOMMENDED_ORIENTATION_FRAME_MAP, "recommended_orientation_axis_map")
        self._mapping_persisted = has_persisted_mapping
        self._mapping_source = "runtime_config" if has_persisted_mapping else "built_in_recommended"
        self.config["mapping_verified"] = self._mapping_is_recommended()
        self._gripper_open_width_m = float(defaults.get("gripper_open_width_m", 0.095))
        self._gripper_force_n = float(defaults.get("gripper_force_n", 1.0))
        self._execution_mode: str | None = None
        self.lock = threading.RLock()
        self.state: dict[str, Any] = self._empty_state()
        self._session_id: str | None = None
        self._client_id: str | None = None
        self._anchor_controller: dict[str, list[float]] | None = None
        self._anchor_tcp: dict[str, list[float]] | None = None
        self._absolute_position_offset: list[float] | None = None
        self._orientation_correction: list[float] = [0.0, 0.0, 0.0, 1.0]
        self._orientation_calibration_status = "DIRECT_MAPPING"
        self._orientation_calibrated_at: float | None = None
        self._pending_anchor: dict[str, Any] | None = None
        self._sequence = 0
        self._pairing_stop = threading.Event()
        self._pairing_thread: threading.Thread | None = None
        self._pose_receive_times_ns: list[int] = []
        self._last_frame_grip = False
        self._last_frame_trigger = 0.0
        # ``tracking_valid`` is a safety edge, rather than a command stream:
        # one loss must request HOLD, but repeated invalid telemetry must not
        # repeatedly restart hardware braking.
        self._last_frame_tracking_valid = False
        if self.trace_logger:
            self.trace_logger.append({"record_type": "event", "event": "adapter_mapping_loaded",
                                      "monotonic_ns": time.monotonic_ns(),
                                      "source": self._mapping_source, "persisted": self._mapping_persisted,
                                      "verified": self.config["mapping_verified"],
                                      "position_axis_map": [list(row) for row in self._position_axis_map],
                                      "orientation_axis_map": [list(row) for row in self._orientation_axis_map]})

    def _mapping_is_recommended(self) -> bool:
        return self._position_axis_map == self._recommended_position_axis_map and self._orientation_axis_map == self._recommended_orientation_axis_map

    def _empty_state(self) -> dict[str, Any]:
        return {"adapter": "pico", "state": "IDLE", "session_id": None,
                "connected": False, "paired": False, "tracking_valid": False,
                "anchor_active": False, "last_input_age_s": None,
                "last_error": None, "gripper_position": None,
                "clutch_signal": False, "input_clutch": False, "input_sequence": 0,
                "input_position_m": [0.0, 0.0, 0.0],
                "input_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "input_rotation_degrees": [0.0, 0.0, 0.0],
                "mapped_input_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "mapped_input_rotation_degrees": [0.0, 0.0, 0.0],
                "absolute_orientation_preview_xyzw": [0.0, 0.0, 0.0, 1.0],
                "absolute_orientation_preview_degrees": [0.0, 0.0, 0.0],
                "input_grip": False, "input_trigger_value": 0.0,
                "last_target_pose": None, "last_target_status": "NONE",
                "target_rotation_degrees": [0.0, 0.0, 0.0],
                "last_target_osc_sequence": None, "last_target_sent_at": None,
                "target_pose_mode": "ABSOLUTE_DIRECT",
                "orientation_tracking_mode": "RELATIVE_ANCHORED",
                "orientation_command_enabled": False,
                "orientation_calibration_status": "DIRECT_MAPPING",
                "orientation_correction_xyzw": [0.0, 0.0, 0.0, 1.0],
                "orientation_correction_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "anchor_tcp_source": None,
                "orientation_calibrated_at": None,
                "input_pose_age_ms": None, "last_pose_timing_ms": None,
                "pose_rx_hz": None, "pose_received_count": 0,
                "updated_at": time.time()}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = copy.deepcopy(self.state)
            if self.state["updated_at"]:
                result["last_input_age_s"] = max(0.0, time.time() - self.state["updated_at"])
            if self.state.get("last_target_sent_at"):
                result["last_target_age_ms"] = max(0.0, (time.time() - float(self.state["last_target_sent_at"])) * 1000.0)
            else:
                result["last_target_age_ms"] = None
            received_ns = self.state.get("input_received_monotonic_ns")
            result["input_pose_age_ms"] = None if not received_ns else max(0.0, (time.monotonic_ns() - int(received_ns)) / 1e6)
            result.pop("input_received_monotonic_ns", None)
            result["mapping"] = {"translation_gain": self._translation_gain,
                                 "rotation_gain": self._rotation_gain,
                                 "min_gain": self.MIN_GAIN, "max_gain": self.MAX_GAIN,
                                 "adjustable": not bool(self.state.get("anchor_active")),
                                 "execution_mode": self._execution_mode,
                                 "verified": bool(self.config.get("mapping_verified", False)),
                                 "persisted": self._mapping_persisted,
                                 "source": self._mapping_source,
                                 "position_axis_map": [list(row) for row in self._position_axis_map],
                                 "orientation_axis_map": [list(row) for row in self._orientation_axis_map],
                                 "recommended_position_axis_map": [list(row) for row in self._recommended_position_axis_map],
                                 "recommended_orientation_axis_map": [list(row) for row in self._recommended_orientation_axis_map]}
            return result

    def update_mapping(self, session_id: str, client_id: str, position_axis_map: Any,
                       orientation_axis_map: Any, verified: bool = False) -> dict[str, Any]:
        del verified  # The server determines verification from the fixed physical-arm recommendation.
        position_map = _frame_map(position_axis_map, "position_axis_map")
        orientation_map = _frame_map(orientation_axis_map, "orientation_axis_map")
        with self.lock:
            if self._session_id:
                authorized = session_id == self._session_id and client_id == self._client_id
            else:
                osc_session = (self.osc.state() or {}).get("session") or {}
                authorized = session_id == str(osc_session.get("id", "")) and client_id == str(osc_session.get("client_id", ""))
            if not authorized:
                raise PermissionError("只有当前控制台/PICO 会话所有者可以修改坐标矩阵")
            was_tracking = bool(self.state.get("anchor_active"))
            if was_tracking and self._anchor_tcp and self._anchor_controller:
                # Rebase the position offset against the current TCP reference
                # before replacing C_p. This makes an online matrix edit take
                # effect without producing a position jump at the edit frame.
                current_position = list(self.state.get("input_position_m") or self._anchor_controller["position_m"])
                osc_state = self.osc.state()
                execution_tcp = (osc_state.get("execution") or {}).get("measured_tcp_pose")
                command_tcp = (osc_state.get("command") or {}).get("target_tcp")
                tcp_reference = execution_tcp if isinstance(execution_tcp, dict) else command_tcp
                if not isinstance(tcp_reference, dict):
                    tcp_reference = self._anchor_tcp
                current_tcp = _vector(tcp_reference.get("position_m"), 3, "TCP position")
                # Keep the current physical point as the new zero so changing
                # the matrix cannot make the arm jump at the edit frame.
                self._anchor_controller["position_m"] = current_position
                self._anchor_tcp["position_m"] = current_tcp
                self._absolute_position_offset = list(current_tcp)
            self._position_axis_map = position_map
            self._orientation_axis_map = orientation_map
            self.config["position_axis_map"] = [list(row) for row in position_map]
            self.config["orientation_axis_map"] = [list(row) for row in orientation_map]
            self.config["mapping_verified"] = self._mapping_is_recommended()
            self._mapping_persisted = False
            self._mapping_source = "runtime_memory"
            self.state.update({"updated_at": time.time(), "mapping_applied_while_tracking": was_tracking,
                               "last_error": None})
            result = {"ok": True, "accepted": True, "mapping_verified": self.config["mapping_verified"],
                    "persisted": False,
                    "applied_while_tracking": was_tracking,
                    "position_axis_map": [list(row) for row in position_map],
                    "orientation_axis_map": [list(row) for row in orientation_map]}
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "adapter_mapping_updated",
                                          "monotonic_ns": time.monotonic_ns(), "verified": self.config["mapping_verified"],
                                          "position_axis_map": result["position_axis_map"],
                                          "orientation_axis_map": result["orientation_axis_map"]})
            return result

    def mapping_persisted(self) -> dict[str, Any]:
        """Record that the current validated mapping was durably written."""
        with self.lock:
            self._mapping_persisted = True
            self._mapping_source = "runtime_config"
            result = {"persisted": True, "mapping_verified": bool(self.config["mapping_verified"]),
                      "position_axis_map": [list(row) for row in self._position_axis_map],
                      "orientation_axis_map": [list(row) for row in self._orientation_axis_map]}
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "adapter_mapping_persisted",
                                          "monotonic_ns": time.monotonic_ns(), "verified": result["mapping_verified"],
                                          "position_axis_map": result["position_axis_map"],
                                          "orientation_axis_map": result["orientation_axis_map"]})
            return result

    def begin_pairing(self, session_id: str, client_id: str) -> None:
        osc = self.osc.state(); session = osc.get("session") or {}
        if session.get("state") != "ACTIVE" or session.get("id") != session_id or session.get("client_id") != client_id:
            raise PermissionError("PICO requires the caller's active OSC session")
        with self.lock:
            self._session_id, self._client_id = session_id, client_id
            self._last_frame_grip = False
            self._last_frame_trigger = 0.0
            self._last_frame_tracking_valid = False
            self._execution_mode = str(session.get("execution_mode") or "shadow")
            self._anchor_controller = self._anchor_tcp = None
            self._absolute_position_offset = None
            self._orientation_correction = [0.0, 0.0, 0.0, 1.0]
            self._orientation_calibration_status = "DIRECT_MAPPING"
            self._orientation_calibrated_at = None
            self._orientation_calibrated_at = None
            self._sequence = int((osc.get("command") or {}).get("sequence") or 0)
            self.state = self._empty_state()
            self.state.update({"state": "PAIRING", "session_id": session_id, "updated_at": time.time()})
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "adapter_pairing_started",
                                          "monotonic_ns": time.monotonic_ns(), "session_id": session_id,
                                          "client_id": client_id, "execution_mode": self._execution_mode})
            self._pairing_stop.set()
            self._pairing_stop = threading.Event()
            self._pairing_thread = threading.Thread(target=self._pairing_heartbeat_loop, name="nero-pico-pairing-heartbeat", daemon=True)
            self._pairing_thread.start()

    def update_sensitivity(self, session_id: str, client_id: str, translation_gain: Any,
                           rotation_gain: Any, hardware_high_gain_confirmed: bool = False) -> dict[str, Any]:
        try:
            translation = float(translation_gain)
            rotation = float(rotation_gain)
        except (TypeError, ValueError) as exc:
            raise ValueError("灵敏度必须是有限数字") from exc
        if not math.isfinite(translation) or not math.isfinite(rotation):
            raise ValueError("灵敏度必须是有限数字")
        if not self.MIN_GAIN <= translation <= self.MAX_GAIN or not self.MIN_GAIN <= rotation <= self.MAX_GAIN:
            raise ValueError("平移与旋转增益范围必须为 25% 至 100%")
        with self.lock:
            if session_id != self._session_id or client_id != self._client_id:
                raise PermissionError("只有当前 PICO 会话所有者可以修改灵敏度")
            if self.state.get("anchor_active"):
                return {"ok": False, "accepted": False, "recoverable": True,
                        "reason": "release_grip_first", "message": "请先松开 Grip 再调整灵敏度",
                        "adjustable": False}
            self._translation_gain, self._rotation_gain = translation, rotation
            self.config["translation_gain"], self.config["rotation_gain"] = translation, rotation
            self.state["updated_at"] = time.time()
            return {"ok": True, "accepted": True, "translation_gain": translation,
                    "rotation_gain": rotation, "adjustable": True}

    def paired(self) -> None:
        with self.lock:
            if not self._session_id:
                raise RuntimeError("PICO pairing has not been started")
            self.state.update({"state": "READY", "connected": True, "paired": True, "updated_at": time.time(), "last_error": None})
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "adapter_paired",
                                          "monotonic_ns": time.monotonic_ns(), "session_id": self._session_id})
            self._pairing_stop.set()

    def connection_lost(self, reason: str) -> None:
        """Make socket loss visible and safe without discarding a valid pairing.

        The gateway can reconnect a headset with the same short-lived pairing
        record.  Keeping the adapter binding lets the first new Grip frame
        resume the OSC session, but it must never retain an active anchor or
        advertise a live PICO connection while no input socket exists.
        """
        try:
            self.stop(reason)
        except Exception as exc:
            with self.lock:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self.state.update({"state": "READY" if self._session_id else "IDLE",
                               "connected": False, "paired": bool(self._session_id),
                               "anchor_active": False, "tracking_valid": False,
                               "updated_at": time.time()})
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "adapter_connection_lost",
                                          "monotonic_ns": time.monotonic_ns(), "reason": reason,
                                          "session_id": self._session_id})

    def _pairing_heartbeat_loop(self) -> None:
        while not self._pairing_stop.wait(1.0):
            with self.lock:
                if self.state.get("state") != "PAIRING" or not self._session_id or not self._client_id:
                    return
                session_id, client_id = self._session_id, self._client_id
            try:
                self.osc.heartbeat(client_id, session_id)
            except Exception as exc:
                with self.lock:
                    self.state.update({"state": "ERROR", "last_error": f"{type(exc).__name__}: {exc}", "updated_at": time.time()})
                return

    def _command(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Do not hold the adapter lock while entering the OSC/control layer.
        # The latter may wait for the shadow runtime, and holding this lock
        # would block the websocket receive path and state polling together.
        self._ensure_osc_session()
        with self.lock:
            if not self._session_id or not self._client_id:
                raise RuntimeError("PICO has no bound OSC session")
            session_id, client_id = self._session_id, self._client_id
            # Monotonic time avoids wall-clock corrections producing duplicate
            # or unexpectedly delayed command sequence values.
            self._sequence = max(self._sequence + 1, time.monotonic_ns() // 1_000_000)
            sequence = self._sequence
        if kind == "track_tcp":
            return self.osc.track_tcp(session_id, client_id, sequence, payload["target_pose"])
        if kind == "hold":
            # Grip release is a deadman HOLD, not a session handoff. The
            # console facade keeps the hardware OSC lease alive while the
            # arm brakes; manual Disconnect remains the only disconnect path.
            input_hold = getattr(self.osc, "osc_input_hold", None)
            if callable(input_hold):
                return input_hold(str(payload.get("reason", "PICO HOLD")))
            return self.osc.hold(session_id, client_id, sequence, str(payload.get("reason", "PICO HOLD")))
        if kind == "gripper":
            return self.osc.gripper(session_id, client_id, sequence, payload)
        raise ValueError(f"unsupported PICO OSC command {kind}")

    def _ensure_osc_session(self) -> None:
        """Resume OSC after FREEDRIVE without recreating receiver pairing."""
        with self.lock:
            client_id = self._client_id
            execution_mode = self._execution_mode or "shadow"
        if not client_id:
            raise RuntimeError("PICO has no bound receiver session")
        session = (self.osc.state() or {}).get("session") or {}
        if session.get("state") == "ACTIVE" and session.get("client_id") == client_id:
            return
        result = self.osc.start_session(client_id, execution_mode)
        state = (result or {}).get("state") or {}
        fresh = (result or {}).get("session") or state.get("session") or {}
        fresh_id = str(fresh.get("id") or fresh.get("session_id") or "")
        if not fresh_id:
            raise RuntimeError("OSC session could not be resumed")
        with self.lock:
            self._session_id = fresh_id
            self._sequence = int((state.get("command") or {}).get("sequence") or 0)
            self.state.update({"session_id": fresh_id, "state": "READY", "updated_at": time.time(), "last_error": None})

    @staticmethod
    def _ack(kind: str, result: dict[str, Any]) -> dict[str, Any]:
        """Build a small websocket ACK; full state remains available via snapshot()."""
        ack = {"ok": bool(result.get("ok", False)), "type": kind}
        nested = result.get("result")
        if isinstance(nested, dict) and "accepted" in nested:
            ack["accepted"] = bool(nested["accepted"])
        elif "accepted" in result:
            ack["accepted"] = bool(result["accepted"])
        return ack

    def heartbeat(self) -> None:
        with self.lock:
            session_id, client_id = self._session_id, self._client_id
        if session_id and client_id:
            self.osc.heartbeat(client_id, session_id)
            with self.lock:
                self.state["updated_at"] = time.time()

    def tracking(self, pose: dict[str, Any]) -> dict[str, Any]:
        position = _vector(pose.get("position_m"), 3, "controller position")
        orientation = _normalise(_vector(pose.get("orientation_xyzw"), 4, "controller orientation"))
        pending_anchor = None
        with self.lock:
            self.state.update({"tracking_valid": bool(pose.get("tracking_valid", True)), "updated_at": time.time(), "last_error": None,
                               "input_position_m": position, "input_orientation_xyzw": orientation,
                               "input_rotation_degrees": _rotation_degrees(orientation),
                               "mapped_input_orientation_xyzw": _map_absolute_orientation(self._orientation_axis_map, orientation),
                               "mapped_input_rotation_degrees": _rotation_degrees(_map_absolute_orientation(self._orientation_axis_map, orientation)),
                               "absolute_orientation_preview_xyzw": _map_absolute_orientation(self._orientation_axis_map, orientation),
                               "absolute_orientation_preview_degrees": _rotation_degrees(_map_absolute_orientation(self._orientation_axis_map, orientation)),
                               "orientation_command_enabled": bool(pose.get("clutch", self.state.get("clutch_signal", False))),
                               "input_clutch": bool(pose.get("clutch", False)),
                               "input_sequence": int(pose.get("_pico_sequence") or 0),
                               "input_received_monotonic_ns": time.monotonic_ns()})
            pending_anchor = dict(self._pending_anchor) if self._pending_anchor else None
            tracking_valid = self.state["tracking_valid"]
        if pending_anchor and self._osc_hold_ready():
            anchor_result = self.anchor_begin(pending_anchor)
            if anchor_result.get("accepted", anchor_result.get("anchor_active", False)):
                # The first frame that completes the HOLD transition must
                # also be submitted as a target.  Previously this branch only
                # created the anchor and returned, so the first Grip press
                # appeared unresponsive until the following PICO frame.
                first_target = self.pose({**pending_anchor, "tracking_valid": tracking_valid, "clutch": True})
                return {"ok": bool(first_target.get("ok", True)), "type": "tracking", "tracking_valid": tracking_valid,
                        "anchor_activated": True, "anchor_sequence": int(pending_anchor.get("_pico_sequence") or 0),
                        "anchor_result": anchor_result, "target_result": first_target}
        # Tracking is a high-rate message. Returning the full adapter snapshot
        # here caused a deepcopy and a large JSON ACK per sample.
        return {"ok": True, "type": "tracking", "tracking_valid": tracking_valid}

    def _osc_hold_ready(self) -> bool:
        snapshot = self.osc.state()
        diagnostics = snapshot.get("diagnostics") or {}
        return str(diagnostics.get("trajectory_state") or "") == "HOLD_READY"

    def input_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Process one newest combined PICO Pose + Grip + Trigger frame."""
        position = _vector(frame.get("position_m"), 3, "input_frame.position_m")
        orientation = _normalise(_vector(frame.get("orientation_xyzw"), 4, "input_frame.orientation_xyzw"))
        grip = bool(frame.get("grip", False))
        trigger = float(frame.get("trigger_value", 0.0))
        if not math.isfinite(trigger) or trigger < 0.0 or trigger > 1.0:
            raise ValueError("input_frame.trigger_value must be between 0 and 1")
        tracking_valid = bool(frame.get("tracking_valid", True))
        sequence = int(frame.get("_pico_sequence") or 0)
        mapped_orientation = _map_absolute_orientation(self._orientation_axis_map, orientation)
        with self.lock:
            previous_grip = self._last_frame_grip
            previous_trigger = self._last_frame_trigger
            previous_tracking_valid = self._last_frame_tracking_valid
            self._last_frame_grip = grip
            self._last_frame_trigger = trigger
            self._last_frame_tracking_valid = tracking_valid
            self.state.update({
                "input_grip": grip, "input_trigger_value": trigger,
                "input_clutch": grip, "input_sequence": sequence,
                "input_position_m": position, "input_orientation_xyzw": orientation,
                "input_rotation_degrees": _rotation_degrees(orientation),
                "mapped_input_orientation_xyzw": list(mapped_orientation),
                "mapped_input_rotation_degrees": _rotation_degrees(mapped_orientation),
                "absolute_orientation_preview_xyzw": list(mapped_orientation),
                "absolute_orientation_preview_degrees": _rotation_degrees(mapped_orientation),
                "orientation_command_enabled": grip,
                "tracking_valid": tracking_valid, "updated_at": time.time(),
                "input_received_monotonic_ns": int(frame.get("_gateway_received_monotonic_ns") or time.monotonic_ns()),
            })

        if not tracking_valid:
            # The headset can publish a run of invalid frames while its
            # tracker is reacquiring.  HOLD is required when tracking is
            # actually lost during an active clutch, but issuing it for every
            # 90 Hz invalid frame repeatedly resets OSC's BRAKING transition.
            # That made the first subsequent Grip press wait through multiple
            # brake windows.  Latch the loss instead; a later valid frame can
            # establish the normal fresh-feedback anchor.
            with self.lock:
                tracking_was_active = bool(self._anchor_controller or self._pending_anchor)
            if previous_tracking_valid or tracking_was_active:
                result = self.stop("PICO tracking lost")
                result.update({"event": "tracking_lost", "input_frame_sequence": sequence})
            else:
                result = {"ok": True, "type": "input_frame", "accepted": False,
                          "recoverable": True, "event": "tracking_unavailable",
                          "message": "等待有效的 PICO 位姿信号",
                          "input_frame_sequence": sequence}
            return result

        if previous_grip and not grip:
            # stop() updates clutch_signal before entering the potentially
            # slower OSC HOLD path, so the console reflects the release edge
            # immediately even if hardware control is temporarily unavailable.
            result = self.stop("PICO right Grip released")
            result.update({"event": "grip_release", "input_frame_sequence": sequence})
        elif grip and not self._anchor_controller:
            anchor = self.anchor_begin({
                "position_m": position, "orientation_xyzw": orientation,
                "_pico_sequence": sequence,
            })
            if not anchor.get("anchor_active"):
                result = dict(anchor)
                result.update({"event": "grip_press_pending", "input_frame_sequence": sequence})
                return result
            result = self.pose({**frame, "clutch": True})
            result.update({"event": "grip_press", "input_frame_sequence": sequence})
        elif grip:
            result = self.pose({**frame, "clutch": True})
        else:
            result = {"ok": True, "type": "input_frame", "accepted": True, "anchor_active": False}

        if abs(trigger - previous_trigger) >= 0.001:
            gripper_result = self.gripper(trigger)
            result["gripper"] = gripper_result
        result.update({"type": "input_frame", "grip": grip, "trigger_value": trigger,
                       "input_frame_sequence": sequence})
        return result

    def anchor_begin(self, pose: dict[str, Any], activate_clutch: bool = True) -> dict[str, Any]:
        position, orientation = _vector(pose.get("position_m"), 3, "controller position"), _normalise(_vector(pose.get("orientation_xyzw"), 4, "controller orientation"))
        osc = self.osc.state()
        if not self._osc_hold_ready() and str((osc.get("diagnostics") or {}).get("trajectory_state") or "") == "BRAKING":
            with self.lock:
                self._pending_anchor = dict(pose)
                self.state.update({"state": "WAITING_HOLD", "anchor_active": False, "clutch_signal": False,
                                   "input_clutch": True, "updated_at": time.time()})
            return {"ok": True, "type": "anchor_begin", "accepted": False, "pending": True,
                    "recoverable": True, "reason": "hold_settling", "message": "等待机械臂安全 HOLD 完成后建立离合"}
        execution = osc.get("execution") or {}
        session = osc.get("session") or {}
        tcp = execution.get("measured_tcp_pose")
        tcp_source = "measured_tcp_pose"
        if not isinstance(tcp, dict) and str(session.get("execution_mode") or "shadow") == "shadow":
            # Shadow mode may not expose a physical measured TCP. The OSC
            # command target is the current simulated TCP reference and is a
            # safe, absolute starting point for the relative position anchor.
            tcp = (osc.get("command") or {}).get("target_tcp")
            tcp_source = "shadow_command_target"
        if not isinstance(tcp, dict):
            with self.lock:
                self._pending_anchor = None
                self.state.update({"state": "READY", "anchor_active": False,
                                   "orientation_calibration_status": "WAITING_FEEDBACK",
                                   "last_error": "等待机械臂位姿，无法初始化位置锚点", "updated_at": time.time()})
            return {"ok": True, "type": "anchor_begin", "accepted": False,
                    "anchor_active": False, "recoverable": True,
                    "reason": "measured_tcp_pose_unavailable",
                    "message": "等待机械臂位姿，无法初始化位置锚点"}
        with self.lock:
            self._pending_anchor = None
            self._anchor_controller = {"position_m": position, "orientation_xyzw": orientation}
            self._anchor_tcp = {"position_m": _vector(tcp.get("position_m"), 3, "TCP position"), "orientation_xyzw": _normalise(_vector(tcp.get("orientation_xyzw"), 4, "TCP orientation"))}
            mapped_anchor_position = _map(self._position_axis_map, position, self._translation_gain)
            self._absolute_position_offset = [self._anchor_tcp["position_m"][index] - mapped_anchor_position[index] for index in range(3)]
            mapped_anchor_orientation = _map_absolute_orientation(self._orientation_axis_map, orientation)
            self._orientation_correction = _normalise(_multiply(
                self._anchor_tcp["orientation_xyzw"], _inverse(mapped_anchor_orientation)))
            self._orientation_calibration_status = "RELATIVE_ANCHORED"
            self._orientation_calibrated_at = time.time()
            self.state.update({"state": "TRACKING", "anchor_active": True, "tracking_valid": True, "updated_at": time.time(), "last_error": None,
                               "clutch_signal": bool(activate_clutch), "input_clutch": bool(activate_clutch),
                               "input_position_m": position, "input_orientation_xyzw": orientation,
                               "input_rotation_degrees": _rotation_degrees(orientation),
                               "input_sequence": int(pose.get("_pico_sequence") or 0),
                               "input_received_monotonic_ns": time.monotonic_ns(),
                               "orientation_tracking_mode": "RELATIVE_ANCHORED",
                               "orientation_calibration_status": self._orientation_calibration_status,
                               "orientation_correction_xyzw": list(self._orientation_correction),
                               "orientation_correction_matrix": _quaternion_to_matrix(self._orientation_correction),
                               "anchor_tcp_source": tcp_source,
                                "orientation_calibrated_at": self._orientation_calibrated_at})
            if self.trace_logger:
                self.trace_logger.append({"record_type": "event", "event": "anchor_begin",
                                          "monotonic_ns": time.monotonic_ns(), "pico_sequence": int(pose.get("_pico_sequence") or 0)})
            return {"ok": True, "type": "anchor_begin", "anchor_active": True,
                    "orientation_tracking_mode": "RELATIVE_ANCHORED",
                    "anchor_tcp_source": tcp_source,
                    "orientation_correction_xyzw": list(self._orientation_correction),
                    "orientation_correction_matrix": _quaternion_to_matrix(self._orientation_correction),
                    "calibrated_at": self._orientation_calibrated_at}

    def pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        adapter_started_ns = time.monotonic_ns()
        gateway_received_ns = int(pose.get("_gateway_received_monotonic_ns") or adapter_started_ns)
        pico_sequence = int(pose.get("_pico_sequence") or 0)
        with self.lock:
            # Use gateway arrival timestamps rather than adapter processing
            # timestamps. A busy OSC worker may supersede old samples, but
            # the displayed rate must still describe the PICO input stream.
            self._pose_receive_times_ns.append(gateway_received_ns)
            if len(self._pose_receive_times_ns) > 32:
                del self._pose_receive_times_ns[:-32]
            pose_rx_hz = None
            if len(self._pose_receive_times_ns) >= 2:
                elapsed_ns = self._pose_receive_times_ns[-1] - self._pose_receive_times_ns[0]
                if elapsed_ns > 0:
                    pose_rx_hz = (len(self._pose_receive_times_ns) - 1) * 1e9 / elapsed_ns
            self.state["pose_rx_hz"] = pose_rx_hz
            self.state["pose_received_count"] = int(self.state.get("pose_received_count", 0)) + 1
        if self.trace_logger:
            self.trace_logger.append({"record_type": "sample", "event": "adapter_pose_enter",
                                      "monotonic_ns": adapter_started_ns, "pico_sequence": pico_sequence})
        if not bool(pose.get("tracking_valid", True)):
            return self.stop("PICO controller tracking lost")
        position, orientation = _vector(pose.get("position_m"), 3, "controller position"), _normalise(_vector(pose.get("orientation_xyzw"), 4, "controller orientation"))
        with self.lock:
            mapped_orientation = _map_absolute_orientation(self._orientation_axis_map, orientation)
            self.state.update({"input_position_m": position, "input_orientation_xyzw": orientation,
                               "input_rotation_degrees": _rotation_degrees(orientation),
                               "mapped_input_orientation_xyzw": list(mapped_orientation),
                               "mapped_input_rotation_degrees": _rotation_degrees(mapped_orientation),
                               "absolute_orientation_preview_xyzw": list(mapped_orientation),
                               "absolute_orientation_preview_degrees": _rotation_degrees(mapped_orientation),
                               "input_clutch": bool(pose.get("clutch", self.state.get("clutch_signal", False))),
                               "input_sequence": pico_sequence,
                               "input_received_monotonic_ns": gateway_received_ns})
            if not self._anchor_controller or not self._anchor_tcp:
                return {
                    "ok": True,
                    "type": "pose",
                    "accepted": False,
                    "recoverable": True,
                    "reason": "anchor_missing",
                    "message": "请先按住 Grip 建立控制起点",
                }
            # The adapter owns these dictionaries and replaces them only on a
            # new anchor. Shallow list copies are sufficient and avoid a
            # recursive deepcopy for every controller sample.
            anchor_tcp = {
                "position_m": list(self._anchor_tcp["position_m"]),
                "orientation_xyzw": list(self._anchor_tcp["orientation_xyzw"]),
            }
        anchor_controller_position = list(self._anchor_controller["position_m"])
        anchor_controller_orientation = list(self._anchor_controller["orientation_xyzw"])
        # Preserve the original relative position mapping: the configured
        # matrix maps PICO tracking coordinates directly into the robot frame.
        mapped_position = _map(self._position_axis_map, position, self._translation_gain)
        orientation_correction = list(self._orientation_correction)
        mapped_anchor_orientation = _map_absolute_orientation(self._orientation_axis_map, anchor_controller_orientation)
        relative_orientation = _normalise(_multiply(_inverse(mapped_anchor_orientation), mapped_orientation))
        attenuated_orientation = _attenuate_relative_rotation(relative_orientation, self._rotation_gain)
        mapped_q = _normalise(_multiply(orientation_correction, _multiply(mapped_anchor_orientation, attenuated_orientation)))
        target = {"position_m": [self._absolute_position_offset[index] + mapped_position[index] for index in range(3)],
                  "orientation_xyzw": mapped_q}
        with self.lock:
            self.state.update({"mapped_input_orientation_xyzw": list(mapped_q),
                               "mapped_input_rotation_degrees": _rotation_degrees(mapped_q),
                               "target_rotation_degrees": _rotation_degrees(mapped_q),
                               "orientation_command_enabled": True,
                               "last_target_pose": copy.deepcopy(target), "last_target_status": "SUBMITTING",
                               "last_target_sent_at": time.time()})
        osc_submit_started_ns = time.monotonic_ns()
        result = self._command("track_tcp", {"target_pose": target})
        osc_submit_finished_ns = time.monotonic_ns()
        nested_result = result.get("result") if isinstance(result.get("result"), dict) else result
        with self.lock:
            self.state.update({"last_target_status": "ACCEPTED" if result.get("ok") else "REJECTED",
                               "last_target_osc_sequence": nested_result.get("accepted_sequence")})
        if self.trace_logger:
            self.trace_logger.append({"record_type": "sample", "event": "adapter_osc_return",
                                      "monotonic_ns": osc_submit_finished_ns, "pico_sequence": pico_sequence,
                                      "osc_sequence": result.get("result", {}).get("accepted_sequence") if isinstance(result.get("result"), dict) else result.get("accepted_sequence"),
                                      "ok": result.get("ok"), "recoverable": result.get("recoverable", False)})
        if not result.get("ok"):
            rejected = result.get("result") if isinstance(result.get("result"), dict) else result
            if bool(rejected.get("recoverable")):
                # A workspace/safety rejection is an input condition, not a
                # transport failure. Keep the anchor and pairing alive so the
                # next pose can recover when the operator moves back inside
                # the valid workspace.
                with self.lock:
                    self.state.update({
                        "state": "TRACKING_LIMITED",
                        "tracking_valid": True,
                        "last_error": str(rejected.get("reason", "target rejected")),
                        "updated_at": time.time(),
                    })
                return {
                    "ok": True,
                    "type": "pose",
                    "accepted": False,
                    "recoverable": True,
                    "reason": str(rejected.get("reason", "target rejected")),
                    "message": "目标超出当前安全工作空间，请向反方向移动",
                    "safe_target_pose": rejected.get("safe_target_pose"),
                }
            raise RuntimeError(f"OSC rejected PICO target: {result}")
        timing = {
            "gateway_to_adapter_ms": max(0.0, (adapter_started_ns - gateway_received_ns) / 1e6),
            "adapter_compute_ms": max(0.0, (osc_submit_started_ns - adapter_started_ns) / 1e6),
            "osc_submit_ms": max(0.0, (osc_submit_finished_ns - osc_submit_started_ns) / 1e6),
            "adapter_total_ms": max(0.0, (osc_submit_finished_ns - gateway_received_ns) / 1e6),
            "monotonic_ns": osc_submit_finished_ns,
        }
        with self.lock:
            self.state.update({"state": "TRACKING", "tracking_valid": True, "updated_at": time.time(), "last_error": None,
                               "last_pose_timing_ms": dict(timing)})
        ack = self._ack("pose", result)
        ack["timing_ms"] = {key: value for key, value in timing.items() if key != "monotonic_ns"}
        return ack

    def reset_anchor(self, session_id: str, client_id: str) -> dict[str, Any]:
        """Rebase the original position anchor without changing coordinate frames."""
        osc = self.osc.state() or {}
        session = osc.get("session") or {}
        with self.lock:
            authorized = (session_id == self._session_id and client_id == self._client_id and
                          session.get("state") == "ACTIVE" and session.get("id") == session_id and
                          session.get("client_id") == client_id)
            pose = {"position_m": list(self.state.get("input_position_m") or []),
                    "orientation_xyzw": list(self.state.get("input_orientation_xyzw") or []),
                    "tracking_valid": bool(self.state.get("tracking_valid")),
                    "_pico_sequence": int(self.state.get("input_sequence") or 0)}
        if not authorized:
            raise PermissionError("只有当前控制台/PICO 会话所有者可以重置初始位置")
        if not pose["tracking_valid"] or len(pose["position_m"]) != 3 or len(pose["orientation_xyzw"]) != 4:
            return {"ok": True, "accepted": False, "reason": "pico_pose_unavailable", "message": "等待有效的 PICO 位姿信号"}
        result = self.anchor_begin(pose, activate_clutch=False)
        result["type"] = "anchor_reset"
        result["message"] = "初始位置已重置；不会立即移动机械臂，后续只跟随手柄相对变化"
        return result

    def gripper(self, value: Any) -> dict[str, Any]:
        fraction = max(0.0, min(1.0, float(value)))
        width = self._gripper_open_width_m * (1.0 - fraction)
        result = self._command("gripper", {"mode": "position", "width_m": width, "force_n": self._gripper_force_n})
        with self.lock:
            self.state.update({"gripper_position": fraction, "updated_at": time.time()})
        return self._ack("gripper", result)

    def stop(self, reason: str) -> dict[str, Any]:
        with self.lock:
            self._anchor_controller = self._anchor_tcp = None
            self._absolute_position_offset = None
            self._orientation_correction = [0.0, 0.0, 0.0, 1.0]
            self._orientation_calibration_status = "DIRECT_MAPPING"
            self._orientation_calibrated_at = None
            self._pending_anchor = None
            session_id, client_id = self._session_id, self._client_id
            self.state.update({"anchor_active": False, "clutch_signal": False, "input_clutch": False,
                               "orientation_command_enabled": False,
                               "orientation_calibration_status": "DIRECT_MAPPING",
                               "orientation_correction_xyzw": [0.0, 0.0, 0.0, 1.0],
                               "orientation_correction_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                               "orientation_calibrated_at": None,
                               "updated_at": time.time()})
        osc_session = (self.osc.state().get("session") or {}) if session_id else {}
        session_is_current = bool(session_id and osc_session.get("state") == "ACTIVE" and
                                 osc_session.get("id") == session_id and osc_session.get("client_id") == client_id)
        result = self._command("hold", {"reason": reason}) if session_is_current else {"ok": True, "already_stopped": True}
        with self.lock:
            self.state.update({"state": "READY" if self._session_id else "IDLE", "anchor_active": False, "tracking_valid": False, "updated_at": time.time(), "last_error": None})
        return self._ack("hold", result)

    def disconnected(self, reason: str) -> None:
        self._pairing_stop.set()
        try:
            self.stop(reason)
        except Exception as exc:
            with self.lock: self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self._session_id = self._client_id = None
            self._execution_mode = None
            self._hardware_high_gain_confirmed = False
            self.state.update({"state": "IDLE", "session_id": None, "connected": False, "paired": False,
                               "anchor_active": False, "updated_at": time.time()})
