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


class PicoInputAdapter:
    """Convert raw headset input to absolute base-frame OSC targets."""

    def __init__(self, osc: Any, config: dict[str, Any]) -> None:
        defaults = {"mapping_verified": True, "translation_gain": 1.0, "rotation_gain": 1.0,
                    "position_axis_map": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    "orientation_axis_map": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    "gripper_open_width_m": 0.095, "gripper_force_n": 1.0}
        defaults.update(config or {})
        self.osc, self.config = osc, defaults
        self.lock = threading.RLock()
        self.state: dict[str, Any] = self._empty_state()
        self._session_id: str | None = None
        self._client_id: str | None = None
        self._anchor_controller: dict[str, list[float]] | None = None
        self._anchor_tcp: dict[str, list[float]] | None = None
        self._sequence = 0
        self._pairing_stop = threading.Event()
        self._pairing_thread: threading.Thread | None = None

    def _empty_state(self) -> dict[str, Any]:
        return {"adapter": "pico", "state": "IDLE", "session_id": None,
                "connected": False, "paired": False, "tracking_valid": False,
                "anchor_active": False, "last_input_age_s": None,
                "last_error": None, "gripper_position": None, "updated_at": time.time()}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = copy.deepcopy(self.state)
            if self.state["updated_at"]:
                result["last_input_age_s"] = max(0.0, time.time() - self.state["updated_at"])
            result["mapping"] = {"translation_gain": self.config.get("translation_gain", 1.0),
                                 "rotation_gain": self.config.get("rotation_gain", 1.0),
                                 "verified": bool(self.config.get("mapping_verified", False))}
            return result

    def begin_pairing(self, session_id: str, client_id: str) -> None:
        osc = self.osc.state(); session = osc.get("session") or {}
        if session.get("state") != "ACTIVE" or session.get("id") != session_id or session.get("client_id") != client_id:
            raise PermissionError("PICO requires the caller's active OSC session")
        with self.lock:
            self._session_id, self._client_id = session_id, client_id
            self._anchor_controller = self._anchor_tcp = None
            self._sequence = int((osc.get("command") or {}).get("sequence") or 0)
            self.state = self._empty_state()
            self.state.update({"state": "PAIRING", "session_id": session_id, "updated_at": time.time()})
            self._pairing_stop.set()
            self._pairing_stop = threading.Event()
            self._pairing_thread = threading.Thread(target=self._pairing_heartbeat_loop, name="nero-pico-pairing-heartbeat", daemon=True)
            self._pairing_thread.start()

    def paired(self) -> None:
        with self.lock:
            if not self._session_id:
                raise RuntimeError("PICO pairing has not been started")
            self.state.update({"state": "READY", "connected": True, "paired": True, "updated_at": time.time(), "last_error": None})
            self._pairing_stop.set()

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
        if not self._session_id or not self._client_id:
            raise RuntimeError("PICO has no bound OSC session")
        self._sequence = max(self._sequence + 1, int(time.time() * 1000))
        if kind == "track_tcp":
            return self.osc.track_tcp(self._session_id, self._client_id, self._sequence, payload["target_pose"])
        if kind == "hold":
            return self.osc.hold(self._session_id, self._client_id, self._sequence, str(payload.get("reason", "PICO HOLD")))
        if kind == "gripper":
            return self.osc.gripper(self._session_id, self._client_id, self._sequence, payload)
        raise ValueError(f"unsupported PICO OSC command {kind}")

    def heartbeat(self) -> None:
        with self.lock:
            if self._session_id and self._client_id:
                self.osc.heartbeat(self._client_id, self._session_id)
                self.state["updated_at"] = time.time()

    def anchor_begin(self, pose: dict[str, Any]) -> dict[str, Any]:
        position, orientation = _vector(pose.get("position_m"), 3, "controller position"), _normalise(_vector(pose.get("orientation_xyzw"), 4, "controller orientation"))
        osc = self.osc.state()
        tcp = (osc.get("execution") or {}).get("measured_tcp_pose") or (osc.get("command") or {}).get("target_tcp")
        if not isinstance(tcp, dict):
            raise RuntimeError("OSC did not publish a current TCP pose")
        with self.lock:
            self._anchor_controller = {"position_m": position, "orientation_xyzw": orientation}
            self._anchor_tcp = {"position_m": _vector(tcp.get("position_m"), 3, "TCP position"), "orientation_xyzw": _normalise(_vector(tcp.get("orientation_xyzw"), 4, "TCP orientation"))}
            self.state.update({"state": "TRACKING", "anchor_active": True, "tracking_valid": True, "updated_at": time.time(), "last_error": None})
            return self.snapshot()

    def pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        if not bool(pose.get("tracking_valid", True)):
            return self.stop("PICO controller tracking lost")
        position, orientation = _vector(pose.get("position_m"), 3, "controller position"), _normalise(_vector(pose.get("orientation_xyzw"), 4, "controller orientation"))
        with self.lock:
            if not self._anchor_controller or not self._anchor_tcp:
                raise RuntimeError("PICO pose rejected: define Anchor with right Grip first")
            translation = [position[index] - self._anchor_controller["position_m"][index] for index in range(3)]
            mapped = _map(self.config["position_axis_map"], translation, float(self.config.get("translation_gain", 1.0)))
            relative_q = _multiply(_inverse(self._anchor_controller["orientation_xyzw"]), orientation)
            # PICO and NERO frame mapping is represented as a signed axis map;
            # apply it to the quaternion vector part and re-normalise.
            xyz = _map(self.config["orientation_axis_map"], relative_q[:3], float(self.config.get("rotation_gain", 1.0)))
            mapped_q = _normalise([*xyz, relative_q[3]])
            target = {"position_m": [self._anchor_tcp["position_m"][index] + mapped[index] for index in range(3)],
                      "orientation_xyzw": _normalise(_multiply(self._anchor_tcp["orientation_xyzw"], mapped_q))}
            result = self._command("track_tcp", {"target_pose": target})
            if not result.get("ok"):
                raise RuntimeError(f"OSC rejected PICO target: {result}")
            self.state.update({"state": "TRACKING", "tracking_valid": True, "updated_at": time.time(), "last_error": None})
            return result

    def gripper(self, value: Any) -> dict[str, Any]:
        fraction = max(0.0, min(1.0, float(value)))
        width = float(self.config.get("gripper_open_width_m", 0.095)) * (1.0 - fraction)
        with self.lock:
            result = self._command("gripper", {"mode": "position", "width_m": width, "force_n": float(self.config.get("gripper_force_n", 1.0))})
            self.state.update({"gripper_position": fraction, "updated_at": time.time()})
            return result

    def stop(self, reason: str) -> dict[str, Any]:
        with self.lock:
            self._anchor_controller = self._anchor_tcp = None
            osc_session = (self.osc.state().get("session") or {}) if self._session_id else {}
            session_is_current = bool(
                self._session_id
                and osc_session.get("state") == "ACTIVE"
                and osc_session.get("id") == self._session_id
                and osc_session.get("client_id") == self._client_id
            )
            result = self._command("hold", {"reason": reason}) if session_is_current else {"ok": True, "already_stopped": True}
            self.state.update({"state": "READY" if self._session_id else "IDLE", "anchor_active": False, "tracking_valid": False, "updated_at": time.time(), "last_error": None})
            return result

    def disconnected(self, reason: str) -> None:
        self._pairing_stop.set()
        try:
            self.stop(reason)
        except Exception as exc:
            with self.lock: self.state["last_error"] = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self._session_id = self._client_id = None
            self.state.update({"state": "IDLE", "session_id": None, "connected": False, "paired": False,
                               "anchor_active": False, "updated_at": time.time()})
