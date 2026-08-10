"""Run the OSC scoring trajectory through the real shadow-mode HTTP chain.

Unlike ``osc_pink_benchmark.py``, this exercises the service process,
asynchronous Pink worker, stale-result handling, final safety gate and shadow
state publication.  It never opens CAN or sends a hardware command.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8765"
CLIENT_ID = "osc-runtime-shadow-benchmark"
CONFIG_PATH = ROOT / "config" / "teleop.json"
DT_S = 0.02
HOLD_S = 10.0
STATE_SAMPLE_STRIDE = 5  # 10 Hz; state RPC must not throttle 50 Hz commands.
DELTA_POSITION_M = [0.050, -0.040, 0.040]
DELTA_RPY_RAD = [math.radians(12.0), math.radians(-10.0), math.radians(15.0)]


def _request(path: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=payload, method=method,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    # First Pink process startup is allowed to warm its numerical backend.
    # Subsequent control requests remain bounded by the OSC control budget.
    with urllib.request.urlopen(request, timeout=20.0) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not decoded.get("ok"):
        raise RuntimeError(decoded.get("error", f"request {path} failed"))
    return decoded["data"]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unit_quaternion(value: list[float]) -> list[float]:
    magnitude = _norm(value)
    return [item / magnitude for item in value]


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left; rx, ry, rz, rw = right
    return _unit_quaternion([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def _rpy_quaternion(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = (value * 0.5 for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return _unit_quaternion([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _slerp(start: list[float], end: list[float], fraction: float) -> list[float]:
    cosine = sum(left * right for left, right in zip(start, end))
    end = list(end)
    if cosine < 0.0:
        cosine, end = -cosine, [-value for value in end]
    if cosine > 0.9995:
        return _unit_quaternion([left + fraction * (right - left) for left, right in zip(start, end)])
    angle = math.acos(_clip(cosine))
    sin_angle = math.sin(angle)
    return _unit_quaternion([
        math.sin((1.0 - fraction) * angle) / sin_angle * left + math.sin(fraction * angle) / sin_angle * right
        for left, right in zip(start, end)
    ])


def _orientation_error(actual: list[float], target: list[float]) -> float:
    cosine = abs(sum(left * right for left, right in zip(_unit_quaternion(actual), _unit_quaternion(target))))
    return 2.0 * math.acos(_clip(cosine))


def _trapezoid_fraction(progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    if progress <= 0.25:
        return (8.0 / 3.0) * progress * progress
    if progress < 0.75:
        return 1.0 / 6.0 + (4.0 / 3.0) * (progress - 0.25)
    return 1.0 - (8.0 / 3.0) * (1.0 - progress) ** 2


def _score(track_position_m: list[float], track_orientation_rad: list[float], holds: list[dict[str, Any]]) -> dict[str, Any]:
    position_rms = math.sqrt(sum(value * value for value in track_position_m) / len(track_position_m))
    orientation_rms = math.sqrt(sum(value * value for value in track_orientation_rad) / len(track_orientation_rad))
    s_track = 0.5 * (_clip(1.0 - position_rms / 0.010) + _clip(1.0 - orientation_rms / math.radians(5.0)))
    hold_scores = [0.5 * (_clip(1.0 - item["position_drift_m"] / 0.003) + _clip(1.0 - item["orientation_drift_rad"] / math.radians(1.5))) for item in holds]
    s_hold = min(hold_scores)
    max_joint = max(item["joint_drift_rad"] for item in holds)
    s_null = _clip(1.0 - max_joint / math.radians(2.0))
    score = 0.0 if min(s_track, s_hold, s_null) <= 0.0 else 100.0 * (0.60 * s_track ** -4 + 0.25 * s_hold ** -4 + 0.15 * s_null ** -4) ** -0.5
    return {"osc_score": score, "pass": score > 97.0, "track": {"position_rms_mm": position_rms * 1000.0, "orientation_rms_deg": math.degrees(orientation_rms), "score": s_track}, "hold": {"score": s_hold, "phase_scores": hold_scores, "worst_position_drift_mm": max(item["position_drift_m"] for item in holds) * 1000.0, "worst_orientation_drift_deg": math.degrees(max(item["orientation_drift_rad"] for item in holds))}, "nullspace": {"score": s_null, "worst_joint_drift_deg": math.degrees(max_joint)}}


def _pose(pose: dict[str, Any]) -> tuple[list[float], list[float]]:
    return [float(value) for value in pose["position_m"]], _unit_quaternion([float(value) for value in pose["orientation_xyzw"]])


def run(static_only: bool = False) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    delta_position = [0.0, 0.0, 0.0] if static_only else DELTA_POSITION_M
    delta_rpy = [0.0, 0.0, 0.0] if static_only else DELTA_RPY_RAD
    # Benchmark input trajectory only; it is not an OSC runtime limit.
    max_linear = 0.048
    max_angular = float(config["limits"]["angular_speed_rad_s"]) * 0.80
    duration = (4.0 / 3.0) * max(
        _norm(delta_position) / max_linear,
        _norm(delta_rpy) / max_angular,
    )
    movement_steps = max(1, int(round(duration / DT_S)))
    hold_steps = int(round(HOLD_S / DT_S))
    started: dict[str, Any] | None = None
    sequence = 0
    track_position_m: list[float] = []
    track_orientation_rad: list[float] = []
    holds: list[dict[str, Any]] = []
    control_periods_s: list[float] = []

    def state() -> dict[str, Any]:
        return _request("/api/osc/state")

    def heartbeat(session_id: str) -> None:
        _request("/api/osc/session/heartbeat", "POST", {"client_id": CLIENT_ID, "session_id": session_id})

    def command(session_id: str, position: list[float], orientation: list[float]) -> None:
        nonlocal sequence
        sequence += 1
        _request("/api/osc/command", "POST", {
            "session_id": session_id,
            "client_id": CLIENT_ID,
            "sequence": sequence,
            "type": "track_tcp",
            "acknowledgement_only": True,
            "payload": {"target_pose": {"position_m": position, "orientation_xyzw": orientation}},
        })

    def latest_actual(timeout_s: float = 2.0) -> tuple[list[float], list[float], list[float], dict[str, Any]]:
        # A session may need one Pink startup cycle before its first coherent
        # execution sample; later state reads normally return immediately.
        # A cold Pinocchio/Pink child can take several control periods before
        # it publishes its first coherent sample. This is benchmark setup,
        # not part of the measured tracking trajectory.
        deadline = time.monotonic() + timeout_s
        while True:
            snapshot = state()
            execution = snapshot.get("execution") or {}
            timing = (snapshot.get("diagnostics") or {}).get("timing") or {}
            if isinstance(timing.get("actual_dt_s"), (int, float)):
                control_periods_s.append(float(timing["actual_dt_s"]))
            tcp = execution.get("measured_tcp_pose")
            joints = execution.get("observed_joint_state_rad")
            if isinstance(tcp, dict) and isinstance(joints, list):
                position, rotation = _pose(tcp)
                return position, rotation, [float(value) for value in joints], snapshot
            if time.monotonic() >= deadline:
                raise RuntimeError(f"OSC shadow state did not publish current TCP and joints within {timeout_s:g} s")
            time.sleep(0.01)

    def warm_up_solver(session_id: str, position: list[float], orientation: list[float]) -> None:
        """Exclude cold numerical-process startup from the scored trajectory."""
        deadline = time.monotonic() + float(config["solver"].get("startup_timeout_s", 30.0)) + 2.0
        while time.monotonic() < deadline:
            command(session_id, position, orientation)
            heartbeat(session_id)
            try:
                latest_actual(timeout_s=0.10)
                return
            except RuntimeError:
                # Keep the session leased while the child imports Pinocchio.
                time.sleep(0.10)
        raise RuntimeError("Pink solver did not publish a first shadow sample during startup")

    def run_period(callback: Any, steps: int, session_id: str, sample_track: bool) -> None:
        next_tick = time.monotonic()
        last_heartbeat = next_tick
        for step in range(1, steps + 1):
            target_position, target_rotation = callback(step / steps)
            command(session_id, target_position, target_rotation)
            now = time.monotonic()
            if now - last_heartbeat >= 1.0:
                heartbeat(session_id)
                last_heartbeat = now
            next_tick += DT_S
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            if sample_track and (step % STATE_SAMPLE_STRIDE == 0 or step == steps):
                actual_position, actual_rotation, _, snapshot = latest_actual()
                error = (snapshot.get("diagnostics") or {}).get("tcp_error") or {}
                if isinstance(error.get("position_norm_m"), (int, float)) and isinstance(error.get("orientation_angle_rad"), (int, float)):
                    track_position_m.append(float(error["position_norm_m"]))
                    track_orientation_rad.append(float(error["orientation_angle_rad"]))
                else:
                    track_position_m.append(_norm([actual - target for actual, target in zip(actual_position, target_position)]))
                    track_orientation_rad.append(_orientation_error(actual_rotation, target_rotation))

    def wait_until_arrived(target_position: list[float], target_rotation: list[float], session_id: str) -> None:
        """Arrival remains part of tracking; hold starts only after it settles."""
        deadline = time.monotonic() + float(config.get("osc", {}).get("arrival_timeout_s", 5.0))
        next_tick = time.monotonic()
        last_heartbeat = next_tick
        while time.monotonic() < deadline:
            command(session_id, target_position, target_rotation)
            now = time.monotonic()
            if now - last_heartbeat >= 1.0:
                heartbeat(session_id)
                last_heartbeat = now
            next_tick += DT_S
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            actual_position, actual_rotation, _, snapshot = latest_actual()
            error = (snapshot.get("diagnostics") or {}).get("tcp_error") or {}
            track_position_m.append(float(error.get("position_norm_m", _norm([actual - target for actual, target in zip(actual_position, target_position)]))))
            track_orientation_rad.append(float(error.get("orientation_angle_rad", _orientation_error(actual_rotation, target_rotation))))
            if bool(((snapshot.get("diagnostics") or {}).get("arrival") or {}).get("reached")):
                return
        raise RuntimeError("OSC did not reach the target before the arrival timeout")

    def hold_period(phase: str, target_position: list[float], target_rotation: list[float], session_id: str) -> None:
        p_hold, r_hold, q_hold, _ = latest_actual()
        max_position = max_orientation = max_joint = 0.0
        next_tick = time.monotonic()
        last_heartbeat = next_tick
        for step in range(1, hold_steps + 1):
            command(session_id, target_position, target_rotation)
            now = time.monotonic()
            if now - last_heartbeat >= 1.0:
                heartbeat(session_id)
                last_heartbeat = now
            next_tick += DT_S
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            if step % STATE_SAMPLE_STRIDE == 0:
                p_now, r_now, q_now, _ = latest_actual()
                max_position = max(max_position, _norm([now - initial for now, initial in zip(p_now, p_hold)]))
                max_orientation = max(max_orientation, _orientation_error(r_now, r_hold))
                max_joint = max(max_joint, max(abs(now - initial) for now, initial in zip(q_now, q_hold)))
        snapshot = state()
        holds.append({
            "phase": phase,
            "position_drift_m": max_position,
            "orientation_drift_rad": max_orientation,
            "joint_drift_rad": max_joint,
            "end_solver_debug": (snapshot.get("solver") or {}).get("debug"),
            "end_trajectory_state": (snapshot.get("diagnostics") or {}).get("trajectory_state"),
        })

    try:
        started = _request("/api/osc/session/start", "POST", {"execution_mode": "shadow", "client_id": CLIENT_ID})
        session_id = started["session"]["id"]
        p_start, r_start = _pose(started["state"]["command"]["target_tcp"])
        p_goal = [start + delta for start, delta in zip(p_start, delta_position)]
        r_goal = _quaternion_multiply(r_start, _rpy_quaternion(delta_rpy))
        warm_up_solver(session_id, p_start, r_start)

        run_period(
            lambda x: ([start + delta * _trapezoid_fraction(x) for start, delta in zip(p_start, delta_position)], _slerp(r_start, r_goal, _trapezoid_fraction(x))),
            movement_steps, session_id, True,
        )
        wait_until_arrived(p_goal, r_goal, session_id)
        hold_period("goal_hold", p_goal, r_goal, session_id)
        run_period(
            lambda x: ([goal - delta * _trapezoid_fraction(x) for goal, delta in zip(p_goal, delta_position)], _slerp(r_goal, r_start, _trapezoid_fraction(x))),
            movement_steps, session_id, True,
        )
        wait_until_arrived(p_start, r_start, session_id)
        hold_period("return_hold", p_start, r_start, session_id)
        report = _score(track_position_m, track_orientation_rad, holds)
        report.update({
            "trajectory_duration_s": duration,
            "samples": {"track": len(track_position_m), "hold_each": hold_steps // STATE_SAMPLE_STRIDE},
            "holds": holds,
            "control_period_ms": {
                "mean": 1000.0 * sum(control_periods_s) / len(control_periods_s),
                "min": 1000.0 * min(control_periods_s),
                "max": 1000.0 * max(control_periods_s),
            },
            "transport": "shadow OSC HTTP chain; 50 Hz commands / 10 Hz state sampling",
        })
        return report
    finally:
        if started is not None:
            try:
                _request("/api/osc/session/stop", "POST", {"reason": "OSC runtime shadow benchmark complete"})
            except Exception:
                pass


if __name__ == "__main__":
    print(json.dumps(run(static_only="--static-only" in sys.argv), ensure_ascii=False, indent=2))
