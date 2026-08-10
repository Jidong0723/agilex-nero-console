"""Deterministic shadow benchmark for the OSC Pink differential-IK settings.

The benchmark implements ``OSC控制器简化测试与综合评价指标.md``:
one 6D, 80%-speed outbound motion, a 30 s hold, the reverse motion and a
second 30 s hold.  It operates only on the project's Pinocchio/Pink model;
no OSC session, CAN transport or robot hardware is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from motion.teleop_kinematics_server import Solver


CONFIG_PATH = ROOT / "config" / "teleop.json"
DT_S = 0.02
HOLD_S = 10.0
DELTA_POSITION_M = np.array([0.050, -0.040, 0.040])
DELTA_RPY_RAD = np.deg2rad([12.0, -10.0, 15.0])


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate SO(3) by its logarithmic map without quaternion ordering."""
    from scipy.spatial.transform import Rotation

    relative = start.T @ end
    return start @ Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec() * fraction).as_matrix()


def _rotation_error_rad(actual: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(actual.T @ target) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _trapezoid_fraction(progress: float) -> float:
    """Normalized 25% accelerate / 50% cruise / 25% decelerate profile."""
    progress = max(0.0, min(1.0, progress))
    if progress <= 0.25:
        return (8.0 / 3.0) * progress * progress
    if progress < 0.75:
        return 1.0 / 6.0 + (4.0 / 3.0) * (progress - 0.25)
    return 1.0 - (8.0 / 3.0) * (1.0 - progress) ** 2


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pose(solver: Solver, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = solver.fk(joints.tolist())
    return np.asarray(data["position_m"], dtype=float), np.asarray(data["rotation"], dtype=float)


def _score(track_position_m: list[float], track_orientation_rad: list[float], holds: list[dict[str, Any]]) -> dict[str, Any]:
    position_rms_m = float(np.sqrt(np.mean(np.square(track_position_m))))
    orientation_rms_rad = float(np.sqrt(np.mean(np.square(track_orientation_rad))))
    position_track = _clip(1.0 - position_rms_m / 0.010)
    orientation_track = _clip(1.0 - orientation_rms_rad / math.radians(5.0))
    s_track = 0.5 * (position_track + orientation_track)
    hold_scores = [
        0.5
        * (
            _clip(1.0 - item["position_drift_m"] / 0.003)
            + _clip(1.0 - item["orientation_drift_rad"] / math.radians(1.5))
        )
        for item in holds
    ]
    worst_position_drift_m = max(item["position_drift_m"] for item in holds)
    worst_orientation_drift_rad = max(item["orientation_drift_rad"] for item in holds)
    worst_joint_drift_rad = max(item["joint_drift_rad"] for item in holds)
    s_hold = min(hold_scores)
    s_null = _clip(1.0 - worst_joint_drift_rad / math.radians(2.0))
    forced_failures: list[str] = []
    if worst_joint_drift_rad > math.radians(5.0):
        forced_failures.append("hold joint drift exceeds 5 deg")
    if worst_position_drift_m > 0.010:
        forced_failures.append("hold TCP position drift exceeds 10 mm")
    if worst_orientation_drift_rad > math.radians(5.0):
        forced_failures.append("hold TCP orientation drift exceeds 5 deg")
    # The updated specification uses a weighted negative fourth-power mean.
    # It penalizes a weak tracking, hold, or null-space score strongly rather
    # than allowing the other two dimensions to compensate for it.
    score = 0.0 if min(s_track, s_hold, s_null) <= 0.0 else 100.0 * (
        0.60 * s_track ** -4 + 0.25 * s_hold ** -4 + 0.15 * s_null ** -4
    ) ** -0.5
    return {
        "osc_score": score if not forced_failures else 0.0,
        "raw_osc_score": score,
        "pass": not forced_failures and score > 95.0,
        "forced_failures": forced_failures,
        "track": {
            "position_rms_mm": position_rms_m * 1000.0,
            "orientation_rms_deg": math.degrees(orientation_rms_rad),
            "score": s_track,
        },
        "hold": {
            "worst_position_drift_mm": worst_position_drift_m * 1000.0,
            "worst_orientation_drift_deg": math.degrees(worst_orientation_drift_rad),
            "score": s_hold,
            "phase_scores": hold_scores,
        },
        "nullspace": {
            "worst_joint_drift_deg": math.degrees(worst_joint_drift_rad),
            "score": s_null,
        },
    }


def run(settings: dict[str, float]) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    runtime_config = json.loads((ROOT / "config" / "runtime.json").read_text(encoding="utf-8-sig"))
    solver_config = config["solver"]
    settings = {
        "posture_cost": float(solver_config.get("posture_cost", 0.005)),
        "damping_cost": float(solver_config.get("damping_cost", 0.05)),
        "frame_position_cost": float(solver_config.get("frame_position_cost", 1.0)),
        "frame_orientation_cost": float(solver_config.get("frame_orientation_cost", 1.0)),
        "frame_gain": float(solver_config.get("frame_gain", 1.0)),
        "frame_lm_damping": float(solver_config.get("frame_lm_damping", 0.0)),
        "joint_center_cost": float(solver_config.get("joint_center_cost", 0.0)),
        "joint_center_deadband": float(solver_config.get("joint_center_deadband", 0.70)),
        **settings,
    }
    solver = Solver(
        ROOT / solver_config["urdf"],
        [float(value) for value in runtime_config["sdk"]["task_tcp_offset_from_flange_m"]],
        DT_S,
    )
    q_start = np.asarray(config["shadow_initial_joints_rad"], dtype=float)
    p_start, r_start = _pose(solver, q_start)
    p_goal = p_start + DELTA_POSITION_M
    r_goal = r_start @ _rotation_from_rpy(DELTA_RPY_RAD)
    # Benchmark input trajectory only; it is not an OSC runtime limit.
    max_linear = 0.048
    max_angular = float(config["limits"]["angular_speed_rad_s"]) * 0.80
    duration = max(
        # The benchmark synchronizes translation and rotation using the norm
        # of each 6D input component; these are test-trajectory rates only.
        float(np.linalg.norm(DELTA_POSITION_M)) / max_linear,
        float(np.linalg.norm(DELTA_RPY_RAD)) / max_angular,
    ) * (4.0 / 3.0)
    soft_margin = np.asarray(config["safety_supervisor"]["fixed_model_margin_rad"], dtype=float)
    soft_lower = (solver.hard_lower + soft_margin).tolist()
    soft_upper = (solver.hard_upper - soft_margin).tolist()
    q = q_start.copy()
    previous_velocity = np.zeros(7)
    track_position_m: list[float] = []
    track_orientation_rad: list[float] = []
    holds: list[dict[str, Any]] = []

    def advance(target_position: np.ndarray, target_rotation: np.ndarray, record_track: bool) -> None:
        nonlocal q, previous_velocity
        # scipy accepts a scalar-last quaternion.  Pinocchio receives xyzw
        # from the controller and the Solver owns its conversion to wxyz.
        from scipy.spatial.transform import Rotation

        target_xyzw = Rotation.from_matrix(target_rotation).as_quat().tolist()
        result = solver.solve(
            {
                "joint_angles_rad": q.tolist(),
                "target_position_m": target_position.tolist(),
                "target_orientation_xyzw": target_xyzw,
                "last_sent_joint_velocity_rad_s": previous_velocity.tolist(),
                "joint_speed_limit_rad_s": [float(limits["joint_speed_rad_s"])] * 7,
                "joint_acceleration_limit_rad_s2": [
                    float(settings.get("joint_acceleration_limit_rad_s2", solver_config["ruckig_max_acceleration"]))
                ] * 7,
                "soft_lower_rad": soft_lower,
                "soft_upper_rad": soft_upper,
                "posture_reference_rad": q_start.tolist(),
                "posture_cost": settings["posture_cost"],
                "damping_cost": settings["damping_cost"],
                "frame_position_cost": settings["frame_position_cost"],
                "frame_orientation_cost": settings["frame_orientation_cost"],
                "frame_gain": settings["frame_gain"],
                "frame_lm_damping": settings["frame_lm_damping"],
                "joint_center_cost": settings["joint_center_cost"],
                "joint_center_deadband": settings["joint_center_deadband"],
                "condition_limit": float(limits["singularity_condition_max"]),
                "dt_s": DT_S,
            }
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Pink solve failed"))
        previous_velocity = np.asarray(result["pink_joint_velocity_rad_s"], dtype=float)
        q = q + previous_velocity * DT_S
        # The OSC state snapshot is published after the final joint position
        # for this period has been integrated/applied.  Score that applied
        # state against this period's reference, rather than against the
        # feedback sample that predates the output by one control tick.
        if record_track:
            actual_position, actual_rotation = _pose(solver, q)
            track_position_m.append(float(np.linalg.norm(actual_position - target_position)))
            track_orientation_rad.append(_rotation_error_rad(actual_rotation, target_rotation))

    movement_steps = int(round(duration / DT_S))
    hold_steps = int(round(HOLD_S / DT_S))
    for step in range(1, movement_steps + 1):
        fraction = _trapezoid_fraction(step / movement_steps)
        advance(p_start + DELTA_POSITION_M * fraction, _slerp(r_start, r_goal, fraction), True)

    def settle(target_position: np.ndarray, target_rotation: np.ndarray) -> None:
        """Keep the normal control law running until the scoring hold starts."""
        stable_steps = int(round(float(config.get("osc", {}).get("arrival_dwell_s", 0.25)) / DT_S))
        max_steps = int(round(float(config.get("osc", {}).get("arrival_timeout_s", 5.0)) / DT_S))
        position_tolerance = float(config.get("osc", {}).get("arrival_position_tolerance_m", 0.0005))
        orientation_tolerance = float(config.get("osc", {}).get("arrival_orientation_tolerance_rad", math.radians(0.25)))
        velocity_tolerance = float(config.get("osc", {}).get("arrival_joint_velocity_tolerance_rad_s", 0.005))
        consecutive = 0
        for _ in range(max_steps):
            advance(target_position, target_rotation, True)
            actual_position, actual_rotation = _pose(solver, q)
            position_error = float(np.linalg.norm(actual_position - target_position))
            orientation_error = _rotation_error_rad(actual_rotation, target_rotation)
            if position_error <= position_tolerance and orientation_error <= orientation_tolerance and float(np.max(np.abs(previous_velocity))) <= velocity_tolerance:
                consecutive += 1
                if consecutive >= stable_steps:
                    return
            else:
                consecutive = 0
        raise RuntimeError("ideal Pink controller did not reach the target before the arrival timeout")

    for hold_index, (target_position, target_rotation) in enumerate(((p_goal, r_goal), (p_start, r_start))):
        settle(target_position, target_rotation)
        q_hold_start = q.copy()
        p_hold_start, r_hold_start = _pose(solver, q)
        max_position_drift = max_orientation_drift = max_joint_drift = 0.0
        for _ in range(hold_steps):
            advance(target_position, target_rotation, False)
            p_now, r_now = _pose(solver, q)
            max_position_drift = max(max_position_drift, float(np.linalg.norm(p_now - p_hold_start)))
            max_orientation_drift = max(max_orientation_drift, _rotation_error_rad(r_now, r_hold_start))
            max_joint_drift = max(max_joint_drift, float(np.max(np.abs(q - q_hold_start))))
        holds.append(
            {
                "phase": "goal_hold" if hold_index == 0 else "return_hold",
                "position_drift_m": max_position_drift,
                "orientation_drift_rad": max_orientation_drift,
                "joint_drift_rad": max_joint_drift,
            }
        )
        if hold_index == 0:
            for step in range(1, movement_steps + 1):
                fraction = _trapezoid_fraction(step / movement_steps)
                advance(p_goal - DELTA_POSITION_M * fraction, _slerp(r_goal, r_start, fraction), True)
    report = _score(track_position_m, track_orientation_rad, holds)
    report.update({"settings": settings, "trajectory_duration_s": duration, "profile": "trapezoid: 25% accelerate, 50% cruise, 25% decelerate", "samples": {"track": len(track_position_m), "hold_each": hold_steps}})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posture-cost", type=float)
    parser.add_argument("--damping-cost", type=float)
    parser.add_argument("--frame-position-cost", type=float)
    parser.add_argument("--frame-orientation-cost", type=float)
    parser.add_argument("--frame-gain", type=float)
    parser.add_argument("--frame-lm-damping", type=float)
    parser.add_argument("--joint-center-cost", type=float)
    parser.add_argument("--joint-center-deadband", type=float)
    parser.add_argument("--joint-acceleration-limit", type=float, default=None)
    args = parser.parse_args()
    settings = {
        key: value
        for key, value in {
            "posture_cost": args.posture_cost,
            "damping_cost": args.damping_cost,
            "frame_position_cost": args.frame_position_cost,
            "frame_orientation_cost": args.frame_orientation_cost,
            "frame_gain": args.frame_gain,
            "frame_lm_damping": args.frame_lm_damping,
            "joint_center_cost": args.joint_center_cost,
            "joint_center_deadband": args.joint_center_deadband,
        }.items()
        if value is not None
    }
    if args.joint_acceleration_limit is not None:
        settings["joint_acceleration_limit_rad_s2"] = args.joint_acceleration_limit
    print(json.dumps(run(settings), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
