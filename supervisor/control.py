from __future__ import annotations

import json
import secrets
import math
import threading
import time
from urllib.request import Request, urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .logging import JsonlExperimentLogger
from .authority import (
    CommandRevoked, CommandStream, ArmWriter, ControlSupervisor, HardwareTransportOwner,
    ServoMode, ServoWriteRevoked, TransportRobotProxy,
)
from nero_backend.robot import NeroRobot
from motion.safety import arm_status_is_no_solution_only
from shared.schemas import jsonable, now_iso
from motion.teleop import OperationalSpaceServo


GRIPPER_TIP_CANDIDATE_OFFSET_M = (0.175, 0.0, -0.0235)


def candidate_tool_pose_from_flange(flange_pose: Any) -> dict[str, Any] | None:
    """Return the unverified gripper-tip candidate expressed in the base frame."""
    if not isinstance(flange_pose, (list, tuple)) or len(flange_pose) != 6:
        return None
    try:
        values = [float(value) for value in flange_pose]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    roll, pitch, yaw = values[3:]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Matches the project's RPY convention: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    offset = GRIPPER_TIP_CANDIDATE_OFFSET_M
    position = [
        values[index] + sum(rotation[index][axis] * offset[axis] for axis in range(3))
        for index in range(3)
    ]
    return {
        "position_m": position,
        "rpy_rad": values[3:],
        "offset_from_flange_m": list(offset),
        "source": "NERO AGX gripper URDF candidate",
        "verified": False,
    }


class LeaseError(RuntimeError):
    pass


@dataclass
class ControlLease:
    token: str
    owner: str
    expires_monotonic: float
    acquired_at: str

    def public(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "acquired_at": self.acquired_at,
            "remaining_s": max(0.0, self.expires_monotonic - time.monotonic()),
        }


class LeaseManager:
    def __init__(self, default_ttl_s: float = 30.0) -> None:
        self.default_ttl_s = default_ttl_s
        self._lease: ControlLease | None = None
        self._lock = threading.RLock()

    def acquire(self, owner: str, ttl_s: float | None = None) -> ControlLease:
        with self._lock:
            if self._lease is not None:
                raise LeaseError(f"motion control is already leased to {self._lease.owner}")
            ttl = self._ttl(ttl_s)
            self._lease = ControlLease(secrets.token_urlsafe(24), owner, time.monotonic() + ttl, now_iso())
            return self._lease

    def require(self, token: str, extend_s: float | None = None) -> ControlLease:
        with self._lock:
            if (
                self._lease is None
                or self._lease.expires_monotonic <= time.monotonic()
                or not secrets.compare_digest(self._lease.token, token)
            ):
                raise LeaseError("motion lease is missing, expired, or owned by another client")
            self._lease.expires_monotonic = time.monotonic() + self._ttl(extend_s)
            return self._lease

    def release(self, token: str | None = None, force: bool = False) -> ControlLease | None:
        with self._lock:
            lease = self._lease
            if lease is None:
                return None
            if not force and (not token or not secrets.compare_digest(lease.token, token)):
                raise LeaseError("only the current lease owner may release the lease")
            self._lease = None
            return lease

    def current(self) -> ControlLease | None:
        with self._lock:
            if self._lease is not None and self._lease.expires_monotonic > time.monotonic():
                return self._lease
            return None

    def take_expired(self) -> ControlLease | None:
        with self._lock:
            if self._lease is not None and self._lease.expires_monotonic <= time.monotonic():
                expired = self._lease
                self._lease = None
                return expired
            return None

    def _ttl(self, ttl_s: float | None) -> float:
        return min(120.0, max(5.0, float(ttl_s or self.default_ttl_s)))


class OperationalSpaceController:
    """Sole USB-CAN owner and unified operational-space control facade."""

    def __init__(self, config: Path | str | dict[str, Any], robot: NeroRobot | None = None) -> None:
        backend = robot or NeroRobot(config)
        config_data = backend.config
        safety_config = config_data.get("safety", {})
        self.allow_electronic_emergency_stop = bool(
            safety_config.get("allow_electronic_emergency_stop", False)
        )
        service_config = config_data.get("control_service", {})
        self.leases = LeaseManager(float(service_config.get("lease_ttl_s", 30.0)))
        log_dir = Path(service_config.get("log_dir", "logs/control"))
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.logger = JsonlExperimentLogger(log_dir / f"control-service-{stamp}.jsonl", {"component": "nero_control_service"})
        self._running = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._log_lock = threading.Lock()
        self._action_lock = threading.RLock()
        self._handoff_lock = threading.RLock()
        self._authority_epoch_lock = threading.RLock()
        self._status_lock = threading.Lock()
        # The SDK/CAN object is not assumed thread-safe. Every active SDK call
        # goes through this one transport lock, including feedback reads.
        self._transport_lock = threading.RLock()
        self._robot_read_lock = self._transport_lock
        self._transport_owner = HardwareTransportOwner(backend)
        self._transport_reset_lock = threading.Lock()
        self._transport_reset_scheduled = False
        self._hardware_transport_available = True
        self.robot = TransportRobotProxy(self._transport_owner, backend)
        self.supervisor = ControlSupervisor()
        self._status_cache: tuple[float, dict[str, Any]] | None = None
        self._active_action: dict[str, Any] | None = None
        self._action_observers: list[Callable[..., None]] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.RLock()
        self._status_monitor: threading.Thread | None = None
        self._status_monitor_interval_s = 0.2
        self._last_sdk_read_duration_ms: float | None = None
        teleop_path = Path(__file__).resolve().parents[1] / "config" / "teleop.json"
        teleop_config = json.loads(teleop_path.read_text(encoding="utf-8-sig")) if teleop_path.is_file() else {}
        self.osc_servo = OperationalSpaceServo(self, Path(__file__).resolve().parents[1], teleop_config)
        # Legacy adapters and callers retain this name, but it is the same
        # single OSC servo instance, never a second control loop.
        self.teleop = self.osc_servo

    # ---- Sole transport-owner entry points ---------------------------------
    # Teleop and HTTP handlers deliberately use these methods rather than the
    # backend object.  The lock makes one SDK transaction active at a time and
    # the supervisor token prevents a stale real-time loop from writing after a
    # HOLD, FAULT, or Leader transition.

    def authority_status(self, control: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.supervisor.snapshot().public()
        control = control or {}
        raw_mode = str(control.get("mode") or "DISCONNECTED").upper()
        connected = bool(control.get("connected"))
        if raw_mode in {"FAULT", "DEGRADED", "EMERGENCY_DAMPING"}:
            hardware_mode = "FAULT"
        elif raw_mode in {"TRANSITIONING", "MODE_TRANSITION"}:
            hardware_mode = "TRANSITIONING"
        elif raw_mode in {"FREEDRIVE", "LEADER"}:
            hardware_mode = "FREEDRIVE"
        elif not connected or raw_mode in {"DISCONNECTED", "NONE"}:
            hardware_mode = "DISCONNECTED"
        else:
            # HOLD, FOLLOWER and TELEOP_CPV_VELOCITY are hardware execution
            # details within the same Follower control role.
            hardware_mode = "HOLD"

        if hardware_mode == "FREEDRIVE":
            control_role = "LEADER"
        elif hardware_mode == "HOLD":
            control_role = "FOLLOWER"
        else:
            control_role = "NONE"
        state.update({
            "hardware_mode": hardware_mode,
            "control_role": control_role,
            "safety_state": "FAULT" if hardware_mode == "FAULT" or state["arm_writer"] == "SAFETY" else "NORMAL",
            "feedback_fresh": bool(control.get("connected") and control.get("robot")),
        })
        return state

    def _set_authority(
        self,
        writer: ArmWriter,
        servo_mode: ServoMode,
        reason: str,
        *,
        command_stream: CommandStream | None = None,
        advance_epoch: bool = False,
        transport_exclusive_category: str | None = None,
    ) -> dict[str, Any]:
        with self._authority_epoch_lock:
            # Promote Transport Owner first.  Any P1/P2 envelope that was
            # queued under the old epoch is then rejected at the SDK boundary,
            # even if it is waiting behind another transport call.
            expected_epoch = None
            if advance_epoch:
                expected_epoch = self.supervisor.snapshot().epoch + 1
                self._transport_owner.advance_epoch(
                    expected_epoch,
                    exclusive_category=transport_exclusive_category,
                )
            state = self.supervisor.transition(
                writer,
                servo_mode,
                reason,
                command_stream=command_stream,
                advance_epoch=advance_epoch,
            )
            if expected_epoch is not None and state.epoch != expected_epoch:
                raise RuntimeError("Supervisor and Transport Owner epoch diverged")
        self._log("arm_write_authority", **state.public())
        return state.public()

    def prepare_teleop_hardware(self) -> dict[str, Any]:
        """Enter a fresh SERVO/HOLDING epoch without allowing old CPV output."""
        self._require_operational_control(allow_stale=True)
        with self._handoff_lock, self._transport_lock:
            mode = self.robot.get_control_state().get("mode")
            if mode == "FREEDRIVE":
                self._set_authority(ArmWriter.MODE_TRANSITION, ServoMode.SUSPENDED, "exit FREEDRIVE for teleop", advance_epoch=True)
                result = self.robot.hold_follower_without_position_target(
                    "teleop start exits FREEDRIVE"
                ).to_dict()
                if not result.get("ok"):
                    self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "FREEDRIVE exit failed")
                    raise RuntimeError(f"cannot exit FREEDRIVE: {result}")
            state = self._set_authority(ArmWriter.SERVO, ServoMode.HOLDING, "teleop session prepared", advance_epoch=True)
            # Prime CPV while the session is still in HOLDING. The first
            # vendor CPV call may enable drives and switch motion mode; doing
            # that with a measured position hold prevents the first live pose
            # sample from being a partially-dispatched batch.
            epoch = int(state["control_epoch"])
            try:
                self.robot.call(
                    "p1", "prime_cpv_position_from_feedback",
                    command_epoch=epoch,
                    category="servo_position",
                    execute_guard=lambda: self.supervisor.allows_servo(None, epoch),
                    dispatch_timeout_s=5.0,
                )
            except TimeoutError as exc:
                self._schedule_transport_reset(f"CPV position prime timeout: {exc}")
                raise RuntimeError(f"CPV prime timed out; control service reset scheduled: {exc}") from exc
            return state

    def _schedule_transport_reset(self, reason: str) -> None:
        """Ask the independent service reset path to quarantine a stuck SDK call."""
        with self._transport_reset_lock:
            if self._transport_reset_scheduled:
                return
            self._transport_reset_scheduled = True

        def request_reset() -> None:
            try:
                body = json.dumps({"reason": reason}, ensure_ascii=False).encode("utf-8")
                request = Request(
                    "http://127.0.0.1:8765/api/control/reset",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=1.0):
                    pass
            except Exception:
                # The external watchdog is best effort; the caller still
                # reports the transport failure and the operator can use Reset.
                pass

        threading.Thread(target=request_reset, name="transport-fault-reset", daemon=True).start()

    def read_teleop_joint_limits(self) -> dict[str, Any]:
        with self._transport_lock:
            return self.robot.read_teleop_joint_limits()

    def read_teleop_feedback(self) -> dict[str, Any]:
        with self._transport_lock:
            # Feedback is advisory for the servo loop; CPV output must win
            # whenever both requests are queued on the single CAN owner.
            return self.robot.call("p2", "read_teleop_feedback", category="teleop_feedback")

    def teleop_stream_active(self) -> bool:
        with self._transport_lock:
            value = self.robot.continuous_stream_active
            return bool(value() if callable(value) else value)

    def grant_teleop_tracking(self, session_id: str, epoch: int) -> bool:
        state = self.supervisor.snapshot()
        if state.writer is not ArmWriter.SERVO or state.epoch != int(epoch):
            return False
        self._set_authority(ArmWriter.SERVO, ServoMode.TRACKING, f"teleop tracking {session_id}")
        return True

    def mark_teleop_stopping(self, session_id: str, epoch: int, reason: str) -> bool:
        state = self.supervisor.snapshot()
        if state.writer is not ArmWriter.SERVO or state.epoch != int(epoch):
            return False
        self._set_authority(ArmWriter.SERVO, ServoMode.STOPPING, f"teleop stopping {session_id}: {reason}")
        return True

    def servo_can_write(self, session_id: str, epoch: int) -> bool:
        return self.supervisor.allows_servo(session_id, epoch)

    def send_servo_position(self, joints: list[float], session_id: str, epoch: int) -> dict[str, Any]:
        guard = lambda: self.supervisor.allows_servo(session_id, epoch)
        if not guard():
            raise ServoWriteRevoked("SERVO write authority is not valid for this teleop epoch")
        values = [float(value) for value in joints]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            self.trigger_safety_fault("non-finite or malformed servo position")
            raise ValueError("servo position must contain seven finite values")
        with self._transport_lock:
            if not guard():
                raise ServoWriteRevoked("SERVO write authority was revoked before transport dispatch")
            # The same guard is evaluated again by Transport Owner immediately
            # before the SDK call, so an old P1 batch cannot survive an epoch
            # revocation while it waits in the queue.  If the vendor call
            # blocks, quarantine the whole service instead of letting a
            # partially dispatched velocity batch survive into a new session.
            dispatch_timeout_s = float(
                self.robot.config.get("control_service", {}).get(
                    "servo_position_dispatch_timeout_s",
                    self.robot.config.get("control_service", {}).get("servo_velocity_dispatch_timeout_s", 0.75),
                )
            )
            try:
                return self.robot.call(
                    "p1", "send_cpv_position", values, command_epoch=epoch,
                    category="servo_position", execute_guard=guard,
                    dispatch_timeout_s=max(0.1, dispatch_timeout_s),
                )
            except TimeoutError as exc:
                self._schedule_transport_reset(f"CPV position timeout: {exc}")
                raise

    def begin_teleop_stop(self, reason: str) -> dict[str, Any]:
        """Atomically invalidate old output while leaving SERVO able to brake."""
        with self._handoff_lock:
            state = self._set_authority(ArmWriter.SERVO, ServoMode.STOPPING, reason, advance_epoch=True)
            self.teleop.freeze_for_authority_change(int(state["control_epoch"]), reason)
            return state

    def latch_teleop_hold(self, reason: str) -> dict[str, Any]:
        return self._set_authority(ArmWriter.SERVO, ServoMode.HOLDING, reason)

    def suspend_arm_writes(self, reason: str) -> dict[str, Any]:
        return self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, reason, advance_epoch=True)

    def trigger_safety_fault(self, reason: str, *, stop_confirmed: bool = False) -> dict[str, Any]:
        """P0 path: revoke normal writers and hold the measured joint pose."""
        with self._handoff_lock:
            state = self._set_authority(ArmWriter.SAFETY, ServoMode.STOPPING, reason, advance_epoch=True)
            self.teleop.freeze_for_authority_change(int(state["control_epoch"]), reason)
            try:
                with self._transport_lock:
                    safety_timeout_s = float(
                        self.robot.config.get("control_service", {}).get(
                        "safety_position_dispatch_timeout_s",
                        self.robot.config.get("control_service", {}).get("safety_velocity_dispatch_timeout_s", 0.2),
                        )
                    )
                    batch = self.robot.call(
                        "p0", "prime_cpv_position_from_feedback",
                        dispatch_timeout_s=max(0.05, safety_timeout_s),
                    )
            except Exception as exc:
                batch = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            final = self._set_authority(ArmWriter.SAFETY, ServoMode.HOLDING, f"FAULT: {reason}")
            result = {"ok": bool(batch.get("ok", True)), "reason": reason, "stop": "STOP_CONFIRMED" if stop_confirmed else "STOP_UNCONFIRMED", "batch": batch, "authority": final}
            self._log("p0_safety_fault", result=result)
            return result

    def add_action_observer(self, observer: Callable[..., None]) -> None:
        self._action_observers.append(observer)

    def start(self) -> dict[str, Any]:
        try:
            startup_timeout_s = float(
                self.robot.config.get("control_service", {}).get(
                    "startup_connect_timeout_s", 12.0
                )
            )
            result = self._transport_owner.call(
                "p2", "connect", dispatch_timeout_s=startup_timeout_s
            )
        except Exception as exc:
            # The localhost console and Shadow teleoperation must remain
            # usable when the USB-CAN adapter is unplugged or temporarily
            # unavailable. Hardware actions stay gated by fresh feedback and
            # a later Reset reconnects this process to the adapter.
            connect_stage = str(
                getattr(self._transport_owner.backend, "connect_stage", "unknown")
            )
            error_text = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TimeoutError):
                error_text += f" (connect_stage={connect_stage})"
            result = {
                "connected": False,
                "error": error_text,
                "connect_stage": connect_stage,
                "can_feedback_recovery": {
                    "status": "unavailable",
                    "enabled": False,
                    "frame_sent": False,
                    "reason": "USB-CAN connection is unavailable",
                },
            }
            if isinstance(exc, TimeoutError):
                # A vendor/CAN call already executing in a Python thread cannot
                # be cancelled safely. Quarantine this Transport Owner for the
                # rest of the process. Shadow mode remains available; Reset
                # creates a fresh process for the next hardware attempt.
                self._hardware_transport_available = False
        # A normal SDK connection failure (for example cando_open returning
        # false) is also an unavailable transport. Keep the HTTP/Shadow
        # service alive, but do not report that the hardware channel is ready
        # or start a status-monitor thread that can only retry a dead handle.
        if not bool(result.get("connected")) and (
            result.get("error") or str(result.get("connect_stage", "")).startswith("failed:")
        ):
            self._hardware_transport_available = False
        self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "startup HOLD; no ARM writer granted")
        self._running.set()
        # Establish one immutable startup snapshot before accepting HTTP
        # operator actions.  Action threads must never be the first caller
        # that waits on a just-powered controller.
        try:
            if not self._hardware_transport_available:
                raise TimeoutError(result.get("error", "hardware startup timed out"))
            self._refresh_status_snapshot()
        except Exception as exc:
            startup_error = str(result.get("error") or "USB-CAN connection is unavailable")
            if "cando_open failed" in startup_error.lower() or "failed to open can bus" in startup_error.lower():
                startup_error += "; adapter was detected but the USB-CAN channel could not be opened (it may be busy or require a USB power-cycle)"
            offline_control = {
                "mode": "DISCONNECTED",
                "connected": False,
                "reason": "USB-CAN connection is unavailable",
                "error": startup_error,
                "robot": {},
                "can_feedback_recovery": result.get("can_feedback_recovery", {}),
            }
            with self._status_lock:
                self._status_cache = (time.monotonic(), {
                    "timestamp": now_iso(),
                    "control": offline_control,
                    "can_feedback_recovery": offline_control["can_feedback_recovery"],
                    "active_action": None,
                    "gripper": {"raw": {"ok": False, "error": offline_control["error"]}},
                    "tool_pose_candidate": None,
                    "gripper_hold": {"supported": False, "active": False},
                    "broker": self.authority_status(offline_control),
                    "safety_priority": ["power_switch", "operator_takeover", "leased_action"],
                    "electronic_emergency_stop_enabled": self.allow_electronic_emergency_stop,
                    "service_health": self.health(),
                    "feedback_ready": {"ok": False, "reason": "USB-CAN connection is unavailable"},
                })
        self._watchdog = threading.Thread(target=self._watchdog_loop, name="nero-lease-watchdog", daemon=True)
        self._watchdog.start()
        if self._hardware_transport_available:
            self._status_monitor = threading.Thread(target=self._status_loop, name="nero-status-monitor", daemon=True)
            self._status_monitor.start()
        recovery = result.get("can_feedback_recovery", {})
        self._log("can_feedback_probe", result=recovery)
        if recovery.get("frame_sent"):
            self._log("can_feedback_recovery", result=recovery)
        self._log("service_start", result=result)
        return result

    def close(self) -> dict[str, Any]:
        self._running.clear()
        self.teleop.stop_session("control service shutdown")
        if self._status_monitor is not None and self._status_monitor is not threading.current_thread():
            self._status_monitor.join(timeout=1.0)
        if self._watchdog is not None and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=1.0)
        # Shutdown must also be safe after a failed initial USB-CAN connect.
        # In that state there is no controller to hold or disconnect, but the
        # HTTP process should still be able to terminate cleanly.
        if not self._hardware_transport_available:
            self._transport_owner.close()
            self._log("service_stop", hold=None, result={"ok": False, "error": "hardware transport quarantined after startup timeout"})
            return {"hold": None, "disconnect": {"ok": False, "error": "hardware transport unavailable"}}
        try:
            mode = self.robot.get_control_state().get("mode")
        except Exception:
            mode = "DISCONNECTED"
        hold_result = None
        if mode not in {"DISCONNECTED", "FREEDRIVE", "EMERGENCY_DAMPING", "FAULT"}:
            hold_result = self.robot.hold_follower_without_position_target(
                "control service shutdown"
            ).to_dict()
        try:
            result = self.robot.disconnect()
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._transport_owner.close()
        self._log("service_stop", hold=hold_result, result=result)
        return {"hold": hold_result, "disconnect": result}

    def status(self) -> dict[str, Any]:
        # Never perform a live SDK read in an HTTP request. The monitor owns
        # refreshes; this endpoint remains responsive while an SDK call stalls.
        with self._status_lock:
            if self._status_cache is not None:
                stamp, snapshot = self._status_cache
                result = dict(snapshot)
                result["status_age_s"] = max(0.0, time.monotonic() - stamp)
                lease = self.leases.current()
                result["lease"] = lease.public() if lease else None
                result["active_action"] = self._active_action_public()
                result["service_health"] = self.health()
                result["teleop"] = self.teleop.status()
                result["broker"] = self.authority_status(result.get("control"))
                result["feedback_ready"] = self._feedback_readiness(
                    result.get("control", {}), result["status_age_s"]
                )
                return result
        return self._refresh_status_snapshot()

    @staticmethod
    def _feedback_readiness(control: Any, status_age_s: float = 0.0) -> dict[str, Any]:
        """Describe whether it is safe to send a non-emergency operator command.

        This deliberately consumes only an already-read status snapshot.  A
        freshly powered NERO can have CAN traffic without valid NERO feedback;
        attempting a Leader transition in that state used to enter the SDK's
        enable retry loop and made the page look frozen.
        """
        if not isinstance(control, dict):
            return {"ok": False, "reason": "control status is unavailable"}
        if not control.get("connected"):
            return {"ok": False, "reason": "NERO CAN connection is unavailable"}
        if status_age_s > 1.0:
            return {"ok": False, "reason": f"robot feedback snapshot is stale ({status_age_s:.1f} s)"}
        robot = control.get("robot")
        if not isinstance(robot, dict):
            return {"ok": False, "reason": "robot feedback is not available"}
        joints = robot.get("joint_angles_rad")
        flange = robot.get("flange_pose")
        arm_status = robot.get("arm_status")
        mode = control.get("mode")
        backend = control.get("freedrive_backend")
        leader_age = control.get("leader_feedback_age_s")
        leader_hz = control.get("leader_feedback_hz")
        valid_joints = isinstance(joints, list) and len(joints) == 7 and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in joints
        )
        valid_flange = isinstance(flange, list) and len(flange) == 6 and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in flange
        )
        valid_leader = (
            mode == "FREEDRIVE"
            and backend in {"leader", "drag_teach"}
            and valid_joints
            and valid_flange
            and isinstance(leader_age, (int, float))
            and math.isfinite(leader_age)
            and leader_age <= 0.5
            and (not isinstance(leader_hz, (int, float)) or leader_hz > 0)
        )
        if valid_leader:
            return {
                "ok": True,
                "reason": "fresh Leader feedback is available; HOLD can exit zero-force mode",
            }
        valid_arm_status = isinstance(arm_status, dict) and arm_status.get("arm_status") is not None
        if valid_joints and valid_flange and valid_arm_status:
            return {"ok": True, "reason": "fresh follower feedback is available"}
        recovery = control.get("can_feedback_recovery")
        if isinstance(recovery, dict) and recovery.get("status") == "failed":
            return {
                "ok": False,
                "reason": "startup CAN feedback recovery did not observe known NERO feedback",
            }
        if not (valid_joints and valid_flange and valid_arm_status):
            missing = []
            if not valid_joints:
                missing.append("seven joint angles")
            if not valid_flange:
                missing.append("flange pose")
            if not valid_arm_status:
                missing.append("arm status")
            return {"ok": False, "reason": f"fresh NERO feedback is incomplete: {', '.join(missing)}"}
        return {"ok": True, "reason": "fresh follower feedback is available"}

    def _refresh_status_snapshot(self) -> dict[str, Any]:
        started = time.monotonic()
        with self._robot_read_lock:
            control = self.robot.get_control_state()
            robot_state = control.get("robot") or {}
            tool_pose_candidate = candidate_tool_pose_from_flange(robot_state.get("flange_pose"))
            try:
                gripper = self.robot.read_gripper().to_dict()
            except Exception as exc:
                gripper = {"raw": {"ok": False, "error": f"{type(exc).__name__}: {exc}"}}
        self._last_sdk_read_duration_ms = (time.monotonic() - started) * 1000.0
        lease = self.leases.current()
        snapshot = {
            "timestamp": now_iso(),
            "control": control,
            "robot": robot_state,
            "can_feedback_recovery": control.get("can_feedback_recovery", {}),
            "lease": lease.public() if lease else None,
            "active_action": self._active_action_public(),
            "gripper": gripper,
            "tool_pose_candidate": tool_pose_candidate,
            "gripper_hold": self.robot.get_gripper_hold_state(),
            "broker": self.authority_status(control),
            "safety_priority": ["power_switch", "operator_takeover", "leased_action"],
            "electronic_emergency_stop_enabled": self.allow_electronic_emergency_stop,
            "service_health": self.health(),
        }
        snapshot["feedback_ready"] = self._feedback_readiness(control)
        with self._status_lock:
            self._status_cache = (time.monotonic(), snapshot)
        return snapshot

    def _status_loop(self) -> None:
        while self._running.is_set():
            # During hardware teleop the P1 CPV stream owns the transport.
            # Concurrent P2 status reads can race the vendor SDK's motion
            # calls and block the backend's single CAN owner.
            teleop_session = self.teleop.status().get("session", {}) if self.teleop else {}
            teleop_state = teleop_session.get("state")
            teleop_mode = teleop_session.get("mode")
            if teleop_state in {"STARTING", "STOPPING"} or teleop_mode == "hardware":
                time.sleep(self._status_monitor_interval_s)
                continue
            try:
                self._refresh_status_snapshot()
            except CommandRevoked:
                # During an exclusive mode transition the Transport Owner
                # deliberately rejects ordinary P2 reads.  Preserve the last
                # good snapshot rather than presenting a false disconnection.
                pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                with self._status_lock:
                    self._status_cache = (time.monotonic(), {
                        "timestamp": now_iso(),
                        "control": {
                            "mode": "DISCONNECTED",
                            "connected": False,
                            "reason": "USB-CAN connection is unavailable",
                            "error": error,
                        },
                        "active_action": self._active_action_public(),
                        "service_health": self.health(),
                        "feedback_ready": {
                            "ok": False,
                            "reason": "USB-CAN connection is unavailable",
                        },
                    })
            time.sleep(self._status_monitor_interval_s)

    def observation(self, include_motor_states: bool = False) -> dict[str, Any]:
        with self._robot_read_lock:
            return self.robot.get_observation(include_motor_states=include_motor_states).to_dict()

    def health(self) -> dict[str, Any]:
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        active = self._active_action_public()
        return {
            "role": "nero-control-service",
            "pid": __import__("os").getpid(),
            "status_monitor_alive": bool(self._status_monitor and self._status_monitor.is_alive()),
            "watchdog_alive": bool(self._watchdog and self._watchdog.is_alive()),
            "active_action": active,
            "job_count": len(jobs),
            "last_sdk_read_duration_ms": self._last_sdk_read_duration_ms,
            "running": self._running.is_set(),
            "hardware_transport_available": self._hardware_transport_available,
            "teleop": self.teleop.status(),
        }

    def teleop_status(self) -> dict[str, Any]:
        return self.teleop.status()

    def broker_status(self) -> dict[str, Any]:
        """Read-only ownership and safety state for diagnostics/UI."""
        snapshot = self.status()
        return {
            **self.authority_status(snapshot.get("control")),
            "teleop": self.teleop.status(),
            "active_action": self._active_action_public(),
            "feedback_age_s": snapshot.get("status_age_s"),
        }

    def teleop_kinematics(self) -> dict[str, Any]:
        return self.teleop.kinematics()

    def teleop_start(
        self,
        mode: str | None = None,
        confirm_hardware: bool = False,
        client_id: str = "anonymous",
        execution_mode: str | None = None,
        input_source: str | None = None,
    ) -> dict[str, Any]:
        return self.teleop.start_session(mode, confirm_hardware, client_id, execution_mode, input_source)

    def teleop_intent(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.teleop.submit_intent(body)

    def teleop_pico_intent(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.teleop.submit_pico_intent(body)

    def teleop_heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        return self.teleop.heartbeat(client_id, session_id)

    def teleop_stop(self, reason: str = "teleop stopped") -> dict[str, Any]:
        result = self.teleop.stop_session(reason)
        self._log("teleop_stop", reason=reason, result=result)
        return result

    # ---- Public OSC facade -------------------------------------------------
    # OSC accepts only base-frame absolute targets and mode/end-effector
    # commands. Clutch and relative-pose handling live in legacy input
    # adapters and never appear in this interface or its state.
    def osc_start(
        self,
        client_id: str = "anonymous",
        execution_mode: str = "shadow",
    ) -> dict[str, Any]:
        result = self.osc_servo.start_session(
            execution_mode=execution_mode,
            input_source="joystick",
            client_id=client_id,
        )
        return {"ok": True, "state": self.osc_state(), "session": result.get("session", {})}

    def osc_stop(self, reason: str = "OSC session stopped") -> dict[str, Any]:
        result = self.hold(reason)
        return {"ok": bool(result.get("ok")), "result": result, "state": self.osc_state()}

    def osc_heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        """Renew an OSC command session without exposing an input adapter."""
        self.osc_servo.heartbeat(client_id, session_id)
        return {"ok": True, "state": self.osc_state()}

    def osc_command(self, body: dict[str, Any]) -> dict[str, Any]:
        command_type = str(body.get("type", "")).strip().lower()
        payload = dict(body.get("payload") or {})
        acknowledgement_only = bool(body.get("acknowledgement_only", False)) and command_type == "track_tcp"
        if command_type in {"track_tcp", "move_tcp"}:
            result = self.osc_servo.submit_absolute_target(body, mode=command_type)
        elif command_type in {"hold", "stop"}:
            result = self.hold(str(payload.get("reason", "OSC HOLD requested")))
        elif command_type == "freedrive":
            result = self.freedrive(
                str(payload.get("reason", "OSC FREEDRIVE requested")),
                bool(payload.get("recover_emergency", False)),
                bool(payload.get("preserve_gripper", False)),
            )
        elif command_type == "gripper":
            width = payload.get("width_m")
            result = self.command_gripper(
                str(payload.get("mode", "")),
                float(width) if width is not None else None,
                float(payload.get("force_n", 1.0)),
                bool(payload.get("preserve_on_freedrive", False)),
            )
        elif command_type == "joint_target":
            token = str(body.get("token", ""))
            result = self.execute(token, {"type": "joint_target", **payload}, body.get("timeout"))
        else:
            raise ValueError("OSC command type must be track_tcp, move_tcp, hold, stop, freedrive, gripper, or joint_target")
        response = {"ok": bool(result.get("ok", result.get("accepted", False))), "result": result}
        # Continuous target updates can arrive at the servo rate.  Returning
        # the full state (including raw CAN feedback) for every update turns
        # command acknowledgement into the bottleneck.  Clients opting into
        # this compact response consume the normal /api/osc/state snapshot.
        if not acknowledgement_only:
            response["state"] = self.osc_state()
        return response

    def osc_state(self) -> dict[str, Any]:
        servo = dict(self.osc_servo.status())
        # Deliberately exclude every legacy clutch/input-adapter concept.
        for key in ("clutch_active", "anchor_id", "relative_pose", "tcp_anchor", "input_source", "pose_mapping_verified"):
            servo.pop(key, None)
        raw_session = servo.get("session")
        session = (
            {key: value for key, value in raw_session.items() if key != "input_source"}
            if isinstance(raw_session, dict) else raw_session
        )
        snapshot = self.status()
        current_tcp = None
        solver = dict((servo.get("last_result") or {}).get("solver") or {})
        if isinstance(solver.get("tcp"), dict):
            try:
                current_tcp = OperationalSpaceServo._pose_from_tcp(solver["tcp"])
            except (TypeError, ValueError):
                current_tcp = None
        target_tcp = servo.get("reference_pose")
        tcp_tracking_error = self._tcp_tracking_error(current_tcp, target_tcp)
        raw_diagnostics = dict(servo.get("diagnostics") or {})
        last_result = dict(servo.get("last_result") or {})
        last_output = dict(servo.get("last_output") or {})
        execution_mode = str((session or {}).get("execution_mode") or "shadow")
        shadow = execution_mode == "shadow"
        pink = dict(last_result.get("solver") or {})
        gate = {
            "status": last_output.get("status", "held"),
            "accepted": bool(last_result.get("ok", False)),
            "limited": bool(last_result.get("gate_limited", False)),
            "reason": last_result.get("gate_reason"),
        }
        diagnostics = {
            "loop_count": raw_diagnostics.get("loop_count", 0),
            "tcp_error": tcp_tracking_error,
            "pink": pink,
            "ruckig": last_result.get("ruckig"),
            "safety_gate": gate,
            "timing": raw_diagnostics.get("timing", {}),
            "trajectory_state": raw_diagnostics.get("trajectory_state"),
            "trajectory_brake_reason": raw_diagnostics.get("trajectory_brake_reason"),
        }
        recent_batches = list(raw_diagnostics.get("recent_cpv_batches") or [])
        observed_joints = last_output.get("final_joint_target_rad") if shadow else (snapshot.get("robot") or {}).get("joint_angles_rad")
        return {
            "schema_version": "nero.osc.v2",
            "state_sequence": servo.get("state_sequence", 0),
            "session": session,
            "command": {
                "target_tcp": target_tcp,
                "final_joint_target_rad": last_output.get("final_joint_target_rad"),
                "final_joint_velocity_rad_s": last_output.get("final_joint_velocity_rad_s"),
                "sequence": last_output.get("sequence", (session or {}).get("sequence", 0)),
                "epoch": last_output.get("epoch", raw_diagnostics.get("motion_epoch")),
                "output_status": last_output.get("status", "held"),
            },
            "execution": {
                "mode": execution_mode,
                "commanded_joint_state_rad": last_output.get("final_joint_target_rad"),
                "observed_joint_state_rad": observed_joints,
                "observed_source": "simulated_final_output" if shadow else "measured_can_feedback",
                "output_count": raw_diagnostics.get("output_count", 0),
                "accepting_targets": bool(servo.get("input_enabled")),
                "current_tcp_pose": current_tcp,
            },
            "diagnostics": diagnostics,
            "transport": {
                "connected": bool((snapshot.get("control") or {}).get("connected")),
                "reason": (snapshot.get("control") or {}).get("reason"),
                "can_health": snapshot.get("feedback_ready"),
                "latest_hardware_feedback": snapshot.get("robot"),
                "participation": "not_participating" if shadow else "active",
                "cpv_dispatch_count": raw_diagnostics.get("cpv_dispatch_count", 0),
                "last_cpv_dispatch": None if shadow else (recent_batches[-1] if recent_batches else None),
            },
            "solver": servo.get("solver"),
            "workspace": servo.get("workspace"),
            "robot": snapshot.get("robot") or (snapshot.get("control") or {}).get("robot"),
            "gripper": snapshot.get("gripper"),
            "authority": self.authority_status(snapshot.get("control")),
            "active_action": snapshot.get("active_action"),
        }

    @staticmethod
    def _tcp_tracking_error(current: Any, target: Any) -> dict[str, Any] | None:
        """Return base-frame target-minus-current TCP error, when both poses exist."""
        if not isinstance(current, dict) or not isinstance(target, dict):
            return None
        try:
            current_position = [float(value) for value in current["position_m"]]
            target_position = [float(value) for value in target["position_m"]]
            current_orientation = [float(value) for value in current["orientation_xyzw"]]
            target_orientation = [float(value) for value in target["orientation_xyzw"]]
        except (KeyError, TypeError, ValueError):
            return None
        if (len(current_position), len(target_position), len(current_orientation), len(target_orientation)) != (3, 3, 4, 4):
            return None
        values = current_position + target_position + current_orientation + target_orientation
        if not all(math.isfinite(value) for value in values):
            return None
        current_norm = math.sqrt(sum(value * value for value in current_orientation))
        target_norm = math.sqrt(sum(value * value for value in target_orientation))
        if current_norm < 1e-9 or target_norm < 1e-9:
            return None
        dot = abs(sum(a * b for a, b in zip(current_orientation, target_orientation)) / (current_norm * target_norm))
        orientation_angle = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        position_vector = [target_position[index] - current_position[index] for index in range(3)]
        return {
            "position_vector_m": position_vector,
            "position_norm_m": math.sqrt(sum(value * value for value in position_vector)),
            "orientation_angle_rad": orientation_angle,
        }

    def handoff_to_console(self, reason: str = "operator returned to the control console") -> dict[str, Any]:
        """Atomically transfer CPV teleoperation ownership to a Follower hold.

        This is intentionally the sole path used by the console takeover and
        the teleop page's return action.  It prevents a console command from
        reaching the arm between a residual CPV command and a verified hold.
        """
        self._require_operational_control()
        with self._handoff_lock:
            stopped = self._stop_teleop_for_mode_transition(reason)
            if not stopped.get("ok"):
                stopped["handoff"] = {"stage": "teleop_stop", "reason": stopped.get("reason")}
                self._log("teleop_console_handoff", reason=reason, result=stopped)
                return stopped

            transition = self._set_authority(
                ArmWriter.MODE_TRANSITION,
                ServoMode.SUSPENDED,
                "official Follower/HOLD transition",
                advance_epoch=True,
                transport_exclusive_category="follower_hold_transition",
            )
            direct_handoff = self.teleop.abandon_session_without_braking(
                int(transition["control_epoch"]),
                f"{reason}: revoke teleop for Follower/HOLD",
            )
            handoff_detail = direct_handoff.get("direct_handoff", {})
            if not handoff_detail.get("threads_stopped", False):
                self._transport_owner.complete_epoch_transition(
                    int(transition["control_epoch"]), "follower_hold_transition"
                )
                self._set_authority(
                    ArmWriter.NONE,
                    ServoMode.SUSPENDED,
                    "Follower/HOLD aborted: teleop workers did not stop",
                )
                return {"ok": False, "reason": "teleop workers did not stop", "transition": transition}

            requested = {"type": "follower_hold", "reason": reason, "handoff": True}
            hold = self._run_observed_operator_action(
                requested,
                lambda: self._transport_follower_hold(reason, int(transition["control_epoch"])),
            )
            if hold.get("ok"):
                self._set_authority(
                    ArmWriter.SERVO,
                    ServoMode.HOLDING,
                    "console follower hold confirmed without position target",
                    command_stream=CommandStream.NONE,
                )
            else:
                self._set_authority(ArmWriter.SAFETY, ServoMode.HOLDING, "console hold failed")
            result = {
                "ok": bool(hold.get("ok")),
                "handoff": {
                    "stage": "hold_confirmed" if hold.get("ok") else "hold_failed",
                    "reason": reason,
                    "teleop": stopped["teleop"].get("handoff", {}),
                    "cpv_stop": stopped["cpv_stop"],
                    "direct_handoff": handoff_detail,
                    "hold": hold.get("result", {}),
                },
                "revoked_lease": stopped["revoked_lease"],
                "hold": hold,
            }
            self._log("teleop_console_handoff", reason=reason, result=result)
            return result

    def _stop_teleop_for_mode_transition(self, reason: str) -> dict[str, Any]:
        """Shared P1 brake + CPV-zero barrier for official mode changes."""
        self._mark_active_preempted(reason)
        self.robot.request_preempt(reason)
        teleop = self.teleop.stop_session(f"{reason}: brake before mode transition")
        if not teleop.get("handoff", {}).get("servo_stopped", False):
            return {"ok": False, "reason": "teleop servo did not stop", "teleop": teleop}
        cpv_stop = self._transport_stop_cpv_for_mode_transition(reason)
        if not cpv_stop.get("ok"):
            return {"ok": False, "reason": "CPV did not stop", "teleop": teleop, "cpv_stop": cpv_stop}
        revoked = self.leases.release(force=True)
        return {
            "ok": True,
            "teleop": teleop,
            "cpv_stop": cpv_stop,
            "revoked_lease": revoked.public() if revoked else None,
        }

    def _transport_follower_hold(self, reason: str, command_epoch: int) -> dict[str, Any]:
        with self._transport_lock:
            try:
                return self.robot.call(
                    "p2",
                    "hold_follower_without_position_target",
                    reason,
                    command_epoch=command_epoch,
                    category="follower_hold_transition",
                ).to_dict()
            finally:
                self._transport_owner.complete_epoch_transition(
                    command_epoch, "follower_hold_transition"
                )

    def teleop_recenter(self) -> dict[str, Any]:
        return self.teleop.recenter()

    def action_status(self, action_id: str) -> dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(action_id)
            if job is None:
                raise KeyError(f"unknown action_id: {action_id}")
            result = dict(job)
        if result.get("started_monotonic") is not None and result.get("finished_at") is None:
            result["elapsed_s"] = round(time.monotonic() - result["started_monotonic"], 3)
        result.pop("started_monotonic", None)
        return result

    def submit_action_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """Submit a single-flight operator or leased action without blocking HTTP."""
        kind = str(body.get("kind", "")).strip()
        timeout = float(body.get("timeout", 25.0))
        # Reject before creating a job thread.  This keeps the HTTP/UI path
        # responsive when a just-powered robot has not started CAN feedback.
        if kind in {"hold", "freedrive", "recover-stale-leader", "gripper"}:
            self._require_operational_control()
        with self._jobs_lock:
            running = next((item for item in self._jobs.values() if item.get("status") in {"queued", "running"}), None)
            if running is None and self._active_action is not None:
                running = {"action_id": self._active_action.get("id", "unknown"), "status": "already_running"}
            if running is not None:
                return {"action_id": running["action_id"], "status": "already_running", "deduplicated": True}
            action_id = secrets.token_urlsafe(10)
            job = {
                "action_id": action_id, "kind": kind, "status": "queued",
                "timeout_s": max(1.0, timeout),
                "created_at": now_iso(), "started_at": None, "finished_at": None,
                "started_monotonic": None, "elapsed_s": 0.0, "result": None, "error": None,
            }
            self._jobs[action_id] = job

        def run() -> None:
            with self._jobs_lock:
                job["status"] = "running"
                job["started_at"] = now_iso()
                job["started_monotonic"] = time.monotonic()
            operator_action = None
            if kind != "execute":
                with self._action_lock:
                    operator_action = {"id": action_id, "owner": "operator", "type": kind, "started_at": job["started_at"], "preempt_requested": False, "preempt_reason": None}
                    self._active_action = operator_action
            try:
                result = self._run_submitted_action(kind, body, timeout)
                with self._jobs_lock:
                    if job.get("status") != "timed_out":
                        job["status"] = "completed" if result.get("ok", False) else "failed"
                    job["result"] = result
            except Exception as exc:
                with self._jobs_lock:
                    if job.get("status") != "timed_out":
                        job["status"] = "failed"
                    job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if operator_action is not None:
                    with self._action_lock:
                        if self._active_action and self._active_action.get("id") == action_id:
                            self._active_action = None
                with self._jobs_lock:
                    job["finished_at"] = now_iso()
                    job["elapsed_s"] = round(time.monotonic() - (job["started_monotonic"] or time.monotonic()), 3)

        threading.Thread(target=run, name=f"nero-action-{action_id}", daemon=True).start()
        return {"action_id": action_id, "status": "queued", "deduplicated": False}

    def _run_submitted_action(self, kind: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        if kind == "execute":
            return self.execute(str(body.get("token", "")), dict(body.get("action", {})), timeout).get("result", {})
        reason = str(body.get("reason", "operator request from local control page"))
        if kind == "hold":
            return self.hold(reason)
        if kind == "recover-stale-leader":
            return self.recover_stale_leader_to_hold(reason)
        if kind == "freedrive":
            return self.freedrive(reason, bool(body.get("recover_emergency")), bool(body.get("preserve_gripper", False)))
        if kind == "emergency-damping":
            return self.emergency_damping(reason)
        if kind == "gripper":
            width = body.get("width_m")
            return self.command_gripper(str(body.get("mode", "")), float(width) if width is not None else None, float(body.get("force_n", 1.0)), bool(body.get("preserve_on_freedrive", False)))
        raise ValueError(f"unsupported action kind: {kind}")

    def acquire(self, owner: str, ttl_s: float | None = None) -> dict[str, Any]:
        lease = self.leases.acquire(owner=owner, ttl_s=ttl_s)
        self._log("lease_acquired", owner=owner)
        return {"token": lease.token, **lease.public()}

    def renew(self, token: str, ttl_s: float | None = None) -> dict[str, Any]:
        return self.leases.require(token, ttl_s).public()

    def release(self, token: str) -> dict[str, Any]:
        lease = self.leases.release(token)
        self._log("lease_released", owner=lease.owner if lease else None)
        return {"released": lease.public() if lease else None}

    def execute(self, token: str, action: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        mode = self.robot.get_control_state().get("mode")
        if mode == "FAULT":
            state = self.robot.read_state()
            recovery_allowed = (
                action.get("type") == "cartesian_pose"
                and arm_status_is_no_solution_only(state.arm_status)
                and state.joint_enable_status == [True] * 7
            )
            if not recovery_allowed:
                self._require_operational_control()
        else:
            self._require_operational_control()
        lease = self.leases.require(token, max(30.0, float(timeout or 25.0) + 5.0))
        preempt_epoch = self.robot.preempt_epoch()
        action_id = secrets.token_urlsafe(8)
        with self._action_lock:
            if self._active_action is not None:
                raise LeaseError(
                    f"another action is already running for {self._active_action['owner']}"
                )
            self._active_action = {
                "id": action_id,
                "owner": lease.owner,
                "type": str(action.get("type", "unknown")),
                "started_at": now_iso(),
                "preempt_requested": False,
                "preempt_reason": None,
            }
        self._log("action_requested", owner=lease.owner, action=action)
        self._emit_action_event(action_id, "requested", action, owner=lease.owner)
        try:
            if mode == "FREEDRIVE" and action.get("type") != "stop":
                self._log(
                    "freedrive_auto_exit_requested",
                    owner=lease.owner,
                    action=action,
                    reason="explicit action requested while in FREEDRIVE",
                )
                transition = self.robot.hold_follower_without_position_target(
                    "automatic Leader/FREEDRIVE exit before explicit action"
                ).to_dict()
                self._log("freedrive_auto_exit", owner=lease.owner, result=transition)
                if not transition.get("ok"):
                    result = {
                        "ok": False,
                        "reason": "FREEDRIVE auto-exit failed; action was not dispatched",
                        "freedrive_auto_exit": transition,
                    }
                    self._log("action_completed", owner=lease.owner, action=action, result=result)
                    self._emit_action_event(action_id, "completed", action, result, lease.owner)
                    return result
                preempt_epoch = self.robot.preempt_epoch()
            result = self.robot.execute_action(
                action, timeout=timeout, expected_preempt_epoch=preempt_epoch
            ).to_dict()
            self._log("action_completed", owner=lease.owner, action=action, result=result)
            self._emit_action_event(action_id, "completed", action, result, lease.owner)
            return result
        except Exception as exc:
            self._emit_action_event(
                action_id,
                "failed",
                action,
                {"ok": False, "reason": f"{type(exc).__name__}: {exc}"},
                lease.owner,
            )
            raise
        finally:
            with self._action_lock:
                if self._active_action and self._active_action.get("id") == action_id:
                    self._active_action = None

    def hold(self, reason: str = "operator requested hold") -> dict[str, Any]:
        self._require_operational_control()
        result = self.handoff_to_console(reason)
        self._log("operator_hold", result=result)
        return result

    def recover_stale_leader_to_hold(
        self, reason: str = "operator authorized stale Leader recovery"
    ) -> dict[str, Any]:
        """Explicit, supported-arm recovery from an unresponsive Leader residue."""
        revoked = self._operator_takeover("recover-stale-leader", reason)
        requested = {"type": "recover-stale-leader", "reason": reason}
        result = self._run_observed_operator_action(
            requested,
            lambda: self.robot.hold_position(
                reason, recover_stale_leader=True
            ).to_dict(),
        )
        self._log("stale_leader_recovery", revoked=revoked, result=result)
        return result

    def freedrive(
        self,
        reason: str = "operator requested freedrive",
        recover_emergency: bool = False,
        preserve_gripper: bool = False,
    ) -> dict[str, Any]:
        self._require_operational_control()
        with self._handoff_lock:
            stopped = self._stop_teleop_for_mode_transition(reason)
            if not stopped.get("ok"):
                return stopped

            # The epoch barrier now invalidates every queued command that
            # predates the confirmed CPV stop.  The official Leader call is
            # the only allowed current-epoch P2 command.
            transition = self._set_authority(
                ArmWriter.MODE_TRANSITION,
                ServoMode.SUSPENDED,
                "direct FREEDRIVE transition",
                advance_epoch=True,
                transport_exclusive_category="freedrive_transition",
            )
            direct_handoff = self.teleop.abandon_session_without_braking(
                int(transition["control_epoch"]),
                f"{reason}: revoke teleop for direct FREEDRIVE",
            )
            handoff_detail = direct_handoff.get("direct_handoff", {})
            if not handoff_detail.get("threads_stopped", False):
                self._transport_owner.complete_epoch_transition(
                    int(transition["control_epoch"]), "freedrive_transition"
                )
                self._set_authority(
                    ArmWriter.NONE,
                    ServoMode.SUSPENDED,
                    "direct FREEDRIVE aborted: teleop workers did not stop",
                )
                result = {
                    "ok": False,
                    "reason": "teleop workers did not stop; official FREEDRIVE was not entered",
                    "direct_handoff": handoff_detail,
                    "revoked_lease": stopped["revoked_lease"],
                }
                self._log("safety_freedrive", recover_emergency=recover_emergency,
                          preserve_gripper=preserve_gripper, result=result)
                return result
            requested = {
                "type": "freedrive",
                "reason": reason,
                "recover_emergency": recover_emergency,
                "preserve_gripper": preserve_gripper,
            }
            result = self._run_observed_operator_action(
                requested,
                lambda: self._transport_enter_freedrive(
                    reason, recover_emergency, preserve_gripper,
                    int(transition["control_epoch"]),
                ),
            )
            if result.get("ok"):
                self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "official FREEDRIVE active")
            else:
                # Direct transition failures must not surprise the operator by
                # injecting a CPV stop stream or a position hold target.
                self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "direct FREEDRIVE transition failed")
            result["direct_handoff"] = handoff_detail
            result["teleop_brake"] = stopped["teleop"].get("handoff", {})
            result["cpv_stop"] = stopped["cpv_stop"]
            result["revoked_lease"] = stopped["revoked_lease"]
        self._log(
            "safety_freedrive",
            recover_emergency=recover_emergency,
            preserve_gripper=preserve_gripper,
            result=result,
        )
        return result

    def _transport_stop_cpv_for_mode_transition(self, reason: str) -> dict[str, Any]:
        with self._transport_lock:
            try:
                return self.robot.call(
                    "p2", "stop_cpv_for_mode_transition", reason,
                    category="mode_transition_cpv_stop",
                )
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _transport_enter_freedrive(
        self,
        reason: str,
        recover_emergency: bool,
        preserve_gripper: bool,
        command_epoch: int,
    ) -> dict[str, Any]:
        with self._transport_lock:
            try:
                return self.robot.call(
                    "p2",
                    "enter_freedrive",
                    reason,
                    recover_emergency=recover_emergency,
                    preserve_gripper=preserve_gripper,
                    command_epoch=command_epoch,
                    category="freedrive_transition",
                ).to_dict()
            finally:
                self._transport_owner.complete_epoch_transition(
                    command_epoch, "freedrive_transition"
                )

    def command_gripper(
        self,
        mode: str,
        width_m: float | None,
        force_n: float,
        preserve_on_freedrive: bool,
        resume_teleop: bool = False,
    ) -> dict[str, Any]:
        self._require_operational_control()
        reason = f"operator requested gripper {mode}"
        teleop_before = self.teleop.status().get("session") or {}
        # Gripper I/O does not change the arm control mode. Keep an active
        # teleoperation session; arm mode transitions still take over.
        teleop_preserved = teleop_before.get("state") == "ACTIVE"
        revoked = None if teleop_preserved else self._operator_takeover("gripper", reason)
        requested = {
            "type": "gripper",
            "mode": mode,
            "width_m": width_m,
            "force_n": force_n,
            "preserve_on_freedrive": preserve_on_freedrive,
        }
        def command() -> dict[str, Any]:
            arm_hold = None
            control_mode = self.robot.get_control_state().get("mode")
            should_hold = control_mode == "FREEDRIVE" or (
                revoked is not None
                and control_mode not in {"EMERGENCY_DAMPING", "FAULT", "DISCONNECTED"}
            )
            if should_hold:
                arm_hold = self.robot.hold_follower_without_position_target(
                    "operator gripper command requires follower hold"
                ).to_dict()
                if not arm_hold.get("ok"):
                    raise RuntimeError(f"could not hold the arm before gripper command: {arm_hold}")
            result = self.robot.command_gripper(
                mode=mode,
                width_m=width_m,
                force_n=force_n,
                preserve_on_freedrive=preserve_on_freedrive,
            )
            result["arm_hold"] = arm_hold
            return result

        result = self._run_observed_operator_action(requested, command)
        result["teleop_resumed"] = False
        result["teleop_preserved"] = teleop_preserved
        self._log("operator_gripper", revoked=revoked, result=result)
        return result

    def clear_gripper_hold(self) -> dict[str, Any]:
        requested = {"type": "gripper_clear_hold"}
        result = self._run_observed_operator_action(requested, self.robot.clear_gripper_hold)
        self._log("operator_gripper_hold_cleared", result=result)
        return result

    def release_gripper_zero_force(self) -> dict[str, Any]:
        self._require_operational_control()
        reason = "operator requested gripper zero force"
        revoked = self._operator_takeover("gripper_zero_force", reason)
        requested = {"type": "gripper_zero_force", "reason": reason}
        def release() -> dict[str, Any]:
            arm_hold = None
            control_mode = self.robot.get_control_state().get("mode")
            should_hold = control_mode == "FREEDRIVE" or (
                revoked is not None
                and control_mode not in {"EMERGENCY_DAMPING", "FAULT", "DISCONNECTED"}
            )
            if should_hold:
                arm_hold = self.robot.hold_follower_without_position_target(
                    "operator gripper zero force requires follower hold"
                ).to_dict()
                if not arm_hold.get("ok"):
                    raise RuntimeError(f"could not hold the arm before releasing gripper: {arm_hold}")
            result = self.robot.release_gripper_zero_force()
            result["arm_hold"] = arm_hold
            return result

        result = self._run_observed_operator_action(requested, release)
        self._log("operator_gripper_zero_force", revoked=revoked, result=result)
        return result

    def get_gripper_teaching_params(self) -> dict[str, Any]:
        result = self.robot.get_gripper_teaching_params()
        self._log("operator_gripper_teaching_params_read", result=result)
        return result

    def set_gripper_teaching_friction(self, teaching_friction: int) -> dict[str, Any]:
        self._require_operational_control()
        result = self.robot.set_gripper_teaching_friction(teaching_friction)
        self._log("operator_gripper_teaching_friction_set", result=result)
        return result

    def emergency_damping(self, reason: str = "operator requested emergency damping") -> dict[str, Any]:
        if not self.allow_electronic_emergency_stop:
            raise RuntimeError(
                "electronic emergency stop is disabled by the non-disabling control policy; "
                "use quick hold, or use the physical power switch for a true emergency"
            )
        revoked = self._operator_takeover("electronic_emergency_stop", reason)
        requested = {"type": "emergency_damping", "reason": reason}
        result = self._run_observed_operator_action(
            requested, lambda: self.robot.e_stop(reason).to_dict()
        )
        self._log("safety_emergency_damping", revoked=revoked, result=result)
        return result

    def _watchdog_loop(self) -> None:
        while self._running.is_set():
            time.sleep(0.1)
            now = time.monotonic()
            with self._jobs_lock:
                jobs = list(self._jobs.values())
            for job in jobs:
                if job.get("status") == "running" and job.get("started_monotonic") is not None:
                    elapsed = now - float(job["started_monotonic"])
                    if elapsed > float(job.get("timeout_s", 45.0)) and not job.get("timeout_marked"):
                        job["timeout_marked"] = True
                        job["status"] = "timed_out"
                        self.robot.request_preempt(f"action {job['action_id']} exceeded watchdog deadline")
            expired = self.leases.take_expired()
            if expired is not None:
                reason = f"motion lease expired for {expired.owner}"
                self._mark_active_preempted(reason)
                self.robot.request_preempt(reason)
                result = self.robot.hold_follower_without_position_target(reason).to_dict()
                self._log("lease_expired_hold", owner=expired.owner, result=result)
            # Browser/PICO adapters are not trusted to remain scheduled when
            # a page is hidden, reloaded, or the network drops.  Their OSC
            # heartbeat is therefore a real ownership lease: terminate the
            # session through the same official Follower/HOLD path rather
            # than leaving an ACTIVE session waiting for a later reconnect.
            if self.osc_servo.heartbeat_expired():
                reason = "OSC session heartbeat expired"
                try:
                    result = self.hold(reason)
                    self._log("osc_heartbeat_expired_hold", result=result)
                except Exception as exc:
                    self._log("osc_heartbeat_expired_hold_failed", error=f"{type(exc).__name__}: {exc}")

    def _log(self, record_type: str, **payload: Any) -> None:
        with self._log_lock:
            self.logger.append({"record_type": record_type, "timestamp": now_iso(), **jsonable(payload)})

    def _active_action_public(self) -> dict[str, Any] | None:
        with self._action_lock:
            if self._active_action is None:
                return None
            return {key: value for key, value in self._active_action.items() if key != "id"}

    def _mark_active_preempted(self, reason: str) -> None:
        with self._action_lock:
            if self._active_action is not None:
                self._active_action["preempt_requested"] = True
                self._active_action["preempt_reason"] = reason

    def _operator_takeover(self, operation: str, reason: str) -> dict[str, Any] | None:
        """Revoke model control before applying an operator-selected control mode."""
        self.teleop.stop_session(f"operator takeover: {operation}")
        revoked = self.leases.release(force=True)
        self._mark_active_preempted(reason)
        self.robot.request_preempt(reason)
        public = revoked.public() if revoked else None
        self._log("operator_takeover", operation=operation, revoked=public, reason=reason)
        return public

    def _require_operational_control(self, allow_stale: bool = False) -> None:
        readiness = self._cached_feedback_readiness()
        with self._status_lock:
            cached = self._status_cache
        mode = (
            cached[1].get("control", {}).get("mode")
            if cached is not None
            else "UNKNOWN"
        )
        if allow_stale and not readiness.get("ok") and cached is not None:
            control = cached[1].get("control", {})
            if (
                control.get("connected")
                and control.get("mode") not in {"FAULT", "DISCONNECTED"}
                and self._feedback_readiness(control, 0.0).get("ok")
            ):
                return
        if not readiness.get("ok"):
            raise RuntimeError(
                "robot control is unavailable until fresh CAN feedback is restored: "
                f"{readiness.get('reason', 'unknown feedback error')} (mode={mode})"
            )
        if mode in {"FAULT", "DISCONNECTED"}:
            raise RuntimeError(
                f"robot control is unavailable while mode={mode}; restore fresh CAN feedback first"
            )

    def _cached_feedback_readiness(self) -> dict[str, Any]:
        """Return readiness without issuing another SDK read from an action thread."""
        with self._status_lock:
            cached = self._status_cache
        if cached is None:
            return {"ok": False, "reason": "no status snapshot has been received yet"}
        stamp, snapshot = cached
        return self._feedback_readiness(
            snapshot.get("control", {}), max(0.0, time.monotonic() - stamp)
        )

    def _emit_action_event(
        self,
        action_id: str,
        lifecycle: str,
        requested_action: dict[str, Any],
        executed_action: dict[str, Any] | None = None,
        owner: str | None = None,
    ) -> None:
        for observer in tuple(self._action_observers):
            try:
                observer(
                    action_id=action_id,
                    lifecycle=lifecycle,
                    requested_action=requested_action,
                    executed_action=executed_action,
                    owner=owner,
                )
            except Exception as exc:
                self._log(
                    "action_observer_error",
                    action_id=action_id,
                    lifecycle=lifecycle,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _run_observed_operator_action(
        self,
        requested_action: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        action_id = secrets.token_urlsafe(8)
        self._emit_action_event(action_id, "requested", requested_action, owner="operator")
        try:
            result = operation()
        except Exception as exc:
            self._emit_action_event(
                action_id,
                "failed",
                requested_action,
                {"ok": False, "reason": f"{type(exc).__name__}: {exc}"},
                "operator",
            )
            raise
        self._emit_action_event(action_id, "completed", requested_action, result, "operator")
        return result


# Compatibility import for existing local integrations. New code must use
# OperationalSpaceController; both names refer to the same sole CAN owner.
RobotControlBroker = OperationalSpaceController
