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
    CommandRevoked, CommandStream, ArmWriter, ControlSupervisor, HardwareTxOwner,
    ServoMode, ServoWriteRevoked, TransportRobotProxy,
)
from nero_backend.robot import NeroRobot
from shared.schemas import jsonable, now_iso
from motion.osc import OscRuntime, pose_from_tcp
from .telemetry import TelemetryReader
from .hardware_maintenance import HardwareMaintenance

def absolute_pose_from_sdk_rpy(tcp_pose: Any) -> dict[str, list[float]] | None:
    """Convert the SDK's base-frame XYZ/RPY feedback into the public pose form."""
    if not isinstance(tcp_pose, (list, tuple)) or len(tcp_pose) != 6:
        return None
    try:
        x, y, z, roll, pitch, yaw = (float(value) for value in tcp_pose)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, z, roll, pitch, yaw)):
        return None
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the SDK/URDF convention used
    # throughout this project.  The published quaternion is XYZW.
    half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return {
        "position_m": [x, y, z],
        "orientation_xyzw": [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
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


class _ControllerOscCommandPort:
    """Private adapter that prevents the OSC servo from reaching the controller."""

    def __init__(self, controller: "OperationalSpaceController") -> None:
        self._controller = controller

    def require_operational_control(self) -> None: self._controller._require_operational_control()
    def prepare_osc_hardware(self) -> dict[str, Any]: return self._controller.prepare_osc_hardware()
    def osc_stream_active(self) -> bool: return self._controller.osc_stream_active()
    def grant_osc_tracking(self, session_id: str, epoch: int) -> bool: return self._controller.grant_osc_tracking(session_id, epoch)
    def mark_osc_stopping(self, session_id: str, epoch: int, reason: str) -> bool: return self._controller.mark_osc_stopping(session_id, epoch, reason)
    def servo_can_write(self, session_id: str, epoch: int) -> bool: return self._controller.servo_can_write(session_id, epoch)
    def publish_servo_position(self, command: dict[str, Any], session_id: str, epoch: int) -> dict[str, Any]: return self._controller.publish_servo_position(command, session_id, epoch)
    def servo_transport_diagnostics(self) -> dict[str, Any]: return self._controller.servo_transport_diagnostics()
    def wait_for_servo_result(self, mailbox_revision: int, timeout_s: float) -> dict[str, Any]: return self._controller.wait_for_servo_result(mailbox_revision, timeout_s)
    def latch_osc_hold(self, reason: str) -> dict[str, Any]: return self._controller.latch_osc_hold(reason)
    def trigger_safety_fault(self, reason: str) -> dict[str, Any]: return self._controller.trigger_safety_fault(reason)


class _NeroOscRxPort:
    """OSC-private receive cache reader; never reaches HardwareTxOwner."""

    def __init__(self, backend: NeroRobot) -> None:
        self._backend = backend

    def read_cached_feedback(self) -> dict[str, Any]:
        return self._backend.read_cached_osc_feedback()


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
        self._transport_owner = HardwareTxOwner(backend)
        self._transport_reset_lock = threading.Lock()
        self._transport_reset_scheduled = False
        self._hardware_transport_available = True
        self.robot = TransportRobotProxy(self._transport_owner, backend)
        self.telemetry = TelemetryReader(self.robot)
        self.maintenance = HardwareMaintenance(self.robot)
        self.supervisor = ControlSupervisor()
        self._status_cache: tuple[float, dict[str, Any]] | None = None
        self._last_cpv_mode_entry: dict[str, Any] | None = None
        self._cpv_profile_cache: dict[str, Any] = {"status": "not_read"}
        self._active_action: dict[str, Any] | None = None
        self._action_observers: list[Callable[..., None]] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.RLock()
        self._status_monitor: threading.Thread | None = None
        self._status_monitor_interval_s = 0.2
        self._osc_diagnostics_stop = threading.Event()
        self._osc_diagnostics_thread: threading.Thread | None = None
        self._osc_diagnostics_interval_s = max(
            0.02,
            1.0 / max(1.0, float((service_config.get("osc_diagnostics_hz") or 20.0))),
        )
        self._last_sdk_read_duration_ms: float | None = None
        osc_path = Path(__file__).resolve().parents[1] / "config" / "osc.json"
        osc_config = json.loads(osc_path.read_text(encoding="utf-8-sig")) if osc_path.is_file() else {}
        task_tcp_offset = (config_data.get("sdk") or {}).get("task_tcp_offset_from_flange_m")
        if task_tcp_offset is None:
            # Dependency-injected fakes used by the non-hardware test suite
            # commonly carry an empty config.  Their task geometry still
            # resolves from the same project runtime source as production.
            runtime_path = Path(__file__).resolve().parents[1] / "config" / "runtime.json"
            runtime_defaults = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
            task_tcp_offset = (runtime_defaults.get("sdk") or {}).get("task_tcp_offset_from_flange_m")
        if not isinstance(task_tcp_offset, list) or len(task_tcp_offset) != 3:
            raise ValueError("runtime sdk.task_tcp_offset_from_flange_m must contain three values")
        # The solver receives its task point only from runtime SDK config.
        # Keeping it out of osc.json prevents SDK/Pink transform drift.
        osc_config["tcp"] = {
            "offset_from_link7_m": [float(value) for value in task_tcp_offset],
            "verified": True,
            "source": "runtime.sdk.task_tcp_offset_from_flange_m",
        }
        self._osc = OscRuntime(
            _ControllerOscCommandPort(self), _NeroOscRxPort(backend),
            Path(__file__).resolve().parents[1], osc_config,
        )

    # ---- Sole transport-owner entry points ---------------------------------
    # OSC and HTTP handlers deliberately use these methods rather than the
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
            # HOLD, FOLLOWER and OSC_CPV are hardware execution
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

    def prepare_osc_hardware(self) -> dict[str, Any]:
        """Enter a fresh SERVO/HOLDING epoch without allowing old CPV output."""
        self._require_operational_control(allow_stale=True)
        with self._handoff_lock:
            mode = self.robot.get_control_state().get("mode")
            if mode == "FREEDRIVE":
                self._set_authority(ArmWriter.MODE_TRANSITION, ServoMode.SUSPENDED, "exit FREEDRIVE for osc", advance_epoch=True)
                result = self.robot.hold_follower_without_position_target(
                    "osc start exits FREEDRIVE"
                ).to_dict()
                if not result.get("ok"):
                    self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "FREEDRIVE exit failed")
                    raise RuntimeError(f"cannot exit FREEDRIVE: {result}")
            state = self._set_authority(ArmWriter.SERVO, ServoMode.HOLDING, "osc session prepared", advance_epoch=True)
            # Prime CPV while the session is still in HOLDING. The first
            # vendor CPV call may enable drives and switch motion mode; doing
            # that with a measured position hold prevents the first live pose
            # sample from being a partially-dispatched batch.
            epoch = int(state["control_epoch"])
            try:
                prime_result = self.robot.call(
                    "p1", "prime_cpv_position_from_feedback",
                    command_epoch=epoch,
                    category="servo_position",
                    execute_guard=lambda: self.supervisor.allows_servo(None, epoch),
                    dispatch_timeout_s=5.0,
                )
                if isinstance(prime_result, dict):
                    self._last_cpv_mode_entry = dict(prime_result.get("cpv_mode_entry") or {})
            except TimeoutError as exc:
                self._schedule_transport_reset(f"CPV position prime timeout: {exc}")
                raise RuntimeError(f"CPV prime timed out; control service reset scheduled: {exc}") from exc
            return state

    def _schedule_transport_reset(self, reason: str) -> None:
        """Ask the independent service reset path to quarantine a stuck SDK call."""
        with self._transport_reset_lock:
            if self._transport_reset_scheduled:
                self._log("transport_reset_already_scheduled", reason=reason)
                return
            self._transport_reset_scheduled = True
        # This record is intentionally written before the localhost reset
        # request. The reset helper can kill this process moments later, so a
        # post-request log entry is not reliable evidence of the trigger.
        self._log("transport_reset_scheduled", reason=reason)

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
                self._log("transport_reset_request_failed", reason=reason)

        threading.Thread(target=request_reset, name="transport-fault-reset", daemon=True).start()

    def _osc_cached_feedback(self) -> dict[str, Any]:
        """Read only the OSC-owned RX snapshot; never enqueue a P2 call."""
        sample = self._osc.rx_snapshot()
        if not sample:
            return {}
        return {
            "joint_angles_rad": list(sample.get("joints") or []),
            "joint_velocity_rad_s": list(sample.get("velocities") or []),
            "sdk_joint_timestamp": sample.get("sdk_joint_timestamp"),
            "joint_feedback_hz": sample.get("joint_feedback_hz"),
            "received_at_monotonic_ns": sample.get("received_monotonic_ns"),
            "timestamp_monotonic_ns": sample.get("fresh_received_at_monotonic_ns"),
            "revision": sample.get("revision"),
            "age_s": sample.get("age_s"),
            "last_error": sample.get("last_error"),
        }

    def read_osc_cpv_parameters(self) -> dict[str, Any]:
        """Compatibility facade for the hardware-maintenance diagnostic API."""
        profile = self.maintenance.read_cpv_parameters()
        self._cpv_profile_cache = dict(profile)
        return profile

    def sync_cpv_profile_to_osc_limits(self) -> dict[str, Any]:
        """Apply current OSC limits only while the arm is safely idle/HOLD."""
        session = self._osc.status().get("session") or {}
        if session.get("state") == "ACTIVE":
            raise RuntimeError("stop the active OSC session before changing the CPV profile")
        control = self.status().get("control") or {}
        if control.get("mode") != "HOLD":
            raise RuntimeError(f"CPV profile can only change in HOLD (current mode={control.get('mode')})")
        readiness = self._cached_feedback_readiness()
        if not readiness.get("ok"):
            # A full status snapshot includes TCP, arm, and gripper reads.
            # On some NERO firmware revisions it can take several seconds,
            # although the lightweight joint-feedback read remains current.
            # Do not weaken the freshness guard: validate against the
            # OSC-owned receive snapshot before changing profile. This is a
            # cache read and deliberately never contends with TX ownership.
            feedback = self._osc_cached_feedback()
            joints = feedback.get("joint_angles_rad") if isinstance(feedback, dict) else None
            if not (
                isinstance(joints, list)
                and len(joints) == 7
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in joints)
            ):
                raise RuntimeError(
                    "fresh hardware feedback is required: "
                    f"{readiness.get('reason', 'minimal joint feedback unavailable')}"
                )
        speed, acceleration = self._osc.cpv_limits()
        result = self.robot.call("p0", "configure_cpv_profile", speed, acceleration, acceleration,
                                 category="cpv_profile_configuration")
        profile = self.read_osc_cpv_parameters()
        if profile.get("status") != "available":
            raise RuntimeError(f"CPV profile was written but complete read-back is unavailable: {profile.get('missing')}")
        return {"ok": True, "requested": {"cv_rad_s": speed, "acc_rad_s2": acceleration, "dcc_rad_s2": acceleration},
                "result": result, "readback": profile}

    @staticmethod
    def _telemetry_percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return float(ordered[index])

    def collect_readonly_hardware_telemetry(self, sample_count: int = 50) -> dict[str, Any]:
        """Measure OSC RX snapshot cadence without changing robot control state.

        This is deliberately unavailable while OSC tracking is active: a
        telemetry probe reads the independent OSC RX cache. It measures
        parameter-query timing and snapshot cadence, not actuator response
        to a new position target.
        """
        count = max(10, min(200, int(sample_count)))
        if (self._osc.status().get("session") or {}).get("state") == "ACTIVE":
            raise RuntimeError("stop the active OSC session before read-only hardware telemetry")
        readiness = self._cached_feedback_readiness()
        if not readiness.get("ok"):
            raise RuntimeError(f"fresh hardware feedback is required: {readiness.get('reason', 'unknown')}")
        cpv = self.read_osc_cpv_parameters()
        samples: list[dict[str, Any]] = []
        previous_started_ns: int | None = None
        for _ in range(count):
            started_ns = time.perf_counter_ns()
            feedback = self._osc_cached_feedback()
            finished_ns = time.perf_counter_ns()
            samples.append({
                "requested_perf_counter_ns": started_ns,
                "received_perf_counter_ns": finished_ns,
                "read_duration_s": max(0.0, (finished_ns - started_ns) / 1e9),
                "inter_request_s": None if previous_started_ns is None else max(0.0, (started_ns - previous_started_ns) / 1e9),
                "feedback_timestamp_monotonic_ns": feedback.get("timestamp_monotonic_ns"),
            })
            previous_started_ns = started_ns
            # This does not infer a controller cycle; it samples the
            # independent RX snapshot while the arm is idle.
            time.sleep(0.02)
        durations = [float(row["read_duration_s"]) for row in samples]
        intervals = [float(row["inter_request_s"]) for row in samples if row["inter_request_s"] is not None]
        p50_duration = self._telemetry_percentile(durations, 0.50) or 0.0
        p95_duration = self._telemetry_percentile(durations, 0.95) or p50_duration
        p50_interval = self._telemetry_percentile(intervals, 0.50) or 0.02
        p95_interval = self._telemetry_percentile(intervals, 0.95) or p50_interval
        # The device feedback timestamp is known to be read-complete in the
        # current SDK. This is an observation-delay estimate, not a claim of
        # physical servo response latency.
        feedback_delay_s = max(0.001, min(0.15, p95_duration + 0.5 * p95_interval))
        feedback_jitter_s = max(0.0, min(0.05, p95_duration - p50_duration))
        calibration = {
            "source": "read_only_idle_can_probe",
            "sample_count": count,
            "feedback_delay_s": feedback_delay_s,
            "feedback_jitter_s": feedback_jitter_s,
            "actuator_response_identified": False,
            "actuator_response_note": "requires an explicitly authorized CPV position-response experiment",
        }
        return {"status": "available", "collected_at": now_iso(), "cpv_parameters": cpv,
                "feedback": {"read_duration_p50_s": p50_duration, "read_duration_p95_s": p95_duration,
                             "inter_request_p50_s": p50_interval, "inter_request_p95_s": p95_interval,
                             "samples": samples}, "calibration": calibration}

    def osc_stream_active(self) -> bool:
        value = self.robot.continuous_stream_active
        return bool(value() if callable(value) else value)

    def grant_osc_tracking(self, session_id: str, epoch: int) -> bool:
        state = self.supervisor.snapshot()
        if state.writer is not ArmWriter.SERVO or state.epoch != int(epoch):
            return False
        self._set_authority(ArmWriter.SERVO, ServoMode.TRACKING, f"osc tracking {session_id}")
        return True

    def mark_osc_stopping(self, session_id: str, epoch: int, reason: str) -> bool:
        state = self.supervisor.snapshot()
        if state.writer is not ArmWriter.SERVO or state.epoch != int(epoch):
            return False
        self._set_authority(ArmWriter.SERVO, ServoMode.STOPPING, f"osc stopping {session_id}: {reason}")
        return True

    def servo_can_write(self, session_id: str, epoch: int) -> bool:
        return self.supervisor.allows_servo(session_id, epoch)

    def publish_servo_position(self, command: dict[str, Any], session_id: str, epoch: int) -> dict[str, Any]:
        """Publish one CPV target without waiting for SDK/CAN completion."""
        guard = lambda: self.supervisor.allows_servo(session_id, epoch)
        if not guard():
            raise ServoWriteRevoked("SERVO write authority is not valid for this osc epoch")
        values = [float(value) for value in command.get("joint_target_rad") or []]
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            self.trigger_safety_fault("non-finite or malformed servo position")
            raise ValueError("servo position must contain seven finite values")
        if not guard():
            raise ServoWriteRevoked("SERVO write authority was revoked before transport dispatch")
        return self._transport_owner.publish_cpv({**command, "joint_target_rad": values, "epoch": epoch}, execute_guard=guard)

    def servo_transport_diagnostics(self) -> dict[str, Any]:
        """Read the latest CPV send result without touching SDK/CAN."""
        cpv = self._transport_owner.cpv_diagnostics()
        active = (self._transport_owner.diagnostics().get("active") or {})
        timeout_s = float(self.robot.config.get("control_service", {}).get(
            "servo_position_dispatch_timeout_s",
            self.robot.config.get("control_service", {}).get("servo_velocity_dispatch_timeout_s", 0.75),
        ))
        if (
            active.get("method") == "send_cpv_position"
            and float(active.get("age_ms") or 0.0) > timeout_s * 1000.0
        ):
            progress = getattr(self._transport_owner.backend, "cpv_dispatch_progress", lambda: {})()
            can = getattr(self._transport_owner.backend, "can_dispatch_diagnostics", lambda: {})()
            detail = {"transport": active, "cpv_progress": progress, "can": can}
            self._log("cpv_position_timeout", diagnostics=detail)
            self._schedule_transport_reset(f"asynchronous CPV position timeout: {detail}")
        return cpv

    def wait_for_servo_result(self, mailbox_revision: int, timeout_s: float) -> dict[str, Any]:
        return self._transport_owner.wait_cpv_result(mailbox_revision, timeout_s)

    def begin_osc_stop(self, reason: str) -> dict[str, Any]:
        """Atomically invalidate old output while leaving SERVO able to brake."""
        with self._handoff_lock:
            state = self._set_authority(ArmWriter.SERVO, ServoMode.STOPPING, reason, advance_epoch=True)
            self._osc.freeze_for_authority_change(int(state["control_epoch"]), reason)
            return state

    def latch_osc_hold(self, reason: str) -> dict[str, Any]:
        return self._set_authority(ArmWriter.SERVO, ServoMode.HOLDING, reason)

    def suspend_arm_writes(self, reason: str) -> dict[str, Any]:
        return self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, reason, advance_epoch=True)

    def trigger_safety_fault(self, reason: str, *, stop_confirmed: bool = False) -> dict[str, Any]:
        """P0 path: revoke normal writers and hold the measured joint pose."""
        with self._handoff_lock:
            state = self._set_authority(ArmWriter.SAFETY, ServoMode.STOPPING, reason, advance_epoch=True)
            self._osc.freeze_for_authority_change(int(state["control_epoch"]), reason)
            try:
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
        # The OSC-owned RX worker reads only SDK receive caches. Start it
        # after connection has been attempted and before serving clients; it
        # remains independent from the serial TX/P0/P1/P2 owner.
        self._osc.start()
        self._running.set()
        self._osc_diagnostics_stop.clear()
        self._osc_diagnostics_thread = threading.Thread(
            target=self._osc_diagnostics_loop,
            name="nero-osc-diagnostics",
            daemon=True,
        )
        self._osc_diagnostics_thread.start()
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
            }
            with self._status_lock:
                self._status_cache = (time.monotonic(), {
                    "timestamp": now_iso(),
                    "control": offline_control,
                    "active_action": None,
                    "gripper": {"raw": {"ok": False, "error": offline_control["error"]}},
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
        self._log("service_start", result=result)
        return result

    def close(self) -> dict[str, Any]:
        self._running.clear()
        self._osc_diagnostics_stop.set()
        self._osc.close()
        if self._osc_diagnostics_thread is not None and self._osc_diagnostics_thread is not threading.current_thread():
            self._osc_diagnostics_thread.join(timeout=1.0)
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
            cached = self._status_cache
        # Do not hold the status-cache mutex while consulting osc/health.
        # OSC state calls take the osc mutex; retaining this lock here
        # creates a lock inversion with the monitor during a session start.
        if cached is not None:
            stamp, snapshot = cached
            result = dict(snapshot)
            result["status_age_s"] = max(0.0, time.monotonic() - stamp)
            lease = self.leases.current()
            result["lease"] = lease.public() if lease else None
            result["active_action"] = self._active_action_public()
            result["service_health"] = self.health()
            result["osc"] = self._osc.status()
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
        tcp = robot.get("tcp_pose")
        arm_status = robot.get("arm_status")
        mode = control.get("mode")
        backend = control.get("freedrive_backend")
        leader_age = control.get("leader_feedback_age_s")
        leader_hz = control.get("leader_feedback_hz")
        valid_joints = isinstance(joints, list) and len(joints) == 7 and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in joints
        )
        valid_tcp = isinstance(tcp, list) and len(tcp) == 6 and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in tcp
        )
        valid_leader = (
            mode == "FREEDRIVE"
            and backend in {"leader", "drag_teach"}
            and valid_joints
            and valid_tcp
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
        if valid_joints and valid_tcp and valid_arm_status:
            return {"ok": True, "reason": "fresh follower feedback is available"}
        if not (valid_joints and valid_tcp and valid_arm_status):
            missing = []
            if not valid_joints:
                missing.append("seven joint angles")
            if not valid_tcp:
                missing.append("TCP pose")
            if not valid_arm_status:
                missing.append("arm status")
            return {"ok": False, "reason": f"fresh NERO feedback is incomplete: {', '.join(missing)}"}
        return {"ok": True, "reason": "fresh follower feedback is available"}

    def _refresh_status_snapshot(self) -> dict[str, Any]:
        control, robot_state, gripper, duration_ms = self.telemetry.status_sample()
        self._last_sdk_read_duration_ms = duration_ms
        lease = self.leases.current()
        snapshot = {
            "timestamp": now_iso(),
            "control": control,
            "robot": robot_state,
            "lease": lease.public() if lease else None,
            "active_action": self._active_action_public(),
            "gripper": gripper,
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
            # During hardware osc the P1 CPV stream owns the transport.
            # Concurrent P2 status reads can race the vendor SDK's motion
            # calls and block the backend's single CAN owner.
            osc_session = self._osc.status().get("session", {})
            osc_state = osc_session.get("state")
            # ``mode`` is an adapter-qualified compatibility value such as
            # ``joystick_hardware``. The canonical execution selector is
            # ``execution_mode``. Checking the old field left the expensive
            # full P2 status sample active during CPV tracking, where it could
            # hold the sole CAN owner long enough to time out a P1 batch.
            execution_mode = osc_session.get("execution_mode")
            if osc_state in {"STARTING", "STOPPING"} or execution_mode == "hardware":
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

    def _osc_diagnostics_loop(self) -> None:
        """Persist low-cost OSC samples without touching the CAN transport.

        The servo already keeps a bounded cycle trace in memory.  This thread
        snapshots that state at a lower rate and writes only the fields needed
        to diagnose real-hardware oscillation.  It must never call the SDK or
        the full status path: doing so could contend with the CPV TX owner.
        """
        while not self._osc_diagnostics_stop.is_set():
            try:
                snapshot = self._osc.status()
                session = dict(snapshot.get("session") or {})
                if session.get("execution_mode") == "hardware":
                    diagnostics = dict(snapshot.get("diagnostics") or {})
                    timing = dict(diagnostics.get("timing") or {})
                    execution = dict(snapshot.get("execution_sample") or {})
                    last_output = dict(snapshot.get("last_output") or {})
                    batches = list(diagnostics.get("recent_cpv_batches") or [])
                    cycle_trace = dict(diagnostics.get("cycle_trace") or {})
                    trace_tail = list(cycle_trace.get("recent_cycles") or [])
                    last_result = dict(snapshot.get("last_result") or {})
                    latest_batch = batches[-1] if batches else {}
                    gate = {
                        "triggered": bool(last_result.get("gate_limited") or latest_batch.get("final_gate_limited")),
                        "limited": bool(last_result.get("gate_limited")),
                        "reason": last_result.get("gate_reason"),
                        "final_gate_limited": bool(latest_batch.get("final_gate_limited")),
                    }
                    self._log(
                        "osc_realtime_diagnostic",
                        monotonic_ns=time.monotonic_ns(),
                        session={
                            "state": session.get("state"),
                            "session_id": session.get("session_id"),
                            "execution_mode": session.get("execution_mode"),
                            "sequence": session.get("sequence"),
                            "motion_epoch": session.get("motion_epoch"),
                            "last_input_age_s": session.get("last_input_age_s"),
                        },
                        trajectory_state=diagnostics.get("trajectory_state"),
                        trajectory_brake_reason=diagnostics.get("trajectory_brake_reason"),
                        timing=timing,
                        last_result=last_result,
                        gate=gate,
                        last_output=last_output,
                        execution={
                            "sample_id": execution.get("sample_id"),
                            "target_generation": execution.get("target_generation"),
                            "joint_state_rad": execution.get("joint_state_rad"),
                            "measured_joint_state_rad": execution.get("measured_joint_state_rad"),
                            "joint_velocity_rad_s": execution.get("joint_velocity_rad_s"),
                            "measured_joint_velocity_rad_s": execution.get("measured_joint_velocity_rad_s"),
                            "position_error_m": execution.get("position_error_m"),
                            "orientation_error_rad": execution.get("orientation_error_rad"),
                        },
                        latest_cpv_batch=batches[-1] if batches else None,
                        cycle_trace_tail=trace_tail[-3:],
                    )
            except Exception as exc:
                self._log(
                    "osc_realtime_diagnostic_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._osc_diagnostics_stop.wait(self._osc_diagnostics_interval_s)

    def observation(self, include_motor_states: bool = False) -> dict[str, Any]:
        return self.telemetry.observation(include_motor_states)

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
            "osc": self._osc.status(),
        }

    def broker_status(self) -> dict[str, Any]:
        """Read-only ownership and safety state for diagnostics/UI."""
        snapshot = self.status()
        return {
            **self.authority_status(snapshot.get("control")),
            "osc": self._osc.status(),
            "active_action": self._active_action_public(),
            "feedback_age_s": snapshot.get("status_age_s"),
        }

    def osc_kinematics(self) -> dict[str, Any]:
        return self._osc.kinematics()

    # ---- Public OSC facade -------------------------------------------------
    # OSC accepts only base-frame absolute targets and mode/end-effector
    # commands. Clutch and relative-pose handling live in legacy input
    # adapters and never appear in this interface or its state.
    def osc_start(
        self,
        client_id: str = "anonymous",
        execution_mode: str = "shadow",
    ) -> dict[str, Any]:
        result = self._osc.start_session(
            execution_mode=execution_mode,
            client_id=client_id,
        )
        return {"ok": True, "state": self.osc_state(), "session": result.get("session", {})}

    def osc_stop(self, reason: str = "OSC session stopped") -> dict[str, Any]:
        session = self._osc.status().get("session") or {}
        if session.get("state") == "ACTIVE" and session.get("execution_mode") == "shadow":
            stopped = self._osc.stop_session(reason)
            result = {"ok": True, "reason": reason, "robot_commands_sent": False,
                      "handoff": stopped.get("handoff", {})}
        else:
            result = self.hold(reason)
        return {"ok": bool(result.get("ok")), "result": result, "state": self.osc_state()}

    def osc_heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        """Renew an OSC command session without exposing an input adapter."""
        self._osc.heartbeat(client_id, session_id)
        return {"ok": True, "state": self.osc_state()}

    def osc_command(self, body: dict[str, Any]) -> dict[str, Any]:
        command_type = str(body.get("type", "")).strip().lower()
        payload = dict(body.get("payload") or {})
        acknowledgement_only = bool(body.get("acknowledgement_only", False)) and command_type == "track_tcp"
        if command_type in {"track_tcp", "move_tcp"}:
            result = self._osc.submit_absolute_target(body, mode=command_type)
        elif command_type in {"hold", "stop"}:
            reason = str(payload.get("reason", "OSC HOLD requested"))
            session = self._osc.status().get("session") or {}
            result = self._osc.request_shadow_hold(reason) if session.get("execution_mode") == "shadow" else self.hold(reason)
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
        else:
            raise ValueError("OSC command type must be track_tcp, move_tcp, hold, stop, freedrive, or gripper")
        response = {"ok": bool(result.get("ok", result.get("accepted", False))), "result": result}
        # Continuous target updates can arrive at the servo rate.  Returning
        # the full state (including raw CAN feedback) for every update turns
        # command acknowledgement into the bottleneck.  Clients opting into
        # this compact response consume the normal /api/osc/state snapshot.
        if not acknowledgement_only:
            response["state"] = self.osc_state()
        return response

    def osc_state(self) -> dict[str, Any]:
        servo = dict(self._osc.status())
        session = servo.get("session")
        snapshot = self.status()
        execution_sample = dict(servo.get("execution_sample") or {})
        solver = dict((servo.get("last_result") or {}).get("solver") or {})
        estimated_tcp = None
        measured_tcp = None
        if isinstance(execution_sample.get("estimated_tcp_pose"), dict):
            estimated_tcp = dict(execution_sample["estimated_tcp_pose"])
        if isinstance(execution_sample.get("measured_tcp_pose"), dict):
            measured_tcp = dict(execution_sample["measured_tcp_pose"])
        elif isinstance(solver.get("tcp"), dict):
            try:
                estimated_tcp = pose_from_tcp(solver["tcp"])
                measured_tcp = estimated_tcp
            except (TypeError, ValueError):
                estimated_tcp = None
                measured_tcp = None
        target_tcp = self._osc.target_pose()
        sampled_target = execution_sample.get("target_tcp") if isinstance(execution_sample.get("target_tcp"), dict) else target_tcp
        estimated_tracking_error = self._tcp_tracking_error(estimated_tcp, sampled_target)
        measured_tracking_error = self._tcp_tracking_error(measured_tcp, sampled_target)
        for tracking_error in (estimated_tracking_error, measured_tracking_error):
            if isinstance(tracking_error, dict) and execution_sample:
                tracking_error.update({
                "sample_id": execution_sample.get("sample_id"),
                "target_generation": execution_sample.get("target_generation"),
                })
        raw_diagnostics = dict(servo.get("diagnostics") or {})
        last_result = dict(servo.get("last_result") or {})
        last_output = dict(servo.get("last_output") or {})
        active_session = bool(isinstance(session, dict) and session.get("state") == "ACTIVE")
        execution_mode = str((session or {}).get("execution_mode") or "shadow") if active_session else "idle"
        shadow = execution_mode == "shadow"
        if not active_session:
            # A stopped shadow sample is historical data, never current robot
            # execution.  Keeping it here used to make hardware HOLD look like
            # simulated movement in the browser.
            execution_sample = {}
            estimated_tcp = None
            # ``execution.measured_tcp_pose`` always means the configured OSC
            # control frame evaluated from the same joint sample as Pink.  No
            # control sample exists while idle/FREEDRIVE, so never substitute
            # the SDK's separately-defined TCP here.
            measured_tcp = None
            sampled_target = None
            estimated_tracking_error = None
            measured_tracking_error = None
            last_output = {"status": "held", "final_joint_target_rad": None,
                           "final_joint_velocity_rad_s": [0.0] * 7, "sequence": 0,
                           "epoch": raw_diagnostics.get("motion_epoch")}
        pink = dict(last_result.get("solver") or {})
        gate = {
            "status": last_output.get("status", "held"),
            "accepted": bool(last_result.get("ok", False)),
            "limited": bool(last_result.get("gate_limited", False)),
            "reason": last_result.get("gate_reason"),
        }
        diagnostics = {
            "loop_count": raw_diagnostics.get("loop_count", 0),
            # ``tcp_error`` is kept as the public compatibility alias, but it
            # now has the unambiguous physical/plant meaning used by the UI.
            "tcp_error": measured_tracking_error,
            "estimated_tracking_error": estimated_tracking_error,
            "measured_tracking_error": measured_tracking_error,
            "pink": pink,
            "ruckig": last_result.get("ruckig"),
            "safety_gate": gate,
            "timing": raw_diagnostics.get("timing", {}) if active_session else {},
            "cycle_trace": raw_diagnostics.get("cycle_trace", {}),
            "state_estimator": raw_diagnostics.get("state_estimator", {}) if active_session else {},
            "shadow_transport": raw_diagnostics.get("shadow_transport", {"enabled": False}) if active_session else {"enabled": False},
            "trajectory_state": raw_diagnostics.get("trajectory_state"),
            "trajectory_brake_reason": raw_diagnostics.get("trajectory_brake_reason"),
            "arrival": servo.get("arrival"),
        }
        recent_batches = list(raw_diagnostics.get("recent_cpv_batches") or [])
        tx_owner = self._transport_owner.diagnostics()
        cpv_mailbox = dict(tx_owner.get("cpv_mailbox") or {})
        hardware_robot = snapshot.get("robot") or (snapshot.get("control") or {}).get("robot") or {}
        hardware_raw = hardware_robot.get("raw") if isinstance(hardware_robot, dict) else {}
        osc_rx = self._osc_cached_feedback()
        hardware_feedback = {
            # Canonical task TCP from the SDK's configured transform. It is
            # available at all times, including FREEDRIVE.
            "tcp_pose": absolute_pose_from_sdk_rpy(hardware_robot.get("tcp_pose")) if isinstance(hardware_robot, dict) else None,
            "tcp_source": hardware_raw.get("tcp_pose_source") if isinstance(hardware_raw, dict) else None,
            "joint_angles_rad": osc_rx.get("joint_angles_rad") or (hardware_robot.get("joint_angles_rad") if isinstance(hardware_robot, dict) else None),
            "joint_velocity_rad_s": osc_rx.get("joint_velocity_rad_s"),
            "arm_status": hardware_robot.get("arm_status") if isinstance(hardware_robot, dict) else None,
            "arm_status_feedback": (snapshot.get("control") or {}).get("arm_status_feedback"),
            "joint_feedback_source": (snapshot.get("control") or {}).get("joint_feedback_source"),
            "timestamp": hardware_robot.get("timestamp") if isinstance(hardware_robot, dict) else None,
            "feedback_age_s": osc_rx.get("age_s", snapshot.get("status_age_s")),
            "rx_revision": osc_rx.get("revision"),
            "rx_error": osc_rx.get("last_error"),
        }
        feedback_mailbox = {
            "revision": osc_rx.get("revision"),
            "sdk_timestamp": osc_rx.get("sdk_joint_timestamp"),
            "received_monotonic_ns": osc_rx.get("received_at_monotonic_ns"),
            "feedback_age_s": osc_rx.get("age_s"),
            "joint_feedback_hz": osc_rx.get("joint_feedback_hz"),
            "last_error": osc_rx.get("last_error"),
        }
        diagnostics["mailboxes"] = {"feedback": feedback_mailbox, "cpv": cpv_mailbox}
        observed_joints = execution_sample.get("joint_state_rad")
        if not isinstance(observed_joints, list):
            observed_joints = last_output.get("final_joint_target_rad") if shadow else (snapshot.get("robot") or {}).get("joint_angles_rad")
        measured_joints = execution_sample.get("measured_joint_state_rad") if active_session else (snapshot.get("robot") or {}).get("joint_angles_rad")
        estimated_joints = execution_sample.get("estimated_joint_state_rad") if active_session else None
        commanded_joints = last_output.get("final_joint_target_rad")
        joint_error = None
        if isinstance(commanded_joints, list) and isinstance(measured_joints, list) and len(commanded_joints) == len(measured_joints) == 7:
            error = [float(command) - float(measured) for command, measured in zip(commanded_joints, measured_joints)]
            joint_error = {"per_joint_rad": error, "max_abs_rad": max(abs(value) for value in error),
                           "norm_rad": math.sqrt(sum(value * value for value in error))}
        return {
            "schema_version": "nero.osc.v2",
            "state_sequence": servo.get("state_sequence", 0),
            "session": session,
            "command": {
                "target_tcp": sampled_target if active_session else None,
                "target_generation": servo.get("target_generation", execution_sample.get("target_generation")),
                "final_joint_target_rad": last_output.get("final_joint_target_rad"),
                "final_joint_velocity_rad_s": last_output.get("final_joint_velocity_rad_s"),
                "sequence": last_output.get("sequence", (session or {}).get("sequence", 0)),
                "epoch": last_output.get("epoch", raw_diagnostics.get("motion_epoch")),
                "output_status": last_output.get("status", "held"),
            },
            "execution": {
                "mode": execution_mode,
                "sample_id": execution_sample.get("sample_id"),
                "sample_monotonic_ns": execution_sample.get("sample_monotonic_ns"),
                "commanded_joint_state_rad": commanded_joints,
                "observed_joint_state_rad": observed_joints,
                "measured_joint_state_rad": measured_joints,
                "estimated_joint_state_rad": estimated_joints,
                "joint_target_error": joint_error,
                "observed_source": "simulated_cpv_feedback" if shadow else "measured_can_feedback" if active_session else "none",
                "output_count": raw_diagnostics.get("output_count", 0),
                "accepting_targets": self._osc.accepting_targets(),
                "estimated_tcp_pose": estimated_tcp,
                "measured_tcp_pose": measured_tcp,
                "feedback_age_s": execution_sample.get("feedback_age_s") if active_session else None,
                "estimated_feedback_delay_s": execution_sample.get("estimated_feedback_delay_s"),
                "solver_latency_s": execution_sample.get("solver_latency_s"),
                "dispatch_interval_s": execution_sample.get("dispatch_interval_s"),
            },
            "diagnostics": diagnostics,
            "transport": {
                "connected": bool((snapshot.get("control") or {}).get("connected")),
                "reason": (snapshot.get("control") or {}).get("reason"),
                "can_health": snapshot.get("feedback_ready"),
                "hardware_feedback": hardware_feedback,
                "feedback_mailbox": feedback_mailbox,
                "participation": "shadow_simulated" if shadow else "active" if active_session else "not_participating",
                "cpv_dispatch_count": raw_diagnostics.get("cpv_dispatch_count", 0) if shadow else cpv_mailbox.get("sent_count", 0),
                "last_cpv_dispatch": None if shadow else (cpv_mailbox.get("last_success") or (recent_batches[-1] if recent_batches else None)),
                "last_cpv_mode_entry": None if shadow else self._last_cpv_mode_entry,
                "cpv_parameters": dict(self._cpv_profile_cache or raw_diagnostics.get("cpv_parameters", {"status": "not_read"})),
                "hardware_tx_owner": tx_owner,
                "cpv_mailbox": cpv_mailbox,
            },
            "solver": servo.get("solver"),
            "workspace": servo.get("workspace"),
            "gripper": snapshot.get("gripper"),
            "authority": self.authority_status(snapshot.get("control")),
            "active_action": snapshot.get("active_action"),
        }

    def osc_calibrate_readonly_hardware(self, sample_count: int = 50) -> dict[str, Any]:
        """Persist only the feedback-latency portion measurable without motion."""
        telemetry = self.collect_readonly_hardware_telemetry(sample_count)
        calibration = dict(telemetry["calibration"])
        feedback = dict(telemetry["feedback"])
        cpv_values = dict((telemetry.get("cpv_parameters") or {}).get("values") or {})
        def conservative_limit(name: str) -> float | None:
            values = cpv_values.get(name)
            if not isinstance(values, list):
                return None
            finite = [float(value) for value in values if value is not None and math.isfinite(float(value)) and float(value) > 0.0]
            return min(finite) if len(finite) == 7 else None
        cpv_speed = conservative_limit("cv")
        cpv_acceleration = conservative_limit("acc")
        applied_runtime = self._osc.apply_hardware_calibration(
            calibration, feedback, dict(telemetry["cpv_parameters"])
        )
        config_path = Path(__file__).resolve().parents[1] / "config" / "osc.json"
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(self._osc.configuration_json(), encoding="utf-8")
        temporary.replace(config_path)
        return {"ok": True, "telemetry": telemetry, "applied": {
            "shadow_transport": applied_runtime["shadow_transport"],
            "cpv_profile_limits": {"max_joint_speed_rad_s": cpv_speed, "max_joint_acceleration_rad_s2": cpv_acceleration},
            "state_estimator": applied_runtime["state_estimator"],
            "actuator_response_identified": False,
        }, "state": self.osc_state()}

    def osc_sync_cpv_profile_to_osc_limits(self) -> dict[str, Any]:
        """Official SDK configuration write, constrained to current OSC limits."""
        written = self.sync_cpv_profile_to_osc_limits()
        calibration = self.osc_calibrate_readonly_hardware(20)
        return {"ok": True, "written": written, "calibration": calibration["applied"], "state": self.osc_state()}

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
        the osc page's return action.  It prevents a console command from
        reaching the arm between a residual CPV command and a verified hold.
        """
        # Full status polling is intentionally paused while hardware CPV is
        # tracking. A stale aggregate snapshot must never block the safety
        # path that revokes the stream and returns to Follower/HOLD.
        self._require_operational_control(allow_stale=True)
        with self._handoff_lock:
            stopped = self._stop_osc_for_mode_transition(reason)
            if not stopped.get("ok"):
                stopped["handoff"] = {"stage": "osc_stop", "reason": stopped.get("reason")}
                self._log("osc_console_handoff", reason=reason, result=stopped)
                return stopped

            transition = self._set_authority(
                ArmWriter.MODE_TRANSITION,
                ServoMode.SUSPENDED,
                "official Follower/HOLD transition",
                advance_epoch=True,
                transport_exclusive_category="follower_hold_transition",
            )
            direct_handoff = self._osc.abandon_session_without_braking(
                int(transition["control_epoch"]),
                f"{reason}: revoke osc for Follower/HOLD",
            )
            handoff_detail = direct_handoff.get("direct_handoff", {})
            if not handoff_detail.get("threads_stopped", False):
                self._transport_owner.complete_epoch_transition(
                    int(transition["control_epoch"]), "follower_hold_transition"
                )
                self._set_authority(
                    ArmWriter.NONE,
                    ServoMode.SUSPENDED,
                    "Follower/HOLD aborted: osc workers did not stop",
                )
                return {"ok": False, "reason": "osc workers did not stop", "transition": transition}

            requested = {"type": "follower_hold", "reason": reason, "handoff": True}
            def follower_hold() -> dict[str, Any]:
                try:
                    return self.robot.call("p2", "hold_follower_without_position_target", reason,
                                           command_epoch=int(transition["control_epoch"]),
                                           category="follower_hold_transition").to_dict()
                finally:
                    self._transport_owner.complete_epoch_transition(
                        int(transition["control_epoch"]), "follower_hold_transition")
            hold = self._run_observed_operator_action(requested, follower_hold)
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
                    "osc": stopped["osc"].get("handoff", {}),
                    "cpv_stop": stopped["cpv_stop"],
                    "direct_handoff": handoff_detail,
                    "hold": hold.get("result", {}),
                },
                "revoked_lease": stopped["revoked_lease"],
                "hold": hold,
            }
            self._log("osc_console_handoff", reason=reason, result=result)
            return result

    def _stop_osc_for_mode_transition(self, reason: str) -> dict[str, Any]:
        """Shared P1 brake + CPV-zero barrier for official mode changes."""
        self._mark_active_preempted(reason)
        self.robot.request_preempt(reason)
        osc = self._osc.stop_session(f"{reason}: brake before mode transition")
        if not osc.get("handoff", {}).get("servo_stopped", False):
            return {"ok": False, "reason": "osc servo did not stop", "osc": osc}
        try:
            cpv_stop = self.robot.call("p2", "stop_cpv_for_mode_transition", reason,
                                       category="mode_transition_cpv_stop")
        except Exception as exc:
            cpv_stop = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not cpv_stop.get("ok"):
            return {"ok": False, "reason": "CPV did not stop", "osc": osc, "cpv_stop": cpv_stop}
        revoked = self.leases.release(force=True)
        return {
            "ok": True,
            "osc": osc,
            "cpv_stop": cpv_stop,
            "revoked_lease": revoked.public() if revoked else None,
        }

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

    def hold(self, reason: str = "operator requested hold") -> dict[str, Any]:
        # HOLD removes motion authority; it must remain available while the
        # full status cache is intentionally paused during hardware TRACK.
        # The handoff itself reads/holds through the serial transport and is
        # still rejected for a known disconnected or FAULT controller.
        self._require_operational_control(allow_stale=True)
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
            lambda: self.robot.hold_follower_without_position_target(reason).to_dict(),
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
            stopped = self._stop_osc_for_mode_transition(reason)
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
            direct_handoff = self._osc.abandon_session_without_braking(
                int(transition["control_epoch"]),
                f"{reason}: revoke osc for direct FREEDRIVE",
            )
            handoff_detail = direct_handoff.get("direct_handoff", {})
            if not handoff_detail.get("threads_stopped", False):
                self._transport_owner.complete_epoch_transition(
                    int(transition["control_epoch"]), "freedrive_transition"
                )
                self._set_authority(
                    ArmWriter.NONE,
                    ServoMode.SUSPENDED,
                    "direct FREEDRIVE aborted: osc workers did not stop",
                )
                result = {
                    "ok": False,
                    "reason": "osc workers did not stop; official FREEDRIVE was not entered",
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
            def enter_freedrive() -> dict[str, Any]:
                try:
                    return self.robot.call("p2", "enter_freedrive", reason,
                                           recover_emergency=recover_emergency,
                                           preserve_gripper=preserve_gripper,
                                           command_epoch=int(transition["control_epoch"]),
                                           category="freedrive_transition").to_dict()
                finally:
                    self._transport_owner.complete_epoch_transition(
                        int(transition["control_epoch"]), "freedrive_transition")
            result = self._run_observed_operator_action(requested, enter_freedrive)
            if result.get("ok"):
                self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "official FREEDRIVE active")
            else:
                # Direct transition failures must not surprise the operator by
                # injecting a CPV stop stream or a position hold target.
                self._set_authority(ArmWriter.NONE, ServoMode.SUSPENDED, "direct FREEDRIVE transition failed")
            result["direct_handoff"] = handoff_detail
            result["osc_brake"] = stopped["osc"].get("handoff", {})
            result["cpv_stop"] = stopped["cpv_stop"]
            result["revoked_lease"] = stopped["revoked_lease"]
        self._log(
            "safety_freedrive",
            recover_emergency=recover_emergency,
            preserve_gripper=preserve_gripper,
            result=result,
        )
        return result

    def command_gripper(
        self,
        mode: str,
        width_m: float | None,
        force_n: float,
        preserve_on_freedrive: bool,
        resume_osc: bool = False,
    ) -> dict[str, Any]:
        # HOLD is the safety escape from a tracking session. Hardware status
        # polling is deliberately paused while CPV owns the transport, so a
        # stale full snapshot must never prevent this revocation/handoff.
        self._require_operational_control(allow_stale=True)
        reason = f"operator requested gripper {mode}"
        osc_before = self._osc.status().get("session") or {}
        # Gripper I/O does not change the arm control mode. Keep an active
        # teleoperation session; arm mode transitions still take over.
        osc_preserved = osc_before.get("state") == "ACTIVE"
        revoked = None if osc_preserved else self._operator_takeover("gripper", reason)
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
        result["osc_resumed"] = False
        result["osc_preserved"] = osc_preserved
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
            if self._osc.heartbeat_expired():
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
        self._osc.stop_session(f"operator takeover: {operation}")
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
