from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from shared.schemas import ExecutedAction, GripperState, RobotState, as_list, jsonable, now_iso, unwrap_msg
from motion.safety import SafetyConfig, SafetyLayer, arm_status_has_error, arm_status_is_no_solution_only


class MotionPreempted(RuntimeError):
    """Raised when an operator safety command preempts an active action."""


def call_safe(name: str, func: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "value": jsonable(func())}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "call": name}


def call_joint_indexed(name: str, getter: Callable[[int], Any], joint_count: int = 7) -> list[dict[str, Any]]:
    return [call_safe(f"{name}({idx})", lambda idx=idx: getter(idx)) for idx in range(1, joint_count + 1)]


class NeroRobot:
    def __init__(self, config: dict[str, Any] | Path | str) -> None:
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text(encoding="utf-8-sig"))
        self.config = config
        self.sdk_config = config.get("sdk", {})
        self.motion_config = config.get("motion", {})
        self.logging_config = config.get("logging", {})
        self.safety = SafetyLayer(SafetyConfig.from_dict(config.get("safety", {})))
        self.robot: Any | None = None
        self.gripper: Any | None = None
        self._preempt_event = threading.Event()
        self._preempt_lock = threading.Lock()
        self._preempt_epoch = 0
        self._command_lock = threading.RLock()
        self._transition_lock = threading.RLock()
        self._control_mode = "DISCONNECTED"
        self._cpv_dispatch_progress: dict[str, Any] = {"state": "idle"}
        self._freedrive_backend: str | None = None
        self._last_control_reason = "not connected"
        self._leader_feedback_lock = threading.Lock()
        self._leader_joint_angles: list[float] | None = None
        self._leader_sdk_timestamp: Any | None = None
        self._leader_seen_monotonic: float | None = None
        self._leader_feedback_hz: float | None = None
        self._emergency_latched = False
        self._freedrive_recovery: dict[str, Any] = {
            "state": "idle",
            "stage": None,
            "error": None,
        }
        self._gripper_requires_enable_clear = False
        self._cpv_stream_started = False
        self._cpv_auto_mode_before_stream: bool | None = None
        self._arm_status_lock = threading.Lock()
        self._arm_status_revision = 0
        self._arm_status_snapshot: dict[str, Any] = {
            "revision": 0, "received_monotonic_ns": None,
            "sdk_timestamp": None, "ctrl_mode": None, "mode_feedback": None,
        }
        self._arm_status_observer_installed = False
        self._connect_stage = "idle"
        self._task_tcp_offset_m = self._configured_task_tcp_offset()

    def _configured_task_tcp_offset(self) -> list[float]:
        """Return the sole task-point offset shared by SDK feedback and OSC."""
        offset = self.sdk_config.get("task_tcp_offset_from_flange_m")
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            raise ValueError("sdk.task_tcp_offset_from_flange_m must contain three finite metres")
        try:
            values = [float(value) for value in offset]
        except (TypeError, ValueError) as exc:
            raise ValueError("sdk.task_tcp_offset_from_flange_m must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sdk.task_tcp_offset_from_flange_m must be finite")
        return values

    def _apply_task_tcp_offset(self) -> dict[str, Any]:
        """Apply the process-local SDK TCP setting after each connection."""
        if self.robot is None:
            raise RuntimeError("robot not connected")
        setter = getattr(self.robot, "set_tcp_offset", None)
        if not callable(setter):
            raise RuntimeError("installed NERO SDK does not provide set_tcp_offset")
        pose = [*self._task_tcp_offset_m, 0.0, 0.0, 0.0]
        result = setter(pose)
        if result is False:
            raise RuntimeError("NERO SDK rejected task TCP offset")
        return {"offset_from_flange_m": list(self._task_tcp_offset_m), "sdk_result": jsonable(result)}

    @property
    def connect_stage(self) -> str:
        return self._connect_stage

    def connect(self) -> dict[str, Any]:
        self._connect_stage = "import_can_sdk"
        import can
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        firmware_name = str(self.sdk_config.get("firmware", "v120")).lower()
        firmware = getattr(NeroFW, {"default": "DEFAULT", "v111": "V111", "v112": "V112", "v120": "V120"}[firmware_name])
        interface = self.sdk_config.get("interface", "agx_cando")
        channel = str(self.sdk_config.get("channel", "0"))
        bitrate = int(self.sdk_config.get("bitrate", 1_000_000))
        local_loopback = bool(self.sdk_config.get("local_loopback", False))
        receive_own_messages = bool(self.sdk_config.get("receive_own_messages", False))
        with self._leader_feedback_lock:
            self._leader_joint_angles = None
            self._leader_sdk_timestamp = None
            self._leader_seen_monotonic = None
            self._leader_feedback_hz = None
        with self._arm_status_lock:
            self._arm_status_revision = 0
            self._arm_status_snapshot = {
                "revision": 0, "received_monotonic_ns": None,
                "sdk_timestamp": None, "ctrl_mode": None, "mode_feedback": None,
            }
            self._arm_status_observer_installed = False
        if interface == "agx_cando":
            self._connect_stage = "detect_usb_can"
            available = can.detect_available_configs(["agx_cando"])
            if not any(str(item.get("channel")) == channel for item in available):
                raise RuntimeError(f"USB-CAN channel {channel} is unavailable; detected configurations: {available}")
        self._connect_stage = "create_sdk_config"
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=firmware,
            interface=interface,
            channel=channel,
            bitrate=bitrate,
            local_loopback=local_loopback,
            receive_own_messages=receive_own_messages,
        )
        self._connect_stage = "create_sdk_arm"
        self.robot = AgxArmFactory.create_arm(cfg)
        try:
            self._connect_stage = "open_sdk_can_bus"
            self.robot.connect()
            self._install_arm_status_observer()
            self._connect_stage = "configure_task_tcp"
            task_tcp = self._apply_task_tcp_offset()
            self._connect_stage = "initialize_gripper"
            self.gripper = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self._connect_stage = "wait_follower_feedback"
            ready = self.wait_feedback_ready(float(self.motion_config.get("feedback_timeout_s", 3.0)))
            self._connect_stage = "read_initial_state"
            initial_state = self.read_state()
            initial_arm_status = initial_state.arm_status or {}
            leader_feedback = None
            if self._enum_int(initial_arm_status.get("teach_status")) == 0x01:
                self._control_mode = "FREEDRIVE"
                self._freedrive_backend = "drag_teach"
                self._last_control_reason = "connected while drag teaching is active"
            else:
                # A single cached Leader packet can survive a previous process.
                # Only treat Leader as active after repeated, changing feedback.
                self._connect_stage = "check_leader_feedback"
                self._read_leader_feedback()
                leader_feedback = self._wait_sustained_leader_feedback(
                    timeout=0.6, newer_than=self._leader_sdk_timestamp, required_updates=2
                )
                if leader_feedback is not None:
                    self._control_mode = "FREEDRIVE"
                    self._freedrive_backend = "leader"
                    self._last_control_reason = "connected while leader feedback is active"
                elif ready.get("ok"):
                    arm_status = self._enum_int((initial_state.arm_status or {}).get("arm_status"))
                    if arm_status == 0x01:
                        self._control_mode = "EMERGENCY_DAMPING"
                        self._emergency_latched = True
                        self._last_control_reason = "connected while electronic emergency stop is latched"
                    elif arm_status == 0x00:
                        self._control_mode = "HOLD"
                        self._emergency_latched = False
                        self._last_control_reason = "connected with normal follower feedback"
                    elif arm_status == 0x02 and arm_status_is_no_solution_only(initial_arm_status):
                        # NERO reports NO_SOLUTION after a stale/invalid
                        # Cartesian target.  It is recoverable with the next
                        # validated Cartesian target and is not an electrical
                        # or emergency fault; keep the service in safe HOLD so
                        # a service reset does not re-present a permanent FAULT.
                        self._control_mode = "HOLD"
                        self._emergency_latched = False
                        self._last_control_reason = (
                            "connected with recoverable NO_SOLUTION; awaiting a valid Cartesian target"
                        )
                    elif arm_status == 0x06:
                        # JOINT_BRAKE_NOT_RELEASED is a recoverable controller
                        # state, not a latched safety fault. It can appear with
                        # all drives disabled or with a partially enabled set
                        # after a prior HOLD/teleop transition. The explicit
                        # HOLD path owns re-enabling the joints, so do not
                        # promote this state to FAULT merely because the
                        # enable bitmap is mixed.
                        self._control_mode = "HOLD"
                        self._emergency_latched = False
                        self._last_control_reason = (
                            "connected in safe HOLD; joint brakes are not fully released"
                        )
                    else:
                        self._control_mode = "FAULT"
                        self._last_control_reason = f"connected with arm_status={arm_status}"
                else:
                    self._control_mode = "FAULT"
                    self._last_control_reason = "connected without fresh follower or leader feedback"
        except Exception:
            failed_stage = self._connect_stage
            call_safe("disconnect_after_connect_error", self.robot.disconnect)
            self.robot = None
            self.gripper = None
            self._connect_stage = f"failed:{failed_stage}"
            raise
        self._connect_stage = "connected"
        return {
            "config": jsonable(cfg),
            "channel": call_safe("get_channel", self.robot.get_channel),
            "is_ok": call_safe("is_ok", self.robot.is_ok),
            "firmware": call_safe("get_firmware", self.robot.get_firmware),
            "gripper_is_ok": call_safe("gripper.is_ok", self.gripper.is_ok),
            "feedback_ready": ready,
            "task_tcp": task_tcp,
        }

    def disconnect(self) -> dict[str, Any]:
        if self.robot is None:
            return {"ok": True, "value": "not_connected"}
        result = call_safe("disconnect", self.robot.disconnect)
        self.robot = None
        self.gripper = None
        self._control_mode = "DISCONNECTED"
        self._cpv_stream_started = False
        self._cpv_auto_mode_before_stream = None
        self._last_control_reason = "disconnected"
        return result

    def bind_tx_owner_thread(self, thread_id: int) -> dict[str, Any]:
        """Bind CAN send to the HardwareTxOwner worker after connection."""
        if self.robot is None:
            raise RuntimeError("robot is not connected")
        context = getattr(self.robot, "_ctx", None)
        getter = getattr(context, "get_comm", None)
        comm = getter() if callable(getter) else None
        binder = getattr(comm, "bind_tx_owner_thread", None)
        if not callable(binder):
            return {"bound": False, "reason": "CAN comm does not expose a TX owner gate"}
        binder(int(thread_id))
        return {"bound": True, "thread_id": int(thread_id)}

    def _install_arm_status_observer(self) -> None:
        """Capture immutable 0x2A1 snapshots after the SDK parser runs."""
        context = getattr(self.robot, "_ctx", None)
        register = getattr(context, "register_parser_packet_fun", None)
        if not callable(register):
            return
        register(self._capture_arm_status_frame)
        self._arm_status_observer_installed = True

    def _capture_arm_status_frame(self, frame: Any) -> None:
        if int(getattr(frame, "arbitration_id", -1)) != 0x2A1 or self.robot is None:
            return
        try:
            status = self.robot.get_arm_status()
            message = getattr(status, "msg", None)
            ctrl_mode = self._enum_int(getattr(message, "ctrl_mode", None))
            mode_feedback = self._enum_int(getattr(message, "mode_feedback", None))
            sdk_timestamp = getattr(status, "timestamp", None)
        except Exception:
            return
        with self._arm_status_lock:
            self._arm_status_revision += 1
            self._arm_status_snapshot = {
                "revision": self._arm_status_revision,
                "received_monotonic_ns": time.monotonic_ns(),
                "sdk_timestamp": jsonable(sdk_timestamp),
                "ctrl_mode": ctrl_mode,
                "mode_feedback": mode_feedback,
            }

    def arm_status_snapshot(self) -> dict[str, Any]:
        with self._arm_status_lock:
            return dict(self._arm_status_snapshot)

    def stop(self, reason: str = "stop requested") -> ExecutedAction:
        # Normal stops do not create a new J-mode position target.
        return self.hold_follower_without_position_target(reason)

    def request_preempt(self, reason: str = "operator preempt") -> None:
        with self._preempt_lock:
            self._last_control_reason = reason
            self._preempt_epoch += 1
            self._preempt_event.set()

    def preempt_epoch(self) -> int:
        with self._preempt_lock:
            return self._preempt_epoch

    def get_control_state(self) -> dict[str, Any]:
        state: RobotState | None = None
        error: str | None = None
        if self.robot is not None:
            try:
                state = self.read_state(include_motor_states=False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        with self._leader_feedback_lock:
            leader_age = (
                time.monotonic() - self._leader_seen_monotonic
                if self._leader_seen_monotonic is not None else None
            )
            leader_hz = self._leader_feedback_hz
        leader_active = self._control_mode == "FREEDRIVE" and self._freedrive_backend == "leader"
        return {
            "mode": self._control_mode,
            "reason": self._last_control_reason,
            "connected": self.robot is not None,
            "preempt_requested": self._preempt_event.is_set(),
            "robot": state.to_dict() if state is not None else None,
            "joint_feedback_source": (
                "leader" if self._control_mode == "FREEDRIVE" and self._freedrive_backend == "leader"
                else "follower"
            ),
            "freedrive_backend": self._freedrive_backend,
            "leader_feedback_age_s": leader_age if leader_active else None,
            "leader_feedback_hz": leader_hz if leader_active else None,
            "emergency_latched": self._emergency_latched,
            "freedrive_recovery": dict(self._freedrive_recovery),
            "continuous_stream_active": self._cpv_stream_started,
            "arm_status_feedback": self.arm_status_snapshot(),
            "error": error,
        }

    def hold_follower_without_position_target(
        self, reason: str = "CPV handoff requested"
    ) -> ExecutedAction:
        """Leave CPV in a verified Follower hold without creating a J target.

        This is the only OSC HOLD handoff: no J-mode switch and no ``move_j``
        command are sent after the CPV stream has reached zero velocity.
        """
        self.request_preempt(reason)
        transition_epoch = self.preempt_epoch()
        if self.robot is None:
            return self._control_result(
                "follower_hold", reason, False, False,
                {"error": "robot not connected", "position_target_sent": False},
            )
        with self._transition_lock:
            previous_mode = self._control_mode
            self._control_mode = "TRANSITIONING"
            try:
                if previous_mode == "EMERGENCY_DAMPING" or self._emergency_latched:
                    raise RuntimeError("emergency stop is latched; cannot enter follower hold")
                cpv_active = self._cpv_stream_started or previous_mode == "TELEOP_CPV"
                if cpv_active:
                    self._stop_cpv_stream()

                if previous_mode == "FREEDRIVE" and self._freedrive_backend == "drag_teach":
                    with self._command_lock:
                        self._set_drag_teach(False)
                    self._gripper_requires_enable_clear = True
                    if not self._wait_drag_teach(False, timeout=1.5):
                        raise RuntimeError("drag teaching exit was not confirmed")

                # CPV velocity is already a Follower-side controller mode.
                # Re-sending the vendor Follower configuration after a settled
                # CPV stream can reset its linkage gains and cause a torque
                # transient. Only issue it when returning from a real Leader
                # or drag-teach mode.
                needs_follower_mode = previous_mode == "FREEDRIVE"
                with self._command_lock:
                    if needs_follower_mode:
                        self.robot.set_follower_mode()
                    enabled, enable_status = self._enable_all_with_retry(
                        timeout=float(self.motion_config.get("enable_timeout_s", 8.0))
                    )
                    if not enabled:
                        raise RuntimeError(f"required joints are not enabled: {enable_status}")
                joints = self._read_stable_follower_joints(timeout=0.8)
                if len(joints) != 7:
                    raise RuntimeError("stable follower seven-joint feedback is unavailable")
                if self.preempt_epoch() != transition_epoch:
                    raise MotionPreempted("follower hold transition was superseded")

                self._control_mode = "HOLD"
                self._cpv_stream_started = False
                self._freedrive_backend = None
                self._last_control_reason = reason
                return self._control_result(
                    "follower_hold", reason, True, True,
                    {
                        "cpv_stream_stopped": cpv_active,
                        "cpv_zero_confirmed": True,
                        "follower_feedback_joint_angles_rad": list(joints),
                        "position_target_sent": False,
                        "follower_mode_command": (
                            "sent_returning_from_leader_or_drag"
                            if needs_follower_mode else "skipped_already_follower_cpv"
                        ),
                        "joint_enable_status": enable_status,
                    },
                )
            except Exception as exc:
                self._control_mode = "EMERGENCY_DAMPING" if self._emergency_latched else "FAULT"
                self._last_control_reason = f"follower hold failed: {type(exc).__name__}: {exc}"
                return self._control_result(
                    "follower_hold", reason, False, True,
                    {"error": f"{type(exc).__name__}: {exc}", "position_target_sent": False},
                )

    def enter_freedrive(
        self,
        reason: str = "operator takeover",
        recover_emergency: bool = False,
        preserve_gripper: bool = False,
    ) -> ExecutedAction:
        self.request_preempt(reason)
        transition_epoch = self.preempt_epoch()
        if self.robot is None:
            return self._control_result("freedrive", reason, False, False, {"error": "robot not connected"})
        # NERO's official Leader mode owns the integrated gripper channel.
        # Enter it without sending or promising any gripper command.
        with self._transition_lock:
            if self._control_mode == "FREEDRIVE" and self._freedrive_backend in {"leader", "drag_teach"}:
                feedback = self._wait_fresh_leader_feedback(timeout=0.3)
                if feedback is not None:
                    self._last_control_reason = reason
                    return self._control_result(
                        "freedrive", reason, True, False,
                        {
                            "already_active": True,
                            "leader_feedback": feedback,
                            "freedrive_feedback": feedback,
                            "gripper_policy": "unavailable_in_official_leader_mode",
                            "legacy_preserve_request_ignored": bool(preserve_gripper),
                        },
                    )
            self._control_mode = "TRANSITIONING"
            self._freedrive_recovery = {
                "state": "running" if self._emergency_latched else "idle",
                "stage": "starting" if self._emergency_latched else None,
                "error": None,
            }
            leader_enable_action = "emergency_recovery"
            enable_status: list[bool] = []
            try:
                if self._emergency_latched:
                    if not recover_emergency:
                        raise RuntimeError(
                            "emergency stop is latched; recovery requires explicit supported-arm confirmation"
                        )
                    feedback = self._recover_emergency_to_freedrive()
                    self._freedrive_backend = "leader"
                else:
                    with self._command_lock:
                        try:
                            raw_enable_status = self.robot.get_joints_enable_status_list()
                            enable_status = [bool(value) for value in (raw_enable_status or [])[:7]]
                        except Exception:
                            enable_status = []
                        if len(enable_status) == 7 and all(enable_status):
                            enabled = True
                            leader_enable_action = "skipped_already_enabled"
                        else:
                            enabled, enable_status = self._enable_all_with_retry(
                                timeout=float(self.motion_config.get("enable_timeout_s", 8.0))
                            )
                            leader_enable_action = "enabled_missing_or_unreadable_joint"
                        if not enabled:
                            raise RuntimeError(
                                f"not all joints could be enabled after retries: {enable_status}"
                            )
                        baseline_timestamp = self._leader_sdk_timestamp
                        self.robot.set_leader_mode()
                    feedback = self._wait_sustained_leader_feedback(
                        timeout=3.0, newer_than=baseline_timestamp
                    )
                    if feedback is None:
                        # The arm was held immediately before this transition;
                        # return to follower mode if Leader feedback cannot be
                        # proven instead of leaving an ambiguous control state.
                        with self._command_lock:
                            self.robot.set_follower_mode()
                        raise RuntimeError("sustained leader/zero-force feedback was not confirmed")
                    self._freedrive_backend = "leader"
                if self.preempt_epoch() != transition_epoch:
                    raise MotionPreempted("freedrive transition was superseded")
                self._control_mode = "FREEDRIVE"
                self._emergency_latched = False
                self._freedrive_recovery = {
                    "state": "succeeded",
                    "stage": "leader_feedback_confirmed",
                    "error": None,
                }
                self._last_control_reason = reason
                return self._control_result(
                    "freedrive", reason, True, True,
                    {
                        "leader_feedback": feedback,
                        "freedrive_feedback": feedback,
                        "gripper_policy": "unavailable_in_official_leader_mode",
                        "legacy_preserve_request_ignored": bool(preserve_gripper),
                        "leader_enable_action": leader_enable_action,
                        "joint_enable_status": enable_status,
                    },
                )
            except Exception as exc:
                self._control_mode = "EMERGENCY_DAMPING" if self._emergency_latched else "FAULT"
                self._last_control_reason = f"freedrive failed: {type(exc).__name__}: {exc}"
                if recover_emergency:
                    self._freedrive_recovery = {
                        "state": "failed",
                        "stage": self._freedrive_recovery.get("stage"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return self._control_result(
                    "freedrive", reason, False, True,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

    def e_stop(self, reason: str = "emergency stop requested") -> ExecutedAction:
        self.request_preempt(reason)
        result = {"ok": False, "error": "robot not connected"}
        if self.robot is not None:
            with self._command_lock:
                result = call_safe("electronic_emergency_stop", self.robot.electronic_emergency_stop)
        self._control_mode = "EMERGENCY_DAMPING" if result.get("ok") else "FAULT"
        self._emergency_latched = bool(result.get("ok"))
        self._last_control_reason = reason
        return self._control_result("emergency_damping", reason, bool(result.get("ok")), self.robot is not None, result)

    def _control_result(
        self, action_type: str, reason: str, ok: bool, sent: bool, result: dict[str, Any]
    ) -> ExecutedAction:
        action = {"type": action_type, "reason": reason}
        return ExecutedAction(now_iso(), action, action, ok, sent, reason, result)

    def wait_feedback_ready(self, timeout: float = 3.0, poll_interval: float = 0.05) -> dict[str, Any]:
        if self.robot is None:
            return {"ok": False, "error": "robot not connected"}
        required = {
            "arm_status": self.robot.get_arm_status,
            "joint_angles": self.robot.get_joint_angles,
            "tcp_pose": self.robot.get_tcp_pose,
        }
        start = time.monotonic()
        last: dict[str, Any] = {}
        while time.monotonic() - start <= timeout:
            last = {name: getter() is not None for name, getter in required.items()}
            if all(last.values()):
                return {"ok": True, "waited_s": round(time.monotonic() - start, 3), "fields": last}
            time.sleep(poll_interval)
        return {"ok": False, "timeout_s": timeout, "fields": last}

    def read_state(self, include_motor_states: bool = False) -> RobotState:
        self._require_connected()
        raw = {
            "arm_status": call_safe("get_arm_status", self.robot.get_arm_status),
            "joint_angles": call_safe("get_joint_angles", self.robot.get_joint_angles),
            "joint_enable_status": call_safe("get_joints_enable_status_list", self.robot.get_joints_enable_status_list),
        }
        motor_states: list[dict[str, Any]] = []
        driver_states: list[dict[str, Any]] = []
        if include_motor_states:
            motor_states = call_joint_indexed("get_motor_states", self.robot.get_motor_states)
            driver_states = call_joint_indexed("get_driver_states", self.robot.get_driver_states)
            raw["motor_states"] = motor_states
            raw["driver_states"] = driver_states
        joint_values = self._float_list(as_list(unwrap_msg(raw["joint_angles"])), 7)
        if self._control_mode == "FREEDRIVE" and self._freedrive_backend == "leader":
            leader = self._read_leader_feedback()
            if leader is not None:
                joint_values = leader["joint_angles_rad"]
                raw["leader_joint_angles"] = leader
                # In Leader mode NERO may keep follower pose feedback stale.
                # Use SDK FK of live joints, then its configured task transform.
                # The intermediate flange pose never leaves this adapter.
                fk_result = call_safe("fk(leader_joint_angles)", lambda: self.robot.fk(joint_values))
                fk_pose = self._float_list(as_list(fk_result.get("value")), 6) \
                    if fk_result.get("ok") else None
                if fk_pose is not None:
                    raw["tcp_pose"] = call_safe(
                        "get_flange2tcp_pose(fk_leader_joints)",
                        lambda: self.robot.get_flange2tcp_pose(fk_pose),
                    )
                    raw["tcp_pose_source"] = "sdk_tcp_from_leader_fk"
        if "tcp_pose" not in raw:
            raw["tcp_pose"] = call_safe("get_tcp_pose", self.robot.get_tcp_pose)
            raw["tcp_pose_source"] = "sdk_tcp_from_follower_feedback"
        return RobotState(
            timestamp=now_iso(),
            joint_angles_rad=joint_values,
            tcp_pose=self._float_list(as_list(unwrap_msg(raw["tcp_pose"])), 6),
            arm_status=unwrap_msg(raw["arm_status"]),
            joint_enable_status=as_list(unwrap_msg(raw["joint_enable_status"])),
            motor_states=motor_states,
            driver_states=driver_states,
            raw=raw,
        )

    def read_teleop_feedback(self) -> dict[str, Any]:
        """Return the minimal cached feedback set required by CPV velocity servo.

        This intentionally avoids pose/driver status reads.  The SDK
        updates joint and motor feedback from its CAN receive path, so the
        teleop cache thread can remain independent from the HTTP status path.
        """
        self._require_connected()
        raw_joints = call_safe("get_joint_angles", self.robot.get_joint_angles)
        raw_motors = call_joint_indexed("get_motor_states", self.robot.get_motor_states)
        joints = self._float_list(as_list(unwrap_msg(raw_joints)), 7)
        joint_message = raw_joints.get("value") if isinstance(raw_joints, dict) else None
        # ``call_safe`` makes SDK messages JSON-safe before they reach this
        # adapter.  Their timestamp/hz metadata is therefore normally a dict,
        # not an object with attributes.  Losing it made the feedback worker
        # treat all post-startup hardware samples as stale.
        if isinstance(joint_message, dict):
            sdk_joint_timestamp = joint_message.get("timestamp")
            joint_feedback_hz = joint_message.get("hz")
        else:
            sdk_joint_timestamp = getattr(joint_message, "timestamp", None)
            joint_feedback_hz = getattr(joint_message, "hz", None)
        velocities: list[float | None] = []
        for motor in raw_motors:
            value = unwrap_msg(motor)
            velocity = value.get("velocity") if isinstance(value, dict) else getattr(value, "velocity", None)
            velocities.append(float(velocity) if isinstance(velocity, (int, float)) and math.isfinite(float(velocity)) else None)
        return {
            "joint_angles_rad": joints,
            "joint_velocity_rad_s": velocities if len(velocities) == 7 else None,
            # These are metadata on the SDK's cached CAN message, not a new
            # CAN request.  Keep both the device/SDK timestamp and when this
            # process received the cached value.
            "sdk_joint_timestamp": jsonable(sdk_joint_timestamp),
            "joint_feedback_hz": joint_feedback_hz,
            "received_at_monotonic_ns": time.monotonic_ns(),
        }

    def read_gripper(self) -> GripperState:
        self._require_connected()
        if self.gripper is None:
            return GripperState(timestamp=now_iso(), raw={"ok": False, "error": "gripper unavailable"})
        status = call_safe("get_gripper_status", self.gripper.get_gripper_status)
        msg = unwrap_msg(status)
        width = float(msg["value"]) if isinstance(msg, dict) and isinstance(msg.get("value"), (int, float)) else None
        force = float(msg["force"]) if isinstance(msg, dict) and isinstance(msg.get("force"), (int, float)) else None
        mode = msg.get("mode") if isinstance(msg, dict) else None
        return GripperState(timestamp=now_iso(), width_m=width, force_n=force, mode=mode, status=msg if isinstance(msg, dict) else None, raw=status)

    def command_gripper(
        self,
        mode: str,
        width_m: float | None = None,
        force_n: float = 1.0,
        preserve_on_freedrive: bool = False,
    ) -> dict[str, Any]:
        self._require_connected()
        requested_preserve_on_freedrive = bool(preserve_on_freedrive)
        if self.gripper is None:
            raise RuntimeError("gripper unavailable")
        if self._control_mode == "FREEDRIVE" and self._freedrive_backend == "drag_teach":
            raise RuntimeError(
                "gripper commands are unavailable during NERO drag teaching; "
                "click HOLD before operating the gripper"
            )
        mode = str(mode).lower()
        if mode not in {"open", "grip", "position"}:
            raise ValueError(f"unsupported gripper mode: {mode}")
        if mode == "open":
            width_m = float(self.motion_config.get("gripper_open_width_m", 0.095))
        elif mode == "grip":
            width_m = 0.0
        if width_m is None:
            raise ValueError("width_m is required for position mode")
        width_m = float(width_m)
        force_n = float(force_n)
        if not math.isfinite(width_m) or not math.isfinite(force_n):
            raise ValueError("gripper width and force must be finite")
        min_width = float(self.safety.config.min_gripper_width_m)
        max_width = float(self.safety.config.max_gripper_width_m)
        min_force = float(self.motion_config.get("gripper_min_force_n", 0.2))
        max_force = float(self.safety.config.max_gripper_force_n)
        if not min_width <= width_m <= max_width:
            raise ValueError(f"gripper width must be within [{min_width}, {max_width}] m")
        if not min_force <= force_n <= max_force:
            raise ValueError(f"gripper force must be within [{min_force}, {max_force}] N")

        before = self.read_gripper()
        before_health = self._gripper_health(before)
        recoverable_faults = {"driver_disabled", "feedback_unavailable"}
        blocking_faults = [
            fault for fault in before_health.get("faults", [])
            if fault not in recoverable_faults
        ]
        if blocking_faults:
            return {
                "ok": False,
                "mode": mode,
                "target_width_m": width_m,
                "force_n": force_n,
                "preserve_on_freedrive": False,
                "requested_preserve_on_freedrive": requested_preserve_on_freedrive,
                "sent": False,
                "commands": [],
                "before": before.to_dict(),
                "gripper": before.to_dict(),
                "health": before_health,
                "gripper_hold": self.get_gripper_hold_state(),
            }

        commands: list[dict[str, Any]] = []
        recovered_from_disabled = (
            not before_health.get("driver_enabled", False)
            or self._gripper_requires_enable_clear
            or self._control_mode == "FREEDRIVE"
        )
        attempts = int(self.motion_config.get("gripper_enable_attempts", 3))
        settle_s = float(self.motion_config.get("gripper_verify_settle_s", 0.25))
        after = before
        health = before_health
        for _ in range(max(1, attempts)):
            with self._command_lock:
                if recovered_from_disabled:
                    from pyAgxArm.protocols.can_protocol.msgs.effector.agx_gripper.default import (
                        ArmMsgGripperCtrl,
                    )
                    command = call_safe(
                        "gripper_enable_clear_width",
                        lambda: self.gripper._send_msg(ArmMsgGripperCtrl(
                            value=round(width_m * 1e6),
                            force=round(force_n * 1e3),
                            status_code=0x03,
                        )),
                    )
                else:
                    command = call_safe(
                        "move_gripper_m",
                        lambda: self.gripper.move_gripper_m(value=width_m, force=force_n),
                    )
            commands.append(command)
            time.sleep(settle_s)
            after = self.read_gripper()
            verification_hold = (
                {"mode": "grip", "force_n": force_n, "baseline_width_m": before.width_m}
                if mode == "grip" else None
            )
            health = self._gripper_health(after, verification_hold, check_drift=False)
            if command.get("ok") and health["ok"]:
                break
            if any(fault not in recoverable_faults for fault in health.get("faults", [])):
                break
        ok = bool(commands[-1].get("ok")) and health["ok"]
        if ok and self._control_mode != "FREEDRIVE":
            self._gripper_requires_enable_clear = False
        return {
            "ok": ok,
            "mode": mode,
            "target_width_m": width_m,
            "force_n": force_n,
            "preserve_on_freedrive": False,
            "requested_preserve_on_freedrive": requested_preserve_on_freedrive,
            "sent": bool(command.get("ok")),
            "command": commands[-1],
            "commands": commands,
            "recovered_from_disabled": recovered_from_disabled,
            "before": before.to_dict(),
            "gripper": after.to_dict(),
            "health": health,
            "gripper_hold": self.get_gripper_hold_state(),
        }

    def clear_gripper_hold(self) -> dict[str, Any]:
        return self.get_gripper_hold_state()

    def release_gripper_zero_force(self, timeout: float = 1.0) -> dict[str, Any]:
        """Clear automatic hold and disable the gripper drive."""
        self._require_connected()
        if self.gripper is None:
            raise RuntimeError("gripper unavailable")

        before = self.read_gripper()
        hold = self.clear_gripper_hold()
        attempts: list[dict[str, Any]] = []
        after = before
        deadline = time.monotonic() + max(0.2, float(timeout))
        while time.monotonic() < deadline:
            with self._command_lock:
                attempts.append(call_safe("disable_gripper", self.gripper.disable_gripper))
            time.sleep(0.1)
            after = self.read_gripper()
            foc = (after.status or {}).get("foc_status", {})
            if foc.get("driver_enable_status") is False:
                return {
                    "ok": True,
                    "zero_force": True,
                    "before": before.to_dict(),
                    "gripper": after.to_dict(),
                    "commands": attempts,
                    "gripper_hold": hold,
                }

        return {
            "ok": False,
            "zero_force": False,
            "reason": "gripper drive-disable feedback was not confirmed",
            "before": before.to_dict(),
            "gripper": after.to_dict(),
            "commands": attempts,
            "gripper_hold": hold,
        }

    def get_gripper_hold_state(self) -> dict[str, Any]:
        # Kept in the status schema for compatibility with older clients.
        return {
            "supported": False,
            "active": False,
            "mode": None,
            "target_width_m": None,
            "force_n": None,
            "baseline_width_m": None,
            "alarm": None,
            "recovery_used": False,
            "updated_at": None,
            "reason": "NERO official Leader mode does not support active integrated-gripper hold",
        }

    def get_gripper_teaching_params(self) -> dict[str, Any]:
        self._require_connected()
        if self.gripper is None:
            raise RuntimeError("gripper unavailable")
        result = call_safe(
            "get_gripper_teaching_pendant_param",
            lambda: self.gripper.get_gripper_teaching_pendant_param(
                timeout=1.0, min_interval=0.0
            ),
        )
        value = result.get("value")
        message = value.get("msg") if isinstance(value, dict) else None
        return {
            "ok": bool(result.get("ok")) and isinstance(message, dict),
            "params": message,
            "raw": result,
        }

    def set_gripper_teaching_friction(self, teaching_friction: int) -> dict[str, Any]:
        self._require_connected()
        if self.gripper is None:
            raise RuntimeError("gripper unavailable")
        if self._control_mode != "HOLD":
            raise RuntimeError("gripper teaching parameters may only be changed in HOLD")
        teaching_friction = int(teaching_friction)
        if not 1 <= teaching_friction <= 10:
            raise ValueError("teaching_friction must be within [1, 10]")
        before = self.get_gripper_teaching_params()
        params = before.get("params") or {}
        command = call_safe(
            "set_gripper_teaching_pendant_param",
            lambda: self.gripper.set_gripper_teaching_pendant_param(
                teaching_range_per=int(params.get("teaching_range_per") or 100),
                max_range_config=float(params.get("max_range_config") or 0.0),
                teaching_friction=teaching_friction,
                timeout=1.5,
            ),
        )
        time.sleep(0.2)
        after = self.get_gripper_teaching_params()
        applied = (after.get("params") or {}).get("teaching_friction")
        return {
            "ok": bool(command.get("ok")) and bool(command.get("value")) and applied == teaching_friction,
            "requested_teaching_friction": teaching_friction,
            "before": before,
            "command": command,
            "after": after,
        }

    def monitor_gripper_hold(self) -> dict[str, Any] | None:
        return None

    def _gripper_health(
        self,
        state: GripperState,
        hold: dict[str, Any] | None = None,
        check_drift: bool = True,
    ) -> dict[str, Any]:
        status = state.status or {}
        foc = status.get("foc_status") if isinstance(status, dict) else None
        if not isinstance(foc, dict):
            return {"ok": False, "reason": "夹爪状态反馈不可用", "faults": ["feedback_unavailable"]}
        fault_keys = (
            "voltage_too_low", "motor_overheating", "driver_overcurrent",
            "driver_overheating", "driver_error_status",
        )
        faults = [key for key in fault_keys if bool(foc.get(key))]
        if not bool(foc.get("driver_enable_status")):
            faults.append("driver_disabled")
        if hold and check_drift and hold.get("mode") == "grip":
            baseline = hold.get("baseline_width_m")
            if baseline is not None and state.width_m is not None:
                max_drift = float(self.motion_config.get("gripper_max_release_drift_m", 0.001))
                if float(state.width_m) - float(baseline) > max_drift:
                    faults.append("opening_drift")
            requested_force = float(hold.get("force_n") or 0.0)
            if requested_force > 0 and (
                state.force_n is None or abs(float(state.force_n)) < min(0.1, requested_force * 0.2)
            ):
                faults.append("grip_force_missing")
        return {
            "ok": not faults,
            "reason": "夹爪正常" if not faults else "夹爪异常: " + ", ".join(faults),
            "faults": faults,
            "driver_enabled": bool(foc.get("driver_enable_status")),
            "width_m": state.width_m,
            "force_n": state.force_n,
        }

    def send_cpv_position(self, joints: list[float]) -> dict[str, Any]:
        """Send one continuous CPV position sample.

        OSC is the only caller.  It has already applied kinematics, trajectory,
        and final safety checks before this method serializes the CAN batch.
        """
        if self.robot is None:
            raise RuntimeError("robot is not connected")
        if not isinstance(joints, list) or len(joints) != 7:
            raise ValueError("CPV position requires seven joint values")
        values = [float(value) for value in joints]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CPV position contains non-finite values")
        started_ns = time.monotonic_ns()
        self._cpv_dispatch_progress = {
            "state": "starting", "started_monotonic_ns": started_ns,
            "joint_index": None, "joint_target_rad": None,
        }
        joint_sent_ns: list[int] = []
        with self._command_lock:
            cpv_entry = self._enter_cpv_stream_if_needed()
            move = getattr(self.robot, "move_cpv_pos", None)
            if move is None:
                raise RuntimeError("NERO SDK does not expose move_cpv_pos")
            for index, value in enumerate(values, start=1):
                self._cpv_dispatch_progress = {
                    "state": "sending", "started_monotonic_ns": started_ns,
                    "joint_index": index, "joint_target_rad": value,
                    "sent_joint_count": len(joint_sent_ns),
                }
                move(joint_index=index, pos=value)
                joint_sent_ns.append(time.monotonic_ns())
        finished_ns = time.monotonic_ns()
        self._cpv_dispatch_progress = {
            "state": "completed", "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": finished_ns, "sent_joint_count": len(joint_sent_ns),
        }
        self._control_mode = "TELEOP_CPV"
        return {
            "ok": True,
            "motion_mode": "CPV_POSITION",
            "joint_target_rad": values,
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": finished_ns,
            "joint_sent_monotonic_ns": joint_sent_ns,
            "batch_duration_ms": (finished_ns - started_ns) / 1e6,
            "batch_skew_ms": ((max(joint_sent_ns) - min(joint_sent_ns)) / 1e6) if joint_sent_ns else 0.0,
            "cpv_mode_entry": cpv_entry,
            "can_diagnostics": self.can_dispatch_diagnostics(consume_slow=True),
        }

    def _enter_cpv_stream_if_needed(self) -> dict[str, Any]:
        """Enter CPV once and require a post-command Arm Status revision."""
        if self._cpv_stream_started:
            return {"entered": False, "confirmed": True, **self.arm_status_snapshot()}
        if self.robot is None:
            raise RuntimeError("robot is not connected")
        enabled, enable_status = self._enable_all_with_retry(
            timeout=float(self.motion_config.get("enable_timeout_s", 8.0))
        )
        if not enabled:
            raise RuntimeError(f"required joints are not enabled: {enable_status}")
        if not self._arm_status_observer_installed:
            raise RuntimeError("cannot enter CPV: Arm Status RX revision observer is unavailable")
        baseline = self.arm_status_snapshot()
        auto_mode = True
        getter = getattr(self.robot, "get_auto_set_motion_mode_enabled", None)
        if callable(getter):
            auto_mode = bool(getter())
        setter = getattr(self.robot, "set_auto_set_motion_mode_enabled", None)
        if callable(setter):
            setter(False)
        try:
            mode = getattr(getattr(self.robot, "OPTIONS", None), "MOTION_MODE", None)
            cpv = getattr(mode, "CPV", "cpv")
            command_sent_ns = time.monotonic_ns()
            self.robot.set_motion_mode(cpv)
            confirmed = self._wait_for_cpv_mode_revision(
                baseline_revision=int(baseline["revision"]),
                timeout=float(self.motion_config.get("cpv_mode_confirm_timeout_s", 1.0)),
            )
            if confirmed is None:
                current = self.arm_status_snapshot()
                raise RuntimeError(
                    "CPV mode confirmation timed out: "
                    f"baseline_revision={baseline['revision']}, current={current}, "
                    f"command_sent_monotonic_ns={command_sent_ns}"
                )
        except Exception:
            if callable(setter):
                setter(auto_mode)
            raise
        self._cpv_auto_mode_before_stream = auto_mode
        self._cpv_stream_started = True
        return {
            "entered": True, "confirmed": True,
            "baseline_revision": baseline["revision"],
            "command_sent_monotonic_ns": command_sent_ns,
            **confirmed,
        }

    def _wait_for_cpv_mode_revision(self, baseline_revision: int, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.05, timeout)
        while time.monotonic() < deadline:
            snapshot = self.arm_status_snapshot()
            if (
                int(snapshot.get("revision") or 0) > int(baseline_revision)
                and snapshot.get("ctrl_mode") == 0x01
                and snapshot.get("mode_feedback") == 0x05
            ):
                return snapshot
            time.sleep(0.005)
        return None

    def can_dispatch_diagnostics(self, *, consume_slow: bool = False) -> dict[str, Any]:
        """Read bounded SDK/CAN TX diagnostics without issuing a CAN query."""
        robot = self.robot
        if robot is None:
            return {"available": False, "reason": "robot is not connected"}
        try:
            context = getattr(robot, "_ctx", None)
            getter = getattr(context, "get_comm", None)
            comm = getter() if callable(getter) else None
            diagnostics = getattr(comm, "send_diagnostics", None)
            if not callable(diagnostics):
                return {"available": False, "reason": "SDK CAN diagnostics are unavailable"}
            return {"available": True, **jsonable(diagnostics(consume_slow=consume_slow))}
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def cpv_dispatch_progress(self) -> dict[str, Any]:
        """Non-blocking introspection for a potentially stuck CPV batch."""
        return dict(self._cpv_dispatch_progress)

    def continuous_stream_active(self) -> bool:
        """Whether a CPV stream still owns the controller motion mode."""
        return bool(self._cpv_stream_started)

    def stop_cpv_for_mode_transition(self, reason: str = "mode transition requested") -> dict[str, Any]:
        """Quiesce CPV before an official Follower or Leader transition.

        This intentionally does not select J mode or create a position target.
        The caller owns the subsequent official mode transition.
        """
        if self.robot is None:
            raise RuntimeError("robot not connected")
        with self._transition_lock:
            was_active = bool(
                self._cpv_stream_started
                or self._control_mode == "TELEOP_CPV"
            )
            if was_active:
                self._stop_cpv_stream()
            self._last_control_reason = reason
            return {
                "ok": True,
                "cpv_was_active": was_active,
                "cpv_zero_confirmed": bool(was_active),
                "post_cpv_dwell_s": 0.0,
                "position_target_sent": False,
            }

    def _stop_cpv_stream(self, timeout: float = 0.8) -> None:
        """Quiesce CPV position mode by holding the latest measured joints."""
        state = self.read_state(include_motor_states=True)
        joints = [float(value) for value in state.joint_angles_rad]
        if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
            raise RuntimeError("CPV position handoff requires fresh seven-joint feedback")
        move = getattr(self.robot, "move_cpv_pos", None)
        if move is None:
            raise RuntimeError("NERO SDK does not expose move_cpv_pos for CPV handoff")
        try:
            with self._command_lock:
                for index, value in enumerate(joints, start=1):
                    move(joint_index=index, pos=value)
            if not self._wait_joint_velocity_settled(timeout=timeout):
                raise RuntimeError("CPV position hold was sent but joint motion did not settle")
        finally:
            previous_auto_mode = self._cpv_auto_mode_before_stream
            setter = getattr(self.robot, "set_auto_set_motion_mode_enabled", None)
            if previous_auto_mode is not None and callable(setter):
                setter(previous_auto_mode)
            self._cpv_auto_mode_before_stream = None
            self._cpv_stream_started = False

    def prime_cpv_position_from_feedback(self) -> dict[str, Any]:
        """Start CPV with a measured joint-position hold, never a velocity."""
        state = self.read_state(include_motor_states=False)
        joints = [float(value) for value in state.joint_angles_rad]
        if len(joints) != 7 or not all(math.isfinite(value) for value in joints):
            raise RuntimeError("CPV position prime requires fresh seven-joint feedback")
        return self.send_cpv_position(joints)

    def _wait_joint_velocity_settled(
        self, timeout: float = 0.8, poll_interval: float = 0.04
    ) -> bool:
        threshold = float(self.motion_config.get("cpv_handoff_velocity_tolerance_rad_s", 0.04))
        required_samples = int(self.motion_config.get("cpv_handoff_stable_samples", 3))
        deadline = time.monotonic() + max(0.1, timeout)
        stable_samples = 0
        while time.monotonic() < deadline:
            state = self.read_state(include_motor_states=True)
            velocities: list[float] = []
            for item in state.motor_states:
                payload = unwrap_msg(item)
                value = payload.get("velocity") if isinstance(payload, dict) else getattr(payload, "velocity", None)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    velocities.append(abs(float(value)))
            if len(velocities) == 7 and max(velocities, default=0.0) <= threshold:
                stable_samples += 1
                if stable_samples >= required_samples:
                    return True
            else:
                stable_samples = 0
            time.sleep(poll_interval)
        return False

    def _read_stable_follower_joints(
        self, timeout: float = 0.8, poll_interval: float = 0.04
    ) -> list[float]:
        """Sample follower feedback after mode handoff, rejecting moving data."""
        tolerance = float(self.motion_config.get("hold_feedback_stability_rad", 0.01))
        required_samples = int(self.motion_config.get("hold_feedback_stable_samples", 3))
        deadline = time.monotonic() + max(0.1, timeout)
        previous: list[float] | None = None
        stable_samples = 0
        latest: list[float] = []
        while time.monotonic() < deadline:
            state = self.read_state()
            joints = state.joint_angles_rad or []
            if len(joints) != 7 or not all(math.isfinite(float(value)) for value in joints):
                previous = None
                stable_samples = 0
                time.sleep(poll_interval)
                continue
            latest = [float(value) for value in joints]
            if previous is not None:
                drift = max(abs(latest[index] - previous[index]) for index in range(7))
                stable_samples = stable_samples + 1 if drift <= tolerance else 0
                if stable_samples >= required_samples:
                    return latest
            previous = latest
            time.sleep(poll_interval)
        return []

    def read_cpv_parameter(self, joint_index: int, name: str) -> Any:
        """Read one CPV setting without changing the controller configuration."""
        if self.robot is None:
            raise RuntimeError("robot is not connected")
        if joint_index < 1 or joint_index > 7:
            raise ValueError("joint_index must be in [1, 7]")
        if name not in {"acc", "dcc", "cv", "pp", "kp", "ki"}:
            raise ValueError(f"unsupported CPV parameter: {name}")
        getter = getattr(self.robot, f"get_cpv_{name}", None)
        if getter is None:
            raise RuntimeError(f"NERO SDK does not expose get_cpv_{name}")
        with self._command_lock:
            # A zero timeout makes every asynchronous CAN getter report
            # ``None`` before its response can arrive.  This is still a
            # read-only query; bound its wait so telemetry cannot monopolise
            # the transport owner.
            return getter(joint_index=joint_index, timeout=0.15, min_interval=0.0)

    def configure_cpv_profile(self, cv_rad_s: float, acc_rad_s2: float, dcc_rad_s2: float) -> dict[str, Any]:
        """Set the official CPV profile and verify every SDK ACK/read-back.

        This only changes the vendor controller profile and never dispatches
        a joint-position or velocity target.  CPV parameters are persisted
        in the motor controller's Flash, so values already matching the
        requested profile are deliberately not written again.
        """
        values = {"acc": float(acc_rad_s2), "dcc": float(dcc_rad_s2), "cv": float(cv_rad_s)}
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("CPV cv, acc and dcc must be finite positive values")
        if self.robot is None:
            raise RuntimeError("robot is not connected")
        rows: list[dict[str, Any]] = []
        with self._command_lock:
            # The v112/v120 SDK examples explicitly select CPV once before
            # issuing settings.  OSC is already in the verified CPV/Follower
            # HOLD state when it calls this method.  Disable the SDK's
            # per-request auto mode write so settings cannot repeatedly race
            # that mode message on the CAN channel.
            auto_mode = True
            if hasattr(self.robot, "get_auto_set_motion_mode_enabled"):
                auto_mode = bool(self.robot.get_auto_set_motion_mode_enabled())

            current: dict[int, dict[str, Any]] = {
                joint_index: {name: self.read_cpv_parameter(joint_index, name) for name in values}
                for joint_index in range(1, 8)
            }
            if hasattr(self.robot, "set_auto_set_motion_mode_enabled"):
                self.robot.set_auto_set_motion_mode_enabled(False)
            try:
                row_data = {
                    joint_index: {"joint_index": joint_index, "before": current[joint_index],
                                  "acknowledgements": {}, "readback": {}}
                    for joint_index in range(1, 8)
                }
                # Configure the two accel limits first.  They use the same
                # requested value and are independent of the CV profile.
                # This also prevents a firmware-rejected CV from leaving the
                # requested acceleration envelope only partly configured.
                for name in ("acc", "dcc", "cv"):
                    for joint_index in range(1, 8):
                        before = current[joint_index]
                        value = values[name]
                        if before.get(name) is not None and abs(float(before[name]) - value) <= 1e-6:
                            acknowledgement: bool | None = None  # already persisted; do not consume Flash life
                        else:
                            setter = getattr(self.robot, f"set_cpv_{name}", None)
                            if setter is None:
                                raise RuntimeError(f"NERO SDK does not expose set_cpv_{name}")
                            acknowledgement = bool(setter(joint_index=joint_index, **{name: value}, timeout=1.0))
                        # Some v120 firmware revisions apply a setting but do
                        # not emit the legacy 0xAC acknowledgement expected by
                        # the inherited v112 SDK.  Read-back is therefore the
                        # final safety authority; a false ACK is recorded but
                        # only rejected when the persisted value disagrees.
                        readback = self.read_cpv_parameter(joint_index, name)
                        row_data[joint_index]["acknowledgements"][name] = acknowledgement
                        row_data[joint_index]["readback"][name] = readback
                        if readback is None or abs(float(readback) - value) > 1e-6:
                            raise RuntimeError(
                                f"CPV {name} read-back mismatch at J{joint_index}: "
                                f"requested={value}, ack={acknowledgement}, before={before.get(name)}, got={readback}"
                            )
                rows = [row_data[joint_index] for joint_index in range(1, 8)]
            finally:
                if hasattr(self.robot, "set_auto_set_motion_mode_enabled"):
                    self.robot.set_auto_set_motion_mode_enabled(auto_mode)
        return {"ok": True, "profile": values, "joints": rows}

    def _move_gripper(self, width_m: float, force_n: float) -> dict[str, Any]:
        if self.gripper is None:
            raise RuntimeError("Gripper unavailable.")
        with self._command_lock:
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.P)
            command = call_safe("move_gripper_m", lambda: self.gripper.move_gripper_m(value=width_m, force=force_n))
        time.sleep(float(self.motion_config.get("gripper_settle_s", 1.0)))
        status = self.read_gripper()
        return {"ok": bool(command.get("ok")), "command": command, "gripper": status.to_dict()}

    def _wait_hold_stable(
        self, target: list[float], timeout: float = 1.0, poll_interval: float = 0.05
    ) -> tuple[bool, float | None]:
        start = time.monotonic()
        last_error: float | None = None
        stable_samples = 0
        while time.monotonic() - start <= timeout:
            state = self.read_state()
            actual = state.joint_angles_rad or []
            if len(actual) == 7:
                last_error = max(abs(float(actual[i]) - float(target[i])) for i in range(7))
                stable_samples = stable_samples + 1 if last_error <= 0.05 else 0
                if stable_samples >= 3:
                    return True, last_error
            time.sleep(poll_interval)
        return False, last_error

    def _wait_ctrl_mode(self, expected: int, timeout: float, poll_interval: float = 0.05) -> bool:
        start = time.monotonic()
        while time.monotonic() - start <= timeout:
            state = self.read_state()
            ctrl_mode = (state.arm_status or {}).get("ctrl_mode")
            if self._enum_int(ctrl_mode) == expected:
                return True
            time.sleep(poll_interval)
        return False

    def _set_drag_teach(self, active: bool) -> None:
        from pyAgxArm.protocols.can_protocol.msgs.piper.default import ArmMsgMotionCtrl

        self.robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x01 if active else 0x02))

    def _wait_drag_teach(
        self, active: bool, timeout: float, poll_interval: float = 0.05
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.read_state()
            teach_status = self._enum_int((state.arm_status or {}).get("teach_status"))
            if (teach_status == 0x01) is active:
                return True
            time.sleep(poll_interval)
        return False

    def _enable_all_with_retry(
        self, timeout: float = 8.0, poll_interval: float = 0.20
    ) -> tuple[bool, list[bool]]:
        """Enable only feedback-confirmed disabled joints.

        Re-sending a global SDK ``enable()`` to an already enabled NERO arm
        can create a visible torque transient. Every control path uses this
        helper instead of a blanket enable command.
        """
        deadline = time.monotonic() + timeout
        status: list[bool] = []
        attempted: set[int] = set()

        while time.monotonic() < deadline:
            try:
                raw_status = self.robot.get_joints_enable_status_list()
                status = [bool(value) for value in (raw_status or [])[:7]]
            except Exception:
                status = []
            if len(status) != 7:
                # Unknown feedback is not permission to re-enable all joints.
                time.sleep(poll_interval)
                continue
            disabled = [index for index, value in enumerate(status, start=1) if not value]
            if not disabled:
                return True, status

            # Never repeat an enable command within one transition.  If a
            # controller has accepted it but feedback is late, wait for that
            # feedback rather than re-enabling an already live drive.
            pending = [joint_index for joint_index in disabled if joint_index not in attempted]
            if not pending:
                time.sleep(poll_interval)
                continue
            for joint_index in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                joint_timeout = max(0.1, remaining)
                try:
                    self.robot.enable(joint_index=joint_index, timeout=joint_timeout)
                except TypeError:
                    self.robot.enable(joint_index)
                attempted.add(joint_index)
                time.sleep(poll_interval)

            raw_status = self.robot.get_joints_enable_status_list()
            status = [bool(value) for value in (raw_status or [])[:7]]
            if len(status) == 7 and all(status):
                return True, status
            time.sleep(poll_interval)
        return False, status

    def _reconnect_after_emergency_reset(self) -> None:
        if self.robot is not None:
            call_safe("disconnect_after_emergency_reset", self.robot.disconnect)
        self.robot = None
        self.gripper = None
        time.sleep(0.2)
        self.connect()
        if self.robot is None:
            raise RuntimeError("SDK reconnect after emergency reset failed")

    def _recover_emergency_to_freedrive(self) -> dict[str, Any]:
        attempts = int(self.motion_config.get("emergency_recovery_attempts", 2))
        settle_s = float(self.motion_config.get("emergency_reset_settle_s", 0.8))
        enable_timeout = float(self.motion_config.get("enable_timeout_s", 8.0))
        last_error = "recovery did not run"
        for attempt in range(1, max(1, attempts) + 1):
            try:
                self._freedrive_recovery.update({"stage": f"reset_attempt_{attempt}"})
                with self._command_lock:
                    # Reset has no acknowledgement frame. Repeating the
                    # idempotent command reduces the chance of a dropped CAN
                    # frame leaving the controller latched.
                    self.robot.reset()
                    time.sleep(0.10)
                    self.robot.reset()
                    time.sleep(settle_s)
                    self._freedrive_recovery.update({"stage": "waiting_for_post_reset_feedback"})
                    self._reconnect_after_emergency_reset()
                    self._wait_post_reset_feedback(
                        timeout=float(self.motion_config.get("emergency_recovery_feedback_timeout_s", 4.0))
                    )

                    # Establish zero-force semantics before drives regain
                    # torque, avoiding a transient follower-position target.
                    self._freedrive_recovery.update({"stage": "entering_leader"})
                    baseline_timestamp = self._leader_sdk_timestamp
                    self.robot.set_leader_mode()
                    time.sleep(0.15)
                    self._freedrive_recovery.update({"stage": "enabling_joints"})
                    enabled, enable_status = self._enable_all_with_retry(
                        timeout=enable_timeout
                    )
                    if not enabled:
                        raise RuntimeError(
                            f"attempt {attempt}: not all joints enabled: {enable_status}"
                        )
                feedback = self._wait_sustained_leader_feedback(
                    timeout=3.0, newer_than=baseline_timestamp
                )
                if feedback is None:
                    raise RuntimeError(
                        f"attempt {attempt}: sustained leader feedback was not confirmed"
                    )
                self._freedrive_recovery.update({"stage": "leader_feedback_confirmed"})
                return feedback
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
        raise RuntimeError(
            f"emergency recovery failed after {max(1, attempts)} attempts: {last_error}"
        )

    def _wait_post_reset_feedback(self, timeout: float) -> dict[str, Any]:
        """Wait for the controller to leave emergency status after reset."""
        deadline = time.monotonic() + max(0.1, float(timeout))
        last_status: Any = None
        while time.monotonic() < deadline:
            raw = call_safe("get_arm_status_after_reset", self.robot.get_arm_status)
            value = unwrap_msg(raw)
            if isinstance(value, dict):
                last_status = value
                arm_status = self._enum_int(value.get("arm_status"))
                teach_status = self._enum_int(value.get("teach_status"))
                if arm_status != 0x01 and teach_status != 0x01:
                    return {"ok": True, "arm_status": arm_status, "teach_status": teach_status}
            time.sleep(0.05)
        raise RuntimeError(
            f"post-reset non-emergency feedback was not confirmed: {jsonable(last_status)}"
        )

    def _read_leader_feedback(self) -> dict[str, Any] | None:
        self._require_connected()
        message = self.robot.get_leader_joint_angles()
        if message is None:
            return None
        values = getattr(message, "msg", None)
        angles = self._float_list(list(values) if isinstance(values, (list, tuple)) else values, 7)
        if angles is None:
            return None
        sdk_timestamp = getattr(message, "timestamp", None)
        hz_value = getattr(message, "hz", None)
        with self._leader_feedback_lock:
            if sdk_timestamp != self._leader_sdk_timestamp:
                self._leader_seen_monotonic = time.monotonic()
            self._leader_sdk_timestamp = sdk_timestamp
            self._leader_joint_angles = list(angles)
            try:
                self._leader_feedback_hz = float(hz_value) if hz_value is not None else None
            except (TypeError, ValueError):
                self._leader_feedback_hz = None
            age = (
                time.monotonic() - self._leader_seen_monotonic
                if self._leader_seen_monotonic is not None else None
            )
        return {
            "joint_angles_rad": list(angles),
            "sdk_timestamp": jsonable(sdk_timestamp),
            "hz": self._leader_feedback_hz,
            "age_s": age,
        }

    def _wait_fresh_leader_feedback(
        self,
        timeout: float,
        newer_than: Any | None = None,
        poll_interval: float = 0.02,
    ) -> dict[str, Any] | None:
        if newer_than is None:
            with self._leader_feedback_lock:
                newer_than = self._leader_sdk_timestamp
        start = time.monotonic()
        last: dict[str, Any] | None = None
        while time.monotonic() - start <= timeout:
            last = self._read_leader_feedback()
            with self._leader_feedback_lock:
                current_timestamp = self._leader_sdk_timestamp
                age = (
                    time.monotonic() - self._leader_seen_monotonic
                    if self._leader_seen_monotonic is not None else None
                )
            if last is not None and age is not None and age <= 0.25:
                if newer_than is None or current_timestamp != newer_than:
                    return last
            time.sleep(poll_interval)
        return None

    def _wait_sustained_leader_feedback(
        self,
        timeout: float,
        newer_than: Any | None = None,
        required_updates: int = 3,
        poll_interval: float = 0.05,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        previous = newer_than
        updates = 0
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            current = self._wait_fresh_leader_feedback(
                timeout=min(0.5, max(0.05, remaining)),
                newer_than=previous,
                poll_interval=min(0.02, poll_interval),
            )
            if current is None:
                continue
            current_timestamp = current.get("sdk_timestamp")
            if current_timestamp == previous:
                continue
            previous = current_timestamp
            last = current
            updates += 1
            if updates >= required_updates:
                return last
            time.sleep(poll_interval)
        return None

    def _raise_if_preempted(self) -> None:
        if self._preempt_event.is_set():
            raise MotionPreempted(self._last_control_reason)

    @staticmethod
    def _enum_int(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _require_connected(self) -> None:
        if self.robot is None:
            raise RuntimeError("NeroRobot is not connected.")

    @staticmethod
    def _float_list(values: Any, expected_len: int) -> list[float] | None:
        if not isinstance(values, list) or len(values) < expected_len:
            return None
        return [float(value) for value in values[:expected_len]]
