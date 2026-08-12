"""Control-Supervisor ownership state for the sole NERO hardware writer."""

from __future__ import annotations

import queue
import threading
import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class CommandRevoked(PermissionError):
    """A queued hardware request belongs to an obsolete control epoch."""


class ServoWriteRevoked(CommandRevoked):
    """A queued P1 write became stale before reaching the SDK."""


class ArmWriter(str, Enum):
    SERVO = "SERVO"
    MODE_TRANSITION = "MODE_TRANSITION"
    SAFETY = "SAFETY"
    NONE = "NONE"


class ServoMode(str, Enum):
    TRACKING = "TRACKING"
    STOPPING = "STOPPING"
    HOLDING = "HOLDING"
    SUSPENDED = "SUSPENDED"


class CommandStream(str, Enum):
    CPV_POSITION = "CPV_POSITION"
    CPV_VELOCITY = "CPV_VELOCITY"
    CPV_ZERO = "CPV_ZERO"
    POSITION_HOLD = "POSITION_HOLD"
    NONE = "NONE"


@dataclass(frozen=True)
class WriteAuthority:
    writer: ArmWriter
    servo_mode: ServoMode
    epoch: int
    reason: str
    command_stream: CommandStream = CommandStream.NONE

    def public(self) -> dict[str, Any]:
        return {
            "arm_writer": self.writer.value,
            "servo_mode": self.servo_mode.value,
            "command_stream": self.command_stream.value,
            "control_epoch": self.epoch,
            "reason": self.reason,
        }


class ControlSupervisor:
    """Serialises ownership changes before the transport can write hardware."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = WriteAuthority(ArmWriter.NONE, ServoMode.SUSPENDED, 0, "service not started")

    def snapshot(self) -> WriteAuthority:
        with self._lock:
            return self._state

    def transition(
        self,
        writer: ArmWriter,
        servo_mode: ServoMode,
        reason: str,
        *,
        command_stream: CommandStream | None = None,
        advance_epoch: bool = False,
    ) -> WriteAuthority:
        with self._lock:
            epoch = self._state.epoch + 1 if advance_epoch else self._state.epoch
            if command_stream is None:
                if writer is ArmWriter.SERVO and servo_mode in {ServoMode.TRACKING, ServoMode.STOPPING}:
                    command_stream = CommandStream.CPV_POSITION
                elif writer is ArmWriter.SERVO and servo_mode is ServoMode.HOLDING:
                    command_stream = CommandStream.POSITION_HOLD
                else:
                    command_stream = CommandStream.NONE
            self._state = WriteAuthority(writer, servo_mode, epoch, reason, command_stream)
            return self._state

    def allows_servo(self, session_id: str | None, epoch: int) -> bool:
        del session_id  # OSC validates session identity; epoch gates cross-session output.
        with self._lock:
            return (
                self._state.writer is ArmWriter.SERVO
                and self._state.servo_mode in {ServoMode.TRACKING, ServoMode.STOPPING, ServoMode.HOLDING}
                and self._state.epoch == int(epoch)
            )

    def allows_transition(self) -> bool:
        with self._lock:
            return self._state.writer is ArmWriter.MODE_TRANSITION

    def allows_safety(self) -> bool:
        with self._lock:
            return self._state.writer is ArmWriter.SAFETY


class HardwareTxOwner:
    """The only SDK/CAN TX owner, with P0/P1/P2 request priority.

    A vendor call already in progress cannot be interrupted safely from Python,
    but queued P0 work always wins at the next SDK-call boundary. Every SDK
    operation that can emit a CAN frame runs on this one thread; the vendor
    CANDO RX thread remains independent and can receive while TX is active.
    """

    _PRIORITY = {"p0": 0, "p1": 1, "p2": 2}
    _STOP = object()

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._queue: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._epoch_lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._epoch = 0
        self._pending_epoch: int | None = None
        self._exclusive_category: str | None = None
        self._active: dict[str, Any] | None = None
        self._active_lock = threading.Lock()
        # The realtime position stream is deliberately not a Queue.  It is a
        # capacity-one mailbox: an unfinished SDK send may be slow, but it
        # must never cause historical CPV targets to accumulate behind it.
        self._cpv_lock = threading.RLock()
        self._cpv_mailbox: dict[str, Any] | None = None
        self._cpv_revision = 0
        self._cpv_last_dispatched_revision = 0
        self._cpv_results: dict[int, dict[str, Any]] = {}
        self._cpv_sent_count = 0
        self._cpv_superseded_count = 0
        self._cpv_revoked_count = 0
        self._cpv_failed_count = 0
        self._cpv_last_success: dict[str, Any] | None = None
        self._cpv_last_result: dict[str, Any] | None = None
        self._cpv_last_target: list[float] | None = None
        self._cpv_last_velocity: list[float] | None = None
        self._cpv_last_finished_ns: int | None = None
        self._cpv_result_changed = threading.Condition(self._cpv_lock)
        self._wake = threading.Condition()
        self._last_work_was_cpv = False
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="nero-hardware-tx", daemon=True)
        self._thread.start()

    def publish_cpv(
        self,
        command: dict[str, Any],
        *,
        execute_guard: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace the pending CPV target without waiting for CAN."""
        values = command.get("joint_target_rad")
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError("CPV mailbox requires seven joint targets")
        if not all(isinstance(value, (int, float)) and value == value and abs(float(value)) != float("inf") for value in values):
            raise ValueError("CPV mailbox contains non-finite joint target")
        with self._epoch_lock:
            epoch = int(command.get("epoch", self._epoch))
        now_ns = time.monotonic_ns()
        with self._cpv_result_changed:
            self._cpv_revision += 1
            revision = self._cpv_revision
            previous = self._cpv_mailbox
            if previous is not None:
                previous_revision = int(previous["mailbox_revision"])
                self._cpv_superseded_count += 1
                self._cpv_results[previous_revision] = {
                    "status": "superseded", "mailbox_revision": previous_revision,
                    "control_sample_id": previous.get("control_sample_id"),
                    "superseded_monotonic_ns": now_ns,
                }
            entry = {
                **command,
                "joint_target_rad": [float(value) for value in values],
                "epoch": epoch,
                "mailbox_revision": revision,
                "published_monotonic_ns": int(command.get("published_monotonic_ns") or now_ns),
                "execute_guard": execute_guard,
            }
            self._cpv_mailbox = entry
            self._cpv_results[revision] = {
                "status": "pending", "mailbox_revision": revision,
                "control_sample_id": entry.get("control_sample_id"),
                "published_monotonic_ns": entry["published_monotonic_ns"],
            }
            self._trim_cpv_results()
            self._cpv_result_changed.notify_all()
        with self._wake:
            self._wake.notify()
        return {"status": "pending", "mailbox_revision": revision, "published_monotonic_ns": entry["published_monotonic_ns"]}

    def wait_cpv_result(self, mailbox_revision: int, timeout_s: float) -> dict[str, Any]:
        """Bounded barrier for a mode handoff; never used by steady tracking."""
        deadline = time.monotonic() + max(0.001, float(timeout_s))
        with self._cpv_result_changed:
            while True:
                result = dict(self._cpv_results.get(int(mailbox_revision), {}))
                if result.get("status") in {"sent", "superseded", "revoked", "failed"}:
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"status": "timeout", "mailbox_revision": int(mailbox_revision)}
                self._cpv_result_changed.wait(remaining)

    def cpv_diagnostics(self) -> dict[str, Any]:
        with self._cpv_lock:
            pending = dict(self._cpv_mailbox) if self._cpv_mailbox else None
            if pending is not None:
                pending.pop("execute_guard", None)
                pending["pending_age_ms"] = max(0.0, (time.monotonic_ns() - int(pending["published_monotonic_ns"])) / 1e6)
            return {
                "mailbox_revision": self._cpv_revision,
                "last_dispatched_revision": self._cpv_last_dispatched_revision,
                "pending": pending,
                "sent_count": self._cpv_sent_count,
                "superseded_count": self._cpv_superseded_count,
                "revoked_count": self._cpv_revoked_count,
                "failed_count": self._cpv_failed_count,
                "last_result": dict(self._cpv_last_result) if self._cpv_last_result else None,
                "last_success": dict(self._cpv_last_success) if self._cpv_last_success else None,
            }

    def _trim_cpv_results(self) -> None:
        if len(self._cpv_results) <= 256:
            return
        for revision in sorted(self._cpv_results)[:-192]:
            self._cpv_results.pop(revision, None)

    def call(
        self,
        priority: str,
        method: str,
        *args: Any,
        command_epoch: int | None = None,
        category: str | None = None,
        execute_guard: Callable[[], bool] | None = None,
        dispatch_timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._closed.is_set():
            raise RuntimeError("hardware transport is closed")
        with self._epoch_lock:
            envelope_epoch = self._epoch if command_epoch is None else int(command_epoch)
            envelope_category = category or priority
        if threading.current_thread() is self._thread:
            return self._execute(
                method, args, kwargs, priority, envelope_epoch,
                envelope_category, execute_guard,
            )
        done = threading.Event()
        envelope: dict[str, Any] = {
            "method": method, "args": args, "kwargs": kwargs,
            "priority": priority, "command_epoch": envelope_epoch,
            "category": envelope_category, "execute_guard": execute_guard, "done": done,
            "queued_monotonic_ns": time.monotonic_ns(),
        }
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        self._queue.put((self._PRIORITY[priority], sequence, envelope))
        with self._wake:
            self._wake.notify()
        completed = done.wait(
            None if dispatch_timeout_s is None else max(0.001, float(dispatch_timeout_s))
        )
        if not completed:
            raise TimeoutError(
                f"hardware transport call {method} did not finish within "
                f"{float(dispatch_timeout_s):.3f} s"
            )
        if "error" in envelope:
            raise envelope["error"]
        return envelope.get("result")

    def advance_epoch(self, epoch: int, *, exclusive_category: str | None = None) -> int:
        """Atomically invalidate queued non-P0 work from older epochs."""
        with self._epoch_lock:
            target = int(epoch)
            if target < self._epoch:
                raise ValueError(
                    f"transport epoch cannot move backward ({target} < {self._epoch})"
                )
            if self._pending_epoch is not None and target < self._pending_epoch:
                raise ValueError(
                    f"transport epoch cannot move behind pending epoch "
                    f"({target} < {self._pending_epoch})"
                )
            # Publish the barrier immediately.  The SDK call already inside a
            # vendor DLL cannot be interrupted, but no queued or mailbox CPV
            # command from the old epoch may start after this point.
            self._epoch = target
            self._exclusive_category = exclusive_category
            self._pending_epoch = None
        with self._cpv_result_changed:
            pending = self._cpv_mailbox
            if pending is not None and int(pending.get("epoch", -1)) != target:
                revision = int(pending["mailbox_revision"])
                self._cpv_mailbox = None
                self._cpv_revoked_count += 1
                self._cpv_results[revision] = {"status": "revoked", "mailbox_revision": revision,
                                               "control_sample_id": pending.get("control_sample_id"),
                                               "revoked_monotonic_ns": time.monotonic_ns(), "reason": "epoch advanced"}
                self._cpv_result_changed.notify_all()
        with self._wake:
            self._wake.notify()
        return target

    def complete_epoch_transition(self, epoch: int, category: str) -> None:
        """Reopen normal current-epoch dispatch after an exclusive transition."""
        with self._epoch_lock:
            if self._epoch == int(epoch) and self._exclusive_category == category:
                self._exclusive_category = None

    def epoch(self) -> int:
        with self._epoch_lock:
            return self._epoch

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        self._queue.put((99, sequence, self._STOP))
        with self._wake:
            self._wake.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def diagnostics(self) -> dict[str, Any]:
        """Return the current serial CAN dispatch without touching the SDK."""
        with self._active_lock:
            active = dict(self._active) if self._active else None
        if active is not None:
            active["age_ms"] = max(0.0, (time.monotonic_ns() - int(active["started_monotonic_ns"])) / 1e6)
        return {"active": active, "queued": self._queue.qsize(), "epoch": self.epoch(),
                "cpv_mailbox": self.cpv_diagnostics()}

    def _loop(self) -> None:
        while True:
            envelope = self._take_high_priority_envelope()
            if envelope is self._STOP:
                return
            if envelope is not None:
                self._dispatch_envelope(envelope)
                self._last_work_was_cpv = False
                continue
            cpv = self._take_cpv() if not self._last_work_was_cpv else None
            if cpv is not None:
                self._dispatch_cpv(cpv)
                self._last_work_was_cpv = True
                continue
            envelope = self._take_any_envelope()
            if envelope is self._STOP:
                return
            if envelope is not None:
                self._dispatch_envelope(envelope)
                self._last_work_was_cpv = False
                continue
            cpv = self._take_cpv()
            if cpv is not None:
                self._dispatch_cpv(cpv)
                self._last_work_was_cpv = True
                continue
            with self._wake:
                self._wake.wait(0.05)

    def _take_high_priority_envelope(self) -> dict[str, Any] | None:
        with self._queue.mutex:
            if not self._queue.queue or self._queue.queue[0][0] > self._PRIORITY["p1"]:
                return None
        _, _, envelope = self._queue.get_nowait()
        return envelope

    def _take_any_envelope(self) -> dict[str, Any] | None:
        try:
            _, _, envelope = self._queue.get_nowait()
            return envelope
        except queue.Empty:
            return None

    def _take_cpv(self) -> dict[str, Any] | None:
        with self._cpv_result_changed:
            entry = self._cpv_mailbox
            self._cpv_mailbox = None
            return dict(entry) if entry is not None else None

    def _dispatch_envelope(self, envelope: dict[str, Any]) -> None:
        try:
            dispatch_started_ns = time.monotonic_ns()
            with self._active_lock:
                self._active = {"method": envelope["method"], "priority": envelope["priority"],
                                "category": envelope["category"], "command_epoch": envelope["command_epoch"],
                                "started_monotonic_ns": dispatch_started_ns}
            result = self._execute(envelope["method"], envelope["args"], envelope["kwargs"], envelope["priority"], envelope["command_epoch"], envelope["category"], envelope.get("execute_guard"))
            dispatch_finished_ns = time.monotonic_ns()
            if isinstance(result, dict):
                result = dict(result); queued_ns = int(envelope.get("queued_monotonic_ns") or dispatch_started_ns)
                result.setdefault("transport_queued_monotonic_ns", queued_ns)
                result.setdefault("transport_dispatch_started_monotonic_ns", dispatch_started_ns)
                result.setdefault("transport_dispatch_finished_monotonic_ns", dispatch_finished_ns)
                result.setdefault("queue_delay_ms", max(0.0, (dispatch_started_ns - queued_ns) / 1e6))
                result.setdefault("transport_duration_ms", max(0.0, (dispatch_finished_ns - dispatch_started_ns) / 1e6))
            envelope["result"] = result
        except Exception as exc:
            envelope["error"] = exc
        finally:
            with self._active_lock: self._active = None
            envelope["done"].set()

    def _dispatch_cpv(self, entry: dict[str, Any]) -> None:
        revision = int(entry["mailbox_revision"])
        started_ns = time.monotonic_ns()
        try:
            with self._epoch_lock:
                valid_epoch = int(entry["epoch"]) == self._epoch and self._exclusive_category is None
            guard = entry.get("execute_guard")
            if not valid_epoch or (callable(guard) and not guard()):
                raise ServoWriteRevoked("CPV mailbox command revoked before SDK dispatch")
            values = [float(value) for value in entry["joint_target_rad"]]
            if not all(math.isfinite(value) for value in values):
                raise ServoWriteRevoked("CPV mailbox contains non-finite target")
            # Revalidate against the actual interval between successful CAN
            # batches.  Computation samples may have been superseded while a
            # slow SDK call ran, so the OSC-cycle dt is no longer sufficient.
            last_finished_ns = self._cpv_last_finished_ns
            previous_target = self._cpv_last_target
            previous_velocity = self._cpv_last_velocity
            final_gate_limited = False
            if last_finished_ns is not None and previous_target is not None and previous_velocity is not None:
                actual_dt = max(0.001, (started_ns - last_finished_ns) / 1e9)
                velocity = [(target - previous) / actual_dt for target, previous in zip(values, previous_target)]
                max_speed = float(entry.get("max_joint_speed_rad_s", float("inf")))
                max_acceleration = float(entry.get("max_joint_acceleration_rad_s2", float("inf")))
                bounded_velocity = []
                for desired, prior in zip(velocity, previous_velocity):
                    bounded = max(-max_speed, min(max_speed, desired))
                    max_delta = max_acceleration * actual_dt
                    bounded = max(prior - max_delta, min(prior + max_delta, bounded))
                    bounded_velocity.append(bounded)
                final_gate_limited = any(abs(actual - desired) > 1e-9 for actual, desired in zip(bounded_velocity, velocity))
                velocity = bounded_velocity
                values = [previous + command * actual_dt for previous, command in zip(previous_target, velocity)]
            else:
                velocity = [float(value) for value in entry.get("joint_velocity_rad_s") or [0.0] * 7]
            with self._active_lock:
                self._active = {"method": "send_cpv_position", "priority": "p1", "category": "servo_position",
                                "command_epoch": entry["epoch"], "mailbox_revision": revision, "started_monotonic_ns": started_ns}
            result = self.backend.send_cpv_position(values)
            finished_ns = time.monotonic_ns()
            result = dict(result or {})
            result.update({"status": "sent", "mailbox_revision": revision,
                           "control_sample_id": entry.get("control_sample_id"),
                           "target_generation": entry.get("target_generation"),
                           "published_monotonic_ns": entry["published_monotonic_ns"],
                           "transport_dispatch_started_monotonic_ns": started_ns,
                           "transport_dispatch_finished_monotonic_ns": finished_ns,
                           "queue_delay_ms": max(0.0, (started_ns - int(entry["published_monotonic_ns"])) / 1e6),
                           "transport_duration_ms": max(0.0, (finished_ns - started_ns) / 1e6),
                           "compute_to_send_delay_ms": max(0.0, (started_ns - int(entry["published_monotonic_ns"])) / 1e6),
                           "finished_monotonic_ns": int(result.get("finished_monotonic_ns") or finished_ns),
                           "joint_target_rad": list(values), "joint_velocity_rad_s": list(velocity),
                           "final_gate_limited": final_gate_limited})
            with self._cpv_result_changed:
                self._cpv_sent_count += 1; self._cpv_last_dispatched_revision = revision
                self._cpv_last_target = list(values); self._cpv_last_velocity = list(velocity); self._cpv_last_finished_ns = finished_ns
                self._cpv_last_success = dict(result); self._cpv_last_result = dict(result); self._cpv_results[revision] = dict(result)
                self._trim_cpv_results(); self._cpv_result_changed.notify_all()
        except ServoWriteRevoked as exc:
            with self._cpv_result_changed:
                self._cpv_revoked_count += 1; result = {"status": "revoked", "mailbox_revision": revision, "error": str(exc), "control_sample_id": entry.get("control_sample_id")}; self._cpv_last_result = dict(result); self._cpv_results[revision] = result; self._cpv_result_changed.notify_all()
        except Exception as exc:
            with self._cpv_result_changed:
                self._cpv_failed_count += 1; result = {"status": "failed", "mailbox_revision": revision, "error": f"{type(exc).__name__}: {exc}", "control_sample_id": entry.get("control_sample_id")}; self._cpv_last_result = dict(result); self._cpv_results[revision] = result; self._cpv_result_changed.notify_all()
        finally:
            with self._active_lock: self._active = None

    def _execute(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        priority: str,
        command_epoch: int,
        category: str,
        execute_guard: Callable[[], bool] | None,
    ) -> Any:
        # The pending-epoch barrier closes the gap after a currently running
        # SDK call finishes: old queued work cannot win the next dispatch.
        with self._dispatch_lock:
            with self._epoch_lock:
                current_epoch = self._epoch
                pending_epoch = self._pending_epoch
                exclusive_category = self._exclusive_category
            if priority != "p0" and (
                pending_epoch is not None
                or command_epoch != current_epoch
                or (exclusive_category is not None and category != exclusive_category)
            ):
                error_type = ServoWriteRevoked if priority == "p1" else CommandRevoked
                detail = (
                    f"pending epoch {pending_epoch}"
                    if pending_epoch is not None
                    else (
                        f"exclusive category {exclusive_category}"
                        if exclusive_category is not None
                        else f"transport epoch {current_epoch}"
                    )
                )
                raise error_type(
                    f"queued {category}/{method} belongs to epoch {command_epoch}; {detail}"
                )
            if execute_guard is not None and not execute_guard():
                raise ServoWriteRevoked(
                    f"queued {category}/{method} was revoked before SDK dispatch"
                )
            result = getattr(self.backend, method)(*args, **kwargs)
            # Connecting necessarily emits SDK traffic before a CAN comm
            # exists. Bind the gate immediately afterwards so every later
            # send must originate from this owner thread.
            if method == "connect":
                binder = getattr(self.backend, "bind_tx_owner_thread", None)
                if callable(binder):
                    binder(threading.get_ident())
            return result


class TransportRobotProxy:
    """Compatibility proxy: default Broker operations are P2 transactions."""

    def __init__(self, owner: HardwareTxOwner, backend: Any) -> None:
        self._owner, self._backend = owner, backend

    def call(
        self,
        priority: str,
        method: str,
        *args: Any,
        command_epoch: int | None = None,
        category: str | None = None,
        execute_guard: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._owner.call(
            priority, method, *args, command_epoch=command_epoch,
            category=category, execute_guard=execute_guard, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._backend, name)
        # This only flips NeroRobot's local cooperative-preemption event.  It
        # never touches the SDK/CAN transport and must remain immediately
        # available while a cancellable P2 action is in progress.
        if name == "request_preempt" and callable(attribute):
            return attribute
        if callable(attribute):
            return lambda *args, **kwargs: self._owner.call("p2", name, *args, **kwargs)
        return attribute
