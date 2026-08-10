"""Control-Supervisor ownership state for the sole NERO hardware writer."""

from __future__ import annotations

import queue
import threading
import time
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
        del session_id  # Session identity is checked by TeleopController; epoch gates cross-session output.
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


class HardwareTransportOwner:
    """The only active SDK caller, with P0/P1/P2 request priority.

    A vendor call already in progress cannot be interrupted safely from Python,
    but queued P0 work always wins at the next SDK-call boundary.  P2 calls are
    intentionally dispatched only through this owner and are therefore never
    concurrent with feedback or CPV writes.
    """

    _PRIORITY = {"p0": 0, "p1": 1, "p2": 2}

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
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="nero-hardware-transport", daemon=True)
        self._thread.start()

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
            # Publish the barrier before waiting for an in-flight SDK call.
            # The next queued non-P0 command will see it and be revoked rather
            # than slipping through between that call and this promotion.
            self._pending_epoch = target
        with self._dispatch_lock:
            with self._epoch_lock:
                self._epoch = target
                self._exclusive_category = exclusive_category
                if self._pending_epoch == target:
                    self._pending_epoch = None
                return self._epoch

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
        self._queue.put((99, sequence, None))
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while True:
            _, _, envelope = self._queue.get()
            if envelope is None:
                return
            try:
                dispatch_started_ns = time.monotonic_ns()
                result = self._execute(
                    envelope["method"], envelope["args"], envelope["kwargs"],
                    envelope["priority"], envelope["command_epoch"],
                    envelope["category"], envelope.get("execute_guard"),
                )
                dispatch_finished_ns = time.monotonic_ns()
                if isinstance(result, dict):
                    result = dict(result)
                    queued_ns = int(envelope.get("queued_monotonic_ns") or dispatch_started_ns)
                    result.setdefault("transport_queued_monotonic_ns", queued_ns)
                    result.setdefault("transport_dispatch_started_monotonic_ns", dispatch_started_ns)
                    result.setdefault("transport_dispatch_finished_monotonic_ns", dispatch_finished_ns)
                    result.setdefault("queue_delay_ms", max(0.0, (dispatch_started_ns - queued_ns) / 1e6))
                    result.setdefault("transport_duration_ms", max(0.0, (dispatch_finished_ns - dispatch_started_ns) / 1e6))
                envelope["result"] = result
            except Exception as exc:  # deliver the original SDK error to caller
                envelope["error"] = exc
            finally:
                envelope["done"].set()

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
            return getattr(self.backend, method)(*args, **kwargs)


class TransportRobotProxy:
    """Compatibility proxy: default Broker operations are P2 transactions."""

    def __init__(self, owner: HardwareTransportOwner, backend: Any) -> None:
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
