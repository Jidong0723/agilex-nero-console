from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable


NERO_FEEDBACK_IDS = frozenset(
    [*range(0x251, 0x258), *range(0x261, 0x268), *range(0x2A1, 0x2AA)]
)
CAN_PUSH_CONTROL_ID = 0x151


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _listen(
    bus: Any,
    duration_s: float,
    monotonic: Callable[[], float],
) -> tuple[int, Counter[int]]:
    deadline = monotonic() + max(0.0, float(duration_s))
    total = 0
    known: Counter[int] = Counter()
    while monotonic() < deadline:
        remaining = max(0.0, deadline - monotonic())
        message = bus.recv(min(0.1, remaining))
        if message is None:
            continue
        total += 1
        arbitration_id = int(message.arbitration_id)
        if arbitration_id in NERO_FEEDBACK_IDS:
            known[arbitration_id] += 1
    return total, known


def _result(
    status: str,
    *,
    enabled: bool,
    frame_sent: bool = False,
    probe_frames: int = 0,
    verification_frames: int = 0,
    known: Counter[int] | None = None,
    reason: str,
) -> dict[str, Any]:
    known = known or Counter()
    return {
        "status": status,
        "enabled": enabled,
        "frame_sent": frame_sent,
        "probe_frame_count": probe_frames,
        "verification_frame_count": verification_frames,
        "known_feedback_ids": [f"0x{value:X}" for value in sorted(known)],
        "timestamp": _timestamp(),
        "reason": reason,
    }


def probe_and_recover_can_feedback(
    *,
    interface: str,
    channel: str,
    bitrate: int,
    speed_percent: int,
    enabled: bool = True,
    silence_timeout_s: float = 3.0,
    verification_timeout_s: float = 3.0,
    max_attempts: int = 1,
    bus_factory: Callable[..., Any] | None = None,
    message_factory: Callable[..., Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Probe NERO feedback and send one target-free CAN-push recovery frame."""
    if not enabled:
        return _result("disabled", enabled=False, reason="automatic CAN feedback recovery is disabled")
    if interface != "agx_cando":
        return _result(
            "disabled",
            enabled=False,
            reason=f"automatic recovery is only supported for agx_cando, not {interface}",
        )
    if int(max_attempts) < 1:
        return _result("failed", enabled=True, reason="maximum recovery attempts is zero")

    if bus_factory is None or message_factory is None:
        import can

        bus_factory = bus_factory or can.Bus
        message_factory = message_factory or can.Message

    bus = None
    try:
        bus = bus_factory(
            interface=interface,
            channel=channel,
            bitrate=int(bitrate),
            local_loopback=False,
            receive_own_messages=False,
        )
        probe_frames, probe_known = _listen(bus, silence_timeout_s, monotonic)
        if probe_known:
            return _result(
                "not_needed",
                enabled=True,
                probe_frames=probe_frames,
                known=probe_known,
                reason="fresh NERO CAN feedback was already present",
            )
        if probe_frames:
            return _result(
                "failed",
                enabled=True,
                probe_frames=probe_frames,
                reason="CAN traffic was present but no known NERO feedback IDs were observed",
            )

        speed = min(100, max(0, int(speed_percent)))
        payload = bytes([0x01, 0xFF, speed, 0x00, 0x00, 0x00, 0x01, 0x00])
        message = message_factory(
            arbitration_id=CAN_PUSH_CONTROL_ID,
            is_extended_id=False,
            data=payload,
        )
        bus.send(message)
        verification_frames, verification_known = _listen(
            bus, verification_timeout_s, monotonic
        )
        if verification_known:
            return _result(
                "recovered",
                enabled=True,
                frame_sent=True,
                probe_frames=probe_frames,
                verification_frames=verification_frames,
                known=verification_known,
                reason="NERO CAN feedback resumed after one CAN-push recovery frame",
            )
        return _result(
            "failed",
            enabled=True,
            frame_sent=True,
            probe_frames=probe_frames,
            verification_frames=verification_frames,
            reason="recovery frame was sent but no known NERO feedback IDs were observed",
        )
    except Exception as exc:
        return _result(
            "failed",
            enabled=True,
            reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if bus is not None:
            try:
                bus.shutdown()
            except Exception:
                pass
