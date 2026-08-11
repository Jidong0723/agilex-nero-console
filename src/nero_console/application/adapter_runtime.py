"""HTTP-process ownership for input adapters and camera resources.

The hardware backend exposes OSC only.  This module lives in the HTTP process
and is the sole owner of policy, camera, and headset adapter lifecycle.
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Protocol

from supervisor.camera_resource import SharedCameraResource
from supervisor.pi05_adapter import Pi05InputAdapter
from supervisor.pico_adapter import PicoInputAdapter


class OscClientPort(Protocol):
    """The complete robot-facing surface available to input adapters."""

    def state(self) -> dict[str, Any]: ...
    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]: ...
    def track_tcp(self, session_id: str, client_id: str, sequence: int, target_pose: dict[str, Any]) -> dict[str, Any]: ...
    def hold(self, session_id: str, client_id: str, sequence: int, reason: str) -> dict[str, Any]: ...
    def gripper(self, session_id: str, client_id: str, sequence: int, payload: dict[str, Any]) -> dict[str, Any]: ...


class OscClient:
    """Narrow adapter client backed by the OSC process proxy."""

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def state(self) -> dict[str, Any]:
        return self._broker.osc_state()

    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]:
        return self._broker.osc_heartbeat(client_id, session_id)

    def _command(self, session_id: str, client_id: str, sequence: int, kind: str, payload: dict[str, Any], *, acknowledgement_only: bool = False) -> dict[str, Any]:
        return self._broker.osc_command({
            "session_id": session_id,
            "client_id": client_id,
            "sequence": sequence,
            "type": kind,
            "acknowledgement_only": acknowledgement_only,
            "payload": payload,
        })

    def track_tcp(self, session_id: str, client_id: str, sequence: int, target_pose: dict[str, Any]) -> dict[str, Any]:
        return self._command(session_id, client_id, sequence, "track_tcp", {"target_pose": target_pose}, acknowledgement_only=True)

    def hold(self, session_id: str, client_id: str, sequence: int, reason: str) -> dict[str, Any]:
        return self._command(session_id, client_id, sequence, "hold", {"reason": reason})

    def gripper(self, session_id: str, client_id: str, sequence: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._command(session_id, client_id, sequence, "gripper", payload)


class AdapterRuntime:
    """Own input adapters in the HTTP process after OSC is available."""

    def __init__(self, broker: Any, project_root: Path, runtime_config: dict[str, Any]) -> None:
        self._lock = threading.RLock()
        self.osc: OscClientPort = OscClient(broker)
        pi05_path = project_root / "config" / "pi05.json"
        pi05_config = json.loads(pi05_path.read_text(encoding="utf-8"))
        self.cameras = SharedCameraResource(pi05_config["cameras"])
        self.pi05 = Pi05InputAdapter(self.osc, pi05_path, self.cameras)
        self.pico = PicoInputAdapter(self.osc, dict(runtime_config.get("pico_adapter") or {}))

    def close(self) -> None:
        with self._lock:
            self.pi05.close()
            self.pico.disconnected("adapter runtime shutdown")
            self.cameras.close()

    def health(self) -> dict[str, Any]:
        try:
            pi05 = self.pi05.snapshot()
        except Exception as exc:
            pi05 = {"state": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}
        return {"adapters": {"pi05": pi05, "pico": self.pico.snapshot(), "cameras": self.cameras.snapshot()}}

    def pi05_state(self) -> dict[str, Any]: return self.pi05.snapshot()
    def pi05_update_config(self, body: dict[str, Any]) -> dict[str, Any]: return self.pi05.update_config(body)
    def pi05_start(self, session_id: str, client_id: str) -> dict[str, Any]: return self.pi05.start(session_id, client_id)
    def pi05_stop(self, reason: str) -> dict[str, Any]: return self.pi05.stop(reason)
    def camera_state(self) -> dict[str, Any]: return self.cameras.snapshot()
    def camera_update_config(self, body: dict[str, Any]) -> dict[str, Any]: return self.cameras.update_config(body.get("cameras", body))
    def camera_activate(self) -> dict[str, Any]:
        result = self.cameras.activate()
        self.pi05.cameras_ready()
        return result
    def camera_deactivate(self) -> dict[str, Any]:
        if self.pi05.snapshot().get("state") == "RUNNING":
            self.pi05.stop("cameras closed by operator")
        return self.cameras.deactivate()
    def camera_devices(self) -> list[dict[str, Any]]: return self.cameras.devices()
    def camera_frame_jpeg(self, source: str) -> bytes | None: return self.cameras.frame_jpeg(source)
    def pico_state(self) -> dict[str, Any]: return self.pico.snapshot()
    def pico_begin_pairing(self, session_id: str, client_id: str) -> None: self.pico.begin_pairing(session_id, client_id)
    def pico_paired(self) -> None: self.pico.paired()
    def pico_disconnected(self, reason: str) -> None: self.pico.disconnected(reason)

    def pico_message(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if kind == "heartbeat":
            self.pico.heartbeat(); return None
        if kind == "anchor_begin": return self.pico.anchor_begin(payload)
        if kind == "pose": return self.pico.pose(payload)
        if kind == "gripper": return self.pico.gripper(payload.get("value", 0.0))
        if kind in {"anchor_release", "hold"}:
            return self.pico.stop("PICO operator HOLD" if kind == "hold" else "PICO right Grip released")
        raise ValueError("unsupported PICO adapter message")
