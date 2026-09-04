"""HTTP-process ownership for input adapters and camera resources.

The hardware backend exposes OSC only.  This module lives in the HTTP process
and is the sole owner of policy, camera, and headset adapter lifecycle.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from supervisor.camera_resource import SharedCameraResource
from supervisor.dataset_recorder import DatasetRecorder
from supervisor.logging import AsyncJsonlTraceLogger
from supervisor.pi05_adapter import Pi05InputAdapter
from supervisor.pico_adapter import PicoInputAdapter


class OscClientPort(Protocol):
    """The complete robot-facing surface available to input adapters."""

    def state(self) -> dict[str, Any]: ...
    def start_session(self, client_id: str, execution_mode: str) -> dict[str, Any]: ...
    def heartbeat(self, client_id: str, session_id: str) -> dict[str, Any]: ...
    def track_tcp(self, session_id: str, client_id: str, sequence: int, target_pose: dict[str, Any]) -> dict[str, Any]: ...
    def hold(self, session_id: str, client_id: str, sequence: int, reason: str) -> dict[str, Any]: ...
    def osc_input_hold(self, reason: str) -> dict[str, Any]: ...
    def gripper(self, session_id: str, client_id: str, sequence: int, payload: dict[str, Any]) -> dict[str, Any]: ...


class OscClient:
    """Narrow adapter client backed by the OSC process proxy."""

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    def state(self) -> dict[str, Any]:
        return self._broker.osc_state()

    def start_session(self, client_id: str, execution_mode: str) -> dict[str, Any]:
        return self._broker.osc_start(client_id, execution_mode)

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

    def osc_input_hold(self, reason: str) -> dict[str, Any]:
        return self._broker.osc_input_hold(reason)

    def gripper(self, session_id: str, client_id: str, sequence: int, payload: dict[str, Any]) -> dict[str, Any]:
        return self._command(session_id, client_id, sequence, "gripper", payload)


class AdapterRuntime:
    """Own input adapters in the HTTP process after OSC is available."""

    def __init__(self, broker: Any, project_root: Path, runtime_config: dict[str, Any]) -> None:
        self._lock = threading.RLock()
        self._runtime_config_path = project_root / "config" / "runtime.json"
        self._runtime_config = runtime_config
        self.osc: OscClientPort = OscClient(broker)
        pi05_path = project_root / "config" / "pi05.json"
        pi05_config = json.loads(pi05_path.read_text(encoding="utf-8"))
        self.cameras = SharedCameraResource(pi05_config["cameras"])
        self.pi05 = Pi05InputAdapter(self.osc, pi05_path, self.cameras)
        trace_dir = project_root / "runtime" / "logs" / "pico"
        self.pico_trace_logger = AsyncJsonlTraceLogger(
            trace_dir / f"pico-adapter-{time.strftime('%Y%m%dT%H%M%S')}.jsonl",
            {"component": "pico_adapter"},
        )
        self.pico = PicoInputAdapter(self.osc, dict(runtime_config.get("pico_adapter") or {}), self.pico_trace_logger)
        self.dataset = DatasetRecorder(self.osc, self.cameras, self.pico, project_root.parent / "dataset")

    def close(self) -> None:
        with self._lock:
            self.pi05.close()
            self.dataset.close()
            self.pico.disconnected("adapter runtime shutdown")
            self.pico_trace_logger.close()
            self.cameras.close()

    def health(self) -> dict[str, Any]:
        try:
            pi05 = self.pi05.snapshot()
        except Exception as exc:
            pi05 = {"state": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}
        return {
            "adapters": {"pi05": pi05, "pico": self.pico.snapshot(), "cameras": self.cameras.snapshot()},
            "pico_trace_logging": {"enabled": True, "path": str(self.pico_trace_logger.path.resolve()), "schema": "pico-trace.v1"},
        }

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
    def dataset_state(self) -> dict[str, Any]: return self.dataset.state()
    def dataset_episodes(self) -> dict[str, Any]: return self.dataset.episodes()
    def dataset_start(self, body: dict[str, Any]) -> dict[str, Any]: return self.dataset.start(body)
    def dataset_stop(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.dataset.stop(str(body.get("status", "completed")), str(body.get("reason", "")))
    def pico_state(self) -> dict[str, Any]: return self.pico.snapshot()
    def pico_reset_anchor(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.pico.reset_anchor(str(body.get("session_id", "")), str(body.get("client_id", "")))
    def pico_begin_pairing(self, session_id: str, client_id: str) -> None: self.pico.begin_pairing(session_id, client_id)
    def pico_paired(self) -> None: self.pico.paired()
    def pico_connection_lost(self, reason: str) -> None: self.pico.connection_lost(reason)
    def pico_disconnected(self, reason: str) -> None: self.pico.disconnected(reason)
    def _persist_pico_config(self, updates: dict[str, Any]) -> None:
        with self._lock:
            self._runtime_config.setdefault("pico_adapter", {}).update(updates)
            temporary = self._runtime_config_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self._runtime_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")
            temporary.replace(self._runtime_config_path)

    def pico_update_sensitivity(self, body: dict[str, Any]) -> dict[str, Any]:
        result = self.pico.update_sensitivity(str(body.get("session_id", "")), str(body.get("client_id", "")),
                                              body.get("translation_gain", 1.0), body.get("rotation_gain", 1.0),
                                              bool(body.get("hardware_high_gain_confirmed", False)))
        if result.get("accepted"):
            self._persist_pico_config({"translation_gain": result["translation_gain"], "rotation_gain": result["rotation_gain"]})
        return result
    def pico_update_mapping(self, body: dict[str, Any]) -> dict[str, Any]:
        result = self.pico.update_mapping(str(body.get("session_id", "")), str(body.get("client_id", "")),
                                          body.get("position_axis_map"), body.get("orientation_axis_map"),
                                          bool(body.get("mapping_verified", False)))
        if result.get("accepted"):
            self._persist_pico_config({"position_axis_map": result["position_axis_map"],
                                       "orientation_axis_map": result["orientation_axis_map"],
                                       "mapping_verified": bool(result.get("mapping_verified", False))})
            result.update(self.pico.mapping_persisted())
        return result

    def pico_message(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if kind == "input_frame":
            return self.pico.input_frame(payload)
        raise ValueError("unsupported PICO message; input_frame is required")
