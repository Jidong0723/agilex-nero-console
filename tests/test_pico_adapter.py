from __future__ import annotations

import math

from supervisor.pico_adapter import PicoInputAdapter


class Broker:
    def __init__(self) -> None:
        self.commands = []
        self.heartbeats = []
        self.state = {"session": {"state": "ACTIVE", "id": "osc-1", "client_id": "browser", "execution_mode": "shadow"},
                      "command": {"sequence": 10, "target_tcp": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}},
                      "execution": {"current_tcp_pose": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}}}
    def osc_state(self): return self.state
    def osc_command(self, command): self.commands.append(command); return {"ok": True, "result": {"accepted": True}}
    def osc_heartbeat(self, client_id, session_id): self.heartbeats.append((client_id, session_id)); return {"ok": True}


def adapter() -> tuple[PicoInputAdapter, Broker]:
    broker = Broker(); value = PicoInputAdapter(broker, {})
    value.begin_pairing("osc-1", "browser"); value.paired()
    return value, broker


def test_pico_anchor_converts_raw_pose_to_absolute_osc_target():
    value, broker = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.pose({"position_m": [1.02, 2, 3], "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True})
    command = broker.commands[-1]
    assert command["type"] == "track_tcp"
    assert all(math.isclose(actual, expected) for actual, expected in zip(command["payload"]["target_pose"]["position_m"], [0.12, 0.2, 0.3]))
    assert "anchor" not in command["payload"]


def test_pico_release_loss_and_disconnect_hold():
    value, broker = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.stop("Grip released")
    assert broker.commands[-1]["type"] == "hold"
    value.disconnected("network dropped")
    assert broker.commands[-1]["type"] == "hold"
    assert value.snapshot()["connected"] is False


def test_pico_gripper_and_heartbeat_use_standard_osc_interface():
    value, broker = adapter()
    value.gripper(.25); value.heartbeat()
    assert broker.commands[-1]["type"] == "gripper"
    assert math.isclose(broker.commands[-1]["payload"]["width_m"], 0.07125)
    assert broker.heartbeats == [("browser", "osc-1")]


def test_repeated_disconnect_after_osc_session_end_is_idempotent():
    broker = Broker()
    value = PicoInputAdapter(broker, {})
    value.begin_pairing("osc-1", "browser"); value.paired()
    broker.state["session"]["state"] = "IDLE"
    value.disconnected("socket already closed")
    value.disconnected("operator disconnected again")
    snapshot = value.snapshot()
    assert snapshot["state"] == "IDLE"
    assert snapshot["session_id"] is None
    assert snapshot["last_error"] is None
    assert broker.commands == []
