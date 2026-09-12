from __future__ import annotations

import math
import json
import tempfile
import threading
from pathlib import Path

from nero_console.application.adapter_runtime import AdapterRuntime
from supervisor.pico_adapter import (
    DEFAULT_FRAME_MAP,
    RECOMMENDED_ORIENTATION_FRAME_MAP,
    RECOMMENDED_POSITION_FRAME_MAP,
    PicoInputAdapter,
)


class Broker:
    def __init__(self) -> None:
        self.commands = []
        self.heartbeats = []
        self._state = {"session": {"state": "ACTIVE", "id": "osc-1", "client_id": "browser", "execution_mode": "shadow"},
                      "command": {"sequence": 10, "target_tcp": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}},
                      "execution": {"measured_tcp_pose": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]}}}
    def state(self): return self._state
    def track_tcp(self, session_id, client_id, sequence, target_pose):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "track_tcp", "payload": {"target_pose": target_pose}}); return {"ok": True, "result": {"accepted": True}}
    def hold(self, session_id, client_id, sequence, reason):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "hold", "payload": {"reason": reason}}); return {"ok": True}
    def gripper(self, session_id, client_id, sequence, payload):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "gripper", "payload": payload}); return {"ok": True}
    def heartbeat(self, client_id, session_id): self.heartbeats.append((client_id, session_id)); return {"ok": True}


def adapter() -> tuple[PicoInputAdapter, Broker]:
    broker = Broker(); identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    value = PicoInputAdapter(broker, {"position_axis_map": identity, "orientation_axis_map": identity})
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


def test_pico_socket_loss_holds_but_preserves_pairing_for_safe_reconnect():
    value, broker = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.connection_lost("gateway input timeout")
    snapshot = value.snapshot()
    assert broker.commands[-1]["type"] == "hold"
    assert snapshot["state"] == "READY"
    assert snapshot["connected"] is False
    assert snapshot["paired"] is True
    assert snapshot["anchor_active"] is False


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
    broker._state["session"]["state"] = "IDLE"
    value.disconnected("socket already closed")
    value.disconnected("operator disconnected again")
    snapshot = value.snapshot()
    assert snapshot["state"] == "IDLE"
    assert snapshot["session_id"] is None
    assert snapshot["last_error"] is None
    assert broker.commands == []


def test_pico_translation_gain_is_linear_and_runtime_configured():
    value, broker = adapter()
    value.update_sensitivity("osc-1", "browser", 0.5, 1.0)
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.pose({"position_m": [1.1, 2, 3], "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True})
    assert math.isclose(broker.commands[-1]["payload"]["target_pose"]["position_m"][0], 0.15)


def test_pico_recommended_frame_map_keeps_simulated_axes_consistent():
    value, broker = adapter()
    value.update_mapping("osc-1", "browser", RECOMMENDED_POSITION_FRAME_MAP, RECOMMENDED_ORIENTATION_FRAME_MAP)
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    for raw, expected_delta in [([1.1, 2.0, 3.0], [-0.1, 0.0, 0.0]),
                                ([1.0, 2.1, 3.0], [0.0, 0.0, 0.1]),
                                ([1.0, 2.0, 3.1], [0.0, -0.1, 0.0])]:
        value.pose({"position_m": raw, "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True})
        actual = broker.commands[-1]["payload"]["target_pose"]["position_m"]
        expected = [0.1 + expected_delta[0], 0.2 + expected_delta[1], 0.3 + expected_delta[2]]
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual, expected))


def test_pico_snapshot_exposes_transmitted_base_frame_target_pose():
    value, broker = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.pose({"position_m": [1.02, 2, 3], "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True})
    target = broker.commands[-1]["payload"]["target_pose"]
    snapshot = value.snapshot()
    assert snapshot["last_target_status"] == "ACCEPTED"
    assert snapshot["last_target_pose"] == target
    assert snapshot["last_target_age_ms"] is not None
    assert snapshot["target_pose_mode"] == "ABSOLUTE_DIRECT"


def test_pico_frame_mapping_can_be_changed_only_when_not_anchored():
    value, _ = adapter()
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    changed = value.update_mapping("osc-1", "browser", identity, identity, True)
    assert changed["accepted"] is True
    assert changed["mapping_verified"] is False
    recommended = value.update_mapping("osc-1", "browser", RECOMMENDED_POSITION_FRAME_MAP, RECOMMENDED_ORIENTATION_FRAME_MAP)
    assert recommended["mapping_verified"] is True
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    changed_while_tracking = value.update_mapping("osc-1", "browser", RECOMMENDED_POSITION_FRAME_MAP, RECOMMENDED_ORIENTATION_FRAME_MAP)
    assert changed_while_tracking["accepted"] is True
    assert changed_while_tracking["applied_while_tracking"] is True


def test_pico_frame_mapping_change_is_used_by_absolute_orientation_output():
    value, broker = adapter()
    rotate_z_90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = value.update_mapping("osc-1", "browser", rotate_z_90, rotate_z_90, False)
    assert result["accepted"] is True
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    angle = math.radians(20)
    hand_q = [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]
    value.pose({"position_m": [1, 2, 3], "orientation_xyzw": hand_q, "tracking_valid": True})
    output = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    expected = [0, math.sin(angle / 2), 0, math.cos(angle / 2)]
    assert abs(sum(a * b for a, b in zip(output, expected))) > 1.0 - 1e-6


def test_pico_orientation_anchor_preserves_current_tcp_without_jump():
    value, broker = adapter()
    tcp_angle = math.radians(20)
    hand_angle = math.radians(-35)
    tcp_q = [0, math.sin(tcp_angle / 2), 0, math.cos(tcp_angle / 2)]
    hand_q = [math.sin(hand_angle / 2), 0, 0, math.cos(hand_angle / 2)]
    broker._state["execution"]["measured_tcp_pose"]["orientation_xyzw"] = tcp_q
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": hand_q})
    value.pose({"position_m": [1, 2, 3], "orientation_xyzw": hand_q, "tracking_valid": True})
    orientation = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    assert abs(sum(a * b for a, b in zip(orientation, tcp_q))) > 1.0 - 1e-6
    assert math.isclose(math.sqrt(sum(item * item for item in orientation)), 1.0, abs_tol=1e-6)
    snapshot = value.snapshot()
    assert snapshot["orientation_tracking_mode"] == "RELATIVE_ANCHORED"
    assert snapshot["orientation_calibration_status"] == "RELATIVE_ANCHORED"


def test_pico_regrip_reanchors_at_the_current_tcp():
    value, broker = adapter()
    hand_q = [0, 0, math.sin(math.radians(25) / 2), math.cos(math.radians(25) / 2)]
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    value.pose({"position_m": [1, 2, 3], "orientation_xyzw": hand_q, "tracking_valid": True})
    first = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    value.stop("Grip released")
    broker._state["execution"]["measured_tcp_pose"]["orientation_xyzw"] = [0, math.sin(math.radians(70) / 2), 0, math.cos(math.radians(70) / 2)]
    value.anchor_begin({"position_m": [4, 5, 6], "orientation_xyzw": hand_q})
    value.pose({"position_m": [4, 5, 6], "orientation_xyzw": hand_q, "tracking_valid": True})
    second = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    expected_second = broker._state["execution"]["measured_tcp_pose"]["orientation_xyzw"]
    assert abs(sum(a * b for a, b in zip(first, hand_q))) > 1.0 - 1e-6
    assert abs(sum(a * b for a, b in zip(second, expected_second))) > 1.0 - 1e-6


def test_pico_absolute_orientation_tracks_one_to_one_after_calibration():
    value, broker = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    angle = math.radians(30)
    value.pose({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, math.sin(angle / 2), math.cos(angle / 2)], "tracking_valid": True})
    orientation = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    assert math.isclose(abs(orientation[2]), math.sin(angle / 2), abs_tol=1e-6)
    assert math.isclose(abs(orientation[3]), math.cos(angle / 2), abs_tol=1e-6)


def test_pico_rotation_gain_attenuates_relative_rotation_from_anchor():
    value, broker = adapter()
    assert value.update_sensitivity("osc-1", "browser", 1.0, 0.5)["accepted"] is True
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    angle = math.radians(40)
    value.pose({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, math.sin(angle / 2), math.cos(angle / 2)], "tracking_valid": True})
    target = broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"]
    assert math.isclose(abs(target[2]), math.sin(math.radians(20) / 2), abs_tol=1e-6)
    assert math.isclose(abs(target[3]), math.cos(math.radians(20) / 2), abs_tol=1e-6)


def test_pico_sensitivity_rejects_values_outside_attenuation_range():
    value, _ = adapter()
    for translation, rotation in [(0.2, 1.0), (1.0, 0.2), (1.01, 1.0), (1.0, 1.01)]:
        try:
            value.update_sensitivity("osc-1", "browser", translation, rotation)
        except ValueError as exc:
            assert "25% 至 100%" in str(exc)
        else:
            raise AssertionError("out-of-range PICO gain was accepted")


def test_pico_anchor_requires_measured_tcp_feedback():
    value, broker = adapter()
    broker._state["session"]["execution_mode"] = "hardware"
    broker._state["execution"] = {}
    result = value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    assert result["accepted"] is False
    assert result["reason"] == "measured_tcp_pose_unavailable"
    assert value.snapshot()["orientation_calibration_status"] == "WAITING_FEEDBACK"


def test_pico_shadow_anchor_uses_current_command_tcp_when_measurement_is_unavailable():
    value, broker = adapter()
    broker._state["execution"] = {}
    result = value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    assert result["anchor_active"] is True
    assert result["anchor_tcp_source"] == "shadow_command_target"
    value.pose({"position_m": [1.01, 2, 3], "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True})
    assert broker.commands[-1]["type"] == "track_tcp"


def test_pending_first_grip_sends_target_on_first_frame_after_hold_ready():
    value, broker = adapter()
    broker._state["diagnostics"] = {"trajectory_state": "BRAKING"}
    pending = value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1],
                                  "_pico_sequence": 7})
    assert pending["pending"] is True
    broker._state["diagnostics"] = {"trajectory_state": "HOLD_READY"}
    result = value.tracking({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1],
                             "tracking_valid": True, "clutch": True, "_pico_sequence": 7})
    assert result["anchor_activated"] is True
    assert result["target_result"]["accepted"] is True
    assert broker.commands[-1]["type"] == "track_tcp"


def test_first_grip_and_following_movement_send_immediately():
    value, broker = adapter()
    initial = {
        "position_m": [1.0, 2.0, 3.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grip": True,
        "trigger_value": 0.0,
        "tracking_valid": True,
        "_pico_sequence": 1,
    }

    pressed = value.input_frame(initial)

    assert pressed["event"] == "grip_press"
    assert broker.commands[-1]["type"] == "track_tcp"

    value.input_frame({
        **initial,
        "position_m": [1.15, 2.0, 3.0],
        "_pico_sequence": 2,
    })

    assert broker.commands[-1]["type"] == "track_tcp"
    target = broker.commands[-1]["payload"]["target_pose"]
    assert math.isclose(target["position_m"][0], 0.25, abs_tol=1e-9)


def test_pico_sensitivity_rejects_active_anchor_without_changing_values():
    value, _ = adapter()
    value.anchor_begin({"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]})
    result = value.update_sensitivity("osc-1", "browser", 0.5, 0.5)
    assert result["reason"] == "release_grip_first"
    assert value.snapshot()["mapping"]["translation_gain"] == 1.0


def test_pico_sensitivity_persists_both_default_gains():
    class PicoStub:
        def update_sensitivity(self, *args):
            return {"ok": True, "accepted": True, "translation_gain": 0.5, "rotation_gain": 0.75}

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        runtime = object.__new__(AdapterRuntime)
        runtime._lock = threading.RLock()
        runtime._runtime_config_path = root / "runtime.json"
        runtime._runtime_config = {"pico_adapter": {"translation_gain": 1.0, "rotation_gain": 1.0}}
        runtime.pico = PicoStub()
        result = runtime.pico_update_sensitivity({"session_id": "osc-1", "client_id": "browser", "translation_gain": 0.5, "rotation_gain": 0.75})
        persisted = json.loads(runtime._runtime_config_path.read_text(encoding="utf-8-sig"))
    assert result["accepted"] is True
    assert persisted["pico_adapter"]["translation_gain"] == 0.5
    assert persisted["pico_adapter"]["rotation_gain"] == 0.75


def test_recommended_mapping_is_separate_from_the_runtime_mapping_and_is_exposed():
    value = PicoInputAdapter(Broker(), {})
    mapping = value.snapshot()["mapping"]
    assert mapping["position_axis_map"] == RECOMMENDED_POSITION_FRAME_MAP
    assert mapping["orientation_axis_map"] == RECOMMENDED_ORIENTATION_FRAME_MAP
    assert mapping["recommended_position_axis_map"] == RECOMMENDED_POSITION_FRAME_MAP
    assert mapping["recommended_orientation_axis_map"] == RECOMMENDED_ORIENTATION_FRAME_MAP
    assert mapping["verified"] is True
    assert mapping["persisted"] is False
    assert mapping["source"] == "built_in_recommended"


def test_mapping_persistence_marks_saved_custom_mapping_and_keeps_recommendation_fixed():
    class PicoStub:
        def __init__(self): self.persisted = False
        def update_mapping(self, *_args):
            return {"ok": True, "accepted": True, "mapping_verified": False,
                    "position_axis_map": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "orientation_axis_map": [[0, 1, 0], [1, 0, 0], [0, 0, -1]]}
        def mapping_persisted(self):
            self.persisted = True
            return {"persisted": True}

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        runtime = object.__new__(AdapterRuntime)
        runtime._lock = threading.RLock()
        runtime._runtime_config_path = root / "runtime.json"
        runtime._runtime_config = {"pico_adapter": {"position_axis_map": RECOMMENDED_POSITION_FRAME_MAP,
                                                      "orientation_axis_map": RECOMMENDED_ORIENTATION_FRAME_MAP}}
        runtime.pico = PicoStub()
        result = runtime.pico_update_mapping({"session_id": "osc-1", "client_id": "browser",
                                              "position_axis_map": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                              "orientation_axis_map": [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                                              "mapping_verified": True})
        persisted = json.loads(runtime._runtime_config_path.read_text(encoding="utf-8-sig"))
    assert result["persisted"] is True
    assert persisted["pico_adapter"]["mapping_verified"] is False
    assert persisted["pico_adapter"]["position_axis_map"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert RECOMMENDED_POSITION_FRAME_MAP == [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]


def test_mapping_lifecycle_is_logged_without_per_frame_diagnostics():
    class Trace:
        def __init__(self): self.events = []
        def append(self, event): self.events.append(event)

    trace = Trace()
    value = PicoInputAdapter(Broker(), {}, trace)
    value.begin_pairing("osc-1", "browser")
    value.update_mapping("osc-1", "browser", RECOMMENDED_POSITION_FRAME_MAP, RECOMMENDED_ORIENTATION_FRAME_MAP)
    value.mapping_persisted()
    assert [event["event"] for event in trace.events if event.get("record_type") == "event"] == [
        "adapter_mapping_loaded", "adapter_pairing_started", "adapter_mapping_updated", "adapter_mapping_persisted",
    ]


def test_input_frame_edges_anchor_release_and_trigger_are_server_side():
    value, broker = adapter()
    base = {"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True}
    first = value.input_frame({**base, "grip": False, "trigger_value": 0.0, "_pico_sequence": 1})
    assert first["anchor_active"] is False
    pressed = value.input_frame({**base, "grip": True, "trigger_value": 0.5, "_pico_sequence": 2})
    assert pressed["event"] == "grip_press"
    assert value.snapshot()["anchor_active"] is True
    assert any(item["type"] == "gripper" for item in broker.commands)
    released = value.input_frame({**base, "grip": False, "trigger_value": 0.5, "_pico_sequence": 3})
    assert released["event"] == "grip_release"
    assert value.snapshot()["clutch_signal"] is False
    assert value.snapshot()["anchor_active"] is False
    assert broker.commands[-1]["type"] == "hold"


def test_tracking_loss_requests_hold_once_until_valid_tracking_returns():
    value, broker = adapter()
    base = {"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1],
            "grip": True, "trigger_value": 0.0}
    value.input_frame({**base, "tracking_valid": True, "_pico_sequence": 1})
    assert value.snapshot()["anchor_active"] is True

    lost = value.input_frame({**base, "tracking_valid": False, "_pico_sequence": 2})
    assert lost["event"] == "tracking_lost"
    holds_after_loss = len([item for item in broker.commands if item["type"] == "hold"])
    assert holds_after_loss == 1

    repeated = value.input_frame({**base, "tracking_valid": False, "_pico_sequence": 3})
    assert repeated["event"] == "tracking_unavailable"
    assert len([item for item in broker.commands if item["type"] == "hold"]) == holds_after_loss


def test_input_frame_updates_absolute_orientation_preview_while_grip_is_released():
    value, broker = adapter()
    angle = math.radians(40)
    hand_q = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
    result = value.input_frame({"position_m": [1, 2, 3], "orientation_xyzw": hand_q,
                                "grip": False, "trigger_value": 0.0,
                                "tracking_valid": True, "_pico_sequence": 1})
    assert result["accepted"] is True
    assert broker.commands == []
    snapshot = value.snapshot()
    assert snapshot["orientation_command_enabled"] is False
    assert snapshot["absolute_orientation_preview_xyzw"] == hand_q
    assert snapshot["last_target_pose"] is None

    value.input_frame({"position_m": [1, 2, 3], "orientation_xyzw": hand_q,
                       "grip": True, "trigger_value": 0.0,
                       "tracking_valid": True, "_pico_sequence": 2})
    assert broker.commands[-1]["type"] == "track_tcp"
    assert broker.commands[-1]["payload"]["target_pose"]["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert value.snapshot()["orientation_command_enabled"] is True


def test_released_grip_keeps_orientation_preview_live_but_does_not_send_motion():
    value, broker = adapter()
    base = {"position_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1],
            "tracking_valid": True, "trigger_value": 0.0}
    value.input_frame({**base, "grip": True, "_pico_sequence": 1})
    sent_count = len([item for item in broker.commands if item["type"] == "track_tcp"])
    angle = math.radians(55)
    hand_q = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
    released = value.input_frame({**base, "orientation_xyzw": hand_q,
                                  "grip": False, "_pico_sequence": 2})
    assert released["event"] == "grip_release"
    assert len([item for item in broker.commands if item["type"] == "track_tcp"]) == sent_count
    assert broker.commands[-1]["type"] == "hold"
    snapshot = value.snapshot()
    assert snapshot["orientation_command_enabled"] is False
    assert snapshot["absolute_orientation_preview_xyzw"] == hand_q
