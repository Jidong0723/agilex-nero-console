from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from supervisor.pi05_adapter import Pi05InputAdapter


class _Camera:
    def read(self):
        return np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8)

    def close(self):
        pass


class _Policy:
    def __init__(self, *_args):
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        # Ten-step LIBERO chunk. Every step is converted against one shared
        # pre-inference feedback pose; only the first five are executed.
        return {"actions": np.asarray([[1., 0., 0., 0., 0., 0., 1.]] * 10, dtype=np.float32)}

    def close(self):
        pass

    def is_alive(self):
        return True


class _Broker:
    def __init__(self):
        self.commands = []
        self.fail_control = False
        self.session = {"state": "ACTIVE", "id": "session-1", "client_id": "client-1", "execution_mode": "shadow"}

    def state(self):
        if self.fail_control: raise RuntimeError("simulated OSC outage")
        return {"session": dict(self.session), "command": {"target_tcp": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}},
                "execution": {"measured_tcp_pose": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}}, "gripper": {"width_m": .02}}

    def track_tcp(self, session_id, client_id, sequence, target_pose):
        if self.fail_control: raise RuntimeError("simulated OSC outage")
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "track_tcp", "payload": {"target_pose": target_pose}})
        return {"ok": True, "result": {"accepted": True}}
    def gripper(self, session_id, client_id, sequence, payload):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "gripper", "payload": payload})
        return {"ok": True}
    def hold(self, session_id, client_id, sequence, reason):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "hold", "payload": {"reason": reason}})
        return {"ok": True}
    def heartbeat(self, client_id, session_id):
        if self.fail_control: raise RuntimeError("simulated OSC outage")
        return {"ok": True}


class Pi05AdapterTests(unittest.TestCase):
    def test_websocket_state_is_disconnected_when_handshake_fails(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        class BrokenPolicy:
            def __init__(self, *_args):
                raise RuntimeError("handshake rejected")
        try:
            with patch("supervisor.pi05_adapter.OpenPIClient", BrokenPolicy):
                adapter._connection_stop.set()
                adapter._connection_thread.join(timeout=1.0)
                adapter._connection_stop.clear()
                adapter._connection_thread = __import__("threading").Thread(target=adapter._connection_loop, daemon=True)
                adapter._connection_thread.start()
                deadline = time.monotonic() + 1.0
                while adapter.snapshot()["model_state"] != "DISCONNECTED" and time.monotonic() < deadline:
                    time.sleep(.02)
                snapshot = adapter.snapshot()
        finally:
            adapter.close()
        self.assertEqual(snapshot["model_state"], "DISCONNECTED")
        self.assertTrue(snapshot["websocket_error"])
        self.assertNotEqual(snapshot["connections"]["policy"]["state"], "ok")

    def test_websocket_state_is_invalidated_after_inference_disconnect(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        class DisconnectingPolicy(_Policy):
            def infer(self, observation):
                raise RuntimeError("socket closed")
        with patch("supervisor.pi05_adapter.OpenPIClient", DisconnectingPolicy):
            adapter.start("session-1", "client-1")
            deadline = time.monotonic() + 1.0
            while adapter.snapshot()["model_state"] != "DISCONNECTED" and time.monotonic() < deadline:
                time.sleep(.02)
            snapshot = adapter.snapshot()
            adapter.close()
        self.assertEqual(snapshot["model_state"], "DISCONNECTED")
        self.assertIn("socket closed", snapshot["websocket_error"])
    def test_action_chunk_flows_to_absolute_osc_target(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        with patch("supervisor.pi05_adapter.OpenPIClient", _Policy):
            adapter.start("session-1", "client-1")
            deadline = time.monotonic() + 1.0
            while len([item for item in broker.commands if item["type"] == "track_tcp"]) < 5 and time.monotonic() < deadline:
                time.sleep(.01)
            snapshot = adapter.snapshot()
            adapter.close()
        motions = [item for item in broker.commands if item["type"] == "track_tcp"][:5]
        self.assertEqual(len(motions), 5)
        motion = motions[0]
        gripper = next(item for item in broker.commands if item["type"] == "gripper")
        self.assertEqual(motion["type"], "track_tcp")
        # π0.5 +X is configured as NERO -X; 1 * .05 * .2 = .01 m.
        self.assertAlmostEqual(motion["payload"]["target_pose"]["position_m"][0], -.01, places=6)
        self.assertEqual(gripper["type"], "gripper")
        self.assertAlmostEqual(gripper["payload"]["width_m"], .01)
        # Targets are not cumulative inside the chunk: all five use the same
        # feedback pose captured for the inference observation.
        for item in motions:
            self.assertAlmostEqual(item["payload"]["target_pose"]["position_m"][0], -.01, places=6)
        self.assertEqual(snapshot["action_chunk_length"], 10)
        self.assertEqual(len(snapshot["absolute_tcp_chunk"]), 10)
        self.assertEqual(snapshot["inference_base_tcp"]["position_m"], [0., 0., .3])

    def test_osc_outage_does_not_stop_inference_or_action_chunk_publication(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        with patch("supervisor.pi05_adapter.OpenPIClient", _Policy):
            adapter.start("session-1", "client-1")
            broker.fail_control = True
            deadline = time.monotonic() + 1.0
            while adapter.snapshot()["chunk_sequence"] < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            snapshot = adapter.snapshot()
            adapter.close()
        self.assertGreaterEqual(snapshot["chunk_sequence"], 1)
        self.assertIsNotNone(snapshot["action_chunk"])
        self.assertEqual(snapshot["last_error"], None)
        self.assertIn("simulated OSC outage", snapshot["osc_error"])
        self.assertNotEqual(snapshot["state"], "ERROR")

    def test_pi05_snapshot_does_not_query_osc_during_control_outage(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            broker.fail_control = True
            started = time.monotonic()
            snapshot = adapter.snapshot()
            elapsed = time.monotonic() - started
        finally:
            adapter.close()
        self.assertLess(elapsed, 0.5)
        self.assertEqual(snapshot["action_chunk"], None)
        self.assertNotEqual(snapshot["state"], "ERROR")

    def test_rejects_duplicate_camera_indices(self):
        adapter = Pi05InputAdapter(_Broker(), Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        with self.assertRaises(ValueError):
            adapter.update_config({"cameras": {"external": {"index": 2}, "wrist": {"index": 2}}})
