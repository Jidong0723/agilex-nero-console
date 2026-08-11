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
        # Translation, axis-angle, gripper: a valid π0.5 Action Chunk.
        return {"actions": np.asarray([[1., 0., 0., 0., 0., 0., 1.]], dtype=np.float32)}

    def close(self):
        pass


class _Broker:
    def __init__(self):
        self.commands = []
        self.session = {"state": "ACTIVE", "id": "session-1", "client_id": "client-1", "execution_mode": "shadow"}

    def state(self):
        return {"session": dict(self.session), "command": {"target_tcp": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}},
                "execution": {"measured_tcp_pose": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}}, "gripper": {"width_m": .02}}

    def track_tcp(self, session_id, client_id, sequence, target_pose):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "track_tcp", "payload": {"target_pose": target_pose}})
        return {"ok": True, "result": {"accepted": True}}
    def gripper(self, session_id, client_id, sequence, payload):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "gripper", "payload": payload})
        return {"ok": True}
    def hold(self, session_id, client_id, sequence, reason):
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "hold", "payload": {"reason": reason}})
        return {"ok": True}
    def heartbeat(self, client_id, session_id): return {"ok": True}


class Pi05AdapterTests(unittest.TestCase):
    def test_action_chunk_flows_to_absolute_osc_target(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        with patch("supervisor.pi05_adapter.OpenPIClient", _Policy):
            adapter.start("session-1", "client-1")
            deadline = time.monotonic() + 1.0
            while len(broker.commands) < 2 and time.monotonic() < deadline:
                time.sleep(.01)
            adapter.stop()
        self.assertGreaterEqual(len(broker.commands), 2)
        motion, gripper = broker.commands[:2]
        self.assertEqual(motion["type"], "track_tcp")
        # π0.5 +X is configured as NERO -X; 1 * .05 * .2 = .01 m.
        self.assertAlmostEqual(motion["payload"]["target_pose"]["position_m"][0], -.01, places=6)
        self.assertEqual(gripper["type"], "gripper")
        self.assertAlmostEqual(gripper["payload"]["width_m"], .01)
        self.assertEqual(adapter.snapshot()["action_chunk_length"], 1)

    def test_rejects_duplicate_camera_indices(self):
        adapter = Pi05InputAdapter(_Broker(), Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        with self.assertRaises(ValueError):
            adapter.update_config({"cameras": {"external": {"index": 2}, "wrist": {"index": 2}}})
