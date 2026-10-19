from __future__ import annotations

import unittest
import threading

from nero_console.application.adapter_runtime import OscClient
from motion.osc import _OperationalSpaceServo


class _FastBroker:
    def __init__(self) -> None:
        self.fast_calls = []
        self.rpc_calls = []

    def osc_track_tcp_nonblocking(self, session_id, client_id, sequence, target_pose):
        self.fast_calls.append((session_id, client_id, sequence, target_pose))
        return {
            "ok": True,
            "accepted": True,
            "session_id": session_id,
            "accepted_sequence": sequence,
            "mailbox": "osc_target_latest_value",
        }

    def osc_command(self, body):
        self.rpc_calls.append(body)
        raise AssertionError("high-rate track_tcp must not use generic osc_command")


class FastOscTargetTests(unittest.TestCase):
    def test_track_tcp_uses_nonblocking_mailbox_ingress(self) -> None:
        broker = _FastBroker()
        client = OscClient(broker, "pi05")
        pose = {
            "position_m": [0.1, 0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

        result = client.track_tcp("session", "client", 17, pose)

        self.assertTrue(result["ok"])
        self.assertEqual(len(broker.fast_calls), 1)
        self.assertEqual(broker.fast_calls[0][:3], ("session", "client", 17))
        self.assertEqual(broker.fast_calls[0][3], pose)
        self.assertEqual(broker.rpc_calls, [])

    def test_servo_target_mailbox_replaces_pending_value(self) -> None:
        servo = object.__new__(_OperationalSpaceServo)
        servo._target_mailbox_lock = threading.Lock()
        servo._target_mailbox = None
        servo._target_mailbox_revision = 0
        servo._target_mailbox_last = {"status": "empty"}
        servo._fast_input_state = {
            "session_id": "session", "client_id": "client",
            "session_state": "ACTIVE", "trajectory_state": "RUNNING", "sequence": 0,
        }
        servo.limits = {
            "workspace_min_m": [-1.0, -1.0, -1.0],
            "workspace_max_m": [1.0, 1.0, 1.0],
            "min_tcp_z_m": -1.0,
        }
        pose = {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        self.assertTrue(servo.publish_target_nonblocking("session", "client", 1, pose)["accepted"])
        self.assertTrue(servo.publish_target_nonblocking("session", "client", 2, pose)["accepted"])
        pending = servo._take_target_mailbox()

        self.assertEqual(pending["sequence"], 2)
        self.assertEqual(pending["mailbox_revision"], 2)
        self.assertIsNone(servo._take_target_mailbox())


if __name__ == "__main__":
    unittest.main()
