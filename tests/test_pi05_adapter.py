from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from supervisor.pi05_adapter import Pi05InputAdapter, _pack_array, _unpack_array


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
        # H20 Nero chunk. Every step is converted against one shared
        # pre-inference feedback pose; only the first five are executed.
        return {"actions": np.asarray([[.01, 0., 0., 0., 0., 0., 1.]] * 20, dtype=np.float32)}

    def close(self):
        pass

    def is_alive(self):
        return True


class _InitiallyUnavailablePolicy:
    """Keep constructor background threads off a live local policy tunnel."""
    def __init__(self, *_args):
        raise RuntimeError("test policy connection disabled")


class _Broker:
    def __init__(self):
        self.commands = []
        self.fail_control = False
        self.track_delay_s = 0.0
        self.session = {"state": "ACTIVE", "id": "session-1", "client_id": "client-1", "execution_mode": "shadow"}

    def state(self):
        if self.fail_control: raise RuntimeError("simulated OSC outage")
        return {"session": dict(self.session), "command": {"target_tcp": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}},
                "execution": {"measured_tcp_pose": {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}}, "gripper": {"width_m": .02}}

    def track_tcp(self, session_id, client_id, sequence, target_pose):
        if self.fail_control: raise RuntimeError("simulated OSC outage")
        started_at = time.perf_counter()
        if self.track_delay_s:
            time.sleep(self.track_delay_s)
        self.commands.append({"session_id": session_id, "client_id": client_id, "sequence": sequence, "type": "track_tcp", "payload": {"target_pose": target_pose},
                              "started_at": started_at, "completed_at": time.perf_counter()})
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
    def setUp(self):
        # Pi05InputAdapter starts its connection owner during construction.
        # Never let that thread pick up an operator's real 127.0.0.1:8000
        # tunnel before an individual test explicitly installs its own policy.
        self._initial_policy_patch = patch("supervisor.pi05_adapter.OpenPIClient", _InitiallyUnavailablePolicy)
        self._initial_policy_patch.start()

    def tearDown(self):
        self._initial_policy_patch.stop()

    def test_openpi_msgpack_packer_preserves_rgb_array_shapes(self):
        """The live OpenPI server requires Packer's binary ndarray markers."""
        import msgpack
        observation = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
        }
        payload = msgpack.Packer(default=_pack_array).pack(observation)
        decoded = msgpack.unpackb(payload, object_hook=_unpack_array)
        self.assertEqual(decoded["observation/image"].shape, (224, 224, 3))
        self.assertEqual(decoded["observation/wrist_image"].shape, (224, 224, 3))
        self.assertEqual(decoded["observation/state"].shape, (8,))

    def test_observation_uses_direct_nero_base_pose_and_gripper_ratio(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            observation = adapter._observation(broker.state(), np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8))
        finally:
            adapter.close()
        expected_ratio = .02 / .095
        np.testing.assert_allclose(observation["observation/state"], [0., 0., .3, 0., 0., 0., expected_ratio, -expected_ratio])

    def test_action_decoding_uses_chunk_start_and_left_multiplied_base_rotation(self):
        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            half = np.pi / 4
            base = {"position_m": [.1, .2, .3], "orientation_xyzw": [0., 0., np.sin(half), np.cos(half)]}
            target, ratio = adapter._target_from_action([.01, -.02, .03, np.pi / 2, 0., 0., .25], base)
        finally:
            adapter.close()
        self.assertEqual(target["position_m"], [.11, .18000000000000002, .32999999999999996])
        # Exp(+X 90°) · Rz(+90°) expresses the incremental axis in NERO base coordinates.
        np.testing.assert_allclose(target["orientation_xyzw"], [.5, -.5, .5, .5], atol=1e-6)
        self.assertEqual(ratio, .25)

    def test_action_delta_scale_reduces_tcp_delta_but_not_gripper_ratio(self):
        adapter = Pi05InputAdapter(_Broker(), Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            adapter.config["execution"]["action_delta_scale"] = .25
            target, ratio = adapter._target_from_action([.04, -.08, .12, 0., 0., 0., .75],
                                                        {"position_m": [.1, .2, .3], "orientation_xyzw": [0., 0., 0., 1.]})
        finally:
            adapter.close()
        self.assertAlmostEqual(target["position_m"][0], .11)
        self.assertAlmostEqual(target["position_m"][1], .18)
        self.assertAlmostEqual(target["position_m"][2], .33)
        self.assertEqual(ratio, .75)

    def test_invalid_gripper_ratio_rejects_action_without_clipping(self):
        adapter = Pi05InputAdapter(_Broker(), Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            with self.assertRaisesRegex(RuntimeError, "row unknown.*1.01"):
                adapter._target_from_action([0., 0., 0., 0., 0., 0., 1.01], {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]})
            diagnostic = adapter.snapshot()["gripper_ratio_diagnostic"]
        finally:
            adapter.close()
        self.assertEqual(diagnostic["status"], "rejected")
        self.assertEqual(diagnostic["action_index"], None)
        self.assertAlmostEqual(diagnostic["raw_ratio"], 1.01)

    def test_gripper_ratio_endpoint_roundoff_is_normalized_and_diagnosed(self):
        adapter = Pi05InputAdapter(_Broker(), Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        try:
            _target, high_ratio = adapter._target_from_action(
                [0., 0., 0., 0., 0., 0., 1.0005],
                {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}, action_index=3)
            _target, low_ratio = adapter._target_from_action(
                [0., 0., 0., 0., 0., 0., -0.0005],
                {"position_m": [0., 0., .3], "orientation_xyzw": [0., 0., 0., 1.]}, action_index=4)
            diagnostic = adapter.snapshot()["gripper_ratio_diagnostic"]
        finally:
            adapter.close()
        self.assertEqual(high_ratio, 1.0)
        self.assertEqual(low_ratio, 0.0)
        self.assertEqual(diagnostic["status"], "boundary_normalized")
        self.assertEqual(diagnostic["action_index"], 4)
        self.assertAlmostEqual(diagnostic["raw_ratio"], -0.0005)

    def test_boundary_roundoff_chunk_continues_without_hold(self):
        """A tiny float32 endpoint overshoot must not stop AutoDL control."""
        class BoundaryPolicy(_Policy):
            def infer(self, observation):
                return {"actions": np.asarray([[.01, 0., 0., 0., 0., 0., 1.0005]] * 20, dtype=np.float32)}

        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        try:
            with patch("supervisor.pi05_adapter.OpenPIClient", BoundaryPolicy):
                adapter.start("session-1", "client-1")
                deadline = time.monotonic() + 1.0
                while len([item for item in broker.commands if item["type"] == "track_tcp"]) < 5 and time.monotonic() < deadline:
                    time.sleep(.01)
                snapshot = adapter.snapshot()
        finally:
            adapter.close()
        self.assertNotEqual(snapshot["state"], "ERROR")
        self.assertGreaterEqual(len([item for item in broker.commands if item["type"] == "track_tcp"]), 5)
        self.assertFalse(any(item["type"] == "hold" for item in broker.commands))
        self.assertEqual(snapshot["gripper_ratio_diagnostic"]["status"], "boundary_normalized")

    def test_rejected_osc_target_stops_entire_chunk_and_requests_hold(self):
        broker = _Broker()
        broker.fail_track = True
        original_track = broker.track_tcp
        def rejected_track(*args, **kwargs):
            if broker.fail_track:
                return {"ok": False, "reason": "workspace rejected"}
            return original_track(*args, **kwargs)
        broker.track_tcp = rejected_track
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .01
        with patch("supervisor.pi05_adapter.OpenPIClient", _Policy):
            adapter.start("session-1", "client-1")
            deadline = time.monotonic() + 1.0
            while adapter.snapshot()["state"] != "ERROR" and time.monotonic() < deadline:
                time.sleep(.01)
            snapshot = adapter.snapshot()
            adapter.close()
        self.assertEqual(snapshot["state"], "ERROR")
        self.assertIn("workspace rejected", snapshot["last_rejection"])
        self.assertEqual(len([item for item in broker.commands if item["type"] == "track_tcp"]), 0)
        self.assertTrue(any(item["type"] == "hold" for item in broker.commands))

    def test_workspace_preflight_rejects_whole_chunk_before_first_dispatch(self):
        class OutsideWorkspacePolicy(_Policy):
            def infer(self, observation):
                return {"actions": np.asarray([[1., 0., 0., 0., 0., 0., .5]] * 20, dtype=np.float32)}

        broker = _Broker()
        original_state = broker.state
        def state_with_workspace():
            snapshot = original_state()
            snapshot["workspace"] = {"min_xyz_m": [-.6, -.6, 0.], "max_xyz_m": [.6, .6, .7], "min_tcp_z_m": 0.}
            return snapshot
        broker.state = state_with_workspace
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        with patch("supervisor.pi05_adapter.OpenPIClient", OutsideWorkspacePolicy):
            adapter.start("session-1", "client-1")
            deadline = time.monotonic() + 1.0
            while adapter.snapshot()["state"] != "ERROR" and time.monotonic() < deadline:
                time.sleep(.01)
            snapshot = adapter.snapshot()
            adapter.close()
        self.assertEqual(snapshot["state"], "ERROR")
        self.assertIn("outside the OSC workspace", snapshot["last_rejection"])
        self.assertFalse(any(item["type"] == "track_tcp" for item in broker.commands))
        self.assertTrue(any(item["type"] == "hold" for item in broker.commands))

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
        self.assertTrue(snapshot["websocket_error"])
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
        # AutoDL values are physical Nero metres; no LIBERO sign or scale is applied.
        self.assertAlmostEqual(motion["payload"]["target_pose"]["position_m"][0], .01, places=6)
        self.assertEqual(gripper["type"], "gripper")
        self.assertAlmostEqual(gripper["payload"]["width_m"], .095)
        # Targets are not cumulative inside the chunk: all five use the same
        # feedback pose captured for the inference observation.
        for item in motions:
            self.assertAlmostEqual(item["payload"]["target_pose"]["position_m"][0], .01, places=6)
        self.assertEqual(snapshot["action_chunk_length"], 20)
        self.assertEqual(len(snapshot["absolute_tcp_chunk"]), 20)
        self.assertEqual(snapshot["inference_base_tcp"]["position_m"], [0., 0., .3])

    def test_slow_inference_does_not_burst_action_chunk_dispatches(self):
        class SlowPolicy(_Policy):
            inference_completed_at = None

            def infer(self, observation):
                time.sleep(.06)
                response = super().infer(observation)
                type(self).inference_completed_at = time.perf_counter()
                return response

        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .03
        try:
            with patch("supervisor.pi05_adapter.OpenPIClient", SlowPolicy):
                adapter.start("session-1", "client-1")
                deadline = time.monotonic() + 2.0
                while len([item for item in broker.commands if item["type"] == "track_tcp"]) < 5 and time.monotonic() < deadline:
                    time.sleep(.005)
                motions = [item for item in broker.commands if item["type"] == "track_tcp"][:5]
                snapshot = adapter.snapshot()
        finally:
            adapter.close()
        self.assertEqual(len(motions), 5)
        self.assertIsNotNone(SlowPolicy.inference_completed_at)
        self.assertLess(motions[0]["started_at"] - SlowPolicy.inference_completed_at, .03)
        self.assertGreaterEqual(snapshot["inference_ms"], 50.0)
        for previous, current in zip(motions, motions[1:]):
            self.assertGreaterEqual(current["started_at"] - previous["completed_at"], .024)

    def test_inference_latency_excludes_camera_capture_time(self):
        class SlowCamera(_Camera):
            def read(self):
                time.sleep(.05)
                return super().read()

        class TimedPolicy(_Policy):
            def infer(self, observation):
                time.sleep(.02)
                return super().infer(observation)

        broker = _Broker()
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = SlowCamera()
        adapter.config["execution"]["period_s"] = .01
        try:
            with patch("supervisor.pi05_adapter.OpenPIClient", TimedPolicy):
                adapter.start("session-1", "client-1")
                deadline = time.monotonic() + 2.0
                while adapter.snapshot()["inference_ms"] is None and time.monotonic() < deadline:
                    time.sleep(.005)
                inference_ms = adapter.snapshot()["inference_ms"]
        finally:
            adapter.close()
        self.assertIsNotNone(inference_ms)
        self.assertGreaterEqual(inference_ms, 15.0)
        self.assertLess(inference_ms, 40.0)

    def test_slow_osc_dispatch_stretches_chunk_without_catchup_burst(self):
        broker = _Broker()
        broker.track_delay_s = .05
        adapter = Pi05InputAdapter(broker, Path(__file__).resolve().parents[1] / "config" / "pi05.json")
        adapter.cameras = _Camera()
        adapter.config["execution"]["period_s"] = .03
        try:
            with patch("supervisor.pi05_adapter.OpenPIClient", _Policy):
                adapter.start("session-1", "client-1")
                deadline = time.monotonic() + 2.0
                while len([item for item in broker.commands if item["type"] == "track_tcp"]) < 5 and time.monotonic() < deadline:
                    time.sleep(.005)
                motions = [item for item in broker.commands if item["type"] == "track_tcp"][:5]
        finally:
            adapter.close()
        self.assertEqual(len(motions), 5)
        for previous, current in zip(motions, motions[1:]):
            self.assertGreaterEqual(current["started_at"] - previous["completed_at"], .024)

    def test_osc_outage_rejects_chunk_and_holds_before_more_rows_execute(self):
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
        self.assertEqual(snapshot["state"], "ERROR")
        self.assertIn("simulated OSC outage", snapshot["last_rejection"])
        self.assertTrue(any(item["type"] == "hold" for item in broker.commands))

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
