"""Unified presets/contracts against fake OSC and an ephemeral WebSocket server."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import msgpack
import numpy as np
from websockets.sync.server import serve

from supervisor.inference_config import canonical, endpoint_uri, prepare_update, project, save_config
from supervisor.inference_contract import action_rows, make_observation, metadata_compatibility
from supervisor.openpi_client import OpenPIClient, _pack_array, _unpack_array
from supervisor.pi05_adapter import Pi05InputAdapter
from tests.test_pi05_adapter import _Broker, _Camera, _InitiallyUnavailablePolicy

ROOT = Path(__file__).parents[1]


class UnifiedConfigTests(unittest.TestCase):
    def config(self):
        return project(json.loads((ROOT / "config/pi05.json").read_text(encoding="utf-8")))

    def test_legacy_active_values_win_and_projection_has_one_persisted_source(self):
        old = self.config()
        old["schema_version"] = "nero.autodl-inference.v1"
        old["model"].update(host="localhost", port=9911)
        old["execution"].update(expected_chunk_steps=13, replan_steps=4)
        old["inference_profiles"]["autodl"].pop("expected_chunk_steps")
        old["inference_profiles"]["autodl"].pop("replan_steps")
        result = project(old)
        self.assertEqual(result["model"]["port"], 9911)
        self.assertEqual(result["execution"]["expected_chunk_steps"], 13)
        self.assertEqual(result["inference_profiles"]["autodl"]["replan_steps"], 4)
        persisted = canonical(result)
        self.assertNotIn("host", persisted["model"])
        self.assertNotIn("connection", persisted)
        self.assertNotIn("expected_chunk_steps", persisted["execution"])
        self.assertEqual(project(persisted), result)

    def test_presets_remember_independent_lengths_across_disk_reload(self):
        local = prepare_update(self.config(), {"inference_profile": "local5090", "execution": {"expected_chunk_steps": 13, "replan_steps": 4}})
        cloud = prepare_update(local, {"inference_profile": "autodl", "execution": {"expected_chunk_steps": 20, "replan_steps": 9}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pi05.json"
            save_config(path, cloud)
            reloaded = project(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(reloaded["execution"]["replan_steps"], 9)
        local = prepare_update(reloaded, {"inference_profile": "local5090"})
        self.assertEqual((local["execution"]["expected_chunk_steps"], local["execution"]["replan_steps"]), (13, 4))

    def test_invalid_groups_leave_original_unchanged(self):
        original = self.config()
        for H, K in ((0, 1), (10, 11), (True, 1), (10, 0), (10.5, 3), (10, "5")):
            with self.subTest(H=H, K=K), self.assertRaises(ValueError):
                prepare_update(original, {"execution": {"expected_chunk_steps": H, "replan_steps": K}})
        with self.assertRaises(ValueError):
            prepare_update(original, {"execution": {"expected_chunk_steps": 20}})
        with self.assertRaises(ValueError):
            prepare_update(original, {"policy_endpoint": {"host": "localhost", "port": True}})
        self.assertEqual(original, self.config())
        self.assertEqual(endpoint_uri("wss://example.com/policy", 9911), "wss://example.com:9911/policy")
        self.assertEqual(endpoint_uri("::1", 9911), "ws://[::1]:9911")


class UnifiedContractTests(unittest.TestCase):
    def test_state7_and_contract_failures(self):
        image = np.zeros((224, 224, 3), np.uint8)
        obs = make_observation([.1, .2, .3], [0, 0, 0], .5, image, image, "pick")
        self.assertEqual(obs["observation/state"].dtype, np.float32)
        self.assertEqual(obs["observation/state"].shape, (7,))
        self.assertEqual(metadata_compatibility({}, 10)["status"], "pending")
        for metadata in ({"state_dim": 8}, {"action_dim": 8}, {"action_horizon": 20}):
            self.assertEqual(metadata_compatibility(metadata, 10)["status"], "incompatible")
        self.assertEqual(metadata_compatibility({"valid_action_dim": 7, "action_dim": 32}, 10)["status"], "pending")
        for rows in ([[0] * 7] * 9, [[0] * 8] * 10, [[float("nan")] * 7] * 10, [[True] * 7] * 10):
            with self.assertRaises(RuntimeError):
                action_rows({"actions": rows}, 10)

    def test_both_presets_use_real_shared_websocket_and_state7(self):
        requests = []
        def handler(socket):
            socket.send(msgpack.packb({"state_dim": 7, "valid_action_dim": 7, "action_horizon": 13}))
            try:
                for raw in socket:
                    requests.append(msgpack.unpackb(raw, object_hook=_unpack_array))
                    socket.send(msgpack.Packer(default=_pack_array).pack({"actions": np.zeros((13, 7), np.float32)}))
            except Exception:
                pass
        with serve(handler, "127.0.0.1", 0, compression=None) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for name in ("local5090", "autodl"):
                    base = project(json.loads((ROOT / "config/pi05.json").read_text(encoding="utf-8")))
                    config = prepare_update(base, {"inference_profile": name,
                        "policy_endpoint": {"host": "127.0.0.1", "port": server.socket.getsockname()[1]},
                        "execution": {"expected_chunk_steps": 13, "replan_steps": 4}})
                    client = OpenPIClient(config["model"]["host"], config["model"]["port"], 2)
                    try:
                        image = np.zeros((224, 224, 3), np.uint8)
                        response = client.infer(make_observation([0, 0, .3], [0, 0, 0], .5, image, image, "pick"))
                        self.assertEqual(len(action_rows(response, 13)), 13)
                        self.assertTrue(client.is_alive())
                    finally:
                        client.close()
            finally:
                server.shutdown(); thread.join(2)
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(request["observation/state"].shape, (7,))
            self.assertEqual(request["observation/image"].dtype, np.uint8)


    def test_connection_owner_recovers_from_websocket_close_and_text_error(self):
        for failure in ("close", "text"):
            with self.subTest(failure=failure):
                count = [0]
                def handler(socket):
                    socket.send(msgpack.packb({"state_dim": 7, "action_horizon": 10}))
                    try:
                        for raw in socket:
                            request = msgpack.unpackb(raw, object_hook=_unpack_array)
                            self.assertEqual(request["observation/state"].shape, (7,))
                            count[0] += 1
                            if count[0] == 1:
                                if failure == "text":
                                    socket.send("temporary model error")
                                else:
                                    socket.close()
                                continue
                            socket.send(msgpack.Packer(default=_pack_array).pack({"actions": np.zeros((10, 7), np.float32)}))
                    except Exception:
                        pass
                with serve(handler, "127.0.0.1", 0, compression=None) as server, tempfile.TemporaryDirectory() as tmp:
                    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                    path = Path(tmp) / "pi05.json"
                    config = json.loads((ROOT / "config/pi05.json").read_text(encoding="utf-8"))
                    for name in ("local5090", "autodl"):
                        config["inference_profiles"][name].update(host="127.0.0.1", port=server.socket.getsockname()[1])
                    config["model"]["request_timeout_s"] = 1
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with patch.object(Pi05InputAdapter, "_local_forward_listening", return_value=True):
                        adapter = Pi05InputAdapter(_Broker(), path)
                        try:
                            adapter.cameras = _Camera(); adapter._ensure_worker()
                            deadline = time.monotonic() + 6
                            while time.monotonic() < deadline:
                                if adapter.state["websocket_diagnostics"]["connect_successes"] >= 2 and adapter.state["protocol_compatibility"]["status"] == "compatible":
                                    break
                                time.sleep(.02)
                            self.assertGreaterEqual(adapter.state["websocket_diagnostics"]["connect_successes"], 2)
                            self.assertEqual(adapter.state["protocol_compatibility"]["status"], "compatible")
                            self.assertFalse(adapter.osc.commands)
                        finally:
                            adapter.close(); server.shutdown(); thread.join(2)


class UnifiedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config/pi05.json"
        self.path.parent.mkdir()
        self.path.write_text((ROOT / "config/pi05.json").read_text(encoding="utf-8"), encoding="utf-8")
        with patch("supervisor.pi05_adapter.OpenPIClient", _InitiallyUnavailablePolicy):
            self.adapter = Pi05InputAdapter(_Broker(), self.path)
        self.adapter._connection_stop.set(); self.adapter._connection_wakeup.set()
        self.adapter._connection_thread.join(2)
        self.adapter._osc_state_stop.set(); self.adapter._osc_state_thread.join(2)

    def tearDown(self):
        self.adapter.close(); self.tmp.cleanup()

    def test_save_failure_keeps_config_and_socket(self):
        policy = type("Policy", (), {"close": lambda _: self.fail("must not close old socket")})()
        self.adapter._policy = policy
        before = copy.deepcopy(self.adapter.config)
        disk = self.path.read_bytes()
        with patch("supervisor.inference_config.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.adapter.update_config({"inference_profile": "autodl"})
        self.assertEqual(self.adapter.config, before)
        self.assertEqual(self.path.read_bytes(), disk)
        self.assertIs(self.adapter._policy, policy)
        self.adapter._policy = None

    def test_active_and_priming_output_rejects_all_connection_changes(self):
        before = copy.deepcopy(self.adapter.config)
        for flag in ("execution_enabled", "priming"):
            self.adapter.state[flag] = True
            try:
                with self.assertRaisesRegex(RuntimeError, "停止"):
                    self.adapter.update_config({"inference_profile": "autodl"})
                with self.assertRaisesRegex(RuntimeError, "停止"):
                    self.adapter.update_config({"execution": {"expected_chunk_steps": 13, "replan_steps": 4}})
            finally:
                self.adapter.state[flag] = False
        self.assertEqual(self.adapter.config, before)

    def test_declared_incompatibility_cannot_prime_hardware(self):
        self.adapter.cameras = _Camera()
        self.adapter.osc.session["execution_mode"] = "hardware"
        self.adapter._policy = type("Policy", (), {"metadata": {"state_dim": 8}, "close": lambda _: None})()
        with self.assertRaisesRegex(RuntimeError, "state_dim"):
            self.adapter.start("session-1", "client-1")
        self.assertFalse(self.adapter.osc.commands)

    def test_preview_can_validate_protocol_before_any_tcp_pose_exists(self):
        a = self.adapter
        class Preview:
            def infer(self, observation):
                return {"actions": np.zeros((10, 7), np.float32)}
            def close(self):
                pass
        a._osc_snapshot = {"session": None, "execution": {}, "command": {}}
        a._policy = Preview(); a.cameras = _Camera(); a._ensure_worker()
        deadline = time.monotonic() + 1
        while a.state["protocol_compatibility"]["status"] != "compatible" and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(a.state["protocol_compatibility"]["status"], "compatible")
        self.assertEqual(a.state["last_response_steps"], 10)
        self.assertFalse(a.osc.commands)

    def test_switch_discards_old_preview_and_executes_exact_non_half_prefix(self):
        a = self.adapter
        entered, release = threading.Event(), threading.Event()
        class Slow:
            def infer(self, observation):
                entered.set(); release.wait(2)
                return {"actions": np.zeros((20, 7), np.float32)}
            def close(self):
                release.set()
        a._policy = Slow(); a.cameras = _Camera(); a._ensure_worker()
        self.assertTrue(entered.wait(1))
        a.update_config({"execution": {"expected_chunk_steps": 13, "replan_steps": 4}})
        self.assertIsNone(a.state["action_chunk"])
        self.assertNotEqual(a.state["state"], "ERROR")
        class Current:
            def infer(self, observation):
                return {"actions": np.asarray([[.001, 0, 0, 0, 0, 0, .5]] * 13)}
            def close(self):
                pass
        a._policy = Current()
        a.config["execution"].update(period_s=.001, arrival_wait_s=0)
        a.start("session-1", "client-1")
        deadline = time.monotonic() + 2
        while a.state["executed_steps"] < 4 and time.monotonic() < deadline:
            time.sleep(.005)
        a.stop("test finished")
        self.assertGreaterEqual(a.state["executed_steps"], 4)
        self.assertEqual(a.state["last_response_steps"], 13)


if __name__ == "__main__":
    unittest.main()
