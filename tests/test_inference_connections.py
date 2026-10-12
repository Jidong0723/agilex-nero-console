"""Independent entry/protocol diagnostics; loopback fake services only."""
import copy
import json
import socketserver
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgpack
from websockets.sync.server import serve

from supervisor.inference_connection import listener_matches, local_listener
from supervisor.openpi_client import OpenPIClient
from supervisor.pi05_adapter import Pi05InputAdapter
from tests.test_pi05_adapter import _Broker

ROOT = Path(__file__).parents[1]


@contextmanager
def diagnostic_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pi05.json"
        path.write_bytes((ROOT / "config/pi05.json").read_bytes())
        with patch.object(Pi05InputAdapter, "_connection_loop"), patch.object(Pi05InputAdapter, "_osc_state_loop"):
            adapter = Pi05InputAdapter(_Broker(), path)
        try:
            yield adapter
        finally:
            adapter.close()


class ListenerTests(unittest.TestCase):
    def test_address_port_family_and_wildcards(self):
        cases = [("127.0.0.1:8000", "127.0.0.1", True),
                 ("127.0.0.2:8000", "127.0.0.1", False),
                 ("192.168.1.20:8000", "127.0.0.1", False),
                 ("0.0.0.0:8000", "127.0.0.1", True),
                 ("[::]:8000", "::1", True),
                 ("[::1]:8000", "ws://[::1]/policy", True),
                 ("[::1]:8000", "127.0.0.1", False),
                 ("0.0.0.0:8000", "::1", False),
                 ("[::1]:8000", "localhost", True),
                 ("127.0.0.1:8001", "127.0.0.1", False)]
        for endpoint, host, expected in cases:
            with self.subTest(endpoint=endpoint, host=host):
                self.assertEqual(listener_matches(endpoint, host, 8000), expected)

    def test_windows_wrong_address_and_inspection_errors(self):
        output = " TCP 192.168.1.20:8000 0.0.0.0:0 LISTENING 42\n TCP [::1]:8000 [::]:0 LISTENING 43"
        with patch("supervisor.inference_connection.os.name", "nt"), patch("supervisor.inference_connection.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=output)):
            self.assertFalse(local_listener("127.0.0.1", 8000))
            self.assertTrue(local_listener("::1", 8000))
        for result in (SimpleNamespace(returncode=1, stdout=""), OSError("missing tool")):
            kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with patch("supervisor.inference_connection.subprocess.run", **kwargs):
                self.assertIsNone(local_listener("localhost", 8000))
        self.assertIsNone(local_listener("model.example", 8000))

    def test_linux_ss_address_column(self):
        output = "LISTEN 0 128 127.0.0.2:8000 0.0.0.0:*\nLISTEN 0 128 [::]:8000 [::]:*"
        with patch("supervisor.inference_connection.os.name", "posix"), patch("supervisor.inference_connection.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=output)):
            self.assertFalse(local_listener("127.0.0.1", 8000))
            self.assertTrue(local_listener("::1", 8000))


class ConnectionStatusTests(unittest.TestCase):
    def render(self, adapter):
        adapter._last_connection_probe = 0
        with adapter.lock:
            adapter._refresh_connections_locked()
            return copy.deepcopy(adapter._connections)

    def test_both_profiles_listener_green_with_protocol_failure(self):
        with diagnostic_adapter() as adapter:
            for profile in ("local5090", "autodl"):
                adapter.config["inference_profiles"]["active"] = profile
                adapter._forward_listener = True
                adapter._listener_checked_at = time.time()
                adapter.state.update(model_state="DISCONNECTED", websocket_error="HTTP 200 is not a WebSocket")
                adapter._connection_stage = "failed"
                result = self.render(adapter)
                self.assertEqual(result["service_entry"]["state"], "ok")
                self.assertEqual(result["policy"]["state"], "bad")
                self.assertEqual(result["ssh_forward"], result["service_entry"])

    def test_unknown_stale_tcp_fallback_and_old_revision(self):
        with diagnostic_adapter() as adapter:
            adapter._forward_listener = None
            adapter._listener_checked_at = time.time()
            self.assertEqual(self.render(adapter)["service_entry"]["state"], "neutral")
            adapter._connection_event(0, "tcp_connected")
            self.assertEqual(self.render(adapter)["service_entry"]["basis"], "tcp_connection")
            self.assertEqual(self.render(adapter)["service_entry"]["state"], "ok")
            adapter._tcp_evidence["checked_at"] -= 11
            adapter._listener_checked_at -= 11
            self.assertEqual(self.render(adapter)["service_entry"]["basis"], "pending")
            before = copy.deepcopy(adapter._tcp_evidence)
            adapter._connection_revision = 1
            adapter._connection_event(0, "tcp_error", "old result")
            self.assertEqual(adapter._tcp_evidence, before)
            adapter._connection_event(1, "tcp_error", "connection refused")
            self.assertEqual(self.render(adapter)["service_entry"]["state"], "bad")

    def test_metadata_contract_failure_does_not_mark_connected_websocket_red(self):
        with diagnostic_adapter() as adapter:
            adapter.state.update(model_state="CONNECTED", protocol_compatibility={"status": "incompatible", "message": "state8"})
            adapter._connection_event(0, "tcp_connected")
            self.assertEqual(self.render(adapter)["policy"]["state"], "ok")

    def test_endpoint_update_clears_evidence(self):
        with diagnostic_adapter() as adapter:
            adapter._connection_event(0, "tcp_connected")
            adapter._forward_listener = True
            adapter._listener_checked_at = time.time()
            adapter.update_config({"policy_endpoint": {"host": "127.0.0.1", "port": 9913}})
            self.assertFalse(adapter._tcp_evidence)
            self.assertIsNone(adapter._forward_listener)
            self.assertEqual(self.render(adapter)["service_entry"]["state"], "neutral")


class TransportStagesTests(unittest.TestCase):
    def test_tcp_success_websocket_failure_uses_exactly_one_connection(self):
        count = []
        class HttpService(socketserver.BaseRequestHandler):
            def handle(inner):
                count.append(1)
                request = inner.request.recv(8192)
                self.assertIn(b"Upgrade: websocket", request)
                inner.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        with socketserver.TCPServer(("127.0.0.1", 0), HttpService) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            events = []
            try:
                with self.assertRaises(Exception):
                    OpenPIClient("127.0.0.1", server.server_address[1], 1, lambda stage, detail=None: events.append(stage))
                self.assertEqual(events, ["tcp_connected", "protocol_error"])
                self.assertEqual(len(count), 1)
            finally:
                server.shutdown(); worker.join(2)

    def test_metadata_validation_and_no_probe_connections(self):
        for metadata, valid in (([7], False), ({"state_dim": 7}, True)):
            with self.subTest(metadata=metadata):
                count = []
                def handler(ws):
                    count.append(1)
                    ws.send(msgpack.packb(metadata))
                    try:
                        ws.recv()
                    except Exception:
                        pass
                with serve(handler, "127.0.0.1", 0, compression=None) as server:
                    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
                    events = []
                    try:
                        args = ("127.0.0.1", server.socket.getsockname()[1], 1, lambda stage, detail=None: events.append(stage))
                        if valid:
                            client = OpenPIClient(*args); client.close()
                        else:
                            with self.assertRaisesRegex(RuntimeError, "metadata"):
                                OpenPIClient(*args)
                        self.assertEqual(events[:2], ["tcp_connected", "websocket_connected"])
                        self.assertEqual(events[-1], "metadata_received" if valid else "protocol_error")
                        self.assertEqual(len(count), 1)
                    finally:
                        server.shutdown(); worker.join(2)


if __name__ == "__main__":
    unittest.main()
