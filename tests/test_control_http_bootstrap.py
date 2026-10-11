from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.nero_control_server import (
    BackendProcessProxy,
    ControlRequestHandler,
    ExclusiveThreadingHTTPServer,
    ServiceRuntime,
)


class BlockingBroker:
    entered = threading.Event()
    release = threading.Event()

    def __init__(self, _config: Path) -> None:
        self.entered.set()
        self.release.wait(2.0)

    def start(self) -> dict[str, object]:
        return {"connected": False, "reason": "test hardware unavailable"}

    def health(self) -> dict[str, object]:
        return {"running": True}

    def close(self) -> None:
        return None


class ControlHttpBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        BlockingBroker.entered.clear()
        BlockingBroker.release.clear()
        self.runtime = ServiceRuntime()
        ControlRequestHandler.runtime = self.runtime
        self.server = ExclusiveThreadingHTTPServer(("127.0.0.1", 0), ControlRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        BlockingBroker.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1.0)
        self.runtime.close()

    def test_page_and_health_respond_while_backend_constructor_is_blocked(self) -> None:
        bootstrap = threading.Thread(
            target=self.runtime.initialize,
            args=(Path("unused.json"), BlockingBroker),
            daemon=True,
        )
        bootstrap.start()
        self.assertTrue(BlockingBroker.entered.wait(0.5))

        with urlopen(f"{self.base_url}/", timeout=0.5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"NERO", response.read())
        with urlopen(f"{self.base_url}/api/health", timeout=0.5) as response:
            payload = json.load(response)
        self.assertTrue(payload["data"]["http_ready"])
        self.assertFalse(payload["data"]["control_backend_ready"])
        self.assertEqual(payload["data"]["control_backend_phase"], "starting")

        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.base_url}/api/status", timeout=0.5)
        self.assertEqual(raised.exception.code, 503)

        BlockingBroker.release.set()
        bootstrap.join(1.0)
        with urlopen(f"{self.base_url}/api/health", timeout=0.5) as response:
            ready = json.load(response)
        self.assertTrue(ready["data"]["control_backend_ready"])

    def test_disconnected_hardware_is_not_labeled_as_service_offline(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "web" / "console" / "app.js"
        app = app_path.read_text(encoding="utf-8")
        self.assertIn('"\\u786c\\u4ef6\\u5df2\\u8fde\\u63a5"', app)
        self.assertIn('"\\u786c\\u4ef6\\u672a\\u8fde\\u63a5"', app)
        self.assertNotIn('control.connected ? "服务在线" : "服务离线"', app)

    def test_cpv_group_http_forwards_complete_values_and_explicit_save(self) -> None:
        from unittest.mock import Mock, patch
        broker = Mock()
        broker.osc_set_cpv_parameters.return_value = {"saved_startup": False}
        values = {"cv_rad_s": 2, "acc_dcc_rad_s2": 10}
        with patch.object(self.runtime, "require_broker", return_value=broker):
            request = Request(f"{self.base_url}/api/osc/cpv-parameters",
                              data=json.dumps({"values": values}).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=1) as response:
                self.assertFalse(json.load(response)["data"]["saved_startup"])
            broker.osc_set_cpv_parameters.assert_called_once_with(values, False)

    def test_cpv_group_rpc_budget_accounts_for_sequential_firmware_acks(self) -> None:
        from unittest.mock import Mock
        connection = Mock()
        connection.recv.side_effect = [
            {"kind": "result", "id": 1, "result": {}},
            {"kind": "result", "id": 2, "result": {}},
        ]
        proxy = BackendProcessProxy(Mock(), connection)
        proxy.call("osc_set_cpv_parameters", {"cv_rad_s": 2, "acc_dcc_rad_s2": 10}, False)
        connection.poll.assert_called_with(60.0)
        proxy.call("osc_state")
        connection.poll.assert_called_with(30.0)


if __name__ == "__main__":
    unittest.main()
