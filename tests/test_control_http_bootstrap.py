from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from scripts.nero_control_server import (
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


if __name__ == "__main__":
    unittest.main()
