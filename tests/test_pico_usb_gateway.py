from __future__ import annotations

import json
import unittest

from scripts.nero_control_server import PicoGateway


class _Adapter:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def pico_begin_connection(self, session_id: str, client_id: str) -> None:
        self.events.append(("start", session_id, client_id))

    def pico_connected(self) -> None:
        self.events.append(("connected",))

    def pico_connection_lost(self, reason: str) -> None:
        self.events.append(("lost", reason))

    def pico_message(self, kind: str, payload: dict) -> dict:
        self.events.append(("message", kind, payload["_pico_sequence"]))
        return {"ok": True, "event": kind, "accepted": True}

    def pico_disconnected(self, reason: str) -> None:
        self.events.append(("disconnected", reason))

    def pico_state(self) -> dict:
        return {"state": "WAITING_FOR_USB"}


class _Runtime:
    def __init__(self) -> None:
        self.adapter = _Adapter()

    def require_adapters(self) -> _Adapter:
        return self.adapter


class _Server:
    def shutdown(self) -> None:
        pass


class _Connection:
    remote_address = ("127.0.0.1", 1)

    def __init__(self, messages: list[dict]) -> None:
        self.messages = [json.dumps(message) for message in messages]
        self.sent: list[dict] = []

    def recv(self, timeout: float | None = None) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise TimeoutError()

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _frame(sequence: int = 1) -> dict:
    return {"type": "input_frame", "sequence": sequence, "position_m": [0, 0, 0],
            "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True,
            "grip": False, "trigger_value": 0}


class PicoUsbGatewayTests(unittest.TestCase):
    def _gateway(self, runtime: _Runtime) -> PicoGateway:
        gateway = PicoGateway(runtime, {"host": "0.0.0.0", "port": 8768,
                                        "idle_timeout_s": 0.05, "message_timeout_s": 0.01})
        gateway._server = _Server()
        gateway.start_session("osc-1", "browser")
        return gateway

    def test_first_input_frame_is_accepted_without_pairing(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        connection = _Connection([_frame()])
        try:
            gateway._handle_connection(connection)
        finally:
            gateway.close()
        self.assertEqual(connection.sent[0]["type"], "connected")
        self.assertEqual(gateway._received_count, 1)
        self.assertIn(("message", "input_frame", 1), runtime.adapter.events)

    def test_pair_message_is_not_a_valid_first_message(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        connection = _Connection([{"type": "pair", "code": "123456"}])
        try:
            gateway._handle_connection(connection)
        finally:
            gateway.close()
        self.assertEqual(gateway._received_count, 0)
        self.assertTrue(any(item.get("ok") is False for item in connection.sent))

    def test_status_exposes_only_loopback_usb_endpoint(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        try:
            status = gateway.status()
        finally:
            gateway.close()
        self.assertEqual(status["host"], "127.0.0.1")
        self.assertEqual(status["transport"], "usb_adb")
        self.assertEqual(status["ws_url"], "ws://127.0.0.1:8768")
        self.assertNotIn("pair_code", status)
        self.assertNotIn("pairing_id", status)
