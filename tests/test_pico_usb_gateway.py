from __future__ import annotations

import json
import subprocess
import threading
import unittest
from unittest.mock import patch

from scripts.nero_control_server import PicoGateway, _unity_euler_zxy_degrees


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

    def close(self, *args, **kwargs) -> None:
        self.closed = True


class _BlockingConnection(_Connection):
    def __init__(self) -> None:
        super().__init__([_frame()])
        self.closed = False
        self.received_first_frame = threading.Event()

    def recv(self, timeout: float | None = None) -> str:
        if self.messages:
            value = super().recv(timeout)
            self.received_first_frame.set()
            return value
        while not self.closed:
            threading.Event().wait(0.005)
        raise ConnectionAbortedError("connection closed")


def _frame(sequence: int = 1) -> dict:
    return {"type": "input_frame", "sequence": sequence, "position_m": [0, 0, 0],
            "orientation_xyzw": [0, 0, 0, 1], "tracking_valid": True,
            "grip": False, "trigger_value": 0}


class PicoUsbGatewayTests(unittest.TestCase):
    def test_unity_zxy_euler_decomposition_uses_unity_display_order(self) -> None:
        # A pure X turn exposes the order unambiguously and should use the
        # headset's 0–360 degree display range, rather than an axis-angle.
        half = 2.0 ** -0.5
        self.assertEqual(_unity_euler_zxy_degrees([half, 0.0, 0.0, half]), [90.0, 0.0, 0.0])
        self.assertEqual(_unity_euler_zxy_degrees([0.0, 0.0, -half, half]), [0.0, 0.0, 270.0])

    def _gateway(self, runtime: _Runtime) -> PicoGateway:
        gateway = PicoGateway(runtime, {"host": "0.0.0.0", "port": 8768,
                                        "idle_timeout_s": 0.05, "message_timeout_s": 0.01,
                                        "adb_executable": "adb-not-installed-for-tests"})
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

    def test_unity_euler_diagnostics_are_retained_without_entering_control_payload(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        frame = _frame()
        frame["euler_degrees"] = [359.5, 1.25, 180.0]
        connection = _Connection([frame])
        try:
            gateway._handle_connection(connection)
            self.assertEqual(gateway.status()["input_frame_euler_degrees"], [359.5, 1.25, 180.0])
            self.assertEqual(gateway.status()["input_frame_unity_euler_from_quaternion_degrees"], [0.0, 0.0, 0.0])
        finally:
            gateway.close()

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

    def test_repeat_start_for_same_session_is_idempotent(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        try:
            gateway.start_session("osc-1", "browser")
            self.assertEqual([event for event in runtime.adapter.events if event[0] == "start"], [("start", "osc-1", "browser")])
        finally:
            gateway.close()

    def test_replacing_session_closes_old_socket_without_losing_new_session(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        connection = _BlockingConnection()
        worker = threading.Thread(target=gateway._handle_connection, args=(connection,))
        worker.start()
        self.assertTrue(connection.received_first_frame.wait(1.0))
        gateway.start_session("osc-2", "browser")
        worker.join(1.0)
        try:
            self.assertFalse(worker.is_alive())
            self.assertTrue(connection.closed)
            self.assertEqual(gateway.status()["session_id"], "osc-2")
            self.assertFalse(any(event[0] == "lost" for event in runtime.adapter.events))
        finally:
            gateway.close()

    def test_reconnect_usb_reports_success_and_device_errors(self) -> None:
        runtime = _Runtime()
        gateway = self._gateway(runtime)
        good = [
            subprocess.CompletedProcess(["adb", "devices"], 0, "List of devices attached\npico\tdevice\n", ""),
            subprocess.CompletedProcess(["adb", "reverse"], 0, "", ""),
            subprocess.CompletedProcess(["adb", "reverse", "--list"], 0, "pico tcp:8768 tcp:8768\n", ""),
        ]
        try:
            with patch("scripts.nero_control_server.subprocess.run", side_effect=good):
                result = gateway.reconnect_usb(source="test")
            self.assertTrue(result["ok"])
            self.assertEqual(gateway.status()["connection_stage"], "waiting_for_headset")
            with patch("scripts.nero_control_server.subprocess.run", return_value=subprocess.CompletedProcess(["adb", "devices"], 0, "List of devices attached\n", "")):
                result = gateway.reconnect_usb(source="test")
            self.assertFalse(result["ok"])
            self.assertIn("no authorized PICO", result["message"])
        finally:
            gateway.close()
