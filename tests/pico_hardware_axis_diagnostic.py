"""Bounded live PICO-to-NERO axis diagnostic with mandatory hardware opt-in.

It starts an isolated OSC hardware session, injects one 1-cm PICO axis step at
a time, waits for measured TCP feedback, returns to the anchor and finally
issues HOLD.  Use only in a clear, supervised robot workspace.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from websockets.sync.client import connect


CLIENT_ID = "pico-hardware-axis-diagnostic"
IDENTITY_Q = [0.0, 0.0, 0.0, 1.0]


def request(http: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    value = Request(f"{http}{path}", data=encoded, method="POST" if body is not None else "GET")
    if encoded is not None:
        value.add_header("Content-Type", "application/json")
    with urlopen(value, timeout=10) as response:
        payload = json.load(response)
    if not payload.get("ok"):
        raise RuntimeError(f"{path} failed: {payload}")
    return dict(payload["data"])


def vector(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise RuntimeError(f"invalid measured TCP position: {value!r}")
    return [float(item) for item in value]


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def norm(value: list[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def osc_state(http: str) -> dict[str, Any]:
    return request(http, "/api/osc/state")


def measured_position(http: str) -> list[float]:
    execution = osc_state(http).get("execution") or {}
    pose = execution.get("measured_tcp_pose") or execution.get("estimated_tcp_pose") or {}
    return vector(pose.get("position_m"))


def wait_for_target(http: str, target: list[float], timeout_s: float,
                    tolerance_m: float) -> tuple[list[float], float]:
    deadline = time.monotonic() + timeout_s
    last = measured_position(http)
    while time.monotonic() < deadline:
        last = measured_position(http)
        error = norm(subtract(last, target))
        if error <= tolerance_m:
            return last, error
        time.sleep(0.05)
    return last, norm(subtract(last, target))


def wait_for_anchor(http: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pico = request(http, "/api/adapters/pico/state")
        if bool(pico.get("anchor_active")) and pico.get("state") == "TRACKING":
            return
        time.sleep(0.05)
    raise RuntimeError("PICO Grip did not establish a TCP anchor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-hardware", action="store_true", help="required: permits a supervised hardware session")
    parser.add_argument("--http", default="http://127.0.0.1:8765")
    parser.add_argument("--ws", default="ws://127.0.0.1:8768")
    parser.add_argument("--distance-m", type=float, default=0.01)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.004,
                        help="maximum measured TCP target error; defaults to the runtime 4-mm tolerance")
    args = parser.parse_args()
    if not args.arm_hardware:
        parser.error("refusing live motion without --arm-hardware")
    if not 0.002 <= args.distance_m <= 0.02:
        parser.error("--distance-m must be 2 mm to 2 cm")
    if not 0.001 <= args.position_tolerance_m <= 0.006:
        parser.error("--position-tolerance-m must be 1 mm to 6 mm")

    started = False
    session_id = ""
    results: list[dict[str, Any]] = []
    try:
        before = osc_state(args.http)
        if (before.get("session") or {}).get("state") == "ACTIVE":
            raise RuntimeError("OSC session is already active; refuse to take over live robot control")
        started_response = request(args.http, "/api/osc/session/start", {
            "client_id": CLIENT_ID, "execution_mode": "hardware"})
        started = True
        session_id = str((started_response.get("session") or {}).get("id") or "")
        if not session_id:
            raise RuntimeError("hardware session did not return an id")
        request(args.http, "/api/adapters/pico/connect", {"session_id": session_id, "client_id": CLIENT_ID})
        pico = request(args.http, "/api/adapters/pico/state")
        ws_url = str((pico.get("gateway") or {}).get("ws_url") or args.ws)
        mapping = (pico.get("mapping") or {}).get("position_axis_map")
        if not isinstance(mapping, list) or len(mapping) != 3:
            raise RuntimeError("PICO position mapping is unavailable")

        with connect(ws_url, compression=None, open_timeout=5, close_timeout=2) as socket:
            sequence = 1
            def send(position: list[float], grip: bool = True) -> None:
                nonlocal sequence
                socket.send(json.dumps({"type": "input_frame", "sequence": sequence,
                                        "position_m": position, "orientation_xyzw": IDENTITY_Q,
                                        "tracking_valid": True, "grip": grip, "trigger_value": 0.0}))
                sequence += 1

            # The gateway intentionally waits for the first input before it
            # accepts the socket, so send it before awaiting its greeting.
            # Later position frames are asynchronous telemetry and have no
            # per-frame WebSocket acknowledgement; feedback is checked below.
            send([0.0, 0.0, 0.0])
            hello = json.loads(socket.recv(timeout=5))
            if not hello.get("ok") or hello.get("type") != "connected":
                raise RuntimeError(f"PICO gateway rejected diagnostic socket: {hello}")
            wait_for_anchor(args.http, 3.0)
            time.sleep(0.15)
            anchor = measured_position(args.http)
            for axis, label in enumerate(("X", "Y", "Z")):
                raw = [0.0, 0.0, 0.0]
                raw[axis] = args.distance_m
                expected_delta = [args.distance_m * float(mapping[row][axis]) for row in range(3)]
                expected_target = [anchor[index] + expected_delta[index] for index in range(3)]
                send(raw)
                measured, target_error = wait_for_target(
                    args.http, expected_target, args.timeout_s, args.position_tolerance_m)
                actual_delta = subtract(measured, anchor)
                expected_distance = norm(expected_delta)
                signed_distance = (sum(actual_delta[index] * expected_delta[index] for index in range(3)) /
                                   expected_distance if expected_distance > 0.0 else 0.0)
                direction_ok = signed_distance >= expected_distance - args.position_tolerance_m
                results.append({"axis": f"PICO +{label}", "raw_delta_m": raw,
                                "expected_nero_base_delta_m": expected_delta,
                                "measured_tcp_delta_m": actual_delta,
                                "measured_distance_along_expected_axis_m": signed_distance,
                                "target_error_m": target_error,
                                "pass": direction_ok and target_error <= args.position_tolerance_m})
                # Return to the same anchor before the next independent axis.
                send([0.0, 0.0, 0.0])
                returned, return_error = wait_for_target(
                    args.http, anchor, args.timeout_s, args.position_tolerance_m)
                if return_error > args.position_tolerance_m:
                    raise RuntimeError(f"TCP did not return to anchor after PICO +{label}: {returned}, error={return_error:.4f} m")
            send([0.0, 0.0, 0.0], grip=False)
        result = {"hardware": True, "distance_m": args.distance_m, "rows": results,
                  "pass": all(row["pass"] for row in results)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pass"] else 1
    finally:
        if started:
            try:
                request(args.http, "/api/adapters/pico/disconnect", {})
            except Exception as exc:
                print(f"WARNING: PICO gateway disconnect failed: {exc}", file=sys.stderr)
            try:
                request(args.http, "/api/osc/session/stop", {"reason": "PICO hardware axis diagnostic complete"})
            except Exception as exc:
                print(f"WARNING: automatic hardware HOLD failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
