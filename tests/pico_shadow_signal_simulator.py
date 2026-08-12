"""Send a bounded 50 Hz PICO button + 6D-pose sequence to the real gateway."""
from __future__ import annotations

import argparse
import json
import math
import time
from urllib.request import urlopen

from websockets.sync.client import connect


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        payload = json.load(response)
    return payload["data"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", default="http://127.0.0.1:8765")
    parser.add_argument("--ws", default="ws://127.0.0.1:8768")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--ticks", type=int, default=200)
    args = parser.parse_args()

    pico = get_json(f"{args.http}/api/adapters/pico/state")
    gateway = pico.get("gateway") or {}
    token, session_id = gateway.get("auth_token"), gateway.get("session_id")
    if not token or not session_id:
        raise RuntimeError("start the PICO WebSocket receiver in Console before running the simulator")

    period = 1.0 / args.hz
    base = [0.0, 0.0, 0.0]
    errors: list[str] = []
    counts: dict[str, int] = {}
    started = time.monotonic()
    signal_started = started
    with connect(args.ws, compression=None, open_timeout=5, close_timeout=2) as socket:
        socket.send(json.dumps({"type": "auth", "session_id": session_id, "token": token}))
        authenticated = json.loads(socket.recv(timeout=5))
        if not authenticated.get("ok"):
            raise RuntimeError(f"authentication failed: {authenticated}")

        signal_started = time.monotonic()
        deadline = signal_started
        for tick in range(args.ticks):
            sequence = tick + 1
            phase = 2.0 * math.pi * tick / 100.0
            # Keep X fixed because the default NERO pose is already close to
            # the configured negative-X workspace boundary.
            position = [base[0], base[1] + 0.012 * math.sin(phase / 2.0), base[2] + 0.008 * math.cos(phase)]
            yaw = 0.10 * math.sin(phase)
            orientation = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
            if tick in {0, 100}:
                kind = "anchor_begin"
            elif tick == 99:
                kind = "anchor_release"  # right Grip released
            elif tick == args.ticks - 1:
                kind = "hold"  # left Menu safety button
            else:
                kind = "pose"
            message = {"type": kind, "session_id": session_id, "sequence": sequence,
                       "position_m": position, "orientation_xyzw": orientation,
                       "tracking_valid": True}
            socket.send(json.dumps(message))
            reply = json.loads(socket.recv(timeout=.25))
            counts[kind] = counts.get(kind, 0) + 1
            if not reply.get("ok"):
                errors.append(str(reply.get("error") or reply))
                break
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))

    finished = time.monotonic()
    elapsed = finished - started
    signal_elapsed = finished - signal_started
    sent = sum(counts.values())
    print(json.dumps({"ticks_requested": args.ticks, "ticks_sent": sent, "elapsed_s": elapsed,
                      "signal_elapsed_s": signal_elapsed, "measured_hz": sent / signal_elapsed, "counts": counts,
                      "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
