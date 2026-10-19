"""No-motion normal/RTC WebSocket contract and end-to-end latency probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervisor.openpi_client import OpenPIClient
from supervisor.inference_contract import action_rows
from supervisor.inference_rtc import VERSION, compatibility


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18762)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--report", type=Path, default=Path("runtime/rtc_websocket_report.json"))
    p.add_argument("--check-errors", action="store_true")
    args = p.parse_args()
    client = OpenPIClient(args.host, args.port, 30)
    try:
        compatibility(client.metadata, {"expected_chunk_steps": 20, "period_s": .05})
        image = np.random.default_rng(11649).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        obs = {"observation/image": image, "observation/wrist_image": image.copy(),
               "observation/state": np.asarray([0, 0, .3, 0, 0, 0, .5], np.float32),
               "prompt": "Pick up the yellow cube."}
        times = {}
        for mode in ("normal", "rtc"):
            values = []
            for i in range(args.repeats):
                context = {"version": VERSION, "phase": "replan", "run_epoch": 1, "request_id": i+1,
                           "observation_perf_ns": i*350_000_000, "timeline_origin_perf_ns": 0,
                           "freeze_until_perf_ns": (i+1)*350_000_000, "action_row0_offset_s": 0,
                           "observation_tick": i*7, "plan_revision": i, "freeze_until": i*7+7,
                           "period_s": .05, "absolute_prefix": [[0, 0, .3, 0, 0, 0, 1, .5]]*13}
                began = time.perf_counter()
                response = client.infer(obs if mode == "normal" else dict(obs, rtc_context=context))
                values.append((time.perf_counter()-began)*1000)
                action_rows(response, 20)
                if mode == "rtc":
                    assert response["rtc"]["request_id"] == context["request_id"]
                    assert response["rtc"]["sampler"] == "rtc_vjp"
            times[mode] = {"samples_ms": values, "p50_ms": float(np.percentile(values, 50)),
                           "p95_ms": float(np.percentile(values, 95)), "max_ms": max(values)}
        checks = {}
        if args.check_errors:
            peer = OpenPIClient(args.host, args.port, 30)
            try:
                try:
                    peer.infer(obs)
                    raise AssertionError("Competing peer was not rejected")
                except RuntimeError as error:
                    assert "busy" in str(error)
                    checks["competing_peer_rejected"] = True
            finally:
                peer.close()
            client.close()
            time.sleep(.1)
            invalid_requests = {
                "state8_rejected": dict(obs, **{"observation/state": np.zeros(8, np.float32)}),
                "invalid_version_rejected": dict(obs, rtc_context=dict(context, version="unknown")),
                "invalid_prefix_rejected": dict(obs, rtc_context=dict(context, absolute_prefix=[[0]*8]*13)),
                "invalid_image_rejected": dict(obs, **{"observation/image": np.zeros((3, 224, 224), np.uint8)})}
            for name, request in invalid_requests.items():
                peer = OpenPIClient(args.host, args.port, 30)
                try:
                    try:
                        peer.infer(request)
                        raise AssertionError(f"{name} did not fail")
                    except RuntimeError:
                        checks[name] = True
                finally:
                    peer.close()
                time.sleep(.03)
            peer = OpenPIClient(args.host, args.port, 30)
            try:
                action_rows(peer.infer(obs), 20)
                checks["reconnect_after_rejection"] = True
            finally:
                peer.close()
        report = {"metadata": client.metadata, "latency": times, "checks": checks, "hardware_commands_sent": 0,
                  "input": "synthetic RGB/state; serialization+SSH+WebSocket+GPU+decode included"}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: {n: v for n, v in stats.items() if n != "samples_ms"} for k, stats in times.items()}))
    finally:
        client.close()


if __name__ == "__main__":
    main()
