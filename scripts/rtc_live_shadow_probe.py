"""Bounded RTC validation using the running console's cameras and shadow OSC.

Loads current inference code without restarting the hardware service. Images
come through its existing JPEG preview API (not raw camera buffers). All OSC
commands are tied to a newly created shadow session; gripper writes are denied.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.inference_config import prepare_update
from supervisor.pi05_adapter import Pi05InputAdapter


class Console:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.identity = None
        self.dispatches = []

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(self.url+path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise RuntimeError(str(result))
        return result["data"]

    def state(self):
        return self.call("/api/osc/state")

    def check(self, snapshot=None):
        session = (snapshot if snapshot is not None else self.state())["session"]
        if (session.get("state") != "ACTIVE" or session.get("execution_mode") != "shadow"
                or (session.get("id"), session.get("client_id")) != self.identity):
            raise RuntimeError("Shadow ownership lost; refusing commands")

    def heartbeat(self, client_id, session_id):
        result = self.call("/api/osc/session/heartbeat", {"client_id": client_id, "session_id": session_id})
        self.check(result["state"])
        return result

    def command(self, session_id, client_id, sequence, kind, payload):
        # Server validates the session identity and mode for each TCP target.
        self.check()
        began = time.perf_counter()
        result = self.call("/api/osc/command", {"session_id": session_id, "client_id": client_id,
                           "sequence": sequence, "type": kind, "payload": payload,
                           "acknowledgement_only": kind == "track_tcp"})
        if kind == "track_tcp":
            self.dispatches.append({"started_s": began, "finished_s": time.perf_counter(),
                                    "accepted": result.get("ok"), "target": payload["target_pose"]})
        return result

    def track_tcp(self, session_id, client_id, sequence, target):
        return self.command(session_id, client_id, sequence, "track_tcp", {"target_pose": target})

    def hold(self, session_id, client_id, sequence, reason):
        return self.command(session_id, client_id, sequence, "hold", {"reason": reason})

    def gripper(self, *args):
        raise RuntimeError("Live shadow probe forbids hardware gripper commands")


class PreviewCameras:
    def __init__(self, console):
        self.console = console
        self.lock = threading.RLock()
        self.frames, self.frame_times_ns = {}, {}
        self.frame_version = 0
        self.config = {}
        self.cameras = self
        self.dataset_frame_history, self.dataset_frame_sizes = {}, {}
        self.last_error = None
        self.poll_metrics = []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.produce, daemon=True)
        self.thread.start()

    def produce(self):
        while not self.done.is_set():
            try:
                began = time.perf_counter_ns()
                # Capture timestamps precede the corresponding JPEG read;
                # never relabel an old frame with its HTTP receipt time.
                status = self.console.call("/api/cameras/state")
                if not status["ready"]:
                    raise RuntimeError("Console cameras unavailable")
                frames, stamps = {}, {}
                for key in ("external", "wrist"):
                    stamps[key] = status["sources"][key]["timestamp_monotonic_ns"]
                    with urlopen(self.console.url+f"/api/cameras/frame/{key}.jpg", timeout=2) as response:
                        raw = response.read()
                    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None or bgr.shape != (224, 224, 3):
                        raise RuntimeError("Invalid preview image")
                    frames[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frames, self.frame_times_ns = frames, stamps
                    self.frame_version += 1
                    self.last_error = None
                    self.poll_metrics.append({"elapsed_ms": (time.perf_counter_ns()-began)/1e6,
                        "frame_age_ms": {k: (time.perf_counter_ns()-v)/1e6 for k,v in stamps.items()}})
            except Exception as exc:
                self.last_error = str(exc)
            self.done.wait(.025)

    def snapshot(self):
        with self.lock:
            return {"ready": len(self.frames) == 2, "frame_version": self.frame_version,
                    "last_error": self.last_error}

    def close(self):
        self.done.set()
        self.thread.join(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", default="http://127.0.0.1:8765")
    parser.add_argument("--model-port", type=int, default=18762)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--report", type=Path, default=ROOT/"runtime/rtc_live_shadow_report.json")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error("seconds must be 1..600")
    console = Console(args.console)
    baseline = console.state()
    existing = console.call("/api/pi05/state")
    if baseline["session"]["state"] != "IDLE" or existing.get("execution_enabled") or existing.get("priming"):
        raise RuntimeError("Console must be idle before starting independent shadow validation")
    if not console.call("/api/cameras/state")["ready"]:
        raise RuntimeError("Activate both cameras in the console first")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.report.parent/("rtc_live_"+time.strftime("%Y%m%dT%H%M%S"))
    config_path = run_dir/"config/pi05.json"
    config_path.parent.mkdir(parents=True)
    config = prepare_update(existing["config"], {"policy_endpoint": {"host": "127.0.0.1", "port": args.model_port},
                            "execution": {"mode": "rtc", "expected_chunk_steps": 20, "replan_steps": 6}})
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    camera = PreviewCameras(console)
    adapter = None
    report = {"passed": False, "console": args.console, "camera_source": "live JPEG84 previews via console HTTP",
              "state_source": "OSC simulated CPV feedback (not measured robot trajectory)",
              "logs": str(run_dir/"runtime/logs/inference_runs"), "samples": []}
    try:
        adapter = Pi05InputAdapter(console, config_path, camera)
        deadline = time.perf_counter()+45
        while time.perf_counter() < deadline:
            if adapter.snapshot()["model_state"] == "CONNECTED" and camera.snapshot()["ready"]:
                break
            time.sleep(.1)
        else:
            raise RuntimeError("Model or camera connection timed out")
        report["metadata"] = adapter.snapshot()["policy_metadata"]
        client = "rtc-shadow-"+uuid.uuid4().hex
        started_session = console.call("/api/osc/session/start", {"execution_mode": "shadow", "client_id": client, "input_source": "pi05"})
        session = started_session["session"]
        console.identity = (session["id"], client)
        console.check(started_session["state"])
        adapter.start(session["id"], client)
        started = time.perf_counter()
        while time.perf_counter()-started < args.seconds:
            console.heartbeat(client, session["id"])
            state = adapter.snapshot()
            sample = {k: copy.deepcopy(state.get(k)) for k in
                      ("state", "last_error", "rtc", "chunk_sequence", "executed_steps", "inference_ms")}
            sample["elapsed_s"] = time.perf_counter()-started
            report["samples"].append(sample)
            if state.get("last_error"):
                raise RuntimeError(state["last_error"])
            if len(report["samples"]) % 10 == 0:
                print(json.dumps(sample), flush=True)
            time.sleep(.1)
        report["passed"] = adapter.state["executed_steps"] >= 20
        report["elapsed_s"] = time.perf_counter()-started
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        if adapter:
            try:
                adapter.stop("Bounded live shadow validation finished")
                report["final_adapter"] = adapter.snapshot()
                report["passed"] &= bool(report["final_adapter"].get("rtc", {}).get("hold_confirmed"))
            except Exception as exc:
                report["cleanup_error"] = str(exc)
                report["passed"] = False
            adapter.close()
        if console.identity:
            final = console.state()
            report["final_osc"] = final
            report["hardware_cpv_dispatch_count"] = final["transport"]["cpv_dispatch_count"]
            report["hardware_mailbox_sent_delta"] = (final["transport"]["cpv_mailbox"]["sent_count"]
                        - baseline["transport"]["cpv_mailbox"]["sent_count"])
            report["passed"] &= report["hardware_mailbox_sent_delta"] == 0
            try:
                console.check(final)
                console.call("/api/osc/session/stop", {"reason": "Bounded shadow validation complete"})
            except Exception as exc:
                report["session_cleanup_error"] = str(exc)
                report["passed"] = False
        camera.close()
        report["camera_poll_metrics"] = camera.poll_metrics
        report["camera_last_error"] = camera.last_error
        intervals = np.diff([d["started_s"] for d in console.dispatches])*1000
        report["dispatches"] = console.dispatches
        report["dispatch_interval_ms"] = {"p50": float(np.percentile(intervals, 50)),
              "p95": float(np.percentile(intervals, 95)), "max": float(max(intervals))} if len(intervals) else None
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k not in ("samples", "dispatches", "final_adapter", "final_osc", "metadata", "camera_poll_metrics")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
