"""Wall-clock RTC soak with fake OSC/cameras; no robot backend is imported."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_inference_rtc import RuntimeTests
from supervisor.openpi_client import OpenPIClient


class RemotePolicy:
    def __init__(self, port):
        self.client = OpenPIClient("127.0.0.1", port, 30)
        self.metadata = self.client.metadata
        self.requests = []

    def infer(self, observation):
        self.requests.append(observation["rtc_context"]["request_id"])
        return self.client.infer(observation)

    def close(self):
        self.client.close()

    def is_alive(self):
        return self.client.is_alive()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--report", type=Path, default=Path("runtime/rtc_shadow_report.json"))
    p.add_argument("--model-port", type=int, help="Use real remote model via an existing tunnel; OSC/cameras remain fake")
    args = p.parse_args()
    fixture = RuntimeTests()
    fixture.setUp()
    if args.model_port:
        fixture.policy = RemotePolicy(args.model_port)
        fixture.a._policy = fixture.policy
        fixture.a.state["protocol_compatibility"] = fixture.a._metadata_status(fixture.policy.metadata)
    started = time.perf_counter()
    error = None
    try:
        fixture.a.start("session-1", "client-1")
        while time.perf_counter()-started < args.seconds:
            if fixture.a.state.get("last_error"):
                raise RuntimeError(fixture.a.state["last_error"])
            time.sleep(.05)
    except Exception as exc:
        error = str(exc)
    finally:
        fixture.a.stop("No-motion soak complete")
        targets = [c for c in fixture.broker.commands if c["type"] == "track_tcp"]
        intervals = [1000*(b["started_at"]-a["started_at"]) for a, b in zip(targets, targets[1:])]
        report = {"elapsed_s": time.perf_counter()-started, "dispatches": len(targets),
                  "requests": len(fixture.policy.requests), "min_interval_ms": min(intervals, default=None),
                  "max_interval_ms": max(intervals, default=None), "hold_confirmed": fixture.a.state.get("rtc", {}).get("hold_confirmed", False),
                  "passed": error is None, "error": error,
                  "hardware_commands_sent": 0, "backend": "fake OSC/cameras; real AutoDL inference" if args.model_port else "fake OSC/cameras, 150 ms fake inference"}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report))
        fixture.tearDown()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
