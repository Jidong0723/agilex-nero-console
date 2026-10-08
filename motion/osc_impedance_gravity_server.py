"""Small JSONL gravity worker entry point for diagnostics and deployment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion.osc_impedance_dynamics import GravityModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--model-revision", default="official-nero-2dc30fca68cbf4e04d1d0bc15c123d026380ece7")
    parser.add_argument("--joint-conventions-json", default="[]")
    parser.add_argument("--tool-profiles-json", default="{}")
    args = parser.parse_args()
    model = GravityModel(
        Path(args.urdf),
        conventions=json.loads(args.joint_conventions_json),
        tool_profiles=json.loads(args.tool_profiles_json),
        model_revision=args.model_revision,
    )
    sys.stdout.write(json.dumps({"ready": True, "nq": 7, "nv": 7}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("kind") == "self_test":
                response = model.self_test(int(request.get("samples", 32)))
            elif request.get("kind") == "gravity":
                response = model.compute_gravity(
                    request.get("q_actual_rad"),
                    tool_profile=str(request.get("tool_profile", "bare_flange")),
                    sample_id=int(request.get("sample_id", 0)),
                )
            else:
                response = {"ok": False, "error": "unknown request kind"}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
