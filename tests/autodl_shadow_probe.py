"""Live, no-hardware verification of the AutoDL OpenPI policy tunnel.

This sends synthetic but schema-correct NERO observations through the same
adapter used by the web console.  The OSC port is an in-memory Shadow double:
no CAN, robot, or physical gripper command is opened.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supervisor.pi05_adapter import Pi05InputAdapter


class _Camera:
    def read(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return image, image.copy()

    def close(self):
        pass


class _ShadowOsc:
    def __init__(self) -> None:
        self.commands: list[tuple[str, dict]] = []
        self.session = {"state": "ACTIVE", "id": "autodl-shadow-probe", "client_id": "autodl-shadow-probe", "execution_mode": "shadow"}

    def state(self):
        pose = {"position_m": [0.0, 0.0, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        return {"session": self.session, "command": {"target_tcp": pose}, "execution": {"measured_tcp_pose": pose},
                "workspace": {"min_xyz_m": [-0.6, -0.6, 0.0], "max_xyz_m": [0.6, 0.6, 0.7], "min_tcp_z_m": 0.0},
                "gripper": {"width_m": 0.02}}

    def heartbeat(self, *_args):
        return {"ok": True}

    def track_tcp(self, _session, _client, _sequence, target):
        self.commands.append(("track_tcp", target)); return {"ok": True}

    def gripper(self, _session, _client, _sequence, target):
        self.commands.append(("gripper", target)); return {"ok": True}

    def hold(self, _session, _client, _sequence, reason):
        self.commands.append(("hold", {"reason": reason})); return {"ok": True}


def main() -> int:
    osc = _ShadowOsc()
    adapter = Pi05InputAdapter(osc, PROJECT_ROOT / "config" / "pi05.json")
    adapter.cameras = _Camera()
    try:
        adapter.start("autodl-shadow-probe", "autodl-shadow-probe")
        deadline = time.monotonic() + 120.0
        while len([item for item in osc.commands if item[0] == "track_tcp"]) < 5 and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = adapter.snapshot()
        tracks = [item for item in osc.commands if item[0] == "track_tcp"]
        if len(tracks) != 5 or snapshot.get("action_chunk_length") != 10:
            raise RuntimeError(f"AutoDL shadow probe failed: tracks={len(tracks)}, state={snapshot.get('state')}, error={snapshot.get('last_error')}")
        actions = np.asarray(snapshot["action_chunk"], dtype=np.float64)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid AutoDL action chunk: shape={actions.shape}")
        print(f"ACTION_CHUNK={actions.shape}; EXECUTED={len(tracks)}; FIRST_TARGET={snapshot['decoded_first_target']}")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
