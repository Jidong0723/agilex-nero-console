# Local HTTP API

Base URL: `http://127.0.0.1:8765`. Success responses use `{"ok": true, "data": {}}`; errors use `{"ok": false, "error": "ErrorType: message"}`.

## OSC

- `POST /api/osc/session/start`: `{"client_id":"...","execution_mode":"shadow|hardware"}`.
- `POST /api/osc/session/heartbeat`: renew the session ownership lease.
- `POST /api/osc/session/stop`: end the session through a safe HOLD handoff.
- `POST /api/osc/command`: command envelope with session, client, monotonic sequence, type, and payload.
- `GET /api/osc/state`: canonical `nero.osc.v2` state snapshot.
- `GET /api/osc/kinematics`: solver and task-point diagnostics.

Supported command types are `track_tcp`, `move_tcp`, `hold`, `stop`,
`freedrive`, and `gripper`. TCP commands always contain an
absolute base-frame pose:

```json
{
  "session_id": "...",
  "client_id": "...",
  "sequence": 1,
  "type": "track_tcp",
  "payload": {
    "target_pose": {
      "position_m": [0.20, 0.10, 0.30],
      "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
    }
  }
}
```

OSC has no clutch, relative-pose, anchor, or input-source fields. Session
heartbeats are its only ownership lease; input adapters translate local state
to absolute TCP targets before calling OSC.

## Input adapters

- π0.5: `GET /api/pi05/state`, `POST /api/pi05/config`, `POST /api/pi05/start`, and `POST /api/pi05/stop`.
- Cameras: `GET /api/cameras/state`, `GET /api/cameras/list`, `POST /api/cameras/config`, `POST /api/cameras/activate`, and `POST /api/cameras/deactivate`.
- PICO: `GET /api/adapters/pico/state`, `POST /api/adapters/pico/connect`, `GET /api/adapters/pico/resolve`, and `POST /api/adapters/pico/disconnect`. Connecting creates a six-digit code valid for 120 seconds. The PICO APK resolves the code to a one-time `ws://<PC-IP>:8768/pair/<pairing_id>` endpoint, then sends a `{"type":"pair", "code":"...", "pairing_id":"...", "gateway_url":"..."}` first message. The computer-side service has no PICO SDK dependency.

PICO and π0.5 run in the HTTP process. PICO owns the authenticated WebSocket
connection and device anchors,
camera and policy lifecycle, and can only read OSC state, renew a session, or
send absolute `track_tcp`, `hold`, and `gripper` commands.

## Operator actions

`POST /api/actions`, `/api/safety/hold`, `/api/safety/freedrive`,
`/api/operator/gripper`, and `/api/operator/handoff-to-console` remain
operator endpoints. The removed `/api/teleop/*` API has no compatibility path.
