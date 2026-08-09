# Local HTTP API

Base URL: `http://127.0.0.1:8765`

Success responses use `{"ok": true, "data": {}}`; errors use `{"ok": false, "error": "ErrorType: message"}`.

## Read endpoints

- `GET /api/health` — service health and action-job summary.
- `GET /api/status` — cached robot, gripper, lease, authority, teleoperation, and feedback readiness.
- `GET /api/actions/{id}` — asynchronous operator-action status.
- `GET /api/teleop/status` — current teleoperation session.
- `GET /api/broker/status` — writer and authority summary.
- `GET /api/teleop/kinematics` — solver and kinematics information.

## Motion lease and action endpoints

- `POST /api/lease/acquire` — `{"owner":"client","ttl_s":30}`.
- `POST /api/lease/renew` — `{"token":"...","ttl_s":30}`.
- `POST /api/lease/release` — `{"token":"..."}`.
- `POST /api/action` — `{"token":"...","action":{...},"timeout":25}`.

Supported actions include `joint_target`, `cartesian_pose`, `cartesian_delta`, `gripper`, and `stop`. Values remain subject to server-side safety checks.

## Operator and teleoperation endpoints

- `POST /api/actions`
- `POST /api/safety/hold`
- `POST /api/safety/freedrive`
- `POST /api/operator/gripper`
- `POST /api/teleop/session/start`
- `POST /api/teleop/session/heartbeat`
- `POST /api/teleop/intent`
- `POST /api/teleop/session/recenter`
- `POST /api/teleop/session/stop`
- `POST /api/teleop/handoff-to-console`

The API is intended for trusted localhost clients. It is not an authenticated remote-control API.

## Pose teleoperation

`POST /api/teleop/intent` uses clutch-scoped relative poses.  Send
`clutch_begin` first, keep the returned `anchor_id`, then send `pose` samples
with `relative_pose.position_m` and `relative_pose.orientation_xyzw`.  Send
`clutch_release` with the same anchor to request a Ruckig braking stop.
`tcp_velocity` is no longer accepted.

PICO devices connect only to the separately configured WebSocket gateway
(default `ws://<PC-LAN-IP>:8768`).  The local Follow Mode card creates the
PICO session and displays a short-lived pairing code; the APK cannot create a
hardware session on its own.
