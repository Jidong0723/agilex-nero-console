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

## Operational Space Controller

`OperationalSpaceController` is the sole owner of the NERO CAN transport.
New clients should use these endpoints rather than the legacy teleoperation or
operator endpoints:

- `POST /api/osc/session/start`: `{"client_id":"...","execution_mode":"shadow|hardware"}`.
- `POST /api/osc/command`: command envelope with `session_id`, `client_id`,
  monotonically increasing `sequence`, `type`, and `payload`.
- `POST /api/osc/session/stop`: end the session through a safe HOLD handoff.
- `GET /api/osc/state`: unified robot, TCP target, solver, authority, gripper,
  feedback, and safety state.

Supported OSC command types are `track_tcp`, `move_tcp`, `hold`, `stop`,
`freedrive`, `gripper`, and compatibility-only `joint_target`. TCP commands
use an absolute base-frame pose:

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

OSC has no clutch, relative-pose, or anchor fields. Those are optional
client-side input-adapter concepts; adapters translate them into absolute TCP
targets before calling OSC.

`GET /api/osc/state` publishes `execution.current_tcp_pose`,
`command.target_tcp`, and `diagnostics.tcp_error`. The error is expressed in the base frame as
target-minus-current `position_vector_m`, its `position_norm_m`, and the
shortest quaternion `orientation_angle_rad`. The top-level `robot` field is
the canonical cached robot-feedback snapshot; clients should not read the
compatibility copy nested under `control.robot`.

The canonical snapshot is `nero.osc.v2`. Its `command` section reports the
target TCP, final gated joint target/velocity, sequence, epoch, and output
status. `execution` reports the mode, commanded and observed joints, observed
source, and a mode-independent final-output count. `diagnostics` contains the
complete Pink result (including the kinematic projection), Ruckig result,
TCP error, final safety-gate result, and timing. `transport` contains CAN
health, the latest hardware feedback, and CPV dispatch data; in shadow mode it
explicitly reports `not_participating` with no CPV dispatch.

The bundled WebAdapter is one such adapter. It is a browser-local component:
on connect or **重锚定**, it captures the current OSC TCP target as its local
reference; it converts joystick increments to absolute base-frame TCP poses;
and it sends only OSC `track_tcp` commands. Re-anchoring never calls a legacy
teleoperation endpoint and never changes OSC state by itself.

## Pose teleoperation

`POST /api/teleop/intent` uses clutch-scoped relative poses.  Send
`clutch_begin` first, keep the returned `anchor_id`, then send `pose` samples
with `relative_pose.position_m` and `relative_pose.orientation_xyzw`.  Send
`clutch_release` with the same anchor to request a Ruckig braking stop.
`tcp_velocity` is no longer accepted.

PICO devices connect only to the separately configured WebSocket gateway
(default `ws://<PC-LAN-IP>:8768`). That gateway remains a compatibility input
adapter. The Web console currently exposes WebAdapter only; a future PICO
adapter should translate its input to absolute TCP targets before calling OSC.
