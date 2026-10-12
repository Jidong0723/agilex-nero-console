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

- 模型推理（5090／AutoDL 共用 `pi05` 路由）：`GET /api/pi05/state`, `POST /api/pi05/config`, `POST /api/pi05/start`, and `POST /api/pi05/stop`.
  输入统一为双 RGB + `float32[7]` 状态，响应为有限数值 `H×7` 动作。配置请求可同时包含
  `inference_profile`、`policy_endpoint: {host, port}` 和
  `execution: {expected_chunk_steps: H, replan_steps: K}`；H/K 必须成组提交且为正整数，K≤H。
  输出开启／正在启动时拒绝连接与 H/K 修改。配置按预设持久化；状态中的
  `protocol_compatibility` 区分待响应验证、兼容和不兼容，`last_response_steps` 显示最近返回长度。
  `connections.service_entry` 独立报告服务入口，字段包含 `state`（`neutral/warn/ok/bad`）、
  `endpoint`、`subtitle`、`basis`（`pending/local_listener/tcp_connection`）、
  `checked_at`（Unix 秒或 null）与 `message`。证据十秒未刷新即待检测，检测失败报告未知。
  `connections.policy` 报告 WebSocket 握手及 metadata 接收状态，附 `stage`；
  连接成功与 7D/H 数据契约兼容独立表达。`connections.ssh_forward` 保留为入口的兼容投影。
  入口检测与实际连接在后台完成，切换配置时丢弃旧版本证据。
  模型的夹爪输出使用闭区间 `[0, 1]`；仅在配置的 `gripper.ratio_boundary_tolerance`
  内对端点浮点舍入归一化，超过该范围仍会拒绝整块并请求 HOLD。状态中的
  `gripper_ratio_diagnostic` 给出动作行、原始比例和处理结果。
- Cameras: `GET /api/cameras/state`, `GET /api/cameras/list`, `POST /api/cameras/config`, `POST /api/cameras/activate`, and `POST /api/cameras/deactivate`.
- PICO USB/ADB: `GET /api/adapters/pico/state`, `POST /api/adapters/pico/connect`, and `POST /api/adapters/pico/disconnect`. Run `scripts/pico_usb_connect.ps1` after connecting the headset; the APK connects to `ws://127.0.0.1:8768` through `adb reverse tcp:8768 tcp:8768`. The first WebSocket message is directly an `input_frame`; no pairing code, QR code, LAN address, or `pair` message is used. The computer-side service has no PICO SDK dependency.

PICO and model inference run in the HTTP process. PICO owns the USB-forwarded WebSocket
connection and device anchors,
camera and policy lifecycle, and can only read OSC state, renew a session, or
send absolute `track_tcp`, `hold`, and `gripper` commands.

### PICO input-frame coordinate contract

`input_frame.position_m` and `input_frame.orientation_xyzw` use the PICO
tracking frame (`+X` right, `+Y` up, `+Z` forward; quaternion order `xyzw`).
The adapter converts it to the NERO base frame with
`[[0, 0, -1], [1, 0, 0], [0, 1, 0]]`: PICO right maps to NERO `+Y`, up to
NERO `+Z`, and forward to NERO `-X`. The same frame conversion is applied to
position and orientation. A held Grip establishes a relative pose anchor, so
a position-only input frame keeps the commanded TCP orientation unchanged.

## Operator actions

`POST /api/actions`, `/api/safety/hold`, `/api/safety/freedrive`,
`/api/operator/gripper`, and `/api/operator/handoff-to-console` remain
operator endpoints. The removed `/api/teleop/*` API has no compatibility path.
