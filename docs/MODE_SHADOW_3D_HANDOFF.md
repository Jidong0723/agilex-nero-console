# 真机模式、影子模式与三维反馈投影交接说明

本文说明当前项目中真机执行、影子执行和网页三维投影的代码结构、共享部分与独有部分。本文只描述现有代码，不定义新的运行逻辑。

## 1. 总体架构

```text
网页 / 摇杆 / PICO
       |
       v
TeleopController
  输入、会话、TCP 目标
  Pink / Pinocchio / Ruckig
  SafetySupervisor
  统一控制循环 _loop
       |
       +---------------------------+
       |                           |
       v                           v
  影子执行                     真机执行
  shadow_joints                真实反馈线程
  不发硬件命令                 Servo authority
                               CPV -> CAN
       |                           |
       +-------------+-------------+
                     v
          teleop.status / last_result
                     |
                     v
             Canvas 三维反馈投影
```

主要文件：

- `motion/teleop.py`：真机和影子的共同控制核心。
- `supervisor/control.py`：Broker、硬件权限、Transport Owner 和安全接管。
- `supervisor/authority.py`：写权限、control epoch 和硬件传输队列。
- `nero_backend/robot.py`：NERO SDK/CAN Backend。
- `scripts/nero_control_server.py`：HTTP、状态聚合和 PICO Gateway。
- `web/console/app.js`：网页输入、轮询、Canvas 绘制。
- `config/teleop.json`：solver、影子初始关节和 workspace 配置。

## 2. 模式模型

模式由三个字段共同决定：

```python
execution_mode  # shadow 或 hardware：是否操作真机
input_source    # joystick 或 pico：输入来源
mode            # 兼容旧接口的组合名称
```

定义在 `motion/teleop.py::normalize_session_selection()`。

| 模式 | execution_mode | input_source | 是否发真实硬件命令 |
|---|---|---|---|
| `shadow` | `shadow` | `joystick` | 否 |
| `joystick_hardware` | `hardware` | `joystick` | 是 |
| `pico_hardware` | `hardware` | `pico` | 是 |
| `pico_shadow` | `shadow` | `pico` | 否 |
| `hardware` | `hardware` | `joystick` | 是 |

结论：PICO 是输入源，不等于真机。`pico_hardware` 是 PICO 输入驱动真机，`pico_shadow` 是 PICO 输入驱动虚拟关节。

## 3. 两种模式共用的结构

### 3.1 会话生命周期

真机和影子都通过同一个 `TeleopController`，共享：

```python
start_session()
stop_session()
heartbeat()
recenter()
submit_intent()
```

共同启动步骤：

1. 标准化模式字段。
2. 生成 `session_id`，检查活动会话和 client ownership。
3. 进入 `STARTING`。
4. 配置关节限制和 `SafetySupervisor`。
5. 启动 Pink/Pinocchio solver。
6. 初始化 Ruckig。
7. 建立 `ACTIVE` 会话并启动统一 `_loop()`。

共同停止步骤：失效输入、处理制动、停止控制线程、关闭 solver、清理 session/intent/anchor。

### 3.2 输入和目标位姿

两种模式共用：

- 键盘、网页摇杆和 PICO 输入解析。
- `clutch_begin`、`pose`、`clutch_release`。
- anchor、reference revision 和 session sequence。
- 相对位置、四元数和 TCP reference pose。
- deadman、输入超时、workspace 检查。

`submit_intent()` 负责更新目标状态，不直接向 CAN 发送速度。

### 3.3 共同的计算和安全链

```text
reference_pose
  -> Pink / Pinocchio IK
  -> 输入滤波
  -> SafetySupervisor.limit_velocity()
  -> Ruckig
  -> SafetySupervisor.final_gate()
  -> final_velocity
```

两种模式均使用：

- URDF、TCP offset、FK。
- Pink 关节速度求解。
- Ruckig 速度/加速度轨迹。
- 关节限制、workspace、flange 高度限制。
- 延迟预算和最终安全门。
- `HOLD_READY`、`RUNNING`、`BRAKING`、`FAULT` 状态。

### 3.4 共同的控制循环逻辑

`TeleopController._loop()` 负责：

1. 读取当前会话和 intent。
2. 选择当前关节状态 `q` 和速度 `qd`。
3. 检查输入、deadman、轨迹和反馈条件。
4. 提交 Pink 请求并轮询最新结果。
5. 计算 Ruckig 轨迹。
6. 执行最终安全门。
7. 根据模式选择“更新虚拟状态”或“发送 CPV”。
8. 更新诊断、轨迹和 `last_result`。

## 4. 真机模式独有结构

真机由 `execution_mode == "hardware"` 触发。

### 4.1 硬件预检和权限交接

真机启动会执行：

```python
broker._require_operational_control()
authority = self._hardware_preflight()
broker.prepare_teleop_hardware()
```

其中 `_hardware_preflight()` 读取真实关节限制并初始化限制交集。`prepare_teleop_hardware()` 负责退出 FREEDRIVE、进入 Follower/Servo、推进 epoch 和建立写权限。

之后还要执行：

```python
broker.grant_teleop_tracking(session_id, epoch)
```

### 4.2 真实反馈线程

真机启动：

```python
_start_feedback_worker()
_wait_for_feedback()
```

反馈链路：

```text
TeleopController
  -> RobotControlBroker.read_teleop_feedback()
  -> HardwareTransportOwner
  -> NeroRobot.read_teleop_feedback()
  -> SDK joint/motor feedback
```

控制循环使用真实：

```python
q = feedback.joints
qd = feedback.velocities
feedback_age = now - feedback.timestamp
```

真机独有保护：

- 反馈轻微过期时降速。
- 反馈严重过期且正在运动时进入 FAULT。
- feedback 不足 7 轴时停止输出。
- 每次写入前检查当前 SERVO authority 和 epoch。

### 4.3 真机输出

最终速度通过：

```python
broker.send_servo_velocity(final_velocity, session_id, epoch)
```

完整链路：

```text
TeleopController
  -> RobotControlBroker
  -> HardwareTransportOwner
  -> TransportRobotProxy
  -> NeroRobot.send_cpv_velocity()
  -> pyAgxArm SDK
  -> CAN / CPV
```

真机故障会进入：

```python
broker.trigger_safety_fault(reason)
```

然后撤销普通写权限、推进 epoch、发送 7 轴零速度并保持 SAFETY/HOLDING。

## 5. 影子模式独有结构

影子由 `execution_mode == "shadow"` 触发。

### 5.1 不访问硬件

影子不执行：

```python
broker._require_operational_control()
_hardware_preflight()
broker.prepare_teleop_hardware()
_start_feedback_worker()
_wait_for_feedback()
broker.grant_teleop_tracking()
```

初始状态来自 `config/teleop.json`：

```json
"shadow_initial_joints_rad": [-0.02, 0.5, -0.8, 1.2, 0.0, 0.2, 0.0]
```

### 5.2 虚拟关节状态

影子循环使用：

```python
q = self.shadow_joints
qd = self.trajectory["velocity_rad_s"]
feedback_age = 0.0
```

经过同样的 Pink、安全门和 Ruckig 后，不调用 `send_servo_velocity()`，而是使用最终安全速度积分：

```python
self.shadow_joints = [
    current + command * actual_dt
]
```

因此影子反映的是“安全门之后的虚拟运动”，不是未经限制的输入回放。

### 5.3 影子故障不操作真实机器人

`_fault_zero(reason, shadow)` 中只有 `shadow == False` 才调用 `broker.trigger_safety_fault()`。影子模式只将虚拟轨迹置为 FAULT、清零虚拟输出并记录：

```python
robot_commands_sent=False
```

不会发 CAN、切换机器人控制模式、申请真实 Servo authority 或发送 P0 零速度。

## 6. 三维反馈投影结构

三维投影是网页 Canvas 可视化，不是独立的机器人反馈模块。

### 6.1 网页状态数据流

`web/console/app.js::refresh()` 每 500 ms 并行请求：

```text
GET /api/status
GET /api/teleop/status
GET /api/broker/status
```

结果写入：

```javascript
state.status
state.teleop
state.broker
```

之后执行：

```javascript
render()
drawWorkspace(teleop)
```

### 6.2 Canvas 的实际输入

机械臂链条和 TCP 主要读取：

```javascript
teleop.last_result.solver.tcp
```

使用字段：

```javascript
tcp.position_m
tcp.rotation
tcp.link_positions_m
```

目标 TCP 来自：

```javascript
teleop.reference_pose
```

工作空间来自：

```javascript
teleop.workspace
```

页面下方的文字 TCP 姿态则读取：

```javascript
control.robot.tcp_pose
```

因此要区分：

1. `control.robot.tcp_pose`：状态文字显示。
2. `teleop.last_result.solver.tcp`：Canvas 机械臂链条、TCP 和姿态轴。

### 6.3 真机投影链路

```text
NERO CAN 真实关节反馈
  -> feedback worker
  -> TeleopController._loop(q = real feedback joints)
  -> Pink/Pinocchio FK
  -> last_result.solver.tcp
  -> app.js::drawWorkspace()
```

真机 Canvas 是根据最新真实关节反馈重新做 URDF FK 的投影，不是直接绘制 SDK 提供的 link 坐标。文本 TCP 状态则来自 `NeroRobot.read_state()` 的硬件状态路径。

### 6.4 影子投影链路

```text
shadow_initial_joints_rad
  -> shadow_joints
  -> TeleopController._loop(q = shadow_joints)
  -> Pink/Pinocchio FK
  -> last_result.solver.tcp
  -> app.js::drawWorkspace()
```

影子 Canvas 使用虚拟关节积分状态，不依赖 USB-CAN，也不依赖真实反馈；但仍使用相同 URDF、TCP offset、Pink/Pinocchio FK 和前端绘制代码。

### 6.5 Canvas 绘制步骤

`web/console/app.js::drawWorkspace()` 绘制：

1. workspace 底面。
2. workspace 网格。
3. X/Y/Z 坐标轴。
4. 最低 flange 高度平面。
5. `link_positions_m` 连杆折线。
6. 实际 TCP 点和姿态轴。
7. `reference_pose` 的目标 TCP `T_ref`。
8. 实际 TCP 到目标 TCP 的虚线。
9. 目标姿态轴。

它是二维 Canvas 轴测投影，不是 WebGL，也不是物理仿真。核心映射为：

```javascript
x_screen = center_x + (x - y) * scale * 0.72
y_screen = base_y - (z - minZ) * scale * 0.88 - (x + y) * scale * 0.28
```

## 7. 共享与独有对照

| 结构 | 真机 | 影子 | 三维投影 |
|---|---:|---:|---:|
| 会话、client ownership、heartbeat | 共享 | 共享 | 读取显示 |
| 输入、clutch、deadman、TCP 目标 | 共享 | 共享 | 目标姿态显示 |
| Pink / Pinocchio / FK | 共享 | 共享 | 提供 TCP/link 数据 |
| Ruckig | 共享 | 共享 | 结果间接显示 |
| SafetySupervisor | 共享 | 共享 | 显示 gate/trajectory |
| 真实 CAN 反馈 | 独有 | 不使用 | 文字状态可用 |
| `shadow_joints` 积分 | 不作为主状态 | 独有 | 影子投影输入 |
| Servo authority / epoch | 独有 | 不需要真实写权限 | 只显示状态 |
| CPV/CAN 输出 | 独有 | 不发送 | 不直接绘图 |
| P0 safety fault | 独有 | 不触发硬件动作 | 只显示 FAULT |
| Canvas 几何投影算法 | 不区分 | 不区分 | 前端独有 |

## 8. 关键边界和交接注意事项

1. 判断执行目标时优先看 `execution_mode`，不要只看旧的 `mode` 字符串。
2. 判断输入来源时看 `input_source`；PICO 不自动代表真机。
3. 判断是否真实写入时看 `send_servo_velocity()` 和 `robot_commands_sent`。
4. 影子模式仍然运行 Pink、Ruckig 和 SafetySupervisor，不是简单的输入回放。
5. Canvas 的机械臂链条主要来自 `teleop.last_result.solver.tcp`，不是直接来自 `control.robot.tcp_pose`。
6. 真机文本 TCP 和 Canvas TCP 经过不同数据路径，短时间内可能存在采样延迟差异。
7. 真机进入 FAULT 会触发 Broker 的 P0 安全路径；影子只冻结虚拟轨迹。

## 9. 推荐阅读顺序

```text
motion/teleop.py::normalize_session_selection()
motion/teleop.py::TeleopController.start_session()
motion/teleop.py::TeleopController._loop()
motion/teleop.py::TeleopController._start_feedback_worker()
supervisor/control.py::RobotControlBroker.prepare_teleop_hardware()
supervisor/control.py::RobotControlBroker.send_servo_velocity()
nero_backend/robot.py::read_state()
nero_backend/robot.py::read_teleop_feedback()
motion/teleop.py::KinematicsClient
web/console/app.js::refresh()
web/console/app.js::render()
web/console/app.js::drawWorkspace()
```

最关键的模式分叉是：

```python
if execution_mode == "shadow":
if not shadow:
if shadow:
```
