# 可选 OSC 阻抗输出

主仓库 Pink、绝对 TCP 输入、速度/加速度限制和最终安全门保持不变。
`motion/osc_impedance.py` 只将最终关节位置目标转换为 MIT 命令，
`nero_backend/osc_impedance.py` 负责 MIT 模式交接和发送校验。
MIT 与 CPV 共用 `HardwareTxOwner` 的容量一邮箱、epoch、generation 撤销、
发送结果和唯一 CAN 写入线程；没有第二条硬件发送链。

## 选择与启动

Web 上的“原控制 CPV / 阻抗控制”按钮只选择下次启动的输出。
也可 `POST /api/osc/output-mode`，请求为
`{"mode":"impedance","client_id":"当前客户端身份"}`；`mode` 也可为 `cpv`。
真机必须已确认 HOLD，独立关节反馈新鲜且停稳；IDLE 本身不代表 HOLD，
断线及 FREEDRIVE 不允许切换。影子会话允许 IDLE 或停稳 HOLD_READY。
运行、制动、故障、交接、活动任务/租约均拒绝。
已有会话被结束并撤销目标；切换不会自动运动或恢复旧目标。
前端按钮只调用 OSC 接口，是否可切换使用 `output_switch.allowed/reason`。
网页摇杆依据通用会话身份和权限 epoch 失效清除锚点，不判断输出模式；
选择当前模式不结束会话，切换拒绝也不主动重置摇杆。

选择原子保存到 `runtime/osc_output_mode.json`，首次默认 CPV。
配置损坏时回退 CPV，原因在状态的 `output_switch.selection_error` 中；
重新选择 CPV 可修复保存文件。重启恢复选择不会启动会话或发送硬件命令。
硬件退出或保存失败时不改变选择、不恢复旧目标。

状态新增 `output_mode`、`output_switch`、`impedance`。
`impedance.command` 为最新计算命令，实际发送结果仍在
`transport.cpv_mailbox.last_success`（保留兼容字段名）中，包含实际力矩和输出模式。
`impedance.last_mode_entry` 记录首帧实测锚点与新的模式反馈。

## LX 提取范围与假设

- 七关节 kp=3.5、kd=0.3、MIT v_des=0；Pink 速度仍用来形成连续位置目标。
- 重力映射、有效性检查、限幅、变化率限制来自 LX 的 `mit_dynamics.py`。
- 使用主仓库同一 URDF、关节约定和隔离 `.conda/nero-kinematics/python.exe`。
- LX 跟踪配置使用 bare_flange、倍率 1.3、alpha_ramp_s=5，忽略夹爪与负载质量。
  **这不是带夹爪/负载的精确动力学模型，不得将仿真当作负载安全证明。**
- 与 LX `first_frame_gravity=fresh` 一致，alpha 初始即为1.3；
  硬件首帧 dt=0，因此新鲜重力模型有效，但发送力矩为零；
  随后每关节以100 Nm/s进入目标，绝对限幅16 Nm（v120）。
  没有新增在线调参入口；alpha 不发生改变时无需倍率渐变。
- 新结果最大60 ms；允许至100 ms的短期缓存，关节误差分别不超过0.02/0.05 rad。
  不可用结果触发停止，不以静默零前馈继续运动。
- 硬件发送前及每关节发送时再检查反馈150 ms、重力100 ms、epoch与目标撤销。

未提取 LX 完整 OSC、静态 IK、手动托举、摩擦补偿、实验调参或控制链配置切换。
默认 CPV 不导入阻抗模块、不启动重力进程，原 TCP 请求格式保持兼容。

## HOLD / 退出

操作员 HOLD、结束会话、FREEDRIVE 和故障使用官方 Follower 退出 MIT，
要求新的非 MIT 模式反馈及稳定关节反馈；未确认退出时保持故障并禁止 CPV。
PICO 松 Grip 的输入 HOLD 保留会话及当前 MIT 位置锚点，只刷新有效重力；
这与操作员结束控制的 HOLD 区分，下一次输入不会复活旧目标。
停止先结束输出线程，再退出硬件模式，最后关闭重力进程。

## 验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe tests/osc_output_offline_acceptance.py
node --check web/console/app.js
```

离线验收运行真实 Pink 与隔离 Pinocchio，完成 CPV→MIT→CPV、X 2 mm往返、
收敛检查与 HOLD；不会创建硬件后端或连接现有服务。
MIT 使用明确标注的 PD 力矩替身，不是完整机械臂动力学。
该替身在零力矩首帧后的前馈爬升期间会发生启动偏移（首次测量约37 mm），
不能将此数字外推为实际硬件位移，也不能据此宣称实机无瞬态。

实机验收必须先确认现场可运动：CPV、阻抗、切回 CPV，
各沿基座 X 做2 mm往返并手动 HOLD，仅停稳后切换。
记录反馈模式、实测 TCP、关节、首帧、发送结果与力矩。
出现超范围位移、模式错误、反馈/重力过期或新故障立即 HOLD，
不增加位移范围或自行调参。尚未进行实机测试时不得声称全部验收通过。

### 本次记录（2026-10-08）

- 新增39项测试通过，覆盖输出、模式首帧、发送线程、双向选择、持久化、
  HTTP拒绝、反馈/重力过期、停止与目标撤销。
- 完整187项版本曾通过（21项跳过）；加入最后两项首帧反馈测试后，
  最新189项中有1项原有 Pi05 计时阈值偶发失败（21项跳过）。
  `test_slow_inference_does_not_burst_action_chunk_dispatches` 读到47 ms，
  阈值50 ms。另一轮原有15 ms阈值用例读到14.9999999994 ms，单独复测通过。
  未修改自动控制实现或其测试阈值，不能称最新全量回归无失败。
- 实际 Pink/Pinocchio 离线 CPV→阻抗→CPV 双向收敛及 HOLD_READY通过；
  两个CPV阶段均未启动阻抗运行组件；整个离线流程无硬件调用。
- JavaScript语法、Python编译和Git差异检查通过；Pink配置、PICO、
  自动控制实现、运行配置及运动学服务器无修改。
- MIT PD替身启动偏移约38 mm，属于需要关注的前馈爬升瞬态，
  不作为实际硬件预测，也不作为实机安全验收通过的证据。
- 未重启当前服务，未做实机运动。真实固件的 Follower退出是否会更新
  非MIT反馈仍待确认：SDK的 Follower请求本身不显式选择运动模式，
  如非MIT反馈不出现，本实现保持故障并阻止CPV，不猜测退出成功。

### HOLD 切换最小修复（2026-10-08）

- 操作顺序：手动 HOLD → 等待停稳和按钮可用 → 选择输出 → 连接输入。
  真机除硬件 HOLD 外还要求通用权限状态 HOLDING；不会将 OSC_CPV/OSC_MIT
  的 Follower角色误当作已经停止。影子会话结束后保留其执行上下文供 IDLE 选择。
- 模式按钮不直接重置摇杆；会话结束、身份或权限 epoch 变化才清除输入。
  请求代次和状态序列隔离旧响应；持久化新选择后推进状态序列，避免旧轮询覆盖。
- 阻抗准备允许综合状态缓存因预热变旧，但独立七关节 RX 必须新鲜，
  并通过原有硬件读取检查连接、急停与关节故障；首帧60 ms反馈要求保持不变。
- 相关49项 Python 回归通过（含下述前端套件入口），Node真实前端函数的
  模拟 DOM/HTTP交互12场景通过；不等同于真实浏览器或实机验收。
- 实际 Pink/隔离 Pinocchio的 CPV→阻抗→CPV、2 mm往返及 HOLD通过，
  CPV阶段未加载阻抗组件；动力学替身仍有约38 mm启动偏移，不作为实机预测。
- 最终全量199项：176项通过、21项跳过、2项原有 Pi05计时测试失败。
  推理耗时读到14.9999999994 ms（阈值15 ms），块派发间隔读到16 ms
  （阈值24 ms）；两项单独复测通过。未修改自动控制实现或放宽测试阈值。
- JavaScript语法、Python编译与Git差异检查通过。未重启运行服务、未发送实机运动命令；
  安全停止后重启控制服务并刷新页面，才能进行现场验证。
