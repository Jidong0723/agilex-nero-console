# OSC 影子链路测试说明

## 适用范围

此测试只验证完整的 OSC 影子 HTTP 链路：HTTP `track_tcp` → OSC → Pink → 最终安全门 → 模拟关节位置应用 → `/api/osc/state`。不打开 CAN、不发送真实 CPV、不驱动真机。

运行命令：

```powershell
.\.venv\Scripts\python.exe tests\osc_runtime_shadow_benchmark.py
```

## 测试轨迹

- 初始姿态：`config/teleop.json` 的 `shadow_initial_joints_rad`。
- 6D 同步目标：`X +50 mm`、`Y -40 mm`、`Z +40 mm`，`Roll +12°`、`Pitch -10°`、`Yaw +15°`。
- 目标线速度和角速度均为当前配置上限的 80%，采用 25% 加速、50% 匀速、25% 减速的梯形时间轨迹。
- 完成正向运动后，沿完全相反的轨迹返回初始 TCP 位姿。

## 阶段边界

1. **动态跟随阶段**：包括目标轨迹发送，以及轨迹结束后仍用同一 Pink 控制律等待实际到位的过程。该阶段持续计入跟随误差。
2. **到位条件**：位置误差 ≤ 0.5 mm、姿态误差 ≤ 0.25°、最大关节速度 ≤ 0.005 rad/s，连续满足 250 ms。
3. **保持阶段**：仅在实际到位后开始；正向目标与返回初始目标各保持 10 s。保持中持续运行正常 Pink 闭环，不切换 HOLD、不重置姿态参考、不重锚定。
4. **失败边界**：若在 `config/teleop.json` 的 `osc.arrival_timeout_s` 内未到位，本次测试失败；旧 epoch、旧目标代次、乱序或超时的 Pink 解不得用于输出。

## 指标与评分

令动态阶段的 TCP 位置/姿态 RMS 误差为 `Ep`、`ER`；每个保持阶段相对其开始时实际状态的最大 TCP 漂移为 `Dp`、`DR`；两个保持阶段、七个关节中的最大关节偏移为 `Dq`。

\[
S_{track}=\frac12\operatorname{clip}(1-Ep/10\text{ mm},0,1)+
\frac12\operatorname{clip}(1-ER/5^\circ,0,1)
\]

\[
S_{hold}=\min_{\text{两次保持}}\left[
\frac12\operatorname{clip}(1-Dp/3\text{ mm},0,1)+
\frac12\operatorname{clip}(1-DR/1.5^\circ,0,1)\right]
\]

\[
S_{null}=\operatorname{clip}(1-Dq/2^\circ,0,1)
\]

\[
\text{OSC Score}=100[0.60S_{track}^{-4}+0.25S_{hold}^{-4}+0.15S_{null}^{-4}]^{-1/2}
\]

验收要求：完整影子 HTTP 链路连续运行三次，三次中的**最低 OSC Score 严格大于 97**。同时，正常输出的每个关节加速度不得超过 `5 rad/s²`；安全故障停车单独记录，不计为正常控制输出。

## 结果解读

- 影子模式的 `execution.observed_source` 必须为模拟应用状态，`transport.participation` 为 `not_participating`。
- 所有 TCP 误差、关节状态和输出批次应来自同一 OSC 样本；不以 CAN/CPV 批次作为影子控制进度。
- 单次最高分不构成验收，必须报告三次最低分及各次的动态、保持、零空间子分数。
