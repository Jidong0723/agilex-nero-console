# NERO H20 推理式 RTC

普通模式与 RTC 共用 WebSocket 协议、单份 π0.5 模型参数和原有 OSC 输出接口。RTC 只扩展模型推理控制源；没有修改 OSC 求解器、伺服循环、公共命令协议或硬件参数。

## 自动预览与推理控制

2026-10-19 TCP反馈刷新修复：模型模块新增独立20 ms实测关节/FK生产线程，与完整状态、夹爪读取和日志解耦；保留原始时间戳和100 ms阈值。状态接口 `measured_tcp_feedback` 提供样本年龄、缓存年龄、实际刷新耗时/间隔及错误。已受控重载，真实双相机/模型预览60秒只读采样最大TCP年龄49.9384 ms、超过100 ms次数0。两次最多2秒的真机启动均被独立的HOLD冷启动衔接保护拦截、CPV增量0，未完成2秒运动验收，不能将只读长测当成真机控制通过。详见 `runtime/diagnostics/rtc_feedback_fix_20261019.md`。此前待加载的首D步精确保留HOLD补丁随本次重启加载。

连接和实测观测就绪后，普通和RTC模式都自动生成动作预览。无需控制会话，也无需点击按钮。每次响应后等待50 ms，模型请求串行处理；实际刷新频率由模型和网络耗时决定，并非20 Hz推理承诺。RTC空闲预览使用无`rtc_context`的普通H20采样，页面明确标记“未进行RTC衔接”。

“开始推理控制”会撤销在途预览、使旧响应失效，重新采集观测并生成可执行首块。只有控制阶段使用真实队列进行RTC引导。“停止推理控制”确认HOLD后恢复预览；故障原因保留，恢复预览不会恢复控制。预览不会发送TCP、夹爪、HOLD或续租命令。断线、缺少观测或非法动作只影响预览状态；上一块保留并标记过期。

观测读取复用现有只读关节缓存和FK实现，独立于OSC执行样本；避免HOLD冻结的TCP时间戳。相机直接读取共享原始RGB，使用同一时钟比较新鲜度；缺少实测TCP／夹爪时等待，不填占位值。控制台相机映射已保存为外部0／腕部1。状态接口新增`preview`、`chunk_source`、`chunk_updated_at`、`chunk_stale`及分影子／真机的`control_readiness`，原接口和配置投影保留。

真机RTC证据核验结果见 [RTC_TRAINING_AUDIT.md](RTC_TRAINING_AUDIT.md)。按用户后续要求，训练契约未核验仅作信息提示，不再阻止手动启动真机RTC。按钮仍依赖实际观测、动作协议、配置和停止确认状态；绿色入口及WebSocket卡片只表达连接。动作缩放已从界面、配置和解码链路删除，按模型物理单位原值解码；旧配置中的缩放字段只在迁移时丢弃。“段末到位等待上限（ms）”只用于普通模式段末等待。

2026-10-18原始RGB验收：`runtime/inference_native_preview_20261018.json` 的60秒预览通过，动作块从175更新至356，控制会话始终IDLE，硬件CPV邮箱增量0。`runtime/inference_native_shadow_supervised_20261018.json` 监督60秒：本次执行100个影子目标后，模型第6行动作超出工作空间，正确确认HOLD并停止；剩余观察期间继续预览，执行计数不再增加，硬件CPV邮箱增量0。较早的一轮在92个目标后同样越界。报告保留失败，不能宣称RTC任务稳定通过；未放宽工作空间或自动开始真机动作。

最终观测读取版本再次通过60秒预览，报告为`runtime/inference_native_preview_final_20261018.json`，控制步数0、硬件邮箱增量0，预览约3～4.6 Hz。118项推理／RTC／夹爪／日志回归通过；控制服务和模式相关65项检查中64项通过、1项依赖条件不满足跳过。后续增加的未确认HOLD不恢复预览专项在8项预览测试中通过。12项前端测试及Edge模拟后端桌面／窄屏检查通过，确认按钮最终名称未被旧渲染覆盖。

原生路径可重复验证（要求现有会话空闲、双相机已连接）：

```powershell
.venv\Scripts\python.exe scripts/inference_console_acceptance.py --mode preview --seconds 60 --report runtime/preview_acceptance.json
.venv\Scripts\python.exe scripts/inference_console_acceptance.py --mode shadow --seconds 60 --report runtime/shadow_acceptance.json
```

## 控制故障恢复与只读预览

控制故障不再终止普通推理工作线程。RTC旧请求退出后，即使HOLD尚未确认，仍允许无控制命令的普通动作预览；控制恢复和预览各自维护状态。停止确认仍是再次控制的硬条件，绝不因预览恢复而自动执行动作或绕过HOLD。

故障时右侧原停止按钮改为“重试停止”（HOLD未确认）或“退出异常”（已停止）。后者清除当前故障提示，将原故障保存在技术详情的上次控制故障与原运行日志中，恢复待机预览；左侧按钮在观测、协议和停止条件满足后显示“重新开始推理控制”。旧请求退出中、图像／实测TCP缺失或过期、协议不兼容时，明确显示原因。仅WebSocket连接正常不代表存在可用的新观测或合法动作。

预览保留有限H×7模型原始输出，包括越界夹爪比例；可视化解码的夹爪仅钳制显示，原始行不改写。超出实际控制容差时显示“仅展示，不执行”并禁止启动。非有限／非法形状响应依然不能伪装成有效实时块，保留上一块并标注过期。所有预览与恢复测试均要求零新增运动命令；退出异常不会开始控制，也不会清除OSC／硬件安全FAULT。

## 当前部署

- 当前克隆实例：`ssh -p 43198 root@connect.nmb2.seetacloud.com`
- 服务目录：`/root/autodl-tmp/nero_rtc_v1`
- 远端模型端口：`127.0.0.1:8002`
- 权重：`/root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649/`
- assets：以上 checkpoint 内的 `assets/nerovla/nero_chunk7_v1/norm_stats.json`，使用分位数归一化。
- OpenPI 基准：`215abfb217dbac7d5f1273282331b9b1866c0479`，克隆实例有已有未提交改动；部署清单记录所用文件摘要，不能仅凭 Git SHA 认定一致。

现有 OpenPI 源码和旧服务入口均保留。新入口直接加载 JAX checkpoint，显式设置 H20、内部维度32、有效维度7、`pi05=True`、`discrete_state_input=False`。没有使用官方 RTC BC 权重或重新训练。

## 两窗口使用

窗口 A（本机 PowerShell，保持打开）：

```powershell
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 `
  -N -L 127.0.0.1:18762:127.0.0.1:8002 `
  -p 47642 root@connect.nmb2.seetacloud.com
```

窗口 B 登录远端：

```powershell
ssh -p 47642 root@connect.nmb2.seetacloud.com
```

服务未运行时，在远端 Bash 执行：

```bash
bash /root/autodl-tmp/nero_rtc_v1/start.sh
```

启动脚本设置 PATH、BOTO_CONFIG 和 OpenPI Python 路径，使用 `uv run --frozen --no-sync`。正常与 RTC 采样都预热，并执行无运动验证；通过后才监听8002。已有服务运行时不要重复启动。`service.log`、`deployment_manifest.json`、`validation_report.json` 位于服务目录。页面停止输出会释放模型推理运行，不会关闭远端模型或用户自己的 SSH 转发。

控制台“模型推理”保存 `127.0.0.1:18762`、H=20，选择“普通”或“RTC”。普通模式仍执行前 K 行；RTC 使用完整 H20 时间轴、固定50 ms，显式设置 `inference_delay` 和 `execute_horizon`。两种模式均按原始物理动作解码，不提供动作缩放。保存设置不会开始运动。

本地已保存的普通 H10/K5 配置没有被自动覆盖；重启本地服务后才会加载本次代码。本轮未重启本地真机服务。

## RTC 时间轴与契约

原请求四字段保持不变：双 `224×224×3 RGB uint8`、`float32[7]` 状态和 prompt。动作始终是物理单位的 `20×7`：每行相对同一次观测起点，位置增量位于基坐标系，`R目标=Exp(Δr)·R观测`，夹爪为绝对比例；行间不累加。

RTC 额外携带 `rtc_context`，版本升级为 `nero.rtc.v2`。metadata 必须声明 `action_time_contract=observation_time_plus_row_index_times_period`、`action_row0_offset_s=0`。指定11649权重的时间语义由用户于2026-10-18确认：第i行目标时间严格为 `t_obs+i×50 ms`，末行在 `t_obs+950 ms`，不是1000 ms。普通请求仍不携带上下文，四字段协议不变。

上下文包含运行／请求／计划编号、`observation_perf_ns`、`timeline_origin_perf_ns`、`freeze_until_perf_ns`、相应网格索引和剩余绝对目标。这些是本机同一单调时钟中的数值，云端只验证算术关系并回显，绝不与云端时钟相减。`t_obs` 取实测TCP样本的统一时基时间，不能取缓存读取或网络返回时刻。双图仍按既有新鲜度／偏差门限检查，保留各自采集时间；门限内异步观测不等于图像与TCP严格同刻采样，不能以此宣称训练采集同步已核验。

固定发送网格以首次 `t_obs` 为原点。新观测一般不恰好落在网格上，因此旧绝对目标按新块每行的真实时间重采样后才作为云端前缀；位置与夹爪线性插值，姿态使用最短弧四元数插值。云端随后相对本次观测起点编码并归一化。冻结行数按 `ceil((freeze_time-t_obs)/50ms)` 计算，不再以取整后的观测槽号替代时间差。

响应以观测起点解码后，在原有发送网格上对新轨迹重采样；替换起点取冻结边界、已承诺／已执行进度、当前时刻的下一个未来网格点的最大值。保留之前的旧轨迹，不外推超过新块第19行的时间。过期新行永不重排到返回时刻；发送线程等锁后也复查时间，错过槽位则HOLD，不追赶式补发。拼接检查包括前一个已执行目标，不能只检查新块内部。

首次启动发送`phase=bootstrap_hold`及实测HOLD构成的20行静止参考，冻结前350ms，云端从首块即进行RTC VJP引导。metadata必须声明`rtc_bootstrap_mode=hold_prefix_v1`，响应sampler必须为`bootstrap_hold_vjp`，不接受普通采样静默回退。旧的空前缀bootstrap和无上下文普通采样在云端仍可兼容，但新客户端不使用空前缀启动。

首块返回后仍复查起点漂移并丢弃过期行；比如恰好300ms返回，最早候选仍为第7行350ms，不重排第0行。若候选距HOLD超过2mm／0.5°，在原时间网格内规划独立五次平滑衔接段，终点为仍未过期的模型原始行；之后的模型行及其时间戳不变。衔接限制为位移≤20mm、姿态≤5°、速度≤0.05m/s及0.2rad/s（同时受已发布更小限值约束）、加速度≤0.3m/s²及1rad/s²、夹爪比例变化≤2/s；保留至少两行原始轨迹用于下一次重规划。衔接及边界均验证，超出可用时间／安全包络则HOLD，不能把任意预测变成可执行动作。

衔接段的真实目标进入队列和下一次RTC前缀，并冻结到衔接末尾，不能被在途重规划提前改写；已跳过时间槽记录为真实HOLD，而非虚构执行模型动作。日志保存原始响应、绝对目标、首行偏差、跳过行数及实际衔接队列，便于复盘。首个未来槽即发起重规划，不额外等待K步。OSC模块及公共接口不变。

当前调度已替换为官方固定参数语义，见下节；旧的历史最大耗时、额外一槽及最小重规划间隔不再使用。反馈异常、队列耗尽、无可用未来或错过控制时间槽仍 HOLD 并等待手动重启。

### 固定 inference_delay / execute_horizon（云端已部署，1秒真机控制验收通过）

参考官方 `src/eval_flow.py` 与 `src/model.py`：每轮执行旧块前 D 行和新块 `[D:E]` 行，下一轮将新块索引前移 E 行。D=`inference_delay`，E=`execute_horizon`；引导权重使用 `get_prefix_weights(D, H-E, H, "exp")`，不把整段可用队列长度当成重叠长度。

默认 D=8（400 ms）、E=10（500 ms），均为可编辑启动配置，不是实测延迟或性能保证。校验 `0 ≤ D ≤ E < H20`、`D+E ≤ H20`；退休字段 `rtc_min_replan_steps` 在读取旧配置时丢弃，旧 API 写入会明确拒绝。普通模式 H/K 行为不变。

实机使用绝对观测时间：第 i 行仍在 `t_obs+i×50 ms`，延迟冻结边界取 `t_obs+D×50 ms` 向上对齐本地发送网格，同时保护已承诺目标和启动衔接。下一轮按本轮真实观测时间加 E 步推进，等待该边界上的新实测样本，不以较晚的请求准备时刻作为周期原点，也不改写样本时间。首次 bootstrap 后立即补充轨迹，是从静止启动的特殊阶段。迟于冻结边界的响应不得拼接执行，明确报告 `inference_delay exceeded`，保持 HOLD，由用户调整配置或排查延迟。D=0 允许官方零延迟算法测试，但真实异步服务通常无法满足零延迟截止时间。

2026-10-19 17:09 后续验收：实际网页点击后，运行 `20261019T170954-eb1f4ff4` 完成约1.021秒真机RTC控制，21个目标、硬件CPV批次增量57、实测TCP最大位移34.8mm，本轮无RTC故障；最终IDLE／HOLD／NORMAL，预览继续。当前部署固定D=7、E=8（并未改变新安装缺省8／10）。完整记录见 `runtime/diagnostics/rtc_one_second_acceptance_20261019.md`。这不是长时间稳定性或永久无抖动保证。

冷启动的真实H20静止HOLD前缀采用官方 `ones` 权重，引导全部有效7维；运行重规划仍采用 `exp` 和 `H-E` 重叠，不改变普通采样。metadata增加 `rtc_bootstrap_attention=full_hold_v2`，必须与云端协调部署。安全衔接器在原零末端速度五次曲线不可行时，尝试匹配后三个原始未来样本的五次延续曲线，仍逐点检查原有位移、姿态、速度、加速度和夹爪边界；不平移后续模型行的时间戳，不取消启动保护。

本次还复现固件模式切换的参数恢复竞态：读到CV=2而跳过写入后，后续读回已变为1.5。显式参数恢复函数仅对这种跳过后不匹配补写一次并重新核验；已写过仍不匹配继续报错。另观察到停止返回HOLD后CV再次为1.5，已用维护接口恢复2／10／10；这不代表固件重置来源已根治。OSC模块本轮未修改。

模型端继续沿用 OpenPI 递减时间采样方向及已验证的 VJP 符号。新增 metadata `rtc_schedule=fixed_delay_execute_horizon_v1`，请求与回显携带 D、E、`prefix_attention_horizon`；缺少新能力的旧云端服务会被拒绝用于 RTC，普通预览仍兼容。必须协调部署云端 `codec.py`、`serve.py` 与本地修改。

2026-10-19 后续部署：SSH43198实例已更新，GPU无运动验证通过。实际网页启动发现并修复完整物理前缀长度与引导重叠长度混用的数组错误。再次网页启动在首块约336 ms返回后，因无法在现有启动安全包络内衔接HOLD而拒绝运动；两次CPV发送增量均0、最终正常HOLD，不能宣称已完成1秒真机验收。随后补充的本地首D步严格保留HOLD修正通过单测，尚未重载。详情见 `runtime/diagnostics/rtc_fixed_de_delivery_20261019.md`，仍需解决冷启动衔接，不能通过取消保护验收。

2026-10-19已完成AutoDL与本地协调升级，更新 `inference/rtc/{codec,serve,validation}.py` 并通过RTX4090启动验证。旧 `nero.rtc.v1` 服务会被新客户端明确拒绝用于RTC；不能只热更客户端后继续旧RTC会话。已有服务目录名 `nero_rtc_v1` 是历史路径，不代表新版协议仍为v1。模型权重、assets、普通采样和引导符号未改变。远端升级前文件保存在 `backup-before-v2-20261018T225338`；当前服务日志为 `service-v2-recovery.log`。

启动反馈修复：普通预览仅验证7D/H动作响应，不能覆盖RTC能力不兼容状态；状态接口按当前metadata复核RTC版本，提前禁用控制并说明原因。浏览器启动异常独立保留，不会被成功预览轮询或会话清理抹掉。已实际通过网页点击开始／停止影子RTC：200个目标、正常HOLD、最终IDLE、真机CPV写入增量0。该验收不代表真机任务性能或完整训练溯源验证。

模型目标和夹爪调用由模型控制源串行发送。停止先撤销发送许可，再等待已有同步调用结束，通过原接口确认HOLD；确认前拒绝重新启动。现有 OSC 仍负责最终权限、IK、工作空间和硬件约束。TCP成功而夹爪失败记录为部分执行并停止，不宣称二者是物理原子动作。

当前 RTC 观测约束：图像年龄≤150 ms、双图时差≤75 ms、实测TCP及缓存年龄≤100 ms、图像／TCP时差≤150 ms、夹爪SDK标记停更≤150 ms。首次启动最多等待1秒收齐缓存。bootstrap漂移阈值为2 mm／0.5°。这些仅属于模型控制源，不改变共享采样或 OSC 阈值。

OSC 的 `sample_monotonic_ns` 与相机／缓存的 `perf_counter_ns` 在比较前统一到 QPC 时间基准。Windows Python 3.12 中两者可能有不同起点，不能直接相减。影子 CPV 的 HOLD 样本允许复用静止姿态，但必须有新鲜缓存、启用中的影子仿真、HOLD_READY、七轴当前仿真关节与采样关节一致、七轴速度为零的证据；真机新鲜度检查保持不变。

在不重启当前控制服务的情况下，可加载最新模型模块，借用控制台的相机预览和真实 OSC 影子会话做有限时验证：

```powershell
.venv\Scripts\python.exe scripts/rtc_live_shadow_probe.py --model-port 18762 --seconds 60
```

要求控制台会话空闲、双相机已接入。脚本只创建 shadow 会话，拒绝夹爪写入，结束时确认 HOLD 并释放自己的会话。它使用已有 JPEG84 预览 HTTP 接口，图像时间采用读取前状态的保守下界，无法取得与 JPEG 原子绑定的精确采集时间；因此它是诊断桥接，不等同于前端进程内直接消费原始 RGB 缓存的验收。TCP／关节来自 OSC 仿真，不代表真机轨迹。报告同时记录失败、相机桥接耗时及硬件 CPV 邮箱发送数，不通过时不会自动重试。

## 采样与验证

参考 [RTC 官方引导实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py) 和 [论文异步算法](https://arxiv.org/html/2506.07339v2)。OpenPI 从t=1积分到0，步长为负：`f=x-t·v`，`g=J_fᵀ[W·(Y-f)]`，使用 `v_rtc=v-λg`。默认10步、指数前缀权重、引导上限5；只在有效前缀和有效7D构造约束残差。零引导直接调用普通采样，避免0×NaN。KV缓存只预填充一次，VJP仅对动作变量求导。

无运动 WebSocket 测试（包含上传、SSH、GPU、返回及解码）：

```powershell
.venv\Scripts\python.exe scripts/rtc_websocket_probe.py --port 18762
```

无硬件持续运行测试（假OSC、假相机、150 ms假推理）：

```powershell
.venv\Scripts\python.exe scripts/rtc_shadow_probe.py --seconds 600
```

两者结果分别保存到 `runtime/rtc_websocket_report.json` 和 `runtime/rtc_shadow_report.json`。后者验证调度与停止行为，不代表真实OSC动力学或任务效果。单元测试另覆盖十分钟虚拟时间、150～300 ms延迟抖动、冻结前缀、超时、部分执行、旧结果与手动重启。

## 真机 RTC 验收门槛

新实例的旧训练数据副本为20 Hz，但为8D状态；当前H20 checkpoint携带7D统计。尚未取得能够将指定checkpoint、实际训练变换和第0行动作时间对齐关联起来的完整证据。模型加载与协议验证通过，不等于这些训练语义已核实。

服务默认发布 `training_contract_verified=false`，当前客户端将其作为未核验信息展示，不以此阻止手动启动真机RTC。实测反馈、动作校验、工作空间与OSC权限检查继续有效。核实训练产物后，可使用 `serve.py --training-contract /path/to/verified-contract.json` 加载以下部署契约，服务会校验字段和证据文件摘要：

```json
{
  "checkpoint_dir": "/root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649",
  "state_dim": 7,
  "action_dim": 7,
  "action_horizon": 20,
  "period_s": 0.05,
  "row_zero_alignment": "next_control_slot",
  "discrete_state_input": false,
  "normalization": "quantile",
  "action_contract": "base_relative_observation_start_rotvec_gripper_absolute",
  "evidence_files": ["/path/to/actual/training-config", "/path/to/actual/dataset-alignment-record"]
}
```

字段及证据文件的存在检查不能替代对训练语义的审查；不能凭示例创建“已验证”记录。完成证据核验、真实OSC Shadow及现场验收后，再开展真机对比。凭据不写入仓库。

## 2026-10-15 实施验证记录

本地基准为 `61954aa`。指定新实例 RTX4090 已加载上述权重，服务在8002运行，模型和两种采样路径共用参数。部署清单及云端报告的本地副本在 `runtime/rtc_remote/`；本地运行报告在 `runtime/rtc_*report.json`。

| 检查 | 结果 |
|---|---|
| 原推理及相关回归 | 108项通过 |
| RTC 专项 | 16项通过，包括停止屏障、未确认HOLD、其他会话保护、部分执行和手动重启 |
| 控制及OSC相关回归 | 所选100项中21项因依赖未具备跳过，其余通过；该批包含早期11项RTC测试 |
| 网页 | 11项前端测试通过；实际Edge模拟后端桌面／窄屏检查通过 |
| 零引导 | 同噪声、同配置下与普通路径逐元素一致 |
| 引导方向 | 可解析速度场通过；真实模型同噪声加权前缀误差13.114→2.673（归一化空间，非TCP毫米误差） |
| 云端同步耗时，12次 | 普通中位81.0 ms；RTC中位112.5 ms、P95 117.6 ms |
| 完整SSH/WebSocket链路，最终20次 | RTC中位173.3 ms、P95 236.8 ms、最大495.9 ms |
| 协议异常 | 竞争客户端、8D状态、未知版本、非法前缀及错误图像被拒绝；拒绝后可重新连接 |
| 十分钟实时时钟测试 | 假OSC／相机／150 ms推理，11,996个目标、1,500次请求，HOLD确认成功 |
| 真实模型接假控制端，30秒 | 597个目标、75次请求，HOLD确认成功 |

Windows上两项持续测试的发送间隔约31～79 ms；固定50 ms时间槽没有变成严格实时保证。最终链路样本出现约496 ms尖峰，150～300 ms不能作为已保证上限。真实OSC动力学Shadow、训练时间对齐证据和真机任务性能仍待验收；本轮没有发送机械臂运动命令。

## 2026-10-18 克隆实例影子验证

已改用 `connect.nmb2.seetacloud.com:47642`，确认 RTX4090、指定11649权重与服务文件，启动模型并预热。当前本机转发为18762→远端8002。凭据未写入文件。

- `runtime/rtc_websocket_20261018.json`：普通与RTC各20次；RTC中位192.1 ms、P95 206.2 ms、最大225.2 ms。竞争连接、8D状态、未知版本、非法前缀／图像拒绝及重连检查通过。
- `runtime/rtc_live_shadow_20261018_diagnostic2.json`：真实双相机JPEG预览＋现有OSC影子仿真，107个目标、13个返回块后，因图像年龄超过150 ms进入HOLD。发送间隔中位48.5 ms、P95 61.4 ms、最大64.2 ms。硬件CPV发送和邮箱发送增量均为0；HOLD确认成功。其他短轮次亦触发过图像／TCP时间差保护，不能判为端到端稳定通过。
- 真云端模型＋模拟相机／控制端的持续尝试曾触发冻结前缀响应超时；随后有报告的30秒运行通过：`runtime/rtc_remote_shadow_20261018.json`，595个目标、75次请求，HOLD确认，零硬件命令。30秒成功不能覆盖较长持续尝试中的超时。

本次修复模型模块的Windows时钟混用及影子静止HOLD启动阻断。独立探针已使用新代码；运行中的控制台未重启，网页进程尚未加载该修复。OSC和硬件模块没有修改。当前控制台会话已释放、输出停止、双相机保持接入、模型连接保留。下一阶段需在受控重载后使用同进程原始RGB缓存验证，区分真实采集停顿和HTTP预览桥接开销，再做更长稳定性测试。训练契约仍未核实，真机RTC门槛保持关闭。
