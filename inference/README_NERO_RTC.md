# NERO 1.01 异步结构的 RTC 推理改造（2026-10-06）

## 范围

仅修改 `supervisor/pi05_adapter.py` 的推理输入、请求、动作块接入/发送，以及 `config/pi05.json` 的 AutoDL playback_mode。新增的 `supervisor/autodl_rtc.py` 仅准备推理上下文；`inference/` 文件仅运行在GPU推理服务。没有修改CAN、SDK、Pink、Ruckig、CPV、OSC实现、相机、PICO、采集器、网页或日志模块。

冻结 `archive/releases/1.01` 不变；修改前当前串行适配器及配置留在 `archive/autodl_rtc_before_20261006`。新版本复用后续已验证的实际发送时钟与反馈来源修复，并非覆盖冻结1.01再丢掉这些修复。

## 执行

1. 模型仍输出H10×7，20Hz，每块共同参考观测TCP；旧数据/统计/权重不变。
2. 运动与推理异步，一次最多一个推理在途。最新块安装前不发起依赖旧参考的下一次RTC请求，不积压历史块。
3. 将实际安装的旧计划，按**新观测的每个50ms时间点**采样成绝对TCP和绝对夹爪前缀。旧窗口之外不外推、不重复末行冒充承诺。
4. 推理服务器逆当前动作倍率，将绝对前缀重新表达为新观测起点下的动作，再经过原NERO的动作变换与训练组归一化。只约束前7维，不约束25个填充维。
5. 使用推理时RTC的VJP引导：预计推理期间将执行的前缀高权重，剩余重叠衰减，未重叠后缀自由预测。这里“高权重”是采样约束，不是宣称数学上精确钳制。
6. 服务器应答包含protocol、reference_chunk_id、delay/overlap和权重。普通服务器不能被静默当RTC使用。
7. 返回后丢弃按实际时刻已过期的前缀；RTC引导的块不再叠加客户端原两周期位置融合。首块/无前缀块仍保留原启动过渡，且其实际计划过渡也纳入下一次RTC前缀。夹爪不平均、不增加运输锁闭或阶段门禁。
8. 保留原实际发送时钟、至少一个50ms间隔、不追赶补发。只有窗口耗尽才用原保留会话HOLD，绝不恢复“每块制动、停稳再推理”。
9. 模型观测复用现有新鲜关节缓存和同一模型只读FK，不新建CAN或底层运动控制器。

## 推理实现与来源

推理时RTC的权重/VJP方法参考Physical Intelligence的Apache-2.0开源实现：

- https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
- https://arxiv.org/abs/2506.07339

本实现适配本工程固定OpenPI JAX pi0.5的adaRMS、KV缓存、时间1→0符号和自定义TCP动作。未复制ALOHA双臂硬编码符号、设备驱动或归一化路径。

## 服务器

原包：`/root/autodl-tmp/nero_h10_9999_20261005`，新代码：`rtc_20261006`。读取原 `checkpoint` 和原训练统计，不修改原 `serve_nero.py`、`start_server.sh`、OpenPI源码或检查点。

新服务启动前做预热和数值预检；预检不通过不得开放WebSocket。预检、服务日志保存在 `rtc_20261006/audit`。

现有端口继续8000，SSH隧道不改。切换工具只匹配这个包里的明确推理进程，SIGTERM结束，不杀控制台/训练/其他任务，不强制kill。

```bash
/root/autodl-tmp/nero_h10_9999_20261005/openpi/.venv/bin/python \
 /root/autodl-tmp/nero_h10_9999_20261005/rtc_20261006/rtc_service_transition.py start-rtc
```

恢复普通推理：

```bash
/root/autodl-tmp/nero_h10_9999_20261005/openpi/.venv/bin/python \
 /root/autodl-tmp/nero_h10_9999_20261005/rtc_20261006/rtc_service_transition.py restore-original
```

## 回档

停止AutoDL输出、结束OSC会话后执行：

```powershell
& 'E:\nero-agilex\agilex-nero-console\archive\autodl_rtc_before_20261006\rollback.ps1'
```

这恢复改造前串行源文件，只把配置的playback_mode设回serial_h10，保留操作员当前语言、倍率和相机配置；需按原方式重置服务加载。恢复普通服务按上方命令。新增RTC文件可保留，不会被旧适配器调用。

用户所命名1.01依然使用原 `archive/releases/1.01/rollback.ps1`，与“回到改造前串行版”不是同一操作。

## 限制

RTC是在模型采样中约束计划连续性，不解决全部实际跟踪滞后，也不是实测进度驱动控制器。推理延迟大于剩余窗口仍可能HOLD；不得把过期行重新计时来伪造连续。

H10只有450ms采样跨度，RTC额外开销、重叠长度和实际夹爪调用需实测。离线/零机器人检查通过不证明抓取成功。此改造不得自动启动真机输出，后续应影子模式验证，再由用户开展实机测试。
