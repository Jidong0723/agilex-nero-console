# 当前唯一云端 RTC 启动方式

当前实例：`connect.nmb1.seetacloud.com:12112`，用户`root`。实例再次克隆后，只替换SSH主机与端口；不要沿用旧实例地址。密码仅在SSH提示时输入，不写入脚本。

生产入口固定为：

```bash
bash /root/autodl-tmp/nero_rtc_production/start_production.sh
```

该入口指向`nero_rtc_runtime_guard_20261021`。固定11649权重、exp、20次RTC求值、cap1000、末端细化10步/0.001、timestamp_frozen、BF16、普通10步、6/6预热；RTC_*环境变量不会覆盖采样参数，不接受实验命令行参数。实际D/E来自控制台请求，网页仍应保存6/6。

## 窗口一：登录并启动云端模型

本机PowerShell：

```powershell
ssh -p 12112 root@connect.nmb1.seetacloud.com
```

登录后执行Bash：

```bash
bash /root/autodl-tmp/nero_rtc_production/start_production.sh
```

首次启动需要加载权重并预热，可能约数分钟。出现`READY`且`rtc_prepared_signature_count=6`后才进入正常服务。冷编译的十几至几十秒日志属于启动阶段，不是已就绪后的单块延迟。前台运行时保留此窗口，Ctrl+C会停止模型。

如需断开SSH后继续运行，可改用后台启动（不要与上面的前台方式同时启动）：

```bash
nohup bash /root/autodl-tmp/nero_rtc_production/start_production.sh >> /root/autodl-tmp/nero_rtc_production/service.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/nero_rtc_production/service.log
```

后台方式下，Ctrl+C只退出tail日志。启动器有锁和8002监听检查；已有服务时不再加载第二份模型，但端口被占用并不自动证明占用者是正确模型，仍需检查网页metadata。

## 窗口二：保持SSH转发

本机另开PowerShell：

```powershell
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -L 127.0.0.1:18762:127.0.0.1:8002 -p 12112 root@connect.nmb1.seetacloud.com
```

保持窗口打开；没有持续文字输出属于正常情况。如果已有转发且网页双卡绿色，不要重复建立相同本机端口转发。

## 控制台核对

- AutoDL云端：`ws://127.0.0.1:18762`，H20；RTC inference_delay=6、execute_horizon=6。
- 服务入口与模型WebSocket卡片均绿色；动作预览自动更新，不表示机械臂正在控制。
- metadata应包含`rtc_signature_guard=warmed_shapes_and_static_options_v1`、`rtc_prepared_signature_count=6`、`rtc_prefix_mask_mode=timestamp_frozen`、exp/cap1000/20步/末端细化10步。
- 普通推理与RTC使用同一服务，不需分别启动；网页选择模式。不需要第三个WebSocket终端。
- 本版本仍有端到端延迟尖峰，6/6不保证永远在300ms内返回；超限或接缝不连续仍会安全停止，不能称为已完成真机平顺性验收。

## 旧版本处理

9个旧云端RTC部署目录及当前目录中的5个离线probe已打包校验后从可运行位置移除。归档位于`/root/autodl-tmp/nero_rtc_archives/`，只保留压缩包用于恢复历史证据；不要解压后当生产启动。权重、assets、训练集和本地报告未删除。

生产目录内旧名字`start.sh`、`start_terminal_refined.sh`现均为同一生产入口的兼容别名，不会启动旧ones/linear或旧8/8配置。本地仓库研究用脚本保留，不属于云端可运行部署；部署时只使用本文件中的生产入口。
