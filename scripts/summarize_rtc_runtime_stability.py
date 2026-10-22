"""Render the no-motion runtime audit; never accesses robot interfaces."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runtime/rtc_runtime_stability_20261021'


def main():
    report=json.loads((OUT/'probe.json').read_text(encoding='utf-8'))
    rows=report['rows'];table=[]
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    colors={'normal':'#64748b','bootstrap_hold':'#16a34a','replan':'#2563eb'}
    for phase,color in colors.items():
        idx=[i for i,r in enumerate(rows) if r['phase']==phase]
        elapsed=np.array([rows[i]['elapsed_ms'] for i in idx])
        server=np.array([rows[i]['timing']['diagnostic_total_ms'] for i in idx])
        axes[0].plot(idx,elapsed,'o-',color=color,label=phase)
        axes[1].plot(idx,server,'o-',color=color,label=phase+' server total')
        table.append(f"| {phase} | {len(idx)} | {np.median(elapsed):.1f} | {np.percentile(elapsed,95):.1f} | {max(elapsed):.1f} | {max(server):.1f} |")
    rtc=[i for i,r in enumerate(rows) if r['phase']!='normal']
    sample=[rows[i]['timing']['synchronized_sample_ms'] for i in rtc]
    axes[1].plot(rtc,sample,'x--',color='#f59e0b',label='RTC synchronized sampler')
    axes[0].axhline(300,ls='--',color='#dc2626',label='D6 nominal budget')
    for axis in axes:axis.set_ylabel('Latency (ms)');axis.grid(alpha=.2);axis.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('Sequential NO-MOTION request index; NOT action step')
    fig.suptitle('Post-warmup: stable model compute does not imply stable end-to-end latency')
    fig.tight_layout();fig.savefig(OUT/'latency.png',dpi=150);plt.close(fig)
    audit=report['metadata']['normalization_audit']
    text='''# RTC编译、时间索引与归一化专项诊断

## 论坛描述与本项目是否相同

论坛没有提供原帖与复现代码，不能确认其“CUDA graph失败”解释。我们的4090服务是JAX/XLA，并非直接使用PyTorch CUDA graph。JAX的输入形状、dtype、pytree或静态参数变化会引入新的编译；首块与后续块数值不同并不必然需要新图。[JAX官方说明](https://docs.jax.dev/en/latest/jit-compilation.html)

云端新进程实际捕获到8次模型编译，耗时依次约30.89、12.07、28.91、12.27、26.00、12.92、26.38、14.14秒。这些都在开放端口之前完成。它证明漏预热可能造成几十秒冷启动，但不能反推此前的530ms尖峰或抖动一定是编译导致，也没有证据指向CUDA graph capture失败。

此前已经有普通、RTC首块及后续的预热，也预热了三个文本桶。缺口是显式固定noise的诊断分支没有全覆盖，并且运行中没有未预热签名的快速拒绝机制。此次没有宣称以前所有普通RTC调用都漏预热。

## 已实施的优化

1. 普通路径预热内部/显式noise；RTC按64/128/200文本桶分别预热内部/显式noise，共6个在线采样签名。HOLD首块与后续RTC仍复用采样函数，参考和权重均固定为`[1,20,32]`。mask、D/E、参考有效长度是数值变化，不改变模型图形状。
2. `PreparedSampler`在服务就绪后拒绝未预热的RTC形状、dtype、pytree或静态采样选项。失败会走现有停止保护，而不是让执行队列等几十秒。它不是CUDA graph捕获成功证明，也不能阻止驱动重置、XLA内部缓存失效、显存竞争等所有问题。
3. 结果带签名摘要、是否已预热、同步采样、准备与后处理时间。显式离线调参可以编译新签名，但不扩展在线允许集，且不应与真机控制并行。
4. 启动时校验实际checkpoint quantile归一化/反归一化往返，包括分位数以外的值。旧绝对目标先相对本次观测做7D坐标转换，再归一化一次，最后补32D；仅7D有引导权重。没有裁剪归一化值到[-1,1]，也没有让补齐的25维参与物理约束。

## 无运动实测

54次请求：普通18次、HOLD首块18次、后续RTC18次。覆盖三个文本桶、随机/固定noise，各重复3轮。输入来自保存观测、固定静止参考，未创建OSC会话，没有机械臂命令；这不是任务成功率或真实启动验收。最长文本故意超过原token上限，沿用原tokenizer截断到200，日志有明确告警，不代表真实任务文本被改动。

| 路径 | 次数 | 端到端中位/ms | P95/ms | 最大/ms | 云端含诊断最大/ms |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+f'''

36次RTC均`prepared_before_call=true`，首块和后续块在相同文本/noise类型下签名相同；同步采样范围{min(sample):.1f}～{max(sample):.1f}ms。READY之后未发现新增模型`jit(fun)`编译。另有一项小型JAX算子编译（日志重复打印两行），不应误写为“所有编译为零”。日志中未找到capture失败，但没有做CUDA profiler，不能证明永不失败。

归一化往返：state最大绝对误差{audit['roundtrip_max_abs']['state']:.3g}，actions为{audit['roundtrip_max_abs']['actions']:.3g}。使用指定11649 assets，同一stats摘要为`{report['metadata']['norm_stats_sha256']}`。这只是数值路径一致性，不替代完整训练产物溯源，也不证明模型预测一定准确。不同物理维度的量程本就不同；不擅自改成毫米等权损失，否则将改变当前IIGDM引导定义。

![按请求观察端到端和模型耗时](latency.png)

## 剩余延迟与“往回跳”的区别

本次后续RTC最大端到端约559ms，但对应云端推理不足167ms；另一个普通请求约449ms、云端推理约88ms。它们没有伴随模型重编译。差额包含编解码、发送接收、SSH/网络、事件循环及本地线程调度，不能仅凭相减进一步归罪某个具体网络节点。54次成功响应之外还记录{report['busy_rejections']}次竞争busy；没有把它吞掉算成功延迟。

6/6继续保持，不因这些尖峰回到抖动更大的8/8，不放宽接缝和新鲜度保护。模型纯计算稳定仍不等于端到端300ms有保证；首块超过预算保持HOLD，后续只有尚有旧参考且同刻位置/姿态连续才允许替换，否则停止。

本地`Timeline.merged()`用原观测时间戳采样，只替换大于等于`freeze_until / committed_until / next_tick / 当前时间下一槽`最大值的未来槽；`complete()`要求严格按next_tick递增。因此未发现为了补偿延迟而“时间索引倒退一步”的逻辑。但模型预测的空间方向仍可能反向，这与重放过去的时间槽不是一回事；同刻2mm/0.5°接缝检查也不能保证整块内部速度/加速度完全平顺。之前episode16的mask压缩和非整步冻结错位修复继续保留。

## 交付边界

只更新云端模型服务与模型推理侧诊断文件，OSC、硬件参数、相机和其他控制源未修改；本地控制服务无需重启。保持exp、20次求值、末端细化10步、cap1000、D/E=6/6，不以调大引导掩盖传输尖峰。没有新的真机运动或Git提交/推送。

云端生产软链接已更新为`/root/autodl-tmp/nero_rtc_runtime_guard_20261021`，旧部署目录保留。最终状态：入口和WebSocket均正常、普通预览持续更新、控制关闭、OSC IDLE/held。63项RTC/运行时/传感器相关测试、47项配置/普通/预览/反馈测试及5项调度测试通过，共115项选择性回归；不是全仓库测试，也不代替真机验收。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    print(OUT/'report.md')


if __name__=='__main__':main()
