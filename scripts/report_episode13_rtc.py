"""Report recorded jitter and seeded inference-only exp/grid ablations."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
EP=ROOT.parent/'dataset/episodes/episode_000013'
OUT=EP/'analysis/rtc_diagnosis'
EXP=ROOT/'runtime/rtc_episode13_optimization_20261021'


def main():
    report=json.loads((EXP/'terminal_grid/rollouts.json').read_text(encoding='utf-8'))
    selected=['linear20','exp20','exp20r4','exp20r6']
    rows=[]
    for row in report['rows']:
        xyz=np.asarray(row['trajectory']);m=dict(row['metrics'])
        for frequency in (2,4):
            hp=signal.sosfiltfilt(signal.butter(3,frequency,fs=20,btype='highpass',output='sos'),xyz,axis=0)*1000
            m[f'highpass_{frequency}hz_rms_mm']=float(np.sqrt(np.mean(np.sum(hp[10:-10]**2,axis=1))))
        poses=np.concatenate([np.asarray(c['old'])[:6] for c in row['chunks']])
        rotation=Rotation.from_quat(poses[:,3:7]);angles=np.degrees((rotation[1:]*rotation[:-1].inv()).magnitude())
        m['orientation_step_p95_deg']=float(np.percentile(angles,95))
        rows.append(dict(profile=row['profile'],seed=row['seed'],**m))
    (OUT/'optimization_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    fields=['jerk_p95_m_s3','seam_error_max_mm','highpass_4hz_rms_mm']
    titles=['Target jerk P95 (m/s³)','Maximum same-time seam error (mm)','Target >4 Hz RMS (mm)']
    for axis,field,title in zip(axes,fields,titles):
        for seed in (0,1,2):
            values=[next(r[field] for r in rows if r['profile']==name and r['seed']==seed) for name in selected]
            axis.bar(np.arange(4)+(seed-1)*.24,values,width=.24,label='seed '+str(seed))
        axis.set_xticks(np.arange(4),selected,rotation=15);axis.set_title(title);axis.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('120 steps / same recorded observations / same noise; no robot motion')
    fig.tight_layout();fig.savefig(OUT/'exp_grid_comparison.png',dpi=160);plt.close(fig)
    table=[]
    for name in selected:
        values=[r for r in rows if r['profile']==name]
        table.append(f"| {name} | {np.median([r['jerk_p95_m_s3'] for r in values]):.2f} | {max(r['seam_error_max_mm'] for r in values):.2f} | {np.median([r['highpass_4hz_rms_mm'] for r in values]):.3f} | {np.median([r['highpass_2hz_rms_mm'] for r in values]):.3f} | {np.median([r['compute_median_ms'] for r in values]):.1f} |")
    text='''# episode_000013：剩下的小幅抖动，以及改为exp的验证

## 结论与边界

这段真机数据确实更顺了，但模型目标仍有毫米级反复修正，接缝的速度折角也没有完全消失。本轮按要求将生产运行重规划改为官方exp权重，并改善现有末端积分网格，不修改OSC、不做执行端低通或跨块平均、不放宽保护阈值。

离线结果支持“比直接切exp更少尖锐修正”，**不支持“所有频段都比linear更平滑”**。特别是2–4Hz附近的目标来回摆动仍未一致改善。本轮没有启动真机运动，不能宣称彻底解决。

## 1. 先准确界定这段episode

episode实际采集6.6905秒，对应run `20261021T131831-905dddfb` 的RTC全局step约22.7到156.5。完整run持续约17秒，344次目标发送；episode内只有134次。`diagnosis.json`、actual_splices.png是**整个run**重放，不应把它的全部指标当作这段短episode。本文片段指标来自high_frequency.json。

片段内目标单步位置变化：中位0.865mm、P95 3.151mm、最大4.961mm。实测相对目标的位置误差：中位1.116mm、P95 4.174mm、最大9.762mm。控制源参考与实际发送目标重放一致，未发现旧索引错误复发。

整次run结束于夹爪预测-0.008814超出当前下界-0.005；这发生在短episode结束之后，不是该片段抖动的解释。本轮没有扩大夹爪容差。

![完整run目标、实测与接缝，短episode只覆盖其中一部分](actual_splices.png)

## 2. 小抖动从哪里来

在片段中对共同20Hz网格做2Hz高通，目标位置向量RMS约0.831mm、实测约0.862mm。按同一频带估计115ms时移后，相关性约0.890。时移只用于离线比较，不是伪造反馈，也不是精确辨识出的控制器纯延迟。图中灰色浅线为原时序、深色为时移后实测。

这说明模型给出的目标本身就含有小幅来回变化，机械臂随后追踪它；不能仅凭抖动手感断言硬件独立振荡。接缝前后±1step覆盖约49.1%的加速度样本，却占约84.8%的目标加速度平方和，支持仍有更新时的速度折角。它是描述性相关证据，不是“84.8%的物理能量由RTC造成”。

谱上2Hz以上峰值约2.14Hz，不是严格的3.33Hz重规划共振；有些变化也可能是任务修正而非无用噪声。本分析共同网格最高只能解释10Hz以内，不能排除更高频机械振动。原生机器人采样间隔中位19.85ms、最大34.75ms，不能与50ms模型目标周期混为一谈。

![短episode的毫米级目标与实测变化及频谱](high_frequency.png)

整run发送节奏P95约54ms、最大57.1ms，末次快速提交约0.69ms，不是过去同步RPC卡42ms的那种故障。当前目标依然是20Hz离散更新，不是硬实时承诺。CPV日志在该run进入模式后于13:18:31读回CV2、ACC/DCC10；不能用后来HOLD期间读到的1/1/1倒推本次运行一直用了1/1/1。

## 3. 为什么不只换一个exp字符串

论文附录A.4的消融指出指数衰减整体最好、线性接近：
[论文与Figure8](https://arxiv.org/html/2506.07339v2#A4)。本实现的exp仍是官方get_prefix_weights公式，前D行权重1、中间软衰减、参考末端之后0。

但soft mask允许未来计划改变，**不是数学上逐行等式硬约束**。IIGDM也不是每一步都将输出强制覆盖成参考。现有网格只在t=0.001做一次末端求值；软权重小于1时，这一次修正会留下残差。直接换exp，三种噪声中同刻接缝最大达到5.62mm，jerk也增大；单纯加到30步增加约50ms计算，未稳定消除这些问题。

因此新增显式可选的`terminal_refinement_steps`，只细化数值积分：总计仍20次速度场求值，14步从t=1到0.02，随后5步几何下降至0.001，最后一步到0。IIGDM VJP公式、cap1000、exp mask、物理7D归一化与观测时间戳不变。让变化快、引导刚性的去噪末端得到多次求值，而不是依赖最后一次较大的残差修正。

**这不是官方仓库原样均匀积分，也不是完整论文超参数复现**。cap和积分配置延续本项目先前的明确优化。启动静止HOLD仍保留ones，运行重规划才是exp。`start.sh`的默认均匀网格仍可作对照。

## 4. 同观测、同噪声、连续前缀对照

使用这段episode内前120步，3组固定噪声；各方案独立推进自己的旧计划，D=E=6。每轮新观测取记录邻近状态及相机JPEG；同一seed共享初始普通块及逐轮噪声。没有机器人接口、没有硬件写入。

这是固定记录观测开环反事实重放，不是闭环真机仿真。状态最近邻有时间偏差，原始推理RGB和噪声未保存；不同预测不会反过来改变图像。探索性的180步基线后段超出短episode相机时域，已排除出正式结论；脚本现已拒绝超出采集时段的重放。

下表jerk、RMS、计算采用三组指标的中位数，接缝最大取三组最大值。jerk是目标有限差分，不是实测机械臂动力学。

| 方案 | jerk P95中位/(m/s³) | 同刻接缝最大/mm | >4Hz RMS中位/mm | >2Hz RMS中位/mm | 计算中位/ms |
|---|---:|---:|---:|---:|---:|
'''+'\n'.join(table)+'''

![逐种子对照，不只展示最好的一组](exp_grid_comparison.png)

![120步轨迹示例，seed0](G:/codex-jidong/agilex-nero-console/runtime/rtc_episode13_optimization_20261021/terminal_grid/rollouts.png)

选择exp20r6：相对直接exp20，最大接缝降至约1.03mm、三组jerk P95中位下降约58%，计算无明显增加。相对linear，jerk中位下降约42%、>4Hz RMS中位约下降17%，但seed0的>4Hz RMS略升，>2Hz总分量的中位数反而上升。这些限制不能省略。

episode10也做了三种噪声、各120步交叉复核：exp20r6接缝最大约1.80mm、没有旧ones的大幅接缝；不过加速度P95三组均比linear高，jerk也并非逐组改善。因此这是一版针对exp收敛与尖锐修正的折中，而非跨任务全面胜出的参数。

另外对照了exp20r10（同样20次求值，10个末端细化步）：接缝最大约0.90mm、>4Hz RMS三组中位约0.199mm，比r6略小，但jerk P95中位约13.05，比r6的12.82略高，没有一致优势。因此本次保留已做episode10交叉复核的r6，不宣称它是全局最优。额外结果保存在`runtime/rtc_episode13_optimization_20261021/extra_terminal/`。

## 5. 如何交付与继续判定

生产配置：H20、D6/E6、exp、20次求值、cap1000、terminal_noise_time=0.001、terminal_refinement_steps=6、BF16、compact prefix。普通采样仍10步。metadata必须明确报告`terminal_refined_multistep_v2`和实际采样配置，不能仅看模型名字。

客户端对exp也保留原迟到回复保护：实际旧参考覆盖、旧目标存在、同刻位置差≤2mm、姿态差≤0.5°，超限仍HOLD。没有更改100ms实测TCP新鲜度、安全包络、硬件参数或OSC增益。

新云端目录为`/root/autodl-tmp/nero_rtc_exp_production_20261021`，旧linear目录保留以便显式回退。恢复入口仍为`/root/autodl-tmp/nero_rtc_production/start_production.sh`。本轮只做无运动复核与停止状态下受控加载，不自动开始推理控制，不提交或推送Git。

部署后不带采样覆盖字段，对真实request6/10/17各跑seed0/1/2及重复seed0：前6行参考位置残差中位0.064mm、最大0.175mm；同噪声最大输出差0，普通返回仍为20×7。它们不是全部实际冻结时段或执行段：观测有非整数step偏移，例如request6实际替换从row6.705开始，其同刻接缝0.54–1.42mm，后续软区允许偏离更多。不能把前6行结果扩展成“整个执行段都在0.175mm内”。详情见`deployed_exp/result.json`。

相关RTC/网格/IPC/预览/反馈/调度79项测试通过。先前已知的夹爪停止读缓存测试不在这79项中，未宣称完整测试集全绿。实际本地服务已受控重载，恢复外部0/腕部1相机；metadata确认exp/20步/refinement6、v2网格、普通10步及参考覆盖保护。两张连接卡片绿色，自动预览序号持续增加，控制关闭、机械臂HOLD且无控制会话。

下一段真机记录应保持同任务、同初始姿态，核对目标与实测2–4Hz和4Hz以上分量、姿态与夹爪、追踪误差、任务完成情况。若目标平滑而实测仍独立振荡，再进入底层控制/接触/机械诊断；不能用全局低通掩盖任务修正或把新的真机结果提前宣称为通过。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    print(OUT/'report.md')


if __name__=='__main__':main()
