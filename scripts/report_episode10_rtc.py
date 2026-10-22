"""Summarize saved, no-motion episode-10 RTC experiments with lineage plots."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.inference_rtc import Target, sample_rows, PERIOD_NS

EP = ROOT.parent/'dataset/failed_episodes/episode_000010'
OUT = EP/'analysis/rtc_diagnosis'
RUN = ROOT/'runtime/logs/inference_runs/20261021T123838-09d7b497'
EXPERIMENT = ROOT/'runtime/rtc_episode10_optimization_20261021/validation3seeds'


def main():
    events = [json.loads(line) for line in (RUN/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    replies = {e['context']['request_id']:e for e in events if e['event']=='rtc_response'}
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = ['#64748b', '#dc2626', '#2563eb']
    lineage = []
    for rid, color in zip((8, 9, 10), colors):
        reply = replies[rid]; context = reply['context']
        origin = context['timeline_origin_perf_ns']; observed = context['observation_perf_ns']
        rows = [Target(tuple(r[:3]), tuple(r[3:7]), r[7]) for r in reply['absolute_targets']]
        x = (observed-origin)/PERIOD_NS + np.arange(20)
        points = np.asarray(reply['absolute_targets'])[:, :3]*1000
        for axis, values in zip(axes, points.T):
            axis.plot(x, values, '.-', color=color, label=f'Request {rid} full prediction')
            if context['phase'] == 'replan':
                end = min(len(context['absolute_prefix']), context['prefix_attention_horizon'])
                axis.axvline(x[0]+end-1, color=color, ls='--', alpha=.5)
        previous = sample_rows(rows, observed, origin+59*PERIOD_NS)
        current = sample_rows(rows, observed, origin+60*PERIOD_NS)
        lineage.append(dict(request=rid, source_row_at_tick60=(origin+60*PERIOD_NS-observed)/PERIOD_NS,
                            prefix_length=len(context['absolute_prefix']),
                            predicted_step59_to60_mm=float(np.linalg.norm(np.subtract(current.position, previous.position))*1000)))
    for k, axis in enumerate(axes):
        axis.axvspan(59,60,color='gold',alpha=.25,label='Same physical interval: tick 59 to 60')
        axis.set_ylabel('XYZ'[k]+' target (mm)');axis.set_xlim(50,66);axis.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Original RTC target step (50 ms), not response-relative time')
    figure.suptitle('A sharp tail transition appears, then becomes the next constrained reference')
    figure.tight_layout();figure.savefig(OUT/'roughness_lineage.png',dpi=160);plt.close(figure)
    (OUT/'roughness_lineage.json').write_text(json.dumps(lineage,indent=2),encoding='utf-8')
    experiment=json.loads((EXPERIMENT/'rollouts.json').read_text(encoding='utf-8'))
    table=[]
    for row in experiment['rows']:
        m=row['metrics']
        table.append(f"| {row['profile']} / {row['seed']} | {m['target_step_max_mm']:.2f} | {m['acceleration_p95_m_s2']:.2f} | {m['jerk_p95_m_s3']:.1f} | {m['seam_error_max_mm']:.2f} | {m['compute_median_ms']:.1f} |")
    text = '''# episode_000010：接上了位置，但还没有接顺运动

## 先说结论

相比 episode_000004 的同刻接缝误差中位数约24.25mm，本次已降至0.57mm，改善确实很大。但291个实际目标仍有最大22.50mm/50ms的跳步，块内最大也达21.33mm。位置接缝小，不等于速度和加速度连续。

本次离线对照支持的优化是：保留IIGDM公式、坐标语义、原观测时间轴、D=6/E=6、cap1000及末端细化20步，只将运行重规划从ones改为官方已有linear权重。它逐渐释放旧计划尾部约束，减少把急变锁进后续块的现象。启动HOLD仍使用ones；普通推理仍独立10步。没有修改OSC。

## 真机记录说了什么

本次匹配推理run为`20261021T123838-09d7b497`，46次请求/响应、44次拼接。逐条重放请求参考和实际发送目标，位置误差最大均为0，未发现此前的旧块索引错位。相机图像正常。实际发送间隔中位50.01ms、P95 53.97ms、最大58.41ms；节奏有抖动，但无法解释全部目标急变。

| 指标 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|
| 新旧目标同一时刻位置差/mm | 0.57 | 2.38 | 3.70 |
| 实际连续目标位置步长/mm | 2.57 | 13.97 | 22.50 |
| 非接缝处目标位置步长/mm | 2.41 | 12.97 | 21.33 |
| 实测相对目标位置误差/mm | 5.51 | 33.61 | 53.42 |

接缝处只占位置步长平方和约23.9%，大部分粗糙度已在块内。该平方和是描述性指标，不是机械能。历史普通推理不是相同闭环、任务、噪声的配对实验，不据此声称RTC一定优于/劣于普通模式。

![实际发送目标、实测与旧计划尾部](actual_splices.png)

![最明显的三处接缝](worst_seams.png)

## 一个急变如何穿过三个chunk

看同一物理区间tick59→60：request8尚未约束的尾部只变化0.96mm；request9把这一区间预测成23.46mm，而且恰好跨过实际13行参考的末尾；request10再次取它作为参考后，仍预测17.19mm的变化。实际切换时跳22.50mm，但同刻新旧差仅1.02mm。

这条证据链支持：在这些样本中，ones的参考末端突然从强约束变成自由预测，会出现急变；它下一轮进入前缀后，又被很强地继承。不能把所有急变都归于这个原因，模型、观测变化及跟踪误差仍参与闭环。单纯加大cap或更多步，只会更精确地贴住参考，不自动令参考平滑。

![同一目标时间在三次预测中的演变；虚线为各自实际参考末行](roughness_lineage.png)

## 对照，而不是凭感觉调参

使用同一episode保存的双图、状态和实际prompt，三种固定噪声，每种连续240步，D=E=6；每轮旧参考由上一轮该方案自己的新预测推进6行得到。ones/linear使用相同初始普通块、相同逐轮噪声和相同观测。

这是**固定记录观测的反事实开环重放**：没有机器人接口、没有硬件写入，不是闭环仿真。JPEG近邻帧和近邻状态不是原始请求精确RGB；记录各样本时间误差，也没有重现实际网络延迟。不同方案轨迹不同，但使用相同记录图像，因此不能据此判断任务成功率。

| 方案/seed | 最大目标步长/mm | 加速度P95/(m/s²) | jerk P95/(m/s³) | 同刻接缝最大/mm | 计算中位/ms |
|---|---:|---:|---:|---:|---:|
'''+'\n'.join(table)+'''

加速度与jerk均为目标序列按50ms有限差分，不是实测机械臂动力学。linear在三组中使加速度P95下降约78–83%、jerk P95下降约84–86%，计算没有明显增加；代价是接缝位置误差略增，最大约2.11mm。exp首轮出现约6.49mm同刻接缝最大误差，未选择。

同一输出的姿态也做了核对：ones三组的单步姿态角P95为3.37–3.78°，linear为1.99–2.17°；最大值从5.95–10.80°降至2.79–3.45°。夹爪比例的最大单步变化也下降。仍有明显运动速度，不能把变顺等同于适宜当前真机速度。

下面仅画seed0；另外两组完整数据见rollouts.json。

![同观测/同噪声连续240步对照](G:/codex-jidong/agilex-nero-console/runtime/rtc_episode10_optimization_20261021/validation3seeds/rollouts.png)

## 还没解决什么

最后一次端到端556.5ms尖峰耗尽了有效引导参考，触发`late response has no guided old-plan continuation`。这是另一个时延问题，不是本次换权重就能治好的，不能靠取消HOLD保护掩盖。

现有轻量OSC快照未发布limits，RTC的额外笛卡尔速度预检查是条件执行；底层OSC自身保护仍在。本轮不擅自改变它或放大范围，但不能把离线轨迹评估当成安全执行证书。

linear仍保持原有迟到接收条件：真实参考覆盖、旧目标存在，同刻位置差≤2mm、姿态差≤0.5°；超限仍停止。弱约束不保证每一块都能过检查。D/E不增加，TCP观测100ms阈值不放宽。末端细化积分仍是明确标记的非官方均匀网格；linear schedule本身来自官方get_prefix_weights。

## 部署与复核

新版本目录`/root/autodl-tmp/nero_rtc_linear_production_20261021`已启动，canonical生产链接已指向新目录；原ones版本目录保留。WebSocket实际metadata确认linear/20步/cap1000/terminal_noise_time=0.001，普通10步、reference覆盖保护开启。启动HOLD保持ones。

部署后不使用采样覆盖字段，重放本episode request7/10/12，各使用seed0/1/2及重复seed0：前6行位置残差最大0.925mm、中位0.207mm，重复噪声输出复现。见`runtime/rtc_episode10_optimization_20261021/deployed_linear/result.json`及prefix_residuals.png；这个结论只针对这些样本的冻结前缀，不能外推到所有未来行。

本地相关RTC/诊断/IPC/预览/反馈/调度76项通过；另跑夹爪观测8项，7项通过、1项既有失败：`test_shadow_stop_and_other_session_make_no_reader_call`仍假设停止后不读反馈，与现有自动预览读取行为不一致，本轮不修改它以掩盖失败。`git diff --check`提示已有config/pi05.json四行尾部空白，未重写用户配置。

确认控制关闭、无数采、HOLD后受控重载本地服务，恢复外部0/腕部1相机。重载后识别到FREEDRIVE，仅调用HOLD恢复有效反馈；没有发TCP位移目标、没有启动推理控制。两张连接卡片绿色、普通自动预览恢复H20，真机控制仍关闭。

本轮没有启动真机运动。离线曲线更顺，不等于真机已经验收；下次应保存同任务的新episode，对比实测TCP、目标、姿态、抓取效果和停止原因。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    print(OUT/'report.md')


if __name__ == '__main__':main()
