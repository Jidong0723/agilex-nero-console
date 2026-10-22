"""Explain timestamped RTC mask regression from saved no-motion evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'dataset/failed_episodes/episode_000016/analysis/rtc_diagnosis'
DATA=ROOT/'runtime/rtc_episode16_20261021'


def main():
    paired=json.loads((DATA/'paired_masks.json').read_text(encoding='utf-8'))
    smaller=json.loads((DATA/'timing_6.json').read_text(encoding='utf-8'))
    modes=['legacy_compressed','nominal','timestamp_frozen']
    labels=['Old compressed overlap','Keep nominal H-E','Timestamp-frozen rows']
    colors=['#dc2626','#d97706','#2563eb']
    sample=[r for r in paired['rows'] if r['request_id']==6 and r['seed']==0]
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
    for mode,label,color in zip(modes,labels,colors):
        row=next(r for r in sample if r['mode']==mode)
        mask=[r['guidance_weight'] for r in row['diagnostics']['prefix_residuals']]
        axes[0].plot(np.arange(len(mask)),mask,'o-',label=label,color=color)
        for k,axis in enumerate(axes[1:]):
            axis.plot(np.arange(20),np.asarray(row['predicted_positions'])[:,k]*1000,'.-',label=label,color=color)
    ref=np.asarray(sample[0]['reference'])*1000
    for k,axis in enumerate(axes[1:]):
        axis.plot(np.arange(len(ref)),ref[:,k],'.--',color='#334155',label='Actual old reference')
        axis.set_ylabel('XYZ'[k]+' target (mm)')
    for axis in axes:
        axis.axvline(sample[0]['source_row'],color='#7c3aed',ls=':',label='Actual first replaced target')
        axis.axvspan(len(ref)-1,13,color='gray',alpha=.08)
        axis.grid(alpha=.2);axis.set_xlim(5,13)
    axes[0].set_ylabel('Guidance weight');axes[0].legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('New chunk row at ORIGINAL observation time; missing reference is not extrapolated')
    fig.suptitle('Request6: first executed target is row8.717, NOT one of the well-converged rows0..7')
    fig.tight_layout();fig.savefig(OUT/'timestamp_mask_comparison.png',dpi=160);plt.close(fig)
    summary=[]
    for label,rows in [(m,[r for r in paired['rows'] if r['mode']==m]) for m in modes]+[('timestamp_frozen_D6E6',smaller['rows'])]:
        gap=[r['same_time_position_mm'] for r in rows]
        summary.append(dict(mode=label,median_mm=float(np.median(gap)),max_mm=float(max(gap)),
                            continuity_pass=sum(r['continuity_allowed'] for r in rows),n=len(rows)))
    (OUT/'mask_ablation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    table='\n'.join(f"| {r['mode']} | {r['median_mm']:.2f} | {r['max_mm']:.2f} | {r['continuity_pass']}/{r['n']} |" for r in summary)
    text='''# episode_000016：8/8抖动回归的原因、修复与恢复6/6

## 结论

这次不是主观感觉：实际发送目标出现最大26.79mm/50ms跳步，同刻新旧计划最大相差26.33mm。主因是8/8短重叠放大了两个实现问题：有效参考缩短时，exp衰减区被一并压短；实际冻结时间有非整步偏移，而mask仍只按配置8行构造。及时返回的大跳变还绕过了只对迟到回复生效的连续性检查。

已修复mask的参考有效区与时序映射，所有回复均做2mm/0.5°同刻连续性检查；按用户明确选择，恢复D6/E6，保留exp、H20、20次求值、末端细化10步、cap1000及terminal_noise_time=0.001。不动OSC，不加执行端低通，不放宽安全阈值，不启动真机运动。

## 真机证据

对应run：20261021T151523-814e408a。完整run282次目标发送、33次拼接；episode采集13.888秒，覆盖其中256个目标。

| 指标 | 数值 |
|---|---:|
| episode内目标步长中位/P95/最大 | 4.30 / 15.95 / 26.79mm |
| 完整run同刻接缝误差中位/P95/最大 | 6.85 / 25.38 / 26.33mm |
| 实测相对目标误差中位/P95/最大 | 8.55 / 38.07 / 67.90mm |
| 完整run发送间隔P95/最大 | 54.40 / 56.71ms |
| 完整run推理延迟中位/最大 | 245.39 / 530.69ms |

重放请求参考及实际发送目标，与日志的位置差均为0，没有发现旧chunk索引错位复发。末次track_tcp提交约0.33ms，不是之前同步RPC阻塞问题。最终530.7ms尖峰触发参考覆盖不足的HOLD，是另外一个时延问题，不能取消保护掩盖。

![真实发送目标、实测以及未执行旧尾部](actual_splices.png)

![最坏接缝：及时返回仍发生厘米级切换](worst_seams.png)

## 从request6看清问题

request6在观测后约221ms返回，比D8的400ms预算早，但实际首次替换是新块row8.717。旧计划同一时刻几乎不动，下一步仅0.81mm；新目标与它相差26.33mm，实际发送一步跳26.79mm。参数没改回旧cap5：模型报告确实是exp/20步/cap1000/末端10步。

由于实际观测间隔约8.0–8.7步，且原计划转到固定执行网格再重采样参考，有效参考只剩10或11行，而非理想12行。不能外推或伪造缺失的参考。

旧代码把`end=min(有效参考行数,H-E)`直接用于官方exp公式。request6有效参考10行，所以公式的end从12变成10，row9权重由约0.287降到0.077。真实冻结边界又在row8与row9之间，row8仍会由旧计划执行，但旧mask将它当可变软区；这样第一个实际新目标插值到row9附近时，引导已非常弱。

前8行残差确实仅约0.06mm，并不矛盾：这些行根本不是第一个新执行目标。**上一轮过于依赖整步离线样本和前8行误差，没有覆盖真实首个替换时刻与参考缩短，验证不足。**

## 修复方式

1. `guidance_bounds()`把名义衰减末端与物理参考有效长度分开。仍以H-E为exp的end；缺失行只将自己的权重归零，不再压缩前面所有行的衰减曲线。
2. 根据`freeze_until_perf_ns − observation_perf_ns`映射真实冻结行数。request6冻结到row8.717，因此row0..8都在冻结区，mask start应为9，而不是盲用8。只做实际时间戳坐标转换；不平移观测、不使用历史最大时延加一步、不增加预测延迟预算。整步观测时与官方离散公式一致。
3. 无论回复是否迟到，在实际替换时刻检查新旧位置差≤2mm、姿态差≤0.5°，超限HOLD。原来早回221ms的26mm跳变将被拒绝，不会因为“及时返回”获得豁免。保持过期行丢弃、已承诺区不替换和队列耗尽停止。

IIGDM场和官方exp函数本身不变。非整步时序映射是适配真实系统的修复，不宣称官方模拟器已包含这一处理；本项目的末端细化积分依然不是官方原样均匀网格。

![同一观测和噪声下的mask及预测；紫线才是真正交接位置](timestamp_mask_comparison.png)

## 配对对照

选取request6/9/11/14/32/34，每个3种固定噪声。使用保存的真实状态、旧参考、时间戳以及最近JPEG；原推理RGB和噪声没有完整保存。比较实际首个替换时刻的新输出与旧计划，连同姿态一起判断是否满足2mm/0.5°。每组18次，零硬件命令。

| 方案 | 同刻位置误差中位/mm | 最大/mm | 连续性检查通过 |
|---|---:|---:|---:|
'''+table+'''

6/6比较会更早替换、参考名义末端为14；它不是同一个物理执行时刻，不能把数字当作闭环任务成功率。也没有因此宣称所有chunk内部都平滑。它支持恢复更长引导重叠、减少参考期外执行这一选择。

8/8在修复后仍有部分样本超过2mm，因此没有把“修复后改善很多”说成“8/8可稳定交付”。用户明确选择允许恢复6/6，优先平顺与安全。

另试验了更小终端时间及cap1e4/1e5，前者接缝虽进一步缩小，却出现约45mm的块内步长；后者出现米级异常目标。全部仅在固定噪声无运动诊断中测试，拒绝采用。生产cap仍1000、终端时间仍0.001，生产启动参数仍拒绝cap>1000。这说明不能只看接缝数字就继续加大引导。

## 交付与边界

本地保存6/6；云端同样预热6/6，真实请求仍携带唯一D/E值。当前默认mask模式为timestamp_frozen。普通推理链路不变，OSC与硬件参数配置不变。本轮不提交或推送Git。

6/6名义预算约300ms；530ms尖峰仍可能安全停止，且新的全响应检查会拒绝不连续块。没有自动恢复控制，不以运行不断为目标绕过保护。只做无运动验证与停止状态加载，本次没有新的真机验收，平顺性仍需下一段实测对比。

### 实际部署复核

云端生产入口已指向`/root/autodl-tmp/nero_rtc_timestamp_production_20261021`，发布的metadata确认timestamp_frozen及6/6预热配置。本地保存6/6，外部相机0／腕部相机1；服务入口与WebSocket均正常，自动预览继续更新，控制未启动。

部署后使用保存观测进行无固定噪声协议验证：一次HOLD首块端到端319.1ms、一次重规划252.0ms，首块已经超过300ms，不能宣称启动必然成功。随后12次重规划尝试中，1次遇到预览占用的busy，另11次成功，端到端中位240.9ms、最大298.0ms；云端纯推理约150.7–161.6ms。busy与319.1ms样本均保留，不因后续成功而剔除。保存观测请求没有包含实时采集耗时，不是完整真机控制时延验收。

RTC时序、引导诊断、积分网格、紧凑输入、传感器与快速通道回归共60项通过。早先一次并行复跑的调度P95测试出现78ms计时波动，单独及最终分组复跑通过，未放宽阈值。本报告的18/18仅是固定参考配对接缝检查，不保证闭环运行永不触发保护。

普通适配器、统一配置、预览、反馈与调度另52项通过，合计112项选择性回归，并非全仓库测试。调度测试因使用同目录测试辅助模块，需用`unittest discover -s tests -p test_inference_scheduling.py`执行；直接按包导入曾报辅助模块找不到，随后按正确入口通过。源码编译和允许Windows CRLF的差异空白检查通过。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8');print(OUT/'report.md')


if __name__=='__main__':main()
