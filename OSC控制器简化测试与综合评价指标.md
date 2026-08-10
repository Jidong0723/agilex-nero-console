# OSC 控制器简化测试与综合评价指标

## 1. 测试条件

仅执行一次自动测试：

1. 机械臂从远离关节限位和奇异位形的中间姿态开始。
2. 同时施加 6D TCP 目标变化：
   - 平移：`X +50 mm，Y -40 mm，Z +40 mm`
   - 旋转：`Roll +12°，Pitch -10°，Yaw +15°`
3. 目标轨迹采用当前允许 TCP 最大速度的 **80%**，各自由度同步到达。
4. 到达目标后保持 **30 s**。
5. 按完全相反的轨迹返回初始位姿，再保持 **30 s**。
6. 全程记录：
   - 目标与实际 TCP 位姿；
   - 7 个关节角。

该测试一次覆盖：6D 耦合跟随、正反向运动、停止稳定性和冗余关节慢速自旋。

---

## 2. 单一综合指标：OSC 综合得分

\[
\text{OSC Score}
=
100\left(
0.80S_{\mathrm{track}}
+
0.15S_{\mathrm{hold}}
+
0.05S_{\mathrm{null}}
\right)
\]

得分范围为 `0–100`，越高越好。

### 2.1 动态跟随得分 \(S_{\mathrm{track}}\)

在往返运动阶段计算：

- TCP 位置跟踪 RMS 误差：\(E_p\)
- TCP 姿态跟踪 RMS 误差：\(E_R\)

\[
S_{\mathrm{track}}
=
\frac{1}{2}
\operatorname{clip}\left(1-\frac{E_p}{10\ \mathrm{mm}},0,1\right)
+
\frac{1}{2}
\operatorname{clip}\left(1-\frac{E_R}{5^\circ},0,1\right)
\]

动态跟随占总分 **60%**。

### 2.2 长期保持得分 \(S_{\mathrm{hold}}\)

在两个 30 s 保持阶段分别计算并取较差值：

- TCP 最大位置漂移：\(D_p\)
- TCP 最大姿态漂移：\(D_R\)

\[
S_{\mathrm{hold}}
=
\frac{1}{2}
\operatorname{clip}\left(1-\frac{D_p}{3\ \mathrm{mm}},0,1\right)
+
\frac{1}{2}
\operatorname{clip}\left(1-\frac{D_R}{1.5^\circ},0,1\right)
\]

长期稳定占总分 *25%**。

### 2.3 零空间稳定得分 \(S_{\mathrm{null}}\)

在两个 30 s 保持阶段，计算任一关节的最大净漂移：

\[
D_q=\max_j |q_j(t_{\mathrm{end}})-q_j(t_{\mathrm{start}})|
\]

\[
S_{\mathrm{null}}
=
\operatorname{clip}\left(1-\frac{D_q}{2^\circ},0,1\right)
\]

零空间稳定占总分 **15%**，用于发现 TCP 基本稳定但某个关节缓慢自旋的问题。

---

## 3. 得分解释

| OSC Score | 评价 |
|---:|---|
| `≥ 90` | 优秀，可作为推荐参数 |
| `85–89` | 良好，可实机使用 |
| `75–84` | 基本可用，仍需优化 |
| `< 75` | 控制效果不足 |

## 4. 强制失败条件

即使综合得分较高，出现以下任一情况也应直接判定该组参数失败：

- 30 s 保持期间任一关节净漂移超过 `5°`；
- TCP 位置漂移超过 `10 mm`；
- TCP 姿态漂移超过 `5°`；
- 出现持续运动、失控或无法停车。
