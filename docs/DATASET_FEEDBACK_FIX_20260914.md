# 原始采集机械臂反馈修复

现场结果：episode 14 两路各记录 292 张图，机器人状态 0。旧结束逻辑删除了该部分数据，原始拒绝原因没有被记录，不能再还原当时具体缺失字段。

代码缺陷：采集反馈中的夹爪开度来自汇总状态缓存，而该缓存会在 hardware OSC 会话中暂停刷新。空夹爪字段会使整条关节记录被静默拒绝；非空旧值也可能使夹爪记录冻结。

修复：采集反馈装饰器通过原后端的 read_gripper 读取 SDK 接收缓存（get_gripper_status 仅读取 parser 缓存），并记录读取时刻和来源。没有新增 CAN 请求、运动指令或控制算法改动。缺字段拒绝计数和原因对页面及归档可见；缺一路流的 episode 保留在 failed_episodes，只有所有流均为空才按原规则删除。

新增只读诊断地址：GET /api/dataset/feedback，返回与采集相同的反馈读取结果。

验证：26 项相关测试通过（1 项历史测试跳过）。服务实际反馈经过采集序列化路径，60 次读取成功写入 60 行，拒绝 0；该检查使用内存输出，不是正式 episode，也不构成运动中 50 Hz 的验收。正式复验需要下一次用户操作采集。

时间戳补充修复：采集进程启动时测量 `monotonic_ns -> perf_counter_ns` 偏移，把 SDK 新鲜时间转换到与相机、关节观测相同的 `perf_counter_ns` 时钟域。每条机器人记录新增有效的 `latest_feedback_age_s`、`feedback_fresh_perf_counter_ns` 和 `feedback_recording_delay_s`。该修改只在数据采集器和校验器中生效，不参与运动控制。

第一版本恢复方式仍见 archive/README_FIRST_VERSION_RESTORE.md；恢复到独立目录，不覆盖现有程序。已被旧结束逻辑删除的 episode 14 图像不能通过此次修改恢复。
