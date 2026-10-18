# 11649 checkpoint 训练契约核验（2026-10-18）

核验对象：`connect.nmb2.seetacloud.com:47642` 上的
`/root/autodl-tmp/nero/checkpoints/pi05_nero_chunk7_full_h20/nero_h20_from11649_pro6000_b32/11649`。

结论：**证据不足，保持 training_contract_verified=false**。没有生成“已验证”契约，也没有改变云端启动参数或重启模型服务。

已核对：指定 checkpoint 存在 params 与 assets；正在运行的部署声明 state7、有效 action7、H20、50 ms，并通过真实请求返回 H20。其 `assets/nerovla/nero_chunk7_v1/norm_stats.json` SHA256 为 `27521fbad1f2d8d4e840d58de701def2ec51f4b6e4ba95294bca4e41ef388402`。上述事实验证部署接口，不证明训练时间语义。

实例保留的 `/root/autodl-tmp/nero_chunk_lerobot/meta/info.json` 声明20 Hz、105个episode、72777帧，但状态为8D、动作为7D。该文件 SHA256 为 `60437c6c9a0ac515a54597f7ac9b529103103191688a3ef291adf3bea555b357`。

实例当前 `/root/openpi/src/openpi/policies/nero_chunk_policy.py` 要求8D状态，将动作位置相对块起点编码，旋转用目标旋转左乘起点逆旋转；SHA256 为 `f6021ce805899efdbcee727e1c858fbb2c22cf2cd3a423f55812b0eae1bf427c`。当前 `pi05_nero_chunk_lora` 训练配置为H10。不能将这份旧配置或旧数据直接视为指定7D/H20完整权重的训练证据。

指定 checkpoint 目录中的 params 是模型数组与 Orbax 元数据，未发现能够将该 checkpoint 绑定到实际H20训练入口、7D预处理或动作第0行时间对齐的训练配置记录。目录名称、模型加载成功及统计维数均不足以补齐该关联。

训练溯源仍缺少的证据：

1. 与11649实际产物绑定的训练配置／启动记录及源版本，确认H20、连续7D状态及所用归一化。
2. 实际训练数据转换和采样代码／记录，证明20 Hz及每块相对观测起点、基坐标旋转、绝对夹爪语义。
3. 同一数据管线的观测与动作索引对齐证据，确认第0行对应下一控制时间槽，而非当前槽或其他偏移。

取得并审查这些证据后，才可生成带摘要的已验证训练契约并受控重载云端服务。按用户后续要求，该核验标记仅作信息提示，不再阻止手动启动真机RTC；反馈、工作空间、协议、权限和故障停车检查仍有效。没有将未核验标记伪造为已核验。
