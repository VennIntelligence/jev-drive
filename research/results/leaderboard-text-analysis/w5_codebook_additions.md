# W5 对第一轮 codebook 的增补建议

第一轮 `out/taxonomy.md` 主要编码榜单定向机制，以下 C 档论文披露的普通方法设计无法准确放入既有类。W5 CSV 在 `category` 中暂用这些建议类；它们只表示论文的描述及被报告的对照，不表示代码已核验、增益可迁移或因果已确立。原 codebook 类别名称保持原样。

| category | 适用判据与边界 | W5 例子 |
| --- | --- | --- |
| `language_action_alignment_training` | 用语言/问答/解释与动作或规划联合训练；训练期从未来轨迹造标签仍属于此类，只有测试时读未来真值才用原 `future_label_conditioning`。 | LinkVLA、SteerVLA、Senna、LVLDrive、Poutine |
| `efficient_action_decoding` | 改变动作解码流程以减低自回归或推理成本；若只提速而 DS 不变须如实记录。 | LinkVLA C2F、FIVE-VLA |
| `output_representation` | 对 waypoint、路径/速度分离、离散 token、soft label 等输出表征做可定位改变。不能把标准 PID 跟踪算入原 `control_interface_selection`。 | CarLLaVA、SimLingo-BASE、LinkVLA |
| `navigation_input_conditioning` | 改变推理时提供的 GPS target point、导航命令等**输入条件**；与 `output_representation` 分开。 | LinkVLA Table 7 |
| `hierarchical_semantic_control` | 高层语义策略向低层动作策略发 meta-action 指令，两个模块有明确接口。 | SteerVLA |
| `pretrained_backbone` | 替换或初始化视觉/语言骨干并报告对照；需备注架构和模型规模是否同时变化。 | CarLLaVA、FIVE-VLA、DrivoR |
| `scene_token_compression` | 用 register 或其他模块压缩视觉 patch token 供规划器使用；与骨干预训练初始化分开。 | DrivoR Table 4b |
| `recurrent_action_memory` | 动作预测读取先前时刻动作表征作为隐状态，并有推理时记忆更新规则。 | FIVE-VLA |
| `dagger_expert_posttraining` | 用学生闭环状态下的专家示范再训练；包含专家候选扩展和接管时机设计；不因训练期特权专家而推断测试期特权访问。 | RoG-DAgger |
| `dataset_sampling_or_source` | 特定数据来源、规模或事件分桶抽样用于训练，且论文披露了对照或明确训练路径；单独效应未知时在记录中说明。 | CarLLaVA、Kyber-E2E、Poutine、Senna |
| `manual_motion_planner` | 手写行为规划与候选轨迹 cost 实现正式推理；只有直接改写**榜单计分代理**的子项权重才用原 `manual_scorer_reweighting`。 | Kyber-E2E |
| `future_latent_conditioning` | 用未来观测的潜变量监督规划器，并明确测试期换成预测未来潜变量。与原 `future_label_conditioning` 的测试时未来真值输入严格区分。 | DriveFuture |
| `geometry_supervision` | 训练期几何 teacher、相机 pose 和跨视角几何约束；需注明哪些模块在推理时保留。 | OmniSpace |
| `multimodal_sensor_fusion` | LiDAR、图像等多传感器直接用于规划；跨传感器协议的分数不能机械比较。 | LVLDrive |
| `masked_scene_token_reconstruction` | 在紧凑 scene-token bottleneck 上施加训练期遮蔽特征重建；训练辅助分支推理时移除。 | NTR |
| `learned_candidate_scoring` | 论文披露学习式候选轨迹评分并选轨，但未说明评分目标是榜单总分或子分数；若明确用榜单分数监督，才可升级到原 `metric_proxy_candidate_selection`。 | DriveFuture、NTR |
| `inference_reasoning_mode` | 比较推理时 CoT/理由生成开关、温度或解码方式；不能把基座消融直接推到最终提交。 | Poutine Table 1 |
| `evaluation_metric_variant` | 论文明确自行采用某项指标的特殊实现；记录该口径，不把口径本身当成方法增益。 | LVLDrive 的逐步延续碰撞判定 |

现有类的 W5 例子：`metric_early_termination`（CarLLaVA、SimLingo-BASE），`metric_proxy_candidate_selection`（DrivoR），`metric_reward_finetuning`（Poutine），`ego_state_fusion`（Senna），`manual_scorer_reweighting`（DrivoR 的 NAVSIM 配置披露）。这些例子只按 W5 的论文证据编码。代码可核性与实际提交配置由第一轮记录另行界定。
