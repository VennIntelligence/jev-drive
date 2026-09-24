# 榜单专用增益 codebook（阶段 2 草案）

编码单位是一个入选方法在一个榜单上的固定仓库提交。`yes` 表示源码中存在对应机制且有 `findings.jsonl` 证据；它**不**自动证明该机制对已发表分数有正向贡献。`no` 表示该机制对该单元适用、关键代码已读而未见；`NA` 表示机制不适用于该协议，或公开代码不足以判断。复审时分别检验机制存在、论文披露、分数消融和部署有效性。

| 类别 | 定义和 yes 判据 | 正例 | 易混淆的反例 |
| --- | --- | --- | --- |
| `metric_proxy_optimization` | 推理时使用学习得到、以榜单子分数/总分为目标的评分头选取或搜索轨迹；需要评分头训练/调用的代码链。 | TOAD 用预测 PDM 子分数做 CEM 奖励。 | 只用普通轨迹概率选最大值，没有与榜单评分项对应的监督。 |
| `metric_aligned_training` | 训练损失、奖励或监督目标显式匹配榜单评分、离散栅格或特定预测时距。 | DriveMA 训练奖励调用 RFS；AD-MLP 按 0.5 m 评测栅格重加权。 | 通用 L1 轨迹损失或普通专家行为克隆。 |
| `metric_weight_tuning` | 推理时人工设定榜单子指标/候选评分权重，以该权重排序而非只用模型概率。 | DrivoR v2 手工改 NC、DAC、DDC、TTC、EP 权重。 | 学得评分头的权重由训练得到，归 `metric_proxy_optimization`。 |
| `benchmark_prompt_specification` | 推理提示明确给模型榜单的计分公式或仿真规则，可能使回答适配协议。 | DriveVLA-M0 提示中写 PDMS 评分权重。 | 只告诉模型“安全驾驶”，无具体榜单规则。 |
| `benchmark_split_adaptation` | 使用与正式评测有重叠或高度近似的划分选择推理参数/模型；记录划分和许可。 | TOAD 在 warmup-two-stage 上调整 CEM 参数，而其场景与 navhard 有交集。 | 在严格不相交开发集上常规选模型。 |
| `offline_candidate_library` | 推理候选来自目标源域离线聚类的大型固定轨迹库，并与在线候选合用。 | GTRS 将 nuPlan 轨迹词表与扩散候选并用。 | 仅使用模型内部可学习的少量 mode query。 |
| `ego_kinematics_shortcut` | 在短时开环规划中，当前/历史自车速度、加速度和轨迹直接进入预测器，分数可能由运动外推取得。 | AD-MLP 仅用 ego 状态；BEV-Planner++ 直连 ego 状态到规划头。 | 仅用可部署里程计做坐标变换，模型规划不消费该状态。 |
| `future_label_conditioning` | **测试时**模型输入或模态选择读取由同一待评估样本未来真值生成的标签。 | BEV-Planner++ 从真值未来轨迹生成转向命令并送入测试规划头。 | 训练时以未来轨迹做监督，测试不读取未来真值。 |
| `answer_conditioned_rationale` | 训练用解释文本由已知未来行动答案条件生成，使解释质量不能独立证明决策推理。 | AutoVLA 教师提示含未来 GT 最佳行动并要求解释对齐。 | 测试时未知答案的在线解释；如测试直接读未来真值，归 `future_label_conditioning`。 |
| `test_time_candidate_selection` | 正式提交汇合多个 checkpoint/额外采样的候选轨迹，并按内部预测分数选一条；需代码显示提交路径。 | RAP-DINO Waymo 脚本拼接两权重候选后取全局最高分。 | 单权重固有多模态头按模型概率输出。 |
| `control_interface_selection` | 闭环 agent 在同一模型的不同输出接口中选择一个来驱动车控，选择涉及手工控制器。 | TFv6 以 route 控转向、target speed 控纵向，而非 waypoint 控制。 | 模型只输出一种轨迹，由通用控制器执行。 |
| `handcrafted_control_override` | 闭环推理后按固定场景/故障阈值覆盖模型车控，直接针对违规或 blocked。 | TF++ 停车牌规则；BLUE 长时间静止后强制油门。 | 正常 PID 跟踪模型轨迹，无额外触发覆盖。 |
| `simulator_pose_access` | 声称传感器输入的 agent 在推理时直接读模拟器对象真实位姿/变换，用于定位或路线。 | SparseDriveV2 Bench2Drive agent 读 CARLA world 中 LiDAR actor 变换。 | 由 GNSS/IMU 推断的位置或公开传感器标定。 |
| `metric_early_termination` | 按路线距离/计分公式主动提前停止，把路线完成率换成更高违规系数。 | TF++ 的 `STOP_AFTER_METER`。 | 故障紧急制动或安全原因停车，未刻意控制路线长度。 |
| `simulator_controller_patch` | 方法评测入口修补模拟器轨迹到车控的共享实现，绝对分数依赖修补版本。 | WA-JEPA 的 HUGSIM `arctan2` 航向修正。 | 修改自己的模型控制策略；若所有基线同版重测，不能推出相对不公。 |
| `inference_failure_brake_fallback` | 推理异常时发送制动轨迹使仿真继续计分，需记录失败比例；此类 yes 只表示潜在路径。 | WA-JEPA HUGSIM 客户端失败后送全零轨迹。 | 直接中止场景、不计分，或正常规划停车。 |

## 变更日志草案

- 阶段 1 的 `metric_reward_training`、`metric_grid_alignment`、`benchmark_target_shaping` 合并为 `metric_aligned_training`，分别保留奖励、栅格权重、时距目标三种子机制在发现描述中。
- `infraction_rule_override` 与 `stuck_recovery_override` 合并为 `handcrafted_control_override`；停车牌与脱困触发写在具体发现中。
- `benchmark_weight_tuning` 更名为 `metric_weight_tuning`，使名称明确指推理评分权重。
- 其余类别保留；对分数效应缺配对消融的记录保留为机制发现，并在发现中标注证据限制。
