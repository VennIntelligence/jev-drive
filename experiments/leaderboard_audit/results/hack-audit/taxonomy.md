# 榜单专用增益 codebook（阶段 2 定稿）

## 编码规则

编码单位是方法在特定榜单上的固定仓库提交。`yes` 表示可定位的机制及 `findings.jsonl` 记录，**不表示**已证明它提高论文主分数；`no` 表示适用且关键路径已读却未见；`NA` 表示协议不适用或公开源码不足以判断。分数效应、披露、真车有效性和置信度由发现分别记录。

## 最终类别

### metric_proxy_candidate_selection

- 定义与判据：训练榜单子分数/总分的代理评分头，最终输出在已生成的固定候选轨迹中选择；须有评分监督或论文说明及推理选择路径。若正式路径继续以该代理反复生成新候选，则整个代理优化路径归下一类，初始 argmax/回退不单独重复编码。
- 正例：RAP-DINO Waymo 评分头拟合 RFS；DrivoR 与 GTRS 按预测 PDM 子分数选择。
- 易混反例：TOAD 虽以 DrivoR argmax 初始化并可回退，正式路径反复生成新候选，归 `metric_proxy_test_search`；普通轨迹概率 argmax 不够。

### metric_proxy_test_search

- 定义与判据：测试时以榜单评分代理为目标反复优化或搜索新轨迹；初始化时对已有候选的评分属于此搜索流程。
- 正例：TOAD 用预测 PDM 子分数做 CEM 奖励。
- 易混反例：固定候选只重排，归 `metric_proxy_candidate_selection`。

### metric_reward_finetuning

- 定义与判据：训练或微调生成策略时直接调用榜单计分函数作为奖励。
- 正例：DriveMA 的 RL 奖励调用 WOD-E2E 官方 RFS 计算器。
- 易混反例：RAP 用 RFS 监督评分头再选候选，归 `metric_proxy_candidate_selection`。

### benchmark_target_shaping

- 定义与判据：为榜单时距、终点或子指标手工调整轨迹监督目标，能定位目标与协议关系。
- 正例：DrivoR 远期目标提高 navval PDMS，却使 v2 warmup EPDMS 下降；TOAD v1 所用 DrivoR 基座的公开训练命令也启用该目标，实际 checkpoint 训练史需另核。
- 易混反例：普通未来轨迹行为克隆，没有具体协议偏好的目标改造。

### metric_grid_alignment

- 定义与判据：训练损失按榜单碰撞/占用评测的离散栅格尺度调权。
- 正例：AD-MLP 对同一 0.5 m 栅格的误差折半。
- 易混反例：通用连续空间 L1/L2 损失。

### manual_scorer_reweighting

- 定义与判据：推理候选评分公式的子项权重由手工指定且偏离该协议的默认聚合，使轨迹排序改变；不因手写系数就推断曾在测试集调参。
- 正例：DrivoR 与 TOAD v2 评测命令的 NC/DAC/DDC/TTC/EP 改权；GTRS 的候选细排系数。
- 易混反例：仅按榜单默认权重组合预测子分数归 `metric_proxy_candidate_selection`；学习得到的评分头参数亦不算手工改权。

### benchmark_prompt_specification

- 定义与判据：推理提示直接写入榜单评分公式或特定仿真规则，模型能读到协议信息。
- 正例：DriveVLA-M0 系统提示描述 PDMS 权重与非反应式回放。
- 易混反例：仅提示“安全驾驶”，或论文讨论指标而推理提示未包含。

### benchmark_split_adaptation

- 定义与判据：用与正式评测重叠/高度相近的划分选择推理配置，并能定位代码设置与论文划分；须单列官方许可和效应限制。
- 正例：TOAD 在与 navhard 有交集的 warmup-two-stage 上做 CEM 参数消融；DrivoR v2 在同划分上验证推理评分改权。
- 易混反例：严格不相交开发集的常规选模。

### offline_candidate_library

- 定义与判据：推理候选包含从源域轨迹离线聚类的大型固定库，分数依赖其覆盖。
- 正例：GTRS 将 nuPlan 轨迹词表与在线扩散候选并用。
- 易混反例：模型内部少量可学习 mode query。

### ego_only_open_loop_planning

- 定义与判据：针对 nuScenes 三秒 L2 等短时开环协议，规划模型不读环境感知，仅由 ego 历史运动状态和导航标签预测未来。其他榜单的车辆状态输入不直接归入此类。
- 正例：AD-MLP 21 维预制输入，不读相机/LiDAR。
- 易混反例：有视觉输入同时融合 ego 状态，归 `ego_state_fusion`。

### ego_state_fusion

- 定义与判据：针对 nuScenes 三秒 L2 等短时开环协议，视觉规划器把当前/历史 ego 运动状态直接送入规划头或候选打分。其他榜单的车辆状态输入须另证相同的短时模仿捷径，不自动归入此类。
- 正例：SparseOccVLA 的 `temporal_ego_states`；BEV-Planner++ 直连 ego 状态。
- 易混反例：仅用里程计做坐标变换；完全不读视觉的 AD-MLP 归上一类。

### future_label_conditioning

- 定义与判据：**测试时**输入或输出模态选择读取同一样本未来真值生成的标签；若预制推理特征不可追溯须降低置信度。
- 正例：BEV-Planner++ 测试管线从未来 ego 轨迹生成转向命令；AD-MLP 论文/数据代码亦由未来生成命令，但预制 pkl 来源缺失。
- 易混反例：只在训练时用未来轨迹监督；可部署地图导航命令不自动等同真值未来命令。

### multi_checkpoint_candidate_selection

- 定义与判据：正式提交汇合多个 checkpoint 的候选，再按内部预测分数选轨迹。
- 正例：RAP-DINO Waymo 提交脚本拼接多个权重输出后取全局最高分。
- 易混反例：单模型内部多模态头；函数名 NMS 而代码无距离抑制时不认作 NMS。

### control_interface_selection

- 定义与判据：同一网络输出多个控制接口，闭环 agent 选择其一并经手工控制器变成车控。
- 正例：TFv6 默认以 route 控转向、target speed 控纵向，而非 waypoint 接口。
- 易混反例：网络只输出一种轨迹，由常规 PID 跟踪。

### manual_control_override

- 定义与判据：模型输出后按固定场景/故障阈值用手写规则覆盖油门、刹车或转向，包括违规响应和卡住后蠕行。
- 正例：TF++ 停车牌制动和卡住恢复；BLUE 静止计时后强制油门。
- 易混反例：常规 PID 跟踪；推理异常零轨迹回退归 `inference_failure_brake_fallback`。

### simulator_pose_access

- 定义与判据：声称传感器输入的闭环 agent 直接从仿真器 world/actor 读真实位姿/变换用于定位或路线。
- 正例：SparseDriveV2 Bench2Drive agent 读取 CARLA LiDAR actor 变换。
- 易混反例：由 GNSS/IMU 推断位置，或读取公开传感器标定。

### metric_early_termination

- 定义与判据：根据路线距离或乘积计分公式主动提前停止，牺牲完成率以避免后续违规惩罚。
- 正例：TF++ 的 `STOP_AFTER_METER` 选项。
- 易混反例：故障紧急制动、正常交通停车或路线自然结束。

### simulator_protocol_dependency

- 定义与判据：方法评测入口修补共享模拟器轨迹转车控实现；绝对成绩依赖该版本，`yes` 不代表相对同表基线受益。
- 正例：WA-JEPA HUGSIM 入口修正 `arctan2` 参数顺序，论文说明统一重测。
- 易混反例：只修改自身模型后处理；同版统一重测不能推断相对不公平。

### inference_failure_brake_fallback

- 定义与判据：推理异常时发送制动轨迹，使场景继续计分；路径存在与已报分数实际触发须分开。
- 正例：WA-JEPA HUGSIM 客户端异常后发送全零轨迹，聚合器另列可信均值。
- 易混反例：失败即中止而不计分，或正常规划停车。

## 变更日志

- 阶段 1 提议原文在 `out/sources/taxonomy_stage1.md`；阶段 2 回扫全部入选单元后定稿。
- `metric_proxy_optimization` 拆为已有候选重排与测试时 CEM 搜索；RAP Waymo 的 RFS 评分头从 `metric_reward_training` 归入候选重排，DriveMA 的 RFS RL 改名 `metric_reward_finetuning`。
- `ego_kinematics_shortcut` 拆为不读环境的 `ego_only_open_loop_planning` 和仍有视觉输入的 `ego_state_fusion`。
- `stuck_recovery_override` 与 `infraction_rule_override` 合并为 `manual_control_override`，保留各发现触发条件。
- `benchmark_weight_tuning` 更名为 `manual_scorer_reweighting`；`test_time_candidate_selection` 更名为 `multi_checkpoint_candidate_selection`；`simulator_controller_patch` 更名为 `simulator_protocol_dependency`。
- `answer_conditioned_rationale` 不进入最终矩阵：AutoVLA 教师在训练解释标注中见未来动作，论文已披露，缺有/无该提示的分数对照；留在审计页与排除日志，与测试时未来真值输入分开。
- AD-MLP 新增中置信度的 `future_label_conditioning`：论文与数据代码由未来生成命令，预制 21 维 pkl 的字段链缺失。
- GTRS v2 回扫补入 `metric_proxy_candidate_selection`：训练目标包含 PDM 子分数，推理评分头参与候选选轨；旧协议消融不能解释修复后的 45.4。
- 首轮独立复核发现两处漏记：TOAD v1 所用 DrivoR 基座训练命令的长目标归 `benchmark_target_shaping`（权重训练史仅文档可追），TOAD v2 实际评测命令覆盖子分数权重归 `manual_scorer_reweighting`。后一类判据补明须偏离协议默认聚合，避免把 v1 的默认 PDMS 权重重复编码。
- 最终复核澄清评分代理两类的边界：TOAD 的初始固定候选 argmax 与回退是 CEM 搜索的组成部分，按正式最终输出路径只记 `metric_proxy_test_search`；DrivoR、GTRS、RAP 等固定候选最终择一仍记 `metric_proxy_candidate_selection`。
- 报告交叉核查补入 DrivoR v2 的 `benchmark_split_adaptation`：论文 §4.1、§4.2.4 与附录 Table 10 将手调推理权重和交叠 warmup 验证集连成证据链；验证集获主办方认可，不作违规推断。
