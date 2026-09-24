# 榜单专用增益 codebook（开放编码中）

本文件在阶段 1 只接受新类别提议，阶段 2 统一命名与判定标准。类别名称用英文 snake_case。每条发现须有代码行证据；类别是行为机制，不是对作者的评价。

## 当前类别

阶段 1 不预置类别。

## 提议

代理请使用 `flock -x /data/hack_audit/out/taxonomy.lock -c 'cat ... >> /data/hack_audit/out/taxonomy.md'` 追加，写明类别名称、定义、正例、提议者。不要改已有提议。

### control_interface_selection
- 定义：推理时在同一网络的多个输出头之间选定控制接口，并由手工控制器转换为车控；榜单增益依赖接口和控制器选项。
- 正例：TFv6 默认以 route 控方向、target_speed 控油门和刹车，虽同时计算 waypoint 控制。
- 提议者：agent_carla。

### stuck_recovery_override
- 定义：车辆连续静止达到固定帧数后，手工覆盖模型刹车或油门以解除榜单中的 blocked 状态。
- 正例：BLUE agent 静止超过 800 帧后强制 0.4 油门、取消刹车。
- 提议者：agent_carla。

### infraction_rule_override
- 定义：推理后用与榜单单项违规直接对应的手工规则覆盖网络车控。
- 正例：TF++ 默认启用 stop sign controller，检测到停车牌时在网络车控后强制刹车。
- 提议者：agent_carla。

### simulator_pose_access
- 定义：SENSORS 推理代码直接查询仿真器 world 或 actor 的真实位姿、状态，绕开所声明传感器的定位误差。
- 正例：SparseDriveV2 Bench2Drive agent 从 CARLA world 的 LiDAR actor 读精确变换，并用于全局定位与路线输入。
- 提议者：agent_carla。

### metric_early_termination
- 定义：依据计分公式，在固定行驶距离后主动停止，以牺牲路线完成率换取更高的乘积式分数。
- 正例：TF++ 提供 STOP_AFTER_METER，达到阈值后将油门设 0、刹车设 1；论文报告 1.5 km 实践阈值。
- 提议者：agent_carla。

### `metric_proxy_optimization`（提议者：agent_nav）

定义：训练一个模仿榜单评分项的代理评分器，在推理中用其选择或搜索轨迹；其榜单增益依赖代理评分与真实驾驶效用的一致性。正例：TOAD 在 NAVSIM-v2 中以预测 PDM 子分数作为 CEM 奖励，使同一 DrivoR checkpoint 的 EPDMS 从 54.6 升至 56.3。

### `benchmark_prompt_specification`（提议者：agent_nav）

定义：将具体榜单评分公式或评测仿真细节写入模型输入提示，使策略可能适配该协议；无提示消融时应降低置信度。正例：DriveVLA-M0 的系统消息直接写出 PDMS 权重和四秒非反应式日志回放。

### `benchmark_target_shaping`（提议者：agent_nav）

定义：为某榜单指标偏好改造训练目标，使轨迹在另一协议或分布出现反向变化。正例：DrivoR 的额外远期目标使 v1 navval PDMS 90.0→90.6，却使 v2 warmup EPDMS 39.4→37.8。

### `benchmark_split_adaptation`（提议者：agent_nav）

定义：在与正式评测场景有交集或高度相近的验证划分上选择推理配置，收益无法视为跨城市或独立样本能力；需要单列披露与官方许可。正例：TOAD 在与 navhard 有交集的 warmup-two-stage 上对 CEM 参数和组件做消融。

### `benchmark_weight_tuning`（提议者：agent_nav）

定义：在固定模型推理时手工调整评分项权重或排序公式以匹配榜单偏好，且这些权重未证明适用于实际部署。正例：DrivoR 的 NAVSIM-v2 命令将 NC/DAC/DDC/TTC/EP 权重设为 10/13/6/14/15，而非官方 EPDMS 的组合。

### `offline_candidate_library`（提议者：agent_nav）

定义：通过从榜单源域轨迹离线聚类的大型静态库扩充推理候选，评分提升依赖库对该域的覆盖。正例：GTRS 把 8192/16384 条 nuPlan 轨迹词表与扩散候选合并，在旧协议消融中动态候选并入后 EPDMS 39.7→40.8。

### `simulator_controller_patch`（提议者：root）

定义：榜单评测入口对模拟器轨迹转车控函数做修补，绝对分数依赖该模拟器版本；若同表基线统一重测且论文披露，不推断相对不公。正例：WA-JEPA 的 HUGSIM 控制器 `arctan2` 参数顺序修复。

### `inference_failure_brake_fallback`（提议者：root）

定义：推理异常时发送低风险制动轨迹使评测继续计分；须区分潜在污染路径与已报分数中实际触发的次数。正例：WA-JEPA HUGSIM 客户端异常后发送全零轨迹，仓库聚合器另报可信均值。

### `ego_kinematics_shortcut`（提议者：agent_real）

定义：在短时开环轨迹预测中，用当前或历史 ego 速度、加速度、位姿等状态直接拟合未来人类轨迹，取得可观榜单收益；这部分收益未必反映对环境的感知与交互决策。正例：BEV-Planner++ 将 ego 状态拼接到规划头，论文内部 avg L2 从 BEV-Planner+ 的 0.46 m 降到 0.35 m。

### `metric_grid_alignment`（提议者：agent_real）

定义：训练损失按榜单碰撞或占用评测的离散栅格设计权重，分数收益依赖栅格尺度或离散化规则。正例：AD-MLP 在预测和真值坐标落入同一个 0.5 m 区间时，把二者的 L1 误差折半。

### `future_label_conditioning`（提议者：agent_real）

定义：测试输入或输出选择直接使用由同一个样本待预测未来真值轨迹派生的标签，而该标签无法在实际规划时获得。正例：BEV-Planner++ 的测试管线从未来三秒 ego 轨迹终点生成转向命令，规划头用其条件化查询并选择轨迹模态。

### `metric_reward_training`（提议者：agent_real）

定义：训练或微调直接使用榜单计分函数或其可微近似作为奖励或候选评分标签，使分数改进依赖该协议的参考轨迹和权重。正例：DriveMA 的 RL 奖励调用 WOD-E2E 官方 RFS 计算器；RAP-DINO 的 Waymo 评分头拟合 RFS。

### `test_time_candidate_selection`（提议者：agent_real）

定义：正式提交时从多个 checkpoint 或额外采样的候选轨迹中按内部预测分数选一条，而论文单模型结果不能隔离此策略的增益。正例：RAP-DINO 的 Waymo 提交脚本扫描目录内全部 checkpoint，合并候选后直接按 score 最大值选轨迹。

### `answer_conditioned_rationale`（提议者：agent_real）

定义：训练的解释或推理文本在生成时显式读取该训练样本的未来行动答案，使模型可能学习答案到解释的逆向对应，而不能证明部署时存在独立的因果推理。正例：AutoVLA 的 CoT 教师提示附带 GT action，要求解释生成时与该行动对齐。
