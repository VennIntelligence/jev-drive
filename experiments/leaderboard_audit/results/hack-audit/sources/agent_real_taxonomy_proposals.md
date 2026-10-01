
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
