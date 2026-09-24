# nuScenes / BEV-Planner++

- 仓库：[NVlabs/BEV-Planner，固定 commit 01c28d6](https://github.com/NVlabs/BEV-Planner/tree/01c28d6db56a178ee3a65bf017fe7996360ef026)。
- 论文：[Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?](../../papers/bev_planner.pdf)，核对 §3–§4、Table 1–3、附录中按转向命令分组的结果。
- 读过：根 README、`configs/bev_next/bev_planner_plus_plus.py`、`mmdet3d/datasets/pipelines/loading.py` 的 `LoadGTPlaner`、`mmdet3d/datasets/nuscenes_dataset.py` 的样本输入和评测入口、`mmdet3d/models/fbbev/detectors/bev_planner.py` 的测试路径、`mmdet3d/models/fbbev/planner_head/naive_planner.py` 的规划前向与输出、`tools/data_converter/nuscenes_converter.py` 的 ego 轨迹和状态构造。
- 未读：全部 BEV 感知骨干、nuScenes devkit 与其他模型的重实现细节；未运行模型、仓库代码或评测。

## 整体印象

论文主动指出 nuScenes 短时开环 L2 和碰撞率易受 ego 状态影响，并明确说提出的简化基线不适合真实部署。Table 1 中 BEV-Planner、BEV-Planner+、BEV-Planner++ 是同一方法族的不同 ego 状态配置；抽样表 0.35 m 对应 **BEV-Planner++**，不能把无 ego 状态的 BEV-Planner 0.55 m 当作同一个设置。论文另报路缘碰撞率 CCR，用于揭示常用指标无法充分衡量真实安全。

## 发现

- **REAL-NUS-004**，`future_label_conditioning`：测试数据管线由真值未来三秒轨迹的末端位置生成左转、右转、直行命令，并送入规划头及轨迹模态选择。真车推理时没有未来真值；可给车辆导航命令，但不能直接替代这条由未来标签生成的命令。论文提到 driving command，未说明 BEV-Planner++ 的测试命令来自待评测未来轨迹；没有去掉或改用可部署导航命令的独立消融。
- **REAL-NUS-005**，`ego_state_fusion`：`with_ego_status=True` 时，规划头把 ego 状态向量直接与视觉解码后的查询拼接；论文 Table 1 从 BEV-Planner+ 到 ++ 的配置差异是规划头额外融合状态，avg L2 0.46→0.35 m。这是论文内部配置对照，不能据此推断闭环部署增益；无状态的 0.55 m 是另一个配置。

两条完整记录和固定 commit 行号见 [agent_real.jsonl](../findings/agent_real.jsonl)。nuScenes 不同论文的 L2、碰撞率实现及 ego 输入协议可能不同，这里仅比较论文内部设置。

## 局限

未来命令确切改善多少没有消融；真值命令与可部署导航命令的差距需要同权重配对评测。数据转换器在场景首帧没有上一帧 pose 时用下一帧估计速度，在 CAN bus 失败的兜底分支也会读取第一个未来位移；触发比例未知，因此未作为独立发现。论文 Table 2 的图像/速度扰动针对 **VAD-Base**，不是 BEV-Planner++，不能挪作本方法的贡献数字。
