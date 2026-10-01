# 26 格独立静态复核

## 方法

以 out/taxonomy.md 的最终类别定义为判据，只读取 out/review.csv、out/repos.csv 和相应固定本地仓库快照、论文。逐格追溯训练或推理关键路径；所有 26 格对应的 15 个唯一仓库工作树 HEAD 均与 out/repos.csv 的 commit 一致。没有运行模型或仿真器，也没有下载数据或权重。这里的 yes 只证明可定位机制，不证明对榜单分数的因果增益；no 表示审阅关键路径未见该类机制。裁定前未读取其他审计员的发现、矩阵或报告。

## 逐项裁定

| # | 榜单 / 方法 | 类别 | 原值 → 复核 | 理由与定位 |
|---:|---|---|---|---|
| 1 | bench2drive / BLUE | manual_control_override | yes → yes | 闭环 agent 在速度低于 0.1 且卡住计数超过阈值后，连续 15 帧强制油门至少 0.4 并取消刹车，覆盖控制输出。 定位：repos/bench2drive__blue/team_code/agent_simlingo.py:861-879,897; team_code/config_simlingo.py:13-15 |
| 2 | nuscenes / SparseOccVLA | ego_state_fusion | yes → yes | 测试管线保留视觉、can_bus、temporal_ego_states；规划头将直接 ego 状态与视觉特征合并。 定位：repos/nuscenes__sparseoccvla/projects/configs/SparseOccVLA/sparseoccvla_stage3_4d_600q_forecasting.py:191-211; projects/mmdet3d_plugin/models/detectors/sparseoccvla.py:691,725-766 |
| 3 | nuscenes / AD-MLP | ego_only_open_loop_planning | yes → yes | 三秒开环规划从预制 21 维 ego/命令特征进入 MLP；关键推理路径不读相机或 LiDAR。 定位：repos/nuscenes__ad_mlp/pytorch/admlp/planner.py:178-186,203-227,245-271; README.md:31-34 |
| 4 | navsim_v2 / GTRS | metric_proxy_candidate_selection | yes → yes | PDM 子分数监督评分头；推理时对已生成的候选估计子分数并取代理总分最高者。 定位：repos/navsim_v2__gtrs/navsim/agents/gtrs_aug/hydra_loss_fn_aug.py:55-70; navsim/agents/gtrs_aug/hydra_model.py:263-307 |
| 5 | navsim_v2 / DrivoR+TOAD | metric_proxy_test_search | yes → yes | 实际配置开启 CEM；推理循环反复采样并 rollout 新轨迹，用预测 PDM 子分数及正则项选精英、更新分布。 定位：repos/navsim_v2__toad/navsim/planning/script/config/common/agent/drivoR.yaml:43-55; navsim/agents/drivoR/drivor_model.py:352-392 |
| 6 | nuscenes / AD-MLP | future_label_conditioning | yes → yes | 数据代码和论文显示命令由同样本未来位移生成，并进入 21 维规划输入；公开的预制推理 pkl 缺逐字段来源，结论应降置信度。 定位：repos/nuscenes__ad_mlp/pytorch/admlp/stp3/datas/NuscenesData.py:505-532,626-629; pytorch/admlp/planner.py:245-271; papers/ad_mlp.pdf PDF p.2 |
| 7 | bench2drive / TFv6 | simulator_protocol_dependency | no → no | 提交入口是 sensor_agent，直接输出 VehicleControl；未见共享模拟器轨迹转车控实现被此方法评测入口修补。 定位：repos/bench2drive__tfv6/slurm/evaluate.sh:13-23; lead/inference/sensor_agent.py:525-549 |
| 8 | navsim_v2 / DrivoR+TOAD | offline_candidate_library | no → no | 64 条初始轨迹由可学习 query 和网络产生，CEM 在线生成新轨迹；未见源域轨迹离线聚类固定词表。 定位：repos/navsim_v2__toad/navsim/planning/script/config/common/agent/drivoR.yaml:11-12; navsim/agents/drivoR/drivor_model.py:88-93,171-181,334-345 |
| 9 | nuscenes / AD-MLP | benchmark_target_shaping | no → no | 监督目标仍是 pkl 的普通未来 gt；0.5 米同格误差折半属于栅格损失调权，未改变监督终点或时距。 定位：repos/nuscenes__ad_mlp/pytorch/admlp/planner.py:47-78,199-240 |
| 10 | wod_e2e / RAP-DINO | benchmark_target_shaping | no → no | 轨迹损失对固定采样的真实未来做最小距离拟合；RFS 用于评分头标签，未见为 WOD 协议手改未来目标。 定位：repos/wod_e2e__rap/navsim/agents/pad/pad_features.py:113-129; navsim/agents/pad/pad_agent.py:445-501 |
| 11 | bench2drive / SparseDriveV2 | inference_failure_brake_fallback | no → no | run_step 从模型轨迹经 PID 输出 VehicleControl；关键路径没有推理异常时发送制动轨迹的分支。 定位：repos/bench2drive__sparsedrivev2_b2d/leaderboard/team_code/sparsedrive_b2d_agent.py:399-405,491-532 |
| 12 | navsim_v1 / RAP-DINO | metric_reward_finetuning | no → no | PDM/RFS 函数生成评分头监督并以 BCE 拟合；策略轨迹本身用模仿损失，未见榜单计分函数作策略微调奖励。 定位：repos/navsim_v1__rap/navsim/agents/pad/pad_agent.py:452-529 |
| 13 | hugsim / UniAD | simulator_pose_access | no → no | 客户端从 FIFO 接收 obs/info，确实把 info 中的 ego_pos/ego_rot 作变换，但没有直接访问仿真器 world/actor；按本类严格判据不计。 定位：repos/hugsim__uniad/tools/closeloop/e2e.py:219-249,270-277; tools/closeloop/dataparser.py:22-27,57-68 |
| 14 | navsim_v1 / DriveVLA-M0 | offline_candidate_library | no → no | 记忆模块检索失败案例用于适配；推理候选由可学习 query/轨迹头产生，未见源域轨迹聚类的大型固定库。 定位：repos/navsim_v1__drivevla_m0/navsim/agents/EpisodeDrive/action_decoder.py:154-169,198-213; README.md:42-54 |
| 15 | nuscenes / SparseOccVLA | ego_only_open_loop_planning | no → no | 测试路径明确读取多相机视觉并交给规划/VLM，故不满足只读 ego 运动与导航的定义。 定位：repos/nuscenes__sparseoccvla/projects/mmdet3d_plugin/models/detectors/sparseoccvla.py:300-371,589-628,696-714 |
| 16 | navsim_v1 / TOAD+DrivoR | benchmark_target_shaping | no → yes | TOAD 文档把加长目标的训练命令关联到 nav1 主基座 checkpoint，评测和提交加载该 checkpoint；源码目标构造与损失可达。训练示例写 agent=drivoR_speed_up，但该 Hydra 配置未随 TOAD 快照公开，实际权重训练史不能逐步复现。 定位：repos/navsim_v1__toad_nav1/README.md:65-99,170-174,251-255; repos/navsim_v1__drivor/README.md:59-88; repos/navsim_v1__toad_nav1/navsim/agents/drivoR/drivor_features.py:222-243; navsim/agents/drivoR/layers/losses/drivor_loss.py:252-271 |
| 17 | wod_e2e / RAP-DINO | offline_candidate_library | no → no | 64 条 proposal 来自可学习 embedding、网络细化后由评分头选取；未见离线聚类的源域轨迹库。 定位：repos/wod_e2e__rap/navsim/agents/pad/navsim_config.py:32; navsim/agents/pad/pad_model.py:96,124-136,160-167 |
| 18 | navsim_v2 / DrivoR+TOAD | future_label_conditioning | no → no | 推理特征取当前/历史 ego 状态及路线导航命令；仓库说明命令基于期望路线，未见从同一样本未来真值生成的测试标签。 定位：repos/navsim_v2__toad/navsim/agents/drivoR/drivor_features.py:60-85; navsim/common/dataclasses.py:199-206; docs/agents.md:79 |
| 19 | hugsim / WA-JEPA | simulator_pose_access | no → no | 客户端读取 FIFO 的 info[ego_box] 建历史运动特征，确实使用模拟器提供的位姿；未调用 world/actor 接口直接取位姿用于定位或路线，按严格判据不计。 定位：repos/hugsim__wa_jepa/close_loop/hugsim_client.py:173-194; close_loop/hugsim_planner.py:196-200,250-268 |
| 20 | navsim_v1 / DriveVLA-M0 | future_label_conditioning | no → no | 推理取 ego_status 的高层导航命令；协议文档说明它基于期望路线，未见测试时读取由同样本未来真值生成的标签。 定位：repos/navsim_v1__drivevla_m0/navsim/agents/EpisodeDrive/drivevla_features.py:80-94; docs/agents.md:79 |
| 21 | navsim_v2 / DrivoR+TOAD | manual_scorer_reweighting | no → yes | 实际 NAVSIM-v2 评测命令覆盖六项 PDM 子分数权重并开启 CEM；推理用手设权重算候选分数并按综合分选精英，另有 comfort/anchor 手设系数。 定位：repos/navsim_v2__toad/README.md:150-179; navsim/planning/script/config/common/agent/drivoR.yaml:20-25,54-55; navsim/agents/drivoR/drivor_model.py:209-220,352-371 |
| 22 | hugsim / UniAD | control_interface_selection | no → no | 闭环入口只取 planning_head 的单条 sdc_traj 回传；未见同一网络多控制接口之间的选择。 定位：repos/hugsim__uniad/tools/closeloop/e2e.py:247-273; projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:130-199 |
| 23 | wod_e2e / DriveMA-4B | offline_candidate_library | no → no | VLM 根据图像与运动历史生成文本轨迹，提交脚本插值并写出；未见源域轨迹聚类固定候选库参与推理。 定位：repos/wod_e2e__drivema/tools/infer_scripts/run_vllm_infer_VA.py:52-75,636-652; tools/other/convert_to_submission.py:135-188 |
| 24 | hugsim / UniAD | metric_proxy_candidate_selection | no → no | 多运动 query 先融合成一个规划 query，再回归单条 sdc_traj；没有榜单子分数代理头对已有候选排序。 定位：repos/hugsim__uniad/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:160-200; tools/closeloop/e2e.py:247-255,270-273 |
| 25 | hugsim / LTF | inference_failure_brake_fallback | no → no | 推理 RuntimeError 后 traj=None，客户端发送 None 并退出；未发送全零制动轨迹以继续计分。 定位：repos/hugsim__ltf/ltf_e2e.py:56-80 |
| 26 | hugsim / UniAD | metric_proxy_test_search | no → no | 可选碰撞优化依据预测占用调整单轨迹，非以榜单评分代理为目标反复搜索新轨迹。 定位：repos/hugsim__uniad/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:193-238; tools/closeloop/e2e.py:270-273 |

## 一致率和分歧

逐格同意 24/26，一致率 92.31%。原 yes 的 6 格均复核为 yes；原 no 的 20 格中 18 格仍为 no，2 格改判 yes；没有 NA。

1. **第 16 格：navsim_v1 / TOAD+DrivoR / benchmark_target_shaping，no → yes。** 此格按 TOAD 所用的冻结 DrivoR 基座判断。TOAD 仓库 README.md:65-99 把 long_trajectory_additional_poses=2 的训练命令关联到 nav1 主 checkpoint；同库 drivor_features.py:222-243 读取额外未来点并插值成目标，drivor_loss.py:252-271 把它加入轨迹回归。README.md:170-174、251-255 的评测与提交命令加载该 checkpoint。DrivoR 原仓库 README.md:59-88 的 agent=drivoR 训练命令也启用此目标，论文 PDF 第4页公式(2)、第9页表7讨论目标与 navval PDMS 的关系。复现限制：TOAD 文档的训练示例指定 agent=drivoR_speed_up，但固定快照没有对应 Hydra 配置，因此源码不能逐步验证该 checkpoint 的训练史。仍建议主审查员按文档机制将此格改为 yes，明确证据依赖 checkpoint 来源说明，且不把论文消融当作正式提交的因果效应。

2. **第 21 格：navsim_v2 / DrivoR+TOAD / manual_scorer_reweighting，no → yes。** 仓库 README.md:150-179 的实际 NAVSIM-v2 评测命令显式覆盖 noc=10、dac=13、ddc=6、ttc=14、ep=15、comfort=2，并开启 CEM；drivor_model.py:209-220 用这些权重给初始候选排序，:352-371 用同类评分及手设 comfort/anchor 系数选 CEM 精英。配置 drivoR.yaml:20-25,54-55 给出默认值。建议主审查员将此格改为 yes，说明是推理排序权重存在，不推断这些系数曾在正式测试集调参或证明分数增益。

## 解释边界

第 13 格 UniAD 与第 19 格 WA-JEPA 均从 HUGSIM FIFO 的 info 读取模拟器提供的真位姿来构造输入；现行 simulator_pose_access 判据要求直接读取 world/actor 的位姿或变换，所以复核为 no。若研究范围改成“一切模拟器真位姿输入”，应统一扩展类别定义后重扫这些方法，不能把本次 no 解读为完全不使用真位姿。

第 6 格 AD-MLP 的论文和数据代码可追溯由未来位移生成命令，但公开预制推理 pkl 缺逐字段构建链。按现行 taxonomy 的判据复核为 yes，并应保留中置信度与“未直接证明该预制字段在正式评测触发”的限制。
