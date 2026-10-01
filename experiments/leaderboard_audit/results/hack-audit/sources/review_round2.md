# 最终矩阵抽样独立复核

- 抽样：27 格（原 yes 7，原 no 20）；随机种子 `20260924`。
- 同意：27/27，一致率 100.0%；原 yes 同意 7/7；原 no 同意 20/20。
- 全部分歧：无。

## 逐项裁定

| # | 榜单 / 方法 | 类别 | 原判 → 复核 | 独立依据 |
|---:|---|---|---|---|
| 1 | `bench2drive` / `BLUE` | `manual_control_override` | `yes` → **`yes`** | 公开评测选用 LingoAgent；PID 后若静止计数超过阈值，强制最小油门并取消刹车。<br>证据：`repos/bench2drive__blue/gate/evaluation/eval_blue_full.sh:23,237; repos/bench2drive__blue/team_code/agent_simlingo.py:58,861-879` |
| 2 | `nuscenes` / `BEV-Planner++` | `ego_state_fusion` | `yes` → **`yes`** | ++ 配置在测试管线传入 ego 运动特征；视觉 BEV 规划头将其拼入 ego_query。<br>证据：`repos/nuscenes__bev_planner/configs/bev_next/bev_planner_plus_plus.py:184-186,289; repos/nuscenes__bev_planner/mmdet3d/models/fbbev/planner_head/naive_planner.py:245,300-304` |
| 3 | `navsim_v2` / `TOAD` | `manual_scorer_reweighting` | `yes` → **`yes`** | NAVSIM-v2 评测命令覆盖 NC/DAC/DDC/TTC/EP 权重，模型用这些系数给候选及 CEM 轨迹排序；偏离默认聚合。<br>证据：`repos/navsim_v2__toad/README.md:150-179; repos/navsim_v2__toad/navsim/agents/drivoR/drivor_model.py:207-218; repos/navsim_v2__toad/navsim/planning/script/config/common/agent/drivoR.yaml:20-25` |
| 4 | `navsim_v2` / `GTRS` | `manual_scorer_reweighting` | `yes` → **`yes`** | Navhard 的 GTRS-Aug 推理入口进入手写子项系数公式并 argmax；公式含 imitation 与非协议默认的 NC/DAC/DDC/TTC/EP 等权重。<br>证据：`repos/navsim_v2__gtrs/docs/gtrs_inference.md:83-120; repos/navsim_v2__gtrs/navsim/agents/gtrs_aug/hydra_model.py:289-307; repos/navsim_v2__gtrs/docs/metrics.md:28-33` |
| 5 | `navsim_v2` / `TOAD` | `benchmark_split_adaptation` | `yes` → **`yes`** | 论文以 warmup-two-stage 作 CEM 消融/配置依据，并明确该划分与 navhard-two-stage 交集；论文同时记载榜单作者认可其作验证集。<br>证据：`papers/toad.pdf:p.5及Fig.4; repos/navsim_v2__toad/docs/splits.md:79-94` |
| 6 | `navsim_v2` / `TOAD` | `metric_proxy_test_search` | `yes` → **`yes`** | 公开评测开启 use_cem，循环采样新控制、预测榜单子分数、挑选精英并更新分布，最终输出新轨迹或基座回退。<br>证据：`repos/navsim_v2__toad/README.md:175-179; repos/navsim_v2__toad/navsim/agents/drivoR/drivor_model.py:226-245,340-392` |
| 7 | `carla_lb2` / `TF++` | `manual_control_override` | `yes` → **`yes`** | 闭环 sensor agent 默认启用停车牌控制，且卡住后可强制蠕行油门或安全箱制动；这些覆盖 PID 输出。<br>证据：`repos/carla_lb2__tfpp/team_code/sensor_agent.py:126-130,630-676` |
| 8 | `nuscenes` / `AD-MLP` | `benchmark_target_shaping` | `no` → **`no`** | 训练路径用真实未来轨迹与普通 L1 监督；未见针对榜单时距、终点或子指标改造的轨迹目标。<br>证据：`repos/nuscenes__ad_mlp/pytorch/admlp/train.py:48-69; repos/nuscenes__ad_mlp/pytorch/admlp/planner.py:199-240` |
| 9 | `nuscenes` / `AD-MLP` | `manual_scorer_reweighting` | `no` → **`no`** | 推理直接回归单条轨迹；代码未给已有候选的榜单子项手工加权排序。<br>证据：`repos/nuscenes__ad_mlp/pytorch/admlp/planner.py:227-240,246-272` |
| 10 | `wod_e2e` / `RAP` | `manual_scorer_reweighting` | `no` → **`no`** | Waymo 提交路径按学习到的单一分数头 argmax；无手写 NC/DAC 等子项系数公式。<br>证据：`repos/wod_e2e__rap/README.md:198-215; repos/wod_e2e__rap/navsim/agents/rap_dino/rap_model.py:156-167` |
| 11 | `bench2drive` / `SparseDriveV2` | `inference_failure_brake_fallback` | `no` → **`no`** | 正式 agent 直接执行模型与 PID 并返回控制；没有捕获推理异常后发送全零制动轨迹的路径。<br>证据：`repos/bench2drive__sparsedrivev2_b2d/leaderboard/team_code/sparsedrive_b2d_agent.py:489-512` |
| 12 | `navsim_v1` / `RAP` | `metric_reward_finetuning` | `no` → **`no`** | NAVSIM-v1 公开训练命令启用 pdm_scorer，计算分数用于监督评分头；未把榜单函数作为策略 RL 奖励。<br>证据：`repos/navsim_v1__rap/README.md:140-155; repos/navsim_v1__rap/navsim/agents/rap_dino/rap_agent.py:450-481,518-530` |
| 13 | `hugsim` / `UniAD_SIM` | `simulator_pose_access` | `no` → **`no`** | 客户端从主机传来的 info 读取 ego_pos 并组装模型输入；未调用 world/actor 读取实时真实位姿或变换。<br>证据：`repos/hugsim__uniad/tools/e2e.sh:8-11; repos/hugsim__uniad/tools/closeloop/dataparser.py:21-70` |
| 14 | `navsim_v1` / `DriveVLA-M0` | `offline_candidate_library` | `no` → **`no`** | 离线 memory 储存失败案例的潜在特征/监督并作适应；推理候选由动作解码器生成，而非源域离线聚类固定大词表。<br>证据：`papers/drivevla_m0.pdf:5-7; repos/navsim_v1__drivevla_m0/navsim/agents/EpisodeDrive/action_decoder.py:154-207` |
| 15 | `nuscenes` / `SparseOccVLA` | `metric_grid_alignment` | `no` → **`no`** | 规划训练使用 anchor 分类与轨迹差分 L1，未按碰撞占用评测的离散格尺度折减同格误差。<br>证据：`repos/nuscenes__sparseoccvla/projects/configs/SparseOccVLA/sparseoccvla_stage3_4d_600q_forecasting.py:94-101; repos/nuscenes__sparseoccvla/projects/mmdet3d_plugin/models/detectors/sparseoccvla.py:544-562` |
| 16 | `navsim_v1` / `TOAD` | `future_label_conditioning` | `no` → **`no`** | 测试特征来自 AgentInput 的当前/历史 ego 状态和导航命令；未来轨迹仅在 target builder 中作训练监督。<br>证据：`repos/navsim_v1__toad_nav1/navsim/agents/drivoR/drivor_features.py:38-60,207-229` |
| 17 | `navsim_v2` / `TOAD` | `metric_grid_alignment` | `no` → **`no`** | 轨迹损失是连续坐标 L1 与多样性；式中 /0.5 是连续距离的评分辅助量，没有离散格内误差调权。<br>证据：`repos/navsim_v2__toad/navsim/agents/drivoR/layers/losses/drivor_loss.py:260-284` |
| 18 | `hugsim` / `WA-JEPA` | `simulator_pose_access` | `no` → **`no`** | 客户端使用主机 info 的 ego_box 构建历史位姿；未直接从仿真器 world/actor 查询位姿。<br>证据：`repos/hugsim__wa_jepa/close_loop/hugsim_client.py:173-211; repos/hugsim__wa_jepa/close_loop/hugsim_planner.py:197-200` |
| 19 | `navsim_v1` / `DriveVLA-M0` | `future_label_conditioning` | `no` → **`no`** | 测试提示命令取当前 ego_status.driving_command；未来轨迹另在 target builder 中生成，未证同一样本未来真值形成测试命令。<br>证据：`repos/navsim_v1__drivevla_m0/navsim/agents/EpisodeDrive/drivevla_features.py:85-106,150-160; repos/navsim_v1__drivevla_m0/navsim/common/dataclasses.py:200-204` |
| 20 | `navsim_v2` / `TOAD` | `metric_proxy_candidate_selection` | `no` → **`no`** | DrivoR 先选基座候选作 CEM 锚点/回退；TOAD 正式推理反复生成并评估新轨迹，按 codebook 归测试搜索而非固定候选选择。<br>证据：`repos/navsim_v2__toad/navsim/agents/drivoR/drivor_model.py:218-245,340-392; repos/navsim_v2__toad/README.md:175-179` |
| 21 | `hugsim` / `UniAD_SIM` | `control_interface_selection` | `no` → **`no`** | 客户端只取网络生成的 sdc_traj 作为输出；未在多种控制接口间选一路径再接手工控制器。<br>证据：`repos/hugsim__uniad/tools/closeloop/e2e.py:250-274; repos/hugsim__uniad/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:175-198` |
| 22 | `wod_e2e` / `RAP` | `benchmark_target_shaping` | `no` → **`no`** | Waymo 微调命令设 5 秒轨迹时域，损失直接拟合未来轨迹；未见专为榜单终点/子指标手工变形的监督目标。<br>证据：`repos/wod_e2e__rap/README.md:175-191; repos/wod_e2e__rap/navsim/agents/rap_dino/rap_agent.py:485-501` |
| 23 | `hugsim` / `UniAD_SIM` | `metric_proxy_candidate_selection` | `no` → **`no`** | 规划 query 聚合后回归一条 sdc_traj；未训练榜单子分数代理头来重排已有候选。<br>证据：`repos/hugsim__uniad/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:175-198; repos/hugsim__uniad/tools/closeloop/e2e.py:250-274` |
| 24 | `hugsim` / `NAVSIM` | `inference_failure_brake_fallback` | `no` → **`no`** | RuntimeError 时发送 None 并退出客户端，没有发送全零轨迹使场景继续计分。<br>证据：`repos/hugsim__ltf/ltf_e2e.py:56-78` |
| 25 | `hugsim` / `UniAD_SIM` | `metric_proxy_test_search` | `no` → **`no`** | 测试可做碰撞优化，但目标是预测占用避碰，不调用榜单评分代理反复搜索新轨迹。<br>证据：`repos/hugsim__uniad/projects/configs/stage2_e2e/base_e2e.py:57,440-449; repos/hugsim__uniad/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py:193-235` |
| 26 | `bench2drive` / `BLUE` | `simulator_pose_access` | `no` → **`no`** | 闭环定位由 GPS/IMU 与 UKF 及公开全局路线完成；未直接查询 CARLA actor 的真实位姿。<br>证据：`repos/bench2drive__blue/team_code/agent_simlingo.py:351-401,468-494,1943-1949` |
| 27 | `bench2drive` / `SparseDriveV2` | `multi_checkpoint_candidate_selection` | `no` → **`no`** | 评测 agent 从配置读取一个 ckpt_path 并加载一个模型；未汇合多个 checkpoint 候选后全局选轨。<br>证据：`repos/bench2drive__sparsedrivev2_b2d/leaderboard/team_code/sparsedrive_b2d_agent.py:70-113,489-512` |

## 代码可达性与判定边界

此次只静态阅读抽中条目的固定本地提交与相应论文；没有运行模型、评测或仿真器，也没有下载权重/数据。`yes` 确认公开代码和论文中的机制及其入口，不证明其在每次运行都触发或独立提高主分数。

- BLUE 以 `gate/evaluation/eval_blue_full.sh` 指定的 `LingoAgent` 为准；同文件的 `ShadowLingoAgent` 注释掉的蠕行代码不计入证据。TF++ 的默认停车牌控制和卡住恢复在闭环 sensor agent 的实际返回控制前生效。
- TOAD v2 以 README 的 `navhard_two_stage` 评测/提交命令为准：`use_cem=true` 且覆盖评分子项权重。初始 DrivoR 选轨提供 CEM 锚点与回退，不把该正式 TOAD 单元重复编码为固定候选重排。其 warmup 划分与 navhard 有交集，论文明确说明榜单作者认可用作验证集；这不等于训练集泄漏。
- RAP 的代码保留多种榜单路径。NAVSIM-v1 公开训练启用 `pdm_scorer=True`，分数用于学习评分头；WOD-E2E 微调命令设 `pdm_scorer=False`，RFS 计算用于分数头监督，二者均非策略 RL 奖励。WOD 提交用学习分数 argmax，不能据此判手工子项改权。
- UniAD 的 HUGSIM 客户端和 WA-JEPA 均接收仿真主机 `info` 中的真实 ego 信息；本类别的窄定义要求 agent 直接访问 world/actor 位姿接口，因此复核为 `no`。此裁定不否认主机信息可能影响定位或历史运动输入。BLUE 的 world-coordinate 全局路线只用于 GPS 参考/导航，不是实时 actor 位姿。
- UniAD 的可选碰撞优化使用预测占用，不以榜单分数代理为目标；SparseDriveV2 的单模型闭环脚本没有推理异常制动轨迹，也没有多 checkpoint 候选汇合；LTF 出错时发送 `None` 并退出。
- DriveVLA-M0 的离线失败案例 memory 用于检索与适应，候选由动作解码器生成；测试导航命令来自当前 `ego_status`，现有本地代码不能证明它取自同一样本未来真值。
- AD-MLP 和 SparseOccVLA 的普通轨迹 L1 监督不足以构成榜单目标塑形或手工评分改权；TOAD 的连续坐标损失及 `/0.5` 辅助量不是按离散评测栅格进行误差折半。
