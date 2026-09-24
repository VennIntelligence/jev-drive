# 榜单分数中可见的协议依赖：报告主体草案

## 范围与读法

截至 2026-09-24，抽样日志覆盖 7 个榜单、36 个候选条目；入选 20 个有代码的方法榜单单元（A 档 16、B 档 4），另记录 14 个 C 档排除项与两个 privileged 规则基线。每个单元固定仓库 commit，对照论文、README、推理/训练/评测路径，只作静态审计；没有运行模型、仿真器或评测，也没有下载权重和数据。交付依据为 `out/sampling.csv`、`out/repos.csv`、20 页 `out/audits/`、`out/findings.jsonl`、`out/taxonomy.md` 和 `out/matrix.csv`。

最终 codebook 有 19 类。矩阵是 20×19，共 29 个 `yes`，落在 17 个单元上；另有 214 个 `no`、137 个 `NA`。**`yes` 只表示固定代码有对应机制且有发现记录，不表示已证明该机制提高论文主分数；`no` 表示适用且已读关键路径未见；`NA` 表示协议不适用或公开代码不足以判断。** 31 条最终发现逐条记载代码位置、部署可用性、披露、消融数字或 `none`。这些计数不能相加为“分数中 hack 的比例”：同一方法可能有多项机制，很多发现缺同权重消融，而且不同榜单、划分和指标不能互换。

## 七个榜单

| 榜单 | 入选与覆盖 | 主要可见机制 | 成绩解释的边界 |
| --- | --- | --- | --- |
| Bench2Drive 0.0.3，闭环 DS | 3 项：TFv6、BLUE、SparseDriveV2；A 2、B 1。4 个矩阵 `yes`、5 条发现。 | TFv6 选 route+target speed 控制接口并在复现配置启用蠕行、停车牌规则；BLUE 也有静止后强制油门；SparseDriveV2 的 SENSORS agent 从 CARLA world 读取 LiDAR actor 精确位姿。 | TFv6 的启发式联合开关约对应 95→94 DS，没有单项贡献；BLUE 的蠕行触发率未知；SparseDriveV2 未做真值位姿替换消融。未把其未进入模型 Collect 的俯视图当成输入泄漏。 |
| CARLA Leaderboard 2.0，闭环 DS | 仅 TF++ 1 项 A 档；CarLLaVA、SimLingo-BASE 缺对应分数的公开闭环入口，Kyber-E2E 无 agent 代码，依抽样规则如实少于 3 项。2 个 `yes`、3 条发现。 | 论文披露按行驶距离提前停车以利用 DS/路线完成率的乘积结构；agent 还用停车牌刹车与卡住后蠕行规则覆盖模型控制。 | 抽样 5.56 DS 属 MAP，固定脚本默认 SENSORS；提前终止示例 0.96→5.10 DS 来自 Town13 validation，不能当官方 MAP test 的同配置增益。 |
| NAVSIM v1，navtest PDMS | 4 项 A 档：TOAD+DrivoR、DriveVLA-M0、RAP-DINO、DrivoR。6 个 `yes`、6 条发现。 | 三项以预测 PDMS/子分数重排已有轨迹；TOAD `nav1` 分支用 CEM 再生成候选；DrivoR 为进度偏好扩展远期训练目标；DriveVLA-M0 的推理系统消息写明 PDMS 和非反应式回放。 | TOAD 论文 94.7 与后续 `nav1` README 94.9 对应不同候选/迭代/评分器；DriveVLA-M0 抽样 94.1 属 Scale，固定公开链主要是 Base 92.3。不能把可见 Base 路径直接解释为 Scale 增益。 |
| NAVSIM v2，navhard 双阶段 EPDMS | 3 项 A 档：DrivoR+TOAD、DrivoR、GTRS。6 个 `yes`、6 条发现。 | TOAD 用五轮 CEM 优化学习的榜单代理分数，并在与 navhard 有交集的 warmup 上选参数；DrivoR 手工改候选评分权重；GTRS 合并离线大轨迹词表与动态候选，再用手定公式细排。 | TOAD 论文同基模型 54.6→56.3 EPDMS；交集验证划分获主办方认可，不能等同违规，也没有其独立提分数字。GTRS 固定论文/README 为旧协议 42.1，后续修复版转录为 45.4；旧版候选池消融不能拆分 45.4。 |
| nuScenes 短时开环规划 | 3 项 A 档：SparseOccVLA、AD-MLP、BEV-Planner++。6 个 `yes`、6 条发现。 | AD-MLP 完全不读环境感知，另以未来 GT 衍生高层命令；SparseOccVLA 与 BEV-Planner++ 在视觉规划中融合 ego 状态；BEV-Planner++ 本地测试管线还用未来真值命令条件化并选模态；AD-MLP 损失对齐 0.5 m 栅格。 | AD-MLP 不带命令的 ego 状态 avg L2 为 0.35 m，完整模型修正后为 0.29 m；不能把 0.29 全归历史运动。BEV 公开 test pipeline 实读 val 标注，命令贡献无单项消融。三篇 L2/碰撞率的样本与实现协议不一定相同，勿按小数直接比较驾驶能力。 |
| WOD-E2E，开放环 RFS | 3 项：DriveMA-4B、RAP-DINO、AutoVLA；A 2、B 1。3 个 `yes`、3 条最终发现；AutoVLA 最终无确认项。 | DriveMA 在 RL 奖励中调用官方 RFS；RAP 以 RFS 训练轨迹评分头，并在提交时汇合多个 checkpoint 候选后选最高分。 | RFS 是未来路线/人类偏好评分，不是闭环道路驾驶。DriveMA 训练阶段差值包含多种奖励和优化；RAP 论文写 NMS ensemble，公开脚本实际是全候选 argmax，且无单/双 checkpoint RFS 差。AutoVLA 的 Waymo 专用提交路径未公开，训练 CoT 教师看未来动作的候选观察因缺独立提分证据未纳入最终矩阵。 |
| HUGSIM，436 场景闭环 HD-Score | 3 项：WA-JEPA、UniAD、LTF；A 1、B 2。2 个 `yes`、2 条发现，均在 WA-JEPA；其余两项在已读路径中无确认的有利分数机制。 | WA-JEPA 评测入口修正模拟器轨迹转航向控制器，并在推理失败时发送制动轨迹让场景继续计分。 | 论文说明控制器修正对同表所有方法统一重测，不能说 WA-JEPA 获相对优势；缺逐场景失败日志，不能说 0.4462 已被制动回退抬高。UniAD/LTF 固定适配代码不含完整 436 场景复测配置，LTF 固定直行命令可能反而伤害转弯成绩。 |

## 分类体系概览

| 机制簇 | 最终类别及矩阵 `yes` 数 | 解释 |
| --- | --- | --- |
| 榜单指标直接进入决策 | `metric_proxy_candidate_selection` 5、`metric_proxy_test_search` 2、`metric_reward_finetuning` 1 | 分别是学习评分头重排固定候选、测试时反复搜索新轨迹、把官方分数直接用于策略微调；RAP 的 RFS 评分头属于第一种，DriveMA 的 RFS RL 属第三种。 |
| 训练目标、排序式与样本覆盖 | `manual_scorer_reweighting` 2、`benchmark_target_shaping` 1、`metric_grid_alignment` 1、`benchmark_prompt_specification` 1、`benchmark_split_adaptation` 1、`offline_candidate_library` 1、`multi_checkpoint_candidate_selection` 1 | 这几类分别定位手工评分权重、目标时距、评测栅格、推理提示、验证划分、离线词表和多权重提交；存在代码机制时仍须独立核对分数效应。 |
| 开环 ego 与未来标签 | `ego_only_open_loop_planning` 1、`ego_state_fusion` 2、`future_label_conditioning` 2 | 当前/历史 ego 状态在车上可能取得，但未来同样本的真值命令不可取得；AD-MLP 的两个来源须分别编码。 |
| 闭环接口、手工规则与仿真协议 | `control_interface_selection` 1、`manual_control_override` 3、`simulator_pose_access` 1、`metric_early_termination` 1、`simulator_protocol_dependency` 1、`inference_failure_brake_fallback` 1 | 规则和接口可能有真实用途；CARLA world 真值位姿及为乘积计分主动停车不能直接部署。HUGSIM 共享协议修正不表示相对同表基线优势。 |

## 值得注意的十条发现

1. **[B2D-TFV6-001](https://github.com/kesai-labs/lead/blob/730bc1a2f44d5f28312dd55f0ca958e94a24c038/lead/inference/config_closed_loop.py#L29-L35)，控制接口。** TFv6 同时计算 waypoint 跟踪与 route+target speed 控制，默认选择后者。论文披露 target speed 头；没有同一 checkpoint 两接口的 Bench2Drive DS 配对数，不能把 95.28 DS 差额归给接口。
2. **[B2D-SPARSE-001](https://github.com/swc-17/SparseDriveV2/blob/e42ea59dd4946238dc65097495a9aa0708121fae/leaderboard/team_code/sparsedrive_b2d_agent.py#L360-L370)，仿真器真值位姿。** 声称 SENSORS 的 SparseDriveV2 agent 向 CARLA world 查询 LiDAR actor 的精确 transform 并用于路线与模型元数据；真车无法查询。论文未披露，且没有改用 GNSS/IMU 的同模型 DS 消融。
3. **[LB2-TFPP-001](https://github.com/autonomousvision/carla_garage/blob/f22bc491b3094792aef475149e09a94dcbb526f9/team_code/sensor_agent.py#L673-L676)，乘积指标驱动的提前终止。** TF++ 在固定距离后主动刹停；论文披露实践阈值 1.5 km。Town13 validation 同模型 DS 0.96→5.10，说明指标激励明显；公开脚本默认不开此开关，官方 MAP 提交的参数未固定。
4. **[NAV1-DRIVOR-001](https://github.com/valeoai/DrivoR/blob/fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a/navsim/agents/drivoR/drivor_features.py#L224-L236)，进度目标的跨协议反向效果。** DrivoR 延长目标使 v1 navval PDMS 90.0→90.6，却使 v2 warmup 双阶段 EPDMS 39.4→37.8；这不是 v1 navtest 93.7 的直接分解，却是目标依赖榜单偏好的内部证据。
5. **[NAV2-TOAD-001](https://github.com/valeoai/TOAD/blob/cfa88e008080d0799b2859b5935c257048ec0adf/navsim/agents/drivoR/drivor_model.py#L357-L368)，测试时评分代理搜索。** TOAD 的 CEM 用冻结评分器产生新候选，论文同基模型 navhard EPDMS 54.6→56.3（+1.7）。增益来自额外推理优化，评分代理在真实交互中是否同样可靠未验证；warmup 与 navhard 的交集已在论文披露并获主办方认可。
6. **[REAL-NUS-006](https://github.com/E2E-AD/AD-MLP/blob/4b93ba085ee47474152f282177865796ea577fc0/pytorch/admlp/stp3/datas/NuscenesData.py#L523-L532)，未来标签进入开环输入。** AD-MLP 论文规定按未来三秒 ego 位移生成转向命令；无命令 0.35 m/0.23% 到有命令 0.29 m/0.19% 是其 Table 1 对照。推理所读 21 维 pkl 的生成脚本未公开，逐 token 输入来源不能核实，故为中置信度。
7. **[REAL-NUS-004](https://github.com/NVlabs/BEV-Planner/blob/01c28d6db56a178ee3a65bf017fe7996360ef026/mmdet3d/datasets/pipelines/loading.py#L218-L229)，测试时未来真值命令。** BEV-Planner++ 本地 val 测试管线从待预测未来轨迹末端生成命令，送入规划查询并选输出模态；真实部署无法获得该标签。论文只泛称 driving command，未交代此来源；没有换成导航命令的消融，因此不能给这条路径分配 0.35 m 中的具体份额。
8. **[REAL-WOD-001](https://github.com/Tsinghua-MARS-Lab/DriveMA/blob/de20e6c64878bc3413d26d69362c3be9b3a91533/examples/train/grpo/plugin/trajectory_reward.py#L963-L975)，官方 RFS 作训练奖励。** DriveMA 明确以人工偏好轨迹和官方 RFS 计算 RL 奖励。论文 Table 3 的 7.893→8.009→8.060 是训练阶段/奖励组合对照，不能作为 RFS 这一项或 test 8.079 的独立贡献。
9. **[REAL-WOD-003](https://github.com/vita-epfl/RAP/blob/5fd8630ae54442dd41827de4a9afe1690e3f02bb/navsim/planning/script/run_waymo_submission.py#L29-L42)，多 checkpoint 提交选轨。** RAP 把候选拼接后取预测 score 最大者；论文称双权重 NMS ensemble，而脚本函数未做距离 NMS。实际权重目录清单与单/双模型 RFS 差均未公开，不能把 8.043 直接分解为集成收益。
10. **[HUG-WA-002](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/close_loop/hugsim_client.py#L199-L207)，推理失败后继续计分。** WA-JEPA 异常时送全零制动轨迹；其聚合器指出静止场景仍可获 HD-Score 并另列可信均值。0.4462 的逐场景失败次数未公开，本审计只证明潜在污染路径，**不声称已报分数实际被抬高**。

## 候选实测设计

以下仅定义后续在具备权重、数据和运行许可时应如何检验，**本次均未执行**。

| 要回答的问题 | 控制变量与观测量 |
| --- | --- |
| TFv6 的接口、蠕行和停车牌规则各贡献多少 DS？ | 同一 checkpoint、seed、220 路线，先比较 route+target speed 与 waypoint 控制，再逐项关闭蠕行/停车牌/Kalman；同时记录 DS、blocked、违规和行人/cut-in 反应。 |
| SparseDriveV2 的 CARLA actor 真值位姿收益是多少？ | 同权重、同路线仅将 actor transform 换成 GNSS/IMU 估计，报告定位误差、路线偏差、DS/碰撞；先核实榜单提交是否允许 world 查询。 |
| TF++ 早停和手工控制规则怎样改变完整路线表现？ | 固定 MAP/SENSORS 轨道、checkpoint 和路线，对比 `STOP_AFTER_METER=-1` 与 1500 m，逐项关闭 stop-sign/creep；同时报 RC、IS、DS、normalized DS。 |
| NAVSIM 评分代理优化是否越过真实评分最优点？ | 在 v1、v2 各固定一套权重和候选，把 CEM 轮数、候选数量、评分权重单独变化；同场景报告预测 proxy 与官方 PDMS/EPDMS、碰撞/舒适项及独立划分表现，区分论文版和 README 版 TOAD。 |
| nuScenes 的未来真值命令贡献和替代方案？ | 固定 AD-MLP 与 BEV-Planner++ 权重和 val 样本，分别用未来 GT 命令、可部署地图/路线命令、无命令、乱序命令输入；报告 1/2/3 秒 L2、碰撞率、转弯子集及样本数，并核对预制 pkl 内容。 |
| WOD-E2E 的奖励与多权重选轨是否只适配 RFS？ | DriveMA 固定训练预算对照有/无 RFS 奖励；RAP 固定候选数对照单权重、双权重 argmax 与实际距离 NMS；同时报 RFS、ADE/碰撞代理、运行延迟及新划分表现。 |
| HUGSIM 的失败回退和控制器版本是否改变均值？ | 取得 436 场景逐场景 `n_failures`、`planner_stats.json`、场景清单，在同一控制器下同时报全量与无失败可信均值；若比较旧/新控制器，所有方法同场景同版重测。 |

## 报分与固定公开代码的限制

- **轨道与协议。** Bench2Drive 0.0.3 DS、CARLA LB2 DS、NAVSIM-v1 PDMS、NAVSIM-v2 EPDMS、nuScenes 短时 L2/碰撞、WOD-E2E RFS、HUGSIM HD-Score 均为不同任务；尤其 WOD 是开放环轨迹评价。TF++ 抽样的 5.56 DS 是 MAP，而固定脚本默认 SENSORS，不能与其他轨道混排。
- **同名方法的不同版本。** TOAD v1 已固定公开 `nav1` 分支，但论文 94.7 与后续 README 94.9 用不同迭代/候选/评分器设置；DriveVLA-M0 的 Scale 94.1 缺完整公开提交流程，Base 脚本为 92.3；DrivoR v2 的 54.6 对应 134k SimScale、30 epochs，并非前文 10 epoch 示例；GTRS 修复版 45.4 不能用旧协议 42.1 的消融分解。SparseDriveV2 审的是有 Bench2Drive agent 的专用分支，不是主分支。
- **nuScenes 的样本与输入。** SOTA2 页名写 test，但 BEV-Planner++ 固定测试配置加载 nuScenes val；其论文称 6019 个 val 样本只纳入有完整未来 3 秒轨迹的 5119 个。AD-MLP 的 0.35 m 是去高层命令的论文消融，修正后完整模型是 0.29 m。SparseOccVLA 的 `gt_planning_command` 被推理消费，但公开仓库未给字段生成链；没有足够证据把它也判作由评测未来真值生成。不同论文的评测协议可能不同。
- **WOD 与 HUGSIM 的缺链。** AutoVLA B 档缺 Waymo 专用提交/后处理路径；RAP Waymo 双 checkpoint 具体文件清单不在仓库。WA-JEPA Table 2 为 436 场景统一控制器重测，不能与旧 345 场景分数并比；UniAD 固定客户端六相机与论文该表四相机说明不一致，LTF 固定直行命令与共同 ground-truth command 协议不同，二者缺本次复测日志。WA-JEPA 的控制器修正是共享协议事实，其失败回退是否进入 0.4462 未知。

从目前可量化的内部对照看，榜单指标代理、短时 ego 状态、未来标签和控制规则都可能显著改变所报指标；多数条目仍缺同权重、同样本、同协议的单项消融。因此报告能指出**分数由哪些代码机制共同产生**，但不能给出每个榜单“多少百分比不属于驾驶能力”的可信数字。
