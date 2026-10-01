# 阶段 2 回扫建议（agent_real，2026-09-24）

本轮只读检查 `INDEX.md` 的 20 个入选单元、20 页审计页、四份 `findings/*.jsonl`、`taxonomy.md`、`sampling.csv`、`repos.csv` 与下述争议所涉的固定代码/论文。另看了 Senna、CarLLaVA、SimLingo-BASE 三个未入选页，未计入矩阵。未运行任何仓库代码、模型或评测。以下是定稿建议，不直接更改 taxonomy、findings 或 matrix。检查时发现文件共 32 条；主代理在本轮进行中新增了 `REAL-NUS-006`。

## 先处理的判定

1. **把 nuScenes 的历史状态与未来真值命令分开编码。** AD-MLP 论文 §2.1 明写高层命令取未来 3 秒位移，Table 1 同输入组合无命令 avg L2 0.35 m/碰撞 0.23%，加命令 0.29 m/0.19%。固定代码 [GT 命令构造](https://github.com/E2E-AD/AD-MLP/blob/4b93ba085ee47474152f282177865796ea577fc0/pytorch/admlp/stp3/datas/NuscenesData.py#L505-L532) 也按未来轨迹终点生成 LEFT/RIGHT/FORWARD。`REAL-NUS-006` 已补 `future_label_conditioning`，但预制 `data_nuscene.pkl` 未公开，故其中每个 token 的 21 维内容不可复核，维持中置信度。建议把 `REAL-NUS-001` 的“只用历史运动状态”与影响数字改准：**0.35 m 才是无未来命令的 ego 状态设置；0.29 m 是历史状态加未来标签**。否则同一个 0.06 m 增益会被错误归到可部署的 ego 动力学。
2. **保留 BEV-Planner++ 的未来命令发现，但限定在公开的 nuScenes val 本地评测路径。** [测试配置](https://github.com/NVlabs/BEV-Planner/blob/01c28d6db56a178ee3a65bf017fe7996360ef026/configs/bev_next/bev_planner_plus_plus.py#L279-L289) 调用 `LoadGTPlaner`，且 [ann_file 指向 val](https://github.com/NVlabs/BEV-Planner/blob/01c28d6db56a178ee3a65bf017fe7996360ef026/configs/bev_next/bev_planner_plus_plus.py#L313-L321)；加载器从未来 GT 末端计算命令，`simple_test` 传给规划头，规划头用它条件化查询并选三模态轨迹。`REAL-NUS-004` 代码机制成立，但论文没有无命令、乱序命令或可部署导航命令消融，**不能给 0.35 m 估算中归因于命令的数值**。论文对 command 的泛称没有说明 BEV-Planner++ 测试命令的来源，`disclosed=partial` 可保留并写明边界。此命令也存在于 BEV-Planner/BEV-Planner+ 配置，不能把 ++ 对 + 的 0.11 m 差值算给命令。
3. **BEV-Planner++ 的 ego 状态增益可以更精确地表述。** 固定配置 `bev_planner_plus.py` 与 `bev_planner_plus_plus.py` 的实质差别是规划头 `with_ego_status=True`；论文 Table 1 的 +→++ avg L2 0.46→0.35 m 可作为该配置变化的内部对照，同时保留没有同权重部署闭环验证的限制。论文 Table 2 的图像/速度扰动是 VAD-Base，不能拿来量化 BEV-Planner++；现审计页已正确隔开。
4. **HUGSIM 的修正控制器是全表协议事实，不能写成 WA-JEPA 的相对提分。** `HUG-WA-001` 代码与论文 Appendix A 都成立；但作者说控制器修正后 436 场景、相同聚合和命令下统一重测所有方法，且无修正前后配对分数。建议在报告单列“协议/绝对分数版本依赖”；若矩阵仍将 `simulator_controller_patch` 对 WA-JEPA 编 yes，应明定 yes 只指代码入口存在，不指已证提高 0.4462 或相对优势。更严格的“只记录已证榜单增益”口径应把此项留作协议注释，不计方法专属 hack。
5. **HUGSIM 推理失败制动只可说潜在污染路径。** `HUG-WA-002` 的全零轨迹回退和聚合器“静止仍可得 HD-Score”说明因果途径；缺 0.4462 的逐场景 `n_failures`/有效场景清单，不能断言已报均值包含多少失败回退。若矩阵 yes，定义须为“公开提交代码包含此路径”，报告注明实际触发未知。WA-JEPA、UniAD 使用的 ground-truth driving commands 是论文 Appendix A 的共同协议；LTF 客户端反而把命令固定 straight，可能降分，均不应在没有方法专属增益证据时与 BEV 的未来真值命令机械同编。

## 逐单元回扫

| 榜单 / 单元 | 现有发现 | 建议与报分边界 |
| --- | --- | --- |
| Bench2Drive / TFv6 | `B2D-TFV6-001/002/003` | 三类各有独立代码路径，保留。95.28 DS 是 README 要求开启 Kalman、stop-sign、creeping 的复现设置；约 95→94 是三项联合开关，不要分给单项。控制接口差异无同 checkpoint DS 数字。 |
| Bench2Drive / BLUE | `B2D-BLUE-001` | 保留中置信度；蠕行是否在 90.58 DS 中触发未知。B 档正确，gate 完整训练入口未公开；不把 gate 的 +5.51 DS 归给蠕行。 |
| Bench2Drive / SparseDriveV2 | `B2D-SPARSE-001` | 保留。需保持 `bench2drive` 分支 commit `e42ea59…`，不能回用原 main；SENSORS agent 无条件从 CARLA world 获取 LiDAR actor 真值变换。50 m 俯视 RGB 不在规划模型 Collect 输入，不加视觉泄漏类别。 |
| CARLA LB2 / TF++ | `LB2-TFPP-001/002/003` | 保留早停与后处理路径。5.56 DS 属 MAP 区，仓库默认脚本是 SENSORS；早停 1.5 km 来自论文，公开脚本默认禁用，不能断言固定脚本复现 5.56。Town13 0.96→5.10 是验证集，不是官方 test 消融。 |
| NAVSIM v1 / TOAD+DrivoR | `NAV1-TOAD-001` | 论文 94.7 PDMS 与 README 当前 94.9 不同；main README 要切 `nav1`，固定 main 不能复现 v1 命令，现低置信度合适。若定稿要求代码与报分同提交，应另固定 nav1 commit；否则把此行明确为“算法路径审计，榜单配置未独立复现”。 |
| NAVSIM v1 / DriveVLA-M0 | `NAV1-DVM0-001/002` | 论文 94.1 是 Scale，固定公开脚本跑 Base 92.3，缺 10K memory 与 TTT 的完整 navtest 链。两条发现只对可见 Base 路径成立，不能归因于 94.1；公开等级 A 是否满足“所抽成绩的训练/评测/推理”需复核，至少在抽样备注标成部分可审。提示词无消融，低置信度。 |
| NAVSIM v1 / RAP-DINO | `NAV1-RAP-001` | 保留 PDMS 代理评分择轨；固定代码还以 PDMS 标签监督评分头。若最终 codebook 保留 `metric_reward_training` 与 `metric_proxy_optimization` 两个正交阶段，本单元也具训练阶段证据；若合并，请统一 WOD RAP 的编码。 |
| NAVSIM v1 / DrivoR | `NAV1-DRIVOR-001/002` | 保留；Table 7 远期目标 +0.6 是 navval，v2 −1.6 是 warmup，均不是 93.7 navtest 的直接分解。93.7 是 trainval 版，勿混入 SimScale 94.0/94.6。 |
| NAVSIM v2 / DrivoR+TOAD | `NAV2-TOAD-001/002` | 保留；54.6→56.3 是 navhard EPDMS 同论文对照。warmup-two-stage 与 navhard 有交集且论文称获主办方认可，`benchmark_split_adaptation` 只表示泛化边界，不指违规或泄漏；也未给交集带来多少分。 |
| NAVSIM v2 / DrivoR | `NAV2-DRIVOR-001/002` | 保留；54.6 对应 134k SimScale、30 epochs，而 README 前面的 `Nav2_10epochs` 示例不是该成绩。`NAV2-DRIVOR-001` 还称权重在与 navhard 重叠的 warmup 上验证：若 `benchmark_split_adaptation` 定义要求把每个用此划分定参的方法标 yes，此单元需增第二条代码行证据；若仅记录明确搜索多组超参的 TOAD，则保留 no 并说明界线。 |
| NAVSIM v2 / GTRS | `NAV2-GTRS-001/002` | 保留机制，严禁把固定论文/README 的旧协议 42.1 或候选池旧消融 39.7→40.8 说成修复后 45.4 的分解。45.4 来自后续 DrivoR 论文的指标修复版比较，本固定提交无对应运行记录；建议抽样备注明确版本差，必要时将本项主分数写为“旧版 42.1；修复后转录 45.4”。 |
| nuScenes / SparseOccVLA | `REAL-NUS-003` | 保留 ego 特征发现。代码 `nuscenes_dataset_v2.py#L343` 取 `info['gt_planning_command']`，推理 `sparseoccvla.py#L707-L733` 以它选 command 分组与 anchor；但公开仓库没给此字段生成代码，尚不能证明同 BEV 一样由被预测未来末端计算。建议在页中记为 **未来标签来源待核**，不要直接把 `future_label_conditioning` 标 yes。未来 CAN bus 只走 occupancy 分支，不能据此说提高规划 L2。 |
| nuScenes / AD-MLP | `REAL-NUS-001/002/006` | 将 0.35 m ego 状态与 0.29 m 含未来命令明确拆开；`REAL-NUS-006` 已补，pkl 内容缺失故中置信度。`metric_grid_alignment` 有代码与论文解释，但无单项消融。SOTA2 把 0.35 标作成绩与修正后论文完整模型 0.29 不符，不能直接排在统一协议名次。 |
| nuScenes / BEV-Planner++ | `REAL-NUS-004/005` | 公开 val 测试管线未来 GT 命令条件化/选模态成立，影响数字 `none`；论文 Table 1 ego 状态 +→++ 的 0.11 m 内部差异可列。论文附录称 6019 个 val 样本只纳入完整 3 秒 GT 的 5119 个（约 85%）；这是披露的评测样本定义，没有难度偏差证据，先记协议说明而非提分发现。SOTA2 页名“test”不等于这份公开代码的 val split。 |
| WOD-E2E / DriveMA-4B | `REAL-WOD-001` | 保留 RFS 奖励训练，Table 3 的 RL 阶段差值含其他奖励和优化变化，非 RFS 单项贡献。WOD 是开放环人评轨迹 RFS；轨迹插值/点不足外推未见单独提分证据，不加后处理发现。 |
| WOD-E2E / RAP-DINO | `REAL-WOD-002/003` | 保留 RFS 监督评分头和多 checkpoint 提交策略，但双模型增益 `none`。论文称 NMS ensemble；公开脚本函数名虽为 NMS，实际拼接候选后 `argmax`，距离阈值未用。ckpt 目录内容未发布，不能断言确切两个 seed 与 8.043 结果逐项一致。 |
| WOD-E2E / AutoVLA | `REAL-WOD-004` | 该条只说教师生成训练 CoT 时显式给未来动作答案，论文附录已披露；不等于 Waymo test 提示泄漏。没有去掉答案提示的 RFS 消融，作为“解释可能事后合理化”的中置信度候选合适；若最终 yes 严格要求已证分数提升，可降为审计观察/NA。公开无 Waymo 专用提交入口，B 档且 7.5566 无法由固定仓库复核。 |
| HUGSIM / WA-JEPA | `HUG-WA-001/002` | 见上方 HUGSIM 两条边界。436 场景统一修正控制器；没有场景资产时也无法判 `runnable.txt` 实际筛掉场景。不要把 NAVSIM v2 的 EPDMS 消融移作 HUGSIM 0.4462 的贡献。 |
| HUGSIM / UniAD | 无 | no/NA 结论合理：`info['command']` 是共同输入；公开客户端配置六相机而 WA-JEPA Appendix A 称本表四相机，0.3124 对应的复测配置未固定。`scene_token='062'` 与异常路径更像复现/鲁棒性问题，未见正向提分证据。 |
| HUGSIM / LTF | 无 | no/NA 结论合理：客户端硬编码直行命令而论文同表协议给 GT 命令，可能损害转弯性能；无消融不能写成提分 hack。0.2310 为 WA-JEPA 436 场景复测数，固定 LTF fork 不含完整复测配置。 |

## Codebook 与交付一致性

- **跨榜单归类。** NAVSIM 的 RAP/DrivoR 以 PDMS 标签训练排序头再推理选最大，WOD RAP 以 RFS 标签做同类事，但目前前者多记 `metric_proxy_optimization`、后者记 `metric_reward_training`。建议最终版将“官方指标监督训练”和“该代理评分选轨”明确拆成可并存的两轴，或合成一个 `benchmark_metric_optimization` 类并在 description 标训练/推理阶段；不能因榜单不同而改类。DriveMA 的 RFS 奖励只属于训练阶段。`test_time_candidate_selection` 建议收窄为**多 checkpoint 合并/选轨**，否则所有 NAVSIM 多候选方法也要标 yes，形成大量重复编码。
- **未来命令判定。** `future_label_conditioning` 需证明当前被测样本的未来 GT 派生标签进入推理或输出选模态；仅有可部署导航 route/command、仅用未来标签监督训练，均不满足。BEV-Planner++ 是高置信度；AD-MLP 因推理 pkl 不公开为中置信度；SparseOccVLA 的字段名和选模态代码构成强线索，但字段生成链尚缺，不宜直接判 yes。HUGSIM 的共同 GT command 应单列协议背景。
- **固定仓库与报分。** 优先在 sampling/report 保留“论文/榜单报分”与“固定公开仓库实际可审配置”两列或两句，特别是 TF++ MAP、TOAD v1 `nav1`、DriveVLA Scale、GTRS 修复版、nuScenes val、AutoVLA Waymo、HUGSIM UniAD/LTF。这样不会把 A/B 级别误读成该固定 commit 可复现榜单数字。
- **结构校验。** `repos.csv` 的 20 个本地 clone 的 HEAD 与各自记录 commit 全部一致。在本轮读取到的 32 条 findings 中，按 `repos.csv` 的 URL **或** slug 映射，固定 commit URL、行范围、连续原文和 ≤15 行全部通过，无重复 ID。`repo` 字段却有 23 条 GitHub URL、9 条 slug（agent_real）；合并前应统一格式，或在自动校验器显式兼容两种。18 条无消融记录以 `none` 开头，只有 5 条值恰为 `none`；若验收脚本要求字面 `none`，应规范字段并把解释移至 description。部分 `paper_quote` 仅转述章节，建议补一小句短原文。所有 `disclosed` 均为 yes/partial/no 且非空。
