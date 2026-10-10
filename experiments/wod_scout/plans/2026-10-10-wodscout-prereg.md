# WODSCOUT 预登记：WOD-E2E 线重开之前的能力边界侦察（纵向之外还有什么，标签从哪里来）

2026-10-10，lane WODSCOUT。在任何新读数之前提交并 push。不训练、不微调、不向任何榜提交、不用 test 标签（本地没有，也不重建）。
val 的 rater 轨迹在这里只作分析标签。基线：WLG（两 seed，val RFS 8.187，test 8.099；决策 169、180）。
结果：`results/{q1_spotlight,q2_path_budget,q3_strata,q4_label_supply}.md`，表 `results/<q>/`，图 `figs/`。

已定、不重测、只引用：纵向是主瓶颈（决策 164、168、218 补记）；479 帧上学 rater 的路全部到头（171、173、175）；
selector 不迁移（195）；nuPlan 侧配方零样本无变化（217）；WOD 抽样 450 帧无 agent 框 / 地图（236）；夜间 shipped ADE@3s +0.43 m（131）。

## 0. 预登记之前已经看过的东西（如实记录）

写本文件之前只读了 manifest 和 index，没有读任何分数：

- `test_sequence_frames_for_submission.json` 只有 `sequence -> frame index`，**没有 cluster 标签**；`val_sequence_name_to_scenario_cluster.json` 只覆盖 val 479 个
  序列，10 个 cluster，没有 Spotlight。两份文件的顺序都看不出按 cluster 分块（val 相邻项 cluster 乱序；test 的 frame index 游程无结构）。
- 所以 **test 里哪些帧属于 Spotlight，本地没有任何合法来源可以知道**；官方评测只回 11 个 cluster 的均值。任务书里「看图把 Spotlight 帧分类」这一步的前提不成立，
  Q1 改成下面的间接读法。不通过多次提交反推成员关系（不提交，也不该这么做）。
- WOD-E2E 论文（arXiv 2510.26125）对 Spotlight 的定义只有一句：「Manually selected challenging scenarios」。
- box 上已有 `datasets/waymo_perception`（12 GB 样本）与 `datasets/womd`（2.5 GB 样本），可用于 Q4 的本地比对，不需要新下载。

## Q1. Spotlight（test 独有 cluster）

成员不可知，能回答的只有三件事，各自的读法先写死：

**A1 协变量位移（val 479 rater 帧 对 test 1 505 提交帧，全部自动量）。** 每帧：v0、喂入的 ax、intent、目标帧 FRONT 相机平均亮度（luma；夜间 = luma < 50，
与决策 131 同阈值，但这里是目标帧而不是序列均值，两个 split 同一算法）、shipped 的 lead_prob 与 lead 距离、WLG plan 的 5 s 位移 d5、|y5|、5 s 航向变化、
hold（d5 < 1 m）、两 seed 在 5 s 的分歧。逐项两样本检验（连续量 KS，离散量 χ²），Holm 校正，报 test − val 的差与按序列 bootstrap 的 CI。
汇总：一个 val 对 test 的分类器两样本检验（logistic 与梯度提升各一，按序列 5 折，AUC + 置换 200 次的零分布）。这个分类器只是统计检验，不进任何驾驶模型。
- 判读：AUC 的置换 p < 0.05 且 AUC ≥ 0.60 = 「test 与 val 在这些量上可分」，此时列出驱动它的量；否则 = 「在我们量得到的输入 / plan 统计上 test 与 val 不可分」，
  也就是说 Spotlight 不是一个靠 ego 状态、亮度、lead、plan 形状能认出来的子集。
- d131 的「Spotlight 不是弱光」：成员不可知，无法直接验证。能验证的是它的必要条件：若 test 的夜间占比不高于 val（差的 CI 上界 < +5 个百分点），
  则「test 比 val 多出来的那个 cluster 是弱光」不成立的可能性大；写成「与 d131 一致 / 不一致」，不写成「已验证」。

**A2 榜单上的跨方法结构（`exports/waymo-e2e-2026-10-08/leaderboard.json`，155 行）。** 样本：每个账号 RFS 最高的一行，且 RFS ≥ 7.5。
量：Spotlight 分与其余 10 个 cluster 分的 Spearman（按账号 bootstrap CI）；控制「其余 9 个 cluster 均值」后的偏 Spearman；
留一法用 10 个 cluster 线性预测 Spotlight 的 R²；各 cluster 的账号间标准差；ego-status / 小模型行与大模型行在 Spotlight 上的差相对其余 cluster 的差。
还报：我们（WLG）在 Spotlight 上相对「按其余 10 个 cluster 预测值」的残差。

**「可用的 val 代理」的定义（三条都满足才算）。** val 里存在的某个 cluster C（或事先固定的组合：10 个 cluster 的等权均值）是 Spotlight 的可用代理，当且仅当
(i) 跨账号 Spearman(C_test, Spotlight_test) ≥ 0.70 且 CI 下界 ≥ 0.50；
(ii) 控制其余 cluster 均值后的偏 Spearman CI 下界 > 0（它带的不只是「总体水平」）；
(iii) C 自己的 val → test 是稳的：WLG 在 C 上的 test 分落在 val 分的按序列 bootstrap 95% CI 内。
只满足 (i) 记为「只是总体水平的代理」。都不满足则结论是「val 里没有可用代理」，并给出最接近的 C 和它差在哪一条。

**A3 看图（小步）。** 先看 12 个 test 帧 + 12 个 val 帧（seed 0 随机），用下面的 rubric，只为确认 rubric 可用、自动量没有漏掉明显的东西。
只有当 A1 判「可分」时才做全量：取分类器 out-of-fold 分数最高的 60 个 test 帧与随机 60 个 val 帧，匿名打乱后由子 agent 按 rubric 标注，比较分布。
A1 判「不可分」则不做全量（找不到该看哪些帧），如实写。

Rubric（看图之前写定）：
- 场景：路口 / 多车道直路 / 单车道或居民区 / 停车场或出入口 / 高速或匝道 / 施工区 / 其他
- 光照：白天 / 晨昏 / 夜间有路灯 / 夜间暗 / 眩光或雨雾
- 自车状态（自动）：静止 < 0.5 / 0.5–5 / 5–12 / ≥ 12 m/s
- 前车：无 / 行进前车 < 30 m / 静止或排队前车
- VRU：无 / 路上或路边近处行人（约 15 m 内）/ 骑行者或滑板车 / 只有远处
- 特殊物：无 / 特种车辆 / 路上异物或动物 / 锥桶路障 / 开门或侵入车道的停放车 / 其他异常
- WLG plan 的动作（自动）：hold（d5 < 1 m）/ creep（< 5 m）/ 直行 / 换道（|y5| 1–4 m 且航向变化 < 15°）/ 转弯（航向变化 ≥ 25°）/ 其他

## Q2. val 上非纵向的丢分预算（WLG，479 帧）

口径同 `wod_gap.md`：cluster-mean RFS，帧权重 w = 1 / (cluster 帧数 × 10)，「分」可加；WLG = 两 seed 逐帧分数均值；按 sequence 配对 bootstrap，B 4 000。
构造沿用决策 164（`pp_wod_diag.retime`）：O1 = 自己的 path 配 top-rated 的弧长（只修速度），O2 = top-rated 的 path 配自己的弧长（只修 path）。

- **G0**：WLG 8.187 ± 0.002、对 top 的 gap 1.400 ± 0.005、O1 +0.657、O2 +0.290（决策 218 补记）各在 ± 0.005 内复现，否则停。
- **预算的两个端点**：path-only = O2 增益（下端：只改 path 就能拿回的）；non-longitudinal 上端 = gap − O1 增益（只修速度拿不回的全部，含必须联合的部分）。
  两个都报，带 CI；中间那块（gap − O1 − O2）叫「joint」。
- **分层**：10 个 cluster；情景（由 cluster 标签与 intent、亮度定义，不看分数）：cut-in = `Cut_ins`；绕障 = `Foreign Object Debris` ∪ `Construction` ∪ `Special Vehicles`；
  换道 = `Multi-Lane Maneuvers`；路口转弯 = `Interections` ∩ intent 左 / 右；路口直行 = `Interections` ∩ intent 直行；行人 = `Pedestrian`；骑行者 = `Cyclist`；
  夜间 = 序列亮度 < 50（决策 131 的 `seq_lum.csv`）。情景之间有重叠（夜间与其余），表里写明哪些列可加。
- **不重复计数**：WLG 的总量（O1、O2）就是决策 218 补记的数；WP2 的同一量是决策 164 的 +0.722 / +0.296；决策 168 的「只选路径 +0.367」是 F20 候选族在 WP2 上的
  特权选择（5 条 path 候选），与 O2 是同一块空间的另一种量法，不相加。本 lane 新增的只是这块怎么按 cluster / 情景分、以及 top-rated 的 path 具体做了什么。
- **BEV 选帧规则**：按 w × (O2 − 自己)（seed 均值）降序取前 30 帧，并列按帧名。先看前 10 帧确认 rubric 可用，再看 30 帧。
- **path rubric**（看图之前写定；先由几何量自动归类，再看 BEV + 前视图核对，改判的帧逐个列出）：
  - P1 换道 / 走另一条车道：5 s 横向差 ≥ 2.5 m 且两条轨迹末段航向差 < 15°
  - P2 车道内偏移 / 贴边绕物：5 s 横向差 0.5–2.5 m 且航向差 < 15°
  - P3 转弯几何（两者都转，半径 / 切角不同）：两条轨迹 5 s 航向变化都 ≥ 25°
  - P4 路线不同（一条转、一条不转）
  - P5 我们的 plan 漂 / 摆而 top-rated 走直
  - P6 标签伪影：top-rated 提前结束被 pad，或 top-rated 近静止而方向是噪声
  - P7 其他
  每类报帧数与分；另报这 30 帧占全部 path-only 分的比例（集中度）。

## Q3. 感知侧的缺口：夜间、行人、test 上丢分的类别（WLG，val）

- 分层：夜间 / 白天（luma < 50 / ≥ 120）、`Pedestrian`、`Cyclist`、`Others`、`Single-Lane Maneuvers`、`Interections`（决策 180 里对榜首丢得最多的四类 + 夜间 + 骑行者对照），
  同表列出全部 10 个 cluster。
- 每层：n、RFS（top / 日志 / WLG / shipped）、WLG 丢的分（对 top 的加权 gap）及 CI、O1 增益、O2 增益及 CI、WLG − shipped 及 CI、ADE@3s（对日志）。
- **path / speed / both 的判定线**（每层）：O1 增益 CI 下界 > 0 且 O1 ≥ 2 × O2 = 「speed」；O2 增益 CI 下界 > 0 且 O2 ≥ 2 × O1 = 「path」；
  两个下界都 > 0 且比值在 2 倍以内 = 「both」；其余 = 「undetermined」（样本不够）。
- 夜间另报：夜 − 昼的 gap 差，未匹配与按 v0 档（静止 / 0.5–5 / 5–12 / ≥ 12）重加权两种，带 CI；ADE@3s 夜 − 昼（1 437 帧，rater + extra），对照 shipped（决策 131 的量）。
- val → test 的逐 cluster 对照：WLG 的 val cluster 分与 CI 对 test cluster 分（决策 180 的截图数），标出 test 落在 val CI 外的 cluster。只作描述，test 没有 CI。
- 这里大约 20 个分层 × 3 个对比；贴边的 CI 只算弱，不做多重校正但写明个数。

## Q4. 标签供给清单（清点，不是方法）

- **(a) WOD-E2E 每帧到底有什么**：全量扫描 front3 slim shard（三个 split 的全部帧，slim 只删了 5 路相机），对每条 `E2EDFrame` 递归 `ListFields`（深度 3），
  按 split 统计每个字段路径「非空」的帧数；另报 `preference_trajectories` 分数 ≥ 0 的帧数（train 是否真为 0）、每图 pose 是否填、timestamp 是否全 0。
  确认或推翻决策 236 的抽样结论。
- **(b) 能否连到 WOD Perception / WOMD**：序列 id 格式、车辆平台与相机 rig（用 box 上已有的 perception 样本对标定与分辨率）、时间戳 / 位姿是否给出可对齐的量、
  论文与官方文档、licence 条款（子 agent 做网页调研，逐条标 STATED / INFERRED / NOT FOUND）。不下载任何大数据；若发现需要下载才能判，只报大小。
- **(c) nuPlan（navtrain）上按地图 + agent 框算「rater 式」标签**：清点 navtrain 上已有的可计算分项（PDM 各子分、已缓存的候选打分的规模）；
  **一致性**：需要一个同时有 rater 分与地图 / agent 框的设置。预登记时已知 WOD-E2E val 有 rater 分而无地图 / 框，navtrain 反之；若 (a)(b) 没有翻出交集，
  结论直接写「没有这样的交集，一致性无法测」。
  能测的替代量（不需要地图，写在前面）：val 479 帧 × 3 条 rater 轨迹上，**不拟合任何参数**的运动学量对 rater 排序的成对一致率（同帧内分数不同的轨迹对，
  按序列 bootstrap，机会水平 0.5）：5 s 弧长（进度）、最小纵向加速度（制动）、最大横向加速度、最大 jerk、到日志未来的 ADE。
  判读线：一致率 CI 下界 ≥ 0.70 = 「这一项可计算分量与 rater 排序基本同向」；0.55–0.70 = 弱；CI 含 0.5 = 无关。这只覆盖 rater rubric 里不需要地图的部分
  （效率、制动），安全 / 合法两项无法在 WOD 上算。已有的相邻读数只引用：决策 195（由 simulator 分训练的 selector 的 pick 与 RFS 最优的 Spearman 0.11，混着跨域）、
  决策 173（日志导出的 hindsight 标签对 RFS 为负）。
- 输出：一张清单表（来源、单位与数量、带什么标签、能否到「rater 偏好」、可行性与障碍）。

## 资源

全部 CPU（box，`jevdrive.run.Run`，split 取自 `jevdrive.data.splits` 的 `wod/val`、`wod/test`、`wod/train`）；不用 GPU（预计 0 卡时；若某一步需要推理，先估算并在
超过 6 卡时前停下报告）。box 上另有三条 lane，全量字段扫描限 32 个进程，不动别人的文件。事后加的读数一律标 post hoc。
