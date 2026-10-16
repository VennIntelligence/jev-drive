# 预登记：TTS（serve 端测试时候选选择，不训练），2026-10-16

背景：[决定 323](../../../research/decisions/323.md)（STACK 停：C2 后果头池化 AUC 0.773 不过线，但 road 通道 0.909、梯度对齐 0.708；Cinque 的 plan 是一个 MDN 分量，不能采样，候选只能是
结构化扰动，`results/stack/checks.md` C1 / C2）、[318](../../../research/decisions/318.md)（失败前 3 s plan 已接触）、[321](../../../research/decisions/321.md) / [322](../../../research/decisions/322.md)
（训练侧的几何目标伤避让 / 没有臂采用）、[290](../../../research/decisions/290.md)（策略自己的输出里有这条弯，推一下就转完：×1.25–1.5 曲率是杠杆）、文献（PDM-Closed 在 nuPlan 闭环 92 / 93
对学习式 72 / 73；Diffusion-ES 搜索 + 打分 50 → 92）。用户预先同意（2026-10-16 12:15 JST）：「STACK 不过，下一步加测试时候选选择」。
本文件在任何闭环读数之前写好、提交、推送；权重拟合之前提交；之后只追加补记。

## 1. 开关与候选

- 开关 `JEV_TTS`（`lib/tts.py`；驱动 `experiments/alpasim/lib/sh30_driver.py`（nuPlan held）、`pai_driver.py` + `pai_core.py`（PAI））。未设 / 0 = 关：驱动不 import 该模块，serving 与改动前逐位相同。
  `pai_core.plan` 多返回 `tokens` / `valid` 两个键（`sh30_core.Core.plan` 早就返回），不改任何数值。
- 位置：模型前向之后、`VH.observe` 之后、`serve_fix` 之前；之后的 serving 开关（serve_fix、track_ref gate、…）作用在选中的 plan 上，与原来作用在模型 plan 上一样。
  与 JEV_EXPERT / JEV_PERTURB / JEV_ROUTE / JEV_STOP（nuPlan）、JEV_EXPERT / JEV_TURN_GAIN / JEV_INTENT_HOLD（PAI）互斥（assert）。
- 候选（每个决策 K = 16，候选 0 = 模型自己的 plan）：横向偏移 ±0.5 / ±1.0 m（`stack_cand` 的 lat：沿各位姿法向，0–2 s 从 0 线性涨到 d）；曲率增益 0.8 / 1.2 / 1.4
  （`perturb.branch_poses(P, g, 1)`，允许转得更急）；弧长比例 0.9 / 1.1（同一路径按弧长 × s 重采样；s > 1 沿最后一段直线延长，4 s 位姿也动）；曲率 × 弧长两两组合 6 个（先增益后弧长）。
  注：头训练时的随机候选是 g ∈ [0.7, 1.3]、s ∈ [0.6, 1.1]（s > 1 末端截断），所以 g 1.4 与 s 1.1 的延长在头的训练范围外；拟合用真实几何标签，会把这一点算进去。

## 2. 打分（越低越好）

S_k = w_road · p_road_k + w_agent · p_agent_k + w_route · d_route_k + w_dev · d_dev_k

- p_road / p_agent：STACK 后果头（`lib/conshead.py`，冻结，`$DATA_DIR/runs/route_follow/stack/cons/head.pt`）对候选 k 的接触概率，8 个 0.5 s 格（0–4 s）取最大。输入同训练：喂入槽的 token
  （SLOTS = t − 1.0 s、t），喂入 ego 行前 20 维，候选位姿。一个决策 16 个候选 = 头一次批量前向（memory 编码一次）。槽 2 或 7 无效的决策直接用自己的 plan。
- d_route：候选 8 个位姿到**模型看到的路线记忆折线**（route-memory 特征里有效槽的点，按槽序连成折线，两端沿端段直线延长）的距离均值，每个位姿封顶 5 m；有效点 < 3 个时所有候选此项为 0。
- d_dev：候选 8 个位姿到自己 plan 的 xy 距离均值（m）。
- 选择：k* = argmin S；只有 S_0 − S_k* > margin 才换（向自己 plan 的滞回），否则用自己的 plan。决策之间不带记忆。
- 记录：drive.jsonl 每个推理决策 `tts`（k、名字、增益、段、各候选 S / p_road / p_agent / d_route、ms），换了时另记 `poses_own`。

## 3. 权重拟合（只用 TRAIN 划分的状态；离线；冻结后再读）

- 行：`on1_L18RSG`（RSG-L18-s0 自己的状态）中 C2 留下的行（`stack/cons/own.npz`，t0 未接触、两类标签都有），`rf_recov.dev_of` = False 的 train-log 行。候选中心 = RSG-L18-s0 在行自己输入上的前向 plan
  （C2 的 `plan`）；路线 = 行的喂入特征（`route_mem/feat/on1_L18RSG.npz`）；头的输入 = C2 存的 token 槽与 ego。
- 标签：每行 16 个候选的几何接触（C2 的几何：AgentHinge margin 0 / 侧 margin 0；道路栅格 Pacifica 四角 sdf < 0；0.1–4 s 40 步），任一类任一时刻 = 接触。
- 网格：w_road = 1（定尺度）；w_agent ∈ {0, 0.1, 0.25, 0.5}；w_route ∈ {0, 0.025, 0.05, 0.1, 0.2}（每 m）；w_dev ∈ {0.02, 0.05, 0.1, 0.2, 0.4}（每 m）；margin ∈ {0.02, 0.05, 0.1, 0.2}；共 400 组。
- 每组在 train 行上算：C = 选中候选的接触率；D = 选中候选对自己 plan 的平均偏离（m，全部行）；SW = 换的比例；R = 有路线的行上选中候选的 d_route 减自己的（均值，m）。
- 可行：D ≤ 0.15 m、SW ≤ 0.25、R ≤ 0（选择平均不离路线更远）。规则：可行组里 C 最小；C 在最小值 + 0.002 以内的组里取 w_route 最大，再取 D 最小，再取 margin 最大。
- 停止线（登记）：选中组的 train 接触率比自己 plan 低不到 0.005（绝对）→ 不做闭环读，写 fit.md 停。
- 同报（描述性，不参与选择）：train / 头的 inner 留出 log（crc32("inner:" + log) % 10 == 0，头的梯度没见过）/ dev 行上的接触（agent / road / 任一）自己 vs 选中，log 聚类 CI；
  换的比例与候选类型分布；oracle（真实接触最少的候选，平局取自己）作上限；只看候选类型的基线。注意：train 行是头的训练行（头在这些行上偏乐观），所以 inner 与 dev 是诚实读数。
- 输出：`$DATA_DIR/runs/route_follow/tts/fit/`；冻结权重 `weights.json` 拷进 `experiments/route_follow/results/tts/weights.json`（驱动默认读这个），报告 `results/tts/fit.md`。

## 4. 闸、读数与线

- 逐位闸（每个驱动 2 个场景）：新代码 JEV_TTS 未设 vs 改动前代码（worktree 在 b7b4d5c1），SH30_COMPILE=0（nuPlan：held_n6/c0 前 2 个场景 × 18 决策；PAI：Tokyo 参照第 c000 块前 2 个
  场景，同 gate serve、gains 20、JEV_VCONT 1.0、JEV_LEAD 1）：每个决策 poses 逐位相等。另各跑一次 JEV_TTS=1 的 smoke（同场景）：每个推理决策都有 `tts` 记录、无推理错误、报 ms。闸不过不读。
- 读数（分阶段，pool priority 9）：
  1. held 460 @18：RSG-L18-s0 + JEV_TTS=1 对存档 RSG-L18-s0 读数（配对 452，log 聚类 CI）：零分、at-fault、零分按出路 / 走廊 / 碰撞拆分、held 24 个转弯完成数。
  2. 若 held 零分增加不超过 +3：盒上 PAI 281 seed 0，同 serve（gate、gains 20、VCONT 1.0、LEAD 1）+ JEV_TTS=1，对盒上参照 `$DATA_DIR/runs/route_follow/recovery_envelope/pai/RSG-L18-s0`：
     零分、at-fault（碰撞零分）、严格 36 转弯完成、cor_route_end、被追尾。
  两个读数都报：换的频率、选中的候选类型，按段（approach / junction / straight / none：路线在自车投影 −4 到 +8 m 之间弯 ≥ 20° = junction；否则 0–40 m 内有 ≥ 30° 的弯 = approach；
  否则 straight；有效路线点 < 3 = none）；增加的延迟（`tts.ms` 的 p50 / p95，驱动 total ms 对参照）。
- 线（登记）：作为 serve 选项**采用**，当且仅当 held 零分 ≤ +3 且 at-fault ≤ +3（点估计），**并且**（held 转弯完成 ≥ +3 **或** PAI 严格 36 完成 ≥ 8 且 PAI 零分 ≤ 参照 + 3）。
- 结果：`experiments/route_follow/results/tts/`（`fit.md`、`held.md`、`pai.md`）；`research/next-round/turn-night.md` 末尾记日期条目；读数落地后写决定文件（decisions.md 下一个空号）。
