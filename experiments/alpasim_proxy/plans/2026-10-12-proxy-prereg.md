# PROXY 预注册：一个行为像 private 套件的本地选择集（2026-10-12 02:50 JST，读数前写定）

## 问题

公开 1491 scene 的均分三次选错模型（APY10m10-AB-s0 公开 0.935 → private 0.640；YR10m10-F-s0 公开 0.925 → private 0.80）。
最后一个正式名额要按一个本地集选：这个集在已知 private 成绩的策略上必须复现 private 的排序和大致差距。
本文件定义：候选代理集、打分、有效性检验、闸门、读数顺序。读数之前写定，之后只追加修订（Amendment），不改原文。

## 锚点（private 成绩已知、本地可跑的策略）

private 数字取自排行榜 JSON（Space Gradio `lambda_5`，2026-10-12 拉取，`results/leaderboard_nuplan_20261012.json`）：

| 锚点 | 本地 1491 公开 rollout | private 均分 / at-fault km / GT m |
|---|---|--:|
| A1 `APY10m10-AB-s0`（提交 #1） | 已存（ot3/cf1，编译 pass） | 0.640 / 0.132 / 2.288 |
| A2 `YR10m10-F-s0`（提交 #2） | 已存（ot3/cf1） | 0.800 / 0.529 / 1.733 |
| A3 SimScale LTF 样例（`simscale-lft-release-v0`） | 今晚跑（box 池，原生，torch 2.8） | 0.738 / 0.270 / 2.266 |
| A4 SimScale DiffusionDrive 样例（`simscale-diffusiondrive-release-v0`） | 今晚跑（同上，先 8 scene smoke） | 0.772 / 0.267 / 2.158 |
| A5 garage `model_0014`（五个队提交同一 checkpoint，结果一致） | 今晚尝试搭建（子任务，可能失败） | 0.848 / 0.228 / 0.718 |
| 软锚 WA-JEPA 参考（本地已存） | 已存（c0c） | 榜上 `wajepa-s2` 0.762 / 0.626 / 1.847，配置不是我们的，只报告不进检验 |

锚点在 private 上的排序：均分 APY < LTF < DD < YR < garage；GT YR 最好（garage 除外）、APY 最差；at-fault km APY 最差、YR 最好。
LTF 与 DD 在 at-fault km（0.270 / 0.267）和 GT（2.266 / 2.158）上相差 < 5%，按并列处理，不计入排序检验。

## 指标（与 AUD1 同一份代码定义，src/eval @ 0bb4c4b）

- 平均 scene 分：scene 分的均值，路线折返失败的 12 个 scene 记 0、在分母内。
- GT distance：有 metrics 的 clip 上，截断后（第一次 ≥ 4 m 后丢弃）的逐 clip 最大距离的均值。
- at-fault distance（km）：Σ 行驶距离 / Σ at-fault 标志 / 1000。
- 每决策漂移：k = 4..10 每步离日志路径距离均值的 OLS 斜率（AUD1-drift 的定义），只报告，不进有效性检验。
加权版本：每个 scene 一个权重 w，三个量分别是 w 加权均值 / 加权比值 / 加权均值。实现 `scripts/proxy_fit.py`，并在 w ≡ 1 时
复现 AUD1-drift 的 run-level 表（APY 0.935 / 1.130 / 0.863，YR 0.925 / 0.941 / 0.789），不复现即停。

## 分层变量（只用与 driver 无关的 scene 元数据）

city（拉斯维加斯 / 波士顿 / 匹兹堡 / 新加坡 = LHT）、起步速度 v0（< 1 / 1-5 / 5-10 / > 10 m/s）、近静止（日志路径 < 5 m 且 v0 < 1）、
4 向命令（左 / 直 / 右，AP2 driver 第一步）、首个路线点横向 |y|（< 2 / 2-10 / > 10 m）、首个路线点纵向 x（< 20 / 20-35 / ≥ 35 m）、
日志路径长度（< 5 / 5-20 / 20-40 / 40-60 / > 60 m）、起步加速度（< -1 / -1..1 / > 1）。长度（决策数）在公开集恒为 10，暂缺，等
LONG-HORIZON 线（experiments/alpasim_long/，出现后在那里协调，不重复）。

## 代理候选（固定清单）

- P0：公开 1491 不加权。
- P1：单因子分层重加权：对一个因子，每层一个非负权重（层内 scene 等权），按下面的损失拟合。
- P2：两因子乘积层（所有因子两两组合，层数 ≤ 20），损失加 KL(层分布 || 公开层分布) × λ，λ ∈ {0.01, 0.1}。
- P3：所有因子 one-hot 的指数倾斜（最大熵）重加权，上界：说明「仅靠重加权公开 scene」最多能走多远。

损失（锚点集 A，三个指标）：每个指标先取对数（均分取 log(1 − 分) 即失分率），对锚点去均值（只比较锚点之间的相对位置，不要求
绝对水平），再求与 private 同样处理后的差的平方和。绝对水平另报。拟合只用已到位的锚点；A3–A5 未到之前只用 A1、A2（两个点，
相对位置 = 差值）。

## 有效性检验（判定代理可用；全部满足才算通过）

- V1 方向：A2 − A1 三个指标的符号与 private 相同（均分 +、GT −、at-fault km +）。
- V2 量级（「大致」= 2 倍以内）：均分差 ∈ [+0.08, +0.32]，GT 差 ∈ [−1.12, −0.28] m，at-fault km 比 A2 / A1 ∈ [2, 8]。
- V3 跨锚点排序（A3、A4 到位后；A5 到位则纳入）：均分上锚点排序与 private 的 Kendall τ ≥ 0.6；GT 与 at-fault km 上，去掉并列对后
  不一致对 ≤ 1。
- V4 不过拟合：Kish 有效样本量 ≥ 300 scene；留一锚点：去掉一个锚点重拟合，被留出锚点在均分上的排序位置与 private 一致（≥ 4 个锚点
  时，至少 3 / 4 次一致）。
- 只有 A1、A2 时最多给到「方向 + 量级在两点上可拟合」；两点拟合任何一维以上的权重都过拟合，不能宣布代理可用。

## 另一条路：指标迁移（P4）

即使没有任何重加权通过，也读：哪一个公开统计量（均分、GT、at-fault km、zero 数、≥ 4 m 出路比例、漂移斜率、近静止碰撞）在锚点
之间与 private 均分排序一致。这是「换个看法」而不是「换个 scene 集」，同样用 V3 的 τ 判。

## 闸门与停止

- G0：w ≡ 1 复现 AUD1 数字（见上），否则停。
- G1：每个新锚点跑完全部 1491 scene，driver 无推理 / 输入错误（ot2_loop 的 health 检查）；smoke 不过就不跑全量，写明原因。
- 若 V1 在 P0–P3 全部不过：结论是「公开 scene 不含 private 的失分机制，重加权无效」，记入决定，代理改为等 LONG-HORIZON 内容，不硬凑。

## 不做

不提交任何东西，不碰 alpasim-cli 的 submit；Tokyo 只用于本任务（理由与计划见 README）；box 只经 GPU 池。

## Amendment 1（2026-10-12 02:45 JST，A3–A5 任何结果读出之前）

- 只用 A1、A2 的拟合已跑完（`results/fit_a2/`，权重 `weights.npz` 冻结）。读到的只有 A1、A2 自己的数：重加权在城市 / LHT / 命令 / 路线几何 /
  加速度上都翻不过均分的符号，只有「近静止」与「日志路径 < 5 m」两层以 66–77% 的权重压上 25–29 个 scene 才同时过 V1、V2（Kish ESS 47–58，
  V4 不过）。
- 新增检验 V5（样本外，比 V3 更硬）：`fit_a2` 中冻结的每个代理，不再重拟合，直接算 A3–A5 的三个指标，看锚点排序（与 V3 同一判法）。
  这是本线的主读数；V3（含 A3–A5 重拟合）降为辅读。
- 12 个路线折返 scene（所有 driver 都在开车前失败，主办方说 private 不受影响）在所有代理里权重为 0；G0 仍在全部 1491 上检查。
- 实现补充：锚点的 driver 健康检查对 shipped 样例 driver（LTF、DiffusionDrive）改读 `native-logs/driver.log` 的逐 session 计数
  （`inference_error`，`straight_fallback` + `dynamic_state_fallback` 记为输入错误），`experiments/alpasim/scripts/c0b_chain.py`。
