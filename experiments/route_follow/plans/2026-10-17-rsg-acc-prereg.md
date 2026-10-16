# 预登记：RSG-ACC，把 RSG-L18 的配方搬到 ACC(+GAP) serve 上的状态（一个联合臂 + 一个对照），2026-10-17

本文件在任何读数（离线闸或闭环）之前写好并推送，之后只追加补记。链 [scripts/rf_rsgacc.sh](../scripts/rf_rsgacc.sh)，
读数与闸 [scripts/rf_rsgacc.py](../scripts/rf_rsgacc.py)，结果 `results/rsgacc/`，box `$DATA_DIR/runs/route_follow/rsgacc/`。

## 0. 为什么

- 唯一被采用过的训练课是 RSG-L18（d286）：路线记忆底座 `RB0-F-s0` 上温和一轮（2 500 步、lr × 0.1）只带 S 两个 hinge（扫掠足迹对车、道路栅格）的
  on-policy 行，行来自 `RB0-F-s0` 在 nuPlan onpol-train 1 020 个 scene、18 决策、**旧的默认 serve** 下的 rollout（池 `on1_L18RB0`，17 429 行）。
- 现在 PAI 提交的 serve 是 ACC（`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`，转向权重 10，d327 / d328），候选加 `JEV_GAP=0.7,2.7`（d330）。ACC 下车更快，
  到的状态不同：ACC 新增的 15 个零分大多是更快地进弯 / 过弯、32–38 m/s 的高速漂移、开到停车点（research/route-follow/lon-clips-acc.html）。
  S hinge 从来没见过这些状态。假设：在 ACC(+GAP) serve 下重新滚出 on-policy 行，同一配方在这些状态上给 S hinge，PAI 上 ACC+GAP 读数变好，nuPlan 默认 serve 不变差。
- 不做的：横向形状 / 几何目标（d286 路线 hinge、d303 接近课、d321 JOINT-DAGGER 三次失败）、新转弯线、等待 / 停车行（d326，另一个 agent 的线，数据还在下）。
  只用 S 两个 hinge（RSG 原样），不加速度项（d303 补记 5 允许，但 ACC 下纵向由 ACC 定，模型的速度 plan 只经停车守卫起作用，没有记录支持加）。

## 1. 数据（两条赛道都滚，理由）

- **nuPlan onpol-train**（`alpasim_nuplan/onpol-train`，1 031 个 scene 中 ol_chain 的 6 个 chunk 表 1 020 个，never held 460）：有道路栅格，S 的两个 hinge 都在；
  ACC 新零分里「更快地进弯 / 过弯」是道路 hinge 的事，只有这里有栅格。serve = `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`（sh30_driver，nuPlan 交接 0.5 s，d329 的接线），
  18 决策，`SH30_DUMP`，其余与 RSG 的 rollout 相同（ol_chain roll1 的 run.sh sh30 + c0b_chain.OVERRIDES + after160）。
  注：这些 scene 也在公开 1491 护栏里，与 RSG 自己的行相同（d264 起的协议），不是新泄漏。
- **PAI train160**（`experiments/onpol_pai/results/split/train160.tsv`，160 个 scene，与 281 读数集、hard_exam / control / trainable / hard_train 交集 0，已核）：
  32–38 m/s 高速、PAI 的路口与交通只有这里有；PAI 没有道路栅格，这些行只有车的 hinge。serve 与 d330 阶段 2 相同（ACC+GAP、
  `controller.gains.rel_front_steering_angle_weight=10`、`ALPASIM_FAST=1`），加 `PAI_DUMP=100000`、保留 rollout.asl 给行管线（onp_rows rows / cold → onp_cache → rf_pai_pool）。
  never PAI 281、held 452、任何护栏 scene。
- **滚的策略（规模）**：同一 scene 上换策略是唯一能加行的方向（scene 数固定，PAI 仿真在 box 上逐位可复现，同一策略重滚不出新行）。
  最多 6 个 checkpoint：`RB0-F-s0`、`RB0-F-s1`（RSG 配方的 init 族）、`RSG-L18-s0`、`RSG-L18-s1`（被 serve 的族）、`RSGlr03n5000-L18-s0`、`RSGk51w1-L18-s0`（RSG 等价格，d300 / d311）。
  分阶段：1（`RSG-L18-s0`，nuPlan 4 个 scene + PAI 4 个 scene 的冒烟）→ ~10（`RSG-L18-s0` 全部 chunk，两条赛道）→ 其余 5 个。用几个 checkpoint
  在合池之前按磁盘与墙钟定下并写进补记，之后不改。报告 **行数**（每池行数、scene 数、去重后的 scene-决策数）与 **步数** 分开。
- **每池闸**（不过则该池不进合池）：nuPlan = ol_chain 原样（on_rows tok 的 R1 + interpreted control smoke10eY、ol_labels R1 / R2、rm_build pool、poolgate：route ok、
  route0_share_005、cov20）；PAI = onp_rows G-in（dump 逐位相等份额 ≥ 0.99）、rf_pai_pool 的 served rmem 闸、R1（本 checkpoint 在池输入上重算的 plan 对 cache 的
  `poses_model`，前 4 s、容差 max(0.03 m, 4 fp16 ulp)，超出份额 ≤ 0.01；错 seed ≥ 0.9；box 对 box，不再有 Tokyo / box 舍入差）。
- **dev 留出**：新池按 log（PAI = scene）`crc32(log) % 100 < 15`（rf_recov.is_dev）标 `dev`；dev 行**永不抽**（`--on-weight dev=1:0`），只给离线闸 G4。旧池 `on1_L18RB0` 不动。

## 2. 臂与对照（一个联合臂 + 一个对照；同步数、同命令）

- 合池 `on1_RSGACC` = `on1_L18RB0`（在前，`self`）+ 各新池（`self`，前缀各一个），rf_pait.py merge。
- **臂** `RSGACC-L18-s0`：`--init RB0-F-s0` + RSG-L18-s0 的命令逐字（`--warmup 1 --lr-scale 0.1 --yr-mass 0.1 --label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3
  --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train --on-mode self --seed 0`），`--on on1_RSGACC`，`--on-k` 与 `--on-weight` 取 pilot 选出的格，`--steps 2500 --eval-every 500`。
- **对照** `RSGACC0-L18-s0`：同一命令、同一池、同 `--on-k`，新池行权重 0（`--on-weight pool=<每个新前缀>:0`；新行在池尾、旧行权重相等，抽样逐位等于只有旧池，rm_train 的说明）。
  即 k13 时它就是 RSG-L18-s0 的重训；它量的是「重训噪声 + 同命令」，臂 − 对照 = 新行的作用。

## 3. 剂量 pilot（1 000 步，离线闸，任何全量训练之前）

格（都 1 000 步、seed 0、`--eval-every 500`，其余同上）：
- **a** `--on-k 13`，训练行均匀抽（新行按行数自然占比）；
- **b** `--on-k 13`，旧池 50 % / 新池合计 50 %（新池之间按行数）；
- **c** `--on-k 26`，均匀抽。
对照：`--on-k 13` 与 `--on-k 26` 各一个新行权重 0 的 1 000 步对照（`p0k13`、`p0k26`）。

闸（每格对同 k 的 1 000 步对照，全部成立才算过）：
- **G1** dev ADE ≤ 0.538（RSG-L18-s0 0.5284 + 0.01，rf_recov.trainer_gates 的线）；
- **G2** G-rely：rely.json `read` 为真（路口 − 直路 CI 下界 > 0，路口 ΔADE ≥ 0.01，rm_train 内读）；
- **G3** 等弧长转弯（rf_recov.gate_turn，held 460 决策 0 / 1 行，`L18RSGallheld` 的 held01）对 RSG-L18-s0 ≥ −0.3°；
- **G4** 新 nuPlan 池 dev 行（从未抽到）上重算的 plan 的接触率（车或道路，0.1–4 s，rf_dagger.contacts，t0 已接触的行去掉）比同 k 对照低：
  log 聚类配对差的 95 % CI 上界 < 0。PAI dev 行（只有车）同表报告，不进判据。
选格规则：过闸的格里 G4 相对降幅最大的；与次大差 ≤ 2 个百分点时取剂量小的（a < b < c）。**没有格过 → 停，不做全量，报告。**

## 4. 全量与读数（按成本，--after 串起来；每个 gate job 只在线成立时提交下一段）

全量：选中的格 2 500 步，臂 + 对照。全量离线闸 G1、G2、G4（同上，对全量对照）记录；G1 或 G2 不过 → 停。

读数（臂判定；对照同表报告、不判）：
1. **held 452 @18**（nuPlan，默认 serve，开关全关，rm_chain.sh read，6 chunk；ol_confirm 对存档 `RSG-L18-s0` 读数配对 452 个 scene、log 聚类 bootstrap）：
   零分 ≤ RSG-L18-s0 + 3；GT 距离点估计差 ≤ +0.03 m。两条都成立 → 2；否则停。
2. **公开 1491 @10 护栏**（ol_chain readlist，all_n10 的 10 个 chunk；ol_confirm 对存档 `RSG-L18-s0` 护栏读数配对，1 479 个有 rollout 的 scene）：
   均分差 CI 下界 > −0.01；零分点估计差 ≤ +5。都成立 → 3；否则停。
3. **PAI 281，ACC+GAP**（pai_box.sh，`CHUNKS_FROM` = d330 阶段 2 读数的 chunk 表，同 serve、同 WIZ、`ALPASIM_FAST=1`）对 d330 阶段 2 的 `RSG-L18-s0` 读数
   （`runs/route_follow/acc_gap/read/s2`），按场景配对 bootstrap 10 000 次、seed 0：
   均分 CI 下界 > −0.01 且点估计 > 0；考题 72 点估计 ≥ −0.01；零分 ≤ 参照 + 2；交接 spin ≤ 参照 + 1。另报新增 / 修好零分逐个（按类型）、被追尾、考题 / 对照 / stop17 子表。

**判词**：三段全过 = 采用，PAI 提交候选（需要新镜像，本 lane 不建）；任一段不过 = 不采用，停在那一段。

## 5. 成本估计（box 4 卡、88 核配额）

- rollout：nuPlan 一个 chunk（170 个 scene）约 15 min（RSG 的 train roll 实测），6 个 checkpoint × 6 = 36 个 job；PAI 一个 chunk（32 个 scene）约 25–35 min（d330 阶段 2 实测 24 min / 36 个 scene，
  加 dump 与 asl），6 × 5 = 30 个 job；CPU 约 12 个并发 → 约 2–3 h 墙钟。磁盘：nuPlan dump 约 39 GB / checkpoint（strip 后约 9 GB），PAI asl 约 64 GB / checkpoint（行建完即删）。
- 行管线：每池 CPU 0.5–1 h；合池 0.2 h。pilot：5 次 1 000 步训练（约 3 min 各）+ 前向 + 闸 < 1 h。全量：2 次 2 500 步（约 7 min 各）。
- 读数：held 2 × 6 chunk（约 0.3 h 墙钟），公开 2 × 10 chunk（约 0.5 h），PAI 2 × 8 chunk（约 0.5 h）。合计约 7–9 h 墙钟。

## 6. 会削弱解读的情况

- 1 seed（seed 0；参照 RSG-L18-s0 也是 1 个 seed 的读数）；held 是行的留出，不是配方的留出。
- 新行有一半以上来自 RSG 族自己的 rollout（d286 补记 5：RSG 自己的行在默认 serve 下对 RSG 无增益），所以「ACC 状态」与「第二轮」在 RSG 族的行上分不开；
  按 init 族（RB0 / RSG）分的行数照报。
- nuPlan 上 ACC 本身变差（d329），这些 rollout 里会有 ACC 造成的失败状态；S hinge 只在车 / 道路接触上起作用，不看 GT。
- G4 的 dev 行所在 log 在旧池里有默认 serve 下的行（旧池不动），G4 只是方向闸。
