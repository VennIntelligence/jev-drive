# 预登记：第二轮 on-policy 行（池跟着策略走，side-bias 修法 2），2026-10-13

背景：[research/route-follow/side-bias.md](../../../research/route-follow/side-bias.md) 修法 2；修法 1 的干预读数
[results/rsgboth_sb/summary.md](../results/rsgboth_sb/summary.md)（RSGbothSB：右转内切消失，公开护栏过、均分 +0.011 对 RSG；held 零分 44 / 39；
转弯完成 75 / 88 %，s0 不过）。RSGboth 多出的零分（[results/rsgboth_v/diag.md](../results/rsgboth_v/diag.md)）：人停了车还在挪、追尾、弯道偏宽。
本文件在任何新 rollout、池或训练之前写好、提交并 push；补记只追加，不改原文。

**写本文件时已看过的**：上面三份结果；RSGbothSB-L18-s1 的 held / 公开读数（选它滚池的理由：两个 seed 里零分更少 39 对 44、转弯完成 87.5 % 对 75 %）。
新池的状态、比例、任何新臂的读数都还没有。

## 1. 池

- **rollout**：`RSGbothSB-L18-s1` 在 onpol-train（`experiments/onpol_nuplan/results/split/train.txt`，6 个交错 chunk，与 `on1_L18RB0` 同一份列表）上
  18 决策滚一次，与 RB0-F-s0 滚 `on1_L18RB0` 完全相同：`ol_chain.sh roll1` 原样（run.sh sh30、c0b_chain.OVERRIDES、after160 缓存、nuPlan 默认 serving：
  控制器 gains 与 RB0 那次的 controller-config 相同，`rel_front_steering_angle_weight` 5.0，不设 JEV_VCONT / JEV_LEAD——RB0 那次的 driver 也没有设，
  任务书里写的「VCONT 1.0 LEAD 1」是 PAI 的 serving，在这条 nuPlan 路径上照「完全相同」处理），pool job、卡 1–5。
- **行** `L18SB1`：`on_rows.py geo`（`fut` = 当时的人日志恢复目标，Amendment 2；agent、road 栅格）+ `tok`（R1 与 interpreted control smoke10eY）+
  `ol_labels.py gate`（R1 / R2），`rm_build.py pool`（路线标签 route.npz + 路线记忆特征）→ 池 `on1_L18SB1`；侧别标签副本 `on1_L18SB1s`（`rf_sidetag.py pool`，
  只加 tside / pside / rside）。
- **池闸**（任一不过则停在这里，不训练）：R1 一 ulp 级 + 对照过、R2 过（ol_labels gates.json）；行数 ≥ 10 000、route ok ≥ 0.95、`route0_share_005` ≥ 0.99
  （`poolgate.py`）；cov20 记录（参照 on1_L18RB0 0.461）；sidetag R1（同 inode、tab 原数组逐字节相同、特征文件相同）。
- **报告**：新池右转 out : in（偏外 > 0.5 m : 偏内 < −0.5 m）对 on1_L18RB0 的 321 : 77，左转同样报。

## 2. 臂

公共配方 = RSGboth / RSGbothSB 的温和 schedule：`--init RB0-F-s0`（唯一允许的 init）`--steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1
--label-bank --agent-lam 10 --agent-on-only --shape --ho-w 3 --rm-lr 3e-4 --on-k 13 --on-split alpasim_nuplan/onpol-train --on-mode both`，60 步 preflight smoke，
seed 0 / 1。脚本 [scripts/rf_round2.sh](../scripts/rf_round2.sh)。

- **R2SB-L18-s{0,1}（主）**：`--on on1_L18SB1s`。
- **R2SBU-L18-s{0,1}（次）**：`--on on1_L18U2s` = on1_L18RB0s（r1）+ on1_L18SB1s（r2）的并集。trainer 的 `--on` 只收一个池；并集用已有的合池工具
  （`r2merge.py` → `ol_pool_merge.py`，第二轮 on1_L18R2 / on1_L18R2s 就是这么做的）做成一个池，trainer 不改。
- **侧别权重规则（看数之前定）**：对每个训练池按它自己的 rside 计数，**右转 out : in > 1.5 时配平到 1 : 1**：`rside=R_in:2,rside=R_out:2·n_in/n_out`
  （RSGbothSB 的约定：77 × 2 = 321 × 0.48）；≤ 1.5 时不加权（均匀抽）。左转不动。规则在训练 job 里机械执行，计数与所用权重写进 `weights_<tag>.json`。

## 3. 读数

每臂：held 460 @18（held.py 460 原始；ol_confirm 452 配对，log 聚类 bootstrap，seed 平均）、rf_follow 转弯完成（24 个 held 转弯）、rf_inside（内切份额，
按左右）、rf_tax extract + rf_sidebias held → rf_sidetag read（plan / 实际 2 s 横向，按侧，按 scene 配对）、公开 1491 / 10 决策护栏（1 479 个共同 scene，
ol_confirm）。参照 RSG-L18、RSGboth-L18、RSGbothSB-L18（各两 seed，存档读数）。

## 4. 判定线（seed 平均判，逐 seed 报；RSG = RSG-L18 两 seed 平均）

1. held 零分 ≤ RSG + 3（460 原始，RSG 34 → ≤ 37）；
2. 公开护栏零分 ≤ RSG + 5（RSG 45 → ≤ 50）且均分 ≥ RSG（0.9480）（点估计，报配对 CI）；
3. 转弯完成 ≥ 85 %（24 个 held 转弯）；
4. 右转内移：held 右转 plan 2 s 横向，臂 − RSG 按 scene 配对的 95 % CI 含 0。

四条全过 = 该臂「过线」，可作为替代 RSG 的候选（仍需 PAI 读数，本登记不含）。主臂 R2SB 不过而次臂过：记录，不把次臂升为主结论（多重比较）。
同时报对 RSGbothSB 的配对差（这一轮在修法 1 之上是否有增益）与新池比例是否变平（修法 2 的机制问：策略好了，偏外状态是否自然变少）。
