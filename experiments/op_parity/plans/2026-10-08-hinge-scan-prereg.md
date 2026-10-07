# op_parity hinge-scan 预登记（2026-10-08，任何扫描臂打分之前写定）

## 问题

第 170 条：drivable-SDF hinge 在 lambda 30 / margin 0.5 m（SH30）全量上 navtest +0.87、navhard +1.83，raw plan 出界真实下降。只跑了这一个超参点，且相对 P2H10（lambda 10 / margin 0.3）
lambda 和 margin 同时变了。本线：(1) 在 pilot 规模上把 lambda 和 margin 分开，看有没有比 SHP（30 / 0.5）更好的点；(2) 只有某点按下面的规则胜出才上全量；
(3) 独立地把已有 SH30 全量 checkpoint 放到 WOD val 上（决策 155 的 harness）看跨域转弯。

## 设置

- pilot 规模与 SHP 完全相同：`pp_train.py --arm P2 --frames warp --host`，数据 `navtrain_full.s{2,3,4}of12`，split `navsim/op-parity-s234`，3 000 步 x 64，warmup 100，seed 0，`--eval-every 1000`，无 `--mem`。
  只改 `--hinge-lam` 与 `--hinge-margin`。
- 网格 lambda in {10, 30, 100} x margin in {0.25, 0.5, 1.0}（m），已知点 (30, 0.5) = `SHP-F-s0` 复用，不重训。其余 8 臂：`SC-L<lam>M<margin x 100>-F-s0`
  （L10M25, L10M50, L10M100, L30M25, L30M100, L100M25, L100M50, L100M100）。
- 种子噪声参照：`SHP-F-s1`（同配方 seed 1），只用来给「胜出」设一个高于种子散布的门槛，不是网格点。
- 参照臂：`SHP-F-s0`（所有差值都对它做配对）。
- 打分只走 `jevdrive.bench`（navtest 12 146 token，frames warp）；inside-cut / cannot-make-turn 用 `turn_oracle.py replay` + `Data`（four_dirs 口径，DAC 失败 token 上的 devkit replay）；
  raw plan 出界用 `rh.py proxy --name`（全 token 的 devkit replay 出界 vs raw 8 点 footprint 出界）。

## 读数（每臂，对 SHP-F-s0 的配对差，log 聚类 bootstrap，B 4 000，seed 0，jevdrive.stats）

1. navtest EPDMS（全体）、直行 token EPDMS（逻辑 4 s 航向变化 < 5 deg）、EP（全体）。
2. raw plan 出界率（全 token）；DAC 失败率（全体）。
3. 转弯 > 45 deg：inside-cut 率、cannot-make-turn 率；> 20 deg inside-cut 率。

## 胜出规则（候选臂 c 同时满足才算胜出）

- W1：EPDMS（全体）对 SHP-F-s0 的差，**Bonferroni 校正后的 99.38% CI（alpha 0.05 / 8）下界 > 0**，且点估计 >= |SHP-F-s1 - SHP-F-s0|（种子散布，全体 EPDMS）。
- W2：无直行 / EP 代价：直行 EPDMS 差与 EP 差的点估计均 >= -0.2，且各自 95% CI 上界 >= 0（不显著变差）。
- W3：raw plan 出界率（全 token）点估计不高于 SHP（差 <= 0）。第 170 条的「真实路径改善」口径：不能是只在 replay 出界上换分。
- 多个臂通过时取 EPDMS 差最大者；没有任何臂通过 -> **SH30 保留**，扫描结果作为曲面报告（lambda / margin 主效应、边际趋势）。
- inside-cut / cannot-make-turn 只报告，不作闸门。

## 上全量（当且仅当有胜出点）

胜出点 `SW`：P2H10 配方（12 个 navtrain shard，`navsim/op-parity-full`，10 000 x 128，warmup 300，seed 0 / 1），打分与 SH30 完全相同：navtest、navhard（G 帧）、HUGSIM 64 `spec_plan_smooth`，
raw-vs-replay 几何与 four_dirs replay。对 SH30 和 P2H10 的配对差都报告。判据（打分前写定）：navtest EPDMS 对 SH30 的 CI 排除 0 且为正，直行 / EP 无代价（同 W2），raw plan 出界不升。
达不到 -> 报告 SH30 保留，不改任何结论。

## WOD val（独立，便宜）

`SH30-F-s0 / s1` 现成全量 checkpoint 经决策 155 harness 服务（`pp_hugsim.py onnx` + `pp_wod.py bias` + `wod_zeroshot_openpilot.py --bias`，与 P2H10 完全同一路径），
WOD val 479 个 rater 帧：RFS、ADE@3s / 5s，对 P2H10（seed 均值）和 shipped 的配对 CI（按序列 bootstrap，B 4 000）；转弯帧（intent 左 / 右）5 s 终点相对 top-rated 轨迹的朝内偏差
（`wod_gap_turn_side.py` 口径：朝转弯内侧为正，中位数、> 1 m 朝内帧数，以及与 P2H10 的配对差）。不做 WOD test 提交。此项不进胜出规则，也不改上面任何判据。

## 预算与约束

pilot 9 次训练约 10 分钟 / 次（24 GB VRAM, 6 核, 40 GB RAM），约 1.5 卡时；全量（如触发）2 x ~45 分钟加 HUGSIM；WOD 服务约 0.3 卡时。总预算约 5 卡时。
所有作业走池（`jevdrive.cl submit`），其他线（mixed-domain、s2-thinhead、wod-pref）共享箱子。不改 `research/decisions.md`，不写 HTML，不提交 WOD test。
