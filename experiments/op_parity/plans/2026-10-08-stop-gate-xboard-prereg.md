# stop gate 跨榜（P2H10，仅 serving 侧）预登记

2026-10-08，打分前写。问题：WOD 上让停车帧 adapter bias = 0 的规则（`--stop-gate 0.5`，decision 162 / 164，results/wod_launch.md）放到 navtrain 训练的 P2H10 自己的榜上，是帮、无害还是伤。不训练，不调阈值。

## 臂（P2H10-F-s0 / s1 存档权重，两 seed 取均值）

| 臂 | 定义 |
|:--|:--|
| BASE | 存档结果（navtest：legacy `opi_lb_navtest_warp-cinque_PPP2H10-F-s*__base` devkit CSV，同 harness、同 token、同 metric cache；HUGSIM：存档 `P2H10-F-s*_spec_plan_smooth-rr1 / rr2`，两次重复取均值）。不重跑。 |
| SG | 喂入 ego vx（`ego[4] * 10`）< 0.5 m/s 时 adapter bias 恒为 0（等价于 present = 0）；阈值 0.5 固定，不扫。HUGSIM 里 vx 是模型时钟（x1.25）下喂给 adapter 的那个值，和训练时规则读的是同一个输入。 |
| DN | 所有速度下减去该 seed 在全 navtest 12 146 token 上的 adapter bias 均值（decision 162 的 `biasdenav`）。均值是 label-free 统计（只用 ego），HUGSIM 用同一个向量。 |

实现开关（默认路径不变）：navtest `P2H10-F-s*@warp:sg` / `:dn`（`jevdrive/bench`，`pp_train.PModel.gate / bias_sub`）；HUGSIM `--opts '{"parity": {"stop_gate": 0.5}}'` 或 `{"parity": {"sub_bias": <meanbias npy>}}`（`lib/parity_hugsim.py`）。
一致性检查（读结果前）：SG 在 fed 速度 >= 0.5 的 token 上的 plan 必须与存档 BASE plan 逐位一致（证明 BASE 复用合法）；DN 的均值向量与 SG 的触发 token 数落盘。

## navtest（12 146 token，W 帧）读数

- 主对比：EPDMS（SG - BASE、DN - BASE），两 seed 均值，按 log 分簇的配对 bootstrap（B 10 000，seed 0），附 NC DAC DDC TLC EP TTC LK HC EC 子分。
- 分层（同上，附 CI）：v0 < 0.5 m/s；起步 token（v0 < 2 m/s 且日志 4 s 距离 > 5 m）；行驶（v0 >= 0.5）；转弯 > 20 deg（|dyaw| > 20，与 hinge_strata 同口径）。v0 与 4 s 距离取 `op_probe/joint/navtest_tokens.parquet`（v0、path_len）。
- 触及量：SG 中被置零的 token 数（占比、其中起步 token 数）；plan 与 BASE 不同的 token 数。

## HUGSIM 64 闭环（preset `spec_plan_smooth`，decision 149 / 153 / four_dirs 的口径；每臂 2 seed x 64 场景一次）

- HD（SG / DN - BASE，场景为单位配对 bootstrap，seed 均值），起步停滞数（前 40 步峰值速度 < 1.6 m/s）、卡死数（max_steps 结束）、打转数（航向误差 >= 60 deg）、碰撞数（fg_collision + bg_collision 结束类），沿用 hugsim_spin10 的分类，`units.csv` 现成列。
- 振荡检查（SG）：从 `zs_steps.jsonl` 的 `parity.gated` 取每步开关；gate 翻转数按场景统计；记「振荡场景」= 一个场景内翻转 >= 6 次，或任意 20 个连续步内翻转 >= 3 次。画 / 列翻转最多的几个场景的速度轨迹（速度、gated、HD）与一个起步停滞场景。
- BASE 的噪声参照：rr1 与 rr2 同 seed 的 HD 差。

## 判定（读数前定）

- navtest 以 EPDMS 全集 CI 为准：helps = CI 下界 > 0；hurts = CI 上界 < 0；harmless = CI 整体落在 [-0.10, +0.10]（约 seed 间差的一半）；其余 inconclusive。分层只描述（与全集同标签规则），不单独下结论。
- HUGSIM 以 HD 全集配对 CI 为准：harmless = CI 整体在 [-0.03, +0.03]；helps = 下界 > 0 且起步停滞 + 卡死数不增；hurts = 上界 < 0，或打转数比 BASE 多 >= 2（臂均值），或出现 >= 1 个振荡场景导致该场景 HD 比 BASE 低 >= 0.1。
- 「一个司机跨榜成立」需要 SG 在 navtest 为 harmless 或 helps，且 HUGSIM 不是 hurts。DN 同规则，仅作为 decision 162 提的「navtest 上要付多少」的答案，不当候选。
- 起步停滞 / 卡死在 P2H10 基线上几乎为 0（decision 144：起步停滞约 1，卡死 0），地板效应：helps 在 HUGSIM 上几乎只能经 HD 体现，写入 caveat。

## 预算

navtest 4 个 plan 作业（各 1-2 min GPU）+ 4 组 devkit 打分（CPU）；HUGSIM 4 次 64 场景（SG x 2 seed，DN x 2 seed），每次约 0.75 场景小时，W = 6，预计合计约 1-1.5 卡时。不超过 2 卡时；不留作业。
