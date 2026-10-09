# C0b 预登记：OT30（离轨行）在 AlpaSim 本地闭环上的判定线（2026-10-09，任何 OT30 闭环分数之前写定）

lane C0b。接 decision 198（OT30 的开环结果与 m = 1 离线退步）、199（400 scene 的 SH30 / AP2）。只有本地 run，不向 AlpaSim 提交任何东西。

## 问题

在 op_parity 里 OT30（SH30 配方加约 10% 静态离轨重投影行，`OT30-F-s0 / s1`）把离轨状态下的修回比例从 0.21 提到 0.64。闭环里 SH30 的零分来自 offroad、出 corridor（`left_corridor_laterally`）和 at-fault 碰撞（d199：8 / 7 / 8）。
离轨行能否减少闭环里的 offroad + 出 corridor 零分，同时不增加 at-fault 碰撞零分？d198 同时发现 AlpaSim 输入标准下离线 m = 1（第一次决策，伪造历史）EPDMS 掉 1.49 [-2.69, -0.46]，所以冷启动表现要单独读。

## 设置

- scene：打分当时所有带 `.done` 的公开 shard 的全部 scene（固定一个列表，本批 700 个，part001、002、003、005、007、008、009；列表写进结果目录）。
- driver：`OT30-F-s0`、`OT30-F-s1`，输入标准与 `SH30-F-s0` 完全相同（`run.sh <dir> sh30`，`SH30_TAG=OT30-F-s{0,1}`，冷启动 `backwarp`，不改任何 driver 代码）。对照 `SH30-F-s0`（d199 的同一个 driver / 同一份运行，新增 scene 用同设置补跑，重叠 scene 先验证分数逐一复现）。
- 仿真确定（d199：AP2 两次运行 400 scene 分数逐一相同），所以 run 内没有重复噪声；seed 方差只能由 s0 / s1 两个训练 seed 提供，SH30 只有 s0 一个闭环点（SH30-F-s1 不在本批）。

## 读数

1. 每个 OT30 seed 的 mean scene score、score = 1 的个数。
2. 零分按类计数：at-fault 碰撞、offroad、出 corridor、其他；偏慢 scene（0 < score < 1）个数；at-fault 事件数（`offroad_or_collision_at_fault` 之和）。
3. 每个 OT30 seed 对 `SH30-F-s0`、两个 seed 的均值对 `SH30-F-s0`：按 scene 配对的差，95% bootstrap CI 按 log（nuPlan 的 `日期_车辆`）整簇重采样（10 000 次，seed 0）。
4. 冷启动：决策 0 是三个 driver 输入完全相同的一步（仿真起点相同），读 OT30 与 SH30 在决策 0 的计划终点（4 s）的差的分布（均值、p90、最大、大于 3 m 的 scene 数）；以及零分 scene 中 rollout 在前 3 次决策内就结束的个数（`drive` 调用数 ≤ 3）。

## 线

OT30 成为比赛 driver 候选，当且仅当同时满足：

- 两个 seed 的 mean scene score 平均 对 SH30-F-s0 **>= +0.010**，且这个配对差的 CI 下界 > 0；
- at-fault 碰撞零分的个数（两个 seed 的平均）**不高于** SH30-F-s0 的个数。

冷启动按读数 4 报告，不设线。未过线 = 不进候选，写明哪一条没过。线之外的读数（按类零分、按 shard）只描述，不事后改线。

## 限定（写在前面）

- SH30 只有 s0 一个闭环点：差的方差里含 SH30 自己的 seed 方差，OT30 两个 seed 的平均不能分开这一项。
- scene 取自少数几段 drive 的多个 scene，同一 log 内相关：CI 按 log 整簇。
- 本地渲染，没有官方环境的对分。
