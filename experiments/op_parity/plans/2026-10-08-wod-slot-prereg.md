# wod-slot 预登记：训练 8 真 slot + 1 零 slot，WOD harness 却喂 9 个真 slot，这个错配是不是 P2H10 在 WOD val 上输给 shipped 的一部分原因

2026-10-08。打分前提交。接第 155 / 162 / 163 条与 mixed-domain 结果（`results/mixed_domain.md`）。代码 `scripts/wod_slot.py`，结果 `results/wod_slot.md`。

## 问题

navtrain 训练的臂（P2H10、SH30）训练时 policy 看 8 个真 slot + 1 个零 slot（最老的一个置零，`pp_train.PModel.forward` 里 `H + bias` 之后乘 valid）。
WOD val harness（第 155 条）warm-up 10 s 后喂 policy 9 个真 slot。mixed-domain 发现把最老 slot 置零会让 shipped 的 plan 平均移动 1.09 m，且 mixed 臂
训练时 WOD 行用 8 + 零比 9 真低 0.109 RFS。所以：P2H10 在 WOD val 的 RFS 7.708 vs shipped 8.005（-0.297），其中有多少是 serving / training 的 slot 错配？

## 做法

只改 serving，不重新训练。对 serving ONNX 在 policy 读取的 9 个 slot（队列 `cat_3` 的下标 24..32）里把最老的一个（下标 24）乘 0，
乘在 intent bias 加到过去 slot 之后（与训练一致：先加 bias 再置零），不动模型自己的递归队列。同一 harness（479 rater 帧 + extra，10 s warm-up，
每帧喂两次）。8 + 零的臂以 `<tag>_s8` 命名。9 真的参照用已存预测。

臂（7 个 serving，各约 15 min 一张卡）：shipped（ONNX 零 bias，`bias-P0.npz`）、P2H10-F-s0/s1、SH30-F-s0/s1、WP2-full-s0/s1（WP2 用 9 个真 slot 训练，对照，预期 8 + 零变差）。
臂组 = 两 seed 逐帧分数平均（沿用惯例）。

## 校验（读数前）

- G0 打分路径：用已存预测重算，shipped 8.005、P2H10 7.708（P2H10 臂组 = F-s0 + F-s1）、WP2 8.111，容差 0.002，不过则停。
- G1 构建路径：全 1 mask 的 ONNX 在 harness 上与未加 mask 的 ONNX 逐位相同（几个 target）。
- G2 mask 生效：shipped 的 `_s8` 与 9 真的 plan 差平均（20 个路点）应与 mixed-domain 记录的 1.09 m（33 点 plan，cache 行）同量级（0.4 – 3 m）；
  一并报告。不满足则查 mask 的位置而不是读结果。

## 读数与规则

主对比（配对 bootstrap over sequences，B 4000，479 rater 帧 cluster-mean RFS；ADE@3s / @5s 用 1 437 帧）：

- C1 `P2H10_s8 - P2H10`（错配本身）。
- C2 `P2H10_s8 - shipped`（9 真）与 C3 `P2H10_s8 - shipped_s8`（同一 serving 协议下的比较）。参照缺口：`P2H10 - shipped` = -0.297。
- C4 `SH30_s8 - SH30`，C5 `shipped_s8 - shipped`（不加训练的 shipped 对 slot 数有多敏感），C6 `WP2_s8 - WP2`（对照，预期 < 0）。
- 分层：全部、standstill（v0 < 0.5）、moving（v0 >= 0.5）、turn intent（intent >= 2）。

标签（沿用 `wod_launch_report.lab`）：错配对 P2H10 的缺口 **carries** = C1 点估计 >= 0.5 × 0.297 且 CI 下界 > 0；**part** = >= 0.25 × 0.297 且 CI 下界 > 0；
其余 **not**（含 C1 CI 跨 0）。若 C1 > 0 但 C5 同样 > 0 且同量级，说明收益来自 shipped 本身对 slot 数的偏好而不是训练协议匹配，写明，不记为错配证据。
方向性预期：C1 在 0 到 +0.10 之间（P2H10 的缺口主因是 standstill / 启动偏置，第 162 / 163 条），C6 < 0。读数按这个量级，结果不改标签规则。

## 偏离记录

结果文件列出一切偏离本文的地方。不触碰 `research/decisions.md`、HTML。
