# 预登记：PAI 转向权重降到可提交的 10，重读参照与两个候选（不训练），2026-10-16

背景：官方 CLI `e2e_challenge/competitor_cli/alpasim_challenge.py`（部署 commit 0bb4c4b，`CONTROLLER_GAIN_BOUNDS`）把
`rel_front_steering_angle_weight` 限在 [0, 10]，超出时提交直接 `SystemExit`。10-13 以来所有 PAI 读数都用 20：steer_long gains pilot、
[d309](../../../research/decisions/309.md) 的 RSG-L18-s0-gate 0.586、[d286](../../../research/decisions/286.md) 的 RBSG2 0.599、
[d292](../../../research/decisions/292.md) 的 holdlead +0.052。这些都不是能提交的配置。本文件在任何读数之前写好并推送，之后只追加补记。

## 1. 读数（全部在 box，同一路径）

`pai_box.sh`，gate serve（`JEV_TRACK` 不设 = v8 默认），`SERVE` 默认 `JEV_VCONT=1.0 JEV_LEAD=1`，CONC 4，`CHUNKS_FROM` = Tokyo `r281/rsg` 的 8 个
chunk 列表（281 场景），每 chunk 一个池 job（`--vram 28 --cpu 6 --ram 40`）。和已有 box 参照 `recovery_envelope/pai/RSG-L18-s0`
（gains 20，code b7bcd3d3，8 / 8 chunk 完成）走同一条路径。输出 `$DATA_DIR/runs/route_follow/pai_g10/<label>`。

| label | checkpoint | 转向权重 | serve | 池优先级 |
|:--|:--|--:|:--|--:|
| g10 | RSG-L18-s0 | 10 | 默认 | 15（最先跑；所有配对的参照） |
| g10hl | RSG-L18-s0 | 10 | + `JEV_VHIST=holdlead` | 13 |
| rbsg2g10 | RBSG2-L18-s0 | 10 | 默认 | 13 |
| g5 | RSG-L18-s0 | 5（默认，不设 WIZ） | 默认 | 12 |
| g20（已有） | RSG-L18-s0 | 20 | 默认 | — |

报告：`rf_pai281.py --rows g20=… g10=… [g10hl=… rbsg2g10=… g5=…] --ref g20,g10 --lists exam=hard_exam.txt control=control.txt --vs g10`
（均分、零分分类、交接 spin、转弯代理；按场景配对 bootstrap 10 000 次）。两个池 job：early（g10 的 8 个 chunk 完成后，g10 对 g20）、
full（全部完成后）。只比所有 label 都有分的场景（`rf_pai281.py` 的 common）。

## 2. 判据（读数之前写死）

1. **参照**：g10 是此后所有 PAI 读数的配对参照，也是 PAI 提交配置的底。g10 − g20 只描述上限 10 的代价，不判。
2. **增益取值**：g5 − g10 配对均分 CI 下界 > 0 且零分不多于 g10 → 提交用默认 5（不传 gains）；否则用 10。
3. **候选进入提交配置**（holdlead、RBSG2 各自对 g10）：281 全集配对均分 CI 下界 > −0.01 且点估计 > 0；考题 72 点估计 ≥ −0.01；
   零分 ≤ g10 + 2（容差，按用户 10-16 的规则）；交接 spin ≤ g10 + 1。两个都过 → 下一步读组合 RBSG2 + holdlead（不在本轮）。
4. 本轮不训练、不提交；提交由用户定。

## 3. 成本

4 个读数 × 8 chunk × 约 52 分钟 ≈ 28 job 小时；4 张卡每张 2 个栈（`--vram 28`）共 8 个并行，约 3.5 小时墙钟；g10 先跑，约 1 小时出参照。
证据等级上限：每臂 1 seed、281 场景、按场景配对。
