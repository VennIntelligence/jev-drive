# op_parity：急转弯差距是不是训练问题（预登记，2026-10-06，任何 T 臂打分之前写定）

## 问题

第 144 / 147 / 148 条：P2 在 |航向变化| > 45° 上 DAC 失败 11.3%（WA-JEPA 3.2%），EPDMS 77.5 vs 87.4；hinge λ=10 在转弯里有帮助（右转 DAC −1.58 pp），HUGSIM 不动。视觉冻结。本登记**只改训练**，问：训练分布 / anchor / 损失权重能否闭合急转弯差距。

## 设置（与 unfreeze pilot 的 F 臂同一份）

navtrain 分片 `navtrain_full.s0of12 + s1of12`（17 216 token，训练集 16 884），W 帧，视觉冻结，P2 配方（ego / pose / command adapter，anchor 行 0.25，λ_i 1 / λ_c 3 / λ_d 30），3 000 步 × batch 64，warmup 100，seed 0 / 1，同一 init。脚本 `scripts/pp_train.py`（新增三个开关），链 `scripts/pp_turn_chain.sh`。

## 臂

- **H**：P2 + hinge λ=10（tag `HP`）。基线。
- **T1**：H + 转弯平衡采样（tag `T1P`）。每个 token 的「训练日志航向变化」= 未折叠后的 8 个未来位姿 yaw 的末值 |Δψ|（与 opj 的 dyaw 同口径）。目标质量 <5° / 5–20° / 20–45° / >45° = 0.35 / 0.15 / 0.25 / 0.25（>20° 合计 50%，对 <20° 的 50%）；自然质量 0.533 / 0.195 / 0.169 / 0.104，所以逐 token 权重相对 <5° 档为 1 / 1.17 / 2.25 / 3.67。anchor 行照旧以 0.25 概率标记。
- **T2**：T1 + 在 |Δψ| > 20° 的 token 上关掉 anchor 蒸馏（tag `T2P`）：这些行只做模仿，plan 不再被拉向 shipped。有效 anchor 比例约 0.25 × 0.5 = 0.125。
- **T3**（仅当 turn-gain.md 判为 shrinkage 才跑）：T2 + 2 s 之后的位姿 y / yaw 项权重 ×2（tag `T3P`）。其他判定不跑。

## 读数与判据

- navtest 全量 12 146 token（W 帧，v2 EPDMS，同 devkit），EPDMS 与 DAC 失败 %，按 op_probe joint 分桶（<5 / 5–20 / 20–45 / >45°，左 / 右转，加 >20° 合并桶），两 seed 平均，按 136 个 log 聚类的配对 bootstrap，对 H 与对 WA-JEPA。参照另列 unfreeze pilot 的 F（无 hinge）。
- **护栏**：整体 EPDMS 臂 − H ≥ −0.3，否则该臂判不过。
- **闭合**：>20° 合并桶 EPDMS：(臂 − H) / (WA − H) ≥ 0.5 记「闭合一半以上」。
- 臂同时过护栏且闭合 ≥ 0.5 ⇒ 对该臂和 H 跑 HUGSIM 64（exam 与 spec，两 seed），单独报 23 条转弯路线。否则不跑 HUGSIM。
- 一句话结论按 >20° 桶：训练单独闭合 ≥ 一半 →「是」；0 < 闭合 < 一半 →「部分」；≤ 0 →「否」。

## 前置条件

`experiments/op_probe/results/turn-gain.md` 首行判决：cap-control 或 cap-model → 不训练，停下报告（修复在别处）；shrinkage 或 mixed → 继续。

## 限定

pilot 规模（17 k token、3 000 步），两 seed；全量训练下的转弯效应可能不同。转弯平衡会降低直行 token 的有效样本量。hinge 只看 plan 的 8 个位姿，LQR 回放里的出界不在其内。
