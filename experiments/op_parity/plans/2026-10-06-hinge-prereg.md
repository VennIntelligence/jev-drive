# op_parity：足迹 drivable SDF hinge 预登记（2026-10-06，任何 hinge 臂打分之前写定）

## 问题

第 147 条：P2 的 DAC 失败（navtest 4.4%，WA-JEPA 1.8%）里约 0.9 pp 来自 plan head 的训练目标（thin head 上加 drivable hinge 少 0.93 pp [0.11, 1.69]，分数不掉），其余约 1.9 pp 在冻结的视觉特征上。本登记测最便宜的一步：**在 P2 的全量微调里直接加足迹 SDF hinge，navtest DAC 失败率是否降约 0.9 pp，EPDMS 是否不掉。**

## 臂

- **P2H10**：第 144 条 P2 配方原样（navtrain 全量 103 288 token，W 协议，视觉冻结，plan 通路 + adapter，batch 128，10 000 步，warmup 300，anchor 行 0.25，λ_i 1、λ_c 3、λ_d 30，seed 0 / 1），只在模仿行上加 `hinge_lam * mean(relu(0.3 − sdf))`。plan 的 8 个位姿（后轴系，0.5 .. 4 s）线性插值到 0.1 s，nuPlan 足迹四角（前 4.049 m / 后 −1.127 m / 半宽 1.1485 m）取 drivable SDF 双线性采样。几何与 probe decoder 同一份（`lib/drivable_hinge.py`，对拍 `opb_probe.corners_torch / sdf_at`）。标签：`opb_labels.py` 全 navtrain 12 个分片（约 1–3 min CPU），与 probe 同一 map 层（ROADBLOCK / INTERSECTION / CARPARK_AREA，与 scorer 的 DAC 多边形一致）。
- λ = 10（probe 用值）。**λ = 3 只在小读 EPDMS 掉超线而 DAC 已动时才开。**
- 参照：第 144 条 P2-F-s0 / s1（同 seed 同行流）与 WA-JEPA（同 devkit 同 metric cache）。不重训 P2。
- 注：probe 的 imitation 损失量纲（米上的 Huber）与这里（按 σ 归一的 Huber 求和）不同，所以「λ = 10」在数值上不是同一个相对权重；不做换算，λ = 3 是第二档。

## 读数与统计

- **主读数**（navtest 12 146 token，W 帧，devkit navsim main @ 0a380a9 v2 EPDMS）：EPDMS 与 DAC 失败率（DAC < 1 的 token 占比，%），**hinge − P2**，两 seed 逐 token 平均后按 navtest log 聚类的配对 bootstrap（B 10 000，`jevdrive.stats.paired`）。同 seed 的配对差一并报。副读：各子分、ADE vs log、速度比；对 WA-JEPA 的同口径配对差。
- navhard two-stage，**GIMM 帧（协议 G）**，第 144 条同 harness（`nav_harness.py`），按 stage-1 log 聚类的配对 bootstrap，对 P2-G（已有）与 WA-JEPA。
- HUGSIM 64 scenario，`exam` 与 `spec` 两个 preset，同第 144 条 harness，P2 同口径对照；HD 配对按 scenario。次要：卡死、起步停滞、打转、前景碰撞计数。
- 成功判据（预期值来自第 147 条）：DAC 失败率 −0.9 pp 量级（点估计 ≤ −0.6 视为「到位」），EPDMS hinge − P2 ≥ −0.3；只做报告，不据此改配方。

## 小读闸门（先于全量，一个 seed）

P2H10-s0 用完整配方训练（≈ P2 的 21–62 min，在池里与 P2 同一份开销；hinge 只加 164 个角点采样，可忽略），navtest 全量打分，对 P2-F-s0 配对：

- 停：EPDMS 差 < −0.3（点估计）**且** DAC 失败率降幅 < 0.3 pp → 停，报告；
- 停：DAC 失败率降幅 < 0.3 pp（hinge 没动 DAC，不论分数）→ 停，报告；
- λ = 3 小读：EPDMS 差 < −0.3 而 DAC 降幅 ≥ 0.3 pp → 同一小读再做 λ = 3，同一规则；
- 过：EPDMS 差 ≥ −0.3 且 DAC 降幅 ≥ 0.3 pp → 放开 s1、navhard、HUGSIM。

估算：训练 1 run ≈ 21–62 min（P2 实测 2.8–6.7 it/s，CPU 共享），全程 < 3 h，不另做 profile。分阶段：标签 → s0（1）→ 闸门 → s1 + readouts（其余）。

## 判读（写在结果之前）

- DAC 失败 ≈ −0.9 pp 且 EPDMS 不掉：第 147 条的 (c) 部分在微调里兑现；剩余约 1.9 pp 在视觉表征，下一步是 encoder 解冻 + 稠密 SDF 辅助头。
- DAC 失败降得明显少于 −0.9（< −0.3）：thin head 上的收益不迁移到全模型微调（原因候选：hinge 与 anchor / imitation 的冲突、λ 量纲）；(c) 的幅度要下调。
- EPDMS 掉而 DAC 动：hinge 以进度或安全换 DAC，报告交换。

## 限定（预先写明）

两 seed，视觉冻结，W 协议训练（navhard 读数用 G 帧是 P2-G 的口径：W 训练的检查点喂 GIMM 帧）；HUGSIM 每 scenario 单次，重跑差约 0.2 HD / scenario；hinge 只作用在 plan 的 8 个位姿（后轴系，不含 LQR 回放），第 147 条里约 28% 的失败只在 LQR 回放里出界，这部分 hinge 看不到。
