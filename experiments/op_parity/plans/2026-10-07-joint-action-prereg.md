# op_parity：action 头与 P2 plan 联合训练，让 action 转得像 plan（预登记，2026-10-07，任何 J 臂训练与读数之前写定）

## 问题与依据

第 149 条：HUGSIM `spec` 用 action 头转向，P2 配方把 action 头蒸馏回 shipped Cinque（`pp_train.py` 的 distill 覆盖所有非 plan 列，含 action），而 on-policy action 通路（ONNX `model.on_policy.*`，与 plan 的 off-policy 通路不共享权重）根本不可训练。转弯路线上 action 请求 |κ| 只有 plan 平滑换算的一半；`spec_plan_smooth`（plan 0.5–1.5 s 航向变化 / 弧长）转弯 23 条 HD 0.333 对 spec 0.279，全 64 0.428 对 0.394（4 臂均值）。

训练前探针（navtrain s0of12，8 608 token，v0 > 3 m/s 的 5 716 行，本登记前算）：shipped action 曲率对日志曲率（t + 0.275 s）斜率 **0.63**（r 0.95，符号：action + = 右转，日志左正，取负后一致）；P2 的 teacher plan 平滑曲率对日志同窗口曲率斜率 1.08；action 对 plan 平滑曲率斜率 0.58。即 action 欠转在 NAVSIM 开环就存在（第 130 条：高相机致速度低估），不是 HUGSIM 特有；把 action 标签换掉就能直接改增益。

注意一个陷阱：HUGSIM 客户端给 adapter 的 vy = ay = 0（`lib/parity_hugsim.py`），NAVSIM 有真值 ay。ay / v² 就是当前曲率，action 头若学会抄 ay，到 HUGSIM 就读到「直行」。故 J 臂在 50% 的行上把 vy / ay 置零（独立 rng，不改行序）。

## 设置（与 HP-F 同一份 pilot 配方）

navtrain `navtrain_full.s0of12 + s1of12`，split `navsim/op-parity-full`，W 帧，视觉冻结，P2 配方 + hinge λ 10，3 000 步 × batch 64，warmup 100，seed 0。基线 **HP-F-s0**（已有，action 蒸馏到 shipped）。

J 臂相对 HP 只改：

1. on-policy action 通路可训练（`act_weights()`，lr 同 plan 通路 3e-5）；
2. action[0]（横向）退出蒸馏，换成标签 `k × max(1, v0)²`，Huber（δ 1，按 teacher action[0] 的 std 归一），权重 λ_a = 3，逐行速度权重 min(1, v0 / 3)，作用于所有行（模仿行与 anchor 行；anchor 的 plan 仍蒸馏到 shipped）；action[1]（纵向）仍蒸馏；
3. vy / ay 置零比例 0.5。

| 臂 | tag | 标签 k（右正，1/m） | 问的是 |
|---|---|---|---|
| JC | `JC-F-s0` | 模型自己当前 plan（detach）0.5–1.5 s 平均曲率，= `spec_plan_smooth` 的换算 | action 能否学成「平滑执行自己的 plan」 |
| JL | `JL-F-s0` | 日志曲率 @ t + 0.275 s（4 历史 + 8 未来位姿的三次样条，dψ/ds） | openpilot 自己的 action 定义（第 130 条「一致目标」），增益 1 |
| JW | `JW-F-s0` | 日志 0.5–1.5 s 航向变化 / 弧长 | 与 JC 同换算、标签来源换成日志 |

数值不调：λ_a、窗口、置零比例、速度权重都是此处写定的值。

## 反馈 / 稳定性读数

- 离线（`scripts/pp_joint_probe.py`，训练分片的 dev 行，v0 > 3）：action 增益（对日志曲率的斜率，全部 / 转弯 |κ| > 0.02 / 3–8 m/s / > 8 m/s），action 对 plan 平滑曲率的斜率与相关，**历史反馈增益 g_h**（把 4 个历史位姿与 ay 按过去 1.5 s 多走曲率 d 弯折，g_h = dκ_cmd / d；g_h ≥ 1 表示命令延续车自己过去的曲率，闭环里是积分器），vy / ay 置零后的斜率。
- 闭环（HUGSIM）：打转数、航向率翻号 / 100 移动步、振荡 run（≥ 8 次且 ≥ 15% 移动步，定义同 hugsim_specplan.md）、请求曲率 |κ| 与步间变化（第 132 条的过校正迹象）。

## 第一阶段：小读（seed 0）

HUGSIM `turn23` + `spin10`，四个 checkpoint（HP、JC、JL、JW）各跑 `spec` 与 `spec_plan_smooth`（同一 checkpoint 的 plan 平滑执行，作为该臂的内部参照），全部经 `jevdrive.bench`。

离线先判（不过的臂不进 HUGSIM 判读，仍报告）：gain_all ≥ 0.85；dev ADE 与 drift_off 相对 HP 变化 ≤ 0.03 m（plan 未被拖动）。

**判据（turn23 HD，单 seed）**：

- Δ1 = 臂 spec − HP spec；Δ2 = 臂 spec − 臂 spec_plan_smooth。
- 稳定性护栏：turn23 ∪ spin10 打转数 ≤ HP spec + 1；航向率翻号均值 ≤ 0.5 / 100 移动步；无振荡 run；g_h < 0.5。
- **过**：Δ1 ≥ +0.03 且护栏全过。取过的臂中 turn23 spec HD 最高者进第二阶段（相差 < 0.02 时取 Δ2 更大者）。
- **早停**：没有臂 Δ1 ≥ +0.03 → 本线停，报告（标签训练不能把 action 教成会转）。若某臂 Δ1 ≥ +0.03 但只败在护栏（打转 / 翻号 / 振荡 / g_h）→ 不进第二阶段，转「反馈臂」（下节）。

## 第二阶段：全量（仅第一阶段过时）

胜出臂按 P2H10-F 的全量配方（12 分片，10 000 步 × batch 128，warmup 300，hinge 10）加该臂的 J 改动，seed 0 / 1。读数：

- navtest EPDMS（bench，W 帧）：臂 − P2H10-F ≥ −0.3（plan 共享，护栏）。
- HUGSIM 64 `spec` 与 `spec_plan_smooth`：对 P2H10-F spec、P2H10-F spec_plan_smooth（hugsim_specplan 已存）、P2-F spec、WA-JEPA，配对 bootstrap（场景为单位），turn23 与全 64 分报，打转与翻号。
- 一句话结论：臂 spec 的 turn23 HD ≥ P2H10-F spec_plan_smooth − 0.03 且全 64 臂 − P2H10-F spec 的 CI 在 0 之上 →「action 头追平平滑 plan」；只有前者 →「部分」；都不 →「否」。
- 过了才做 B2D 路口转弯（另行登记）。

## 反馈臂（条件触发，不在本次启动范围）

触发：第一阶段某臂增益到位（Δ1 ≥ 0.03）但败在护栏，或第二阶段臂 spec 明显低于平滑 plan（< −0.03）。设计：在 factor_wm 平面重投影引擎（`experiments/factor_wm/scripts/fw_p2.py`）里由该臂**自己的 action 头**执行横向（clip + 0.2 s 延迟），纵向按 plan；只取 v > 3 m/s 的访问状态（避开第 143 / 146 条的静止起步陷阱），标签 = 引擎恢复路径按同一换算得到的 action 曲率；这些行**只训 action**，plan 不训。读数同第一阶段。触发时另写附录再启动。

## 限定

pilot 规模、单 seed、HUGSIM 每场景单次；turn23 来自 17 个 scene，场景级差 < 0.2 不算证据。vy / ay 置零也改 plan 的输入分布，所以 J 臂与 HP 的 spec 差含 plan 变化，臂内 spec 对 spec_plan_smooth 才是纯 action 对照。HUGSIM 只作评测，不在其场景上训练。
