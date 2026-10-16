# DESIRE-CAP 预登记：底模有没有 desire 通路，我们的 adapter 训练是否把它杀了（2026-10-16，任何 desire 读数之前）

本文件在任何 desire 臂的 plan 读数之前 push；之后只追加补记，不改上文。已知的只有行数（日志 / 路线记忆几何统计，不是模型输出）。

## 1. 问题

真车上 openpilot 在司机打灯时由 DesireHelper 给 desire（laneChangeLeft / Right；低于 20 mph 时 turnLeft / Right），shipped 模型据此执行整条变道。
我们的 AlpaSim driver 从不喂 desire（`PModel.forward` 的 desire 块恒为 0，`jevdrive/op_adapt.policy_feeds`），变道从没学会（第 284 条 0.05 m；
第 299 条 PAI 要求 4 做 1、held 0 / 4；第 302 条反事实行 ≤ 0.15 m；第 304 条策略被拉回车道中心）。第 49 条（CARLA，零状态预热）：Cinque 的
laneChange 脉冲方向对、幅度小（3 s 处 0.6–0.9 m）。第 92 条：turn desire 在真实数据上不让 plan 变好。
**问题**：(a) shipped Cinque（P0）在我们训练用的同一输入协议上，有没有一条「给 desire → plan 横移一条车道 / 转弯」的通路；(b) 谱系里哪一段把它保留、削弱或杀掉。

## 2. desire 怎么进模型（读代码与 ONNX，非读数）

Cinque ONNX：`desire` (8,) 进 `state_desire_q` (132, 1, 8)（20 Hz、6.6 s），新的一步接在末尾；`reshape (33, 4, 8)` 后按 4 步取 max，转置成策略子图的
`_to_copy` (1, 33, 8)：33 个 0.2 s 块，块 32 = 最近 0.2 s。modeld 只送上升沿脉冲（`desire - prev > 0.99`，第 0 位清零），DesireHelper 在变道期间保持
desire 不变，所以模型看到的是一个随时间后移的单脉冲。本实验在 `_to_copy` 上直接放这个脉冲：desire d、脉冲年龄 Δ → 块 32 − round(Δ / 0.2) 的第 d 位 = 1，
其余 0（= modeld 在脉冲后 Δ 秒时的队列内容）。desire 编号（log.Desire）：turnLeft 1、turnRight 2、laneChangeLeft 3、laneChangeRight 4。
其他输入不变：同一帧 token（Cinque 冻结编码器的 W 协议，与第 326 条、训练行同一协议）、同一 ego 行（人的历史 H）、路线记忆臂同一重建记忆。
**开环限制**：帧不随 plan 动。Δ = 0（脉冲刚到）时的输入与真车上变道开始那一刻一致（之前的帧都是保持车道）；Δ > 0 时真车的帧已经在横移，开环里没有，只作剂量 / 时序参考。

## 3. 行

**NAVSIM**（CAP-LEDGER 的行，`navsim/capledger_test`，147 个 test log 的 2 Hz 帧，没有任何 checkpoint 训练过；格子定义见
`experiments/cap_ledger/plans/2026-10-16-cap-ledger-prereg.md` 第 2 节，本文不改）：

| 集合 | 定义 | 行 / log |
|:--|:--|--:|
| **C_fast**（主，变道） | C（自由路直行巡航，5 ≤ v0 ≤ 15，航向变化 < 10°，无路口 / 前车 / 停止线 / 冲突）且 v0 ≥ 8.9 m/s（DesireHelper 的变道最低速 20 mph） | 543 / 89 |
| C_slow | C 且 v0 < 8.9（真车在这个速度打灯给的是 turn desire） | 474 |
| **T**（主，转弯） | T_L ∪ T_R（路口内 4 s 航向变化 ≥ 30°，人在转），desire 取人转的一侧 | 601 + 435 / 86、81 |
| LC_L / LC_R | 人在变道（同 lane group 换到相邻车道，4 s 横移 ≥ 1.5 m） | 96 / 21、189 / 36 |

**PAI 自身状态**（`on1_PAIonp`：PAI train160 的 160 个 scene 上 on-policy 决策行，帧 token 已在 W 协议缓存，路线记忆 `route_mem/feat/on1_PAIonp.npz`；
不在本文任何 checkpoint 的训练集里：RSG-L18 用 on1_L18RB0、RBSG2-L18 用 on1_L18R2，其余只用 NAVSIM）。按喂入的路线记忆（rmem 布局，后轴系，左 +）分：

| 集合 | 定义 | 行 / scene |
|:--|:--|--:|
| P_keep | v0 ≥ 5，turn = straight，0–60 m 路线点全有效，+40 m 横向 \|y\| < 0.5、+60 m < 0.7，0–60 m 航向 < 5° | 2 182 / 65 |
| P_lca（路线要求变道） | v0 ≥ 5，turn = straight，0–60 m 全有效，\|y(+0 m)\| < 1.0，2.0 ≤ \|y(+40 m)\| ≤ 5.5，\|y(+60) − y(+40)\| ≤ 1.0，+30–60 m 航向 < 10°、0–60 m < 25°；方向 = y(+40) 的符号 | 99（L 23 / R 76）/ 15 |
| P_turn（路线要求转弯） | turn = approach（≥ 45° 转弯入口前 0–40 m，表里的 side） | 左 789 / 11、右 754 / 11 |

## 4. 臂

P0（shipped Cinque 权重走 parity 路径，无 adapter）；P2-F、P2H10-F、YR10m10-F、YS-L18、RB0-F、RSG-L18、RBSG2-L18，各 s0、s1（RB0-F 不在用户列表里，
它是 RSG / RBSG2 的前驱，必须有才能定「哪一段」）。阶段读数 = 两 seed 逐行平均。前驱：P2-F ← P0、P2H10-F ← P2-F、YR ← P2H10、YS ← YR、RB0 ← P0、RSG ← RB0、RBSG2 ← RB0。

desire 臂：none；TL / TR / LCL / LCR × Δ ∈ {0, 0.4, 1.0, 2.0} s（共 17 次前向 / 行）。路线记忆臂（RB0、RSG、RBSG2）另跑「记忆清空」（全 0 = served 的空记忆）
× {none, 4 个 desire @ Δ 0}，只报。

## 5. 读数

plan 后轴系（x 前、y 左、航向左 +），在 t = 1、2、4、6、8 s 插值。每行每个 desire 臂对同一行 none 的配对差，朝 desire 一侧取正（左 desire s = +1、右 s = −1）：
响应 R_y(t) = s (y_d(t) − y_none(t))（m），R_ψ(t) = s (ψ_d(t) − ψ_none(t))（°）。另读模型自己的 desire_state 头（8 类 softmax）里该 desire 的概率（只报）。

**主读数**：
- 变道：C_fast、Δ = 0、LCL 与 LCR 两个响应的逐行平均，R_y(4 s) 的均值（log 聚类 bootstrap 95 % CI，B 10 000，seed 0）；同时报中位、左 / 右分开、1 / 2 / 6 / 8 s、R_ψ(4 s)、
  方向正确份额（R_y(4) > 0.1 m）、full（≥ 2.5 m）/ partial（0.5–2.5）/ flat（< 0.5）份额。
- 转弯：T、Δ = 0、desire 与人同侧，R_ψ(4 s) 均值 [CI]；同时报 R_y(4 s)、反侧 desire 的 R_ψ(4 s)（选择性：反侧也朝 desire 转 = 能当选择信号）。

## 6. 登记的线

1. **底模变道通路（P0）**：主读数 ≥ 2.5 m（一条车道）→「底模有完整变道通路」；0.5–2.5 m 且方向正确 ≥ 90 % →「底模有部分（分级）通路」；< 0.5 m →「底模在本协议下没有可用的变道通路」。
   Δ = 0.4 / 1.0 / 2.0 与 6 / 8 s 只报；若 4 s 不到 2.5 m 而 8 s ≥ 3.0 m，加注「通路完整但慢」（不改判）。
2. **底模转弯通路（P0）**：R_ψ(4 s) ≥ 10° →「有」；3–10° →「部分」；< 3° →「没有」。
3. **每阶段保留比**（只在 P0 至少「部分」的那条通路上判）：ratio = 阶段主读数均值 / P0 主读数均值（同一批行，log 聚类 bootstrap 的 CI）。
   ratio < 0.30 且 CI 上界 < 0.50 → **杀掉**；ratio ≥ 0.70 且 CI 下界 > 0.50 → **保留**；其余 → **削弱**。
   **哪一段杀的**：按谱系顺序第一个判「杀掉」、而前驱不是「杀掉」的阶段。两支（P2-F 支、RB0-F 支）各自报。
4. **分级还是平**（每阶段，主读数的行分布）：中位 < 0.5 m = 平；中位 ≥ 2.5 m = 完整；其余 = 分级（部分变道），并报 full / partial / flat 份额。
5. **伴随（只报，不判）**：C_slow 上的两种 desire；LC_L / LC_R 上 none 与同向 LC desire（Δ 0）时 CAP-LEDGER 的变道成功规则（plan 4 s 相对 lane0 中心线的偏移与人同号且 ≥ 人的一半）；
   P_keep 的变道响应；P_lca 朝路线一侧的变道响应；P_turn 朝路线一侧的转弯响应；路线记忆清空的读数；desire_state 概率。

## 7. 闸（读数前）

- G-none：desire 全 0 时（none 臂）走补丁后路径与原 `PModel.forward` 逐位相同（同一批行）。
- G-rep：NAVSIM P0 none 的 2 / 4 s 位姿与 CAP-LEDGER 的 `plans/H/P0.npz` 同行同时刻一致（中位差 < 1e-3 m；该文件存在时）。

## 8. 不做

不训练；不改 serve；不建镜像；不提交。设计（driver 由路线派生 desire、训练保留 desire 响应）只写方案，进 README 与决策条目。

## 9. 成本

一条池链：plans（GPU，约 15 checkpoint × 17 臂 × 约 6 300 行，估 20–30 分钟，单卡 ≤ 16 GB，优先级低于并行训练）→ report（CPU）。挂在 CAP-LEDGER 的 rm_build nav 之后（共用它的行缓存）。
