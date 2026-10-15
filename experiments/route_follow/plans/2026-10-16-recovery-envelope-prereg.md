# 预登记：RECOVERY-ENVELOPE（策略自己状态上的恢复示范 R + 纵向包络 L + 两者合一 RL），2026-10-16

背景：[决定 297](../../../research/decisions/297.md)（自分支示范当模仿：推翻）、[303 补记 5](../../../research/decisions/303.md)（接近课的害处在形状项 L1 / L2，同输入下 4 s 少转 1–1.5°；
速度项 L3 只沿各点自身航向动、几乎没训动）、[315](../../../research/decisions/315.md)（纵向是自身运动的趋势跟随者）、[316](../../../research/decisions/316.md)（专家从策略自己的失败状态 Δ 2 s 救回 33 / 41）、
[317](../../../research/decisions/317.md)（自练前沿很浅：UNDECIDED）、[318](../../../research/decisions/318.md)（路线主导 BLIND：plan 不读后果）、
[lon_inventory](../results/lon_inventory/inventory.md)（nuPlan 自身状态上合格停止线 ≥ 3 m/s 451 个决策 / 85 个场景，≥ 6 m/s 77，≥ 12 为 0；空路比人慢 2 006；弯前比专家快约 80）。
方向由用户批准（10-16 凌晨）；本文件在任何 pilot 之前写好、提交、推送；之后只追加补记。

原则：标签全在策略**自己的**状态上，在该状态自己的时刻、按当时回放的交通计算（交通是非反应式日志回放：策略与日志一旦不同步，人类在同一站位的速度就不是有效目标）。
没有进度提示、没有路线 hinge、没有朝目标位姿的拉。

## 臂（全部在 RSG-L18 配方上，同步数；对照 = 存档 RSG-L18，d297 证明重训逐位相同）

配方 REC（= rf_onk.sh 的 REC）：`--init RB0-F-s0 --steps 2500 --warmup 1 --lr-scale 0.1 --eval-every 500 --yr-mass 0.1 --label-bank --agent-lam 10
--agent-on-only --shape --rm-lr 3e-4 --on-split alpasim_nuplan/onpol-train`，池 on1_L18RB0（S 行，`self`）每 batch 13 行，`--ho-w 3`。

| Tag | 改动 |
|:--|:--|
| `RCVR0-L18-s{0,1}` | R：d316 的恢复行（下文），`--on-mode mixed`（S 行 self，R 行 both） |
| `RCVL-L18-s{0,1}` | L：纵向包络项 `--env-lam W`（on_envelope.py），包络行 + 推快的接近行，`--on-mode self` |
| `RCVRL0-L18-s{0,1}` | R0 + L：`--on-mode mixed --on-lon-all`（包络项也作用在 self 行上） |
| `RCVR-L18-s{0,1}`、`RCVRL-L18-s{0,1}` | 同 R0 / RL0，行换成今晚全场景倒回（rewind_onpol_all，决定 319）的恢复行；材料到了再跑 |

pilot tag：`RCV<臂>P[W]-L18-s0`（1 000 步），正式 tag 留给 2 500 步。

### R（老师恢复，DAgger 式）

- 单元：d316（以及到了以后的全场景倒回）里专家接手的 session，Δ 2 s 与 3 s（d316 的候选），**专家续写完成且交接 h 之后没有任何零分事件**（rf_rewind n-read 的 completed / k_z，
  在产行的那次 session 自己身上判）。
- 行：该 session 里决策 k ≥ h 的状态（2 Hz），直到转弯出口 + 10 m 或 session 结束；第一个零分事件之后的行不要（按上一条本来就没有，照写）；on_rows 的排除规则照旧
  （overlap、corridor、log_future、frame …）。k = h 的行是**策略的精确状态**；k > h 是专家驾驶到达、策略被喂入的状态。报两类行数。
- 目标：该决策上专家的 served plan（8 个位姿 x, y, yaw，专家自己的时序，同一 session，因此与当时回放的交通一致），在行自己的 anchor 系。
- 帧：d316 的 session 是 `SH30_DUMP=0` 跑的，没有帧。做法：同样的 D2 / D3 交接（同 cache、同 pack、同 JEV_EXPERT_FROM 表）以 `SH30_DUMP` 重跑，行从重跑里取（交接前与 d316 的
  逐位 / 容差检查照 d316 的规则报）。重跑做不出来时退回只用 k = h（on1_L18RSG 里同一 scene、同一 k 的行，交接前状态与存档 18 决策 rollout 在容差内），如实写明。
- 训练：已有的 `both` / `mixed` 路径（d297 的 SELF 行同一路径）：R 行 = 模仿项 + 两个 hinge。抽取：R 行占池抽取 20 %（`--on-weight`），每场景上限 0.10（不可行则 0.20，写明）。
- dev：R 行按 log 留出（crc32(log) % 100 < 15，APP 的规则），不进训练。

### L（纵向包络，只有速度）

- 项：on_approach.py 的 L3 推广成**非对称包络**，新模块 `experiments/onpol_nuplan/lib/on_envelope.py`，只由 `rm_train.py --env-lam W` 打开（`--on-lon on_envelope --on-lon-lam 10 W`，
  与 `--app-lam` 同一尺度）；不开时不加任何运算（逐位闸 1）。on_approach 的形状项 L1 / L2 在本轮**一律不用**（不加载 on_approach）。
  位置只沿每个位姿自身（detached）航向移动（on_approach.lon_only / on_lon.timing_only，同一个算子），航向不得梯度。
- 每步（0.1 s 稠密 41 步，on_lon 的几何：后轴 / 前保险杠在行路线上的弧长、路径速度）：`relu(v − v_u(s)) + relu(v_l(s) − v)`，对有界的步取平均；行上没有任何界 → 0。
- 上界（只在「此刻有理由」处）：
  1. **停止线**：前方 0–60 m 第一条 LONG-PROBE 合格线（ln_elig）且（灯：此刻红，且人在该线停过或 4 s 窗内一直红；停车标志：人在该线停过）：
     停止点 = 线 − 1.0 m（on_lon 的 M_B），前保险杠弧长 s_f 上 `v_u = sqrt(2 · 3.0 · (停止点 − s_f))`（专家 A_DEC），停止点以后 v_u = 0。停止线不加 +1 m/s 的余量（余量会允许越线）。
  2. **弯入口**：前方 0–40 m 有严格弯（rf_follow.find_turn，≥ 60°，rf_loninv 的 turn 类）：后轴弧长 s_r 落在 [入口 − 26 m, 出口] 上 `v_u = 专家入口剖面 + 1.0 m/s`
     （rule_expert.Scene.profile(V_E 4.6) 在行路线上，即 rf_loninv 的 proxy v_ref；与真包中位差 0.11 m/s）。
  3. **前车**：专家的让行规则（rule_expert.plan 的 yield）在行自己的状态上：沿路线以专家剖面从当前速度出发 3 s，前方（行路线 ±1.5 m 内、A-eligible，on_lon 的前 4 个）
     任一车在 3 s 内任一时刻的最小弧长 σ 被专家前保险杠触到 → 停止点 = σ − 1.5 m，`v_u = sqrt(2 · 3.0 · (停止点 − s_f)) + 1.0`。
  - 可行地板：每个上界与 `sqrt(max(v0² − 2 · 6.0 · (s − s0), 0))`（专家 A_MAX 下最快的减速）取大：物理上刹不住的部分不罚。
- 下界（只在空路上）：rf_lonol 的 free 格子（40 m 内无 A-eligible 前车、60 m 内无任何停止线、60 m 内路线转角 < 30°、车速 ≥ 2 m/s）：
  `v_l(s) = 0.85 · min(v_ref(s), sqrt(v0² + 2 · 1.5 · (s − s0)))`，v_ref = 专家空路速度（proxy profile：max(vlog, 3) 加曲率上限），1.5 = 专家 A_ACC（从当前速度可达）。
  这里用了人在该站位的速度（vlog）作为路的「可开速度」：只在空路（无前车、无灯、无弯）上，交通不同步不影响它；有任何理由的地方不用下界。
- 行（on1_L18RSG 的这些类行，按 log 留出 dev 同上）：stop / turn / lead / free；推快的接近行（下节）同样打标签。行可以同时有上界与下界之外的多个理由，取各上界的最小。
- 抽取：S 行仍每 batch 13 行，包络行另加 7 行（on-k 20，APP 的做法；APP0C 两个 seed 证明这种池混合本身中性），其中停止线行占池抽取约 5 %（≈ 1 行 / batch），
  推快行约 1 行，其余按 turn / lead / free 均分。
- W：pilot 并行跑 W 1 与 W 3，取过全部闸的最小 W；都不过 → L 不跑全量。

### 推快的接近（AlphaNIST 的「shaped start」）

- nuPlan 自身状态里没有快的停止线状态（≥ 12 m/s 为 0），用 d317 的 JEV_PERTURB（perturb.py，sh30_driver.py；1 s 窗后速度比 ≈ 1 + 2(s − 1)）在策略距合格停止线或严格弯入口
  40–60 m 的决策上推快：s 使速度 +25–40 %（需要时加 n），2–3 档；之后策略自己开。session 以 SH30_DUMP 跑，行进 on_rows 管线，按上面的包络打标签，作 self 行。
- 场景：lon_inventory 的 85 个 ≥ 3 m/s 合格线场景 + 64 个转弯场景（d316 的 train64）。GPU：池 job。报行数、推后达到的速度对推前。

## 闸（今晚跑）

1. **逐位**：所有新开关关时 = RSG-L18-s0 配方逐位相同（50 步，标量与最终权重 max |diff| 0，旧代码 = 本次改动前的 commit 的 worktree）；另：`--on-lon-all` 在 self 模式下与不加逐位相同；
   `--env-lam` 的 pack 若全行 ok = False，与不加逐位相同（项为 0 但图里多了运算：只要求 loss 与权重相同）。
2. **pilot**（1 000 步，seed 0，每臂；R 先用 d316 的行）：
   - dev ADE ≤ 0.538；G-rely 为 true（rm_train 的 rely.json：junction 行 ADE(mask) − ADE(on) ≥ 0.01、CI 下界 > 0）；
   - **同输入转弯闸（新）**：rf_appzeros.py 的决策 0 / 1 仪器（held 460 RSG-L18-s0 rollout 的决策 0 / 1 是日志回放状态，任何模型输入相同；pool on1_L18RSGallheld 的 k ≤ 1 行），
     开环前向，位于 ≥ 45° 弯 [入口 − 40, 入口 + 10) m 内的行，4 s 航向朝弯内（h4）：pilot − RSG 的均值 ≥ −0.3°（APP1X 在闭环里是 −1.0 到 −1.5°）。
     仪器校验：同一开环读数对 APP1X-L18-s0 应 ≤ −0.5°；读不出 APP1X 的欠转则该闸记为「仪器不灵」并报告，不作通过依据。
   - L：dev 行（留出 log 的 on1_L18RSG 包络行）上，(a) 合格停止线前的超速（上界违反的均值）比 RSG-L18-s0 降 ≥ 20 %；(b) A-speed（弯入口包络上的超速）下降，log 聚类 CI 下界 > 0；
     (c) 空路 dev 行 plan / 喂入速度比（4 s plan 平均速度 / 喂入速度，中位）上升，CI 下界 > 0。
   - R：留出 log 的 R 行上模仿误差（对专家目标的 8 点 ADE；训练器的模仿损失是同一目标上的 dist，ADE 是它可读的版本）比 RSG-L18-s0 降 ≥ 20 %。
   - RL：R 与 L 的闸都要过。
3. 报 pilot 结果与花费。

## 全量读数与线（主会话的登记，原文照录）

Registered lines for the full round (main session's; copy into the prereg): vs RSG-L18 2-seed means on held 452 paired, log-clustered bootstrap: H1 harmless: zeros ≤ +3, at-fault ≤ +3, GT ≤ +0.03 m (points; CIs reported); H2 turning: held 24 turns completion seed mean ≥ +3 for R and RL; PAI 281 (seed 0, vs RSG-L18-s0-gate): strict 36 completion ≥ 10 (reference 4), all-281 zeros ≤ reference + 3; L / RL: PAI cor_route_end ≤ reference − 3 and rear-ended ≤ reference + 3; public guard: score CI low > −0.01, zeros ≤ +5. Adopt an arm only if H1 + guard pass and its own H2 line passes.

链（rf_recov.sh，一次提交，全部 `--after`）：行 → pilot → 训练（臂 × seed）→ held 460 @18（rm_chain.sh read，6 块）→ 公开 1491 @10 护栏（ol_chain readlist + ol_confirm，SLOW 的做法）
→ PAI 281（盒上，gate serve，gains 20，同 v2pai / jlon）seed 0 → 报告。

## 补记规则

用户 01:20 JST 指示（主会话转达）：pilot 过闸的臂由本会话直接起全量；L 不等倒回材料；R0 / RL0 用 d316 的行，全场景倒回的行到了再起 R / RL。

## 补记 1（实现会话，10-16 00:25 CST，任何 pilot 之前）：前车上界改成按时刻

登记的前车上界（rule_expert 的让行：3 s 窗内前车最小弧长，任一时刻触到就停在它前 1.5 m）在 on1_L18RSG 自己的前车行上 90 % 违反、平均超 2.95 m/s：
它等于「每辆在动的前车后面都要准备停下」，会把已经跟得过远（d315：时距 1.84 × 人）的策略教得更慢。改为同一停距与 A_DEC 制动包络、但对**同一时刻**前车的位置：
第 t 步前保险杠速度 ≤ sqrt(2 · 3.0 · (σ(t) − 1.5 − s_f(t))) + 1.0（σ(t) = 合格前车在 t 时刻的最近角点弧长，on_lon pack 的 a_sig），可行地板同上；前车类 = t0 时 40 m 内有合格前车。
登记形式保留为 `lead_lit` 只作记录。改后自身行的违反：停止线 10.5 %（平均超 0.10 m/s）、弯 14.5 %（0.09）、前车 6.3 %（0.08）、空路下界 54 %（缺 0.59）。

## 补记 2（实现会话，10-16 00:32 CST，W 0.3 / 0.1 的 pilot 之前）：L 的剂量

W 1 / W 3 pilot 都不过守卫闸：dev ADE 0.5713 / 0.5976（线 0.538）、G-rely 假（junction 差 0.0027 / −0.0073）；W 3 同输入转弯 −0.50°（不过），W 1 −0.20°（过）。
机制闸在 W 1 就大幅过：空路缺速 0.61 → 0.09 m/s、plan / 喂入 0.96 → 1.06；弯入口超速 −63 %；停止线超速 −20 %。读法：剂量过大。按登记字面 L 不跑全量；本补记在看到任何新结果之前登记一个剂量延伸：
W 0.3 与 W 0.1 两个 pilot（同闸），外加诊断 pilot W 1 只有上界（空路行抽取权重 0，`RCVLP1u`，不是候选臂，只用来定位害处是否来自下界）。取过全部闸的最小 W 起全量；仍都不过则 L 停。
同输入转弯闸的噪声参照（开环，同仪器）：APP0C-L18-s0（W 0 对照）−0.31° [−0.49, −0.15]，RSG-L18-s1 −0.12° [−0.23, −0.03]；APP1X −1.70° [−2.22, −1.09]（仪器复现闭环的 −1.5 / −3.1°）。
−0.3° 的线落在 checkpoint 间噪声内，判读时并报。

## 补记 4（实现会话，10-16 02:05 CST，R / RL 的 k 6 pilot 之前）：全场景倒回的行

主会话（02:45 JST）：全场景倒回（d319）label == True、去掉 within_tol == False，不加形状闸。重跑带帧后 on1_RCVall = 952 行 / 84 单元（k = h 82、k > h 870；dev 143 行 / 13 单元 / 4 log），
帧与目标对源逐位一致。pilot（1 000 步，k 3）：R：ADE 0.5284、G-rely 过、等弧长转弯 +0.17°，留出 log 模仿 1.377 → 1.259（−8.6 %，不过）；
RL（上界 W 1、下界 W 0.1）：ADE 0.5366、G-rely 过、停止线 −28 %、A-speed −75 %、空路比 +0.02、等弧长 +0.75°，模仿 −11.9 %（不过）。
按补记 3 的 R 剂量规则对 R 与 RL 各跑 k 6（RL 写作 `k6_1l0.1`），闸不变；过闸即起全量，不过则停。
