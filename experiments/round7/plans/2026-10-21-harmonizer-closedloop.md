# Harmonizer 闭环配对：登记（2026-10-21 晚，写于任何开 Harmonizer 的闭环读数之前）

来源：第 435 条（开环：路口转弯 4 s 计划开时更紧 0.3–0.5 m，纵向 2 / 4 s 移 0.18 / 0.45 m）与框架草稿「Harmonizer 风险」。PAI 决赛复测开
Harmonizer，公开榜与我们所有读数都关着。问题：开 Harmonizer 后，提交 serve 下的 281 分数变多少？会不会改变「R6DAG 是否比 NTV0 好」？
结果写 `experiments/round7/results/harmonizer_closedloop.md`。代码 `experiments/round7/scripts/harmcl.{sh,py}`。

## 对象与读数

- 模型：**NTV0-L18-s0**（已提交 `p1021a`）与 **R6DAG-L18-s1**（当前最好，第 434 条）。各一次读，1 seed。R6DAG s0 不读（卡时）。
- 读数：PAI 281 全集，提交 serve：`SERVE="JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 HARMONIZER=1"`，
  `WIZ=controller.gains.rel_front_steering_angle_weight=10`，`ALPASIM_FAST=1`，`CONC=4`，分块表 `CHUNKS_FROM=$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0/chunks`
  （8 块，与已存关读数同表）；`pai_box.sh submit`（VRAM 30 / RAM 40，同第 435 条的开渲染）+ `pai_fix.sh tax` → facts。
  池优先级 21（低于 DAG2 读数的 22，不插队）。
- 关读数（已存，不重读）：`$DATA_DIR/runs/pai_fix/facts/NTV0-L18-s0-281ag.csv`（读于 `runs/defect_fix/pai_accgap/NTV0-L18-s0`）、
  `R6DAG-L18-s1-281ag.csv`（读于 `runs/round6/dag/pai281/R6DAG-L18-s1-ag`）。同分块表、同 serve、同 g10、同 FAST；驱动代码在两次关读数之后
  只加了默认关的开关（`JEV_ACC_VSET` 等、`HARMONIZER`、`DRV`），默认路径不变。
- 冒烟：`pai_box.sh` + `HARMONIZER=1` 已在第 435 条整链跑过池（`harm.sh` ON 变体，驱动换成日志回放），渲染端一致，不另做冒烟。
  **第一块开跑后先查渲染日志**：`native-logs/renderer-0.log` 里 `"enable_harmonizer": true` 与 `DifixModelFactory: checkpoint ready at
  .../harmonizer_nontemporal.pt`；缺任一 → 取消全部开读数，不读任何结果。

## 比较（4 份读数的共同场景：都有分且无 driver RPC 错误）

1. 每个模型 **ON − OFF**：均分差，场景自助法 95 % CI（`jevdrive.stats.paired`，10 000 次，seed 0）；零分数与新增 / 修好零分；
   pai_fix `fclass` 各类（overshoot / route-turn / offroad / AF-front / side / zero-other；部分分 rear-hit / slow / hit）的计数差与 CI。
2. **转弯出路**：facts `primary` 分 内侧（`off_junc_inside` + `off_curve_inside`）/ 外侧（`off_junc_outside`）/ 路口直行 / 直路，ON 与 OFF 计数、
   新增 / 修好；另报 `turn_deg ≥ 35°` 场景（路口转）上的均分 ON − OFF。
3. **起步**：LAUNCH-37（`launch_mask`：交接速度 < 1 m/s 且日志走 > 5 m）上起步数 ON / OFF。
4. **双重差分**：逐场景 (R6DAG − NTV0)_ON − (R6DAG − NTV0)_OFF，场景自助法 CI；同时报 ON 下 R6DAG − NTV0 本身（关下为 +0.072 [+0.034, +0.111]）。

## 噪声参照（读数之前定）

- 关重读逐场景确定：渲染器逐位确定（第 435 条，300 / 300 帧相同）；PAI-FIX「重读逐场景确定」（第 406 条）；第 436 条 8 个 rollout 重读得分与存档全同。
  因此 ON − OFF 里没有重读噪声，全是 Harmonizer 的效应（含闭环里画面变化的混沌传播）。
- 「小扰动」的翻动量参照：第 370 条 holdlead（serve 小改）对 NTV0 +0.0004 [−0.014, +0.013]，零分 +2，类间互相抵消；
  种子差：NTV0 s1 − s0 −0.006 [−0.026, +0.013]，R6DAG s1 − s0 +0.040 [+0.014, +0.068]（新 3 / 修 13 零分）。

## 读法（写在读数之前）

- **ON − OFF 是否「有实质影响」**（每个模型分开）：CI 不含 0 **且** |点估计| ≥ 0.02（281 上约 5.6 个场景分，按第 426 条 0.6 倍折算约 17 PCS）。
  CI 含 0 且 |点| < 0.02 → 「在小扰动水平内」。其余（CI 含 0 但 |点| ≥ 0.02，或 CI 不含 0 但 |点| < 0.02）→「方向待定」，照写不加解读。
- **类**：某类计数差的 CI 不含 0 才说该类变了；与开环预测对照：开时转得更紧 → 预期内侧出路增加、外侧出路与 route-turn 零分减少；
  两个模型内侧出路都 +≥ 3 且外侧不增 → 「与开环一致」。
- **起步**：任一模型 LAUNCH-37 起步数降 ≥ 3 → 标出（纵向开环位移最大的一层之一）。
- **提交读法**：决赛用 ON，所以按 ON 下的 R6DAG − NTV0 定：
  - ON 下差的 CI 下界 > 0 → R6DAG 在开 / 关两种渲染下都更好，Harmonizer 不改变选择；
  - ON 下点 > 0 但 CI 含 0 → 不再有「明确更好」，看 DiD：DiD CI 不含 0（负）→ Harmonizer 缩小了 R6DAG 的优势，记为风险；
  - ON 下点 ≤ 0 → Harmonizer 改变了谁更好，R6DAG 的提交需另议（不在本登记里决定）。
  - DiD 的 CI 含 0 → 「Harmonizer 不改变哪个模型更好」的证据（仅 1 seed、1 读）。
- 绝对水平：若两个模型 ON − OFF 都显著为负且同量级 → 记为「Harmonizer 统一扣分」，进入「开渲染行作训练增强」的排期讨论（第 7 轮），本次不做。

## 成本

开渲染约 99 s / 场景（第 435 条，FAST，CONC 4），2 × 281 ≈ 15.5 job-h 的栈时间（每卡 2 栈，约 8 卡时），CPU 侧 tax 两个小 job。
rollout 产出按 `pai_box.sh` 默认保留（零分场景留 rollout.asl，视频不留），作为新的开 Harmonizer 参照读数，不另 dump。

## 补记 1（10-21 18:20，暂停；读数之前）

主会话按用户决定让 DAgger 第 2 次迭代的 281 读数优先：R6DAG-s1 开读数的排队块 c003–c007 与 harmcl-report 已被主会话取消（c003 已确认渲染日志
载入 Harmonizer 权重，18:13）。在跑的 11 块（NTV0 c000–c007、R6DAG-s1 c000–c002）照跑完。先写**部分报告**：NTV0 ON − OFF 用它全部完成的场景；
R6DAG-s1 ON − OFF 只用 c000–c002；DiD 只在 c000–c002 的共同场景上；明确标「部分」（n、宽 CI）。部分报告需要的 R6DAG c000–c002 facts 由一个
只用 CPU 的 tax job（`R6DAG-L18-s1-281agH-p3`，vram 0.1）补出，原 tax 因依赖被取消而失败。读法规则不变，但部分结果不按「提交读法」下结论，
不写 decisions 行。主会话通知后补交 R6DAG-s1 c003–c007（先删这些块的 `SUBMITTED` 标记，`pai_box.sh submit` 只补未完成块）+ tax + 报告。
