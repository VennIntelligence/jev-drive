# op_parity turn selector bench 预登记：第 191 条的 selector（N7）接上 SH30，走 `jevdrive.bench` 出官方 navtest / navhard 数（2026-10-08，任何本线新数读之前写定）

## 问题

第 191 条：navtrain held-out 标签上训练的非特权 selector（臂 N7：ego + 指令 + plan、未池化冻结视觉 token、SH30 隐状态、自路沿 margin；5 个初始化平均）在 navtest 记录航向变化 |dyaw| ≥ 20° 的 3 154 个 token 上
增益 +2.75 [+1.55, +3.82]（F19 × pc，无 EC）。这还不是榜单数，四个缺口：(1) 桶由 logged 4 s 航向变化定义，是特权的；(2) 没在直行 / 小弯（navtest 的 74%）上跑过，训练只有转弯；
(3) 增益在 `pc` 口径、无扩展舒适度（EC）下估值；逐帧切换候选可能丢 EC / HC；(4) 走的是分析脚本，不是 bench 模型。
本线只问：**把 SH30 + 这个 selector 当作 bench 模型，官方 navtest EPDMS（全 12 146 token，含 EC，两个 SH30 seed，同第 170 条 89.55 的口径）和 navhard（两阶段）上，相对 SH30 和 WA-JEPA（navtest 91.71）的增益是多少，转弯增益有多少活到了官方分数。**

## 已有的、不重测

第 170、178、186、187、190、191 条的全部数。权重不变：N7 按 191 登记的协议（F19 × pc、配置取 191 在 navtrain OOF 上选出的那一个、5 个初始化种子 200..204、同一份 28 323 行训练集）；**191 没存权重**，所以按相同协议重训一次（唯一变化：把 `fit_nn` 的网络构造抽成函数并存下 state_dict，RNG 消耗顺序不变），
并用 **G-repro** 确认 +2.75 复现。不在 navtest 上重新选配置、臂、门限或 epoch，不重训到别的臂。

## 模型与门

bench 模型名 `SH30-F-s{0,1}@warp:tsA` / `:tsB` / `:ts0`（navhard 按第 170 条用 `@gimm`）。流水线：SH30 plan → 导出 8 个位姿（现有 `export`）→ **select 阶段**（同一份 SH30 权重前向取隐状态 / 路沿，对导出位姿套 191 的 19 个候选 `turn_ceiling.transform`，算自路沿 margin C，
N7 头 5 个初始化平均，picks_of（恒等 m = 0））→ 被选候选的位姿写成该模型的 pred 文件 → 既有 `score` / `harness` / `collect`。门只决定哪些 token 允许非恒等：

- **A**：每个 token 都经 selector。
- **B（主变体，现在指定）**：只在 SH30 **自己的**导出位姿 4 s 末航向的绝对值 ≥ **20°** 的 token 上经 selector，其余恒等。这是 191 桶的模型侧类比（同一个 20° 门限、同一个量的模型版），门限不调。选 B 而不是 A 的理由：selector 只在转弯 token 上训练，A 让它在没见过的分布上工作；B 把已验证的支撑集和部署行为对上。
- **ts0**：强制恒等，只用于恒等闸门。
- (C) 另一种门（navtrain 上调门限等）**不做**；B 的门限不在任何官方数出来后改。

门的覆盖率（对 logged 桶的 precision / recall）、门内 / 门外增益分开报。

## 读数

主读数 R1：**B，navtest 全 12 146 token，官方 EPDMS × 100（含 EC），两个 seed 的均值，对 SH30（同两个 seed）的 Δ，按 log 配对 cluster bootstrap 的 95% CI**（`jevdrive.bench report`，B = 10 000）。
其余：R2 分桶（直行 / 20–45° / > 45°、左 / 右，bench strata）的 Δ，门内 / 门外 Δ；R3 分解：同一批 logged 转弯 token 上 `pc` 无 EC（候选分数表，+2.75）→ raw 无 EC（同样的 pick，候选子分直接取）→ bench 无 EC（bench 子分重算）→ 含 EC 的官方 Δ；
R4 子分 DAC / NC / TTC / EP / LK / EC / HC / DDC / TLC 的 B 对 SH30；R5 navhard（B）combined / stage 1 / stage 2 对 SH30，以及对 WA-JEPA 的 navtest、navhard 差；R6 A 的 navtest 全量（navhard 的 A 看预算）。

## 线（固定，看到官方数后不改）

- **闸门（不过就停下报告，不打分）**：G-id：`ts0` 在一组整 log 的冒烟集上（约几百 token，`navsim/op-parity-tsbench-smoke`，整 log 以保 EC 配对）的每 token 分数与归档的 SH30-F-s0 / s1 bench 分数逐 token 完全相等（最大绝对差 ≤ 1e-9）；
  G-repro：重训的 N7 在 3 154 个 logged 转弯 token 上读 191 同一套特征与候选分数表，F19 × pc 增益落在 [+2.45, +3.05]（191 的 +2.75 ± 0.30），95% CI 下界 > 0；G-feat（只诊断，不停）：本线从模型前向重算的自路沿 margin 与 191 的 margins.npz 在这 3 154 token 上的最大绝对差。
- **入账线**（主）：B 的 R1 官方 navtest EPDMS 对 SH30 的 Δ，95% CI **下界 > 0**，且 **logged 直行 token 的 Δ 点估计 ≥ −0.10**。两条都过 = 这一条算榜单数（entry real）。
- 事先写好的结论：
  1. 过线：作为 SH30 + selector 的官方数报告；全量方法线继续（191 的「值得全量方法」落到官方分数上）。
  2. 转弯桶（logged ≥ 20°，官方含 EC）Δ 的 95% CI 下界 > 0，但全量 Δ 的下界 ≤ 0（桶占 26%，pc +2.75 折到全量约 +0.7，12 k token 的 CI 可能放不下）：增益真实但不满足入账线，**不算入账**；写成桶内结果，说明要入账需要多大的桶内增益或更多 token。
  3. 转弯桶 raw 无 EC 的 Δ > 0，但含 EC 的 Δ 比 raw 无 EC 少一半以上（EC / HC 吃掉）：结论是逐帧独立 pick 不是可部署形态，下一步是带时间一致性的 selector（上一帧 pick 作输入或迟滞），在 navtrain 序列上设计，不在 navtest 上调；
  4. 直行 Δ < −0.10（B 门漏过非转弯 token 或 selector 影响了门内的直行类 token）：不入账，门要重新设计；A 的直行 Δ 说明 selector 在没见过的分布上的风险。
  5. 桶内官方 Δ 的 95% CI 下界 ≤ 0 且点估计 ≤ 0：191 的 +2.75 是 pc 口径 / 非因果 credit 的产物，191 降级，线按此结束。
  这些情形不互斥时（例如 2 与 3 同时发生）都写出来。
- 期望（不是线）：桶内官方 Δ 约 +2 ~ +2.7，全量 Δ 约 +0.5 ~ +0.7，EC 损失 < 0.3。

## 成本与停止

预算：约 3 h 墙钟、2 card-h、80 core-h。导出 / select 是 GPU 小作业（每个 seed 一次前向 + 候选 margin），navtest 打分 6 分片 × 约 13 核 × 约 5 min ≈ 6 core-h 一次，4 次 navtest（A / B × 2 seed）+ navhard（B 2 次，A 看预算）。
冒烟实测外推全量，超过预算 1.5 倍就停下报告。冒烟只用来跑通和 G-id，不读任何非恒等分数作为结论。

## 不做

HUGSIM / 闭环、WOD、重训 selector、在非转弯 token 上训练、改 N7 的臂 / 配置 / 门限、任何在官方数之后的门或变体选择。不改既有 labels、检查点、缓存和 bench 归档，新结果写新 run 目录。
