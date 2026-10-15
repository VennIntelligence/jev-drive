# 预登记：SELF-REWIND（策略自己的前沿；不训练，只跑 rollout），2026-10-16

**为什么。** AlphaNIST（arXiv 2610.13354）从部分完成状态的缓存里采训练起点，偏向策略约一半能解的起点（权重 ε + sqrt(p(1 − p))，
p 是滚动成功率）。这一读只回答：RSG-L18-s0 在它**自己**失败的训练转弯上，有没有这样的前沿。做法：把策略自己失败的 rollout 倒回 Δ 秒，
对状态做小扰动（确定性策略要有 N 个不同的续写），再把车交回策略**自己**，读它自己的完成率 p̂(单元, Δ)。

决策 316（`plans/2026-10-16-rewind-train64-prereg.md`）在同一批 nuPlan @18 训练池 64 个严格转弯上做了同样的倒回，交给的是规则专家。
本读数整体复用那个仪器：rf_rewind.sh / rf_rewind.py、30 决策的 session（after280 缓存）、存档 tfail.json 的单元 / k_ref、专家包、
sh30_driver 的交接方式（同一个 session，不重启，不注入状态）。

写于任何 rollout 之前。已看过的只有：d316 的结果文件与存档 18 决策 rollout 上算出的单元表（`rf_rewind.sh self units`，下面的数）。
补记只追加，不改原文。

## 扰动机制（按动作，不瞬移，不改模拟器）

- AlpaSim 没有中途注入状态的接口，模拟器不动。扰动是动作：策略从决策 0 开到 h，与 identity 相同（同一服务）；从 h 起 n 个决策，
  **served plan** = `branch_poses(模型 plan, g, s)`（onp_log_driver.branch_poses，P3 的变换：每个位姿的航向与每段的航向角乘 g、段长不变，
  再按 s 倍弧长重采样；s < 1 = 更慢），之后又是模型的原始 plan。控制器跟踪这个 plan，所以扰动后的状态是控制器与车辆动力学真实走出来的。
- 实现：`sh30_driver.py` 一个开关 `JEV_PERTURB`（`experiments/alpasim/lib/perturb.py`；文件行 `<scene> <k> <g> <s> <n>`），不设 = 与今天逐位相同
  （只多一个 `if PT.ON`）；与 JEV_EXPERT / JEV_EXPERT_FROM、JEV_TRACK、JEV_ROUTE、JEV_STOP 互斥（启动时 assert）。记录多 `perturb` {g, s, k0, n}
  与 `poses_model`，`poses` 是 served plan。perturb.py 的 branch_poses 是 onp_log_driver 的拷贝，selftest 与源码逐位比对（已过）。
- 变体：每个 (单元, Δ) 8 个，g ∈ {0.7, 0.85, 1.15, 1.3} × s ∈ {0.9, 1.1}，n = 2 个决策（1 s，2 Hz）。
- **「仍是策略自己的状态」界**：扰动窗末（决策 h + n 的时刻）相对同 stage identity 的位移：横向（identity 航向系）≤ 0.5 m、航向 ≤ 5°、
  速度比 0.9–1.1（identity 车速 < 1 m/s 时改为 |Δv| ≤ 0.1 m/s）。每个变体报位移；界外的变体单列，不进 p̂。另报「动了」（横向 > 2 cm
  或航向 > 0.2° 或速度比偏 > 1 %）：没动的变体就是 identity 的重放。
- smoke 之后可以调 g、s、n（例如界外太多就收 g，几乎都没动就加 n），写成补记，在 10 单元 stage 之前；调完的变体集合对 10 单元和其余单元相同。

## 单元与 Δ（存档 18 决策 rollout 上算，`self.json`；与 d316 tfail.json 的 k_ref 42 / 42 相同）

- **失败单元（32 个）**：d316 的 A 层去掉 short（short 是慢，不是转弯错误）：event 23（k_ref = k_z）、off 9（k_ref = 第一次 > 2.5 m）。
- **对照单元（10 个）**：同 64 个里 21 个完成转弯中，存档 rollout 越过入口（路线站点 ≥ entry 的第一个决策 k_entry）最晚的 10 个；k_ref = k_entry。
  它们的 k_entry 只有 4–8（6, 6, 4, 4, 5, 8, 8, 7, 7, 6），所以 h = k_ref − 2Δ 在 Δ 3 s 只有 4 个 ≥ 1、Δ 5 / 8 s 一个都没有。为了让对照有数，
  **对照在 Δ 3 s 取 h = max(1, k_entry − 6)**，实际 Δ_eff = 1.5–3.0 s（每个单元记在 `delta_eff`）；Δ 5 / 8 s 对照不可倒回。这一点是对照的弱处，
  事先写明。
- Δ ∈ {3, 5, 8} s，h = k_ref − 2Δ，要求 h ≥ 1（决策 0 是强制日志步）。**不可倒回的 (单元, Δ)**：失败单元 Δ 3 s 2 / 32（2f355af4 k_ref 0、
  2bbd57bb k_ref 5），Δ 5 s 11 / 32，Δ 8 s 29 / 32（只有 1b945907、918656f9 两个 k_ref 17 与 131c5de1 k_ref 18 可倒回）；对照 Δ 5 / 8 s 全部。
  即可倒回：失败 30 / 21 / 3，对照 10 / 0 / 0。

## 每个 (单元, Δ) 跑什么

- `D<Δ>-g<g>s<s>`：8 个扰动变体（上面），每个变体一个 launch，launch 里只有该 Δ 可倒回的场景。
- `xpD<Δ>`：规则专家 XP-L12-E46 从同一个 h 起接管（d316 的 JEV_EXPERT_FROM），Δ 3 / 5 / 8 s 都跑（Δ 3 s 与 d316 的 D3 同一 h，用来核对可复现，
  并列报 d316 的数）。
- `id`：策略单独开 30 个决策（同 stage 的 identity）：扰动 session 的 h 前检查、位移的参照、以及「不扰动时策略自己的续写」。

## 完成（严格）与量

- **严格完成** = rf_follow 的 completed（零分前越过 exit + 10 m 且离路线 ≤ 2.5 m）AND v(t_h + 3 s) ≥ 2 m/s AND 不是蠕变（G-crawl，d297 /
  rf_selfpool：t_h 之后、零分前、路线站点 < entry 的控制器样本上 box5 平滑速度的最小值 < 0.5 v_e = 2.3 m/s，v_e = rf_p4.V_E_OTHER 4.6）。
  t_h = 第一个扰动决策的时刻（identity 用同一个 h 的时刻）。同时报 rf_follow 的普通 completed。
- 每个 (单元, Δ)：**p̂ = 严格完成数 / 分母内变体数**（分母 = 位移在界内 且 h 前检查在 d316 容差内：h 前 served plan ≤ 0.15 m、轨迹 ≤ 5 cm、
  |v(h) − v_id(h)| ≤ 0.05 m/s，对同 stage 的 id）；普通 p̂；identity 在同一 h 的严格完成；专家在同一 h 的严格 / 普通完成。
- 每个 Δ（失败单元）：p̂ 分布（0 / (0, 0.2) / [0.2, 0.8] / (0.8, 1) / 1 的单元数）、平均 p̂ 与日志聚类 bootstrap 95 % CI（10 000 次）、p̂ > 0 的单元数；
  按变体类型（g < 1、g > 1、s < 1、s > 1）的严格完成 / 分母内变体；严格完成里 s < 1 的占比；严格完成里过 d293 形状闸的数（rf_p4.gate_of("other")，
  只算 h 之后的轨迹，同 d316）；**行产量** = 严格完成变体里 k ≥ h、零分前、站点 ∈ [entry − 20, exit + 10] m 的决策数之和（2 Hz，rf_follow 的 junction 段，
  同 d316 的 rows_turn），另报其中过闸的、以及贡献行的单元数。
- 对照：同样的 p̂（Δ 3 s，Δ_eff 1.5–3.0 s）。
- 位移：每个 Δ、每个变体的 |横向|、|航向|、速度比的中位 / 最大、界内数、「动了」数；窗末后 1 s 的横向。

## 登记线（读数之前写下）

- **FRONTIER**：存在某个 Δ，失败单元中 p̂ ∈ [0.2, 0.8] 的 ≥ 8 个，且对照的平均 p̂ ≥ 0.7（同一 Δ 的对照；该 Δ 没有可倒回的对照时用对照在全部 Δ 上的合并平均，
  实际上就是 Δ 3 s 那一行）。
- **NONE**：每个失败单元在每个 Δ 上 p̂ < 0.1。
- **SLOW-ONLY**：FRONTIER 成立，但在成立的 Δ 上失败单元的严格完成有 ≥ 80 % 来自 s < 1 的变体（完成的是减速，不是策略）。
- 其余：**UNDECIDED**，按 Δ 报。
- 注：Δ 8 s 只有 3 个可倒回的失败单元，按构造到不了 8 个，Δ 8 s 只作描述。p̂ 的分辨率是 1 / 8（界外变体会让分母更小）。

## smoke 闸（2 个单元：失败 1b945907（event，k_ref 17，h = 11 / 7 / 1）、对照 14315cc0（k_entry 8，h = 2））

- JEV_PERTURB 不设的 `id`（与 SMOKE 的 `idB`）对 d316 存档的 identity（rewind_train64 pilot10 / rest54 的 id）：在最大 h 之前 plan ≤ 0.15 m、
  轨迹 ≤ 5 cm、每个 h 上 |Δv| ≤ 0.05 m/s。
- JEV_PERTURB 设了的 session：h 前对同 stage id 在上述容差内；扰动正好打在 h .. h + n − 1；窗末位移按变体报，界内比例决定是否调变体（补记）。
- 不过（开关不设时离开 d316 容差，或 h 前偏离）：停在 smoke，报告，不改模拟器。

## 分阶段与成本

- 每个 stage：每个 launch 一个池 job（`rws-<tag>`，--vram 20 --cpu 6 --ram 40，history key alpasim-read4 = nuPlan held read 的同一服务），
  读数（CPU job：n-self-read → n-self-table → n-self-check）`--after` 全部 launch，优先级 0（盒空闲）。
- smoke（上面 2 个单元：id、idB、xpD3 / D5 / D8、24 个变体 launch）→ 检查 + 补记 → **pilot10**（失败单元按 scene 排序取每第 4 个 = 8 个，
  对照每第 5 个 = 2 个）与 **rest32**（其余 32 个）一次提交，rest32 的全部 job `--after` pilot10 的读数 job（其 n-self-check 是闸：
  h 前容差内 ≥ 80 %、界内变体 ≥ 50 %、无缺失 rollout，不过则 rest32 不跑）；最后合并读数 CPU job `--after` rest32 的读数。
- 规模：变体 8 ×（30 + 10 + 21 + 3 = 64）= 512 个 rollout，专家 64，identity 42，smoke 约 60，合计约 680 个 rollout、约 85 个 launch。
  按 d316 实测每 rollout 约 8 s job 时间、每个 launch 约 45–60 s 启动：约 1.5 + 1.4 ≈ 3 job-h，每卡 4 个 job ≈ 0.75 card-h，4 卡墙钟约 40 min。
  远低于 5 card-h，不做 profiling。

## 交付

- 代码：`experiments/alpasim/lib/perturb.py`、`sh30_driver.py`（JEV_PERTURB）、`experiments/route_follow/scripts/rf_rewind.{sh,py}`（`self` 模式、
  `n-self-*`；不另写 runner）。
- 结果：`experiments/route_follow/results/rewind_self/`（rewind_self.md、逐单元 CSV、逐变体 CSV、read.json），图 `figs/rewind_self_bev.png`
  （2–3 个单元：identity、变体、专家）。盒：`$DATA_DIR/runs/route_follow/rewind_self/`。
- 决策文件 research/decisions/317.md + decisions.md 一行 + research/next-round/turn-night.md 末尾一段 + route_follow README 一行。

## 补记 1（smoke 之后、pilot10 之前，2026-10-15 23:40 盒时间）

- smoke（`$DATA_DIR/runs/route_follow/rewind_self/smoke`，be43648a；1b945907 h = 11 / 7 / 1，14315cc0 h = 2）：**机制过闸**。开关不设的 id 与 d316 存档 identity
  逐位相同（2 / 2：plan 0、轨迹 0、每个 h 上 Δv 0，结局相同），idB 与 id 逐位相同；32 个扰动 session 在 h 前与 id 逐位相同（32 / 32），扰动正好打在
  决策 h、h + 1 上。
- **位移**：横向与航向都远在界内（Δ 3 / 5 s 横向 ≤ 1 cm、航向 ≤ 0.84°；Δ 8 s（h = 1）横向 ≤ 0.14 m、航向 ≤ 4.3°，g 1.3 s 1.1 最大）。
  **速度比全部出界**：窗末速度比 ≈ 1 + 2 (s − 1)（s 0.9 → 0.80–0.84，s 1.1 → 1.17–1.22），32 / 32 界外。所以按登记的规则调 s，g 与 n 不变：
  **s ∈ {0.96, 1.04}**（预期窗末速度比约 0.92 / 1.08）；变体 = g {0.7, 0.85, 1.15, 1.3} × s {0.96, 1.04}，n = 2。pilot10 的 n-self-check 仍是闸
  （界内变体 ≥ 50 %，否则 rest32 不跑）。
- 顺带看到的（2 个单元，不作判词，写出来是因为它们会影响读法）：窗末后 1 s 速度比继续放大（s 0.9 → 0.51–0.78，s 1.1 → 1.28–1.63），即策略把
  速度扰动当成自己的趋势跟着走（d315 的纵向趋势跟随）；这是策略自己的续写，不排除。失败单元 1b945907 的 24 个变体全部失败（corridor，和 identity
  一样），专家 Δ 3 / 5 / 8 s 都完成；对照 14315cc0 的 identity 完成，但 8 个变体全部失败（6 个 at-fault 碰撞、全部被追尾，速度变了 ±20 % 以后与
  回放车的时序对不上），专家从同一 h 也失败（碰撞）。即 ±20 % 的速度扰动对这个对照不算「小」；s 收到 ±4 % 正是为此。
