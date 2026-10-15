# 预登记：RECOVERY-REWIND on nuPlan train 64（不训练，只跑 rollout），2026-10-16

**为什么。** 决策 314 在 PAI 36 个严格转弯（考场）上读到：把 RSG-L18-s0 自己的转弯失败倒回 Δ = 1 / 2 / 3 s 交给规则专家，
专家完成 85 / 81 / 85 %。那是考场。用户要的是**训练转弯**上的同一读数，作为「恢复课」的行来源：nuPlan @18 训练池的 64 个严格转弯
（`experiments/route_follow/results/entry_speed_rows/README.md` 的 Data 节；盒上 `$DATA_DIR/runs/route_follow/entry_speed/follow/scenes.csv`
标签 RSG-L18-s0@18，turn = True 的 64 行：completed 21、event 23、off 9、short 11）。本读数只回答：在训练转弯上，专家能不能从策略自己的
状态接手并完成转弯、恢复轨迹是否过 d293 形状闸、能产出多少行。不训练。

写于任何非 smoke rollout 之前。只看过存档的 follow 表（64 个转弯的结局、k_z、k_cross）。补记只追加，不改原文。

## 服务路径与交接机制

- nuPlan @18 的服务就是 held read 的服务：`rm_chain.sh readlist RSG-L18-s0`（= ol_fast.sh 的 sim：sh30 driver、c0b_chain 的 OVERRIDES、
  after160 缓存，18 个决策、2 Hz，`force_gt_duration` 0.5 s = 决策 0 强制跟日志）。训练池的 rollout（`onpol_long/roll/RSG-L18-s0`）也走这条路径。
- d314 的 `rew` 单元在 `onp_log_driver.py`（PAI 的 pai_driver 子类），nuPlan 用的是 `sh30_driver.py`，不能直接用。sh30_driver 已有 JEV_EXPERT
  （rf_expert.sh box-read 用它在 held 上跑专家），缺的只是「从第 k 个决策起」。所以只在 driver 侧加一个开关，不改模拟器：
  `rule_expert.py` 的 `JEV_EXPERT_FROM`（一个整数或 `<scene> <k>` 文件），sh30_driver 在决策 k 之前照常服务模型的 plan，从 k 起服务
  `rule_expert.plan(位姿, 实测速度)`（同 JEV_EXPERT 的调用）。同一个 session，不重启：交接时位姿、速度、控制器状态、模型的历史和路线记忆
  都是策略自己的。模型每步照常运行并记录（`poses_model`）。
- 正确性检查（smoke 闸）：每个交接 session 在 h 之前的 served plan 与控制器轨迹必须与 identity 逐位相同，v(h) 与 identity 相同；
  smoke 加跑第二个 identity（idB）核对盒上可复现。不通过就停在 smoke，报告，不改模拟器。
- 专家 XP-L12-E46（rule_expert 默认：lookahead 1.2 s、入弯速度 4.6 m/s）。专家包：`rf_expert.py pack` 在这 64 个场景上，配置取存档训练池
  rollout 的 generated config（与 held 的 pack_held 同一做法）。
- 策略对照 = 新跑的 identity rollout（盒上同一 session 逐位可复现，交还策略就是 identity 本身）。

## 单元与时刻

- 单元集合与结局取**新跑的** identity（同一代码、同一盒），不取存档：rf_follow 的定义原样（turn、completed、event / off / short、k_z）。
  存档结局与新跑结局的一致数单列；不一致的场景按新跑归层。
- 决策时间轴：2 Hz，k = 0 .. 17；Δ 秒 = 2Δ 个决策。h_Δ = k_ref − 2Δ，要求 1 ≤ h ≤ 17（决策 0 是强制日志步）；不满足的 (单元, Δ) 列为「不可倒回」，
  不进分母（另报以全部单元为分母的保守数）。
- **A（失败的转弯，k_ref = 失败决策）**：
  - event：k_ref = k_z（rf_follow：评估器 2 Hz 序列上第一个零分步，at-fault 前 / 侧碰、offroad 或 dist_to_gt ≥ 4 m）；
  - off：k_ref = 第一个满足「路线站点 ≥ entry − 20 m、|横向| > 2.5 m、k < k_z」的决策（2 Hz 锚点，rf_follow 的站点 / 横向）；
    没有这样的决策时（只在越过 exit + 10 m 处 > 2.5 m）k_ref = k_cross；
  - short：k_ref = 17（读数的最后一个决策：此时确定没完成）。short 是慢，没有转弯错误的时刻；按这个定义 h = 15 / 13 / 11，专家只剩 1–3 s，
    事先预计 short 子层几乎不能完成。所以 A 的表同时按 event / off / short 拆开报。
- **B（完成的转弯，k_ref = 第一次 d293 形状违规的决策）**：在 identity 的 10 Hz 控制器轨迹上（站点 ∈ [entry − 40, exit]、零分前），
  按 rf_shape 的 0.25 m 弧长网格逐点检查：|κ| > kmax、|dκ/ds| > dk、v²|κ| > alat；以及入口速度（轨迹第一次越过 entry 处的 box5 平滑速度）> v_entry，
  时刻记为越过入口那一刻。最早一个超限点的时刻 t_v，k_ref = t0 ≤ t_v 的最后一个决策。门限：nuPlan 没有 topo 类，取 d293 的全部转弯人类 p95
  （`rf_p4.gate_of("other")`：kmax 0.184 1/m、dk 0.0425 1/m²、alat 4.42 m/s²，入口速度 V_E_OTHER 4.6 + 1 = 5.6 m/s）。
  没有任何违规的完成转弯 = 「完成且像人」对照层：不交接，只计数。
- **C**：专家从决策 0 开（xp0），全部 64 个。

## 量（每层、每个 Δ）

- 专家完成率（rf_follow 规则：零分前越过 exit + 10 m 且离路线 ≤ 2.5 m），日志聚类 bootstrap 95 % CI（同一 log 的场景一起抽，10 000 次）。
- 交接状态：v(h)、|横向|(h)、h 时站点相对入口；h 后 0–1 s 的速度轨迹（专家对 identity）。
- 回走廊：h 后、零分前第一次 |横向| ≤ 1 m 并保持 1 s 的时刻 − t_h（settle）。
- 蠕变：v(h + 3 s) ≥ 2 m/s 的比例（h + 3 s 超出 9 s 片段时不计，报分母）。
- 追尾：h 后有 collision_rear；at-fault：零分且是前 / 侧碰撞。
- 形状闸：同上门限，只在 h 之后的轨迹段（t ≥ t_h、站点 ∈ [entry − 40, exit]、零分前）上算；入口在 h 之前已过时入口速度项记为过；只对 completed 计「过闸」。
- B 另报：h 之后同一时间段上，专家与 identity 的 |横向| 中位 / 最大、峰值曲率、峰值横向加速度（交接前两者按构造相同）。
- 行产量：交接 session 中 k ≥ h、零分前、站点 ∈ [entry − 20, exit + 10] m（rf_follow 的 junction 段）的决策数，按单元求和，
  并单列「完成且过闸」单元的行数（恢复课实际可用的 N）。
- 校准（不改判词）：人类日志轨迹过同一个闸的比例（64 个转弯），说明 PAI 定的门限在 nuPlan 上是否偏紧 / 偏松。
- 参照：xp0 在 A / B 单元上的完成数，「专家从起点也失败」的单元列出。

## 登记线

- **A**：主 Δ = 2 s 上专家完成率 ≥ 0.6（分母 = 可倒回的 A 单元）。Δ 1 / 3 s 并列报告，不改判词。
- **B**：Δ = 2 s 上过闸的完成数 / 交接数 ≥ 0.5。
- **C**：专家从起点完成率 ≥ 0.8（其余是专家无论如何都失败的）。
- 附加（不改判词，决定能不能当标签）：蠕变 ≥ 80 %、追尾与 at-fault 计数、行产量。

## 规模与成本

- rollout：identity 64 + xp0 64 + D1–D3 各约 64（A 43 + B 中有违规的 ≤ 21，减去不可倒回的）≈ 320，smoke 另加 idB。
- 实测：held read 一个 job 77 个场景 600 s（服务 556 s），VRAM 峰值 11.6–13.8 GB（XP-L12-E46 read，2026-10-15）。按每 rollout 约 8 s job 时间，
  320 个约 45 job-min；每个 launch 另加约 45 s 启动。每层 4 个 chunk job（`rwn-<tag>-c<i>`，--vram 20 --cpu 6 --ram 40，history key
  alpasim-read4 = nuPlan held read 的同一服务），phase A → n-tfail（CPU）→ D1–D3 → read（CPU），全部一次排队、`--after` 串起来。
  一个 stage 墙钟约 40 min，全部 < 1.5 card-h，低于 3 h，不做 profiling。
- 优先级 6（低于在跑的 lane，`cl top` 先看）。分阶段：smoke 2 个场景（一个 A、一个 B，加 idB）→ 检查交接 → 10 个场景 → 其余 52 个 → 64 个合并读数。

## 交付

- 代码：`experiments/alpasim/lib/rule_expert.py`（JEV_EXPERT_FROM）、`sh30_driver.py`（一行）、`experiments/route_follow/scripts/rf_rewind.{py,sh}`
  （`train64` 模式、`n-*` 子命令；不另写 runner）。
- 结果：`experiments/route_follow/results/rewind_train64/`（rewind_train64.md、逐单元 CSV、read.json），图 `experiments/route_follow/figs/` 下 A、B 各一张 BEV。
- 盒：`$DATA_DIR/runs/route_follow/rewind_train64/`（stage 目录、pack、jobs.txt）。
- README 一段、decisions.md 一行 + 决策文件。

## 补记 1（smoke 之后，按登记规则停在 smoke，2026-10-15 18:50 盒时间）

- smoke 1（`$DATA_DIR/runs/route_follow/rewind_train64/smoke`，00e12cf6；A 单元 4109b79d84ea5053 event k_z 13，h = 11 / 9 / 7；B 单元 537935a8 无违规）：
  交接 session 在 h 前**不是**逐位相同：D1 / D2 的模型 plan 在 k 7 起与 identity 差 7.8 mm（同一 ego 行），h 时轨迹差 ≤ 5 mm、v(h) 差 ≤ 0.005 m/s；
  D3（h = 7）逐位相同；idB 与 id 逐位相同。当时以为是 launch 的场景组成不同（D launch 只有 1 个场景、id 有 2 个），改成 D launch 跑 id chunk 自己的
  场景表（0fecfb10，没有交接的场景重放 identity）。
- smoke 2（`.../smoke2`，0fecfb10；同一 A 单元 + 两个完成转弯 537935a8、bba5d587，两者都无违规，B 层没有交接）：组成相同也不逐位相同：
  3 个交接 session 在 h 前 plan 最大差 0.14 m、轨迹 3.5 cm、v(h) 0.029 m/s；D launch 里的 identity 重放 5 / 6 与 id 同分同结局。
  原因在渲染器：第一个 plan 差异处，driver 收到的 CAM_F0 JPEG 字节数就不同（例如 4109b79d 在 ts 3.017 s），ego 行相同；这一次 idB、D1、D2
  彼此逐位相同而 id 是不同的那个。即 nuPlan 路径（sh30 + NRE 渲染，RENDER_AGGREGATED）的同场景 session **跨 launch 不逐位可复现**，
  偶尔渲染出不同帧（smoke 1 里 idB = id，smoke 2 里 idB ≠ id）。pipeline_profile.md 的「逐位相同」在这里不成立。
- 量级：4109b79d 上 id 与其他 session 的锚点差 k 7 / 9 / 11 时 2 mm / 1.4 cm / 3.5 cm（idB 同样大），交接状态在 identity 的 run-to-run
  散布之内；但这不是登记的逐位条件。
- 判定：按「若 h 前不逐位相同：停在 smoke，报告，不改模拟器」停。没有提交 10 个和全部场景的读数；没有结果文件和决策条目。
- smoke 附带看到的（单个单元，不作判词）：A 单元 Δ 1 s 专家 offroad，Δ 2 / 3 s 都是 short（专家减到 4.6 m/s，18 个决策 = 9 s 片段内到不了 exit + 10 m）；
  xp0 在 3 个场景里只完成 1 个（2 个 short）。9 s 片段对按 4.6 m/s 过弯的专家可能太短，这会压低 A 与 C 的完成率，与倒回本身无关。
- 恢复的选项（待用户定）：(a) 把登记条件放宽为「h 前与 identity 的差在 identity 自身 run-to-run 散布内」，对照取同 launch 的 identity 重放；
  (b) 让渲染确定（模拟器 / 渲染器侧改动，超出本任务）；(c) 对每个单元在同一 launch 里同时跑 id 与 D（每场景多 rollout），仍受渲染器影响。
