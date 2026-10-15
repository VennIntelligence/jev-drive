# 预登记：RECOVERY-EXPERT REWIND（不训练，只跑 rollout），2026-10-16

**为什么。** 用户的问题（读完 SimScale CVPR 2026 与我们的 P4 / d297 教训之后）：把策略**自己**闭环里转弯失败的那一刻倒回 Δ 秒，
带着那一刻的速度和航向把车交给规则专家（吸引子），专家能不能救回来：转完弯、回到走廊。救不回来，就没有可学的恢复示范；
救得回来，恢复出来的轨迹就是下一课的候选标签。这一读只回答这一件事。

背景：规则专家 XP-L12-E46 从 held 场景起点开能完成 22 / 24 个 held 转弯（`results/rule_expert/README.md`，预登记 Amendment 1），
PAI 281 的 36 个严格转弯的专家读数在 Tokyo 跑（`research/next-round/turn-night.md` 第 7 节）。P3 / P4（d290、d297）的教训：
从中途状态冷启动会出「速度棘轮蠕变」（窗末速度 0.1–1.8 m/s、被追尾）和入口交接时原地枢转；所以这里**不冷启动**，见下面的交接机制。

写于任何 rollout 之前（只看过存档的 Tokyo r281/rsg 的转弯结局分布：event 21、off 6、completed 5、short 4，以及 box_det 的确定性结论）。
补记只追加，不改原文。

## 交接机制（不重启，不 seed）

- 盒上 AlpaSim PAI 的同一场景 rollout 逐位可复现（`results/p4_self/box_det.md`：盒对盒 115 / 115 个 session 的 plan、轨迹、分数逐位相同；
  盒对 Tokyo 从不逐位相同，所以全部在盒上跑，失败状态取盒上 identity 自己的 rollout）。
- 倒回 = 同一场景重开一个 session，RSG-L18-s0 从决策 0 开，与 identity 逐位相同地开到决策 h，从 h 起 driver 返回规则专家的 plan
  （`onp_log_driver.py` ONP_CELLS 新加的 `rew` 单元，`rule_expert.plan(位姿, 实测速度)`，就是 pai_driver 在 JEV_EXPERT 下的同一调用）。
  模型照常每步运行并记录（`poses`），served plan 记为 `poses_branch`。
- 所以交接时的车速、航向、位置、控制器 MPC 状态、模型的历史与路线记忆、仿真里其他车，全是策略自己的，不是从日志或冷启动重建的。
  这正是 P4 自状态交接（src self）的同一路径，只是交接后接管的是专家而不是推过的 plan。
- 服务配置同存档参照 r281/rsg：`SH30_TAG=RSG-L18-s0 JEV_VCONT=1.0 JEV_LEAD=1`、gains 20、`send_recording_ground_truth=true`，
  **`JEV_TRACK=0`**：专家只能在旧路径上服务（pai_driver 对 JEV_EXPERT 强制 path 0），按 d309 的同路径配对规则，策略对照也用 path 0。
  结果头写明 path 0；不与 gate 路径的读数配对。
- 专家变体：XP-L12-E46（lookahead 1.2 s、入弯速度 4.6 m/s，rule_expert 默认），专家包 = Tokyo `pack_pai` 里这 36 个场景的 npz（原样拷到盒）。

## 单元与时刻

- 场景：`results/pai281/strict36.txt` 的 36 个严格转弯（盒上缺的 5 个 usdz 正在从 hf-mirror 下载，到了就补跑；到不了的在结果里列出）。
- 每个场景 phase A 两个 rollout：`id`（策略单独开，= 失败本身 = 对照）与 `xp0`（专家从起点开，= 参照）。
- 转弯与结局按 `rf_pai_follow.one`（rf_follow 的定义，PAI 版）：严格转弯 = 路线上 30 m 内净航向变化 ≥ 60°，入口 / 出口取变化的 10 / 90 %；
  **completed** = 车轨迹在第一个零分事件（at-fault 前 / 侧碰、offroad、left_corridor_laterally）之前越过 exit + 10 m，且越过处离路线 ≤ 2.5 m；
  否则 event / off / short_rear / short_slow。路线是每个 rollout 自己的 AlpaSim MAP 路线（沿它自己的位姿重建），同 rf_v2pai turns。
- **t_fail**（只对 identity 失败的弯）：
  - outcome = event：t_fail = 第一个零分事件时刻（rf_pai_follow 的 tz，评估器 2 Hz 序列上第一次置位）；
  - outcome = off：t_fail = 路线站点 ≥ entry − 20 m 之后、tz 之前，第一次 |横向偏差| > 2.5 m 的控制器采样时刻（2.5 m 是完成定义里的走廊）；
  - completed / short：没有失败时刻，不倒回（short 是慢 / 被追尾，没有可倒回的转弯错误；单列）。
  - k_fail = now ≤ t_fail 的最后一个决策；h_Δ = k_fail − 10 Δ，Δ ∈ {1, 2, 3} s；h < 17（运行时前 1.7 s 强制跟日志）的 Δ 不跑，按「不可倒回」列出。
- phase B：每个可倒回的 (场景, Δ) 一个 rollout `xpDΔ`：策略开到 h，专家从 h 开到 rollout 结束（PAI 片段 20 s，即场景结束；不另设 15 s 截断）。
- **策略对照**：同一 rewound 状态交还 RSG-L18-s0 = 同一 session 继续 = identity rollout 本身（盒上逐位可复现）。因此对照的完成率按单元选择
  就是 0（单元就是 identity 的失败）；配对线退化为「专家完成率 − 0」。这一点事先写明，不当作专家的额外证据；对照的价值在于核对：
  每个 xpDΔ session 在 h 之前的 served plan 与控制器轨迹必须与 identity 逐位相同（倒回正确性检查），smoke 另跑一个 `idB`（第二个 identity）。
- 第二层（其余 34 个零分场景）：本轮不跑（延后），主层结果出来后再按成本决定，另写补记。

## 量

- 完成率（上面的 completed），每个 Δ。
- 交接时状态：v(h)、|横向偏差|(h)、对路线的航向误差、h 时站点相对入口。
- **速度是否带过去**（smoke 闸）：v(h) 等于 identity 在同一时刻的速度（差 < 1e-6 m/s），且 h 之前 plan / 轨迹逐位相同；报告 h 后 0–1 s 的速度轨迹
  （专家与 identity 并列：只应按加速度上限偏离，不应跳变到 0 或日志速度）。
- 回走廊：h 之后、零分事件之前，第一次 |横向偏差| ≤ 1.0 m 并保持 1 s 的时刻减 t_h（settle，s）；未回到记为未回。
- 蠕变检查：v(h + 3 s) ≥ 2 m/s 的比例；同时报 rollout 末速度。
- 被追尾：h 之后有 collision_rear；at-fault：零分且 collision_at_fault（kind = collision）。
- 形状闸（d293，rf_p4.gate_of：本类 p95 曲率、p95 曲率变化率、横向加速度 ≤ 人 p95、入口速度 ≤ 本类 v_e + 1 m/s），只在 h 之后的轨迹段上算
  （t ≥ t_h、站点 ∈ [entry − 40, exit]、零分前）；入口在 h 之前已过时入口速度项记为过。只对 completed 计「过闸」（候选标签）。
- 参照：xp0 在全部 36 个与在失败单元上的完成数；「专家从起点也失败」的单元列出。盒上 identity 与 Tokyo r281/rsg 的结局一致性也列出。

## 登记线

- **RECOVERS**：存在某个 Δ，专家完成率 ≥ 0.6，且对策略对照的配对差 ≥ +0.3（对照按上面的说明是 0，所以第二条在第一条成立时自动成立；照写）。
  报告 bootstrap 95 % CI（场景重采样 10 000 次）。
- **NO RECOVERY**：三个 Δ 的专家完成率都 < 0.4：没有可学的恢复示范，停这条线。
- 0.4–0.6：不确定；按失败类型（corridor / offroad / collision / off）拆开读，不当作通过。
- 附加（不改判词，决定标签能不能用）：通过的 Δ 上蠕变检查 ≥ 80 %、被追尾与 at-fault 计数、形状闸通过率。蠕变检查 < 80 % 时恢复轨迹不能直接作标签
  （d290 的速度棘轮问题），写明。
- 若 smoke 显示交接没带上速度（v(h) 与 identity 不同，或 h 前不逐位相同）：停在 smoke，报告，不改模拟器。

## 规模与成本

- rollout：36 × 2（phase A）+ 约 27 × 3（phase B，按 Tokyo 的 27 个 event / off 估）≈ 153，smoke 再加 1。
- P4 在盒上的实测：一个 launch 一个 rollout 约 241–359 s（含启动），3 个 launch 并排一张卡；一个场景 job 约 phase A 5 min + phase B 6 min + extract 2 min ≈ 13 min，
  每卡约 3 个 job 并行 → 36 个 job 在 3 卡上约 1 h 墙钟、约 3 card-h。低于 3 h，不做 profiling。
- 池：`rf_rewind.sh fanout`，每场景一个 job（`rw-<id8>`，--vram 22 --cpu 4 --ram 32，history key alpasim-p4c1），优先级 6（低于在跑的 lane）。
  分阶段：smoke 1 个场景（e37aa159，corridor 晚而宽，三个 Δ 都可倒回；SMOKE=1 加 idB）→ 检查 → 约 10 个 → 全部 + collect。

## 交付

- 代码：`experiments/onpol_pai/lib/onp_log_driver.py`（rew 单元）、`experiments/route_follow/scripts/rf_rewind.{py,sh}`。
- 结果：`experiments/route_follow/results/rewind/`（rewind.md：每个 Δ 专家对策略、回走廊、末速、at-fault；rewind.csv 逐单元；read.json），
  图 `experiments/route_follow/figs/rewind_bev.png`（2–3 个场景：identity、xp0、三个 Δ 的路径与交接点）。
- 盒：`$DATA_DIR/runs/route_follow/rewind/run`（每场景 launch 目录、a / all.pkl、tfail.json）。
- README 一段、decisions.md 一行 + 决策文件。

## 补记 1（smoke 之后、第一批正式读数之前，2026-10-15 17:10 盒时间）

- smoke 场景 e37aa159（`$DATA_DIR/runs/route_follow/rewind/smoke`，提交 5a8beed6）：identity = event / corridor，t_fail 15.1 s，k_fail 151，h = 141 / 131 / 121。
  **交接机制过闸**：三个 xpDΔ session 在 h 之前 served plan 与控制器轨迹与 identity 逐位相同（3 / 3，最大差 0），v(h) 与 identity 相同
  （7.77 / 7.57 / 6.59 m/s，差 0）；第二个 identity（idB）分数相同。h 后 1 s 专家把车速降 2.7 / 2.5 / 1.7 m/s（向 V_E 4.6 m/s 减速，不超过
  专家的 A_DEC 3 m/s²），不是跳变：速度确实从策略的状态带过去了。
- smoke 的单场景数（不作判词）：Δ 1 s completed；Δ 2 / 3 s 都是 short_rear（专家减到 4.6 m/s 后被回放的后车追尾，PAI 在追尾处截断）；
  xp0（专家从起点）也是 short_rear。可见「被追尾」可能是专家的主要失败方式。
- 由此事先追加（探索性，不改登记线和判词）：按专家结局（completed / event / off / short_rear / short_slow）拆开每个 Δ 的计数；
  「未被追尾时的完成率」= completed / (单元数 − short_rear) 单列，标为探索性。
- 盒上 5 个缺失 usdz 已从 hf-mirror 下到（fetch.log 5 个 ok），36 个场景全部可跑。分阶段：10 个场景（17:10 提交），检查后再提交其余 26 个 + collect。
