# 预登记：nuPlan 上的 openpilot 式纵向（JEV_ACC，serve 侧，不训练），配对 RSG-L18-s0，held 460 / 18 决策，2026-10-16

PAI 臂的设计与补记照 [2026-10-16-pai-acc-prereg.md](2026-10-16-pai-acc-prereg.md)（PAI 读数：对 g10 +0.105 [+0.066, +0.145]，零分 76 对 76，新增 15 / 修好 15，
转弯代理 6/12 对 12/12；另一个代理在分析，本文件不重复）。用户 10-16 要求在 nuPlan 上也跑一次，因为盒子空闲。
本文件在任何读数（冒烟之外）之前写好并推送，之后只追加补记。

## 0. 预期（写在读数前）

**平或略差。** 理由：
- [d226](../../../research/decisions/226.md) / [d239](../../../research/decisions/239.md)：nuPlan 700 公开场景上 `JEV_LEAD` −0.086、`JEV_VCONT` −0.007、两者 −0.071（P2H10-F 时期）；
  nuPlan 给 progress 计分的方式与 PAI 不同（PAI 同一开关 +0.086）。ACC 臂按 PAI 的规则带着 VCONT + LEAD，所以已知的 lead 代价会算进来；第 3 节的 VL 臂把它拆出来。
- [d280](../../../research/decisions/280.md)：榜单 GT 是到整条日志路径的最近点距离，落后对 GT 贡献 0，只有**越过日志终点**才加纵向部分（public 上 28 % 的 GT，集中在人起步慢的场景）。
  ACC 只会让车更快，所以 GT 只可能在越终点上变差，横向部分不该动。
- [d252](../../../research/decisions/252.md)：navtest 上一刀切速度缩放没有净正档，快 5 % 就新增 NC 失败。
- [d278](../../../research/decisions/278.md)：nuPlan 上的「慢」主要是巡航落后，对榜单 GT 是 0，对分数只经由 progress（上界小）。
- nuPlan 的 scene score 是 `min(progress / 0.8, 1)`，held 上 RSG-L18-s0 非零场景大多已饱和；提速能拿的分很少，能丢的（零分）多。

## 1. 臂与 nuPlan 的接线（`experiments/alpasim/lib/sh30_driver.py`，本次提交）

与 PAI 同一规则（serve_fix.py (d)，参数不变：a_lat 2.4、弯前减速 1.5、ACC 下降 2.5 m/s²、A_CRUISE 表、守卫 2 m / 1 m/s²、路径延长 300 m），只改驱动的输入：

- **交接**：nuPlan `force_gt_duration_us` = 0.5 s、控制步 0.5 s。AlpaSim 0bb4c4b `unbound_rollout.py`：`closed_loop_start_us = render_start + 0.5 s`，
  `force_gt_period = [render_start, closed_loop_start]`（含端点），所以第 0 步（目标时刻 0.5 s）被强制回放，第 1 次 drive 调用（会话时钟 0.5 s）就是交接步，
  此时车速是日志（人）的速度（controller.py 的 handoff 步还会把动力学状态强制成日志值）。`v_hand` = 第一个 `t_s ≥ 0.5 s` 的调用的车速
  （`FX.ACC_T_HAND_NUPLAN`，`Serve(..., acc_t_hand)`；PAI 仍是 1.6 s，PAI 路径逐位不变，见第 2 节）。会话时钟 `t_s = (now − 第一次调用) / 1e6`。
- **弯道限速的路线**：nuPlan route 消息（`route_generator_type: MAP`，车前约 40–80 m，`route_start_offset_m: 40`）按 d287 的方式作记忆合并
  （`rmem.Tracker.polyline`，与 JEV_ROUTE 同一个 pose 查找）。RSG-L18-s0 有 ROUTE-MEM 分支，所以 ACC 直接用模型自己的 `s.rt`；别的 checkpoint 另建一个 Tracker。
  会话前约 40 m 没有路线覆盖，此处不限（PAI 同）。
- **lead**：`JEV_LEAD=1` 的 openpilot lead MPC，不改；nuPlan 每次调用推进 10 个 planner tick（0.5 s）。
- **停车守卫**：读模型**上一决策**的 plan；nuPlan 上一决策是 0.5 s 前（PAI 是 0.1 s），所以守卫比 PAI 晚最多 0.5 s 生效。
- **路径延长**：同 PAI。nuPlan 没有 JEV_TRACK（参照也没有），返回的 8 个位姿直接交给官方 MPC。
- **nuPlan 的特殊处**（参照存档读数的输入统计，不是结果）：held 452 个场景交接车速中位 6.0 m/s（10 / 25 / 75 / 90 分位 0.0 / 2.8 / 8.5 / 11.0），
  26 % 低于 3 m/s，**没有 ≥ 16 m/s 的交接**（最大 14.7），会话内最大车速 ≥ 16 m/s 的只有 0.4 %。PAI 上 ACC 的主要收益来自高速交接后模型掉速（90ab8c24 26 m/s）
  与被追尾，这一类在 nuPlan held 上几乎不存在。低速交接时 `v_set` = 模型自己 2–4 s 段速度，ACC 只是用 A_CRUISE 的 1.6 m/s² 更快地爬到它，
  正是 d280 的「人起步慢、我们跑到人前面」——**越终点 GT 风险**，单独报告。nuPlan 的其他车辆回放日志、不避让（physics / traffic NONE），
  更快到达 = 更早贴上日志里减速的前车，lead MPC 与守卫是唯一的刹车。

- 关时逐位相同：
  - `experiments/alpasim/scripts/acc_check.py`（本次扩展）：(1) JEV_ACC 未设，4 组开关 × 3 000 个随机决策，与 ACC 之前的 serve_fix（fc245afc~1）0 差异；
    (3) **JEV_ACC=1 且 PAI 默认交接**，2 组开关 × 3 000 个带路线与时钟的随机决策，与 PAI ACC 读数所用的 serve_fix（dd4c8d24）0 差异；(4) nuPlan 交接锁在 0.5 s；(2) 原有行为检查。
    Mac 上全过（本次提交前运行）。
  - 驱动：box 上闭环闸（`rf_nuacc.sh` 的 ident 作业）：改动前代码（worktree 在 dd4c8d24）对新代码 JEV_ACC 未设，`SH30_COMPILE=0`，smoke5 的前 2 个场景 × 18 决策，
    每个决策 poses 逐位相等（`rf_tts.py ident` 的同一比较）。闸不过，held 读数不开始（pool 的 `--after` 链：失败的上游让下游失败）。

## 2. 冒烟（不计入判据）

同一链里、闸之前：新代码 JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 跑 8 个非 held 场景（smoke5 + 3 个不同 log 的 train 场景）× 18 决策。
闸要求：每个推理决策都有 `fix.acc`，无推理错误，`v_hand` 在第 1 次调用（0.5 s）锁定，胜出项里出现 `set`。
冒烟的分数不看，不判。

## 3. 读数（box，GPU pool，PRIO 13，官方模式）

- 模式：nuPlan 的 `ALPASIM_FAST` 默认仍是 0（`sim_fast.py` 文档：nuPlan 路径未验证；git log 到 dd4c8d24 没有改默认），本次三个臂都是官方相机集。
- 参照：存档读数 `route_mem/read/RSG-L18-s0`（2026-10-13 02:44，`rm_chain.sh read RSG-L18-s0 <i> 6`，无 serve 开关，官方模式，held_n6 的 6 个 chunk 表）。
  之后代码路径有实质变化（5633b19c 起每个 read 4 个并发 rollout / renderer cache 5，rmem 的 route v2 改动，驱动里加了多个默认关的开关），
  所以**同时重读参照**，同一代码、同一模式、同一时间段：
  - `ref` = `RSG-L18-s0-re1016`：无 serve 开关（= 存档参照的配置）。**判据的配对对象。**
  - `acc` = `RSG-L18-s0-acc`：`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1`（PAI 臂的 serve）。
  - `vl` = `RSG-L18-s0-vl`：`JEV_VCONT=1.0 JEV_LEAD=1`，无 ACC。描述性：拆出 lead / VCONT 本身在 RSG-L18 上的代价（d226 是 P2H10-F 时期）。
  - 存档 `RSG-L18-s0` 对 `ref`：描述性（同配置重读的一致性）。
  每臂 6 个 chunk（`rm_chain.sh read RSG-L18-s0 <i> 6 <sfx>`，`--env` 传开关，`CL_HIST_KEY=alpasim-read4`，20 GB / 6 核），全部 `--after` 冒烟闸。
- 报告作业（CPU，`--after` 18 个读数作业）：`rf_nuacc.sh report` → `ol_confirm.py`（log 聚类 bootstrap 10 000 次，base = ref）、`rf_follow.py`（held 24 个转弯完成）、
  `rf_nuacc.py report`（GT 拆分：用 `gt_decomp/scripts/gd.py` 的评估器重实现，每个 rollout 取 GT 最大的那一步，横向部分 = `lateral_dist_to_gt_trajectory`，
  越终点部分 = GT − 横向；配对差的 log 聚类 CI；ACC 各项胜出比例、守卫比例、交接车速；新增 / 修好的零分逐个列出）。
  输出 `$DATA_DIR/runs/route_follow/nuacc/report/nuacc.md`（+ `nuacc.json`、`held18/confirm.md`、`follow/`），拷回 `experiments/route_follow/results/nuacc/`。
- 只比较各臂都有分的场景（ol_confirm 的规则，约 452 个）。

## 4. 判据（读数之前写死；acc 对 ref，按场景配对，log 聚类 bootstrap）

ACC 在 nuPlan held 上**过**当且仅当全部成立：
1. held 均分配对差的 95 % CI 下界 > −0.01；
2. 零分 ≤ ref + 2；
3. 横向 GT（cross-track，上面的定义）配对差点估计 ≤ +0.05 m（余量依据：d300 不同配置之间 GT 差 ±0.04 m 量级，TTS 的 +0.133 m 是明显变差）；
4. at-fault 里程（总 km / at-fault 事件数，榜单的 Average Distance Between At-Fault Incidents）点估计 ≥ 0.9 × ref（held 上约 20 个事件，±2 个约 ±10 %）。

单独报告、不作闸：越终点 GT（预期变差）、总 GT、at-fault 零分、零分按碰撞 / 出路 / 走廊拆分、转弯完成、vl 臂与存档参照。
**选槽规则不变**：nuPlan 最后一个槽按 held 18 决策的横向 GT、零分、at-fault 里程选，**从不按公开均分**。
**公开 1491 / 10 决策护栏只在 held 过时才读**（不在本次链里；过了由主会话另行提交，d300 的 `rf_lrsteps.sh guard` 路径）。

## 5. 成本

冒烟 + 闸约 15 分钟（3 个短 job 并行）；held 3 臂 × 6 chunk，每 chunk 约 10 分钟，4 卡并行约 1–2 波；报告约 10 分钟。墙钟约 45–60 分钟，约 3.5 card-h。
证据等级上限：1 seed、452 场景、按场景配对。

## 补记 1（闸作废，读数之前，2026-10-16 18:15 CST）

- 闸作业 1016-175111-d316 失败：改动前代码（dd4c8d24）对新代码 JEV_ACC 未设，2 个场景 / 36 个决策，poses 最大差 0.258 m，不逐位相同；18 个 held 作业与报告因依赖失败，没有任何读数。
- 逐决策看差（`smoke/n_old` 对 `n_new`）：场景 0c318d79 前 13 个决策逐位相同，之后 4–47 mm 漂移；场景 24567f5a 第 0 个决策相同，第 1 个决策 **anchor 与 hist 逐位相同而 plan 差 16 mm**
  （同样的位姿输入、不同的输出 = 渲染图像或推理不确定），之后车位置漂移到 ≤ 2 cm。代码改动会从第 0 个决策起就出现（JEV_ACC 未设时驱动走的是原分支），这里不是。
  同时 nuPlan 快速模式那条线今天测得：盒上官方 nuPlan 读数本身**跑与跑之间不逐位可复现**（今天的官方重跑对存档 held：154 个场景里 141 个分数相同，152 条里 101 条路径相同；
  `experiments/alpasim/results/pai_fast.md` nuPlan 节）。所以闸的「闭环逐位相同」在 nuPlan 上不可达，与本次改动无关。
- 闸改为：Mac 上 `acc_check.py` 的代码级逐位相同（第 1 节，已过）+ 同代码重跑噪声作为解释尺度（上面那条线的测量；本读数里 stored 对 re1016 的配对也给出同一量）。
- ACC 冒烟部分（同一闸作业）全过：8 个场景、144 个推理决策都有 `fix.acc`，0 错误，交接全部锁在 0.5 s。胜出项 lead 129 / model 7 / ramp 4 / set 4，守卫 stop 22 / brake 9：
  nuPlan 上 lead MPC 几乎总是起作用的那一项（日志回放的前车），ACC 自己的 set / curve 很少决定速度——预示 acc 与 vl 臂会很接近。冒烟不判。
- 重交：`rf_nuacc.sh held`（同 3 个臂、同判据、同报告，无闸依赖），PRIO 13，官方模式（nuPlan 快速模式无加速，默认仍 0）。
