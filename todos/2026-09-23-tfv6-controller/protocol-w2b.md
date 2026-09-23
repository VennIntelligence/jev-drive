# W2b 协议：修正静止切向后重跑 C/D，渐进放量

状态：**冻结**，2026-09-24，由 Mac 侧写定。上游：[protocol.md](protocol.md)（W2，除本页改动的部分外全部沿用）、[diagnosis-tangent.md](diagnosis-tangent.md)、`research/decisions.md` 第 31 条。

## 为什么重跑

W2 冻结的轨迹变换规定：首点切向取"原点 → p0"的方向，|p0| ≥ 5 cm 即采用。车辆静止时 TFv6 的近端 waypoint 会有几厘米的横向抖动，这条规则会给出接近 90° 的切向，经过后轴修正后造出假目标点（phantom）。C、D 两臂的 W2 结果因此作废。A、B 两臂执行的 control 不经过这个变换，W2 的 A、B 结果沿用。

## 唯一的改动：切向规则

沿点列从原点累加弧长 s_i（s_0 = |p_0|）。**s_i < L（L = 1.389 m，actor origin 到 rear axle 的距离）时 t_i = x 轴（车头方向）**；s_i ≥ L 时沿用 W2 的差分切向，包括原来的退化处理。其余一律不变：r_i = p_i − L·t_i + (L, 0)、y 取反、N = 8、trajectory_dt = 0.25 s、不外推、控制器配置、D 臂的两个 opt-in、作者启发式、四臂 shadow 与 guard、`controller_speed()`。

理由：车走过的弧长不到一个 L 时，航向不可能转过很大角度，把这段的切向取为当前车头方向，误差上界是 L·(1 − cos Δψ)。W2 日志的离线复算显示，这条规则下 phantom 为 0（[diagnosis-tangent.md](diagnosis-tangent.md)）。

## 臂、路线、seed

只跑 C、D 两臂，路线和 TM seed 与 W2 的第 1、2 级完全相同：Dev10 × {0, 1, 2}、v1 保留集 6 条 × {0, 1, 2}，共 96 个 case。A、B 使用 W2 的正式结果，按 (级, 路线, seed) 配对。

## 运行时不变量（每个 case 跑完立即检查）

任何一条不满足，**立即停机**：停掉自己启动的 CARLA 和 route 进程，发 BLOCKED，写明是哪个 case、哪条不变量。不打补丁续跑。

- **I1** phantom = 0：所有 tick 都满足 ¬(‖r_0 − p_0‖ > 0.3 m 且 ‖p_0‖ < 0.3 m)。agent 内加 fail-fast 断言，同时离线复核。
- **I2** 没有 shadow guard 失败、shadow 不一致或 agent crash。基础设施失败仍按 W2 的规矩处理：最多重跑 3 次。
- **I3** 日志完整：frames.jsonl 覆盖全部 tick，官方 results/infractions 齐全，case 表里没有重复的 key。

## 渐进放量（每一级的门槛都是机械条件，满足就自动进入下一级，不等 Mac）

| 级 | 内容 | case | 进入下一级的条件（全部满足） |
|---|---|---:|---|
| S0 离线 | 单测；在 W2 全部 202 个 case 和 D2 日志上，用新规则重算 r_i | 0 | 新规则下 phantom = 0；直行时 r_i = p_i；弧长 ≥ L 的点与 W2 的结果逐位相同；坐标核对补上**静止段和起步段**（B 臂轨迹，按 W2 的方法比较 r_i 与后轴实际位置），给出误差表 |
| S1 金丝雀 | C：3514 seed 0/1/2、28154 seed 0、25318 seed 0；A/B 不变性：28154 seed 0 B、26405 seed 1 A（用新代码跑） | 7 | I1–I3 全部通过；**3514 的三个 seed 都不再在 3 s 内发生 static 碰撞**（修复确实生效）；28154/0/B 与 26405/1/A 的 DS 和官方状态与 W2 原结果一致（这两个在 D2 中 3/3 复现，用来证明新代码没有改动 A/B 的行为） |
| S2 小批 | Dev10 seed 0 的 C、D（3514 的 C 已在 S1 跑过，复用） | 19 | I1–I3 全部通过 |
| S3 全集 | 其余 C、D case | 72 | 每个 case 都检查 I1–I3 |

S1 中 C 臂的 5 个 case 是正式结果，计入全集（它们的路线和 seed 都属于全集），不重复跑；A/B 的两个不变性 case 只作检查，不计入。每一级结束时写一行 status.jsonl（进度、不变量、耗时）。

## 分析与判定

与 W2 完全相同：DS 配对差按路线整组做 cluster bootstrap（10000 次），主效应 C − B，D − C 与 A − B 只描述；判定、第 3 级（220 条）的继续条件、所有次要指标和机制描述，都照 protocol.md 执行。另外加两项：
- W2 与 W2b 的 C、D 并列对照：每个 case 的 DS 变化，以及 phantom 相关事件是否消失。
- 按 D1 的方法重做同轨迹反事实和 DS 拆账，只针对新的 C、D。

报告：`report-w2b.md`，中文，表格加图。
