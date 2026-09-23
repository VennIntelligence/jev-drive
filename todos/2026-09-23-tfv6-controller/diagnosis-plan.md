# W2 之后的诊断：差在哪、是什么情况

状态：open，2026-09-24。上游：[report.md](report.md)、`research/decisions.md` 第 31 条。

W2 只给了总账：C − B（我们的控制器对作者 waypoint PID）没检出、路线间差异极大；A − B（route + target speed 对 waypoint）+14 DS。这一轮**只诊断、不改控制器**，目的是在决定改什么之前，先知道每一分 DS 是在什么情况下、以什么方式丢的。结论都是描述性的，不做新的效应检验。

## 手上已有的数据

202 个有效 case 的逐 tick 日志（`frames.jsonl`）里每 tick 都有：TFv6 的 waypoint、route、target speed，四个臂各自的 raw / final control（shadow，即不执行、只按同一时刻的输入算出来的 control），实际执行的 control，C/D 的 controller reason，真值位姿与速度，启发式状态；另有官方 `infractions.json`（带 frame 和行驶里程）。**没有**其他车辆和行人的状态。

这意味着不跑 CARLA 就能做反事实：在 C 的轨迹上，每个 tick 都知道 B 在同样的输入下会给什么 control。

## D1：只用现有日志（零 CARLA）

1. **DS 拆账。** B2D 的 DS = RC × 违规惩罚系数的连乘。对每一对（C 对 B、A 对 B，同路线同 seed），把 DS 差按对数分解到 RC 损失和每一类违规，汇总成"每条路线、每一类原因各占多少 DS"。回答：C 丢的分，是没开完、撞了、偏离路线，还是 min speed？
2. **分歧时间线。** 同一 (路线, seed) 的四个臂从同一个出生状态开始，控制不同才会分开。对每对找第一个分歧 tick（真值位置差 > 1 m 或速度差 > 1 m/s），记下当时的情境：速度段、TFv6 意图（target speed 为 0 = 模型要停；plan 的首段 / 8 点平均隐含速度）、plan 曲率（转弯与否）、stop sign / creep 状态；再记下分歧之后第一个违规事件及其时间差。按情境汇总：分歧多发生在起步、跟车减速、停车、还是转弯？
3. **同轨迹反事实。** 在每条轨迹上比较实际执行的 control 和其他臂的 shadow control。按情境分箱（速度段 × TFv6 意图 × plan 曲率），报 C 与 B 在纵向（throttle − brake）和横向（steer）上的差值分布，A 与 B 同样。这个量不受轨迹分歧影响，直接看两个控制律在同一输入下哪里系统性地不一致。注意 PI 类控制器在非执行臂里的积分状态会漂，这一点要在结果里单列。
4. **纵向跟踪。** plan 隐含的速度曲线与实际速度的差：B、C 各自是否跟得上 TFv6 要的速度，滞后多少，在起步和减速停车时各差多少。
5. **横向相对 route。** 用每 tick 的 `route_prediction` 算车相对模型 route 的横向偏移：B、C 走 waypoint 时是不是系统性地切弯或外偏，A 走 route 时又如何。回答 A − B 的 14 DS 里有多少可能来自横向。
6. **故事清单。** 按 DS 损失排序，列出前 10 个最有信息量的 (路线, seed, 臂, 时间窗)，每个一句话写清"发生了什么"，作为 D2 的候选。

产出：本目录 `diagnosis-logs.md`（中文，表格加图，图按 `research/plot_style.py`，每张图后 1–3 句说明看什么）和 `results/diagnosis/` 下的小 CSV。D1 结束后停下，等 Mac 审过再决定 D2。

## D2：少量 CARLA（D1 审过之后，预算约 2 小时）

- **可重复性。** 同 seed 重跑 A 臂时 2091 差了 40 DS，所以单个 case 的大差距可能只是噪声。对 D1 故事清单里 |C − B| 或 |A − B| 最大的几组，把涉及的臂在同一 seed 下各重跑 2 次，看差距是否还在。
- **录像。** 这些重跑开 CARLA recorder 并渲染离屏追车视角，在关键时间窗截 contact sheet、剪短 mp4，让人能直接看到车在那一刻遇到了什么（其他车、路口、行人）。逐帧日志里没有的场景上下文由这一步补齐。
- 具体清单由 D1 的结果决定，写进 D1 报告，由 Mac 批准。
