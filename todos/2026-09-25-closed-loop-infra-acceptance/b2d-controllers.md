# Bench2Drive 控制器验收：给控制器一份已知是好的 plan

状态: 预注册（2026-09-25 15:26 CST（commit 9ca6151），写于任何控制器臂运行之前；expert a 已在跑，还没看它的结果）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)
代码: `scripts/b2d_expert_agent.py`（PDM-Lite + 轨迹日志）、`scripts/b2d_zeroshot_agent.py` 的 `"replay"`、`scripts/infra_ctl_accept.sh`

## 问题

Alpamayo 和 openpilot 的 B2D 分数里，有多少是控制器造成的？诊断里已经看到 Zoo PID 只执行了 plan 横向偏移的 4%、碰撞后
把车顶住（[Alpamayo 闭环诊断](../2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md) 第 7 节），但那是在模型 plan
上量的，plan 本身好不好说不清。这里把 plan 换成一个已知是好的：**专家自己开出来的未来轨迹**。控制器拿到它还开不好，
就是控制器（或适配层）的问题，与模型无关。

## 设计

**专家。** PDM-Lite（privileged rule-based expert：直接读仿真器里的路线、所有 actor、红绿灯，不看相机），用 SimLingo
仓库自带的 Bench2Drive 副本里的 `leaderboard/team_code/autopilot.py` 原样运行；它在公开的 B2D 220 条上 DS 97.0
（`pdm_lite_b2d_traj/merged.json`），本文 20 条路线上公开结果都是 DS 100。`b2d_expert_agent.py` 只是它的子类，每个 tick 记
一行 hero 的仿真器位姿、速度和它施加的控制量（`expert.jsonl`）。PDM-Lite 需要那棵树里 `CarlaDataProvider.active_scenarios`
的登记，所以**所有臂（专家和控制器）都在 SimLingo 的 Bench2Drive 副本里跑**。这棵树和官方 0.0.4 只差 D1（去掉 4000 tick
上限）、D2（完成阈值 90）、D5（active_scenarios 记账）等（[SimLingo catalogue](../2026-09-25-simlingo-catalogue/README.md)
的差异表），不改任何 actor 行为；专家和控制器臂之间的比较不受影响，但这里的 DS 不能和官方树上的数字直接比。

**已知是好的 plan。** 控制器臂用 `b2d_zeroshot_agent.py`，配置和考试时一样（相机 rig、规划节奏、控制器、开关），只是
不连模型：每次规划时，plan = 专家日志里**同一经过时间**（从各自第一个 agent tick 起算）之后 0.25–5 s 的后轴轨迹，
用 hero 当前的仿真器后轴位姿换到 rig 坐标系（x 前、y 左）。这就是一个完美 planner 在这一刻会给的 plan：从车现在的位置
出发，回到专家的路径，并按专家的时刻表走。之后的路径与考试完全相同：plan 交给控制器，位姿用传感器 PoseFilter，
Zoo PID 的 target point 来自它自己的 RoutePlanner。控制器跟得好，车就会复现专家的轨迹，scenario 也按专家那次展开。

**路线（20 条，TM seed 0）。** Alpamayo 横向修复 re-smoke 的 16 条（2390、24211、1711、2373、3564、1833、1852、1956、2668、
4183、11381、1825、2084、2086、2091、2115，覆盖绕行、路口左右转、起步、行人、红灯），加 4 类它们没有覆盖的场景，每类取
bench2drive220 里 id 最小的一条：HighwayCutIn **2286**（高速）、HardBreakRoute **24330**（前车急刹，Town10HD）、
SequentialLaneChange **17563**（连续变道）、T_Junction **26458**（小镇 T 字路口，Town07）。选择只看场景类型和 id，
没看任何结果。

**臂。**

| 臂 | 控制器 | 规划节奏 | 用在哪 |
|---|---|---|---|
| E-a | PDM-Lite 专家 | 20 Hz（自己的控制） | 参考；它的日志就是其他臂的 plan |
| E-b | PDM-Lite 专家，同配置重跑 | — | 参考的噪声（同 seed 两次之间的差） |
| Z2 | Zoo PID（Bench2DriveZoo UniAD/VAD 官方 PID），F1 forward-only，`zoo_cadence: plan`（每次规划算一次、保持到下次） | 2 Hz | Alpamayo 考试（`full220-alpamayo-zoopid-f1`） |
| Z5 | 同 Z2 | 5 Hz | openpilot 考试的节奏（Lebowski plan_every 4 × 20 Hz 相机 = 5 Hz）；这里用 Alpamayo 的 10 Hz rig、plan_every 2 得到同样的 5 Hz，相机只影响成本不影响控制 |
| F2 | 预注册的固定控制器（`scripts/b2d_controller.py` preset carla，20 Hz 路径跟踪，`controller_config.json`） | 2 Hz | Alpamayo 预注册 smoke / `full220-alpamayo` |
| F5 | 同 F2 | 5 Hz | 分开“控制器”与“节奏”两个因素 |
| P1 | Zoo PID 纵向 + 固定控制器 20 Hz 横向 | 2 Hz | 横向修复 P1（诊断第 8 节） |
| P2 | Zoo PID 纵向 + Zoo 转向 PID 瞄准 plan 1.5 s 处的点 | 2 Hz | 横向修复 P2（诊断第 8 节） |

TCP partner 不在这里：它不是 plan → control 的控制器，而是一个自带 PID 的学出来的 agent，它的 1.5 m/s 上限是出厂的低速
油门上限（`b2d_partner.py`），在结论里单列。HUGSIM 的控制器在 [hugsim-controllers.md](hugsim-controllers.md)。

## 指标（跑之前定死）

跟踪误差都用仿真器真值（`ticks.jsonl` 的 `truth`，后轴），对 **E-a 的后轴路径**（折线）：

- **e_lat**（cross-track error）：车后轴到专家路径最近点的横向距离。
- **e_lon**（along-track error）：车在专家路径上的弧长位置 s_r(t) 减去专家在同一经过时间的弧长 s_e(t)；负是落后。
- **e_head**：车航向与专家路径在投影点处航向之差。
- 统计范围：车速 ≥ 1 m/s 的 tick，从第一个 tick 到（控制器臂自己的）第一次碰撞或专家日志结束，取先到者；所有路线合并
  （pooled）报 median 与 p95，另报逐路线。
- **lat_ratio**（诊断 7.3 的定义）：车速 ≥ 2 m/s、plan 在 2 s 处横向 ≥ 1 m 的规划，2 s 后实际横向位移 / 计划横向位移的
  中位数。这里的 plan 就是专家的，比值应接近 1。
- 结果：官方 per-route DS、RC、状态、违规（碰撞、闯灯、route deviation、blocked、timeout）；“卡住”= `Agent got blocked`
  或 route timeout（这棵树没有 4000 tick 上限）。

## 判据（跑之前定死）

每个控制器臂分别判，全部满足才算 **pass**：

| # | 判据 | 门槛 | 理由 |
|---|---|---|---|
| A1 横向跟踪 | pooled e_lat | median ≤ 0.30 m 且 p95 ≤ 1.00 m | 半个车道约 1.75 m；p95 1 m 以内不会压线到别的车道 |
| A2 纵向跟踪 | pooled \|e_lon\| | median ≤ 2.0 m 且 p95 ≤ 8.0 m | 2 m 约是 8 m/s 下 0.25 s；超过这个量级 scenario 的时序（路口让行、前车急刹）就会和专家那次不同 |
| A3 横向执行 | lat_ratio | ≥ 0.7 | Zoo PID 在模型 plan 上是 0.04，固定控制器 0.64；给专家 plan 应当接近 1 |
| A4 结果 | 20 条平均 DS | ≥ E-a 平均 DS − 5 | 5 DS 约是一条路线从 100 掉到 0 的四分之一 |
| A5 不卡住 | E-a 完成、控制器臂卡住（blocked / timeout）的路线数 | 0 | 卡住就是诊断里 TickRuntime 的来源 |
| 只报告 | 控制器臂有、E-a 没有的碰撞（路线数）；e_head；E-a 与 E-b 的差 | — | E-a / E-b 之差给出“同一个专家重跑一次”的噪声，用来读 A4 |

如果 E-a 自己在某条路线上没完成，那条仍然参与跟踪统计（plan 仍是专家的），A5 不计它，A4 照算。E-a 与 E-b 的平均 DS
差超过 5 时，A4 的门槛放宽到这个差（写成偏离）。基础设施失败（server 崩溃、没有结果文件）的 attempt 重试一次；
重试后仍没有结果的路线按缺失报，不算进 A4。

## 运行

`scripts/infra_ctl_accept.sh all 0`（tmux 窗口 `ctl-plumb`，GPU 0，CARLA index 700–735）：先 E-a（5 个 worker），收集它的
日志，再把 E-b 和六个控制器臂并行（各 1 个 worker）。输出 `$DATA_DIR/runs/infra-accept/b2d-ctl/`。计算：
`scripts/infra_ctl_score.py`（写完再跑）。

## 偏离

**偏离 1（2026-09-25 16:00，控制器臂跑了前 1–3 条之后、看全部结果之前）：plan 的来源从“按经过时间”改成“从车现在的位置、
按专家的节奏”。** 预注册里 plan = 专家在“同一经过时间”之后的轨迹。第一批路线里（2390 上 Z2、F2 都以 `Agent got blocked`
结束，专家 6 s 就开完）看到这个定义本身有两个问题，和控制器无关：(1) 车一落后，plan 的第一个点就在车前几米（e_lon 中位数
约 −7 m 时第一个点在 7 m 外），这是一个要求“瞬移”的不连续 plan，没有任何 planner 会这样输出；(2) 专家日志到头以后 plan
全部塌缩到终点一个点，Zoo PID 从 waypoint 间距读期望速度，读到 0 就刹车，于是停在终点前几米，被判 blocked。新的定义
（`b2d_zeroshot_agent._replay_path`）：把车的后轴投影到专家路径上得到弧长 s0，取专家在 s0 处的时刻 t*，plan = 专家在
t* 之后 0.25–5 s 的位置；专家在 s0 停过（红灯、让行，停在 [ta, tb]）时 t* = 经过时间截到 [ta, tb]，也就是专家等多久
plan 就等多久、不会更久。这样 plan 连续、从车的位置出发、速度剖面就是专家的，仍然是“完美 planner 此刻会给的 plan”。
代价：车落后时 scenario 的时序不再强制和专家那次相同。离线单元检查（直线 5 m/s、在 20 m 处停 6 s）：准时、落后、
等待中、迟到、早到五种情况下 t* 和 plan 都符合定义。指标和判据不变（A2 的 e_lon 仍按经过时间对齐，所以落后照样被记为
纵向误差）。旧定义下的控制器臂（Z2、Z5、F2、F5、P1 各 1–3 条）全部作废，移到 `discarded/`；E-a、E-b 不受影响。

另有一个与此无关的基础设施修复：SimLingo 的 Bench2Drive 副本在 `leaderboard/` 下带一个普通 package `team_code`，
Bench2DriveZoo 的 `team_code` 没有 `__init__.py`（namespace package），不管 sys.path 顺序如何都会被前者遮住，Zoo PID
的 wrapper 因此 import 失败（Zoo 臂第一次启动全部 crash，0 tick）。改成按文件路径加载 Zoo 的 `planner.py`，
行为不变（同一份文件）。

## 结果

（跑完再填。）
