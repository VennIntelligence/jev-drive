# D3a：W2/W2b 全量日志筛查（零 CARLA）

按 [`diagnosis-plan-d3.md`](diagnosis-plan-d3.md)，只读取 canonical W2 的 96 个 A/B 与 W2b 的 96 个 C/D case；没有运行或更改驾驶。脚本 [`b2d_tfv6_d3a.py`](../../scripts/b2d_tfv6_d3a.py) 输出 [`results/diagnosis/d3/`](results/diagnosis/d3/) 的逐事件 CSV。停滞定义为真值前向速度绝对值连续 `<0.5 m/s` 达到 **100 tick/5 s**；出生点附近的等待也保留。路口距离取 evaluator 路线 XML 中的转向稀疏节点或 scenario trigger 的最近距离，**不是** CARLA 地图的 `is_junction`；分支距离用同一 XML 路线节点的分段线性近似，不等于未记录的 dense global plan。因此 D3a 不会单凭这项几何近似裁定 R1/R2。

## 全量结果

[`all-deviations.csv`](results/diagnosis/d3/all-deviations.csv) 有 **12** 次官方 route deviation，全部是保留集 **2084 和 27529** 的 C/D，各路线×各 seed×两臂各一次；A/B 均无。2084/C/seed 1 的官方终态在 **176.85 s**，其余 11 次约 **22.90–27.20 s**；该迟发个案在偏离终态前有长时间停滞，不能把终态时间误作首次错过路口时间。

[`all-stalls.csv`](results/diagnosis/d3/all-stalls.csv) 有 **115** 段 ≥5 s 停滞，出现在 10 条路线，分臂 A/B/C/D 为 **18/39/33/25** 段。因此“停住 ≥5 s”并不只发生在 2091；需区分正常停车、交通阻塞与持续到官方 tick 上限的失败。

| 路线 | ≥5 s 段 | ≥30 s 段 |
|---:|---:|---:|
| 2084 | 9 | 2 |
| 2091 | 27 | 16 |
| 3255 | 15 | 2 |
| 3514 | 6 | 4 |
| 17569 | 3 | 2 |
| 25424 | 12 | 7 |
| 26405 | 2 | 2 |
| 27494 | 14 | 3 |
| 27529 | 18 | 0 |
| 28154 | 9 | 0 |

每段的起止 step/time、地图坐标、XML 近路口距离、是否已开始运动及中位速度均保存在 CSV。2091 的超时分布也**跨臂/seed**：seed 0 的 C/D、seed 1 的 C、seed 2 的 B 是官方 TickRuntime；同 seed 的其他臂有时能完成。因此不能把 2091 整条路线标成必然无法通行。

## 2084、27529：事件前时间线

[`route-timeline.csv`](results/diagnosis/d3/route-timeline.csv) 逐 tick 保留车速、真值坐标、到 XML 转向/trigger 距离、实际 steer、C shadow steer 是否到 .8 限幅、TFv6 route 与 waypoint 的世界坐标，以及各自到 XML 正/直行分支的距离。包含每个出事 C/D 的路口接近窗口、官方终态前 10 s 和同 seed B 的路口接近窗口。[`route-first-wrong.csv`](results/diagnosis/d3/route-first-wrong.csv) 保存以“终点到直行支路比到 XML 正支路至少近 2 m”为门槛的首次 tick，但列名中的 `wrong` **只表示此 XML 近似指标**。

| route / seed | C 官方偏离 step | D 官方偏离 step | C 路口接近窗口首次 waypoint 倾向直行 step | B 同指标 step | D 同指标 step |
|---|---:|---:|---:|---:|---:|
| 2084 / 0 | 512 | 516 | 197 | 319 | 197 |
| 2084 / 1 | 3536 | 522 | 201 | 229 | 196 |
| 2084 / 2 | 473 | 479 | 198 | 229 | 201 |
| 27529 / 0 | 474 | 465 | 310 | 415 | 310 |
| 27529 / 1 | 459 | 457 | 310 | 289 | 305 |
| 27529 / 2 | 520 | 543 | 320 | 418 | 322 |

例如 27529/0/C 在 step 300 位于 `(273.3,−249.8)`、速度约 **3.26 m/s**；到 step 310 位于 `(271.4,−249.8)`、约 **3.74 m/s**，而 B 在 step 310 仍在 `(271.6,−249.8)`、仅 **0.42 m/s**。C 的 waypoint 末点从近似“正/直行分支”距离 **0.5/0.0 m** 变为 **2.3/0.1 m**。但 B 在相同 XML 转向点附近也会短时满足同一指标，表明 XML 中从直线到右转支路的约 19 m 稀疏跳段不够精细，不能把表中首次 tick 当成真实 planner 首次错误 tick。W2/W2b 日志没有模型输入的 target point、command、Kalman 状态或 RoutePlanner pop 事件；R1 与 R2 仍待 D3b 补录。已记录的 C steer 原始值与限幅标记可在相同时间线核查 R3，但规划分支尚未可靠重建，暂不裁决。

## 2091：停滞与同输入开环重算

[`2091-stall-timeline.csv`](results/diagnosis/d3/2091-stall-timeline.csv) 对 2091 所有 A/B/C/D seed 的每个 ≥5 s 段逐 tick 给出 TFv6 首段、作者 B 定义 `2‖wp1−wp3‖`、8 点平均速度、模型 target speed、C reason、B shadow throttle/brake、实际 control 与后处理是否改写。A/B 的后轴几何也用 W2b 规则离线重算，避免旧 phantom。最早起步停滞段中，C 三个 seed 分别停 **41.10 / 47.85 / 12.50 s**，D 为 **12.35 / 46.05 / 12.30 s**；C 的首段速度中位数约 **0.12 / 0.14 / 0.14 m/s**，8 点平均约 **2.12 / 2.21 / 2.38 m/s**。同段 B shadow 给出 >.05 throttle 的 tick 比例约 **85.9% / 85.3% / 80.8%**，C 的 `stop_hold` 比例约 **38.6% / 32.8% / 37.2%**；A/B/C/D 已记录的起步段后处理均没有改写 throttle/brake。这直接支持 S1 的**控制律输入不一致**与其导致的局部刹车保持，反对把起步段主要解释为 S3 后处理互锁；S2 真实让行还要看 D3b 的灯与 actor 记录。

[`2091-openloop.csv`](results/diagnosis/d3/2091-openloop.csv) 在每个静止段固定**同一条已记录速度序列**，用冻结 C 的 `ConditionalPI(kp=.5,ki=.25)`、`stop_hold` 门槛，分别喂首段、作者 B 定义、8 点平均期望速度。首段分支与新 C 原始 throttle/brake 在最早段的 ±0.05 内相符 **773/822、930/957、248/250 tick**；偏差来自段前 PI 状态、原控制器几何/时间保护等未在局部重算中重建的状态。把期望速度换成作者定义时，三个 seed 原本最早停滞段的“给油 >.05 且不刹车”tick 分别为 **792/822、937/957、233/250**；首段定义仅 **461/822、546/957、147/250**，8 点平均为 **821/822、955/957、247/250**。这证明同一输入下起步律对期望速度定义敏感，**不证明**替代定义在闭环中能完成路线，也没有把替代控制写入驾驶。

2091 后续停滞不是同一故事的简单重复：例如 seed 0/C 在 step 3145–3999 的 42.75 s 段，作者定义在约全部 tick 请求油门，而首段分支 0/855 tick；但 seed 1/C 在 step 1325–2441 的 55.85 s 段，作者定义只在 19/1117 tick 请求油门，连替代首段目标也不能解释为必然起步。模型与交通状态在闭环中已改变，D3b 需要用灯、actor、target point 和因子实验区分 S1/S2/S3。

## D3b 可行性与墙钟预算

九个指定 (route,seed) 的 W2b/W2 B、C 历史单 case 墙钟逐一估算：B+C 各按实测，E/F 各按同组 B/C 较慢的一臂，合计 **13,019 worker-s**。两台 CARLA 并行的纯驾驶下界约 **1.81 h**，加 20% 启停/调度余量约 **2.17 h**；六个 route-deviation 组若都需 Kalman 隔离，按 C 实测另加约 **0.20 h**。渲染与诊断开销另外计入，计划总量约 **2.5–3.0 h**，未超过 **3.5 h** 删除阈值。补录和混合臂可执行，D3b 按任务自动开始。最终估时与实际偏差将在总报告列出。
