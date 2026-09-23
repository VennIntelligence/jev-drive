# W2 协议：TFv6 轨迹 + 我们的控制器，闭环对照

状态：**冻结**，2026-09-23。第 1 阶段（[implementation.md](implementation.md)）已由 Mac 侧审过，冻结时做的两处澄清见文末「冻结记录」；判定规则、臂、路线、seed、指标都没有因第 1 阶段的数字而改动。此后本页不再修改，偏离写进报告。

上游：[remote_carla_research.md](../remote_carla_research.md) 的 W2。W1 的结论（lateral v2 候选按冻结门槛不通过，见 [lateral-v2-report.md](../2026-09-23-controller-next/lateral-v2-report.md) 和 `research/decisions.md` 第 30 条）决定了主臂用 **production 控制器**。

## 问题

TFv6（`tfv6_resnet34`，三 seed ensemble，B2D DS 约 95 的强 planner）输出的 8 点 waypoint，交给我们的控制器去跟，和交给 TFv6 自带的 waypoint PID 去跟，闭环驾驶表现有没有可测的差别。

## 臂

四个臂共用同一个模型、同一份 checkpoint、同一套传感器和同一套作者启发式（stop sign、creeping、紧急制动等），**唯一的差别是"模型输出 → (steer, throttle, brake)"这一步**。

| 臂 | 横向 | 纵向 | 角色 |
|---|---|---|---|
| A | 官方：route + lateral PID（`steer_modality=route`） | 官方：target speed + PID（`throttle/brake_modality=target_speed`） | 官方基线，作者原样 |
| B | 官方 waypoint PID（`steer_modality=waypoint`） | 官方 waypoint PID（`throttle/brake_modality=waypoint`） | 表示对照：同样吃 waypoint、用作者的 PID |
| C | 我们的 production 控制器吃 8 点 waypoint | 同一控制器的纵向 | **主臂** |
| D | C + `pursuit_frame="rear_slip"` + `steer_inverse="ackermann"` | 同 C | 探索臂，不参与判定 |

- **主效应 = C − B**（控制器本身）。A − B 是表示（route+speed 对 waypoint）的效应，只描述。D − C 只描述，用来看 W1 的候选在真实 planner 下是否值得另开阶段。
- 作者启发式如果是在 PID 之后覆盖 control（例如 creeping 改 throttle、紧急制动改 brake），C、D 臂在我们的控制器之后按完全相同的规则覆盖；如果某个启发式本身依赖 route 或 target speed 模态，照原样在四个臂里都执行，并在实现说明里逐条列出。任何一条启发式做不到四臂一致，停下来报告。
- 控制器配置：C 用当前 `main` 上 `scripts/b2d_controller.py` 的默认值，D 只加上面两个 opt-in。truth-pose ceiling 一律关闭。TFv6 的 waypoint 每 tick 重新规划、本身就是车体局部轨迹，所以 C、D 臂的控制不用世界位姿：`Controller` 在两次规划之间只用 SPEED 和 gyro 做短时传播。lateral v2 的 GNSS/IMU + fixed-k PoseFilter 照样运行并逐帧记录（TFv6 的 GNSS 在 actor origin，`gnss_x_m=0`，后轴 x=−1.389 m），只作诊断，不进控制。

## 轨迹接口（C、D）

TFv6 `pred_future_waypoints`：8 点，+0.25…+2.0 s，原点是 actor origin（后轴前 1.389 m），x 向前，**y 向右**。三 seed ensemble 取作者 agent 里实际使用的那个融合结果，不自己另做平均。

我们的控制器：原点 rear axle，x 向前，**y 向左**，`update(traj_xy, t, trajectory_dt=0.25)`，N=8，**不外推**到 5 s。

变换：设 p_i 是第 i 点（已把 y 取反），t_i 是该点处轨迹的单位切向（中心差分，首点用 p_0 − 0 的方向，末点用后向差分；相邻点距离 < 5 cm 时沿用上一个切向，全部重合时取 x 轴），L=1.389 m，则

  r_i = p_i − L·t_i + (L, 0)

即未来的后轴位置，放在当前后轴坐标系里。直行时 r_i = p_i。

## 第 1 阶段：实现与核对（冻结之前）

1. **wrapper**：新增 `scripts/b2d_tfv6_controller_agent.py`（或在现有 `b2d_tfv6_visual_agent.py` 的基础上拆出一个无 GUI 的 agent），参数 `--arm {A,B,C,D}`；A、B 不经过我们的代码，直接设作者的模态参数。每 tick 记录：模型输出（route、waypoint、target speed）、变换后的 r_i、四个臂各自的最终 control、真值 ego pose（transform、速度、角速度）、启发式是否触发。逐帧文件放 `/data`。
2. **坐标离线核对**：用 B 臂在 Dev10 里挑一段直行、一段左转、一段右转（在同一条或几条路线上，TM seed 0），录下来。对每个 tick 把 r_i 放回世界系，与真值后轴在 t + 0.25·i 的实际位置比较，报每个 i 的纵向/横向误差中位数和 P95。再对三个错误版本（不取反 y、不做原点平移、平移符号反了）算同样的数字，正确版本在左右转上都必须明显更小，且横向误差没有随转向方向翻号的系统偏差。输出一张图（按 `research/plot_style.py`）和一张表。
3. **单元测试**：变换函数（直行恒等、纯旋转、切向退化）；四臂 control 路径的选择；启发式覆盖在 C/D 臂中的顺序。控制器、TCP、pose 原有测试全部继续通过（`OPENBLAS_CORETYPE=Barcelona`）。
4. **smoke**：四个臂各在 Dev10 的一条路线上跑完一遍，确认不崩、日志齐全。smoke 不计入结果。
5. **墙钟和并发**：TFv6 是第三方代码，照原样跑，不改。只测：单个 CARLA + agent 的 tick 时间、显存峰值、CPU；在 3090（24 GB）上同时跑 1、2、3 个 CARLA server 各自的吞吐，选总吞吐最高且没有崩溃的并发数。估算下面第 2、3 级的总墙钟，写进实现说明。
6. **nondeterminism 基线**：A 臂 TM seed 0 在 Dev10 上跑两遍（这 20 次属于正式 case，在冻结之后跑，见第 2 阶段），这里只需要确认同一个 case 跑两次日志可以对齐。

并发数实测为 2（3 个 server 在 24 GB 上 CUDA OOM）。第 1 阶段的产出写进本目录 `implementation.md`（英文、中文都可，写实测数字），然后发信号 `NEED_GO`，等 Mac 侧冻结。

## 第 2 阶段：正式运行（冻结之后）

路线与 seed：

| 级 | 路线 | TM seed | 臂 | case 数 |
|---|---|---|---|---:|
| 1 | Dev10（`drivetransformer_bench2drive_dev10.xml`） | 0、1、2 | A、B、C、D | 120 |
| 1r | Dev10 | 0（重跑） | A | 10 |
| 2 | v1 保留集 6 条（`todos/2026-09-22-b2d-controller/results/holdout.xml`：3072、2084、2050、25318、28154、27529） | 0、1、2 | A、B、C、D | 72 |
| 3 | B2D 完整 220 条 | — | — | 需要决策，不自动跑 |

- 1r 用来量化同 seed 重跑的非确定性（DS 的重跑差分布），作为解释配对差时的噪声参照。
- 顺序：按 (路线, seed) 为单位，把四个臂放在同一批里连着跑，避免臂和时间段（GPU 温度、其他负载）混在一起。
- 基础设施失败（CARLA 崩溃、加载失败、超时）同一 case 最多重跑 3 次，全部记录；**驾驶失败不重跑**。某个 (路线, seed) 任何一臂最终没有结果，这一组配对整体剔除并列在报告里。
- 第 1 级跑完、分析完，**自动进入第 2 级**，不需要等 Mac（第 2 级无论第 1 级结果如何都跑）。第 2 级跑完后分析、写报告、发 `DONE`。

## 指标

主指标：每个 (路线, seed) 的 DS（Driving Score）。配对差 C − B，对 30 对（第 1 级）或 18 对（第 2 级）取均值；95% CI 用**按路线的 cluster bootstrap**（重抽路线，路线内的 seed 一起带走），10000 次。A − B、D − C 同样的方法。

次要指标（同样配对报告）：RC（route completion）、完成数、违规分项（碰撞按对象分、闯灯、stop、偏离车道、agent blocked、min speed 等，按 B2D 的分类）每 km 次数、SR（零违规完成）。

控制器层面（只对 B、C、D，在第一次碰撞之前的片段上统计，碰撞后单列）：
- **跟踪误差**：tick t 的计划点 r_i（i = 2、4，即 +0.5 s、+1.0 s）与真值后轴在 t + 0.25·i 的实际位置之差，分纵向、横向，报中位数和 P95；只统计计划速度 > 1 m/s 的 tick。
- 纵向、横向加速度 P95，纵向、横向 jerk P95，steer-rate P95。

## 判定（事先登记）

- **控制器有可测效应**：第 1、2 级合并（16 条路线、48 对）的 C − B 平均 DS 差，95% CI 不含 0。方向两边都算（我们的控制器更差也是"有效应"）。
- **无可测效应**：CI 含 0，同时报 CI 的宽度，说明能排除多大的效应。
- 单独第 1 级或第 2 级的结果只描述，不下结论。
- **第 3 级（220 条）的继续条件**：合并 CI 不含 0，或者 |C − B| 点估计 ≥ 3 DS 且第 1、2 级方向一致。满足时发 `NEED_GO` 请求决策，附上按第 1 级实测墙钟估算的 220 条耗时；不满足时写明不继续。
- 失败照实报告。不追加臂、不追加参数网格、不换指标、不挑路线。

## 交付

本目录：
- `implementation.md`：第 1 阶段实测。
- `results/`：`cases.csv`（每个 case 一行：级、路线、seed、臂、DS、RC、违规分项、完成、墙钟、重跑次数）、`paired.csv`、`tracking.csv`、`summary.json`。逐帧文件留在 `/data/runs/b2d/tfv6-w2/`。
- `figs/`：按 `research/plot_style.py`；至少一张配对 DS 差（每条路线一行，C−B、A−B、D−C 三组点和 CI），一张跟踪误差对比。
- `report.md`：中文，表格加图，每张图后 1–3 句说明看什么。回答"问题"一节，判定按上面登记的规则。

## 冻结记录

2026-09-23，Mac 侧审完第 1 阶段后冻结。执行者提的两个问题的裁定：

1. **位姿契约**：按上面「臂」一节改写后的表述执行。原稿写"位姿来源与 lateral v2 的生产臂一致"，这对 route oracle 成立（那里控制器要在世界系里对一条固定参考线），对每 tick 重规划的局部 waypoint 不适用；实现的做法是对的，改的是协议措辞。
2. **坐标核对**：通过。正确变换在左、右转的 8 个 horizon 上横向误差中位数都最低；+1 s 时为左 .43 / 右 .44 m，三个错误版本是左 .77–1.69、右 .92–2.63 m。有符号中位数在 8 个 horizon 上左右各 3 正 5 负，没有随转向翻号的持续偏置。剩下的 .4 m 量级误差包含模型预测误差和 B 臂 PID 的实际跟踪误差，不归因于坐标变换。
