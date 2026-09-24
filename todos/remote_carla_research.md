# CARLA 闭环控制器研究线：交接任务书（探索类）

状态: open，2026-09-23 写。读者：在 Mac 上另开一个会话、负责这条研究线并指挥 Tokyo box 本地 agent 的主持者。
这条线和论文主线（reaction decoder、Qwen 视频特征、P4/P5，见 `research/prediag-2026-09/README.md`）关系不大，所以从主线会话里拆出来。机械性的准备工作（驱动、运维规则、TFv6 安装和官方复现）写在 [remote_carla.md](remote_carla.md)，由 Tokyo 的 agent 直接执行，本线以它的产出为起点。

## 这条线要回答的问题

**轨迹控制器的好坏，对一个强 planner 的闭环驾驶表现有没有可测的影响？** 分三层：

1. 在 route oracle（无模型、无交通）下，控制器能否把转弯的跟踪误差降到有意义的水平，并且在 held-out 弯上成立。
2. 换成真实 learned planner（TFv6，B2D DS 约 95）的轨迹，我们的控制器相对 planner 自带的 PID，在 DS、完成率、违规、舒适性上有没有差别。
3. TCP（弱 planner，DS 约 60）上同样的问题，而且要去掉 target-point 仲裁的混淆。

## 已有的事实（先读，不要重新推导）

| 文档 | 内容 |
|---|---|
| `todos/2026-09-22-b2d-controller/`、`todos/2026-09-23-tcp-controller/`、`todos/2026-09-23-lateral-followup/` | Tokyo 通宵三轮：控制器实现、Dev10 正式对照、真实 TCP 纵向对照、短前视/Hermite/后轴传播三个横向候选，全部未通过默认替代验收 |
| [lateral-physics.md](2026-09-23-controller-next/lateral-physics.md) | 经验项 k·v²·ω 就是 PhysX 线性轮胎的后轴侧偏，k(v)=(1+1/v)/(c·g)，c≈11/rad；body-heading 门槛会结构性地惩罚更好的后轴跟踪（β_r P95≈4.9°）；加 k 后急右弯误差主要来自控制器几何（未建模 Ackermann，曲率只有假设的 0.886；pursuit 用车头方向而不是后轴速度方向）；执行器不饱和 |
| [lateral-v2-protocol.md](2026-09-23-controller-next/lateral-v2-protocol.md) 与 `lateral-v2-report.md` | slip-frame pursuit + Ackermann 反解的预注册闭环检验：6 arm × 11 路线（4 开发窗 + 8 个按几何预选的 held-out 弯）× 11 组配对扰动，含 truth-pose ceiling。主线会话正在 GPU box 上跑完它，报告落地后这条线由本会话接管 |
| [tcp-trajectory-contract.md](2026-09-23-controller-next/tcp-trajectory-contract.md) | TCP 4 点 / 0.5 s / y 向右，原点是 GNSS 点（≈rear axle，差 1 cm，已用 B2D 真实标注核实）；**paired-v2 的"原生横向"96% 的 tick 在追 target point，不是网络轨迹** |
| [planner-scout.md](2026-09-23-controller-next/planner-scout.md) | 选定 TFv6 `tfv6_resnet34`；waypoint 原点是 actor origin（rear axle 在其后 1.389 m），y 向右，8 点 +0.25…+2 s；官方 95 DS 用的是 route + target speed + 自带 PID，waypoint 默认不参与控制 |
| [gpubox-port.md](2026-09-23-controller-next/gpubox-port.md) | 控制器栈已能在 GPU box headless 运行；OpenBLAS kernel 差异导致逐位比对要设 `OPENBLAS_CORETYPE=Barcelona` |

控制器已有的 opt-in（commit `e71e616`）：`pursuit_frame="rear_slip"`、`steer_inverse="ackermann"`、truth-pose ceiling（仅诊断）、`update()` 接受 N 点 + `trajectory_dt`。

## 工作包

### W1：接住 lateral v2 的结论
读 `lateral-v2-report.md`。候选若在 held-out 上成立，把它作为 W2/W3 里"我们的控制器"那一臂；若不成立，W2/W3 用 production 控制器，并把失败模式写清楚。门槛按冻结协议判，不事后改。协议里预先登记的 heading 新指标（course error）和旧 body-heading 指标都要继续并列报告。

### W2：TFv6 + 我们的控制器（核心）
前提：[remote_carla.md](remote_carla.md) 第 5、6 步完成，官方 TFv6 在本地 Dev10 上的分数已知。
- 写 wrapper：TFv6 waypoint → rear-axle 原点（x −1.389 m，外加 yaw 变化的 d·sin Δψ 项）、y 取反、8 点 dt=0.25 s，用 N 点接口直接给控制器，不外推到 5 s。先在录下的 TFv6 输出上离线核对坐标（直行、左转、右转各一段）。
- 臂（先预注册再跑）：
  - A：TFv6 官方（route + target speed + 官方 PID），基线；
  - B：TFv6 waypoint + 官方风格 waypoint PID（隔离"用 waypoint 而不是 route"这个表示变化本身的影响）；
  - C：TFv6 waypoint + 我们的控制器（W1 选出的那一版）。
  B 对 C 才是控制器本身的效应；A 对 B 是表示的效应。
- 带交通的 B2D 本身不确定（同 TM seed 背景车也会变），每臂至少 2–3 个 seed，报配对差和 CI，不拿单次结果下结论。
- 分级：Dev10 → 已有的 6 条 holdout → 只有前两级有正信号才上完整 220 条。每一级的继续条件事先写进协议。
- 指标：DS、RC、违规分项、完成数；控制器对 planner 轨迹的跟踪误差；纵向/横向加速度和 jerk；碰撞前后分开统计。

### W3：TCP 三臂（去掉仲裁）
TCP 环境只在 Tokyo 上（Python 3.8、torch 2.2，3090 上可用）。臂：原生 PID 只吃 waypoint（关掉 target-point 仲裁）/ 我们的控制器吃同样 4 点（N=4，dt=0.5，y 取反，无原点平移）/ 官方原样（仲裁开，只作参照）。比前两臂。优先级低于 W2，TFv6 的结论出来后再决定是否做。

### W4（可选）：route oracle 下的纵向和停车
上一轮纵向 PI 已有结论，除非 W2 暴露出纵向问题，否则不再投入。

## 做法约束

- **先预注册再跑**：每个工作包先写协议（臂、路线、seed、指标、通过条件、继续条件），commit 冻结后才启动第一例。失败照实报告，不追加参数网格、不放宽门槛、不挑有利区间。
- **只用 Tokyo 跑 CARLA 闭环**（RTX 3090 24 GB、32 核）。GPU box 由主线占用（特征抽取、head、VLM），这条线不要往那边派任务；确实需要时先和用户确认。
- 报告节奏按 `CLAUDE.md`：长任务顺利时每 3–5 小时报一次；出错、卡住、需要决策或完成时立即报。
- 结论落地：阶段结论写进本目录的报告和 `research/decisions.md`（按 research/README.md 的规矩，错了就地改）；英文的使用说明更新到 `docs/b2d-controller.md`、`docs/b2d-tcp-controller.md`。

## 交付

每个工作包一份中文报告（表格、带 CI 的配对结果、图按 `research/plot_style.py`），最后一份总结回答开头那个问题：控制器对强 planner 的闭环表现有没有可测影响，多大，靠什么机制。
