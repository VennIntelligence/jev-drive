# openpilot 自己的横向控制路径放到 B2D `drive` 臂（预登记，诊断）

2026-10-04 写，在任何 B2D 闭环运行之前。main 的决定：第 118 条的 HUGSIM「不伤」线没过（非打转 HD −0.009 [−0.058, +0.042]，下界 −0.058 < −0.02；代价来自卡在停车计划），B2D 作为诊断补跑。接第 107、113（第 6 点：均匀低通在 B2D 掉 DS −4.11，B2D 没有起步打转）、118 条。

## 前提核实（读代码和箱子上的 cfg，不是事后）

`drive` 臂在车道保持段**今天已经**用 action 头曲率：`op_arb.sh` 的 `drive` 臂 `lat_exec` 默认 `curv`（箱子 `lsc_b2d/jobs/*/op/cfg/drive.json` 确认），`b2d_zeroshot_agent._curvature_steer` 把 `info["curvature"]`（= action[0] / max(1, v)²）经自行车几何和 CARLA 转向曲线转成 steer，只受 controller 的 steer_rate（2.0 /s）限制。路口 / 命令区（`zone_m`）与「openpilot 路径和路线分歧」时由路线几何 + P7 pursuit 转向，不是 openpilot。所以本臂相对 shipped **只增加** openpilot 在车上的三样东西：modeld 的 v ≤ 0.3 保持、controlsd 的 latActive 与 `clip_curvature`（jerk 5 / max(v,1)²、横向加速度 3 m/s²、|κ| ≤ 0.2）、lateralDelay 0.2 s 纯延迟。路口转弯因此不经过本臂改动的路径（还是路线几何），这是要如实报的限定；「action 曲率够不够转」只能在车道保持段的弯道和 `div` 接管段上读。

## 臂 `opc`

- env `OP_CTRL='{"delay": 0.2}'`；`lib/op_ctrl.OpLateral` 以 CARLA 的 tick（20 Hz，DELTA 0.05 s）步进，内部 100 Hz 做 clip_curvature，延迟以秒计（20 个 100 Hz 拍 = 0.2 s；CARLA 无时钟放大，所以不乘 1.25）。
- 曲率 → steer 的几何**沿用 shipped**（`_curvature_steer`）：angle = atan(wheelbase · κ)，wheelbase 与 max_steer_deg 取自 P7 配置，除以 max_steer · CARLA 转向曲线的速度缩放，再受 steer_rate 限制。要记一笔：P7 若 `steer_inverse=ackermann`，shipped 的这条路径本来就用名义角，未改。
- 非 openpilot 所有（路线 / 区 / 分歧）的 tick 上 OpLateral 跟随当前 steer 对应的曲率（controlsd「不 active 时期望曲率 = 当前曲率」），交还时从轮子现位置出发。
- 纵向、`drive` 的所有仲裁和路径不变（env 之外与 shipped 同代码）。tick 日志新增 `opc` 记录（命令曲率、保持后曲率、clip 后曲率、实现曲率）。
- 19 路线 × seed 2/3，一张卡，对照 vmerge2 的 `v2-drive-s2/s3`（同代码，env 惰性）。先 smoke 一条路线（17280）。

## 判据（先定）

- **过线**：DS 配对差点估计 ≥ −2 **且** 路线聚类 bootstrap 95% CI 上界 > 0。
- 不过线：不采用；原样报告。不调参、不加臂。
- 其余只报告：RC、碰撞（vehicle / layout）、红灯、逐路线变化、实现的低速 c（同 lowspeed_ctrl 的读数）、车道保持段弯道上 action 曲率与实现曲率的比。

## 限定

单次、2 seed；seed 间同路线可差 20–30 DS，所以点估计要求是 −2 而不是严格显著；shipped 与 opc 的差只有 clip + 延迟 + 保持，预期效应小，CI 会宽。
