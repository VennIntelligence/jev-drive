# openpilot 真实横向控制路径代替计划跟踪：HUGSIM 起步打转（预登记）

2026-10-05 写，在这个臂的任何闭环运行之前（离线回放与离线 c 已做，见下）。接第 113、114、115（含用户实车经验：低速起步横向接管、没有打转）、117 条。
用户的洞察：openpilot 在真车上不是跟踪计划的位置，而是用模型的 desired curvature 经 controlsd 送到车上；计划看起来会跳，车却开得平顺。
我们的两个闭环（HUGSIM PR#57 iLQR、B2D `drive` 的横向 pursuit）跟踪的是计划位置。假设：换成 openpilot 的真实横向路径，起步打转消失，且不要第 113 条低通的 HD 代价。

## 1. 源码核实的路径（openpilot master ec95db3f，2026-10-02；opendbc 35f7e081；本地 tmp/opsrc，不入库）

| 环节 | 源码 | 常数 / 行为 |
|---|---|---|
| modeld 输出 | `selfdrive/modeld/modeld.py` `get_action_from_model` | 有 `action` 头的模型（Cinque 有，slice 2062–2066）：desired_curvature = action[0] / max(1, v)²；没有才用 `get_curvature_from_plan`（plan 在 action_t 的 yaw）。`LAT_SMOOTH_SECONDS = 0.0`（不平滑）；v ≤ `MIN_LAT_CONTROL_SPEED = 0.3` 时保持上一次的曲率 |
| action_t | 同上 | lat_action_t = lateralDelay + `LAT_SMOOTH_SECONDS` + DT_MDL（帧延迟 0.05）+ DT_MDL/2（0.025）；考试喂的是 (0.275, 0.525)，即 lateralDelay = 0.2 s |
| lateralDelay | `selfdrive/locationd/lagd.py` | 初值 steerActuatorDelay + 0.2，在线估计夹在 [0.15, 0.65] s；估的是「期望横向加速度 → 实际横向加速度」的纯延迟 |
| controlsd | `selfdrive/controls/controlsd.py` | 100 Hz；latActive 要求 v > max(minSteerSpeed, 0.3)（或 steerAtStandstill）；不 active 时期望曲率 = 当前实测曲率 |
| 曲率限制 | `selfdrive/controls/lib/drive_helpers.py` `clip_curvature` | 每 10 ms 变化 ≤ MAX_LATERAL_JERK 5.0 / max(v, 1)² × 0.01，横向加速度 ≤ 3.0 m/s²，|κ| ≤ MAX_CURVATURE 0.2 |
| 横向控制器 | `latcontrol_torque.py` | KP 按速度插值 [1, 1.5, 2, 3, 5, 7.5, 10, 15, 30] m/s → [250, 120, 65, 30, 11.5, 5.5, 3.5, 2.0, 0.8]，KI 0.15；setpoint 取 lat_delay 之前的请求；jerk 前馈 LP 1.2 Hz、前视 0.19 s、增益 0.3 |
| 代表车 | opendbc `toyota/interface.py`、`values.py` | RAV4 TSS2：扭矩控制，steerActuatorDelay 0.12（lagd 初值 0.32 s），steerRatio 14.3，wheelbase 2.690 m，minSteerSpeed 0，steerAtStandstill 否；LTA（角度控制）车的方向盘角速率上限 ≤ 5 m/s 时 0.3°/10 ms 增、0.36°/10 ms 减 |

**对第 113 条第 2 点的更正**：曲率速率上限 5 / max(v, 1)² 在 0.5–3 m/s 不起作用（源码确认；每 0.25 s 步允许 1.25 / max(v,1)² 1/m）。但「模型期望曲率的等效 c 约 0.2–0.5」是按 **plan 推曲率**（`get_curvature_from_plan`）算的；Cinque 在车上跑的是 **action 头**，它对计划 1 s 方向几乎不响应（下面第 3 节）。

## 2. 臂 `opctrl`（唯一主臂，参数事先定，不调）

- 模型调用与考试完全相同（action_t (0.275, 0.525)，时钟放大 1.25，速度 1.25 v）；server 已有的 `curvature`（action[0] / max(1, 1.25 v)²）作为计划最后一行带给仿真器。
- 仿真器侧 `lib/op_ctrl.py`（tree `opctrl` = fixed + `patches/hugsim/optional/op-ctrl.patch`，env `OP_CTRL='{}'`）：modeld 的保持规则 → controlsd 100 Hz 的 latActive 与 `clip_curvature`（逐字移植）→ 车：实现曲率 = 期望曲率延迟 lateralDelay。**车辆响应模型的选择**：用 openpilot 自己对闭合转向回路的模型（lagd 辨识的纯延迟，latcontrol_torque 的 setpoint 也按这个延迟取）；扭矩到转角的 EPS 动力学 openpilot 不建模、我们也没有数据，所以不加。延迟 = 0.25 仿真秒（= action_t 里的 lateralDelay 0.2 模型秒 × 1.25 时钟放大），与模型被告知的延迟自洽；在 4 Hz 仿真器上等于「本步的期望曲率在下一步实现」。代表车 RAV4 TSS2 的 minSteerSpeed 0、不在静止时转向。LTA 角速率上限只做离线敏感度，不进闭环。
- HUGSIM 的转向状态每步设为 atan(L · 本步平均实现曲率)（L = 2.7），所以仿真器一步的航向变化 = v · κ̄ · 0.25 s；**纵向不动**（仍是 PR#57 iLQR 的加速度）：openpilot master 的 LongControl 只有 pid / stopping 两态，没有专门的起步逻辑；第 117 条已证起步纵向不是杠杆。
- 计分用的 planned_traj 是原计划（曲率行在 closed_loop 里剥掉）。

## 3. 离线（已做，决定要不要上闭环之前的读数）

考试基线 64 个运行的前 40 步（10 s），CPU onnxruntime 按考试同样的喂法回放 video.mp4 帧（`experiments/hugsim/scripts/opctrl_replay.py`，2 121 步；
2.5 s 横向复现误差中位 0.007 m、p90 0.08 m），每步记 action 头曲率、plan 推曲率和计划 1 s 方向 φ1；`opctrl_offline.py` 读数（`results/op_control_stack/offline.json`）。
c 的定义与 lowspeed_ctrl.md 相同：下一 0.25 s 步实现的航向变化 / 度 φ1，过原点回归，|φ1| < 15°，场景聚类 bootstrap。openpilot 路径 = `OpLateral` 作用在回放的 action 曲率与日志速度上（开环，同一输入）。

| v (m/s) | PR#57 iLQR（日志） | openpilot 路径 c | 两步 c2：iLQR / openpilot | 模型侧：action 曲率的即时航向 / 度 φ1 | 对照：plan 推曲率 |
|---|---|---|---|---|---|
| < 1 | 0.055 [0.049, 0.065] | −0.003 [−0.008, 0.002] | 0.131 / −0.003 | −0.002 [−0.006, 0.003] | 0.027 [−0.001, 0.050] |
| 1–2 | 0.145 [0.130, 0.158] | 0.026 [0.014, 0.038] | 0.311 / 0.055 | 0.024 [0.013, 0.036] | 0.082 [0.054, 0.103] |
| 2–3 | 0.247 [0.228, 0.274] | 0.003 [−0.077, 0.051] | 0.521 / −0.010 | −0.019 [−0.096, 0.033] | −0.058 [−0.246, 0.065] |
| < 3 合并 | 0.127 [0.106, 0.156] | 0.013 [0.001, 0.024] | 0.277 / 0.027 | 0.010 [−0.003, 0.021] | 0.047 [0.009, 0.075] |

- iLQR 这一列复现了第 113 条（0.066 / 0.146 / 0.307，那里用全部步）。openpilot 路径的 c 在 < 1 与 2–3 m/s 约为 0，1–2 m/s 为 0.026，合并 0.013：落在第 111 条临界值 0.018–0.034 附近或以下，比 iLQR 小一个量级。
- 原因在模型侧：action 头的期望曲率与计划 1 s 方向在低速几乎无关（同号比例 0.56；与 plan 推曲率相关 0.66–0.74，但幅度约为其 1/2–1/5）。所以第 113 条「等效 c 0.2–0.5」只对 plan 推曲率成立，不对车上真正用的 action 头成立。
- 环路增长（第 111 条窗口核，openpilot 路径加一步延迟）：G = 5.85 / 9.34 时 iLQR 在 1–2 m/s 为 1.82 / 2.34、2–3 m/s 为 2.44 / 3.31；openpilot 路径 1–2 m/s 为 0.96 / 1.07（transfer 取 CI 上界 1.05 / 1.18），< 1 与 2–3 m/s 的点估计不放大（CI 上界 ≤ 1.15）。
- 符号：HUGSIM 转向角与下一步航向变化同号 0.98；action 曲率与 plan 推曲率正相关，所以 openpilot 曲率（+ 右）与 HUGSIM 转向（+ 右）同号。
- 限定：开环（回放的是 iLQR 跑出来的帧，闭环里 φ1 会不同）；回放用 mp4 帧；2–3 m/s 档样本少、CI 宽。

**离线预测**：打转应大幅减少；风险在 HD（欠转）。

## 4. HUGSIM 闭环（全部 64 个场景，原生 Cinque，单次）

- 小烟测（不计分）：scene-0013-medium-00（基线打转）、scene-3000_3200-medium-00（第 113 条丢分最多）；检查曲率行被剥掉、sim.log 有 `op_ctrl` 行、不崩。
- 对照：考试基线 `cinque-fixed`；同日基线重跑 `cinque-fixed-base3`（tree fixed，同一 server、同一批）；另报第 113 / 114 条的两次重跑。
- 打转、按起步事件、HD 配对差、崩溃处理的定义与 `lowspeed_report.py` / `launch_long_report.py` 完全相同（脚本 `experiments/hugsim/scripts/opctrl_report.py`）。
- **线（两条都要过）**：(i) 64 个里打转 ≤ 2；(ii) 基线不打转的 54 个场景 HD 配对差（场景 bootstrap 95% CI）下界 > −0.02（对考试基线）。另报对各次重跑的配对差、RC、碰撞、max_steps、本臂闭环的 c（同一读数）。
- 事先写下的风险：action 头的曲率比 plan 推的小（离线中位约 1/2–1/5），正常转弯可能欠转（出路线 / 撞路沿），这会体现在线 (ii)。

## 5. B2D（`drive` 臂，19 路线 × seed 2、3）

只在第 4 节两条线都过时跑。钩子（把 `drive` 臂的横向 pursuit 换成同一路径：Cinque 的 action 曲率 → `lib/op_ctrl.py` → CARLA 转向，按 CARLA 车的转向曲线换算）在 HUGSIM 结果出来后实现，**先把钩子设计写进本文件末尾并提交，再烟测，再全量**。对照 shipped `drive` 臂（vmerge2 的 seed 2、3）。线：DS 配对差（路线聚类 bootstrap）点估计 ≥ −2，且 CI 上界 > 0。

## 判读

- HUGSIM 两条线过、B2D 过：两个闭环榜都改用 openpilot 的真实横向路径。这不是逐榜 trick：它就是 openpilot 在车上的控制方式，标注「用 openpilot 自己的控制栈代替基准的计划跟踪器」。
- HUGSIM 过、B2D 不过：HUGSIM 单列。
- 打转过、HD 不过：看丢分是否来自欠转（转弯场景出路线 / 撞路沿），这说明 action 头在仿真器里欠响应，不再加臂，报告。
- 打转不过：action 头的离线 c 在闭环里不成立（闭环 c 另报），报告机制。
- 不调参数、不追加臂；要加先在本文件追加并提交。
