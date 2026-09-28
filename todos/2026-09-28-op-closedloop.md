# openpilot 进 Bench2Drive 闭环：base 路线跟随器 + openpilot modifier（op-arb）

状态: running（计划写于 2026-09-28 15:40 CST，任何 op-arb 闭环数字之前）
主题: [research/openpilot-closedloop-integration.md](../research/openpilot-closedloop-integration.md)（结果与设计讨论）
相关: [decisions.md](../research/decisions.md) 第 31、33、38、41、49、51 条；[openpilot 迁移](2026-09-24-zeroshot-exam/openpilot-migration.md) D1–D5；
[night-queue-3](2026-09-26-night-queue-3.md) CL 节与 [OPL] 条目
代码: `scripts/op_arb_agent.py`（agent，各仲裁模式）、`scripts/op_arb_server.py`（openpilot server，多返回 lead / meta / desire 头和无 desire 的 twin）、
`scripts/op_arb.sh`（启动）、`jevdrive/op_arb_report.py`（读数）
box run dir: `$DATA_DIR/runs/op_arb/`

## 目标

用户提议：一个很小的 base 只管起步和按路线过弯，openpilot 作为 modifier 叠在上面，需要让行、减速、停车、绕行时由 openpilot 接管或混入。
这里先诊断 openpilot 在闭环里「起不了步」「不转弯」的原因（带逐 tick 数字），再在少量 dev 路线上比较 2–3 种仲裁方式，并做 base-only 消融：
openpilot 必须证明自己加了东西，否则分数是 base 挣的。论文立场（trick or trade）要求 openpilot 外面加的东西通用、小。

## Setup（所有臂相同，只有交给 P7 的轨迹不同）

- 模型：openpilot Cinque（op-adapt 与 WL 两条线改的就是它），20 Hz 每 tick 一步，原生 road + wide rig 每 tick 渲染、同帧配对，5 s 预热刹车保持，
  后轴变换，路线 desire（路口前 20 m turnLeft / Right，变道段 laneChange），即 night-queue-3 CL2 的 openpilot 路径原样；P7 控制器（`P7.json`，5 Hz 交 plan）。
- base：路线几何（Bench2Drive 发给每个 agent 的 dense route，`RouteAdapter` 的 rejoin path，只用传感器位姿）+ 速度 governor
  （设定速度 8 m/s、路径曲率上的横向加速度 ≤ 2 m/s²、加速度 ≤ 1.5 m/s²、路线终点停车）。base 没有任何感知。
- 臂（`op_arb_agent.py` 的 mode）：

| 臂 | 横向 | 纵向 | 对应 openpilot 真车用法 |
|:--|:--|:--|:--|
| native | openpilot plan | openpilot plan | CL2（第 33 条的「openpilot 自己开」） |
| base | base | base | openpilot 关掉（消融） |
| oshadow（只诊断，不作候选） | base | base + 真值 governor（路径内前车 / 行人 IDM，路线上红黄灯停） | — |
| acc | base | min(base, openpilot lead 头上的 IDM) | chill 模式（ACC + 驾驶员管路口） |
| e2e | base | acc + 行驶中 openpilot plan 作速度约束，plan 造成的停车锁存到 openpilot 放行 | experimental 模式（端到端纵向） |
| switch | 行驶中、不在路口转弯区时 openpilot plan；否则 base | 同左；起步与转弯区用 e2e | D3（TCP 伙伴）换成 base |

- IDM 参数（写死）：s0 2.5 m、T 1.2 s、a 1.5 m/s²、b 2.0 m/s²；lead 头用第 0 个 lead、lead_prob > 0.5，x 从相机量起，减去 2.06 m 到前保险杠。
- 路口转弯区：路线 LEFT / RIGHT 命令段前 15 m 到后 5 m（与 D3 相同）。

## 路线（写死，`scripts/op_arb.sh routes`）

全部取自 `bench2drive_0.0.4_val.xml`（Bench2Drive 0.0.4 的 val 路线，与 220 考卷只重 2 条，所以不是 G / K 的考卷路线）：
每个 scenario 类型取 XML 顺序里第一条不在 Town12/13 的路线（没有就取第一条）。
- phase 1（诊断，6 条）：SignalizedJunctionLeftTurn、VanillaSignalizedTurnEncounterRedLight、HardBreakRoute、NonSignalizedJunctionLeftTurn、ParkingExit、DynamicObjectCrossing。
- phase 2（评测，10 条，与 phase 1 不重）：SignalizedJunctionRightTurn、VanillaSignalizedTurnEncounterGreenLight、T_Junction、VanillaNonSignalizedTurn、StaticCutIn、
  MergerIntoSlowTrafficV2、ConstructionObstacle、VehicleTurningRoutePedestrian、OppositeVehicleTakingPriority、SignalizedJunctionLeftTurnEnterFlow。
- TM seed 0，单次。

## 步骤

1. phase 1：native、oshadow 两臂 × 6 条（server 上开 twin：同帧再步进一个 desire = 0 的 session）。读数（`jevdrive/op_arb_report.py diag`）：
   - 起步：静止 tick 按真值分三类（前方 30 m 内无车无人且无红黄灯 / 路径内前车 ≤ 15 m / 红黄灯 ≤ 30 m），报 plan 速度（1、3、5 s）、plan 5 s 位移、action accel、
     lead_prob 与 lead x、meta 的 gas_press，以及各信号区分「该走」与「该等」的 AUC。
   - 行驶中：自由路段 plan 速度相对当前速度的偏差（有没有「一直推迟」的低速偏置）；接近真值 hazard（前车 TTC、红灯、行人）时 plan 减速与 lead 头的 AUC。
   - 转弯：路线 LEFT / RIGHT 段上，openpilot plan（有 desire）和 twin（无 desire）在 1、2、3、5 s 处相对路线的横向偏差；native 臂的路口通过率。
2. 根据 phase 1 定 e2e 的放行信号（release：gas / planx / none）和阈值，**在 phase 2 任何数字之前写进本文执行日志**。
3. phase 2：base、acc、e2e、switch 四臂 × 10 条。读数：DS、RC、SR、碰撞、闯灯、blocked，逐臂对 base 的配对差（10 条，只作方向），openpilot 约束起作用的 tick 比例与距离比例，
   按真值 hazard 分的违规。
4. 写 research/openpilot-closedloop-integration.md。

## 判据（写在任何数字之前）

- 「openpilot 加了东西」= 某个候选臂对 base 的配对 DS 差为正，且碰撞 + 闯灯数少于 base，并且这些改进发生在 openpilot 约束起作用的 tick 附近（违规按时刻归因）。
  10 条路线、单 seed，只能给方向，不给 CI 结论。
- 「openpilot 是 modifier 不是装饰」= 候选臂里 openpilot 约束（lead / plan）是 binding 的 tick 占行驶 tick 的比例 ≥ 5%，且去掉它（= base）结果变差。
- trick 风险按「为 B2D 加了什么 openpilot 真车部署里没有的东西」逐项列：base 的路线跟随（真车上是驾驶员 / 导航）、设定速度（驾驶员）、起步放行（驾驶员按 resume）。

## 资源与预算

- CARLA 只在 GPU 6（测试卡，与 G 的 pilot 共卡，按每卡 ≤ 6 个 server 数实际个数），本 lane 2 个 server（index 160–161），核 144–167；SCH 行 `op-arb`。
- 预算 ≤ 20 worker·h；估计 phase 1 12 次路线运行、phase 2 40 次，每次约 6–10 min，共约 6–9 worker·h。
- 停进程只按记录的 PID。

## 执行日志
