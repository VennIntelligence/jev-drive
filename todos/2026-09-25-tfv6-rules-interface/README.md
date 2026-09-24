# TFv6 规则 × 控制接口，以及 Bench2Drive 按 hazard family 拆分

状态: running（预登记 2026-09-25 00:30 box 时间提交，写于任何计分 run 之前；此前只跑过 smoke，smoke 的 run 不计入结果）
主题: ../../research/leaderboard-vs-ability.md（7.4 节实验 1 与 6）、../../research/capability-vs-leaderboard.md；决策背景 ../../research/decisions.md 第 31、32、35 条
前提: [P5 v0](../2026-09-24-p5-carla-pairs-v0.md)（配对生成器、`_rng` 重置、藏 hazard actor、天气 null 全部沿用），
[TFv6 controller W2/W2b/D3](../2026-09-23-tfv6-controller/report-w2b.md)（A/B 臂的 config 切换、作者规则的审计沿用；本实验不用我们的 controller）

## 目标

两个实验，都来自第 35 条的「怎么才能定下来」。

**实验 1：TFv6 拿分通道的 2% 反应率是「假的」还是「真的」。** 第 32 条里 TFv6 的 target speed 通道（route + target speed 接口，
第 31 条说它比 waypoint 接口高 14 DS）在 P5 配对考卷上定向翻转率（directional flip rate，expert 真反应的帧里考生输出与 expert 同向、
且超过考生自己天气噪声地板的比例）只有 2.0%，waypoint 通道 39.4%。两种解释：

- **假的（规则掩盖）**：作者的手写规则替网络刹了车，网络自己不需要反应；关掉规则，target speed 通道就会反应（第 35 条写的推翻证据：
  「关掉规则后 target speed 通道的翻转率显著上升」）。
- **真的**：网络的 target speed 通道对突发 hazard 确实不敏感，高分来自接口本身（route 横向 + 近二值的速度档位）。

还有第三种可能要一起检验：**P5 的 2% 是读数的问题**——target speed 近乎二值，天气 null 上就整档跳，噪声地板 τ = 7.85 m/s，
一次 10 → 4 m/s 的降档在闭环里已经足以触发作者 PID 的刹车（`speed / target > 1.1` 即刹），但在开环计分里不算翻转。
这只有在 TFv6 自己开车的闭环里才看得出来，所以本实验在闭环里做配对。

**实验 6：Bench2Drive 按 hazard family 拆分。** 不看 220 条的总分，而是按 scenario family 报 DS（Driving Score，路线完成度 × 违规折扣）
和 SR（Success Rate，走完且无违规的路线比例），带路线 bootstrap CI 和重复运行的 run-to-run 噪声；把公开的逐路线结果（能找到的）
和我们 TFv6 的各臂放在同一张表里。要回答：榜单前几名之间不到 1 分的差距是否在评测噪声以内；谁在 hazard 避让上真的更好。

## 代码审计：规则到底能做什么（写于任何计分 run 之前）

LEAD `cvpr2026`（`730bc1a`）里，README 要求复现 95.28 DS 时打开的三项启发式，逐行读过（`lead/inference/sensor_agent.py`、
`config_closed_loop.py`、`closed_loop_inference.py`）：

| 规则 | 什么时候动 | 能不能替网络对突发 hazard 刹车 |
|---|---|---|
| creeping（`ForceMovePostProcessor`） | 车速 < 0.1 m/s 连续超过 1100 tick（55 s）后，强制油门 ≥ 0.4、取消刹车，持续 20 tick | 不能。它只会**加油门**。所谓 LiDAR safety box 只在 creeping 进行中检查前方点云，有障碍就取消这次 creeping 并刹车，也就是只能撤销规则自己的动作 |
| stop sign（`StopSignPostProcessor`） | 网络检测到的停车牌框距离 < 1 m 且车在动时刹停；进入范围时 40 tick 内把油门限到 0.1 | 不能，只对停车牌起作用 |
| Kalman（`use_kalman_filter`） | 滤 GPS，影响 target point 的计算 | 不能，它不碰 control，只改网络的一个输入 |

另外两件事：(i) 接口 A 自己的刹车逻辑（`pred_target_speed < 0.01` 或 `speed / pred_target_speed > 1.1` 即刹）和接口 B 的
（期望速度 < 0.4 m/s 或速度比 > 1.1）属于**接口**，不是规则，两种规则设置下都一样；(ii) P5 的开环读数取的是网络输出，本来就在后处理之前，
规则开关在开环里只能通过 Kalman 改 target point 间接影响它。

**据此的预测（登记在这里，结果出来后对照）**：规则开关对 hazard 反应没有可测影响；「规则掩盖」这一支从机制上就很难成立。
README 的 95→94 这 1 分预计集中在停车牌路线（`VanillaNonSignalizedTurnEncounterStopsign`）和长时间停滞的路线上。
真正需要测的是另外两支：闭环里 target speed 通道到底反不反应。

## Setup

### 四个臂（2 × 2，只改作者自己暴露的 config 开关）

| 臂 | 接口（interface） | 规则（rules） | 说明 |
|---|---|---|---|
| **A1** | A：route + target speed（作者默认） | on：Kalman + stop sign + creeping（README 复现配置） | 即官方 95 分的配置；W2 的 A 臂 |
| A0 | A | off：LEAD `ClosedLoopConfig` 默认值（三项全关） | |
| B1 | B：三个 modality 全设 `waypoint` | on | W2 的 B 臂 |
| B0 | B | off | |

模型：`tfv6_resnet34`，三 seed ensemble（`model_0030_{0,1,2}.pth`），作者的 `SensorAgent` 原样实例化（`scripts/tfv6_rules_agent.py`），
预处理、PID、后处理一行不改。规则开关走作者自己的 `LEAD_CLOSED_LOOP_CONFIG` 环境变量（`scripts/tfv6_rules_run.sh`），接口开关就是 W2 用过的
modality 赋值。agent 只做记录：每 tick 两个通道的控制量（作者 `ensemble()` 每 tick 两套 PID 都算）、后处理前的控制、每个后处理器改了什么、
实际执行的控制、ego 真值、target speed（解码标量、分布期望、0 档概率）、waypoint 隐含的 2 s 速度，以及 scenario hazard actor 的位置。
这不是用户正在调的那个固定控制器；本实验不用任何我们的 controller。

所有 run 装 P5 的 `_rng` 重置 hook（`B2D_RESEED_AFTER_BUILD=1`），TM seed 0。这让背景车流的抽样与官方不同（相当于换了一个随机种子），
所以我们的 DS 与论文的 DS 不是同一次抽样，只和我们自己的各臂、各次重复直接配对比较。

### 路线

- **B2D-209**：Bench2Drive 220 条去掉 11 条已知会让 server 段错误的路线（`docs/carla.md`），209 条，原 id。每条路线恰好一个 scenario，
  44 个 scenario 类型各 5 条（去掉段错误后有 6 类少于 5 条）。公开结果也限制在同样的 209 条上比较（另报它们自己的 220 条）。
- **配对（实验 1）**：P5 v0 的 9 个 hazard family（PedestrianCrossing、DynamicObjectCrossing、VehicleTurningRoutePedestrian、
  ParkingCrossingPedestrian、OppositeVehicleRunningRedLight、HardBreakRoute、StaticCutIn、ParkingCutIn、HighwayCutIn）× 5 条 = **45 条路线**，
  都在 B2D-209 里。Light（红绿灯）按 P5 v1 的决定归 R 层，不进。每条路线一个配对，seed 0，三个世界：
  - x⁺ = B2D-209 里的那条路线本身（完整跑，计官方分）；
  - x⁻ = 同一 XML，scenario 照常运行，hazard actor 每 tick 藏到地下 500 m（HardBreakRoute：只把「前车急刹」换成空操作），P5 v0 原样；
  - null = x⁺ 只换天气（白天换晴夜、夜晚换正午），P5 v0 原样。
  x⁻ 与 null 在 scenario 触发后 20 s 截断（只需要 scenario 窗口）；x⁺ 不截断。

### 重复与 seed

- TM seed 全部为 0；TFv6 ensemble 固定。
- **A1 在 B2D-209 上跑两遍**（rep0、rep1，配置逐字相同），两遍之差就是「同一方法、同一配置、再跑一次」的 run-to-run 噪声，
  即榜单上一个条目单次评测的噪声。其余三个臂各一遍。
- 配对部分每臂一遍（45 对）。

## 实验 1 的指标与判据（跑之前写死）

### 每对（每臂）的量

- t_trig：scenario 触发时刻（blackboard `ScenarioRouteNumber0`）。
- t_div：x⁺ 与 x⁻ 的 ego 位置差 ≥ 1 cm 或航向差 ≥ 0.1° 的第一个 tick（与 P5 相同）。t_div < t_trig 的对记 `background_drift` 丢掉。
- **碰撞**：从 leaderboard 的 criterion event（带 frame）里取 x⁺ 与 hazard actor（`hidden.json` 里的 id）的碰撞；
  HardBreakRoute 没有 hazard actor，取触发后与任何车辆的碰撞。t_col 是 x⁺ 里第一次这种碰撞。
- **行为反应统计量** S = min over t ∈ W of [v⁺(t) − v_other(t)]，W = [t_trig, min(t_trig + 15 s, t_col, 两个 run 较早结束者)]，
  v 是 ego 真值速度（20 Hz），按仿真时间对齐。x⁻ 上的 S 是配对的反应，null 上的 S_null 是同一臂的噪声。
  S 为负 = 有 hazard 的世界里车更慢。窗口截在碰撞之前，所以「撞上了停下来」不会被算成反应。
- **τ_arm** = max(1.0 m/s，该臂 null 上 |S_null| 的 95 分位数)。null 从第一帧起就因为天气不同而可能分叉，这正是 TFv6 自己开车时
  「无关扰动」造成的速度差，和 P5 用 null 定每个考生噪声地板是同一个意思。
- **反应**：S ≤ −τ_arm。**反应率 RR(arm)** = 有效对里反应的比例。
- **null 误报率 FR(arm)**：null 上 S_null ≤ −τ_arm 的比例（样本内，约 5% 以下，只作检查）；以及按 P5 的做法，
  null 路线随机分两半、一半定 τ 一半量，两个方向平均（样本外，判据用这个）。
- **hazard 碰撞率 HC(arm)**：x⁺ 发生上述碰撞的对的比例。
- **规则归因**（规则 on 的两臂）：x⁺ 在 [t_trig, t_trig + 20 s] 里有没有任何后处理器改动控制；以及在 x⁺ 与 x⁻ 的**执行控制**第一次不同的
  那个 tick，后处理前的控制是否两侧相同（相同 = 这次分叉是规则造成的）。
- **开环读数（描述）**：t_trig ≤ t < t_div 的帧上 ego 两侧逐 tick 相同、只有图像不同，报两个通道的 Δ（target speed 标量、
  分布期望、waypoint 2 s 速度）的分布，作为 TFv6 自己开车时的「on-policy P5」。

所有比例按路线 bootstrap（每条路线一对，重抽 10 000 次）给 95% CI；臂间对比是同一批路线上的配对差，同样按路线重抽。
按 family 分开报，合并按对等权。

### 判据

| 结果 | 读法 |
|---|---|
| RR(A0) − RR(A1) ≥ +15 个百分点且配对 CI 不含 0；**或** A1 里 ≥ 25% 的有效对第一次控制分叉由规则造成 | **规则掩盖**：第 35 条的推翻证据成立，「拿分通道不反应」是规则造成的假象 |
| RR(A1) 的 CI 下界 > A 的样本外 FR，且 RR(A1) ≥ 50%，且 RR(A0) − RR(A1) 的 CI 含 0 | **网络在闭环里真的反应，规则无关**；P5 的 2% 是开环读数（近二值输出 + 天气噪声地板）的问题，不是网络看不见 |
| RR(A1) 与 RR(A0) 都不高于 A 的样本外 FR（RR − FR 的 CI 含 0）或都 ≤ 20% | **网络的 target speed 通道确实不反应**；再看 HC(A) 与 B2D-209 上 A 的 DS：若 HC(A) 不高于 HC(B) 而 DS 仍高，分数来自接口（比如整体更慢、更晚进入冲突区），逐对查进入 scenario 时的车速 |
| 落在以上几行之间 | 按 family 如实报，不下总结论 |

接口的读法另报一行：RR(A) − RR(B) 与 HC(A) − HC(B)（规则 on、off 各一次），回答「会反应的 waypoint 通道」在闭环里是不是真的反应得更多、撞得更少。

### B2D-209 上的 2 × 2（DS 与 SR）

每臂报 209 条的平均 DS、SR；两个主效应（接口 = (A1 + A0 − B1 − B0) / 2，规则 = (A1 + B1 − A0 − B0) / 2）与交互作用，
按路线 cluster bootstrap 给 CI。按 family 拆开再报一次。预期（见代码审计）：规则主效应 ≈ +1 DS，集中在停车牌与停滞路线；
接口主效应与第 31 条同号（A 高）。

## 实验 6 的 family 划分、指标与判据（跑之前写死）

### family

两套划分都报。

1. **Bench2Drive 官方的 5 项能力**（`tools/ability_benchmark.py` 原样：Overtaking、Merging、Emergency_Brake、Give_Way、Traffic_Signs；
   一个 scenario 可以属于几项），这是论文里常报的「multi-ability」口径。
2. **我们的 10 个 hazard family**（互斥，44 个 scenario 各归一个，`jevdrive/tfv6_rules.py` 的 `FAMILY`）。前五个是「突发 hazard」（E 层）：

| family | scenario | 路线数（220 / 209） |
|---|---|---|
| vru_emerging（遮挡后冲出的行人或自行车，「鬼探头」） | DynamicObjectCrossing、ParkingCrossingPedestrian | 10 / 10 |
| vru_crossing（路口横穿的行人、自行车） | PedestrianCrossing、VehicleTurningRoutePedestrian、VehicleTurningRoute、CrossingBicycleFlow | 20 |
| cut_in | StaticCutIn、ParkingCutIn、HighwayCutIn | 15 |
| lead_hard_brake（前车急刹） | HardBreakRoute | 5 |
| junction_violator（他车闯灯、抢行、堵路口） | OppositeVehicleRunningRedLight、OppositeVehicleTakingPriority、BlockedIntersection | 15 |
| unprotected_turn（无保护左转等让行） | NonSignalizedJunctionLeftTurn / RightTurn / LeftTurnEnterFlow、SignalizedJunctionLeftTurn / LeftTurnEnterFlow / RightTurn | 30 |
| merge_lane_change | EnterActorFlow、InterurbanActorFlow、InterurbanAdvancedActorFlow、HighwayExit、MergerIntoSlowTraffic、MergerIntoSlowTrafficV2、SequentialLaneChange、ParkingExit | 40 |
| obstacle_bypass（绕行静止障碍） | Accident、AccidentTwoWays、ConstructionObstacle、ConstructionObstacleTwoWays、HazardAtSideLane、HazardAtSideLaneTwoWays、ParkedObstacle、ParkedObstacleTwoWays、VehicleOpensDoorTwoWays、InvadingTurn | 50 |
| emergency_vehicle | YieldToEmergencyVehicle | 5 |
| routine_control（R 层：路口、红绿灯、停车牌、失控恢复） | T_Junction、VanillaNonSignalizedTurn、VanillaNonSignalizedTurnEncounterStopsign、VanillaSignalizedTurnEncounterGreenLight / RedLight、ControlLoss | 30 |

209 条里每个 family 的实际路线数在结果表里写明（段错误路线全在 Town12/13，分散在几个 family 里）。

### 指标

- 每个方法（我们的四臂 + 找得到逐路线结果的公开方法）、每个 family：平均 DS、SR（Bench2Drive 官方定义：走完且除 min speed 外没有违规）、
  主要违规类型计数。**CI**：family 内按路线 bootstrap（10 000 次），95% percentile。
- **run-to-run 噪声**：A1 的 rep0 与 rep1 逐路线配对。每条路线的单次噪声方差估计 σ̂²_r = d_r² / 2（d_r 是两次 DS 之差），
  209 条总分的单次噪声 SD = sqrt(Σ σ̂²_r) / n；family 同理。另报两次完全相同的路线比例和 |d_r| 的分布。
- **方法间比较**：同一批路线上的配对差（总分和每个 family），两种 CI 并报：(a) 路线 bootstrap（「换一批同类路线，差距还在不在」）；
  (b) run-to-run：差是否超过 2 × sqrt(2) × 单次噪声 SD（「同一批路线再各跑一次，名次会不会翻」）。噪声只在 TFv6 A1 上量得到，
  套用到其他方法是假设，写明。

### 判据

| 结果 | 读法 |
|---|---|
| 榜单前几名（公开逐路线结果能配对的）两两总分差 < 2 × sqrt(2) × 单次噪声 SD，且路线 bootstrap CI 含 0 | 这些名次差在评测噪声以内，不能排序 |
| 某方法在 SUDDEN 五个 family 合并的 SR 上比另一方法高，且路线 bootstrap CI 不含 0、差值超过 run-to-run 噪声 | 这个方法在突发 hazard 上确实更好 |
| 总分差在噪声内而某个 family 的差显著 | 报出来：总分掩盖了 family 间的取舍 |
| 公开逐路线结果一个也找不到 | 实验 6 只报我们四臂的 family 拆分和噪声；跨方法只能用各论文自报的 multi-ability 分数（没有逐路线，不能给 CI），明确写这是限制 |

## 估计与分批（批量之前）

smoke 实测（00:03–00:13，box 满载：另有两个 GPU 任务、一个进程占 35 核、load 55 / 50 核）：A1 跑完整条 27515（Town03）
1077 tick、wall 557 s；每 tick 462 ms，其中 agent 436 ms（三 seed forward 226 ms，其余是作者的 LiDAR / GPS 预处理），
`world.tick` 只有 23 ms，所以瓶颈是 agent 的 CPU / GPU 争用，不是渲染。显存每 worker 约 11 GB（CARLA 约 8 + TFv6 约 2.5）。
第 31 条在 Tokyo 3090 上不争用时 agent 约 100 ms / tick，这里 01:15 之后其他任务结束，预计落在 150–250 ms / tick 之间，
批量开头 30 分钟的实测替换这个估计（写进结果一节）。

工作量（按 tick 计；B2D 完整路线按 800 tick、截断的 x⁻ / null 按 500 tick）：

| 批 | 内容 | run 数 | tick | 1 张卡 6–8 worker，约 30 tick/s 合计 |
|---|---|--:|--:|--:|
| T1 | 配对 45 条 × 4 臂 × {x⁺ 完整, x⁻, null} | 540 | 约 32 万 | 约 3 h |
| T2 | A1 rep0：B2D-209 里其余 164 条（与 T1 的 A1 x⁺ 合成完整 209 条） | 164 | 约 13 万 | 约 1.2 h |
| T3 | A1 rep1：209 条整遍重跑 | 209 | 约 17 万 | 约 1.5 h |
| T4 | A0、B1、B0：各自其余 164 条 | 492 | 约 39 万 | 约 3.6 h |

合计约 9 h，超过 3 h，而且超过 GPU 的可用窗口（01:15 起两卡由本实验与 SimLingo 分，04:00 起 HUGSIM 要一部分）。
这里没有可优化的「我们的代码」：瓶颈在作者 agent 内部（三 seed forward + LiDAR 预处理），按规则第三方代码原样跑，只能调并发。
所以**按 T1 → T2 → T3 → T4 的顺序跑，时间用完就停，停在哪里如实报**：T1 回答实验 1；T1 + T2 + T3 回答实验 6 的噪声与 TFv6 的 family 拆分；
T4 只补 B2D-209 上的 2 × 2 DS（没有 T4 时，2 × 2 只在 45 条配对路线上报）。并发：一张卡上 worker 数从 6 起，显存 ≤ 80 GB，
CPU 不超过我们在 gpu-plan.md 里登记的份额。基础设施失败（server 段错误等）每个 run 最多重试一次，仍失败的记缺失、不补跑。

## 步骤

- [x] agent（作者 SensorAgent 原样 + 只读记录）、runner、路线 XML（209 条 + 45 对 x⁻ / null）、family 表：`00bf79f`
- [ ] smoke：4 臂 × 2–4 个 run，GPU 0 ≤ 20 GB；核对 x⁺ 与 x⁻ 在触发前逐 tick 相同、规则开关确实生效、计时
- [x] 本文件提交（任何计分 run 之前）；估时补进上一节；gpu-plan.md 登记
- [ ] 公开逐路线结果收集（CPU，`research/results/b2d-family/public/`）
- [ ] 批量（01:15 起）
- [ ] 分析：配对、2 × 2、family、噪声；图；decisions 条目；第 35 条按规则更新

## 结果

跑完再填。
