# openpilot 进 Bench2Drive 闭环：为什么起不了步、不转弯，以及「base + openpilot modifier」怎么接

2026-09-28。计划、登记与执行日志在 [todos/2026-09-28-op-closedloop.md](../todos/2026-09-28-op-closedloop.md)，小表在 [results/op_arb/](results/op_arb/)，
代码 `scripts/op_arb_agent.py`（agent 与各仲裁模式）、`scripts/op_arb_server.py`（openpilot server）、`scripts/op_arb.sh`（启动）、`jevdrive/op_arb_report.py`（读数）、`jevdrive/op_arb_figs.py`（图）。
本文供用户与 main 讨论；所有数字都是 GPU 6 测试卡上的小规模 pilot（phase 1 诊断 6 条路线、phase 2 评测 10 条路线，TM seed 0 单次），只能给方向。

## 结论先行

> **用户决定（2026-09-28 ~19:00）：五种仲裁都不满意，集成设计搁置，以后再好好调。** 下面第 4 节的「推荐设计」只作为本轮 pilot 里最好的一个记录，不是定案。
> 随后做了两个诊断，见第 7 节：**同均速、不用 openpilot 的 base 与 e2e 同一水平（DS 67.7 对 66.2，违规 7 对 8），e2e 的收益可以完全用开得慢解释**；
> 幽灵停车一半是 openpilot 在这些位置本来就想减速（只看时 33–50%），一半是闭环把减速放大（80%、更深），再被 CARLA 的低速刹停和我们的锁存变成长时间停车；去掉 20 s 兜底 DS 36.4、5 条 blocked。

1. **起不了步是模型的静止先验，不是执行层、不是缺车速输入、也不是把前车读错。** native（openpilot 原生 plan → P7）6 / 6 条诊断路线车速始终为 0；
   静止且前方无障碍时 plan 5 s 只走 0.97 m（中位），action accel −0.02，meta 头预测「驾驶员此刻踩刹车」0.80；画面估的自车速度 −0.02 m/s（知道自己停着），前车读得很准（lead_prob 1.00、x 3.9 m）。
   一旦被带到 1 m/s 以上，plan 反而要加速（1–4 m/s 时 5 s 后比当前快 3.5–6 m/s）：缺的只是真车上驾驶员按 resume 的那一下。
2. **不转弯是因为 openpilot 不发起路口转弯，turn desire 在路口前不改变 plan。** 转弯前 0–20 m，带 turnLeft desire 的 plan 在 15 m 处只跟了路线横移的 −5%，不给 desire 的 twin 是 −10%；进弯以后两者都跟 100%。
   闭环里把路口交给 openpilot（oplat）10 / 10 条失败，DS 8.5，3 条 route deviation。
3. **base 本身就很强，这是本研究最需要讨论的一点。** 一个没有任何感知、只按路线走、8 m/s 巡航的 base，在 10 条 dev 路线上 DS 56.6、RC 93.5（native openpilot 是 0，第 33 条 / CL3 / K0 / K1 是 7–10）。
   Bench2Drive 的短路线里，大部分分数来自「按路线走完」，而这正是 openpilot 结构上做不了的那部分。
4. **openpilot 当 modifier 加了东西，但信号弱、噪声大。** 纵向 = min(base, openpilot lead 头 IDM, openpilot plan)、plan 引起的停车锁存到 openpilot 放行（e2e）：DS 66.2，对 base +9.7 [−5.5, +27.0]（4 条好 / 3 条差 / 3 条同）。
   好在：StaticCutIn 36 → 100（lead 头避免了两次追尾）、VehicleTurningRoutePedestrian 25 → 70（没撞行人）、两条路口 42 → 60（没闯灯）；差在：plan 在闭环里频繁把车带停在无事可停的地方
   （42 次锁存里 30 次真值「无障碍」，41 次放行里 24 次靠 20 s 兜底），车均速 0.96 m/s 对 base 2.69，T_Junction 与 MergerIntoSlowTraffic 因此超时或丢分。
   只接 lead 头（acc）两次运行 58.2 / 62.9，对 base +1.6 / +6.3，主要就是 StaticCutIn 那一条；**同配置两次运行在一条路线上差 47 DS**，所以 10 条单 seed 只能给方向。
5. **让 openpilot 横向驾驶（switch、oplat）明显更差**：DS 26.1 / 8.5，对 base −30.5 [−42.6, −19.1] / −48.1。openpilot 自己开时 7 / 10 条出车道、5 条 blocked。
6. **推荐**：base（路线几何 + 巡航）管横向与起步，openpilot 只做纵向 modifier（lead 头 ACC + 行驶中的 plan 约束），停车锁存由 openpilot 自己放行。
   这与 openpilot 真车 experimental 模式的分工一致（驾驶员管路线与 resume），加在外面的东西是通用的几何与一条 IDM；最大的 trick 风险不在这些规则，而在第 3 点：B2D 的分数大半是 base 的。
   论文里 openpilot 的贡献必须用「对 base 的配对差」和按 hazard 归因的违规来报，不能报绝对 DS。
7. **行人在 CARLA 里不要指望 openpilot**（第 55 条：真实 nuScenes 上 AUC 0.83，CARLA P5 上 0.506，只用真实数据改 vision 也带不过来）。27297 上 e2e 没撞行人，更可能是它当时正被 plan 压得很慢。

## 1. 背景：已有的闭环读数

名词：DS（Driving Score，Bench2Drive 闭环总分，0–100）、RC（Route Completion，路线完成百分比）、SR（Success Rate，完成且除 min-speed 外零违规的路线比例）；
P7 是我们验收过的轨迹执行器（第 41 条）；desire 是 openpilot 唯一的「意图」输入，8 类 one-hot（无、左转、右转、左变道、右变道……），按 modeld 的做法只在上升沿给一个脉冲；
openpilot 没有 route、target point，也没有车速输入（车速只从画面估）。

| 来源 | 接法 | 结果 |
|:--|:--|:--|
| 迁移文档 D1（09-25，5 条） | openpilot 原生 plan → Zoo PID | 20 个路线 × 配置里 12 个从头到尾没动 |
| 迁移文档 D5 smoke3（9 条） | 纯 Lebowski / + TCP 起步与路口伙伴 / TCP 单独 | DS 9.9 / 65.4 / 73.9；纯 openpilot 路口 0 / 3；加伙伴后模型只开了 32% 的时间，分数低于 TCP 单独 |
| night-queue-3 CL0 / OPL（09-26） | Cinque / Lebowski 原生 plan → P7；+ TCP 起步 | 3 / 3 条 blocked；加 TCP 起步后伙伴接管 89% 的 tick，按登记不接受 |
| night-queue-3 CL3 / CL6（220 条 × 3 seed） | Cinque `temporal` 上的薄 head 轨迹 → P7 | DS 8.0–9.8、8.8，碰撞 410–470 次 |
| night-queue-4 K0 / K1（220 条 × 3 seed，本文补读） | 同一特征上 `ridge_late` / TFv6 的 route + target speed 表示 | DS 7.2–7.8 / 7.0–7.4，RC 33–35 / 21–22，blocked 54–75 次 |
| 作者执行层参照（CL10） | BridgeDrive / BLUE / SimLingo | DS 96.1 / 88.8 / 86.8 |

所以「openpilot 在 B2D 上 DS 8–10」这个数的来源其实有两类：原生 plan 根本不起步（D1、CL0、OPL），和我们在 openpilot 特征上学的读出（CL3–CL6、K0/K1）一路撞车、卡死。
后者说明「学一个小的 route-conditioned 读出当 base」这条路已经被试过：K1 用的正是 TFv6 高分所依赖的 route + target speed 表示（第 31 条），闭环 DS 没有比 K0 高。

## 2. 失败诊断（phase 1，6 条路线）

两个臂在同一条 openpilot 流上（Cinque，20 Hz，原生 road + wide rig，5 s 预热，后轴变换，路线 desire，P7 5 Hz），只差谁开车：
**native** 是 openpilot 自己的 plan 开；**oshadow** 是只用来诊断的特权司机开（路线几何 + 真值前车 / 行人 IDM + 路线上的红灯停车），openpilot 同帧跑在 shadow 里，
server 上另开一个同帧、desire 恒为 0 的 twin session。每个 openpilot step（每 tick）记下它的全部输出头：plan 的位置与速度、action 头的 accel、
lead 头（前车距离、速度、存在概率 lead_prob）、meta 头（openpilot 预测的驾驶员 gas / brake 踩踏概率，0 s 与 2 s）、desire_pred、pose（画面估的自车速度），
以及只用于评估的真值上下文（路径内前车 / 行人距离、路线上下一个停止线距离与灯色）。路线（Bench2Drive 0.0.4 val，不是 220 考卷）：
334 SignalizedJunctionLeftTurn、27787 VanillaSignalizedTurnEncounterRedLight、24721 HardBreakRoute、26872 NonSignalizedJunctionLeftTurn、26537 ParkingExit、17749 DynamicObjectCrossing。

![op-arb phase 1](figs/op_arb_p1_diag.png)

(a) 静止 tick 上 openpilot plan 5 s 处的位移，按真值情境分组（箱 = 四分位，须 = 5–95%）；(b) oshadow 带着车在无障碍路段行驶时，plan 5 s 处速度与当前速度之差；(c) 路口转弯前 0–20 m 和转弯中，
plan 在前方 15 m 处的横向偏移占路线横向偏移的比例（1 = 跟着路线转，0 = 直行），有 route desire 与无 desire 的 twin 并排。

### 2.1 起不了步：openpilot 在静止时的 plan 本身就是「再等等」

native 6 / 6 条路线车速始终为 0，60 s 后被判 blocked（DS 0）。原因不在执行层：P7 在这些 tick 上做的就是 plan 让它做的事。静止 tick 上的 openpilot 输出：

| 情境（真值） | 步数 / 路线 | plan v@3 s | plan x@5 s | action accel | lead_prob | lead x | gas@0 s | brake@0 s | 画面估车速 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 无障碍（30 m 内无车无人、40 m 内无红黄灯） | 1 416 / 5 | 0.13 m/s | **0.97 m** | −0.02 | 0.13 | 36 m | 0.03 | 0.80 | −0.02 |
| 红 / 黄灯 ≤ 30 m | 1 819 / 3 | 0.02 | 0.18 | 0.01 | 0.19 | 35 | 0.01 | 0.92 | −0.02 |
| 路径内前车 ≤ 15 m | 5 597 / 3 | 0.01 | 0.08 | −0.04 | 1.00 | 3.9 | 0.02 | 0.86 | −0.03 |

（中位数；native 与 oshadow 合并，分臂的表见 [standstill.csv](results/op_arb/p1/standstill.csv)。）

四点读法：

1. **不是缺车速输入。** 画面估出的自车速度在静止时是 −0.02 m/s，模型知道自己停着。
2. **不是把静止前车读错。** 真有前车时 lead_prob 1.00、lead x 3.9 m，读得很准；没有前车时 lead_prob 0.13。起步失败在无障碍路段照样发生。
3. **是「停着就继续停着」的先验。** 无障碍时 plan 5 s 只走 0.97 m，action accel 甚至是负的；meta 头给出的「驾驶员此刻踩刹车」概率 0.80，「踩油门」0.03。
   这和 openpilot 的真车用法一致：它的训练数据里，车停稳以后何时走由驾驶员决定（按 resume 或踩油门），模型学到的是「人类在这种画面里通常还停着」的条件期望。
   plan 对情境**有排序**（x@5 s 区分「该走」与「该等」的 AUC 0.86，lead x 0.91，见 [standstill_auc.csv](results/op_arb/p1/standstill_auc.csv)），但绝对量只有 1 m 级，任何照 plan 走的执行层都不会起步。
4. **它不看红绿灯，至少静止时不看。** 27787 的起点就在停止线后 3 m：绿灯的 10 s 和红灯的 20 s 里 plan x@5 s 都在 0.1 m 左右；变化的只是有横穿车流时 gas@2 s 从 0.13 掉到 0.01。
   所以「红灯还是绿灯」这件事，openpilot 在停止线前的输出里几乎没有。

**一旦在走，它就会加速。** oshadow 把车带起来以后，无障碍路段上 plan 5 s 处的速度比当前快：1–2 m/s 时 +6.0 m/s，2–4 m/s 时 +3.5 m/s，4–8 m/s 时 ±0.5 以内（图 (b)，[cruise.csv](results/op_arb/p1/cruise.csv)），
8 m/s 以上是 −2.5（它在城区想开 6 m/s 左右）。也就是说起步失败**只发生在静止这一个点上**，不是 D4 当时推测的「低速时 plan 一直偏慢」：只要有人把车带过 1 m/s，openpilot 自己就会开起来。
这正是真车上「engage while rolling」和「驾驶员按 resume」补上的那一格。

### 2.2 不转弯：openpilot 不发起路口转弯，turn desire 几乎不起作用

oshadow 路线上的 3 个左转路口（334、27787、26872），比较前方 15 m 处 plan 的横向偏移与路线本身的横向偏移（图 (c)，[turn.csv](results/op_arb/p1/turn.csv)）：

| 位置 | 步数 | 路线在 15 m 处的横移 | plan（route desire）跟了多少 | twin（无 desire）跟了多少 | desire_pred 里「要转弯」的概率 |
|:--|--:|--:|--:|--:|--:|
| 转弯前 0–20 m | 79 | 2.8 m | **−0.05** | −0.10 | 0.38（两者相同） |
| 转弯中 | 169 | 6.6 m | 1.04 | 0.98 | 0.62（两者相同） |

转弯前，带 turnLeft desire 的 plan 在 15 m 处几乎是直的（只跟了路线横移的 −5%），和不给 desire 的 twin 没有区别；车已经被 base 带进弯以后，openpilot 才顺着画面里的弯继续（跟 100%），这时有没有 desire 也一样。
desire_pred（模型自己预测的驾驶员意图）在两个 session 里逐位相同，说明它只看画面，不受 desire 输入影响。
这与第 49 条开环里「laneChange desire 方向总对但幅度只有半条车道」一致，而且更弱：**turn desire 在路口前不改变 plan**，也就不存在「给 desire 就能让 openpilot 转弯」这条路。
openpilot 在真车上本来就不在路口转弯（驾驶员接管方向盘），它学到的是车道保持加「已经在弯里就顺着弯」。

### 2.3 它在闭环里能提供什么：前车强、红灯弱、行人样本太少

oshadow 行驶中（> 2 m/s）的 tick，按真值「必须减速」事件对「无障碍」行驶 tick 算 AUC（[hazard_auc.csv](results/op_arb/p1/hazard_auc.csv)）：

| 事件（真值） | 步数 / 路线 | plan 3 s 内减速量 | −action accel | lead_prob | brake@0 s | hard brake |
|:--|--:|--:|--:|--:|--:|--:|
| 前车 TTC < 4 s 且 < 30 m | 80 / 2 | 0.95 | **0.97** | 0.73 | 0.93 | 0.74 |
| 红 / 黄灯 < 30 m | 108 / 2 | 0.68 | 0.64 | 0.43 | 0.76 | 0.73 |
| 行人在路径内 < 20 m | 30 / 1 | 0.83 | 0.91 | 0.63 | 0.76 | 0.44 |

（对照：无障碍行驶 718 步。）前车类 hazard 上 openpilot 的纵向信号很强，这是 lead 头和 ACC 的本行；红灯只有 0.64–0.76，意味着它会闯一部分红灯；
行人只有 1 条路线 30 步，不下结论。**行人反应在 CARLA 里不能指望 openpilot 自己给**：第 55 条（op-adapt）测到原版 Cinque 在真实 nuScenes 上读得出 ≤ 10 m 的车道内行人（AUC 0.83，信息在 trunk 里），
但在 CARLA P5 行人上是 0.506 的随机水平，只用真实数据改 vision 层也带不过来；第 42 条的 CARLA probe 同样是 0.51。所以这里的 0.9 很可能是行人从停着的车后出来时对「车」的反应（推测，要 P5 式配对才能分开），
行人是一个 sim 域差，不是 openpilot 在真车上缺的能力。这直接影响 modifier 的触发设计：在 CARLA 里，openpilot 能可靠触发的只有前车类（lead 头、plan 减速），红灯部分可靠，行人与 VRU 基本不行。

## 3. 仲裁方案

共同的 base（每个方案都一样）：路线几何（Bench2Drive 发给所有 agent 的 dense route，用传感器位姿 rejoin，`RouteAdapter`）+ 速度 governor（设定速度 8 m/s、曲率上横向加速度 ≤ 2 m/s²、加速度 ≤ 1.5 m/s²、终点停车）。
它没有任何感知，只做两件事：按路线走，和「驾驶员设定的巡航速度」。openpilot 是 modifier：它能让车更慢、停下、（switch / oplat 里）自己转方向，但不能让车比 base 更快。

| 方案 | 机制 | openpilot 管什么 / base 管什么 | 真车上对应 | trick 风险 |
|:--|:--|:--|:--|:--|
| base（消融） | base 单独开，openpilot 只记日志 | 0 / 全部 | openpilot 关掉 | — |
| acc | 纵向 = min(base 速度剖面, openpilot lead 头上的 IDM)，横向 = base | 跟车、跟停、前车起步 / 路线、巡航、红灯（不管） | chill 模式（ACC + 驾驶员管路口） | 低：IDM 是通用跟车律，lead 头是 openpilot 原生输出 |
| e2e | acc + 车速 ≥ 1 m/s 时 openpilot plan 的位置剖面也当上限（min）；plan 造成的停车锁存，等 plan 5 s 位移 > 2 m 持续 1 s 放行，20 s 兜底 | 以上 + 红灯 / 弯道 / 行人减速 / 路线、巡航、起步、兜底放行 | experimental 模式（端到端纵向）+ 驾驶员按 resume | 中：放行阈值是在 6 条诊断路线上定的一个数；20 s 兜底是「不耐烦的驾驶员」 |
| switch | 车速 ≥ 2 m/s 且不在路口转弯区时 openpilot 原生 plan 开（横纵都是它）；起步、转弯区（左右转命令前 15 m 到后 5 m）用 e2e | 直路上横纵全归它 / 起步、路口转向 | D3 的 TCP 伙伴换成 base | 中：转弯区用的是路线命令，与 D3 相同 |
| oplat | switch 去掉转弯区：openpilot 带 desire 自己过路口 | 除起步和锁存外全部 | 纯 openpilot + resume | 低，但诊断预期它不会转弯 |
| 小的学习型 base（未跑） | route-conditioned 读出（K1 式）代替几何 base | — | — | 高且已失败：K0 / K1 闭环 DS 7–8 |
| desire 脉冲过路口（未单独跑） | 路口前按路线发 turn desire，让 openpilot 自己转 | — | 驾驶员打灯 | 诊断显示 turn desire 不改变 plan，已含在 oplat 里 |
| 混合（blending）而不是切换 | 横向按置信度混 openpilot 与 base 的偏移 | — | — | 诊断显示直路上两者重合（|plan − route| 0.2°），转弯前 openpilot 是直的，混合只会把转弯拉直，未跑 |

### 3.1 pilot 结果（phase 2，10 条 dev 路线，TM seed 0）

路线（Bench2Drive 0.0.4 val，与 phase 1 不重）：27043 SignalizedJunctionRightTurn、15102 VanillaSignalizedTurnEncounterGreenLight、24944 T_Junction、27870 VanillaNonSignalizedTurn、22535 StaticCutIn、
37969 MergerIntoSlowTrafficV2、24497 ConstructionObstacle、27297 VehicleTurningRoutePedestrian、9196 OppositeVehicleTakingPriority、28147 SignalizedJunctionLeftTurnEnterFlow。
表来自 [arms.csv](results/op_arb/p2/arms.csv)、[paired.csv](results/op_arb/p2/paired.csv)、[per_route.csv](results/op_arb/p2/per_route.csv)；「openpilot binding」= 非预热 tick 里 openpilot 的约束（lead / plan / 锁存 / 原生 plan）是最紧那一个的比例。

| 方案 | DS | RC | SR | 车辆 / 行人 / 静物碰撞 | 闯红灯 | blocked | 出车道 | openpilot binding（tick / 距离） | 对 base 的 DS 差 [95% CI]，好 / 差 / 同 |
|:--|--:|--:|--:|:--|--:|--:|--:|:--|:--|
| base（openpilot 关） | 56.6 | 93.5 | 0.2 | 5 / 2 / 1 | 4 | 0 | 1 | 0 / 0 | — |
| acc | 58.2 | 85.3 | 0.3 | 2 / 2 / 2 | 4 | 0 | 1 | 0.20 / 0.19 | +1.6 [−14.0, +19.1]，1 / 2 / 7 |
| acc2（同配置重复运行，见执行日志的 bug 一条） | 62.9 | 93.3 | 0.3 | 3 / 2 / 1 | 4 | 0 | 1 | 0.19 / 0.23 | +6.3 [−0.3, +19.2]，1 / 1 / 8 |
| **e2e** | **66.2** | 89.9 | 0.3 | 4 / 0 / 1 | 3 | 0 | 1 | 0.76 / 0.60 | **+9.7 [−5.5, +27.0]，4 / 3 / 3** |
| switch | 26.1 | 56.1 | 0.0 | 10 / 0 / 6 | 0 | 5 | 7 | 0.62 / 0.87 | −30.5 [−42.6, −19.1]，1 / 9 / 0 |
| oplat | 8.5 | 16.8 | 0.0 | 7 / 0 / 6 | 1 | 2 | 4 | 0.48 / 0.89 | −48.1 [−63.7, −33.2]，0 / 10 / 0 |
| 参照：native（phase 1 的 6 条） | 0.0 | 0.0 | 0 | — | — | 6 | — | 1 | — |

![op-arb phase 2](figs/op_arb_p2_ds.png)

逐路线 DS，每个点是一个方案在一条路线上的单次运行。看三件事：base / acc / acc2 在 7 条路线上重合（openpilot 的 lead 头只在 22535 上决定了结果）；e2e 在 9196、27043、27297 上高出一截、在 24944、37969 上掉下来；switch / oplat 几乎全在底部。

读法：

- **openpilot 的 lead 头是可靠的 modifier，但在这 10 条里用得上的只有一条。** acc 的 binding 有 82% 发生在真值 30 m 内无车的 tick 上（它对 30 m 以外的慢车也在减速），这些不改变结果；
  22535（StaticCutIn）是唯一一条 base 追尾、acc 不追尾的路线。
- **openpilot 的 plan 约束同时带来好处和「幽灵停车」。** e2e 里 plan 是最紧约束的 tick 有 69% 真值无障碍、18% 红黄灯；42 次锁存（plan 把车带停）里 30 次在无障碍处，7 次红灯、4 次前车。
  plan 的停车有真信号（三条路线上避免了闯灯或碰撞），但假阳性更多；锁存的放行信号（plan 5 s 位移 > 2 m）只放行了 17 次，24 次靠 20 s 兜底。
  推测机制：闭环里车一旦按 plan 减速，模型 5 s 的隐状态历史就看到自己在减速，接着预测继续减速直到停（人类数据里「开始减速」常常就是「要停」），这是一个自我强化的停车吸引子，
  开环 shadow 里（phase 1，车由别人开）看不到。验证：同一位置在 oshadow（别人开）与 e2e（自己开）下 plan 的减速量配对比较，或在 plan 约束里只用 plan 相对它自己当前速度估计的减速（`plan_form` rel）再跑一次。
- **B2D 的 DS 奖励慢。** e2e 平均车速 0.96 m/s，base 2.69，DS 却更高：DS 只按违规打折，慢本身几乎不扣分（这些运行每条都有十几次 min-speed 违规，但 15102 上 base 带着 20 次 min-speed 违规照样 DS 100）。这与第 38 条「保守驾驶本身抬高 hazard SR」是同一件事，
  所以 e2e 的 +9.7 里有多少是「看见了」、多少是「开得慢所以撞不上」，这 10 条分不开。按路线看 e2e 比 base 少掉的违规：27043、9196 各少一次闯红灯（碰撞仍在），27297 少两次撞行人（但多一次闯红灯），
  而 22535 的两次追尾 acc 也避免了（那是 lead 头的功劳）；这些位置 e2e 的车速都很低。
- **openpilot 自己横向开会离开路线。** switch 让 openpilot 在直路上接管横纵，7 / 10 条出车道、5 条 blocked，碰撞多数记在 base 接回之后（车已经偏出车道）；oplat 连路口也给它，3 条直接 route deviation。
  这与 phase 1 里 shadow plan 在直路上与路线只差 0.2° 不矛盾：shadow 里车一直在路线上，闭环里 openpilot 的横向误差会累积，而且它在低速（大部分时间 < 2 m/s）的横向本来就差（第 49 条 desire 幅度随车速变小是同一现象）。机制没有单独隔离。

## 4. 推荐设计

```mermaid
flowchart LR
  R[B2D route + sensor pose] --> B[base: route geometry + set speed + curvature cap]
  C[road + wide cameras 20 Hz] --> O[openpilot Cinque, route desire]
  O -->|lead head| A[IDM on lead]
  O -->|plan, v >= 1 m/s| P[plan position profile]
  O -->|plan x@5s > 2 m for 1 s| L[stop latch release]
  B -->|lateral path| M
  B -->|speed profile| M[min of speed profiles]
  A --> M
  P --> M
  L --> M
  M --> P7[P7 executor, 5 Hz]
```

- **分工**：横向永远是 base（路线几何）；纵向 = min(base 巡航剖面, lead 头 IDM, 行驶中的 plan 剖面)；plan 造成的停车锁存，openpilot 自己的 plan 给出「要走」时放行，20 s 兜底。
  这就是 e2e 臂，也是 openpilot experimental 模式在真车上的分工：驾驶员给路线、设定速度、按 resume，模型管纵向。openpilot 占多少：非预热 tick 里 76% 由 openpilot 的约束决定速度（按距离 60%），横向 0%。
- **为什么不是 switch / blending**：openpilot 在直路上的横向与路线重合（没东西可加），进路口前不转（加进来只会把弯拉直），自己开时累积偏离（switch −30 DS）。横向给它唯一有意义的场景是绕行，而第 49 条和 Q1 都说它不绕。
- **为什么不是小的学习型 base**：K0 / K1 已经是「openpilot 特征上的 route-conditioned 读出」，闭环 DS 7–8；几何 base 零训练就是 56.6。学出来的 base 只在它能看见东西时才值得，而那是 openpilot 该做的事。
- **trick 风险（逐项）**：路线几何与设定速度（B2D 给每个 agent 的路线；真车上是导航与驾驶员，通用，风险低）；20 s 兜底放行（真车上驾驶员的耐心，是 B2D 专用的数字，中）；放行阈值 2 m / 1 s（6 条诊断路线上定的一个数，中）；
  IDM 参数（通用教科书值，低）。最大的风险是结构性的：base 单独 56.6，所以任何「openpilot + base」的绝对 DS 都主要是 base 的；只有对 base 的配对差才是 openpilot 的。
- **下一步要修的一件事**：plan 约束的幽灵停车。候选（先登记再跑）：`plan_form` rel（只取 plan 相对它自己当前速度估计的减速）；或 plan 约束只在 meta brake_press 同时高时生效（`plan_gate` brake，phase 1 里 brake@0 对前车 / 红灯的 AUC 0.93 / 0.76）。
  两者都已在 `op_arb_agent.py` 里实现为开关，没跑。

## 5. 要不要等适配线（op-adapt B / C、世界模型想象训练）

不用等，大部分集成与它们正交；但有两处会随它们变，要留接口。

| 部分 | 依赖适配线吗 | 现在能做 |
|:--|:--|:--|
| base（路线几何、巡航、曲率限速）、P7、起步锁存状态机、IDM | 不依赖 | 已实现（`op_arb_agent.py`），可以直接上 220 条 |
| 仲裁接口（openpilot 只输出 plan / lead / meta，外面取 min） | 不依赖：适配后的 Cinque 输出同样的头，换权重即可 | 已实现；op-adapt 的 PyTorch port 与 ORT 同输出，换进 server 是一行 |
| 静止起步 | 不依赖。这是训练数据里「resume 由人触发」造成的先验，vision 层的适配改不了它；世界模型里的想象训练理论上能学会「何时走」，但在那之前锁存 + 放行是必要的 | 保持锁存；想象训练出来后，放行信号直接换成它的 plan |
| 路口转弯 | 不依赖。desire 在路口前不改变 plan；除非适配线加 route / nav 输入（目前两条线都没有），base 一直负责 | — |
| 行人 / VRU 反应 | **依赖，而且目前两条线都不解决**：第 55 条说 CARLA 行人是 sim 域差，真实数据的适配带不过来 | 闭环考 VRU 时要么接 sim + real 混训后的模型，要么明写「openpilot 在 CARLA 看不见行人」 |
| plan 的幽灵停车 | 可能依赖：如果是闭环自我强化（推测），在世界模型里做 on-policy 的想象训练正好对着它；vision 适配不会改 | 先用 `plan_form` rel / `plan_gate` brake 两个开关在 dev 上测，登记后跑 |

所以现在就能推进的是：用 e2e 的仲裁在 220 条（或 dev + held-out）上给出 base、acc、e2e 三臂 3 seed 的配对差，作为「openpilot 当 modifier 能加多少」的基线；
适配线出来以后，同一个 agent 只换 server 里的权重，再跑同一组配对，读数就是「适配加了多少」。

## 预算

CARLA 只在 GPU 6（与 G 的 pilot 共卡，本 lane 2 个 server），phase 1 约 1.3 worker·h、两次 smoke 约 0.5、phase 2（五臂加一次 base 重跑）约 7.7，合计约 9.5 worker·h（上限 20）。
单 tick 约 0.5–1.1 s 墙钟，主要是共卡时两路 1928×1208 相机的渲染；openpilot + twin 每步约 90 ms。

## 6. 讨论的开放问题

1. **base 占了大半分数，论文怎么报？** 建议只报对 base 的配对差（按 hazard family 与违规归因），不报绝对 DS；或者换一个 base 做不到的考卷（第 38 条说 B2D 的差距在规划 / 让行类路线）。要不要这样定？
2. **慢是不是作弊？** e2e 均速是 base 的 1/3，DS 却更高。是否在读数里加一个进度 / 速度护栏（例如对 base 的平均车速不低于 x%），或者报「每 km 违规」而不是 DS？
3. **20 s 兜底放行算不算 trick？** 它对应真车上驾驶员不耐烦按 resume；去掉它 e2e 会在幽灵停车处永久停住。要不要把「兜底放行次数」作为一个必须报的数？
4. **起步放行交给谁？** 现在是 openpilot 自己的 plan（5 s 位移 > 2 m）。phase 1 里它在 8 段「该等」里 0 误放，但在 13 段「该走」里只放了 4 段。要不要允许一个同样小的外部信号（例如只看 lead 头：前车离开即走），代价是红灯前会走（它不看灯）？
5. **幽灵停车的机制**：是闭环自我强化（推测），还是 CARLA 画面里的什么东西（路边停车、路口形状）让它想停？用 oshadow 与 e2e 同一位置的 plan 配对就能分开，要不要先做这个再上 220 条？
6. **横向要不要彻底放弃 openpilot？** 当前证据（switch −30、oplat −48、desire 不转弯、不绕行）都说放弃；唯一的保留是绕行（第三层），那需要外部的模式头（Q2 / X 那条线），与 openpilot 无关。

## 7. 两个诊断（用户搁置设计后，2026-09-28 19:00–20:30）

登记写在 [todo](../todos/2026-09-28-op-closedloop.md) 的「后续」一节，早于任何诊断数字；小表 [results/op_arb/diag/](results/op_arb/diag/)。同样 10 条 dev 路线、单 seed，两个新臂各 10 条，约 4.5 worker·h（全 lane 合计约 14）。

### 7.1 同均速对照：e2e 的 +9.7 可以完全用「开得慢」解释

`baseslow` = base 单独开，每条路线的设定速度按 e2e 在该路线的平均车速换算（1–2.5 m/s 为主），openpilot 只记录。实测平均车速与 e2e 基本对上（例如 15102 0.99 / 0.86、22535 1.60 / 1.38、28147 0.65 / 0.69）。

| 臂 | DS | RC | 车辆 / 行人 / 静物碰撞 | 闯红灯 | blocked | 碰撞 + 闯灯合计 | 对 base 的 DS 差 [95% CI] |
|:--|--:|--:|:--|--:|--:|--:|:--|
| base（8 m/s） | 56.6 | 93.5 | 5 / 2 / 1 | 4 | 0 | 12 | — |
| **baseslow（同均速，无 openpilot）** | **67.7** | 89.0 | 2 / 0 / 1 | 4 | 0 | **7** | +11.1 [−4.4, +28.1] |
| e2e（有 20 s 兜底） | 66.2 | 89.9 | 4 / 0 / 1 | 3 | 0 | 8 | +9.7 [−5.5, +27.0] |
| e2enofb（无兜底） | 36.4 | 48.5 | 2 / 0 / 0 | 2 | 5 | 4 | −20.1 [−42.5, +1.5] |

按登记的读法（baseslow 的违规数 ≤ e2e）判：**「e2e 对 base 的 +9.7 可以用开得慢解释」**。一个完全不看路、只是开得和 e2e 一样慢的 base，DS 67.7、违规 7 次，与 e2e 的 66.2、8 次同一水平；
22535 的追尾、27297 的撞行人，baseslow 同样避免了。也就是说，这 10 条上 openpilot 的纵向 modifier 没有显出「看见了」的贡献。去掉 20 s 兜底以后（e2enofb），13 次锁存只有 5 次靠 openpilot 自己的信号放行，
5 条路线 blocked，DS 掉到 36.4；它违规最少（4 次）只是因为大部分路线没开完（RC 48.5）。有兜底时 41 次放行里 17 次是 openpilot 放的、24 次是兜底。

### 7.2 幽灵停车：一半是 openpilot 看见了什么，一半是闭环把它放大，再被 CARLA 的低速刹停和锁存固定下来

事件：e2e 里真值无障碍处的 30 次锁存（[phantom_events.csv](results/op_arb/diag/phantom_events.csv)、[phantom_summary.csv](results/op_arb/diag/phantom_summary.csv)）。

| 读数 | 闭环 e2e | shadow：base（8 m/s 经过同一位置） | shadow：baseslow（约 1–2.5 m/s 经过） |
|:--|--:|--:|--:|
| **登记的读数**：plan 开始 binding 那一步「想停」（v(5 s) < 1 m/s 或 < 0.3 · v(0)）的比例 | 3%（1 / 30） | 3%（基线 0%） | 20%（基线 1.7%） |
| 那一步的 plan 3 s 内减速量（中位，负 = 要加速） | −1.83 m/s | +0.18 | −0.66 |
| meta 「驾驶员在踩刹车」（中位） | 0.36 | 0.17 | 0.31 |
| lead_prob（中位） | 0.12 | 0.04 | 0.26 |
| 事后：最后一次减速的整段里 plan 曾要求降到当前速度 60% 以下的比例（min v(3 s)/v(0) 的中位） | 80%（0.34） | 33%（0.71） | 50%（0.69，18 段可比） |

按登记的判格，两个 shadow 都 ≤ 20%，字面上落在「闭环自我强化」一支。但这个登记读数量错了时刻：plan 开始 binding 时，**闭环里的 plan 自己也还不想停**（3%，而且在要加速），
binding 只是因为 plan 加速得比 base 慢。所以另加了事后读数（标明事后，不改判格）：看停车前最后一段减速的全程。

读法：
1. **不是凭空停。** 同一位置只看不开时，openpilot 也在 33%（8 m/s 经过）到 50%（慢速经过）的位置要求明显减速，慢速经过时 20% 的位置「想停」，是它自己基线的 12 倍。这些位置多在路口前后（30 次里 13 次在左转命令前后，4 次在路口内）。
2. **闭环把减速放大。** 自己开的时候，同样的位置 80% 要求减到 60% 以下，最深到当前速度的 34%（只看时是 70%）。车一慢，模型的历史就看到自己在减速，接着预测更慢，这是登记里怀疑的自我强化，证据是减速深度的差，不是有无的差。
3. **真正把「减速」变成「停车」的是两件和 openpilot 无关的事。** 一是 CARLA 的低速刹停：e2e 里有 39 次车速在 0.15 s 内从 > 1 m/s 掉到 < 0.2 m/s，当时刹车只有 0.05–0.3（base 臂里只有 1 次），
   plan 要的其实是降到 1 m/s 左右（事件里 plan 的 v(5 s) 中位 4 m/s 以上），不是停。二是我们的锁存：只要 plan binding 后车速 < 0.2 m/s 就按「openpilot 让停的」处理，交给它的静止先验去放行，而第 2.1 节已经说明它在静止时几乎不会说走。
   这两件事都是本研究的仲裁设计与仿真器的问题，修正方向写在第 4 节的「下一步」，设计已搁置，没有再跑。

### 7.3 对设计的含义（供以后重新调时用）

- 这 10 条上，openpilot 纵向 modifier 的分数收益等于「开得慢」的收益。要说明 openpilot 看见了东西，读数必须带同均速对照；而且要换一套 base 慢开也躲不过的考题（例如必须按时通过的让行、对向车流）。
- 锁存不该在「plan 要减速到低速」时触发；CARLA MKZ 在 1–2 m/s 的轻刹会直接刹停，执行层要么避开这一段，要么锁存判据改成「plan 自己要求停」（v(5 s) ≈ 0），而不是「车停了」。

## 10. 标定链审计：CARLA rig 是不是已经等价于「收敛后的 liveCalibration」（2026-09-29，只读源码 + CPU 数值，没有 CARLA）

问题来自用户：真车上 calibrationd 是为了吸收「用户装得不准」而在线学 rpy；模拟器里外参是构造出来的精确值，应当直接喂先验，不需要任何 warm-up。结论先说：**我们已经在喂了，不需要 warm-up，也没有要改的代码。**

**源码版本**：box 上没有 openpilot 源码（只有 ONNX 权重，Lebowski 的 host 队列逻辑移植自 516ec1e6）；本节对照 `commaai/openpilot` master（2026-09-29 拉取，`openpilot/selfdrive/locationd/calibrationd.py`、`selfdrive/modeld/modeld.py`、`common/transformations/{camera,model}.py`）。master 里 `liveCalibration` 已改名 `extrinsicsCalibration`，字段不变。Cinque / Lebowski 的图像输入与 warp 用法与 master modeld 一致（`frames.py` 就是从这里移植的，见其 docstring）。

### 10.1 调用链

1. **calibrationd**（4 Hz）用 `cameraOdometry`（modeld 的 pose 头，即模型自己看出来的相机运动）在「直行且 > 15 mph 且偏航率小」时估：`rpyCalib` = (0, −atan2(trans_z, trans_x), atan2(trans_y, trans_x)) 复合到当前 rpy 上（roll 恒设 0，注释里写明模型输入不做 roll 校正），块平均（100 帧一块，至少 5 块有效才 `calibrated`）；同一处还平均了模型输出的 `wideFromDeviceEuler`（广角相对 device 的欧拉角）与 `height`（来自 pose 头的 `road_transform` 的 z）。初始值 rpy = 0、wide = 0、height = 1.22 m；`calStatus` 只是 uncalibrated / calibrated / invalid / recalibrating 的标记。
2. **modeld**：只取 `rpyCalib`，road 与 wide 两个相机**共用同一个** `device_from_calib_euler`：`get_warp_matrix(rpy, K_cam, bigmodel_frame)`，K 来自 `DEVICE_CAMERAS` 的硬编码（road 焦距 2648，wide 567，1928×1208，主点在图像中心）。`wideFromDeviceEuler`、`height`、`calStatus` **不进 warp，也不进任何模型输入**（其余输入只有 desire、traffic convention、lateral control params）。在收到第一条 calibration 之前 `model_transform` 是全零，输出 `valid = false`（是「有没有收到」，不看 `calStatus`）。
3. **模型输出**：plan / lane lines / lead 都在 calib 系（device 系按 rpyCalib 转正的路面对齐系，原点在相机，x 前 y 右 z 下）；pose 头里的 `wide_from_device_euler`、`road_transform` 是模型对自己外参的估计，只被 calibrationd 消费。
4. **下游**：locationd / paramsd / lagd / controlsd 里的 `device_from_calib = rot_from_euler(rpyCalib)` 只用来把 device 系的 IMU 角速度、cameraOdometry 转到 calib 系（车身系），`lagd` 另用 `calib_valid` 作门；这些是真车控制与状态估计链，我们的闭环里没有（plan 直接进我们自己的控制器）。
5. **写死常数处**：相机焦距与主点（`camera.py`）、model 内参 910 / 455、`MEDMODEL_CY = 47.6`、calibrationd 的初值高度 1.22 m。**没有任何一处用相机高度做输出换算**——plan 的 z 我们没有用，高度只是模型从图像里自己推断的量。

### 10.2 逐项对照

| 量 | openpilot 期望（收敛后） | 我们喂的 | 一致？ |
|:--|:--|:--|:--|
| `rpyCalib`（road 与 wide 共用） | 相机光轴相对「车行进方向」的残差；CARLA 相机水平、正对车头，车行进方向即光轴 → 收敛值 0 | `scripts/zeroshot_policy_server.py` 的 warp 用 `get_warp_matrix(np.zeros(3), …)` | 是（roll 本来就恒 0） |
| warp 矩阵 | `model.get_warp_matrix` | `frames.get_warp_matrix`；与上文 master 公式独立重推对比，max abs diff 2e-13（rpy = 0 与两组非零 rpy，road 与 wide）；服务端预算的 gather 下标与参考矩阵逐位相同 | 是 |
| 内参 K（road / wide） | 2648 / 567，主点 (964, 604) | CARLA fov 反推焦距 2648.000 / 567.000，相对误差 3e-8；主点为图像中心 | 是 |
| 地平线（模型帧行） | road 47.6，wide 151.8 ↔ 相机主点行 604 | 反算 warp 后恰落在 (964, 604) | 是 |
| road 与 wide 的相对外参（`wideFromDeviceEuler`） | 真车上是小的固定值（模型估的） | 两台 CARLA 相机位置与朝向完全相同（spec 逐项差 0），等价于 wide_from_device = 0；没有任何代码读它 | 是（构造上精确；模型自己估的值我们没存，见 10.4） |
| 相机高度 | calibrationd 初值 1.22 m，只是模型估计的输出，不进 warp | CARLA 相机装在 1.433 m（挡风玻璃上沿，理由见 `zeroshot_rigs.py`）；输出换算不用高度 | **不同，但不是输入**：比名义高 0.21 m，20 m 处地面点在 road 模型帧上下移 9.7 px（wide 4.8 px），模型得自己吸收 |
| `calStatus` / `validBlocks` | modeld 不看；selfdrived / lagd 才门控 | 不存在 | 无关 |
| 车身俯仰 / 侧倾 / 坡度 | 真车同样不补偿（calibrationd 是慢变量，且只学 pitch / yaw） | CARLA 车体在加减速时俯仰，相机随车 | 与真车同类，没补偿 |
| 广角镜头 | comma wide 是鱼眼，被 openpilot 当 567 焦距的针孔用（训练数据里就带着畸变） | CARLA wide 是真针孔（119.1°） | **不同**（外围几何不同），不是标定项，本次不量 |

### 10.3 数值验证

脚本 `scripts/check_op_calibration.py`（只读，纯 CPU，退出码 = 是否全通过）：上表「warp 矩阵、内参、地平线、相对外参、gather 下标」各项全过；另外量了「如果 rpy 错了模型帧会移多少」：俯仰错 0.5° 使 road 模型帧内容移 7.9 px（1° 是 15.9 px），偏航 0.5° 横移 7.9 px（与 910 px 焦距的 tan 一致）。
box 上没有存下来的 CARLA 相机模型帧（`p5_openpilot/check/…/model_frame_example.npy` 是缓存管线的产物，不是 CARLA rig 的渲染），所以「在存下的帧上看地平线行」这一步用解析反算代替：地平线与地面点落点的误差在 1e-13 px 量级。

### 10.4 结论与没做的事

- **(a)** 已经是精确先验：rpy = 0、内参与 road / wide 相对外参都与 CARLA 构造一致，且从第 0 帧起就在喂（没有 uncalibrated 阶段，也没有 modeld 那个「收到 calibration 之前 warp 全零」的窗口）。calibrationd 的 warm-up 在这里没有对应物，不需要延长 warm-up 来「等标定」。此前 wl2 prereg 里的「20 s 慢瞬态」只可能来自模型时序状态或场景，不是标定。
- **(b)** 没发现输入错配。两个非标定的差别值得记着：相机高度 1.433 vs 名义 1.22（模型得自己适应；此前 ego-gap 里 plan 横移回归增益 0.98，没有尺度偏的迹象，见 todos/2026-09-29-wl2-prereg.md），以及 wide 的针孔 vs 鱼眼。两者都是「与训练分布的差别」，不是标定没喂。
- **(c)** 不需要修。没有加 opt-in 开关：改任何值都会让输出偏离精确标定。要做 wl2 prereg 建议 2 的标定对照时，只要把 `zeroshot_policy_server.py` 里 `np.zeros(3)` 换成参数即可，现成的偏移量表在 10.3。
- 一项 CPU 之外才能补的读数：模型自己对外参的估计（`wide_from_device_euler`、`road_transform` 的 z，即它认为的相机高度）在 `decode()` 里被丢掉了。存下来能直接看「模型觉得 CARLA 相机装在多高、wide 相对 road 偏了多少」，是对本节 (b) 的第一手证据；随下一次有 CARLA 的 run 顺带写进 `plans.jsonl` 即可（prereg 建议 2 已有同一条）。
