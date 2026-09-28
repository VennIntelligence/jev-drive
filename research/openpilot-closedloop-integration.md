# openpilot 进 Bench2Drive 闭环：为什么起不了步、不转弯，以及「base + openpilot modifier」怎么接

2026-09-28。计划、登记与执行日志在 [todos/2026-09-28-op-closedloop.md](../todos/2026-09-28-op-closedloop.md)，小表在 [results/op_arb/](results/op_arb/)，
代码 `scripts/op_arb_agent.py`（agent 与各仲裁模式）、`scripts/op_arb_server.py`（openpilot server）、`scripts/op_arb.sh`（启动）、`jevdrive/op_arb_report.py`（读数）、`jevdrive/op_arb_figs.py`（图）。
本文供用户与 main 讨论；所有数字都是 GPU 6 测试卡上的小规模 pilot（phase 1 诊断 6 条路线、phase 2 评测 10 条路线，TM seed 0 单次），只能给方向。

## 结论先行

【P2 结论待填】

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

【P2 表：各方案 DS / RC / SR / 碰撞 / 闯灯 / blocked / openpilot binding 比例】

## 4. 推荐设计

【待 P2】

## 5. 要不要等适配线（op-adapt B / C、世界模型想象训练）

【待写】

## 6. 讨论的开放问题

【待写】
