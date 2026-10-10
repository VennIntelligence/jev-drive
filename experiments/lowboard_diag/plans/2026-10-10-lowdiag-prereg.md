# LOWDIAG：两块低分闭环榜的失分诊断，一套共用分类（预登记，2026-10-10）

lane LOWDIAG。任何逐 rollout 读数之前写定并 push。不训练；仿真器的物体框、地图、日志路径只作标签。模板是第 227 条。
**范围（2026-10-10 用户收缩，仍在任何读数之前）**：只用已有 rollout 与日志，每榜一个 seed（PAI：base `P2H10-F-s0`，served
`JEV_VCONT=1.0 JEV_LEAD=1`，60 scene，FIX1 的 `ab_*_s0`；HUGSIM：`SH30-F-s0` `spec_plan_smooth` all64）；不重跑，缺的信号记
「未确定」（HUGSIM 的闭环 P0 臂已取消）；navhard 不做；回答顺序：PAI corridor 问题 > 每榜各类份额 > 底模输出是否正确
（时间不够时只做 longitudinal 类）。类定义、优先级与阈值即下文，与首版（commit fe0bd0aa，完整推导与细节在那里）相同。

**单位**：失败单位 = PAI 零分 rollout / HUGSIM end ≠ complete 的 run；失分 = 1 − score / 1 − HD。PAI 非零的 progress 失分与
HUGSIM complete 的扣分各单列一行，不强行归类。事件 E = 第一个置零 flag 的计分步 / 结束步；类型 K ∈ {agent, boundary（offroad /
bg）, corridor（left corridor / off_route）, none}。ref = 日志 ego 路径 / 录制路线；onset = E 前最后一个 |lat| < 1.5 m 的时刻；
ref 在 [s_on − 10 m, s_on + 20 m] 的类型：turn（R < 15 m 或 |Δψ| ≥ 30°）、bend（15–50 m 或 10–30°）、straight。

**原始 flag（多选，全部报计数）**：O1 driver 异常；O2 仿真 / scorer（PAI hand-over spin；HUGSIM 对向 / 横穿车在 ego 刹停的反事实下
仍撞上、被追尾）；L1 碰撞且物体 ≥ 2 s 前已在已行驶路径的扫掠带内、正面、从 E − 3 s 起 4 m/s² 刹停可免；L2 turn / bend 上的
boundary / corridor 事件且 onset 处 v² / R > 4 m/s²；L3（只作 flag）onset 车速 > 日志 1.2 倍；L4 无事件的不走 / 走得慢
（follow：多数时间 lead 限速在起作用或走廊 20 m 内有物体；free：其余）；R1 E 时 |lat| ≥ 1.5 m 且 onset 后与 ref 航向差 ≥ 20°
（另一个出口 / 该转没转 / 不该转而转）；R2 ref 为 turn 的 boundary / corridor 事件；D ref 为 straight 的横移（航向差 < 20°）；
C1 其余碰撞；C2 ref 非 turn 的 boundary 事件。

**互斥类（按顺序取第一个命中）**：(1) other = O1 / O2；(2) longitudinal = L1 / L2 / L4；(3) route = R1 / R2，以及 K = corridor 的
全部剩余单位（子类 drift / bend）；(4) clearance = C1 / C2 与 K = boundary 的 D。已知偏向：先离开日志车道再撞上的归
longitudinal（若刹停可免），从多选表可读出。

**in-plan 与 lead time**：served plan 的 ego 框扫掠命中与 K 相同的事件（agent：与被撞物体同时刻的框相交；boundary：PAI 碰到
road edge / HUGSIM 路面覆盖率低于 fd_hugsim 的线；corridor：plan 上的点离 ref ≥ 4 m）；in-plan = E 前 3 s 内至少一个决策命中；
lead = E − 最后一段连续命中的第一个决策（封顶 8 s）。

**底模输出是否正确（base right）**：底模 = shipped 权重不带 adapter。PAI：同一 token 的离线 replay，窗口 [E − 4 s, E − 0.5 s]
（corridor / boundary 用 [t_on − 2 s, t_on + 2 s]）；served plan 命中事件的决策里，候选不命中的份额 ≥ 50% 记为 right：
speed = served path 配 P0 速度剖面；path = P0 path 配 served 速度剖面（corridor 另加：P0 在共同弧长处到 ref 的横向误差小 ≥ 1 m 的
决策 ≥ 50%）；any = P0 plan 原样；L4：P0 的 2 s 弧长 ≥ served 的 1.5 倍的决策 ≥ 50%。HUGSIM：同一时刻的 P0 plan 没存，
只读 run 自己记录的 lead 头（E 前 ≥ 1.5 s 有一步 lead_prob ≥ 0.5 且 a_need ≥ 1.5 m/s²，仅 K = agent）；闭环 P0 一列未确定。

**PAI corridor 零分的子类（按顺序）**：在 onset 处量 ref 类型（另加 fork：ref 在 s_on + 30 m 处相对 ego 直线延长横移 ≥ 3 m 而
|Δψ| < 10°）、[t_on − 4 s, t_on] 内喂入 command 的众数 cmd_fed、driver 自己的规则套在日志路径上的 cmd_true、发给 driver 的
route 在 s_on + 30 m 处到日志路径的横向偏差、以及同一 token 上强制 command（左 / 直 / 右）的 plan：effect = 强制成 cmd_true 与
强制成直行的 4 s 终点横向差的中位。(1) route 信息缺失或错误：ref 非 straight 且（cmd_fed ≠ cmd_true 或 route 偏差 ≥ 3 m），
或 ref straight 而 cmd_fed 非直行且 ego 朝该方向离开；(2) 信息正确但没用上：cmd 正确、effect < 1 m；(3) 转不过去：cmd 正确、
effect ≥ 1 m，plan 朝正确方向但不够 / 太晚 / 切内角；(4) drift：ref straight 的其余。读法：(1)+(2) ≥ 50% → route 工作线存在，
瓶颈是 route 信息 / command 通道；(3) ≥ 50% → 转弯能力；(4) ≥ 50% → 第 205 条的横向漂移，不是 route；都不到 → 只报份额。

**核对**：每榜先取 5–6 个代表单位出 BEV，用眼核对每个 flag；此步只改实现错误，阈值或定义要改则写补记并注明看过哪些单位。
**事先的预期（不是判据）**：PAI corridor 类最大，drift 与 route 信息都不小；HUGSIM 最大的是 other（脚本对向车），其次
longitudinal；base right 在 longitudinal 类里高于其余类。

## 补记 1（2026-10-10，stage 1 之后、全量表之前）

看过 PAI s0 的 9 个代表单位的 BEV（213dfdac、24a50fcc、6b986b30、b988494a、605bf77a、1d6e30bc、b45734f4、a28b6685、a45776a3）以及全部 32 个
单位的首轮 flag 列表之后，改两处定义，其余不动：
1. **O2c（被追尾）加一个条件：被撞物体车速 > 0.5 m/s。** a45776a3 里 ego 低速向右偏出日志车道，车身侧后部擦到路边静止的车，
   原规则把它判成「被追尾」；静止物体不可能追尾。改后该单位按其余规则归类。
2. **fork 的参照改成 ref 自己在 s_on − 10 m 处的切线**（原文是 ego 在 onset 的位姿的直线延长）。6b986b30 的 ref 是直路，
   但 onset 时 ego 的航向已经偏了约 9°，按 ego 的延长线量 ref「横移」≥ 3 m，被误判成 fork。
HUGSIM 的 O2b（对向 / 横穿车）没有写反事实里物体在 E 之后外推多久：主读数取 E + 10 s（执行者看过 2510_2710-extreme-00 之后从
3 s 改的），E + 3 s 与只到 E 两个口径一并报，不另改定义。
另加只作描述、不参与分类的列：`obj_on_ref`（被撞物体是否站在日志路径的扫掠带里，用来把 L1 分成「日志车道里的前车」与
「ego 自己偏过去撞上的」；原 L1 对静止物体的「已在已行驶路径里」恒成立，这一点在读 longitudinal 类时要用这一列）、`gap_ahead`
（onset 前 2 s ego 正前方走廊内最近物体的间距）、`w_max`（事件前最大 yaw rate）。
