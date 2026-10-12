# ZEROS-MECH 预登记：on-policy S 族零分的三个机制假设（转弯时机与指令、绕前车反射、走哪条车道），在已存 18 决策 held rollout 上只用 CPU 读

2026-10-12 写，写定并 push 于任何读数之前。来源：用户看完 `YS-L18-s1` 的零分片段
（[research/onpol-long/zeros-ys-l18.html](../../../research/onpol-long/zeros-ys-l18.html)、
[experiments/onpol_long/results/read/zeros_YS_s1.md](../../onpol_long/results/read/zeros_YS_s1.md)）后提出三个机制，下午讨论约定先在已存
rollout 上读、再决定是否训练修法。本稿之前只看过：数据格式（driver 日志字段；scene loader 的 traffic objects 与 vector map 字段；一个 scene
一个时刻上 asl 里的 actor 位姿与日志轨迹插值相同）、已发布的 15 段片段描述与 43 个零分清单。没有算任何登记量。
记号同 [research/recipes.md](../../../research/recipes.md)；「推」= 由记录的数算出，「估」= 判断。

## 0. 问题与三个假设

S 族（`YS-L18`，及第 2b 轮 `YS2G-L18`）在 held 上剩下的零分主要来自哪种机制？每条假设一个读数、一条判词线；数不支持就写不支持，不默认确认。

- **H1 转弯时机 / 指令。** plan 头单峰（`jevdrive/openpilot/model.py` `mdn_mu` 一条 33 × 15 轨迹；模仿是对日志 8 个未来位姿的 huber 回归，
  `experiments/op_parity/scripts/pp_train.py` `Losses.dist`），转弯起点不确定时 plan 是一条抹平的缓弯（**H1-smooth**）。指令是第一个路点的 one-hot
  （`experiments/alpasim/lib/ap2_inputs.py` `route_cmd`：第一个 ≥ 5 m 的路点，y > 2 m 为 LEFT），路线消息从约 40 m 外开始（第 266 条：10 m 锚点只有 30%
  的决策在消息里），所以指令提前几秒亮起、却不告诉路口在哪（**H1-command**）；直路上 2.7° 的 ego 航向误差就能经力臂翻转指令（第 205 条）。主会话看的片段：
  47da19c4、f42258b6 入弯晚、走外侧；a3882e6a、a3dd6d53 提前切内侧；5ddfd1fe 18 个决策全是 LEFT 而日志直行 8 s、车拐进第一个左口；ded4ce0e 在决策 8
  （4.5 s）S → L，正是车开始偏到直行日志左边的时刻。晚而宽的右转也是已知的 S 外移（第 233、234、244 条：road hinge 把 48–70° 右转的 plan 推离内侧边缘）。
- **H2 绕前车反射。** S 的 agent hinge（`lib/agent_hinge.py`；on-policy 行上 `experiments/onpol_nuplan/scripts/on_train.py` 的 `self` 模式）罚 own plan
  0.5–4 s 扫掠足迹与前方 |横向| < 3 m 物体日志框的接近：与 ego 横向范围重叠的物体 margin 0.3 m，旁边的 0 m；自 P2H10S 起只看形状（`--shape`：去掉沿航向的
  位置梯度），所以清掉一个框只能横移、不能减速（第 230 条：去形状之前梯度 96% 向后）。on-policy 行没有模仿目标、没有车道项。车道内的慢前车正是「前方 3 m
  内的物体」，所以假设这门课教出了横向绕行反射。要称的反证：四个碰撞片段里 bf00c76f、8051c60f 向左漂进停着的车、没有绕行动机（日志偏右，车不偏），
  bd70654a 是左转进了停着的车所在车道，只有 7bbb6cfc 像挤过空隙；第 220 条 10 决策碰撞 18% 直撞前车、27% 绕行 / 转弯且 plan 扫掠穿过物体、45% 无意图漂移；
  第 202 条（C1）没有前车碰撞，全是出车道后的侧碰。第 1 轮池 agent hinge 在 8.4% 的行上激活（ol1 预登记补记 9）。
- **H3 哪条车道是我的。** S 的 road hinge 栅格由 ROADBLOCK、ROADBLOCK_CONNECTOR、INTERSECTION、LANE、LANE_CONNECTOR 组成
  （`experiments/onpol_nuplan/scripts/on_rows.py` `ROAD_LAYERS`），不带方向：对向车道在 on-policy 课里不花代价，人行道在栅格外、被罚。corridor hinge K 有害
  （第 264 条），车道中心 hinge C 被标签闸排除（日志 46% 违反）；两者都不是方向信号。road hinge 的课确实迁移到了访问状态（出路零分基底 33 → YS 14.5），
  所以若零分确实去了对向车道，带方向的栅格是候选修法。

## 1. 数据（只读，box）

- **label（8 个）**：`YS-L18-s0/s1`、`YNONE-L18-s0/s1`、`YS2G-L18-s0/s1`（`$DATA_DIR/runs/onpol_long/read/<tag>/run`），`YR10m10-F-s0/s1`
  （`$DATA_DIR/runs/alpasim_long/held/h18/<tag>`）。族：**YS**（第 1 轮，两 seed 合并）、**YS2G**（第 2b 轮，两 seed 合并，现最佳）、**YNONE**（不加行的续训，
  第二对照，不作基线）、**基底 = `YR10m10-F-s0`**（YS 的起点，「课改了什么」的参照）；`YR10m10-F-s1` 是同配方另一训练 seed，只作描述。
- **scene**：8 个 label 都有计分 rollout 的 held scene（`alpasim_nuplan/onpol-held@v1` 的 460 减去路线折返、没有 rollout 的，ao_e1 同规则，预期 452），39 个 log；
  每个 label 一个 rollout。
- **每个决策**（`driver-logs/drive.jsonl`，同一 scene 取最后一个 session，ao_e1 规则）：`anchor`（后轴位姿，rollout local 系）、served `poses`（0.5–4 s 的
  8 个位姿，anchor 系）、`cmd`（0 LEFT / 1 STRAIGHT / 2 RIGHT / 3 UNKNOWN）、`route0`、`hist`。
- **评估器序列**（`metrics.parquet`，2 Hz，t_k = 17 000 + 500 000 k µs，另有 t = 0）：dist_to_gt_trajectory、offroad、collision_any / front / lateral / rear、
  left_corridor_laterally、wrong_lane（评估器自己的逆向车道旗标，只作参照）；rollout 汇总 `aggregate/results-summary.json`。
- **模拟器状态（特权，只作标签）**：asl 只留了失败 rollout（YS 每 seed 约 45 个），所以 actor 状态从 run 自己的 scene loader 重建：配置
  `physics_update_mode: NONE`、`min_traffic_duration_us: 0`，其他车辆回放日志（docs/alpasim.md），即 `data_source.traffic_objects` 的日志轨迹（10 Hz，local 系）
  在 t_k 线性插值；闸 G-act（第 7 节）在所有留了 asl 的 rollout 上检验。日志路径 = `data_source.rig.trajectory`（后轴，10 Hz，0–9.5 s）。地图 =
  `data_source.map`（trajdata VectorMap）：车道来自 nuPlan baseline paths，含 lane connector，中心线按行驶方向排列；车道多边形用评估器自己的
  `eval.scorers.offroad._get_lane_polygon`，中心线方向用 `_get_center_line_yaw_at_projection`。ego：车辆配置 5.176 × 2.297 m，后轴到车尾 1.127 m，
  即车头在后轴前 4.049 m。路线：ao_e1 的 `RouteGeneratorMap` 复现（G0 已在第 266 条过，99.89%）。
- **行侧（H2）**：第 1 轮池 `$DATA_DIR/runs/op_parity/cache/on1_L18Y`（17 306 行，基底在 train 1031 scene 上的 rollout；`agent.npz` 32 个 actor 在
  t0 + 0, 0.5 … 4 s 的框，行自身系；`tab.npz` 的 served plan、speed）与 `$DATA_DIR/runs/onpol_nuplan/rows/L18Y/active.npz`（每行 own plan 的 agent
  hinge，0.3 / 0 m，41 步平均）。

## 2. 共同定义

- **在场决策**：k ≤ k_end，k_end = 评估器截断步（第一个 dist ≥ 4 m 或 collision_any 或 offroad 的 k，long_read.py `cut` 同规则），没有则 17。决策级比例
  只在在场决策上算（之后的状态分数看不到）；R1 的时机量用全部 18 个决策（车的实际航迹）。
- **零分与事件**（ol_clips.py `table` 同规则，YS-L18-s1 可与 zeros.csv 逐条对齐）：得分 0 的 rollout，对汇总里计了的三类（collision_at_fault、offroad、
  left_corridor_laterally）各取计时序列（collision_any、offroad、left_corridor_laterally）第一个 > 0 的时刻，类型 = 最早者，事件决策 k_ev = 该时刻所在的决策
  （t = 0 记 0）。另记 at-fault 接触决策 k_c = 第一个 collision_front 或 collision_lateral > 0 的 k。
- **日志路径与横向偏差**：日志后轴折线，末端沿末航向直线延长 40 m（on_rows.py 同惯例）；点投到折线（ao_e1 `project`，按上一决策的站位限窗）得站位 s 与
  带符号横向（左正）。ego 偏差 e(k) = anchor 的横向。
- **对向一侧**：右行（美国三城）为左，新加坡（左行）为右；s_onc = +1 / −1。
- **CI**：log 聚类 bootstrap，10 000 次，`default_rng(0)`，jevdrive.stats 的约定（抽 log，和之比）；两个 label 比较时同一次抽样作用于两边（各 label 访问的
  决策不同、不能逐单位配对，所以是两个和之比的差）。中位数的 CI 同样抽 log、取加权中位数。
- **族合并**：YS = YS-L18-s0 与 s1 的决策 / 事件合在一起；YS2G、YNONE 同。

## 3. R1 时机与指令（H1）

定义：
- 日志航向 ψ_log(t_k)：GT 轨迹 yaw（展开）在 t_k 插值；总转角 Δ = ψ_log(t_17) − ψ_log(t_0)。**转弯 scene** |Δ| ≥ 45°，**直行 scene** |Δ| < 15°，其余为
  「缓弯」，不进 R1 的线。转向 σ = sign(Δ)。ego 航向 ψ_e(k) = anchor yaw（展开）。
- 累计比例 c_log(k) = σ(ψ_log(t_k) − ψ_log(t_0)) / |Δ|，c_e(k) = σ(ψ_e(k) − ψ_e(0)) / |Δ|。**t_log / t_e**：c 第一次到 0.5 的时刻（决策间线性插值）；ego 到不了
  记「未转够」（censored，中位数里按 +∞，另报比例）。**delta = t_e − t_log**（s）。
- **空间版**：t_log 时日志的站位 s_log½，t_e 时 ego 投到日志路径的站位 s_e½（决策间插值），**Δs = s_e½ − s_log½**（m，正 = 沿路更靠后才转够 = 空间上晚）。
  **细化（读数前写定）**：入弯早晚是空间概念（片段里的「晚而宽」「提前切内侧」都是路上的位置），时间版混了车速（车比日志慢就「晚」），所以下面的早 / 晚分类
  用 Δs 的符号；时间版 delta 照单报分布，并给出按 delta 符号分类的同一张表作敏感性。
- t_cmd = 第一个 cmd ∈ {LEFT, RIGHT} 的决策时刻；报 t_e − t_cmd、t_log − t_cmd（指令提前量）、指令方向与 σ 一致的比例。每个决策另存复现路线在 ego 站位前方
  60 m 内的航向变化 route60。
- **转弯 scene 上的 corridor 零分**：事件时 ego 在日志弧的内侧（σ·e(k_ev) > 0）或外侧，与 Δs 的符号组成 **早-内 / 晚-外** / 早-外 / 晚-内（未转够算晚）。
- **直行 scene**：在场决策中 cmd ≠ STRAIGHT 的比例，按 label。直行 scene 上的 corridor 零分：**漂移起点 k_d** = 最小的 k ≤ k_ev，使 |e(k)| > 1 m 且此后到 k_ev
  一直同号、不减（相邻决策回落 ≤ 0.1 m 算不减）；没有则 k_d = k_ev。记 k ∈ [k_d − 3, k_d] 内有无 cmd ∈ {LEFT, RIGHT}（起点时或之前 1.5 s 内有非直行指令，
  ded4ce0e 型）、是否在窗内由 STRAIGHT 翻过来、指令方向是否与漂移方向相同、该处 route60（≥ 20° 记「路线前方真转」，否则记「力臂翻转」，第 205 条）。
  **基率**：直行 scene 的干净 rollout（三类旗标都没有）上，k = 3..17 的每个 4 决策窗里有非直行指令的比例。
- **峰值角速度（H1-smooth）**：转弯 scene 上 ω(k) = (ψ(k+1) − ψ(k)) / 0.5 s，k = 0..16，ego 与日志各取峰值 |ω|；label 的**不足比例** = 1 − Σ峰_e / Σ峰_log
  （和之比）。副量：曲率版（1 s 窗的航向变化 / max(弧长, 2 m) 的峰值，排除车速）；描述：日志未来 4 s 转角 ≥ 30° 的决策上 plan 自身 4 s 转角与日志从同一时刻
  起 4 s 转角之比。

报：delta 与 Δs 的中位数 / IQR，按 label 与族，YS − 基底（中位数差，CI）；早-内 / 晚-外 计数按 label；直行 scene 的指令比例；漂移起点的指令窗表与基率；
不足比例按 label、YS − 基底。

线（在 YS 合并上判；YS2G 同样算、只描述）：
- **H1-command 成立**：(a) 直行 scene 上的 corridor 零分里 ≥ 一半在漂移起点时或之前 1.5 s 内有非直行指令，**且** (b) 转弯 scene 上的 corridor 零分里 早-内
  个数 ≥ 晚-外 个数。(a) 或 (b) 的事件不足 4 个时该条记「不可读」，判词只按另一条给「部分」。(a) 过线而指令与漂移同向的比例 < 一半、或基率 ≥ (a) 的比例时，
  判词注明「指令与漂移只是同时出现」。
- **H1-smooth 支持**：8 个 label（含基底）的峰值角速度不足比例都 ≥ 0.20 且 CI 下界 > 0。**不是 S 课造成**：YS − 基底的不足比例差的 CI 含 0（或 YS 不比
  基底差）。曲率版与之不同向时判词注明（那说明是车速、不是 plan 形状）。

## 4. R2 绕前车（H2）

定义：
- actor 在 t_k 的状态（第 1 节，闸 G-act）：中心、yaw、长宽；速度 = 0.5 s 后向差分（轨迹开头则前向）。变到 ego 后轴系（x 前、y 左）。ego 速度 = anchor 的
  0.5 s 后向差分（k = 0 用 `hist`）。
- **前车框**：0 < gap ≤ 15 m（gap = 物体四角最小 x − 4.049 m）、物体中心 |y| < 1.5 m、|Δψ| < 30°。**前车条件（lead）** = 框里有车且
  (v_lead < v_ego − 1 m/s 或 (v_lead < 0.5 m/s 且 v_ego > 1 m/s))，多辆取 gap 最小者。**对照（no-lead）** = 框里没有车（不论速度）。框里有车但不满足速度条件
  记「跟车」，单列。细化：对照取「框里没车」而不是「不满足 lead」，免得跟车状态混进对照。
- plan 横向量，t = 2、3、4 s：y_h(t) = plan 点在 ego 航向系的 y；y_log(t) = plan 点到日志路径的带符号横向；**d(t) = y_log(t) − e(k)**（plan 相对日志路径的
  横向位移）。**主量 |d(3)|**；带符号量 s_onc·d(3)（正 = 向对向一侧）；y_h、y_log 作副量同表报。**细化（读数前写定）**：单子的「|offset at 3 s|」在弯道上
  会被道路曲率（y_h）或车已有的偏差（y_log）主导，线用 d(3)。
- **碰撞**：所有 label 中汇总 collision_at_fault > 0 的 rollout，窗 k ∈ [k_c − 6, k_c]（接触前 3 s）。窗内有 lead →「有前车」；窗内有 |d(3)| ≥ 0.5 m →「横移」。
  四类：**绕前车**（有前车且横移）、直撞（有前车、不横移）、无前车横移、漂移（都没有）。被撞物体 = k_c 时与 ego 框重叠最大（无重叠取最近）的 actor，按它在
  k_c − 2 时 ego 系里的位置归类（本车道前方 / 邻道 / 横穿 / 对向 / 其他），并记是否静止（< 0.5 m/s）、是否就是窗内的前车。
- **行侧（描述，不进线）**：`on1_L18Y` 中 active agent > 0 的行；用 `AgentHinge(agent.npz, margin 0.3, side 0.0)` 在 served plan 上重算每步每物体的 hinge，
  取最大处的物体（binding object），按它在 t0 的行自身系位置分：**本车道同向前车**（中心在 ego 框中心之前、|y| < 1.5 m、|Δψ| < 30°）、**邻道**
  （1.5 ≤ |y| < 3 m、|Δψ| ≤ 60°）、**横穿**（60° < |Δψ| ≤ 120°）、**对向**（> 120°；单子里与横穿合称 crossing）、其他；报比例，再只看 hinge 值最高的 10% 的行；
  另报 binding object 满足完整前车条件（含速度）的比例、binding 步是否在车道内 0.3 m margin 下。

报：按 label 与族，lead / no-lead / 跟车 三类决策的个数与 |d(3)|、s_onc·d(3)（均值，CI）；YS − 基底、YS − YNONE（lead 下与 no-lead 下，CI）；描述：差中差
（lead − no-lead，再 YS − 基底）、YS2G − 基底；|d(3)| 按 gap 分箱（0–5、5–10、10–15 m）的图；碰撞四类计数与被撞物体类别，按 label；行侧比例。

线（在 YS 合并上判）：
- **H2 成立**：(i) lead 下 YS − 基底的 |d(3)| ≥ +0.20 m 且 CI 下界 > 0，同时 no-lead 下 YS − 基底 < +0.10 m；**或** (ii) YS 的 at-fault 碰撞里绕前车类
  ≥ 一半且比例高于基底（YS 碰撞 ≥ 4 个才可读）。
- **H2 被否**：lead 下 YS − 基底的 CI 含 0，且绕前车类 < 1/4。其余记「未确立」。

## 5. R3 车道方向（H3）

定义：
- 点 p、航向 ψ 的车道类：取 p 周围 10 m 内的车道（`get_lanes_within`），用评估器的车道多边形判 p 在不在内；对每条包含 p 的车道取中心线在投影处的方向
  ψ_lane，Δ = wrap(ψ_lane − ψ)。**同向** = 有一条 |Δ| < 60°；否则 **对向** = 有一条 |Δ| > 120°；否则 **横穿**；都不含为 **无**（出车道）。这个优先级就是
  带方向的栅格会怎么判：任一同向车道覆盖的地方都合法。本地图里路口由 lane connector 覆盖，所以路口里的点按 connector 判，单子说的「无（路口 / 出车道）」
  在这里基本只剩出车道；另记「路口重叠」旗标（包含 p 的车道里有两条方向差 > 30°）作描述。
- 点：**后轴（主量）**、车头中心（后轴前 4.049 m，副量）、plan 4 s 终点（位置与 plan 航向）、日志自己在 t_k 的位姿（闸 G-dir）。

报：在场决策中后轴在对向车道的比例按 label（YS − 基底，CI）与按决策序号（图）；零分事件前 2 s（k ∈ [k_ev − 4, k_ev]）有任一决策后轴在对向车道的比例，按类型
与 label；R2 的「lead 下横移」决策（lead 且 |d(3)| ≥ 0.5 m）里 plan 终点在对向车道的比例；车头版；评估器 wrong_lane 的同一比例（参照：评估器用未 wrap 的
角差，自己注明误报多）。

线（在 YS 合并上判）：
- **H3 得到一个臂**：YS 的 corridor + collision 零分（按最早失败分类）里 ≥ 25% 在事件前 2 s 有后轴在对向车道（事件 ≥ 8 个才可读）。
- **on-policy 课让它变差**：YS 的对向决策比例高于基底且差的 CI 下界 > 0。

## 6. YS-L18-s1 的 43 个零分逐条分类（给主会话对照片段）

每个事件：scene、类型、k_ev、scene 类（转弯 L / R、直行、缓弯，Δ）；R1（Δs、delta、事件时内 / 外、类；直行 / 缓弯 scene 的漂移起点、起点窗内指令、是否翻转、
方向是否一致）；R2（事件前 3 s 内有无前车、横移、碰撞四类与被撞物体）；R3（事件前 2 s 是否在对向车道、事件时后轴的车道类）。规则化的「与哪条假设相符」：
H1-cmd（直行 / 缓弯：漂移起点窗内有非直行指令；转弯：早-内）、H1-wide（转弯：晚-外）、H2（窗内有前车且横移）、H3（前 2 s 在对向）、起步（offroad 在
k_ev = 0）、都不是（漂移）。一个事件可以同时符合几条。表与 zeros.csv 逐条对齐，类型或决策不一致的列出。

## 7. 闸与停的规则（写死）

- **G-cov**：≥ 99% 的 (label, scene) 有 18 个决策记录与 metrics；scene 级脚本错误 ≤ 1%，否则停、修代码、整体重跑（不挑 scene）。
- **G-act**（actor 重建）：所有留了 asl 的 rollout 上，asl 的 actor_poses 与重建在同一时刻位置差 ≤ 0.05 m 的 (actor, 时刻) ≥ 99%，asl 的 actor 都在重建里，
  且 EGO 的 asl 位姿与 driver `anchor` 差 ≤ 0.05 m 的决策 ≥ 99%。不过 → R2 的前车条件只在留了 asl 的 rollout 上算（只够碰撞那部分），lead / no-lead 对比
  不做，写明。
- **G-dir**（车道方向）：日志自己的位姿（452 × 18）里有车道包含的那些，同向 ≥ 95%、对向 ≤ 1%。不过 → R3 不做，写明原因。
- **G-row**：重算的每行 agent hinge 与 active.npz 差 ≤ 1e-4 的行 ≥ 99.9%。不过 → 行侧不报，写明。
- **G-zero**：重算的 YS-L18-s1 零分的类型与决策与 zeros.csv 一致；不一致的逐条列出（不停）。
- 定义、阈值、label 集合、主量在读数后不改；登记读数只跑一遍（调试只在 ≤ 5 个 scene 上跑，看代码不看汇总）；新加的切片标「探索」，不进判词。

## 8. 与三个候选臂的对应（读数前写定）

- **A**（on-policy 行上把本车道同向物体从横向 agent hinge 里拿掉，交给只管纵向的跟车距离项）：H2 成立 → 是；否则否（行侧比例只作描述）。
- **B**（带方向的 road 栅格）：H3 得到一个臂 → 是；否则否。
- **C**（把路线路点给 adapter）：H1-command 成立 → 是；只有直路那条成立 → 写明只支持「直路上的指令抖动」这一半；否则否。

## 9. 代价与产出

CPU only：scene loader 每 scene 约 7 s（第 266 条实测），452 scene 用 `jevdrive.par`（≤ 64 进程）几分钟；读约 600 个 asl 的 actor 位姿；行侧 17 306 行
CPU torch 秒级。产出：box `$DATA_DIR/runs/zeros_mech/collect/<ts>/`（per_decision.parquet、per_rollout.parquet、gates.json、errors.csv）、
`$DATA_DIR/runs/zeros_mech/rows/<ts>/`；提交 `results/zm_report.md` 与小 csv / json、`figs/`；决策条目与中文 HTML 页 `research/zeros-mech/index.html`。
