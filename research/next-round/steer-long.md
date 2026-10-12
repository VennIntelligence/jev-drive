# 急转与纵向（专家 B）：plan 到执行器这一段缺什么

2026-10-12 起稿。执行者：专家 B（协调会话派）。状态：**预登记已写（第 3 节），读数之前提交**；第 4 节以后在读数后补。
记号同 plan.md：d### 指 `research/decisions/###.md`；「测」有记录读数，「推」由测得的数算出，「估」没有测量依据。盒上路径以 `$DATA_DIR`（`/root/autodl-tmp/ujs`）起，
Tokyo 路径以 `/data` 起。本稿不训练、不起闭环、不碰别的 lane 的 job。

## 0. 一句话

（读数后填）

## 1. 前提与已有证据

**用户看完全部 PAI 回放后的观察（前提，不重测）**：(1) 车几乎总比人类日志慢，越开越慢，会无故停下；(2) 低速时小扰动让控制器失稳，尤其转弯中途，打转、翻车；
高速时前方有不确定的东西，plan 开始横扫，车突然打一个这个速度下根本不需要的大方向，然后出路。用户的假设：真 openpilot 车在高速上不会这样，plan 到执行器这一段有东西错了或缺了。
协调会话的读法：不是单位错，是缺阻尼（链路里没有任何随速度变化的转角 / 横向加速度 / 转角速率限制），加上模型对自身偏航历史的环路增益 > 1（d100 / d106 / d111）。

**控制链路，读源码（AlpaSim 0bb4c4b `src/controller/alpasim_controller`，测）**：
- driver 返回的轨迹 = plan 的 8 个 0.5 s 点加锚点（t0 时的 ego 位姿），在 plan 自己的时钟上线性插值成 0.1 s 点，固定在 local 系，直到下一次推理（`navsim_transfuser_challenge/trajectory.py`）。
  nuPlan 每 0.5 s 推理一次（2 Hz），PAI 每 0.1 s 一次；PAI 上另有 serving 开关 `JEV_VCONT=1.0 JEV_LEAD=1`（d226，Tokyo 的 pai281 读数都开着），nuPlan 全关。
- MPC（`nonlinear_mpc.py`）：20 × 0.1 s horizon，每 0.1 s 重解；cost 只从第 10 步起计（1.0–2.0 s 的位置与朝向），权重纵 2 / 横 1 / 朝向 1、加速度 0.1，
  命令变化（rterm）转角 5、加速度 1；转角状态与命令都限 ±0.785 rad，与速度无关；**没有转角速率限制、没有横向加速度限制、没有速度跟踪**。
- 车辆模型（`vehicle_model.py`，Ford Fusion 参数）：5 m/s 以下运动学自行车，以上线性轮胎的动力学自行车（侧偏刚度 93.5 / 176.2 kN/rad，**没有轮胎饱和**，所以仿真里横向加速度没有物理上限）；
  转角一阶滞后 τ = 0.1 s，加速度同。轴距 2.85 m。
- rterm 的量纲：5 × (Δδ)²，Δδ 以 rad 计；在 30 m/s，Δδ = 0.05 rad 就是 v²·tanδ / L ≈ 16 m/s² 的横向加速度，cost 只有 0.0125，而 1 m 的横向位置误差在 11 个 horizon 步上计 11。
  即在高速段这个惩罚几乎不约束转角（推）。
- 每个会话写一份 `controller/alpasim_controller_<session>.csv`：每 0.1 s 的位姿（local 系）、vx、vy、wz、转角命令 `u_steering_angle`、转角状态 `front_steering_angle`、加速度。
  held 452 的 18 决策 run 与 Tokyo 的 pai281 run 都留着（测：nuPlan 每 label 452 份，91 行 = 9 s；PAI 每 label 281 份，200 行 = 20 s）。

**openpilot 自己在这一段做的事（tmp/opsrc ec95db3f `selfdrive/controls/lib/drive_helpers.py`，lib/op_ctrl.py 原样搬过）**：`clip_curvature`：
曲率变化率 ≤ MAX_LATERAL_JERK 5.0 m/s³ / v²（每 10 ms 一步）、|κ| ≤ MAX_LATERAL_ACCEL_NO_ROLL 3.0 m/s² / v²（加路面横坡补偿）、|κ| ≤ 0.2 1/m；
再加车上的转角速率限制（Toyota LTA：5 m/s 时每 10 ms 0.3° 方向盘、25 m/s 时 0.15°）和真实车辆的转向延迟（lagd 辨识 0.15–0.65 s）。AlpaSim 这条链路上这几样一样都没有。

**已有证据**：
- d220（SWV1）：66 个 at-fault 碰撞 rollout 全部在事件前 3 s 内已有 served plan 的扫掠框压到被撞物体，跟踪误差 p95 2–4 cm；横向加速度限速（按 1–4 m/s² 降速）一个都避不开。
  → 在 nuPlan 的碰撞上「plan 先动」已经测过，但那是限速（纵向），不是限曲率（横向），且没有分速度段。
- d237（LOWDIAG，PAI served 单 seed）：31 / 32 个零分在事件前 3 s 内 served plan 已命中同类事件，中位提前 1.8 s；跟踪偏差中位 0.054 m；
  4 个 spin、2 个 corridor 零分是 22.6 / 30.3 m/s 起步的 spin。
- HUGSIM 打转线：d90 打转先在 plan 里（20 / 21 次计划先偏 12–65°）；d96 低速时把历史帧去旋转，10 个打转 9 个消失，偏差每步 ×1.5、历史偏航滞后两步跟上；
  d100 / d111 环路每步 ×1.6–2.8，真实 WOD 起步上同样存在（模型自身性质），能否发散取决于控制器传递 c，HUGSIM 0.19 比临界 0.02–0.03 大约 10 倍；
  d106 适配训不掉；d110 navhard 一半以上 token 要早期大转向（任何规则的「必须不伤」集合）；d113 / d114 规则侧均匀低通压住打转但伤正常转向、死区挡不住；
  d117 纵向起步规则不压打转。
- 纵向：PAI 失分里纵向占 43 %（d237；d241 更正后 33.6–44.8 %）；d278 巡航起步 5–10 m/s 比人慢 0.87–0.92 m/s；d280 巡航落后对 GT 贡献 0；
  long_oracle：navtest 上任何统一速度缩放都不胜过 served；d226：nuPlan 上 VCONT −0.007、LEAD −0.086，PAI 上 VCONT + LEAD +0.148。

## 2. 假设

横向（对应观察 2）：
- **HL1 plan 先动**：大转向是 plan 先给出的（模型在 1–2 s horizon 上要了一个大曲率），MPC 只是照做。杠杆：driver 侧限 plan 的曲率与曲率变化率。
- **HL2 控制器放大**：plan 正常，MPC 的转角超出 plan 所需（过冲、来回）。杠杆：官方旋钮 `rel_front_steering_angle_weight`。
- **HL3 偏航历史环路**：plan 对自车偏航历史的响应 > 1，表现为 plan 的横向量随 0.5 s 一步一步同号放大（单调发散，d96 型）或正负交替且幅度增长（振荡）。杠杆：低速去旋转（d96）或曲率变化率限制。

纵向（对应观察 1）：
- **HV1 模型自己要慢**：速度差主要是 plan 的速度低于人类（v_log − v_plan），不是 ego 没跟上 plan。
- **HV2 执行链让它慢**：差额主要在 v_plan − v_ego（MPC 没有速度项、只追 1–2 s 位置；PAI 上 VCONT 从 ego 速度起步、LEAD 只减速）。
- **HV3 无故停车的来源**：plan 自己停（P）/ LEAD 限速（L，仅 PAI）/ VCONT 混合（V，仅 PAI）/ MPC 没把位置差追上（M）。

## 3. 预登记（读数之前写，提交后才读任何数）

### 3.1 数据与 label（只读已存 rollout，不起闭环）

| 赛道 | label（checkpoint，serving） | scene | 来源 |
|:--|:--|:--|:--|
| nuPlan 18 决策 held | `YR10m10-F-s0/s1`（第 2 次提交的 checkpoint，开关全关） | 452 | `$DATA_DIR/runs/alpasim_long/held/h18/<tag>` |
| 同上 | `YS-L18-s0/s1`、`YS2G-L18-s0/s1`（开关全关） | 452 | `$DATA_DIR/runs/onpol_long/read/<tag>/run` |
| PAI 281 | `YR10m10-F-s0`（ybase）、`YS-L18-s0`（ys0）、`YS-L18-s1`（ys1），`JEV_VCONT=1.0 JEV_LEAD=1` | 281 | Tokyo `/data/runs/onpol_long/pai281/{ybase,ys0,ys1}/runs/*` |

每个 rollout 用到：控制器 CSV（按 session uuid 对上）、driver 的 `drive.jsonl`（anchor、served plan `poses`，PAI 另有 `poses_model` 与 `fix`）、
评估器 2 Hz 序列（`metrics.parquet`：`dist_traveled_m`、`gt_dist_traveled_m`、`min_distance_to_obstacle_m`、`offroad`、`collision_any`、`left_corridor_laterally`）、
summary 的 score 与 flag。nuPlan 另用 zeros_mech 的 `per_decision.parquet`（`$DATA_DIR/runs/zeros_mech/collect/20261012-155215`，日志位姿 2 Hz）作人类横向加速度的分布与零分类别核对。
navhard「必须不伤」集合：`$DATA_DIR/runs/hugsim-spinattr/navhard_ref.csv`（d110 的 5 912 个 token，PDM 参考在 1 / 2 s 的朝向、2 s 的位置、v0）。

### 3.2 每 0.1 s 的量（控制器 CSV 的每一行 t）

当前生效的 plan = t0 ≤ t 的最近一次推理（PAI 上 `infer` 为真的记录）。返回轨迹按 3 节第 1 条的规则重建（锚点 + 8 点，plan 时钟上线性插值）。
- v = vx，自车横向加速度 **A_exec = |vx · wz|**，转角 δ、转角命令 u、转角速率 **δ̇ = |Δδ| / 0.1 s**；
- 命令的横向加速度 **A_cmd = vx² · |tan u| / 2.85**；
- plan 在 MPC horizon 上的曲率 κ_path = 返回轨迹在绝对时间 [t + 1, t + 2] s 段的朝向差 / 弧长（plan 时钟夹在 4 s 内），**A_path = |κ_path| · v_ref²**，v_ref = 该段弧长 / 时长；
- 追上参考所需：参考点 = 返回轨迹在 t + 1.5 s 的位置，换到当前自车系 (x, y)，κ_reach = 2y / (x² + y²)，**A_req = |κ_reach| · max(v, 1)²**（参考点距离 < 1 m 时记缺失）；
- 曲率变化率：新 plan 与上一个 plan 在同一绝对时间段 [t + 1, t + 2] 上 κ_path 之差 / 两次推理的间隔，乘 v_ref² 得 **J_plan**（m/s³）；
- served plan 在锚点系 τ = 1 s、2 s 的横向偏移 **y1、y2**，2 s 处的朝向 **h2**（每次推理一个值）。

速度段（按 v）：**低 < 5 m/s**（控制器的运动学段），**中 5–15**，**高 ≥ 15**。

### 3.3 事件集

- **零分事件**：score = 0 的 rollout，事件时刻 t_ev = 最早一个 flag（at-fault 碰撞看 `collision_any`、`offroad`、`left_corridor_laterally`）在 2 Hz 序列上首次 > 0 的时刻；
  类型按最早者（同 zeros_mech 的 ol_clips 规则）。窗口 W = [t_ev − 3 s, t_ev]。
- **剧烈横向标记**：零分事件的 W 里 A_exec ≥ 6 m/s² 连续 ≥ 0.3 s，或 |vy / vx| ≥ 0.2（vx ≥ 3 m/s），记「打转样」；与类型并列，不互斥。
- **高速大转向片段**（所有 rollout，不限零分）：v ≥ 15 m/s 且 A_exec ≥ 3 m/s²（openpilot 的上限）连续 ≥ 0.3 s；报每 100 km 的次数、其后 3 s 内出零分的比例；
  与人类日志在同速度段的 A ≥ 3 m/s² 比例对照（nuPlan：per_decision 的日志朝向差 / 0.5 s × 日志速度，2 Hz；自车同样降到 2 Hz 再比）。

### 3.4 匹配

同赛道、同速度段、来自「干净 rollout」（score > 0 且三个 flag 都没有）的全部 0.1 s 行，各量的 p99 作阈值 θ（按赛道 × 速度段，label 合并）。事件的速度段按 W 里第一个越线行的 v，没有越线按 t_ev − 1 s 的 v。

### 3.5 判定规则（每个零分事件）

- onset_plan = W 里 A_path > θ(A_path) 的第一行；onset_steer = W 里 A_cmd > θ(A_cmd) 的第一行。
- **plan 先动**：onset_plan 存在且 ≤ onset_steer（或转角始终不越线）。
- **转角先动且放大**：onset_steer 存在且早于 onset_plan 0.2 s 以上（或 plan 始终不越线），并且在 onset_steer 这一行 A_cmd > 1.5 × A_req。
  转角先动但 A_cmd ≤ 1.5 × A_req 记「转角先动，参考所需」（MPC 在追一个位置偏差）。
- **都不越线**：记「安静」（漂移型，阈值上看不到横向异常）。
- **环路特征**（W 再往前 1 s，按 0.5 s 取 plan：nuPlan 每次推理，PAI 每 5 次取 1 次）：
  振荡 O = y2 序列在 |y2| ≥ 0.3 m 的项里 ≥ 2 次变号，且最后一个半周期峰值 ≥ 1.3 × 第一个；
  单调发散 R = 连续 ≥ 3 个 plan 的 h2 与前 0.5 s 自车偏航率同号、|h2| 每步至少 ×1.2，且自车 |wz| 同期也在增。
- 放大比：W 内 A_cmd 峰值行的 A_cmd / A_req。

**登记线（按速度段分别读，nuPlan 与 PAI 分开报，合并只作描述）**：
- 高速段零分事件中 plan 先动 ≥ 60 % → **HL1 成立**，提案 (a) 排第一；
- 任一速度段转角先动且放大 ≥ 40 % → **HL2 成立**，提案 (b) 排第一；两者都不到 → 描述，(a) (b) 都只作探索臂；
- 低速段打转样事件中 O 或 R ≥ 30 % → **HL3 在 AlpaSim 可见**；
- 事件数 < 10 的格子只描述，不判。

### 3.6 慢与停

- **速度比**：每行 v_ego / v_log，v_log 取**位置对齐**：自车已走 d_e(t)（`dist_traveled_m`），找日志走到同一 d 的时刻 t'（`gt_dist_traveled_m` 反插），取日志在 t' 的速度。
  只用 v_log ≥ 2 m/s、事件之前、自车未越过日志终点的行。按自车已走距离分箱（0–10、10–25、25–50、50–100、> 100 m）与距上次停车（v < 0.5 m/s）的时间分箱（0–2、2–4、4–6、6–9、9–20 s）报中位数；
  趋势 = log(v_ego / v_log) 对已走距离的斜率（每 100 m），log 聚类 bootstrap（nuPlan 按 log，PAI 按 scene）95 % CI。**线**：斜率 CI 上界 < 0 → 「越开越慢」成立。
- **分解**：v_log − v_ego = (v_log − v_plan) + (v_plan − v_ego)，v_plan = 模型 plan 首 1 s 的平均速度（PAI 用 `poses_model`，另报 served 对 model 的差 = serving 去掉的部分）。
  **线**：|v_log − v_plan| 的均值 ≥ 2 / 3 总差 → HV1；|v_plan − v_ego| ≥ 2 / 3 → HV2；之间 → 两者都有。
- **无故停车**：v < 1 m/s 连续 ≥ 1 s 的片段，起点满足 位置对齐的 v_log ≥ 2 m/s（人在这里没停）、`min_distance_to_obstacle_m` ≥ 10 m（评估器量的到任何物体最近距离）、在任何 flag 之前、未越过日志终点。
  每片段按起点前 2 s 内的 plan 归类，优先级 P > L > V > M：
  P = 模型 plan 2 s 弧长 < 1 m；L（仅 PAI）= `fix.lead.src == 'mpc'` 且 `cut` ≥ 0.5 m；V（仅 PAI）= 模型 plan 2 s 弧长 ≥ 1 m 而 served < 1 m 且非 L；
  M = served plan 2 s 弧长 ≥ 1 m 而之后 1 s 自车仍 < 1 m/s；其余记 other。**线**：某一类 ≥ 50 % → 提案 (c) 针对那一类；P ≥ 50 % → serving 修不了，只记。
- 红灯不进规则（日志的灯态没读）：人在这里没停只说明人过的时候不用停，自车晚到可能碰上红灯。看片时逐个核对。

### 3.7 限曲率的离线反事实与选择规则

规则（driver 侧，作用在返回轨迹上，速度剖面不动）：沿 plan 重积分朝向，每段曲率夹到 |κ| ≤ a_max / max(v_ref, 3)²，
跨推理的曲率变化率夹到 j_max / max(v_ref, 3)²（与上一个 plan 在同一绝对时间比）。网格 a_max ∈ {2, 3, 4, 5} m/s²，j_max ∈ {2.5, 5, 无}。
- **作用于事件**：W 内有 plan 被改动（τ = 1–2 s 的位置差 > 0.10 m），且被改动的第一个 plan 不晚于 onset_plan，并把该 plan 的 |y2| 减小 ≥ 50 %。
- **误伤**：(i) 干净 rollout 的推理中被改动 > 0.10 m 的比例；(ii) 人类日志 2 Hz 横向加速度超过 a_max 的比例（同 v 分箱）；
  (iii) navhard PDM 参考：由 h1、h2、x2、v0 推的 1–2 s 曲率乘 v0²，超过 a_max 的 token 比例（只是「必须转」的代理，AlpaSim 的规则不进 navhard 打分）。
- **选择规则**：误伤 (i) ≤ 2 %、(ii) ≤ 1 %、(iii) ≤ 1 % 的组合里，取作用于高速 plan 先动事件最多的一个；并列取更宽松的。
  **线**：最好的组合作用于 < 25 % 的高速 plan 先动事件 → 限曲率不作提交候选，只作探索臂。

### 3.8 闸（不过就停，修数据后整体重读）

- G-cov：≥ 99 % 的 rollout 对上控制器 CSV 与 driver 记录（按 session）。
- G-frame：推理时刻 CSV 位姿与 driver anchor ≤ 0.05 m 的比例 ≥ 99 %（同一 local 系；CSV 记录在步末，取 t0 那一行）。
- G-ev：nuPlan 六个 label 的零分 rollout 数与类型和 zeros_mech `per_rollout.parquet` 逐条一致（不一致的列出，不改规则）。
- G-rec：重建的返回轨迹与 CSV 里的 `x_ref_0 / y_ref_0`（MPC horizon 第 0 步的参考点，即轨迹在当前时刻 t 的插值，换到 rig 系）核对：重建轨迹在 t 处换到同一 rig 系后差 ≤ 0.05 m 的行 ≥ 95 %。

### 3.9 看片

从零分事件里按「速度段 × 判定」分层各取 1–3 个，共 10–20 个 PAI 事件，用 Tokyo 上的前视 mp4（`~/Desktop/pai_videos` 的源文件）截 t_ev − 3 s 到 t_ev 每 0.5 s 一帧拼图，逐个记：前方有什么、plan 在横扫还是一步打过去、
车身先动还是画面里的路先变、红灯 / 停止线 / 前车。表进第 4 节，与判定不一致的逐条列出，不改判定。

### 3.10 代价（估）

CPU：nuPlan 6 × 452 + PAI 3 × 281 个 rollout，每个几百行，Mac 上几分钟；数据压缩后约 1 GB 拉回 Mac（`tmp/`）。0 卡时。

## 4. 读数

（待读）

## 5. 提案（排序）

（待读）

## 6. 没能核实的

（待读）

## 修订记录

| 日期 | 改了什么 |
|:--|:--|
| 2026-10-12 | 预登记（第 3 节），读数之前 |
