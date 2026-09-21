# 轨迹怎么变成 CARLA 的控制量

状态: 设计结论，2026-09-22。写给要实现 `scripts/b2d_agent.py` 里那个 stand-in driver 的替代品的人。
背景: [carla-efficiency.md](carla-efficiency.md)（成本）、[frozen-vlm-planner.md](frozen-vlm-planner.md)（planner 输出什么）、
[decisions.md](decisions.md) 第 16–18 条（闭环要不要做）。

## 问题

开环 benchmark（NAVSIM、Waymo E2E、nuScenes）上的 planner，包括我们自己的 frozen VLM + 薄 head，
输出的都是一条 **trajectory**（未来若干秒的 ego 位姿序列，ego frame，固定采样率）。
CARLA 的 leaderboard 每个 tick（20 Hz，`fixed_delta_seconds = 0.05`）要的是一个
`carla.VehicleControl`（throttle ∈ [0,1]、steer ∈ [−1,1]、brake ∈ [0,1]）。
中间这一步叫什么、谁来学、要付什么代价，是这份文档的全部内容。

我们的 planner 的输出形状把问题收窄了：**在 K 条 k-means anchor 组成的 trajectory vocabulary 上打分，
取 argmax**，输出是 Waymo 格式的 5 s、4 Hz、20 个 (x, y)，ego frame，原点在后轴，x 向前 y 向左，
第一个点在 t + 0.25 s（`jevdrive/traj.py`）。K ≥ 1024（decisions 第 8 条）。

结论先写在前面：**用 CARLA 自带的 `VehiclePIDController` 的结构，参数从它自己的默认值起步，
控制器保持固定、不学；轨迹在 5 Hz 更新、由控制器用 dead reckoning 重投影后在 20 Hz 跟踪。
"好模型自己学会用控制器"这句话是对的，但它说的是 head 通过固定控制器做 closed-loop fine-tuning，
不是把控制器做成可学的。** 下面是理由和边界。

## CARLA 已经给了什么

`$CARLA_ROOT/PythonAPI/carla/agents/navigation/`（tarball 里，不在 PyPI wheel 里，要加 `sys.path`），
MIT license，四个文件。这是在 box 上读源码得到的，不是文档。

### `controller.py`：`VehiclePIDController`

一个 `PIDLongitudinalController` 加一个 `PIDLateralController`，各自的 error 放进一个
`deque(maxlen=10)`，积分项是这 10 个样本的和乘 dt（所以其实是 0.5 s 的滑动窗，不是真积分），
微分项是最近两个样本的差除 dt。

| 环 | error 的定义 | 单位 | 输出 |
|---|---|---|---|
| 纵向 | `target_speed − current_speed` | **km/h** | clip 到 [−1, 1]；≥ 0 走 throttle（上限 `max_throttle`），< 0 走 brake（上限 `max_brake`） |
| 横向 | ego 前向向量与「ego → 目标 waypoint」向量的夹角，叉积定号 | rad | clip 到 [−1, 1]，再限幅到 `±max_steering`，**每 tick 最多变 0.1** |

默认值要分两处看，这是一个陷阱：类本身的默认是 `K_P=1.0, K_I=0, K_D=0, dt=0.03`，
但没有人这样构造它；`LocalPlanner` 传进去的是：

| 参数 | `LocalPlanner` 的值 |
|---|---|
| dt | **1/20 = 0.05**（和 leaderboard 一致） |
| 横向 K_P / K_I / K_D | **1.95 / 0.05 / 0.2** |
| 纵向 K_P / K_I / K_D | **1.0 / 0.05 / 0**（error 以 km/h 计） |
| max_throttle / max_brake / max_steering | 0.75 / **0.3** / 0.8 |
| 目标速度 | 20 km/h 固定，或 `follow_speed_limits` |

两点值得记：纵向 K_P=1.0 配 km/h 的 error 意味着差 1 km/h 就给满油门上限，实际是 bang-bang；
`max_brake=0.3` 是巡航用的软刹车，Bench2Drive 的 cut-in / 行人场景要的急刹它给不出，用的时候要提到 1.0。

### `local_planner.py`：目标 waypoint 和目标速度怎么选

`LocalPlanner` 跟的是地图 waypoint（`sampling_radius` 2 m 一个），不是轨迹。每个 tick：

1. `min_distance = 3.0 + 0.5 · v`（v 单位 m/s），把队列头部所有距离小于它的 waypoint 弹掉。
2. 目标 waypoint = 弹完之后的队首，也就是**第一个比 3 m + 0.5 s × v 远的点**（8 m/s 时约 7 m）。
   这就是它的 lookahead 规则，pure pursuit 式的，只是 lookahead 用在选点上、转向用 PID 算。
3. 目标速度是一个常数（`set_speed`），横向和纵向完全解耦。

`basic_agent.py` 在这之上加了红灯和前车检测（用**特权** actor 列表，不是传感器），
`behavior_agent.py` 加了按 TTC 的跟车逻辑（`Normal`：max 50 km/h、安全时距 3 s、刹车距离 5 m）。
这两个文件对我们只有一个用处：证明这套 PID 在 CARLA 的车上用这组增益能开。
它们的感知全是特权的，进 `Track.SENSORS` 一行都不能用。

### 车和物理

leaderboard 的 ego 是 `vehicle.lincoln.mkz_2020`。steer 命令乘以前轮的 `max_steer_angle` 得到转角，
再被 `steering_curve` 按速度打折；PhysX 在 0.05 s 内做 substep。轴距、最大转角、`steering_curve` 的具体值
**这次没有查**（要一个活着的 server 调 `get_physics_control()`，本文档不起 server），
见最后一节。它们只影响 feedforward 那一项，不影响结论。

## 别人怎么做

### Bench2Drive 上发表过数字的 agent

来自 web 上的源码（Bench2DriveZoo `uniad/vad` 分支 `498c1f7`、`tcp/admlp` 分支 `8a08b07`，
TCP、carla_garage、SimLingo、CaRL 各自的 main）。报的是**他们做了什么**，不是我们必须做什么。

| Agent | 横向 | 纵向 | aim point | 目标速度 | 模型输出 | 推理频率 |
|---|---|---|---|---|---|---|
| UniAD / VAD（Bench2DriveZoo） | PID 0.75 / 0.75 / 0.3，窗 40，error = 到 aim point 的角度 / 90° | PID 5.0 / 0.5 / 1.0，窗 40，`delta = clip(v_des − v, 0, 0.25)`，throttle ≤ 0.75 | 中点离 ego 最接近 **4 m** 的那一段的起点 | 相邻 waypoint 平均间距 × 2（隐含 0.5 s 间隔） | 6 点 × 0.5 s = 3 s | **每 tick（20 Hz）** |
| AD-MLP（Bench2DriveZoo） | 同上 | 同上 | 同上 | 同上 | 同上 | **每 10 tick（2 Hz）**，中间 hold |
| TCP | 同上，再和 control branch 按 0.3 / 0.7 混合 | 同上 | 同上 | 同上 | 4 点 × 0.5 s | 每 tick |
| TransFuser++ LB2 / PDM-Lite / SimLingo | PID 3.12 / 0.64 / 1.38，窗 6，error = 到 lookahead 点的 heading 差 / 90° | 线性回归（6 个特征）或 PID；SimLingo 用 PID 1.75 / 1.0 / 2.0 | route 上 `clip(0.98·v_kph + 1.9, 24, 105)` 个点（0.1 m 一个）= **2.4–10.5 m** | 分类头直接出目标速度（TF++）或 0.5 s→1 s 段长 × 2（SimLingo） | 10 个 1 m 间距 checkpoint | 每 tick |
| CaRL（RL） | 无控制器 | 无控制器 | — | — | **直接 [steer, acc]** | 10 Hz，action repeat 2 |

三件事从这张表里读出来：

1. **发表过的 Bench2Drive 数字全都来自 trajectory → 固定 PID → control。** 唯一的例外是 CaRL，
   它是 RL 训出来的、直接出控制量，代价是没有任何开环可比性。
2. **这套 PID 是同一份代码传下来的**：TransFuser CVPR'21 的 `PIDController` 逐字复制到 TCP、InterFuser、
   LAV、ThinkTwice、Bench2DriveZoo。`speed 5.0 / 0.5 / 1.0`、`brake_ratio 1.1`、`clip_delta 0.25`、
   `max_throttle 0.75` 五年没动过。它和 CARLA 自带的那个在结构上是同一个东西（角度误差 PID + 速度误差 PID），
   区别只在积分是滑动窗均值、error 归一化到 /90°、aim point 固定 4 m 而不是随速度变。
3. **它们的纵向实际上都是 bang-bang**：`delta ≤ 0.25` 而 K_P=5，P 项上限 1.25，油门几乎总在 0.75 饱和；
   brake 是布尔量，clip 之后是 0 或 1。UniAD 的 agent 还硬编码 `speed > 5 m/s → throttle = 0`，
   把它封在 18 km/h 以下。所以"baseline 的控制器"没有什么精细可言，
   分数差异来自 planner 和数据，不来自控制器，这一点对我们是好消息。

另一个对我们直接有用的事实：**Bench2DriveZoo 的 AD-MLP 是每 10 tick 推理一次、中间 hold**，
UniAD/VAD 是每 tick。也就是说 Bench2Drive 对 agent 的推理频率没有规定，5 Hz 的 replan
有先例，而且我们每 4 tick 比 AD-MLP 的每 10 tick 更密。

### openpilot

comma.ai 的 openpilot 是"真车上的 trajectory → control"的参照。它今天的结构（master `521db4c`，2026-09）：

| 部件 | 现状 | 补偿的是什么 |
|---|---|---|
| 模型输出 | 直接出 **desired curvature 和 desired acceleration**（`modelV2.action`）。lateral MPC 2024-01 删掉，longitudinal MPC 2025-08 从训练和 Experimental mode 里删掉 | — |
| 延迟 | 执行器延迟 `action_t` 是**模型的输入**：policy 被告知延迟后输出"延迟之后那一刻"的目标；`lagd` 在线学转向延迟（0.15–0.65 s） | 真实 EPS 和整条 pipeline 的滞后 |
| 横向 | `LatControlTorque`：在**横向加速度空间**做 PID（KP 0.8、KI 0.15，低速按表放大到 250），feedforward = 目标横向加速度 − 路面横坡 − offset，再加 friction 项；输出线性映射到 EPS 力矩 | EPS 力矩增益、静摩擦/迟滞、横坡、装配 roll 偏差 |
| 在线标定 | `paramsd`（Kalman：steer ratio、轮胎刚度、角度零偏、roll），`torqued`（力矩↔横向加速度的斜率、offset、friction） | 未知轮胎、湿滑路面、传感器零偏 |
| 纵向 | MPC 只看雷达前车；`LongControl` 是 accel error 的积分项 + feedforward，默认 ki=0，**就是纯 feedforward** | 车厂 PCM 的响应误差下沉到 opendbc |
| 频率 | 模型 20 Hz，控制 100 Hz，中间是 zero-order hold + 限幅（横向 jerk ≤ 5/v²、横向加速度 ≤ 3 m/s²） | 舒适、法规 |
| 谁被训练 | **没有任何东西是穿过真车控制器训练的。** 0.11 起 policy 在一个 learned simulator 里训，动作经过一个**手写的** Vehicle Model（含延迟和 domain randomization）变成位姿，真车上的 torque PID、lagd、torqued 都不在 loop 里 | — |

放到 CARLA 里看，这张表右边一列几乎全部是 CARLA 没有的问题：状态精确、车辆模型已知且确定、
执行器延迟是零（tick k 的 control 在 tick k+1 的物理里生效）、没有横坡、没有 EPS。
把这些去掉，openpilot 的控制器退化成 `κ_des → steer angle`（他们自己的 `LatControlAngle` 就是这个）
加 `accel = a_target`。**真正剩下的、值得借的结构只有两条**：目标量按"延迟之后那一刻"定义，
以及决策频率低于执行频率时用 hold + 限幅衔接。这两条在下面的推荐里都用上了。

openpilot 还回答了"谁学什么"这个问题的一半：即使是真车，他们也选择**固定控制器 + policy 在仿真里
学会对着这个固定的动作语义开车**，而不是学控制器。这和"好模型自己学会用控制器"是同一个意思。

## 谁学什么：四条路

"好模型学会用控制器"是这一节的轴。先把话说准：这句话成立的前提是**控制器在训练时也在 loop 里**。
我们的 stage A head 是在 Waymo 开环数据上做 imitation 训出来的，控制器不在 loop 里，
所以它学到的是"人类会开出的轨迹"，不是"这个 PID 跟得好的轨迹"。这两者差多远，正是闭环分数
相对开环指标掉多少的来源之一，也是控制器要做好的原因：控制器跟得越准，这个差距就越接近纯粹的
planner 误差，闭环数字才解释得清。

| 路 | 手调什么 | 学什么 | 和开环数字的可比性 | 推理开销 | 对 K 词表的适配 |
|---|---|---|---|---|---|
| A. trajectory → 固定 PID → control | 6 个增益、lookahead、限幅（都有现成值） | 只有 head，开环 | **最好**：闭环 = 同一个 head + 一个和 UniAD/VAD 同类的固定控制器 | ~0（20 Hz 的 numpy，藏在 sim 后面） | 好：见下 |
| B. trajectory → 可微 / 可学控制器，head 穿过它训练 | 控制器的结构、训练它的 loss、一个可微的车辆模型 | 控制器 + head 共同适应 | **差**：head 被改过，开环指标描述的不再是部署的那个 head | 小 | argmax 不可微，只能 soft 选择或 policy gradient |
| C. 模型直接出 20 Hz 的 control | 没有控制器，但要 control 标签或 RL | 全部 | **无**：输出空间不同，没有开环指标可报 | **7.3×**：每 tick 推理 520.9 对 71.1 ms/tick（[docs/bench2drive-cost.md](../docs/bench2drive-cost.md)） | 词表没有意义了 |
| D. trajectory → 固定 PID + 小的学出来的 residual（看跟踪误差） | PID 基座 + residual 的训练 | residual（planner 不变） | **好**：head 不变，residual 是 planner 无关的附件，可以做 ablation 报出来 | 很小 | 好 |

逐条说：

**A 是 baseline 全体的做法，也是我们唯一保留开环可比性的做法。** 它的"手调"其实不存在：
CARLA 和 TCP 两组增益都在，两组都在这台车上跑过成千上万条路线。它把学习负担全放在 head 上，
而 head 的学习信号来自开环数据。要让"模型学会用控制器"，正确的动作是**之后**对 head 做
closed-loop fine-tuning（在 CARLA 里跑、控制器固定、以 driving score 或 DAgger 式的 expert 标签作信号），
而不是换控制器。我们的输出是 K-way 的 argmax，这件事反而顺手：**动作空间是 K 个离散 anchor，
policy gradient 直接可用**，AnchorVLA（decisions 第 18 (d) 条，K=100 anchors 上做 RL）就是这么做的。

**B 在 CARLA 里没有东西可学。** 可学控制器的价值在于吸收 plant 的未知（openpilot 那张表的右列），
CARLA 的 plant 是已知的、确定的、可以写成一个 kinematic bicycle 加 `steering_curve`。
一个学出来的控制器最后学到的就是这个模型，我们可以直接写下来当 feedforward。它的代价却是实打实的：
head 一旦穿过控制器训练，Waymo 上的 RFS 就不再描述部署的那个 head。

**C 是 CaRL 的路，对我们不可行也不值得。** 不可行：Qwen3-VL-4B 单次 60–140 ms，20 Hz 要 50 ms；
每 tick 推理的实测是 0.096× 实时。不值得：control 是 Lincoln MKZ 2020 在 PhysX 里的 throttle/brake 语义，
是所有表示里最不可迁移的一种，Waymo 和 NAVSIM 上都没有对应标签，整篇论文的开环部分就没了。

**D 的实际用途比看上去小。** residual 能修的是"控制器跟不上轨迹"，而在已知 plant 上 feedforward
已经把这一项修掉了。它可能修的另一件事是"head 系统性地晚刹车"，但那是 planner 的缺陷，
在控制器里补上等于把它从开环指标里藏起来。留作 ablation 的选项，不进主数字。

### K 词表这个输出形状意味着什么

一个直觉上的担心是：控制器要跟的是 1024 条固定形状里的一条，连续两次决策可能在两条 anchor 之间跳，
而且 anchor 的"分辨率"可能太粗。在 box 上量了现成的 Waymo K=1024 词表（`runs/waymo_stage_a/half_val/20260920-152600/vocab_K1024.npy`）：

| 时刻 | anchor 之间的中位最近邻距离 | x 的中位 | x 的范围 |
|---|---:|---:|---|
| 0.25 s（第 1 点） | **0.009 m** | 1.40 m | [−0.2, 7.3] |
| 0.5 s | 0.021 m | 2.81 m | [−0.4, 14.7] |
| 1.0 s | 0.056 m | 5.67 m | [−0.9, 29.4] |
| 5.0 s（第 20 点） | 0.601 m | 26.9 m | [−5.1, 146.0] |

nuScenes 的 K=1024（3 s、2 Hz）同一量级：0.5 s 处 0.017 m，3 s 处 0.29 m。

读法：**控制器只消费轨迹的前 0.5–1 s，那一段 anchor 之间的间距是毫米到厘米级，词表在控制器眼里
就是连续的**；离散性全部在 5 s 那一端，而那一端是 head 用来打分的，不是控制器用来跟踪的。
所以 K 词表对控制器的选择没有约束，它约束的是 head。另外 86/1024 条 anchor 在 0.25 s 处 |p| < 0.2 m，
即"停住"的轨迹是词表里一个明确的 mode，head 选中它时控制器的目标速度自然是 0，不需要额外的刹车规则。

真正的形状问题是另一个：**词表是 Waymo 的真实驾驶分布，x 到 146 m 是高速路的 anchor**，
Bench2Drive 是城区、限速 30–60 km/h。这不是控制器的事，是 head 的 transfer 的事，闭环分数会直接回答。

## 推荐

**用 A。控制器固定，结构取 CARLA 的 `VehiclePIDController`，参数从 `LocalPlanner` 的默认值起步，
再加一项按已知车辆模型算的 feedforward。**

具体形状（写给下一步实现的人；不改 `b2d_agent.py` 的接口，新文件）：

| 步骤 | 做法 | 来源 |
|---|---|---|
| 输入 | 5 Hz：一条 (20, 2) 的 ego-frame 轨迹加它对应的帧的时间戳。20 Hz：speedometer 的前向速度、IMU 的 yaw rate | 都是 `Track.SENSORS` 允许的 |
| 重投影 | 每个 20 Hz tick 用 `v·dt` 和 `ω·dt` 做 dead reckoning（用运动学推算当前位姿），把轨迹从"拍摄那一刻的 ego frame"转到"现在的 ego frame" | openpilot 的 `action_t` 思路，只是延迟已知 |
| aim point | 沿轨迹弧长插值，取 `max(3.0, 0.5·v)` m 处的点 | `LocalPlanner` 的 `min_distance` 规则 |
| 横向 | steer = feedforward + PID。feedforward：pure pursuit 曲率 `κ = 2y / (x² + y²)`，转成前轮转角 `δ = atan(κ·L)`，除以 `max_steer_angle` 和 `steering_curve(v)`。PID：CARLA 的 1.95 / 0.05 / 0.2 作用在到 aim point 的角度误差上；每 tick 变化 ≤ 0.1，限幅 0.8 | `controller.py` |
| 纵向 | 目标速度 = 重投影后轨迹在 0.25–1.0 s 之间的弧长 / 0.75 s；PID 1.0 / 0.05 / 0（km/h）；`max_throttle 0.75`，**`max_brake 1.0`**（不是 0.3） | `controller.py`，刹车上限改 |
| 停车 | 目标速度 < 0.4 m/s 时 brake = 1、throttle = 0 | TCP 的 `brake_speed` |
| 不做 | 任何基于特权 actor 的红灯 / 前车规则；任何 residual | 见下面的语义一节 |

每条选择背后的算术：

**5 Hz 决策、20 Hz 执行，中间靠控制器重投影，而不是靠别的。** 现在的配置（decimate 4 + overlap）里，
tick k 拍的帧，结果在 tick k+1 或 k+2 落地，然后用到 tick k+5，所以一条轨迹被执行时的年龄是
0.05–0.30 s 的仿真时间。这个年龄有两部分。第一部分是几何的：ego 已经动了。不重投影的话，
8 m/s 直行 0.3 s 是 2.4 m，"0.5 s 后的点"变成了"0.2 s 后的点"，lookahead 短了一半；
转弯时 yaw rate 20°/s 累积 6° = 0.105 rad，直接进横向 PID 是 0.2 的 steer，每 4 tick 抖一次。
重投影之后这一部分归零：IMU gyro 噪声 0.001 rad/s，0.3 s 积出 0.0003 rad；speedometer 是精确的前向速度，
位置误差 < 1 cm。第二部分是决策的：0.3 s 前还没开始刹的前车现在刹了，这条轨迹不知道。
这一部分任何控制器都修不了，它是 replan 频率的代价，真车上一样有，而且 AD-MLP 的 0.5 s 比我们的更长。
所以：**重投影是必须的、便宜的、够用的；决策延迟留给 head 和 replan 频率，不要试图在控制器里补。**

**feedforward 值得写，因为它是 CARLA 相对真车唯一的免费午餐。** 已知轴距和最大转角，pure pursuit 的
steer 在 CARLA 里就是接近精确的开环解，PID 只剩下修 `steering_curve` 和轮胎侧偏的小残差。
Bench2DriveZoo 的 PID 不带 feedforward，所以要靠 K_P 0.75 硬拉，代价是 4 m aim point 上的震荡；
PDM-Lite 用更长的 lookahead（2.4–10.5 m）换稳定。feedforward 让我们两头都不用选。
需要的三个常数（轴距、`max_steer_angle`、`steering_curve`）一次 `get_physics_control()` 就有。

**目标速度从轨迹的弧长读，不从相邻两点读。** TCP 系用 `|wp1 − wp0| × 2` 这种单段差分，在 0.5 s 采样上
噪声很大，这是它们 brake 要做成布尔量、throttle 要 clip 到 0.25 delta 的原因之一。我们是 4 Hz，
0.25–1.0 s 之间三段的弧长平均把这个噪声压掉了；而且用重投影后的轨迹，目标速度本身就带着"延迟之后"的语义。

**为什么起点是 CARLA 的增益而不是 TCP 的。** 两组都是在这台车上验证过的；CARLA 的那组 error 单位是
物理单位（rad、km/h），配 feedforward 更自然；TCP 的那组是为没有 feedforward 的情况调的。
实现里把两组都做成 config，Dev10 上各跑一遍，选跟踪误差小的。这不是研究问题，是半天的事。

### 这会改变评测的哪些语义

必须和分数一起报的：

| 项 | 我们的做法 | 和公开数字可比吗 |
|---|---|---|
| 推理频率 5 Hz，control 20 Hz | agent 侧的选择，仿真仍 20 Hz | **可比**。AD-MLP 2 Hz、InterFuser 10 Hz 都有发表数字；leaderboard 没有规定 |
| 决策延迟 1–2 tick（overlap） | 比标准协议**更严**：标准是 tick k 的帧、tick k 的控制 | 可比，对我们不利，如实写 |
| ego 位姿 | speedometer + IMU 做 dead reckoning，**不用** `hero_actor.get_transform()` | `Track.SENSORS` 合规。**现在的 `_steer_to_route` 用的是特权 transform，必须换掉** |
| GNSS | 不用，或只用于沿 route 的粗定位。leaderboard 的 GNSS 噪声 5e-6°，按纬度折算约 0.5 m（自己算的，未抽样验证） | 和 baseline 相同的传感器 |
| 红灯 / 前车规则 | **不做**。Bench2DriveZoo 的 UniAD/VAD 也没有 | 可比。加了就是在评一个 rule-based 系统 |
| `max_brake 1.0` | 增益，不是语义 | 可比 |
| `MinimumSpeedRouteTest` | leaderboard 会惩罚爬行；目标速度不能保守到常低于限速 | 提醒，不是变更 |

### 一个能开的 driver 对成本的影响

这是现在 3.1 h 里最大的水分，能量出来。`runs/b2d/full220`（220 条、8 worker、共享卡）：

| 量 | 值 |
|---|---:|
| 路线长度（`bench2drive220.xml` 的 waypoint 折线） | 均值 105 m，中位 104，p10 65，p90 139，最长 222；合计 23.1 km |
| 跑完的路线 | 209 / 220 |
| 撞到 4000 tick 上限的 | **118 / 209 = 56%** |
| 没撞上限的 91 条 | tick 数 1281–2701，均值 1779，**与路线长度不相关**（r = −0.01），下限正好是 `ActorBlockedTest` 的 60 s = 1200 tick |
| 所有路线的 tick 均值 | 3033 |
| 每 tick 均值（共享卡） | 124.6 ms |
| 每条路线 tick 之外的固定开销 | 68.5 s（中位） |

读法：stand-in **一条路线也没有真正跑完**。一半在原地蹭着凑满 4000 tick，另一半卡住 60 s 被
`AgentBlockedTest` 终止。所以 3033 tick 这个均值和路线长度没有任何关系，它量的是两个 failure mode 的时长。

一个能开的 driver：105 m 在 8 m/s 上是 13 s，加起步、一个场景交互（红灯、cut-in、行人）10–30 s，
按 600–1200 tick 估（30–60 s，Bench2Drive 的路线就是这样设计的短路线）。每条路线 wall =
tick × 124.6 ms + 68.5 s，8 个 worker：

| driver | tick / 路线 | 220 条 | 其中固定开销占 |
|---|---:|---:|---:|
| 现在的 stand-in（实测） | 3033 | 3.1 h | 15% |
| 能开的，悲观 | 1200 | **1.4 h** | 31% |
| 能开的，乐观 | 600 | **1.1 h** | 48% |

也就是说**换 driver 比这篇文档之前的任何一项优化都省得多（2–3×），而且换完之后下一个瓶颈就是每条路线
68 s 的固定开销**（world reload、scenario 构建、server 错峰），不再是 tick。这个估计的两个边界：
一个真开得好的 driver 不会卡住，但会守红灯、会让行，tick 可能高于 1200；一个开得差的 head 仍会撞上限，
数字会回到 3 h。

### 第一步

一个 py3.8、只依赖 numpy 的 `scripts/b2d_controller.py`，接口是 `update(traj_xy, t_frame)`（5 Hz）和
`step(speed, yaw_rate) -> VehicleControl`（20 Hz），内容就是上面那张表。验证不需要 planner：

1. 先用 route 本身当"轨迹"。`set_global_plan` 重载一下保留 downsample 之前的 1 m 间距 route
   （现在 stand-in 拿到的是 `downsample_route(…, 50)` 之后 50 m 一个点的版本，这也是它转不过弯的原因之一），
   取 ego 前方 5 s、按限速的匀速轨迹喂给控制器。这个 driver 不看红灯不看车，会撞，但**能跑完路线**，
   而且它的 tick 数就是上面那张表里"能开的 driver"的下界，把成本模型修准。
2. Dev10（官方推荐的 10 条 ablation 子集，约 10 分钟）上报：route completion、横向跟踪误差（aim point 处的
   横向偏差 RMS）、速度跟踪误差、是否触发 `AgentBlockedTest`，CARLA 增益和 TCP 增益各一遍。
3. 然后才接 planner：policy server 把选中的 anchor（20×2）回传，替换掉现在"算完就扔"的那条路径。

工作量一天以内。要记进 todo 的数字：Dev10 上的 tick/路线、两组增益的跟踪误差、220 条的新 wall clock。

## 没查清的，和怎么查清

| 项 | 状态 | 怎么定 |
|---|---|---|
| MKZ 2020 的轴距、`max_steer_angle`、`steering_curve` | 未查（不起 server） | 下次任何 server 活着时 `vehicle.get_physics_control()` 一行 |
| CARLA 的增益在 8–14 m/s 上是否稳（`LocalPlanner` 默认 20 km/h） | 未测 | 第一步的 Dev10，两组增益对比 |
| 能开的 driver 每条路线多少 tick | 估 600–1200 | 第一步的 route-following run 给下界，接 planner 后给真值 |
| GNSS 噪声 5e-6° ≈ 0.5 m 这个换算 | 自己算的 | 抽 100 个样本看 std；只影响要不要用 GNSS 做粗定位 |
| Bench2DriveZoo 的 waypoint 轴向（index 1 是否 forward） | 未逐行验证 | 与我们无关，只在复现 UniAD 数字时要对 |
| Waymo 词表在 CARLA 城区分布上的 coverage | 未量 | 用 route-following run 记录的 ego 轨迹反查 oracle minADE，同 planner v0 的 K sweep 口径 |
| head 在 CARLA 红灯前会不会选"停住"的 anchor | 不是控制器问题 | 闭环分数本身 |

## 会推翻这个推荐的证据

- Dev10 上 route-following driver 的横向跟踪误差 > 0.5 m 或触发 `AgentBlockedTest`，且两组增益都如此：
  那说明 CARLA 的 PID 结构在这个 lookahead 上不够，换成 PDM-Lite 的长 lookahead + 它的增益（同为固定 PID，
  推荐不变，参数变）。
- 接上 planner 之后，闭环失败集中在"轨迹合理但跟丢"而不是"轨迹本身错"（看 aim point 偏差和 anchor 选择的
  逐 tick 记录能分开）：那时 D（residual）才值得做，而且要作为 ablation 单列。
- 决定做 closed-loop fine-tuning：控制器仍然固定，变的是 head 的训练信号，推荐不变。

## 参考

- CARLA `PythonAPI/carla/agents/navigation/{controller,local_planner,basic_agent,behavior_agent}.py`，0.9.15，box 上读的。
- Bench2DriveZoo `team_code/pid_controller.py`、`uniad_b2d_agent.py`、`vad_b2d_agent.py`（分支 `uniad/vad`）；
  `ADMLP/model.py`、`TCP/model.py`（分支 `tcp/admlp`）。
- TCP: https://github.com/OpenDriveLab/TCP；carla_garage（TF++ / PDM-Lite）: https://github.com/autonomousvision/carla_garage；
  SimLingo: https://github.com/RenzKa/simlingo；CaRL: https://github.com/autonomousvision/CaRL。
- openpilot master 与 RELEASES.md: https://github.com/commaai/openpilot；"Learning to Drive from a World Model"（arXiv:2504.19077，2025-04）。
- AnchorVLA（arXiv:2607.03182）：在 anchor 词表上做 RL 的先例，见 [decisions.md](decisions.md) 第 18 条。
