# 轨迹怎么变成 CARLA 的控制量

状态：2026-09-23修订。v1完整基线、v2–v4开发与恢复smoke已结束；v4正式双seed对照60条全部结束，不满足默认替代条件。真实模型上的新控制器收益尚未验证。
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

结论先写在前面：**控制器保持固定、不学；CARLA/TCP 风格 PID 与独立 pure pursuit 经过分层验证后选择默认。
轨迹在 5 Hz 更新、由控制器按时间裁剪并用 dead reckoning 重投影后在 20 Hz 跟踪。
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

纵向K_P=1.0配km/h误差时，差1km/h的比例项就超过.75油门上限，较大误差容易饱和；
这不能推出所有tick都是bang-bang（只切换最大动作），实际占比应从完整控制记录测量。
原max_brake=.3限制制动权限；本轮统一上限1.0，但有完整制动权限不等于已经验证紧急避碰。

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

本地 Bench2Drive 0.0.4、官方 leaderboard-2.0 和 master 的 RouteScenario 都固定请求
`vehicle.lincoln.mkz_2020`，本轮按这个 blueprint 标定。2026-09-22 的实测记录位于
`/data/runs/b2d/controller/calibration-units/`（physics、response、steering_unit_probe、controller_config）。

| 量 | 本次测量 |
|---|---:|
| 轴距 | 2.8604714914 m |
| 后轴相对 actor 原点的 x 偏移 | −1.3886332202 m |
| 最大前轮转角 | 69.9999924° |
| steering_curve | (0,1)、(20,0.9)、(60,0.8)、(120,0.7) |
| steering_curve 横轴单位 | **km/h**；活体诊断曲线探针已核验 |

探针临时使用 [(0,1),(10,0.1)] 曲线，在约2 m/s、steer=0.05时观察到约1.22°前轮角，
符合按7.2 km/h读取约0.35倍率的量级；若横轴是m/s会预测约2.87°。这验证单位，不能据此声称
实际转向完全等于理想自行车。原始响应记录仍存在左右差异、轮胎侧偏与速度跟踪偏差；目标10 m/s
的右转 sweep 实际均速仅0.021 m/s，属于失效采样，不能拿它拟合正常巡航转向增益。

三个常数提供几何起点；转角执行响应、道路坡度、轮胎动力学、GNSS误差都需独立验证。
不再使用“CARLA plant 完全已知，因此 feedforward 近乎精确”的前提。

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

1. 本表调研的多种方法使用trajectory→固定PID→control；TCP另有学习的control branch混合，CaRL直接预测控制量。
   这份来源集合不足以概括所有Bench2Drive方法，比较时也不能把完整TCP等同单一PID。
2. **这套 PID 是同一份代码传下来的**：TransFuser CVPR'21 的 `PIDController` 逐字复制到 TCP、InterFuser、
   LAV、ThinkTwice、Bench2DriveZoo。`speed 5.0 / 0.5 / 1.0`、`brake_ratio 1.1`、`clip_delta 0.25`、
   `max_throttle 0.75` 五年没动过。它和 CARLA 自带的那个都是角度/速度误差反馈，但离散语义不同：CARLA 的积分为 sum(error)×dt、
   微分为差分/dt；TCP 用零填充窗 mean(error)、微分不除dt，且角度归一化到/90°、aim固定4 m。
   两者不能只换增益和窗口就声称复现。
3. TCP式纵向中，delta上限.25与K_P=5使比例项上限达到1.25，高于.75油门限幅；
   brake为布尔量。但油门是否经常饱和取决于实际误差，不能仅凭增益就断言“几乎总在饱和”。
   原调研记录的UniAD实现还在speed>5m/s时清油门，这是代码中的油门规则，并不是车速必然不超过5m/s的证明。
   本轮6m/s真实仿真确实发现速度与踏板循环；对真实模型驾驶表现的影响仍须固定planner做controller-only对照。

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

CARLA 不需要复现真实 EPS 的所有复杂性，但模拟器仍有转向执行响应、轮胎侧偏、坡度和噪声；
本轮传感器路径也不直接获得精确状态。可以借用的结构是目标量按对应时间定义、低频决策与高频执行分开，
不能从“仿真”直接推出零延迟、无横坡或纯前馈足够。
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

**A 是本轮起点，保留 planner 输出与开环指标的可比性。** CARLA/TCP 提供已有参考增益，
但统一轨迹采样、坐标、延迟和限幅后的适配仍须验证。它把学习负担全放在 head 上，
而 head 的学习信号来自开环数据。要让"模型学会用控制器"，正确的动作是**之后**对 head 做
closed-loop fine-tuning（在 CARLA 里跑、控制器固定、以 driving score 或 DAgger 式的 expert 标签作信号），
而不是换控制器。我们的输出是 K-way 的 argmax，这件事反而顺手：**动作空间是 K 个离散 anchor，
policy gradient 直接可用**，AnchorVLA（decisions 第 18 (d) 条，K=100 anchors 上做 RL）就是这么做的。

**B 本轮不选。** 学习控制器可能吸收模型误差，但 CARLA 的实际动力学不能简单等同自行车模型；
是否值得学习须先量化固定控制器的残差。当前优先把坐标、时间、定位和固定控制基线做清楚。
head 一旦穿过控制器训练，还需要重新报告部署 head 的开环指标。

**C 是 CaRL 的路，对我们不可行也不值得。** 不可行：Qwen3-VL-4B 单次 60–140 ms，20 Hz 要 50 ms；
每 tick 推理的实测是 0.096× 实时。不值得：control 是 Lincoln MKZ 2020 在 PhysX 里的 throttle/brake 语义，
是所有表示里最不可迁移的一种，Waymo 和 NAVSIM 上都没有对应标签，整篇论文的开环部分就没了。

**D 留作后续消融。** residual 能修的是"控制器跟不上轨迹"；先实测固定几何/反馈控制的剩余误差，
不能假定 feedforward 已经修完。若它补的是"head 系统性地晚刹车"，那是 planner 的缺陷，
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

**固定控制器、不训练控制器的方向保留；先做独立 CARLA/TCP PID 参考与 pure pursuit 候选，
经过离线、无交互 CARLA、完整路线三层验证后再选默认。** 原稿的 pure pursuit + aim-bearing PID
不作为默认：两项都根据前方目标点角度转向，不能把后者解释为只修跟踪残差。

实现与协作材料集中在 [controller 工作目录](../todos/2026-09-22-b2d-controller/README.md)，
完整任务、验证、验收门槛见 [plan.md](../todos/2026-09-22-b2d-controller/plan.md)，
接口见 [contract.md](../todos/2026-09-22-b2d-controller/contract.md)。

| 部件 | 本轮定义 |
|---|---|
| 输入 | `update(traj_xy, t_frame)`：20×2、4 Hz 时间采样、5 s horizon，5 Hz 更新；`step(t_now, speed_mps, yaw_rate_rps)`：20 Hz 返回 throttle/steer/brake |
| 坐标 | 后轴原点，x 前 y 左；内部 yaw rate 左转正，CARLA steer 右转正，仅在边界转换 |
| 延迟 | 保留运动历史，从 trajectory 源帧到当前帧完整 SE(2) 重投影；同/倒序、未来、超出历史、stale 参考有明确 reason |
| lookahead | CARLA 风格为 `3 + 0.5v`，TCP 风格固定 4 m；`max(3, 0.5v)` 是另一个候选，不能标为 CARLA 默认；从当前位置在剩余路径的局部投影起算弧长 |
| 横向参考 | CARLA 角度 PID 1.95/0.05/0.2；TCP 归一化角度 PID 0.75/0.75/0.3，各自保留来源离散语义 |
| 横向候选 | 单独 pure pursuit：κ=2y/(x²+y²)，δ=atan(Lκ)，经测得转向映射变成 steer；不叠加到 aim point 的角度 PID |
| 纵向 | CARLA 1.0/0.05/0、km/h 误差，与 TCP 5/0.5/1、clipped delta/布尔制动分别作为适配对照；三种 preset 的速度读法在对照时固定 |
| 速度窗口 | 以绝对 trajectory 时间读取 `[age, age+0.25]`；原 `[age+0.25, age+1]` 作为可选 reference，单独比较停车误差 |
| 输出边界 | throttle≤0.75、brake≤1、|steer|≤0.8、每 0.05 s Δsteer≤0.1，油门和刹车互斥；统一限幅使 TCP 适配版不等同官方完整控制分支 |
| 主横向指标 | 后轴到 route/reference 的有符号法向距离；aim_xy/bearing 仅诊断，truth/estimated/短时 controller cross-track 分开 |
| 第一阶段输入 | policy=none 的 dense-route 诊断轨迹，每 4 tick 同步生成；真实 planner 输出接口稍后独立接入 |

**aim point 的 y 不是横向跟踪误差。** 半径 20 m 的圆弧，即使车在圆上完全贴合，前方弧长 3 m
的目标点也有 y=20×(1−cos(3/20))=0.2246 m，bearing=0.075 rad。
以它做误差会把完美跟踪判失败；把完整 CARLA P 项叠加到 pure pursuit 还会额外给出 0.14625 steer。
若后续需要 feedforward + feedback，feedback 应根据真正的 cross-track 和路径切线 heading error 定义，
先独立证明作用，不能沿用这个相加式。

**几何补偿和时间补偿是两件事。** 刚体变换不会改变相邻点距离；因此把旧轨迹重投影后仍取原第1–4点，
目标速度仍来自旧时间段。必须按 `age=t_now−t_frame` 裁剪/插值时间窗口。8 m/s 在3 s内匀减速时，
未来0.25–1 s的平均速度比当前参考速度低约1.67 m/s，追上自己的 command 不代表按原定时轨迹停车。
报告同时给相对 command 与原定时 reference 的速度 RMS，并独立验收停点误差。

**20 Hz 控制需要每 tick 的运动输入。** 原 decimate+overlap stub 在非 policy tick 保持旧控制，
不能直接声称已有 20 Hz 重投影控制环。本轮主线程每 tick 收 SPEED/IMU/GNSS、推进位姿和 step，
轨迹源 frame/time 在采样时冻结；相机包独立按完整帧路由。后台推理不写 PID 状态。
原稿预计0.05–0.30 s age及厘米级补偿精度只是估计；本轮需要用延迟注入和同帧定位真值实测。

**不能仅凭已有增益宣布适配成功。** 原模型输出点间隔、aim 规则、TCP 的 control branch 混合、
状态输入与我们不同。先在独立无交互路线验证坐标、定位和 plant 响应，再比较控制方法，
最后在 Dev10 保留所有交互失败；不根据单条结果无限调参。

### 这会改变评测的哪些语义

必须和分数一起报的：

| 项 | 我们的做法 | 和公开数字可比吗 |
|---|---|---|
| 推理频率 5 Hz，control 20 Hz | agent 侧的选择，仿真仍 20 Hz | 低频推理有先例；正式对照需固定并记录传感器协议与执行节拍 |
| 决策延迟（overlap） | 源帧时间与控制时间独立记录，不预设恰为1–2 tick | 需实测age，并在固定协议下比较；延迟对分数的净影响不先下结论 |
| ego 位姿 | GNSS/IMU/speed融合；hero真值仅写诊断日志，不参与控制 | 仅移除 transform 不足以证明可提交；dense route 本身是诊断 oracle，正式 planner 须单独核对输入协议 |
| GNSS | route driver 用于定位，原始/融合后误差同帧对后轴真值报告 median/p90/p95 | 经纬度噪声换算不能代替定位标定 |
| 红灯 / 前车决策 | 本轮route oracle不做让行/避碰；真实模型可以在输出轨迹中做这些决策，并不需要显式规则 | 旧稿仅凭“都没有手写规则”称可比不成立。当前policy=none的驾驶成绩不能与完整模型比较 |
| `max_brake 1.0` | 增益，不是语义 | 可比 |
| `MinimumSpeedRouteTest` | 固定 B2D 0.0.4 的 penalty 字典是 `[0.7, "unused"]`；criterion 定期生成事件 | 报原始事件数、percentage及官方整体penalty；事件或criterion FAILURE不能直接解释为爬行扣分 |

### 一个能开的 driver 对成本的影响

这是现在 3.1 h 里最大的水分，能量出来。`runs/b2d/full220`（220 条、8 worker、共享卡）：

| 量 | 值 |
|---|---:|
| 路线长度（`bench2drive220.xml` 的 waypoint 折线） | 均值 105 m，中位 104，p10 65，p90 139，最长 222；合计 23.1 km |
| harness finished 的路线（不等同驾驶完成） | 209 / 220 |
| 撞到 4000 tick 上限的 | **118 / 209 = 56%** |
| 没撞上限的 91 条 | tick 数 1281–2701，均值 1779，**与路线长度不相关**（r = −0.01），下限正好是 `ActorBlockedTest` 的 60 s = 1200 tick |
| 所有路线的 tick 均值 | 3033 |
| 每 tick 均值（共享卡） | 124.6 ms |
| 每条路线 tick 之外的固定开销 | 68.5 s（中位） |

读法：旧稿把这些短/长耗时直接归因为 blocked/capped；这个解释必须用官方 status、completion 和事件核对。这些数字提示计时被失败/截断主导；runner finished 仅表示 evaluator 返回。真实完成率必须重读各 attempt 的 results.json，不能把 finished 当跑完或仅凭 tick 数反推出所有驾驶状态。

一个能开的 driver：105 m 在 8 m/s 上是 13 s，加起步、一个场景交互（红灯、cut-in、行人）10–30 s，
按 600–1200 tick 估（30–60 s，Bench2Drive 的路线就是这样设计的短路线）。每条路线 wall =
tick × 124.6 ms + 68.5 s，8 个 worker：

| driver | tick / 路线 | 220 条 | 其中固定开销占 |
|---|---:|---:|---:|
| 现在的 stand-in（实测） | 3033 | 3.1 h | 15% |
| 能开的，悲观 | 1200 | **1.4 h** | 31% |
| 能开的，乐观 | 600 | **1.1 h** | 48% |

上表是旧硬件/worker协议下的预算假设，**不是更换控制器已经取得2–3×收益**。固定开销可能占比上升，
但Tokyo单GPU、不同地图和路线失败分布需重新实测。这个估计的两个边界：
一个真开得好的 driver 不会卡住，但会守红灯、会让行，tick 可能高于 1200；一个开得差的 head 仍会撞上限，
数字会回到 3 h。

### 实施顺序与验收

本轮按 [plan.md](../todos/2026-09-22-b2d-controller/plan.md) 的 T0–T8 实施：基线归档、车辆与定位标定、
NumPy controller、独立控制验证、双频率 agent、CLI/报告、完整 smoke、Dev10/保留路线、冻结配置与结论。
route 是控制诊断输入，不含避障/让行；中心线可能穿过 ConstructionObstacleTwoWays 等场景障碍物，
不能预先承诺能跑完，或把任何 blocked 都归咎 lookahead。

先用独立解析圆弧/停车轨迹验收，再在无交互 CARLA 开发路线验收，最后完整无cap smoke与原版Dev10。
Dev10 carla/tcp各10条；pursuit通过前置门槛后加入，默认与最强参考再做seed=1完整配对及预定6条保留路线。
全部attempt、驾驶失败、基础设施失败、缺失/截断都进报告，不能只展示成功片段或挑最佳重试。
只有之后固定planner/checkpoint/input做controller-only替换，才能讨论跑榜分数提升。

## 已查清与仍待验证

| 项 | 状态 | 下一步 |
|---|---|---|
| MKZ 2020 几何与转向曲线单位 | 已实测；见上方常数与 calibration-units 原始文件 | 继续检查真实响应与G2动态误差，失效sweep不拟合 |
| CARLA/TCP/pursuit 控制能力 | 离线R20m/6m/s RMS=.4671/.1546/.0091m；原四条实车开发CARLA与pursuit各4/4通过、TCP26966失败；新增真实S弯三组均有gate失败 | v1已冻结，v2修复输入轨迹后重新验证；不使用保留集调参 |
| 每条路线多少tick | Dev10seed0每组7条<600、1条600–1200、2条>1200；失败25424为4000tick | 旧600–1200预算不能约束碰撞停滞；见完整成本表 |
| GNSS raw/fused pose误差 | G2全部12例融合p90≤.425m，pursuit为.167–.392m；raw/fused完整分位数已存档 | 真值与控制分离；碰撞中负速度故障单独保留 |
| Waymo词表CARLA覆盖与停车决策 | 未验证 | 固定planner的后续闭环，不由route诊断代替 |

## 当前实验结果与数据边界

Dev10两个TM seed三组均9/10驾驶完成，两轮误差几乎一致；这里只改变TrafficManager种子，并未控制全部随机源。路线均值横向RMS为CARLA .2711m、TCP .4947m、pursuit .2951m；
平均completion为94.726%、94.914%、94.726%。因此pursuit没有满足预定门槛的默认替代证据。六条保留集三组均5/6完成，路线均值横向RMS为.4488/.7373/.5679m（CARLA/TCP/pursuit），仍不支持替换。
完整表、图、源文件哈希与失败尝试在[文章素材](../todos/2026-09-22-b2d-controller/article-notes.md)，
成本分解在[Tokyo实测段](../docs/bench2drive-cost.md#tokyo-controller-diagnostic-complete-dev10-seed0-2026-09-22)。

当前实际模型为policy=none。TCP只是控制器适配preset，未运行TCP神经网络。
2091路口的事件确认自车与背景车碰撞；没有让行规划使其与单纯横向跟踪试验不同，不能把碰撞全归因于PID。
25424施工场景的中心线碰撞后长期停滞；全部失败仍计入分母，没有剔除这些路线来改善主分数。

还发现一个输入边界：停车起步或路线偏移时，ego原点到首个future waypoint的连接段可把轨迹导数推到配置巡航速度以上。
例如3514首点距原点4.845m，按0.25s解释为19.381m/s，而后续点段仍为8m/s。
命令与该reference的误差是同一输入轨迹语义，不能代替G2独立真值巡航指标；三组冻结同一实现，原数据不篡改。
v2随后修复了该空间路线至定时轨迹的边界：从估计后轴位置及heading构造重接路径，再按路径弧长定时，保持原巡航/减速与控制参数。原dense route继续用作独立真值误差参考，修复后的完整开发验证见下节。

覆盖审计发现原四条G2与六条保留集均无足够交替曲率。新增开发S弯17563@6m/s三组都完成，但CARLA/TCP/pursuit速度RMS为.663/.922/.636m/s，均超过.5；TCP横向RMS也达到.905m。不能把此前4/4描述成完整S弯验收通过。另在实际6.357°坡上完成5s静止制动诊断，真值位移0m；它只证明静止hold，不证明坡上接近与减速过程。

## v2–v4闭环反馈与驾驶表现目标（2026-09-23）

空间路线定时已修复：从估计ego后轴与航向重接原dense route，再按弧长采样。此后六项G2开发条件包含实际S弯与6m/s直线。
v3 PI Kp=1、Ki=.25未消除全部波动；唯一追加的Kp=.5复验中，CARLA横向配PI与pursuit max均6/6通过，pursuit additive仍5/6。
候选pursuit max已在正式Dev10前冻结；不依据正式结果继续调增益。[完整迭代和图](../todos/2026-09-22-b2d-controller/iteration-v2.md)。

新smoke在罗盘NaN处暴露未处理异常。最长.2s的陀螺仪预测及超时制动/复位修复后，126项测试通过，原2390闭环完成100%，
一帧降级后恢复，控制p99=.265ms。此处既保留失败输入，也保留重放与新的闭环结果，不能只写最后成功。

用户明确目标是驾驶表现，不要求Driving Score必然提高。DS不直接惩罚加速度/jerk；官方Smoothness另算。
18对完整v3/v4闭环案例的纵向jerk RMS等权均值下降25.9%，横向加速度RMS上升2.0%，说明收益必须分项报告。
[物理诊断完整数据](../todos/2026-09-22-b2d-controller/results/comfort-v3-v4-v1/README.md)不是官方舒适性分数，也不是模型收益证明。
旧tcp-smoke/tcp-fast确实加载过真实TCP checkpoint，但本轮route oracle尚未做真实TCP的新旧控制器对照。
后续应冻结模型checkpoint、相机输入与推理节拍，比较轨迹跟踪、速度稳定、转弯、停车和舒适性，驾驶得分作为另一个观察量。
TCP原生waypoint时域和学习的control branch须先明确，不得凭空补出5s轨迹后声称只换了控制器。

正式结果已完成：候选16/20驶完全程，对CARLA横向＋同一PI的17/20；全程横向RMS为.402600m对.363227m，高10.84%。
候选DS均值59.147虽高于参考53.811，仍不满足默认替代条件。保留CLI carla/vendor，PI/max作为显式可选配置；
不再追加未触发的保留集，也不把未执行的G2 seed1补充复跑记为通过。
[最终报告与全部验收偏离](../todos/2026-09-22-b2d-controller/final-report.md)已封存；下一阶段先做真实TCP纵向两组、三路线的小规模对照。

## 会推翻候选选择的证据

- 两个PID在**无交互、物理可行、定位已达标**路径都RMS>0.5 m或发生控制导致blocked，才触发PDM-Lite对照；
  Dev10路径穿障碍或时间/定位错误不是换PID的充分理由。
- 独立pursuit未通过G1/G2，或复跑完成率/控制失败劣于最强参考，不选为默认。近似持平时保留通过门槛的简单参考。
- 合理轨迹反复跟丢且固定控制已定位明确剩余误差，才考虑residual，并单列ablation；证据用后轴cross-track、
  路径切线heading、真实轨迹与事件时间，不用aim_y当跟踪误差。
- closed-loop fine-tuning可以改变head的训练信号；控制器是否固定及版本必须随结果报告。

## 参考

本轮新增本地证据：Bench2Drive `0.0.4` / `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` 的
`leaderboard/leaderboard/scenarios/route_scenario.py`（固定MKZ请求）、`utils/statistics_manager.py`（MinimumSpeed unused、真实results语义）、
`scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py`（定期最低速度事件）；
CARLA实测为 `/data/runs/b2d/controller/calibration-units/`。以下外部agent条目为原research来源记录，
本轮主要重核本地锁定版本，不声称重新审查其所有最新分支。

- CARLA `PythonAPI/carla/agents/navigation/{controller,local_planner,basic_agent,behavior_agent}.py`，0.9.15，box 上读的。
- Bench2DriveZoo `team_code/pid_controller.py`、`uniad_b2d_agent.py`、`vad_b2d_agent.py`（分支 `uniad/vad`）；
  `ADMLP/model.py`、`TCP/model.py`（分支 `tcp/admlp`）。
- TCP: https://github.com/OpenDriveLab/TCP；carla_garage（TF++ / PDM-Lite）: https://github.com/autonomousvision/carla_garage；
  SimLingo: https://github.com/RenzKa/simlingo；CaRL: https://github.com/autonomousvision/CaRL。
- openpilot master 与 RELEASES.md: https://github.com/commaai/openpilot；"Learning to Drive from a World Model"（arXiv:2504.19077，2025-04）。
- AnchorVLA（arXiv:2607.03182）：在 anchor 词表上做 RL 的先例，见 [decisions.md](decisions.md) 第 18 条。
