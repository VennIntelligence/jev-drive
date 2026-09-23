# TCP 预测轨迹的物理契约：原点、坐标轴、时间戳与 target 仲裁

状态: done（源码结论 + 已有日志的经验核对 + Bench2Drive 原始 anno 核对；训练侧原点已确认是 GNSS 点，见第 6 节。此前这里写的是"训练侧原点还剩一处残余不确定"）
日期: 2026-09-23
上游: [../2026-09-23-tcp-controller/](../2026-09-23-tcp-controller/)（Tokyo box 上的 TCP-final-report.md、protocol-v2.md）

目标：把真实 TCP 的 waypoint 接到我们的 trajectory controller（`update(traj_xy, t_frame)`，输入为 rear-axle 原点、x 向前、y 向左、+0.25..+5 s 共 20 点）之前，先从源码确定 TCP 输出点的物理含义。此前的报告把"原点未确认"和"target 仲裁怎么处理"列为 open question，导致横向结论无法迁移到真实模型。

源码版本：`Bench2DriveZoo@8a08b07`（Tokyo `/data/third_party/Bench2DriveZoo`），`Bench2Drive@7ec25d1`（Tokyo 与 GPU box 的 commit 相同，GPU box 路径 `~/data/third_party/Bench2Drive`）。我们自己的代码在 Tokyo repo `~/mycode/jev-drive@54e4406`，本地 `main` 上还没有。下文用 `Zoo/` 和 `B2D/` 作为这两个仓库的前缀。

## 1. 训练标签怎么生成

TCP（Trajectory-guided Control Prediction，同时输出轨迹分支和控制分支的端到端模型）在 Bench2Drive 上的标签由两段代码生成。

**原始 anno（每帧的 json 标注）。** `B2D/tools/data_collect.py:792-799` 写入 `'x': tick_data['pos'][0]`、`'y': tick_data['pos'][1]`、`'theta': tick_data['compass']`。`theta` 就是 IMU compass：`data_collect.py:79` 取 `input_data['IMU'][1][-1]`。`pos` 的赋值不在已发布的代码里。`data_collect.py` 全文只在第 793-794 行读过它，而 `Env_Manager` 是一个缺了 `run_step` 的片段类。文档 `B2D/docs/anno.md:120-121` 只说它是 "current position in world coordinates"。这个类里有两个候选的换算函数，分别是 `gps_to_location`（`data_collect.py:1644-1652`，墨卡托反变换，得到 CARLA world 米制坐标）和 TCP 旧版的 `_get_position`（`data_collect.py:699-702`，`(gps-mean)*scale`）。GNSS 和 IMU 都装在车辆 actor 坐标的 `x=-1.4, y=0, z=0` 处（`data_collect.py:626-640`），GNSS 的噪声是 lat/lon 各 σ=5e-6°（`data_collect.py:362-368`），折合约 0.557 m。

**npy 汇总。** `Zoo/tools/gen_tcp_data.py:10-11` 设 `FUTURE_FRAMES = 4*5  # 10hz --> 2hz`。数据集是 10 Hz（`data_collect.py:20` `frame_rate = 10.0`）。第 188-194 行把当前帧 `i` 作为输入，未来点取 `full_seq_x[i+5:i+25:5]`，也就是帧 i+5、i+10、i+15、i+20，对应 **+0.5、+1.0、+1.5、+2.0 s**。第 174 行给 `y` 留了注释 `# TODO(yzj): need to align sign`，但代码里没有做任何符号处理。

**Dataset 里的局部化。** `Zoo/TCP/data.py:161-175`：

```python
ego_theta = self.theta[index][0] - np.pi/2 # compass on left hand (0, -1)
R = [[cos θ, sin θ], [-sin θ, cos θ]]          # rotation by -θ
waypoint_k = R · (future_xy_k - ego_xy)        # "left hand"
```

CARLA 的 compass 以北（world −y）为 0、顺时针为正，所以 compass − π/2 等于 CARLA yaw。我们的日志核对了这一点，见第 4 节。旋转 −yaw 把 CARLA world 向量（左手系）变换到 CARLA 车体系，结果是 **x 向前、y 向右**，单位 m，没有任何归一化（`data.py:175` 直接赋值）。target point 用同一个 R、同一个 ego 点做局部化（`data.py:182-189`）。loss 是 L1（`Zoo/TCP/train.py:61`），speed 输入要除以 12（`train.py:37`）。

归纳一下：标签原点是 t0 时刻 `pos` 所代表的那个车上参考点，朝向取 t0 的 compass 朝向，未来点是**同一个参考点**在 +0.5k s 时的位置。参考点的选择只能通过未来的 yaw 变化影响标签，因为 Δp = Δr + d·(h(ψ_t) − h(ψ_0))，其中 r 是 rear-axle 位置、d 是参考点相对 rear axle 的纵向偏移、h(ψ) 是朝向为 ψ 的单位向量。这一项在直道上为零，在弯道上的大小约为 d·sin Δψ。

| 项 | 训练标签 | 证据 |
|---|---|---|
| 参考点 | `pos`。源码支持它是 GNSS 点（actor x=−1.4 m），但赋值代码未发布 | data_collect.py:626-640, 1644-1652, 793 |
| 朝向 | t0 的 compass − π/2 = CARLA yaw | data.py:163；日志核对 |
| 轴 | x forward，y **right** | data.py:167-172；model.py:296-319 |
| 时间 | +0.5/+1.0/+1.5/+2.0 s（10 Hz 数据的第 5/10/15/20 帧） | gen_tcp_data.py:11, 192 |
| 单位/归一化 | m，无归一化 | data.py:161-175 |
| 噪声 | GNSS σ≈0.557 m/轴，每个标签点是两次独立采样之差，≈0.79 m/轴，零均值 | data_collect.py:363-365 |

## 2. 官方 agent 在推理时怎么用

**ego pose 与 target point。** `Zoo/team_code/tcp_b2d_agent.py:95-113` 用 `fsolve`，从路线第一个点的 world 坐标和 GPS 反解出 lat_ref/lon_ref。`:204-221` 用 `gps_to_location` 把 GNSS 读数换算成 `pos`（`:362-370`），交给 `RoutePlanner(4.0, 50.0)`。planner 先弹出 4 m 内的路线节点，再返回 `route[1]`（`Zoo/team_code/planner.py:83-118`）。`:223-231` 用 `theta = compass − π/2` 和训练时同一个 R 算出 target point，也就是 forward/right 下的米制坐标。传感器位置与采集时一致，GNSS/IMU 都在 x=−1.4（`:141-156`）。leaderboard 给 GNSS 加的噪声也相同（`B2D/leaderboard/leaderboard/autoagents/agent_wrapper.py:207-213`）。ego 车型是 `vehicle.lincoln.mkz_2020`（`B2D/leaderboard/leaderboard/scenarios/route_scenario.py:149`）。

**轨迹分支 PID（`Zoo/TCP/model.py:286-362`，参数在 `Zoo/TCP/config.py:35-53`）。**
- 先交换坐标轴，得到 `[right, forward]`（`:296-297`）。
- desired speed = Σ‖p_{i+1}−p_i‖·2/3，即 3 段相邻点距离的平均值除以 0.5 s，不包含原点到 p1 这一段（`:304-307`）。
- aim：遍历 3 对相邻点，找中点范数最接近 `aim_dist=4.0` m 的那一对，取这一对的**前一个点** `waypoints[i]`（`:310-313`）。
- angle = atan2(right, forward)/90°，右为正，和 CARLA steer 正向一致（`:317`）。angle_last 是 p4−p3 的方向，angle_target 是 target point 的方向（`:318-319`）。
- **仲裁**（`:325-331`）：满足 `|angle_target| < |angle|` 时改用 target，另外当 `|angle_target − angle_last| > 0.3`（27°）且 target 的 forward 分量小于 10 m 时也改用 target。steer 是 PID(0.75, 0.75, 0.3, 窗口 40) 作用于 angle_final 的输出。
- 纵向：`brake = desired < 0.4 or speed/desired > 1.1`，throttle 是 PID(5, 0.5, 1) 作用于 clip(desired−speed, 0, 0.25) 的输出，上限 0.75（`:336-341`）。

**分支融合。** 这一步发生在 agent 里，和上面 control_pid 内部的 target 仲裁是两回事。`PLANNER_TYPE` 有三种取值（`tcp_b2d_agent.py:277-307`）：
- `only_traj`：只用轨迹 PID。
- `only_ctrl`：只用控制分支输出的离散动作。
- `merge_ctrl_traj`：steer 和 throttle 取 α=0.5 的线性混合，brake 取两个分支中较大的那个。

最后还有一层官方尾部处理（`:316-329`）：当 speed > 1.0 m/s（|steer|>0.07 时）或 > 1.5 m/s 时，throttle 上限压到 0.05，否则为 0.5；brake>0 时二值化为 1。我们的 paired-v2 实验用的是 `only_traj`，并去掉了这层尾部处理（Tokyo `docs/b2d-tcp-controller.md:5,34`）。

## 3. 我们的 adapter 目前的假设与差异

真实 TCP 实验**从来没有把 TCP 的点喂给我们的 trajectory controller**。`scripts/b2d_tcp_comparison_agent.py` 保留原生横向 PID 和仲裁，只比较纵向。纵向用的是 3 段相邻点平均速度（`:116-117`，不含原点段），这个量与原点无关。setup 记录里写的是 `prediction_axes='forward/right; physical origin unconfirmed'`（`:79-80`），route reference 里另外记了 `rear_axle_offset_m=-1.38863`（`:206`）。可视化那一侧把 GNSS 当作原点：`b2d_tcp_visual_agent.py:155-156` 写的是 `prediction_origin=[-1.4, 0.0]`，最终报告也把 actor 和 −1.4 m 两种原点都标成"仅作可视化假设"（`TCP-final-report.md:55`）。所以现在不存在一个"错了的转换"，缺的是一个可以直接接上的契约。如果要接到 controller 上，下面这些差异都要处理：

| 差异 | TCP 实际 | controller 期望 | 量级 / 后果 |
|---|---|---|---|
| 横向符号 | y right | y left（`b2d_controller.py:3`） | 不翻转就会镜像转向，属于致命错误 |
| 原点 | GNSS 点，actor x=−1.400（推理侧已实测，训练侧见第 5 节） | rear axle，actor x=−1.38863（`docs/b2d-controller.md:55`） | 相差 **0.0114 m**，可以忽略，见下表 |
| 若误把原点当作 actor 中心 | — | — | 纵向偏 1.389 m。controller 会在前面补一个 (0,0) 点（`b2d_controller.py:228`），第一段就会多出 1.389 m，near-window 速度偏 +1.389/0.5 = **+2.78 m/s**。弯道上的横向误差见下表 |
| 时间 | 4 点，+0.5..+2.0 s，dt=0.5 | 20 点，+0.25..+5 s，dt=0.25；shape 必须是 (20,2)（`:189`），`np.arange(21)*trajectory_dt`（`:231`） | 现有 API 放不进 4 点，要么改 API，要么伪造 3 s 的数据 |
| 时间零点 | 模型输入帧的采集时刻（GNSS/IMU/三路相机同帧，见 TCP-final-report.md:61） | `t_frame` 要对应 source pose（`:224-229`） | 应传该帧的 sim timestamp，不能传推理完成的时刻 |
| 位置噪声 | 标签和推理 pose 都带 GNSS 噪声，σ≈0.56 m | controller 不建模噪声 | 只影响 target point 和标签方差，不引入偏置 |

原点假设在弯道里造成的横向误差（d·sin(κvt)，κ=1/R）：

| R (m) | v (m/s) | Δψ@2 s (rad) | 若原点为 actor：横向@2 s (m) | 纵向@2 s (m) | 实际 GNSS vs rear axle：横向@2 s (m) |
|---:|---:|---:|---:|---:|---:|
| 15 | 5 | 0.667 | 0.859 | 0.297 | 0.0070 |
| 30 | 8 | 0.533 | 0.706 | 0.193 | 0.0058 |
| 100 | 12 | 0.240 | 0.330 | 0.040 | 0.0027 |
| 500 | 15 | 0.060 | 0.083 | 0.002 | 0.0007 |

读法：如果训练原点真是 actor 中心而我们按 rear axle 解释，路口转弯 2 s 处会差将近 0.9 m。这和 TCP 在直道上的横向 MAE（约 0.4–0.5 m，见第 4 节）同一量级，所以这个问题值得确认。按 GNSS 解释的话，误差比标签噪声小两个数量级。

## 4. 已有日志的经验核对（paired-v2，6 个 attempt，9807 个受控 tick）

用 Tokyo `/data/runs/b2d/tcp-controller/paired-v2/*/attempts/*/1/tcp-control.jsonl` 里的 `model_input.gps_projected`、`compass_used`、`prediction.raw_waypoints` 和同帧 truth（actor 的 xyz 与 yaw）做了离线核对，只读日志，没有改动任何文件。

| 核对项 | 结果 | 结论 |
|---|---|---|
| 推理 pose 相对 actor 的偏移（1773 两臂，yaw 覆盖 −91°..−3°，可辨识） | 拟合 `pos = s·actor + c + d·heading + e·right`：d = −1.406 / −1.436 m，e = 0.001 / −0.005 m，残差 std 0.553 m | 推理 pos 是 GNSS 点（x=−1.4），噪声和 σ=5e-6° 吻合 |
| 同上，直道路线 1711/24211 | yaw 范围只有 ±1°，d 和常数 c 共线，无法辨识 | 不用于结论 |
| fsolve 解出的 lat/lon_ref | 每条路线都有一个常数平移（最大 ~820 m），尺度 s−1 在 0.02–0.30% | route 节点和 pos 用的是同一组 ref，平移相互抵消；尺度误差在 50 m 处不超过 15 cm |
| compass − π/2 − yaw | 均值 0.0000°，最大 0.022° | 朝向约定得到确认，compass 没有噪声 |
| 预测 forward / GT rear-axle 前进距离（稳速 >3 m/s，中位数比值） | 各路线在 +0.5/+1/+1.5/+2 s 上都落在 0.92–1.07 | dt=0.5 s、+0.5..+2.0 s 得到确认。如果 dt 不对，比值会整体偏到 0.5 或 2 |
| 横向符号 | 预测 right 与 GT right 的相关系数在 4 个 horizon 上为 +0.26..+0.29（moving，n=1908） | 与 right-positive 一致。直道信号弱，主要证据仍然是源码 |
| 用弯道辨识原点 | 三条路线在 2 s 内都没有 Δψ 绝对值 >10° 的 moving tick | **这批日志辨识不了训练原点** |
| 原生仲裁 | 9427/9807 = **96.1%** 的 tick 上 angle_final 取的是 target 的角度 | paired-v2 的"原生横向"实际上几乎都在追导航 target point，不是在跟踪网络轨迹 |

## 5. 建议

**变换（TCP raw → controller 输入）：**

```
p_k   = (x_k, -y_k)          k = 1..4          # right -> left, meters, no origin shift
t_k   = 0.5 * k  s after t_frame
t_frame = sim timestamp of the GPS/IMU/camera frame fed to the network
```

原点不做平移。GNSS 点在 rear axle 后面 1.14 cm，引起的误差低于 1 cm，而 GNSS 噪声是 0.56 m。如果想做到严格，可以沿局部切向加 0.0114 m，但不建议为此引入额外的代码路径。controller 自己补的 (0,0) 点和标签的 t0 原点一致，所以"第一个未来点必须和原点处的运动相容"（`docs/b2d-controller.md:45`）这条约束天然满足。

**horizon：告诉 controller 只有 2 s，不外推到 5 s。** 把 `update` 改成接受 (N,2) 加上 `trajectory_dt`，`_times = arange(N+1)*dt`，替代写死的 20 和 21。TCP 直接传 N=4、dt=0.5，不做重采样。`_speed_at` 本来就是在时间和弧长上做线性插值，所以这样和线性重采样到 0.25 s 等价，也不会凭空造出几何。可行性：stale_timeout 为 0.5 s，near window 读 [age, age+0.25]，最坏读到 0.75 s，远在 2 s 以内。additive lookahead 在 12 m/s 时是 3+6=9 m，小于 2 s 弧长约 24 m；即使超出，也会被 `min(station+distance, arc[-1])` 截住（`b2d_controller.py:394`）。外推到 5 s（恒速或恒曲率）会给 controller 喂 3 s 模型没有预测过的轨迹。这违反 protocol-v2.md:21 的"不补点到 5 s"，横向结论也会混入外推模型的效果。

**仲裁：做 controller-only 比较时必须关掉。** 我们的 controller 只看轨迹，看不到 target point。原生侧如果保留仲裁，就等于在 96% 的 tick 上额外拿到了导航 oracle 信号，这样比出来的是"信息量"，不是"控制器"。建议分三个臂：
- (a) 原生 control_pid，强制 `use_target_to_aim=False`，即只用 waypoint 的 aim。
- (b) 我们的 controller，吃同一组 4 点。
- (c) 官方原样（仲裁开），只作为 reference。

controller-only 的结论只看 a vs b。c vs a 单独量化仲裁本身的贡献。三个臂都保持 `only_traj`，并使用同一套 envelope。

**残余不确定和便宜的核对。** 训练侧 `pos` 的赋值代码没有发布，所以"标签原点 = GNSS 点"目前是推断，不是源码事实。支持它的证据有四条：采集和推理的传感器布置完全一致；`gps_to_location` 就在采集类里；推理用 GNSS 点算 target point，训练必须用同一个点，否则 target 输入会有系统偏差；直道稳速下 forward 比值约为 1。能和它竞争的假设是 `pos` 取自 CARLA API 的 actor location。直道日志无法区分这两个假设，但下面两个核对都很便宜：

1. **anno 直接比对（最便宜，约 1 个 mini clip）。** 用 `B2D/tools/download_mini.sh` 下一个 clip，逐帧计算 `R(−yaw)·(anno[x,y] − bounding_boxes.ego_vehicle.location)`。如果结果是 ≈(−1.40, 0) 且带 σ≈0.56 m 的噪声，就是 GNSS；如果是 ≈(0,0) 且没有噪声，就是 actor。y 的符号是否需要翻转（gen_tcp_data.py:174 的 TODO）也可以在这里一起看到。
2. **标签 replay。** 在同一个 clip 上跑 `gen_single_route` 加 `CARLA_Data.__getitem__`，得到的标签和由 `ego_vehicle.location/rotation` 算出的 rear-axle 与 actor 两种 GT 位移做比较。在转弯段用横向残差对 sin Δψ 回归，斜率就是原点偏移 d（预期约 −0.011 m）。

如果核对结果是 actor，唯一需要改的是在变换里加上 `p_rear(t_k) = p_k − 1.389·h(ψ_k)`，其中 ψ_k 取相邻点的切向，第 0 个点平移 −1.389 m。其余建议不变。

## 6. 训练侧原点核对：Bench2Drive anno 的 `pos` 是 GNSS 点（H1 成立）

第 5 节的核对 1 和核对 2 已经做完，结论是 **H1**：anno 里的 `x, y`（`gen_tcp_data.py:172-174` 读的就是这两个 key，`data.py` 以它为原点做局部化）是 GNSS 传感器位置，带 GNSS 噪声，不是 CARLA actor location。第 5 节的变换建议（不做原点平移，只翻转 y）不变，−1.389 m 的纵向平移**不需要**。

**数据。** 从 HF `rethinklab/Bench2Drive` 下了 `B2D/tools/download_mini.sh` 列表里的 3 个 clip，压缩包合计 0.95 GB，解压后连同 tar.gz 共 2.0 GB，放在 GPU box `~/data/datasets/bench2drive-mini/`（HF 走 `proxy_on`）。脚本是 [tcp_label_origin/check_pos_origin.py](tcp_label_origin/check_pos_origin.py)，只读 anno，在 box 上用 `ssh autodl '~/data/envs/jevdrive/bin/python - <root>' < check_pos_origin.py` 运行。和 `gen_tcp_data.py` 一样丢掉每个 clip 的最后一帧。

**逐帧偏移。** 对每帧计算 d = R(−yaw)·(anno[x,y] − ego_vehicle.location[x,y])，yaw 取 `bounding_boxes` 中 ego 的 `rotation[2]`，d 落在 CARLA 车体系（x 向前、y 向右）。H1 预测 d ≈ (−1.40, 0)、每轴抖动约 0.56 m；H2 预测 d ≡ (0, 0)。

| clip | 帧数 | yaw 跨度 | mean d (fwd, right) m | std d m | 相邻帧 d_fwd 相关 | 带常数项的拟合 d_fwd / d_right m |
|---|---:|---:|---|---|---:|---|
| DynamicObjectCrossing_Town02_Route13_Weather6 | 213 | 4° | (−1.377, +0.087) | (0.525, 0.536) | −0.03 | 不可辨识（yaw 跨度太小） |
| OppositeVehicleTakingPriority_Town13_Route600_Weather2 | 133 | 18° | (−1.388, +0.075) | (0.389, 0.559) | −0.00 | −1.26 / +0.08 |
| VehicleTurningRoute_Town15_Route443_Weather1 | 227 | 91° | (−1.383, +0.061) | (0.443, 0.499) | +0.07 | −1.437 / −0.043，常数 c=(0.05, 0.12) |
| 合计 | 573 | | **(−1.382 ± 0.019, +0.074 ± 0.022)** | (0.464, 0.527) | | |

读法：三个 clip 的纵向均值都落在 −1.38..−1.39，标准误 0.02 m，和 GNSS 安装位置 x=−1.4 吻合，和 H2 的 0 相差约 70 个标准误。抖动在 0.39–0.56 m，相邻帧几乎不相关，符合每 tick 独立采样的 GNSS 噪声。VehicleTurningRoute 转了 91°，带世界系常数项拟合时常数 c 接近 0，d_fwd=−1.44，说明这个偏移确实固定在车体上，不是 lat/lon_ref 带来的世界系平移。横向有一个 +0.07 m 的小均值（约 3 个标准误），量级远小于噪声，对 controller 没有影响，这里不追究。`theta − π/2` 与 actor yaw 的最大差为 0.0000°，anno 的 theta 没有噪声，这和第 4 节在推理日志上的结论一致。

**标签 replay 与 y 符号。** 按 `data.py:161-175` 在全部 anchor 上重建 TCP waypoint 标签，和 actor 真实未来位移在 t0 车体系中的投影（x 向前、y 向右）比较，只取速度 >1 m/s 的 anchor，n=374：

| horizon | corr(label_y, GT right) | label_y 对 GT right 的斜率 | GT right 范围 m |
|---|---:|---:|---|
| +0.5 s | +0.10 | +0.46 | −0.9..+1.0 |
| +1.0 s | +0.32 | +0.68 | −2.1..+1.6 |
| +1.5 s | +0.55 | +0.81 | −3.0..+2.3 |
| +2.0 s | +0.73 | +0.85 | −3.9..+4.0 |

- **y 是 right-positive。** 相关系数全部为正，并随 horizon 增大（短 horizon 上信号被约 0.8 m 的双采样噪声淹没）。在 |Δψ|>10° 的 42 个 +2 s anchor 上，label_y 与 GT right 的符号 42/42 一致。`gen_tcp_data.py:174` 的 `TODO(yzj): need to align sign` 不需要任何处理：标签就是 CARLA 车体系的 forward/right，和第 1 节的源码推断相同。接 controller 时仍然要做第 5 节的 y → −y。
- **弯道独立估计原点。** 标签减去 actor 位移的横向残差应为 d·sin Δψ + 噪声。在 1496 个 (anchor, horizon) 点上回归（最大 |Δψ| 30°），得到 **d = −1.48 ± 0.22 m**，截距 −0.01。这个估计和逐帧静态偏移互相独立，结果同样指向 GNSS 点。斜率小于 1 也是同一个效应：右转时 GNSS 点在后方，横向位移比 actor 中心少 1.4·sin Δψ。

**结论。** 训练标签和推理 pose 的原点都是 GNSS 点，也就是 actor x=−1.4、在 rear axle（x=−1.389）后方 1.1 cm。第 5 节的契约 `p_k = (x_k, −y_k)`、不做原点平移，现在从"推断"升级为"数据核对过"。第 5 节末尾那条"如果核对结果是 actor"的备选分支作废。
