# Task 10 评测协议修订 v2（Mac 接手后，冻结于看到任何 v2 结果之前）

状态：**冻结**。规则仍是 [controller-eval-rules.md](controller-eval-rules.md)；本文件替换 v1 操作协议（[controller-eval-protocol.md](controller-eval-protocol.md)，`21cb250`）中下列各条，其余条款（指标公式、censoring、统计方法、判定逻辑、L3 护栏）不变。v1 的 160 个 route-oracle case 与 149 个 "expert" case 保留，只作描述，不进判定。

## 为什么改

读 v1 的代码和临时结果时发现三处会让结论偏掉的问题：

1. **L1 route oracle 的输入和打分目标不是同一条轨迹。** 喂给控制器的是 `RouteAdapter.trajectory`，它从当前位置直接按巡航速度铺开，车静止时要求"立刻 8 m/s"；打分用的却是起步加速度 ≤2 m/s² 的理想时间轨迹。A/B 忠实执行了阶跃（2 s 内到 8 m/s），于是比打分参考领先约 5 m 并一直保持，`time_xy_rms` 被放大到 4 m 以上；C 自带的起步限幅恰好接近打分器的斜坡。例：24816/p01，A/B/C 在 1、2、3 s 时的速度分别为 3.0/8.2/8.2、3.0/9.0/7.8、1.7/5.2/7.4 m/s。v1 L1-oracle 的 primary 差异主要由这个接口伪差决定，不能判定控制器优劣。
2. **L1 "expert" 参考由 C 自己开出来。** 它是真值位姿下的 `prod-truth` pursuit 控制器，C/D 去跟踪自己家族的轨迹。改用 LEAD 自带 expert 也不行：LEAD expert 的横向 PID（kp 3.118、ki 0.641、kd 1.378、speed_scale 0.9755）和纵向线性回归参数与 TFv6 的 A 执行层逐项相同，A 就是 PDM-Lite expert 的控制器，那样的参考偏向 A。
3. **L2 可行/不可行分类用了控制器自己的跟踪误差。** `a_req=(v_plan−v_ego)/0.5` 里的 `v_plan−v_ego` 就是速度跟踪误差，跟得差的控制器会有更多帧被划成"不可行"而从可行指标里剔除。
4. 另外，8 条路线只有 8 个 bootstrap cluster，CI 不稳。

## L1 参考（替换 v1 的 oracle / expert 两类）

两类参考都由 [b2d_controller_eval_refs.py](../../scripts/b2d_controller_eval_refs.py) 只根据路线的稠密全局路径和名义出生点（每条路线一个 40 tick 的 A/p00 probe 取得）生成，**同一条时间参数化轨迹既作为输入喂给所有控制器（fixed-trace 接口，A/B 经 v1 已核对的作者公式适配器），也作为打分目标**：

- `ramp`：route oracle 规律，起步 ≤2 m/s²、按路线巡航速度（`l1-cruises.json`）巡航、终点 2 m/s² 减速停车，速度在时间上做两次 1 s 平滑以限制 jerk。
- `profile`：中性的 stop-and-go 剖面，种子由路线 ID 的 SHA-256 决定。每 30–60 m 换一个巡航档（0.5/0.75/1.0/1.25×巡航，上限 10 m/s），按曲率限速（横向加速度 ≤2 m/s²），路线长于 80 m 时中途停一次、长于 160 m 时停两次（停留 2.5–4 s），终点停车；起步 ≤1.5 m/s²、制动 ≤2.5 m/s²，同样两次 1 s 平滑。

两类都在物理上可行，与 A/B/C/D 都无关。L1 显著更好要求 paired route-cluster bootstrap 95% CI 上界 <0 **在 ramp 和 profile 上同时成立**。primary 公式、censoring、`extra_jerk_rms` 等沿用 v1。

描述性附表：v1 的 step-oracle（车静止时阶跃到巡航，归为"不可行起步指令的响应"）和 C 驾驶生成的 expert 参考交叉表，均不进判定。

## L1 路线与扰动

- held-out 扩到 **40 条**：v1 的 8 条加 32 条新抽，来自 bench2drive220，按城镇分层随机（小地图 20、Town12 8、Town13 4，种子 20260925），见 `controller-eval/selection-v2.json`、`l1-v2-heldout.xml`。排除所有在控制器开发中实际开过的 27 条路线（W2/W2b、D2/D3、search、A-defects、lateral-v2、TCP 控制器工作）和 Task 10 dev 路线。
- 扰动：p01–p03（v1 登记的 GNSS/IMU 噪声种子和出生偏移）。每条路线一个 CARLA 进程，3 路并行；基础设施故障（零 tick 的 RPC 超时或 setup error）只补缺失 case，每路线最多 3 次，其他异常立即停批。
- 接口扰动（`short_2s`、`sparse_5s`、`stop_jitter`，定义同 v1）在 profile 参考上跑，路线为 40 条中按列表顺序每隔两条取一条（约 13 条），p01，四个控制器。鲁棒性按 v1 定义报告，不进判定。

## L2 可行性分类（替换 v1 的 a_req 与 a_internal）

分类只用 plan 自身：`a_plan = (v_[0.5,1.0] − v_[0,0.5]) / 0.5`，其中两段平均速度由 plan 在 0.5 s 和 1.0 s 处的点求得（TFv6：`|p1|/0.5` 与 `|p3−p1|/0.5`，waypoint 间隔 0.25 s；TCP：`|p0|/0.5` 与 `|p1−p0|/0.5`，间隔 0.5 s）。可行 iff `−4 ≤ a_plan ≤ 3` 且 `a_lat = v_[0.5,1.0]²·|κ| ≤ 6`（κ 为 v1 的三点曲率；速度 <0.5 m/s 时横向项置零）。TFv6 的 A 执行 route/target-speed head，但分类统一用同一帧的 waypoint head，使四个 arm 的分类口径一致。v1 的相邻段内部加速度判据删除：在 dev 数据（W2/W2b 的 66 个 A case，36655 帧）上它中位数 6 m/s²、P95 32 m/s²，主要是 waypoint head 的噪声。同一 dev 数据上新判据 a_plan 中位数 −0.04，P5/P95 为 −4.34/5.26，可行帧占 83.6%。旧的 ego 判据作为敏感性分析另报。

跟踪目标 `v_plan`、各 L2 指标、判定阈值不变。

## L2/L3 路线

扩到 **16 条** 带 scenario 的路线：v1 的 8 条加 8 条新抽（与 L1 新抽路线不重叠），见 `l23-v2-heldout.xml`。种子 0、1；TFv6 跑 A/B/C/D，TCP 跑 N/A/B/C/D。TCP wrapper 先在 dev 路线 24240 冒烟，检查 model forward、native PID、帧对齐和实际选中 control 后才进正式批。

## 冻结后的三处接口修正（2026-09-25，均在前两条 held-out 路线的逐 case 抽查中发现，此时没有任何汇总结果）

第一批 ramp 在 24816、25845 跑完后抽查，A 几乎每例碰撞或 blocked。逐 tick 看原始记录后改了三处，原始尝试保留在 `void-*` 目录，之后整批从头重跑：

1. **A 的 route 输入。** A 的 route PID 原生输入是与车速无关的空间 route（导航给的 1 m 间隔 checkpoint）。v1 适配器从 5 s 时间轨迹里重采样 route，低速时整条轨迹不到 1 m，8 个 checkpoint 全塌到一点，瞄点失效（v1 pilot 中 "A replay 3.0 s 撞静态物体" 即此）。现在 A 拿参考路径在自车前方 1–8 m 的几何（进度单调），目标速度仍取时间轨迹 p2–p4。
2. **plan 刷新率。** 名义条件 plan 每 tick 刷新（20 Hz）。TFv6 闭环里 route/waypoint 每 tick 都在变（W2 日志核实），v1 的 5 Hz 让为 20 Hz 设计、带逐 tick 微分项的 A/B 每 4 tick 吃一次瞄点跳变；C 自带按位姿历史补偿过时 plan 的机制。5 Hz 过时 plan 改作接口模式 `stale_5hz`。
3. **plan 的坐标系。** 名义条件下参考轨迹用真值后轴位姿转到车体系（理想 planner，相当于感知模型直接在车体系里出 plan）；控制器自己的车速、IMU 等输入仍是带噪估计。v1 用带 GNSS 噪声的滤波位姿转换，前方 4 m 的 route 点相邻 tick 横跳 ±0.3 m，被 A 的微分项放大成发散蛇行；TFv6 里 A 的 route 来自模型的 route head，不经过定位。经噪声位姿转换的 plan 改作接口模式 `pose_plan`。

修正后两条路线 12 例：四个控制器 CTE 均 0.01–0.05 m；A/B 仍 blocked，但原因是真实缺陷：起步近全油门领先参考约 4.7 m，巡航掉速 0.15–0.7 m/s，终点前 plan 速度归零即刹停、没有位置环，停在终点前 1.4 m（A）/3.5 m（B）。报告需说明：固定时间轨迹不像闭环 planner 那样每 tick 从自车位置重规划，时间滞后分项在闭环中会被部分吸收，因此 CTE、速度误差、时间滞后三项分开报告，闭环影响由 L2 衡量。

接口模式因此为 `short_2s`、`sparse_5s`、`stop_jitter`、`stale_5hz`、`pose_plan` 五种。

## 第四处修正：ramp 参考加曲率限速（2026-09-25）

ramp 跑到 22/40 条路线时查碰撞（C 在 2143 切弯撞人行道，A 在 23687/24092/2903 弯道偏出约 1 m），顺带核查参考本身，发现 ramp 全程按巡航速度，没有曲率限速：40 条路线中 13 条的最大横向加速度超过 4 m/s²，最高 9.6 m/s²（半径约 7 m 的弯以 8 m/s 通过），违反"L1 只用物理可行参考"。profile 参考最大 2.3 m/s²，不受影响。修正：ramp 与 profile 用同一个曲率限速（a_lat ≤ 2 m/s²），其余规律不变。参考生成是确定的，限速不起作用的路线参考文件逐字节不变，这些路线已跑的结果保留（打分器逐字节核对喂入文件）；参考改变的 14 条路线中已跑的 8 条移入 `void-l1-ramp-uncurved/` 重跑。上述碰撞都发生在低速（1.5–3 m/s）急弯，与 ramp 的过弯超速无关，重跑后再看是否复现。
