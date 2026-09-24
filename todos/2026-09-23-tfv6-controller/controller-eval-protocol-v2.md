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
