# B2D 控制器复验：P5 按原样过验收（时间索引 replay）

状态: 预注册（2026-09-25 20:20 CST，写于任何本批臂运行之前）
上级: [闭环基础设施验收](../2026-09-25-closed-loop-infra-acceptance.md)；第一轮验收：[b2d-controllers.md](b2d-controllers.md)
代码: `scripts/b2d_zeroshot_agent.py`（`controller_preset: pursuit`、`replay_plan: time`）、`scripts/infra_ctl_accept.sh v2`、
`scripts/infra_ctl_score.py`、`scripts/test_infra_ctl_p5.py`

## 问题

第一轮验收里没有控制器通过；固定控制器 2 Hz（F2）横向合格、19/20 完成，但纵向比专家时刻表落后约 3 m、多撞 6 条路线。
[tfv6-controller 的 L1 v3](../2026-09-23-tfv6-controller/controller-scorecard.md)（决策 41）已经在 D 的横向上做完了一版纵向
重设计，P5 是终版：近静止减速一律刹车、PCHIP 读 plan 时间并做前馈、按实测 plan 周期自适应的过期阈值、前馈裁到
[−4, +3] m/s²、低频 plan 停在终点前时按位置逼近终点；它是 L1 上唯一在 1 Hz 与 2 Hz plan 下都 14/14 完成的控制器，但还没有
跑过闭环。这里**不调任何参数**，把 P5 原样放进验收，看它在专家 plan 上能不能跟住、分数接近专家。

## P5 是什么、怎么接进来

- **配置**：`todos/2026-09-23-tfv6-controller/controller-eval/P5.json`（preset `pursuit`，`longitudinal_mode: accel`，
  `plan_interp: pchip`，`feedforward_tau_s 0.3`，`adaptive_stale`、`stale_factor 2.5`，`feedforward_limit [−4, 3]`，
  `terminal_approach`，`low_speed_brake_mps 2`，横向 = D：`pursuit_frame rear_slip`、`steer_inverse ackermann`）。
  代码是 `scripts/b2d_controller.py`，最后一次改动就是 P5 的 commit `f8ea8bd`，与 scorecard 的 L1 v3 同一份。
  油门上限沿用默认 0.75（P5 没有改它）。
- **适配**：考试 agent 原来把 controller config 除 `rear_axle_offset_m` 以外全部交给 `Controller`；P5.json 里还有一个
  pose-adapter 键 `pose_lateral_coefficient_s2_per_m`（0.0107，`b2d_controller.ADAPTER_KEYS`），L1 harness（`b2d_agent.py`）
  把它交给 `PoseFilter`，L2/L3 的 TFv6 agent 也同样硬编码进 `PoseFilter`。考试 agent 现在同样处理：这个键进 `PoseFilter`，
  其余原样进 `Controller(preset="pursuit")`。固定控制器的 config 没有这个键，取 0 即 `PoseFilter` 的默认值，F2 行为不变。
- **核对（不改变行为）**：`scripts/test_infra_ctl_p5.py` 把考试 agent 的构造路径和 L1 / L2 用的
  `pursuit_from_config(P5.json)` 放在同一条合成 plan 流上（2 Hz 与 1 Hz，起步、停车 7 s、再起步，带 1 cm 横向噪声），
  1200 个 tick 的 throttle / steer / brake 逐位相同。控制器输入的约定（`step(t, controller_speed(v), -world_gyro)`、plan 为后轴
  rig 系 x 前 y 左、0.25 s 间隔 20 点、GNSS 在 x = −1.4）与 L1 harness 相同；F2 在同一接口上 lat_ratio 0.87，说明坐标约定对。
  没有在 box 上重跑一个 L1 case：L1 的参考轨迹与原始日志在另一台机器上，控制器代码与 scorecard 那次是同一 commit，
  构造又是逐位相同，重跑只会重复验证同一件事。

## 协议改动：replay plan 改成时间索引、平滑追上（第一轮必改项 2）

第一轮的 plan 来源（偏离 2）：把车投影到专家路径得 s0，取专家在 s0 附近 ±2 m 停留的时间窗 [ta, tb]，t* = 经过时间截到
这个窗。车停在专家停车点后 2 m 以外时 t* 被冻住，plan 不再推进，这会放大没有位置反馈的控制器的“卡住”。

新定义（`replay_plan: "time"`，`_replay_path`）：τ = 经过时间，落后量 d = s_e(τ) − s0，

  s(t) = s_e(τ + t) − d · e^(−t / 2 s)，t = 0.25 … 5 s，再截到 s ≥ s0 并对 t 取单调不减（planner 不要求倒车）。

τ 永远前进，所以 plan 不会冻住；第一个点在 s0 + v_e·0.25 s + 0.12 d，不会跳到车前。车领先时 plan 放慢或原地等；专家在
停车点等待时，落后的车被带到停车点、然后和专家同时起步。专家日志之后仍按最后速度直线外推 10 s（第一轮的偏离 2 (1)）。
合成轨迹上的单元检查（准时、落后 10 m、领先 3 m、专家等待时车在 8 m 外 / 已到停车点、日志之后、起点）都与公式逐点一致。

这个改动影响所有控制器臂，所以**用新协议重跑 F2 作为配对参考**（臂 f2t），P5 的比较都对 f2t 和专家两边配对。
仍然不变的限制：plan 是专家的轨迹、不对被控车的偏差作反应；e_lon 仍按经过时间对齐（落后照样计为纵向误差）。

## 臂

| 臂 | 控制器 | plan 节奏 | 用途 |
|---|---|---|---|
| E-a / E-b | PDM-Lite 专家（第一轮的两遍，**复用**） | 20 Hz | 参考；E-a 的日志就是各臂的 plan |
| f2t | 固定控制器（`b2d_controller.py` carla，同第一轮 F2） | 2 Hz（Alpamayo 10 Hz rig，plan_every 5） | 新协议下的配对参考 |
| p5x2 | P5 | 2 Hz（plan_every 5） | Alpamayo 考试节奏 |
| p5x5 | P5 | 5 Hz（plan_every 2） | openpilot 考试节奏 |
| p5x1 | P5 | 1 Hz（plan_every 10） | 低频 VLA；成本与其他臂相同，一起跑 |

专家不重跑：专家不经过 replay，协议改动不影响它；第一轮 E-a / E-b 在同一棵树、同一 seed 上逐路线 |ΔDS| 平均 0.26。

## 判据（跑之前定死，与第一轮相同）

每臂分别判，A1–A5 全部满足才算 **pass**；一个控制器只在它通过的节奏下被验收。

| # | 判据 | 门槛 |
|---|---|---|
| A1 横向跟踪 | pooled e_lat | median ≤ 0.30 m 且 p95 ≤ 1.00 m |
| A2 纵向跟踪 | pooled \|e_lon\| | median ≤ 2.0 m 且 p95 ≤ 8.0 m |
| A3 横向执行 | lat_ratio | ≥ 0.7 |
| A4 结果 | 20 条平均 DS | ≥ E-a − 5 = **90.5** |
| A5 不卡住 | E-a 完成而本臂 blocked / timeout 的路线数 | 0 |

只报告、不作判据：完成数、卡住数、碰撞数与“专家没撞它撞了”的路线数、signed e_lon 中位数（负 = 落后，“lag”）、e_head；
每臂对 E-a 的逐路线配对差（DS、完成、碰撞数），以及 P5 各臂对 f2t 的配对差（再加逐路线 e_lon 中位数），均值 + route
bootstrap 95% CI（10 000 次，seed 0；`infra_ctl_score.py`）。基础设施失败的 attempt 由 runner 重试一次（`--max-attempts 2`），
仍没有结果的路线按缺失报、不进 A4，重试次数随结果报告。

**如果 P5 不过**：只诊断、不调参。按判据拆横向（A1 / A3）与纵向（A2、lag）；逐条看失败路线的场景类型
（施工绕行、路口转弯、前车急刹、cut-in、行人、红灯）和控制器 `reason`（`tracking` / `stop_hold` / terminal approach / stale），
写清楚是哪一层、哪类场景，交给用户决定是否继续。

## 运行与成本

`ACCEPT_OUT=$DATA_DIR/runs/infra-accept/b2d-ctl-v2 B2D_RUN_EXTRA="--client-threads 8" scripts/infra_ctl_accept.sh v2 2
f2t:740 p5x2:743 p5x5:746 p5x1:749 f2t:752 p5x2:755`（tmux `jev`，GPU 2，6 个 CARLA server，index 740–757）。
估计：第一轮 F2 一臂 20 条合计约 84 worker-min（当时同卡 12–17 个 server），4 臂约 340 worker-min，6 个 runner 约 1 h 墙钟；
GPU 2 上同时还有 P5 v1 的 6 个 server，所以按 1–1.5 h 计。这个 harness 的成本已在 [profiling.md](profiling.md) 量过
（瓶颈是 GPU 渲染，不在我们的代码里），本批不再单独 profiling。
