# CAP-AUDIT 预登记：我们的 AlpaSim driver 在全部已存决策上的能力记分卡（不只零分），按能力 × 场景类型对人类日志与参考 driver 的差距

2026-10-12 写，写定并 push 于任何读数之前。本稿之前只看过：数据格式（run 目录结构、controller CSV 列、metrics.parquet 的指标名、评估器
corridor = 4 m 与 offroad / wrong_lane 的定义、nuPlan 原始 log 的 scenario_tag 表）、300 个 scene 的标签计数（不含任何 driver 量）。
没有算任何登记量。代码 `experiments/cap_audit/scripts/ca.py`（box 端 collect / tags / percep / bev）与 `ca_report.py`（Mac 端读数）。

## 0. 问题

到目前为止只诊断过零分（第 268、272、274 条）。组织方确认 private 与 public 同长（10 决策）、1000 个不同 scene、按 nuPlan log 均匀抽样；
private 比 public 难，本地「退化但活下来」的决策在 private 上就是零分。问题：**在全部决策上（含 90% 不得零分的 scene），按能力、按场景类型，
我们的 driver 比 (a) 人类日志、(b) 同一渲染上跑的参考 driver 差在哪里？** 只描述与定位，不训练，不重滚。

## 1. 数据（只读，box，0 卡时）

**主读数 = 10 决策 public（与 private 同长）**，1491 个 public scene 里有 rollout 的（预期 1479，12 个路线折返 scene 无 rollout），136 个 log。
每个 label 每 scene 一个 rollout，同一 label 多个 run 目录按排序「先出现的 run 目录胜」（aud1_collect 规则）。

| label | 角色 | run 目录（`$DATA_DIR/runs/`） |
|:--|:--|:--|
| `YR10m10-F-s0` | 我们，提交 #2 | `alpasim/ot3/a` + `cf1/a` manifest 的 `YR10m10-F-s0`、`@new`、`@fresh`（aud1_drift_collect 的 dirs()） |
| `YS2G-L18-s0/s1` | 我们，最后一个名额候选（seed 均值） | `onpol_long/read10/YS2G-L18-s{0,1}/b*` |
| `YS-L18-s0/s1` | 我们，第 1 轮（seed 均值） | `onpol_long/read10/YS-L18-s{0,1}/a*` |
| `APY10m10-AB-s0` | 我们的家族，提交 #1（只描述，不进参考集） | 同 YR 的 manifest |
| `WA-JEPA` | 参考 | `alpasim/c0c` manifest 的 `WA-JEPA (reference)` |
| `LTF` | 参考（SimScale 样例） | `alpasim_proxy/anchors/ltf{,2}/runs/LTF-*/2*` |
| `DD` | 参考（SimScale 样例） | `alpasim_proxy/anchors/dd/runs/DD-*/2*`（不含 `dd_smoke`） |
| `G0014` | 参考（garage model_0014） | `alpasim_proxy/anchors/garage{,2}/runs/G0014-*/2*`（不含 `garage_smoke`） |

**次读数 = 18 决策 held**（`alpasim_nuplan/onpol-held@v1`，有 rollout 的约 452 scene，39 个 log）：`YR10m10-F-s0`（`alpasim_long/held/h18/`）、
`YS-L18-s0/s1`、`YS2G-L18-s0/s1`（`onpol_long/read/<tag>/run`）、`APY10m10-AB-s0`（`alpasim_long/held/a18/`）、参考 `LTF` / `DD` / `G0014` /
`WA-JEPA`（`alpasim_long/fam/{ltf,dd,garage,wajepa0-3}`，各一次）。只用于：主读数旗标格在 18 决策上方向是否复现，以及感知列（见 2.5）。
`runs/zeros_mech/collect/` 不读不写；输出全部在 `runs/cap_audit/`。

**每个 rollout 的来源**：`controller/alpasim_controller_<rid>.csv`（ego 后轴位姿、速度 vx / vy，10 Hz，rollout local 系；所有 driver 都有）、
`rollouts/<scene>/<rid>/metrics.parquet`（评估器 2 Hz 序列：lateral_dist_to_gt_trajectory、dist_to_gt_trajectory、offroad、collision_*、
min_distance_to_obstacle_m、min_distance_to_lane_boundary_m、wrong_lane）、`aggregate/results-summary.json`（scene score 与旗标）；我们的 driver
另有 `driver-logs/drive.jsonl`（served plan、anchor、cmd）。**特权 scene**：run 自己的 scene loader（zm_read 的 `Lanes` / `objects` / `obj_state`，
第 268 条闸 G-act 已证明日志回放的他车位姿与 asl 逐点相同）：日志后轴路径（10 Hz）、vector map 车道多边形与中心线方向、他车日志轨迹（只有车辆：
shipped cache 是 vehicle-only，模拟里没有行人）。

## 2. 每个决策的能力量

决策 k = 1..K（K = 10 主读数 / 18 次读数），t_k = 17 000 + 500 000 k µs。ego 位姿取 controller CSV 在 t_k 的行；日志位姿取日志路径在 t_k 的插值。
**保留步**：每个 driver 自己的评估器截断，即第一个 dist_to_gt ≥ 4 m 或 collision_any 或 offroad 的步（含）之前的决策（long_read.cut 同规则）。
每个量先在 scene 内对保留步取均值（或标明的 min / max / 斜率），再在格内对 scene 取均值。全部量定向为「越大越差」。
「日志值」= 人类日志在同一时刻同一函数算出的值；**缺口（deficit）= driver 值 − 日志值**；同一 scene 同一时刻的日志值对所有 driver 相同，
所以「我们的缺口 − 参考的缺口」= 我们 − 参考（日志项抵消，只在 driver 自己的保留步不同处不完全抵消）。

### 2.1 横向（几何）
1. `lat_log`：|对日志路径的横向偏差|（m，评估器 lateral_dist_to_gt_trajectory）；日志 = 0。
2. `lane_ctr`：|对同向车道中心线的横向偏差|（m）。车道 = 包含后轴点、中心线方向与 ego 航向差 < 60° 的车道里中心线最近的一条；没有同向车道的决策
   该量为空（记入 `nolane` / `onc`）。缺口 = ego − 日志（日志自己也不总在中心，第 264 条日志 46% 违反车道中心 hinge）。
3. `head_lane`：|航向 − 该车道中心线方向|（deg），缺口 = ego − 日志。
4. `head_path`：|航向 − 日志路径在 ego 投影点处的切向|（deg）；日志 = 0。在转弯格里即曲率跟踪误差。
5. `drift`：scene 内 `lat_log` 对 k 的 OLS 斜率（m / 决策，≥ 3 个保留步）；日志 = 0。
6. 描述（不进旗标）：转弯 scene 上每个决策按「同弧长处航向超前 / 落后」×「内侧 / 外侧」分四类（第 268 条的早-内 / 晚-外），偏差 > 1 m 才计；
   我们的 served plan 在路径前方 10 m 处的横向偏差（`plan_off10`，第 266 条 E1 的量，日志路径代替路线，d266 路线与日志 99.89% ≤ 0.05 m）。

### 2.2 纵向
7. `speed_err`：|v_ego − v_log|（m/s）；有符号均值作描述（负 = 比人慢）。
8. `behind`：max(0, s_log(t) − s_ego(t))，在最后一个保留步（m；s = 沿日志路径的弧长，路径末端直线外延 40 m）；落后于人。
9. `ahead`：max(0, s_ego(t) − s_log(t))，同上（m）；超前于人 = 开得比人快 / 冲过人停的位置。
10. `lead_close`：ego 本车道前方有车（后轴系 0 < gap ≤ 30 m，|y| < 1.5 m，航向差 < 30°）的决策上，max(0, THW_log − THW_ego)（s，THW = gap / max(v, 0.5)，
    封顶 4 s；日志没有前车时 THW_log = 4）。scene 内无此类决策则为空。
11. `stop_speed`：日志近停的决策（v_log < 0.5 m/s）上的 v_ego（m/s）；日志 ≈ 0。scene 内无此类决策为空。
12. `brake_miss`：日志在 1 s 内减速 ≥ 1.0 m/s 的决策上，max(0, Δv_ego − Δv_log)（m/s，Δ = 1 s 后减当前）：没有像人一样刹。场景里的停止线、
    让行、前车减速都落在这里（模拟里他车回放日志、不反应）。

### 2.3 合法性
13. `onc`：后轴在对向车道（zm_read `Lanes.cls` = onc，即所在车道只有与航向差 > 120° 的）的保留步比例；缺口 = ego − 日志。
14. `nolane`：后轴不在任何车道多边形内的比例；缺口 = ego − 日志（停车场 / 上下客区日志自己也会不在车道里）。
15. `margin`：scene 内最小「车身多边形到附近车道并集边界」距离（m，并集不含车身时为 0，评估器 min_distance_to_lane_boundary_m 同定义，自己对 ego
    与日志用同一函数算）；缺口定向为 日志 − ego（正 = 比人更贴边）。
16. `cor_half`：scene 内 max lat_log ≥ 2.0 m 的 scene 比例（走廊 4 m 的一半；「活下来但用掉一半走廊」）；日志 = 0。

### 2.4 交互
17. `gap_min`：scene 内 ego 车身到最近车辆框的最小距离，截到 [0, 5] m；缺口定向为 日志 − ego（正 = 比人更近）。
18. `near_miss`：scene 内 ego 最小距离 < 0.5 m 而日志同时段最小距离 ≥ 1.0 m 的 scene 比例。
评估器自己的 min_distance_to_obstacle_m 作闸（G-gap）。

### 2.5 只有我们有的两列（不进旗标，用于归类）
- **serving**：`plan_exec` = |served plan 在 +1.0 s 的位姿 − 执行后 t_k + 1 s 的 controller 位姿|（m）；`plan_off10`（2.1 第 6 项）。
- **感知**：road_probe（第 275 条）held 重跑 `YS-L18-s1` 的 8 129 个决策上，出厂 P0 头的 ego 车道线 20 m 横向误差（左右均值）与两线都在（prob > 0.5）
  的比例（rp.py `shipped`，不改代码）；按 scene 接到场景类型。只覆盖 held 的 scene 与这一个 checkpoint。不另做 GPU 重放（非零分 rollout 的 asl
  已删，重放要重滚，不在本线范围）。

## 3. 分组（列）

格 = 一个能力量 × 一个分组；分组都是边际的（不做全交叉）：
- **全部**；
- **转弯几何**（日志在 t_0 → t_K 的航向变化 Δψ）：直行 |Δψ| < 15°、缓弯 15–30°、靠路沿一侧转弯（右行交通的右转、新加坡的左转）≥ 30°、横穿对向转弯
  （右行的左转、新加坡的右转）≥ 30°；
- **起始速度**（日志 v(t_0)）：< 0.5（起步）、0.5–5、5–10、≥ 10 m/s；
- **nuPlan 场景标签**（原始 log 的 scenario_tag，取中心 token 时刻 −1.5 s 到 +4.0 s 内所有 lidar_pc 的标签并集；一个 scene 可属多个族）：
  - `intersection`：on_intersection、traversing_intersection、*_traffic_light_intersection*、on_all_way_stop_intersection、starting_*_intersection_traversal
  - `turn_start`：starting_left_turn、starting_right_turn、starting_*_cross_turn、starting_*_noncross_turn、starting_high_speed_turn
  - `stop_tl`：on_stopline_*、stationary_at_traffic_light_*、stopping_*、accelerating_at_traffic_light*、accelerating_at_stop_sign、accelerating_at_crosswalk
  - `pedestrian`：near_pedestrian_*、traversing_crosswalk
  - `pickup`：on_pickup_dropoff、traversing_pickup_dropoff
  - `lead`：following_lane_with_*、stationary_in_traffic、*_with_lead、behind_long_vehicle、near_multiple_vehicles
  - `stationary`：stationary
  - `obstacle`：near_construction_zone_sign、near_trafficcone_on_driveable、near_long_vehicle、near_high_speed_vehicle
  - `lane_change`：changing_lane*
  - `plain`：除速度类（*_magnitude_speed、high_lateral_acceleration、high_magnitude_jerk）外没有任何标签。
- 格的最小样本：≥ 30 个 scene 且 ≥ 10 个 log 有定义值；否则只列不判。

## 4. 读数、旗标与归类

**我们的三行**：`YR10m10-F-s0`、`YS2G`（两 seed 的 scene 级均值）、`YS-L18`（同）。**参考集** R = {WA-JEPA, LTF, DD, G0014}。
CI：按 log 聚类的 bootstrap（jevdrive.stats 惯例，default_rng(0)，10 000 次抽 log，比值 = 格内 scene 值之和 / scene 数），所有 driver 用同一次抽样。

- 每格报：日志值、每个 driver 的值与缺口；「我们的超额缺口」= 我们 − 参考 的配对差（同一 scene 集，两边都有定义的 scene）。
- **旗标（登记线）**：对我们的某一行，格内 m* = min_{r ∈ R}（我们 − r）；在同一次 bootstrap 下 m* 的 2.5% 分位 > 0，即「比每个参考都差，且超出 CI」。
  **强旗标**：m* 的 bootstrap z（均值 / 标准差）> 3.75（约为格数 ~300 的 Bonferroni 双侧 0.05）。
- **敏感性**：只用我们与该参考都得分 > 0 的 scene（「都活下来」）重算 m*；报旗标格在此子集上是否仍成立。
- **复现**：18 决策 held 上同一格的 m* 点估计是否同号（参考各一次、我们的 seed 均值）；只报，不判。
- **最差 5 格**（画 BEV）：在 `YS2G` 行的旗标格里按 m* 的 2.5% 分位除以该量在全体 driver scene 值上的标准差排序（标准化的下界）；
  若旗标格少于 5 个，用 `YR10m10-F-s0` 行补，再不足就按同一排序取未旗标格并注明。每格取 m* 贡献最大的 scene，在我们缺口最大的决策画
  ego / 参考执行路径、日志路径、车道（带方向）、他车框、我们的 served plan（zm_read 的 bev 机制）。

**归类规则（写死，读数前）**，对每个旗标格：
- 量属 2.2（speed_err、behind、ahead、lead_close、stop_speed、brake_miss）→ **纵向**；
- 量属 2.4：最近车辆在 ego 前方本车道（x > 0、|y| < 1.5 m）的决策占该格 near-miss / 最小距离决策的多数 → **纵向**，否则 → **几何**（第 268 / 202 条：
  碰撞多是出车道后的侧碰）；
- 量属 2.1 / 2.3：
  - **serving**：该格我们的 `plan_exec` 比本行全部 scene 的均值高、配对差 CI 不含 0，而 `plan_off10` 不高（CI 含 0 或更低）；
  - **感知**：格属直行 / 缓弯 / plain / 起始速度类（非转弯、非路口），且感知列在该格的 20 m 误差中位 ≥ 0.30 m 或两线都在比例 < 0.80（第 272 条线）；
  - **几何**：其余（转弯、路口、上下客区），即 plan 本身画错了路。
- **杠杆**：几何且量是 margin / nolane / onc / gap_min（S 的 road / agent hinge 直接管的）→ **hinge 标签续训**（on-policy 行上的 S 族 hinge，
  注意第 274 条带方向栅格已失败）；几何且量是 lat_log / lane_ctr / head_* / drift / cor_half（没有现成 hinge 管跟踪与时机）→ **新标签课**（第 266 条 E3
  短程锚点标签 / 转弯几何标签）；纵向 → **新标签课 L**（research/self-consequence 的前车、必停点、入弯速度 hinge）；感知 → **感知侧修法**
  （第 263 条标定 / 俯仰、第 275 条：冻结 token 上加 road head 不够）；serving → serving 修法（三类之外，单列）。

## 5. 闸（任何读数前在 collect 的 gates.json 里检查；不过就停、报告，不读）

- **G-cov**：主读数每个 label ≥ 1470 个 scene 有 metrics.parquet 与 controller CSV；held 每个 label ≥ 440。
- **G-pose**：controller 在 t_k 的行存在于 ≥ 99% 的（label, scene, k ≤ 保留步）。
- **G-lat**：自算的有符号横向偏差绝对值与评估器 lateral_dist_to_gt_trajectory 在 ≥ 95% 的决策上相差 ≤ 0.10 m。
- **G-gap**：自算的 ego 车身最小车距与评估器 min_distance_to_obstacle_m 的 Spearman ≥ 0.9（评估器用圆角 / 收缩，不要求逐点相同）。
- **G-dir**：日志自己在同向车道的决策比例 ≥ 0.95、在对向车道 ≤ 0.01（同第 268 条 G-dir）。
- **G-tag**：≥ 95% 的 scene 找到中心 token 并有至少一个标签。

## 6. 产出

`experiments/cap_audit/results/`：`cap_table.md`（三行 × 全部格，含旗标与归类）、`cells.csv`（全部数）、`flags.csv`、`gates.json`、`tags.csv`；
`figs/`：热图（能力 × 分组，`YS2G` 与 `YR` 两幅，色 = m* 标准化，旗标格加框）、5 格 BEV。中文 HTML `research/cap-audit/index.html`。
新的决定行写进 research/decisions.md。

## 7. 估算

CPU：zm_read collect 在 452 scene × 8 label 上 22 s（64 worker）；本线主读数 1479 × 8 label、每决策多两个 shapely 并集与车距，估 5–15 分钟；
held 452 × 9 label 估 2–5 分钟；感知导出 < 5 分钟。无 GPU。总计远小于 1 小时，不走 staged launch。

## 8. 补记（读数前）

**补记 1（2026-10-12 18:25，任何登记量读出之前）。** 第一次全量 collect 的 G-lat 不过（pub 0.856、held 0.828，线 0.95），G-cov / G-pose / G-gap
（Spearman 0.990）/ G-dir（同向 0.995、对向 0.004）都过。原因是尺子不同，不是数据：评估器的 lateral_dist_to_gt_trajectory 量的是**车身多边形质心**
到完整日志折线的距离（`eval/scorers/ground_truth.py`），我们自算的是后轴；有航向误差时两者差到 0.1 m 以上。改为在质心处自算一次（`lat_c`）只用于
G-lat，线不变；登记量 `lat_log` 本来就是评估器自己的值，不受影响；后轴的有符号偏差 `lat` 只用于转弯的内 / 外描述。另一处与第 2 节文字不同：
评估器的 margin 用其自己的车身收缩与圆角，自算值与评估器的 Pearson 0.99（6 scene 冒烟），登记量用自算值（ego 与日志同一函数），照原文。
