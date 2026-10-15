# 预登记：纵向诊断 LON-DIAG（不训练；Tokyo CPU 读存档 + 盒上一个 GPU 开环探针），2026-10-15

背景：[research/zero-taxonomy.md](../../../research/zero-taxonomy.md)（RSG-L18-s0，PAI 281，gains 20，旧的未保护 serve 路径 `r281/rsg`）把 PAI 离 0.94 的
36.8 分里 13.5 分记在「慢」、6.0 分记在「人停了车没停」（`cor_route_end` 17 个零分）。用 `results/taxonomy/pai281_rsg.csv` 按考题 72
（`experiments/alpasim/results/pai_exam/hard_exam.txt`）再切一刀：

| 主因 | 全 281：个数 / 丢分 | 考题 72：个数 / 丢分 | 已知事实 |
|:--|--:|--:|:--|
| `cor_route_end` | 17 / 17.0 | **11 / 11.0**（考题 38 个零分里的 11 个） | 17 个里自车到人停车点时速度比全部 ≥ 1.0（1.00–1.56）：这些场景**不慢** |
| `low_slow_free` | 73 / 31.7 | 17 / 6.5 | 速度比中位 0.59；39 个被追尾，追尾时日志车速中位 19.2 m/s、自车 7.5 |
| `low_slow_behind_lead` | 12 / 6.4 | 1 / 0.6 | |
| `low_late_start` | 3 / 2.4 | 2 / 1.5 | |

已定的机制，不再重读：慢 = 速度历史输入的棘轮（d285：换日志速度历史恢复 83 %；d294：喂入速度趋势系数 +0.71–0.86，新奇度不是主因；
第一下变短不是棘轮引起、棘轮在之后接管）；serve 侧 hold 去掉 86 % 的短但 39 个前撞（vhist）、holdlead 全集 +0.052 而考题持平（d292）；
训练侧 `--ego-drop-speed 0.5`（RSGallV）去掉 75 % 的短但公开零分 +8.5、GT +0.056 m（allrows）；`--on-mode both`（RSGboth）开到人的速度但 held
零分 +7、多出的零分 75 % 发生时比人快：追尾、人停着它往前蹭、弯里跑宽（rsgboth_v）；纵向不对几何响应：会为前车刹（B2），不为横穿 / 对向 /
停止线减速（d310）；冻结 token 读得出红灯（AUC 0.936，stopline_probe），但 B 项在池里只亮 0.5 % 的行、RSGB 失败（traffic-light-probe）；
开环上 RSG 自己的 plan 在 on-policy 行上 1.9 % 越过红灯线、平均越 13.7 m，人类未来 0 %（results/lon/lonprobe）。

**本次要回答的、还没有读数的问题**：
1. 17 个「人停了车没停」里，人为什么停（红灯 / stop sign / 排队 / 评估器口径 / 其他），车在人停车点前 3 s 有没有任何减速意图；
2. 慢的 73 个里，短从哪一步开始、起点处场景里有什么、交接时（喂入历史 = 日志速度）plan 是否已经短、短是否随日志车速升高（训练分布上限）、
   回路的不动点（plan 速度 = 喂入速度的那个速度）在哪；
3. 开环上，策略的纵向输出读的是**速度历史**还是**画面**：同一帧换历史，plan 的速度跟着历史走多少；红灯前把人已在减速的历史换成匀速历史，plan 还刹不刹；
   对前车同样做一遍作阳性对照；人速 ≥ 15 m/s 的行上 plan 是否本来就短。

先写线后算；修改只追加补记。

## 数据（只读）

- PAI 281 RSG-L18-s0：Tokyo `/data/runs/route_night/r281/rsg`（16 chunk；`runs/*/driver/drive.jsonl` 每决策 `ego`(20)、`hist`、`poses`、`poses_model`、`fix`、`cmd`、`rmem`；
  `controller/*.csv` 10 Hz；`rollouts/<scene>/<rid>/metrics.parquet`、`rollout.asl`（actor 位姿 + 类别）、前视 mp4）；rf_tax extract `/data/tmp_tax/pai_rsg.pkl`
  （日志路径与速度、路线、车道图、停止线 `pai_wl.pkl` 同目录或按 rf_tax 头部重抽）。零分类 / 低分类用 `results/taxonomy/pai281_rsg.csv`，不重跑。
- 考题 72：`experiments/alpasim/results/pai_exam/hard_exam.txt`。
- 开环：盒上 rm_train.build 的离线 store（`navtrain_full.s0of12`，`--yr-mass 0`，与 `rf_lonread.py` 同法）+ checkpoint RSG-L18-s0；红灯线标签 = LONG-PROBE 的
  `lc_ctx`（stopline_probe 的 nuPlan 标签路径：路线 → 停止多边形 type 2 → lane connector → 日志 `traffic_lights`）；前车 = on_lon 的 A-eligible 规则。
  ego 行的速度项 = rm_train `EGO_SPEED`（列 4–9、11、12、14、15、17、18：vx vy ax ay 与 4 个历史位姿的 x / y）。

定义（公用）：plan 速度 v_p = served plan 从后轴起 2 s 弧长 / 2（开环用 poses 的 0.5–2 s 段）；喂入速度 v_f = ego 行的 vx；日志同位置速度 v_log = rf_tax 的
位置对齐（自车弧长 → 日志弧长）；短 = v_p − v_f；「自由」决策 = 路线走廊 40 m 内无同向在动或静止车辆、60 m 内无停止线、60 m 内路线转角 < 30°、不在路口。
CI 用场景聚类 bootstrap（B 10 000，seed 0，百分位）。

## 读数 1：`cor_route_end` 17 个（Tokyo CPU）

每个场景记：
- 人的停车原因（规则，互斥，按顺序）：**E** 评估器口径 = 日志末速 ≥ 1 m/s（人没停，自车比人快、先到终点）或日志全长 < 2 m（人没动）；
  **Q** 排队 = 人停车时刻、人停车点前方路线走廊 15 m 内有静止（< 0.5 m/s）车辆（asl）；**L-STOP** = 人停车点前 0–12 m 有 STOP 类 wait line；
  **L-LIGHT** = 同范围只有 UNKNOWN 类 wait line（PAI 地图灯口的停止线不标类型，traffic-light-probe）；**N** = 以上都不是。
  并记红灯行为规则的 24 个场景（`results/red/pai_red24_rsg.csv`）命中与否。
- 车的反应：自车到达人停车点的时刻 t_r（弧长对齐），t_r 前 3 s 内每个决策的 v_p / v_f；**减速意图** = t_r 前 3 s 内任一决策 v_p ≤ 0.7 v_f；
  到达时车速、比人早到几秒（已有 `ahead_of_log_s`）、`fix` 里 lead 是否触发。
- 人的减速剖面：人在停车点前 40 m 的 v_log(s)，与车同弧长的 v 画在一张图上。

线：
- **L1（车读不读「该停」）**：14 个非 E 场景里 ≥ 10 个没有减速意图 → 「不读」（问题在信号，不在时机）；≤ 5 个 → 「读到了但晚 / 弱」（问题在幅度）；中间 → 混合，按原因类分报。
- **L2（信号来自哪里）**：L-LIGHT + Q ≥ 8 → 标签在 nuPlan 里有（灯态 / 前车），可以在训练输入 / 标签上做，不需要 PAI 的东西；N ≥ 5 → 要看片才知道人为什么停，进「给用户看」清单。
- 产物：`results/lon_diag/route_end.csv`、`route_end.md`，接触表 `figs/lon_diag_route_end.png`（每场景：车到人停车点时的前视帧 + BEV，标原因类与 v_p / v_f 曲线；给用户看片用）。

## 读数 2：慢的 73 个 + 跟车 12 个（Tokyo CPU）

逐决策表（全 281 场景都算，慢类只是切片）：k、t、v_f、v_p、v_log、短、自由标志、走廊内最近车辆距离与速度、60 m 内路线最大转角、60 m 内停止线距离、路口标志、考题标志。

- **R2a 交接处的短**：k = 0–2（历史还是日志的）的 v_p / v_f 中位按 v_log 分箱（< 8、8–12、12–16、≥ 16 m/s）。
  线：≥ 16 箱 < 0.85 且 < 8 箱 ≥ 0.95 → **分布上限**（训练里没见过这么快，plan 一上来就短），与读数 3(iii) 合读；各箱都 ≥ 0.95 → 交接时不短，短全在回路里。
- **R2b 起点**：每个慢场景的起点 = 第一个 v_p < 0.9 v_f 且之后 3 个决策都不回到 0.95 的决策。起点前 1 s 的场景特征（走廊内车辆 40 m 内、停止线 60 m 内、转角 ≥ 30° 60 m 内、路口）
  对同场景起点前 3–6 s 的对照决策的相对比例 RR（场景聚类 CI）。线：任一特征 RR ≥ 2 且 CI 下界 > 1 → 有触发物；全部 CI 含 1 → **无触发**的棘轮（第一下变短没有画面原因）。
- **R2c 不动点**：全 281 的自由决策上，v_p 对 v_f 的分箱均值曲线（1 m/s 一箱）与斜率 g（OLS，场景聚类 CI）；不动点 v* = 曲线与对角线的交点。
  线：g < 1 且 v* ≤ 10 m/s → 策略有一个低于 PAI 日志车速的「舒服速度」，棘轮衰减到它；g ≥ 0.95 → 没有不动点，短是常数偏移。报考题 / 非考题分开。
- **R2d 跟车 12 个**：跟车时自车与前车的间距（m 与 s）对日志同位置人的间距；线：中位时距 ≥ 1.5 × 人 → 跟得远，否则跟得慢（前车本身慢）。
- 产物：`results/lon_diag/slow.md`、`decisions.parquet`（Tokyo `/data/runs/route_follow/lon_diag/`，表太大不进 git）、`figs/lon_diag_fixed_point.png`、
  `figs/lon_diag_onsets.png`（8 个起点场景的前视帧 + v 曲线；给用户看片用）。

## 读数 3：开环反事实（盒，GPU 池 1 个 job，约 1 卡时；RSG-L18-s0，frames 不动，只改 ego 行）

闸：开关全关时 plan 与 rm_train 的 dev 评估逐位相同（rf_lonread 的做法）。

- **(i) 历史增益**：自由直路行（无 A-eligible 前车 40 m 内、无红灯线 60 m 内、60 m 内无 ≥ 30° 转弯）上，速度项 ×{0.6, 0.8, 1.0, 1.2, 1.4}（vx vy ax ay 与历史位姿按同比例缩放，
  历史位姿沿各自方向径向缩放），v_p 对缩放后 v_f 的斜率 γ。线：γ ≥ 0.6 → 历史主导（与 d294 闭环 0.71–0.86 一致）；γ ≤ 0.3 → 画面主导。按速度箱分报。
- **(ii) 红灯前的刹车来源**：红灯线 10–40 m 前、人速 ≥ 6 m/s、人的历史已在减速（ax < −0.5 m/s²）的行。刹车量 b = (v_f − v_p) / v_f。
  **H** = 人自己的历史；**F** = 匀速历史（vx 不变、ax = 0、历史位姿按当前速度等距、vy = ay = 0）。
  线：中位 b_F / b_H < 0.3 → **plan 从历史读「停」，不从灯读**；0.3–0.7 → 部分；≥ 0.7 → 读灯，闭环失败在别处（时机 / serve）。
  阳性对照：A-eligible 前车 ≤ 30 m 且正在接近的行，同一 H / F；线：前车 b_F / b_H ≥ 0.7 而红灯 < 0.3 → **缺的专门是灯 / 线这一路**。
  再记一组绿灯线前同条件的行作阴性对照（H 下 b 应接近 0）。
- **(iii) 上限**：自由行按人速分箱（< 8、8–12、12–16、≥ 16 m/s）的 v_p / v_f 中位（H 历史）；并报训练 store 的行速分布 p50 / p90 / p99 / max。
  线：≥ 16 箱 < 0.85 → 分布上限成立；与 R2a 合读。
- 产物：`results/lon_diag/openloop.md` / `.json`、`figs/lon_diag_history_gain.png`、`figs/lon_diag_red_brake.png`。

## 读完怎么分桶（先写好）

- **只改训练输入、不用精心设计**：读数 3(i) γ ≥ 0.6 且 3(ii) 红灯 b_F / b_H < 0.3 → 速度历史 dropout / 噪声 + 红灯行过采样（traffic-light-probe 第 2 条）是同一个杠杆的两面；
  但 allrows 已证明单做 dropout 公开零分 +8.5（offroad / corridor），所以这一桶要带横向代价一起报，不能单独算「免费」。
- **要讨论**：停止信号的形式（灯态 / 停止线作输入 vs B 项标签 vs 后果行）、速度目标来源、慢修好后横向零分上升的交互（d310 的高速直走类）；读数 1 的 N 类与 L-STOP 类（PAI 考 stop sign，nuPlan 没有标签）。
- **评估器口径、不做**：读数 1 的 E 类。
- **给用户看片**：读数 1 的接触表全部 17 个（原因类与红灯 / stop sign 需要眼睛确认）、读数 2 的 8 个起点帧。

## 代价

Tokyo CPU（≤ 8 worker，两张卡上的 expert 读不受影响）约 2–4 h 人机时间；盒上 1 个 GPU 池 job 约 1 卡时（排在 rw-* 之后）；不起新仿真，不训练。
