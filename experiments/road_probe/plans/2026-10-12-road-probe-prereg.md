# ROAD-PROBE 预登记：nuPlan 城市道路几何（尤其路口与弯道）在冻结 openpilot（Cinque）视觉 token 里有没有

2026-10-12 写，写定并 push 于任何探针读数之前。（编号注：EYES-VS-HEAD 在 decisions.md 里后来改号为第 272 条；本稿中的「第 269 条」都指它。）来源：第 269 条说 openpilot 随附的车道线 / 路沿头在我们失败的地方（转弯、路口、出路前 2 s）
不报车道（直行两条 ego 线都 > 0.5 只有 0.240，转弯 0.003，零分前 2 s 0.005），plan 和头错在同一处。问题是：这些路的几何是**不在 token 里**，
还是**在 token 里但随附头解不出来**。这是能力边界探针（stage 1，离线），不是训练线：协调会话据此决定是否值得做一个学出来的 road head。
本稿之前只看过数据格式与数量（训练行的 tab / meta 键、行数、log 单元数、排除原因计数；held 读数只在零分 rollout 留了 asl；dump 的键），
没有算任何标签、没有训练任何头、没有读任何登记量。

## 1. 数据（只读，box）

- **训练 token（已存，不重算）**：onpol_long 三组 on-policy 行的冻结视觉 token，`$DATA_DIR/runs/op_parity/cache/on1_{L18Y,L18YS,L18P}@warp/front.npy`
  （(n, 8, 32, 512) fp16，driver dump 的 8 个 slot 帧经 `pp_prep.encoder`，on_rows.py tok；该处 gate R1 把它与 served plan 对齐到 0.03 m），
  行 = YR10m10-F-s0、YS-L18-s0/s1、P2H10S-F-s0 在 1020 个 train scene（`alpasim_nuplan/onpol-train`，28 个 log 单元 = `c0b_report.log_of`）上的
  18 决策 rollout，合计 69 419 行（排除 overlap / corridor / log_future / frame 约 6%，按 on_rows 的规则，不改）。ego 向量（20 维，含 command）取 tab。
- **held 帧（新跑）**：`YS-L18-s1` 在 `alpasim_nuplan/onpol-held`（460 scene，18 决策，alpasim_long 的 driver 路径，与 onpol_long read1 同一套参数）
  上重跑一遍并打开 `SH30_DUMP`，3 个 pool job；token 用同一个 `pp_prep.encoder`、同一种 slot 配对（on_rows make_front 的写法）。这样 held 是无选择偏差的
  全部决策，且与训练 token 同一条管线。
- **R3 零分窗口（已存）**：第 269 条登记的 98 个 corridor + offroad 零分（`YR10m10-F-s0` 64 + `YS-L18-s1` 34；初稿把两边数目写反，读数前改正，总数 98 不变），事件与 k_ev 取 zeros_mech
  `per_rollout.parquet`，窗口 W = 决策 max(0, k_ev − 4) … k_ev；token 用存的 driver 消息（`$DATA_DIR/runs/eyes_vs_head/msgs/<label>/`）按 live driver
  设置重放取 served token（ol_scorer.py replay 的写法：compiled、gpu 合成）。
- **折**：train / held 已按 log 单元分开（on_split.py）；训练内部 sha256("rpval|" + log) % 7 == 0 的 log 单元作 val，只用于选 step 与 ridge λ。
  held 与 R3 的 scene 不进训练、不参与任何选择。

## 2. 标签（特权，只作标签与评分）

run 自己的 scene loader（after160 重建 cache 的 trajdata 矢量地图，第 269 条 ev.py 的 `Geo` / `cross` / `log_y` 原样 import），后轴系 x 前、y 左正，
与 ev.py、第 269 条同一坐标与符号。对每个决策（anchor = drive.jsonl / meta 的 anchor，cam = 相机在后轴系的位置）：
- d ∈ {10, 20, 40} m：横线 x = cam_x + d；y_log(d) = 日志后轴路径（终点外延 60 m 直线）与它的第一个交点（= 路线在 d 处的位置）。
- **ego 车道边界** L(d) / R(d)：所有车道 left / right edge（含 connector）与横线的交点集 B(d)（0.3 m 内合并、|y| ≤ 15 m），取 y_log(d) 左侧最近与右侧
  最近的交点；两侧都有且 2.0 ≤ L − R ≤ 6.0 m 才可评。即「路线在 d 处所在车道」的两条边界，不是车现在所在车道。
- **路沿** EL(d) / ER(d)：ROAD_AREA 并集边界的交点集 R(d) 中 y_log(d) 左侧 / 右侧最近的交点（nuPlan 没有单独的 road edge，同第 269 条）。
- **车道中心朝向** h20：包含点 (cam_x + 20, y_log(20)) 的车道中，中心线切向与日志路径在该点切向最接近的那条，在投影点的切向减 ego 航向（wrap）。
- **路线转角**：日志路径从后轴投影点 s0 到 s0 + 40 m 的航向变化绝对值；**直行** < 5°，**转弯** ≥ 20°，其间为「缓弯」只作描述。
- **到路口距离**：沿日志路径从 s0 起每 2 m 取点（0–60 m），第一个处在路口（ev.py `Geo.junction`：两条中心线相差 > 30° 的车道多边形同时包含）的弧长；无则 "> 60"。
- **对向车道**（探索）：zeros_mech `Lanes.cls` 对后轴的分类 == onc（所有包含车道都与 ego 航向差 > 120°、没有 < 60° 的）。

## 3. 探针

全部只读冻结 token（8 slot × 32 × 512）与 20 维 ego（含 command），不读地图、日志、路线几何。输出 14 个量：L / R / EL / ER × {10, 20, 40}（12 个横向 y，m）、
h20（rad）、对向 logit。
- **P（主）**：第 243 条 HEAD1 的头（`experiments/corridor/scripts/head1_train.py` 的 `build()`，3 层 encoder + 2 层 query decoder，d 256，3.4 M 参数）
  原样 import，14 个 query × 1 通道。回归目标按 fit 集均值 / 标准差标准化，Huber δ = 1（标准化单位），对向 BCE，缺失标签屏蔽；8 000 step、batch 256、
  lr 3e-4、wd 0.05、warmup 300、每 500 step 在 val 上取最好的；均匀抽样（不加权）。两个 seed，**登记读数 = 两个 seed 预测的平均**，单个 seed 另报。
- **线性底**：newest slot（slot 7）的 32 × 512 token 展平 + ego → ridge，λ ∈ {1e1, 1e2, 1e3, 1e4, 1e5} 在 val 上选；每个输出一样。
- **blind 底**：P 的同一个头不读 token（只有 ego + command），一个 seed：告诉我们路的几何有多少是 ego / command 本身就能猜的。
- **学习曲线**：P（seed 0）用 fit log 单元的 1/8、1/4、1/2（按 sha256("rplc|" + log) 排序取前段，嵌套），加全量，共 4 点。
- **随附头参照行**：P0（`load_pmodel("P0")`，不加 adapter）在同一组 token 上（held：P0(front, ego, tc = (1, 0))；R3：第 269 条 eyes_vs_head 重放存下的 P0 头，
  同一批决策）的线 1 / 线 2 当 L / R、路沿 0 / 1 当 EL / ER，**不看概率**（概率另报存在率），并报撤掉 0.689 压缩（绕相机 y 除以 0.689，第 104 条）的版本。

## 4. 读数与判词线

误差 = |预测 − 标签|（m），只在标签可评处算。中位数与 p90 的 CI：按 scene 整簇的 bootstrap（B = 2000，百分位；jevdrive.stats 只做均值，偏离写明）。
- **R1**（held 直行决策）：20 m 处 L、R 各自及合并的误差中位数与 p90；10 / 40 m、EL / ER、h20 同表描述。
- **R2**（held 转弯决策，转角 ≥ 20°）：同 R1。
- **R3**（98 个零分窗口）：每个窗口内，每一侧取 20 m 误差在窗口决策上的中位数；窗口误差 = 两侧的平均；**R3 中位 = 98 个窗口误差的中位数**；
  **R3 份额 = 两侧窗口中位都 ≤ 0.5 m 的窗口比例**。只用 P（两 seed 平均）判。
- **判词（只在 R3）**：R3 中位 < 0.3 m 且份额 ≥ 0.6 →「**眼睛可修**」（路的几何在冻结 token 里，学一个 road head 值得做）；R3 中位 > 1.0 m →
  「**冻结的眼睛看不见城市道路**」（要么解冻 / 换编码器，要么地图）；其间 →「**未定**」，附学习曲线：用 R3 误差（与 held 转弯误差）对训练 log 单元数做
  log 线性拟合，斜率 < 0（CI 不含 0）写成「加数据还在降」，否则「加数据不再降」。
- 副线（描述，与第 269 条直接可比）：R1 中位 < 0.30 m 且 p90 < 0.75 m（第 269 条「眼睛可用」的 (ii)(iii)）是否达到；并报「最近边界误差」
  （第 269 条的口径：预测 y 到 B(20) 最近交点）对 P 与随附头。
- 还报：学习曲线（4 点 × R1 / R2 / R3）；转弯决策误差按到路口距离（在路口内、0–10、10–20、20–40、40–60、> 60 m）；P − 随附头、P − blind、P − 线性的差。
- **探索（便宜才做）**：对向分类的 AUC（held 全部决策；R3 窗口上另报），正例数。

## 5. 闸与停止规则

- G-roll：held 重跑 ≥ 450 个 scene 有 rollout，且平均分与存的 `YS-L18-s1` held 读数（0.83 左右）相差 < 0.03（重跑是同一个 driver）；
- G-tok：held dump 上 encoder token + `YS-L18-s1` checkpoint 复算的 plan 与 dump 存的 served plan 在 8 个位姿上差 ≤ 0.10 m 的决策 ≥ 95%；
- G-rep：R3 重放的 served plan 与存的 drive.jsonl 在 4 s 位姿差 ≤ 0.10 m 的决策 ≥ 95%（同第 269 条）；
- G-frame：每个 scene 第 0 个决策的 anchor 与日志后轴 ≤ 0.10 m 的 ≥ 99%（地图与行同一坐标系；训练、held、R3 分别）；
- G-lab：held 直行决策上 20 m 的 L、R 都可评的 ≥ 70%（否则标签定义在城市里不成立，停、查）；
- G-sign：held 直行上 L(20) 的中位 > 0 > R(20) 的中位。
任何闸不过：停、写原因，不读登记量。阈值、窗口、判词线读后不改；读后想到的变体只能写成探索。

## 6. 预算与产出

估：held 重跑 3 个 job 共约 1 卡时（渲染占大头）；token 约 10 分钟；R3 重放约 5 分钟；训练 P × 2 seed、学习曲线 3 点、blind 共 6 个 run，每个按 HEAD1 的速度
约 10–15 分钟（先 smoke 实测）；标签 CPU 约半小时。合计约 2.5 卡时、一天内。分级启动：smoke（两三个 log 单元）→ 全量。
产出 `experiments/road_probe/`：README、INDEX 一行、`scripts/rp.py`（labels / tok / replay / train / read）、`results/rp_report.md`（闸、登记表、线与判词、
学习曲线）、2–3 张图（误差按距离与场景类型；4–6 个 BEV 例子：地图、探针输出、随附头、plan，含零分窗口）；research/decisions.md 一条。大文件留 box
`$DATA_DIR/runs/road_probe/`。

## 7. 读数前的补充（2026-10-12，写代码时，任何标签统计、训练与读数之前）

- 事件表用第 269 条的逐事件表 `experiments/eyes_vs_head/results/events_P0.csv`（group、scene、type、k_ev，与 zeros_mech `per_rollout.parquet` 同源），
  取 group ∈ {YR10m10-F-s0, YS-L18-s1}、type ∈ {corridor, offroad}，正好 98 个。
- R3 的随附头：不读 eyes_vs_head 存的重放文件，而在本线自己的重放里对同一组 served token 用同一个调用（`load_pmodel("P0")(H, ego, tc)`，ev.py 的写法）
  再算一次，与 token 一一对应；数值与第 269 条相同（同一函数、同一输入）。
- R3 的分母：窗口误差在至少一侧可评的窗口上取中位；「两侧都 ≤ 0.5 m」的份额以两侧都可评的窗口为分母；两种窗口数都报。
