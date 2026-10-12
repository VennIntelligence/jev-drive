# LOCAL-MAP 预登记：把 2–4 s 前看到或收到的道路几何带到零分窗口里，能不能补上冻结眼睛在路口里的缺口

2026-10-12 写，写定并 push 于任何读数之前。来源：第 275 条，冻结 token 上的道路探针在路口还在 20–60 m 外时把 20 m 处 ego 车道边界读到
0.42–0.47 m，路口内与路口前 20 m 内 1.1–1.7 m，98 个出路零分窗口中位 1.26 m。问题：**一个规则的局部地图（记忆）能不能救它**——把更早决策的探针
车道边界估计、和更早收到的 route 路点（那时路口还在 40–80 m 外，路点已覆盖路口）用 driver 收到的 egomotion 刚体变换到零分窗口决策的坐标系，
和窗口里的地图比。无训练、无 GPU。本稿之前只看过数据格式（msgs pickle 的消息种类、route 消息 20 个路点约一半 NaN、首点约 40 m 外；tab / labels /
pred 的键与行数；asl 里 EGO actor 位姿 = rig 位姿沿航向 +1.461 m，即 `rig_to_bounding_box`），没有算任何登记量。

## 1. 数据（只读，box `$DATA_DIR/runs/`）

- **零分组**：第 272 / 275 条的 98 个 corridor + offroad 零分（`YR10m10-F-s0` 64 + `YS-L18-s1` 34，事件表 `experiments/eyes_vs_head/results/events_P0.csv`），
  每个 rollout 的全部 18 个决策（`road_probe/tok/zero/tab.npz`：label、scene、k、t0、anchor、cam、served plan `poses`，1 764 行）。
  窗口 W = 决策 max(0, k_ev − 4) … k_ev（按 tab 的 k，与第 275 条 `r3_windows` 完全相同）。
- **held 组**：第 275 条 `YS-L18-s1` held 重跑的 8 129 个决策（`road_probe/tok/held/tab.npz`）。
- **探针输出**：第 275 条登记的 P = `road_probe/pred/P-p1000-s0.npz` 与 `-s1.npz` 的平均（行与 tab 一一对应），每个决策 L / R × 10 / 20 / 40 m 的横向 y。
- **地图标签**：第 275 条 `road_probe/labels/{zero,held}.parquet` 原样（ev.py / rp.py 的 `Geo` / `cross` / `log_y`：横线 x = cam_x + D 上路线所在车道的左右边界 L(D) / R(D)，
  两侧都有且宽 2–6 m 才有定义）；车道中心 c(D) = (L + R) / 2；`turn40`、`jdist` 同第 275 条。
- **driver 收到的 route 与 egomotion**：零分组用存的 driver 消息 `eyes_vs_head/msgs/<label>/<scene>.pkl`（`route_request`：rig 系 20 个路点、`timestamp_us`；
  `driver_ego_trajectory`：local 系 rig 位姿）。held 组的 asl 只为零分 / 失败 rollout 保留，没有 route 消息：用 AlpaSim 自己的
  `RouteGenerator`（MAP、`route_start_offset_m = 40`，`ao_e1.py` 的写法）在每个决策的 anchor 上重算，闸 G-route 检查它与 driver 记录的 `route0`
  以及零分组消息逐点一致。

## 2. 坐标与搬运

local 系 = 世界平移；rig = 后轴（G-frame 核），x 前、**y 左正**（ev.py / rp.py 的约定；图里照第 275 条画成右正）。决策的时间序号 kt = (t0 − 17 000) / 500 000。
决策 k 往回 d 个决策（d ∈ {2, 4, 6, 8}，即 1–4 s；d = 0 是当前决策）的估计用刚体变换带到 k 的坐标系：p_k = in_ego(a_k, to_local(a_{k−d}, p))，a = 该决策的
anchor（driver 由 egomotion 得到的 local 系 rig 位姿）；route 路点用 route 时间戳处 egomotion 的 rig 位姿。k − d 不存在（kt < d 或该决策被跳过）则无估计。

## 3. 估计（规则，读前定死）

- **L1 带过来的探针**：每一侧把 k − d 的探针三个点 (cam_x + 10, y10)、(cam_x + 20, y20)、(cam_x + 40, y40)（k − d 的坐标系）带到 k，连成折线；
  在 k 的横线 x = cam_x + D 上取折线**按点序第一个交点**（线性插值，**不外推**）；不覆盖则无估计。d = 0 即第 275 条探针本身。
- **L2 带过来的 route**：k − d 收到的路点（去掉 NaN）带到 k，连成折线，同样取第一个交点 ŷc(D)，误差 |ŷc − c(D)|。d = 0 = 当前收到的路点（底线：预期够不到
  10 / 20 m）。另报 **L2-new**（固定规则，非 oracle）：在 kt 及之前收到的全部 route 中，取覆盖 D 的**最新**一条。副量：点 (cam_x + D, c) 到带过来路点折线的最近距离。
- **L3 融合**：车道位置用 L2 的中心（同一个 d，或 L2-new），宽度用 L1 在同一 d、同一 D 的 L̂ − R̂；L1 在该 d 不覆盖时用当前（d = 0）探针的宽度。
  L̂ = ŷc + w / 2，R̂ = ŷc − w / 2。

## 4. 读数

- **Z（零分窗口，登记）**：对每个（线、d、D ∈ {10, 20, 40}）：L1 / L3 按第 275 条 R3 的口径——每个窗口每一侧取窗口决策上误差的中位，窗口误差 = 两侧平均；
  L2 的窗口误差 = 中心误差在窗口决策上的中位。报窗口中位、p90、窗口 bootstrap 的 95% CI（B 2000，`rp.r3_summary` 的做法）和**覆盖窗口数**
  （估计与标签都有定义的窗口；第 275 条两侧标签都有定义的是 45 个）。另报 **oracle**（每个窗口在 d ∈ {2, 4, 6, 8} 里取误差最小的 d），明确标为 oracle，不进判词。
- **H（held 转弯，路口前 0–20 m，登记）**：`turn40 ≥ 20°` 且 0 < `jdist` ≤ 20 m 的决策（第 275 条 junction 表的 0–10、10–20 m 两档），每个决策两侧合并
  （L2 为中心误差），中位、p90、按 scene 整簇 bootstrap（`rp.boot_q`，B 2000）。描述性另报路口内（`jdist` = 0）与 20–60 m 两档、held 直行。
- **判词（Z、D = 20 m）**：任一条线在某个**固定** d ∈ {2, 4, 6, 8}（或 L2-new）上窗口中位 < 0.5 m，且覆盖窗口 ≥ 23（45 的一半）→「**局部地图可行**」，写明是哪条线、哪个 d；
  三条线所有固定 d 的窗口中位都 > 1.0 m →「**从没看清过**」；其间 →「**未定**」，附 d 的曲线。共 3 线 × 5 规则，有多重比较，表里全部报出。
  覆盖不足 23 的（线、d）只描述，不判。
- **plan 检查（Z）**：每个窗口决策，存的 served plan（后轴折线，从原点起）在 D = 10、20 m 的横向位置若落在带过来的车道 [R̂, L̂] 之外，记为「车道保持规则与 plan
  不一致」；对 L1 与 L3、每个 d，数窗口：可评窗口、带过来的地图说 plan 出车道的窗口、真实地图标签说出车道的窗口、两者都说的、只有带过来的地图说的（误报），
  以及第一次不一致的决策相对 k_ev 的提前量。

## 5. 闸与停止规则（任何一个不过：停、写原因，不读登记量）

- **G-frame**：98 个零分 rollout 的每个决策，消息里 egomotion 链在 t0 的 rig 位姿与（i）tab 的 anchor 差 < 1 cm，（ii）asl 里模拟器 EGO actor 位姿减去
  `rig_to_bounding_box`（沿航向 1.461 m）后差 < 5 cm；两项各 ≥ 99% 的决策。
- **G-carry**：零分组每个有 k − 1 的决策，把 k − 1 收到的路点带到 k，到 k 收到的路点折线的距离（只取两者弧长重叠的路点）中位 < 0.05 m、p95 < 0.20 m。
- **G-sign**：held 直行（`turn40` < 5°）上 L1 d = 2、D = 20 m 的 L̂ 中位 > 0 > R̂ 中位；G-carry 用镜像变换（航向取反）的距离中位 > 正确变换的 10 倍。
- **G-d0**：L1 d = 0 在 Z 上复现第 275 条 R3（窗口中位 1.255 m，差 < 0.005，可评窗口 45），在 H 的 0–10 / 10–20 m 两档复现 junction 表（1.149 / 1.683 m，差 < 0.005）。
- **G-route**：held 重算路点首点与 driver 记录的 `route0` 差 ≤ 0.05 m 的决策 ≥ 99%；零分组重算路点与消息逐点差 ≤ 0.05 m 的 ≥ 99%（不过只停 held 上的 L2 / L3）。

阈值、窗口、规则读后不改；读后想到的变体只能写成探索。

## 6. 预算与产出

CPU（AlpaSim venv，`n_cpus()` 个 worker、每个一线程），读消息、重算路点、带估计约 20 分钟。产出 `experiments/local_map/`：README、INDEX 一行、
`scripts/lm.py`（box：read）与 `scripts/lm_report.py`（Mac：表与图）、`results/lm_report.md`、两张图（Z 与 H 上误差随 d 的曲线；4 个 BEV 例子：地图、带过来的探针、
带过来的 route、served plan、日志路径）；research/decisions.md 一条。大文件留 box `$DATA_DIR/runs/local_map/`。
