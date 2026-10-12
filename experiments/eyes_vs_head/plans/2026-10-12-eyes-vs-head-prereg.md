# EYES-VS-HEAD 预登记：零分时 openpilot 自己的感知头（车道线、路沿、lead）看对了没有

2026-10-12 写，写定并 push 于任何读数之前。来源：协调会话要决定是否值得做一个零样本的「openpilot 眼睛 + 规则规划器」driver，需要知道：在已存的
18 决策 held rollout 上，driver 得零分（corridor / offroad / collision）时，冻结 openpilot 模型自己的感知头是否把场景读对了而 plan 仍然离路或撞车
（「眼对、脑错」），还是感知头在重建图像上本身就错了（「眼错」）。本稿之前只看过：数据格式（driver 日志字段、rollout.asl 只留给零分 / 失败 rollout、
地图元素类型：ROAD_LANE 有 left_edge / right_edge，nuPlan 地图里没有 ROAD_EDGE 元素、只有 ROAD_AREA 多边形）、swv1 重放文件的键名。没有算任何登记量。

## 0. 已知事实与偏离

- **存储里没有感知头。** driver 日志（`driver-logs/drive.jsonl`）只存 served plan、anchor、command；完整输出向量不存。rollout.asl 只保留了零分 / 失败
  rollout（run 结束时删掉 score > 0 的 asl，ol_chain.sh `read1`），里面有 driver 收到的全部消息（CAM_F0 图像、ego 轨迹、路线、drive 请求）。
  所以感知头靠离线重放得到：col1 / swv1 的办法（`experiments/alpasim/scripts/swv1_replay.py`）：把记录的 driver 侧消息按到达顺序喂给真正的 driver 类
  （`sh30_driver.Driver`，served checkpoint），在同一组视觉 token 上同时取 served checkpoint 的输出（**FT**）和随附 policy 权重、不加 adapter 的输出
  （**P0** = `load_pmodel("P0")`）。一张卡、一个 pool job。
- **「冻结 openpilot 的眼睛」= P0。** policy 网络（输出 plan、lane_lines、road_edges、lead 等全部头）在我们的 checkpoint 里是训练过的，adapter 偏置加在 token 上；
  所以 served checkpoint 的车道线头不一定是 openpilot 的。登记量全部用 **P0**；FT 只作描述（第二列）。
- **R1 的样本偏离任务书。** 任务书要 YR10m10-F-s0 与 YS-L18-s0 的「全部决策」，但图像只留在零分 rollout 里。所以：
  - **R1-ctrl（主，无选择偏差）**：col1 的对照组（`$DATA_DIR/runs/alpasim/col1/lead/msgs/ctrl`，P2H10-F-s0 在公开 part001 / part002 的 200 个 scene 上
    重跑并保留了全部 asl，10 决策），所有决策；
  - **R1-held（次，偏向失败 scene）**：YR10m10-F-s0、YS-L18-s0（两个 seed 都做：YR10m10-F-s1、YS-L18-s1）零分 rollout 的全部决策，按「事件窗前」
    （k < k_ev − 4）与「事件窗内」分开报。
  「眼睛可用」线只在 R1-ctrl 的直行决策上判。

## 1. 数据（只读，box）

- **label**：`YR10m10-F-s0`（基底）、`YR10m10-F-s1`、`YS-L18-s0`、`YS-L18-s1`（43 个零分即 zeros_mech 的事件表）、`YS2G-L18-s0/s1`；每个 label 的零分
  rollout（asl 在）。YNONE 不做（不是候选）。
- **重放**：每个 label 用它自己的 checkpoint（`SH30_TAG`），driver 设置同 held read（cold backwarp、motion 1、synth gpu、不 compile：约 1 cm 的差）；
  ctrl 用 P2H10-F-s0。每决策存 served plan、P0 plan、anchor、相机位置、command，以及 FT / P0 的 lane_lines、lane_lines_prob、road_edges、lead、lead_prob。
- **地图与日志（特权，只作标签）**：run 自己的 scene loader（held 的 after160 配置；ctrl scene 也在这个重建 cache 里）：车道 left_edge / right_edge（车道线真值，
  含 lane connector 的虚拟边界）、ROAD_AREA 多边形并集的边界（路沿真值；nuPlan 地图没有单独的 road edge）、日志后轴路径、日志交通（zeros_mech G-act：模拟器
  逐点回放日志交通）。事件（类型、k_ev、at-fault 接触 k_c、被撞物 id）取 zeros_mech 的 `per_rollout.parquet`（box `$DATA_DIR/runs/zeros_mech/collect/20261012-155215`）。

## 2. 定义

- **头的解码**（`jevdrive/openpilot/model.py` `decode`）：lane_lines = mdn_mu → (4, 33, 2)，取 y；prob = sigmoid(lane_lines_prob)[1::2]；
  road_edges = mdn_mu → (2, 33, 2)；lead = mdn_mu → (3, 6, 4)，取 [0, 0] 的 x、y、v；lead_prob = sigmoid(lead_prob)[0]。点在 X_IDXS = 192 (i / 32)² m（相机前方），
  y 右正。转到后轴系：(cam_x + X, cam_y − y)（左正）。线 0、1、2、3 = 左左、左、右、右右；ego 车道 = 线 1、2。
- **G-sign（符号闸）**：R1-ctrl 直行决策上线 1 的 y 中位数 < 0 < 线 2 的中位数，且按上面的符号算的 10 m 误差中位数小于把 y 取反后的。不过就停、查约定，不调别的。
- **地图横截面**：在后轴系 x = cam_x + d（d = 10、20、40 m）处的横线（|y| ≤ 15 m）与所有车道边界折线的交点集 B(d)（0.3 m 内的交点合并：相邻车道共用边界）、
  与路面边界的交点集 R(d)。
- **横向误差**：存在的线（prob > 0.5）在 d 处的 y 与 B(d) 中最近交点的距离；路沿头与 R(d) 中最近交点的距离；交点集为空则该点不可评。
  这是「最近边界」误差，最大约半个车道宽（约 1.75 m），偏错一整条车道会被记成小误差；所以另外记「ego 车道被正确框住」（见下）。
- **直行 / 转弯**：按该决策的路线 command（STRAIGHT 对 LEFT / RIGHT）；另报 junction（后轴同时在两条方向差 > 30° 的车道多边形里）与城市（log 名的地点）。
- **ego 车道框住**（d = 10 m）：线 1、2 都存在，各自与 B(10) 中的某个交点相距 ≤ 0.5 m，且这两个交点在 B(10) 中相邻（中间没有别的边界）。

## 3. 读数与判词线

**R1 眼睛总体质量。** 对 R1-ctrl 与 R1-held 的每个决策：线 0–3 的存在率（prob > 0.5），存在的线在 10 / 20 / 40 m 的误差中位数与 p90，路沿头同；按直行 / 转弯、
junction、城市分；P0 为主，FT 并列。
**线「眼睛可用」（R1-ctrl，直行决策，P0，全部满足）**：(i) 线 1、2 同时存在 ≥ 80% 的决策；(ii) 存在的线 1、2 在 20 m 的误差中位数 < 0.30 m；
(iii) p90 < 0.75 m。依据：OpenLane 上的单目 3D 车道检测在 0–40 m 的横向（x）误差约 0.3 m（PersFormer 2022、Anchor3DLane 2023 都在 0.3 m 左右）；
1.9 m 宽的车在 3.5 m 车道里每侧约 0.8 m 余量，p90 0.75 m 是规则 lateral 还能不出车道的上限（估）。我们自己没有 navtest 上的车道线读数；
vm3（CARLA）用过 1.2 的横向比例修正，所以若中位数达标而误差带系统偏置（同号），另报带符号中位数。转弯决策用同一套量只作描述。

**R2 零分。** 每个零分事件（6 个 label；登记的判词用 `YS-L18-s1` 的 43 个 + 基底 `YR10m10-F-s0` 的零分，其余 label 描述），窗口 W = 决策 max(0, k_ev − 4) … k_ev
（事件前 2 s，第 266 条：served plan 在此时已偏离路线），对每个决策 j：
- eyes_right_j：ego 车道框住，且线 1、2 在 10 m 与 20 m 的误差都 ≤ 0.5 m；
- eyes_wrong_j：线 1 或 2 不存在（prob < 0.5），或在 10 m 或 20 m 误差 > 1.0 m；
- crosses_j：served plan（FT，真正执行的）在相机前 0–40 m 内的某个位姿，其后轴横向越过预测的线 1 或线 2（在同一 x 上插值），或越过预测的路沿；记越过的一侧。

事件分类（P0 头）：
- **(a) 眼对、脑错**：存在 j ∈ W，eyes_right_j 且 crosses_j，且越过的一侧与事件时车相对日志的偏离侧相同（zeros_mech 的 e(k_ev) 符号；碰撞不要求同侧）；
- **(b) 眼错**：非 (a)，且 W 中至少一半可评的决策 eyes_wrong_j；
- **(c) 模糊**：其余；记子因：c1 眼对但 plan 留在看到的车道内、c2 误差在 0.5–1 m 之间或交点不可评、c3 越过的一侧与偏离侧相反。

按零分类型（corridor / offroad / collision）× label 报计数，附逐事件表。
**线**：`YS-L18-s1` 与基底 `YR10m10-F-s0` 合并的 corridor + offroad 零分中 (a) ≥ 60% → 「在这些头上做规则 lateral 有支持」；(b) ≥ 40% → 「眼睛在这些
scene 上不够」；都不满足 → 不支持，写明。两个 driver 分别也报。FT 头的同一分类只作描述。

**R3 碰撞时的 lead 头。** 6 个 label 的每个 at-fault 碰撞（k_c ≥ 0），窗口 k_c − 4 … k_c：lead 头报到被撞物 = 存在 j，lead_prob > 0.5，lead 位置与被撞物框上离相机
最近的点相距 ≤ 2 m，且 lead v 与被撞物速度在 ego 航向上的分量相差 ≤ 2 m/s。分三类：报到、报了别的（prob > 0.5 但不匹配）、没报（prob ≤ 0.5）。
被撞物在 k_c − 2 时相对 ego 的航向差 > 120° 为对向（zeros_mech H3），60–120° 横穿，其余同向；lead 头只看前方同向，所以对向与横穿单独报，不算进主比例。
无判词线（描述读数），但写清同向碰撞中「报到」的比例。

**R4（探索，便宜才做）。** R1 的决策上，日志后轴路径在 10 / 20 m 处相对预测 ego 车道（线 1、2 存在时）的位置：在车道内的比例、离预测车道中心 > 0.5 m 的比例。

## 4. 闸与停止规则

- G-msg：选中的零分 rollout ≥ 99% 有 asl、能抽出消息；
- G-rep：重放的 served plan 与 drive.jsonl 存的 plan 在 4 s 位姿差 ≤ 0.10 m 的决策 ≥ 95%（不过就停：重放没有复现 driver）；
- G-frame：每个 rollout 第 0 个决策的 anchor 与日志后轴位姿 ≤ 0.10 m 的 ≥ 99%（地图与 rollout 同一坐标系）；
- G-sign：见第 2 节。
任何闸不过：停、写原因，不读登记量。阈值、窗口、类别规则读后不改；读后想到的变体只能写成探索。

## 5. 产出

`experiments/eyes_vs_head/`：README、`scripts/ev.py`（msgs / replay / geo / read）、`results/ev_report.md`（登记表、线与判词、逐事件表）、2–3 张图
（误差直方图；4–6 个 BEV 例子：地图、P0 车道线头、plan、日志，每类一个）；research/decisions.md 一条。大文件留在 box `$DATA_DIR/runs/eyes_vs_head/`。
预算：CPU 半天以内，重放一张卡（pool job）。
