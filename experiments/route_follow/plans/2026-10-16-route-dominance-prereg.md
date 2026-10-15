# ROUTE-DOMINANCE 预登记：失败转弯里是路线输入压过了道路与他车，还是 plan 根本不读后果（开环，不训练，不新跑闭环）

2026-10-16。写在任何模型输出之前。脚本 `experiments/route_follow/scripts/rf_rdom.py` + `rf_rdom.sh`。

## 问题

用户假设：RSG-L18-s0（以及更差的 APP1X-L18-s0）在转弯里撞路沿 / 杆 / 车，是因为**路线输入主导 plan**，「只顾着转」，看不见几何与他车。
对立读法：plan 根本不读后果（d220：66 个碰撞 rollout 在撞前 3 s 全部落在策略自己的 plan sweep 里；critic pilot：视觉 token 在 plan + ego + 路线之外对零分没有增量）。
反证：带路线的 RSG-L18 比不带路线的 YS2G 少 15.5 个 held 零分（d286）。两种读法的处方相反（路线权重 / 置零 对 后果标签），先读再设计。

## 器具

帧固定，只换输入行，开环，rf_lonol.py 的做法：rm_train.build 的 store（一个离线 shard s0 只用来填 store，`--yr-mass 0`），`--on` = 策略自己的 on-policy 池，
池行按训练时的方式喂（缓存的 Cinque token、ego 行、路线记忆 200 维），checkpoint 按 build 的 `--rm-from` 加载。

**状态来源（读之前已核）**：
- held：`on1_L18RSGallheld` = RSG-L18-s0 在 held 460 / 18 决策上的 rollout（`runs/onpol_long/roll/RSG-L18-s0-held`），7 837 行、447 个 scene，带 token、路线记忆
  （`runs/route_mem/feat/on1_L18RSGallheld.npz`）、仿真器 agent 框（agent hinge 标签）、道路栅格（road hinge 标签）。这次 run 的零分与 held 读数
  `runs/route_mem/read/RSG-L18-s0` 逐 scene 相同（34 / 34，同一批 scene）。
- train（复现集）：`on1_L18RSG` = RSG-L18-s0 自己在 train 1 020 个 scene 上的 18 决策 rollout（17 474 行），同样的标签。d316 的 42 个 A 层单元出自这批 rollout。
- **APP1X-L18-s0 自己的 held rollout 没有缓存 token**（`runs/route_mem/read/APP1X-L18-s0` 跑时没开 SH30_DUMP，driver-logs 里没有 dump），
  也没有 APP1X 的 on-policy 池；不新跑闭环，所以 APP1X 只读「APP1X 权重在 RSG 的同一批状态上」，它自己的 52 个新零分状态不可读，照实报告。

**零分与转弯**（rf_appzeros.py 的 extract / 定义原样，对两个 run 目录重跑 extract）：
- 零分 scene = score 0 且 failure_reason ∈ {offroad, collision_at_fault, left_corridor_laterally}（「route folds back」类 8 个是场景生成错误，不算）；
  类 = offroad / collision / corridor。零步 kz = 第一个 offroad、前向 / 侧向碰撞或 GT 距离 ≥ 4 m 的决策（rf_follow 的 k_z）。
- 路线 = AlpaSim MAP 路线逐锚点重生成（rf_follow.regen_full）；转弯 = 该折线上 ≥ 45° 的转弯（rf_appzeros.turns）；
  **转弯零分** = 零步站点前后 [s0 − 10, s_zero + 60] m 内有 ≥ 45° 转弯入口（rf_appzeros 的 route_turn / held_turn 类，二者合并）。bend / straight 的零分只描述。
- **失败状态** = 转弯零分 rollout 里 k ∈ [kz − 6, kz − 1] 且在池里的决策（撞前 0.5–3 s）。
- **对照状态** = 非零分 rollout 里**完成的转弯**（≥ 45° 转弯，入口 ≥ s0 − 10，车的最大站点过了出口 + 10 m）上，站点相对入口 ∈ [−25, +25] m 的池内决策。
  失败状态相对其最近入口的站点分布同时报告。

## 三种路线输入（帧、ego 其余各维不变）

- **served**：池里存的路线记忆（策略当时实际收到的）+ 原命令。
- **zeroed**：路线记忆 200 维全置 0（即所有 valid 位为 0；训练 `--rm-drop 0.25` 与 G-rely 的置零方式，`ego[:, 20:] = 0`），命令保留。
- **swapped**：另一个出口的路线。每个状态在 nuPlan 车道图上（corridor/lib/lanegraph.py，rf_cfpool geom 的规则）：用 served 记忆的有效点（转到地图系）
  做 viterbi 匹配得到路线的节点路径；从路线在自车投影处的节点（在路口连接段内时退到它的前驱车道）出发的 40 m 内所有下游路径里，取出口航向与路线路径差 ≥ 45°、
  离自车航向 ≤ 120° 的路径中差最大的一条（rf_cfpool 的 j0）；其中心线（自车投影起 110 m）经 rmem.window 变成 200 维特征，有效位与 served 取交集
  （覆盖范围相同，只换几何）；命令按 navigation.command_from_route 的规则用该中心线在自车前 42.1 m 处的横向符号（±2 m）重算。找不到另一出口的状态不做 swap，
  报告数目。
- 描述（不进判定线）：**zeroS** = zeroed + 命令改 S。

## 闸（任何读数之前）

- G-rep：served 输入下本器具的 plan 对池里存的 served plan（compiled serving）8 个位姿距离中位 ≤ 0.05 m、p95 ≤ 0.3 m（读的状态上）。
- G-det：同一输入两次前向逐位相同。
- G-feat：用本器具的特征构造（served 记忆有效点 → rmem.window）重建 served 特征，有效位逐位相同、坐标最大差 ≤ 0.05（特征单位，即 1 m）在 ≥ 95 % 状态上；
  不过则 swapped 臂作废，只报 served / zeroed。
- G-swap-sanity：swapped 臂在对照状态上 4 s 终点向另一出口一侧的移动中位 > 0（策略在这些状态上读路线；否则 swap 读数只作描述）。

## 量

每个状态、每个输入（每个 checkpoint）：
- 相对 served plan 的横向位移（在 served 位姿的航向系里）：1.0 / 1.5 / 2.0 s 的平均 |Δlat|、4 s |Δlat|；2 s 与 4 s 航向差（deg）。
- 接触（hinge 同一几何，0.1–4.0 s 的 40 个稠密步，t = 0 不计）：
  - 道路：Pacifica 四角在道路栅格（road.npz：ROADBLOCK | ROADBLOCK_CONNECTOR | INTERSECTION | LANE | LANE_CONNECTOR，road hinge 的标签）上 sdf < 0；
  - 他车：AgentHinge 距离 < 0（margin 0；hinge 自己的计数规则：物体在自车框中心前方、|横向| < 3 m 或已重叠）；
  - 接触 = 任一。t = 0 已接触（道路或他车）的状态不进比例分母，单独报告数目。
- 指标：「zeroed 下无接触」「swapped 下有接触」。0.1–2.0 s 的同样指标只作描述。

## 登记的判定线

主读数 = RSG-L18-s0 在自己状态上，held + train 合并（两者同一策略、同一规则；log 聚类），分母 = served plan 有接触（且 t = 0 未接触）的失败状态，
F = 其中 zeroed 下无接触的比例；log 聚类百分位 bootstrap（B = 10 000，seed 0，比例 = 和之比）：
- **ROUTE-OVERRIDE**：F ≥ 50 % 且 CI 下界 > 30 %。
- **BLIND**：F ≤ 20 % 且 CI 上界 < 35 %。
- 否则 **UNDECIDED**，按失败类（offroad / collision / corridor）与 checkpoint（RSG / APP1X 权重）拆开报。
- held-only 与 train-only 各自按同样的线读；若两者落在相反的两条线（一个 ROUTE-OVERRIDE、一个 BLIND），总判定降为 UNDECIDED。

次要（登记，不定判词）：
- S1：在有 swap、served plan 无接触的状态上，swapped plan 进入接触的比例（路线把 plan 带进障碍物），失败对对照。
- S2：zeroed 的位移分布（m）：失败、对照与全部池行（按 rf_lonol 的 turn60 ≥ 30° / 直路分桶），即路线在这些状态上到底把 plan 挪了多少。
- S3（基线）：对照状态里 served 有接触者的 zeroed 解除率，以及 served 无接触者被 zeroed **引入**接触的比例（失败、对照各一）。解除率在失败与对照上相同时，
  解除不能归因为「路线把 plan 推进接触」。
- APP1X-L18-s0 权重在同一批 RSG 状态上的同样读数（ROUTE-OVERRIDE / BLIND 线照算，标为「RSG 状态上」）。

## 读法（写在读之前）

- ROUTE-OVERRIDE → 下一轮处方在路线侧：on-policy 行也做路线置零（现在 `rm_drop_pool false`：池行从不置零）、或降路线支路 lr。
- BLIND → 处方在后果侧：去掉路线不救这些 plan，路线不是推手；需要后果标签（agent / road hinge 打在策略自己失败的状态上、critic）。
- UNDECIDED → 按类报，处方不定。

## 成本

一个 GPU 池作业（`--vram 12`，同 rf_lonread），两个池各建一次 store，两个 checkpoint × 4 个输入 ≈ 25 k 行前向，预计 < 0.5 卡时；CPU：rf_appzeros extract
（1 480 个 rollout，16 worker）、车道图 swap（navsim2，选出的状态），接触在盒上 CPU 算。超过 1 卡时先报告。
