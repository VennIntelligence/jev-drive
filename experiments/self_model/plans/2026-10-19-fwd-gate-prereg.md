# SELF-FWD 离线闸：自我前向模型（预登记，2026-10-19，写于任何输出之前）

来源：[practice-plan.md](../../../research/next-round/practice-plan.md) 第 4 项；论点 [knowing-vs-doing.md](../../../research/knowing-vs-doing.md)；
旧稿 [self-model.md](../../../research/next-round/self-model.md)（车身课 V1 = 对自己 plan 的扫掠判接触，这里换成「判在会执行出来的轨迹上」）。
相关：d308（蠕变：执行每 2 s 比 plan 少转 2.5°，三分之二来自从车位姿重规划）、d322（价值评论家没学会，AUC 0.846）。

## 问题

在 AlpaSim nuPlan 默认 serve 的 rollout 上，一个小网络能否从「plan 8 点 + 决策时自车状态」预测出接下来 4 s 实际执行出来的轨迹（含之后 7 次重规划与控制器）？
预测的执行扫掠判出路 / 接触，是否比直接在 plan 上判更准？(b) 不过 = plan 已足够代表执行，这条机制没有可吃的空间。

## 数据（盒上已有，不新滚）

- 决策 = drive.jsonl 的一条 `drive` 记录（2 Hz，k = 0..17）。输入：`poses`（served plan，行坐标系 = 该决策的 `anchor` 后轴位姿）、`ego`（20 维：速度、加速度、3 个历史位姿）。
- 目标「执行轨迹」：同一 session 第 k+1..k+8 个决策的 `anchor`（模拟器自车后轴，0.5 s 网格，正好对上 plan 的 8 个时刻）变到第 k 个 anchor 系。只用 k + 8 <= 17 的决策（k <= 9）。
- 训练：on1_JT 的四个 S 来源 pool 行（`runs/onpol_nuplan/rows/L18RB0 / L18RTB1 / L18YS / L18Y`，onpol-train 场景，28 个 log unit），pool 的排除规则照旧。
  按 log unit 留验证：crc32(logu) % 7 == 0 的 log 只做早停与次要读数。
- 测试（闸）：onpol-held 场景（16 个 log unit，训练里没有）：
  - T1 = RSG-L18-s0 在 held 上的 rollout（`runs/onpol_long/roll/RSG-L18-s0-held`，pool 行 `L18RSGallheld`，有道路栅格与模拟器车框）：闸 (a) 与 (b)。
  - T2 = JTN-L18-s0 held 460 读数（`runs/route_mem/read/JTN-L18-s0`，默认 serve，没有栅格）：只做闸 (a)，所有 k <= 9 的决策。
  两个策略都不在训练来源里。

## 模型（先定，不按测试结果调）

- MLP 3 x 256（SiLU），输入 plan 24 维 + ego 20 维（训练集标准化），输出残差 exe − plan（x, y, yaw 共 24 维）；Huber（xy，delta 0.5 m）+ yaw（rad × 2）；
  AdamW lr 1e-3、wd 1e-4、batch 512、60 epoch 余弦，按验证 log 的 4 s 末点中位误差选 epoch。seed 0。
- 参照：B0「执行 = plan」；B1 同输入的岭回归（线性）。

## 闸 (a)：前向模型准不准

在 T1、T2 上各自：4 s 末点误差 ‖p̂(4 s) − e(4 s)‖ 的中位 **< 0.20 m**，且「明显好于执行 = plan」：中位不超过 B0 中位的一半，并且逐行误差差（B0 − 模型）的均值按 log 聚类 bootstrap（jevdrive.stats，B = 10 000）95 % CI 下界 > 0。T1、T2 都过才算过。
另报每个时刻（0.5..4 s）的误差、纵向 / 横向分解。

## 闸 (b)：判后果准不准

T1 的 pool 行（k <= 9）。事件在实际执行扫掠上判（8 个执行位姿 + 原点，`drivable_hinge.interp_matrix` 插到 0.1 s，步 1..40，与 S hinge 同一套足迹代码）：
- 出路：4 个足迹角在道路栅格（pool `road.npz`，S 的 road hinge 栅格）上的 SDF 最小值 < 0；t0 已出路的行（`road_off_t0`）不计入出路判定。
- 接触：`agent_hinge.AgentHinge`（margin 0、原车框、counted = 前方且横向 3 m 内或已重叠）的最小有符号距离 < 0。
- 主事件 = 出路或接触；分数 = −min(角点 SDF 最小值, 车框距离最小值)（越大越危险）。同一函数分别作用在 plan 上与预测执行上。
- **过闸：AUC(预测执行) − AUC(plan) >= 0.03，且按 log 聚类 bootstrap（重抽 log unit，B = 10 000，seed 0）95 % CI 下界 > 0。**
- 次要：出路、接触各自的 AUC；验证 log（训练来源的留出 log）上同样读数。

## 分层（只报告，不作闸）

- 转弯：plan 4 s 航向 |ψ8| >= 15° 为弯，< 5° 为直，中间为缓弯。
- 速度：决策时速度 < 1、1–5、5–10、>= 10 m/s。

## 前置检查（不过就先修代码，不读闸）

- 执行 0.5 s 位姿对 plan 第 1 个位姿的中位差 < 0.10 m（坐标系对齐；d308：控制器跟 plan 到 2–3 cm）。
- 每个 pool 行都能在 drive.jsonl 里按 (scene, k, anchor) 唯一配上 session。

## 判后

- (a)(b) 都过：设计（不训练）S hinge 作用在冻结前向模型的预测执行位姿上，估代价。
- (b) 不过：结论「plan ≈ 执行，这条机制没有可吃的空间」，第 4 项关闭。
- (a) 不过、(b) 过：机制有空间但模型不够准，不写进训练；记下差在哪个时刻 / 分层。

成本：CPU 解析 + 一张卡几分钟训练，全程一个 pool 作业（优先级 10）。
