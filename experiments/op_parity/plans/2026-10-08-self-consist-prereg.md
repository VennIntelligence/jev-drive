# op_parity self-consist 预登记：模型自己的路沿 / lead 输出，带的信息有没有地图约束带的多（2026-10-08，任何本线读数之前写定）

## 问题

第 170 条的 hinge 用 NAVSIM HD 地图的可行驶区域当约束，按榜专用，WOD 没有地图。第 177 条要的是不靠榜单标签的方法。
想法：用模型自己的感知输出（openpilot 的 road_edges 和 lead）当约束。训练之前先问一个只读问题：
模型自己的输出对「这条 plan 会不会出界 / 撞」的预测力，和地图约束比差多少。本线不训练，不写新模型代码。

## 已有的、不重测

- 第 88 / 104 条：路沿输出在 NAVSIM 相机高度（1.87 m）下缩尺约 0.7，左右对称、随距离不变；虚拟相机降高不消 DAC 一项（第 104 条第 9 点）。
  本线不再测缩尺本身，只测「一个全局 scale / offset 校准后，边界信息够不够用」。
- 第 165 条：冻结 Cinque 特征读边界 SDF 比 WA-Cf 粗 0.25–0.35 m；第 166 条：真值边界喂 plan 通路无用。本线读的是模型已有的 road_edges 头，不是新读出头。
- 第 157 条：lead_x 近距多读（< 6 m 约 2.0 m，线性到 10 m 为 0），固定余量规则两榜不过。本线只量 lead 信号的预测力，不做执行规则。
- 第 170 条：SH30 / P2H10 的 DAC 失败、split、strata 直接复用，不重打分。

## 对象与数据

- navtest 12 146 token。plan 来源 5 个：`cinque@gimm`（出货，存档 heads 含 road_edges 与 lead）、`P2H10-F-s{0,1}`、`SH30-F-s{0,1}`（fine-tune，存档 plan 没有存 road_edges，
  纯推理重跑一次，同一 torch 端口、warp 帧、batch 128，同时存 road_edges(均值 + log-std) 与 lead_prob / lead_x / lead_v；重跑 plan 与存档导出位姿的最大差 > 0.05 m 就停下查，不继续）。
- 逐 token 失败标签：bench 存档 units.csv，DAC 失败 = `DAC < 1`；碰撞类失败 = `NC < 1 or TTC < 1`。P2H10 / SH30 用 seed 各自的 plan 与分数，Cinque 用 `cinque@gimm`。
- 转角分桶用 bench strata 的口径：logged 4 s 航向变化 |dyaw|：直行 < 5°、20–45°、> 45°、> 20°（= 前两者之和）；左 / 右 = dyaw 的符号。
- 校准集 navtrain：fine-tune 用 `navtrain_full.s0of12@warp`（8 608 token）上同一推理；Cinque 用 `lb_navtrain` 存档 heads（3 000 token）。校准只读 navtrain，navtest 不参与任何拟合。
  fine-tune 在这些 token 上训练过（in-sample），记为限定。

## 量

**几何。** 后轴系 t0（x 前，y 左），足迹 = nuPlan Pacifica 四角（前 4.049 / 后 −1.127 / 半宽 1.1485，与 hinge 同），plan 的 8 个位姿（0.5–4 s）线性插到 0.1 s，与 hinge 同一套插值。
路沿头 road_edges (2, 33, 2) = [y, z] 在 X_IDXS（0–192 m 二次间隔）：左 = 0、右 = 1，按 `offroad_roadedge.py` 的约定落到自车系：
`ex = X_IDXS + cam_x`，`ey = cam_y − y_op`（cam = tab 里该 token 的相机在车上的位置）。

**自路沿 margin。** 足迹每个角、每个 0.1 s 时刻，在其纵向位置 cx 上对 (ex, ey) 线性插值（cx < ex[0] 取首点）得 yL(cx)、yR(cx)；
`mL = min(yL(cx) − cy)`，`mR = min(cy − yR(cx))`（路内为正），token 的 margin `m = min(mL, mR)`。
变体：(a) raw；(b) 校准：`ey' = s·ey ± b`（左 +、右 −，一组全局 (s, b)，见下）；(c) **地图 margin**（参考上界）= hinge 同一 SDF 栅格
（`runs/op_probe/labels/navtest.npz`）在同一批角点、时刻上的最小值。预测量 = −margin（越小越危险）。

**校准。** 对每个 navtrain token，在自车系 x ∈ {5, 10, 15, 20, 25, 30} m 的截面上，取模型左 / 右路沿 y，以及 SDF 栅格零交叉给出的地图边界
（从 y = 0 向左 / 右走到 SDF 变号，亚格点线性插值；ego 点 SDF ≤ 0 的截面丢掉）。
拟合 `y_map = s·y_edge + b`（左）/ `y_map = s·y_edge − b`（右），两侧共享 (s, b)，Huber（δ = 1 m）IRLS，每个模型（臂 × seed）各拟合一次。报告 (s, b) 与残差 MAD。

**覆盖（可用性）。** token「可用」= 左右路沿在 plan 足迹覆盖的纵向范围（0 到 max cx）内的每个 X_IDXS 点上：有限、yL > yR、宽度 yL − yR ≥ 2.5 m（raw 口径，校准后同样检查）。
报告可用比例：全体、> 20°、> 45°、左 / 右。主 AUC 在可用 token 上算；同时报「不可用当最危险」的全体 AUC 作敏感性。
**覆盖判据：** > 20° token 的可用比例 < 70 % 时，不管 AUC 多少，结论写「不是处处可用的约束信号」。

**AUC 与召回。** 目标 = DAC 失败，预测 = −margin；AUC（rank 形式）与 FPR = 10 % 处的召回（阈值取负例预测的 90 分位）。
分桶：全体、> 20°、20–45°、> 45°、其中左 / 右；另给分侧读数（左转看 mL，右转看 mR，即内侧路沿）作诊断。
CI = 按 log 聚类 bootstrap（136 个 log，B = 1 000，`default_rng(0)`）。fine-tune 的数 = 两个 seed 各自 AUC 的均值（CI 在 token 上先对两 seed 同 log 重采样再均值）。

**对照（只作上下文，不入判线）。**
1. 乱序对照：校准后的 margin 里，路沿读数在同一转角桶内随机换 token（plan 不动），AUC 应回到基线。
2. 仅 plan 的基线：|plan 末端航向变化| 与 plan 4 s 距离两个标量的 AUC（不看路沿，不看地图）。
3. 地图 margin 变体即参考上界。

**lead 与碰撞。** lead_prob > 0.5（第一个时刻）才算有 lead，否则 clearance = +∞。
clearance = min over t ∈ {0, 0.5, ..., 4 s} 的 `(lead_x + cam_x + lead_v·t) − (plan_front(t))`，plan_front = plan 后轴位姿的纵向位置 + 4.049 m（沿自车 x，忽略横向）。
两个版本：raw；按第 157 条的近距偏差修正（lead_x 减 2.0 m 当 < 6 m，线性到 10 m 为 0，常数照搬，没有在 navtest 上重拟合）。
预测量 = −clearance，目标 = NC / TTC 失败；AUC 与 FPR 10 % 召回，分：全体、有 lead 的 token、lead_x < 15 m 的 token。对照：仅自车速度的 AUC。
**lead 一项没有预登记判线**，只做描述，并说明和第 157 条一致与否。

**WOD（第 4 项）。** 存档的 WOD val 预测（`processed/wod_zeroshot/preds/op_cinque_*`）只有 plan / lead，没有 road_edges（已查文件键，没有读任何结果）。
所以不做；只报重跑成本并停下。

## 判线（读数前写定）

对象：navtest > 20° token，校准后自路沿 margin（可用 token）对 DAC 失败的 AUC。取 P2H10 与 SH30 共四个 (臂, seed) 的均值作判线读数（这是未来 label-free hinge 会训的模型）；Cinque 并排报告。
读数用点估计；CI 跨线时说明跨线，结论仍按点估计。

| AUC | 结论 |
|:--|:--|
| < 0.65 | 想法结束，直说 |
| 0.65–0.75 | 弱，报告并停 |
| > 0.75 | 报告 label-free hinge pilot 需要什么（不启动） |

另：若地图 margin 的 AUC 本身 < 0.85，说明足迹 SDF 对 devkit DAC 的上界读数不干净，要在结论里写清楚这点（计划轨迹 vs 被 LQR 跟踪的轨迹的差）。

## 算力与做法

推理纯 GPU 小任务：fine-tune 4 臂 × (navtest 12 146 + navtrain 8 608) token，预计合计 < 0.15 卡时，经 `python -m jevdrive.cl submit`，在 tmux `jev` 里经 `scripts/tmux_run.sh`；
分析纯 CPU，在 `jevdrive.run.Run` 内。分阶段：先 `navtest-first64` 式的 64 token 试跑核对重跑 plan 与存档一致，再全量。
turn-ceiling 占用 CPU 配额，先 `python -m jevdrive.cl probe`，只取空闲部分。

## 已知的限定（写在前面）

- 命中率衡量的是信息量，不是 hinge 能不能训出来：自监督 margin 用模型自己的输出当约束，训练时模型可以同时改约束与 plan。
- 足迹 vs 路沿的 margin 是对 plan 的，devkit DAC 对的是 LQR 跟踪后的轨迹。
- fine-tune 在校准集上训练过；校准只有一组 (s, b)，不随距离变。
- DAC 失败率约 4 %（> 20° 约 8 %），> 45° 桶 token 数 1 517，AUC 的 CI 会宽。
