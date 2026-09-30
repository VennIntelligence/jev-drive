# Log expert audit：真实 log 里的专家轨迹有多少、原生 openpilot plan 在上面错多少

状态: running（2026-09-30 22:5x 开；本文「预登记」一节写于任何 slice 计数和误差数字之前，之后的偏离都记在「偏离」一节）
主题: [../research/decisions.md](../research/decisions.md) 第 47 条（严格 S 形 nudge 的定义）；op-adapt r2 [2026-09-29-op-adapt-r2-prereg.md](2026-09-29-op-adapt-r2-prereg.md)
协调: 不碰 op-adapt r2 的 run dir、chain 文件和进程；只读它的 `t/teacher/{wod,nav,nus}`（原生 Cinque plan，已跑完，文件时间 09-30 20:00–20:25）和 `index/`。纯 CPU，用 r2 没占的核（0–7、152–174）。不训练，不写 decisions。

## 问题

冻结的 openpilot（Cinque）在四类行为上跨 benchmark 有差距：从停止起步（start-from-stop）、路口前起转（turn onset）、窄路 / 绕障的横向机动（nudge、bypass）、恢复。
r2 现在的监督只有 CARLA 行人对加对原生 plan 做 13 种扰动的规则打分，表达不了这些行为。假设：真实 log 里人类自车的未来轨迹本身就是这些行为的正确专家轨迹，而且在真实域里。
训练之前先量两件事，按数据集 × 行为 slice：

1. **量**：多少帧、多少**独立事件**（同一个 log 里同一次机动的相邻帧合并）、多少个不同的 log。
2. **原生 plan 的误差**：对人类未来的 ADE（average displacement error，逐点位移误差的时间平均），拆成纵向和横向，与同数据集的「正常直行」对照 slice 比，按 log 聚类 bootstrap 出 CI。

恢复不在范围（log 里没有离开分布的起点）。

## 预登记（写于任何 slice 计数与误差之前）

### 数据与帧集合

| 数据集 | 事件计数用的帧（全 train split） | 原生 plan 来源 | 输入协议 |
|:--|:--|:--|:--|
| WOD-E2E train | 2 037 段，偶数帧（5 Hz，与 r2 缓存对齐） | r2 `t/teacher/wod`（op，fp16 port） | 前视，5 Hz 缓存，9 个 5 Hz context（1.6 s），未校准；另报纵向 ×1.06 校准 |
| WOD-E2E val | 479 段，偶数帧，只计数 | 误差用 `processed/wod_zeroshot/preds/op_cinque` 的 1 445 个 val 帧（官方协议，3 相机，10 s 历史），原始与 ×1.06 都报 | 见左 |
| NAVSIM navtrain | 103 288 个 token（2 Hz，相邻 token 相隔 0.5 s） | 主：r2 `t/teacher/nav`（NAVSIM 2 Hz sample-and-hold，全部 103 288，decisions 第 36 条记为 off-protocol）；辅：`runs/skill_pack/n1/feat/lb_n{1,2}train_*` 的 `native`（GIMM 补帧，60 k 个 token） | 见左 |
| nuScenes train | 全部 train scene（含 r2 的 50 个 dev scene），偶数 step 中每隔一个（5 Hz） | r2 `t/teacher/nus`（20 Hz 钟，每 keyframe 间隔 10 step） | 见左 |

缓存 plan 是每个 5 Hz slot 一条（33 点，openpilot 相机系），用 `op_adapt_score.plan_at` 转到后轴自车系（相机前移 `cam_x`：WOD 1.519、NAVSIM 1.646、nuScenes 1.70 m）后取 0.5…4.0 s 共 8 点。
误差只在有完整 9 个 context 的 slot 上算（`ctx` 全 ≥ 0），以免起段补零 hidden 的 plan 混进来；事件计数不受此限。不在本 lane 里重跑任何推理。

### 运动学量（全部只从自车位姿算，各数据集同一公式）

t0 自车系（x 前、y 左，后轴），未来位置 `p_k`（k = 1…8，t_k = 0.5 k s，`p_0` = 0），段 `seg_k = p_k − p_{k−1}`，段速 `sp_k = |seg_k| / 0.5`，段航向 `ψ_k = atan2(seg_k.y, seg_k.x)`（度），**只在 `sp_k ≥ 0.8 m/s` 时有效**。
`v0`、`v_{−0.5}`、`v_{−1}`：t0、−0.5 s、−1 s 的速度（WOD past 速度状态；NAVSIM `vel`；nuScenes 位姿 ±0.25 s 中心差分），`vmax1` = 三者最大。
`dpsi_past` = 过去段 `p(−1.0 s) → p(−0.5 s)` 在 t0 自车系里的航向绝对值（度），该段长 < 0.4 m 时记 0。
所有 slice 共用 4 s 的人类未来（NAVSIM 只有 4 s）；这比第 47 条的 5 s 短，是**偏离第 47 条的地方**，见下，另用第 47 条原规则（5 s，0.25 s 网格）在 WOD val 479 个 rater 帧上复现一次作实现检查（该条记 log nudge 15 帧）。

### Slice 定义（可重叠；括号里是 base 阈值）

| slice | 条件 |
|:--|:--|
| **start**（从停起步） | `vmax1 ≤ 0.5`，且 `|p_8| ≥ 3.0`，且 `max sp_k ≥ 1.5` |
| **stay**（镜像对照，停着不动） | `vmax1 ≤ 0.5`，且 `|p_8| ≤ 0.5` |
| **stop**（巡航到停） | `v0 ≥ 3.0`，且 `sp_7 ≤ 0.5` 且 `sp_8 ≤ 0.5`（3.5 s 内停住并保持到 4 s） |
| **turn_onset**（起转前） | 未在转（`dpsi_past < 10`），未来会转（有效 `|ψ_k|` 最大 ≥ 30），起转时刻 `t_s`（第一个有效 `|ψ_k| ≥ 10` 的 `t_k`）在 [0.5, 3.0] s |
| **in_turn**（已在转） | 未来会转（同上）且 `dpsi_past ≥ 10` |
| **nudge**（无转弯的 S 形横向） | 无转弯（有效 `|ψ_k|` 最大 < 30），`v0 ≥ 3`，8 段全有效且 `sp_k ≥ 2`；峰值 `peak = max|y_k|`、末端 `y_end = y_8`、末端航向 `h_end = ψ_8`、最大航向 `h_max`：`return`：`peak ≥ 1.0` 且 `|y_end| < 0.5 peak`；`hold`：`peak ≥ 1.0`、`1.0 ≤ |y_end| < 2.5`、`|h_end| < 5`、`h_max > 2 |h_end|`；nudge = return ∪ hold |
| **lane_change** | 同一「无转弯 / 全有效」前提，`|y_end| ≥ 2.5`、`|h_end| < 10`、`h_max > 2 |h_end|`（单列，不算 nudge） |
| **control**（稳态直行） | `v0 ≥ 5`，全部 `|sp_k − v0| ≤ 1.5`，有效 `|ψ_k|` 最大 ≤ 5，`max|y_k| ≤ 0.75`，`dpsi_past < 5`，且有 intent / command 时须为直行 |
| **all** | 有 plan 的全部帧（只作背景行） |

turn_onset 另按 intent 拆：WOD `intent` ∈ {2, 3}、NAVSIM `driving_command` 为左 / 右的占比；nuScenes 没有 command，不拆。
已知局限：没有地图，弯道里的 nudge 与弯道分不开（第 47 条同）；4 s 内 `y_end` 比 5 s 小，会少算慢速的 hold；`ψ_k` 在 0.5 s 段上有噪声，阈值取得比第 47 条的 3 点平滑宽。

**敏感性**（只用于事件计数，不用于误差）：loose：start `|p_8| ≥ 1.5`、`vmax1 ≤ 0.8`；stop `v0 ≥ 2`、`sp ≤ 0.8`；turn 30 → 20、10 → 7；nudge `peak ≥ 0.7`、lane_change `|y_end| ≥ 2.0`。strict：start `|p_8| ≥ 6`、`vmax1 ≤ 0.3`；stop `v0 ≥ 5`、`sp ≤ 0.3`；turn 30 → 45、10 → 15；nudge `peak ≥ 1.5`。

### 事件与计数

- **frames**：满足条件的 slot 数（各数据集自己的采样率：WOD 5 Hz、nuScenes 5 Hz、NAVSIM 2 Hz），同时报 **seconds** = frames / 采样率，两者可跨数据集比较。
- **independent events**：同一 (数据集, log) 里满足条件的帧按时间排序，相邻两帧时间差 > 1.0 s 就断开，每个连续段记 1 个事件；WOD log = 20 s segment，NAVSIM log = log_name，nuScenes log = scene（同一次驾驶会切成多个 scene，所以 nuScenes 的独立性比 WOD 弱，读数时注明）。**distinct logs** = 至少有 1 个事件的 log 数。
- 事件数在整个 train split（含无 plan 的帧）上算；带 plan 的事件数（`events_plan`）另列，作为误差那一列的样本量。

### 误差读数

对每个带 plan 的帧，plan（后轴自车系，同一 t0）与人类未来逐点比：`e_k = plan_k − p_k`。
- `ADE_H` = 平均 `‖e_k‖`（t_k ≤ H），`lonADE_H` = 平均 `|e_k.x|`，`latADE_H` = 平均 `|e_k.y|`，H ∈ {1, 2, 3, 4} s；另报有符号纵向偏差 `bias_lon_3` = 平均 `e_3s.x`（负 = plan 比人慢 / 短）。
- 与 control 的比：`ratio = slice 均值 / control 均值`，同一数据集内。
- **CI**：按 log 聚类 bootstrap（有放回重抽 log，slice 与 control 用同一次重抽，2 000 次，seed 0，取 2.5 / 97.5 分位）；均值是帧加权。CI 同时给 slice 均值与 ratio。
- WOD 报 raw 与 ×1.06（plan 的 x 乘 1.06）两版，NAVSIM 报两个 plan 来源，nuScenes 只有 raw。

### 判读规则（写死）

1. **量的分档**（train split，独立事件数）：A ≥ 1 000；B 100–999；C < 100。
2. **原生误差「明显高于 control」**：在 slice 对应的分量上 ADE@3s 与 ADE@4s 的 ratio 的 95% CI 下界都 > 1.5。对应分量：start / stop → `lonADE`；turn_onset / nudge / lane_change → `latADE`；同时也看总 `ADE`，但以对应分量为准。
   跨数据集：至少两个数据集都有 ≥ 30 个带 plan 的事件且点估计 ratio > 1.5 才算「跨数据集一致」；只有一个数据集达标的记为「单数据集」。
3. **归类**：A 档 + 明显高于 control + 跨数据集一致 → **log 模仿候选**；A 档但 ratio CI 含 ≤ 1.5 → 「数据够但原生已经不差，模仿的边际小」；B / C 档且明显高于 control → 「需要仿真数据补量」；C 档 → 「log 里基本没有」。
4. **ratio 的解读限度**：单条人类未来不是唯一正确答案（多模态），原生 plan 与它不同不等于错。所以只用 slice 对 control 的相对差，不读绝对 ADE 为「错」；start 与 stop 分量偏差的符号一起看。
5. 与 stay 比的镜像检查：start 的纵向误差应明显大于 stay（停着不动时原生应该也不动）；若 stay 的 lonADE 也高，说明原生在停车帧上有系统偏差，start 的高误差不能全算作「不会起步」。

### 预算与不做的事

约 4 h 墙钟，CPU 核 ≤ 数百 core·h，GPU 不用（缓存 plan 已有）。不训练；不改任何已登记协议；不写 decisions；不删除文件；每个 devkit / NAVSIM 进程带 `OPENBLAS_CORETYPE=Haswell`；不用 `pkill -f`。

## 步骤
- [ ] 预登记提交（本节），推送，box 拉取
- [ ] `scripts/log_expert_audit.py build`：三个数据集的帧表（位姿量 + 未来 8 点 + 缓存 plan）到 `$DATA_DIR/processed/log_expert_audit/`
- [ ] 实现检查：第 47 条原规则在 WOD val 479 帧上复现；缓存 plan 与 wod_zeroshot preds 在重叠帧上数值一致
- [ ] `analyze`：slice、事件、误差、bootstrap → `research/results/log-expert-audit/*.csv`
- [ ] 图、结果表、报告

## 偏离
（暂无；之后的偏离在这里记：改了什么、为什么、在看到哪些数字之后）

## 结果
跑完再填。
