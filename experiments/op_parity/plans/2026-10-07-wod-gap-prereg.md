# op_parity WOD gap 预登记：WP2 的 RFS 丢在哪、不训练能否拿回（2026-10-07，任何本线打分之前写定）

## 问题

第 163 条：WOD val（479 rater 帧）上 shipped 8.005、WP2 8.111、logged future 8.13、top-rated rater 轨迹 9.59；test 榜首约 8.17（高于日志）。
问：WP2 相对「可达到的分」（top-rated 9.59、日志 8.13）丢在哪些帧、是速度剖面还是路径；其中有没有不重新训练就能拿回的部分。
全部离线，用 box 上存档的 val 预测（`preds/op_cinque`、`op_cinque_WP2-full-s0/s1`、`op_cinque_WP1-full-s0/s1`），不训练、不 serving、不提交 test。

## 不重测

- 第 162 条：P2H 的损失是 bias 常数项、纯速度剖面（O1 / O2）。这里只复用它的 retime 算子和分层定义，不重跑 P2H。
- 第 163 条：WP2 / WP1 / shipped 的 RFS、ADE、夜昼表，直接引用；本线脚本先复现 8.005 / 8.111 / 8.012 作为等价检查。
- 第 79 条（RFS 不动是因为翻转帧太少）、第 131 条（shipped 夜间）只作口径。

## 臂与参照

| 名字 | 轨迹 |
|:--|:--|
| shipped | `preds/op_cinque` |
| WP2 | s0、s1 各自打分后逐帧平均（「seed mean」，与 wod_parity.md 同口径）；几何量（落点偏差、类型）每个 (帧, seed) 算一次，权重各 1/2 |
| WP1 | 同上（只作对照行） |
| log | `future` 的 xy |
| top | 每帧 rater 分最高的轨迹（并列取第一个）；second / worst 只作参考行 |

Split：`jevdrive.data.splits.load("wod/val")`，`run.use_split`；断言 rater 帧的序列集合 = split 成员。

## A. Gap map

**逐帧量**（RFS port 的常数原样引用；脚本内的分解函数必须逐帧复现 `W.rater_feedback_score`，差 < 1e-9）：
- 每个 horizon h ∈ {3 s, 5 s} 的分量 s_h = max_r score_r · 0.1^max(norm_rh − 1, 0)；未触地板时 RFS = (s_3 + s_5) / 2。
- gap_i(A, R) = RFS_i(R) − RFS_i(A)。聚合用榜单口径（cluster 内平均再对 cluster 平均）。为了让分层贡献可加，定义帧权重
  w_i = 1 / (n_cluster(i) · n_clusters)，Σ w_i · gap_i 恰等于 cluster-mean 的 gap；一个子集的「丢分点数」= Σ_{i∈子集} w_i · gap_i，子集内平均 gap 另列。
- 地板：RFS_i ≤ 4 + 1e-9 记为 floored。报告 floored 帧占比、它们占总 gap 的份额；另报「不在任何单个 rater 信任域内」(inside = False) 的占比。
- horizon：loss_3 = (top − s_3) / 2，loss_5 = (top − s_5) / 2，floor 抬分 = RFS − (s_3 + s_5) / 2（≥ 0），三项加权和 = 总 gap。
- 模式：帧（对臂 A）分为 (a) 在 top 轨迹信任域内（两个 horizon 都在）；(b) 不在 top 的、但完整落在另一条较低分 rater 的域内；(c) 不在任何域内。

**纵向 / 横向归因规则**（先写定）：
1. 度量内：在 top 轨迹自己的纵 / 横坐标系里，3 s、5 s 的带符号纵向偏差 e_lon（正 = 比 top 走得远）和横向偏差 e_lat，各除以该帧的阈值。
   取两个 horizon 中 max(|e_lon|/thr, |e_lat|/thr) 较大的那个 horizon 定类型：
   `inside`（两个 horizon 都 ≤ 1）、`ahead`（纵向超、横向不超、e_lon > 0）、`behind`（纵向超、e_lon < 0）、`lateral`（只横向超）、`both`（都超）。
2. 交换（O1 / O2，`pp_wod_diag.retime`）：对 (A, R)：O1 = A 的路径配 R 的逐点弧长；O2 = R 的路径配 A 的弧长。
   纵向份额 = (RFS(O1) − RFS(A)) / gap，横向份额 = (RFS(O2) − RFS(A)) / gap（非线性，两者不必加到 1）。
   **速度剖面「承载」gap**：纵向份额 ≥ 0.5 且 O1 − A 的 CI 不含 0；**路径承载**：横向份额 ≥ 0.5 且 O2 − A 的 CI 不含 0；0.25–0.5 记「部分」。
   (A, R) 跑：(WP2, top)、(WP2, log)、(shipped, top)、(log, top)、(WP2, shipped)。
3. 逐帧标签（给排名表用）：该帧 gap ≥ 0.5 时，O1 拿回 ≥ 一半而 O2 没有 → speed；反之 → path；都拿回 → either；都没有 → joint。

**失败类型**（排名表的行）= 度量内类型 {ahead, behind, lateral, both} × 情境 {stopped v0 < 0.5 m/s, moving}，只统计 gap > 0 的 (帧, seed)；
`inside` 但 gap > 0 的归「lower-rated mode / 其他」。每行：帧数、丢分点数（Σ w·gap，对 top）、占总 gap 的比例、对 log 的 gap（WP2 − log）、
逐帧标签分布（speed / path）、同一批帧上 shipped 与 log 的 gap、log 是否同类型的比例。排名按丢分点数。

**分层**（与 wod_p2h_diag 相同定义）：速度档 stopped < 0.5 / slow 0.5–5 / mid 5–12 / fast ≥ 12；stopped 里 log 5 s 位移 < 1 m（stays）与 ≥ 1 m（moves）；
launch（v0 < 2 且 log 5 s > 5 m）；intent 转弯（2、3）/ 直行（1）；lead（shipped lead_prob > 0.5）；夜（luma < 50）/ 昼（≥ 120）；scenario cluster。
每层：n、top / log / WP2 / shipped 的 RFS、WP2 的丢分点数、WP2 − log、WP2 − shipped（带 CI）、floored 占比、O1 / O2（对 top）。n < 8 的层不报。

**日志自己的 gap map**（A = log，R = top）：同一套表。另报「rater 偏好什么」：在 log 不在 top 域内的帧上，
带符号 e_lon（3 s / 5 s，米；负 = rater 比日志走得远）、|e_lat|、5 s 路程比 top / log、前 1 s 路程比（起步 / 制动早晚）、
top 相对 log 的平均减速度差（由 5 s 路程和 v0 估）、是否有 lead、速度档。分类：rater 更慢 / 更快 / 横向不同 / 两者，各自帧数与丢分点数。

**图**：排名前几的失败类型里，每类取丢分最大的 1 帧和丢分处于该类中位数的 1 帧，共 6–10 张 BEV（WP2 两 seed、shipped、log、3 条 rater 轨迹带分数，
3 s / 5 s 点标出），存 `figs/wod_gap/`，结果文档逐张写看什么。选帧规则即此，不手挑。

**早停**：全局 O1 / O2 和类型表给出清楚答案（一类承载 ≥ 0.5）后，不再加新的分层。

## B. Seed ensemble

- ENS2 = WP2 s0 与 s1 轨迹逐点平均；ENS3 = (ENS2 + shipped) / 2。
- 指标：RFS（479）、ADE@3s / @5s 对日志（1 437 帧，wod_parity 口径）、ADE 对 top 轨迹（479，官方口径）并列。
- 配对 bootstrap（按序列）对 WP2 seed mean 和对 shipped。

## C. 速度剖面修正

- 算子：plan 的逐点弧长乘 k，路径不变（`retime` 到缩放后的弧长；超出末端沿末段方向外推）。对两个 seed 各自做，RFS 逐帧取 seed 平均。
- 网格：k ∈ {0} ∪ {0.50, 0.52, …, 1.50}（一个网格，全局与分层共用）。
- **C1 全局 k**：总是跑。**C2 分层 k**（只有 A 判定纵向承载或部分承载时才跑）：C2a 按速度档 4 个 k；C2b 速度档 × lead 8 个 k。
- 拟合目标：训练折上 Σ w_i · RFS_i（w 为全样本的帧权重，固定）。分层时各格独立取 argmax（加权和可加），并列取最接近 1 的 k。
- **折外**：按序列 5 折（rater 帧一序列一帧），折分配 `default_rng(seed)` 随机置换，重复 20 次（seed 0–19）；每帧的折外分 = 20 次的平均；
  报告折外 RFS、与 WP2 的配对差和 CI；旁边列全样本拟合（in-sample）的数和选到的 k，乐观度 = in-sample − 折外。
- 另列 oracle 上限（不可部署，只标尺）：每帧取 top 的弧长（即 A 的 O1）。

## 判读（写在结果之前）

- **可拿回（recoverable）**：某个不训练的操作（B 或 C）的 **折外**（B 无拟合参数，直接读）d RFS 对 WP2 的 CI 下界 > 0。CI 含 0 → 不算。
- 统计：配对 bootstrap，按序列重采样，B = 4 000，`default_rng(0)`，百分位 2.5 / 97.5，RFS 用 cluster-mean 聚合（wod_parity.py 同一写法）；
  ADE 用 `jevdrive.stats.paired(groups=sequence, n_boot=4000)`。
- 479 帧、对比很多（分层 × 臂 > 50 个）：单个贴边的 CI 只读作弱；结论只建立在全局规则达标、且两个 seed 同向的读数上。
- 事后追加的任何臂 / 分层在结果文档里标 post hoc。

## 约束

CPU 离线；val rater 标签只用于分析和 k 折拟合；不向 Waymo 提交任何东西；不改 research/decisions.md 与任何 HTML。
