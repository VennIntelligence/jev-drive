# GT-DECOMP 预登记：榜单 "Distance to GT Trajectory" 拆成纵向（沿路径前后）与横向（离路径）两部分，我们对参考 driver

2026-10-12 写，写定并 push 于任何读数之前。本稿之前只看过：评估器代码（下文 §1 引的行）、一个 YR rollout 的 metrics.parquet 指标名与时间戳、
一个 run 目录的 `eval-config.yaml` 与 `results-summary.json` 字段名（含该 chunk 的 run 级均值，这是 aud1 已经报过的量）。没有算任何登记量。
代码 `experiments/gt_decomp/scripts/gd.py`（box 端 collect，Mac 端 report）。CPU，0 卡时，`runs/gt_decomp/` 之外不写。

## 0. 问题

第 278 条：我们在全部已存决策上只在巡航落后人类一项比每个参考都差（5–10 m/s 起步慢 0.87–0.92 m/s，5 s 后落后 4.1 m）。提交 #2
（YR10m10-F-s0）榜单 GT 距离 private 1.733 m，本地 0.941 m，直行基线 1.15 m。问题：**我们的 GT 距离里多少是沿路径的前后（纵向），多少是离开人类路径
（横向）；我们比参考多出的 GT 距离在哪一部分；纵向课 L（research/self-consequence）是不是下一个训练臂、该登记什么预期。**

## 1. 评估器的定义（读代码得到，box `~/data/third_party/alpasim` @ 0bb4c4b；e2e_challenge 分支 2023534 同文件同行）

- `src/eval/src/eval/scorers/ground_truth.py` 第 59–60、106–108 行：每个评估步 `dist_to_gt_trajectory = full_gt_linestring.distance(ego_polygon.centroid)`，
  **到整条日志路径（折线）的最近点距离，不按时间对齐**。`full_gt_linestring` = 日志 ego 轨迹（rig 位姿经 `transform_ego_coords_rig_to_aabb` 变到车身框中心，
  `accumulator.py` 第 178–180 行）的 2D 折线。前两步置 0（第 67 行）。本定义自 2025-11-17 首个公开提交起未变（`git log` 三个版本同一行）。
- 同文件第 99–105 行另有 `dist_to_gt_location` = 到**同一时刻**日志位置的欧氏距离（时间对齐，即 ADE 类量）；第 109–116 行 `lateral_dist_to_gt_trajectory`
  = 同上最近点距离，但投影越过日志路径终点时只取垂直于终段方向的分量（第 20–40 行）。
- 时间聚合 MAX（第 173 行）。聚合前截断：`eval_relevant` 之前的步去掉、第一个 `collision_any or offroad` 之后的步去掉（`aggregation/processing.py`
  第 37–64 行），第一个 `dist_to_gt_trajectory >= 4.0` 之后的步去掉（`aggregation/main.py` 第 155–160 行，事件步本身保留，`modifiers.py` 第 100–112 行）。
  榜单列 = run 内对 clip 的均值（aud1_private_gap.md §1，`aggregate_over_clips`）。
- 评估步：t = 0、17 000、17 000 + 500 000 k µs（k = 1..10），即每个决策一步，前两步（0 与 17 000）为 0。

**由定义推出的预期（登记，读数检验）**：榜单 GT 是到路径的最近距离，**沿路径落后对它的贡献恒为 0**；纵向只能经「越过日志路径终点」进入
（`dist_to_gt_trajectory − lateral_dist_to_gt_trajectory`，只在投影超出终点时非 0，只有比人快才会发生）。所以登记预期：我们（落后人类）的榜单 GT 里
纵向部分 ≈ 0，比参考多出的部分几乎全在横向；时间对齐误差 `dist_to_gt_location`（第 271 条 ADE 类量）大部分在纵向。

## 2. 数据（只读）

与第 278 条 cap_audit 同一套 rollout、同一 label → run 目录规则（`ca.py run_dirs / pick_rollouts`，导入不复制）：主读数 10 决策 public，1479 scene / 136 log，
10 个 label：我们 `YR10m10-F-s0`（提交 #2）、`YS2G-L18-s0/s1`、`YS-L18-s0/s1`（seed 均值行 `YS2G` / `YS-L18`）、`APY10m10-AB-s0`（提交 #1，只描述）；
参考 `WA-JEPA`、`LTF`、`DD`（DiffusionDrive）、`G0014`（garage 0014）。次读数 18 决策 held（`alpasim_nuplan/onpol-held@v1`，452 scene），同表只描述。
分组沿用 cap_audit `ca_report.groups`：转弯几何 straight / gentle / kerb_turn / across_turn，起步速度 v<0.5 / 0.5-5 / 5-10 / ≥10，nuPlan 标签组。

## 3. 闸

- **G1a 聚合复现**：从每个 rollout 的 `metrics.parquet` 序列按 §1 的截断规则取 MAX，得到的 `dist_to_gt_trajectory` 与 `results-summary.json` 里该 rollout 的
  值之差 ≤ 1e-3 m，每个 label ≥ 99.9 % 的 rollout；每个 run 目录里对 clip 取均值与其 `metrics_results[0].dist_to_gt_trajectory` 之差 ≤ 1e-3 m（全部 run 目录）。
  `dist_to_gt_location`、`lateral_dist_to_gt_trajectory`、`progress_rel_to_total`（LAST）、scene score（§6 规则）同样检查。不过 = 停，报告。
- **G1b 几何复现**：用 scene loader 的日志 rig 轨迹经 rig→车身框中心偏移得到 GT 折线，用 controller CSV 在 t_k 的 rig 位姿加同一偏移得到 ego 中心，
  shapely 重算每步 `dist_to_gt_trajectory` 与 `dist_to_gt_location`；与 parquet 之差 ≤ 1e-3 m 的步 ≥ 99 %（每个 label，保留步）。
  不过：榜单拆分（§4 A）只用评估器自己的三个量，不受影响；时间对齐拆分（§4 B）照算，但写明实测误差分布，并把 |cross| 换成评估器的
  `dist_to_gt_trajectory`、along 用 sqrt(max(e² − d², 0)) 带符号（符号取几何）作为替代读法。

## 4. 拆分（每个评估步 k，每个 rollout 的保留步）

**A. 榜单 GT 的拆分（主问题）**：d_k = dist_to_gt_trajectory，l_k = lateral_dist_to_gt_trajectory，o_k = d_k − l_k（越过终点的纵向部分，≥ 0）。
rollout 的 GT = max_k d_k；在 argmax 步拆成 GT_lat = l_{k*}、GT_over = d_{k*} − l_{k*}。逐 driver 报：均值 GT、GT_lat、GT_over、GT_over 占 GT 的比例、
GT² 里纵向的份额 Σ(d² − l²) / Σd²（argmax 步）、有 o_k > 0 的 rollout 数；逐决策 k 的均值 d_k、l_k。

**B. 时间对齐误差的拆分（解释第 271 条，也回答「前后 vs 左右」的直觉）**：在人类路径框里，s_h(t_k) = 日志在 t_k 的中心点在 GT 折线上的投影弧长，
s_e = ego 中心的投影弧长，**along_k = s_e − s_h（正 = 领先人类）**，**cross_k = 带符号最近点距离（左正）**。逐 driver 报：均值 e_k、均值 along（带符号）、
均值 |along|、均值 |cross|、纵向份额 Σalong² / Σ(along² + cross²)；逐决策 k 的均值 along（滞后曲线，图）。

**配对**：我们每行（YR、YS2G、YS-L18）减每个参考，逐分量（GT、GT_lat、GT_over；|along|、along、|cross|），scene 级配对差，log 聚类自助（ca_report.Boot，
10 000 次，seed 0，按 log 重抽、比值和）。**「我们多出的 GT 在纵向」的判据**：对四个参考里的每一个，GT_over 差占 GT 差的比例 ≥ 0.5 且 GT_over 差的 CI 下界 > 0。
否则写「多出的部分在横向」（GT_lat 差 CI 下界 > 0 时）或「无可分辨差」。

## 5. 反事实（CPU，不重滚，只改时间不改路径）

- **(a) 我们的纵向换成人类的**：把 ego 的 10 Hz controller 轨迹按弧长参数化，取每个 s 处的带符号横向偏差 c(s)（在 GT 折线框里）；反事实下 ego 在 t_k 位于
  人类的弧长 s_h(t_k)，横向保持 c(s_h(t_k))。s_h 超出 ego 实际到过的最大弧长时 c 取最后一个值（**外推**，单独报外推步占比，再报只在到过的弧长内取的下界版本）。
  反事实 d_k = |c(s_h(t_k))|（人类不越过终点，o = 0），截断规则照 §1（≥ 4 m；碰撞 / 出路的截断时刻沿用原 rollout，不能重模拟）。报反事实 GT 均值 − 实际 GT
  均值（配对，CI）。登记预期：差 ≈ 0 或略正（落后时没走到弯道后段，最大值取不到；换成人类节奏会走到）。
- **(b)** 同法用于每个参考。
- **(c) 算术（标明只是算术）**：在「横向不变」下，什么样的沿路径落后能把本地 0.941 m 变成 private 1.733 m？按 §1 定义答案是「任何落后都不能」；
  只有领先并越过终点能加纵向分量。另算：在本地每 clip 的结构下（<4 m 的 clip 均值、≥4 m 的 clip 均值），1.733 m 需要多少比例的 clip 横向离开 4 m 走廊。

## 6. 进度分（scene score）

scene score = 0（collision_at_fault / offroad / left_corridor_laterally）否则 progress_score；progress_score = 1 若日志路径 < 5 m，否则
min(clip(progress_rel_to_total, 0, 1) / 0.8, 1)（`aggregation/scene_score.py` 第 49–97 行，`eval-config.yaml` 的 5.0 / 0.8）。逐 driver 报：非零 scene 里因进度
< 1 的个数、平均扣分 Σ(1 − score) / N_scene（只算非零 scene，即纵向课最多能挣回的分）、按起步速度 / 转弯几何分组；配对我们 − 参考（CI）。
另报：纵向反事实 (a) 下进度恒为 1（人类节奏）时的平均分上界 = 实际均分 + 上面的扣分（标明是上界：不改变任何零分）。

## 7. 判读与建议（写死规则）

1. 若 §4 A 判「多出的部分在横向」且 §5 (a) 的 |差| < 0.05 m：**纵向课 L 不会动榜单 GT 距离**；它的杠杆只在 scene score 的进度项，预期上界 = §6 我们的进度扣分
   减参考里最小的扣分（登记为课 L 在 public 10 决策上的预期 Δscore 上界），GT 距离登记预期 0 ± 0.05 m（或略升）。榜单 GT 的杠杆是横向，下一臂应针对横向离开路径。
2. 若 §4 A 判「多出的部分在纵向」：课 L 同时动 GT 与进度，预期 ΔGT = −(GT_over 差)。
3. 两者之间：照实写各自份额。

## 8. 产物

`results/gt_decomp.md`（表）、`results/*.csv`、`figs/gd_split.png`（每 driver 横 / 纵堆叠条）、`figs/gd_lag.png`（逐决策沿路径滞后曲线，我们对参考）、
`research/decisions.md` 一行 + `decisions/<NNN>.md`、中文页 `research/gt-decomp/index.html`。box：`$DATA_DIR/runs/gt_decomp/collect-{pub,held}/<ts>/`。
