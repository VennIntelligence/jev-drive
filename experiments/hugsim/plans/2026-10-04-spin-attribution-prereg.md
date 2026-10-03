# HUGSIM 打转归因：谁打转、闭环增益、Lebowski 重合、navhard 约束（预登记）

2026-10-04 写，在任何新特征提取或统计之前。纯 CPU，不跑闭环、不占卡。依据：决定 88 / 90 / 92 / 94 / 96 / 100 / 106 / 109，
`results/launch_lean.md`、`history_derotate.md`、`controller_spin.md`。已知（读日志，不算结果）：HUGSIM 从不静止起步（step 0 v = 1.0，
单帧预热 5 s 后）；原生 Cinque 10 个打转里 6 个在 ~3.5 s 内开始、3 个在中途停车 / 蠕行之后、1 个在 10 m/s；非打转场景也有走走停停；
64 个场景里 46 个 launch lean ≥ 1°。目标是归因，不选规则或模型修法。

## 数据与标签
- 标签：`results/spin/spin_episodes.csv`，agent = cinque / lebowski，controller = fixed（PR#57），64 个场景各一行；Cinque 10 / 64、Lebowski 11 / 64。
  it_dw3 / pilot（决定 98 / 101 / 106 的 ln1 / ln3）的打转标签只在有逐场景日志时用，没有就说没有，不外推。
- 日志：box `hugsim-exam/scored-op/{cinque,lebowski}-fixed`，`hugsim-derot/cinque-fixed-{base,derot3,sel3}`，每个 run 的 `zs_steps.jsonl`
  （v、theta、pos、plan、raw_plan、model_pos、lead_prob、engaged）、`infos.pkl`（obj_boxes、ego）、`ground.ply`；`results/hugsim-exam/scenarios.csv`
  （数据集、n_ahead、ahead_min_b、route heading 变化、turn）；`results/lean/lean_per_scenario.csv`（d100 的 launch lean）。
- 不用 GPU。若 CPU 上能跑 openpilot ONNX（`OPModel(name, "cpu")`），问题 2 的局部增益在 CPU 上做；跑不动就只做日志推断。

## 防泄漏
特征是起步窗口的量，不能被打转本身污染：打转 run 的窗口截到 onset 前一步（`spin_episodes.onset` / `start`），非打转用同样的绝对窗口
（前 K = 16 步 = 4 s 和前 40 步 = 10 s，打转 run 截到 min(K, onset)）；时间类特征用「窗口内占比」，不用绝对步数。这降低但不消除泄漏
（onset 本身由速度和朝向定义），所以结果是描述性的，不是预测模型。

## 问题与判读
**Q1 谁打转（Cinque 原生，64 个，10 对 54）。** 特征：|launch lean|（d100）；窗口内 v < 3 / v < 1 m/s 占比；起步次数（v 从 < 1 到 ≥ 1.5 的次数）；
前方障碍（obj_boxes 中车道内 |lateral| < 2 m、0–40 m 的最近距离；scenarios.csv 的 n_ahead / ahead_min_b；日志 lead_prob 均值）；
数据集（四类，另报 nuScenes 与其余）；难度；路线 heading 变化与 turn（scenarios.csv）；起步朝向与路线首段方向之差；路宽代理
（ground.ply 在 3–25 m 前方的横向可行驶范围，左右各一）；`engaged`。车道线 / 路沿置信度不在日志里（需要模型前向），**只在 Q2 的 CPU 前向可用时补，
否则标「未测」**。单特征 AUC（10 对 54，Mann–Whitney，bootstrap 2000 次 95% CI，分层重采样）。「分得开」= CI 不含 0.5；因为约 15 个特征，
同时报在 5% 水平期望偶然出现的个数（约 0.75），并对 CI 下界 ≥ 0.5 但 p 不小的特征说明。多变量只在满足 (i) 特征数 ≤ 3、
(ii) 特征选择在留一折内部完成、(iii) leave-one-out AUC 的 CI 不含 0.5 且高于最好单特征的 LOO 值，才报；否则写「n = 10 个正例撑不起」。
**阻挡假设**（d96：9 个去旋转后不打转的里 5 个停在原先绕开的障碍后面）：定义「起步窗口里车道内 25 m 内有障碍」并与「窗口内 v < 3 m/s 占比
≥ 50%」联合；检验打转组该条件的比例高于非打转（Fisher 单侧）；再看 10 个打转里 onset 前计划是否在障碍侧偏离。判「成立」要求
OR > 1 且单侧 p < 0.1 且至少 6 / 10 的打转满足；「否」= 打转组比例不高于非打转；其余「不确定」。
同样的特征表对 Lebowski（11 个）重做一遍，看方向是否一致。

**Q2 非打转场景的环路增益。** 样本：≥ 10 个非打转场景，含 launch lean 最强的 5 个与走走停停（窗口内 v < 1 m/s 次数最多）的 5 个，
对照 d100 的 3 个打转场景 × 3 日志和其余打转日志。读数与 d100 一致：step 1–2 局部增益（日志历史上叠 ±w 假偏航，w = 2°/s）、大信号增益
（日志历史 vs 去旋转历史的 1 s 计划方向差 / 历史偏航）、控制器传递 c、由此得到的每步增长倍数。**CPU ONNX 能跑才做前向版本**；否则只做
日志推断版本：φ1 的步间增长、φ ~ λ + s·H 的回归（H = 过去 6 步偏航，窗口内 v < 3）、c（Δθ ~ φ）。判读：比较打转与非打转的三个量
——增益 s（是否更低）、扰动 λ（起步第 1–2 步的 |φ| 是否更小）、低速窗口长度（v < 3 的步数是否更短）；哪个量的组间差异最大且 CI 不含 0
就当主解释，几个都不显著就写「分不开」。

**Q3 Lebowski 重合。** 逐场景表：Cinque / Lebowski 打转标记；Jaccard（10 和 11 的交 / 并）。零假设（两者独立、边际固定）下的期望交集
= 10·11/64 = 1.7，超几何 p 一并报。能找到 it_dw3 / pilot（决定 98 / 101 / 106）的逐场景标签就加入。

**Q4 navhard 约束。** stage 2（stage 1 参考）里参考（PDM-Closed 或人类参考，先查 d94 臂用哪个，选 per-token 表里有的）前 1 s、2 s 的朝向变化
≥ 5° / ≥ 10° 的 token 数（占比）；原生、rot0、straight、selector 在这些 token 以及其余 token 上的 EPDMS（d94 的 per-token 分数）；给配对差
（臂 − 原生）与 token 级 bootstrap CI。输出一个「必须不伤」集合：stage 1 全部 + stage 2 中参考 early turn ≥ 5° 的 token。
若 per-token 表不在 box 上就说明，不重算。

## 不做
不选规则或模型修法；不改 decisions；不碰 docs/web-reader.md；不 `pkill -f`；不占卡。
