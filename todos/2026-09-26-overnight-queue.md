# 夜间队列（2026-09-26 深夜，5–6 h 空卡）：只排预登记过、不需要新决定、跑错也不亏的活

状态: 待排（写于 2026-09-26 00:50，用户睡前；由 elicitation agent 按卡空出的顺序领取）
上游: [激发计划](2026-09-26-elicitation-program.md)、[fast-perception](2026-09-26-fast-perception.md)、[I3 HUGSIM 配对](2026-09-25-reactivity-program/i3-hugsim-pairs.md)（已渲染 65 个场景）、
[融合前诊断](2026-09-25-fusion-diagnostics.md)、[reactivity 计划](2026-09-25-reactivity-program.md)
夜间汇总写在 `tmp/2026-09-26-overnight.md`（agent 维护），本文件只定队列与规则。

## 规则

- 不做闭环；不做任何「看了数再选参数」的事；每项按各自 todo 里已登记的配置原样跑，新增的只有 seed 与数据切片。
- 每个 > 1 min 的作业进 tmux，写 `log.txt` / `events.jsonl`，在 `runs/schedule.md` 登记卡号；卡由中央调度员分，本文件不指定。
- 任何一项超估计 2 倍先停，不自行加资源；结果按到达顺序追加到夜间汇总，判格留给白天。

## 队列（按优先级；卡·时是估计）

| # | 任务 | 卡·时 | 核 | 买到什么 | 依据 / 预登记位置 |
|:--|:--|--:|--:|:--|:--|
| 1 | **E2 批量**：navtrain 为主、WOD 并列，SAM 抹行人 + inpainting + 安慰剂对；先过 16 对目检与残留检出 ≤ 10% 再批量 | 3–5 | 20 | 真实帧配对 | 激发计划 E2 |
| 2 | **补 seed**：M-C 各 arm（配对差分双流 / 只 Qwen / 只 openpilot / hard-example / 均匀）× seed {1, 2} 于 P5 v1 BA 与 PDM 集；NAVSIM 与 WOD 的 `temporal` 薄 head（ridge、cls）× seed {1, 2}；Q9b 网格 head × seed {1, 2} | 2–3 | 16 | 去掉「全部单 seed」这条限定；报 3 seed 的均值与极差 | 第 42 条 M-C 网格、第 40 条 (iii)、开环对比、fusion Q9b；配置一字不改 |
| 3 | **E3 挖掘（CPU）→ 只在孪生帧上抽 navtrain Qwen 特征** | 1–2 | 40 | E3 训练前置；不整库抽 | 激发计划 E3；抽取器与 P3e / E1 同一个 |
| 4 | **fast-perception 可选 D-depth**：YOLO26x-depth 给接地点定深度，替换平地抬升，只对胜出检测器做；报 P5 hazard 行人与 nuScenes 行人召回对平地 / oracle 高度的位置 | < 1 | 8 | SAM 主误差是平地抬升（0.49 → oracle 0.77），一个深度头能修多少 | fast-perception「可选 D-depth」 |
| 5 | **I3 配对上跑现有考生**：65 个 HUGSIM 3DGS 场景的 x⁺ / x⁻ / null 索引已建；抽 openpilot `temporal` 与 Qwen `L18_last`，套 P5 v1 BA 集上的 M-C head（零样本，不重训）与 `ridge_late`，报翻转率与 null false-flip | 1–2 | 12 | 真实外观配对上的第一个读数，与 E2 互为对照 | I3 子文档的标签规则与索引；judge 同 P5 |
| 6 | **E4c（CPU）**：reaction latency 曲线（两个 v1 集、全部考生）与 WOD 人类 onset 锚 | 0 | 8 | 主考卷该用哪一段窗口 | 激发计划 E4c |
| 7 | **E5**：fast-perception 表出来后按登记补齐检测器名再跑 | 0.5 | 8 | 20 Hz student | 激发计划 E5 |

## 早上要看的

1. E2 的验证表（残留检出、安慰剂地板）和造出来的对数；批量若未过验证就停在验证。
2. 补 seed 后每张主表的 3 seed 极差是否小于原 CI 半宽。
3. E3 可行性统计（两个数据集）。
4. E4c 曲线与 WOD onset 分布：PDM-Lite 0.4 s 与 BA 5.6 s 谁更靠近人类。
5. I3 上 M-C 零样本有没有动。

## 偏离与澄清日志

（执行时追加，时间戳为 box 时钟、早于受影响的数字。）

- 2026-09-26 01:12 CST [SEEDS]（第 2 项，写于任何新 seed 的数字之前）「seed」对每个 head 改的是什么，其余一字不改（λ 网格、特征、judge、bootstrap 种子、null 半分都不动）：
  (a) **M-C**（`reactivity_mc`，v1 BA 与 PDM 两套，`op_streams_vis`，预登记网格）：唯一的随机性是按 base 路线分 5 折的排列 `p5_exam.folds(t, pairs, seed)`；seed s 就是这个排列种子。
  prior（`ridge_late`）与各 arm 都在 fold 内重拟合，所以 prior 也随 seed 变；内层 λ 选择是 `GroupKFold`（确定性），考试里 null 半分与路线 bootstrap 的种子保持 0。新参数 `--fold-seed`（默认 0）。
  (b) **NAVSIM 薄 head**（`navsim_heads`）：seed s → log 分组外层 5 折 `_group_folds(seed=s)`、cls λ 的内层 20% log 划分 `_group_folds(seed=s+1)`、K = 1024 k-means 词表 `traj.kmeans(seed=s)`；s = 0 即原配置（0 / 1 / 0）。
  新参数 `--seed`；打分用官方 devkit 原样（PDMS v1.1、EPDMS main @ 0a380a9，navtest；navhard two-stage EPDMS），打分名 `heads_s<s>_<arm>`。
  (c) **WOD 第 40 条 (iii) 薄 head**（`drive_backbones.heads_train`，train 训、完整 val 评）：seed s → K = 1024 词表的 k-means 种子（`waymo_heads.vocabularies(seed=s)`）与 cls λ 的内层 sequence 划分（`Halves` 的 `GroupShuffleSplit(random_state=s)`，经 `train_context(seed=s)`）。
  `ridge_late` 的 λ 是 `GroupKFold`、拟合是闭式、train / val 固定，**没有随机性**，seed 1、2 预期与 seed 0 逐位相同（照跑，作为核对）。读数：479 个 rater 帧的 RFS（`heads_readout`）与第 22 条的 ADE 口径（s_ego 第 1–9 档、pre_onset）。新参数 `--seed`。
  (d) **Q9b 网格 attention head**（`fusion_q9b fit`，v1 BA 与 PDM）：seed 是 `_train` 的 seed（`torch.manual_seed` + 内层 20% 路线 `GroupShuffleSplit`）。配对 arm 原 run 已经按登记跑了 seed 0–2（`Q9b pair grid s0/s1/s2`），直接读已存结果，不重跑；
  hard-example 网格 arm 原 run 只有 seed 0，新参数 `--hard-seed` 跑 1、2（同一次 fit 会顺带重算配对 arm 的 s0–s2，逐项核对与原 run 相同）。
  核对：每个脚本先在 seed 0 上重跑，与已存 run 比（逐位相同，或 GPU fp32 噪声级），过了才跑 1、2。报每张主表 seed 0 / 1 / 2 的点值、3 seed 均值与极差，以及极差是否小于原 CI 半宽。
  估计：GPU 0 上 (a) 约 15 min、(c) 约 25 min、(d) 约 20 min、(b) 拟合 2 min；devkit 打分 8 个 arm × 约 23 min，两路并行约 1.5 h。
- 2026-09-26 01:31 CST [SEEDS]（seed 0 重跑核对之后、seed 1 / 2 的任何数字之前）核对结果（`research/results/elicitation/seeds/seed0_reproduction.csv`）：
  M-C（BA）重跑与已存 run 的 obs 预测最大差 0.46 mm、`criteria.csv` 逐项相同；NAVSIM 与 WOD 的 `ridge ego` / `ridge_late` 预测最大差 ≤ 2 mm（WOD）/ 8 µm（NAVSIM），算是复现。
  **分类头（`cls ego K1024`、`cls_late`）不能逐位复现**：同一 seed 0 重跑，NAVSIM navtest 上 31% 的 token、WOD val 上 47–52% 的帧选到了不同的 anchor（GPU 上 k-means 的 scatter 累加与 L-BFGS 的非确定性，推测），
  但汇总量几乎不动（navtest ADE 0.8646 对 0.8654 m；WOD RFS cluster mean 差 0.00–0.05）。所以分类头的「seed 间差」里本来就混着同 seed 的 run-to-run 噪声。处理：不改代码（改了就不再是已存 run 的配方），
  分类头多报一行「seed 0 重跑」（s0′：WOD 直接用这次重跑；NAVSIM 另打分 `heads_s0r_<arm>`），极差按 s0 / s1 / s2 算，s0′ 只作 run-to-run 噪声的参照。

## 结果

（按到达顺序记在 `tmp/2026-09-26-overnight.md`；早上汇总后把判格写回各自的 todo。）

### 补 seed（第 2 项，2026-09-26 01:12–04:18，GPU 0 约 1 h、GPU 3 5 min，devkit 打分 CPU 2 × 12 线程约 2.3 h）

口径见偏离日志 [SEEDS] 01:12 / 01:31。代码 `scripts/seeds_overnight.sh`、`jevdrive/elicit_seeds.py`（各 head 只加了 seed 参数，默认值不变），
表 [research/results/elicitation/seeds/](../research/results/elicitation/seeds/)（`mc_seeds.csv`、`navsim_seeds.csv`、`wod_seeds.csv`、`q9b_seeds.csv`、`seed0_reproduction.csv`）。
「半宽」是原 run（seed 0）的 95% CI 半宽；「极差 < 半宽」即 seed 带来的变化小于原本报的抽样不确定性。

**M-C（P5 v1 BA，seed = 路线分折排列）**：

| arm | 读数 | s0 / s1 / s2 | 均值 | 极差 | 半宽 | 三个 seed 都过判据 |
|:--|:--|:--|--:|--:|--:|:--|
| 配对双流 Cinque | 行人翻转 % | 43.3 / 43.6 / 42.4 | 43.1 | 1.2 | 7.8 | 是 |
| 配对双流 Cinque | cut-in 对 prior Δ pp | +3.1 / +4.6 / +2.8 | +3.5 | 1.8 | 2.6 | |
| 配对双流 Lebowski | 行人翻转 % | 41.6 / 39.2 / 41.9 | 40.9 | 2.7 | 8.8 | 是 |
| 配对双流 Lebowski | cut-in 对 prior Δ pp | +6.7 / +8.3 / +5.0 | +6.7 | 3.3 | 5.3 | |
| 只 Qwen Cinque / Lebowski | 行人翻转 % | 42.1 / 41.9 / 40.4；39.2 / 36.7 / 37.0 | | ≤ 2.5 | 8.3–8.7 | 否 / 是 / 否（cut-in 贴边） |
| 只 openpilot Cinque / Lebowski | 行人翻转 % | 6.9 / 6.2 / 8.4；17.7 / 18.0 / 19.5 | | ≤ 2.2 | 4.3–6.9 | 否 |
| hard-example / 均匀 | 行人翻转 % | 0–3.7（全部 seed） | | ≤ 1.2 | | 否 |

样本外 null false-flip 所有 arm × seed 在 4.4–5.6%。PDM-Lite 集上所有 arm 所有 seed 行人翻转 ≤ 1.3%（与 E4 一致）。

**NAVSIM（seed = log 分折、内层划分、k-means 词表）**，navtest，官方 devkit：

| 行 | PDMS s0 / s1 / s2 | 极差 / 半宽 | EPDMS s0 / s1 / s2 | 极差 / 半宽 | navhard EPDMS s0 / s1 / s2 |
|:--|:--|:--|:--|:--|:--|
| Cinque `cls_late` | 77.9 / 77.4 / 77.8 | 0.48 / 0.61 | 77.4 / 76.9 / 77.3 | 0.46 / 0.61 | 19.8 / 19.5 / 19.0 |
| Lebowski `cls_late` | 77.3 / 77.2 / 76.8 | 0.47 / 0.63 | 76.7 / 76.5 / 76.3 | 0.39 / 0.64 | 17.5 / 19.0 / 18.7 |
| Cinque `ridge_late` | 73.5 / 73.7 / 73.5 | 0.13 / 0.66 | 73.9 / 74.0 / 73.9 | 0.15 / 0.69 | 16.8 / 16.5 / 16.8 |
| Lebowski `ridge_late` | 72.4 / 72.4 / 72.4 | 0.02 / 0.66 | 72.9 / 72.9 / 72.9 | 0.01 / 0.69 | 17.0 / 17.0 / 17.0 |
| `cls ego` | 68.4 / 67.7 / 68.4 | **0.77 / 0.71** | 67.8 / 67.2 / 67.6 | 0.68 / 0.76 | 13.6 / 14.3 / 13.3 |
| Cinque `cls_late` − `cls ego` | +9.4 / +9.7 / +9.5 | 0.29 / 0.63 | +9.6 / +9.8 / +9.7 | 0.23 / 0.64 | |
| Lebowski `cls_late` − `cls ego` | +8.8 / +9.5 / +8.4 | **1.10 / 0.62** | +8.8 / +9.3 / +8.6 | **0.72 / 0.61** | |
| `ridge_late` − `ridge ego`（两模型） | +10.9 / +11.0 / +10.9；+9.8 / +9.8 / +9.8 | ≤ 0.13 / 0.69 | +9.6 / +9.8 / +9.7；+8.7 / +8.7 / +8.7 | ≤ 0.15 / 0.72 | |

s0′（seed 0 重跑，分类头的 run-to-run 噪声）：Cinque / Lebowski `cls_late` PDMS 77.8 / 77.3，`cls ego` 68.3，与 s0 差 ≤ 0.12。`ridge ego` 三个 seed 的分数逐位相同（三个 seed 选到同一个 λ，闭式解与分折无关）。

**WOD 第 40 条 (iii)（seed = k-means 词表 + 内层划分）**，479 个 rater 帧：

| 行 | s0 / s1 / s2（s0′） | 均值 | 极差 | 半宽 | 极差 < 半宽 |
|:--|:--|--:|--:|--:|:--|
| RFS cluster mean，Cinque `cls_late` | 7.64 / 7.50 / 7.50（7.65） | 7.55 | 0.14 | — | — |
| RFS cluster mean，Lebowski `cls_late` | 7.73 / 7.46 / 7.57（7.68） | 7.59 | 0.27 | — | — |
| RFS 配对，Cinque `cls_late` − `cls ego` | +0.38 / +0.19 / +0.29（+0.39） | +0.29 | 0.18 | 0.17 | **否** |
| RFS 配对，Lebowski `cls_late` − `cls ego` | +0.43 / +0.17 / +0.34（+0.42） | +0.31 | 0.26 | 0.16 | **否** |
| RFS 配对，`cls ego` − `ridge ego` | +0.22 / +0.23 / +0.18 | +0.21 | 0.06 | 0.18 | 是 |
| RFS 配对，`ridge_late` − `ridge ego`（两模型） | +0.40 / +0.40 / +0.40；+0.42 ×3 | | 0 | 0.18 | 是（确定性） |
| ADE 第 1–9 档 pre_onset Δ vs `ridge ego`，`ridge_late`（两模型） | −0.294 ×3；−0.318 ×3 | | 0 | 0.13 | 是（确定性） |
| 同上，`cls_late`（两模型） | +0.21 / +0.05 / +0.20；+0.18 / +0.06 / +0.22 | | 0.16 | 0.29–0.32 | 是 |

**Q9b 网格 attention head（seed = `_train` seed）**，P5 v1 BA 行人翻转 %：

| arm | Cinque s0 / s1 / s2 | 极差 / 半宽 | Lebowski s0 / s1 / s2 | 极差 / 半宽 |
|:--|:--|:--|:--|:--|
| 配对（原 run 已有） | 46.1 / 52.2 / 45.3 | 6.9 / 8.2 | 37.2 / 50.5 / 48.5 | **13.3 / 10.9** |
| hard-example（新跑 s1、s2） | 3.4 / 2.5 / 0.0 | 3.4 / 2.9 | 0.2 / 2.5 / 0.7 | 2.2 / 0.4 |

hard 重跑时同一次 fit 顺带重算的配对 arm s0–s2 与原 run 逐项相同（行人翻转到小数点后 4 位）。PDM 集上两种 arm 所有 seed ≤ 4.5%。

**读法**：

1. **主结论全部不随 seed 变**：M-C 配对双流在三个 seed 上都过判据（行人 42–44% / 39–42%，极差只有原 CI 半宽的 1/6–1/3），hard-example 与均匀对照在三个 seed 上都是行人 ≈ 0；
   NAVSIM 上 `temporal` 对同族 ego 的 +8.4 到 +11 分在三个 seed 上都远离零；`ridge_late` 类 head 基本确定（极差 ≤ 0.15 分）。「全部单 seed」这条限定对这些读数可以去掉。
2. **两处超过原 CI 半宽，都在分类头**：WOD 上 `cls_late` − `cls ego` 的 RFS 增益 3 seed 均值 +0.29 / +0.31，seed 1 只有 +0.19 / +0.17（s0 的 +0.38 / +0.43 与它的重跑 s0′ 一致，所以不是 run-to-run 噪声，是词表 / 内层划分的 seed 效应）；
   第 40 条里「Lebowski `cls_late` 7.73 够到原生 7.89」这一格依赖 seed 0，3 seed 均值 7.59，要改成「补上缺口的一部分，是否够到原生随 seed 变」（推测，要白天按第 40 条的判据重算 3 seed 均值）。
   NAVSIM 上 Lebowski `cls_late` − `cls ego` 极差 1.1 PDMS，超过半宽 0.62，但三个 seed 都在 +8.4 以上，结论不变。
3. Q9b 配对 arm 的 seed 离散大（Lebowski 37 → 51%），原先只读 seed 0 偏保守；hard 对照三个 seed 都 ≤ 3.4%。
4. navhard EPDMS（无 CI，只是 aggregate）的 seed 极差 0.3–1.6 分，和 navhard 上各行之间的差同量级，navhard 的单 seed 行间比较要谨慎。
