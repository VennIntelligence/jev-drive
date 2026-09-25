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

## 结果

（按到达顺序记在 `tmp/2026-09-26-overnight.md`；早上汇总后把判格写回各自的 todo。）
