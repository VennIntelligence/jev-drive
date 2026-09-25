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

## 结果

（按到达顺序记在 `tmp/2026-09-26-overnight.md`；早上汇总后把判格写回各自的 todo。）
