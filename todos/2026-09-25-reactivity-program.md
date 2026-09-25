# reactivity 计划：决定性检查、方法实验、仪器加固（排 GPU 用）

状态: 待排（预登记，写于任何新抽取之前，2026-09-25）
上游: [op-temporal-p5-and-route](2026-09-25-openpilot-temporal-p5-and-route.md)（实验 1、2a 已出，2b 在跑）、第 25 条（continuation prior + reaction decoder）、第 32 条（P5 v0）、第 40 条
主线: openpilot 冻结 vision；闭环考试仍暂停（memory：infra 验收未过）；本文件全部是开环或离线渲染。
不重复做的（别人已做，只当动机引用）：真实帧抹行人的 model-agnostic 考试（2609.22582，6 个 ckpt，1.9%）、meta-action 考试（CoLT-Drive，11 个）、闭环 shift（Fail2Drive，7 个）。
所以**不再**把更多公开 planner 拉过 P5 当 audit。

## 依赖图

```
D0 vision 层行人 probe ──┬─ 结果决定 M-A 单独够不够 ──► M-A（冻结 vision，重训 policy）
                         └─ vision 里没有行人 ─────────► M-C（双流 reaction head）
2b（在跑）──────────────────────────────────────────────► M-A 的 desire 输入用不用
I1 P5 加固（第二 expert、扩路线）── 独立，先排 CPU/CARLA ─► M-A / M-C 的考卷与 Δ 标签
I2 continuation share 表 ── 独立，CPU
I3 HUGSIM 3DGS 开环配对 ── 独立，GPU 渲染
I4 worldmodel-4B 是否 action-conditioned ── 独立，调研 + 一次前向
```

## 决定性检查

| # | 问题 | 做法 | 判据 | 资源 | 预算 |
|---|---|---|---|---|---|
| D0 | 行人信息是 vision 就没有，还是 policy 丢的 | 在 P5 那批 stream 上把 tap 从 `temporal` 换成 `driving_vision` 输出（`OPModel(taps=[...])` 已支持任意 ONNX tensor），Cinque + Lebowski；同一 hazard probe（按路线 5 折），按 family 报 x⁺/x⁻ AUC，与 `temporal`（行人 0.50–0.53）和 Qwen `L18_last`（0.62–0.88）并排 | 行人 family 上 vision AUC 对 `temporal` 的配对 Δ 的 CI 整体 > 0 且 AUC ≥ 0.60 → 信息在 vision 里，M-A 单独可行；否则 M-A + M-C | 1 卡 < 10 GB，12 核 | 抽取 24 min（同上一轮）+ probe 分钟级；工程半天 |
| 2b | intent → desire 进 backbone 有没有用 | 在跑，见上游 todo | 原生 plan 在 Intersections 上的 Δ；全集主表不变差 | 已占 GPU 2 | 已在预算内 |

## 方法实验（文章的「打法」）

共同规则：judge 一字不改（第 22 条口径 + P5 定向翻转率 / null false-flip）；训练与考试按路线或 sequence 分开；每个 arm 必带 **hard-example reweighting 对照**（第 21 条 D1，survey 标「没人做过」的那一格）；先在 P5 v0 的 165 对上做 smoke，I1 出来后再上加固版。

| # | 内容 | 训练信号 | 数据 | 判据 | 资源 | 预算 |
|---|---|---|---|---|---|---|
| M-A | 冻结 `driving_vision`，重训 temporal policy（结构照 comma：冻结特征上的小 temporal transformer；输出照原生 plan 头，不改 judge） | (i) WOD imitation（52 万帧）；(ii) + CARLA 配对差分 loss：Δ 只在 x⁺/x⁻ 之间有标签，null 上约束为零；(iii) 对照：(i) + 按 s_ego 的 hard-example 重加权 | WOD train（vision 特征需重抽，`temporal` 不够）、P5 pair 观测帧、nuScenes 作 held-out | 主：P5 按 family 翻转率对冻结 `ridge_late`（cut-in 86–89%、行人 0%）；副：WOD RFS 对原生 8.005 / 7.886 与 `cls_late` 7.73；静止帧对 cv；null false-flip 不高于 5–7% | 抽 vision 特征：1 卡，52 万帧 × 2 模型约 70 min；训练：1 卡 < 20 GB，每 arm 估 1–3 h（policy 几十 M 到几百 M 参数，特征常驻 RAM） | 抽取 1.5 h + 3 arm × 2 模型 ≈ 6–18 h GPU；工程 2 天（policy 结构复刻、loss、数据管线） |
| M-C | 双流 reaction head：openpilot 冻结当 continuation + 第 2 层；reaction 输入 = 带行人信息的特征（Qwen `L18_last` / 原生 video token，已抽）⊕ openpilot `temporal`；输出对 prior 的修正 Δ，plan 层相加 | 配对差分（同 M-A (ii)）；对照 hard-example 重加权；对照单流（只 Qwen、只 openpilot） | P5 观测帧 + P4 训练帧，特征全在 `processed/carla_p5/features` 与 op-p5 | 行人 family 翻转率离开 0（≥ 20% 且 CI 下端 > null false-flip）而 cut-in 不掉、null false-flip 不动 | 1 卡 < 10 GB，或 CPU | 每 arm 分钟级；工程 1 天 |

M-A 只在 D0 判「信息在 vision 里」时是完整方案；否则 M-A 负责静止 / 路口 / cut-in 强度，M-C 负责行人，两者一起是第 25 条的结构。

## 仪器加固（文章的「尺子」）

| # | 内容 | 为什么 | 资源 | 预算 |
|---|---|---|---|---|
| I1 | P5 v1：第二个 expert（PDM-Lite，需作者 scenario_runner fork）；路线 25 → ≥ 100 条；3 TM seed；两个行人 family 与 Light 进合并 | v0 合并 CI [32, 60]、行人 reactive 帧只有 45 + 76；单 expert（BehaviorAgent）不提前减速 | CARLA：8 实例并发，每实例 CPU 约 8 核、VRAM 7–9 GB；不需要模型 | 生成约 385 × 4 run，每 run 3–5 min → 8 并发约 8–10 h；先 2 对端到端 profiling |
| I2 | continuation share 表：同一 ego-only / cv / ctrv 基线跑过 nuScenes、WOD-E2E、NAVSIM、B2D（开环 shadow）、HUGSIM（已有 cv），报占顶分比例 | 文章第一张表；数字多数已有，缺统一口径 | CPU | 1 天整理，NAVSIM 基线若缺则 devkit 跑 < 1 h |
| I3 | 真实外观配对：HUGSIM 3DGS 沿 logged 轨迹渲染有 / 无 hazard actor 两版（attack planner 生成 actor），开环，不跑闭环控制器；标签用 expert 在 3DGS 场景几何上两侧重跑或 CV 规则 | 回应 2609.22582 的「编辑真实帧」：几何一致的重建替代 inpainting；survey 标「没人做过」 | 1 卡，渲染 2.4–6 GB / 实例；场景已在盘上 | 先 5 个场景验证 actor 插入与遮挡正确（1 天工程），再 50 场景 × 2 版渲染约 2–3 h |
| I4 | `commaai/worldmodel-4B` 是否接 action / plan 条件 | 若是，M-A 有 on-policy 环境，绕开 CARLA 域差 | 1 卡加载一次 | 半天调研 |

## 排程建议

1. 今天：D0（GPU 2 与 2b 共卡，< 10 GB）、I2（CPU）、I4（任一卡一次前向）。
2. D0 出来后：M-C 立刻上（特征都在盘上），M-A 先抽 vision 特征（1 卡 70 min），policy 复刻工程并行。
3. I1 与 I3 独立，CPU/CARLA 与渲染卡各占一路，与上面不抢。
4. 任何一步超预算 2 倍先停下来报；结论写回 decisions.md（D0、M-* 进第 40 条下面或新开第 42 条；I1 进第 32 条）。
