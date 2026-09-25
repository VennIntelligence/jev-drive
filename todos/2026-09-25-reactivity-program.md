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

## 执行记录

执行者：reactivity agent（2026-09-25 15:50 起）。I1、I3、I4 由三个子执行代理做，各写子文档
[i1-p5v1](2026-09-25-reactivity-program/i1-p5v1.md)、[i3-hugsim-pairs](2026-09-25-reactivity-program/i3-hugsim-pairs.md)、
[i4-worldmodel](2026-09-25-reactivity-program/i4-worldmodel.md)，结论由本文件汇总。M-A 等用户决定，不启动训练。

### 偏离与澄清日志（每条写于看到受影响的数字之前）

1. **D0 的 tap 口径（2026-09-25 15:55，抽取之前）**。表里写「`driving_vision` 输出」，而 Cinque / Lebowski 是单个合并的 ONNX，没有独立的
   `driving_vision.onnx`。按第 40 条 tap 表（[driving-backbones](2026-09-24-driving-backbones/README.md)）定：**主 tap 是 `vision`**
   （进时间模块之前的当前步 vision encoder pooled 输出：Cinque `mean` 512 维、Lebowski `view_40` 3072 维）；**次要 tap 是 `hidden`**
   （进 feature 队列的向量：Cinque 16 384 维 = 32 个未池化 vision token，Lebowski 512 维）。判定只用主 tap；`hidden` 只描述，
   若它过线而 `vision` 不过，记为「次要证据，需复现」，不改判定。
2. **D0 的「行人 family」（同上）**。P5 v0 里 hazard actor 是行人的 family 有四个：DynamicObjectCrossing、ParkingCrossingPedestrian、
   PedestrianCrossing、VehicleTurningRoutePedestrian。判据作用在**四个合并的行人集合**上（x⁺ 对 x⁻ 的观测帧，按路线 bootstrap，
   500 次，与 `probe_auc_paired` 同一套重采样）；逐 family 的 AUC 与配对 Δ 只描述。判定按模型分开：某个模型的 `vision` 在行人集合上
   AUC ≥ 0.60 且对同模型 `temporal` 的配对 Δ 的 95% CI 下端 > 0，就算「信息在该模型的 vision 里」；两个模型中至少一个过线 → M-A 单独可行
   （用过线的那个模型）；都不过 → M-A + M-C。
3. **D0 的 probe 与抽取（同上）**。probe、fold、λ 选择、训练行一字不改（`p5_exam.probes`），只加 examinee；抽取器、stream、渲染与实验 1 相同，
   只多存 `vision` / `hidden` 两个数组。新抽取的 `temporal` 与实验 1 已存的逐位比较，作为等价性检查。同一次 `p5_exam run` 也会给出这些 tap
   上 `ridge_late` 的翻转率，只描述，不进 D0 判定。
4. **M-C 的具体化（2026-09-25 16:10，写于任何 M-C 拟合之前，也早于 D0 的数字）**。表里只给了结构和判据，下面把会影响数字的选择全部定死：
   - **结构**：`pred(x) = prior(x) + Δ(x)`。prior 是实验 1 的 examinee 本身，即 openpilot `temporal` 上的 `ridge_late`
     （`p5_exam.heads` 原样，按路线 5 折，只在 role = train 的帧上拟合，所以对 pair 帧是样本外）；Δ(x) = W·z(x) + b，**线性**（thin，闭式解）。
     z 是 reaction 输入：Qwen `L18_last`（P3(d″) 原生视频 token，已抽）⊕ 同一模型的 `temporal`，每列按训练行标准化，每个 stream 再乘 1/√d_stream，
     使两路总方差相等（不让 2560 维的 Qwen 靠维数压过 512 维的 openpilot）。输出是整条 20 × 2 未来轨迹的修正，考试照旧读 2 s 速度。
     主 arm 用 Cinque，Lebowski 是复现。
   - **配对差分 arm（主）**：外层 fold f 内，训练行是 fold ≠ f 的全部 pair 帧（obs 的 x⁺/x⁻，含 non-reactive 帧，以及 null 的 x⁺/x_null），
     损失 Σ‖W(z⁺ − z⁻) − [(y⁺ − y⁻) − (p⁺ − p⁻)]‖²，y 是 expert 两侧各自的未来轨迹，p 是 prior；null 对的 expert 差≈0，所以「null 上约束为零」由它们自带。
     另加「直行帧上修正项为零」：fold ≠ f 的 role = train 帧上 μ‖W(z − z̄)‖²，μ 取使两项总权重相等（μ = n_pair / n_train），固定不调；
     μ = 0 只作敏感性描述。ridge λ 在训练 fold 内按路线分组 3 折选（网格 10^[−1..5]，最小化留出路线上的配对 MSE）。
   - **hard-example 重加权对照**：同一 z、同一 prior、同样的训练帧（fold ≠ f 的 role = train 帧，加上训练 fold 的 pair 帧，各当普通单帧），
     逐帧拟合残差 y − p 的加权 ridge，权重 w = s_ego / mean(s_ego)，s_ego = 同一批训练行上 `ridge ego` 的逐帧 ADE（Keyframe-Focused IL 2106.06452 的做法）；
     λ 同样按路线 3 折选（加权 MSE）。同时报 w ≡ 1 的均匀 imitation 一行作参照（只描述）。
   - **单流对照**：配对差分 arm，z 只用 Qwen `L18_last`，或只用 openpilot `temporal`。
   - **考试**：`p5_exam.exam` 一字不改（τ_model 由各自 null 定，定向翻转率、样本外 null false-flip、路线 bootstrap CI）。
   - **判据**（表里的三条写成数）：(a) 行人：四个行人 family 的 reactive 帧合并（v0 实际上只有 DynamicObjectCrossing 45、ParkingCrossingPedestrian 76 有量），
     翻转率 ≥ 20% 且路线 bootstrap CI 下端 > 该 arm 的样本外 null false-flip；(b) cut-in 不掉：三个 cut-in family 合并的翻转率对 prior 的逐帧配对差，
     路线 bootstrap 95% CI 上端 ≥ 0（没有显著变差）；(c) null false-flip 不动：样本外 null false-flip ≤ 7%。三条都过，这个 arm 判「过」。
     主问题是配对差分 arm 过、而 hard-example 对照不过（第 21 条 D1 的那一格）；两者都过，则配对结构不是必要的，照实写。
5. **M-C 的 λ 网格碰边（2026-09-25 16:15，写于看到 v0 smoke 的翻转数之后，所以只算事后敏感性，不改判定）**。预登记网格 10^[−1..5]：
   配对差分 arm 在 10 个 fold × 模型里多数选到下边 0.1，hard / uniform 选到上边 1e5。选 λ 只看内层留出路线上的 MSE，与翻转数无关，
   但放宽网格是看到结果之后才决定的，因此：主判定仍用预登记网格那次 run；另跑一次 10^[−5..7] 作敏感性，并加一个只描述的诊断——
   训练 fold 内配对目标（2 s 速度差，|Δ_expert| > 0.5 m/s 的对）上拟合值对 expert 值的斜率，按行人 / cut-in 分组，用来区分
   「训练集上都拟合不了」（表征）和「拟合得了但跨路线不泛化」（数据量）。
6. **D0 exam 里 Cinque `hidden` 只做 probe，不拟合 `ridge_late`（2026-09-25 16:12，D0 的任何 probe 数字出来之前）**。16 384 维的 ridge 每个 fold
   要 5 次 16 384² 的 CPU eigh，实测第一个 fold 卡了 9 分钟以上，估计全程约 6 h，超预算远不止 2 倍。`ridge_late` 翻转率在 D0 里本来只描述，
   D0 判定用的是 probe，所以这一个 tap 去掉 head、保留 probe；其余 tap 不变。同时按 main 的排程把 D0 exam 与 M-C 敏感性挪到 GPU 1（与 I4 共卡），
   我方 CPU 收到 196–201。
