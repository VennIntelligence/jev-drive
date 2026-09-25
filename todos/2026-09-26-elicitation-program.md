# 激发计划：冻结组件 + 配对差分，从 CARLA 走到真实开环数据（E1–E5）

状态: 待排（预登记，2026-09-26，写于任何新拟合、新挖掘之前）
上游: 第 25 条（continuation prior + reaction decoder）、第 42 条（M-C 在 P5 v1 上行人 43%）、第 43 条（融合诊断，2026-09-26 就地修正）、
[reactivity 计划](2026-09-25-reactivity-program.md)、[融合前诊断](2026-09-25-fusion-diagnostics.md)、[fast-perception](2026-09-26-fast-perception.md)
主线: 不拼接、不重训 backbone。组件冻结（ego 历史、openpilot `temporal` + lead 头、Qwen `L18_last` pooled、快检测器 embedding），
能力靠**配对差分**激发：head 只在 x⁺ / x⁻ 的差上有标签，直行帧与 null 上约束为零；对照永远带均匀 imitation 与 hard-example 重加权。评测只看 reactivity（P5 口径）与 WOD 第 22 条口径。

## 已经知道的（本计划的起点）

| 事实 | 数字 | 出处 |
|:--|:--|:--|
| 同一批 Qwen ⊕ openpilot 特征、同一批 P5 v1 行人帧，三种训练信号 | 均匀拼接 0.2–0.7%；hard-example 0%；**配对差分 43.3% [35.0, 50.7]** | Q1-v1、M-C v1 |
| 激发需要的数据量 | v0 134 个行人 reactive 帧 → 0；v1 406 个 → 43% | 第 42 条修正 |
| 各组件里能提取的 | ego：gate；openpilot：车辆纵向反应、20 m 内 lead 召回 0.9；Qwen pooled：行人存在性；openpilot vision 层无行人（AUC 0.51） | 第 40、42 条，Q4c |
| CARLA ↔ Waymo 特征域差 | Waymo 训的 head 读不出 CARLA 特征，domain AUC 1.0 | P4（第 32 条前置） |
| PDM-Lite 集 | 所有考生逐帧翻转 ≈ 0，GT 规则门 4.2%：expert 在相机可见前 0.4 s 已反应 | reactivity todo I1 结果 |
| 快通道时间窗 | ParkingCrossingPedestrian p25 0.53 s；Qwen 流 200–300 ms / 帧、SAM 3 路 558 ms 都来不及 | Q8 |

## 五项

### E1. M-C 的 head 零样本套到 WOD（迁移检查，其余各项的前置）

- **问题**：在 CARLA 配对上激发出来的 reaction head，放到真实特征上还动不动。
- **做法**：取第 42 条 v1 BA 集上过判据的 M-C 双流 head（Cinque 与 Lebowski 各一，预登记网格里选出的那一个，**不重训、不调**），
  prior 换成 WOD train 训的 `ridge_late` openpilot（第 40 条 (iii) 那一个），Δ 加在 prior 上；在 WOD 完整 val 上评。
  特征：WOD 的 `op_*` `temporal` 与 `qwenvid L18_last` 已在盘上（第 40 条与 P3e 抽取），做标准化时用 CARLA 训练行的统计量（head 自带），并列一版用 WOD 行的统计量（描述）。
- **读数**：按 cluster（Pedestrians、Cyclists、Cut_ins、FOD、Intersections、全部）报 (i) 加 Δ 后 − prior 的 RFS 配对 Δ 与 ADE 第 1–9 档 Δ（第 22 条口径，sequence bootstrap）；
  (ii) Δ 的激活率：|Δ| 超过该 head 在 P5 null 上定的 τ 的帧占比，在 straight_yaw 帧对 pre-onset / Pedestrians 帧上分别报（激活该集中在后者）。
- **判据（写死）**：
  - **转移得过去**：Pedestrians 或 Cyclists 上 RFS Δ 的 CI 整体 > 0，且 straight_yaw 帧上激活率 ≤ P5 null false-flip（约 5–7%）、全部 rater 帧 RFS Δ 的 CI 不整体 < 0。
  - **转不过去但无害**：所有 Δ 跨零、激活率在直行帧 ≤ 7%。
  - **有害**：全部 rater 帧 RFS Δ CI 整体 < 0，或直行帧激活率 > 7%。
- **预期**：转不过去但无害（P4 的域差）。若转移得过去，E3 的挖掘对可以只做标签复核；否则 E2/E3 是必需。
- **资源**：CPU，特征在盘上，< 1 h；工程半天（head 的加载与 prior 对接）。

### E2. 真实帧的反事实编辑对（SAM 的位置：离线造对，不进回路）

- **问题**：能不能在 WOD / nuScenes 帧上批量造 x⁻，让配对差分在真实特征上训练。
- **做法**：只做「删除」不做「插入」。用 SAM 3.1（已在 `envs/sam3`）按 pedestrian / cyclist prompt 出 mask，视频一致 inpainting（先用仓库原样的 LaMa 或 ProPainter 之一，登记时定一个），
  x⁺ = 原帧，x⁻ = 抹掉该 actor 的帧；只取 actor 在走廊内、≤ 30 m、像素 ≥ 20 的帧（Q2b 的走廊定义）。**安慰剂对照**（必带）：同样大小的 patch 贴到空路面。
  标签：无 expert，两种并列，各自单独报：(a) log 自己的未来当 x⁺ 侧、x⁻ 侧标签 = CTRA 外推（继续）；(b) 规则裁判（Q6 的三条门）。
- **验证（造对之前先过）**：16 对目检；独立检测器（YOLO26 COCO person）在 x⁻ 上的残留检出率 ≤ 10%（2609.22582 报 8.5%）；安慰剂对上 Qwen 与 openpilot 特征的位移中位数当噪声地板。
- **读数**：在 E1 的 head 上，x⁺ − x⁻ 的 Δ 幅值对安慰剂对的比；用编辑对**训练**的 M-C 在 P5 v1 BA 集上的行人翻转（真实 → 仿真方向的迁移）与 WOD Pedestrians 的 RFS Δ（按 sequence 分折）。
- **判据**：编辑对上 |Δ| 中位数 ≥ 2 × 安慰剂对，且用编辑对训的 head 在 WOD Pedestrians 上 RFS Δ CI > 0 而直行帧激活率 ≤ 7%。
- **资源**：SAM 已量 188 ms / 张单路；WOD 走廊内行人帧约 160（Q2b）太少，扩到 train 全量前视 41.5 万帧里走廊内有行人 / cyclist 的帧（估 1–3%，4–12k 张）：SAM 约 0.5–1 h GPU，inpainting 约 1–3 h GPU；head 训练分钟级。约 3–5 GPU·h，工程 1–2 天。

### E3. 从 log 里挖孪生帧（新增，真实 expert、不编辑图像）

- **问题**：WOD train 52 万帧里，有没有足够多的「ego 历史匹配、未来分叉」的帧对，能当配对差分的标签。
- **做法**：每帧的 ego 历史向量（16 步 × 位置 / 速度 / 加速度，第 40 条 ego 特征）+ intent one-hot；在同 intent 内做最近邻（标准化后 L2，容差 τ_ego 在 null 分布上定：同一 sequence 相邻 0.25 s 的帧对之间的距离 p95），
  排除同一 sequence 与同一路段（用 sequence id 与 GPS 近邻）；未来分叉 = 5 s 未来的纵向速度差 ≥ 2 m/s 或横向偏移差 ≥ 1 m（与 P5 τ_exp 的量级对齐，登记时定死）。
  分叉的一侧当 x⁺（反应）、另一侧当 x⁻（继续）；方向由未来自己给出。**噪声对照**：ego 匹配但未来也匹配的对（孪生 null），它们的特征差分定 τ。
- **可行性统计（本项的第一交付，纯 CPU）**：匹配对数按 τ_ego 与分叉阈值的网格；分叉对里 x⁺ 侧走廊内有 SAM 标出的行人 / 车辆的比例（用 Q2b 已有的 WOD 前视 SAM 检测，20 237 帧；不够再补）；
  分叉对与 pre-onset、Pedestrians、Cut_ins cluster 的重叠。
- **判据**：分叉对 ≥ 1 000 且其中走廊内有可见原因物体的比例显著高于孪生 null 对（CI 不跨零）→ 开训练；否则记「log 里挖不出足够的配对」，只留 E2。
- **训练与读数（可行性过了才做）**：M-C 同一 head，训练行 = 孪生对，按 sequence 分折；WOD Pedestrians / Cut_ins 的 RFS Δ、pre-onset ADE Δ；P5 v1 BA 行人翻转（真实 → 仿真）。对照：均匀 imitation、hard-example。
- **文献核查（写入结果之前）**：按 ego 状态匹配从 log 里挖配对做差分监督有没有先例（关键词 matched pairs / twin frames / counterfactual mining from logs / state-matched contrastive imitation）；有就引，claim 相应缩。
- **资源**：最近邻 52 万 × 100 维 CPU 小时级；SAM 若要补帧按 E2 的量估；head 分钟级。工程 1 天。

### E4. PDM-Lite 集的计分窗口（让 v1 主判定集可用）

- **问题**：PDM-Lite 在因素可见前 0.4 s 已反应，逐帧口径下所有考生 ≈ 0。
- **做法（预登记两种，主判用 (a)）**：(a) **可见门控**：reactive 帧只保留 t ≥ t_vis + L 的帧，L = 考生的感知 + 决策延迟（openpilot 0.1 s、Qwen 流 0.3 s、SAM 0.6 s，按 Q8 与 baselines 实测），并要求该帧的 Δ_expert 仍 > τ_exp；
  (b) **按对计分**：窗口内任意 reactive 帧翻对即算这对过（第 32 条已用过的口径）。两种都报，null false-flip 同样按窗口重算。
- **读数**：所有现有考生（TFv6 两通道、ridge ego、openpilot、Qwen、M-C 各 arm、Q6 规则门）在 PDM-Lite 集上按 (a)(b) 重报；与 BA 集并排。
- **判据**：(a) 下 M-C 双流行人翻转与 BA 集的差在 ±15 pp 内且 null ≤ 7% → PDM-Lite 集成为主判定集；否则继续以 BA 为主、PDM-Lite 只报 (b)。
- **资源**：CPU 分钟级（预测都存着），工程半天。

### E5. 20 Hz student：把 M-C 蒸馏到快通道（等 fast-perception 出延迟后登记细节）

- **问题**：M-C 的 Qwen 流 200–300 ms / 帧，ParkingCrossingPedestrian 一类来不及。
- **做法**：teacher = M-C 双流 head 的 Δ；student 输入 = openpilot `temporal` ⊕ 快检测器 embedding（fast-perception 里延迟 ≤ 20 ms 且 P5 行人 ≤ 20 m 召回 ≥ 0.8 的那个，候选 YOLOE-26 / EfficientSAM3；embedding = 走廊内前 k 个检测的 (类别, x, y, 尺寸, score) 经一个小 MLP，k = 8）；
  标签 = 同一批 P5 v1 配对差分（不是 teacher 输出）+ teacher 的 Δ 当辅助目标，两者各一 arm。对照：openpilot 单流配对差分（第 42 条 7–18%）。
- **读数**：P5 v1 BA 行人翻转、cut-in Δ、null false-flip、非反应帧误刹；端到端延迟（检测 + head）。
- **判据**：行人 ≥ 30% 且 CI 下端 > null、cut-in 不掉、延迟 ≤ 50 ms。
- **资源**：检测器在 P5 观测帧上跑一遍（YOLOE-26x 640 估 < 10 ms / 张，29.8k 张约 5 min）；head 分钟级。工程 1 天。**登记条件**：fast-perception 的延迟 × 召回表出来后再补检测器名与数字，之后才算预登记完成。

## 顺序与依赖

```
E4（CPU，半天）──────────────────────────────► v1 主判定集可用，E2/E3/E5 的仿真侧读数都用它
E1（CPU，半天）──转移得过去──► E2/E3 只做标签复核
      └────────转不过去──────► E2 与 E3 并行（E3 先出可行性统计，纯 CPU）
fast-perception 延迟表 ───────► E5 登记完成 → 跑
```

## 硬约束

- 组件全部冻结；head 只有配对差分、均匀 imitation、hard-example 三种训练信号，每项都三者并列。
- judge 一字不改：WOD 第 22 条口径，P5 v0 / v1 的定向翻转率 + 样本外 null false-flip；E4 新增的窗口口径先登记后使用。
- 训练 / 评估按 sequence（WOD）或路线（P5）分开；E2 / E3 的对不得与评估 sequence 同源。
- 闭环仍暂停；本计划全部开环。
- 任何一步超估计 2 倍先停下报。结论：E1–E3 进 decisions.md 新条目「配对差分从 CARLA 到真实数据」；E4 补进第 42 条；E5 进第 42 条或新条目。

## 交付

1. E1：一张 cluster × {RFS Δ, ADE Δ, 激活率} 的表 + 判格。
2. E2：编辑对样例 16 组图、验证表（残留检出、安慰剂地板）、训练后的 WOD / P5 两向读数。
3. E3：可行性统计表（对数 × 阈值网格、原因物体比例、cluster 重叠）；过线则加训练读数；文献核查一段。
4. E4：PDM-Lite 集在两种窗口下的全考生表，与 BA 并排。
5. E5：student 的翻转 / 延迟表。
小表进 `research/results/elicitation/`。

## 偏离与澄清日志

（执行时追加，时间戳早于受影响的数字。）

## 结果

（待填。）
