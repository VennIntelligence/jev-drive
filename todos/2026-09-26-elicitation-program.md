# 激发计划：冻结组件 + 配对差分，从 CARLA 走到真实开环数据（E1–E5）

状态: 待排（预登记，2026-09-26，写于任何新拟合、新挖掘之前；2026-09-26 按 openpilot 开环对比最终结果修订：E1 加 NAVSIM 列，E3 主数据集改 navtrain，新增 E6 可选）
上游: 第 25 条（continuation prior + reaction decoder）、第 42 条（M-C 在 P5 v1 上行人 43%）、第 43 条（融合诊断，2026-09-26 就地修正）、
[reactivity 计划](2026-09-25-reactivity-program.md)、[融合前诊断](2026-09-25-fusion-diagnostics.md)、[fast-perception](2026-09-26-fast-perception.md)、
[openpilot 开环对比](2026-09-25-openpilot-openloop-comparison.md)（第 37 条修正：冻结 `temporal` + 薄 head 在 WOD / nuScenes / NAVSIM 三个数据集上都超过 ego-only）
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
| 冻结 `temporal` 跨数据集成立 | NAVSIM navtest 全量 12 146 token：Cinque `temporal` + 薄 head PDMS 77.9 / EPDMS 77.4，ego-only 68.4 / 67.8，原生 plan 52.1 / 46.2（2 Hz 协议所致），TransFuser 文献 84.0 / 76.7；单 seed，EPDMS 两边 devkit 版本不同只能说同量级 | 开环对比最终报告，第 37 条 |
| 特征比原生 plan 头对输入协议更鲁棒 | 同一份 2 Hz 输入，原生 plan 掉到 cv 以下，`temporal` 经 head 仍比 ego 高 9.6 EPDMS | 同上 |
| NAVSIM / nuPlan 的附加条件 | 全部 agent 有 GT 轨迹（原因物体不经感知可查）；自带 PDM scorer 可对任意 proposal 打分；scorer 是可调优的代理（第 35 条），只能当标签之一 | 第 35、37 条 |

## 五项

### E1. M-C 的 head 零样本套到 WOD（迁移检查，其余各项的前置）

- **问题**：在 CARLA 配对上激发出来的 reaction head，放到真实特征上还动不动。
- **做法**：取第 42 条 v1 BA 集上过判据的 M-C 双流 head（Cinque 与 Lebowski 各一，预登记网格里选出的那一个，**不重训、不调**），
  prior 换成 WOD train 训的 `ridge_late` openpilot（第 40 条 (iii) 那一个），Δ 加在 prior 上；在 WOD 完整 val 上评。
  特征：WOD 的 `op_*` `temporal` 与 `qwenvid L18_last` 已在盘上（第 40 条与 P3e 抽取），做标准化时用 CARLA 训练行的统计量（head 自带），并列一版用 WOD 行的统计量（描述）。
- **NAVSIM 列（新增）**：同一个 head 套到 navtest，prior 换成开环对比里 navtrain 训的 `temporal` 薄 head；用官方 devkit 报 PDMS / EPDMS 的配对 Δ，并按 navhard 与「走廊内 ≤ 30 m 有行人 / cyclist 的 token」（GT 轨迹判定，不经感知）分组；直行帧激活率同样报。
- **读数**：按 cluster（Pedestrians、Cyclists、Cut_ins、FOD、Intersections、全部）报 (i) 加 Δ 后 − prior 的 RFS 配对 Δ 与 ADE 第 1–9 档 Δ（第 22 条口径，sequence bootstrap）；
  (ii) Δ 的激活率：|Δ| 超过该 head 在 P5 null 上定的 τ 的帧占比，在 straight_yaw 帧对 pre-onset / Pedestrians 帧上分别报（激活该集中在后者）。
- **判据（写死）**：
  - **转移得过去**：Pedestrians 或 Cyclists 上 RFS Δ 的 CI 整体 > 0，且 straight_yaw 帧上激活率 ≤ P5 null false-flip（约 5–7%）、全部 rater 帧 RFS Δ 的 CI 不整体 < 0。
  - **转不过去但无害**：所有 Δ 跨零、激活率在直行帧 ≤ 7%。
  - **有害**：全部 rater 帧 RFS Δ CI 整体 < 0，或直行帧激活率 > 7%。
- **预期**：转不过去但无害（P4 的域差）。若转移得过去，E3 的挖掘对可以只做标签复核；否则 E2/E3 是必需。
- **资源**：CPU，特征在盘上（WOD 与 navtrain / navtest 的 `temporal`、Qwen `L18_last` 都已抽），< 1 h；NAVSIM devkit 打分约 20 min；工程半天（head 的加载与 prior 对接）。

### E2. 真实帧的反事实编辑对（SAM 的位置：离线造对，不进回路）

- **问题**：能不能在 WOD / nuScenes 帧上批量造 x⁻，让配对差分在真实特征上训练。
- **做法**：只做「删除」不做「插入」。用 SAM 3.1（已在 `envs/sam3`）按 pedestrian / cyclist prompt 出 mask，视频一致 inpainting（先用仓库原样的 LaMa 或 ProPainter 之一，登记时定一个），
  x⁺ = 原帧，x⁻ = 抹掉该 actor 的帧；只取 actor 在走廊内、≤ 30 m、像素 ≥ 20 的帧（Q2b 的走廊定义）。**安慰剂对照**（必带）：同样大小的 patch 贴到空路面。
  标签：无 expert，三种并列，各自单独报：(a) log 自己的未来当 x⁺ 侧、x⁻ 侧标签 = CTRA 外推（继续）；(b) 规则裁判（Q6 的三条门）；(c) navtrain 上加 PDM scorer：对「继续」与「刹停」两条 proposal 各打分，分差的符号当标签（第 35 条：scorer 是可调优代理，只作第三来源，不单独定判格）。
- **数据**：WOD 与 navtrain 都做，**navtrain 为主**：它有 GT 3D 框，抹掉的 actor 是不是走廊内的原因物体、有没有抹干净、还有没有被遮挡的第二个行人，都能对着 GT 验；WOD 只能靠 SAM 自己的检出，作为第二数据集。
- **验证（造对之前先过）**：16 对目检；独立检测器（YOLO26 COCO person）在 x⁻ 上的残留检出率 ≤ 10%（2609.22582 报 8.5%）；navtrain 上再按 GT 框核对「被抹 actor 的框内无残留、其他 actor 未受影响」；安慰剂对上 Qwen 与 openpilot 特征的位移中位数当噪声地板。
- **读数**：在 E1 的 head 上，x⁺ − x⁻ 的 Δ 幅值对安慰剂对的比；用编辑对**训练**的 M-C 在 P5 v1 BA 集上的行人翻转（真实 → 仿真方向的迁移）与 WOD Pedestrians 的 RFS Δ（按 sequence 分折）。
- **判据**：编辑对上 |Δ| 中位数 ≥ 2 × 安慰剂对，且用编辑对训的 head 在 WOD Pedestrians 上 RFS Δ CI > 0 而直行帧激活率 ≤ 7%。
- **资源**：SAM 已量 188 ms / 张单路；WOD 走廊内行人帧约 160（Q2b）太少，扩到 train 全量前视 41.5 万帧里走廊内有行人 / cyclist 的帧（估 1–3%，4–12k 张）：SAM 约 0.5–1 h GPU，inpainting 约 1–3 h GPU；head 训练分钟级。约 3–5 GPU·h，工程 1–2 天。

### E3. 从 log 里挖孪生帧（新增，真实 expert、不编辑图像；主数据集 navtrain，WOD 并列）

- **问题**：navtrain（约 10 万 token，GT agent 轨迹齐全）和 WOD train（52 万帧，无 3D 框）里，有没有足够多的「ego 历史匹配、未来分叉」的帧对，能当配对差分的标签。
- **为什么主做 navtrain**：孪生对的价值在于「未来差只能由场景解释」，这句话在 navtrain 上可以直接用 GT 轨迹核对（x⁺ 侧走廊内有接近的行人 / 车辆、x⁻ 侧没有），不经 SAM；WOD 上只能靠 SAM（行人召回 0.36）判原因物体，上界受感知限制。PDM scorer 还能给每一侧的「继续 / 刹停」打分当第三标签。
- **做法**：每帧的 ego 历史向量（16 步 × 位置 / 速度 / 加速度，第 40 条 ego 特征）+ intent one-hot；在同 intent 内做最近邻（标准化后 L2，容差 τ_ego 在 null 分布上定：同一 sequence 相邻 0.25 s 的帧对之间的距离 p95），
  排除同一 sequence 与同一路段（用 sequence id 与 GPS 近邻）；未来分叉 = 5 s 未来的纵向速度差 ≥ 2 m/s 或横向偏移差 ≥ 1 m（与 P5 τ_exp 的量级对齐，登记时定死）。
  分叉的一侧当 x⁺（反应）、另一侧当 x⁻（继续）；方向由未来自己给出。**噪声对照**：ego 匹配但未来也匹配的对（孪生 null），它们的特征差分定 τ。
- **可行性统计（本项的第一交付，纯 CPU）**：两个数据集各一张：匹配对数按 τ_ego 与分叉阈值的网格；分叉对里 x⁺ 侧走廊内有原因物体的比例（navtrain 用 GT 轨迹：≤ 30 m、接近速度 < −0.5 m/s；WOD 用 Q2b 已有的前视 SAM 检测，20 237 帧，不够再补）；
  分叉对与 navhard、pre-onset、Pedestrians、Cut_ins 等分组的重叠；navtrain 上另报 PDM scorer 对两侧「继续 / 刹停」的分差分布，看它与人类未来的符号一致率。
- **判据**：任一数据集上分叉对 ≥ 1 000 且其中走廊内有原因物体的比例显著高于孪生 null 对（CI 不跨零）→ 在该数据集上开训练；两个都不过就记「log 里挖不出足够的配对」，只留 E2。navtrain 上另加一条描述：scorer 分差与人类未来符号一致率（不进判格）。
- **训练与读数（可行性过了才做）**：M-C 同一 head，训练行 = 孪生对，按 sequence / log 分折；navtest 上按有无行人分组的 PDMS / EPDMS 配对 Δ（官方 devkit）与 navhard；WOD Pedestrians / Cut_ins 的 RFS Δ、pre-onset ADE Δ；P5 v1 BA 行人翻转（真实 → 仿真）。对照：均匀 imitation、hard-example。
- **资源补充**：navtrain 的 `temporal` 与 Qwen 特征已在开环对比里抽好；GT 轨迹在 navtrain 的 log 里，走廊判定纯 CPU。
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

### E6（可选，不进主线）. NAVSIM 上那 6 分 PDMS 是不是 R 层配方

- **问题**：冻结 `temporal` + ridge 77.9 PDMS，TransFuser 84.0；差距是表征还是 head 配方。第 35 条判 R 层配方不稀缺，这里只是量一下。
- **做法**：同一份 navtrain 特征，两个 head 各一：(a) WOD 那种 K=1024 轨迹词表分类头（第 40 条 (iii) 的 `cls_late`）；(b) Hydra-MDP 式按 PDM 子指标各一个打分头、推理时加权选 proposal（权重只在 navtrain 内的 held-out 上定）。均匀 imitation 训练，与 E1–E5 的配对差分无关。
- **读数**：navtest PDMS / EPDMS，navhard 并列；报与 ridge 的配对 Δ。
- **判据**：只作描述，不改任何方向；(b) 若 ≥ 84 就在 backbone 一节写「512 维冻结特征 + 配方 head 到 TransFuser 水平」，否则写「差距不在配方」。
- **资源**：特征在盘上；(a) L-BFGS 小时级 GPU，(b) 离线在候选词表上跑 PDM scorer 一次（Hydra-MDP++ 的做法，CPU 多核小时级）+ head 分钟级。工程 1 天。排在 E1–E5 之后，卡空了再做。

## 顺序与依赖

```
E4（CPU，半天）──────────────────────────────► v1 主判定集可用，E2/E3/E5 的仿真侧读数都用它
E1（CPU，半天，WOD + NAVSIM 两列）──转移得过去──► E2/E3 只做标签复核
      └──────────────────────转不过去──────► E2 与 E3 并行（都以 navtrain 为主、WOD 并列；E3 先出可行性统计，纯 CPU）
fast-perception 延迟表 ───────► E5 登记完成 → 跑
E6（可选）──────────────────────► E1–E5 之后、卡空时
```

## 硬约束

- 组件全部冻结；head 只有配对差分、均匀 imitation、hard-example 三种训练信号，每项都三者并列。
- judge 一字不改：WOD 第 22 条口径，P5 v0 / v1 的定向翻转率 + 样本外 null false-flip；E4 新增的窗口口径先登记后使用。
- 训练 / 评估按 sequence（WOD）、log（navtrain / navtest 按官方 split）或路线（P5）分开；E2 / E3 的对不得与评估 sequence / token 同源。
- NAVSIM 的读数一律用官方 devkit 打分并注明版本；EPDMS 与文献只能写「同量级」，不写「超过」；所有 head 报 seed 数。
- 闭环仍暂停；本计划全部开环。
- 任何一步超估计 2 倍先停下报。结论：E1–E3 进 decisions.md 新条目「配对差分从 CARLA 到真实数据」；E4 补进第 42 条；E5 进第 42 条或新条目。

## 交付

1. E1：WOD 一张 cluster × {RFS Δ, ADE Δ, 激活率} 的表，NAVSIM 一张分组 × {PDMS Δ, EPDMS Δ, 激活率} 的表 + 判格。
2. E2：编辑对样例 16 组图、验证表（残留检出、安慰剂地板）、训练后的 WOD / P5 两向读数。
3. E3：navtrain 与 WOD 各一张可行性统计表（对数 × 阈值网格、原因物体比例、分组重叠；navtrain 另有 scorer 符号一致率）；过线则加训练读数；文献核查一段。
4. E4：PDM-Lite 集在两种窗口下的全考生表，与 BA 并排。
5. E5：student 的翻转 / 延迟表。
6. E6（可选）：navtest 上两种 head 对 ridge 的配对 Δ，一段结论。
小表进 `research/results/elicitation/`。

## 偏离与澄清日志

（执行时追加，时间戳早于受影响的数字。）

- 2026-09-26 00:35 CST（box 时钟，下同）[E1] 写于 E1 的任何数字之前。执行分工：本计划由 elicitation agent 负责，E1 自做，E4、E3 可行性各一个子执行代理并行（各自在本日志里记 [E4] / [E3] 条目）。E1 的操作化：
  (1) **head 是哪一个**。M-C 没有存 W；它是按路线 5 折 cross-fit 的，v1 BA 集上「过判据的 head」其实是 5 个 fold head（每个 fold 自己的 λ、自己的训练行标准化统计量）。
  做法：用 `reactivity_mc.fit_fold` 同一段代码、v1 BA 集、`op_streams_vis`、预登记 λ 网格，把 5 个 fold 的 `M-C pair`（双流）W 逐 fold 重算出来，
  先核对它在 obs 行上的预测与 `runs/reactivity/mc-carla_p5v1_ba/20260925-233126/preds_obs.npz` 一致（max |diff| 预期 0 或 fp32 噪声级，≤ 1e-4 m），
  λ 与该 run 的 `mc_fold` 事件一致；不一致就停。迁移用的 Δ = 5 个 fold head 的 Δ 取平均（每个 fold head 用它自己的 CARLA 训练行统计量做标准化、减它自己的训练行均值），
  不重新拟合、不选 λ。Cinque 与 Lebowski 各一（各配同模型的 prior），Cinque 为主。
  (2) **WOD 评测集**。登记写「WOD 完整 val」，但 Qwen `L18_last` 在 val 上只抽过 `qwenvid_p3` 的 19 663 帧（`p2p3_v1` 子集，与 P5 同一个 P3(d″) 抽取器、eager batch 2），
  完整 val（106 360）要再抽约 8.7 万帧（约 9–10 GPU·h）。所以评测集改为 `p2p3_v1` ∩ Qwen = 19 663 帧（其中 rater 帧 478、pre_onset 与 straight_yaw 按子集标记），
  与第 40 条 cross-fit 那一版的帧相同。prior = 第 40 条 (iii) 的 train 训 `A ridge_late op-<model> temporal`（`runs/drive_backbones/heads_train/20260925-110819/p3drive_heads_preds_dir0.npz`），
  openpilot `temporal` 取 `op_<model>_p3_trainval`（考试协议）。s_ego 分档的 decile 边界在完整 val 的 106 360 帧上定（与第 22 条 (iii) 一致），再限到这 19 663 帧。
  (3) **描述版标准化**（「用 WOD 行的统计量」）：两路特征各用 WOD **train** 行的均值 / 标准差（`qwenvid_train_t4` 的 137 533 个 train 行，openpilot 取同一批行；不用评测行），
  其余不变（W 不变，减的均值换成 WOD train 行在该标准化下的均值，即 0）。只描述，不进判格。
  (4) **读数**。cluster = index 的 `cluster` 列：Pedestrian、Cyclist、Cut_ins、Foreign Object Debris、Interections（原拼写）、全部。
  (i) RFS Δ = RFS(prior + Δ) − RFS(prior)，rater 帧（每 sequence 一帧，frame bootstrap 即 sequence bootstrap，`traj.boot_ci`）；ADE Δ 在 s_ego 第 1–9 档（对 log，sequence bootstrap），另报 pre_onset 第 1–9 档一行。
  (ii) **激活率**：P5 的 τ 是 null 对上 |v₂(pred(x⁺)) − v₂(pred(x_null))| 的 p95（v₂ = 2 s 速度），单帧上没有对，所以激活定义为 |v₂(prior + Δ) − v₂(prior)| ≥ τ，
  τ 取 v1 BA run 的 `flip_rates.csv` 里 `M-C pair [<model>]` 的 `tau_model`（Cinque 1.628、Lebowski 1.655 m/s，已存的数，不是新拟合）；分别报 straight_yaw、pre_onset、Pedestrian cluster、全部帧。
  判格里「直行帧」= straight_yaw 帧，7% 门槛照登记。另报 |Δ| 的 ADE 幅值（m）中位数作描述。
  (5) **NAVSIM 列的前提不成立，推迟**：登记写「navtrain / navtest 的 Qwen `L18_last` 都已抽」，实际盘上 NAVSIM 只有 openpilot `temporal`（`runs/navsim_zs/openpilot/<split>/<model>_temporal.npz`），没有任何 Qwen 特征。
  双流 head 在 navtest 上要先抽 Qwen：P3(d″) 抽取器原样，clip = NAVSIM agent 可见的 4 帧（−1.5 / −1.0 / −0.5 / 0 s，2 Hz，与 P5 / WOD 的 0.2 s 间隔不同，这是又一处输入分布差，照记），
  相机 CAM_F0 / CAM_L0 / CAM_R0 → front / front_left / front_right；navtest 12 146 token 约 1.1–1.4 GPU·h，要等 GPU 1 / 2。NAVSIM 列的其余口径现在定死：
  prior = navtrain 训的 `ridge_late <model> temporal`（`runs/navsim_zs/heads/20260925-232810`，与 P5 / M-C 一样是连续回归 prior，Δ 可以直接相加）；`cls_late` + Δ 只描述。
  Δ 在 0.25 s 网格上，取 0.5 … 4.0 s 的 8 个点（索引 1, 3, …, 15）加到 prior 的 (x, y) 上，heading 保持 prior 的不变；官方 devkit（v1.1 出 PDMS、main @ 0a380a9 出 EPDMS，`OPENBLAS_CORETYPE=Haswell`）打分；
  分组：navhard two-stage（EPDMS）、「走廊内 ≤ 30 m 有行人 / cyclist」的 navtest token（GT agent：logged 未来 4 s 路径 ±1.5 m、前方 ≤ 30 m，与 E3 的原因物体定义同一函数）、其余 token；激活率同 (4)(ii)。
  (6) 判格里「转移得过去」的直行帧门槛写的是「≤ P5 null false-flip（约 5–7%）」：取该 head 自己在 v1 BA 上的样本外 null false-flip（Cinque 5.1%、Lebowski 5.0%）；「无害 / 有害」的 7% 照登记。
  两个模型分别判，主判 Cinque。fold head 的重算在 CPU 上做（GPU 都有人），与已存预测的差按 fp32 噪声级核对，λ 必须逐 fold 相同。
  (7) 2026-09-26 00:45 [E1]（WOD 的任何数字出来之前）fold head 在 CPU 上重算只复现到 1.3 cm（fold 0 Cinque，λ = 0.1 在网格下沿，3072 维的解病态，CPU 与 GPU 的 fp32 累加顺序差被放大），
  没过 (1) 的 ≤ 1e-3 m 核对。病态方向正是域外特征可能投影上去的方向，所以不放宽门槛，改在 GPU 上重算（原 run 是 GPU fit，同型号卡），占 GPU 1 几分钟、< 5 GB，gpu-plan 记一行；其余读数仍在 CPU。
  登记的三格判据都是 WOD 上的量（Pedestrians / Cyclists 的 RFS Δ、straight_yaw 激活率、全部 rater 帧 RFS Δ），所以 E1 的判格由 WOD 列下；NAVSIM 列按登记报表，不改判格。

- 2026-09-26 00:30 CST（box 时钟）**[E4] 两种窗口的操作化**（写于任何 E4 数字之前；此前只见过 v1 两套集合的官方逐帧数，即第 42 条已记的那些）。
  输入全部是已存的逐帧打分，不重拟合：PDM-Lite 集 `runs/p5_pairs/exam-d0-carla_p5v1_pdm/20260925-230423`（TFv6 三通道、`ridge ego`、Qwen 与 openpilot 的 `ridge_late`）、
  `runs/reactivity/mc-carla_p5v1_pdm/20260925-230424`（M-C 全部 arm 与 prior）、`runs/fusion_diag/q6gt_carla_p5v1_pdm/20260925-233007`（Q6 GT 规则门）；BA 集用对应的三个 run 并排。
  Q6-SAM 在 v1 上按融合诊断 23:36 的决定没有跑，SAM 状态版考生缺，记为缺项，不补。Q1-v1 的拼接 arm 不在本项考生名单里。
  (1) **可见门控 (a)**：t_vis 取 `pairs.csv` 该对的 `t_vis`（20 Hz tick），观测帧 k 只保留 k ≥ t_vis + L / 0.05；reactive 的定义不变（|Δ_expert| > τ_exp，τ_exp 用官方 run 的值），
  所以「该帧的 Δ_expert 仍 > τ_exp」由保留下来的 reactive 帧自动满足。L 按考生：openpilot 流（`ridge_late op-*`、M-C `prior`、M-C `pair op`）0.1 s（Q8 的 2.3 ms 加一拍相机 tick，按登记的 0.1 s）；
  Qwen 流（`ridge_late L18_*`、M-C `pair qwen`）0.3 s；M-C 双流 arm（`pair`、`pair (mu=0)`、`hard`、`uniform`）取两路较慢者 0.3 s；
  TFv6 三通道 0.1 s（三 seed ensemble 前向在 3090 上约 51 ms、agent 每 tick 约 110 ms，见 todos/remote_carla.md，向上取到 0.1 s）；`ridge ego` 0（只读自车状态，无感知）；
  Q6 GT 规则门 0（特权 GT 状态，无感知延迟，是上界），另加一行只描述的「GT 规则门 + L = 0.6 s」当 SAM 状态版在 SAM 延迟下的上界。
  观测帧间隔 0.2 s，所以 L = 0.1 只去掉 k = t_vis 那一帧，0.3 去掉前两帧，0.6 去掉前三帧。
  τ_model 不重算（沿用官方 run 在全部 null 帧上的值，judge 不改）；null false-flip 按同一窗口重算：null 帧用该 base 的 seed-0 对的 `t_vis`（seed-0 对没有 t_vis 的 fallback null 窗口用 `t_trig`，与 `p5_pairs` 取 null 帧的规则一致）按同一 L 门控，
  样本外口径与官方相同（null 按 base 路线对半，τ 在一半上定、在另一半的门控帧上算，两个方向平均）。翻转率、非反应帧误翻、CI（base 路线 bootstrap 2000 次）其余一字不改。
  (2) **按对计分 (b)**：窗口 = 该对的整个观测窗口（t_vis ≤ k < t_div − 1，即官方的观测帧，不加 L 门控，与第 32 条的按对口径相同）；一对（base_id, seed）在某个 scope 里有 ≥ 1 个 reactive 帧才进分母，
  其中任意一个 reactive 帧定向翻对（同号且 |Δ_model| ≥ τ_model）即算过；CI 按 base 路线 bootstrap。对应的 null 口径：一个 null case（每个 base 一个）只要窗口内任意一帧 |Δ_model| ≥ τ_model 就算误翻，
  报样本内与样本外（同 (1) 的对半 τ）两个数；这是与「任意一帧翻对就算过」对称的地板，预期会远高于逐帧的 5%，判据 (a) 不用它。
  (3) **scope**：官方的合并 family（pooled）、行人（4 个行人 family 的 reactive 帧，同 M-C criteria）、cut-in（3 个 cut-in family）。
  (4) **判据**按登记：(a) 下 M-C 双流（`M-C pair [cinque]`，Lebowski 复现）行人翻转与 BA 集官方逐帧的 43.3%（Lebowski 41.6%）差在 ±15 pp 内，且 (a) 下样本外 null false-flip ≤ 7% → PDM-Lite 集成为主判定集；
  否则 BA 仍为主，PDM-Lite 只报 (b)。以 Cinque 判，Lebowski 不一致时照实写。BA 集上同样算 (a)(b) 作并排描述，不进判据。代码 `jevdrive/elicit_e4.py`，run `runs/elicitation/e4/<time>`。

- 2026-09-26 00:28 CST（box 时钟）**[E3] 可行性统计的操作化**（写于 E3 的任何挖掘、任何数字之前；只看过数据格式）。不训练，纯 CPU（≤ 24 核），代码 `jevdrive/elicit_e3.py`，run `runs/elicitation/e3-feas/<time>`。
  (1) **ego 历史向量**。WOD：第 40 条的 ego 特征原样，`past.npy` 16 步 × (x, y, vx, vy, ax, ay)（0.25 s 间隔、4 s，t0 自车系）= 96 维。navtrain：NAVSIM agent 合法可见的全部 ego 量，
  4 个历史位姿 (x, y, yaw，−1.5 … 0 s、2 Hz) + 4 个速度 + 4 个加速度 = 28 维（`navsim_heads.ego_features` 去掉 command 那 4 维；t0 位姿恒为 0 的列标准差置 1）。
  「16 步」在 NAVSIM 上不存在，只能用这 4 步。两边都按本数据集挖掘池的列均值 / 标准差标准化，距离 = 标准化后的 L2。
  (2) **分层**：同 intent / command 内找近邻（WOD `intent` 0–3；navtrain 取 t0 的 driving command one-hot 的 argmax）。
  (3) **挖掘池与近邻**：WOD 主读数用 train（415 663 帧）**抽稀到 2 Hz**（`frame % 5 == 0`，与 navtrain 的 2 Hz 对齐，也去掉 10 Hz 下几乎重复的相邻帧），且要求 `has_future`；
  全 10 Hz 池只作描述。每帧取**一个**最近邻（同层、不同 sequence / log、且不在同一路段），距离 ≤ τ_ego 才成对；(A, B) 与 (B, A) 去重。
  「同一路段」：navtrain 用 log 里的 `ego2global` 平移，两帧同 `map_location` 且全局距离 < 30 m 即排除；WOD-E2E 的 index 与 past 都在 t0 自车系、没有全局位姿，**只能排除同一 sequence**（照记，不补）。
  另报「不排除同一路段」的 navtrain 对数作描述。
  (4) **τ_ego**：同一 sequence / log 内时间上最近的两帧之间的 ego 向量距离的 p95。WOD：同一 sequence 相隔 3 帧（0.3 s，10 Hz 下最接近且不短于 0.25 s 的间隔）；
  navtrain：同一 log 内相邻 2 Hz token（时间戳差 0.5 s ± 0.05 s，nuPlan / OpenScene 能给的最小间隔）。网格 τ ∈ {0.5, 1, 2} × τ_ego，主格 1 ×。
  (5) **未来与分叉**。WOD `future.npy` 20 点 × 0.25 s = 5 s；navtrain `navtrain_future.npz` 8 点 × 0.5 s = **4 s**（NAVSIM 只有 4 s，登记的「5 s」在 navtrain 上改为 4 s）。
  终端纵向速度 v_T = 最后 1 s 的平均速度（|p_T − p_{T−1 s}| / 1 s）；横向偏移 y_T = 终点在各自 t0 自车系里的 y。分叉 = |Δv_T| ≥ θ_v 或 |Δy_T| ≥ θ_y；网格 (θ_v, θ_y) ∈ {(1, 0.5), **(2, 1)**, (3, 2)}（m/s, m），主格 (2, 1)。
  **孪生 null** = 同样 ego 匹配、但 |Δv_T| ≤ 0.5 m/s 且 |Δy_T| ≤ 0.3 m 的对。
  (6) **哪一侧是 x⁺**：两侧各算未来对自己的匀速直行外推（t0 速度沿 t0 朝向）的 ADE，偏离 continuation 更大的一侧是 x⁺（反应），另一侧 x⁻（继续）；null 对用同一规则挑「x⁺」侧，保证比较对称。
  (7) **原因物体（navtrain，GT）**：x⁺ 那一帧 t0 的 GT agent（log 的 `anns`：框、名字、`gt_velocity_3d`，自车系；先核对它是绝对速度——自车在动时 traffic_cone 的速度中位数 < 0.3 m/s，不是就停）。
  走廊 = 原点 + 该帧自己的 8 个 logged 未来位姿连成的折线，沿最后朝向延长到弧长 30 m（Q2b 事后敏感性那一版 `fusion_q2b.extend`，这里登记为主读数：刹车一侧的 4 s 路径到不了让它刹车的物体）；
  物体在走廊内 = 框 footprint（中心 + 4 角）任一点到折线横距 ≤ 1.5 m、投影弧长 0 < s ≤ 30 m；原因物体 = 走廊内且接近速度（位置单位向量 ·(v_obj − v_ego)，v_ego 取 `ego_dynamic_state` 的 vx, vy）< −0.5 m/s，类别不限，另按 vehicle / pedestrian / bicycle / 其他分列。
  同一个函数（`elicit_e3.corridor_objects`）给 E1 的 NAVSIM 分组用。另报两条描述：走廊内有物体（不要求接近）的比例；x⁻ 侧同一定义的比例（对内配对差）。
  (8) **原因物体（WOD，SAM）**：Q2b 的 SAM 3.1 前视检测（score > 0.5，按该 sequence 的标定平地抬升）只覆盖 val 的 20 237 帧（`p2p3_v1` 子集 + rater 帧），**train 上没有检测**。
  所以 WOD 的原因物体比例在这 20 237 帧**内部**另做一次同口径挖掘（池 = 这 20 237 帧，不再抽稀，其余规则同 (1)–(6)）上算：走廊同 (7)（logged 5 s 路径延长到 30 m、±1.5 m、前方 ≤ 30 m），
  单帧检测没有速度，**不要求接近速度**，类别 = vehicle / pedestrian / cyclist / cone | debris / emergency vehicle 任一。这些 val 对只用于可行性统计，不会进训练（评测 sequence 不得同源）。
  WOD 判据里的「分叉对 ≥ 1 000」用 train 池的对数，「原因物体比例高于 null」用 val 子集的对；两者不同源，照记。
  (9) **判据的检验**：主格上，分叉对的 x⁺ 原因物体比例 − 孪生 null 对的 x⁺ 原因物体比例，95% CI 用按 log / sequence 的 cluster bootstrap（以 x⁺ 帧的 log / sequence 为单位，2000 次）；CI 下端 > 0 且分叉对 ≥ 1 000 → 该数据集「过」。
  (10) **分组重叠**（描述）：navtrain——x⁺ 侧走廊内有行人 / bicycle、t0 静止（v₀ < 0.5 m/s）、command 左 / 右 / 直；navhard 由 navtest 系场景构造，与 navtrain token 按 token 核对重叠（预期为 0，照报）。
  WOD train——pre_onset、straight_yaw、turn_yaw（`waymo.subsets`）、静止起步；cluster 只在 val 上有标，所以 Pedestrian / Cyclist / Cut_ins 等的重叠只在 (8) 的 val 子集对上报。
  (11) **PDM scorer 符号一致率**（描述，不进判格）：navtrain 没有 metric cache，全量缓存超过 1 h CPU，所以**抽样**：主格分叉对里两侧 t0 速度都 ≥ 2 m/s 的对，seed 0 随机取 300 对（600 个 token），
  只为这些 token 建 v1.1 metric cache（`scripts/navsim_zs_score.sh cache v1 navtrain` 加 token 过滤）。每个 token 两条 proposal，几何都沿该帧 logged 未来路径：
  「继续」= 以 t0 速度匀速，「刹停」= 从 t0 速度以 3 m/s² 减速到停；官方 PDMS（v1.1）各打一次，分差 = PDMS(刹停) − PDMS(继续)。
  人类符号：logged 4 s 弧长比匀速弧长短 2 m 以上记「人类减速」，长 2 m 以上记「人类加速 / 继续」，其余不判。报 (a) 逐 token：|分差| ≥ 0.05 且人类可判的 token 上，分差符号与人类符号一致的比例；
  (b) 逐对：sign(分差_x⁺ − 分差_x⁻) 与 sign(人类减速量_x⁺ − 人类减速量_x⁻) 一致的比例。token bootstrap CI。缓存或打分若实测超过 1 h 就停在已完成的子集上，照记。
  (12) **特征可得性**：统计主格分叉对里两侧都有现成特征的对数——navtrain / navtest 只有 openpilot `temporal`（无 Qwen）；WOD train 的 Qwen `L18_last` 只有 thin=4 的 137 533 个 train 行（`qwenvid_train_t4`），openpilot `temporal` 覆盖全部 trainval。
  并按 0.33–0.40 s / 帧 / 卡估双流训练所需的 Qwen 抽取量（GPU·h）。

## 结果

### E4：PDM-Lite 集的两种计分窗口（2026-09-26 00:27，CPU，< 1 min）

代码 `jevdrive/elicit_e4.py`，run `$DATA_DIR/runs/elicitation/e4/20260926-002723`，小表 [research/results/elicitation/e4/](../research/results/elicitation/e4/)（`e4_all.csv` 是全部考生 × 窗口 × scope，`criterion.csv` 是判格，`reactive_lead_times.csv` 是 reactive 帧离可见的时间）。
口径见偏离日志 [E4] 00:30。等价性：「逐帧」一列在两个集合的 29 个考生上逐项复现官方 run 的合并翻转率（最大差 1e-16），样本外 null false-flip 逐项相同（差 0）。

**先看 reactive 帧落在哪**（从因素对相机可见算起，秒）：

| 集合 | 组 | reactive 帧 | p10 | p25 | 中位 | p75 |
|:--|:--|--:|--:|--:|--:|--:|
| PDM-Lite | 行人 | 309 | 0.0 | 0.4 | 0.8 | 2.4 |
| PDM-Lite | cut-in | 474 | 0.3 | 1.6 | 2.8 | 3.6 |
| BehaviorAgent | 行人 | 406 | 0.2 | 1.0 | 2.2 | 3.2 |
| BehaviorAgent | cut-in | 676 | 5.5 | 6.2 | 7.0 | 7.9 |

所以 (a) 的 0.1–0.6 s 门控只去掉 PDM-Lite 行人 reactive 帧的 11–35%、cut-in 的 5–15%，大部分帧早已在任何考生的延迟之后。

**PDM-Lite 集，行人 reactive 帧的定向翻转率 %（[route bootstrap 95% CI]），与 BA 集官方逐帧并排**：

| 考生 | L (s) | BA 逐帧 | PDM 逐帧 | PDM (a) 门控 | PDM (b) 按对 |
|:--|--:|:--|:--|:--|:--|
| TFv6 waypoint 2 s | 0.1 | 29.3 [22.0, 36.7] | 10.4 [3.3, 19.6] | 11.6 [3.8, 22.1] | 25.4 [10.8, 41.7] |
| TFv6 target speed | 0.1 | 0.0 | 6.1 [2.6, 10.4] | 6.5 [2.4, 11.5] | 23.8 [9.7, 39.3] |
| `ridge ego` | 0 | 7.6 [4.9, 10.4] | 11.7 [8.1, 15.7] | 11.7 | 52.4 [37.5, 67.2] |
| openpilot `ridge_late` = M-C prior（Cinque） | 0.1 | 0.2 | 0.6 | 0.7 | 3.2 |
| Qwen `ridge_late L18_last` | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| **M-C 配对双流（Cinque）** | 0.3 | **43.3 [35.0, 50.7]** | 1.0 [0.0, 2.7] | **0.8 [0.0, 2.9]** | 4.8 [0.0, 13.1] |
| M-C 配对双流（Lebowski） | 0.3 | 41.6 [32.4, 50.0] | 0.3 | 0.4 [0.0, 1.2] | 1.6 |
| M-C 只 Qwen / 只 openpilot | 0.3 / 0.1 | 42.1 / 6.9 | 0.6 / 0.6 | 0.8 / 0.7 | 3.2 / 3.2 |
| M-C hard-example / 均匀 | 0.3 | 0.0 / 0.0 | 0.3 / 0.6 | 0.4 / 0.8 | 1.6 / 3.2 |
| Q6 GT 规则门（any） | 0 | 80.0 [75.4, 85.6] | 10.7 [5.7, 16.6] | 10.7 | 36.5 [21.0, 53.5] |
| Q6 GT 规则门，L = 0.6 s（SAM 延迟下的上界，描述） | 0.6 | 80.0 | 10.7 | 12.4 [5.7, 22.8] | 36.5 |

**cut-in 与合并**（同样的列，只列主要考生）：

| 考生 | cut-in：BA / PDM 逐帧 / (a) / (b) | 合并：BA / PDM 逐帧 / (a) / (b) |
|:--|:--|:--|
| TFv6 waypoint 2 s | 32.5 / 13.9 / 14.4 / 26.0 | 30.4 / 12.5 / 13.4 / 25.7 |
| M-C prior（Cinque） | 76.9 / 0.4 / 0.4 / 2.6 | 48.2 / 0.5 / 0.6 / 2.9 |
| M-C 配对双流（Cinque） | 80.0 / 1.7 / 1.9 / 7.8 | 66.3 / 1.4 / 1.5 / 6.4 |
| M-C 配对双流（Lebowski） | 77.2 / 6.5 / 7.3 / 15.6 | 63.9 / 4.1 / 4.8 / 9.3 |
| Q6 GT 规则门 | 34.2 / 0.0 / 0.0 / 0.0 | 50.8 / 4.2 / 4.2 / 16.4 |

null false-flip：(a) 下所有学习考生的样本外 null 仍是 4.5–5.5%（规则门 0，`ridge ego` 0.3%）；(b) 的对称地板是「一个 null case 里任意一帧动了就算误翻」，
学习考生在 PDM-Lite 上是 45–56%（BA 上 25–55%），`ridge ego` 2.2%，规则门 0。也就是 (b) 的按对通过率只有和这个地板并排读才有意义，
M-C 双流行人 4.8% 远在它自己 56% 的 null case 地板之下。

**判定（按登记）**：(a) 下 M-C 双流行人翻转 0.8%（Lebowski 0.4%），对 BA 集官方 43.3%（41.6%）差 −42.5 pp（−41.2 pp），出了 ±15 pp；null 5.1% 过了第二条。
**不过 → BA 集继续为主判定集，PDM-Lite 集只报 (b)。** 两个模型一致。

读法：

1. **登记时的机制假设不成立。** E4 的前提是「PDM-Lite 在视觉证据出现之前就反应，reactive 帧落在考生来得及看之前」，据此设计了按延迟门控。
   实测 PDM-Lite 行人 reactive 帧中位在可见后 0.8 s、cut-in 2.8 s，门控只去掉少数帧，翻转率在门控前后几乎不变（M-C 1.0 → 0.8%）。
   所以 PDM-Lite 集的零不是「早了几百毫秒」，窗口口径修不好它。
2. **更可能的原因是 PDM-Lite 反应的依据不是几何上可见的威胁**（推测）：GT 状态上的规则门（TTC、车道侵入、行人接近，都用特权 GT 位置和速度）在 BA 集行人上 80%、cut-in 34%，
   在 PDM-Lite 集上只有 10.7% 与 0%。也就是 PDM-Lite 减速的那些帧里，连 GT 几何都还没显示「有威胁」；cut-in 里它在对方开始并线之前约 4 s（7.0 → 2.8 s）就开始让，
   读的应该是 scenario 的特权状态（对方的计划路径或触发器），不是画面。M-C 各 arm 与 expert 的同号比例（不看 τ）也掉到接近机会水平：行人 0.57–0.64（BA 0.74–0.87）、cut-in 0.57–0.70（BA 0.96–0.98）。
   验证办法：把 PDM-Lite reactive 帧按「GT 规则门是否已触发」分开计翻转，看考生在已触发的那部分上是否恢复到 BA 的量级（本项没做，不在登记里）。
3. **`ridge ego` 在 PDM-Lite 上反而是按对最高的学习考生（行人 52%）**：它的 τ_model = 0，任何非零差都算动；它只读自车历史，所以这说明 PDM-Lite 集的一部分 reactive 帧上两侧 ego 历史已有亚阈值差
   （t_div 的阈值之下，PDM-Lite 已开始微减速），属于标签伪影（推测），不是 `ridge ego` 看见了什么。
4. TFv6 在 PDM-Lite 集上比 BA 集低（waypoint 29 → 10%），但 target speed 通道反而从 0 升到 6%（TFv6 的训练 expert 就是 PDM-Lite 一族，推测它学到了同样的提前量），只作描述。

![E4](../research/figs/elicit-e4-windows.png)

主要考生在行人（a）与 cut-in（b）reactive 帧上的定向翻转率；灰色是 BA 集官方逐帧，三种蓝 / 橙是 PDM-Lite 集的逐帧、(a) 门控、(b) 按对；误差线是 base 路线 bootstrap 95% CI，点线是 20% 门槛。
要看的是：蓝色两根几乎等高（门控不改变 PDM-Lite 的读数），而且连 GT 规则门在 PDM-Lite 上也从 0.8 掉到 0.1，问题在标签，不在窗口。
