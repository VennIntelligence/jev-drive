# op_parity turn-probe 预登记：冻结的 Cinque 特征里有没有急弯处的路口 / 路沿几何（2026-10-07，任何本线读数之前写定）

## 问题

P2 / P2H 在 navtest 急弯主要丢在切内角（第 153 条）。假设 H：冻结的 Cinque 视觉 token 没有把路口 / 可行驶边界的几何编码到够用的程度，而 WA-Cf 编码了。
H 成立 → 修法是「解冻后段视觉 + 可行驶区域辅助 loss」；H 不成立 → 问题在 plan head（decoder / anchor 的低曲率先验），视觉保持冻结。
本线只做读出探针：不训练任何驾驶模型，不改 P2 / P2H 代码路径，不提交 test。

## 已有的、不重测（打分前已知的数，如实列出）

- 第 146 / 147 条（op_probe dac-localize）：同一批 token、同一种 [stage, E] ridge / MLP 探针已经读过可行驶 SDF：navtest 整体 raster R^2 Cinque V 0.66 对 WA-Cf 0.78
  （MLP 0.80 对 0.89）；沿 logged path 的 corridor MAE 在 PP-turn 上 2.26 对 1.61 m。所以「WA-Cf 整体更好」是已知的，本线的规则 R1 预期成立，不算新信息。
- op_probe view-by-turn：thin decoder 的 DAC 失败率差（Cf − V）随转角变大（> 45° −5.28 pp）；第 160 条 Stage 0 junction look-ahead AUC V 0.805 对 WA-Cf 0.850。
  这些是 decoder / 分类读数，不是几何误差按转角的分层。
- 没有做过的（本线的内容）：几何读出误差按转角分层（0–5 / 5–20 / 20–45 / > 45°）、转弯相对直行的退化量对比、P2H 切内角失败 token 对通过 token、
  距入弯点的距离、navtrain held-out log、VJ21 参照。
- 复用：特征缓存 `runs/op_probe/feats/{P2-F-s0 (V = view_39, 32 × 512), WA (Cf, 32 × 512), VJ21 (raw 32 × 1024 + runs/op_probe/rep/vj21_pca.npz → 512)}`，
  SDF 标签 `runs/op_probe/labels/{navtrain_s23456, navtest}.npz`（第 148 条 hinge 用的同一份），探针实现 `opb_probe.ridge_fit / mlp_fit_predict`，
  P2H 失败标签 `results/four_dirs/nav_tokens_navtest.csv`。不重新抽特征。

## 设计

**行。** 训练 = `navsim/op-parity-s234-train@v1`（25 415 token，navtrain s2–s4 去掉 dev log）；navtrain held-out = `navsim/op-parity-s234-dev@v1`（408 token，log 不相交；
量小，只作 in-domain 对照）；测试 = navtest 全部 12 146 token（`navsim/navtest`）。ridge 的 λ 在训练 split 内部按 log 哈希留出的 10% 上选（不用 dev，也不用 navtest）。

**特征臂**（探针输入一律 z-score 后的 [X, E]，X 都是 32 × 512 = 16 384 维，容量相同）：E（只有 20 维 ego 状态 + command + 历史位姿，floor 对照）、
V（Cinque `view_39`，P2 配方实际吃的 token）、WA（WA-Cf，第 160 条 Stage 1 的 memory 来源）、VJ21（冻结原版 V-JEPA 2.1，PCA 512，只作参照，不进规则）。

**探针。** 线性 = ridge；非线性 = 2 层 MLP（1024 hidden，dropout 0.1，3 000 步，batch 512，seed 0），即 op_probe 的两个探针，原样，不调参。每个臂完全相同。

**目标。** 1 m 的 ego 系可行驶 SDF raster（64 × 48，x −8..56 m，y ±24 m，clip ±10 m）+ 8 个 corridor 距离（沿 logged future 弧长 5 / 10 / 15 / 20 m 处，
路径法向左 / 右到可行驶边界的自由距离，上限 15 m）。

**主指标：边界带 SDF MAE（m）。** 在预测的 raster 上，只算真值 |SDF| ≤ 2 m 且 x ∈ [0, 32) m、|y| < 16 m 的格子的 |pred − true|，每 token 取平均。
选它的理由：(1) 目标与路径无关（ego 系固定网格），探针不需要猜车要往哪走，E 臂不能靠 command / 速度直接拿分，读的是几何本身；(2) 只在边界附近计分，
单位是「路沿位置错了多少米」，不被大片路面内部 / 远处格子稀释；(3) 32 m × ±16 m 覆盖转弯车 4 s 内能到的范围。
**次指标**（只报告）：从预测 raster 沿 logged path 读出的 corridor MAE，按转弯内侧 / 外侧拆开（内侧 = 转向同侧），以及内侧带符号偏差（pred − true，
正 = 特征以为内侧比实际宽）；直接回归的 corridor MAE；整体 raster R^2。

**分层**（logged future 4 s 航向变化的绝对值，153 / 154 口径）：S5 < 5°、5–20°、T20–45、T45 > 45°。
navtest T45 内再分：P2H 切内角失败（`nav_tokens_navtest.csv` 中 bucket = `D1' >45`、key ∈ {P2Hs0, P2Hs1}、inside = 1）、其他 P2H DAC 失败（inside = 0）、
通过（两个 seed 都不在该 bucket）。
**距入弯点**（navtest |Δψ| > 20° 的 token）：入弯点 = logged future（线性插值到 0.1 s）上 |ψ| 首次 ≥ 5° 的位置，d = 到该点的弧长；
已在弯中 = 历史 1.5 s 的航向变化 ≥ 5°。分箱：已在弯中 / d < 5 m / 5–15 m / ≥ 15 m。

**CI。** 按 log 的 cluster bootstrap（ratio of sums），B = 4 000，seed 0，各臂、各分层共用同一组重采样（配对）。

## 判定规则（主指标，MLP 探针，navtest；ridge 的两个点估计必须同号，否则降级为「探针依赖」）

- **G0 有效性**：T45 上 skill = 1 − MAE / MAE(E) 对 V 和 WA 都 ≥ 0.15。不满足 → 该目标不是视觉可读的，本线无结论。
- **R1 差距**：Δ45 = MAE_V(T45) − MAE_WA(T45) 的 CI 下界 > 0，且 Δ45 / MAE_WA(T45) ≥ 0.10。
- **R2 转弯特异**：DiD = [MAE_V(T45) − MAE_V(S5)] − [MAE_WA(T45) − MAE_WA(S5)] 的 CI 下界 > 0（Cinque 从直行到急弯的退化比 WA-Cf 大）。

| 结果 | 结论 | 含义 |
|:--|:--|:--|
| R1 且 R2 | **表征缺（转弯特异）** | 支持「解冻后段视觉 + 可行驶区域辅助 loss」 |
| R1 不成立 | **head 侧** | 视觉保持冻结，查 decoder / anchor 的低曲率先验 |
| R1 成立、R2 不成立 | **整体 encoder 差、非转弯特异** | 探针不支持「急弯处几何缺失」这个专门假设；急弯的额外损失不在这份读出里，倾向 head 侧 + 第 160 条的整体 encoder 质量 |

**失败子集读数（只报告，方向事先写明）**：若 H 在失败机制上成立，应看到 V 的内侧带符号偏差在切内角失败 token 上 > 0（CI 不含 0）且大于 WA 的（配对差 CI 不含 0），
而在通过 token 上不是。已知先验：第 146 条在 P2 的 F 集上 margin 偏差 +0.75（V）对 +0.38（WA-Cf）。

## 小步闸门

先跑 small read：训练行随机 4 000（seed 0），只 ridge，臂 E / V / WA，navtest 全量按同一规则读。G0 不过（两臂 skill 都 < 0.15）→ 线停、报告无效。
否则直接跑全量（预计 < 10 卡分钟，比再评估便宜）；small read 的 R1 / R2 照报，结论以全量为准。

## 预算与偏离

GPU ≤ 1 卡时（预计 ~0.2），经 pool；CPU 报告。偏离只在打分前写进本文件末尾。

## 偏离（打分前）

（暂无）
