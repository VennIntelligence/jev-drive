# E0：CARLA 行人信息在 openpilot 的哪一层消失（逐层 probe）

状态: done（登记写于 2026-09-29 12:40 CST，任何逐层数字之前；结果 12:50 CST）
主题: [research/feature-adapter-domain-shift.md](../research/feature-adapter-domain-shift.md) §5 E0；[research/decisions.md](../research/decisions.md) 第 42、55 条；[op-adapt todo](2026-09-28-op-adapt.md)
代码: `scripts/op_layer_probe.py`（抽特征 + probe），`jevdrive/op_torch.py`（port）
box run dir: `$DATA_DIR/runs/op_layer/`

## 目标

第 42 条说 CARLA 上 openpilot 的 `vision` 层读不出行人（AUC 0.51）；第 55 条在真实 nuScenes 上测到 stage 3 输出 0.83、`temporal` 0.71。
E0 回答一个问题：**CARLA 上行人信息在 stage 1、2、3、4、`vision`、`temporal` 中的哪一层消失**，同一套 probe 在 nuScenes 上做参照。
结果决定 feature adapter（E2）放哪一层，还是判不可行。

名词：
- **stage 1–4**：Cinque vision 编码器（ConvNeXt）的四段；每段末尾（下采样前的 LayerNorm 之后）的特征图分别是 `permute_9`（256 × 32 × 64）、`permute_17`（512 × 16 × 32）、`permute_73`（1024 × 8 × 16，即 P5 trunk 缓存）、`permute_81`（2048 × 4 × 8，1×1 投影到 hidden 之前）。
- **mean + max 池化**：特征图在空间上取均值和最大值再拼接，维数是通道数的 2 倍（512 / 1024 / 2048 / 4096）。第 55 条的 trunk 深度诊断用的就是这个池化。
- **`vision`**：`mean` 输出（512 维池化视觉头）；**`temporal`**：`select_4`（policy 的 512 维时间 token），两者都是第 42 / 55 条一直用的 tap。
- **D0 probe**：第 42 条的 hazard probe，`p5_exam.probes` + `probe_auc_paired_scopes`。

## 数据（写死）

- **CARLA**：P5 v1 BA（`carla_p5v1_ba`），行人 4 个 family，4 414 对 x⁺ / x⁻、42 条路线（scope `pedestrian`）。特征 = 与第 55 条 (c) 同一批帧（P5 index 的全部行，含训练行）。
- **真实**：nuScenes val 150 个 scene 的 6 019 个 keyframe，走廊内行人（`ped_corr`，203 个正例）为主标签，宽走廊（`ped_wide`）、10 m 内走廊（≤ 10 m 档，89 个正例）为描述。
- 输入管线与第 55 条 / 缓存完全一致：nuScenes 用 20 Hz 时钟协议、P5 用 5 Hz 帧保持协议，road + wide 两个相机、(前一帧, 当前帧) 对；Cinque port，fp16（与 trunk 缓存同一数值路径）。
- stage 3：**复用 P5 与 nuScenes 的 trunk 缓存**（`processed/op_adapt/{p5,nusc}/`）池化得到；stage 1、2、4 由 port 从像素重新前向抽取（只抽读数帧）；同一次前向也算一份 stage 3，与缓存池化后的值比对（数值一致性检查，不进判格）。
  `vision` / `temporal` 复用第 55 条 eval 的 `orig_vision` / `orig_temporal`（`runs/op_adapt/train-lam10/*/eval/{nusc,p5}.npz`）。

## probe（写死，与第 55 条同一套）

- **CARLA**：`p5_exam.probes` 原样（hazard 标签、在非观察行上训练、特征按训练折 z-score、类平衡 L2 logistic、λ ∈ {1e-4..1e-1} 在训练折内按路线 20% 分组留出选、5 折按 base 路线分）；
  读数 = x⁺ 对 x⁻ 的样本外 AUC，scope `pedestrian`，路线聚类 bootstrap 500 次；对 `vision` 的配对 Δ = 同一 bootstrap 里 AUC(layer) − AUC(`vision`) 的 2.5 / 97.5 分位。
- **真实**：第 55 条 (a) 的 `oof_probe`（z-score、类平衡 L2 logistic、λ 同上、5 折按 scene 分、样本外 AUC、scene 聚类 bootstrap 500 次），配对 Δ 对 `vision`。
- 复现检查：nuScenes val 上 stage 3 mean + max 的 `ped_corr` AUC 应为 0.831、`temporal` 0.709、`vision` 0.702（第 55 条）；偏差 > 0.01 就先查管线，不报判格。
- 描述性附加（不进判格）：2 层 MLP probe（隐层 256、ReLU、Adam lr 1e-3、wd 1e-2、full-batch 200 步、类平衡损失、同一折 / 同一训练行，超参不调）；
  域分类器 AUC（CARLA P5 帧 vs nuScenes val 帧，逐层，logistic，按路线 / scene 分组的 5 折，两边各取至多 6 000 帧等量）。
- 维数随层不同（512 → 4096），probe 容量随层变化，这是逐层比较的固有混淆；在结果里注明，`vision`（512）和 `temporal`（512）之间的比较不受影响。

## 判据（照 E0 原文写死）

以 CARLA `pedestrian` scope 的 AUC 点估计为准（CI 一并报），下面的「≥ 0.65」「< 0.60」都是点估计：

- **G-3（信息在 stage 3、stage 4 起丢）**：stage 3 AUC ≥ 0.65 **且**对 `vision` 的配对 Δ CI 下界 > 0 → E2 放 P2（stage 3 后）。
- **G-shallow**：stage 3 < 0.60 → 看 stage 2、1；最浅的一个 ≥ 0.65 的层就是 E2 的位置。
- **G-none**：stage 1、2、3 都 < 0.60 → 「CARLA 行人在 openpilot 前三个 stage 里就没被表示」，特征 adapter 在这一类上判不可行，只剩像素级（Cosmos）或重训更深。

## 原文没写死的地方（照第 55 条的口径选，事前记下）

1. **0.60 ≤ stage 3 < 0.65 的灰区**：原文的两条线之间没有格。读法：不判 G-3（要 ≥ 0.65）；按 G-shallow 去看 stage 2 / 1 是否有 ≥ 0.65 的层；都没有则判「弱，未成立」，不当作位置，也不当作 G-none（G-none 要求三层都 < 0.60）。
2. **「最浅的 ≥ 0.65」**：只在 stage 3 < 0.60 时才启用；层的顺序是 stage 1 < 2 < 3 < 4。stage 4 / `vision` / `temporal` 不参与位置判定（adapter 只能放 stage 3 之前或之后，它们是「哪一层丢」的对照）。
3. **「对 `vision` 的配对 Δ」**：用 stage 3 池化对 512 维 `vision`（第 42 条同一个 tap），而不是对 `temporal`。
4. **原文「stage 4 / vision」**：stage 4 取 `permute_81` 的 mean + max 池化（4096 维），`vision` 单列。
5. **真实侧的标签**：走廊内行人（第 55 条 (a) 主标签），与 CARLA 侧「x⁺ 对 x⁻ 的 hazard 可分性」不是同一个任务，只是同一套 probe 配方；并排读要带着这个限定。
6. **「几乎不要 GPU」的预算**：抽特征只前向 P5 读数帧与 nuScenes val 关键帧，估计 ≤ 0.3 GPU·h，上限 0.5（用卡 6，显存 ≤ 15 GB）。

## 各结局的意思（照原文）

- G-3：CARLA 与真实的差主要在 stage 4 这一段对外观的敏感性，恰好是 op-adapt B 调的参数；第二轮 B 加原始 CARLA 帧大概率能把原始 CARLA 带上来。
- G-shallow（有位置）：adapter 放在那一层；G-none：域差在很浅的层，验收原则里的「sim」应定义成 Cosmos 画面（midterm-gaps Q2.2），文中如实写这个限制。

## 执行日志

- 12:39 冒烟（nuScenes 4 个 scene）：抽特征通；发现 nuScenes 的 key 是每个 key slot 一个 token（不是 token[slot]），改了一行，登记不变。
- 12:40–12:46 抽特征（GPU 6、CPU 48–67：`op-train` 行的 owner 已结束，box 上没有 op_adapt 进程，调度表行仍写 granted，按空闲用；20 个 render worker）：
  nuScenes val 6 019 行 1.3 min，P5 46 703 行 4.2 min（显存峰值远低于 15 GB）。同一次前向算的 stage 3 池化与 trunk 缓存池化的最大差 0.023（fp16 存储的舍入，池化特征量级约 1–20），一致。
- 12:47–12:54 probe：复现检查过（nuScenes val `ped_corr`：stage 3 **0.831**、`temporal` 0.709、`vision` 0.702，与第 55 条逐位一致）。
- 总 GPU 用量约 0.2 GPU·h（登记上限 0.5）。box run：`runs/op_layer/{feats,probe,extract-*,probe}`，小表 [research/results/e0-layer/](../research/results/e0-layer/)。
- 偏离：无。

## 结果

CARLA = P5 v1 BA 行人 4 个 family 的 4 414 对 x⁺ / x⁻（42 条路线，D0 probe，路线聚类 bootstrap 500）；真实 = nuScenes val 走廊行人（6 019 keyframe、203 正例、150 scene，第 55 条 (a) probe）。全部线性 probe。

| 层（池化维数） | CARLA AUC [95% CI] | CARLA Δ 对 `vision` [CI] | 真实 AUC [95% CI] | 真实 Δ 对 `vision` [CI] | CARLA MLP（描述） | 真实 MLP（描述） | 域分类器 AUC（描述） |
|:--|:--|:--|:--|:--|--:|--:|--:|
| stage 1（512） | 0.526 [0.508, 0.556] | +0.018 [0.000, +0.040] | 0.715 [0.628, 0.799] | +0.014 [−0.093, +0.120] | 0.533 | 0.727 | 1.000 |
| stage 2（1 024） | 0.523 [0.504, 0.548] | +0.015 [−0.003, +0.033] | 0.740 [0.644, 0.821] | +0.039 [−0.051, +0.136] | 0.538 | 0.787 | 1.000 |
| **stage 3（2 048）** | **0.523 [0.507, 0.544]** | +0.014 [+0.002, +0.028] | **0.831 [0.758, 0.895]** | +0.130 [+0.049, +0.230] | 0.540 | 0.805 | 0.9998 |
| stage 4（4 096） | 0.522 [0.502, 0.546] | +0.013 [−0.002, +0.028] | 0.748 [0.666, 0.813] | +0.047 [−0.004, +0.121] | 0.518 | 0.769 | 0.9994 |
| `vision`（512） | 0.509 [0.499, 0.520] | — | 0.702 [0.599, 0.785] | — | 0.524 | 0.765 | 0.9992 |
| `temporal`（512） | 0.506 [0.499, 0.515] | −0.003 [−0.008, +0.003] | 0.709 [0.598, 0.799] | +0.008 [−0.140, +0.137] | 0.508 | 0.784 | 0.9944 |

全表（含宽走廊、每个 CARLA family）：[summary.md](../research/results/e0-layer/summary.md)、[carla.csv](../research/results/e0-layer/carla.csv)、[real.csv](../research/results/e0-layer/real.csv)、[domain.csv](../research/results/e0-layer/domain.csv)。

**按登记判格**：stage 3 = 0.523 < 0.60，不判 G-3；stage 2 = 0.523、stage 1 = 0.526，三层都 < 0.60 → **G-none**：
CARLA 行人在 openpilot 的前三个 stage 的（池化）特征里读不出来，特征 adapter 在这一类上判**不可行**（按原文，只剩像素级 Cosmos 或重训更深）。
stage 3 对 `vision` 的 Δ CI 下界 > 0（+0.002）但幅度只有 +0.014，AUC 远低于 0.65，所以不构成 G-3。MLP probe 没有改变结论（0.518–0.540）。
真实一侧 stage 3 最高（0.831），stage 4 与 `vision` 明显掉（−0.08 / −0.13，第 55 条「丢在 stage 4 → 投影 → policy」的复现）；CARLA 上没有这个先升后降的形状，各层平躺在 0.51–0.53。

**读法与限定（结论和推测分开）**：
1. **CARLA 与真实不是同一个任务**：CARLA 侧是 x⁺ 对 x⁻ 的配对可分性（场景完全相同，只差行人）；真实侧是走廊内有无行人，probe 可以借场景上下文（城区、路口、人多的街）。所以真实侧 stage 1 的 0.715 不能读成「stage 1 里有行人」，很可能有一部分是上下文；并排读的是形状（有没有一个在某层升起来的行人信号），不是绝对值差。
2. **probe 管线在 CARLA 上是有响应的（阳性对照，描述）**：同一批 P5 帧上，HighwayCutIn（车辆切入）的配对 AUC 随深度升高：stage 1 0.504、stage 3 0.647、`vision` 0.634、`temporal` 0.762；行人 family 里只有 PedestrianCrossing 在浅层有一点信号（stage 1 / 2 0.632 / 0.611，stage 3 起降到 0.60 以下，`temporal` 0.537，836 对），DynamicObjectCrossing 在 stage 3 有 0.613（609 对），VehicleTurningRoutePedestrian（2 659 对，占行人 scope 的 60%）在每一层都是 0.50。行人 scope 的 0.52 主要是被这一个 family 拉平（推测：行人在画面里太小或太远，池化 mean + max 里没有可分的统计量）。
3. **池化 probe 对小而局部的目标不敏感**：mean + max 池化把整张特征图压成一个向量，一个占几十个像素的行人在 stage 1–2 的 32 × 64 / 16 × 32 图上只影响很少几个位置。真实侧的正例是近处的大行人（≤ 10 m 走廊各层 AUC 0.79–0.92，描述），CARLA 配对里的行人多为中远距离（推测，未按距离分档）。所以 G-none 严格说是「池化特征上的线性 / 单隐层可读性」的结论，**不是**「特征图里一个像素也没有」；这个区别没有测。
4. **域分类器 AUC 在每一层都是 ~1.0**（stage 1 就是 1.000），即 CARLA 与真实帧在最浅层就线性可分，这与「画风差在浅层」一致，但它同样不说明行人信息在不在。

## 后续（不属于本登记，未做）

- 空间分辨的读法（stage 1–3 的特征图上，用 x⁺ − x⁻ 的差图落在行人 mask 内的能量占比；或在 mask 内池化后 probe），以及按行人距离 / 像素高度分档：这是把 G-none 从「池化读不出」收紧到「特征图里没有」所必需的，成本约 CPU 分钟 + 0.1 GPU·h。
