# E0：CARLA 行人信息在 openpilot 的哪一层消失（逐层 probe）

状态: 进行中（登记写于 2026-09-29 12:40 CST，任何逐层数字之前）
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

（跑之前的检查、偏离、耗时记在这里）

## 结果

（数字出来之前留空）
