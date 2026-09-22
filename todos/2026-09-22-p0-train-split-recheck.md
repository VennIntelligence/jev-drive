# P0：train split 复核 3d 和 L0（train 训 head，完整 val 评）

状态: done
主题: ../research/frozen-vlm-planner.md

## 目标

第 1、3d、20 条都以同一句限定结尾：**head 只在 239 个 val sequence 上训过，train split 下完之后要整套重做**。
train 的特征 2026-09-22 抽完了（263/263 shard、415 663 帧、2037 个 sequence），所以现在重做。

问的问题和 3d / 20 一个字都没改，只换训练数据：

> 在 **train split 上训 head、在完整 val 上评**的条件下，第 3d 条的 DiD 还是不是正的？
> 第 20 条的分支 2（「线性读出这批冻结特征在 pre_onset 上已经榨干了」）还成不成立？

**判据不在这个 todo 里**，在 [decisions 3d](../research/decisions.md) 的预登记表和
[decisions 20](../research/decisions.md) 的 0.05 m 门槛 / 分支 1 vs 分支 2 / 零和检查里。
这个 todo 只负责把数字算出来，然后**就地回填**那几条。

## 什么结果改哪一条（跑之前写死）

读法沿用 3d 和 20 已经写死的那两套，下表只说「这次的数字落在哪一格时，哪条条目的状态怎么动」。

| 量 | 这次的结果 | 改哪一条、怎么改 |
|:--|:--|:--|
| DiD（ridge_late，pre_onset − straight_yaw） | 显著为正 | 第 3d 条状态**保持「已确认」**，把限定条件从「半 val 预演」升级为「train 训、完整 val 评」；第 1 条的「已证伪」去掉「待 train split 复核」这个保留 |
| | 不显著、半宽小于 0.05 m | 第 3d 条**降级为「待定」**：半 val 上的显著正号在 10 倍数据下没有复现，写清楚原来是 +0.098、现在是多少；第 1 条改成「证据不足以证伪，也不足以支持」 |
| | 不显著、半宽大于 0.05 m | 落 3d 预登记表**第四行（测不动）**。这不可能发生（n 从 791 涨到约 1500，半宽应该缩小），真发生了说明流程有问题，先停下来查 |
| | **显著为负** | 第 3d 条**翻案**：半 val 的结论是训练数据不足造成的，第 1 条的框架要从「已证伪」退回「待定」，并重新评估项目方向 |
| pre_onset 的 delta 本身 | ≤ −0.05 m 且 CI 不跨零 | 第 20 条的**分支 1 触发**（在 arm A 这一层就触发，比 B/C 还强），第 20 条状态从「已确认（分支 2）」**降级为「待定」**并重写 |
| | 落在 [−0.05, +0.05] | 第 20 条的分支 2 在 10 倍数据下**得到确认**，把限定条件从「半 val」改成「train 训」 |
| arm C（只在 train 的 pre_onset 上 fit，约 10 倍旧 n） | pre_onset ΔADE ≤ −0.05 且 CI 不跨零 | 第 20 条**分支 1 触发**，状态降级重写：线性读出没榨干，是旧的 n 太小 |
| | 仍在 [−0.05, +0.05] | 第 20 条分支 2 **确认**，「下一步动表征」这句话可以去掉「待复核」 |
| arm D（MLP） | 比 A 好 | 第 20 条里「MLP 在 2560 维上 53 000 帧过拟合」这句要**就地改掉**：过拟合是样本量问题，不是 head 问题 |
| | 仍比 A 差 | 该句**升级**：10 倍数据下仍然更差，不是样本量问题 |
| s_ego decile 的相对增益 | 仍是倒 U（第 8 档见顶、第 10 档掉回） | 第 20 条最后一节的形状**确认** |
| | 变成单调上升 | 第 20 条那一节**就地改写**：倒 U 是训练数据不足造成的 |
| RFS（479 个 rater 帧） | 视觉相对 ego 有增益且 CI 不跨零 | 第 3d 条「RFS 上视觉买不到任何东西」**翻案**，就地改 |
| | 仍然无增益 | 该句**确认**，并且样本量从 237/242 涨到 479，CI 半宽应该从 ±0.3 缩到约 ±0.21 |

**零和检查照旧**：pre_onset 上任何增益都要和 straight_yaw 的损失并排报。

## Setup

- 数据：Waymo E2E，`$DATA_DIR/processed/waymo_e2e` 的 snapshot（`scripts/snapshot_processed.sh p0`），
  train 415 663 帧 / 2037 sequence，val 106 360 帧 / 479 sequence / 479 个 rater 帧。
- 特征：`qwen_front3`、`L18_mean`（和 3d / L0 同一个 set 同一层，2560 维 float16），356 个 shard。
- 协议：**只换训练数据，其他一律不动**——同样的 late fusion、同样的 `ridge_cv`
  （4 折 sequence-grouped CV 选 λ）、同样的 paired sequence bootstrap、同样的子集定义。
  切分不再是 `val_halves`，而是 `sp.train = train split 的全部帧`、`sp.val = val split 的全部帧`，
  仍然用 `waymo_stage_a.Halves`，所以 inner selection split 的形状没变。
- s_ego：`waymo_l0.ego_surprise` 原样复用。train 上 out-of-fold（同一套 grouped fold），
  val 上用整个 train 训出来的 ego 模型，因此对 val 的每一帧都是样本外。
- arm：A（均匀 `ridge_late`）、B `ego:a1`（半 val 上 fit 半选出来的那套加权）、
  C（只在 train 的 pre_onset 帧上 fit，n 从约 750 涨到约 12 000）、D（MLP，均匀和 `ego:a1`），
  外加 base `ridge ego`，以及 `cls ego` / `cls_late vision+ego` 两个分类头（RFS 那一行要它们）。
  B 的其余 scheme 和两个 control 族**不跑**：L0 已经判过，sq / top25 把所有子集一起弄坏。
- 算力：**CPU only**。GPU 同时有另一个 agent 在抽特征，这个 run 用 `CUDA_VISIBLE_DEVICES=`
  跑，torch 看不到卡。RAM 控制在 60 GB 以内。
- run dir：`$DATA_DIR/runs/waymo_p0/train_split/<ts>/`，tmux 窗口 `p0`。

## Profiling（2026-09-22，跑之前，CLAUDE.md 要求 3 h 以上先做）

25 核、120 GB 的容器里实测：

| 量 | 实测 |
|---|---:|
| float32 GEMM（gram，50 000 × 2560） | **2.37 TFLOPS** |
| float32 GEMM（X W，50 000 × 2560 × 1024） | **3.01 TFLOPS** |
| 一个 shard 的 `L18_mean` 读入（1528 × 2560 float16） | 0.02 s |

**发现一个 70 倍的坑：`ce_solve` 在 CPU 上被 denormal 拖死。** 分类头的反向是 `x.T @ r`，
`r` 是 softmax 减去目标；拟合到后期 logit 变大，softmax 的绝大多数元素掉进 denormal（次正规数）区间，
x86 上对 denormal 的乘加要走微码，慢两个数量级。同一个 GEMM：

| | 一次 `x.T @ r`（20 000 × 2560 × 1024） |
|---|---:|
| 默认 | **3.12 s** |
| `torch.set_flush_denormal(True)` | **0.043 s** |

所以 CPU 路径开局就 flush denormal。这不改结果（被冲掉的量级在 1e-38 以下，远低于 float32 的有效位），
只改速度。GPU 上没有这个问题，所以 L0 那次没碰到。

### 估时（按实测外推）

| 步骤 | n | 估计 wall |
|:--|--:|--:|
| 载入 `L18_mean`（356 shard，522 k × 2560 float16 ≈ 2.7 GB） | — | 1–3 min |
| s_ego（d=100，4 折 + 全量） | 522 k | < 1 min |
| `ridge ego` + arm A / B / C（每臂 5 次 2560 维 gram + eigh） | 414 k fit | 每臂约 20 s，合计 < 5 min |
| arm D（MLP，40 epoch × 2 次拟合 × 2 个加权） | 414 k fit | 20–60 min |
| decile 表 + paired bootstrap + DiD（1000 次 sequence 重采样） | 106 k eval | < 5 min |
| **`cls ego` + `cls_late`（L-BFGS，9 个 λ）** | 414 k fit | **3–4 h**（主导项） |

**总计约 4–5 h。** 所以拆成两个 step：`arms`（ridge + MLP，约 1 h，先把 3d / 20 的主量写出来）
和 `cls`（分类头 + RFS 那两行，约 3–4 h）。先跑 `arms`，结果一到手就回填条目，不等 `cls`。

**为什么不把 cls 也优化掉**：它已经贴着 CPU GEMM 的屋顶（flush denormal 之后，一次
`fun` 调用的 gemm 占 80% 以上），再快只能减 λ 网格或减迭代，两者都会改变选择出来的模型，
让这次的数字和 3d 不可比。宁可花 4 h。

## 步骤

- [x] profiling：GEMM 吞吐、shard 读入、`ce_solve` 的 denormal 坑
- [x] `planner.ce_solve` 在没有 CUDA 时也能给 λ 分批（commit eba2bc6）
- [x] `jevdrive/waymo_p0.py`：train-fit / full-val-eval 的 CPU 路径
- [x] snapshot 钉住 processed 树（另一个 agent 可能触发 reindex）
- [x] step `arms`：s_ego、A / B / C / D、paired、DiD、decile 表、RFS（回归臂）
- [x] 就地回填 decisions 1 / 3d / 20
- [x] step `cls`：`cls ego`、`cls_late`，补 RFS 表和分类头的 DiD
- [x] 每个 arm 的逐帧预测存盘（`preds.npz`），供 P1 直接换 judge，不重训

## 成功标准

1. **A 必须先复现出一个可读的量级**：train 训出来的 `ridge ego` 在完整 val 上的 ADE 应该
   比半 val 的 1.88 更好（更多训练数据），但不应该好到量级不同。差两倍以上就是流程错了，停下来查。
2. **两个方向不再有了**：train/val 是固定切分，只有一个方向。所以 3d 里「两个方向同号」这条
   一致性检查在这里换成「和半 val 两个方向的区间是否相容」。
3. **判据一个字不改**：0.05 m 门槛、3d 的四行读法、分支 1 / 分支 2、零和检查全部照旧。
4. **回填就地改**：不在旧段落下面追加一段新的，按 decisions.md「怎么维护」那一节的规矩改。

## 不做

- 不碰 GPU，不抽任何特征。
- 不跑 B 的其余 scheme 和两个 control 族（L0 已经判过）。
- 不换层、不换 feature set、不换 K——那是 P2 / P3。
- 不动判据。

## 结果

**2026-09-22 跑完。**
run dir：`$DATA_DIR/runs/waymo_p0/train_split/20260922-175708/`，
snapshot `snapshots/p0-20260922-175427`，CPU、tmux 窗口 `p0`。
小表拉回 `research/results/p0-train-split/`，图 [research/figs/p0-decile-relative-gain.png](../research/figs/p0-decile-relative-gain.png)。

规模：train 415 663 帧 / 2037 个 sequence 训，完整 val 106 360 帧 / 479 个 sequence 评，
pre_onset 1510 帧、straight_yaw 46 580 帧、turn_yaw 11 060 帧、rater 479 帧。

| 量 | 半 val（方向 0 / 1） | **train 训、完整 val 评** |
|:--|:--|:--|
| DiD | +0.098 / +0.029 | **+0.111 [+0.059, +0.162]** |
| pre-onset Δ | −0.027 / −0.043 | **−0.016 [−0.062, +0.030]** |
| straight Δ | −0.125 / −0.072 | **−0.127 [−0.158, −0.099]** |
| 全集 Δ | −0.078 / −0.034 | **−0.068 [−0.085, −0.052]** |
| C 的 pre-onset Δ（n_fit） | −0.036（约 750）/ −0.080 | **−0.016 [−0.089, +0.050]（6786）** |
| D MLP 均匀的 pre-onset Δ | +0.248 / +0.030 | **−0.000 [−0.063, +0.057]** |
| RFS：ego → A | 7.258 → 7.265 / 7.112 → 7.096 | **7.056 → 6.968**（n=479，paired Δ −0.087 [−0.183, +0.004]） |

按预登记的「什么结果改哪一条」表：**DiD 显著为正 → 第 3d 条保持已确认、限定条件升级；
第 1 条去掉「待 train split 复核」**；**C 落在 [−0.05, +0.05] → 第 20 条分支 2 确认**；
**D 仍比 A 差 → 「MLP 过拟合」那句就地修正为「归因于样本量是错的」**；
**decile 仍是倒 U → 第 20 条那一节确认**；**RFS 仍无增益 → 该句确认，n 从 237/242 涨到 479**。
三条都已就地回填。

### 实际 wall time vs 估计

| 步骤 | 估计 | 实测 |
|:--|--:|--:|
| 载入 356 个 shard 的 `L18_mean` | 1–3 min | **6 s** |
| s_ego + ridge ego + A / B / C | < 5 min | **1 min 50 s** |
| arm D（MLP × 2） | 20–60 min | **10 min 35 s** |
| 分类头（`cls ego` + `cls_late`，9 个 λ） | 3–4 h | **3 h 41 min**（ego 45 min，late 2 h 56 min） |

前三项都比估计快，主要是 25 核的 float32 GEMM 实测 2.4–3.0 TFLOPS，比按经验假设的 1 TFLOPS 快三倍。
**MLP 的 early stopping 停在第 1 个 epoch**（40 个里），所以它也比估计快——这本身是个结果，
写进第 20 条了。

分类头的两条结果写进了第 3d 条和第 20 条：
`cls_late` 相对 `cls ego` 的 DiD 是 **−0.240 [−0.618, −0.022]**（符号和 ridge 相反，但半宽 0.298
比效应还大，而且两个分类头在 pre_onset 上的绝对 ADE 都比 `ridge ego` 差，所以是在填词表自己的坑）；
RFS 上 `cls ego` 7.343 / `cls_late` 7.292 **赢过** `ridge ego` 7.056，而 ADE 差 0.14 m——
半 val 上是反过来的，十倍数据之后分类头才追上。
