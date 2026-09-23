# P3e：多模态 head（K=1024 词表分类、anchor 截断 diffusion）接在 Qwen 原生视频特征上

状态: running（Stage A / A2、profiling、2B 插入已完成；train split 抽取在跑，之后是 train 训的决定性检查）
主题: ../research/prediag-2026-09/README.md

## 目标

P3(d″) 量出来 Qwen3-VL-4B 原生视频输入（`qwenvid_p3`）是整条阶梯里唯一两个方向 CI 都不跨零的表征（第 24 条），
但读它的一直是 `ridge_late`，一个单 mode 的线性 head。本 todo 回答两件事：

1. **Stage A**：把同一批特征换成两种多模态 head，**最高分那一条 mode（top-1）** 在第 22 条口径下相对 ridge 是好是坏。
   head 是 K=1024 固定词表分类（fixed-vocabulary classification，在 K 条 anchor 上打分、取 argmax；
   第 8 条定 K ≥ 1024）和 DiffusionDrive 式的 anchor 截断 diffusion（truncated diffusion：
   从 anchor 加一小段噪声出发，只走两步 DDIM 去噪，同时给每个 mode 打分）。
2. **Stage B**：为 train split 抽同样的视频特征做 profiling，给出全量 / 分层子采样 / 加空间摘要三个选项的小时数和磁盘，
   交给 lead 定范围。**本 todo 不启动全量抽取。**

**先写在前面：v0 大概率是数据饥饿的。** 半 val 的每个方向只有约 9.6k–10k 帧可 fit（其中 pre-onset 约 750 帧），
而 1024 个类别平均每类不到 10 个正样本；P0 在同样的半 val 上分类头就比 ridge 差，要到 train 训（41 万帧）才在 RFS 上追上
（`cls ego` 7.343 / `cls_late` 7.292 对 `ridge ego` 7.056）。所以 **v0 的作用是把管线打通、给出一个下界**，
不是给多模态 head 定性；「head 输给 ridge」在这一步不是反对 head 的证据。

## Setup

- 数据：冻结的分层子集 `p2p3_v1` 与 `qwenvid_p3` 的交集，19 663 帧 / 479 sequence，
  半 val 两个方向（方向 0：fit 9642 帧 / 241 sequence，评 10 021 / 238；方向 1 反过来），与 P3(d″) 那次逐帧同一批行。
- 特征：`qwenvid_p3` 的 `L18_last`（主 tap）和 `L18_mean`（副 tap），2560 维 float16。
- 词表（vocabulary）：**train split 全部 415 663 条 logged future** 上 k-means（k-means++ 初始化，seed 0）。
  train 与 val sequence 不相交，所以用 train 的轨迹建词表不泄漏；比只用 fit 半的 1 万条覆盖好得多。
  分类头用 K=1024；diffusion 用 K=20（DiffusionDrive 的 anchor 数）。
- base：和整个阶梯一样是 `ridge ego`，所有 Δ 都是 arm − `ridge ego`（ADE 负为好，RFS 正为好）。
- 算力：GPU ≤ 30 GB、≤ 8 CPU worker（`RESOURCE_LEDGER.md` 的 heads 流），估计 < 30 min。
- run dir：`$DATA_DIR/runs/waymo_heads/p3e-v0/<ts>/`，tmux 窗口 `heads`。

## Arms（两个方向都跑）

| arm | head | 条件 | 说明 |
|:--|:--|:--|:--|
| `A ridge_late qwenvid L18_last / L18_mean` | ridge_late | ego + 特征 | P3(d″) 的原行，必须逐位复现后才往下读 |
| `cls ego K1024` | 线性 softmax over 1024 anchor，L-BFGS，λ 由 fit 半的 inner split 选 | ego | 分类头自己的 ego-only 对照 |
| `cls_late qwenvid L18_*` | 同上，ego 分类头的 logit 冻结成 offset（late fusion，第 9 / 10 条） | ego + 特征 | 复用 `planner.Heads.cls`，P0 的同一份代码 |
| `diff ego M20` | 截断 diffusion，20 个 anchor | ego | diffusion 头自己的 ego-only 对照 |
| `diff qwenvid L18_*` | 同上 | ego + 特征 | 小：2 层 256 维 transformer 跨 20 个 mode token + 1 个条件 token |

diffusion 配方（DiffusionDrive 的缩小版，跑之前定死）：轨迹按 train split 的 0.5–99.5 分位范围线性归一到约 [−1, 1]；
DDPM 线性 β（1e-4 → 0.02，1000 步），训练时 t ~ U{1..50}，对 20 个 anchor 同时加噪；去噪器输出每个 mode 相对输入的 offset
（得到 x̂₀；跑的过程中改成相对 anchor 的 offset，见结果节）和一个 score；正样本是离 logged future 最近的 anchor，loss = 正样本 x̂₀ 的 L1 + score 上的 cross-entropy。
推理：从 t=50 的加噪 anchor 出发，DDIM（η=0）走 50 → 25 → 0 两步，取 score 最高的 mode。
ego 和特征各自一个 Linear + LayerNorm 投影再相加成条件 token，所以 100 维 ego 不会被 2560 维特征淹掉。
early stopping 与 `waymo_ladder._train` 同一纪律：fit 半的 sequence-grouped inner split 上按 top-1 ADE 选 epoch 数，
再在整个 fit 半上重训那么多 epoch；eval 半只读一次。推理噪声用固定 seed。

## 报告口径

**主比较只用 top-1 mode**，按第 22 条：pre-onset Δ（s_ego 第 1–9 档，ADE vs log）带 sequence bootstrap CI、DiD、
rater 帧上的 RFS Δ，第 10 档单独报 ADE vs rater_best。这样 top-1 和 ridge 在同一把尺子上。
**诊断列只作诊断，不和 ridge 比**：oracle minADE@K（词表里最好的 anchor / head 自己 20 个 mode 里最好的一条）、
minADE@6 / @20（按 score 排序的前 k 个 mode）、trust-region 覆盖缺口（rater 帧里没有任何 mode 落进任一 rater trust region 的比例，
第 8 条要求 K sweep 必报）、以及 pre-onset 上的 oracle minADE（第 9 条说词表系统性亏待 pre-onset）。
再加一张同 tap 的配对表：head top-1 − `ridge_late`（同 tap）在 pre-onset 第 1–9 档上的 Δ，以及视觉增量
（`cls_late` − `cls ego`、`diff qwenvid` − `diff ego`）对 ridge 的视觉增量（`ridge_late` − `ridge ego`）。

## 成功标准（跑之前写死）

先过**复现门**：`A ridge_late qwenvid L18_last / L18_mean` 的 pre-onset 第 1–9 档 Δ 必须与 P3(d″) 的
−0.053 / −0.041（方向 0）、−0.045 / −0.082（方向 1）逐位一致（|差| < 1e-4）。不一致就停下来查管线，下面一格都不读。

然后按主 tap `L18_last` 读，副 tap `L18_mean` 必须给出同一行才算数，两个 tap 分歧就记「tap 依赖、不下结论」：

| 结果（每个 head 家族分别判） | 读法 | 下一步 |
|:--|:--|:--|
| top-1 − `ridge_late`（同 tap）在 pre-onset 第 1–9 档上**两个方向都 ≤ −0.05 且 CI 不跨零** | 读出端是瓶颈，多模态 head 在这点数据上就买回了 ridge 买不到的东西 | train split 抽取的优先级最高，head 直接搬过去 |
| 该差在 [−0.05, +0.05] 内，且 RFS Δ 不比 `ridge_late` 差 0.10 以上 | head 与 ridge 打平；管线通了，v0 给不出方向 | 由 train split 规模决定，不是 v0 |
| 该差 > +0.05（head 更差），**且**同家族 ego-only 对照相对 `ridge ego` 也差同样量级 | 数据饥饿的下界：坏在 head 在 1 万帧上学不会，不在特征 | 不作为反对 head 的证据；train split 之后重测 |
| 该差 > +0.05，**但** ego-only 对照不差（坏只出在加了特征之后） | 特征接进 head 的方式有问题（过拟合 2560 维，或条件太弱） | 先修接法（降维 / 更强正则 / 空间 token），再谈 train split |

附加读法（不改变上表，只报）：视觉增量在 head 家族内部是否 ≤ ridge 的视觉增量（负号 = head 从特征里提得更多）；
diffusion 的 20 个 mode 相对 20 个原始 anchor 的 oracle minADE 降了多少（= 去噪器到底有没有在 refine，
而不只是一个 20 类分类器）；K=1024 词表在 pre-onset 上的 oracle minADE 与 straight 的比值是否仍是第 9 条的 1.7–2.1 倍。

预期（写在跑之前，推测）：两个 head 的 top-1 ADE 都比 `ridge_late` 差，落在第三行；
RFS 上分类头可能打平或略好（第 10 条：分类输 ADE、赢 trust region），diffusion 的 RFS 取决于 score 学得怎样。

## 插入：Qwen3-VL-2B 同样的原生视频输入（lead 2026-09-23 追加，在 train split 抽取之前）

P3(a) 把 32B 量成 4B 的 clean null（「缺的不是容量」），但更小的一端没测过。如果 2B 在 pre-onset 上和 4B 一样好，
train split 抽取就换 2B（更快、更省显存）。配置与 d″ 逐项相同：三相机、4 帧、stride 2、Qwen 自己的 video 通路、
batch 2、不 compile，同一批 19 663 行（`qwenvid_p3` 的行），特征集 `qwenvid2b_p3`。
tap：4B 的 L18/36 是一半深度，2B 有 28 层，对应 **L14**（mean 和 last）；再加一个更深的 **L21**（3/4 深度，mean 和 last）。
读法与 d″ 相同：`waymo_ladder` 的 P3 ladder（ridge_late、同一个子集、同一个 `ridge ego` base），再按第 22 条 rejudge；
4B 的 `L18_last / L18_mean` 在同一次 run 里并排重算，必须逐位复现 d″。同时记 ms/frame 和峰值显存。

**决策规则（跑之前写死，lead 给定）**：2B 取代 4B 做 train split 抽取，当且仅当存在一个 2B tap，
在**两个方向上**都满足：pre-onset（第 1–9 档）Δ 的点估计 ≤ 4B 同类 tap（mean 对 mean、last 对 last）的点估计 + 0.01 m，
**且** CI 不跨零。否则保留 4B。L21 只作为补充行报，不参与这条规则（规则按「同样相对深度」比较；
如果只有 L21 满足，记下来交 lead，不自动换）。

## Stage A2：修 diffusion 的特征接法（Stage A 落第四行后的预写下一步，跑之前写死）

Stage A 里 diffusion 的坏只出在加了 2560 维 pooled 条件之后（inner split 上也一样）。按预写的下一步先修接法，
不碰 head 其余部分，三个变体，每个 tap 各一份：`pca16` / `pca64`（fit 半上标准化后投到前 16 / 64 个主成分再标准化）、
`drop0.5`（全 2560 维，特征输入 dropout 从 0.1 提到 0.5）。
**选择只看 fit 半**：每个方向在三个变体里按 inner split 的 top-1 ADE 选一个，作为「修过的」`diff qwenvid`；eval 半的数全部并排报，
但读法只用被选中的那个。判据：
- 被选中变体的视觉增量（相对 `diff ego`，pre-onset 第 1–9 档）在两个方向上 CI 都跨零或为负 → 接法修好了（特征不再伤 head），
  diffusion 回到第三行（数据饥饿），它的定性等 train split；
- 仍然两个方向都显著为正 → pooled 向量这条路对这个 head 走不通，条件要换成 Stage B 存的 4×4 空间 token（train split 抽完之后）。

## Stage A3：diffusion 改用 4×4 空间 token 条件（半 val，跑之前写死）

A2 判定 pooled 向量这条路走不通，下一步是空间 token。train 抽取已经先把 P3 val 子集（19 663 行）连同 `L18_grid` 抽完了，
所以半 val 上现在就能跑，不必等 train。arm `diff qwenvid L18_grid 4x4`：ego 条件 token + 48 个空间 token（三相机 × 4×4，
每个 token 自己的可学位置编码，Linear 2560→256 + LayerNorm，输入 dropout 0.1）+ 20 个 mode token，其余与 `diff ego` 完全相同。
判据同 A2：它相对 `diff ego` 的视觉增量（pre-onset 第 1–9 档）两个方向 CI 都跨零或为负 → 空间 token 至少不伤 head，
train split 上的 (b) 用它；两个方向都显著为正 → 在 1 万帧上 token 条件也学不会，(b) 仍然跑（数据量才是被测变量），但预期下调。

## 步骤

- [x] `jevdrive/waymo_heads.py`：词表、`cls` / `diff` arm、两方向驱动、诊断列、第 22 条 rejudge
- [x] 小规模 smoke（一个方向、少 epoch）确认形状和耗时
- [x] Stage A 两方向全跑，结果填下面
- [x] Stage B：train split 视频特征抽取的 profiling（batch、attention 实现、dtype、decode worker、I/O 重叠），
      与 `qwenvid_p3` 的重叠行做数值等价核验（`jevdrive/waymo_qwenvid.py profile`）
- [x] Stage B：全量 / 分层子采样 / 加空间摘要三个选项的小时数和磁盘
- [x] 插入：Qwen3-VL-2B 同输入的 P3 行，按上面的规则决定 train split 用 2B 还是 4B（结论：保留 4B）
- [x] Stage A2：diffusion 条件的三个变体（结论：pooled 向量这条路走不通）
- [x] 启动 train split 抽取（thin=4 + 4×4 空间摘要，`qwenvid_train_t4`，2026-09-23 15:11）
- [x] val 子集 93 个 shard 抽完后：`check`（新旧 val 特征的下游等价，通过）
- [ ] 决定性检查 (a)：`trainfit`，train 训、val 评的 d″ ridge_late（第 22 条口径）
- [ ] (b)：多模态 head 在 train 特征上重跑（diffusion 用空间 token 条件）

## 结果

### Stage A（2026-09-23）

run：box 上 `$DATA_DIR/runs/waymo_heads/p3e-v0/20260923-130350/`，小表拉回 `research/results/p3e-heads/`。
单卡与 vlm 流共享，全程约 7 min。

**复现门过了**：两个 ridge 行的 pre-onset 第 1–9 档 Δ 是 −0.05312 / −0.04106（方向 0）、−0.04541 / −0.08215（方向 1），
与 P3(d″) 的记录差 < 2e-5。

**跑之前没写、跑的过程中改了的两件事**（都只看 fit 半，没有看 eval 半的数来选）：
1. `cls_late` 的 ego offset 改成 **cross-fitted**（fit 半内 4 折 sequence-grouped，out-of-fold 的 ego logit）。
   P0 的写法是在全部训练行上 fit 的 in-sample logit，41 万行时无害；1 万行时 ego 分类头选到网格最小的 λ、几乎背下训练行，
   smoke 里 `cls_late` 与 `cls ego` 分不开。
2. diffusion 的输出从 `x_t + offset` 改成 **`anchor + offset`**（`x_t` 只作为条件）。t=50 的截断噪声在这个归一化下是
   半幅的 0.17、x 方向约 8.7 m，原写法要网络自己穿过 256 维瓶颈把噪声抵消掉，在 fit 半的 inner split 上正样本 mode 的 x̂₀
   停在 6.1 m（原始 anchor 的 oracle 才 1.7 m）；改后同一个网络 refine 到 1.15 m。lr 3e-4/60 epoch 与 1e-3/100 epoch 在 inner split 上无差，保持默认。

主表（第 22 条口径，Δ = arm − `ridge ego`；最后一列是同 tap 的 head top-1 − `ridge_late`）：

| 方向 | arm | pre-onset Δ（第 1–9 档）[CI] | straight Δ | DiD | RFS Δ | top-1 − ridge_late（同 tap，pre-onset）|
|--:|:--|:--|--:|--:|--:|:--|
| 0 | `A ridge_late qwenvid L18_last` | −0.053 [−0.092, −0.018] | −0.033 | −0.020 | +0.016 | — |
| 0 | `A ridge_late qwenvid L18_mean` | −0.041 [−0.083, −0.000] | −0.068 | +0.027 | −0.015 | — |
| 0 | `cls ego K1024` | +1.057 [+0.842, +1.279] | +1.312 | −0.256 | −0.287 | — |
| 0 | `cls_late qwenvid L18_last` | +1.056 [+0.825, +1.291] | +1.558 | −0.502 | −0.381 | +1.109 [+0.886, +1.337] |
| 0 | `cls_late qwenvid L18_mean` | +1.020 [+0.794, +1.243] | +1.521 | −0.502 | −0.304 | +1.061 [+0.840, +1.288] |
| 0 | `diff ego M20` | +0.524 [+0.366, +0.682] | +0.240 | +0.284 | **+0.302** | — |
| 0 | `diff qwenvid L18_last` | +1.044 [+0.766, +1.336] | +0.877 | +0.167 | +0.067 | +1.097 [+0.831, +1.386] |
| 0 | `diff qwenvid L18_mean` | +0.937 [+0.612, +1.205] | +0.979 | −0.042 | +0.117 | +0.978 [+0.674, +1.232] |
| 1 | `A ridge_late qwenvid L18_last` | −0.045 [−0.075, −0.016] | −0.022 | −0.023 | −0.005 | — |
| 1 | `A ridge_late qwenvid L18_mean` | −0.082 [−0.144, −0.016] | −0.045 | −0.037 | −0.054 | — |
| 1 | `cls ego K1024` | +1.339 [+1.016, +1.706] | +1.242 | +0.098 | −0.382 | — |
| 1 | `cls_late qwenvid L18_last` | +1.123 [+0.816, +1.444] | +1.285 | −0.162 | −0.416 | +1.168 [+0.870, +1.485] |
| 1 | `cls_late qwenvid L18_mean` | +1.130 [+0.826, +1.456] | +1.274 | −0.144 | −0.378 | +1.212 [+0.927, +1.522] |
| 1 | `diff ego M20` | +0.571 [+0.322, +0.836] | +0.141 | +0.429 | +0.119 | — |
| 1 | `diff qwenvid L18_last` | +0.989 [+0.707, +1.310] | +0.825 | +0.164 | +0.087 | +1.035 [+0.747, +1.344] |
| 1 | `diff qwenvid L18_mean` | +1.201 [+0.883, +1.501] | +0.983 | +0.218 | −0.152 | +1.283 [+0.971, +1.568] |

n：pre-onset 625 / 640，rater 帧 237 / 242。RFS Δ 的 CI 半宽约 0.3；`diff ego` 方向 0 的 +0.302 是唯一 CI 不跨零的正 RFS（[+0.043, +0.536]）。

诊断（只作诊断，不和 ridge 比）：

| 方向 | arm | top-1 ADE | minADE@6 | minADE@20 | 候选池 | 池 oracle minADE（全部 / pre-onset / straight）| rater 帧无 mode 入 trust region | 池无 anchor 入 trust region | mode oracle − 池 oracle |
|--:|:--|--:|--:|--:|:--|:--|--:|--:|--:|
| 0 | `cls ego K1024` | 3.24 | 1.35 | 0.89 | K=1024 词表 | 0.555 / 0.759 / 0.400 | 0.068 | 0.004 | +0.339 |
| 0 | `cls_late qwenvid L18_last` | 3.40 | 1.36 | 0.89 | K=1024 词表 | 同上 | 0.068 | 0.004 | +0.338 |
| 0 | `diff ego M20` | 2.19 | 1.20 | 1.18 | 20 个 anchor | 1.846 / 2.580 / 1.346 | 0.139 | 0.160 | −0.670 |
| 0 | `diff qwenvid L18_last` | 2.77 | 1.40 | 1.37 | 20 个 anchor | 同上 | 0.152 | 0.160 | −0.480 |
| 1 | `cls ego K1024` | 3.17 | 1.32 | 0.86 | K=1024 词表 | 0.543 / 0.824 / 0.397 | 0.066 | 0.000 | +0.317 |
| 1 | `cls_late qwenvid L18_last` | 3.21 | 1.34 | 0.86 | K=1024 词表 | 同上 | 0.066 | 0.000 | +0.318 |
| 1 | `diff ego M20` | 2.11 | 1.13 | 1.11 | 20 个 anchor | 1.813 / 2.705 / 1.337 | 0.120 | 0.199 | −0.700 |
| 1 | `diff qwenvid L18_last` | 2.65 | 1.37 | 1.33 | 20 个 anchor | 同上 | 0.178 | 0.199 | −0.482 |

（`L18_mean` 各行与 `L18_last` 相差 ≤ 0.1，全表在 `research/results/p3e-heads/p3e_diagnostics.csv`。）

视觉增量（head 家族内部 vision − ego，减去 ridge 的 vision − ego；负 = head 从特征里提得比 ridge 多）：

| 方向 | tap | cls（pre-onset / straight）| diff（pre-onset / straight）|
|--:|:--|:--|:--|
| 0 | L18_last | +0.052 [−0.112, +0.233] / +0.279 | +0.573 [+0.292, +0.857] / +0.670 |
| 0 | L18_mean | +0.004 [−0.148, +0.169] / +0.277 | +0.454 [+0.186, +0.723] / +0.807 |
| 1 | L18_last | −0.171 [−0.366, +0.016] / +0.066 | +0.464 [+0.209, +0.727] / +0.706 |
| 1 | L18_mean | −0.127 [−0.324, +0.053] / +0.078 | +0.712 [+0.422, +1.013] / +0.887 |

**按预写的表读（两个 tap 给同一行，算数）**：
- **分类头落第三行（数据饥饿的下界）**：top-1 比同 tap 的 `ridge_late` 差 1.06–1.21 m，两个方向 CI 都远离零；
  ego-only 的 `cls ego` 相对 `ridge ego` 也差 1.06 / 1.34 m，同一个量级。所以坏在 1 万帧上训 1024 类线性 softmax
  （λ 选到网格底、600 步不收敛），不在特征：家族内部的视觉增量和 ridge 的视觉增量打平（pre-onset 四格 CI 全跨零）。
- **diffusion 落第四行（坏只出在加了特征之后）**：ego-only 的 `diff ego` 相对 `ridge ego` 差 +0.52 / +0.57 m，
  但 RFS 反而 +0.30 / +0.12（方向 0 CI 不跨零）；加上 2560 维 pooled 特征后再差 0.45–0.71 m（CI 全不跨零），
  inner split 上也是一样（`diff ego` 1.93 / 2.08 m 对 `diff qwenvid` 2.43–2.59 m）。按预写的下一步：**先修接法**
  （pooled 条件过拟合；降维、更强正则，或者按 Stage B 的空间摘要做 token 条件），再谈 train split。
- 附加读法：diffusion 的去噪器确实在 refine，20 个输出 mode 的 oracle 比 20 个原始 anchor 低 0.48–0.70 m；
  K=1024 词表的 pre-onset / straight oracle 比值是 1.90 / 2.08，第 9 条的 1.7–2.1 倍原样复现；词表对 rater trust region 的
  覆盖缺口只有 0.4% / 0.0%，但分类头实际打出的前 20 个 mode 有 6.6–7.1% 的 rater 帧一个都不入——缺口在 ranking，不在词表（第 9 条的老结论）。
- 预期写的是「两个 head 都比 ridge 差，落第三行」：分类头应验，diffusion 比预期多一层——ego-only 的 diffusion 在 RFS 上是全表最好的，
  坏的是特征条件。

### Stage B：train split 视频特征抽取的 profiling（2026-09-23）

run：box 上 `$DATA_DIR/runs/waymo_qwenvid/profile/20260923-124248/`，小表 `research/results/qwenvid-train-profile/`。
同一批 240 行（`qwenvid_p3` 抽取顺序的前 240 行，batch 两两配对与原 run 相同）。**限定：整段 profiling 期间 vlm 流的 Qwen3-VL-4B
在同一张卡上跑满 100%**，绝对 ms/frame 被抬高了约 1.4–2 倍（同一个 eager batch 2 配置，单独跑时 397 ms，这里 533–790 ms）；
所以下面按「同一时段内的相对值」读，再乘回单独跑的 397 ms。

| 配置 | ms/frame（共享卡，同一时段）| 相对 eager b2 | 峰值显存 | 与 `qwenvid_p3` 逐位相同的行 | `L18_mean` rel L2 / 最小 cos | `L18_last` rel L2 / 最小 cos |
|:--|--:|--:|--:|--:|:--|:--|
| batch 2，eager（d″ 原配置） | 790 | 1.00 | 7.0 GB | **240 / 240** | 0 / 1 | 0 / 1 |
| batch 8，eager | 787 | 1.00 | 13.2 GB | 0 | 3.3e-3 / 0.99990 | 1.3e-2 / 0.99987 |
| batch 2，`torch.compile` | 557 | 0.70 | 6.2 GB | 0 | 7.6e-3 / 0.99974 | 1.4e-2 / 0.99985 |
| **batch 8，`torch.compile`** | **540** | **0.68** | 10.1 GB | 0 | 7.5e-3 / 0.99971 | 1.4e-2 / 0.99986 |
| batch 8，compile + 4×4 空间摘要 | 492 | 0.62 | 9.9 GB | 0 | 同上 | 同上 |

瓶颈（torch.profiler，batch 8，按 CUDA 时间）：**GPU 计算**。eager 下 CUDA 时间合计约等于 wall time；bf16 GEMM 占 36%（compile 后 51%），
flash attention 17%（compile 后 25%，其中 ViT 的 head dim 64 那支是大头），其余是 elementwise，compile 把它们融掉，这就是 0.68 倍的来源。
batch 从 2 加到 8 几乎不省（1.00 / 0.97），说明 GEMM 已经喂饱了。按 FLOP 估：每帧约 37 TFLOP GEMM（ViT 24 层 × 24 480 patch 约 15，
LM 18 层 × 6150 token 约 22）+ 约 25 TFLOP attention（ViT 每路视频 8160 token 的全注意力约 20），合计约 62 TFLOP。
其它几项量过、不是瓶颈：CPU 读 + 解码 + 预处理一个 clip 315–392 ms/核，6–8 个 worker 的供给是 GPU 消耗的 10 倍以上；
cuDNN attention 对 flash 快 1–3%（交替 A/B 三轮，在噪声内），不换；pixel values 在 loader 里转 bf16 **逐位不变**
（patch embedding 的第一步就是转 bf16），省一半 pinned RAM——第一版 profiling 在 batch 12 × 8 worker 时被 OOM kill，就是这 150 MB/item 的 float32。

**数值等价**：原配置（eager、batch 2）逐位复现 `qwenvid_p3`。更快的配置都不逐位（换 batch 或 compile 改的是 kernel），
差异是 rel L2 1e-2 量级、逐行 cos ≥ 0.9997，是 bf16 kernel 重排的量级。因此抽取任务**先用同一配置把 19 663 行 val 子集重抽一遍**
（约占总量 5%），train 和 eval 永远出自同一个数值配方；重抽完先在新 val 特征上重跑 P3(d″) 的 ridge 行，看 Δ 是否在 CI 内不动，再放后面的 shard。
另：eager batch 8 的 `vit_mean` 有一处尺度异常（rel L2 5.6、cos 仍为 1），compile 配置里没有（rel 9e-4），不影响选中的配置，记在这里。

train split 构成（有 future、4 × 2 帧三相机窗口完整的帧，2037 个 sequence）：

| 方案 | train 行 | pre-onset | turn | straight | 其它 | +val 子集后总行数 | 单独跑（270 ms/帧）| 与 vlm 共享（约 540 ms/帧）| 磁盘：pooled 四个向量 | 磁盘：+4×4 空间摘要 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 全量 | 403 049 | 6 654（1.7%）| 42 265（10.5%）| 189 367（47.0%）| 164 763（40.9%）| 422 712 | **31.7 h** | 63 h | 7.4 GB | 111 GB |
| 分层 thin=4 | 137 533 | 6 654（4.8%）| 42 265（30.7%）| 47 378（34.4%）| 41 236（30.0%）| 157 196 | **11.8 h** | 24 h | 2.7 GB | 41 GB |
| 分层 thin=3 | 166 877 | 6 654（4.0%）| 42 265（25.3%）| 63 100（37.8%）| 54 858（32.9%）| 186 540 | 14.0 h | 28 h | 3.2 GB | 49 GB |
| 分层 thin=2 | 225 956 | 6 654（2.9%）| 42 265（18.7%）| 94 649（41.9%）| 82 388（36.5%）| 245 619 | 18.4 h | 37 h | 4.3 GB | 65 GB |

分层规则（写死）：pre-onset 和 turn_yaw 帧全留；其余帧只留 frame index 是 thin 倍数的（thin=4 即每 0.4 s 一帧，Waymo 10 Hz）。
所有 2037 个 sequence 都在。代价：训练集构成变了（thin=4 时转弯从 10.5% 涨到 30.7%、pre-onset 从 1.7% 到 4.8%），
均匀 loss 下 head 看到的是一个偏转弯的分布；要复原全量的目标，给被稀疏的帧 thin 倍的样本权重即可（它们是等距抽样，权重精确）。
第 20 条的教训是加权本身会动别的子集，所以两种都要报。相邻帧高度相关（0.1 s），稀疏直行帧损失的有效样本远少于名义的 3/4，
而第 3b / 24 条说约束是 sequence 数不是帧数，这一条对子采样有利。

空间摘要：每路相机取**最后一个时间位**的 L18 token（decoder 是 causal，后一个时间位已经 attend 过前一个），平均池化到 4×4，
三相机 48 个 token × 2560 维 fp16 = 240 KB/帧。实测不增加时间（在噪声内），磁盘见上表，全量 111 GB 在 150 GB 预算内。

按 lead 的规则（优化后全量 ≤ 30 h 才跑全量）：单独跑的估计 31.7 h 已经超线，而卡在接下来几小时里和 vlm 流共享、实际更慢，
所以默认是 **thin=4 分层子采样 + 4×4 空间摘要**（单独 11.8 h，41 GB）。按 lead 后来的指示，启动前先做 2B 那一行，等放行。

### 插入：Qwen3-VL-2B（2026-09-23）

抽取：`qwenvid2b_p3`，19 663 行，L14 / L21 的 mean 和 last。为了赶时间用的是 compile、batch 8（d″ 是 eager、batch 2；
4B 上同样的换法让特征动 rel L2 约 1e-2、逐行 cos ≥ 0.9997，对 ridge 的影响应远小于 CI）。1 h 29 min，与 vlm 流共享卡。
ladder run：`$DATA_DIR/runs/waymo_ladder/p3-qwenvid2b/20260923-145747/`，4B 的两行在同一 run 里逐位复现 d″。小表 `research/results/p3-qwenvid2b/`。

| 方向 | arm | pre-onset Δ（第 1–9 档）[CI] | straight Δ | DiD | RFS Δ | 第 10 档 vs rater_best |
|--:|:--|:--|--:|--:|--:|--:|
| 0 | 4B `L18_last`（d″） | −0.053 [−0.092, −0.018] | −0.033 | −0.020 | +0.016 | −0.186 |
| 0 | 4B `L18_mean`（d″） | −0.041 [−0.083, −0.000] | −0.068 | +0.027 | −0.015 | −0.181 |
| 0 | 2B `L14_last` | −0.024 [−0.042, −0.006] | −0.019 | −0.005 | +0.027 | −0.068 |
| 0 | 2B `L14_mean` | −0.034 [−0.084, +0.012] | −0.073 | +0.040 | −0.095 | −0.199 |
| 0 | 2B `L21_last`（补充） | −0.031 [−0.068, +0.003] | −0.028 | −0.003 | −0.028 | −0.106 |
| 0 | 2B `L21_mean`（补充） | −0.030 [−0.073, +0.013] | −0.065 | +0.035 | −0.039 | −0.121 |
| 1 | 4B `L18_last`（d″） | −0.045 [−0.075, −0.016] | −0.022 | −0.023 | −0.005 | −0.220 |
| 1 | 4B `L18_mean`（d″） | −0.082 [−0.144, −0.016] | −0.045 | −0.037 | −0.054 | −0.256 |
| 1 | 2B `L14_last` | −0.036 [−0.064, −0.006] | −0.009 | −0.027 | −0.030 | −0.156 |
| 1 | 2B `L14_mean` | −0.083 [−0.143, −0.015] | −0.041 | −0.042 | −0.096 | −0.270 |
| 1 | 2B `L21_last`（补充） | −0.071 [−0.113, −0.029] | −0.013 | −0.057 | −0.025 | −0.081 |
| 1 | 2B `L21_mean`（补充） | −0.063 [−0.124, −0.003] | −0.039 | −0.023 | −0.070 | −0.225 |

**按预写规则：保留 4B。** `L14_last` 方向 0 是 −0.024，比 4B 的 −0.053 差 0.029（门槛 0.01）；`L14_mean` 点估计两个方向都在 0.01 以内
（−0.034 对 −0.041、−0.083 对 −0.082），但方向 0 的 CI 跨零（上界 +0.012）。补充的 L21 两个 tap 方向 0 的 CI 也都跨零，
所以没有「只有 L21 满足」要交给 lead 的情况。一句话：2B 在方向 1 上和 4B 一样好，方向 0 上弱一截，**d″ 仍是唯一两个方向 CI 都不跨零的表征**；
和 32B 的 null 合起来，Qwen3-VL 家族里 4B 是这个量上的甜点，但 2B 与 4B 的差在方向 0 上只有 1–3 cm，不宜读成「2B 更差」的强结论。

吞吐（同一时段 A/B，compile、batch 8、4×4 摘要，320 行交替两轮）：4B **353 ms/帧**、10.0 GB；2B **287 ms/帧**、8.4 GB（0.81 倍）。
2B 省得不多是因为两者的 ViT 完全一样（24 层 × 1024 维，每帧 24 480 patch，约占一半 FLOP），2B 只省在 LM 上。

### Stage A2：diffusion 条件的三个变体（2026-09-23）

run：`$DATA_DIR/runs/waymo_heads/p3e-a2/20260923-151546/`，小表 `research/results/p3e-heads/a2/`。
inner split 选出的变体：方向 0 `L18_last` → `pca64`（inner top-1 2.09 m）、`L18_mean` → `pca16`（2.11）；方向 1 两个 tap 都 → `pca16`（2.29 / 2.30）。
四个都仍比 `diff ego` 的 inner 1.93 / 2.08 m 差。

| 方向 | tap | 选中的变体 | pre-onset Δ vs `ridge ego` | 视觉增量（vs `diff ego`，pre-onset）| RFS Δ |
|--:|:--|:--|:--|:--|--:|
| 0 | L18_last | pca64 | +0.812 [+0.593, +1.032] | +0.375 [+0.166, +0.577] | +0.250 |
| 0 | L18_mean | pca16 | +0.666 [+0.475, +0.855] | +0.216 [+0.035, +0.407] | +0.308 |
| 1 | L18_last | pca16 | +0.968 [+0.697, +1.264] | +0.448 [+0.220, +0.686] | +0.073 |
| 1 | L18_mean | pca16 | +1.028 [+0.745, +1.333] | +0.545 [+0.346, +0.759] | +0.079 |

**按预写的判据落第二行：视觉增量四格 CI 全为正，pooled 向量这条路对 diffusion head 走不通**，条件要换成 train split 抽出来的 4×4 空间 token。
降维确实减轻了伤害（原版的视觉增量 +0.45–0.71，这里 +0.22–0.55），RFS 也比原版好（方向 0 的 pca 变体 RFS Δ +0.25–0.34，CI 不跨零），但没有一格翻到零以下。
另：同一配置两次 run 之间 `diff ego` 的 pre-onset Δ 从 +0.524 变到 +0.490（GPU 非确定性），这个量级的抖动小于表里任何一个结论依赖的差。

### train split 抽取已启动（2026-09-23 15:11）

选择：**4B（2B 规则未过），thin=4 分层子采样 + 4×4 空间摘要**。理由：优化后的全量估计单独跑 31.7 h、按启动时的共享负载是 41 h 以上，
都超过 30 h 的线（新预算下更多显存和 worker 不改变结论：瓶颈是 GPU 计算，batch 从 2 加到 8 只省 3%）。
配置：compile、batch 8、8 个 loader worker、峰值显存约 10 GB；先抽 P3 val 子集的 93 个 val shard（19 663 行），再抽 263 个 train shard（137 533 行），
合计 157 196 行，写到 `features/qwenvid_train_t4/<shard>/`，每个 shard 写完 meta.json 才算完成，重跑自动跳过已完成的 shard。
tmux 窗口 `jev:qv-train`，run dir `$DATA_DIR/runs/waymo_qwenvid/run/20260923-151127/`。
启动后前 13 个 shard 实测 377–447 ms/帧（卡上同时有 vlm 32B-FP8、p4 CARLA 和本流自己的 head 训练），脚本自报 ETA 约 20 h；
vlm 32B 结束后应回到约 270–350 ms/帧，即约 12–15 h。
ledger 行：`2026-09-23 15:11 | heads | jev:qv-train | <=50 GB (fraction-capped; uses ~10 GB) | 8 workers | ~15 h (11.8 h alone) | Qwen3-VL-4B video features, train thin=4 + P3 val subset, 4x4 grid -> features/qwenvid_train_t4 (resumable per shard)`。

### 下游等价核验：通过（2026-09-23 17:39）

run：`$DATA_DIR/runs/waymo_qwenvid/check/20260923-173851/`。新配置（compile、batch 8）抽出的 19 663 行 val 子集对 `qwenvid_p3`：
`L18_last` rel L2 1.4e-2、逐行 cos 最小 0.99981；`L18_mean` rel L2 7.7e-3、cos 最小 0.99898。同一个 P3(d″) ridge_late 行在两版特征上重拟：

| 方向 | tap | pre-onset Δ，`qwenvid_p3` | pre-onset Δ，`qwenvid_train_t4` | DiD（旧 / 新）| RFS Δ（旧 / 新）|
|--:|:--|:--|:--|:--|:--|
| 0 | L18_last | −0.0531 [−0.092, −0.018] | −0.0545 [−0.094, −0.019] | −0.020 / −0.021 | +0.016 / +0.012 |
| 0 | L18_mean | −0.0411 [−0.083, −0.000] | −0.0411 [−0.083, −0.000] | +0.027 / +0.027 | −0.015 / −0.014 |
| 1 | L18_last | −0.0454 [−0.075, −0.016] | −0.0450 [−0.076, −0.015] | −0.023 / −0.023 | −0.005 / −0.015 |
| 1 | L18_mean | −0.0822 [−0.144, −0.016] | −0.0825 [−0.144, −0.017] | −0.037 / −0.037 | −0.054 / −0.055 |

所有 Δ 的变化 ≤ 1.4 mm，比 CI 半宽小一个数量级：快配置是同一个表征，train 侧的抽取可以直接用。
