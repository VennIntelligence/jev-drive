# P3e：多模态 head（K=1024 词表分类、anchor 截断 diffusion）接在 Qwen 原生视频特征上

状态: running（Stage A 在跑；Stage B 只做 profiling，不启动全量抽取）
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
（得到 x̂₀）和一个 score；正样本是离 logged future 最近的 anchor，loss = 正样本 x̂₀ 的 L1 + score 上的 cross-entropy。
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

## 步骤

- [ ] `jevdrive/waymo_heads.py`：词表、`cls` / `diff` arm、两方向驱动、诊断列、第 22 条 rejudge
- [ ] 小规模 smoke（一个方向、少 epoch）确认形状和耗时
- [ ] Stage A 两方向全跑，结果填下面
- [ ] Stage B：train split 视频特征抽取的 profiling（batch、attention 实现、dtype、decode worker、I/O 重叠），
      与 `qwenvid_p3` 的重叠行做数值等价核验
- [ ] Stage B：全量 / 分层子采样 / 加空间摘要三个选项的小时数和磁盘，交 lead 定

## 结果

跑完再填。
