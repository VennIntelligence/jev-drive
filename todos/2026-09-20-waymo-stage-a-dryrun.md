# Waymo Stage-A 预演：把 val 按 sequence 对半切，先把决定性的那次测量做出来

状态: draft
主题: ../research/frozen-vlm-planner.md

## 目标

回答 [decisions 3c](../research/decisions.md) 那条核心论据在**视觉侧**是否成立：
**在 pre-maneuver-onset 子集（车还没开始转、但接下来要转）上，冻结 VLM 的视觉特征
相对 ego-only 到底买到了什么？** 这是整个项目最关键的一次测量。

nuScenes 那边已经把整条 Stage-A 链路跑通了（[planner v0](2026-09-20-planner-v0.md)），
但 nuScenes 的 pre-onset 子集只有 135 帧，ΔADE 的 CI 半宽 ±0.05 m 而待测差异是 0.034 m，
**在设计上就没有 power**，正反都说明不了。Waymo 同一个子集完整 val 约有 **1750 帧**，
才第一次有资格给出结论。

Waymo train split 还没下载，是剩下最大的一项。所以这一版**把 val 按 sequence 对半切**，
一半训 head、一半评测，先把这次测量做出来，不等 train。

这次测量在 [decisions 3d](../research/decisions.md) 里**已经预登记**：
测什么、带什么限定条件、四种可能结果各怎么读，都在拿到数字之前写死了。

**主量是 difference-of-differences（DiD），不是 onset 上的 delta。**
planner v0 已经证明这两者会给出不同答案：onset 上 ΔADE 显著为负（−0.033，CI [−0.065, −0.003] 不跨零），
但四个子集的 Δ 几乎一模一样（−0.029 / −0.026 / −0.033 / −0.029），
所以"增量集中在 ego prior 最弱处"这个主张**并不能从一个显著的 onset delta 推出来**。要报的是

> **DiD = (pre-onset 上的 vision−ego delta) − (straight 上的 vision−ego delta)**，带自己的 paired bootstrap CI。

DiD 显著为负才支持 [decisions 1](../research/decisions.md) 的框架；
DiD 不显著而两边 delta 都显著为负，诚实的结论就是**"视觉带来一个均匀的小增量"**——
弱得多的故事，论文按那个写，并且要重新评估这个项目值不值得继续。
所以 **straight 子集是对照臂，必须和 pre-onset 一起抽、一起报**，不是可选项。
**跑之前先读那一条**，跑完之后**就地回填它的「结果」字段**（不要新开一条），
填进去的必须包括：paired delta、它的 sequence-bootstrap CI、n、CI 半宽，
以及明确落在那张表的哪一行。如果 CI 半宽大于要分辨的效应量，就直接写 **underpowered**，
和 nuScenes 那次同样处理——**区间宽而不显著不是 null**。

> **这是预演，不是论文的数字。** 论文里 head 必须在 train split 上训、在 val 上评。
> val-internal split 的训练数据少一个数量级，而且和评测半边来自同一批 sequence 的采集条件，
> 所有绝对值都会偏乐观。它能回答的只有"视觉的增量是正的还是零"这个方向性问题，
> 以及整条 Waymo 链路能不能跑通。train 下完之后必须整套重做。

## Setup

- 数据: Waymo Open Dataset E2E（WOD-E2E）val split，479 个 sequence、约 107 k 帧，
  每个 sequence 有且只有一帧带 rater label（固定在 12 s 处，和 test 提交的帧同一个时间点）。
  索引、past/future、rater label 和 scenario cluster 都已经在 `jevdrive/waymo.py` 里。
- Feature: Qwen3-VL-4B-Instruct，完全冻结，`qwen_front3`（FRONT / FRONT_LEFT / FRONT_RIGHT
  三张图一次 forward，native 972×1079，3060 token/帧）。取 **L19–L22 的 image-token mean**
  （[decisions 5](../research/decisions.md)）。
- 目标: 未来 5 s、4 Hz、20 个 XY waypoint。`waymo.py` 里 `future_states` 已经写在
  current rear-axle ego frame（x 前、y 左），**正是提交格式，不需要任何坐标变换**。
- Head: 固定轨迹词表 + 线性分类器，K ≥ 1024（[decisions 8](../research/decisions.md)），
  ego/intent 用 **late fusion**（[decisions 10](../research/decisions.md)），
  另外并排报 ridge 连续回归。
- 对照 backbone（**次要表，不在关键路径上**，见第 5 节）：DINOv2、V-JEPA 2、SigLIP2。
- 算力预算: 主线 feature 抽取 **0.8–1.4 h**（见第 4 节的账），head 训练约 15 min，单卡。
  对照 backbone 另算，**主线 DiD 出来之后再跑**。
- 前提: val 下载完成。下载由另一个 session 管，**本计划不碰网络**。

## 1. 数据切分：sequence-disjoint，且按 cluster 分层

`jevdrive/waymo.py` 新增 `val_halves()`，产出一个确定性的、可复现的对半切分：

- **单位是 sequence，不是 frame。** 同一个 sequence 的帧高度相关，按帧切等于泄漏。
- **按 scenario cluster 分层。** RFS 的榜单聚合方式是"先按 11 个 cluster 各取均值，
  再对出现过的 cluster 做**不加权**平均"（`rfs_by_cluster`）。所以某一半如果缺了一个 cluster，
  它的 RFS 和另一半就不是同一个口径。切分必须保证**两半的 cluster 集合相同**。
- **在每个 cluster 内部按 pre-onset 帧数做贪心配平**（把 sequence 按 pre-onset 帧数降序，
  轮流丢进当前总数较小的那一半，即 LPT 调度）。rater frame 每个 sequence 恰好一帧，
  所以 sequence 数配平了，rater 帧数自动配平。
- cluster 只有一个 sequence 时（如果有），该 cluster 在两半里都留不住，
  **要在报告里显式列出来**，并在 RFS 聚合时说明它被排除了。

切完之后必须报这张表（跑完填）：

| 半边 | sequences | frames | rater frames | pre-onset frames | turning frames | 缺失的 cluster |
|:--|--:|--:|--:|--:|--:|:--|
| A（训 head） | | | | | | |
| B（评测） | | | | | | |

预期量级：每半约 **240 个 sequence、53 k 帧、240 个 rater frame、875 个 pre-onset frame**。
两个方向（A 训 B 评、B 训 A 评）都跑，报两次，这样 479 个 sequence 的 rater 帧全都被评过一次，
RFS 的有效样本从 240 变成 479。**两个方向的结果必须都报**，不能挑好看的那个。

## 2. 要抽哪些帧的 feature（这是省钱的关键）

**不需要全部 107 k 帧。** 目标帧集合是三者的并集，规则确定、和切分无关，所以先算先抽都不浪费：

1. **2 Hz 下采样**：`frame index % 5 == 0`（源是约 10 Hz，`FRAME_DT = 0.1`）。
   约 21.4 k 帧。这是 head 的训练数据和整体评测的主体，和 nuScenes 的 keyframe 频率一致
   （那边 2 Hz、20 k 训练样本，够训 K=1024 的分类器）。
2. **全部 rater-scored 帧**：479 帧。RFS 只能在这些帧上算。
3. **全部 pre-onset 帧**：约 1750 帧。决定性的那次测量在这里，一帧都不能少。

并集约 **23 k 帧**（1 和 3 会有重叠）。

## 3. 评测：RFS 领头，pre-onset 用 ADE，每条主张都引配对差值

按 [decisions 2](../research/decisions.md) 和 [3b](../research/decisions.md)：

- **主结果**：评测半边的 rater 帧上的 **RFS**（`waymo.rater_feedback_score`，官方实现的 bit-exact 移植），
  按 `rfs_by_cluster` 聚合；ADE@3s / ADE@5s（对着**评分最高的那条 rater 轨迹**算，不是对 logged future）
  并排报；ego-only baseline 单独一行。
- **机理**：pre-onset 子集上**只报 ADE**，不报 RFS。这不是样本不够，是结构性的：
  rater 帧固定在 12 s 处，不是按 maneuver onset 时刻挑的，落进 pre-onset 的 rater 帧
  完整 val 也只有 14–35 帧。ADE 这边完整 val 约 1750 帧，够用。
- **每一条比较性的主张都引用 paired bootstrap 的差值和它的 CI**，不引用两行各自的区间。
  重采样的单位是 **sequence**（nuScenes 那边是 scene）。`jevdrive/planner.py` 里的
  `paired()` 已经是这个形状，直接复用，只要把 masks 换成 Waymo 的子集、scenes 换成 sequence。
  要报的对比至少有：

  | 对比 | 在哪个子集上 | 指标 |
  |:--|:--|:--|
  | **DiD = pre-onset 的 delta − straight 的 delta** | — | **ADE@5s（主量，回填 decisions 3d）** |
  | ego-only → late fusion 加视觉 | **pre-onset** | ADE@5s |
  | ego-only → late fusion 加视觉 | **straight**（DiD 的对照臂） | ADE@5s |
  | ego-only → late fusion 加视觉 | 全体评测帧 | RFS，ADE@5s |
  | ego-only → late fusion 加视觉 | turning | ADE@5s |
  | CTRV → ego-only | pre-onset / 全体 | ADE@5s（复现 decisions 3c 的 40 倍差距） |
  | ridge 回归 → 词表分类 | 全体 | RFS，ADE@5s |
  | K=1024 → 4096 → 8192 | 全体 | RFS，trust-region miss，oracle |

  每一行都要带该子集的 **n** 和 **CI 半宽**，这样 power 不够的时候一眼看得出来。
  **DiD 那一行是主量，但它下面四行必须并排报，不能只报 DiD 或只报 pre-onset。**
  单独一个子集的数字没法判断它是真信号还是子集噪声；straight 是 DiD 的对照臂，
  全体和 turning 提供量级参照，几行的符号是否一致本身就是一个可信度信号。

  代码已经就位：`jevdrive/planner.py` 的 `did()` 和 `jevdrive/traj.py` 的 `boot_did()`。
  两个子集共享 sequence，所以 bootstrap 必须**同一次重采样同时作用在两边**（已经这么实现），
  否则会低估相关性、把 CI 算窄。
- **词表**：只用训练半边的 future 做 k-means，评测半边不参与。K 扫 {1024, 4096, 8192}，
  每个 K 报 oracle minADE **和** trust-region 覆盖率（`traj.vocab_coverage`，
  Waymo 上 `region_for(5, 4)` 自动退回官方的 (3 s, 1.0 m) 和 (5 s, 1.8 m)）。
- RFS 本身也要带 **sequence-bootstrap CI**：240 个 rater 帧不算多，而 RFS 有 4.0 的下限，
  分布是截断的，点估计容易骗人。

## 4. GPU 账：抽全部 val 不值，抽目标子集值

`docs/waymo-e2e.md` 在 val 上实测过（256 帧、8 workers、有下载在跑，所以是区间不是点）：

| 设置 | tokens/帧 | ms/帧 | 23 k 帧 | 107 k 帧（全 val） | bytes/帧 |
|:--|--:|--:|--:|--:|--:|
| `qwen_front3` native（默认） | 3060 | 127–216 | **0.8–1.4 h** | 3.8–6.4 h | 47 KB |
| `qwen_front3_l800` | 1725 | 75–134 | 0.5–0.9 h | 2.2–4.0 h | 47 KB |
| `qwen_front` native（只前视） | 1020 | 69–82 | 0.4–0.5 h | 2.1–2.4 h | 47 KB |

**结论：抽目标子集（23 k 帧）而不是全 val（107 k 帧），省 4–5 倍，是明显该做的。**
存储都不是问题（23 k × 47 KB ≈ 1.1 GB）。

**分辨率维持 native front3，不降到 800 px。** 23 k 帧在 native 下也只要 0.8–1.4 h，
省下来的 0.3–0.5 h 不值得冒险：侧视相机正是 turning 帧需要的，而 turning / pre-onset
恰恰是这次测量的全部内容，降分辨率最可能伤到的就是它。`--long-side 800` 留作卡太忙时的 fallback，
`qwen_front`（只前视）留作 ablation，不是默认。

**开跑时机：等 val 下完。** 理由不是怕浪费——目标帧集合的三条规则（`frame % 5 == 0`、rater 帧、
pre-onset 帧）都只依赖单帧自己的 past/future 和 shard 内的信息，**已经在盘上的 shard 现在抽也不会白抽**。
真正的理由是 `waymo.extract_features` 现在把选中的帧写成一个稠密的 memmap，
**没有断点续传、也不能把两次部分抽取合并**；要支持增量就得改代码（给 `<name>.npy` 加一个
frame_name → row 的稳定映射，并在已有 index 里跳过），那点工作量超过它能省下的 1 h。
所以：等 val 完成，一次抽完。如果下载比预期慢很多，再回头做增量那件事。

## 5. 对照 backbone：三个各回答一个问题，但都不在关键路径上

[decisions 12](../research/decisions.md) 把 DINOv3 换掉了（访问申请被作者拒绝，不绕道第三方转存）。
替换的三个都没有门禁，每个回答一个具体问题：

| backbone | checkpoint | 回答什么问题 | 输入 |
|:--|:--|:--|:--|
| DINOv2 | `facebook/dinov2-base` | 纯视觉的单帧自监督能做到多少？（nuScenes 上已有对照） | 单帧 |
| **V-JEPA 2** | `facebook/vjepa2-vitl-fpc64-256` | **时间预训练能不能补上单帧缺的那一块？** | **clip** |
| SigLIP2 | `google/siglip2-so400m-patch14-384` | 增益来自语言对齐，还是来自视觉预训练本身？ | 单帧 |

### V-JEPA 2 怎么喂，以及它测的到底是什么

V-JEPA 2 要的是 **clip 不是帧**（`fpc64` = 每个 clip 64 帧，256×256）。这件事必须摆在明面上：

- **给它的历史窗口，就是我们打算给 Stage B 的那个窗口。** 控制的变量是**时间跨度**，不是帧数：
  用 `waymo.history_rows(df, n_back, stride)` 取同一个跨度（比如 stride 5 = 0.5 s、n_back 7，
  span 3.5 s），再在这个跨度内采满 64 帧喂给它。跨度对齐之后，
  帧率和它预训练时不一致（表观运动速度被改变了），**这是一个已知的 distribution shift，要写进实验细节**。
  另外并排跑一行它的原生 6.4 s clip，让读者看得到多出来的跨度值多少。
- **所以这一行不能读成"V-JEPA 2 是不是更好的 backbone"。** 它比 Qwen 单帧多拿了历史，
  赢了也可能整个是 history 效应而不是 backbone 效应。**唯一能把两者分开的办法是给 Qwen
  同样的窗口**，那就是 Stage B 的事。这一行诚实的读法是：
  **"在 Stage A 的代价下，喂进时间信息到底值不值"**——正好是 Stage B 立不立项的前置证据。
- 相机：只喂 FRONT。64 帧 × 3 相机的解码量不现实，而且 V-JEPA 2 是单视角模型。
  这又是一个和主线（front3）不同的变量，同样要在表里写明。

### 代价（**都是估算，没实测**，跑之前先 benchmark）

| backbone | tokens/样本 | 估计 ms/样本 | 23 k 样本 | 主要瓶颈 |
|:--|--:|--:|--:|:--|
| DINOv2 ViT-B/14，front3 | 3 × ~1000 | ~5 | ~2 min | 解码 |
| SigLIP2 so400m 384，front3 | 3 × 729 | ~20–40 | 0.1–0.3 h | 计算 |
| V-JEPA 2 ViT-L fpc64，front only | ~8192 | ~100–200 | **0.6–1.3 h** | **64 帧 JPEG 解码** |

V-JEPA 2 每个样本要解 **64 张 JPEG**（主线每帧只解 3 张），I/O 是 20 倍。
如果实测下来太贵，先把训练半边降到 1 Hz，**评测侧的 pre-onset 和 straight 两个子集一帧不减**——
DiD 的 power 全在评测侧。

### 排期：主线先跑，对照后补

[decisions 3d](../research/decisions.md) 的 DiD 是关键路径，**对照表不是**。
所以顺序是：先把 Qwen-vs-ego 的主线跑完、把 3d 回填掉，再跑对照。
**如果把 V-JEPA 2 接进来会推迟 DiD，那就先不接。**

## 步骤

- [ ] 0. 等 val 下载完成（另一个 session 管），`waymo.build_index()` 重建索引，`waymo.check()` 全绿
- [ ] 1. `val_halves()`：按 cluster 分层、按 pre-onset 帧数配平的 sequence-disjoint 对半切；出第 1 节那张表
- [ ] 2. 目标帧集合（2 Hz ∪ rater ∪ pre-onset），报实际帧数，确认在 23 k 量级
- [ ] 3. `extract_features`：`qwen_front3` native，只抽目标帧，存 L19–L22 的 mean（加 vis_mean 做对照）
- [ ] 4. 词表：训练半边 k-means，K ∈ {1024, 4096, 8192}，报 oracle minADE + trust-region 覆盖率
- [ ] 5. Head：cls（late fusion 加 ego+intent）、ridge、以及 ego-only / CTRV / CV baseline
- [ ] 6. 评测：RFS 领头 + ADE 并排；**DiD（pre-onset 减 straight）作为主量**，
      加 pre-onset / straight / 全体 / turning 四行并排；所有主张引 paired delta + CI
      （按 sequence bootstrap，两个子集共享一次重采样）
- [ ] 7. A→B 和 B→A 两个方向都跑，两份结果都报
- [ ] 8. **（关键路径的终点）就地回填 [decisions 3d](../research/decisions.md) 的「结果」字段**：
      **DiD 及其 CI**、pre-onset 的 paired delta 及其 CI、两边的 n、
      实际 CI 半宽与 0.037 m 预估的对照、落在预登记表的哪一行；
      不新开条目。同时把 3c 里视觉侧的那一半标上"已在半 val 上预演"
- [ ] 9.（8 完成之后才做）对照 backbone：先 benchmark 三个的实际 ms/样本，再跑 DINOv2 和 SigLIP2
      （单帧，和主线同一套 head 和 split），最后接 V-JEPA 2
- [ ] 10. 对照表：每个 backbone 一行，V-JEPA 2 额外标注它的历史跨度、帧率和只用 FRONT 这三件事

## 成功标准

跑之前写死：

1. **链路**：一条命令从索引跑到 RFS 表，`--horizon 5 --rate 4` 之外不需要改 planner 的代码。
2. **power**：按 [decisions 3d](../research/decisions.md) 的预估，
   pre-onset 约 1750 帧、按 √n 从 nuScenes 的 0.031 m（n=135）缩放是 0.278 倍，
   Waymo 的 ADE 尺度又大约 3 倍，相抵后 delta 的半宽约 **0.026 m**，DiD 再乘 √2 约 **0.037 m**。
   也就是说**只有真实 DiD 明显大于 0.04 m 才判得出来**。
   跑完**必须把实际半宽填进去和这个预估对照**；实际半宽如果和效应同量级，
   那是预登记表的**第四行（测不动）**，不是 null。
   nuScenes 上实测的 DiD 半宽作为标定点写在 [planner v0](2026-09-20-planner-v0.md) 里。
3. **方向性结论**：按 [decisions 3d](../research/decisions.md) 那张预登记的表来判，
   四种读法在拿到数字之前已经定死，**不许事后挑说法**。框架成立与否**由 DiD 决定**：
   DiD 显著为负 → 框架成立；DiD 不显著而两边 delta 都显著为负 → 结论是"均匀的小增量"，
   论文按弱故事写；CI 跨零且区间宽 → underpowered，不许写成"视觉没用"。
4. **一致性**：A→B 和 B→A 两个方向的 ΔADE 符号一致，量级在彼此的 CI 内。不一致说明切分或流程有问题。

## 不做

- 不碰网络，不下载任何东西。
- 不抽 train split 的 feature（还没下完）。
- 不做 refinement、不做 temporal head（Stage B 的事）、不做 gate。
- 不因为对照 backbone 推迟主线：DiD 先出来。
- 不去找 DINOv3 的第三方转存——作者是明确拒绝（decisions 12）。
- 不提交 test（每 30 天 6 次，等论文的数字出来再用）。

## 结果

### 词表表（2026-09-20，**不需要任何 feature，已经跑完**）

run: box 上 `$DATA_DIR/runs/waymo_stage_a/half_val/20260920-152600/`，
一条命令重跑：`scripts/waymo_stage_a.sh`。当时 val 下到 **29 个 shard / 33 208 帧 / 163 个 rater 帧**
（完整 val 是 93 个 shard / 约 107 k 帧 / 479 个 rater 帧），所以**这是彩排，不是最终数**。

切分（cluster 分层、按 sequence 数配平，479 个 sequence 的 cluster 映射从一开始就是全的，所以**这个切分现在就定死了**，
不会随下载变动）：

| 半边 | sequences | clusters | frames | rater frames | pre-onset | straight | turning |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0（fit） | 241 | 10 | 16 704 | 80 | 233 | 7 180 | 1 643 |
| 1（eval） | 238 | 10 | 16 504 | 83 | 256 | 7 363 | 1 791 |

词表在 fit 半边的 logged future 上做 k-means，全部指标在 eval 半边上算：

| K | oracle minADE (m) | oracle minFDE (m) | **没有任何 anchor 能进任一 rater 信任域的帧占比** | oracle minADE / pre-onset | oracle minADE / straight |
|--:|--:|--:|--:|--:|--:|
| 64 | 1.115 | 1.802 | **0.096** | 1.766 | 1.015 |
| 256 | 0.748 | 1.064 | **0.036** | 1.160 | 0.658 |
| 1024 | 0.551 | 0.663 | **0.000** | 0.904 | 0.475 |
| 4096 | 0.440 | 0.456 | 0.000 | 0.725 | 0.361 |
| 8192 | 0.407 | 0.407 | 0.000 | 0.689 | 0.328 |

n：oracle 那几列 16 504 帧，覆盖率那一列只有 **83** 个 rater 帧（RFS 只能在这些帧上算）。

**三条结论。**

1. **nuScenes 上那个"两个口径给出不同 K 结论"的发现，在真的 RFS 几何上复现了。**
   K=64 的 oracle minADE 是 1.115 m，比任何 head 实际能做到的都好得多，看上去够用；
   但 **9.6% 的 rater 帧无论选哪条 anchor 都进不了任何信任域**，直接被压到 4.0 分。
   K=256 还有 3.6%，K=1024 降到 0/83。所以 **K ≥ 1024** 这条（[decisions 8](../research/decisions.md)）
   现在有 Waymo 自己的证据，不再只是从 nuScenes 外推。
2. **0/83 不等于 0。** 83 帧里 0 个失败，单侧 95% 上界是 **3.5%**（1 − 0.05^(1/83)）。
   完整 val 有 479 个 rater 帧、eval 半边约 240，那时才能把上界压到 1% 左右。这一格要重算。
3. **词表对 pre-onset 子集系统性地更差**：每个 K 上 pre-onset 的 oracle minADE 都是 straight 的
   **1.7–2.1 倍**（K=8192 时 0.689 对 0.328）。也就是说"还没起手但要转"的未来轨迹离任何簇心都更远——
   它们本来就是少数派，而 k-means 的目标是总体均方位移，会主动牺牲它们。
   **这是一个设计杠杆，不只是一个观察**：如果最后 pre-onset 是论文的主战场，
   词表可以按子集重采样之后再聚，或者给 pre-onset 单独留一部分 anchor。
   代价是整体 oracle 变差。**还没试，属于推测**，验证方法是同样的表加一行"按 pre-onset 重采样后聚类"。

Waymo 的信任域比 nuScenes 那个 stand-in 宽松（K=64 时 9.6% 对 16.2%），符合预期：
Waymo 有 3 条 rater 轨迹可选，阈值在 5 s 处也更大。两个数不能直接比，只能各自内部比。

### 还没跑的

- feature 抽取：`jevdrive.waymo features_inc` 正在按 shard 增量跑（见下），完了才能跑 head。
- 整条 entry-3d 的彩排（head + RFS + DiD）：等 feature。
