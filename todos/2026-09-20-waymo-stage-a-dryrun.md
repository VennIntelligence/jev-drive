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
- 算力预算: feature 抽取 **0.8–1.4 h**（见下面第 4 节的账），head 训练约 15 min，单卡。
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
  | ego-only → late fusion 加视觉 | **pre-onset** | ADE@5s（**决定性的那一格**，回填 decisions 3d） |
  | ego-only → late fusion 加视觉 | 全体评测帧 | RFS，ADE@5s |
  | ego-only → late fusion 加视觉 | turning | ADE@5s |

  **这三行要并排报，不能只报 pre-onset 那一行。** 一个孤零零的子集数字没法判断它是真信号
  还是子集噪声；全体和 turning 两行提供量级参照，三行的符号是否一致本身就是一个可信度信号。
  | CTRV → ego-only | pre-onset / 全体 | ADE@5s（复现 decisions 3c 的 40 倍差距） |
  | ridge 回归 → 词表分类 | 全体 | RFS，ADE@5s |
  | K=1024 → 4096 → 8192 | 全体 | RFS，trust-region miss，oracle |

  每一行都要带该子集的 **n**，这样 power 不够的时候一眼看得出来。
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

## 步骤

- [ ] 0. 等 val 下载完成（另一个 session 管），`waymo.build_index()` 重建索引，`waymo.check()` 全绿
- [ ] 1. `val_halves()`：按 cluster 分层、按 pre-onset 帧数配平的 sequence-disjoint 对半切；出第 1 节那张表
- [ ] 2. 目标帧集合（2 Hz ∪ rater ∪ pre-onset），报实际帧数，确认在 23 k 量级
- [ ] 3. `extract_features`：`qwen_front3` native，只抽目标帧，存 L19–L22 的 mean（加 vis_mean 做对照）
- [ ] 4. 词表：训练半边 k-means，K ∈ {1024, 4096, 8192}，报 oracle minADE + trust-region 覆盖率
- [ ] 5. Head：cls（late fusion 加 ego+intent）、ridge、以及 ego-only / CTRV / CV baseline
- [ ] 6. 评测：RFS 领头 + ADE 并排 + pre-onset 的 ADE；所有主张引 paired delta + CI（按 sequence bootstrap）
- [ ] 7. A→B 和 B→A 两个方向都跑，两份结果都报
- [ ] 8. **就地回填 [decisions 3d](../research/decisions.md) 的「结果」字段**：
      paired delta、sequence-bootstrap CI、n、CI 半宽、落在预登记表的哪一行；
      不新开条目。同时把 3c 里视觉侧的那一半标上"已在半 val 上预演"

## 成功标准

跑之前写死：

1. **链路**：一条命令从索引跑到 RFS 表，`--horizon 5 --rate 4` 之外不需要改 planner 的代码。
2. **power**：pre-onset 子集上 ΔADE@5s 的 CI 半宽 **< 0.15 m**。
   达不到就说明连 Waymo val 的一半也不够，必须等 train split，这本身就是一个要记下来的结论。
3. **方向性结论**：pre-onset 上 vision-minus-ego 的 ΔADE 按 [decisions 3d](../research/decisions.md)
   那张预登记的表来判，四种读法在拿到数字之前已经定死，**不许事后挑说法**。
   CI 跨零且区间宽只能记为 underpowered，不许写成"视觉没用"。
4. **一致性**：A→B 和 B→A 两个方向的 ΔADE 符号一致，量级在彼此的 CI 内。不一致说明切分或流程有问题。

## 不做

- 不碰网络，不下载任何东西。
- 不抽 train split 的 feature（还没下完）。
- 不做 refinement、不做 temporal head（Stage B 的事）、不做 gate。
- 不提交 test（每 30 天 6 次，等论文的数字出来再用）。

## 结果

（跑完再填。box 上的 run dir 路径写在这里。）
