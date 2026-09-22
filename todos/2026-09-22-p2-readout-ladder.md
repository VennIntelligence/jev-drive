# P2：读出阶梯（同一个 Qwen3-VL-4B，换输入和换 head）

状态: running
主题: ../research/frozen-vlm-planner.md

## 目标

第 20 条落在**分支 2（representation 层面）**，但那个判定有一个自己承认的缺口：
L0 用的全部是 **pooled**（把整帧的 image token 平均成一个 2560 维向量）特征，
而且只喂**当前一帧**。所以「线性读出已经榨干」这句话，严格说只对
「pooled 向量 + 单帧 + ridge」这一个组合成立。P2 把这个组合的三个维度各松一格：

| 松哪一格 | arm | 需要什么 |
|---|---|---|
| 输入的时间 | (b) 把当前帧和更早几帧的 pooled 向量拼起来 | 无，盘上每一帧都有特征 |
| 输入的空间 | (c) 存 spatial token grid（空间 token 网格），配 attention head | 重抽一个子集 |
| 读出的形式 | (c) 里的 attention pooling / transformer head，对照 compute-matched MLP | 同上 |
| 两者一起 | (d) (b)+(c) | 同上 |

**P2 回答的问题**：pre-onset 上的失败到底在 readout、在输入，还是在表征。
如果一个普通的 attention head 就把 pre-onset 买回来了，那第 20 条的分支 2 要被推翻，
归因要改成「pooling 太弱」，而不是「Qwen3-VL 的表征里没有这个信息」。

## Setup

- 数据: Waymo E2E val，93/93 shard、106 360 帧。特征集 `qwen_front3`（pooled，L{09,18,27,36}_{mean,last}）
- 层: `L18_mean`，和第 5 条、`waymo_stage_a.LAYER` 一致
- 协议: 完全照抄第 3d / 20 条——val 按 sequence 对半切（`waymo_stage_a.val_halves`，seed 0），
  一半 fit 一半 eval，λ 用 fit 半内部 4 折 sequence-grouped CV 选，两个方向都报，paired scene bootstrap
- 判据: 第 20 条的 −0.05 m 实用门槛，一字不改（见下面的读表）
- 算力: 一张 RTX PRO 6000 Blackwell。另一个 agent 在同一台机器上跑 CPU-only 的 P0/P1，
  所以 RAM 控制在 60 GB 以内，用自己的 tmux window 和 run dir

## 共用的分层子集（P2 和 P3 共用，选一次就冻结）

P3 要重抽四个 backbone 的特征，按 106 360 帧算不现实（32B 单这一个就是 18 h）。
所以 P2(c) 和整个 P3 共用**同一个分层子集**，选定之后写进文件，谁也不许再动：

- pre_onset 全要（这是主判据的分子）
- rater 帧全要（RFS 的全部样本）
- turn_yaw 全要（第 3c 条的对照臂）
- 其余的 straight / all 帧按 sequence 分层抽样补到约 2 万帧

**半/半的 sequence 切分和 L0 逐字相同**，所以 arm A 可以在完全同一批帧上重算，
每一个对比都是 like-for-like。子集文件存在 box 上
`$DATA_DIR/processed/waymo_e2e/subsets/p2p3_v1.parquet`，
计数和 sha256 提交到 `research/results/p2p3-subset/`。

**s_ego（第 20 条的主定义 kinematic surprise）的算法保持不变**：
ego ridge 在 fit 半上用 sequence-grouped inner fold 算 out-of-fold，eval 半用整个 fit 半训出的模型。
为了让 decile 轴和第 20 条的那张表可比，**s_ego 在完整 val 上算，再限制到子集**，
不在子集内部重算。arm 自己的 ego base 则在子集的 fit 半上拟合，这样各 arm 之间是配对的。

## Arms

| arm | 输入 | head |
|:--|:--|:--|
| **A** | L18_mean（2560） | ridge_late，和 L0 的 arm A 同一套 |
| **b2 / b3** | L18_mean 在 4 个时间槽上拼接（10 240 维） | ridge_late |
| **c-attn** | L18 token grid（3 相机 × 6×8 = 144 token × 2560） | 单 query attention pooling → 线性 |
| **c-tf** | 同上，外加 ego state 当一个 token | 2 层小 transformer |
| **c-mlp** | L18_mean（2560） | compute-matched MLP（对照，和 L0 的 arm D 同族） |
| **d** | b + c 一起 | attention pooling + 时间拼接 |

### (b) 的帧间隔：10 Hz 下取不到正好 0.25 s

第 14 条测出来 Waymo 的 `context.name` 帧间隔是 **10.00 Hz**，也就是 `FRAME_DT = 0.1 s`。
所以「0.25 / 0.5 / 0.75 s 以前」这三个时刻**没有整数帧对应**（−2.5 / −5 / −7.5 帧）。
不去做时间插值（那会造出盘上不存在的特征），改成**两个等间隔的 stride 各跑一遍**：

- **b2**：stride 2 帧 = 0.2 s，窗口 0 / −0.2 / −0.4 / −0.6 s
- **b3**：stride 3 帧 = 0.3 s，窗口 0 / −0.3 / −0.6 / −0.9 s

两个窗口把 0.25–0.75 s 这个区间夹在中间，而且每个槽都是盘上真实存在的那一帧。

**窗口完整性按第 13 条办**：只用每个槽都精确命中的帧（`waymo.history_set`），
而且**整张表**（arm A 也在内）都限制到这批帧上，否则各行看的不是同一批数据。
(b) 不需要重抽任何特征，所以它**不用子集，跑完整的半 val 协议**，是 P2 里样本量最大的一行。

### (c) 的网格怎么选、占多少盘

Qwen3-VL 的 `smart_resize` 把 972×1079 的相机图放到 **960×1088**，patch 16、spatial merge 2，
所以每个相机的 merged token 网格是 **30 × 34 = 1020 个 token**，三个相机合计 3060，
和 `todos/2026-09-21-waymo-train-features.md` 里数出来的 3060 个图像 token 对上。

直接存 3060 × 2560 的 fp16 是 **15.7 MB/帧**，2 万帧就是 **314 GB**，不可接受。
所以每个相机的 token map 用 `adaptive_avg_pool2d` 压到 **6 × 8 = 48 个 token**：

| 量 | 值 |
|---|---:|
| 每帧 token | 3 × 48 = **144** |
| 每帧字节（fp16，2560 维） | 144 × 2560 × 2 = **737 KB** |
| 2 万帧合计 | **约 14.7 GB** |

在 50 GB 的预算之内，而且 30/6 = 5、34/8 = 4.25，`adaptive_avg_pool2d` 处理非整除没有问题。
同一次抽取**同时存 pooled 的 L18_mean**，用来和盘上已有的 `qwen_front3` 逐位核对：
如果不一致，说明这条新路径改了特征定义，(c) 和 A 就不可比，必须先修好再往下跑。

## 时间与存储预算（跑之前估，跑完就地更正）

| 步骤 | 估计 | 依据 |
|---|---:|---|
| (b) ridge，10 240 维，2 个 stride × 2 方向 × (5 次 eigh + gram) | **约 1 h** GPU | gram 是 5×10⁴ × 10 240²，约 1.0e13 FLOP，TF32 下约 100 s 一次 |
| (c) 抽 20k 帧的 token grid | **约 35 min** GPU | 现有 pipeline 125 ms/帧跑到 L36；只跑到 L18 是真正的 early exit，LLM 那半省一半，估 90–100 ms/帧 |
| (c) attention / transformer head | **约 1.5 h** GPU | 14.7 GB fp16 整个放进显存，一次 epoch 是 1 万行 × 144 token 的 attention，秒级 |
| (d) | 约 30 min | 同上 |
| **合计** | **约 3.5–4 h** | 单个 arm 都不到 3 h，按 CLAUDE.md 只需在 (c) 抽取前做一次 200 帧的 profiling |
| 存储 | grid 特征 **约 15 GB**，pooled 增量可忽略 | 盘上还有 1.3 TB |

## 预写的读表（第 20 条的门槛一字不改）

**实用效应门槛：pre-onset 的 ΔADE 要 ≤ −0.05 m，且 CI 不跨零，在两个方向上都成立，
才算「买回了 pre-onset」。** 落在 [−0.05, +0.05] 之内的一律读成「没有实用增益」。

| 结果 | 判定 | 下一步 |
|:--|:--|:--|
| (c) 的 attention / transformer head 达标，(c-mlp) 对照不达标 | **第 20 条的分支 2 被推翻**。归因改成「**pooling 太弱**」，不是「表征里没有」 | 主线转到 readout：spatial token + attention，P3 降级成补充对照 |
| (b) 达标而 (c) 不达标 | 缺的是**时间**不是空间。这和第 12 条对 V-JEPA 2 的读法（「在 Stage A 的代价下喂时间值不值」）连上 | 主线转到时序输入，Stage B 立项 |
| (b) 和 (c) 都达标 | 输入不够，两个维度都不够 | (d) 的数字决定先做哪个 |
| 全都留在 [−0.05, +0.05] | **分支 2 加强**：换 head、换输入都没用，确实要换表征 | P3 的结果成为唯一希望 |
| c-mlp 也达标 | 那不是 attention 的功劳，是容量的功劳，而 L0 的 arm D（同族 MLP）在完整特征上是**变差**的，所以这种情况要先查是不是子集选择造成的 | 在完整半 val 上复跑 c-mlp 再判 |

**decile 曲线按形状读，不按某一档的数值读**：问的是**相对增益在最高几档是不是不再下降**。
第 20 条测到的形状是倒 U——第 8 档最好（−7.8%），第 10 档掉回 −3.6%。
其中第 10 档那一段**不参与判定**（第 20 条已经用 rater 证明那一档的 logged future 本身就有争议，
RFS 5.92、41.7% 被压到下限，「对着 log 算 ADE」在那里不是好目标）。
**要看的是第 4–9 档那段上升还在不在、见顶是不是仍然在第 8 档。**

**零和检查照第 20 条做**：pre-onset 上赚的必须和 straight_yaw 上亏的并排报。

## 步骤

- [ ] 冻结共用子集，写 `p2p3_v1.parquet`，计数提交
- [ ] (b) 完整半 val 上跑 b2 / b3 + 同帧集上的 arm A
- [ ] (c) 200 帧 profiling，核对 pooled 逐位相同
- [ ] (c) 抽 2 万帧 token grid
- [ ] (c) attention / transformer / MLP 对照三个 head
- [ ] (d)
- [ ] 数字写进 decisions 第 23 条

## 结果

跑完再填。run dir 见下。
