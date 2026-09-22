# P3：backbone 阶梯（换表征，同一个子集、同一套 head、同一个 judge）

状态: running
主题: ../research/survey-counterfactual-video-gen.md

## 目标

第 20 条判到**分支 2**：线性读出这批冻结 Qwen3-VL-4B 特征，在 pre_onset 上已经榨干，
下一步要动表征。P3 就是那一步的开环版本：**换 backbone，别的一律不动**——
同一个分层子集、同一套 head 家族、同一个 judge、同一个 −0.05 m 门槛。

**P3 回答两个问题**：
1. 换表征能不能把 pre-onset 买回来（门槛见下），或者改变 s_ego decile 曲线的形状；
2. **H3 这个 DiT 在 Qwen3-VL-32B 之外还加不加东西**。这一问是 P3 独有的：
   第 21 条记的文献事实是 H3 的理解侧**就是 Qwen3-VL-32B 的第 50 层**，
   所以 (a) 和 (b) 并排跑，两者之差就是「生成侧的 DiT 本身值多少」，
   而这正是 `survey-counterfactual-video-gen.md` 第 4 节里空着的那一格。

## Arms

| arm | 模型 | 许可 | 取什么 | 在盘上了吗 |
|:--|:--|:--|:--|:--|
| **a** | Qwen3-VL-32B-Instruct | Apache-2.0 | 第 50 层（H3 的 conditioner 层）和第 32 层（我们 4B 设定 L18/36 的中层类比）的 mean / last | **是**，76 GB 已在 HF cache |
| **b** | MiniMax H3，fl2va DiT | Community License（排除美/欧/英/韩自托管） | 约 1/2 和 2/3 深度两个 block 的 token mean | 否，要过代理下 |
| **c** | Wan2.2-TI2V-5B | Apache-2.0 | 同上配方，便宜的生成式对照 | 否 |
| **d** | V-JEPA 2 ViT-L | — | 4 帧 clip 到当前帧，pooled | **是**，1.3 GB 已在 cache |

(a) 是判别式的 scaling 对照（同一族、同一套预处理，只是大 8 倍）；
(b)(c) 是生成式；(d) 是第 12 条钉下来的视频自监督对照。

### 每个 arm 的配方

**(a) Qwen3-VL-32B**：**和 `qwen_front3` 同一条预处理路径**——三个相机、同样的
`smart_resize` 到 960×1088、同样的 chat template、batch 钉死 4
（`todos/2026-09-21-waymo-train-features.md` 证明过 batch size 是特征定义的一部分）。
64 层，取第 32 层和第 50 层，各存 `_mean` 和 `_last`，hidden 5120。

**(b) MiniMax H3**（按 `research/lit/2026-09-22-round4-h3-deployment.md` 第 6.1 节）：
- 只要 `transformer/` 的 fl2va 分区，**pruned bf16 约 40 GB**（不含可预计算的 AdaLN 分支）常驻；
  32B 不同时在显存里就够，紧张就换 `pruned_fp8_scaled`（21 GB）
- text conditioner 对一个固定的中性 prompt **预先编码一次存盘**，之后复用
- VAE 编码当前帧。**三个相机是拼成一张 16:9 画布还是一个一个过，跑之前定下来并写进本节**
- 按 DriveLaW 的 t=1 / 2502.07001 的 timestep ≈ 200/1000 加**高噪声**，**只走一次 forward**；
  绕开 diffusers 那套把 `num_frames` 卡到 17n+5、一次 5–15 s 的 blocks，直接调 transformer
- 在约 1/2 和 2/3 深度 hook，token mean-pool 成向量；便宜的话扫两个噪声水平

**(c) Wan2.2-TI2V-5B**：同 (b) 的配方，作为便宜的生成式对照（DriveWAM 的基座，24 GB 可跑）。

**(d) V-JEPA 2**：`jevdrive/features.py` 里已有 `VJepaFeatures`，但它是 64 帧的 `fpc64` 权重。
这里喂 **4 帧 clip**（tubelet 是 2 帧，4 帧 = 2 个 tubelet），
**这是一个已知的 distribution shift，读结果时必须带上**；另外它只吃 FRONT 一路相机，
而主线是 front3，这两个变量都要写进表里（第 12 条的原话：这一行不能读成「谁是更好的 backbone」）。

## 共用协议（和 P2 完全相同，见 [p2 的 todo](2026-09-22-p2-readout-ladder.md)）

同一个 `p2p3_v1.parquet` 分层子集（约 2 万帧：pre_onset / rater / turn_yaw 全要，
其余按 sequence 分层抽样），半/半 sequence 切分与 L0 逐字相同，
`ridge_late` over ego + feature，λ 由 fit 半内 4 折 grouped CV 选，两个方向，paired scene bootstrap。
**arm A（Qwen3-VL-4B pooled）在同一子集上重算**，所以每一个对比都是 like-for-like。
每个 arm 的逐帧预测都存盘，P1 那边定下来的 judge 以后可以直接套用，不用重新拟合。

## 时间与存储预算（跑之前估；每个 arm 先用约 200 帧实测再改这张表）

| arm | ms/帧（估） | 2 万帧墙钟（估） | 特征体积 | 依据 |
|:--|--:|--:|--:|:--|
| a Qwen3-VL-32B | **约 1000** | **约 6 h** | 4 × 5120 × 2 B = 41 KB/帧 → **0.8 GB** | 4B 实测 125 ms/帧（ViT 44.5 + LLM 72.8 + 其余）；32B 的 LLM 约 8 倍参数、跑到第 50/64 层，ViT 也更大 |
| b H3 DiT | 30–100 / forward | 20–60 min（每个噪声水平） | 2 层 × d × 2 B，约 **0.2 GB** | h3-deployment 6.1 的推算，**未实测** |
| c Wan2.2-5B | 同量级 | 20–60 min | 约 0.2 GB | — |
| d V-JEPA 2 ViT-L | 约 25 | 约 10 min | 2 × 1024 × 2 B = 4 KB/帧 → **80 MB** | ViT-L、4 帧、256²，比 Qwen 的 3060 token 小一个量级 |
| 下载 | — | H3 约 40 GB、Wan 约 28 GB，按 docs/network-proxy.md 的 10 MB/s 估 **1.1 h / 0.8 h** | — | 代理链路上限约 18 MB/s 且是共享的 |

**只有 (a) 超过 3 h**，所以只有它按 CLAUDE.md 必须先做 profiling pass；
其余每个 arm 也一律先测 200 帧，把实测写回这张表再开全量。
**任何时候只排一个抽取，不并排两个**，GPU 不空转也不打架。

**失败就记录、不阻塞**：一个模型下不下来、跑不起来，**最多调两小时**，
然后把「具体哪一步失败、报什么错」写进结果一节，换下一个 arm。

## 预写的读表（和 P2 共用，门槛来自第 20 条，一字不改）

**一个 arm「买回了 pre-onset」，当且仅当它的 pre-onset ΔADE ≤ −0.05 m
且 CI 不跨零，在两个方向上都成立。** [−0.05, +0.05] 之内一律读成没有实用增益。

| 结果 | 判定 |
|:--|:--|
| 某个 arm 达标 | 表征层面**确实有救**。按 5b 的预登记读法，这是 **confirm**，值得往更大的基座投 |
| 全集 ADE 改善但 pre_onset 仍在 [−0.05, +0.05]，decile 曲线还是同一条倒 U | **kill**：它买到的和 Qwen3-VL-4B 买到的是同一批中等偏离帧，换基座是换汤不换药。**这个负面结果本身可发表** |
| 全集 ADE 变差而 pre_onset 也没改善 | kill，并且顺带证实 continuation bias 的预测 |
| (b) 达标而 (a) 不达标 | **生成侧的 DiT 确实加了东西**——这正是第 4 节空着的那一格，是 P3 最有价值的一种结局 |
| (a) 和 (b) 同样好 | H3 的增量全在它的理解侧（= Qwen3-VL-32B），**DiT 不加东西**，第 21 条「新东西只在 33B DiT 里」要就地改掉 |

**decile 曲线按形状读**：相对增益在第 4–9 档还升不升、见顶是不是还在第 8 档。
**第 10 档不参与判定**（第 20 条已用 rater 证明那一档的 logged future 本身就有争议）。

**零和检查照第 20 条做**：pre-onset 上赚的必须和 straight_yaw 上亏的并排报。

## 步骤

- [ ] (a) 200 帧 profiling → 全量抽取 → 跑 head
- [ ] (b) 下载 → 工程（两小时上限）→ profiling → 抽取 → 跑 head
- [ ] (c) 同上
- [ ] (d) profiling → 抽取 → 跑 head
- [ ] 数字写进 decisions 第 24 条

## 结果

跑完再填。run dir 见下。
