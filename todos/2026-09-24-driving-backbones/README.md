# 驾驶专用 backbone 进 P3 阶梯：openpilot 与 Alpamayo 1.5 的冻结特征 + 我们的 ridge head

状态: running（预登记 2026-09-24，在抽任何特征之前提交）
主题: ../../research/prediag-2026-09/README.md（P3 backbone 阶梯）、../../research/frozen-vlm-planner.md
上游: [zero-shot WOD 考试](../2026-09-24-zeroshot-exam/wod-e2e.md)（第 34 条）、[P3 阶梯计划](../2026-09-22-p3-backbone-ladder.md)（第 24 条）

## 问题

第 34 条量到：两个驾驶专用模型用它们**自己的**输出头 zero-shot 就在 WOD-E2E val 的 rater 帧上拿到 RFS（Rater Feedback Score，
预测轨迹落在三条人工打分轨迹 trust region 里的得分，0–10）8.01（openpilot Cinque）和 7.86（Alpamayo 1.5），
高于我们在 WOD train 上训的最好 head（`cls ego` 7.31）。这可能来自两件完全不同的事：
(1) 它们的**表征**里有我们的通用 backbone（Qwen3-VL、V-JEPA 2 等）没有的驾驶信息；
(2) 表征差不多，赢在**输出头**（多模态 / 具体 mode 的解码、闭环数据训练的 plan head）。
本实验把它们当冻结特征抽取器，接进 P3 阶梯**原封不动**的 head 和 judge，直接回答：
在我们的主线（冻结特征 + 薄 head）里，驾驶专用模型的特征是否比阶梯里已有的通用 backbone 更好。

## 协议：和 P3 一字不改

| 项 | 取值（全部沿用第 23/24 条） |
|:--|:--|
| 帧 | 冻结子集 `p2p3_v1`：20 237 帧 / 479 sequence，sha256(frame names) = `6e6c17b8…7cc5`；pre-onset 1510、rater 479、straight 11 975、turn 4461 |
| 切分 | 第 3d 条的 sequence 不相交半/半 val（seed 0），两个方向各训一次、各评另一半 |
| head | `ridge_late`（先拟合 ego ridge，再用 ridge 在 ego + feature 上拟合它的残差）；ego = 运动学状态 + intent one-hot；λ 在 fit 半内 4 折按 sequence 分组 CV 选；特征用 fit 行的统计量标准化 |
| judge | 第 22 条口径（`waymo_ladder.rejudge`）：主判 pre-onset（车还没开始转、运动学看不出意图的帧）上 ADE vs log 的 Δ，限 s_ego（ego-only 读出自己的 5 s 残差）第 1–9 档；straight Δ、DiD（pre-onset Δ 减 straight Δ）并排；子集 rater 帧上的 RFS Δ；第 10 档单独用 RFS 和 ADE vs rater_best；全集 ADE 只作侧栏 |
| 统计 | 按 sequence 重抽的配对 bootstrap；Δ 一律是 arm − `ridge ego`（ADE 负为好，RFS 正为好） |
| 代码 | 新特征写成和已有 P3 set 相同的平铺格式（`features/<set>/index.parquet + <array>.npy`），然后 `waymo_ladder` 的 `p3` 步骤 + `rejudge` 原样跑 |

**同一张表里的所有 arm 取覆盖的交集**（第 13 条）。V-JEPA 2 和 Qwen 原生视频需要严格完整的 4 帧 stride-2 窗口（19 663 帧），
Alpamayo 要 f−3…f，openpilot 要 f−2，所以主表在 **19 663 帧**上跑，arm A、V-JEPA 2 (d)、Qwen 视频 (d″) 在同一批帧上重算，每一行直接可比。

**追加一个读数（预登记，不改原读数）**：两个方向的 eval 半互不相交、合起来正好是整个子集，所以把两个方向的样本外预测拼起来，
每个 rater 帧恰好被评一次（cross-fit）。在这 479 个 rater 帧上报 RFS（frame mean 与榜单口径 cluster mean）、
在全部 pre-onset 帧上报第 1–9 档的 Δ，都用配对 sequence bootstrap。这比单方向的 n 大一倍，是本实验判定用的读数；逐方向的表照旧并排。

## 模型与特征（抽之前写死）

### openpilot：small（30M）、Cinque v3（382M，`f78ed37d`）、Lebowski（877M，0.11.2）

输入与 WOD 考试完全相同：FRONT / FRONT_LEFT / FRONT_RIGHT 纯旋转重投影成 road / wide 两路 calib-frame model frame（最近邻，
JPEG 自带 YCbCr，色度 2×2 均值），每个 WOD 帧喂两次得到 20 Hz，desire 全 0，traffic convention [1, 0]，action_t (0.275, 0.525)；
后端同考试（small TRT 图精度，Cinque / Lebowski TRT fp16）；Lebowski 按考试的偏离 1 用 `context_rate` 在输出相位上逐 0.2 s 走一步，数值等价。

**唯一的差别是时间上下文的给法**。考试对每个目标帧从零状态暖机 10 s；子集是 2 万帧、每个 sequence 平均 42 帧，逐帧暖机要 400 万步。
这里改成**每个 sequence 从它在 index 里的第一帧开始、零状态、一路流到它最后一个子集帧**，途经的子集帧就地取特征，
和车上 openpilot 从接通起一直跑的方式一样。Lebowski 的两个相位（奇偶帧）各起一条流。3 个 val sequence 的 index 里有缺帧，缺口处重置为零状态。
这样每个子集帧的历史是「它在 sequence 里之前的全部帧」：中位 10.9 s，23% 的帧不到 4.8 s（feature 队列没填满，和车上刚接通时一样）。
**不按历史长度剔帧**；历史不足 4.8 s 的帧另报一行敏感性（同样限制下重算 arm A）。

特征 tap（在 ONNX 图里把已有的中间张量**追加为图输出**，不改任何计算；等价性见下）：

| tap | 是什么 | small | Cinque | Lebowski |
|:--|:--|--:|--:|--:|
| **`temporal`（主 tap）** | off-policy temporal summarizer 在当前步的输出 token，也就是 plan head 直接读的那个向量 | `model/p_select` 512 | `select_4` 512 | `select` 512 |
| `vision` | 当前步 vision encoder 的 pooled 输出（进时间模块之前） | `model/view` 1024 | `mean` 512（32 个 vision token 的均值） | `view_40` 3072 |
| `hidden` | 进 feature 队列的那个向量（输出里的 `hidden_state`） | 512 | 16 384（= 32 × 512 vision token 本身，`vision` 的未池化版） | 512 |
| `plan` | 原生 plan 的 MDN 均值（33 个时刻 × 15 维） | 495 | 495 | 495 |

`plan` 这一行问的是「我们的 ridge 在它自己的 plan 上还能校准出多少」，不是表征问题，单独标注。
另外把每个子集帧的原生 plan 按考试的换算转成 WOD 的 20 × 0.25 s 轨迹，**不拟合任何东西**，在同样的 eval 帧上和 `ridge ego` 配对比，作为参照行。

### Alpamayo 1.5（10B，`nvidia/Alpamayo-1.5-10B` @ `7aba829`）

**相机映射（预登记的近似）**：模型要 cross-left / front-wide / cross-right / front-tele 四路 f-theta 视图。box 上的 WOD 帧只有前三路，
所以四路视图用考试同一个纯旋转重投影（`jevdrive.camgeom`，双线性，看不到的像素填黑）**只从 FRONT / FRONT_LEFT / FRONT_RIGHT** 渲染。
前三路相机覆盖 yaw 约 −68° … +68°：front-tele 完全有像，front-wide 的左右边缘和底部带子缺，cross-left / cross-right 大约只有内侧一半有像
（考试里 7 路相机时 cross 视图覆盖 0.75，这里预计降到 0.4 左右，实测补进结果）。
**不为子集去 GCS 抓侧面相机**：f−3…f 覆盖子集要 54 036 条唯一记录，按考试实测每条约 2.3 MB、约 95 条/分钟，是约 124 GB、约 9.5 h 的网络，
超出这一轮的预算。缺口的影响用考试已经抓好的 479 个 rater 帧的 7 路相机包来量（下面「附带量」）。

其余输入同考试：每路 4 帧（f−3…f，10 Hz），egomotion 历史用 `wod_zeroshot.alpamayo_history`，**no-nav prompt**
（考试里 nav 没有作用；intent 已经在 ego 特征里，给文本会重复计入）。

**只做 prefill**：把融合了 ego 历史 token 的完整 prompt 过一遍 `vlm.model`（不接 lm_head、不生成 reasoning、不跑 flow matching expert），
在 forward hook 里按阶梯的约定池化，和 `QwenFeatures` 的定义一致：

| array | 定义 |
|:--|:--|
| `vit_mean` | vision merger 之前的 ViT token 均值（16 张图全部 patch） |
| `vis_mean` | merger 输出（进 LM 的 image token embedding）的均值 |
| `L{9,18,27,36}_mean` | 第 k 个 decoder layer 输出在全部 image token 位置上的均值 |
| `L{9,18,27,36}_last` | 第 k 层在 prompt 最后一个 token 上的 hidden state |

Alpamayo 的 VLM 是在 Cosmos-Reason2-8B（Qwen3-VL-8B 架构，36 层、hidden 4096）上继续训的，所以 **主 tap 定为 `L18_mean`**：
和 arm A（Qwen3-VL-4B 的 `L18_mean`，36 层里的第 18 层、同一种 image-token 均值）同族、同相对深度、同池化，差别是驾驶数据的训练（外加 8B 对 4B、相机和帧数不同）。
其余 tap 是次要读数，按多重比较对待。

### 对照行（同一次 `p3` 运行、同一批 19 663 帧）

`ridge ego`（base）；A Qwen3-VL-4B `L18_mean`；(d) V-JEPA 2 ViT-L `mean`；(d″) Qwen3-VL-4B 原生视频 `L18_last` / `L18_mean`；
上面三个 openpilot 各 4 个 tap；Alpamayo 的 10 个 array；late fusion 两行（预登记）：Cinque `temporal` ‖ A、Alpamayo `L18_mean` ‖ A。

## 判据（跑之前写死）

**第 20 条门槛照旧**：pre-onset Δ（第 1–9 档）≤ −0.05 m 且 CI 不跨零，两个方向都成立 → 「买回了 pre-onset」。

**本实验的主问题「驾驶专用特征是否比通用特征好」**：每个模型只用它的**主 tap**（openpilot 三个 `temporal`、Alpamayo `L18_mean`）做判定，
和两个通用参照逐帧配对：arm A（`L18_mean`）和 (d″) `L18_last`（阶梯里唯一两个方向 CI 都不跨零的通用行）。三个读数，全部是上面的 cross-fit 合并读数：

| 读数 | n | 越小/越大越好 |
|:--|--:|:--|
| (i) pre-onset，第 1–9 档，ADE vs log | 约 1250 | 小 |
| (ii) 全部帧，第 1–9 档，ADE vs log | 约 17 700 | 小 |
| (iii) rater 帧 RFS（frame mean；cluster mean 并列） | 479 | 大 |

| 结果 | 判定 |
|:--|:--|
| 对**两个**通用参照，至少一个读数的配对 Δ CI 整体偏向驾驶特征，且没有任何读数的 CI 整体偏向通用特征 | **更好** |
| 对称地，至少一个读数的 CI 整体偏向通用特征，且没有读数偏向驾驶特征 | **更差** |
| 两个方向都有（一个读数更好、另一个更差） | **不同，不是更好**：写清楚好在哪、差在哪 |
| 全部 CI 跨零 | **测不出差别**，报每个读数的半宽，说明能排除多大的效应 |

读法上的两条限定，跟着数字走：Alpamayo 的输入是**残缺的**（侧面视野缺一半、底部黑带），所以它的「不更好」不能读成「驾驶 VLM 的表征没有用」，
只能读成「在只有前三路相机时没有用」，附带量里的两个数字决定这句话能说多重；openpilot 的输入和考试完全一样，没有这条限定。

次要 tap、fusion 行、原生 plan 参照行只描述，不参与判定。如果某个主 tap 判为「更好」，下一步是 train 训、完整 val 评（P3‴ 的协议）：
openpilot 的流式抽取在 train 上约 1.5 h，可以做；Alpamayo 在 52 万帧上约 20 h，要用户决定。

## 附带量（同样预登记）

1. **侧面相机缺口对 Alpamayo 特征的影响**：在考试已经抓好的 479 个 rater 帧上（7 路相机包在盘上），同一帧分别用 7 路和只用前三路渲染，
   抽同样的特征，报每个 array 的余弦相似度、相对 L2 差，以及它和帧间差异（同一 array 在 479 帧上的标准差）的比。
2. **侧面相机缺口对 Alpamayo 自己开车的影响**：只用前三路的输入，按考试的配置（K = 6、no-nav、seed 42）在 479 个 rater 帧上跑原生轨迹，
   和考试的 7 路结果（RFS 7.879）配对比。约 25–40 min GPU，排在主抽取之后。

## 成本估计与优化（先测后跑）

| 抽取 | 估计 | 依据 | 计划 |
|:--|:--|:--|:--|
| openpilot 三模型，流式 | 102 762 个 WOD 帧 × 2 步；GPU 约 4 + 8 + 6 min（small / Cinque / Lebowski 单 session 实测吞吐），JPEG 解码 + 重投影 30.8 万张 | smoke 的 steps/s，考试的每张 JPEG 约 78 ms·核 | 一次渲染喂三个模型；序列级并行，CPU 进程池渲染与 GPU 重叠，两张卡各一组 session |
| Alpamayo prefill | 约 3086 token / 帧，8B 的 prefill 约 50 TFLOP，估 0.15–0.3 s / 帧 → 19 944 帧 0.8–1.7 h 单卡 | 考试的 n_prompt | 同长度 prompt 直接 batch（无 padding）；JPEG 在 GPU 上解码与重投影；processor 的缩放在线程池里做，和 GPU 重叠；两张卡各一个进程分半 |

任何一个估计超过 3 h 就先在 200 帧上做 profiling、找瓶颈再改。

**等价性检查（全量之前）**：
- openpilot：(a) 加了 tap 输出的 ONNX 与原 ONNX 在同一段 200 步输入上的主输出最大差；(b) 流式与考试方式（每目标 10 s 暖机）在 rater 帧上的原生轨迹差
  （直接和考试存下的逐帧预测比），以及在 100 帧上 `temporal` 特征的余弦相似度——这一项量的是「定义改了多少」，不是 bug 检查。
- Alpamayo：(a) 前三路渲染与考试渲染在有像素的区域上逐像素一致；(b) batched、hook 池化的特征与单样本 `output_hidden_states=True` 参考在 16 帧上的最大相对差。

## 步骤

- [x] 预登记（本文件），提交
- [ ] openpilot：tap ONNX、流式抽取器、等价性检查、实测成本、全量
- [ ] Alpamayo：前三路渲染器、prefill 特征抽取器、等价性检查、实测成本、全量（两张卡）
- [ ] `p3` 一次跑完全部行 + `rejudge`；cross-fit 合并读数、配对比较；原生 plan 参照行
- [ ] 附带量 1、2
- [ ] 结果写回本文件、decisions 新条目、图

## 结果

（跑完再填）
