# E1：2 × 2 析因 + 空间分辨 probe（Cosmos 能恢复多少 CARLA 行人可读性）

状态: done（登记写于 2026-09-29 13:08 CST，任何数字之前；结果 13:30 CST）
主题: [research/feature-adapter-domain-shift.md](../research/feature-adapter-domain-shift.md) §5 E1；上游 [E0 todo](2026-09-29-e0-layer-probe.md)、[decisions 第 62 条](../research/decisions.md)；数据 [Cosmos G4 全量](2026-09-28-cosmos-pilot.md)
代码: `scripts/op_cosmos_probe.py`（抽特征 + probe）
box run dir: `$DATA_DIR/runs/op_cosmos_probe/`

## 目标

E0 判了 G-none：CARLA 行人在 openpilot 各层的**池化**特征上读不出（AUC 0.51–0.53），真实 nuScenes stage 3 是 0.83。两个问题：

1. **画风占多少**：同一对 x⁺ / x⁻ 渲染成 CARLA 与 Cosmos（G4，只重画外观、行人位置与几何不变）各一份，openpilot 同一条管线、同一套池化 probe 下，Cosmos 那一格恢复多少可读性。
2. **池化是不是把小行人藏掉了**（E0 的限定 2、3）：在特征图上按空间分辨去读，CARLA、Cosmos、nuScenes 三边各做一遍。

名词：**x⁺ / x⁻** = 有 / 无行人的反事实配对；**cell** = 某一层特征图上的一个空间位置（stage 1 / 2 / 3 的图分别是 32 × 64、16 × 32、8 × 16）；**GT 框** = CARLA 给的行人 mask 的外接框（`gt.npz` 的 `box`）。

## 数据（写死）

**Cosmos G4 全量里在起跑时刻已完成的对，全部使用**：`runs/cosmos_full/pairs/*/done.json` 在 2026-09-29 13:01:47 共 **75 对**（Town12，75 个不同 scenario instance；DynamicObjectCrossing 49、PedestrianCrossing 13、ParkingCrossingPedestrian 8、VehicleTurningRoutePedestrian 5）。
名单冻结在 `runs/op_cosmos_probe/pairs_frozen.txt`（起跑后新完成的对不进本次；数量少于 E1 原文的 ≥ 300 对，见下面的功效限定）：

c000v00 c001v00 c008v00 c012v00 c013v00 c014v00 c015v00 c022v00 c041v00 c043v00 c047v00 c049v00 c050v00 c052v00 c054v00 c055v00 c056v00 c057v00 c060v00 c065v00 c066v00 c067v00 c068v00 c069v00 c071v00 c073v00 c075v00 c082v00 c084v00 c088v00 c090v00 c091v00 c094v00 c097v00 c109v00 c118v00 c124v00 c129v00 c131v00 c132v00 c133v00 c138v00 c140v00 c141v00 c146v00 c148v00 c150v00 c151v00 c156v00 c159v00 c161v00 c162v00 c163v00 c167v00 c168v00 c169v00 c170v00 c172v00 c174v00 c175v00 c176v00 c177v00 c178v00 c179v00 c180v00 c182v00 c183v00 c192v00 c193v00 c194v00 c196v00 c197v00 c198v00 c199v00 c203v00

- 四格：{CARLA, Cosmos} × {x⁺, x⁻}，即每对的 `carla_plus / carla_minus / cosmos_plus / cosmos_minus`（`jevdrive.cosmos_full.load_pair`；Cosmos x⁺ 在混合支撑外取 x⁻ 的像素）。四格走**同一条**管线：单相机（Cosmos 的 1280 × 704、64° 相机）→ openpilot road / wide 模型帧（`scripts/cosmos_openpilot.py` 的 `maps` / `model_frames`）→ Cinque port fp16。
- **与 E0 的差别**：E0 的 CARLA 是 P5 的三相机拼接，4 414 对、42 条路线，场景不同；这里 CARLA 格是 Cosmos 全量的单相机 Town12 clip。所以 2 × 2 里的 CARLA 格是**同管线同场景**的基线，不是 E0 的重复；E0 的 P5 数字只作旁证，另外在 P5 上用大样本跑空间 probe（见下）。
- **流协议**：与 P5 / nuScenes 缓存一致，0.2 s 一格。93 帧 20 Hz clip 取第 0, 4, …, 92 步共 24 个 slot，每个 slot 一个图像对（上一个 slot 的帧，当前帧），slot 0 的上一帧为零图；policy 上下文是 9 个 slot（步长 1 slot），slot < 0 处 hidden 为零，流从零状态起（与 `op_adapt_eval.run` 同一条 `stage4_policy`）。
- **读数 slot**：slot 序号 ≥ 9（保证 9 个上下文 slot 的 hidden 都不是零起点的哑元，即 step ≥ 36），且 x⁺ 上 GT 行人 mask 像素数 ≥ 100（1280 × 704 分辨率，行人在画面里看得见）。每个读数 slot 给 x⁺（正）与 x⁻（负）各一行。
- 层与 tap 同 E0：stage 1 / 2 / 3 / 4（`permute_9 / 17 / 73 / 81`）、`vision`（`mean`）、`temporal`（`select_4`）。

## probe（写死）

**池化 probe（第 1 问）**：mean + max 池化特征（512 / 1 024 / 2 048 / 4 096 / 512 / 512 维），z-score、类平衡 L2 logistic、λ ∈ {1e-4..1e-1} 在训练折内按组留出选（`op_adapt_readout.oof_probe`，即 E0 真实侧同一套），5 折，**组 = scenario instance**（同一个场景的所有 slot 与 x⁺ / x⁻ 在同一折），读数 = 样本外 AUC（x⁺ 行对 x⁻ 行），instance 聚类 bootstrap 500 次；格间的 Δ 用同一 bootstrap 抽样。
每一格**在自己格内训练与测试**（in-domain）；这是「这一格的特征里读不读得出来」，回答第 1 问。跨格迁移（在 CARLA 格训、Cosmos 格测，及反向）作为描述附加。
与 E0 的区别：E0 CARLA 侧在 P5 的非观察帧上训练 hazard probe、在观察帧上测，这里 75 对没有独立训练集，只能格内交叉验证；训练行只有约千级，池化维数上千，probe 会偏弱。所以同样读数下 CARLA 格的数字要和 E0 的 0.52 一起看，不能拿它当无偏的上限。

**空间 probe（第 2 问）**：
- **S1 GT 框内 cell 池化**（只有 Cosmos 全量有 GT 框，所以只在 CARLA 格、Cosmos 格上做）：对每个读数 slot，用 road 与 wide 两个模型帧各自的重投影索引把 GT 框（外扩 8 px）映到模型帧上，落到的 cell 取并集 M（stage 1 / 2 / 3 各一份；两个相机的特征图在同一网格里融合，tap 只有一张图，所以取并集）；特征 = 图上 M 内的 mean + max 池化；x⁻ 用**同一个 M**（匹配 cell）。probe 同上，读数 AUC。
  对照（描述）：M 之外的 cell 池化（同 probe）；stage 3 的感受野很大，M 之外仍会漏出行人信息，所以这个对照不是「零」，只用来看信息是不是集中在框内。M 为空的 slot（框在视野外）丢掉，个数照报。
- **S2 stage 3 图上的小 conv 头**（CARLA-Cosmos 单相机两格，P5 CARLA、nuScenes 各一份）：1 × 1 conv（1 024 → 32）→ ReLU → 空间 global max → 线性 → logit；输入是 stage 3 整张 1 024 × 8 × 16 图，通道按训练折 z-score；AdamW lr 1e-3、wd 1e-2、300 步、batch 512（不足则 full-batch）、类平衡损失，超参不调，seed 0。这是「特征图里有没有某个局部位置在响应」的最小读法（max 池化对局部信号敏感）。
  - CARLA-P5：E0 同一批数据与同一折（hazard 标签、在 P5 非观察帧上训练、在观察帧上测、按 base 路线 5 折）；读数 = D0 的 x⁺ 对 x⁻ 样本外 AUC（scope `pedestrian`），路线聚类 bootstrap。对照 = 同批同折的池化线性 stage 3（0.523）。
  - nuScenes：E0 同一批（val 6 019 帧、`ped_corr` 标签、5 折按 scene）；对照 = 池化线性 stage 3（0.831）。
  - Cosmos 两格：同池化 probe 的折与组。
- S1 没有做 nuScenes / P5：没有现成的行人框投影到模型帧；这是本次登记里主动放弃的一项，不是遗漏。
- 描述附加：读数按行人大小分档（x⁺ mask 像素数 ≥ 1 500 的大 / 近行人子集）报池化与 S1 的 AUC；单独报读数 slot 的像素数分布。

## 判据（写死；点估计，CI 一并报）

**主读数**：Cosmos 格 vs CARLA 格的池化 probe，重点 stage 3 和 `temporal`。

- **K-restore**（像素级 sim2real 恢复了信号）：Cosmos 格 stage 3 池化 AUC ≥ 0.65 **且**对 CARLA 格的配对 Δ CI 下界 > 0。
- **K-none**：Cosmos 格 stage 1、2、3 与 `temporal` 的池化 AUC 都 < 0.60 → 画风（Cosmos 能改的那部分）不是让池化读数升起来的因素；解读见下。
- **灰区**：Cosmos 格 stage 3 在 [0.60, 0.65) 或 ≥ 0.65 但 Δ CI 含 0 → 「弱 / 未成立」，不当作恢复也不当作 K-none。
- **E1 原文的 G-a**（保留原判格）：Cosmos 格 `temporal` ≥ 0.70 且对 CARLA 格 Δ CI 下界 > 0 → 渲染域差成立；`temporal` < 0.60 → 不成立。
- **画风份额**（只描述）：(AUC_Cosmos − AUC_CARLA) / (AUC_real − AUC_CARLA)，AUC_real 用 nuScenes val 同层（stage 3 0.831、`temporal` 0.709）。真实一侧与这里不是同一个任务（E0 限定 1），份额只当量级参考。

**空间 probe**：
- **S-local**：CARLA 格 S1（stage 1–3 任一层）≥ 0.65 **而**同格池化 stage 1–3 都 < 0.60 → 信息在但局部，池化把它抹掉了；空间 adapter / 空间 head 可行，E0 的 G-none 收窄为「池化读不出」。
- **S-local-big**：CARLA-P5 上 S2 conv 头 ≥ 0.65（池化线性 0.523）→ 同样结论，且样本大得多（4 414 对）。
- **S-absent**：CARLA 格 S1 各层与 P5 S2 都 < 0.60 → 特征图里也读不出（在这套 probe 的能力内），E0 的 G-none 升为已确认。
- 之间（0.60–0.65）→ 弱，不判位置。
- Cosmos 格 S1 / S2 ≥ 0.65 而 CARLA 格 < 0.60 → 像素级重画把局部信号带回来了（与主读数 K-restore 互相印证）。
- nuScenes S2 与池化 0.831 的关系只描述（max 头是否比池化更高），不进判格。

## 各结局的意思

- K-restore + 份额高：域差的大头是外观，Cosmos 像素级修有效；特征 adapter 如果要做，学的是 CARLA → Cosmos 的映射（E2 的前提之一成立，另一前提是 E0 找到位置，目前没有，S-local 若成立则给出位置）。
- K-none 且 S-absent：行人信息在 CARLA 与 Cosmos 两边都没被表示 —— 或者这些行人对 Cinque 本来就太小 / 太远，与画风无关，需要按大小分档才能分开；
- K-none 但 S-local：池化读不出，局部有，adapter 要在空间图上做。
- 混淆（写在前面）：(1) Cosmos x⁺ 是「x⁻ + 混合区里重画的行人」，混合边界的接缝也可能是可读信号，K-restore 不能排除 probe 读的是接缝；S1 的框内 / 框外对照和 x⁻ 匹配 cell 只能部分区分；(2) 75 对 / 75 个 instance，DynamicObjectCrossing 占 65%，CI 会很宽，Δ CI 大概率含 0 —— 那时只能报点估计与「功效不足」，不下 K-restore；(3) 池化维数上千对约千级训练行，probe 偏弱，与 E0（数万训练行）不可比。

## 预算与执行

GPU 6，显存 ≤ 15 GB，≤ 1 GPU·h（前向约 75 × 4 × 24 = 7 200 个图像对，远小于预算；P5 / nuScenes 的 S2 只读 stage 3 缓存，训练是主要 GPU 用时）。CPU：op-train 行的 48–67（先确认仍空闲）。> 1 min 的作业进 tmux `jev`（`scripts/tmux_run.sh`）。

## 执行日志

- 13:04 冒烟（3 对）通；**偏离**：冒烟那次忘了设 `CUDA_VISIBLE_DEVICES`，落在 GPU 0（Cosmos 的卡）上跑了约 47 秒、显存只有一个 Cinque 前向的量，没有碰 Cosmos 的文件；正式作业都设了 `CUDA_VISIBLE_DEVICES=6`。
- 13:05–13:07 抽特征（GPU 6、CPU 48–63、16 个 decode worker）：75 对 × 4 格 × 24 slot，2 分钟；四格走同一条 `cosmos_openpilot.maps / model_frames` 管线，GT 框与 mask 的外接框逐像素一致（框是 [x0, y0, x1 + 1, y1 + 1]）。读数 slot 共 943 个（每个 slot 一行 x⁺、一行 x⁻，共 1 886 行），覆盖 75 对；x⁺ mask 像素数中位数 1 984（p10 163、p90 13 133）。
- 13:07–13:16 Cosmos 两格的池化、S1、S2 与格间迁移；13:20–13:28 P5 与 nuScenes 的 S2。作业 exit 0。
- **登记之外加的两项描述性读数，都是看到 Cosmos 格数字之后才加的**，不进判格：(a) P5 配对训练桥（在 P5 的 x⁺ / x⁻ 帧上直接训 probe，和 Cosmos 格同一种训练方式，用来检验「CARLA 格读得出来是因为训练方式不同」）；(b) 行人像素数分档（用来检验「读不读得出跟行人大小有关」）。分档的边界（100 / 500 / 1 500 / 6 000 px）是看过像素分布后取的整数，没有调。
- GPU 用量：抽特征 2 分钟 + 全部 probe 约 25 分钟 wall，合计约 0.5 GPU·h（登记上限 1）；显存没有单独监测（最大的张量是 P5 S2 的 4 096 行 × 1 024 × 8 × 16 训练标准化子样，约 2 GB fp32），预计远低于 15 GB。
- box run：`runs/op_cosmos_probe/{feats.npz,meta.csv,pairs_frozen.txt,probe/}`；小表 [research/results/e1-cosmos/](../research/results/e1-cosmos/)。

## 结果

75 对 / 75 个 instance，943 个读数 slot（1 886 行）；下面所有 CARLA / Cosmos 格 AUC 都是**格内**训练与测试（5 折按 instance），CI 是 instance 聚类 bootstrap 500。

### 1. 2 × 2 池化 probe（第 1 问）

| 层 | CARLA 格 AUC [CI] | Cosmos 格 AUC [CI] | Δ Cosmos − CARLA [CI] |
|:--|:--|:--|:--|
| stage 1 | 0.727 [0.696, 0.760] | 0.725 [0.692, 0.760] | −0.002 [−0.021, +0.019] |
| stage 2 | 0.767 [0.735, 0.803] | 0.743 [0.711, 0.776] | −0.025 [−0.050, +0.003] |
| **stage 3** | **0.815** [0.780, 0.848] | **0.746** [0.712, 0.782] | **−0.069** [−0.098, −0.037] |
| stage 4 | 0.803 [0.766, 0.841] | 0.783 [0.746, 0.825] | −0.020 [−0.037, −0.000] |
| `vision` | 0.752 [0.717, 0.789] | 0.708 [0.672, 0.745] | −0.044 [−0.067, −0.015] |
| `temporal` | 0.704 [0.663, 0.744] | 0.680 [0.644, 0.717] | −0.024 [−0.052, +0.009] |

跨格迁移（描述，λ = 1e-2）：CARLA 训 → Cosmos 测 stage 3 0.729、`temporal` 0.659；Cosmos 训 → CARLA 测 0.756、0.701（都低于各自格内，方向没有翻转）。
按 family（stage 3，CARLA / Cosmos）：DynamicObjectCrossing（49 对）0.835 / 0.765，PedestrianCrossing（13 对）0.883 / 0.799，ParkingCrossingPedestrian（8 对）0.694 / 0.712，VehicleTurningRoutePedestrian（5 对）0.585 / 0.568。

**按登记判格**：
- **K-restore 不成立**：Cosmos 格 stage 3 = 0.746 ≥ 0.65，但对 CARLA 格的 Δ = −0.069，CI 整体在 0 以下；Cosmos 没有「恢复」任何东西，因为 CARLA 格本身就读得出来，而且比 Cosmos 格更高。
- **K-none 不成立**（Cosmos 各层 ≥ 0.68）。
- **G-a（`temporal` ≥ 0.70 且 Δ CI 下界 > 0）不成立**：Cosmos `temporal` = 0.680，Δ = −0.024 [−0.052, +0.009]；也不 < 0.60，所以按 E1 原文落在灰区。
- **画风份额**（AUC_real − AUC_C）：stage 3 = 0.831 − 0.815 = 0.016，份额 = −0.069 / 0.016，分母近 0，**没有意义**，不报数。

### 2. 空间分辨 probe（第 2 问）

| 读法 | CARLA 格 | Cosmos 格 | 备注 |
|:--|:--|:--|:--|
| S1 框内 cell 池化，stage 1 | 0.938 [0.917, 0.960] | 0.929 [0.897, 0.955] | 框外 cell：0.558 / 0.557 |
| S1，stage 2 | 0.966 [0.950, 0.978] | 0.920 [0.883, 0.953] | 框外：0.642 / 0.651 |
| S1，stage 3 | 0.950 [0.931, 0.966] | 0.921 [0.892, 0.946] | 框外：0.735 / 0.709 |
| S2 conv 头，stage 3 整张图 | 0.896 [0.865, 0.925] | 0.861 [0.821, 0.898] | 对同格池化 stage 3：Δ +0.082 [+0.056, +0.108] / +0.115 [+0.084, +0.144] |

| S2 conv 头在 E0 的数据上 | 池化线性 stage 3 | conv 头 stage 3 | Δ [CI] |
|:--|--:|--:|:--|
| CARLA P5 行人（4 414 对，E0 协议） | 0.523 | 0.521 [0.509, 0.538] | −0.002 [−0.013, +0.005] |
| CARLA P5 全部 hazard（7 639 对） | 0.544 | 0.552 [0.535, 0.576] | +0.008 [−0.002, +0.018] |
| nuScenes val `ped_corr` | 0.831 | 0.868 [0.804, 0.916] | +0.037 [−0.011, +0.092] |

**按登记判格**：S-local 的前提（CARLA 池化 stage 1–3 都 < 0.60）在 Cosmos 全量的 CARLA 格上不成立（池化已经 0.73–0.82），所以该格上不是「池化抹掉了局部信号」；S1 框内 0.94–0.97、框外 0.56–0.74（stage 3 的感受野把行人信息漏到框外）说明信息集中在框内。**S-local-big**（P5 上 conv 头 ≥ 0.65）**不成立**：0.521，与池化 0.523 相同，P5 的 E0 数据上局部读法也读不出来；**S-absent** 也不成立（要求 CARLA 格 S1 也 < 0.60，实际 0.94+）。Cosmos 格 S1 / S2 与 CARLA 格同高，没有「Cosmos 把局部信号带回来」。

### 3. 描述性附加（登记之外，见执行日志）

行人大小分档（x⁺ mask 像素数，1280 × 704），格内训练的全行 probe 在各档内的 AUC：

| 档（px） | 读数 slot | stage 3 CARLA / Cosmos | `temporal` CARLA / Cosmos |
|:--|--:|:--|:--|
| 100–500 | 294 | 0.569 / 0.577 | 0.528 / 0.542 |
| 500–1 500 | 140 | 0.634 / 0.603 | 0.545 / 0.566 |
| 1 500–6 000 | 262 | 0.923 / 0.830 | 0.791 / 0.764 |
| ≥ 6 000 | 247 | 0.943 / 0.899 | 0.840 / 0.792 |

P5 配对训练桥（P5 的 x⁺ / x⁻ 行上直接训 probe，42 条路线、8 828 行，线性池化）：stage 1 0.559、stage 2 0.562、stage 3 0.598 [0.557, 0.649]、stage 4 0.581、`vision` 0.575、`temporal` 0.542；conv 头 stage 3 0.614 [0.576, 0.660]。

## 读法与限定（结论和推测分开）

1. **E1 的前提被推翻：在这 75 对上 CARLA 行人不是「读不出」**，池化 stage 3 就是 0.815，框内 cell 池化 0.95。所以「Cosmos 恢复了多少 CARLA 的缺口」没有可恢复的东西；Cosmos 格反而略低（stage 3 −0.069、`vision` −0.044，其余层差不显著）。这与 E0 的 0.52 的差**不是 probe 的训练方式造成的**：P5 的 x⁺ / x⁻ 行上按 Cosmos 格的方式直接训练也只有 0.54–0.60。
2. **行人大小是主导变量（数据支持；因果是推测）**：Cosmos 数据里 < 500 px 的读数 slot 在两格都是 0.57，≥ 1 500 px 到 0.92 / 0.83；Cosmos 重画外观并没有让小行人变得可读（0.569 vs 0.577）。这支持「小行人读不出来的原因是分辨率 / 大小，不是画风」。要点：这套 Cosmos 数据的读数 slot 是按「行人在画面里 ≥ 100 px」选的，中位 1 984 px，比 P5 的 42 条评测路线里的行人更近更大的可能性很高（**P5 的行人像素数没有量，这是推测**）；把 E0 的 G-none 解读成「CARLA 行人在 openpilot 里没被表示」不成立，至少要收窄成「P5 评测那一批（多为小 / 远行人）在池化和 conv 头上读不出来」。
3. **Cosmos 格低于 CARLA 格的原因未拆**：可能是混合区的重画损失了一点行人细节，也可能是 Cosmos 的行人本身对 Cinque 更难读（推测）；反过来 x⁺ 中混合区的接缝可能给 probe 一条捷径，但 CARLA 格没有接缝、读数同样高，所以不是接缝在撑起 Cosmos 的数字。
4. **功效与偏差**：75 对，DynamicObjectCrossing 占 65%（49 / 75），VehicleTurningRoutePedestrian 只有 5 对；读数 slot 在一对内相关，CI 按 instance 聚类；probe 训练行只有 1 886，2 048 维。这些格内数字是「能读 vs 不能读」的证据，不是各层精确的排序。
5. **S1 的框内高分有一部分是平凡的**：框内 cell 的特征本来就以行人的外观为主，x⁻ 里同位置是背景；它回答的是「信息在不在特征图的这个位置」（在），不是「下游 head 读不读得到」。
6. **nuScenes 上 conv 头 0.868 vs 池化 0.831**：Δ 的 CI 含 0（[−0.011, +0.092]），不判；只说明 max 头没有让真实侧变差。

## 结局

- 特征 adapter 那条线：**E0 的「不可行」判词收回到只适用于 P5 评测这一批小行人**；在 Cosmos 全量这批（较大行人）上池化特征已经能读，adapter 没有要补的「行人可读性」。E2（CARLA → Cosmos 的特征 adapter）的动机变弱：Cosmos 格读数不高于 CARLA 格，配对监督学不到「让行人更可读」的方向。
- 下一步的直接问题不是画风，而是**小行人**：< 500 px 的行人两格都读不出。要么按大小 / 距离重新选 P5 考卷，要么测输入分辨率（128 × 256 的模型帧上一个 < 500 px 的行人只占几个像素）。这没有做。
