# op-adapt 第二轮（B，sim + real）：预登记

状态: 草案，待用户批准（写于 2026-09-29，任何第二轮训练、任何第二轮读数之前；不跑训练。文中唯一的新数字是第 0 节的 M0 行人大小测量，纯 CPU、没有模型，是设计依据，不是结果）
主题: [research/decisions.md](../research/decisions.md) 第 42、44、45、48、50、55、56、62、63 条；[research/midterm-gaps.md](../research/midterm-gaps.md) 缺口二；[research/feature-adapter-domain-shift.md](../research/feature-adapter-domain-shift.md)
前身: [op-adapt 第一轮](2026-09-28-op-adapt.md)（port、吞吐、B 小试、后续 1 / 2）；数据: [Cosmos G4 全量](2026-09-28-cosmos-pilot.md)；E0 / E1: [e0](2026-09-29-e0-layer-probe.md)、[e1](2026-09-29-e1-cosmos-probe.md)
排期约束: [tmp/2026-09-29-midterm-plan.md](../tmp/2026-09-29-midterm-plan.md)（09-30 中午起 2 卡；D2–D5 卡 A 给本轮，D5–D8 闭环要用本轮选出的模型）

## 为什么要做

第一轮（第 55 条）说明 openpilot 的 vision 后段可以便宜地改、改了不坏（跨数据集漂移 6 cm），但只用真实 nuScenes 监督，真实行人可读性只涨 +0.08（没过 +0.10），CARLA 完全不动（+0.009）。
按验收原则（真学会了，sim 与 real 都能开），这不是正例。缺口二（midterm-gaps 2.1）要的正是一个「同一次轻度适配让 openpilot 在 CARLA 与真实上都更会对行人反应、正常驾驶不变差」的结果，而且要有**行为**读数，不只是线性 probe。

这一轮开始前，E0 / E1 改变了问题的形状：
- E0（第 62 条）：P5 考卷上 CARLA 行人在每一层都读不出（0.51–0.53）。
- E1（第 63 条）：Cosmos 全量前 75 对上，CARLA 行人本来就读得出（池化 stage 3 0.815，框内 cell 0.95），Cosmos 画风反而略低（−0.07）；可读性由行人大小决定（< 500 px 两格都 0.57，≥ 1 500 px 0.92 / 0.83）。
- 所以 CARLA 与真实之间的行人差，很可能主要是**大小 / 分辨率**，不是画风。本轮在冻结登记前先量了这件事（第 0 节），结果决定了 P5 读数的分档和「分辨率修正」要不要进 arm。

名词（第一次出现时注释，之后直接用）：
- **B**：第一轮的适配方式，openpilot Cinque 的 stage 4（vision 编码器最后一段，约 1.1 亿参数）解冻，stage 1–3（**trunk**）冻结并按帧缓存，policy（时序 transformer 与全部输出头）冻结。
- **蒸馏（distillation）**：改后模型在正常帧上的输出贴住原模型的输出；**漂移（drift）**：同一帧上原生 plan 与原模型的差。
- **配对（pair）**：同一世界、同一 ego 轨迹的 x⁺（有行人）/ x⁻（行人藏到地下）；**C / K**：同一对的 CARLA 原画 / Cosmos 重画（第 56 条 G4），所以每对是四元组 C⁺ C⁻ K⁺ K⁻。
- **D0 probe**：第 42 条的配对可分性 probe（冻结特征 + 类平衡 L2 logistic，按 route 分折，x⁺ 对 x⁻ 的样本外 AUC，route 聚类 bootstrap）。
- **翻转（flip）**：P5 考卷上 x⁺ 与 x⁻ 的预测 2 s 速度之差超过考生自己 null 分布的 95 分位，且方向与 expert 一致；**null false-flip**：只换 TM seed 的 null 对上的同一统计量。
- **等效像素（px_eq）**：行人可见 mask 像素数换算到 Cosmos 相机（1280 × 704、64° HFOV、f = 1 024 px）；E1 的大小分档就是这个单位。openpilot 的 road 模型帧（f = 910）看到的面积是它的 0.79 倍，wide 帧 0.20 倍。

## 0. 冻结前的测量 M0：各集合的行人有多大、多远（已跑，CPU）

脚本 `scripts/op_adapt_r2_pedsize.py`（box 上 `runs/op_adapt/r2_pedsize/`），小表 [research/results/op-adapt-r2/pedsize/](../research/results/op-adapt-r2/pedsize/)。
每个集合取它在读数里当「正例」的那些帧：P5 是 D0 行人 scope 的 4 414 个观测帧（42 条路线），Cosmos 是 G4 全量此刻已完成的 93 对，真实是走廊内行人的帧（第一轮的走廊定义，s ≤ 30 m）。
距离一律是本车后轴到行人的水平距离。量法与可比性：

| 集合 | 像素怎么来的 | 距离怎么来的 |
|:--|:--|:--|
| P5 v1 BA（考卷） | 实测：分割视图里行人可见像素（半分辨率、Waymo 前视内参 f = 556.75），面积 × 3.38 | 实测：x⁺ 世界 `actors.npz` |
| Cosmos G4 全量 | 实测：`gt.npz` 的逐帧 mask | 实测：第 2 遍世界的 `actors.npz` |
| WOD train | 实测：对得上走廊行人的 YOLO26x 前视框（面积 × 按焦距换算 × Cosmos 上量的 mask / 框 比 0.49）；原始 mask / 框 比 0.54，与 Cosmos 一致 | YOLO 平地抬升（与标签同源） |
| nuScenes train / val | **推导**：针孔直接在 Cosmos 相机里算（身高 1.75 m、CAM_FRONT 在后轴前 1.70 m），mask / 框高² = 0.206（Cosmos 上量的）；不含遮挡，是上界 | 实测：GT 框 |
| NAVSIM navtrain | 无 | 实测：GT 框 |

| 集合 | n 帧 | px_eq 中位 [p25, p75] | < 100 px | 100–500 | 500–1 500 | ≥ 1 500 | 距离中位 | > 30 m |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| **P5 v1 BA 考卷** | 4 414 | **487** [135, 937] | **21.8%** | 30.8% | 30.5% | 16.9% | **29.9 m** | **50.0%** |
| Cosmos G4，E1 读数 slot | 1 155 | 1 647 [311, 5 747] | 0.0% | 33.0% | 15.6% | 51.4% | 17.0 m | 0.6% |
| Cosmos G4，全部可见 5 Hz 帧 | 1 595 | 859 [169, 4 340] | 11.1% | 32.3% | 16.6% | 40.0% | 20.1 m | 18.4% |
| WOD train（走廊，非 uncertain，前视对上的 1 605 / 2 149） | 1 605 | 1 573 [740, 5 372] | 0.7% | 14.6% | 32.9% | 51.8% | 18.1 m | 4.4% |
| nuScenes train（推导） | 1 344 | 5 261 [1 890, 11 592] | 0 | 0 | 20.2% | 79.8% | 12.9 m | 3.3% |
| nuScenes val（推导） | 203 | 5 793 [1 281, 17 857] | 0 | 0 | 27.1% | 72.9% | 12.4 m | 5.4% |
| NAVSIM navtrain | 4 562 | — | — | — | — | — | 20.9 m | 6.9% |

P5 按 family（px_eq 中位只算可见帧 ≥ 68 px，即 P5 自己的可见门槛）：

| family | 帧 | 行人完全不可见（0 px） | 可见帧 px_eq 中位 | 可见帧 < 500 px | 距离中位 | > 30 m |
|:--|--:|--:|--:|--:|--:|--:|
| VehicleTurningRoutePedestrian | 2 659 | 3.3% | 541 | 46.7% | **34.4 m** | **74.6%** |
| PedestrianCrossing | 836 | **38.9%** | 10 055 | 0.8% | 9.4 m | 0% |
| DynamicObjectCrossing | 609 | 0% | 247 | 55.2% | 18.3 m | 12.3% |
| ParkingCrossingPedestrian | 310 | 29.0% | 210 | 67.7% | 17.5 m | 21.3% |

同一距离档的 px_eq 中位（[px_by_dist.csv](../research/results/op-adapt-r2/pedsize/px_by_dist.csv)）：10–20 m 档 P5 1 487、Cosmos 2 861、WOD 2 478；20–30 m 档 P5 555、Cosmos 274、WOD 841。

读法：
1. **P5 考卷是一批远而小的行人**：一半在 30 m 以外（真实走廊标签按定义只到 30 m，真实正例里 > 30 m 的只有 3–7%），可见帧里 43% < 500 px（E1 上这一档两种画风都只有 0.57），另有 11.4% 的观测帧行人根本不在画面里（PedestrianCrossing 39%、ParkingCrossing 29%）。
   E0 的 0.51 因此至少有一大部分是**考卷的构成**：行人 scope 的 60% 是 VehicleTurningRoutePedestrian，它的行人中位 34 m。这支持第 63 条的推测（「P5 多为小 / 远行人」由推测变为测量），但「小 → 读不出」的因果仍是推测。
2. **Cosmos 训练对与真实训练集在大小上接近**（中位 1 600 px 左右、17–18 m），比考卷大 3 倍、近 12 m。在这样的训练分布上学，最直接会改善的是 ≥ 500 px 的行人；考卷里的 30 m+ 行人是训练里几乎没有的区间（Cosmos 读数 slot 0.6%）。
3. 所以本轮把 P5 的主读数按大小分档（≥ 500 px_eq 的可见帧为主，全体为延续第一轮的副读数，见第 4 节），并把「分辨率修正」作为一个先过零训练门槛再开的 arm（第 2 节 Z）。
4. nuScenes 的像素是按距离推出来的上界（不含遮挡）；WOD 的像素是实测框。两者的距离分布与 Cosmos 读数 slot 同一量级，**真实侧的训练 / 读数行人与 Cosmos 训练对在大小上可比，与 P5 考卷不可比**。

限定：Cosmos 只有 93 对（DynamicObjectCrossing 63 对），全量完成后按同一脚本重跑一次（写进执行日志，不改登记）；P5 距离缺 162 / 4 414 帧（x⁺ 世界里那一 tick 没有该行人的 actor 记录）；WOD 有 25% 的走廊行人帧没有对上前视检测（最高分的匹配在侧前相机），没算。

## 1. 要回答的问题

1. 三个带标签的域一起训（Cosmos 配对、同一批配对的 CARLA 原画、真实 WOD / navtrain / nuScenes），openpilot 能不能在**真实留出**与**CARLA 留出**上同时读出更多行人，并且**原生 plan 真的对行人减速**，同时正常驾驶不坏。这是验收原则的第一个正例候选。
2. 把 COCO YOLO26x 的 image-plane 检测 token（第 50 条配方）在 policy 之前注入、与 (1) 的可训练部分端到端一起训，是否比纯视觉更好，尤其在小 / 远行人上。
3. 效应归因：real-only、sim-only、去掉 CARLA 原画、去掉 Cosmos 各自带走多少。
4. （门槛之后）一个更高分辨率的视图能否让 < 500 px 的行人变得可读。

## 2. Arms

所有 arm 共用：Cinque（3.8 亿参数），port 的 fp16 路径（第一轮 P3 定），trunk 缓存，9 个 5 Hz context slot（1.6 s），原模型作 teacher，同一套数据划分和读数。可训练部分以外全部冻结。

| arm | 可训练 | 训练数据 | 损失 | seed | 回答什么 |
|:--|:--|:--|:--|--:|:--|
| O | —（原模型） | — | — | — | 所有读数的参照；读数与训练前的 M1 一起跑 |
| **A**（主，纯视觉 B） | stage 4 + 辅助头 | sim（C + K 配对）+ real | L_aux + L_pair + λ_b·L_Δplan + λ_d·L_distill | 3 | 纯视觉 sim + real 能不能两边都学会 |
| **D**（主，检测插件） | A 的全部 + 检测 token adapter | 同 A | 同 A | 3 | 检测 token 注入是否在 A 之上再加 |
| A-real | 同 A | 只有 real（sim 只进蒸馏） | L_aux + λ_d·L_distill | 1 | G-b：只加真实数据量能不能过 +0.10，CARLA 动不动 |
| A-sim | 同 A | 只有 sim 配对（real 只进蒸馏，不用真实标签） | L_aux + L_pair + λ_b·L_Δplan + λ_d·L_distill | 1 | sim 监督单独能否带到真实 |
| A-noC | 同 A | K + real（去掉 CARLA 原画） | 同 A | 1 | OpenVLA 的教训：原画作为带标签的域是否必需（P5 是原画） |
| A-noK | 同 A | C + real（去掉 Cosmos） | 同 A | 1 | Cosmos 画风对真实侧是否有贡献（E1 说它对可读性没有） |
| D-only | 只有检测 adapter + 辅助头（stage 4 冻结） | 同 A | 同 A | 1 | 检测 token 单独能做到多少 |
| Z（门槛后开） | A 的全部 + tele 视图 adapter | 同 A | 同 A | 1（过线再补 2） | 分辨率修正 |

**λ_b 的选择（唯一调的超参）**：只在 A seed 0 上跑两个预登记配置 λ_b ∈ {0.3, 1}，用 dev 选：dev 漂移中位 ≤ 0.10 m 且 dev null 减速率比原模型高 ≤ 2 pp 的配置里，选 dev 配对的 Δplan 与 expert Δ 同号率最高的；两个都不满足就选漂移小的。其余 arm 与 seed 继承选中的 λ_b。λ_d = 10（第一轮 dev 选出的），不再调。

### 2.1 A：纯视觉 B 的损失

- **L_aux（可读性，三个域都有）**：第一轮的两个辅助头（`select_4` 上的 MLP、32 个 hidden token 上的 attention pooling），三个输出：走廊内行人有无（主）、宽走廊内行人有无、最近走廊行人距离（只在正例上回归）。
  标签：nuScenes = GT 框（第一轮原样）；WOD = YOLO26x 平地抬升（`uncertain` 帧遮掉行人损失，后续 2 原样）；navtrain = GT 框；Cosmos / CARLA 对 = CARLA actor 位置套同一走廊定义（x⁻ 全部为负，x⁺ 按几何定；行人 px_eq < 68 的 x⁺ slot 标为不可见、不进 L_aux，但进 L_Δplan）。
- **L_pair（配对差分，sim）**：同一 slot 的 x⁺ 与 x⁻ 的主 logit 做 margin ranking，softplus(1 − (z⁺ − z⁻))，只在 x⁺ 行人可见的 slot 上，C 与 K 各算一份。两个成员除行人区外逐像素相同（K 按构造，C 按渲染），所以这个损失只能靠行人区降下来。
- **L_Δplan（行为，sim）**：原生 plan 在 x⁺ 与 x⁻ 上的纵向速度差 Δv̂(t)（t = 1, 2, 3 s）去拟合 BehaviorAgent 在两个世界里的实际未来速度差 Δv*(t)（第 1 遍 CARLA 的完整轨迹，窗口之后仍在记录）；Huber，按训练集 |Δv*| 的标准差归一。
  窗口前段 expert 还没分叉，Δv* ≈ 0，这部分同时是「别在行人还远时就乱减速」的监督。这是 midterm-gaps 的 F4：行为监督**只来自 sim**，真实侧只看它迁不迁得过去。
- **L_distill**：第一轮原样（全部输出去掉 hidden 与 MDN std，按原输出标准差归一的 MSE），加在真实「正常帧」（宽走廊内无行人 / 骑车人）和 sim 的 x⁻ 帧上。
- 优化：AdamW，stage 4 lr 3e-5、新模块 1e-3、wd 0.01，batch 64（按「序列」计：一个 sim 四元组 = 4 条序列），cosine，fp16 + GradScaler（第一轮原样）。

### 2.2 D：检测插件

- 检测：YOLO26x-seg 640（第 45 条的快通道模型），**只用前视相机的原生分辨率图像**（Cosmos 1280 × 704、P5 Waymo 前视、WOD 前视、nuScenes CAM_FRONT、navtrain CAM_F0），每个 5 Hz slot 取 pedestrian / cyclist / vehicle 里 score 最高的 8 个。
- token（第 50 条 image-plane 配方，不抬升不筛走廊）：类别 one-hot、score、框（重投影到 openpilot road 模型帧的归一化坐标，使不同相机可比）、框高 / 焦距（角尺寸）、16 维检测外观（YOLO neck 的 RoIAlign，PCA 在 WOD train 上拟合）。
- 注入：在 stage 4 之后、进入 policy 队列之前的 32 × 512 hidden token（`view_39`）上加一层 cross-attention（hidden 为 query，检测 token 经 MLP 投到 512 维为 key / value，8 头），输出投影**零初始化**、再乘一个初值 0 的标量门。
  所以 D 在第 0 步与原模型逐位相同，蒸馏从零漂移起步；部署时每帧的注入 hidden 进入队列，自然成为后续帧的 context，训练时 9 个 slot 都注入（每个 slot 都要有检测）。约 130 万参数。
- 为什么这算「分辨率修正」的便宜版：YOLO 看的是原生分辨率（Cosmos 上 f = 1 024，比 road 帧的 910 高、比 wide 帧的 455 高一倍多），小行人在检测器里比在 openpilot 的 512 × 256 帧里多得多的像素。第 45 条：YOLO 在 P5 hazard 行人 ≤ 20 m 召回 0.89，图像平面 20–40 m 看见 46–61%。

### 2.3 Z：tele 视图（先过零训练门槛 R0 才开）

- 做法：同一前视图像按 f = 1 820（road 帧焦距的 2 倍）重投影成一张 512 × 256 的 tele 帧（视场约 16°，对准 road 帧中心），过同一个冻结 trunk 和共享的 stage 4，32 个 token 经与 D 相同结构的零初始化 cross-attention 注入 hidden。成本：每 slot 多一次 trunk 前向，trunk 缓存 × 2（约 +200 GB，盘上 1.3 TB 空）。
- **R0（零训练，数据准备时顺带跑，≤ 0.3 GPU·h）**：原模型的 stage 3 在 tele 帧上，E1 的池化 probe 与框内 cell probe，在 Cosmos 对与 P5 的 < 500 px_eq 档各算一次。
  **开 Z** = 任一集合 < 500 档的池化 stage 3 AUC ≥ 0.65（E1 现为 0.57）且对 road 帧同档的配对 Δ CI 下界 > 0；否则 Z 不开，结论写成「在 openpilot 的 trunk 上，放大输入不让小行人变可读」，这同时是第 63 条「分辨率限制」推测的直接检验。
- 为什么不直接开：tele 视图是新的输入流，改动比 D 大，且 E1 的 < 500 档在两种画风下都读不出，放大能否救回是未知的；R0 几乎免费，先答这个问题。

## 3. 数据

### 3.1 三个带标签的域

| 域 | 来源 | 规模 | 标签 | 状态 |
|:--|:--|:--|:--|:--|
| sim-K | Cosmos G4 全量的 K⁺ / K⁻ | 目标 2 000 对；此刻 93 对（lane 13:33 drain 后正在重开） | CARLA actor + 走廊；expert Δ 未来 | 进行中 |
| sim-C | 同一批对的 C⁺ / C⁻（`load_pair(pair, "carla")`） | ≥ sim-K（原画在 controls 阶段就有，不等 Cosmos） | 同上 | 随 CARLA 第 2 遍产出 |
| real-WOD | WOD train | 207 678 个 slot、113 068 帧有标签（走廊 2.4%、宽 9.3%） | YOLO 抬升，11.7% uncertain | trunk 缓存已有 |
| real-nav | NAVSIM navtrain | 103 288 个 token（走廊 4.4%、宽 17.1%） | GT 框 | trunk 缓存已有 |
| real-nus | nuScenes train 650 个 scene（第一轮的训练集） | 约 2.8 万个 keyframe（走廊 4.8%） | GT 框 | trunk 缓存已有 |

sim 的每对取窗口里 slot ≥ 9 的 5 Hz slot（与 E1 同：9 个 context slot 都在窗口内），每对约 15 个训练 slot；每个训练样本是同一 slot 的四元组（sim-noC / noK 的 arm 是二元组）。

**sim 起训的数量规则（写死）**：10-01 12:00 或 sim-K ≥ 1 000 对，先到者起训；到时 sim-K < 500 对，则 sim-K 用已有的全部、sim-C 用全部已渲染的对，并在结果里写明规模；sim-K < 200 对则 A-noC 不跑（K 太少、没有意义），其余照跑。

### 3.2 划分

| 用途 | sim（Town12 的 209 个实例） | real |
|:--|:--|:--|
| train | 75% 实例（全部变体） | nuScenes 650 scene 去掉 dev；WOD train 95% sequence；navtrain 95% log |
| dev（只用于 λ_b 选择与早停检查） | 10% 实例 | nuScenes 50 scene（第一轮同一批）；WOD train 5% sequence；navtrain 5% log |
| test（Cosmos 留出） | 15% 实例，C 与 K 两种画面都考 | — |
| 留出考卷 | P5 v1 BA（42 条行人路线，训练对已按 60 m 排除） | nuScenes val 150 scene、WOD val 60 条 stream（第一轮同一批）、NAVSIM navtest、WOD-E2E val |

按实例划分，所以同一实例的不同天气 / TM seed 变体不会跨 train / test。划分用 seed 0 的固定排列，写进 run dir。

### 3.3 混合与过采样

- 每个 batch 按序列计 50% sim、50% real。sim 内 C : K = 1 : 1（四元组天然如此）；real 内 WOD : nuScenes : navtrain = 45 : 30 : 25（navtrain 是 2 Hz sample-and-hold 的离协议输入，第 36 条，比例压低但保留，因为 NAVSIM 是 no-harm 读数之一，蒸馏在它上面钉住行为）。
- 正例权重：主 BCE 的 pos_weight 使每个域的有效正例率约 25%（第一轮做法）。
- **大小 / 距离过采样**：real 正例按最近走廊行人距离分档，10–20 m 与 20–30 m 档的采样权重 × 2（0–10 m、30 m+ × 1）；sim 的 x⁺ slot 按 px_eq 分档，100–500 与 500–1 500 档 × 2。理由：第一轮的增益在 10–20 m（+0.14）和 > 20 m（+0.10），E1 的可读性断崖在 500–1 500 px；考卷一半在 30 m 以外，但训练里几乎没有，那一档不做人工放大（数据不支持）。
- 蒸馏帧：real 正常帧 + sim x⁻，占每个 batch 的 25% 序列（与上面的比例并行抽，不挤占标签样本）。
- 训练量：每个 arm 120 万条序列（第一轮 50 万的 2.4 倍，因为数据量与域数都多了），约 2–3 个 sim epoch、real 约 1 个 epoch。

## 4. 读数与登记线

全部在留出数据上，改后模型与原模型 O 在同一次前向、同一套 probe 里并列算。「D0 / (a) probe」沿用第一轮与第 42 条的代码，不改。

### 4.1 可读性

| 编号 | 读数 | 集合 | 登记线（主 arm A、D 每个 seed 都要过，才算过） |
|:--|:--|:--|:--|
| R-nus | 走廊行人 `temporal` probe AUC Δ（改后 − O） | nuScenes val（203 / 6 019） | Δ ≥ +0.10 且配对 CI 下界 > 0（第一轮原线） |
| R-wod | 同上 | WOD val：60 条 stream 的读数帧，走廊行人标签用 `labels_wod` 同一规则在 val 上补做（CPU） | Δ ≥ +0.05 且 CI 下界 > 0 |
| S-p5 | D0 `temporal` AUC | P5 v1 BA 行人 scope，**可见帧 ≥ 500 px_eq**（2 092 对，≥ 31 条路线） | ≥ 0.65 且对 O 的配对 Δ CI 下界 > 0 |
| S-p5-all | 同上（第一轮延续） | P5 行人 scope 全部 4 414 对 | ≥ 0.60 且 Δ CI 下界 > 0（第一轮 (c) 原线；只作副判格，不进主格） |
| S-cos | 配对 AUC（E1 的池化 probe，按实例 5 折） | Cosmos test 实例，C 与 K 各一个 | 两种画面 Δ 都 ≥ +0.05 且 CI 下界 > 0 |

每个可读性读数都按大小分档另报（描述）：P5 与 Cosmos 按 px_eq（< 500、500–1 500、≥ 1 500），真实按距离（0–10、10–20、20–30 m）。

### 4.2 行为

| 编号 | 读数 | 集合 | 登记线 |
|:--|:--|:--|:--|
| B-p5 | 原生 plan 2 s 速度的行人翻转率，第 48 条规则 | P5 v1 BA 行人 reactive 帧（ p5_exam.exam 原样，考生 = 改后原生 plan） | 每个 seed：翻转 CI 下界 > 该考生样本外 null false-flip + 10 pp；且 null false-flip ≤ O 的 + 3 pp |
| B-real | 减速率差：Δ = [slow(行人帧) − slow(null 帧)]_改后 − [同]_O；slow = plan 2 s 速度 < 当前速度 − max(1 m/s, 20%) | nuScenes val ∪ WOD val：走廊行人 0–30 m 的帧对「宽走廊内无行人 / 骑车人」的帧，按本车速度 5 档分层匹配 | Δ ≥ +5 pp 且 scene / sequence 聚类 CI 下界 > 0；null 帧的 slow 率比 O 高 ≤ 2 pp |
| B-cos（描述） | 原生 plan 在 Cosmos test 对上的 Δv̂ 与 expert Δv* 同号率 | Cosmos test，C 与 K | — |
| B-log（描述） | 行人帧上 plan 对 log 未来的 ADE 配对差；log 减速的行人帧里 plan 也减速的比例 | nuScenes val、WOD val | — |

真实侧没有反事实，所以 B-real 读的是「有行人比没行人多减速多少」相对原模型的变化；它不能区分「对行人反应」和「对行人常出现的场景反应」，这一点用 null 帧的速度分层和 B-log 部分对冲，结论里要写明。

### 4.3 不坏（no-harm）

| 编号 | 读数 | 登记线 |
|:--|:--|:--|
| N-drift | 正常帧原生 plan 漂移（第一轮 (b) 原样，nuScenes val 与 WOD val） | 两处中位 ≤ 0.10 m、p95 ≤ 0.50 m |
| N-ade | WOD val 原生 plan 对 log 未来的 ADE 相对变化 | CI 上界 ≤ +2% |
| N-lead | lead x 与 lead_prob（第一轮原样） | lead_prob \|Δ\| 中位 ≤ 0.02，lead x \|Δ\| 中位 ≤ 0.5 m |
| N-nav | NAVSIM navtest PDMS（原生 plan，与原模型同一 harness：第 36 条补帧输入、同一 2 000 token 子集或当时的全集，二者同一次跑） | ≥ O − 1 |
| N-rfs | WOD-E2E val RFS（与第 44 条同一 harness） | 配对 Δ 的 CI 下界 ≥ −0.10 |
| N-cutin | P5 cut-in 翻转（车辆不能被弄坏） | 不低于 O 5 pp 以上 |

### 4.4 捷径检查（描述，但出问题就写进判格）

- **行人区换回 null**：Cosmos test 与 P5 上，把 x⁺ 的行人区（mask 膨胀 24 px）换成 x⁻ 的像素再跑一遍，改后模型的 L_aux logit 差与翻转应回到 null 水平；若仍有 ≥ 一半的效应，说明学的是行人区以外的东西（第 56 条 G-d）。
- 域分类 AUC（每层，C / K / real）只描述。

### 4.5 判格与各结局的意思

主判格对 A 和 D 各判一次；「过」= 该 arm 的 3 个 seed 都过该条。

| 结局 | 条件 | 对中期故事的意思 |
|:--|:--|:--|
| **P（正例）** | R-nus、R-wod、S-p5、S-cos、B-p5、B-real、全部 N 都过 | 验收原则第一个正例（表征 + 开环行为）；选中的模型进 D5–D8 闭环；midterm-gaps 2.3 结局 A |
| P-rep（看见但不动） | R-*、S-*、N 过；B-p5 或 B-real 不过 | 两边都学会「看见」，policy 不据此减速（G-c）；故事写成表征正例、行为待补；下一步是 policy LoRA 或更强的 L_Δplan，不是更多数据 |
| P-size（分辨率） | R-* 与 N 过；S-p5 过但 S-p5-all 不过，且分档里 < 500 档不动 | 学会是真的，考卷上剩下的是分辨率；用分档表如实写，并看 D / Z 在 < 500 档是否不同 |
| real-only | R-* 过、N 过，S-* 与 B-p5 不过 | 与第一轮同形；看 A-noC：若 A-noC 也一样，sim 监督没被用上；验收原则仍无正例（结局 C） |
| sim-dominant | S-* 或 B-p5 过，但 R-* 或 B-real 不过，或 N 不过 | sim 压过 real 或学到 CARLA / Cosmos 特有方向（第 44 条的老问题，结局 D）；看捷径检查与 A-noK |
| none | 以上都不成立 | 中期只报第一轮 + E0 / E1 / M0 的诊断链 |

arm 之间的比较（描述，但用来归因，同一套 route / scene 聚类配对 bootstrap）：
- **D 对 A**：S-p5、S-p5-all、B-p5、< 500 档可读性的配对 Δ；CI 下界 > 0 → 「检测插件在纯视觉之上有增益」，写进故事作「trade（工业模型轻适配）+ 结构化感知插件」的一环（第 50 条的真实侧第一次读数）。
- **A-real / A-sim 对 A**：A 的 S-* 只在有 sim 数据时涨、R-* 只在有 real 数据时涨 → 「各域要各自的标签」（OpenVLA 的结论在驾驶上复现）；A-sim 在 R-* 上也涨 → sim 监督可以迁到真实，是更强的结论。
- **A-noC 对 A**：S-p5（原画考卷）掉而 S-cos 的 K 不掉 → 原画必须作为带标签的域（第 1.1 节的第 3 点）；**A-noK 对 A**：R-* 不掉 → Cosmos 对真实侧没有贡献，与 E1 一致，以后 sim 数据可以不经 Cosmos（省 118 GPU·h 一轮）。
- **D-only**：检测 token 单独（stage 4 冻结）若已达 D 的大部分，说明增益来自检测器而不是视觉适配。

## 5. 训练前还要做的（写死，数字出来不改登记）

1. **M1 基线（零训练，约 0.5 GPU·h）**：原模型在全部读数上的数，包括 P5 ≥ 500 px_eq 子集的 D0、按大小分档、原生 plan 在 P5 上的翻转与 null false-flip、B-real 的原模型减速率、NAVSIM / RFS。这些是 O 列，任何训练之前跑完、提交。
2. **R0**（第 2.3 节）。
3. **标签与目标**：Cosmos / CARLA 对的走廊标签和 expert Δv*（第 1 遍世界的 `pose.jsonl`）；WOD val 的走廊行人标签（`labels_wod` 同一规则）。CPU。
4. **缓存**：sim 对的 trunk 缓存（C 与 K，约 2 000 × 24 slot × 4 ≈ 19 万张，按第一轮的 16.6 万张 8 min 推约 10–15 min GPU）；YOLO 检测与 token（sim 对、nuScenes train / val、WOD train 缺的偶数帧、P5 前视、navtrain / navtest 若已有则复用）；teacher 输出。

## 6. 分级启动与 sanity checklist

- **第 1 个单位**：A seed 0（λ_b = 1）跑 2 000 步。检查：loss 全部有限且下降；dev 漂移中位 ≤ 0.10 m；dev 上 L_aux 的 AUC 在 sim 与 real 都高于 O；显存与吞吐在第 7 节估计的 ±30% 内；缓存 → stage 4 → policy 与 port 全前向在 3 对上 `temporal` 相关 ≥ 0.9999（第一轮做过的同一检查）。
- **约 10 个单位**：全部 arm（A 的两个 λ_b、D、A-real、A-sim、A-noC、A-noK、D-only）各跑 10% 的步数，逐条对照：
  1. 完成率 100%，没有 NaN / OOM 重启；
  2. dev 漂移中位 ≤ 0.15 m（10% 步数时，比终线宽）；
  3. dev 上 sim 的 L_pair 配对准确率 > 0.6、real 的 L_aux AUC ≥ O；
  4. D 与 D-only 的门 |α| 离开 0（插件真的在用）；A-real 的门不存在（结构检查）；
  5. dev 配对 Δv̂ 与 Δv* 的同号率 > 0.5（L_Δplan 在学对方向）；
  6. 输出非退化：改后原生 plan 在 dev 正常帧上的速度分布与 O 的 KS 统计量 < 0.1。
  任一条不过就停批，写明哪一条、哪个 arm。
- **全量**：按第 2 节的 seed 数跑完，然后第 4 节的读数一次性跑完（读数代码在全量开始前冻结并提交）。

## 7. 成本与排期（第一轮实测吞吐换算）

第一轮实测：B（stage 4 解冻、trunk 缓存、fp16）384 条序列 / s，即 0.07 GPU·h / 10 万条；共卡测，偏保守。

| 项 | 量 | GPU·h |
|:--|:--|--:|
| 缓存：sim 对 trunk、YOLO 检测与 token、teacher 输出 | 约 20 万 sim 张 + 约 60 万真实 slot 的检测 | 2.5 |
| M1 基线 + R0 | | 0.8 |
| 训练，每个 arm 120 万条序列：0.84 GPU·h 纯前后向 + L_Δplan 的双成员 policy 前向约 +20% + 检测 adapter 约 +10% | ≈ 1.1 / arm | — |
| 训练：A × 4（3 seed + 1 个 λ_b 备选）、D × 3、A-real、A-sim、A-noC、A-noK、D-only | 12 次 | 13.2 |
| 读数：每个模型约 0.8（第一轮的 a / b / c 约 0.3，加 P5 原生 plan 流、Cosmos test、WOD val、NAVSIM navtest、RFS） | 12 个模型 | 9.6 |
| 分级启动的开销（1 + 10 单位） | | 1.5 |
| **合计（不含 Z）** | | **约 28** |
| Z（R0 过线才开）：tele trunk 缓存 1.0 + 训练 1.3 + 读数 0.8；过线再补 2 seed +4.2 | | 3.1（+4.2） |

一张卡一天 24 GPU·h：不含 Z 约 1.2 个卡日。排期（卡 A）：

| 时段 | 做什么 |
|:--|:--|
| 09-30 12:00 – 24:00 | 第 5 节的准备（标签、缓存、检测、teacher），M1、R0；读数代码冻结提交 |
| 10-01 上午 | 1 单位 → 10 单位，对 checklist；12:00 按 3.1 的规则定 sim 规模后起全量 |
| 10-01 下午 – 10-02 | A、D 各 3 seed 与四个对照、D-only；（Z 若开）|
| 10-02 晚 – 10-03 上午 | 读数、判格、写结果；选中的模型交给 D5–D8 闭环 lane |
| 10-03 – 10-06 | 余量：返工或 Z 的补 seed；不在这段开新 arm |

在 10-06 之前放得下，而且留出了约 3 天余量。卡 B 本轮不用。

## 8. 待用户决定的事

1. **P5 主读数改成「可见帧 ≥ 500 px_eq」**（全体 4 414 对降为副判格）：理由是 M0 量出考卷一半在 30 m 外、43% 的可见帧 < 500 px、11% 的帧行人不在画面里，而训练与真实读数的行人都集中在 10–30 m。这是在任何第二轮数字之前改读数定义，但它确实放宽了第一轮 (c) 的口径，需要用户点头。
2. **行为监督只来自 sim 的 BehaviorAgent 未来速度差**（L_Δplan）：有让真实侧 plan 变保守的风险，由 B-real 的 null 帧线（≤ +2 pp）和 N-* 兜底。是否接受这个 expert 作行为老师。
3. **navtrain（2 Hz 离协议）进训练**，比例 25%，同时进 L_aux 与蒸馏；或只进蒸馏。
4. **Z 由 R0 门控**；D 作为「分辨率的便宜修正」无条件跑。是否同意。
5. **sim 规模规则**（3.1）：Cosmos lane 在 13:33 以 93 对 drain，正在重开；如果 10-01 12:00 仍 < 500 对 K，按规则用现有的起训而不是等。是否宁可等到更多对。
6. B-real 的定义（真实侧无反事实，「行人帧比 null 帧多减速」相对原模型的变化）是否足够作为 real 的行为证据，还是只作描述、主判格只用 B-p5。

## 执行日志

- 2026-09-29 13:3x（box 时间）M0：`scripts/op_adapt_r2_pedsize.py`，CPU，数秒；此时 Cosmos 全量有 93 对（lane 状态：13:33 drained，另一个会话正在重开，本 todo 没有碰这个 lane）。
