# op-adapt 第二轮（B，sim + real）：预登记

状态: 已批准（2026-09-29 用户）— 执行等 Cosmos DONE 与 main 的 go（v1 写于 2026-09-29，v2 同日按用户决定改写并批准；任何第二轮训练、任何第二轮读数之前。文中唯一的新数字是第 0 节的 M0 行人大小测量，纯 CPU、没有模型，是设计依据，不是结果）
用户决定（2026-09-29，已并入下文各节）: Q1 P5 主读数 = 可见帧 ≥ 500 px_eq，全体为副读数（批准）；Q2 行为监督**不模仿 expert 轨迹**，改成规则打分（第 2.1 节，本版的主要改动）；Q3 navtrain 只进蒸馏，不进有监督训练；Q4 Z 由 R0 门控、D 无条件跑（批准）；Q5 Cosmos 全量完全跑完之前不起训（取消 v1 的「10-01 12:00 或 K ≥ 1 000」规则），在那之前卡给别的活；Q6 B-real 进主判格；v2 的两个遗留点（批准时定）：(a) L_dir 在「可见但 < 500 px、靠近路径」一档不强制 plan 相同，(b) 让行豁免只给移动的 actor，静止障碍挡路且有安全绕行候选时干等要扣进度分（第 2.1 节）。
主题: [research/decisions.md](../research/decisions.md) 第 42、44、45、48、50、53、55、56、62、63 条；[research/midterm-gaps.md](../research/midterm-gaps.md) 缺口二；[research/feature-adapter-domain-shift.md](../research/feature-adapter-domain-shift.md)
前身: [op-adapt 第一轮](2026-09-28-op-adapt.md)（port、吞吐、B 小试、后续 1 / 2）；数据: [Cosmos G4 全量](2026-09-28-cosmos-pilot.md)；E0 / E1: [e0](2026-09-29-e0-layer-probe.md)、[e1](2026-09-29-e1-cosmos-probe.md)；打分口径与 [WL-2](2026-09-29-wl2-prereg.md) 的 cg 标签对齐
排期约束: [tmp/2026-09-29-midterm-plan.md](../tmp/2026-09-29-midterm-plan.md)（D2–D5 卡 A 给本轮，D5–D8 闭环要用本轮选出的模型）；起训以 Cosmos 全量 lane 的 DONE 为准（Q5）

## 为什么要做

第一轮（第 55 条）说明 openpilot 的 vision 后段可以便宜地改、改了不坏（跨数据集漂移 6 cm），但只用真实 nuScenes 监督，真实行人可读性只涨 +0.08（没过 +0.10），CARLA 完全不动（+0.009）。
按验收原则（真学会了，sim 与 real 都能开），这不是正例。缺口二（midterm-gaps 2.1）要的正是一个「同一次轻度适配让 openpilot 在 CARLA 与真实上都更会对行人反应、正常驾驶不变差」的结果，而且要有**行为**读数，不只是线性 probe。

这一轮开始前，E0 / E1 改变了问题的形状：
- E0（第 62 条）：P5 考卷上 CARLA 行人在每一层都读不出（0.51–0.53）。
- E1（第 63 条）：Cosmos 全量前 75 对上，CARLA 行人本来就读得出（池化 stage 3 0.815，框内 cell 0.95），Cosmos 画风反而略低（−0.07）；可读性由行人大小决定（< 500 px 两格都 0.57，≥ 1 500 px 0.92 / 0.83）。
- 所以 CARLA 与真实之间的行人差，很可能主要是**大小 / 分辨率**，不是画风。本轮在冻结登记前先量了这件事（第 0 节），结果决定了 P5 读数的分档和「分辨率修正」要不要进 arm。

行为监督怎么给，v2 改了。v1 让原生 plan 的 x⁺ / x⁻ 速度差去拟合 BehaviorAgent 的速度差（L_Δplan）。用户否决，理由三条：BehaviorAgent 有特权视野（对相机还看不见的行人就开始反应）；它开得比 openpilot 从人类数据学来的风格差；而且回归一条 expert 曲线会把模型绑到这一个 expert 的刹车曲线上。
v2 换成**规则打分、没有 expert 目标**：一条 plan 好不好，只看它撞不撞、在不在可行驶路面上（为绕行借道可以，只要那块路面可行驶且没人）、有没有无故停滞；为路径上的行人让行 / 停车本身不扣分。
打分对一组候选 plan 进行（openpilot 自己的 plan 加结构化扰动），训练把 plan 拉向打分最好的候选（第 2.1 节），sim 与真实用同一个打分器。

名词（第一次出现时注释，之后直接用）：
- **B**：第一轮的适配方式，openpilot Cinque 的 stage 4（vision 编码器最后一段，约 1.1 亿参数）解冻，stage 1–3（**trunk**）冻结并按帧缓存，policy（时序 transformer 与全部输出头）冻结。
- **蒸馏（distillation）**：改后模型在正常帧上的输出贴住原模型的输出；**漂移（drift）**：同一帧上原生 plan 与原模型的差。
- **配对（pair）**：同一世界、同一 ego 轨迹的 x⁺（有行人）/ x⁻（行人藏到地下）；**C / K**：同一对的 CARLA 原画 / Cosmos 重画（第 56 条 G4），所以每对是四元组 C⁺ C⁻ K⁺ K⁻。
- **D0 probe**：第 42 条的配对可分性 probe（冻结特征 + 类平衡 L2 logistic，按 route 分折，x⁺ 对 x⁻ 的样本外 AUC，route 聚类 bootstrap）。
- **翻转（flip）**：P5 考卷上 x⁺ 与 x⁻ 的预测 2 s 速度之差超过考生自己 null 分布的 95 分位，且 x⁺ 更慢；**null false-flip**：只换 TM seed 的 null 对上的同一统计量。第 48 条的原规则写「方向与 expert 一致」，行人 scope 上 expert 的方向一律是减速，所以是同一条规则，读数不依赖 expert 轨迹。
- **等效像素（px_eq）**：行人可见 mask 像素数换算到 Cosmos 相机（1280 × 704、64° HFOV、f = 1 024 px）；E1 的大小分档就是这个单位。openpilot 的 road 模型帧（f = 910）看到的面积是它的 0.79 倍，wide 帧 0.20 倍。
- **PDM 打分器**：NAVSIM 的 PDM scorer（nuPlan 的 PDM-Closed 规则打分），PDMS = NC · DAC · (5 EP + 5 TTC + 2 C) / 12，NC 不碰撞、DAC 留在可行驶区、EP 相对进度、TTC 碰撞时间、C 舒适度。
- **非反应式（non-reactive）打分**：其他 actor 按记录下来的未来轨迹走，不对候选 plan 做反应；**at-fault**：只算本车负责的事件（本车静止时、或对方从后面撞上来的不算，PDM 规则）。
- **S_jev、候选集 C(s)、Top 集**：第 2.1 节定义的打分、每个场景 s 的候选 plan 集合、分数在最高分 δ 以内的候选。

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

1. 三个带标签的域一起训（Cosmos 配对、同一批配对的 CARLA 原画、真实 WOD / nuScenes），openpilot 能不能在**真实留出**与**CARLA 留出**上同时读出更多行人，并且**原生 plan 真的对行人减速**，同时正常驾驶不坏。这是验收原则的第一个正例候选。
2. 行为监督不用 expert 轨迹、只用规则打分（S_jev）时，plan 能不能学会对行人反应，而且在 sim 与真实两边用同一个打分器监督。
3. 把 COCO YOLO26x 的 image-plane 检测 token（第 50 条配方）在 policy 之前注入、与 (1) 的可训练部分端到端一起训，是否比纯视觉更好，尤其在小 / 远行人上。
4. 效应归因：real-only、sim-only、去掉 CARLA 原画、去掉 Cosmos 各自带走多少；规则打分对 expert 回归（A 对 A-bhv）。
5. （门槛之后）一个更高分辨率的视图能否让 < 500 px 的行人变得可读。

## 2. Arms

所有 arm 共用：Cinque（3.8 亿参数），port 的 fp16 路径（第一轮 P3 定），trunk 缓存，9 个 5 Hz context slot（1.6 s），原模型作 teacher，同一套数据划分和读数。可训练部分以外全部冻结。

| arm | 可训练 | 训练数据 | 损失 | seed | 回答什么 |
|:--|:--|:--|:--|--:|:--|
| O | —（原模型） | — | — | — | 所有读数的参照；读数与训练前的 M1 一起跑 |
| **A**（主，纯视觉 B） | stage 4 + 辅助头 | sim（C + K 配对）+ real（WOD、nuScenes；navtrain 只进蒸馏） | L_aux + L_pair + λ_s·(L_score + L_dir) + λ_d·L_distill | 3 | 纯视觉 sim + real 能不能两边都学会 |
| **D**（主，检测插件） | A 的全部 + 检测 token adapter | 同 A | 同 A | 3 | 检测 token 注入是否在 A 之上再加 |
| A-real | 同 A | 只有 real（sim 只进蒸馏） | L_aux + λ_s·L_score（nuScenes）+ λ_d·L_distill | 1 | G-b：只加真实数据量与真实打分监督能不能过线，CARLA 动不动 |
| A-sim | 同 A | 只有 sim 配对（real 只进蒸馏，不用真实标签） | L_aux + L_pair + λ_s·(L_score + L_dir) + λ_d·L_distill | 1 | sim 监督单独能否带到真实 |
| A-noC | 同 A | K + real（去掉 CARLA 原画） | 同 A | 1 | OpenVLA 的教训：原画作为带标签的域是否必需（P5 是原画） |
| A-noK | 同 A | C + real（去掉 Cosmos） | 同 A | 1 | Cosmos 画风对真实侧是否有贡献（E1 说它对可读性没有） |
| D-only | 只有检测 adapter + 辅助头（stage 4 冻结） | 同 A | 同 A | 1 | 检测 token 单独能做到多少 |
| A-bhv（对照） | 同 A | 同 A | L_aux + L_pair + λ_b·L_Δplan（v1 的 BehaviorAgent 速度差回归，sim）+ λ_d·L_distill；λ_b = 1，不调 | 1 | 规则打分对 expert 回归：同一数据下哪个的行为读数更好、真实侧更不保守 |
| Z（门槛后开） | A 的全部 + tele 视图 adapter | 同 A | 同 A | 1（过线再补 2） | 分辨率修正 |

A-bhv 留作 1 seed 对照。L_Δplan 即 v1 的定义：原生 plan 在 x⁺ 与 x⁻ 上的纵向速度差 Δv̂(t)（t = 1, 2, 3 s）用 Huber 拟合 BehaviorAgent 在两个世界里的实际未来速度差 Δv*(t)（第 1 遍 `pose.jsonl`），按训练集 |Δv*| 的标准差归一。目标从已有的第 1 遍位姿 CPU 算出，训练与读数合计约 2 GPU·h。它不进任何主判格，只回答「不模仿 expert 是不是更好」。

**λ_s 的选择（唯一调的超参）**：只在 A seed 0 上跑两个预登记配置 λ_s ∈ {0.3, 1}，用 dev 选：dev 漂移中位 ≤ 0.10 m 且 dev null 减速率比原模型高 ≤ 2 pp 的配置里，选 dev x⁺ slot 上「原生 plan 的 S_jev − 原模型的 S_jev」均值最高的；两个都不满足就选漂移小的。其余 arm 与 seed 继承选中的 λ_s。λ_d = 10（第一轮 dev 选出的），不再调；δ = 0.1、H = 4 s 等打分器常数写死（第 2.1 节），不调。

### 2.1 A：纯视觉 B 的损失

- **L_aux（可读性，WOD、nuScenes、sim）**：第一轮的两个辅助头（`select_4` 上的 MLP、32 个 hidden token 上的 attention pooling），三个输出：走廊内行人有无（主）、宽走廊内行人有无、最近走廊行人距离（只在正例上回归）。
  标签：nuScenes = GT 框（第一轮原样）；WOD = YOLO26x 平地抬升（`uncertain` 帧遮掉行人损失，后续 2 原样）；Cosmos / CARLA 对 = CARLA actor 位置套同一走廊定义（x⁻ 全部为负，x⁺ 按几何定；行人 px_eq < 68 的 x⁺ slot 标为不可见、不进 L_aux，但进 L_score）。navtrain 不进 L_aux（Q3）。
- **L_pair（配对差分，sim）**：同一 slot 的 x⁺ 与 x⁻ 的主 logit 做 margin ranking，softplus(1 − (z⁺ − z⁻))，只在 x⁺ 行人可见的 slot 上，C 与 K 各算一份。两个成员除行人区外逐像素相同（K 按构造，C 按渲染），所以这个损失只能靠行人区降下来。
- **L_score 与 L_dir（行为，sim 与 nuScenes）**：见下面的打分器；这是 v2 的主要改动。
- **L_distill**：第一轮原样（全部输出去掉 hidden 与 MDN std，按原输出标准差归一的 MSE），加在真实「正常帧」（宽走廊内无行人 / 骑车人；WOD、nuScenes、navtrain）和 sim 的 x⁻ 帧上。
- 优化：AdamW，stage 4 lr 3e-5、新模块 1e-3、wd 0.01，batch 64（按「序列」计：一个 sim 四元组 = 4 条序列），cosine，fp16 + GradScaler（第一轮原样）。

#### 打分器 S_jev（一条 plan 好不好）

对场景 s（一个 5 Hz slot）里的一条候选 plan τ，把本车 footprint 沿 τ 在 H = 4 s 内按 10 Hz 推进（openpilot plan 在 T_IDXS 上的点插值；不做 PDM 的 LQR 跟踪仿真），对照被打分 actor 集 A(s) 的真实未来轨迹（非反应式），算：

```
S_jev(τ) = NC(τ) · DAC(τ) · [ 5·P(τ) + 5·TTC(τ) + 2·C(τ) ] / 12
```

- **NC ∈ {0, 1}（不碰撞，= WL-2 的 cg）**：任一时刻本车框与 A(s) 中任一 actor 框重叠，或本车走廊内（本车当前位姿坐标系下 |y| ≤ 1.75 m、前方）最近 actor 的纵向间距（前保险杠到 actor 参考点，`jevdrive/wl.py::_gap_front` 同一算法）< 2 m，就是 0。两条都只算 at-fault：该时刻本车速度 ≥ 0.5 m/s，碰撞点不在本车后半部。PDM 把撞静物记 0.5，这里一律记 0，与 cg 一致。
- **DAC ∈ {0, 1}（可行驶区）**：每个时刻 footprint 四角都在可行驶区内。可行驶区 = 任意方向的行车道 + 路口 + 停车道（CARLA：OpenDRIVE 的 Driving / Bidirectional / Parking 与 junction，不含 Sidewalk / Shoulder / Border / Median，按路线两侧 ±40 m、0.2 m 栅格离线生成；nuScenes：map expansion 的 `drivable_area` 层）。所以借对向车道或相邻车道绕行不扣 DAC，那块路上有没有车交给 NC 与 TTC。
- **P ∈ [0, 1]（进度，带让行豁免）**：prog(τ) = τ 在 H 末端沿参考路径（CARLA：route 中心线；nuScenes：原模型 plan 的路径）的投影弧长。
  P(τ) = min(1, prog(τ) / max(5 m, max_{k∈N(s)} prog(k)))，N(s) 是 C(s) 中 NC · DAC = 1 的候选。
  **让行豁免（只给移动的 actor，用户 2026-09-29 定）**：若沿车道候选（下面候选集里 `op`、`hold` 两条中至少一条）的 NC = 0 是由行人、骑车人或移动中（≥ 0.5 m/s）的车辆造成的，N(s) 只取**不离开车道**的安全候选（横向偏移 ≤ 1 m）。
  所以：没有人挡路时，无故慢 / 停会被最好的安全候选拉低 P（「不停滞」）；有人在路径上时，停下让行拿满 P，绕行也拿满 P（被 min(1, ·) 截住），两者都不被要求；静止障碍（停着的车、锥桶）挡路且有安全绕行候选时，不豁免，干等的 P 被绕行候选拉低（第 52 条 P6 的「该绕」）。5 m 下限沿用 PDM 的 progress 门槛：谁都走不到 5 m 时全部记 1。
- **TTC ∈ {0, 1}**：PDM 的定义与参数（devkit `default_scoring_parameters` 同值）：本车在运动时，从 τ 上每个时刻按当时速度与朝向外推 1 s，若与同一时刻前方 actor 的框相交记 0。它按框相交判，不按 WL-1 的「走廊内间距 / 接近速度」判，所以 WL-2 去掉 TTC 的理由（横移绕过的过程中对被绕物体接近速度高，被记成 unsafe）在这里不成立：绕开以后外推框不再相交。
- **C ∈ {0, 1}（舒适）**：PDM 的阈值原样（纵向加速度、横向加速度、jerk、yaw rate、yaw 加速度）。

**被打分 actor 集 A(s)（反特权视野）**：只打相机在当前 slot 或 9 个 context slot 里**看得见**的 actor。用户否决 expert 的第一条理由（对看不见的行人先反应）同样适用于一个拿真值未来的打分器，所以打分器也要被相机的视野截断。
sim：hazard 行人按分割视图像素 px_eq ≥ 68（P5 的可见门槛）；背景车辆与骑车人按前视相机 FOV 内且 ≤ 60 m（不做遮挡测试）。真实（nuScenes）：GT 框中心投影在 CAM_FRONT 内且 visibility token ≥ 40%。
代价：一个马上要从遮挡后走出来的行人在它出现之前不影响打分，打分器会认为「照开」是好的；这是有意的，相机不可能知道。描述读数报「用全部 actor 打分与只用可见 actor 打分，Top 集不同的 slot 比例」。

**候选集 C(s)（13 条，没有一条来自 expert）**：
- openpilot 自己的 plan：Cinque 的 plan 头是单一 MDN 均值（33 × 15），没有多个 mode，所以「openpilot 自己的备选」取 desire 条件下的原模型 plan：`op`（desire 0）、`op_L` / `op_R`（laneChangeLeft / Right desire 脉冲，第 49 条：方向总对、幅度不够，作为横向候选之一）。
- 结构化扰动：WL-2 的 11 个候选原样（`jevdrive/wl_traj.py`，候选定义与 WL-2 同一份代码）：`op`、`op_slow`（速度 × 0.5）、`op_stop`（−4 m/s² 到停）、`hold`（参考路径匀速）、`brake_hard`（−6 m/s²）、`brake_mild`（−2 m/s²）、`shift_L` / `shift_R`（±3.0 m，2 s 余弦过渡）、`shift_L_slow` / `shift_R_slow`、`nudge_L`（1.5 m）。真实侧没有 route，`hold` / `brake_hard` 的路径用 `op` 的路径。
- 候选由原模型在同一帧（同一种画面：C 帧用 C 的 plan，K 帧用 K 的 plan）上的输出生成，离线打分并缓存；训练中不在线打分。

**Top 集**：Top(s) = {k ∈ C(s) : S_jev(k) ≥ max_j S_jev(j) − δ}，δ = 0.1。max_j S_jev(j) = 0（候选里没有一条不撞、不出界）的 slot 不进 L_score。

#### L_score：拉向最近的好候选

```
L_score(s) = min_{k ∈ Top(s)} d(μ_θ(s), τ_k)
d(μ, τ) = mean_{t ∈ T_IDXS, t ≤ 4 s} Σ_{c ∈ {x, y, v, a}} Huber( (μ_c(t) − τ_c(t)) / σ_O,c(t) )
```

μ_θ 是改后模型的 plan MDN 均值，σ_O 是原模型在同一帧上的 MDN std（openpilot 自己的尺度）；候选的 v、a 由候选的弧长时间曲线求导。
- 这是 winner-take-all（多假设训练常用的「只罚最近的那个」）：模型自己选离它最近的好候选，不被逼到某一条，也不会被平均到「左绕与右绕的中间」（那正好撞人）。
- **没有 expert 目标**：Top 集完全由规则打分决定，expert 只在 A-bhv 对照里出现。
- **与蒸馏同形**：`op` 在 C(s) 里，训练开始时 μ_θ = `op`。只要原模型 plan 本来就在 Top 集里，损失为 0、梯度为 0，之后只要改后模型漂离，就被拉回最近的好候选（通常就是 `op`）。损失只在打分器判原模型 plan 不够好的 slot 上有推力，推向「改得最少的好 plan」。
- 用在哪些 slot：sim 的 x⁺ slot（C 与 K 各一份，候选各自生成，打分几何相同）；nuScenes train 的 VRU 帧（CAM_FRONT 里可见的行人 / 骑车人在宽走廊 0–30 m 内）。正常帧与 x⁻ 帧交给 L_distill，不上 L_score，避免打分器在普通驾驶上与蒸馏拉扯（打分器在正常帧上与原模型不一致的比例作描述报，是打分器本身的 sanity 读数）。

#### L_dir：配对方向约束（sim）

在同一对的 x⁺ / x⁻ 上（C、K 各一份）：

```
L_dir = g_dir · mean_{t ∈ {1,2,3,4 s}} relu(v̂⁺(t) − v̂⁻(t)) / σ_v  +  g_eq · d(μ_θ(x⁺), sg[μ_θ(x⁻)])
```

- g_dir = 1：行人在该 slot 可见且 px_eq ≥ 500（Q1 的主读数档），且在走廊内或附近（行人 4 s 真实未来与 `op` / `hold` 路径的最小横向距离 ≤ 1.75 + 1.0 m）。x⁺ 不许比 x⁻ 快；慢多少由 L_score 决定，这里只管方向。
- g_eq = 1：行人不可见（px_eq < 68），或可见但与路径无关（最小横向距离 > 1.75 + 3.0 m，且 x⁺ 与 x⁻ 的 Top 集相同）。两个 plan 应相同；x⁻ 一侧 stop-gradient，它已被蒸馏钉住。
- 其余（可见但 < 500 px 且靠近路径，或处在两个距离门槛之间）不加配对约束，只由 L_score 管。「小而近」这一档留空（用户 2026-09-29 批准）：openpilot 读不出这么小的行人，逼两边相同等于教模型忽略路径上的小行人，与 L_score 相反。

#### 非反应式打分的偏差与处理

| 偏差 | 在哪 | 处理 |
|:--|:--|:--|
| 场景行人按本车距离触发：记录下来的行人未来是在 expert 的到达时刻触发的，慢一些的候选在现实里会晚触发 | sim | 触发之前的 slot，hazard 行人按当前位置冻结（静止）打分，不用它将来的轨迹；触发 tick 取录制器的 `_t_trig` |
| 场景 actor 在 expert 通过后被销毁、第 1 遍在通过行人 10 m 后 0.5 s 收尾：慢候选的未来超出记录 | sim | H 截到记录末端；可用长度 < 3 s 的 slot 不进 L_score 与读数 |
| 背景车对 expert 的驾驶有反应（后车跟刹、对向车让行） | sim、真实 | at-fault 规则排除后方来车；x⁺ 与 x⁻ 各用自己世界的第 1 遍 actor（分叉后两边背景车可能不同）；残余偏差不处理，结论里写 |
| 真实行人对 log 里的人类司机有反应（等车过去再走） | 真实 | H 只取 4 s；「让行豁免」不要求绕行也不要求照开；残余偏差写进限定 |
| 不做 PDM 的 LQR 跟踪仿真，plan 直接按点推进 | 两边 | C 项可能偏乐观；N-nav 仍用 devkit 原样（有跟踪仿真） |
| 可见性截断让打分器看不到将要出现的行人 | 两边 | 有意为之；描述报全 actor 与可见 actor 两种 Top 集的分歧率 |

#### 与 WL-2 的口径：哪里相同、哪里不同

| 项 | WL-2（cg 主标签，已批准、不改） | 本轮 S_jev |
|:--|:--|:--|
| 不安全 | 碰撞 或 本车道间距 < 2 m（`_gap_front`） | NC 同一个定义、同一个间距算法 |
| 结果怎么来 | CARLA 闭环分支，反应式世界，碰撞传感器 | 非反应式真值轨迹 + 框重叠；按候选离线算 |
| at-fault | 不区分，任何碰撞都算 | 本车静止、被追尾不算 |
| TTC | 主标签去掉 TTC；次要标签含「走廊间距 / 接近速度 < 1 s」 | PDM 的框相交外推 TTC，作加权项不作门 |
| 进度 | 3 s 行驶距离，原值 | 4 s 投影进度，按安全候选归一，带让行豁免 |
| 可行驶区、舒适 | 没有 | 有（DAC 门、C 加权项） |
| 候选 | 11 个（`wl_traj.py`） | 同样 11 个 + `op_L` / `op_R` |

一致性检查（第 5 节 V1）：把 S_jev 的 NC 用在 WL-1 已有的 2 814 个分支 run 的实际轨迹与记录 actor 上，与 WL 的 cg 标签的一致率要 ≥ 95%（差异只该来自 at-fault 与重叠判法）。WL-2 自己的登记不动；若它以后要换成 S_jev 口径，另行登记。

#### 真实数据上哪里能用打分监督

| 数据 | 有没有 agent 未来轨迹 | 打分监督 | 理由 |
|:--|:--|:--|:--|
| nuScenes train | 有：GT 框 + instance 跟踪，2 Hz keyframe（插值到 10 Hz），map expansion 有 `drivable_area` | **用**（L_score） | 唯一既有真值轨迹又在第一轮协议下训练过的真实集 |
| NAVSIM navtrain | 有（metric cache 里的 agent 未来，devkit 自带 PDM 打分器） | **不用**，只进蒸馏 | (1) Q3：navtrain 只进蒸馏；打分监督就是有监督训练。(2) 第 36、53 条：navtrain 的 2 Hz sample-and-hold 输入是离协议的，在它上面学到的读出搬不到 5 Hz 的 CARLA / WOD（E2 的 navtrain 对到 WOD 上有害）。(3) 用 navtrain 的 PDM 打分训练，N-nav（navtest PDMS）就从「不坏」检查变成被优化的目标，是第 35 条说的 R 层配方，失去独立性 |
| WOD（E2E） | 没有 agent 轨迹（只有相机与 ego），YOLO 单目抬升没有跟踪、距离噪声大 | 不用；只进 L_aux 与蒸馏 | 没有可信的 actor 未来 |

所以真实侧的行为监督只来自 nuScenes（约 2.8 万个 keyframe，其中走廊行人帧 1 344 个（M0），宽走廊 VRU 帧更多但同一量级），比 sim 的 x⁺ slot 小一个量级以上；真实侧的行为读数（B-real、B-score-nus）因此在留出的 nuScenes val 与从未进过打分监督的 WOD val 上都报。

### 2.2 D：检测插件

- 检测：YOLO26x-seg 640（第 45 条的快通道模型），**只用前视相机的原生分辨率图像**（Cosmos 1280 × 704、P5 Waymo 前视、WOD 前视、nuScenes CAM_FRONT、navtrain CAM_F0），每个 5 Hz slot 取 pedestrian / cyclist / vehicle 里 score 最高的 8 个。
- token（第 50 条 image-plane 配方，不抬升不筛走廊）：类别 one-hot、score、框（重投影到 openpilot road 模型帧的归一化坐标，使不同相机可比）、框高 / 焦距（角尺寸）、16 维检测外观（YOLO neck 的 RoIAlign，PCA 在 WOD train 上拟合）。
- 注入：在 stage 4 之后、进入 policy 队列之前的 32 × 512 hidden token（`view_39`）上加一层 cross-attention（hidden 为 query，检测 token 经 MLP 投到 512 维为 key / value，8 头），输出投影**零初始化**、再乘一个初值 0 的标量门。
  所以 D 在第 0 步与原模型逐位相同，蒸馏从零漂移起步；部署时每帧的注入 hidden 进入队列，自然成为后续帧的 context，训练时 9 个 slot 都注入（每个 slot 都要有检测）。约 130 万参数。
- 为什么这算「分辨率修正」的便宜版：YOLO 看的是原生分辨率（Cosmos 上 f = 1 024，比 road 帧的 910 高、比 wide 帧的 455 高一倍多），小行人在检测器里比在 openpilot 的 512 × 256 帧里多得多的像素。第 45 条：YOLO 在 P5 hazard 行人 ≤ 20 m 召回 0.89，图像平面 20–40 m 看见 46–61%。

### 2.3 Z：tele 视图（先过零训练门槛 R0 才开；Q4 已批准）

- 做法：同一前视图像按 f = 1 820（road 帧焦距的 2 倍）重投影成一张 512 × 256 的 tele 帧（视场约 16°，对准 road 帧中心），过同一个冻结 trunk 和共享的 stage 4，32 个 token 经与 D 相同结构的零初始化 cross-attention 注入 hidden。成本：每 slot 多一次 trunk 前向，trunk 缓存 × 2（约 +200 GB，盘上 1.3 TB 空）。
- **R0（零训练，数据准备时顺带跑，≤ 0.3 GPU·h）**：原模型的 stage 3 在 tele 帧上，E1 的池化 probe 与框内 cell probe，在 Cosmos 对与 P5 的 < 500 px_eq 档各算一次。
  **开 Z** = 任一集合 < 500 档的池化 stage 3 AUC ≥ 0.65（E1 现为 0.57）且对 road 帧同档的配对 Δ CI 下界 > 0；否则 Z 不开，结论写成「在 openpilot 的 trunk 上，放大输入不让小行人变可读」，这同时是第 63 条「分辨率限制」推测的直接检验。
- 为什么不直接开：tele 视图是新的输入流，改动比 D 大，且 E1 的 < 500 档在两种画风下都读不出，放大能否救回是未知的；R0 几乎免费，先答这个问题。

## 3. 数据

### 3.1 三个带标签的域

| 域 | 来源 | 规模 | 标签 | 用在哪些损失 | 状态 |
|:--|:--|:--|:--|:--|:--|
| sim-K | Cosmos G4 全量的 K⁺ / K⁻ | 目标 2 000 对；lane 13:39 重开后在跑（本 todo 不碰这个 lane） | CARLA actor（第 1 遍，含窗口之后的未来）+ 走廊 + 可行驶区 | L_aux、L_pair、L_score、L_dir、蒸馏（x⁻） | 进行中 |
| sim-C | 同一批对的 C⁺ / C⁻（`load_pair(pair, "carla")`） | = sim-K 的对（原画在 controls 阶段就有，但按 Q5 一起等全量结束） | 同上 | 同上 | 随 CARLA 第 2 遍产出 |
| real-WOD | WOD-E2E train | 207 678 个 slot、113 068 帧有标签（走廊 2.4%、宽 9.3%） | YOLO 抬升，11.7% uncertain；无 agent 轨迹 | L_aux、蒸馏 | trunk 缓存已有 |
| real-nus | nuScenes train 650 个 scene（第一轮的训练集） | 约 2.8 万个 keyframe（走廊 4.8%） | GT 框 + instance 未来 + `drivable_area` | L_aux、L_score、蒸馏 | trunk 缓存已有；map expansion 若 box 上没有先下（约 0.4 GB） |
| real-nav | NAVSIM navtrain | 103 288 个 token | — | **只进蒸馏**（Q3） | trunk 缓存已有 |

sim 的每对取窗口里 slot ≥ 9 的 5 Hz slot（与 E1 同：9 个 context slot 都在窗口内），每对约 15 个训练 slot；每个训练样本是同一 slot 的四元组（sim-noC / noK 的 arm 是二元组）。

**sim 起训规则（写死，Q5）**：Cosmos 全量 lane 写出 DONE（全部对完成，或 main 让它以某个对数收尾）之前，本轮不起任何训练；在那之前卡给别的活，本轮只做第 5 节里不需要 Cosmos 结果的准备（CPU 为主，GPU 只有零训练的缓存与读数，每项 < 1 GPU·h）。
DONE 之后按最终对数：sim-K ≥ 200 对全部 arm 照跑；< 200 对则 A-noC 不跑（K 太少、没有意义），其余照跑；规模写进结果。

### 3.2 划分

| 用途 | sim（Town12 的 209 个实例） | real |
|:--|:--|:--|
| train | 75% 实例（全部变体） | nuScenes 650 scene 去掉 dev；WOD train 95% sequence；navtrain 95% log（只蒸馏） |
| dev（只用于 λ_s 选择与早停检查） | 10% 实例 | nuScenes 50 scene（第一轮同一批）；WOD train 5% sequence；navtrain 5% log |
| test（Cosmos 留出） | 15% 实例，C 与 K 两种画面都考 | — |
| 留出考卷 | P5 v1 BA（42 条行人路线，训练对已按 60 m 排除） | nuScenes val 150 scene、WOD val 60 条 stream（第一轮同一批）、NAVSIM navtest、WOD-E2E val |

按实例划分，所以同一实例的不同天气 / TM seed 变体不会跨 train / test。划分用 seed 0 的固定排列，写进 run dir。

### 3.3 混合与过采样

- 每个 batch 按序列计 50% sim、50% real。sim 内 C : K = 1 : 1（四元组天然如此）；real 的有标签部分 WOD : nuScenes = 60 : 40（v1 的 45 : 30 : 25 去掉 navtrain 后按比例放大）。
- 正例权重：主 BCE 的 pos_weight 使每个域的有效正例率约 25%（第一轮做法）。
- **大小 / 距离过采样**：real 正例按最近走廊行人距离分档，10–20 m 与 20–30 m 档的采样权重 × 2（0–10 m、30 m+ × 1）；sim 的 x⁺ slot 按 px_eq 分档，100–500 与 500–1 500 档 × 2。理由：第一轮的增益在 10–20 m（+0.14）和 > 20 m（+0.10），E1 的可读性断崖在 500–1 500 px；考卷一半在 30 m 以外，但训练里几乎没有，那一档不做人工放大（数据不支持）。
- nuScenes 的 VRU 帧（L_score 的真实来源）在 nuScenes 份额内 × 2，使真实打分监督每个 batch 约 3–4 条。
- 蒸馏帧：real 正常帧（WOD、nuScenes、navtrain 按 40 : 30 : 30）+ sim x⁻，占每个 batch 的 25% 序列（与上面的比例并行抽，不挤占标签样本）。
- 训练量：每个 arm 120 万条序列（第一轮 50 万的 2.4 倍），约 2–3 个 sim epoch、real 约 1 个 epoch。

## 4. 读数与登记线

全部在留出数据上，改后模型与原模型 O 在同一次前向、同一套 probe 里并列算。「D0 / (a) probe」沿用第一轮与第 42 条的代码，不改。

### 4.1 可读性

| 编号 | 读数 | 集合 | 登记线（主 arm A、D 每个 seed 都要过，才算过） |
|:--|:--|:--|:--|
| R-nus | 走廊行人 `temporal` probe AUC Δ（改后 − O） | nuScenes val（203 / 6 019） | Δ ≥ +0.10 且配对 CI 下界 > 0（第一轮原线） |
| R-wod | 同上 | WOD val：60 条 stream 的读数帧，走廊行人标签用 `labels_wod` 同一规则在 val 上补做（CPU） | Δ ≥ +0.05 且 CI 下界 > 0 |
| S-p5 | D0 `temporal` AUC | P5 v1 BA 行人 scope，**可见帧 ≥ 500 px_eq**（2 092 对，≥ 31 条路线；Q1） | ≥ 0.65 且对 O 的配对 Δ CI 下界 > 0 |
| S-p5-all | 同上（第一轮延续） | P5 行人 scope 全部 4 414 对 | ≥ 0.60 且 Δ CI 下界 > 0（第一轮 (c) 原线；只作副判格，不进主格） |
| S-cos | 配对 AUC（E1 的池化 probe，按实例 5 折） | Cosmos test 实例，C 与 K 各一个 | 两种画面 Δ 都 ≥ +0.05 且 CI 下界 > 0 |

每个可读性读数都按大小分档另报（描述）：P5 与 Cosmos 按 px_eq（< 500、500–1 500、≥ 1 500），真实按距离（0–10、10–20、20–30 m）。

### 4.2 行为

主判格的两条（B-p5、B-real）都不用 S_jev，所以不和训练目标循环；S_jev 读数作副判格，写明它是被优化的量。

| 编号 | 读数 | 集合 | 登记线 |
|:--|:--|:--|:--|
| B-p5（主） | 原生 plan 2 s 速度的行人翻转率，第 48 条规则 | P5 v1 BA 行人 reactive 帧中可见 ≥ 500 px_eq 的帧（`p5_exam.exam` 原样，考生 = 改后原生 plan）；全部 reactive 帧并列作描述 | 每个 seed：翻转 CI 下界 > 该考生样本外 null false-flip + 10 pp；且 null false-flip ≤ O 的 + 3 pp |
| B-real（主，Q6） | 减速率差：Δ = [slow(行人帧) − slow(null 帧)]_改后 − [同]_O；slow = plan 2 s 速度 < 当前速度 − max(1 m/s, 20%) | nuScenes val ∪ WOD val：走廊行人 0–30 m 的帧对「宽走廊内无行人 / 骑车人」的帧，按本车速度 5 档分层匹配；两个数据集另各报一次（WOD val 从未进过打分监督） | Δ ≥ +5 pp 且 scene / sequence 聚类 CI 下界 > 0；null 帧的 slow 率比 O 高 ≤ 2 pp |
| B-score-p5（副） | 原生 plan 的 S_jev 配对 Δ（改后 − O）；另报 NC 失败率（cg） | P5 v1 BA 行人 reactive 帧的 x⁺ 与 x⁻（P5 自己第 1 遍的 actor，Town 与场景都不在训练里） | x⁺：Δ 的 route 聚类 CI 下界 > 0；x⁻：Δ ≥ −0.02（不靠一律减速换分） |
| B-score-nus（副） | 同上 | nuScenes val：VRU 帧；正常帧 | VRU 帧 Δ 的 scene 聚类 CI 下界 > 0；正常帧 Δ 的 CI 下界 ≥ −0.01 |
| B-cos（描述） | 原生 plan 的 S_jev Δ 与「落在 Top 集 0.5 m 内」的比例；x⁺ / x⁻ 2 s 速度差 | Cosmos test，C 与 K | — |
| B-log（描述） | 行人帧上 plan 对 log 未来的 ADE 配对差；log 减速的行人帧里 plan 也减速的比例 | nuScenes val、WOD val | — |
| B-ref（描述） | 参照行：nuScenes val 上 log 的人类轨迹、P5 上 BehaviorAgent 实际轨迹的 S_jev | 同上 | —（只用来看打分器的量级，不作目标） |

真实侧没有反事实，所以 B-real 读的是「有行人比没行人多减速多少」相对原模型的变化；它不能区分「对行人反应」和「对行人常出现的场景反应」，这一点用 null 帧的速度分层和 B-log 部分对冲，结论里要写明。
B-score-nus 在真实侧补上「减速是不是对的」：同一帧上改后 plan 与原 plan 谁更不撞、更不无故停，但它和 L_score 是同一个打分器，在 nuScenes 上训、在 nuScenes val 上读，只能作副判格（第 35 条：对准 metric 的增益属于 R 层）。

### 4.3 不坏（no-harm）

| 编号 | 读数 | 登记线 |
|:--|:--|:--|
| N-drift | 正常帧原生 plan 漂移（第一轮 (b) 原样，nuScenes val 与 WOD val） | 两处中位 ≤ 0.10 m、p95 ≤ 0.50 m |
| N-ade | WOD val 原生 plan 对 log 未来的 ADE 相对变化 | CI 上界 ≤ +2% |
| N-lead | lead x 与 lead_prob（第一轮原样） | lead_prob \|Δ\| 中位 ≤ 0.02，lead x \|Δ\| 中位 ≤ 0.5 m |
| N-nav | NAVSIM navtest PDMS（原生 plan，与原模型同一 harness：第 36 条补帧输入、同一 2 000 token 子集或当时的全集，二者同一次跑）；navtrain 不进打分监督，所以这仍是独立的不坏检查 | ≥ O − 1 |
| N-rfs | WOD-E2E val RFS（与第 44 条同一 harness） | 配对 Δ 的 CI 下界 ≥ −0.10 |
| N-cutin | P5 cut-in 翻转（车辆不能被弄坏） | 不低于 O 5 pp 以上 |

### 4.4 捷径检查（描述，但出问题就写进判格）

- **行人区换回 null**：Cosmos test 与 P5 上，把 x⁺ 的行人区（mask 膨胀 24 px）换成 x⁻ 的像素再跑一遍，改后模型的 L_aux logit 差与翻转应回到 null 水平；若仍有 ≥ 一半的效应，说明学的是行人区以外的东西（第 56 条 G-d）。
- 域分类 AUC（每层，C / K / real）只描述。
- 打分器 sanity（描述）：训练集上原模型 plan 不在 Top 集的 slot 比例（sim x⁺、sim x⁻、nuScenes VRU 帧、nuScenes 正常帧各一个数）；全 actor 与可见 actor 的 Top 集分歧率。

### 4.5 判格与各结局的意思

主判格对 A 和 D 各判一次；「过」= 该 arm 的 3 个 seed 都过该条。

| 结局 | 条件 | 对中期故事的意思 |
|:--|:--|:--|
| **P（正例）** | R-nus、R-wod、S-p5、S-cos、B-p5、B-real、全部 N 都过 | 验收原则第一个正例（表征 + 开环行为，行为监督不含 expert 轨迹）；选中的模型进 D5–D8 闭环；midterm-gaps 2.3 结局 A |
| P-rep（看见但不动） | R-*、S-*、N 过；B-p5 或 B-real 不过 | 两边都学会「看见」，policy 不据此减速（G-c）；故事写成表征正例、行为待补；下一步是 policy LoRA（让 L_score 的梯度进 policy），不是更多数据 |
| P-size（分辨率） | R-* 与 N 过；S-p5 过但 S-p5-all 不过，且分档里 < 500 档不动 | 学会是真的，考卷上剩下的是分辨率；用分档表如实写，并看 D / Z 在 < 500 档是否不同 |
| real-only | R-* 过、N 过，S-* 与 B-p5 不过 | 与第一轮同形；看 A-noC：若 A-noC 也一样，sim 监督没被用上；验收原则仍无正例（结局 C） |
| sim-dominant | S-* 或 B-p5 过，但 R-* 或 B-real 不过，或 N 不过 | sim 压过 real 或学到 CARLA / Cosmos 特有方向（第 44 条的老问题，结局 D）；看捷径检查与 A-noK |
| none | 以上都不成立 | 中期只报第一轮 + E0 / E1 / M0 的诊断链 |

B-score-* 不改变上表的结局，只作并列：若 B-p5 / B-real 过而 B-score 不过，写「减速了但不一定是对的减速」；若 B-score 过而 B-p5 / B-real 不过，写「打分变好但不是靠对行人减速」，并看 Top 集里被选中的是哪类候选。

arm 之间的比较（描述，但用来归因，同一套 route / scene 聚类配对 bootstrap）：
- **D 对 A**：S-p5、S-p5-all、B-p5、< 500 档可读性的配对 Δ；CI 下界 > 0 → 「检测插件在纯视觉之上有增益」，写进故事作「trade（工业模型轻适配）+ 结构化感知插件」的一环（第 50 条的真实侧第一次读数）。
- **A 对 A-bhv（seed 0 对 seed 0）**：B-p5、B-real 的 null 帧 slow 率、B-score-*、N-ade。A 的 B-p5 不低于 A-bhv 且真实 null 帧更不保守 → 「规则打分不输给 expert 回归、且不带 expert 的保守」；反过来就如实写 expert 监督更强，并看差在哪类帧。
- **A-real / A-sim 对 A**：A 的 S-* 只在有 sim 数据时涨、R-* 只在有 real 数据时涨 → 「各域要各自的标签」（OpenVLA 的结论在驾驶上复现）；A-sim 在 R-* 或 B-real 上也涨 → sim 监督可以迁到真实，是更强的结论。A-real 的 B-p5 动了 → nuScenes 上的打分监督单独能带到 CARLA。
- **A-noC 对 A**：S-p5（原画考卷）掉而 S-cos 的 K 不掉 → 原画必须作为带标签的域；**A-noK 对 A**：R-* 不掉 → Cosmos 对真实侧没有贡献，与 E1 一致，以后 sim 数据可以不经 Cosmos（省 118 GPU·h 一轮）。
- **D-only**：检测 token 单独（stage 4 冻结）若已达 D 的大部分，说明增益来自检测器而不是视觉适配。

## 5. 训练前还要做的（写死，数字出来不改登记）

1. **打分器与验证（CPU，Cosmos 跑完之前做完）**：`jevdrive/op_adapt_score.py`（S_jev、候选生成调用 `wl_traj.py`、可行驶区栅格、可见 actor 集），单元测试（框重叠、DAC 在直道与路口、让行豁免、P 的 5 m 下限、cg 与 `_gap_front` 同值）。登记的验证，任一不过就停、回 main：
   - V1：WL-1 的 2 814 个分支 run，实际轨迹 + 记录 actor 上的 NC 与 WL cg 标签一致率 ≥ 95%；
   - V2：nuScenes val 上 log 人类轨迹作为一条「候选」，S_jev ≥ 0.8 的帧 ≥ 90%（地图对齐、框与 footprint 没算错）；
   - V3：P5 与 Cosmos 已完成对上，x⁻ 的 `op` 在 Top 集的 slot ≥ 70%（打分器在没有 hazard 时不该大面积否定原模型）；
   - V4：x⁺ 可见 ≥ 500 px 且在走廊内的 slot 里，Top 集至少含一条减速或横移候选的 ≥ 90%，且 `hold` 的 NC 失败率明显高于 x⁻（打分器看得到行人）。
2. **M1 基线（零训练，约 0.5 GPU·h）**：原模型在全部读数上的数，包括 P5 ≥ 500 px_eq 子集的 D0、按大小分档、原生 plan 在 P5 上的翻转与 null false-flip、B-real 的原模型减速率、B-score 的 O 列与 B-ref 参照行、NAVSIM / RFS。不依赖 Cosmos 的部分先跑；S-cos、B-cos 的 O 列等 Cosmos DONE 后在同一份冻结代码上补。全部在任何训练之前提交。
3. **R0**（第 2.3 节），用当时已完成的 Cosmos 对与 P5；零训练，不受 Q5 限制。
4. **标签与目标**：Cosmos / CARLA 对的走廊标签、可见 actor 集、触发 tick、候选与 S_jev（每对 x⁺ / x⁻ × C / K）；nuScenes train / val 的候选与 S_jev；WOD val 的走廊行人标签（`labels_wod` 同一规则）；A-bhv 用的 expert Δv*（第 1 遍 `pose.jsonl`）。CPU。
5. **缓存**：sim 对的 trunk 缓存（C 与 K，约 2 000 × 24 slot × 4 ≈ 19 万张，按第一轮的 16.6 万张 8 min 推约 10–15 min GPU）；YOLO 检测与 token（sim 对、nuScenes train / val、WOD train 缺的偶数帧、P5 前视、navtrain / navtest 若已有则复用）；teacher 输出，含 `op_L` / `op_R` 的 desire 条件前向（sim 与 nuScenes）。

## 6. 分级启动与 sanity checklist

- **第 1 个单位**：A seed 0（λ_s = 1）跑 2 000 步。检查：loss 全部有限且下降；dev 漂移中位 ≤ 0.10 m；dev 上 L_aux 的 AUC 在 sim 与 real 都高于 O；显存与吞吐在第 7 节估计的 ±30% 内；缓存 → stage 4 → policy 与 port 全前向在 3 对上 `temporal` 相关 ≥ 0.9999（第一轮做过的同一检查）；第 0 步 L_score 在「原模型在 Top 集」的 slot 上恰为 0。
- **约 10 个单位**：全部 arm（A 的两个 λ_s、D、A-real、A-sim、A-noC、A-noK、D-only、A-bhv）各跑 10% 的步数，逐条对照：
  1. 完成率 100%，没有 NaN / OOM 重启；
  2. dev 漂移中位 ≤ 0.15 m（10% 步数时，比终线宽）；
  3. dev 上 sim 的 L_pair 配对准确率 > 0.6、real 的 L_aux AUC ≥ O；
  4. D 与 D-only 的门 |α| 离开 0（插件真的在用）；A-real 的门不存在（结构检查）；
  5. dev x⁺ slot 中原模型不在 Top 集的那些，改后 plan 的 S_jev 高于原模型的比例 > 0.5，且 g_dir slot 上 v̂⁺ > v̂⁻ 的违例率低于 O（L_score 与 L_dir 在学对方向）；A-bhv 这一条换成 dev 配对 Δv̂ 与 Δv* 的同号率 > 0.5；
  6. 输出非退化：改后原生 plan 在 dev 正常帧上的速度分布与 O 的 KS 统计量 < 0.1；dev x⁻ 上 `op_stop` / `brake_*` 类候选成为最近 Top 候选的比例不高于 O 的 + 5 pp（没有学成一律减速）。
  任一条不过就停批，写明哪一条、哪个 arm。
- **全量**：按第 2 节的 seed 数跑完，然后第 4 节的读数一次性跑完（读数代码在全量开始前冻结并提交）。

## 7. 成本与排期（第一轮实测吞吐换算）

第一轮实测：B（stage 4 解冻、trunk 缓存、fp16）384 条序列 / s，即 0.07 GPU·h / 10 万条；共卡测，偏保守。

| 项 | 量 | GPU·h |
|:--|:--|--:|
| 缓存：sim 对 trunk、YOLO 检测与 token、teacher 输出（含 desire 候选前向 +0.3） | 约 20 万 sim 张 + 约 60 万真实 slot 的检测 | 2.8 |
| M1 基线 + R0 | | 0.8 |
| 训练，每个 arm 120 万条序列：0.84 GPU·h 纯前后向 + x⁺ / x⁻ 双成员 policy 前向约 +20%（L_dir，与 v1 的 L_Δplan 同量）+ 检测 adapter 约 +10%；L_score 只是对缓存候选的距离，可忽略 | ≈ 1.1 / arm | — |
| 训练：A × 4（3 seed + 1 个 λ_s 备选）、D × 3、A-real、A-sim、A-noC、A-noK、D-only、A-bhv | 13 次 | 14.3 |
| 读数：每个模型约 0.8（第一轮的 a / b / c 约 0.3，加 P5 原生 plan 流、Cosmos test、WOD val、NAVSIM navtest、RFS；S_jev 打分在 CPU 上） | 13 个模型 | 10.4 |
| 分级启动的开销（1 + 10 单位） | | 1.5 |
| **合计（不含 Z）** | | **约 30**（v1 约 28；+2 来自 A-bhv 与 desire 候选） |
| Z（R0 过线才开）：tele trunk 缓存 1.0 + 训练 1.3 + 读数 0.8；过线再补 2 seed +4.2 | | 3.1（+4.2） |

CPU：候选生成与 S_jev 打分是向量化 NumPy，sim 约 2 000 对 × 15 slot × 2 世界 × 2 画面 × 13 候选、nuScenes 约 3.4 万帧 × 13、P5 读数帧 × 13，合计千万级「候选 × 时刻」的框检查，按 `n_cpus()` 分片约 < 1 h；可行驶区栅格（Town12 路线切片 + P5 各 Town）< 0.5 h。
打分器实现与 V1–V4 约半天人时，放在 Cosmos 跑的这段时间里。

一张卡一天 24 GPU·h：不含 Z 约 1.25 个卡日。排期（卡 A），以 Cosmos 全量 DONE 为零点（Q5）。Cosmos lane 13:39 重开，按 stage 10 实测的 82 对 / h 外推约 09-30 下午到晚上完成，以 lane 的 DONE 文件为准：

| 时段 | 做什么 |
|:--|:--|
| 现在 – Cosmos DONE | 卡给别的活。本轮只做：打分器与 V1–V4（CPU）、可行驶区栅格、real 侧候选与 S_jev、teacher 的 desire 前向与 YOLO（零训练、< 1 GPU·h 一项，找空卡跑）、M1 不依赖 Cosmos 的部分、R0；读数代码冻结提交 |
| DONE + 约 3 h | sim-K 与 sim-C 的 trunk 缓存、sim 候选与 S_jev、M1 的 Cosmos 部分；1 单位 → 10 单位，对 checklist |
| DONE + 约 1 天 | A、D 各 3 seed、四个对照、D-only、A-bhv；（Z 若开） |
| DONE + 约 2 天 | 读数、判格、写结果；选中的模型交给 D5–D8 闭环 lane |
| 到 10-06 | 余量：返工或 Z 的补 seed；不在这段开新 arm |

若 DONE 在 09-30 晚上，结果约 10-02 晚上出，仍在 10-06 前、留约 3 天余量；DONE 每晚一天，D5–D8 闭环拿到模型也晚一天。卡 B 本轮不用。

## 8. 用户决定与剩下的问题

v1 第 8 节的六个问题，用户 2026-09-29 的决定已并入正文：Q1 → 第 4.1、4.2 节；Q2 → 第 2.1 节（规则打分、A-bhv 对照）；Q3 → 第 2.1 节「真实数据上哪里能用打分监督」与第 3.1、3.3 节；Q4 → 第 2.3 节；Q5 → 第 3.1 节起训规则与第 7 节排期；Q6 → 第 4.2 节主判格。

v2 的两个遗留点，用户批准时按 main 的建议定下，已写进第 2.1 节：(a) L_dir 在「可见但 < 500 px、靠近路径」一档不强制 plan 相同；(b) 让行豁免只给移动的 actor（行人、骑车人、移动车辆），静止障碍挡路且有安全绕行候选时干等要扣进度分。没有剩下的问题。

## 执行日志

- 2026-09-29 13:3x（box 时间）M0：`scripts/op_adapt_r2_pedsize.py`，CPU，数秒；此时 Cosmos 全量有 93 对（lane 状态：13:33 drained，另一个会话随后在 13:39 重开，仍在跑；本 todo 没有碰这个 lane）。
- 2026-09-29 v2：按用户对 v1 第 8 节的决定改写（行为监督改为规则打分 S_jev + L_score / L_dir，navtrain 只蒸馏，B-real 进主判格，起训等 Cosmos DONE，加 A-bhv 对照与 B-score 副读数）；仍是草案，没有任何第二轮数字。
- 2026-09-29 v2 批准：用户批准修订版，遗留点 (a)(b) 按上面写定；状态改为已批准，执行等 Cosmos DONE 与 main 的 go。
