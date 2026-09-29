# E1：2 × 2 析因 + 空间分辨 probe（Cosmos 能恢复多少 CARLA 行人可读性）

状态: running（登记写于 2026-09-29 13:08 CST，任何数字之前）
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

（待填）

## 结果

（待填）
