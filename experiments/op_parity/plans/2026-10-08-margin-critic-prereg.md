# op_parity margin-critic 预登记：冻结表征上用地图 margin 直接监督的头，能把任意候选轨迹的可行驶区域 margin 读到多准，代入第 187 条的规则 / 头能回收多少（2026-10-08，任何本线新读数之前写定）

## 问题

第 187 条：navtest > 20° 的 3 154 个 token × 2 个 SH30 seed 上，给每条候选的**地图** SDF 足迹 margin（特权），一参数规则 P3 +3.21 [+2.08, +4.39]、ridge P1 +3.77、树 P2 +4.15（F19 × pc，上限 +9.55）；
换成 SH30 自己的路沿输出逐候选求 margin，规则是 −4.45，「有修法」AUC 0.546（地图 0.706），恒等 DAC 失败 AUC 0.629（地图 0.916）。187 没测的是：**专门为这个量训练一个头时，冻结表征里有没有足够准的边界**。
本线量这条能力边界：在 navtrain 全量（约 10 万 token，不只转弯）上以地图几何为**训练标签**，训练一个只吃 SH30 测试时可见输入的头，预测任意候选轨迹的 margin；在 navtest > 20° 上读 margin 质量与代入 187 的 P3 / P1 / P2 后的增益。
地图只作训练标签；测试时非 PRIV 的臂不读任何 HD-map 几何。navtest > 20° 的 token 不参与 margin 头的拟合、选 epoch、选超参。

## 已有的、不重测（引用）

- 第 187 条：P1 / P2 / P3、N1–N7、E 臂的全部读数与学习曲线；G1 地图 margin 对恒等 DAC 失败 AUC 0.916、G2 自路沿 0.629；「有修法」AUC 地图 0.706、自路沿 0.546。本线直接读 `turn_selinput/select2.pkl` 与 `margins.npz`，只重跑 P1 / P3 作逐位复现闸门。
- 第 186 条：上限 F19 × pc +9.55、L9 × epcap +9.20，E 臂 −0.27。第 179 条：自路沿 margin 校准残差 1.8–2.1 m，AUC 0.641 对地图 0.921。
- 第 165 条：同容量 probe 从冻结 Cinque 特征回归边界带 SDF，navtest > 45° MAE 1.21 m，WA-Cf 0.86 m，ego-only floor 2.0–2.1 m（navtrain s2–s6 训练、池化特征、ridge / MLP）。它回答了「边界带栅格读得多准」，没有回答「逐候选足迹 margin」和「能否用来选」，也没有用全量 navtrain 与未池化 token；本线不重测它的栅格读数，只把它当先验：预期 margin 误差在 1 m 量级。
- 第 166 条：真值边界对切内角无用但对 > 20° DAC 有用；第 190 条：fold 模型与 held-out plan（本线的训练输入来源）。
- 并行线 turn-selnt（`2026-10-08-turn-selector-navtrain-prereg.md`）用 simulator 分数监督 selector；本线的监督是地图几何，不读 navtrain 的 simulator 分数，不重复它的臂。

**已看过、不算盲的数**：上述全部。**没看过的**：任何以地图 margin 为目标的头在 navtrain dev / navtest 上的读数。

## 标签

候选轨迹 (8, 3) 后轴位姿 → `sc_analyze.footprint`（0.1 s 网格 41 步 × 4 角点）→ 地图 SDF 栅格（`op_probe/labels/navtrain_all.npz` / `navtest.npz`，128 × 96，0.5 m，双线性、边界夹取）→ 每步取 4 角点最小 → 沿时间累积最小 → 取 1 / 2 / 3 / 4 s → 截 [−2, 4] m。
与 187 的 `margins.npz` 同一定义（187 的限定照搬：这是 plan 足迹对 SDF 栅格，不是 devkit 打分的 LQR 跟踪轨迹）。训练时标签在 GPU 上由栅格现采（`grid_sample`，align_corners=False，border），逐点 SDF 截 [−3, 6] m。

## 头（MC，一个结构，不调）

隐式 SDF 场 + 足迹查询，margin 由预测的逐点 SDF 按与标签**完全相同**的算子（角点最小、时间累积最小）得出，所以对任意候选都是良定的，且与地图 margin 同单位可直接代入：
- 场景记忆：冻结视觉 token（`front.npy`，SH30 所见的全部 8 帧 × 32 × 512）→ LayerNorm → Linear 512→256 + (帧, 槽) 位置嵌入；SH30 隐状态 `select_4`、`mean`（各 512，训练集上逐维标准化）各一个 token；ego（20 维，含指令）一个 token；2 层 pre-norm Transformer encoder（d 256，4 头，FF 512，dropout 0.1）。
- 查询：点 (x, y)（后轴系，米）→ 8 频 Fourier → MLP → 2 层 cross-attention（只看记忆，查询之间不互看，所以是一个场）→ MLP → 标量 SDF。
- 损失：逐点 Huber(δ 0.5) + 4 个 horizon 的 margin Huber(δ 0.5)，等权。AdamW lr 3e-4、wd 0.05、warmup 300、cosine、bf16 autocast，batch 256 token × 每 token 6 条轨迹；步数 8 000（训练 log ≥ 30%）/ 4 000（< 30%）/ 6 000（参考臂对）。
- 选 epoch：每 500 步在 navtrain dev（`navsim/op-parity-full-dev`，1 789 token，SH30 与全部 fold 模型都没训练过的 log）上算 4 s margin MAE，取最好的检查点。只在 navtrain 上选。

**训练轨迹的采样**（每 token 一个 48 条的库，每步随机取 6 条）：held-out plan 周围的 F33 全族（`turn_ceiling.transform`，c00–c32，原样）+ 8 条连续随机变换（偏移 U[−1.5, 1.5] m、曲率增益 logU[0.6, 1.6]、速度 logU[0.5, 1.5]，F33 外点范围的略宽超集）围绕 held-out plan + 7 条同分布随机变换围绕 logged future（没有 future 的 token 用 plan 代替）。
logged future 只作训练轨迹来源（输入是轨迹、标签是地图几何），不是测试时输入。

**训练输入来自没见过该 log 的模型**：token 的隐状态与 plan 取 `CF5f{j}-F-s0`，j = sha256("cf5|" + log) % 5（第 190 条的折；plan 取 `bench/ol/<shard>/preds/` 的导出）；视觉 token 来自冻结的 shipped 编码器，与折无关。训练集 = `navsim/op-parity-full-train` 中 SDF `ok` 的 token（约 10.1 万）。
**测试输入**（navtest > 20°，3 154 token × SH30-F-s0 / s1）：视觉 token、ego、该 seed 的 SH30 隐状态（`turn_selinput/hidden/`）、`turn_ceiling/poses.npz` 的 33 条候选。输出 Q（2 × 33 × 3 154 × 4），与 `margins.npz` 的 M 同形。

## 臂

| 臂 | 输入 | 训练 token | 用途 |
|:--|:--|:--|:--|
| **MC（主）** | 视觉 8 帧 + 隐状态 + ego | 100% | 判定 |
| MC-30 / MC-10 / MC-3 | 同上 | 按 log 的嵌套子集 30% / 10% / 3%（按 sha256("mc|" + log) 排序取前段；3% ≈ navtest 转弯标签的规模） | 读数 3，学习曲线 |
| MC-V | 视觉 8 帧 + ego（无隐状态） | 100% | 输入对照；G-hidden 不过时替补主臂 |
| MC-E | 只有 ego（+ 轨迹） | 100% | 盲 floor：不看场景时由先验能读到多少 |
| **参考臂 R-WA** | WA-Cf token（`op_probe/feats/WA` 的 `Cf`，32 × 512，NAVSIM 上训练过的编码器）+ ego | navtrain s2–s4（约 2.5 万） | 读数 4：边界信息是否丢在冻结编码器 |
| R-C1（R-WA 的配对基线） | Cinque 最新一帧 32 × 512 + ego | 同一批 token | 与 R-WA 同头、同 token、同步数 |

参考臂选 WA-Cf 的理由：视觉 token 冻结且与折无关，「fold 特征对 shipped 特征」只差隐状态，区分度低；解冻编码器末块要走图像管线，预算内做不了；WA-Cf 是现成的、同形状（32 × 512）的 NAVSIM 监督编码器特征（第 165 条读栅格 0.86 对 1.21 m），同一个头换编码器最直接地回答信息丢在哪。
限定事先写明：WA 在 navtrain 上训练过，它的 navtrain 特征是 in-sample（诊断上界，不是方法）；只有单帧、2.5 万 token。每个臂 1 个训练 seed。

## 读数（navtest > 20°，全部按 log 聚类 bootstrap，`jevdrive.stats`，B = 10 000）

1. **margin 质量**：Q 对 M 的 MAE 与 Pearson r（4 s 与 1 / 2 / 3 s；恒等候选与 F19 全部候选分开）；token 内候选排序（F19 的 19 条候选上 Q 与 M 的 Spearman，逐 token-seed 后取均值；只在 M 的候选间极差 ≥ 0.2 m 的 token-seed 上算）；
   恒等 plan 的 DAC 失败 AUC（−Q 4 s；参考：地图 0.916、自路沿 0.629）；「有修法」AUC（−Q 恒等 4 s；参考：地图 0.706、自路沿 0.546）。AUC 的 CI：log 聚类 bootstrap B = 1 000（同 187）。
2. **选择**：把 Q 当作 187 里 M 的替身，臂 Q3（一参数规则，τ 网格与 P3 相同）、Q1（ridge，E + 全部候选 Q）、Q2（树，行 = 候选），**头按 187 原样在 navtest 内按 log 5 折 cross-fit 重拟合**（`turn_selinput.unit2` 同一代码路径、同折、同重复数 10 / 5 / 10），族 F19 × pc（判定）与 L9 × epcap（一致性）。
   选「重拟合」而不是「在 navtrain 上拟合」的理由：(i) 与 P1–P3 逐折配对，差只来自 margin 的来源；(ii) 在 navtrain 上拟合选择头需要 navtrain 的 simulator 分数，那是 turn-selnt 线的问题，本线的监督只有地图；(iii) 预测 margin 的标定与地图不同，τ / ridge 需要重估。代价：选择头见过 navtest 其余折的 simulator 分数（与 187 的 P / N 臂同等待遇）；margin 头本身从不见 navtest。
   报：增益与 95% / Bonferroni（m = 3，98.33%）CI；回收率对上限 +9.55（`TC.ratio_ci`）；对同头特权臂的回收率 Q_h / P_h（比值的 log 聚类 bootstrap）与配对差 Q_h − P_h；移动比例。
3. **学习曲线**：MC-3 / 10 / 30 / 100 的读数 1 全部项 + Q1 / Q2 / Q3（F19 × pc）。
4. **参考臂**：R-WA 与 R-C1 的读数 1 + Q1 / Q2 / Q3，及配对差 R-WA − R-C1。

## 判定（固定，看到读数后不改）

判定只用主臂 MC（G-hidden 不过时用 MC-V）在 F19 × pc、> 20° 的读数；h ∈ {1, 2, 3} 三个头，Bonferroni m = 3。
- **(a) 冻结表征带着边界**：存在 h，Q_h 的 Bonferroni CI 下界 > 0，**且** Q_h / P_h 的点估计 ≥ 0.5（P_h 取 187 的同头读数：+3.77 / +4.15 / +3.21），**且** 该头在 L9 × epcap 上点估计 > 0。
- **(c) 不在**：三个 Q_h 的 95% CI 都含 0 或整体 < 0，**且** 恒等 DAC 失败 AUC < 0.68（自路沿参考 0.629 + 0.05）。
- **(b) 部分**：其余。分两种写明：(b1) 某个 Q_h 的 Bonferroni 下界 > 0 但对特权臂的回收 < 50%；(b2) 没有显著的选择增益，但 DAC AUC ≥ 0.68（读得出一部分边界，精度不足以逐候选选择）。
- **学习曲线的读法**：30% → 100% 时 DAC AUC 升 ≥ 0.02 或 F19 候选 4 s margin MAE 相对降 ≥ 5% 判「仍受标签数限制」，否则「已饱和」（在这个头下）。(c) 或 (b2) 且饱和 → 「冻结特征缺边界」；(c) 或 (b2) 且仍在升 → 「标签数不够，未定」。
- **floor 的读法**（不进判定）：MC 的 DAC AUC 不高于 MC-E 0.03 以上，则读到的不是场景而是先验（ego / 轨迹形状），写明。
- **参考臂的读法**（不进判定）：R-WA − R-C1 的 DAC AUC 差 ≥ 0.05 或 Q2 增益差的 95% CI 下界 > 0 → 「信息丢在冻结编码器」；否则「同规模下换 NAVSIM 监督的编码器也读不出，问题不在（或不只在）编码器」。

## 闸门（任何一项不过就停下报告）

- **G-label**：本线算子（torch 足迹 + `grid_sample`）对 navtest 33 候选 × 2 seed × 3 154 token 算出的地图 margin 与 187 的 `margins.npz` 的 M 最大绝对差 ≤ 0.02 m（冒烟阶段在全部 3 154 个 token 上做；这一步只读地图，不读分数）。torch 足迹对 `sc_analyze.footprint` 最大差 ≤ 1e-3 m。
- **G-leak**：抽取用的模型折 = sha256 折，且该 log 不在 `navsim/op-parity-cf5f{j}-train`；抽取得到的 plan_pos 与同一 fold 模型存档的 plan 在速度 ≥ 0.5 m/s 的行上前 15 点最大差 < 0.03 m（turn-selnt 线同一判据）；navtrain 与 navtest 不相交；训练集与 dev 不相交。
- **G-hidden**：5 个 fold 模型与 SH30-F-s0 在 navtest 转弯 token 上隐状态逐维 Pearson 相关的中位数 ≥ 0.8。不过 → 主臂换 MC-V，MC 标「隐状态空间不可比」。
- **G-P**：经本线驱动重跑的 P3 与 P1（F19 × pc，全部重复）与 `select2.pkl` 逐位相同。
- **冒烟**：shard 0 的约 600 个 token：抽取 → 轨迹库 → 300 步训练（损失下降；dev MAE 低于常数预测）→ navtest 预测 → G-label → 选择阶段 rep 0。按冒烟实测的 it/s 与抽取耗时外推全量；超过预算（约 5 h 墙钟、6 card-h、80 core-h）1.5 倍就停下报告。

## 运行

`scripts/margin_critic.py`（extract / bank / train / select / report）+ `scripts/margin_critic_chain.sh`，tmux `jev:margin-critic`，STATUS / DONE / ERROR 在 `$DATA_DIR/runs/op_parity/margin_critic/chain/`。全部 GPU 作业经 pool（抽取按 shard `cl fanout`，每个臂一个训练作业），在 `jevdrive.run.Run` 内，split 取自 `jevdrive.data.splits`，缓存 / 并行 / 统计用 `jevdrive.cache` / `par` / `stats`。
只读：sh30_crossfit 的检查点 / 标签、turn_selinput 与 turn_ceiling 的输出、op_probe 的标签与特征、pp_prep 缓存。写：只在 `runs/op_parity/margin_critic/` 下。

## 不做

不新打 simulator 分；不读 navtrain 的 simulator 分数；不解冻编码器；不调头的结构 / 超参（一个配置）；不在 navtest 上选任何东西；不做闭环、reactive、直行对照；车道线输出不做；每臂一个 seed，不做第二个参考臂。

## 修订 1（2026-10-08，全量抽取阶段，任何全量训练 / navtest 读数之前）

**发生了什么**：全量链在抽取阶段停下。shard 1（8 608 行）与 shard 6 的 G-leak 判据「前 15 点最大差 < 0.03 m」不过：shard 1 最大差 0.03125 m（= 2^-5，fp16 在 32 m 附近的 4 个 ulp），比容差多一个量化台阶；其余已完成的 shard 通过。
冒烟（600 行）最大差 0.0156 m。此时没有任何全量训练作业启动，没有任何全量的 margin / 选择读数；已看过的只有冒烟的读数（482 个训练 token、300 步，不是判定对象）。

**为什么改而不是停**：原判据是从 turn-selnt 线照搬的「最大值」判据，那条线只抽 2.8 万个转弯 token；本线抽全部 10.3 万行，最大值判据对行数敏感（一行多一个 fp16 台阶就不过），而闸门要排除的是「用错了折的模型」，错模型的差是另一个量级（turn-selnt 冒烟实测：错折模型的中位差 0.07 m，92% 的行超容差）。
**修订后的 G-leak（b）**（逐 shard）：速度 ≥ 0.5 m/s 的行里，差 ≥ 0.03 m 的行占比 ≤ 0.1%，**且** 最大差 ≤ 0.0625 m（低于错折模型的中位差），**且** 本线自己的反例对照（每折取 100 行用下一折的模型重跑）中 ≥ 50% 的行超过 0.03 m（证明检验有分辨力）。折哈希、log 不在训练 split、navtrain / navtest 不相交等其余各项不变。
逐 shard 的最大差、超容差行数、对照占比写进结果文档。这是对预登记的偏离，在结果与决定条目里写明；判定阈值、头、标签、臂、读数全部不变。

另一处与正文不符、同样在读数前记下：冒烟用 shard 2（不是 shard 0）的前 600 行，因为参考臂的 WA-Cf 特征只覆盖 shard 2–4；R-WA 在 shard 0 上的第一次冒烟因此报错（没有读数）。
