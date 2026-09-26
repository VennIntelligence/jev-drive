# 配对差分走到真实数据（第二轮）：G0 student 零样本、G1 门控、G3 编辑对诊断、G2 HUGSIM 车辆配对重训；附第 40 条 3 seed 重判

状态: 待排（预登记，2026-09-26 上午，写于任何新拟合之前；用户 2026-09-26 决定顺序 G0 → G1 → G3 诊断 → G2，第 40 条重判随时可做）
上游: 第 44 条（E1 零样本有害、E2 编辑对不过、E3 关闭）、第 42 条（M-C 3 seed 过；E4c 人类 onset 0.9–1.9 s；E5 student 52.7%、p95 30 ms）、
第 45 条（YOLO26x-seg 行人召回不劣于 SAM、延迟 1/25）、[I3](2026-09-25-reactivity-program/i3-hugsim-pairs.md)（65 个 3DGS 车辆场景，openpilot 读出零样本 70%、M-C Δ −11 pp）、
[激发计划](2026-09-26-elicitation-program.md)、[夜间汇总](../tmp/2026-09-26-overnight.md)
主线: 组件冻结、配对差分激发不变。本轮只回答一个问题：**CARLA 里激发出来的反应通道，怎样在真实数据上不有害、并在有行人的帧上为正**。全部开环。

## 夜里量到的失败形状（本轮的起点）

| 事实 | 数字 | 含义 |
|:--|:--|:--|
| CARLA 训的 Δ 在真实特征上到处开火 | E1 直行帧激活率 Cinque 16.7%、Lebowski 40.7%（门槛 7%）；I3 null 对上 Δ 也动、τ 被抬高 | 缺的是 gate，不是方向 |
| openpilot 读出跨真实外观成立 | I3（3DGS 车辆配对）`ridge_late` 零样本 70.0% [65.5, 74.5]，null 4.4% | 驾驭训练过的 backbone 自己就能迁移 |
| Qwen 是唯一跨域失效的那一路 | I3 上 Qwen `ridge_late` 2.8%；E1 的害主要来自 Δ 头 | student 不用 Qwen（E5：YOLO embedding + openpilot） |
| 编辑对的信号量 | 抹人在 Qwen `L18_last` 上只比安慰剂多移 21%；openpilot 2.4 倍 | pooled 全图特征对小目标不敏感，未必是编辑质量的问题 |
| 「什么时候该反应」真实数据教得会 | P2(e) 的 gate 沿 s_ego 单调上升 | gate 可以来自 WOD 自己 |
| HUGSIM 没有行人资产 | 只有 3DRealCar 110 辆车 | G2 只能做车辆版 |

## 五项

### G0. E5 的 20 Hz student 零样本上真实数据（最先做）

- **问题**：不含 Qwen 的行人通道（openpilot `temporal` ⊕ YOLO26x-seg 检测 embedding → MLP，P5 v1 BA 配对差分训出，3 seed 行人 51–57%）放到真实特征上是有害、无害、还是有用。
- **为什么先做它**：它的两路输入都不吃 CARLA 外观。检测框在任何渲染器里都是同一个物体；openpilot 特征在 I3 真实外观上零样本 70%。E1 的害来自 Qwen pooled 的 Δ，student 里没有这一路。
- **做法**：student A 与 B（E5 登记的两个 arm，3 seed 各自套，不重训、不调）。YOLO26x-seg 640 fp16 在 WOD val `p2p3_v1` 子集 19 663 帧前视（与 E1 同一批帧，可配对）和 navtest 全部 token 前视上跑一遍，
  走廊内前 8 个检测的 embedding 与 E5 完全同一段代码；prior 分别是 WOD train 训的 `ridge_late` openpilot 和 navtrain 训的薄 head（与 E1 相同）。
- **读数**：与 E1 逐列相同：WOD 全部 rater 帧与 Pedestrians / Cyclists / Cut_ins cluster 的 RFS 配对 Δ、ADE 第 1–9 档 Δ、直行帧与行人帧的激活率；NAVSIM 全部 token 与「走廊内有行人 / cyclist」的 897 个 token 的 PDMS / EPDMS 配对 Δ（官方 devkit，版本注明）。
- **判据（写死）**：
  - **有用**：Pedestrians（WOD）或有行人 token（NAVSIM）的主指标 Δ CI 整体 > 0，直行激活率 ≤ 7%，全部帧 Δ CI 不整体 < 0。
  - **无害但没用**：所有 Δ 跨零、直行激活率 ≤ 7%。
  - **有害**：全部帧 Δ CI 整体 < 0 或直行激活率 > 7%。
- **预期**：至少「无害」；行人上是否为正取决于 CARLA 与真实里行人相对 ego 的分布差。若「有害」，说明 MLP 在 embedding 上也学了 CARLA 的分布，G1 的 gate 是必需。
- **资源**：YOLO 三路 fp16 约 20 ms / 帧，WOD 19.6k + navtest 12.1k × 3 路约 30–40 min 单卡；head 与打分 CPU，devkit 约 20 min。工程半天。

### G1. 门控：gate 来自真实数据，Δ 来自 CARLA 配对（第一优先）

- **问题**：给 Δ 乘一个在真实数据上训的 gate g(x) ∈ [0, 1]，E1 与 I3 的害是否消失，行人帧上是否为正。
- **gate 的三个候选（各一 arm，同一批帧）**：
  - g₁ **P2(e) 的 gate**：WOD train 上按第 23 条 (e) 的配方（`waymo_ladder.gated_arm`，L1 稀疏，输入 ego + openpilot `temporal`）重训，只取 gate 分支；NAVSIM 上在 navtrain 同样训一个。
  - g₂ **hazard 存在性 probe**：YOLO 检测 embedding 上的 logistic probe，标签 = 走廊内 ≤ 30 m 有行人 / cyclist / 接近车辆（WOD 用 Q2b 的 SAM 检测当标签，navtrain 用 GT 轨迹），按 sequence / log 分折；输出经 Platt 校准。
  - g₃ **openpilot 自带信号**：lead_prob 与 TTC（Q4c 的 lead 头，20 m 内召回 0.9）经一个单调映射，无训练。
- **Δ 的两个来源**：M-C 双流（Qwen ⊕ openpilot，E1 那五个 fold head）与 G0 的 student；共 3 × 2 = 6 个 arm，加「无 gate」（E1 / G0 本身）作对照。
- **在 I3 上先过一遍**：同一 gate（g₂ 用 I3 的 GT actor 当标签、g₃ 直接用）乘到 M-C 的 Δ 上，M-C 对 `ridge_late` 的 −11 pp 应变为 ≥ 0，null false-flip ≤ 7%。这是 gate 有没有作用的最便宜检查。
- **读数**：与 E1 / G0 逐列相同，另加 gate 自己的描述：直行帧 / 行人帧上 g 的均值与 g > 0.5 的比例。
- **判据（写死）**：**成立** = 直行激活率 ≤ 7% 且全部帧 Δ CI 不整体 < 0（害消失），且 Pedestrians / 有行人 token 的主指标 Δ CI 不整体 < 0；**有用** = 上面成立且该子集 Δ CI 整体 > 0。三个 gate 里由训练行上的 AUC 选主 arm，不看评测数。
- **预期**：g₂、g₃ 能让害消失；能否为正取决于 Δ 的方向在真实特征上有没有意义，G0 的 student 比 M-C 更可能为正。
- **资源**：全部 CPU（特征、检测、预测都在盘上），gate 训练分钟级；工程一天（三个 gate 的接口）。

### G3. 编辑对（E2）的两项诊断，先于任何视频 inpainting

- **G3a 位移比值的 CARLA 参照**：在 P5 v1 BA 的 x⁺ / x⁻ 对上算与 E2 完全相同的统计：Qwen `L18_last`、openpilot `temporal`、YOLO embedding 三种特征上 |f(x⁺) − f(x⁻)| 的中位数，除以 null（天气）对上的同一量。
  与 E2 的 1.44（Qwen）、2.4（openpilot）并排。**读法写死**：CARLA 上 Qwen 的比值也 < 2 而激发仍成立 → 编辑质量不是 E2 的瓶颈，瓶颈在标签与对数，视频 inpainting 不投；CARLA 上比值 ≥ 3 → 编辑信号确实弱，再登记视频一致 inpainting。CPU 分钟级。
- **G3b 标签 (c)**：E2 已造的 navtrain 911 对上补 PDM scorer 标签（「继续」与「刹停」两条 proposal 各打一次，分差符号当 Δ 方向；E3 里量过它与人类未来的一致率逐对 79%），用它重训 M-C 与 student，
  评测同 E2（P5 BA 行人翻转、WOD Pedestrians RFS Δ、直行激活）。判据同 E2。CPU 加 devkit 打分小时级。
- **G3c（描述）**：WOD 编辑对从 216 扩到 train 全量走廊内行人帧（E2 估 4–12k 张）的 SAM + LaMa 成本表，只估不跑；等 G3a 判定后决定。

### G2. HUGSIM 车辆配对上重训 M-C（等 G0、G1 出来后排）

- **问题**：在真实外观的车辆配对（I3，65 场景，1 832 reactive 帧，static / cutin / oncoming）上用配对差分重训 M-C 与 student，Δ 是否不再有害，WOD Cut_ins 上是否为正。
- **限定先写死**：HUGSIM 只有 3DRealCar 车辆资产，**没有行人**，本项测不到行人缺口；行人 3DGS 资产（从带行人的重建场景导出动态高斯，或 SAM 3D Body 一类的人体重建）是另一个调研项，不在本轮，查到可用资产再单独登记。
- **做法**：训练行 = I3 的 x⁺ / x⁻ / null 帧（按场景 5 折），特征 openpilot `temporal` ⊕ Qwen `L18_last`（已抽）与 YOLO embedding（需在 I3 帧上跑一遍，分钟级）；对照：CARLA 训的同一 head、hard-example 重加权、均匀 imitation。
- **读数**：I3 held-out 场景翻转率与 null；WOD Cut_ins / 全部 rater 帧 RFS Δ、直行激活率；NAVSIM 有接近车辆的 token 的 PDMS Δ。
- **判据**：I3 held-out 翻转 ≥ CARLA 训的 M-C（58.7%）且 null ≤ 7%；WOD Cut_ins RFS Δ CI 不整体 < 0、直行激活 ≤ 7%。
- **资源**：YOLO 在 I3 帧上分钟级 GPU；其余 CPU；工程半天。

### R40. 第 40 条「Lebowski `cls_late` 够到原生」按原判据 3 seed 重判

- **做法**：第 40 条 (iii) 的预登记配对比较（`cls_late` − 原生，frame mean，sequence bootstrap；缺口比例 G）在 seed 0 / 1 / 2 各算一次，再对三个 seed 的预测取平均算一次；Cinque 与 Lebowski 都算。零拟合，预测在 `research/results/elicitation/seeds/` 与 heads_train run dir。
- **判据**：原判据不变（`cls_late` − 原生的 CI 跨零 = 够到）。**报法写死**：三个 seed 中有任一不够到，本格写「是否够到随词表 seed 变，均值 G = …」，第 40 条与中期 artifact 里「收回原生差距的 0.4–0.7」一行改成 3 seed 的 G 范围。
- **预期**：Lebowski 只 seed 0 够到（均值 7.59 对 7.89），Cinque 三个都不够；两模型 G 约 0.3–0.5。
- **资源**：CPU 分钟级。

## 顺序与依赖

```
R40（CPU，随时）
G0（YOLO 30–40 min GPU + CPU）──有用 / 无害──► G1 只做「有 gate 是否更好」
                                └──有害───────► G1 必需
G1（CPU；先在 I3 上验 gate）
G3a（CPU 分钟级）──比值 < 2──► 不投视频 inpainting，做 G3b
                └──比值 ≥ 3──► 另登记视频一致 inpainting
G2 等 G0、G1 出来后排；行人资产另行调研
```

## 硬约束

- 组件冻结；Δ 头不在真实数据上重训（G2 除外，它在 3DGS 配对上训）；gate 只在真实数据上训，且不看评测数选。
- judge 不改：WOD 第 22 条口径、NAVSIM 官方 devkit 注明版本、P5 / I3 的定向翻转率 + null false-flip；所有 head 报 3 seed。
- 训练 / 评估按 sequence、log、场景分开；G0 / G1 的 WOD 评测帧与 E1 同一批（19 663 帧），保证可配对。
- 闭环仍暂停。任何一步超估计 2 倍先停。
- 结论：G0–G3 补进第 44 条（就地修正「三条路都没通」那一句，写明第几条通了、在什么条件下）；G2 进 I3 子文档与第 44 条；R40 就地修正第 40 条与中期 artifact 的那一行。

## 交付

1. G0：WOD 一张 cluster 表、NAVSIM 一张分组表（Δ、激活率）+ 判格。
2. G1：I3 上 gate 前后一张；WOD / NAVSIM 上 3 gate × 2 Δ 的表 + gate 描述 + 判格。
3. G3a：三种特征在 CARLA 与 E2 上的位移比值并排 + 读法；G3b：重训后的 E2 表；G3c：成本表。
4. G2：I3 held-out 与 WOD Cut_ins 的表。
5. R40：3 seed 的配对比较表，第 40 条与 artifact 的修正。
小表进 `research/results/real-data-transfer/`。

## 偏离与澄清日志

（执行时追加，时间戳早于受影响的数字。）

- 2026-09-26 08:30 CST [main] 执行分工与两条澄清（写于本轮任何数字之前）。
  (1) 并行四路：G0 + 共享 YOLO embedding（一个代理）、G1（一个代理）、G3a/G3c/G3b（一个代理）、R40（一个代理）；G2 等 G0、G1 判格出来后再派。
  各代理在本日志里只写自己标签的条目（[G0] / [G1] / [G3] / [R40]），提交前 `git pull --rebase`。
  (2) GPU：YOLO26x-seg 的检测一次性在 5 张卡上分片跑完（WOD val 子集、navtest、I3 帧、g₂ 需要的 WOD train / navtrain 训练帧），由 G0 代理负责，其余代理复用；之后的小 GPU 任务各自挑最空的卡。
  (3) G3a 的读法补一格：比值落在 [2, 3) 时照做 G3b，并在结果里标「灰区」；比值 ≥ 3 时停在 G3a，视频一致 inpainting 的登记交给用户。
  (4) E5 的 embedding 用的是 CARLA `route.json` 走廊与三路相机的平地抬升；真实数据上走廊（WOD / NAVSIM 没有路线中心线）、相机标定与只有前视这三处的操作化，由 [G0] 在任何 G0 数字之前登记，G1 / G2 / G3b 的 student 一律沿用。
- 2026-09-26 08:33 CST（box 时钟，下同）[G3] 分步时间估计与 G3a 的操作化，写于 G3 的任何数字之前（此前只读过 E2 / E5 的已发表数字与代码）。代码 `jevdrive/real_g3.py`，run `runs/real-data-transfer/g3*/<time>`，小表 `research/results/real-data-transfer/g3/`。
  **时间估计**：G3a 工程 30 min + CPU < 5 min；G3c 桌面估算 45 min（复用 E2 WOD 扫描 / 造对 / Qwen 抽取的实测速率）；G3b 若开：PDM 标签（911 个 token 的 v1.1 metric cache + 两条 proposal 各打一次，E3 实测 600 token 16 线程 345 s）约 15 min 墙钟，
  M-C 按 E2 代码重训 + R1–R3 约 20 min，student（等 G0 的 embedding 定义；编辑图上 YOLO 几千张约 5 min GPU、3 seed 拟合 + 读数约 30 min）工程约 2 h。合计约 4.5 h，不含等 G0 的时间。
  **G3a 口径**：(1) 统计量就是 E2 门 (d) 的 `elicit_e2_train.feature_floor`：每路按 x⁺ 行逐维标准化（std ≤ 1e-6 的维取 1）、逐行 RMS 位移，编辑对中位数 / null 对中位数，`_ratio_ci` 原样（2000 次，seed 0），
  分组从 log 换成 base 路线（P5 的独立单位），编辑对与 null 对按路线联合重抽。函数直接 import，不改。
  (2) **对照数的澄清**：本节正文写的「E2 的 1.44（Qwen）」是 R1（训练后 dual head 的 Δ 幅值比），不是特征位移；与 G3a 同一统计量的 E2 数是门 (d)：Qwen **1.21**、openpilot Cinque **2.40**、Lebowski 1.92。
  并排表以门 (d) 为对照；另加一行 R1 同款的 CARLA 描述（M-C 已存的样本外 Δ = `M-C pair` − `prior`，`preds_obs.npz`，|Δ(x⁺) − Δ(x⁻)| 对 |Δ(x⁺) − Δ(x_null)| 的 20 点平均幅值中位数之比），与 E2 的 1.43（E1 head）/ 1.44 并排，不进判格。
  (3) **对与 null**：x⁺ / x⁻ = P5 v1 BA 的 `obs` 表（每对 t_vis ≤ k < t_div − 1 的全部观测帧，即 hazard 已可见），null = `null` 表（seed 0 case 的 x⁺ 对「昼夜互换」的天气世界，同一组 k）。
  **判格 scope = 行人四个 family**（PedestrianCrossing、DynamicObjectCrossing、VehicleTurningRoutePedestrian、ParkingCrossingPedestrian）的 obs 帧，理由：E2 抹的是行人 / 骑车人，比的是「抹掉一个行人」的特征位移；
  分母用全部 null 帧（天气互换与 family 无关，样本多）。描述 scope：全部 family、cut-in 三个 family、行人 family 只取 reactive 帧（|d_expert| > τ_exp）、行人 family 的 null 只取行人 family 的 case、按 `factor_px` 三分位。
  (4) **特征**：Qwen `L18_last`（`p5_pairs.load_features`）、openpilot `temporal`（`p5_openpilot.load`，`op_streams_vis`，与 M-C / E5 同一抽取）、YOLO embedding = E5 的 `processed/elicit_e5/embed.npy`（64 维原样，mask 位也按同一公式逐维标准化）。
  YOLO 的 E2 侧数要等 G0 定真实数据上的 embedding，在 G3b 里补（描述，不进判格）。
  (5) **判格读 Qwen `L18_last` 行人 scope 的点估计**（本节读法写的是「CARLA 上 Qwen 的比值」）：< 2 → 编辑质量不是 E2 的瓶颈，做 G3b；[2, 3) → 做 G3b，标「灰区」；≥ 3 → 停在 G3a。openpilot 与 YOLO 两行只描述。
  (6) **限定先写下**：E2 的分母是「同形补丁贴空路面」的安慰剂（只量管线噪声），CARLA 的分母是昼夜互换（全图外观大变），两者不是同一种 null；按登记用天气 null 判。
  为了不让这一点被数字掩盖，另报一个描述量：两边编辑对的分子本身（各自 x⁺ 标准差单位下的 RMS 位移中位数，CARLA 行人 vs E2 的 0.322 / 0.070 / 0.076）。
- 2026-09-26 08:40 CST（box 时钟）[G1] 分步时间估计与 g₁ / g₃ / 统一接口 / I3 的操作化，写于 G1 的任何数字之前（此前只读过 E1 / E5 / I3 / 第 23 条 (e) 的已发表数字与代码）。
  代码 `jevdrive/real_g1.py`，run `runs/real-data-transfer/g1-*/<time>`，小表 `research/results/real-data-transfer/g1/`。g₂ 的标签与训练行、student 的 Δ 等 G0 的交接说明，另记一条，时间早于 g₂ / student 的任何数字。
  **时间估计**（墙钟）：接口 + g₁ + g₃ 工程 1.5 h；g₁ 训练（WOD train 41 万行 × 3 个 L1 × 2 模型，外加 5 折 OOF；navtrain 同样）GPU 约 40 min；lead 头补跑（I3 8.7k 帧、WOD val 那 19 663 帧所在的流、navtest 12k token、
  AUC 用的训练行抽样）单卡约 1 h，可与 g₁ 并行；I3 过一遍 15 min；WOD 表 CPU 15 min；NAVSIM 官方 devkit 打分（每个 arm v1.1 PDMS 约 3 min、main EPDMS 约 6 min，16 线程，6 路并行）约 1 h；
  g₂ 与 student 两路（G0 交接后）约 2 h；写表、图、第 44 条约 1.5 h。合计约 8–9 h，不含等 G0 的时间。
  (1) **统一接口**：gate 是逐帧标量 g(x) ∈ [0, 1]，gated arm = prior + g(x)·Δ(x)（软乘，不二值化）；Δ 与 prior 都是 E1 / G0 已有的那一个（M-C 用 E1 存下的 `wod_delta_<m>.npz` / `navtest_delta_<m>.npz`，即 CARLA 训练行标准化的主口径）。
  读数函数一字不改：WOD 用 `elicit_e1.readouts(d, prior, g·Δ, τ)`，τ 仍是该 Δ 头在 P5 null 上定的那个（M-C Cinque 1.628、Lebowski 1.655 m/s）；NAVSIM 按 `elicit_e1.run_navsim` 的写法把 g·Δ 加到 `ridge_late` prior 上，官方 devkit 打分，`navsim_table` 同一段配对 bootstrap。
  I3 上 g 逐帧作用（x⁺ 帧用 g(x⁺)、x⁻ / null 帧各用自己的 g），τ 由 `p5_exam.exam` 在 gated 预测的 null 对上照常重定（judge 原样）。gate 与 Δ 按模型配对（Cinque gate × Cinque Δ），主判 Cinque。
  (2) **g₁**（第 23 条 (e) 的配方）：`waymo_ladder.GatedResidual('mlp')`，输入 = 标准化的 [ego（`waymo_p0.load_all` 的 ego_state + intent，与 P5 / I3 的 `ego_input` 同一函数）⊕ op-<m> `temporal`]，
  目标 = 同一批 WOD train 行上 `ridge ego` 的残差，L1 ∈ {0, 1e-3, 1e-2} 按 inner split 的 pre-onset ADE 选，训练与早停是 `waymo_ladder._train` 原样（train_context 的 direction 0：fit = 全部 WOD train，不碰 val）；只取 gate 分支。
  标准化统计量用 gate 自己的 WOD train 行，I3 / WOD val 上同一张映射。NAVSIM：navtrain stage-one 行，输入 [32 维 ego（`navsim_heads.ego_features`）⊕ op-<m> `temporal`]，目标 = `ridge ego` 的 OOF 残差（x, y，8 点），
  早停与 L1 选择用按 log 分的 20% inner split（NAVSIM 没有 pre-onset 子集，改用 inner 整体 ADE），其余同 WOD。单 seed（0），gate 不是 Δ 头，不在「所有 head 报 3 seed」之列；如判格落在边界上再补 seed。
  (3) **g₃**：g₃ = p_lead · h(TTC)，p_lead = sigmoid(lead_prob[0])，lead 取 Q4c 的解码（`mu[:, 0, 0]` 的 x 与 v，selection 0、t = 0）；TTC = x_lead / (v_ego − v_lead)，仅当 v_ego − v_lead > 0.1 m/s，否则 TTC = ∞；
  h(TTC) = clip((6 − TTC) / (6 − 2), 0, 1)，即 TTC ≤ 2 s 全开、≥ 6 s 全关、中间线性。常数现在写死，不调。v_ego = 当前自车速度（WOD / I3：`past` 最后一步的 |(vx, vy)|；NAVSIM：`vel[-1]` 的模）。
  前提核对（任何 g₃ 数字之前）：openpilot 的 lead v 是对地速度——I3 `static` 世界里停着的车，v_lead 中位数应接近 0 而不是 −v_ego；不成立就改用相对速度并照记。
  lead 头输出不在盘上，补跑：WOD 用 `op_<m>_p3_trainval` 的同一流协议，只跑含评测帧的流并核对 `temporal` 与已存逐位相同；I3 用 `p5_openpilot.py` 的 I3 plan；navtest 用 `navsim_zs_openpilot.py feat` 的同一 rollout。三处都只加 `lead` / `lead_prob` 两个原始输出切片，默认输出不变。
  (4) **主 arm 的选择**（不看评测数）：每个数据集 × 模型一次，候选 g₁ / g₂ / g₃，比较量 = gate 在「训练行」上对 g₂ 的 hazard 标签的 AUC；g₁ 与 g₂ 用按 sequence / log 的 5 折 OOF 值，g₃ 无训练直接算。
  训练行 = g₂ 的训练行（WOD 与 navtrain 各自的，具体见 G0 交接后的 g₂ 条目）；g₃ 在这些行上补跑 lead 头。AUC 最高者为主 arm，平手（差 < 0.005）取更简单的（g₃ > g₂ > g₁）。
  (5) **I3 上的检查**（登记原文）：M-C Cinque 的双流翻转对 `ridge_late` prior 的逐帧配对差（`paired_vs_prior` 同一算法，场景 bootstrap）点估计 ≥ 0 且样本外 null false-flip ≤ 7% 记「gate 起作用」；g₁、g₃ 先，g₂ 用 I3 的 GT actor 标签训（按场景 5 折），等 I3 的 YOLO embedding。
  g₁ 在 I3 上用 WOD train 训的那一个（I3 的 ego / op 特征与 P5 同格式），不在 I3 上重训。
  (6) **NAVSIM 范围**：navtest 全部 12 146 token 与 E1 的分组（走廊内行人 / cyclist 897、直行、其余），`ridge_late` prior，PDMS（v1.1，主指标）与 EPDMS（main @ 0a380a9）；
  navhard two-stage 与 `cls_late` prior 不做（g₂ / g₃ 在 navhard 的合成帧上没有输入，`cls_late` 在 E1 里也只是描述）。
  (7) **判格**按 G1 节写死的两格，WOD 与 NAVSIM 各判一次（WOD：straight_yaw 激活率、全部 rater 帧与 Pedestrians 的 RFS Δ；NAVSIM：直行 token 激活率、全部与行人组的 PDMS Δ）；每个 arm 都报，主 arm 的格是 G1 的结论。

## 结果

（待填。）
