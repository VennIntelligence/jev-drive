# op_probe：P2 在 navtest DAC 上输给 WA-JEPA 的信息丢在哪一层（预登记，2026-10-06，写在任何读数之前）

接第 144 / 145 条与 op_parity gap 页：navtest 上 P2 与 WA-JEPA 的 EPDMS 差距约六成是 DAC（P2-s0 失败 537 / 12 146，WA-JEPA 219；
P2-s0 失败而 WA 通过 433 个 token，反向 115）。代码 `scripts/opb_*.py`，结果 `results/dac-localize.md`。

## 0. 问题

在这些 DAC 失败 token 上，路面 / 可行驶区信息是 (a) 冻结的 comma 视觉特征里就没有，(b) 视觉里有但在 temporal / hidden 通路里丢了，
还是 (c) 到 plan head 之前都在、但 plan 没用上（解码 / 策略）。

已定的不重测：第 104 条（NAVSIM 相机高度把世界缩到 0.7，模型读图本身对）；第 112 条（navhard DAC 失败约一半在评分器一侧：多边形窄、跟踪器滞后）；
第 145 条（解冻前视 encoder、1.40 m 虚拟相机 pilot 都 < +0.5）；第 40 / 62 / 89 条（冻结 openpilot 特征的其他可读性）。

## 1. 层（P2 = op_parity P2-F-s0，W 帧，与 88.21 的读数同一缓存）

| 层 | 张量（cinque.ort.onnx 的 port） | 维度 | 说明 |
|:--|:--|:--|:--|
| V | `view_39`，当前帧 | 32 × 512 | 冻结视觉输出（P0 / P1 / P2 相同） |
| M | `add_40`，当前帧 | 32 × 512 | + adapter bias（ego / pose / cmd）+ 两个逐 token MLP 块 |
| T | `select_4` | 512 | 4 层 temporal transformer（288 token）后取出的汇总 token |
| H | `add_54` | 512 | plan head 的隐层（再一个 Gemm 就是 plan） |
| O | plan 均值 | 8 位姿 | 输出本身 |
| E | ego 输入 20 维 | 20 | 平凡基线：只给 ego / pose / cmd |

参照：WA-JEPA 已发布 checkpoint 的 `context_scene`（V-JEPA 2.1 encoder + scene projector，WA-C）与轨迹 head 前的隐层（`traj_out` 的输入，
最后一个 flow 步，WA-H）；shipped P0 的 T / H（看微调改了什么）。WA-JEPA 特征用 bf16 批量前向、固定噪声（只取特征，不用它的计划）。

## 2. 标签（NAVSIM 地图 API，评分器自己的图层）

- 可行驶面 = ROADBLOCK ∪ INTERSECTION ∪ CARPARK_AREA（`PDMScorer` 判 NON_DRIVABLE_AREA 用的多边形类型；`PDMDrivableMap` 里 roadblock connector
  只以 LANE_CONNECTOR 出现，这个判据不认；半径取 80 m 而不是 50 m，覆盖整个栅格）。t0 后轴坐标系（x 前、y 左），有符号距离场 SDF（+ 在内，m），0.25 m 精栅格做 EDT，存 0.5 m（128 × 96，x ∈ [−8, 56)，y ∈ [−24, 24)）。
- 探针目标：1 m 下采样 SDF（64 × 48 = 3 072 格，截断到 ±10 m）。
- 派生量（全部从真值 SDF 算，评估时从预测 SDF 同样算）：
  - **走廊**：沿日志未来路径（8 个位姿 + 末段直线外推）弧长 s ∈ {5, 10, 20} m 处，沿法向到左 / 右边界的自由距离（截断 15 m），6 个数。
  - **自车计划余量**：P2 计划 8 个位姿（线性插到 0.1 s）上 SDF 的最小值减半车宽 1.15 m；< 0 = 计划中心线带车宽出界。
- 检查（小读闸门的一部分）：navtest 上与 metric cache 的 `drivable_area_map` 在 45 m 内的格点一致率 ≥ 98%；日志未来路径在 SDF 内（≥ −0.4 m）的 token ≥ 95%。

## 3. token 集合（navtest，op_parity gap 的逐 token 分项）

- **F**：P2-s0 DAC 失败且 WA-JEPA 通过（433）。**PP**：两者都通过。**R**：WA 失败、P2 通过（115）。**FF**：都失败（104）。
- F 再分（评分器自己的 LQR 回放 + 多边形）：F-plan = 原始计划（不经 LQR）带车框已出界；F-lqr = 只有 LQR 回放出界；F-graze = 出界最大深度 < 0.3 m。
  若 F-lqr + F-graze > 50%，结论只对 F-plan 下，并写明。
- 探针训练：navtrain 5 个分片（`navtrain_full.s2..s6of12`，约 4.3 万 token）；超参（ridge λ）在 `navsim/op-parity-full-dev` 的 log 上选；
  测试只在 navtest（与 navtrain log 不交）。

## 4. 探针与读数

- **探针**：每层一个 ridge（特征标准化，多输出，λ 网格在 dev 上按 R² 选），目标 = SDF 栅格 3 072 维；另一个 ridge 直接回归 6 个走廊距离。
  非线性复核：同输入的 2 层 MLP（1 024 隐层），只报告不判。E（ego 20 维）是平凡基线。
- **M1 走廊误差**：6 个走廊距离的 MAE（m）；技能 = 1 − MAE(层) / MAE(E)，分别在 F、PP 上。
- **M2 出界识别 AUC**：用预测 SDF 在 P2 自己的计划上算余量，按余量排序区分 F 与 PP（AUC，越高 = 该层「知道」这条计划会出界）。
  对照：同一计划、WA-JEPA 特征上的 AUC；真值 SDF 上的 AUC（上限，评分器口径以外的差异）。
- **M3 解码测试**：每层（+ E）训一个 MLP 计划解码器（8 位姿，后轴系），两种目标：D-imit（只模仿日志）与 D-hinge（模仿 + 用真值 SDF 的出界 hinge，
  navtrain 上）。navtest F 与 PP（随机 1 500）上用评分器自己的 `pdm_score` 打分：F 上的 DAC 通过率与 PP 上的 DAC / 分数。
- **消融**（P2 本体，F ∪ PP-1500，同一评分器）：ego 速度 / 加速度置零、位姿历史置零、命令置零、命令换成「直行」、整个 bias 关（present = 0）、
  前视帧换 G（GIMM）/ N（原生 2 Hz 关键帧）、只留当前帧（8 个旧槽置零）。报告 F 的 DAC 通过率、PP 的 DAC 与分数变化。
- CI：navtest 按 log 聚类 bootstrap（B 10 000，`jevdrive.stats`）。

## 5. 判定（写死）

记 sk(层, 集) = M1 技能，auc(层) = M2 AUC（F 对 PP），WA = WA-C 的对应值。

- **(a) 视觉里没有**：sk(V, F) < 0.5 × sk(WA-C, F) 且 auc(V) < 0.65。
- **(b) 视觉有、后面丢**：V 有（sk(V, F) ≥ 0.75 × sk(WA-C, F) 或 auc(V) ≥ 0.75），且从 V 到 H 掉得多：auc(V) − auc(H) ≥ 0.10 或 sk(V, F) − sk(H, F) ≥ 0.15，
  并且这个掉幅在 F 上比在 PP 上大（差的 CI 不含 0）。
- **(c) 到 H 都在、没被用上**：auc(H) ≥ 0.75 且 sk(H, F) ≥ sk(V, F) − 0.10，并且 M3 的 D-hinge(H) 在 F 上的 DAC 通过率比 P2 本体（0）高 ≥ 40 个百分点、PP 的分数降幅 ≤ 1。
- 都不满足 = 不确定，按证据最近的一类写并降级为「弱」，再考虑激活修补（失败 token 与匹配的通过 token 之间换 T / H）。
- 证据强度：判定在 F-plan 子集上也成立且 CI 不跨阈值 → 中；只在全 F 上成立或贴线 → 弱。

## 6. 小读（在放量之前）

navtrain 1 个分片（s2，约 8.6 千）训探针，navtest 400 个 token（F 中随机 200 + PP 中随机 200，seed 0）评估，WA-JEPA 特征只算这些 token。
闸门：(1) 标签检查过（第 2 节）；(2) 抽出的 P2 计划与 op_parity 存档计划逐 token 一致（最大位置差 < 0.05 m）；(3) 评分器在这 400 个上的 DAC 与
devkit CSV 一致 ≥ 99%；(4) 至少一层在 PP 上 sk > 0.2（探针在工作）。小读的 M1 / M2 若已把 (a)/(b)/(c) 分开，照样放量拿 CI；若与预想差很多，报告后再放量。

## 7. 次要（便宜才做）

HUGSIM 起步停滞（第 146 条 X 9/12 对 C 3/12）的同类定位：不在本预登记内，主线完成且剩余预算够时另写附录。

## 附录 1（小读之后、放量之前写，2026-10-06 14:30）

小读（navtrain s2 8 608 token 训探针，navtest F 200 + PP 200）的四个闸门都过：标签与 metric cache 一致 99.97%、日志未来全部在面内；抽出的 P2 计划
与存档逐位相同；评分器 DAC 与 devkit 400/400 一致；PP 上 P2-V 技能 0.41。小读暴露两个读法问题，放量时加以下读数（判定规则第 5 节不改）：

1. **M2 被计划形状混杂**：只用 ego 的探针 AUC 已有 0.74（弯的计划在任何「通用路面」先验下余量都小）。M2 照报，另报（事后、不入判定）
   **余量偏差** = 预测余量 − 真值余量（在 P2 自己的计划上），F / PP / F − PP，按 log 聚类 CI：表示该层的路面估计是否「看见」这条计划出界。
2. **F 的选择偏差**：F 按「WA 通过」挑出，WA 派生的解码器 / 探针在 F 上天然占便宜，P2 派生的天然吃亏（回归均值：E 解码器在 F 上也有 40% 通过）。
   放量时 M1 / M3 同时报 R（WA 失败、P2 通过，115）与 FF，并报按层的**分层全 navtest DAC 失败率**（F / R / FF 全数 + PP 随机 1 500，按各集合大小加权回全体），
   这个数不受选择影响，是 M3 的主读数；F 上的通过率是次读数。
3. 消融同样有回归均值：F 上任何扰动都翻过 25–50%。对照用 P2-F-s1（同配方另一 seed，devkit 读数）在 F 上的通过率。
