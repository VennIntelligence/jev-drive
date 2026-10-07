# op_parity turn-oracle 预登记：把真值可行驶边界几何喂给 plan 通路，急弯切内角消失多少（2026-10-08，任何本线读数之前写定）

## 问题

P2H 在 navtest 急弯的失败主体是切内角（第 153 条，54%）。第 165 条：冻结 Cinque token 读路沿几何比 WA-Cf 粗，但差距大半不是转弯特异的，
且冻结 V-JEPA 2.1 读得一样好却对 planner 无用。所以瓶颈在视觉（特征里没有够用的几何）还是在 plan head（给了好几何也不用）没有定。
本线用 oracle 判：把**真值**可行驶区域 SDF 作为特权输入喂给 plan 通路，看急弯失败少多少。特权输入只是探针，不是方法；不提交 test，不跑 WOD。

## 已有的、不重测

- 第 154 条（转弯采样 / 关 anchor / 后段横向加权）、144（侧机）、135 / 136（宽 FOV）、156 / 159（ego 历史 / ax 反馈）已关闭，不碰。
- 第 160 条 Stage 1：`pp_train --mem wa_cf`（32 个 WA-Cf token 作 adapter memory）在同一 pilot 配方上 +0.95，测试时屏蔽 memory −0.60：
  这条 memory 通道 plan 通路确实会用。本线的 oracle 走同一通道，H0 / MW 的 checkpoint 与 navtest 读数（`RH0-F-s{0,1}`、`RMW-F-s{0,1}`）直接复用，不重训。
- 复用：SDF 标签 `runs/op_probe/labels/{navtrain_s23456, navtest}.npz`（第 148 条 hinge 的同一份，0.5 m，x −8..56 m，y ±24 m）；
  探针 `opb_probe.mlp_fit_predict` 与 turn_probe 的目标 / 特征 / navtest 预测（`runs/op_parity/turn_probe/pred.npz`）；four_dirs 的插桩回放
  （`fd_navsim._init / work`）与桶定义；navtest 打分只走 `python -m jevdrive.bench`。

## 设计

**配方**：第 160 条 Stage 1 的 H0（= P2H pilot：`pp_train --arm P2 --frames warp --hinge-lam 10`，`navsim/op-parity-s234`，3 000 步 × 64，warmup 100）
加 `--mem <kind>`。除 memory bank 的内容外，所有臂的参数量、token 数（32）、步数、行流（同 seed 同行序）、MEM_DROP 0.25 完全相同。
训练代码只在 `pp_train.py` 的 `--mem` 选项里加 4 个 kind，shipped P2 / P2H 路径不动。

**注入方式与理由**：32 个 memory token × 512 维，经 `ParityAdapter` 的 side 通道（LayerNorm → Linear → 2 层 decoder → 加到 9 帧 hidden token 的 bias）。
选它因为 (1) 这是第 160 条已证明 plan 通路会用的通道，oracle 为负时不能归因于「通道本身不通」；(2) 与 MW 同容量，可直接对照「真实特征带来多少 / 真值几何带来多少」；
(3) bench 已支持按 kind 读 bank 和 `:noside` 屏蔽，不需要新 runner。
**编码**：标签 raster 128 × 96（0.5 m）无损放进去：token k = x 方向第 4k..4k+3 行（2 m 一条，自车后 8 m 到前 56 m），384 个值 = clip(SDF, ±6 m) / 3；
维 384–447 填常数 1（LayerNorm 之后幅值仍可恢复），448–511 填 0。只含地图几何（ego 系固定网格），不含 logged path / 路线：不泄漏专家轨迹。

**臂**（tag `TO?-F-s{seed}`）：

| 臂 | bank | 含义 |
|:--|:--|:--|
| H0（复用 RH0） | 无 memory | P2H pilot 基线 |
| OS `sdf_shuf` | 真值 SDF，行在各 data dir 内固定随机置换（保证换到别的 log） | **匹配对照**：同容量、同边缘分布、内容与当前帧无关 |
| OG `sdf_gt` | 当前 token 的真值 SDF | **oracle** |
| PW `sdf_wa` | turn_probe MLP 从 WA-Cf 读出的 1 m SDF，双线性上采样到 0.5 m，同样编码 | 现实质量的几何（好的 encoder） |
| PV `sdf_v` | 同上，从 Cinque `view_39` 读出 | 现实质量的几何（冻结 Cinque） |
| MW（复用 RMW） | WA-Cf 原始 token | 参照（第 160 条） |

PW / PV 的 navtrain bank 用按 log 哈希的 5 折 out-of-fold 预测（MLP 与 turn_probe 完全相同：z-score 的 [X, E]，目标 raster + corridor，3 000 步，seed 0），
navtest bank 直接用 turn_probe 已有的全量训练 MLP 预测。bank 质量（边界带 MAE，navtrain OOF / navtest，以及「真值降到 1 m 再上采样」的分辨率损失）在打分前记录。
测试时各臂用自己的 bank（OS 在 navtest 上也是置换的）。另读 OG `:noside`（屏蔽 memory）。

**读数**（navtest 全部 12 146 token，`jevdrive.bench`；转角分层用 logged 4 s 航向变化：S5 < 5°、T20 > 20°、T45 > 45°；four_dirs 的 R < 15 m 急弯集作次读数）：
EPDMS；DAC 失败率（全体 / T20 / T45）；**T45 切内角率** = T45 token 中 DAC < 1 且 LQR 回放首个出界角在转向同侧（four_dirs `inside`）的比例；
**T45「转不过去」率** = DAC < 1、外侧、航向增益 < 0.9（four_dirs D1u 口径）；raw plan 出界占比；S5 EPDMS（直行代价）；EP、NC + TTC 失败率。
侧别来自对各臂 DAC 失败 token 跑 four_dirs 的插桩回放（同一 `_departure`），并核对回放 DAC 与 bench DAC 一致。
统计：每 token 取 seed 均值，按 log 聚类的配对 bootstrap（`jevdrive.stats.paired`，与 rep s1 表一致），闭合比的 CI 用同一 log 重采样的比值。

**主量**：Q = T45 切内角率；闭合 c = 1 − Q(OG) / Q(OS)。

## 步骤与闸门

1. bank（GPU 一次，约 10 分钟）→ seed 0 四臂训练（各约 4 分钟，过池）→ navtest（OG、OG:noside、OS、PW、PV）→ 回放 → seed-0 闸门。
2. **seed-0 闸门（明确阴性）**：OG 对 OS 在 seed 0 上同时满足「T45 切内角闭合点估计 < 0.25」与「T20 DAC 失败率下降 < 0.4 pp」→ 明确阴性，不跑 seed 1，
   按 seed 0 判「plan head」（报告里标明单 seed）。否则四臂都补 seed 1，按 2-seed 均值判。
3. 「全量 2 seed」在本线指 **pilot 规模的 2 seed**（与第 160 条 Stage 1 相同）：全量 navtrain 每 seed 约 1 卡时 × 4 臂超出 Step 1 的 2 卡时预算，这是打分前声明的口径。
   预计 Step 1 约 1 卡时，CPU 打分（每次 navtest 约 7 分钟 × 9–10 次）是墙钟主项。navhard 不跑：navhard 合成帧没有 SDF 标签，生成标签不属于「便宜」。

## 分支规则（阈值现在定死）

- **视觉是瓶颈**：c ≥ 0.5 且 Q(OS) − Q(OG) 的 95% CI 不含 0。
- **plan head 是瓶颈**：c < 0.25。
- **两者都有**：其余情况（含 c ≥ 0.5 但 CI 含 0）。
- 无效 / 停下报告：Q(OS) 的 2-seed 失败 token-seed 数 < 30（无功效）；或 OS 对 H0 的 navtest EPDMS 差超出 ±0.5（对照本身改变了模型，归因不成立）。
- 报告但不进规则：OG 的 T45 总 DAC 失败是否上升（切内角换成转不过去）、S5 EPDMS 差（< −0.2 记为直行代价）、OG `:noside` 相对 OG 的下降（通道是否被用）。
- PW / PV：存活比 s = (Q(OS) − Q(X)) / (Q(OS) − Q(OG))，只在 oracle 效应 CI 不含 0 时报告；它决定 Step 2 视觉分支值不值得做（PV ≈ 0 而 PW 高 → 需要更好的特征；
  PV 已高 → 几何其实读得出，问题在怎么送进 head）。

## Step 2（规则选出分支后直接继续，打分前在本文件追加补充登记）

视觉分支 = 解冻后段视觉 + 可行驶边界辅助 loss 的 pilot；plan-head 分支 = 先便宜诊断（decoder 的 anchor / 低曲率先验、0.25 anchor 蒸馏行、时域权重、假设选择），再对最优候选做一个 pilot；
「两者」= 先视觉 pilot，再最便宜的 plan-head 干预。结果指向别处则停下报告。

---

## Step 2 补充登记：plan-head 分支（2026-10-08，Step 1 读完之后、Step 2 任何读数之前写定）

**Step 1 结果**（seed 0）：OG 对 OS 的 T45 切内角闭合 −0.08 [−0.19, +0.02]，T20 DAC 失败 +0.29 pp，按闸门为明确阴性，规则判「plan head」。
但 OG 屏蔽 memory 后读数不变（EPDMS +0.02），dev ADE OG / OS / PW / PV 都是 0.61（H0 0.62，MW 0.42）：plan 通路根本没去读 SDF token。
所以 Step 1 没有分开「给了好几何也不用」和「这种编码经 adapter 在 3 000 步内学不会」。Step 2 先把这个分开，再做 pilot。

**已定论、不重测的 head 侧项**：转弯 token 关 anchor 行（第 154 条 T2）、后段横向加权（154 T3）、转弯平衡采样（154 T1）、曲率 / 航向增益封顶（第 148 条第 5 点：斜率 0.94、不封顶）。
「假设选择」不适用：Cinque 的 plan 输出是单假设（495 个均值 + 495 个方差）。因此本分支剩下的问题是 plan 通路**能不能、在什么条件下**用上正确几何。

### Stage A：fresh head 上限（离线，不动 P2）

第 147 / 160 条的 thin decoder（`rep.py decode` 的同一网络、同一 hinge 循环、4 000 步、λ 10、margin 0.3、seed 0），输入 z-score 后拼接：
`V` = [Cinque view_39, E]（复现第 160 条 Stage 0 的 T20 DAC 失败 10.59%，作为实现校验，差 > 0.5 pp 则先查错）；`V+G` = [V, 真值 1 m SDF raster（3 072 维）, E]；
`V+S` = [V, 置换到别的 log 的 SDF raster, E]（匹配对照）；`G` = [真值 SDF raster, E]（只有几何）。
读数：navtest 全部 T20 token（3 154 个）的 DAC 失败率、T45 切内角率、转不过去率；DAC 用 `jevdrive.bench score-poses`（官方路径）并与 four_dirs 回放核对，侧别来自回放。
**判读**：c_A = 1 − Q(V+G) / Q(V+S)（T45 切内角率）。c_A ≥ 0.5 且差的 CI 不含 0 → 真值几何对一个新 head 是可用的，Step 1 的阴性落在 P2 通路 / 注入方式上（「没读进去」）；
c_A < 0.25 → 新 head 拿到真值几何 + hinge 也避不开切内角，问题在目标 / 输出参数化（8 个原始位姿的 hinge、模仿目标本身），与视觉、与 P2 head 的先验都无关；其间 → 部分。

### Stage B：P2 通路上的便宜干预（seed 0，各自带匹配的置换对照，读数与 Step 1 相同）

| 对 | 改动（其余同 Step 1 的 OG / OS） | 检验的解释 |
|:--|:--|:--|
| B1 `OG9 / OS9` | 9 000 步（3 倍） | 学得慢：adapter 输出零初始化 + 几何带来的模仿梯度弱，3 000 步不够 |
| B2 `OGh / OSh` | hinge λ 30、margin 0.5 | 目标不要求：λ 10 / 0.3 m 的 hinge 项只有模仿项的 ~5%，head 没有理由去读几何 |

各对的读数：T45 切内角闭合（OG? 对 OS?）、T20 DAC 失败差、S5 EPDMS、memory-off 读数（通道是否被用）。
**选优与 pilot**：闭合点估计最大、且 ≥ 0.25、且 T20 DAC 失败下降 ≥ 0.4 pp 的那一对为最优候选，补 seed 1（唯一的一个 pilot），按 2-seed 均值用 Step 1 的阈值判
（≥ 0.5 且 CI 不含 0 = 「给对条件，plan 通路能用真值几何消掉一半以上切内角」；< 0.25 = 不能）。两对都不到 0.25 → 不做 pilot，停下报告：
在 adapter memory 通道上，P2 plan 通路在这些便宜变化下都不用真值几何；Stage A 说明信息本身可不可用。
不在本分支内、不做：新的注入结构（卷积编码器、把几何接进 ego MLP）、解冻视觉、全量训练、HUGSIM。结果若指向这些，停下报告。
预算：Stage A 约 0.1 卡时 + CPU 打分；Stage B 约 0.6 卡时 + 4–6 次 navtest；pilot 约 0.3 卡时。
