# 榜单「调料包」（skill pack）：同一个 openpilot 基座，按榜单换前后端薄插件（2026-09-29）

状态：分析 + 方案，**待定**。没有跑新的 GPU 实验；本文的新数字全是已有输出上的 CPU 分析（小表在 [results/skill-pack/](results/skill-pack/)，
脚本 `scripts/skill_pack_nav_decomp.py`、`scripts/skill_pack_wod_posthoc.py`）。文献数字由子 agent 读 arXiv HTML 摘录，标「摘要读」的是模型转述，引用前要对 PDF 核一遍。
相关：[decisions.md](decisions.md) 第 35、37、40（第 6 点 E6）、46、53 条；[tmp/2026-09-29-leaderboard-wod.md](../tmp/2026-09-29-leaderboard-wod.md)（三张榜的对照表）；
正在跑的 NAVSIM 导航 lane [todos/2026-09-29-op-leaderboard.md](../todos/2026-09-29-op-leaderboard.md)；缺口分析 [midterm-gaps.md](midterm-gaps.md)。

## 0. 结论先行

故事设定：基座是轻度适配的 openpilot（Cinque，权重不动，这是「trade」，能力来源）；每张榜配一个薄的「调料包」（skill pack：输入契约适配的前端插件 +
对准该榜 metric 的后端选择 / 校准插件，这是「trick」，分数配方）。按用户 2026-09-29 的澄清，pack **不要求与基座无关**，效果优先，可以直接吃 openpilot 的
原生 plan、`temporal` 特征、desire 输入和 plan 的纵向偏差；哪些部分也能搬到别的基座只作旁注。

1. **NAVSIM 差距的构成**：openpilot 零样本 84.2 PDMS 丢的 15.8 分里，**EP（ego progress，相对 PDM-Closed 的进度）8.7 分、DAC（drivable area compliance，出不出可行驶区）4.3 分**，
   NC（no at-fault collision）1.6、TTC（time to collision）1.1、C（comfort）0。和 93.8–95.3 的榜首相比，按均值公式换算，差距约 60% 在 EP、25% 在 DAC、NC + TTC 约 12%。
   openpilot 的 NC / TTC 已经和 Hydra-MDP、DiffusionDrive 同档；它**开得慢、转弯出界**，不是撞得多。榜首的 EP（90–93）甚至**高于 human log 的 87.5**，这本身就是对准 metric 的信号。
2. **机制归因**：能动 EP 与 DAC 的，文献里有三类，都是「对准 PDM scorer」：(a) 词表 + 按 PDM 子分学出来的打分头（Hydra 系，加权推理一步 DAC +4.4）；(b) 用 PDM 类奖励的 RL（ReCogDrive +4.3、AutoVLA +8.6，主要是 EP 与 DAC）；
   (c) 更大的 backbone 和在 navtrain 上训（EP +5 到 +8）。LiDAR 不是必要条件（榜首四个都是纯相机）；合成数据主要涨 navhard 不涨 navtest；AutoVLA 的 92.1 是 oracle best-of-6，不可比。
3. **我们自己已有的两条证据说明 pack 在 openpilot 上有空间**：E6 的 Hydra 式打分头在同一套候选上 +6.3 PDMS（第 40 条第 6 点，hold 输入的特征）；
   而它和补帧后的原生 plan 恰好**互补**——两者都是 84.2，逐 token 相关只有 0.29，**逐 token 取两者较好的 oracle 是 91.2**。WOD 上，把 Cinque 原生 plan 纵向拉长 6%（按序列两折交叉拟合选 s）
   RFS +0.12 [+0.05, +0.20]，val 8.005 → 8.119，与 test 榜 #2–#6（8.08–8.09）同量级。
4. **N0 已跑（2026-09-29 更新，决策第 64 条）**：「原生 plan ⊕ Hydra 选择」的切换器 + 拉长 1.1 倍的速度候选，参数只在 navtrain 子集（2 428 token）上选，navtest 只打一次分：
   **84.94，对原生 plan +0.77 [+0.28, +1.27]**，按登记「成立」，但只有写在分数之前的预期（+2 到 +4）的三分之一；单纯拉长速度在 NAVSIM 上零和（EP 涨、DAC / TTC 跌）。
   **N1 已跑（决策第 68 条）**：在补帧输入下重抽 navtrain `temporal`、重训打分头、把原生 plan 放进独立候选槽，navtest **87.32，对 N0 +2.38 [+1.94, +2.83]**，对原生 +3.15，按登记「成立」，高出预期（85.0–86.5）；增益主要来自独立原生槽（+1.6 到 +1.9），补帧特征本身只有 +0.25。
5. **诚实的边界**：中期前 NAVSIM 进前 5（≥ 93.5）不现实，现实目标是「零改权重的基座 + 薄 pack 到 DiffusionDriveV2 / GoalFlow 档（88–91）」；WOD 若 test 上也有 +0.1 量级，能进前 10。
   故事应该是「同一个基座，每榜一个 pack 把分拉上去；pack 动分数，配对考卷上的能力不动」，第二半句要真的量。

## 1. NAVSIM：84.2 离 91–95.6 差在哪

### 1.1 我们自己的逐 token 分解

数据：box 上 navtest 全量（12 146 token）的 devkit 逐 token 输出（v1.1 PDMS）：原生 plan + GIMM-VFI 补帧（`v1_navtest_opi_navfull_gimm_g0.2-cinque__base`，即 84.2 那一跑）、
E6 的 Hydra 式打分头（hold 输入的 `temporal`，`v1_navtest_e6_hydra_cinque`）、同候选集的模仿 softmax 对照（`e6_clsref`）。

PDMS 逐 token = NC × DAC × (5·EP + 5·TTC + 2·C) / 12。分解按「先门后分」：1 − M·W = (1 − M) + M·(1 − W)，M = NC·DAC 的损失按 NC、DAC 各自的缺口比例分，
M·(1 − W) 按 5 / 5 / 2 的权重分给 EP / TTC / C。这样被 NC 或 DAC 判零的 token 整个算在门上，逐 token 精确可加（重构误差 0）。

| 行（navtest n = 12 146） | PDMS | 丢分 | NC | DAC | EP | TTC | C | NC 失败 | DAC 失败 | 零分 token |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 原生 plan，GIMM 补帧 | 84.17 | 15.83 | 1.63 | 4.33 | **8.73** | 1.14 | 0.00 | 1.8% | 4.4% | 5.9% |
| Hydra 式打分头（hold 特征，E6） | 84.18 | 15.82 | 1.82 | 6.45 | 6.22 | 1.32 | 0.00 | 2.1% | 6.6% | 8.2% |
| 同候选集模仿 softmax（E6 对照） | 77.86 | 22.14 | 3.02 | 12.49 | 4.82 | 1.81 | 0.00 | 3.8% | 12.9% | 15.4% |

读法：原生 plan 的丢分过半在 EP；零分 token 718 个里 539 个是 DAC 判零、197 个是 NC 判零。两者都有效（NC·DAC > 0）的 10 668 个 token 上，原生 plan 的 EP 是 78.0，Hydra 选择是 83.8；
EP < 0.5 的 token 占 7.5% 对 3.7%。结合原生 plan 4 s 纵向终点比 log 平均短 3.4 m（`results.csv` 的 `lon4`），这是**系统性偏慢**。
按 command 拆（[integration §9](openpilot-openloop-integration.md)）：左 / 右转 EP 只有 68 / 61、DAC 91 / 87，直行 EP 77、DAC 98——慢和出界都集中在转弯。

两者互补的程度：

| 逐 token 取较好者（oracle，不可部署） | PDMS |
|:--|--:|
| 原生 plan 单独 / Hydra 单独 | 84.2 / 84.2 |
| max(原生, Hydra) | **91.2** |
| max(原生, Hydra, 模仿 softmax) | 92.8 |
| E6 留出 log 上词表 1024 条的 oracle（第 40 条 6） | 99.8 |

原生 plan 判零而 Hydra > 0.8 的 token 占 3.3%，反过来 5.0%；逐 token PDMS 相关 0.29。所以「在原生 plan 和打分头之间切换」有约 7 分的 oracle 空间，
一个能学到其中三分之一的切换器就有 +2 分。这是 pack 的第一个组件的依据。

### 1.2 与公开子分对照

公开数字来自各论文 navtest 表（v1 PDMS）。

| 方法 | 传感器 | NC | DAC | TTC | C | EP | PDMS | 出处 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| **openpilot Cinque 原生 plan + GIMM（我们，zero-shot）** | 1×C | 98.3 | 95.6 | 95.4 | 99.96 | 73.1 | **84.2** | 第 37 条 |
| human | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 | 2406.15349 Tab1 |
| TransFuser / LTF | C+L / C | 97.7 / 97.4 | 92.8 / 92.8 | 92.8 / 92.4 | 100 | 79.2 / 79.0 | 84.0 / 83.8 | 2406.15349 Tab1 |
| PDM-Closed（真值感知） | – | 94.6 | 99.8 | 86.9 | 99.9 | 89.9 | 89.1 | 2406.06978 Tab1 |
| Hydra-MDP-V8192-W-EP（ResNet） | C+L | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 | 2406.06978 Tab1 |
| Hydra-MDP++ V2-99 | C | 98.6 | 98.6 | 95.1 | 100 | 85.7 | 91.0 | 2503.12820 Tab1 |
| DiffusionDrive / V2 | C+L | 98.2 / 98.3 | 96.2 / 97.9 | 94.7 / 94.8 | 100 / 99.9 | 82.2 / 87.5 | 88.1 / 91.2 | 2411.15139 / 2512.07745 Tab1 |
| GoalFlow | C+L | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 | 2503.05689 Tab1 |
| iPad R34（摘要读） | C | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 | 2505.15111 Tab1 |
| DriveSuprim ViT-L（摘要读） | C | 98.6 | 98.6 | 95.5 | 100 | 91.3 | 93.5 | 2506.06659 Tab2 |
| AutoVLA SFT / RFT / best-of-6（oracle） | C | 96.9 / 98.4 / 99.1 | 92.4 / 95.6 / 97.1 | 88.1 / 98.0 / 97.1 | 99.9 | 75.8 / 81.9 / 87.6 | 80.5 / 89.1 / 92.1 | 2506.13757 Tab1 |
| RAP-DINO | C | 99.1 | 98.9 | 96.7 | 100 | 90.3 | 93.8 | 2510.04333 Tab1 |
| DrivoR | C | 99.0 | 98.9 | 96.7 | 100 | 90.0 | 93.7 | 2601.05083 Tab1 |
| CLOVER | C | 99.1 | 99.0 | 96.9 | 100 | 91.7 | 94.5 | 2605.15120 |
| DriveZero-Scale | C | 99.2 | 99.4 | 97.3 | 99.9 | 92.5 | 95.3 | 2609.06055 Tab4 |

把我们的某一个子分换成参照方法的值、其余不动（均值公式，换算回 84.2 的尺度；均值公式本身对我们给 81.6，只作份额参照）：

| 参照 | 全换的差 | 只换 NC | 只换 DAC | 只换 TTC | 只换 EP |
|:--|--:|--:|--:|--:|--:|
| DriveZero-Scale（95.3） | +13.2 | +0.8 | +3.3 | +0.8 | **+7.8** |
| RAP-DINO（93.8） | +11.4 | +0.7 | +2.9 | +0.5 | **+6.9** |
| Hydra-MDP++ V2-99（91.0） | +8.1 | +0.3 | +2.6 | −0.1 | **+5.1** |
| DiffusionDriveV2（91.2） | +7.7 | 0.0 | +2.0 | −0.2 | **+5.8** |

读法：EP 是大头，DAC 第二，NC / TTC 合计不到 2 分。和 AutoVLA SFT → RFT 的形状一样（NC / TTC 高、EP 低），RL 或打分头主要把 EP 与 DAC 拉上去。
榜首 EP 92–93 高于 human 87.5：EP 以 PDM-Closed 的进度为分母，开得比人还「积极」才能拿到，这一段分数不是驾驶能力（第 35 条的 R 层 / metric 代理）。

### 1.3 机制逐项归因（论文自己的 ablation）

| 机制 | 论文里的增益（PDMS，除非注明） | 动的子分 | 对 openpilot 的含义 |
|:--|:--|:--|:--|
| 词表 + 学出来的 PDM 子分打分头（Hydra 系） | Hydra-MDP：纯模仿 80.9 → 子分蒸馏 83.0 → 加权推理 85.7 → EP 连续蒸馏 86.5（2406.06978 Tab1）；只蒸馏总分 80.2，低于纯模仿；Hydra-MDP++ 加权 +1.5（2503.12820 Tab3）；GoalFlow 打分器 +4.7（2503.05689 Tab2） | 加权一步 DAC 91.7 → 96.1；EP 小 | 我们 E6 已在冻结 `temporal` 上复现 +6.3；可直接用 |
| backbone 尺度 / 分辨率（navtrain 上训） | Hydra-MDP ResNet 86.5 → ViT-L / V2-99 89.9 / 90.3；Hydra-MDP++ R34 → V2-99 +4.4 | EP +5 到 +8、DAC +2.6 | 基座不换，这一项不在 pack 里；它是 specialist 与我们的结构性差距 |
| 集成 | Hydra-MDP-C（A + B 集成）91.0，对单模型 +0.7 | 小 | Cinque + Lebowski 两个打分头集成，便宜，预期 < 1 |
| 粗到细选择 / 自蒸馏（DriveSuprim，摘要读，EPDMS） | R34 81.4 → 82.4 → 82.7 → 83.1 | EP 83.1 → 88.4 | 次要 |
| RL，奖励为 PDM 类分数 | ReCogDrive 86.5 → 90.8（2506.08052 Tab3）；AutoVLA SFT 80.5 → RFT 89.1（2506.13757 Tab1）；DiffusionDriveV2 88.1 → 91.2（2512.07745） | EP +5 到 +6、DAC +2 到 +3、AutoVLA TTC +10 | 要动基座的 plan 头，中期后 |
| best-of-N / 测试时选择 | AutoVLA best-of-6 92.1 是 **oracle 打分器**选（§4.2），不可比；可部署的是学出来的选择器（CLOVER 64 候选、iPad、DrivoR） | – | oracle 选择不进 pack；学出来的选择器 = 打分头 |
| 规则式后处理 / PDM-Closed 兜底 | Hydra-MDP 不可微后处理 +0.2；没找到学习式条目用 PDM-Closed 兜底 | – | PDM-Closed 要真值地图与感知，不合规；低优先 |
| ego status 与 command | TransFuser 去掉速度 / 加速度 −1.5 到 −2.6（2406.15349 Tab2）；只用 ego 的 MLP 65.6；没有任何论文报「去掉 command」 | – | openpilot 已隐式吃自车运动；command 由导航 lane 负责 |
| LiDAR | 没有打分头方法的干净 LiDAR 开关；LTF 83.8 对 TransFuser 84.0；榜首四个纯相机 | – | 不是差距来源 |
| 合成数据（SimScale、DriveZero-Scale、RAP 恢复扰动） | DriveZero 94.8 → Scale 95.3（navhard 51.5 → 57.1）；SimScale（EPDMS）navtest +1.7 到 +2.9、navhard +5 到 +6；RAP 恢复扰动 v1 0.0、navhard +4.4 | 主要 navhard | navtest 上不值；navhard 值得但要训基座 |
| 世界模型评估轨迹（WoTE） | TransFuser 81.0 → 无未来状态的评估器 83.2 → 带预测 BEV 85.6（2504.01941 Tab3） | DAC、TTC | 与我们 JEPA 主线相关，但是中期后的事 |

小结：pack 能拿的是第 1、3 行和一部分第 2 行的效果（我们的 `temporal` 已经是强表征，E6 说明它在打分头上与 TransFuser 同档）；RL 和合成数据要动基座；backbone 尺度不在 pack 范围。

## 2. WOD-E2E 与 Bench2Drive

**RFS 的形状**（WOD-E2E，arXiv 2510.26125 §3）：每帧 3 条 rater 轨迹，3 s / 5 s 各有随初速缩放的信任域（3 s：横 1.0 m、纵 4.0 m；5 s：1.8 / 7.2 m），在域内拿该 rater 分，域外指数衰减、下限 4。
rater 标签**只有 val 479 帧有**，train 只有 log 轨迹，test 隐藏。所以能直接对准 RFS 的训练信号只有 val 那 479 帧——这正是榜首的做法。

| 条目 | 基础 → 最终 | 增益 | 做法（出处） |
|:--|:--|:--|:--|
| Poutine | test 7.91（SFT）→ 7.99（+GRPO） | RL +0.08 | GRPO，奖励 = RFS，在 < 500 个 val 偏好帧上（2506.11234） |
| DriveMA | 7.741 → 7.893（元动作 SFT）→ 8.060（回合级 GRPO，2B） | 元动作 +0.15，RL +0.17 | RL 用 val 的 479 个 rater 帧（2605.31271 Tab3，split 未写明） |
| NTR | test 单模型 7.998 → 集成 8.046 | 集成 +0.05 | 2605.31116 Tab1 |
| RAP | test 8.04 | – | 两个 checkpoint 的 NMS 集成（2510.04333 附录 A.1） |
| **openpilot Cinque（我们，val）** | 8.005 | – | zero-shot |

我们在 val 479 帧上的 CPU 后处理试算（box 上已有的预测，官方 RFS 移植，95% CI 为 token bootstrap；[wod_val_posthoc.txt](results/skill-pack/wod_val_posthoc.txt)）：

| 行 | RFS（cluster 均） | 压到下限的帧 | 对 Cinque 的配对差 |
|:--|--:|--:|:--|
| Cinque 原生 | 8.005 | 10.0% | – |
| Lebowski 原生 | 7.886 | 12.5% | −0.14 [−0.27, −0.00] |
| 两者轨迹平均 | 7.969 | 10.4% | −0.04 [−0.12, +0.04] |
| **Cinque 纵向 ×s，s 按序列两折交叉拟合（两折都选 1.06）** | **8.119** | 9.0% | **+0.12 [+0.05, +0.20]** |
| oracle：逐帧取 {Cinque, Lebowski} 较好者 | 8.308 | 7.9% | +0.30 |
| oracle：逐帧取纵向 ×{0.80…1.20} 最好者 | 8.691 | 4.4% | +0.71 |
| oracle：逐帧取 {Cinque, Lebowski, cv, cls ego} 最好者 | 8.814 | 2.9% | +0.83 |

读法：openpilot 的 plan 在两张榜上都系统性偏慢（NAVSIM 4 s 短 3.4 m、WOD 最优纵向系数 1.06），一个全局纵向校准在 WOD 上就值 +0.12，与 DriveMA 的 RL 同量级；
逐帧选速度的 oracle 空间 +0.71，但只有 479 个带标签帧，学一个逐帧选择器很容易过拟合。两模型平均没有用（NTR 的集成 +0.05 也小）。
s 在 val 上选、在 val 上报，靠交叉拟合避免同帧泄漏；搬到 test 时 s 固定为 1.06，**test 与 val 不配对，增益可能不同**。

**Bench2Drive**：闭环榜的 pack 是控制接口与规则，不是打分头。第 31 条量到同一 TFv6 checkpoint 换 route + target speed 通道 +14.3 DS、规则约 1 DS；
LEAD（2512.20563）里 expert 可模仿化 +1.4 DS、去 GRU +2.3、多 target point +2.0，TFv6 纯相机 91.6、加 LiDAR 与雷达 95.0–95.2；BridgeDrive 96.34 只在 LEAD 数据上训成立（PDM-Lite 数据 87.99）。
openpilot 在 B2D 上连起步都有问题（standstill prior，[closedloop integration](openpilot-closedloop-integration.md)），中期前 B2D 不在 pack 的范围，只作闭环 lane 的事。

## 3. pack 的组件（按 openpilot 上的预期增益排序）

「前端」= 输入契约适配（让基座看到它训练时的输入）；「后端」= 在基座输出之后、按榜单 metric 选择或校准轨迹。所有组件都不改 openpilot 权重。

| # | 组件 | 端 | 用 openpilot 的什么 | 需要什么 | 预期增益（依据） | 能否搬到别的基座 |
|:--|:--|:--|:--|:--|:--|:--|
| F0 | 输入契约：2 Hz → 20 Hz 补帧（GIMM-VFI）、WOD 10 s × 10 Hz 窗口、5 s @ 4 Hz 输出格式 | 前 | 原生输入协议 | 已做 | NAVSIM hold 52.1 → **84.2**（+32，已测）；WOD 已在契约内 | 只对要高帧率的基座有意义 |
| B1 | **切换器：原生 plan ⊕ Hydra 打分头的选择**（打分头预测的子分给两者打分，取高者；参数 1–3 个） | 后 | 原生 plan、`temporal` | navtrain 子集上逐 anchor 子分标签（CPU）、原生 plan 在同一子集的 devkit 分（导航 lane 的 `none` 臂） | oracle 91.2；学到 1/4–1/2 → **+2 到 +4**（推测） | 打分头部分可以 |
| B2 | **纵向（速度）校准**：原生 plan 纵向 ×s 作为额外候选（NAVSIM 由打分头选 s，WOD 用全局 s） | 后 | 原生 plan 的系统性偏慢 | NAVSIM：navtrain 子集上 CPU 打分；WOD：val 交叉拟合 | WOD **+0.12**（已测，交叉拟合）；NAVSIM EP 丢 8.7 分，预期 +1 到 +3（推测） | 纯 openpilot 特性 |
| B3 | **打分头在补帧输入下重训，原生 plan 进候选集**（Hydra 式五个子分头 + 以原生 plan 为先验的模仿项，替代 E6 的 `cls_late` 模仿项） | 后 | GIMM 输入下的 `temporal`、原生 plan | navtrain 2 万 token 的 GIMM 补帧 + openpilot 重抽特征（GPU）；E6 的逐 anchor 标签复用 | E6 在 hold 特征上 +6.3；补帧特征更好、先验更强 → navtest **87–90**（推测） | 结构可以，先验项是 openpilot 的 |
| B4 | command 条件：command 作打分头输入；若导航 lane 的 A* 过线，A* 的 plan 进候选 | 前 + 后 | desire | 由导航 lane 负责 desire 部分，pack 只加打分头输入 | 转弯 PDMS 74–78 对直行 88；上限几分，但三次读数都没用（midterm-gaps 3.3） | 可以 |
| B5 | 两模型集成（Cinque + Lebowski 打分头 logit 平均） | 后 | 两个 openpilot | 多训一个头（CPU 分钟级） | Hydra-MDP-C +0.7；WOD 轨迹平均 −0.04 → NAVSIM **≤ +0.7**，WOD 不做 | 可以 |
| B6 | WOD 逐帧速度 / 模型选择器（val 479 帧交叉拟合） | 后 | 原生 plan 的纵向缩放族 | val rater 帧 | oracle +0.3 到 +0.7；可学部分预期 +0.05 左右，过拟合风险大 | 可以 |
| — | RL 微调 plan 头（奖励 = PDMS / RFS） | 基座 | plan 头 | 训练代码（op-adapt 的 port 可用） | 文献 +4 到 +9 PDMS、+0.08 到 +0.17 RFS | 改了基座，不算 pack；中期后 |
| — | oracle best-of-N、PDM-Closed 兜底 | – | – | 真值 | 不合规 | 不做 |

排除的理由写清楚：oracle best-of-N 用评测真值选轨迹；PDM-Closed 需要真值地图与感知，NAVSIM agent 的输入里没有；RL 改了 plan 头，已不是「同一个基座」。

## 4. 合法适配 vs 刷榜，以及怎么诚实报告

判线：**只用榜单公开给参赛者的训练 / 验证数据拟合、测试集只打一次分、输入不超出榜单契约**，就是合法适配；这与榜首的做法一致（Hydra 系在 navtrain 上蒸馏 PDM 子分；Poutine / DriveMA 在 WOD val 的 rater 帧上用 RFS 做 RL）。
越线的是：用测试集真值选轨迹（oracle best-of-N）、在测试集上挑臂或多次提交取最好（WOD 每 30 天 6 次配额，我们只交预登记的一个）、利用计分 bug（NAVSIM Issue #172 子分全 1 得 0 分那类）。

但「合法」不等于「是能力」。按第 35 条的分层，B1–B3 学的标签就是评测用的 scorer，是 metric 对准，属于 R 层配方。所以报告要两栏并列：

| 栏 | 内容 | 预期 |
|:--|:--|:--|
| 分数 | 每张榜：基座（F0 only）、基座 + pack，配对 Δ 与 CI；标注 pack 参数量（5 个子分头 × 1024 × 544 ≈ 2.8M）与拟合用的数据 | pack 涨分 |
| 行为变化 | pack 输出与原生 plan 的差：4 s 终点偏移 > 0.5 m 的 token 比例、平均纵向 / 横向差、EP 是否超过 human | 大量 token 纵向变长 |
| 能力读数 | 同一 pack 在配对考卷上的反应：P3 真实外观行人考卷（WOD + OmniRe，[p3-exam-filter.md](p3-exam-filter.md)）的翻转率，NAVSIM 走廊有行人 / cyclist 的 897 个 token 的 PDMS Δ | 不变（等价带内） |

第 53 条的教训要照顾：NAVSIM 训的打分头搬到 CARLA 的 5 Hz 帧上兼容检查不过（top-10 anchor 重叠 0–20%），所以能力读数**只用真实数据的配对考卷**（P3、WOD），不用 P5 / I3。
WOD 的 pack（B2 全局 s）在 P3 上能原样跑；NAVSIM 的 pack 在 P3 上不能跑（输入契约不同），只报 NAVSIM 内的行人 token 子集，并写明这是弱读数。
「pack 动分数不动能力」这句话要预登记成可证伪的：若 P3 翻转率在 pack 前后差超出 ±5 pp（等价带，写死），这句话就不成立，照实写。

## 5. 预登记计划

原则：选臂只在 navtrain（NAVSIM）或 val 交叉拟合（WOD）上做；navtest / navhard / WOD test 各只打一次；所有臂和判线在看任何新数字之前写进 todo。

### 5.1 与导航 lane 的分工

导航 lane（[todos/2026-09-29-op-leaderboard.md](../todos/2026-09-29-op-leaderboard.md)）负责 desire 时刻的臂、GIMM 缓存（navtest / navhard / navtrain 子集）与 WOD 提交包。pack **不碰 desire**，不重跑 GIMM：
直接读 lane 的 `lb_navtest/gimm.npy`、`lb_navtrain/{gimm,plans}` 与 navtrain 上 `none` 臂的 devkit 分；若 lane 的 A* 过线，A* 的 plan 作为 B1 的额外候选。
GPU 步骤（N1）排在 lane 的 navtest / navhard 跑完之后，避免抢卡。

### 5.2 实验（优先级顺序）

| ID | 内容 | 调参 / 测试 | 成本 | 判线（写死） | 最早完成 |
|:--|:--|:--|:--|:--|:--|
| **N0** | B1 + B2：在 lane 的 navtrain 子集（3 000 token，**去掉与 E6 打分头训练集重叠的 token**）上，原生 plan 与其纵向 ×{1.00, 1.05, 1.10, 1.15} 作候选，E6 的 hold 特征打分头给候选与 1 024 个 anchor 打分；切换规则 = 打分头对 native 候选的最近 anchor 的加权分 + 边际 δ，δ 与 s 集在 navtrain 上选 | navtrain 子集选 → navtest 一次 | CPU：逐 anchor 子分标签 ~3 000 × 7 core·s ≈ 6 core·h（24 核 15 min）+ 4 个纵向候选的 devkit 打分 ~1 h + 拟合分钟级；**0 GPU** | navtest 配对 Δ（对原生 plan 84.2）CI 下界 > 0 → 「切换器成立」；另报 left / right / straight 与行人 token 子集 | 10-01 |
| W1 | B2 on WOD：s = 1.06 固定，重写 test 提交包 `cinque_lon1.06`；在 P3 上跑 s = 1.00 / 1.06 的能力读数 | 已交叉拟合 | CPU 分钟级 | test 提交**由用户决定**（占配额 1 次）；P3 翻转率差在 ±5 pp 内 → 「不动能力」 | 10-01 |
| N1 | B3：在 E6 的 2 万个 navtrain token 上 GIMM 补帧 + openpilot 重抽 `temporal`，navtest 用 lane 的缓存重抽；重训五个子分头，模仿项换成「到原生 plan 的距离」，原生 plan 与纵向候选并进候选集；Cinque 与 Lebowski 各一 | navtrain held-out log 选权重 → navtest、navhard 各一次 | GIMM ~12 GPU·h + openpilot 抽特征 ~1.5 GPU·h + 头训练 CPU；**约 14 GPU·h，2 卡 7 h 墙钟**。省钱版：navtrain 用 ego-motion warp（CPU）补帧，~1.5 GPU·h，代价是训练 / 测试补帧器不一致 | navtest PDMS 对 N0 的配对 Δ CI 下界 > 0；navhard 只报点估计 | 10-03 |
| N2 | B5：Cinque + Lebowski 打分头集成 | 同上 | CPU | 报点估计与配对 Δ，不单独判 | 10-04 |
| C1 | 能力读数：NAVSIM 行人 / cyclist 走廊 897 token 的 Δ、pack 前后终点偏移分布、EP 超 human 的 token 比例；WOD P3 | – | CPU | 见第 4 节 | 10-05 |
| W2 | B6（可选）：WOD val 上逐帧速度选择器，两折交叉拟合 | val 交叉拟合 | CPU / 小 GPU | 交叉拟合 Δ 对 W1 CI 下界 > 0 才进 test 包 | 中期后 |

分阶段启动照 CLAUDE.md：N1 先在 200 个 navtrain token 上跑通补帧 + 特征 + 打分，与 E6 的 hold 特征逐 token 对账（维度、NaN、分布），再上 2 万。

### 5.3 中期前能拿到的故事

| 榜 | 基座（F0） | + pack（预期，推测） | 位置（对照 [tmp 表](../tmp/2026-09-29-leaderboard-wod.md)） |
|:--|:--|:--|:--|
| NAVSIM v1 navtest | 84.2 | N0 84.9（已测）；**N1 87.3（已测，[+1.94, +2.83] 对 N0）** | 88–90 约在 DiffusionDriveV2 / GoalFlow / Hydra-MDP++ 档，前 15 左右；前 5（≥ 93.5）够不着 |
| NAVSIM v2 navhard | 33.3 | 36–42（E6 在 hold 特征上 25.7，补帧 + 原生 plan 应更高，推测） | SimWAM 37.6 到 GTRS-E 49.4 之间 |
| WOD-E2E | val 8.005 | val 8.119（已测，交叉拟合）；test 未知 | 若 test 同量级，在 8.08–8.17 那一簇，前 10 |
| Bench2Drive | – | 不在中期范围 | – |

故事一句话：**同一个不改权重的 openpilot，每张榜配一个几 M 参数的 pack（前端补输入契约，后端对准该榜 metric 做选择与速度校准），NAVSIM 从 52 → 84 → 85 → 87（N0、N1 已测）、WOD 从 8.00 → 8.12；
pack 带来的分数主要落在 EP / 进度和信任域上，而真实外观配对考卷上的反应不变——分数可以用配方买，能力要看基座。**
这正好把第 35 条（高分主要是 R 层配方）从「读别人的论文」变成「在自己的基座上做一遍」，是 trick or trade 论点的一个正面、可控的证据。
若 C1 显示 pack 改变了行人反应（任一方向），这句话的后半句要改写，且那本身是一个更有意思的结果。

## 6. N0 结果（2026-09-29）

预登记 [todos/2026-09-29-skill-pack-n0.md](../todos/2026-09-29-skill-pack-n0.md)，小表 [results/skill-pack/n0/](results/skill-pack/n0/)，决策第 64 条。CPU only（13 核，1.5 h，最慢的是在 CPU 上重拟合模仿项）。

| 读数 | 值 |
|:--|:--|
| T（navtrain 子集去掉 E6 训练 token，n = 2 428）：原生 / 原生 ×1.05 / ×1.10 / ×1.15 / Hydra / oracle | 81.94 / 81.93 / 81.95 / 81.92 / 85.99 / 92.95 |
| T 上选中 | s = 1.10、δ = 0.5 → 86.86（39.5% 输出原生） |
| navtest N0 / 原生 / E6 Hydra | **84.94** / 84.17 / 84.18 |
| N0 − 原生，N0 − Hydra（配对，95% CI） | **+0.77 [+0.28, +1.27]**，+0.75 [+0.49, +1.02] |
| 按 command，N0 − 原生 | 左 +1.5、右 +2.3、起步 +5.9、直行 −0.8 |
| N0 的 EP 超过 human 的 token | 19.8% |

读法：切换器从 oracle 的 7 分空间里只拿到约 1 分，瓶颈在打分头的排序（hold 特征上的线性头）。第 3 节表里 B1 的「+2 到 +4」与 B2 的「+1 到 +3」都偏乐观：
B2 单独在 NAVSIM 上是零（PDMS 的 DAC / TTC 会罚快），只有经打分头门控才贡献；WOD 的 RFS 不罚快，所以同一个速度校准在 WOD 上单独有 +0.12。
增益集中在转弯和起步，直行掉 0.8，是「用安全裕度换进度」的形状，符合第 4 节对 metric 对准的预期。中期前 NAVSIM 的现实数字是 85 上下，N1 决定能不能到 87 以上。

## 6.1 N1 结果（2026-09-29）

预登记 [todos/2026-09-29-n1-scorer.md](../todos/2026-09-29-n1-scorer.md)，小表 [results/skill-pack/n1/](results/skill-pack/n1/)，决策第 68 条。GPU 6 共卡，约 5 GPU·h。

| 读数 | 值 |
|:--|:--|
| navtest N1 / N0 / 原生 / E6 Hydra | **87.32** / 84.94 / 84.17 / 84.18 |
| N1 − N0，N1 − 原生（配对，95% CI） | **+2.38 [+1.94, +2.83]**，+3.15 [+2.79, +3.52] |
| 子分 N1（对 N0） | NC 98.58（+0.4）、DAC 96.17（+2.0）、EP 79.42（+2.2）、TTC 95.64（+0.8） |
| 按 command N1 / N0 | 直行 89.56 / 87.09，左 81.22 / 78.98，右 78.60 / 75.93，起步 93.86 / 91.97 |
| 输出原生槽的 token / EP 超过 human 的 token | 66.3% / 23.0% |
| held-out logs 消融 A0 / A1（hold 特征）/ A2（无原生槽）/ A3（借最近 anchor） | 89.46 / 89.21 / 87.53 / 87.82 |

读法：N0 只拿到 oracle 空间的约 1 分，N1 拿到约 3.2 分（对原生），瓶颈是候选集与打分方式，不是特征分布：补帧特征 +0.25，原生 plan 进候选集 +1.93，独立原生槽相对借最近 anchor +1.64。
直行不再掉分。第 5.3 节原先的 87–90 现在 87.3 落在下沿，往 90 走需要更好的头，不是更多补帧。
局限：只有 Cinque、1 seed、navtest 一次；全量拟合的十个头 λ 全在网格下沿 1e-5（网格偏紧，可能还有余量，没有重跑）；navhard 未测。

## 7. 开放问题与风险

1. （N1 已答，第 6.1 节）N0 的切换器用 hold 特征的头借最近 anchor 评原生 plan；N1 干净版本 +2.38，增益来自独立原生槽而非补帧特征。遗留：λ 网格下沿、navhard 与 Lebowski 未测。
2. 纵向拉长在 NAVSIM 上可能用 NC / TTC 换 EP；E6 的打分头 EP 权重选到 2，是五个权重里最大的，说明 metric 本身鼓励这个方向。EP 超过 human 的比例要单列。
3. WOD 的 s = 1.06 是在 val 上拟合的；test 片段不同。test 上若不涨，诚实地写「val 交叉拟合 +0.12，test 未复现」。
4. 导航 lane 若 A* 不过线，转弯的 DAC / EP 损失（左右转 PDMS 74–78）大部分留在那里，pack 能补的主要是直行 EP；N0 按 command 分组报，别让平均数掩盖。
