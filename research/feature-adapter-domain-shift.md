# 特征适配器（「特征配方」）能不能解决画风漂移：文献与我们自己的证据

2026-09-29。文献调研加上 decisions 已有读数的重新解读，没有跑 GPU。接 [midterm-gaps.md](midterm-gaps.md) 缺口二（验收原则没有正例）的 G-a 假设与候选修复 F5 / F6，
读过 decisions 第 36、40、42、44、48、53、55、56、60 条和 [op-adapt todo](../todos/2026-09-28-op-adapt.md)，排期约束见 [中期排期](../tmp/2026-09-29-midterm-plan.md)。

## 0. 结论先行

用户的想法：把 openpilot `temporal`（Cinque policy 输出的 512 维时序 token）/ vision token 或 V-JEPA 2 特征这类中间时空特征拿出来，在下游 head 前面放一个小的、按域调的 feature adapter
（feature adapter：插在冻结 backbone 某一层后面的小网络或逐通道仿射，只训它，用来把一个域的特征挪到另一个域的分布上），让一个模型适配多个域。

| 问题 | 判断 | 依据 |
|:--|:--|:--|
| 文献里这类方法能修什么 | 能修**低层外观**（纹理、颜色、光照、渲染风格）引起的移位，前提是任务信息在 adapter 的输入里还在；修不了内容差（场景、物体分布）、条件分布差和信息已丢失的情况 | IBN-Net、AdaIN、DSBN、surgical fine-tuning、Zhao et al. 2019 的反例、Meta-Sim 的 content gap（第 1 节） |
| 放在 `temporal` 上能不能修 CARLA 行人 | **不能**。CARLA 帧上 `temporal` 和 `vision` 的行人线性 AUC 都是 0.51，信息在 adapter 的输入之前就没了；在这一层对齐分布，最多让 head 不再读错方向（第 44 条标准化到 WOD 统计量只把损害减半，已经测过） | 第 42 条 D0、第 44 条 E1 |
| 放在更浅的层能不能修 | **可能**，取决于 CARLA 行人信息在哪一层消失。真实 nuScenes 上 stage 3 输出线性 AUC 0.83，CARLA 上 stage 3 从没测过；这是整件事唯一的开关，而且测它几乎不要 GPU（P5 的 trunk 缓存已在盘上） | 第 55 条 trunk 诊断；本文 E0 |
| 开环 / 闭环输入契约不同（2 Hz 对 20 Hz） | **不是 adapter 能修的**：缺的是时间轴上的输入，recurrent policy 看不到就没有；修法在输入端（补帧）或按契约重拟合读出头 | 第 36、37、53 条；[openpilot-openloop-integration.md](openpilot-openloop-integration.md) |
| V-JEPA 2 特征 | 情况不同：CARLA 上 V-JEPA 2 冻结特征里有行人信息（配对差分 47%），但 CARLA 训的 Δ 上真实数据仍有害（Q6）。这里是 head 学了 CARLA 特有方向，训练时先把 CARLA 特征映射成「真实感」特征再训 head，文献上有先例（CyCADA、Bewley 2019），但直接用 Cosmos 帧的特征训 head 是它的上界、更简单 | 第 48 条、第 44 条 Q6 |
| Cosmos G4 配对的价值 | 很高：它给了「同一世界、CARLA 画面 ↔ 真实感画面、行人区外逐像素对齐」的四元组（CARLA x⁺ / x⁻，Cosmos x⁺ / x⁻），既能**监督**一个 CARLA→真实感的特征 adapter，也能**分解**域差里多少是画风 | 第 56 条、`jevdrive.cosmos_full.load_pair` |
| 最有价值的用法 | **不是**让真实部署变好，而是**闭环时替代 Cosmos**：Cosmos 每 clip 约 90–220 s，进不了闭环；一个在 G4 配对上训的 CARLA→Cosmos 特征 adapter 可以实时插在 CARLA 闭环里的 openpilot 前段，让适配后的模型「看到」的 CARLA 接近真实感画面。这直接对应缺口一的 H2（渲染域差） | 本文第 5 节 |

**总判断**：想法不是不能做，但要把期望放对。「一个 adapter 修好所有域差」在我们这里不成立：输入契约差不归它管，内容差（CARLA 的行人外观、步态、场景布局本身）它也不管，
它只能修画风，而且只能放在信息还在的那一层。先用两个几乎免费的读数（E0 定位、E1 分解）判断画风占多大份额，再决定要不要训 adapter（E2）。
我的主观估计：E0 显示 CARLA 行人信息在 stage 3 仍在（≥ 0.65）的概率约 40%；在那种情况下 E2 让原始 CARLA 行人 AUC 过 0.60 的概率约一半。整体约 20–25% 的机会拿到「一个 stage 3 adapter 把原始 CARLA 行人读出来」的结果。

---

## 1. 文献：特征层面的域适配 / 域泛化，每一族修什么、要什么、会怎么坏

下表按「移位发生在哪」分族。「要什么」一列里：配对数据 = 两个域同一内容的对应样本；目标标签 = 目标域的任务标注；无标注目标 = 只要目标域的原始输入。

| 族 | 代表工作 | 修什么移位 | 要什么 | 报告的收益 | 已知失效方式 |
|:--|:--|:--|:--|:--|:--|
| 对抗式特征对齐 | DANN（Ganin & Lempitsky, ICML 2015；JMLR 2016）、ADDA（Tzeng et al., CVPR 2017）；输出空间版 AdaptSegNet（Tsai et al., CVPR 2018） | 边缘分布 p(z) 的整体移位 | 无标注目标 + 源标签 | GTA5→Cityscapes 分割，DeepLab-v2 R101 source-only 36.6 → AdaptSegNet 42.4 mIoU | 只对齐边缘分布；类别比例或条件分布不同时，对齐会把不同语义挤到一起，目标误差反而上升（Zhao et al., ICML 2019 给出下界；Ben-David et al., Machine Learning 2010 的理论里联合误差项不受对齐控制） |
| 矩匹配 | DAN / MMD（Long et al., ICML 2015）、Deep CORAL（Sun & Saenko, ECCV-W 2016） | 一阶、二阶统计量的移位 | 无标注目标 | 小到中等 | 同上；只对齐矩，方向性的「伪特征」对齐不掉 |
| 统计量 / 风格归一化 | AdaBN（Li et al., ICLR-W 2017 / Pattern Recognition 2018）、AdaIN（Huang & Belongie, ICCV 2017）、IBN-Net（Pan et al., ECCV 2018）、RobustNet ISW（Choi et al., CVPR 2021）、MixStyle（Zhou et al., ICLR 2021） | **低层外观**：IBN-Net 的核心观察是外观差异主要编码在浅层特征图的逐通道均值 / 方差里，内容差异在深层 | AdaBN 要无标注目标；IN / whitening / MixStyle 只在源上训，零目标 | 域泛化 GTA→Cityscapes（R50）约 29 → 36–37 mIoU 这一量级 | 只动风格统计量；深层的内容差、物体外观本身（不是颜色而是形状、步态）不在它的作用范围。AdaBN 依赖 BN 的 running statistics，**openpilot 的 ConvNeXt 是 LayerNorm，没有可重估的 running stats**，只能换成显式的逐通道仿射 |
| 按域的小参数 | 残差 adapter（Rebuffi et al., NeurIPS 2017）、DSBN（Chang et al., CVPR 2019）、LoRA（Hu et al., ICLR 2022）、VPT（Jia et al., ECCV 2022）、Rein（Wei et al., CVPR 2024） | 「一个 backbone，多个域 / 任务，每个域一小组参数」，正是用户说的「特征配方」 | 每个域要该域的标签（Rebuffi、DSBN 的有监督部分），或源标签 + 强 backbone（Rein） | Rein：冻结 DINOv2 类 VFM + 约 1% 可训参数，只用合成数据训，Cityscapes 68.1 mIoU（无真实数据） | adapter 的容量有限，前提是 backbone 本身对目标域的表征已经够好；Rein 的收益主要来自 VFM 本身的鲁棒性，adapter 是把它用出来 |
| 哪一层该调 | Surgical fine-tuning（Lee et al., ICLR 2023） | 按移位类型选层：输入层面的移位（corruption、外观）调**最前几层**最好；输出层面的移位（标签含义变）调最后几层 | 少量目标标签 | 七个数据集三类移位上，选对层的部分微调追平或超过全量微调 | 选错层几乎无效；这是本文第 3 节「adapter 放哪」的直接依据 |
| 测试时适配 | TENT（Wang et al., ICLR 2021）、TTT（Sun et al., ICML 2020）、CoTTA（Wang et al., CVPR 2022）、回归版 SSA（Adachi et al., ICLR 2025） | 部署时的温和移位（corruption、天气） | 只要测试流 | 在 corruption 基准上明显降错误率 | TENT 以熵最小化为目标，只适用分类；我们的 plan 是回归 / MDN，要 SSA 这类回归专用方法；长时间在线适配有塌缩问题（CoTTA 和后续 continual TTA 工作都在处理这个）；对内容差无效 |
| 像素级 sim2real + 特征一致 | CyCADA（Hoffman et al., ICML 2018）、EPE（Richter et al., TPAMI 2022）、Bewley et al.（ICRA 2019） | 画风，用图像翻译补，再用特征 / 语义一致性约束内容不变 | 无标注目标（非配对） | Bewley：只用仿真控制标签，在真实乡村道路上做到单目车道保持，零真实标签 | 翻译模型可能改内容（我们的 Cosmos v1 就是：配对的两边在行人以外也不同，第 56 条）；翻译器本身的伪影可成为捷径 |
| 配对的特征对齐 | Tzeng et al.（WAFR 2016，弱配对 sim / real 图像上的 pairwise loss + domain confusion）、RCAN（James et al., CVPR 2019，把随机化仿真与真实都映射到规范仿真画面） | 画风，靠「同一内容两种外观」的对应样本直接回归 | **配对数据** | Tzeng：弱配对比只用分布对齐在 PR2 真机任务上更好；RCAN：抓取任务少量或零真实数据迁移 | 配对的质量决定上限；弱配对本身带内容差 |
| 中间表示当迁移接口 | Müller et al.（CoRL 2018，语义分割作接口，1/5 缩比卡车两大洲零微调）、Pan et al.（BMVC 2017）、DU-Drive（Yang et al., ECCV 2018，真实转到仿真的统一域） | 用抽象表示（分割、BEV、检测）把画风整个挡在感知模块里 | 感知模块在目标域上要好 | Müller：零微调迁移到真车 | 抽象表示丢掉的东西（语义类别之外的线索）策略就用不上；感知模块自己的域差仍在 |
| 内容差本身 | Meta-Sim（Kar et al., ICCV 2019）；合成行人检测的经典失败（Vázquez et al., TPAMI 2014） | 场景布局、物体分布、物体外观（不是画风） | 目标域数据来拟合分布 | — | 这类差距**任何特征风格对齐都修不了**；Vázquez 的结论正是「虚拟行人训的检测器到真实上要做域适配」，差距部分在外观本身 |
| 基础模型特征 | DINOv2（Oquab et al., TMLR 2024）、V-JEPA 2（Assran et al., 2025, arXiv 2506.09985）；驾驶里 Drive Anywhere（Wang et al., ICRA 2024）、Mallak & Maalouf（arXiv 2602.09018, 2026） | 冻结 VFM 特征对外观移位天然更稳 | 大规模预训练 | Mallak & Maalouf 在 VISTA 闭环里：冻结 FM 特征的策略在三个因素同时变化时仍 > 85%，从零训的 < 50%；最伤的是乡村→城市、白天→夜晚（各约 −31%），不是天气 | VISTA 是由真实数据重建的仿真，「外观移位」比 CARLA↔真实温和得多；这类结果说明「好 backbone 比 adapter 重要」，不说明 adapter 能补一个已经丢了信息的 backbone |
| 近两年端到端驾驶 sim2real | SimScale（Tian et al., CVPR 2026）：真实 log 的神经渲染仿真 + 伪专家，sim-real co-training，navhard +8.6、navtest +2.9 EPDMS | 状态分布（离开人类轨迹的状态），画风差由「从真实 log 重建」按构造压小 | 真实 log + 渲染器 | 见左 | 不涉及 CARLA 式的画风差；它的收益在状态覆盖，不在外观 |

读法：凡是报告大收益的特征级方法，要么移位本质上是低层外观、任务信息在对齐层还在（IBN、Rein、surgical 的 corruption 设定），要么有目标域标签或配对数据。
「无标签、内容也不同、信息在对齐层之前已丢」这一格，文献里没有正例，而且有理论反例（Zhao et al. 2019）。

### 1.1 VLA 的适配配方：OpenVLA 那条线说明了什么（用户补充的比较）

| 工作 | 结构 | 适配方式与读数 | 每个新域要多少目标域数据 |
|:--|:--|:--|:--|
| OpenVLA（Kim et al., CoRL 2024, arXiv 2406.09246） | Prismatic VLM：DINOv2 + SigLIP 融合的视觉编码器 + Llama 2 7B；Open X-Embodiment 的 97 万条真实示教 | ① 预训练阶段：**微调视觉编码器对 VLA 至关重要**（论文原话：VLM 文献里冻结视觉编码器通常更好，但 VLA 训练里微调它是关键，推测冻结特征缺细粒度空间信息）。② 新机器人上的参数高效适配（成功率）：全量 69.7%，**LoRA r = 32 68.2%**（只训 1.4% 参数），sandwich（视觉编码器 + 最后一层 + token embedding）62.1%，**冻结视觉 47.0%**，只调最后一层 30.3% | 每个目标任务 10–150 条示教（Franka-Tabletop、Franka-DROID） |
| OpenVLA-OFT（Kim, Finn & Liang, RSS 2025, arXiv 2502.19645） | 同 backbone；并行解码 + action chunk + 连续动作 + L1 回归，ALOHA 上加 FiLM | 仍用 LoRA（理由：示教 500 条对预训练 100 万条）；LIBERO 四个 suite 平均 76.5% → 97.1% | LIBERO 每个 suite 500 条（10 个任务）；ALOHA 每个任务 20–300 条 |
| π0（Black et al., 2024, arXiv 2410.24164） | PaliGemma VLM + flow matching 动作专家 | 预训练 + 在高质量任务数据上 post-training | 要目标任务数据 |
| RT-2（Zitkovich et al., CoRL 2023） | VLM 直接出动作 token | 网页数据与机器人数据 co-fine-tuning，整个模型一起训 | 目标机器人示教 |
| SimplerEnv（Li et al., CoRL 2024, arXiv 2405.05941） | 评测：真实数据训的 RT-1 / Octo 等放进仿真 | 仿真画面和真实画面的差会让读数失真；修法是 **visual matching**（把真实背景贴进仿真、调物体纹理，像素层面）加控制的 SysID | 不训练 |

对我们的三个回答：

1. **op-adapt B / B-LoRA / C 就是 OpenVLA 配方在驾驶上的对应物**：冻结大部分、解冻视觉后段或加 LoRA、保原能力（我们用蒸馏，OpenVLA 靠小学习率与 LoRA 的低秩约束）。
   OpenVLA 的表给出两个与我们一致的信号：冻结视觉只调后面（我们「冻结特征 + 薄 head」）明显差于调视觉（47.0 对 69.7）；LoRA 约等于全量（我们测得 B-LoRA 成本 0.06 GPU·h / 10 万样本，对 B 的 0.07 几乎一样便宜）。
   它**没有**测「只调视觉最后一段」这一格（sandwich 同时调了整个视觉编码器），所以不能直接说我们的 stage 4 解冻够不够；我们第 55 条的读数是：stage 4 解冻补回了 stage 3 → `temporal` 落差的约六成。
2. **每个新域要的数据**：OpenVLA 系列所有适配读数都用**目标域的带标签示教**，每任务 10–300 条；它的「视觉泛化」评测（没见过的背景、干扰物、物体颜色）是在真实照片域内、以 97 万条真实示教预训练为前提的。
   没有一个读数是「不给目标视觉域任何示教，只靠适配迁过去」。
3. **能不能迁到没见过的视觉域**：这条线没有正面证据；反过来，SimplerEnv 说明真实数据训的 VLA 放进仿真画面会失真，他们修在**像素层面**（visual matching），不是特征 adapter。
   我们的第 55 条是同一现象的驾驶版：只用真实 nuScenes 监督调 stage 4，原始 CARLA 行人 AUC 只动了 +0.009。按 VLA 的经验，第二轮 B 加了 Cosmos 配对，预期提升 **Cosmos 画面**上的读数，而**原始 CARLA** 仍是没见过的视觉域，不会跟着涨（即 midterm-gaps 2.3 的结局 B）。
   如果验收原则里的「sim」指原始 CARLA，按这个配方就得把**原始 CARLA 帧也作为一个带标签的域放进训练**（G4 的每一对都存了 `carla_plus.mp4` / `carla_minus.mp4`，数据现成），或者给原始 CARLA 单独一个小的按域 adapter（DSBN / residual adapter 式）。

---

## 2. 我们的证据：CARLA 与真实之间的差到底在哪

### 2.1 openpilot Cinque 的层次和已知读数

Cinque 的前向（op-adapt 执行日志读 ONNX 所得）：图像 → ConvNeXt stem → stage 1（256 通道）→ stage 2（512）→ stage 3（1024，27 个 block；输出 `permute_73`，1024 × 8 × 16，即 op-adapt 的 trunk 缓存）
→ stage 4（2048）→ 1×1 投影到 32 × 512 的 hidden token（`view_39`；`vision` tap 是它的池化）→ policy：当前 + 前 8 个 5 Hz hidden（1.6 s）的 4 层 transformer → `select_4`（= `temporal`）→ plan / lead / desire。

| 层（adapter 可以放的位置） | 真实数据上的走廊行人线性 AUC | 原始 CARLA（P5 v1 BA）上的 | 来源 |
|:--|:--|:--|:--|
| P0 像素 | — | — | Cosmos G4 在这一层做翻译 |
| P1 stage 1 / 2 输出 | 未测 | 未测 | — |
| P2 stage 3 输出（trunk） | **0.831**（nuScenes val，mean + max 池化） | **未测**（缓存已在 `processed/op_adapt/p5/`） | 第 55 条 trunk 诊断 |
| P3 hidden / `vision` | 0.702 | 0.509–0.519 | 第 55 条 (a)、第 42 条 D0 |
| P4 `temporal` | 0.709（10 m 内 0.834，> 20 m 0.564） | 0.506 | 第 55 条、第 42 条 |
| P5 head 输入 | — | — | 第 44 条 E1 在这里做过统计量对齐 |

读法：真实数据上，行人信息在 stage 3 有 0.83，stage 4 → policy 丢掉约 0.12；原始 CARLA 上从 `vision` 起就是 0.51。**CARLA 上 stage 3 那一格是空的**，而它决定了第 3 节的全部分支。

### 2.2 哪些是画风差、哪些不是

| 现象 | 性质 | 证据 | adapter 管不管 |
|:--|:--|:--|:--|
| CARLA 行人在 openpilot 里读不出来（0.51），真实里近处读得出（0.83） | **待分解**：画风（渲染材质、光照）和内容（CARLA 行人模型的外形、步态、尺度分布、出现位置）混在一起 | 第 42、55 条；同一个 openpilot 在 CARLA 上读得出车辆 cut-in（翻转 86–89%），说明不是整幅画面都失效，是**类别选择性**的 | 画风那部分，放在信息还在的那一层才管；内容那部分不管 |
| CARLA 配对激发的 Δ 在真实特征上是方向性偏置 | head 学到了 CARLA 特有方向（「CARLA 画面 vs 真实画面」而不是「行人」） | 第 44 条 E1：直行帧激活率 16.7% 不低于行人帧 14.4%；**标准化到 WOD 统计量只把损害减半**（−1.02 → −0.53）；G0 student 害小一个量级但方向同样与场景无关 | 在 head 输入上做矩对齐（CORAL / AdaBN 的对应物）**已经测过，只修了一半**。剩下的一半是 head 在 CARLA 特征上学到的方向本身；这要在训练时就让 head 看到真实感特征，不是部署时补 |
| V-JEPA 2 在 CARLA 上有行人信息，但 CARLA 训的双流 Δ 到真实上仍有害 | 信息在，读出方向是 CARLA 的 | 第 48 条（配对差分 47%）、第 44 条 Q6（WOD / NAVSIM 0 / 3 candidate seed） | 训练时的特征映射（CARLA → 真实感）有意义；但用 Cosmos 帧直接抽 V-JEPA 特征训 head 更简单，adapter 只是它的廉价近似 |
| 2 Hz 输入（NAVSIM）对 20 Hz 训练的 openpilot | **输入契约差**，时间轴上的信息缺失 | 第 36 条：2 Hz × 1.5 s 时间轴横向误差 ×9、纵向 ×25；换 rig（相机、分辨率、畸变）反而基本不掉；第 53 条：NAVSIM 训的读出搬不到 5 Hz CARLA 帧 | **不管**。recurrent policy 没收到的帧，后面的特征 adapter 造不出来；修法是补帧（[openpilot-openloop-integration.md](openpilot-openloop-integration.md)）或按契约重拟合读出 |
| 冻结特征 + 按数据集重拟合的线性读出 | 按域的读出层，最小的「特征配方」 | 第 40 条：原生 plan 在 nuScenes 上比 `ridge ego` 差 0.85–1.03 m，冻结 `temporal` + 重拟合读出把这个分布差**完全吸收**；NAVSIM 上 +9.7 | 这是 adapter 在**信息在、有目标标签**时有效的正例；但它要目标域标签，CARLA↔真实的问题恰恰是一边没有对应标签 |
| 某些 CARLA town 车道线概率 0.03–0.16 | 可能是画风（无纹理路面、夜雾），也可能是 CARLA 路面标线本身 | midterm-gaps H2 | 画风部分可以；E1 顺便分解 |
| 夜间 / 黄昏变暗、白天泛光 | 渲染故障，不是域差 | 第 60 条 | 应当在渲染端修（已修变暗），不应让 adapter 去学 |

**关键区分**：对一个已训好的 backbone，信息「在不在」是**以 adapter 的输入层为准**的。像素里当然有行人；问题是 openpilot 在 CARLA 帧上把它映到了后续层不读的方向，还是在某一层把它压掉了。
线性 AUC 0.51 说明到 `vision` 时已经线性不可读；非线性可读性没测过，但 512 维池化特征上非线性能读出而线性完全不能的情况不常见（推测，E0 顺带用 MLP probe 验证）。

---

## 3. adapter 应该放在哪一层

| 位置 | 能修什么 | 在我们这里的前提 | 监督来源 | 成本 | 评价 |
|:--|:--|:--|:--|:--|:--|
| P0 像素（Cosmos G4、EPE / CARLA2Real） | 画风，全部 | 翻译不改内容 | 已有 | 每 clip 90–220 s，进不了闭环 | 离线训练与考卷可用；第 56 条已解决「配对只差行人」 |
| P1 stage 1 / 2 后，逐通道仿射或小残差卷积 | 低层外观（IBN-Net 的「风格在浅层统计量」） | 行人信息在 stage 1 / 2 可读 | G4 配对：CARLA 帧与 Cosmos 帧的同层特征回归 | 小；前向要重跑 stage 2–4 | **如果 E0 显示 stage 3 在 CARLA 上也读不出，这是唯一可能有效的特征层位置** |
| P2 stage 3 后（trunk 输出） | 中层外观 + 部分形状 | CARLA 上 stage 3 行人可读（E0） | 同上 | 小；trunk 已缓存，adapter + stage 4 + policy 前向很便宜 | **E0 过则首选**：它和 op-adapt B 的可训练区域相邻，可以和 B 的 stage 4 共用训练管线 |
| P3 hidden token / P4 `temporal` | 只剩分布对齐 | 信息在，但 CARLA 上 0.51 | — | 最小 | **排除**：输入里已没有行人信息；第 44 条的矩对齐已在这层附近测过，只修一半 |
| P5 按域的读出头 | 读出方向、输入契约下的统计差 | 目标域标签 | 真实 GT / CARLA expert | 已在用 | 已有正例（第 40 条），但它不跨域，每个域要自己的标签 |

另一个维度是**训练时 vs 部署时**：

- **部署时（sim 一侧）**：适配后的 openpilot 在 CARLA 里跑，前面插 CARLA→真实感 adapter。修的是「CARLA 画面让模型变差」（缺口一 H2、缺口二 G-a）。这是 Cosmos 做不到（太慢）、adapter 能做到的地方。
- **训练时（为真实部署训 head / 训 B）**：把 CARLA 配对的特征先映射成真实感再训。等价于在 Cosmos 帧上训，adapter 只是更便宜；既然 G4 全量 2 000 对会有 Cosmos 帧，训练时直接用 Cosmos 帧，不需要 adapter。

---

## 4. Cosmos G4 配对作为资产

G4 全量每一对存了四段视频（`load_pair(pair, "carla")` 与 `load_pair(pair, "cosmos")`，93 帧，1280 × 704，20 Hz）：

| | 无行人 x⁻ | 有行人 x⁺ |
|:--|:--|:--|
| CARLA 原画 | C⁻ | C⁺ |
| Cosmos 重画 | K⁻（翻译一次） | K⁺（行人像素 + 24 px 外逐像素等于 K⁻） |

这是一个 2 × 2 析因设计：画风（C → K，内容不变）× 内容（⁻ → ⁺，画风不变）。它能给出三样文献里很少有的东西：

1. **配对监督**：(C⁻, K⁻) 与 (C⁺, K⁺) 在同一世界、同一 ego 轨迹上逐帧对应，比 Tzeng et al. 2016 的弱配对强，接近 RCAN 的规范化配对。一个在 stage ℓ 上回归 f(z_ℓ(C)) ≈ z_ℓ(K) 的 adapter 可以直接监督，不需要对抗训练。
2. **画风份额的量化**：在任一层，「行人可读性」在 C 与 K 两种画风下各算一次，差值就是画风造成的部分；K 上仍读不出的那部分是内容（CARLA 行人模型本身）或 Cosmos 也没画对的部分。
3. **捷径检查的内建对照**：K⁺ 与 K⁻ 在行人区外逐像素相同，所以 adapter 若学到 Cosmos 的伪影，只会出现在行人区内，可以用区外特征差为 0 这一构造性质做 null。

限定：K 不是真实；Cosmos 的「真实感」是它训练分布（多为北美 / 欧洲行车记录）的画风，与 nuScenes / WOD 各有差距。所以「画风份额」严格说是「Cosmos 能改掉的那部分」，是画风份额的下界估计（Cosmos 没改到的画风差会被算进「内容」）。
场景来源是 Town12 长路线切片；考卷（P5 v1 BA）在别的 town，adapter 的评测天然是跨场景的。

---

## 5. 预登记实验提案

三个实验按顺序做，前一个决定后一个开不开。全部用现成数据，不需要新渲染。GPU 用排期 D2–D5 卡 A / 卡 B 的空隙。每个实验的判据在任何数字之前写进各自的 todo。

### E0：CARLA 行人信息在 openpilot 哪一层消失（定位）

- **做法**：P5 v1 BA 行人 4 个 family 的 4 414 对，用 op-adapt 已缓存的 stage 3 trunk（`processed/op_adapt/p5/`），跑与第 55 条 trunk 诊断**同一套**的 mean + max 池化 2048 维 probe（D0 的 fold、类平衡 L2 logistic、route 聚类 bootstrap）；
  另用 port 补抽 stage 1、stage 2 输出（每对两帧，几分钟 GPU），同一 probe；同时报一个 2 层 MLP probe（描述）和 domain classifier AUC（CARLA P5 对 nuScenes val，每层，描述）。
- **成本**：CPU 为主，GPU ≤ 0.5 GPU·h。
- **判据（写死）**：CARLA stage 3 行人 AUC ≥ 0.65 且对 `vision` 的配对 Δ CI 下界 > 0 → 「信息在 stage 3、stage 4 起丢」，E2 放 P2；< 0.60 → 看 stage 2 / 1，最浅的一个 ≥ 0.65 的层就是 E2 的位置；都 < 0.60 → 「CARLA 行人在 openpilot 的前三个 stage 里就没被表示」，特征 adapter 在这一类上**判不可行**，只剩像素级（Cosmos）或重训更深。
- **各结局的意思**：第一种结局说明 CARLA 与真实的差主要在 stage 4 这一段对外观的敏感性上，恰好是 op-adapt B 在调的参数，第二轮 B 加原始 CARLA 帧大概率能把原始 CARLA 带上来；第三种结局说明域差在很浅的层，画风以外的内容差可能很大，应当把验收原则里的「sim」定义成 Cosmos 画面（midterm-gaps Q2.2），并在文中如实写这个限制。

### E0 结果（2026-09-29，[todo](../todos/2026-09-29-e0-layer-probe.md)，登记先于数字；决定见 [第 62 条](decisions.md)）

同一套 probe（CARLA：第 42 条 D0，P5 v1 BA 行人 4 414 对；真实：第 55 条 (a)，nuScenes val 走廊行人）逐层读，mean + max 池化，线性；stage 3 用 P5 / nuScenes 的 trunk 缓存，stage 1、2、4 由 port 从像素抽（约 0.2 GPU·h）。

| 层 | CARLA AUC [95% CI] | 对 `vision` 的 Δ [CI] | 真实 AUC [95% CI] |
|:--|:--|:--|:--|
| stage 1 | 0.526 [0.508, 0.556] | +0.018 [0.000, +0.040] | 0.715 [0.628, 0.799] |
| stage 2 | 0.523 [0.504, 0.548] | +0.015 [−0.003, +0.033] | 0.740 [0.644, 0.821] |
| stage 3 | 0.523 [0.507, 0.544] | +0.014 [+0.002, +0.028] | **0.831** [0.758, 0.895] |
| stage 4 | 0.522 [0.502, 0.546] | +0.013 [−0.002, +0.028] | 0.748 [0.666, 0.813] |
| `vision` | 0.509 [0.499, 0.520] | — | 0.702 [0.599, 0.785] |
| `temporal` | 0.506 [0.499, 0.515] | −0.003 [−0.008, +0.003] | 0.709 [0.598, 0.799] |

**判格：G-none**（stage 1–3 全部 < 0.60；stage 3 = 0.523，远低于 G-3 的 0.65）。CARLA 上行人各层平躺在 0.51–0.53，没有真实侧那个 stage 3 升起、stage 4 掉下的形状；2 层 MLP probe 也是 0.52–0.54，域分类器 AUC 每层 ≈ 1.0（描述）。
按 E0 原文，特征 adapter 在 CARLA 行人这一类上判**不可行**，只剩像素级（Cosmos）或重训更深；E2 不开，E1 是否单独有价值见第 62 条。

限定，读的时候要带着：(1) CARLA 侧是配对可分性（场景相同、只差行人），真实侧是走廊内有无行人，probe 可借上下文，所以真实侧 stage 1 的 0.715 不是「浅层有行人」，并排比的是形状；
(2) 池化 probe 对小而局部的目标不敏感，G-none 是「池化特征上读不出」，不是「特征图里没有」，空间分辨的读法没测；
(3) 阳性对照：同批帧上 HighwayCutIn（车辆）的配对 AUC 随深度升高（stage 1 0.504、stage 3 0.647、`temporal` 0.762），管线对车有响应；行人 family 里 VehicleTurningRoutePedestrian（2 659 对）每层 0.50，PedestrianCrossing 只在 stage 1 / 2 有 0.63 / 0.61。
表见 [research/results/e0-layer/summary.md](results/e0-layer/summary.md)。

### E1：2 × 2 析因，画风占多少（分解，也是 midterm-gaps F1 的扩展版）

- **做法**：取 G4 全量里已完成的对（≥ 300 对即开，全部 2 000 对更好），原 Cinque 在 C⁺ / C⁻ / K⁺ / K⁻ 四段上按 stream 跑（port 或 ORT TensorRT，与第 42 条同一喂法），抽 stage 1–3、`vision`、`temporal` 与原生 plan / lead / 车道线概率。每层报：
  (i) 行人配对可读性（D0 probe，x⁺ 对 x⁻）在 C 与 K 上各一个 AUC；(ii) 画风位移 ‖z(K⁻) − z(C⁻)‖ 与内容位移 ‖z(C⁺) − z(C⁻)‖ 的比；(iii) 原生 plan 在 C⁺/C⁻ 与 K⁺/K⁻ 上的减速翻转（行为读数）；(iv) 车道线概率与 lead 的 C / K 差。
- **成本**：openpilot 前向每对 4 × 93 帧，2 000 对约 74 万帧；按 stream 每步只算一帧新视觉，约 1–2 GPU·h。
- **判据（写死）**：`temporal` 上 K 的行人 AUC ≥ 0.70 且对 C 的配对 Δ CI 下界 > 0 → G-a（渲染域差）成立；< 0.60 → G-a 不成立，CARLA 行人的问题主要不是画风。
  额外定义「画风份额」= (AUC_K − AUC_C) / (AUC_real − AUC_C)，AUC_real 用 nuScenes val 同一 probe 同一层的值（`temporal` 0.709，stage 3 0.831）；只描述，不进判格。
- **各结局的意思**：G-a 成立且画风份额 ≥ 0.7 → 域差大部分是画风，特征 adapter 值得训（E2），而且适配后的模型在闭环里需要它；G-a 成立但份额低 → 有画风，但真实一侧更好读，差在内容或 Cosmos 没画到；G-a 不成立 → 画风不是问题，任何 adapter 都不会有用，这个想法到此为止。
- 注意 E1 只用**原模型**，不依赖第二轮 B，可以在 D2 与第二轮并行。

### E2：G4 配对监督的 CARLA→真实感特征 adapter（只在 E0 找到位置、E1 判 G-a 成立时开）

- **做法**：在 E0 定的层 ℓ 后插一个残差 adapter A_ℓ（逐通道仿射 + 一个 1×1 瓶颈卷积，参数 < 1% 的 stage 4），其余全部冻结（原模型；B 第二轮出来后在 B 上再做一遍）。
  损失：‖A_ℓ(z_ℓ(C)) − z_ℓ(K)‖²（逐通道按 K 的标准差归一），另加 `temporal` 层的同类项（让 adapter 对下游有效，而不只是在 ℓ 层像）；训练用 G4 Town12 的对，按 route 留 10% 选 early stop。
  三个 arm：(a) 配对回归 adapter；(b) 零训练的逐通道统计量对齐（AdaIN / CORAL 式，把 C 在 ℓ 层的均值方差换成 K 的全局统计量）；(c) 安慰剂：打乱配对（C 与别的 pair 的 K 配）训同一个 adapter。
- **评测（全部是训练没见过的 town 与原始 CARLA 画面，不经 Cosmos）**：P5 v1 BA 行人 D0 AUC（`temporal`）；P5 cut-in 翻转与 null false-flip（不能坏）；原生 plan 在 P5 null 帧上相对 K 风格的漂移（无法直接测，改报 plan 与原模型的漂移，描述）；
  另在真实 nuScenes val 上确认 adapter **不插**时一切不变（按构造成立），插上时漂移多大（描述：adapter 是 CARLA 专用的「配方」，不该用到真实上）。
- **成本**：ℓ 层特征缓存 + 训练约 1–3 GPU·h（adapter 小，trunk 已缓存时更少）；P5 评测 < 0.5 GPU·h。
- **判据（写死）**：(a) 在 P5 原始 CARLA 行人 `temporal` AUC ≥ 0.60 且对原模型配对 Δ CI 下界 > 0；cut-in 翻转不低于原模型 5 pp 以上；(a) 对 (c) 的配对 Δ CI 下界 > 0（否则是 adapter 容量本身而不是配对在起作用）。(b) 若与 (a) 持平，结论写成「逐通道统计量就够」，更便宜的那个进闭环。
- **各结局的意思**：过 → 一个实时的、CARLA 专用的特征 adapter 可以替代 Cosmos 进闭环，缺口一 H2 有了修复，验收原则的「sim」可以保持为原始 CARLA；(a) 在 E1 的 K 上涨、在 P5 原始 CARLA 上不涨 → adapter 学的是 Town12 特有的东西，或 Cosmos 伪影；(b) ≈ (a) ≈ (c) 都不涨 → 画风差不是逐层可逆的映射，只能在像素层修。

### 预算与排期

| 实验 | GPU·h | 需要的数据 | 何时 | 依赖 |
|:--|--:|:--|:--|:--|
| E0 | ≤ 0.5（多为 CPU） | P5 trunk 缓存（已有）、nuScenes val trunk（已有） | D2，任何时间 | 无 |
| E1 | 1–2 | G4 全量里 ≥ 300 对（D1–D2 出） | D2–D3，与第二轮 B 并行 | 无（只用原模型） |
| E2 | 1–3.5 | G4 全量、P5 v1 BA | D3–D5，只在 E0 / E1 判开时 | E0、E1 |
| 合计 | ≤ 6 | | | |

按 2 卡 330 元 / 天，6 GPU·h 约 80 元，远在中期余量之内。E0 与 E1 同时也是 midterm-gaps F1 的完整版，无论 adapter 做不做都应该先跑。

---

## 6. 诚实的判断：为什么它可能不行

1. **最大的风险是信息在 adapter 的输入之前已丢**。CARLA 帧上 openpilot 从 `vision` 起行人就是 0.51，而 openpilot 是在真实行车视频上训的 ConvNeXt，对 CARLA 行人这种没见过的外观，早期层的响应可能本来就弱。
   如果 E0 在 stage 1–3 都读不出，特征 adapter 就只能放在像素层——那就是 Cosmos 本身。
2. **画风以外的内容差很可能不小**。CARLA 行人模型的外形、步态、尺度，行人出现的位置与方式（BehaviorAgent 场景的横穿脚本），都不是画风；Cosmos 在外观上画真实了，但行人的运动仍是 CARLA 的。
   E1 能量出这部分有多大，adapter 修不了它。
3. **我们已经有一个 adapter 类方法的负面读数**：第 44 条 E1 在 head 输入上做逐维标准化，损害只减半；G1 的真实数据 gate 同样只是整体缩小 Δ。这些都是「在已经读错的层上对齐」，与文献里 Zhao et al. 2019 的失效方式一致。
4. **Cosmos 不是真实**。E2 学的是 CARLA → Cosmos 的映射；它能证明「Cosmos 能改掉的画风差可以在特征层实时改掉」，不能直接证明适配后的模型在真实上更好。真实一侧的读数仍要靠第二轮 B 在 nuScenes / WOD 上的读数。
5. **VLA 的经验站在「用目标域数据训」这一边**：OpenVLA 系列所有适配都用目标域示教（每任务 10–300 条），冻结视觉只调后面的明显更差。对我们，最稳的做法很可能不是 adapter，而是把原始 CARLA 帧也作为一个带标签的域放进第二轮 B 的训练；adapter 是在「闭环里必须实时、而且不想动共享权重」时的补充。

**推荐的第一步**：E0。它几乎不花钱（≤ 0.5 GPU·h，数据已在盘上），一次读数就决定特征 adapter 在我们的 openpilot 上有没有位置可放；紧接着用 E1（1–2 GPU·h）把 midterm-gaps 的 F1 做成完整的 2 × 2 分解。

---

## 参考文献（均已核实存在；arXiv 号给出便于检索）

- Ganin & Lempitsky, Unsupervised Domain Adaptation by Backpropagation, ICML 2015（arXiv 1409.7495）；Ganin et al., Domain-Adversarial Training of Neural Networks, JMLR 2016。
- Tzeng et al., Adversarial Discriminative Domain Adaptation (ADDA), CVPR 2017（arXiv 1702.05464）。
- Long et al., Learning Transferable Features with Deep Adaptation Networks, ICML 2015（arXiv 1502.02791）。
- Sun & Saenko, Deep CORAL, ECCV 2016 Workshops（arXiv 1607.01719）。
- Zhao et al., On Learning Invariant Representations for Domain Adaptation, ICML 2019（arXiv 1901.09453）。
- Ben-David et al., A Theory of Learning from Different Domains, Machine Learning 2010。
- Tsai et al., Learning to Adapt Structured Output Space for Semantic Segmentation (AdaptSegNet), CVPR 2018（arXiv 1802.10349）。
- Hoffman et al., CyCADA, ICML 2018（arXiv 1711.03213）。
- Li et al., Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN), ICLR 2017 Workshop / Pattern Recognition 2018（arXiv 1603.04779）。
- Huang & Belongie, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN), ICCV 2017（arXiv 1703.06868）。
- Pan et al., Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net, ECCV 2018（arXiv 1807.09441）。
- Choi et al., RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening, CVPR 2021（arXiv 2103.15597）。
- Zhou et al., Domain Generalization with MixStyle, ICLR 2021（arXiv 2104.02008）。
- Rebuffi et al., Learning Multiple Visual Domains with Residual Adapters, NeurIPS 2017（arXiv 1705.08045）。
- Chang et al., Domain-Specific Batch Normalization for Unsupervised Domain Adaptation, CVPR 2019（arXiv 1906.03950）。
- Hu et al., LoRA, ICLR 2022（arXiv 2106.09685）；Jia et al., Visual Prompt Tuning, ECCV 2022（arXiv 2203.12119）。
- Wei et al., Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation (Rein), CVPR 2024（arXiv 2312.04265）。
- Lee et al., Surgical Fine-Tuning Improves Adaptation to Distribution Shifts, ICLR 2023（arXiv 2210.11466）。
- Wang et al., Tent: Fully Test-time Adaptation by Entropy Minimization, ICLR 2021（arXiv 2006.10726）；Sun et al., Test-Time Training, ICML 2020（arXiv 1909.13231）；
  Wang et al., Continual Test-Time Domain Adaptation (CoTTA), CVPR 2022（arXiv 2203.13591）；Adachi et al., Test-time Adaptation for Regression by Subspace Alignment, ICLR 2025（arXiv 2410.03263）。
- Tzeng et al., Adapting Deep Visuomotor Representations with Weak Pairwise Constraints, WAFR 2016（arXiv 1511.07111）。
- James et al., Sim-to-Real via Sim-to-Sim (RCAN), CVPR 2019（arXiv 1812.07252）。
- Müller et al., Driving Policy Transfer via Modularity and Abstraction, CoRL 2018（arXiv 1804.09364）。
- Bewley et al., Learning to Drive from Simulation without Real World Labels, ICRA 2019（arXiv 1812.03823）。
- Pan et al., Virtual to Real Reinforcement Learning for Autonomous Driving, BMVC 2017；Yang et al., Real-to-Virtual Domain Unification for End-to-End Autonomous Driving, ECCV 2018。
- Kar et al., Meta-Sim: Learning to Generate Synthetic Datasets, ICCV 2019（arXiv 1904.11621）。
- Vázquez et al., Virtual and Real World Adaptation for Pedestrian Detection, TPAMI 2014。
- Richter et al., Enhancing Photorealism Enhancement, TPAMI 2022（arXiv 2105.04619）。
- Oquab et al., DINOv2, TMLR 2024（arXiv 2304.07193）；Assran et al., V-JEPA 2, 2025（arXiv 2506.09985）。
- Wang et al., Drive Anywhere: Generalizable End-to-end Autonomous Driving with Multi-modal Foundation Models, ICRA 2024（arXiv 2310.17642）。
- Mallak & Maalouf, Robustness Is a Function, Not a Number: A Factorized Comprehensive Study of OOD Robustness in Vision-Based Driving, arXiv 2602.09018（2026，预印本）。
- Tian et al., SimScale: Learning to Drive via Real-World Simulation at Scale, CVPR 2026（arXiv 2511.23369）。
- Kim et al., OpenVLA: An Open-Source Vision-Language-Action Model, CoRL 2024（arXiv 2406.09246）。
- Kim, Finn & Liang, Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT), RSS 2025（arXiv 2502.19645）。
- Black et al., π0: A Vision-Language-Action Flow Model for General Robot Control, 2024（arXiv 2410.24164）。
- Zitkovich et al., RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, CoRL 2023（arXiv 2307.15818）。
- Li et al., Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv), CoRL 2024（arXiv 2405.05941）。
