# 驾驶智能藏在哪里？从 world model 到薄 head 的一条研究线

状态：文献综述，写给组会分享。材料截止 2026-09-20，整理自两轮 deep research 报告
（`research/lit/`，报告本身不进 git）。文中所有数字来自被引论文或官方页面，**不含我们自己的实验结果**；
图由 [figs/survey/make_figures.py](figs/survey/make_figures.py) 生成，数据表就写在脚本里。
论文日期是 arXiv 首次提交的年月，修订另注。「近三个月」指 2026-06-20 到 2026-09-20 之间首发或修订。

---

## 0. 三句话版本

一辆车在路口要在几百毫秒内做出决定。大模型知道很多关于世界的事，但一次 forward 就要几百毫秒，
再逐字生成一段话就更慢。于是这条研究线的核心问题只有一个：**预训练模型内部的表示里，已经有多少
「驾驶智能」？把这份表示变成动作的那一部分，能薄到什么程度？**

读完两轮文献，答案分三层：

1. **信息确实在里面。** 冻结的视觉或视觉语言模型，接一个很小的输出层，已经能在多个 benchmark 上拿到有竞争力的分数。
2. **「读得出」不等于「用得好」，更不等于「开得安全」。** 探针能读出的信息、模型生成时实际用上的信息、
   以及车执行之后不出事，是三件不同的事，中间每一步都有反例。
3. **整个领域收敛的不是「删掉生成」，而是「把计算放到哪儿」。** 生成式训练留下的表示可以在部署时只保留任务需要的
   那一小段计算，但保留多少、何时需要更多，仍然是开放问题，也是这条线最挤的地方。

我们组的切口是：一个完全冻结、没做过任何 driving 适配的 VLM，取中间层的 hidden state，接一个在固定轨迹词表上打分的薄 head，
在 Waymo 的端到端榜单上看能走到哪里、代价多少。这篇文章画的是这个切口周围的地形，不是我们的结果。

---

## 1. 为什么大家都在「变薄」：三条路线的科普

先把几个词说清楚。**End-to-end driving**（端到端驾驶）指一个网络直接从传感器输入映射到未来轨迹或控制量，
不再手工拆成感知、预测、规划三段。**Backbone** 是网络里负责把图像编码成内部表示的大块；**head** 是把这份表示变成
分类、分数或轨迹的小块；**representation**（表示，或者说 latent）是 backbone 输出的那串向量，人不能直接读，
但下游模块能用。

要把「大模型知道的东西」搬进车里，过去三年主要有两条路：

- **World model** 路线：先学会预测未来会发生什么，再在预测的基础上规划。最早的做法是真的生成未来几秒的视频。
- **VLA（vision-language-action）** 路线：拿一个已经会看图说话的 VLM（vision-language model），教它输出动作。
  最早的做法是让它像写作文一样，把场景描述、推理过程和轨迹坐标全部当成文字生成出来。

两条路都从「什么都生成」开始，也都在往「少生成一点」走。但这两条「变薄」的历史，都不是一条直线。

### 1.1 World model：不必画出未来，可能仍要预测未来

一个容易产生的误解是：大家先生成视频（GAIA-1、DriveDreamer、Vista），后来发现视频没用，于是一路删到直接输出动作。
文献支持的是另一幅图景：**未来预测被放到了不同位置，服务于不同目的**。GAIA-1 和 GAIA-2 的主要用途是可控仿真和造数据，
不是在线规划器；从它们到 latent planner 之间不是逐代替换，而是分叉 [1–4]。

![world model roles](figs/survey/f1_world_model_roles.png)

图 1：一个驾驶 world model 可以坐在三个位置。第一行 World4Drive 在部署时仍然对每个意图做 latent rollout（在内部把未来推演几步）再打分；
第二行 DriveLaW 借用视频生成器第一个去噪步的中间特征，RGB 解码器根本不跑；第三行 LAW 和 Drive-JEPA 把「预测未来」写进训练目标，
部署时只剩一个普通 encoder。三者都不输出像素，但保留的在线计算完全不同 [5–8]。

这三条路各自有什么证据？最干净的几组消融是：

| 实验 | 实际改变了什么 | 结果 | 能说明什么 |
|---|---|---|---|
| LAW，Tables 4–5 [5] | 有 / 无 future-latent prediction 辅助任务 | NAVSIM PDMS **77.5 → 84.6** | 学习预测未来有用；但这不是「删掉 RGB decoder」的对照 |
| DriveLaW，Table 6 [7] | planner 用视频生成器第 1 / 5 / 10 个去噪步的特征 | PDMS **89.1 / 86.9 / 23.2** | 越晚的生成状态越不利于决策；早期状态足够 |
| DriveLaW，Appendix A.2 [7] | 部署时缓存第一次 Video DiT 的特征，不解码视频 | 同上 | RGB 解码不是必需；但 Video DiT 和迭代式 Action DiT 都还在 |
| Drive-JEPA，Tables 5–6 [8] | 同一 planner，换 encoder 的预训练方式 | 见图 4 左 | 表示的训练目标很重要；同样不是 decoder 消融 |
| WA-JEPA v2，Table 4c [9] | 同一架构，无未来监督 / 直接回归 / flow matching | EPDMS **91.1 / 90.7 / 91.7** | 保留 stochastic 的未来 latent 建模可能优于进一步简化 |

表 1：只有「实际改变了什么」这一列能回答消融问题。PDMS（Predictive Driver Model Score）是 NAVSIM v1 的综合分，
EPDMS 是 v2 的扩展版，两者不能混排。

三点值得记住。第一，DriveLaW 第 10 步的崩塌不能直接读成「物理信息被视觉细节挤掉了」，表格排除不了 conditioning
分布或优化的问题。第二，删掉 RGB 也不保证实时：DriveLaW v3 在 H20 上的轨迹规划要 **0.71 s** [7]。第三，
针对原问题最严格的版本，即「同一 backbone、同样训练预算、同一 action head、只去掉 RGB decoder、在闭环上证明能力不掉」，
**没有找到已发表的证据**。

### 1.2 VLA：少写轨迹，不等于少做推理

VLA 这边的历史更像四种输出接口的并存，而不是一条单向的路。

![output interfaces](figs/survey/f2_output_interfaces.png)

图 2：十个代表系统按「每次决策生成多少 token」排列。箭头不表示引用或性能关系；最右端的候选打分方式目前只有交通概念探针
和 Jev 风格的 demo，还没有一个完整的驾驶策略，所以最后一步画成虚线 [10–17]。

DriveVLM 和 EMMA 建立的是「通用预训练知识可以参与驾驶」这件事，没有证明长文本是最好的动作通信格式 [10, 11]。
2025 年之后出现的不是一个排序，而是三种压缩同时发生：不写 CoT（chain-of-thought，显式的推理文本）；
用离散的 action token 代替小数文本，比如 AutoVLA 把 5 s 轨迹编成 10 个 token [13]；
从 hidden state 直接接一个连续的 diffusion 或 flow planner，比如 ORION、Alpamayo-R1、Qwen-Drive-1.0 [14–16]。

最新的结果也没有沿这条谱单向右移。NVIDIA 的 Alpamayo 2 Super 把一个 32B 的 reasoner 和一个 2B 的 diffusion action expert
组合在一起，显然不是 tiny head [18]。Qwen-Drive-1.0 的官方说明则明确写道，RL checkpoint 应搭配 reasoning 一起用，
因为 RL 的训练路径本身就是 reasoning-conditioned [16]。

---

## 2. 「变薄」有四种，别混在一起

设想车辆来到一个路口。系统可以先预测未来画面再选轨迹；可以在内部预测几个未来状态；也可以直接从当前表示读出「减速」。
输出越来越短，不代表内部做的事情相同。读这条线的论文，第一步是问清楚：**它删掉的是哪一行？**

| 改变的环节 | 具体少做了什么 | 不能顺便推出什么 |
|---|---|---|
| RGB decoding | 不把预测的未来还原成像素 | 不能说未来预测已经没用 |
| Textual trajectory | 不逐 token 写出一串坐标 | 不能说 reasoning 已经没用 |
| CoT | 不再生成显式推理文本 | 不能假定困难场景的能力不变 |
| Latent rollout | 不再显式推演中间的未来状态 | 不能假定行动后果的信息还在 |

表 2：某一行的成功消融，不是另三行也能删的证据。

这四行里最容易被混成一件事的是中间两行，而 Alpamayo-R1 恰好给了同一平台上的分解 [15]：

![Alpamayo-R1 latency](figs/survey/f3_alpamayo_latency.png)

图 3：同一个模型、同一张 GPU（RTX 6000 Pro Blackwell），三种输出方式的延迟，来自论文 Table 14。
保留 reasoning、只把轨迹从 127 个自回归 token 改成 5 步 flow，延迟从 312 ms 降到 99 ms；再去掉 reasoning 才到 29 ms。
也就是说，**轨迹的序列化方式才是大头，不必先删 reasoning 才能提速**。

但图 3 只有速度。论文同时报告 reasoning 版本在困难场景上规划精度最多提升 12%、闭环的 close-encounter 率下降 35%，
这些收益不能直接记到最快那条路径的账上 [15]。AutoVLA 也提供了反对「CoT 无用论」的证据：低数据量时 reasoning 未必有益，
数据多了以后带 CoT 训练的策略更好，而它的 fast 模式平均 1.07 s、slow 模式 10.5 s，即便只输出 10 个 action token 也不算快 [13]。

### 2.1 本文用的「薄」是什么意思

本文说的 **thin decoder** 指：在一个大的、冻结或轻微适配的 backbone 之上，用一个小的、非自回归的输出 head。
小 MLP、线性层、在固定候选上打分都算；「输出 token 少」或者「用了 diffusion」本身不保证 head 很薄。
按这个定义，图 2 里只有最右一列是 thin decoder，第三列的 flow head 要看迭代几步、有多大。

### 2.2 Jev：把问题改成选择题，解决了哪一半

2026 年 9 月 15 日 TypeSafe 发布了 Jev，描述为一种专有架构，核心是并行采样器和 RLCD 训练 [19]。社区很快出现复现：
hr98w/jev-visual 实现了 shared multimodal prefill（多个问题共享一次图像编码和前缀计算），之后对每个候选答案打分，
并明确声明没有复现专有训练和 calibration [17]。它的数字说明的是均摊效应：64 个问题独立打分要 37.3 s，共享前缀后 2.4 s，
但这是批处理，不是 2.4/64 秒一次的连续控制。

对驾驶来说，Jev 风格的接口意味着：不让模型自由写答案，而是提前给几个候选，直接读它更支持哪一个。
这里有两个必须预先写下的陷阱：

- **候选内部的 softmax 是相对概率，不是安全概率。** 如果正确选项根本不在候选里，模型仍然可以非常确信其中一个。
- **输出一定符合格式，不等于输出一定正确。** 一个从不出格式错误、但持续选错动作的三分类器，只解决了接口问题。

公开的、受控的 Jev 风格驾驶复现（accuracy、calibration、闭环），截止本文**没有找到**。这和「已经有人实现了 shared-prefix scoring」完全兼容。

---

## 3. 喂给 head 的是什么：front-end representation

head 薄了以后，差别就全在前端。两个系统用同样的分类器，也可能看到不同的分辨率、历史窗口和 token。
文献里比较有说服力的，是**同一个 planner 下换前端**的实验：

![front-end ablations](figs/survey/f4_frontend_ablations.png)

图 4：三组「同一 planner、换前端」的消融，全部是 NAVSIM v1 PDMS。左：Drive-JEPA 同一 planning framework 下换 encoder 的预训练，
ImageNet 和 DINOv2 几乎一样，V-JEPA 2 高出 10 分，针对驾驶再预训练的 JEPA 最好 [8]。中：DriveLaW 同一 Action DiT 下，
BEV、VLM、视频生成器三种前端 [7]。右：DriveLaW 用视频生成器第几步的特征喂 planner [7]。三组都不是「等规模、等数据」的受控比较，
但方向一致：**前端的训练目标比架构名字更重要，而越晚的生成状态越不适合做决策。**

| 前端类型 | 已有证据支持什么 | 还没证明什么 | 代表工作 |
|---|---|---|---|
| Frozen VLM latents | 物体存在、数量等概念线性可读；生成答错不等于信息不存在 | 未做 driving 适配的 LLM hidden state 直接接 planner 的完整对照 | Probing Visual Concepts [20]、FROST-Drive [21] |
| 自监督视频 / 单帧特征 | V-JEPA 2 面向视频预测，DINOv2/v3 面向静态稠密特征 | 同一 head 下对未来速度的优势 | Drive-JEPA [8]、V-JEPA 2 [22]、DINOv3 [23] |
| BEV / occupancy | 显式几何便于空间推理 | 不自带开放语义和交规知识；成本要算上视角变换和时序融合 | OccWorld [24] |
| 离散 scene token | 可以和 action token 统一成一个序列 | 重建质量不等于对动力学的充分性 | DrivingGPT [25] |

表 3：四类前端各自的证据边界。四类前端接同一个 frozen linear head 的完整对照，**没有找到**。

一个容易漏掉的 baseline 是 **VLM 自己的 vision encoder**。如果只比 VLM 和 DINOv2，胜负可能来自视觉预训练、分辨率或参数量，
而不是 LLM。FROST-Drive 在这一点上给了一个有意思的数据：同样的 adapter 加 GRU head，冻结的 ImageNet ViT 在它自定义的
val 口径上是 RFS 7.39，换成冻结的 VLM vision encoder 是 8.17，而它的 LLM 部分根本没参与 [21]。
另一篇比较 frozen VLM 和视频生成模型的工作则发现，VLM 在语义上强，视频生成模型在几何和相机运动上强 [26]。

---

## 4. 探针：信息在里面，不等于用得上

**Linear probe**（线性探针）是一种测量：冻结主模型，只在它某一层的表示上训练一个线性分类器，看某个量是否容易被读出来。
它便宜、可解释，是「表示里有没有 X」这类问题的标准工具。这一节讲的是探针文献告诉了我们什么，以及它们留下了什么。

### 4.1 与驾驶最近的探针工作已经做了什么

**Probing Visual Concepts in Lightweight VLMs for Automated Driving** [20] 是最直接的先例。它在 CARLA 生成的反事实图像上，
测物体存在、数量、空间关系、朝向四类概念，对 vision encoder、projector、LLM 逐层训练线性探针，模型包括 Qwen3-VL-2B。
结论有三条：部分朝向信息从 vision encoder 到 LLM 是下降的，但不是所有空间概念都下降；同一模型的探针、自由生成和
constrained output 三种读法结果不一致；沿探针方向做 activation steering 可以改变生成。

这意味着「首次探索驾驶 VLM 的逐层物理信息」「首次发现晚层空间能力下降」「首次证明可读信息不等于输出能力」
这几种表述都已经被占了。剩下的空白是 **时间信息、未来动力学和行动后果**，而不是换一个模型尺寸再画一条逐层曲线。

### 4.2 曲线下降，还不能叫「信息丢失」

假设探针在早层比晚层准。直觉上这像是模型越接近语言输出，越不懂物理。但同一个观察至少有四种解释，
文献给每一种都提供了例子：

| 解释 | 意味着什么 | 支持它的先例 |
|---|---|---|
| 信息从未进入 vision representation | 前端就没编码，后面当然读不出 | 需要先查 encoder 层 |
| 信息在后续计算中被压掉 | 换更早的层，或者改训练 | 驾驶探针里部分朝向的下降 [20] |
| 信息迁移到了 text / prompt token | 换读取位置，pooling 要跟上 | Linear Mechanisms：空间与时序信息可从 visual token 转移到 text token [27] |
| 信息还在，只是线性读不出（readout mismatch） | 换 pooling 或小 MLP，不必重训 | Hidden in plain sight：VLM 生成答错时视觉表示里往往仍有答案 [28] |

表 4：四种机制可能产生几乎相同的一条下降曲线，却指向完全不同的改进。只画曲线区分不了它们。

还要把「晚层导致变化」和「language alignment 导致变化」分开。前者是当前模型内部的定位，后者是关于训练目标的因果主张。
要证明后者，需要 alignment 前后的 checkpoint 或匹配的训练对照；拿同一家族两个尺寸比，不能替代这项控制。

### 4.3 适配会不会把知识擦掉

这条线还有一个相邻问题：如果不冻结、而是用 driving 数据去微调 VLM，原来的知识还在吗？
机器人领域的 Anchor-Align 给了最直接的测量：单纯用行为克隆去适配一个 VLA，它在 GQA 上的通用问答准确率在 1 万步内
相对下降约 94%；加上表示锚定后保留约原水平的 70%。同一篇还逐层读出动作，峰值在第 22 层，R² 为 0.60 [29]。
Qwen-Drive-1.0 在驾驶域给了不完全但直接的数字：混合通用数据做 SFT 后，MMMU 从 73.4 到 72.7，MMBench 从 87.1 到 85.5 [16]。
两者都说明**任务分数上升不保证原有能力保留，而冻结参数也不保证 head 用上了那些知识**。

### 4.4 一个所有 open-loop 研究都绕不开的捷径

nuScenes 上有一篇 2023 年底的论文，标题就是问题：**Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?** [30]
它展示只用车辆自身状态（速度、加速度、yaw rate），不看任何画面，在 open-loop（离线、不执行、只比对记录轨迹）指标上就有竞争力。
原因很简单：未来两三秒的行为，大部分由当前运动状态决定，车正在左转，两秒后大概率还在左转。

对本文的主题，这是最重要的一条方法论提醒：**任何「视觉表示里有驾驶智能」的主张，都必须先和一个只用自身状态的 baseline 比，
增量才算视觉的功劳。** 这也是为什么 Waymo 榜单上至今没有一行官方口径的 ego-only 结果，会成为一个值得注意的空白（见第 6 节）。

---

## 5. 从探针到规划器：trajectory vocabulary 是什么

探针回答「有没有」，但自动驾驶社区认的是榜单。把「读出一个类别」升级成「输出一条轨迹」，同时保持 head 很薄，
最成熟的做法是 **trajectory vocabulary**（轨迹词表）：在训练集的所有未来轨迹上做 k-means 聚类，得到 K 条代表性轨迹作为候选，
模型不再回归坐标，而是给 K 个候选打分，选最高的一条输出。这样输出层就是一个 K 类分类器，可以做得很薄；
代价是词表覆盖不到的轨迹永远输出不了，所以很多方法会在选中的候选上再做一步小幅修正（refinement）。

| 方法 | K，怎么建 | scorer 的监督 | 修正 | 结果 |
|---|---|---|---|---|
| Hydra-MDP [31] | **4096 / 8192**，训练轨迹 k-means | 到 imitation 目标的距离 + 仿真子分数蒸馏 | 无 | NAVSIM v1 PDMS 82.6 → 83.0 |
| Hydra-MDP++ [32] | 8192 | 更多 rule-based 子分数 | 无 | 91.0 PDMS |
| DiffusionDrive [33] | **20 个 anchor** + 噪声 | anchor 分类 + 最佳模式回归 / 去噪 | 2 步截断扩散 | 88.1 PDMS |
| GoalFlow [34] | 4096 / 8192 个**终点**，不是完整轨迹 | 终点距离 / 可驾驶区域 | goal-conditioned flow | 90.3（5 步）/ 88.9（1 步） |
| WoTE [35] | **256**，训练轨迹 k-means | winner-take-all 回归 + world model 奖励 | 先修正后打分 | 88.3 PDMS |
| GTRS [36] | 训练 16384，推理 8192 + 动态候选 | 子分数监督 + vocabulary dropout | 含动态轨迹 | navhard 49.4 EPDMS（挑战赛 ensemble） |
| Auto-JEPA [37] | 从训练轨迹记忆里检索 top-300 | latent 检索 + rule score | 不生成新轨迹 | v1 91.3 PDMS；v2 navtest 89.1 EPDMS |

表 5：「完整轨迹词表」「终点词表」「扩散初始化用的 anchor」不是同一种 K。同一列里的分数可以比，跨列的绝对值不能。

![vocabulary K](figs/survey/f7_vocab_k.png)

图 7：文献里公开的 K 扫描。三条线是三个不同系统，只能在一条线内部比。共同的形状是：K 翻倍带来的增益在几百之后就很小了，
DiffusionDrive 从 20 到 40 个推理候选只涨 0.1 分。词表大小不是主要瓶颈，**scorer 能不能选对**才是 [31, 33, 35]。

回到本文的主题：如果前端是一个冻结的 foundation model，后面接词表 scorer，这件事有没有人做过？
有，而且近三个月很密：

| 近邻 | 冻结的是什么，head 是什么 | 公开结果 | 和「冻结 VLM + 词表打分」的距离 |
|---|---|---|---|
| FROST-Drive [21]，2026-01 | 冻结 InternVL3 的 vision encoder；adapter + GRU 回归 | Waymo test RFS **7.856** | 不用 LLM hidden state，但系统定位最近 |
| Qwen-Drive-1.0 消融 [16]，2026-09 | 未做 driving 适配的 Qwen3.5-4B；训练一个 flow Planning Expert | Waymo **val** RFS 7.88 | frozen LLM feature 接 planner 已有；不是一次性的固定词表分类 |
| DiffAdapterVLA [38]，2026-09 | 冻结已适配的 Cosmos-Reason2-8B；在 LLM 内部插入可训练的 DiffAdapter | NAVSIM v1 88.3 → 90.3；**3.05%** 可训练参数 | 中间层 + 参数效率已有；不是完全静态可缓存的特征 |
| Auto-JEPA [37]，2026-07 | 冻结 V-JEPA 2；predictor + 检索式 scorer | v1 91.3 PDMS | 冻结表示 + 轨迹记忆已有；不是 VLM，head 也不薄 |
| VL-DPO [39]，2026-05 | 冻结 Gemini 2.5 Pro 给 12 条渲染出来的候选打分 | 自定义 Waymo val 7.23 → 7.30 | VLM 评分候选已有；是离线 teacher，不是在线系统 |
| OpenEMMA / LightEMMA 复现 [40] | zero-shot VLM 直接输出轨迹 | Waymo test RFS **5.158 / 6.517** | training-free 的近邻，分数远低于训练过的方法 |

表 6：判定是 partially done。不能再声称「first frozen-VLM planner」；「一次固定词表分类 + 完全静态缓存的中间层特征 + 同 backbone 的
分类 vs 回归对照」这个组合层面还有空间，但只换模型版本或换一层不够。

---

## 6. 赛场：Waymo E2E 榜单长什么样，有多挤

### 6.1 数据和协议

Waymo Open Dataset for End-to-End Driving（WOD-E2E）专门挑长尾场景，官方描述这些场景在日常驾驶中的出现频率低于 0.03%。
数据集论文进了 CVPR 2026 [41]。

| 项目 | 官方口径 |
|---|---|
| 规模 | **4021** 个 20 s 片段：train 2037 / val 479 / test 1505 |
| 相机 | 8 个环视相机，JPEG 嵌在 TFRecord 里 |
| 输入 | 过去 (−4 s, 0] 的 ego 状态，4 Hz；route intent 是 straight / left / right / unknown 四选一 |
| 输出 | 未来 (0, 5] 的 20 个 XY waypoint，4 Hz，车辆后轴坐标系 |
| test | 前 12 s 可见，后 8 s 隐藏 |
| 主指标 | **RFS**（Rater Feedback Score）：预测轨迹和几条人工评过分的参考轨迹比，离所有参考都远时分数向下限衰减；在 3 s、5 s 和 11 类场景上聚合 |
| 次指标 | ADE@5 s：对着评分最高的那条参考轨迹算的平均位移误差 |
| 提交 | 每 30 天 6 次；2026 年**不举办**正式 challenge，榜单保持开放 |

表 7：来自官方 challenge 页面和 proto 定义，2026-09 核对 [42]。

RFS 的设计动机值得一说：传统的 ADE 只问「像不像记录里那唯一一条轨迹」，可是长尾场景往往有好几种合理的做法。
让人来给几条轨迹打分，再看模型选的那条离高分轨迹多近，比单纯对着日志算距离更接近「开得好不好」。
代价是它仍然是 open-loop 的代理指标：不是在线人工评价，也不是执行后的碰撞率。

### 6.2 榜单

下面是所有能核实到日期和出处的 official-test 记录。这不是实时抓取的动态榜单。

| 方法 | Backbone 与更新方式 | Head；主要输入 | **RFS** | **ADE@5s (m)** | 出处 |
|---|---|---|---:|---:|---|
| RAP | DINOv3-H+，全模型约 0.88B，Waymo 阶段解冻 | 多候选回归 + 打分；多相机、4 s ego、route | **8.043** | **2.646** | 2510.04333，2025-10 |
| Poutine | Qwen2.5-VL-3B；SFT + GRPO | 自回归坐标；前 3 相机、4 s ego | 7.986 | 2.742 | 2506.11234，2025-06 |
| SUV ★ | Wan2.2-5B 视频生成器 + action expert，联合训练 | 视频 + flow action；仅前视 | 7.94 | 2.90 | 2608.03084，2026-08 |
| Qwen-Drive-1.0 ★ | Qwen3.5-4B；先 driving 适配，再冻结训 Planning Expert | flow Planning Expert；前 3 相机 × 4 时刻 | 7.78 → 7.91 | 2.65 → 2.67 | 2609.00111，2026-09 |
| Poutine-Base | 同 Poutine，无 RL | 同上 | 7.909 | 2.940 | 2506.11234 |
| MindVLA-U1 | Qwen3-VL-2B；vision 冻结，LLM 更新 | flow action + 语言 | 7.77 → 7.87 | 2.67 → 2.66 | 2605.12624，2026-05 |
| **FROST-Drive** | InternVL3 **vision encoder 冻结** | adapter + GRU；5 相机 | 7.856 | 3.565 | 2601.03460，2026-01 |
| ViT-Adapter-GRU | ViT；冻结边界未公开 | adapter + GRU | 7.849 | 2.889 | FROST Table 6 |
| Fast-dDrive | Qwen2.5-VL-3B；更新 backbone | block discrete diffusion | 7.823 | 2.907 | 2605.23163，2026-05 |
| UniPlan | DiffusionDrive 约 60M；SFT | anchor-conditioned diffusion | 7.780 | 2.842 | FROST Table 6 |
| HMVLM | Qwen2.5-VL-3B；全量训练 | 自回归 + 平滑；5 相机 | 7.737 | 3.072 | 2506.05883，2025-06 |
| DiffusionLTF | DiffusionDrive 约 60M | anchor-conditioned diffusion | 7.717 | 2.977 | 2510.26125，2025-10 |
| NoRD | Qwen2.5-VL-3B；全量训练 | 自回归 action，无 reasoning | 7.709 | 未报 | 2602.21172，2026-02 |
| dVLM-AD | LLaDA-8B + SigLIP2；SFT | discrete diffusion | 7.633 | 3.022 | 2512.04459，2025-12 |
| AutoVLA | Qwen2.5-VL-3B；SFT，RFT 用 LoRA | 自回归 action token | 7.556 | 2.958 | 2506.13757 |
| Swin-Trajectory | Swin 约 36M | MLP 回归 | 7.543 | 2.814 | 2510.26125 |
| NaiveEMMA | 官方 baseline | 自回归；8 相机 | 7.528 | 3.018 | 2510.26125 |

表 8：★ 为近三个月。箭头表示同一论文报告的两个版本（Qwen-Drive 为 SFT → RL）。FROST-Drive 的作者说明其提交因双盲隐藏于公开榜单。
UniPlan 在两份记录里数字不同，此处采用较晚的 FROST 榜单快照。

![Waymo leaderboard](figs/survey/f5_waymo_leaderboard.png)

图 5：表 8 画成散点，按 backbone 的使用方式着色。左上角好。可以看到四件事：整个榜单的 RFS 压在 7.5 到 8.05 之间，
除 FROST-Drive 外 ADE 压在 2.65 到 3.1 之间，差距很小；**唯一一个完全冻结视觉的系统 FROST-Drive，RFS 挤进了微调 VLM 的中游，
ADE 却是全场最差**，说明它选对了大方向但轨迹细节粗；把 backbone 全解冻的小模型 RAP 排第一；
视频生成器路线 SUV 只用一个前视相机也拿到 7.94。

几个榜单外的参照点：zero-shot 直接让 VLM 写轨迹，RFS 只有 5.2 到 6.5，比任何训练过的方法都低两个档次 [40]；
一个社区报告用只看 ego 状态和 route 的 MLP ensemble 在自定义 val 上拿到 7.41，但它用了同一 val 集的其他时刻训练，
不是官方口径 [43]。**官方 test 上没有任何一行 ego-only、constant-velocity 或 route-only 的 baseline。**
从第 4.4 节的角度看，这是榜单上最值得先填的一格。

### 6.3 榜首的代价

| 方法 | 公开的训练配置 |
|---|---|
| RAP | 前期预训练 **4×H100 × 80 h**，不含 Waymo 阶段 |
| Poutine | **4×A100**；预训练 24 h + Waymo 10 h + RL 12 h；另有大模型标注成本 |
| MindVLA-U1 | **8×H200 × 7 h**，50 epochs |
| Fast-dDrive | **8×H100**，3 epochs |
| HMVLM | **8×A100**，3000 iterations |
| NoRD | SFT **16×A100**；Waymo RL **32×A100** |
| AutoVLA | SFT **8×L40S**，5 epochs；另有 RFT |
| FROST-Drive、Qwen-Drive-1.0 | 完整 GPU-hours 未公开 |

表 9：榜单前列的配方大多包含多数据集预训练、全量 VLM 训练、RL、大模型标注和 ensemble，不是「比一个薄 head 大一点」。

### 6.4 有多挤

![crowding timeline](figs/survey/f6_crowding_timeline.png)

图 6：本文引用的论文按主题排在时间轴上，横轴是 arXiv 首次提交的月份，黄色带是最近三个月。
五条线里有四条在最近三个月都有新点：frozen backbone 接 planner 这一行的四个点里三个在带子里；
fast / slow 与 memory 这一行最近三个月来了三篇。**这条线不是没人做，而是同时有很多人在做**。
接下来两节分别讲最后两行。

---

## 7. 快与慢：什么时候需要多想一点

有一个很自然的直觉：人开车大部分时间不需要认真思考，只在少数时刻需要。翻译成系统设计就是
「大多数时候走一条便宜的快路径，必要时才调用大模型」。这个方向在 2025 到 2026 年迅速变成一个小赛道。

![fast slow](figs/survey/f8_fast_slow.png)

图 8：两组配对实验。左：AdaThinkDrive 用同一个 VLA，比较从不思考、总是思考、自适应选择三种策略，
自适应在 NAVSIM v1 上分最高、时间居中 [44]。右：ASSCG 在 nuPlan Hard20 闭环上，比较每帧都调慢模型、固定间隔调用、
学出来的 gate 三种方案；学出来的 gate 和固定间隔一样便宜，分数却最高 [45]。两张图的共同点是：**总是思考并不是上限。**

几篇代表工作各自证明了什么：

| 工作 | 「快 / 慢 / 记忆」的具体含义 | 关键数字 | 边界 |
|---|---|---|---|
| AdaThinkDrive [44]，2025-09 | 同一 VLA 选择 think 或 no-think，用 GRPO 比较两种输出的轨迹质量 | never 88.3 / always 88.9 / adaptive **90.3** PDMS；简单场景 84% 不思考，困难场景 96% 思考 | 三种设置都在处理视觉输入，省的是 CoT，不是 prefill；训练用了 64×H20 |
| CF-VLA [46]，2025-12 | 先给 meta-action，必要时做反事实批评和修正 | adaptive 的 minADE 最好（0.765），avgADE 却不如 no-think（1.561 vs 1.489）；always-think 两项都最差 | 专有数据；不同指标给出不同答案 |
| ASSCG [45]，2026-06 | 序列 gate 决定 Query / Cache / Drop：更新、复用或抑制慢模型的指导 | nuPlan Hard20 闭环 65.00 → **67.28**，每帧成本 0.80 → 0.32 s，慢模型调用率约 19% | 「学会何时调用慢 VLM 并复用旧指导」已经被做了 |
| FIVE-VLA [47]，2026-09 | 不生成 CoT，用一个 recurrent 的 action latent memory 持续保持意图 | Bench2Drive 成功率：无记忆 73.0%，输入过去 10 个 waypoint 70.0%，recurrent memory **77.3%** | 显式历史改善 open-loop 却降低闭环；便宜的 recurrent head 就够用是必须先排除的竞争解释 |
| Driving on Memory [48]，2026-08 | 用同一地点过去驾驶的记忆**代替**当前相机 | NAVSIM 上接近或超过领先系统；Bench2Drive、RealEngine 上明显退化 | 首先是 benchmark audit：NAVSIM 的高分可能主要奖励静态道路规律 |

表 10：fast / slow、动态 gate、缓存旧指导都已有直接先例。这个方向上剩下的问题是「哪些时刻、哪些历史、哪些额外计算真正带来收益，
能不能在付出大模型成本之前识别出来」，而不是再提一个双系统的名字。

Driving on Memory 值得多说一句。它的实验是：把当前相机输入拿掉，换成这辆车（或别的车）过去在同一地点看到的东西，
结果在 NAVSIM 上几乎不掉分。这不是说当前交通不重要，而是说 **NAVSIM 这个 benchmark 对「有没有看当前画面」的检验强度不够**。
对任何声称「视觉表示有用」的工作，这都是一个必须排除的混杂因素。

---

## 8. 证据的台阶，以及延迟怎么比才公平

### 8.1 从「像人」到「会开」中间有几级

一项表示层面的研究可以有价值，而不必证明整辆车会开。关键是让结论和证据的层级对上。

![evidence ladder](figs/survey/f9_evidence_ladder.png)

图 9：常用 benchmark 按「证明了什么」排成台阶。每往右一步都需要新的验证，左边的分数不能直接换成右边的结论。
比如 Waymo RFS 高说明选的轨迹接近人类偏好，但不说明执行之后能纠错；NAVSIM v1 是 non-reactive 的，其他车不会对你的决定做反应。

| Benchmark | 真正测什么 | 已知的限制 |
|---|---|---|
| nuScenes open-loop | 和记录轨迹的 L2 距离、碰撞代理 | ego-status shortcut [30]；不能把低 L2 当驾驶智能 |
| Waymo E2E | 和人工评分轨迹的接近程度（RFS） | open-loop 代理；rater 标签每段只有一帧 |
| NAVSIM v1 | 在 non-reactive 场景里执行候选轨迹，PDMS 综合安全、进展等 | 不是真实多智能体交互；Driving on Memory 显示可被静态记忆利用 [48] |
| NAVSIM v2 | pseudo-simulation，加入偏离专家状态后的评估，EPDMS | 比 v1 难，但仍不是完整闭环；v1 / v2 分数不可互换 [49] |
| Bench2Drive | CARLA 里实际执行策略，DS、成功率、分能力评估 | 2026-08 协议更新，新旧分数不可直接比 [50] |
| Bench2Drive-Robust | 相机失效、ego 状态误差、控制延迟下的闭环 | 是对「薄所以快」这个论点最直接的压力测试 [51] |

表 11：选 benchmark 就是在选要支持的结论。同名指标还要核对实现版本，比如 WA-JEPA v2 把旧口径的 EPDMS\* 和修正后的 EPDMS 分列，
主结果分别是 88.0 和 91.7 [9]。

### 8.2 延迟：head 快不等于系统快

「薄 head 所以实时」是这条线最常见的宣传，也是最容易做假的地方。完整链路包括图像预处理、vision encoder、
LLM prefill（把图像 token 和 prompt 过一遍整个网络）、输出和控制接口；只测 head 的微秒数会漏掉前面全部。
文献里的报告口径五花八门：

| 方法 | 硬件 / batch | 报告的速度 | 计时边界 |
|---|---|---|---|
| Alpamayo-R1 [15] | RTX 6000 Pro Blackwell | 29 / 99 / 312 ms | 含 vision、prefill、action；不是分位数 |
| MindVLA-U1 [52] | H200，B1 | slow 2594 ms；fast 108 ms；action-only 103 ms | 含 vision |
| Fast-dDrive [53] | H100，B1 | 1919 ms；用 SGLang 665 ms | 均值，prefix 复用；无 p95 |
| DiffAdapterVLA [38] | 多 GPU，**batch 8** | 80.4 ms / sample | **批均摊**，不是单请求延迟 |
| Hydra-MDP++ [32] | V100 | 206 / 271 ms | CPU 预处理未说明 |
| DiffusionDrive [33] | RTX 4090 | 45 FPS；去噪部分 7.6 ms | 7.6 ms 不是全流程 |
| WoTE [35] | L20 | 18.7 ms | 文中 total |
| GoalFlow [34] | 未说明 | 1 步 10.4 ms；5 步 49.0 ms | 只算生成 |
| FROST-Drive [21] | 未公开 | 未公开 | 完整协议缺失 |

表 12：不同相机数、帧数、精度、batch 下的数字不能直接排名。可比的最低要求是 batch 1、含 vision 和 prefill、报 p50 和 p95。

还有一个专门针对「取中间层」方案的提醒：**在中间层取特征，不等于提前退出**。如果后面的层照常跑完，就没有任何推理节省；
要真正截断计算、验证输出一致，才能声称省了时间。同样，仿真器愿意等模型算完，不代表模型能跟上真实时间；
Bench2Drive-Robust 把实测的延迟注入控制回路，是检验这一点最直接的办法 [51]。

---

## 9. 结语与阅读路线

回到开头的问题。**驾驶智能藏在哪里？** 文献的答案是：大量可用的信息确实在预训练表示里，冻结它、接一个薄 head，
已经能在 Waymo 上挤进微调 VLM 的中游。**但它的可读性、可执行性和对追加计算的需求不是同一件事。** 探针读得出的东西生成时可能用不上；
选对了大方向的系统轨迹细节可能最差；在某个 benchmark 上「不看当前画面也行」可能只是 benchmark 的问题。
最有价值的工作不是再宣布一次「生成已被证明多余」，而是精确找出：能删什么、不能删什么、删了之后失去什么。

如果只读十篇，建议按这个顺序，每篇解决一个判断：

| 顺序 | 论文 | 先解决的判断 |
|---|---|---|
| 1 | Probing Visual Concepts in Lightweight VLMs for Automated Driving [20] | 逐层探针 + 驾驶 VLM 这个模板已经被占了多少 |
| 2 | Hidden in plain sight [28] | 生成失败为什么不等于表示里没有信息 |
| 3 | Is Ego Status All You Need [30] | 任何 open-loop 结果都要先过 ego-state 这一关 |
| 4 | FROST-Drive [21] | 冻结视觉 + 小 head 在 Waymo 上的直接先例，注意 val 和 test 的区分 |
| 5 | Qwen-Drive-1.0 [16] | 未适配 VLM 接 planner 的最新数字，以及 SFT → RL 在不同指标上的分歧 |
| 6 | Alpamayo-R1 [15] | 轨迹序列化的成本和 reasoning 的成本分开算 |
| 7 | Hydra-MDP [31] | 轨迹词表怎么建、scorer 怎么监督 |
| 8 | Drive-JEPA [8] | 预训练目标对下游 planner 的影响 |
| 9 | ASSCG [45] | 「何时调用慢模型」已经做到哪一步 |
| 10 | Driving on Memory [48] | benchmark 是否真的检验了动态感知 |

---

## 参考文献

1. GAIA-1: A Generative World Model for Autonomous Driving. arXiv:2309.17080, 2023-09.
2. GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. arXiv:2503.20523, 2025-03.
3. DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. arXiv:2309.09777, 2023-09.
4. Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. arXiv:2405.17398, 2024-05.
5. LAW: Enhancing End-to-End Autonomous Driving with Latent World Model. arXiv:2406.08481, 2024-06; v2 2025-02.
6. World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model. arXiv:2507.00603, 2025-07.
7. DriveLaW: Unifying Planning and Video Generation in a Latent Driving World. arXiv:2512.23421, 2025-12; v3 2026-04.
8. Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving. arXiv:2601.22032, 2026-01; v2 2026-07.
9. WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving. arXiv:2608.20974, 2026-08; v2 2026-09.
10. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. arXiv:2402.12289, 2024-02.
11. EMMA: End-to-End Multimodal Model for Autonomous Driving. arXiv:2410.23262, 2024-10.
12. OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model. arXiv:2503.23463, 2025-03.
13. AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. arXiv:2506.13757, 2025-06; v3 2025-11.
14. ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation. arXiv:2503.19755, 2025-03.
15. Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail. arXiv:2511.00088, 2025-10; v2 2026-01.
16. Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving. arXiv:2609.00111, 2026-09. Model card: huggingface.co/Qwen/Qwen-Drive-1.0-4B.
17. Jev Visual. github.com/hr98w/jev-visual, 2026-09 snapshot.
18. NVIDIA, Generate Trajectories, Reasoning Traces, and Auto-Labels with NVIDIA Alpamayo 2 Super. Technical blog, 2026-08.
19. TypeSafe, Introducing System One Models and Jev. typesafe.ai, 2026-09-15.
20. Probing Visual Concepts in Lightweight Vision-Language Models for Automated Driving. arXiv:2603.06054, 2026-03; v2 2026-08.
21. FROST-Drive: Scalable and Efficient End-to-End Driving with a Frozen Vision Encoder. arXiv:2601.03460, 2026-01.
22. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985, 2025-06.
23. DINOv3. arXiv:2508.10104, 2025-08.
24. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. arXiv:2311.16038, 2023-11.
25. DrivingGPT: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive Transformers. arXiv:2412.18607, 2024-12.
26. Which Pretraining Paradigm Better Serves Spatial Intelligence? An Empirical Comparison of Vision-Language and Video Generation Models. arXiv:2605.28132, 2026-05.
27. Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models. arXiv:2601.12626, 2026-01.
28. Hidden in plain sight: VLMs overlook their visual representations. arXiv:2506.08008, 2025-06.
29. Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment (Anchor-Align). arXiv:2607.13429, 2026-07.
30. Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? arXiv:2312.03031, 2023-12.
31. Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation. arXiv:2406.06978, 2024-06.
32. Hydra-MDP++. arXiv:2503.12820, 2025-03.
33. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. arXiv:2411.15139, 2024-11.
34. GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving. arXiv:2503.05689, 2025-03.
35. WoTE: End-to-End Driving with Online Trajectory Evaluation via BEV World Model. arXiv:2504.01941, 2025-04.
36. GTRS: Generalized Trajectory Scoring for End-to-end Multimodal Planning. arXiv:2506.06664, 2025-06.
37. Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving. arXiv:2607.29031, 2026-07.
38. Planning in the Backbone: DiffAdapterVLA for Native Continuous Trajectory Generation with Driving VLMs. arXiv:2609.15322, 2026-09.
39. VL-DPO: Vision-Language-Guided Finetuning for Preference-Aligned Autonomous Driving. arXiv:2605.20082, 2026-05.
40. dVLM-AD. arXiv:2512.04459, 2025-12（其中报告了 OpenEMMA / LightEMMA 在 Waymo test 上的复现分数）.
41. WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios. arXiv:2510.26125, 2025-10; CVPR 2026.
42. Waymo Open Dataset, Vision-based End-to-End Driving Challenge (2025 edition) rules and protos. waymo.com/open/challenges/2025/e2e-driving/, 2026-09 核对.
43. Community ego + intent MLP ensemble report, github.com/manfromnowhere143/perceptionproof, 2026-06-29（非标准 val 协议）.
44. AdaThinkDrive. arXiv:2509.13769, 2025-09.
45. ASSCG. arXiv:2606.25509, 2026-06.
46. CF-VLA (Counterfactual VLA). arXiv:2512.24426, 2025-12.
47. FIVE-VLA. arXiv:2609.18623, 2026-09.
48. Driving on Memory. arXiv:2608.31029, 2026-08. Code: github.com/boschresearch/MemoryDrivoR.
49. Pseudo-Simulation for Autonomous Driving (NAVSIM v2). arXiv:2506.04218, 2025-06.
50. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving. arXiv:2406.03877, 2024-06; 协议更新 2026-08.
51. Bench2Drive-Robust: Benchmarking Closed-Loop Autonomous Driving under Deployment Perturbations. arXiv:2605.18059, 2026-05.
52. MindVLA-U1. arXiv:2605.12624, 2026-05.
53. Fast-dDrive. arXiv:2605.23163, 2026-05.

其他在表 8 中出现、正文未展开的 Waymo 条目：RAP arXiv:2510.04333；Poutine arXiv:2506.11234；SUV arXiv:2608.03084；
HMVLM arXiv:2506.05883；NoRD arXiv:2602.21172；NAVSIM v1 arXiv:2406.15349；ZTRS arXiv:2510.24108。
