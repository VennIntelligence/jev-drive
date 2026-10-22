# 第 9 轮在线环路：文献对照与踩雷清单（2026-10-22 工作笔记）

问题来自 [experiments/round9/plans/2026-10-22-round9-draft.md](../experiments/round9/plans/2026-10-22-round9-draft.md) 第 1 节的用户修订版。
草稿里的环路是：推演者用最新快照在训练场景上开车，产出新 S 行、DAgger 行和逐场景结局；缓冲区按「激活 × 可学性 × 新鲜度」重定权；
场景按当前失败优先抽；P 个学习者各带混合参数 θ，用练习切片上的闭环分做 PBT 式复制和扰动；同时守住已经满分的场景。
背景数字见 [s-rows-and-paper.md](s-rows-and-paper.md) 第 0、3、4 节。本页回答四个问题：
这是不是已知范式、驾驶里谁做过、会踩哪些雷、最小版本用什么默认值。

**核实标记**：[A] 用 arXiv API 核对了 id、标题和一作；[T] 另外读了全文里的相关段落（数字出自这里）；
[S] 只看了摘要或搜索摘要；「出处未核实」= arXiv 已核，会议名凭记忆。

---

## 0. 结论先行

- **范式本身不新**。拆开看：推演者 / 学习者分离是 IMPALA / Ape-X / SEED；在自己访问的状态上聚合、由教师打标签是 DAgger；
  按当前失败选场景是 PLR / SFL；P 个学习者在线调混合参数是 PBT / PB2；模仿锚加后果信号是 DAPG / BC-SAC / RAD 一族；
  守住旧技能是持续学习里的回放加蒸馏。每个零件都有成熟的默认值，也有已知的坑。
- **驾驶里最近的邻居都是 2025–2026 年的**：OPTED（AlpaSim、冻结主干只训解码器、在线蒸馏特权 RL 教师、FIFO 缓冲）几乎是我们 DAgger 支路的同构；
  RoaD（AlpaSim 上一轮闭环 SFT，第二轮反而碰撞翻倍）；RAD（3DGS 上 4 轮 RL 配 1 轮 IL 交替）；Gigapixel（self-play DAgger）；
  CL4AD（GPUDrive 上的 PLR）。**没找到的组合**是：(i) 后果信号是对当前 plan 解析计算的 hinge，不是奖励，也不是教师目标；
  (ii) 在真实日志重建场景上，用闭环分做 PBT 去调**数据混比 / 剂量**；(iii) 把「不打坏已会场景」写进选择适应度。
  这是「没找到」，不是新意证明。三条里 (ii) 最容易被审稿人说成「工程上的超参搜索」，(i) 才是方法核心。
- **对我们最有用的一条结构性观察**：S 行的损失每步都在**当前** plan 上重算，DAgger 行的标签只取决于状态。所以两类行都只有「状态分布旧」，
  没有「标签旧」。RL 里最难的离策略修正（V-trace、重要性比、PPO 截断）对我们基本不适用。旧行的风险只有两条：
  优先级过期，以及状态分布离当前策略太远。这让缓冲区可以放心留旧行，按 DAgger 的理论本来就该留。
- **最大的雷不在算法，在读数**：P = 3、80 个场景、确定性读，单轮适应度差多半是噪声（第 3.1 节）。PB2 论文里 B = 4 的 PBT 跑不过随机搜索，
  PBT 原文也说种群 ≤ 10 时方差大、结果差。最小版本应该先跑「不复制的种群」两三轮，量出学习者之间的差是否大过噪声，再决定开不开 exploit。
- **第二大的雷是遗忘和模仿漂移**，我们自己的历史已经说明了：续训本身伤底座（d259 / d264），轮次聚合到平台（YS2、YSAGG、DAG2），
  聚焦抽样伤 PAI（d386）。文献给的修法很一致：每批都保留原始模仿数据（InstructGPT 发现只加 KL 修不好回退，必须混预训练数据），
  在旧状态上做 BC 或蒸馏（Wołczyk 2024 的比较里 BC 最有效），并且留一个固定 θ 的同步数对照学习者。

---

## 1. 这是不是已知范式：逐族对照

| 族 | 代表作（核实） | 它做什么 | 我们的环路里对应什么 | 差别 |
|:--|:--|:--|:--|:--|
| 在策略 vs 离策略 RL | PPO，Schulman et al., arXiv:1707.06347 [A] | 在策略：只用当前策略的数据，用完即弃 | 推演者用最新快照滚出的行 | 我们**不**用回报估计梯度；S hinge 是对当前 plan 的解析梯度，行可以无修正地重复使用，所以整体上是「离策略的状态 + 在策略的损失」 |
| 推演者 / 学习者架构 | IMPALA，Espeholt et al., ICML 2018, arXiv:1802.01561 [A]；Ape-X，Horgan et al., ICLR 2018, arXiv:1803.00933 [A]；R2D2，Kapturowski et al., ICLR 2019（OpenReview，无 arXiv，出处未核实）；SEED RL，Espeholt et al., ICLR 2020, arXiv:1910.06591 [A] | 推演者异步滚、学习者集中训；IMPALA 用 V-trace 修策略滞后，Ape-X 用带优先级的共享回放 | 池里的 rollout fan-out 作业 + 训练作业 | 我们的推演者贵（3DGS 渲染，PAI 约 200 场景 / 卡时），学习者便宜（1 000 步约 0.07 卡时）。比例和 Atari 正好反过来：**瓶颈在仿真，不在学习者** |
| DAgger 与在线模仿 | DAgger，Ross, Gordon, Bagnell, AISTATS 2011, arXiv:1011.0686 [A]；AggreVaTe，Ross & Bagnell, arXiv:1406.5979 [A]；SMILe，Ross & Bagnell, AISTATS 2010（"Efficient Reductions for Imitation Learning"，无 arXiv，出处未核实）；HG-DAgger，Kelly et al., arXiv:1810.02890 [A] | 学生开、专家在学生的状态上打标签、聚合所有轮次的数据 | DAgger 行（地图教师）就是 DAgger；S 行保留「学生访问的状态」，去掉专家标签 | DAgger 的无悔保证要求**聚合**、每轮都训到收敛；我们是小轮次、不收敛，更接近「在线梯度版 DAgger」。AggreVaTe 的「按 cost-to-go 加权」对应 PLR 的场景优先 |
| PBT 及其后代 | PBT，Jaderberg et al., arXiv:1711.09846 [T]；PB2，Parker-Holder, Nguyen, Roberts, NeurIPS 2020, arXiv:2002.02518 [T]；FIRE PBT，Dalibard & Jaderberg, arXiv:2109.13800 [T] | 并行训练、按适应度复制权重和超参，再扰动 | P 个学习者，θ = (S 压力, DAgger 份额, …) | PBT 原文的种群是 10–80（RL）、32（翻译）、45（GAN）；我们 P = 3，远在它说的「方差大、结果差」区间（见 3.1） |
| PLR / 无监督环境设计 | PLR，Jiang, Grefenstette, Rocktäschel, ICML 2021, arXiv:2010.03934 [T]；Robust PLR，Jiang et al., NeurIPS 2021, arXiv:2110.02439 [T]；ACCEL，Parker-Holder et al., ICML 2022, arXiv:2203.01302 [T]；PAIRED，Dennis et al., NeurIPS 2020, arXiv:2012.02096 [A]；SFL / "No Regrets"，Rutherford et al., arXiv:2408.15099 [A]（NeurIPS 2024，出处未核实） | 按「学得动的程度」给关卡打分，按排名加温度抽，并混入「陈旧度」分布防止分数过期 | 场景按当前失败 / 部分分优先抽，满分场景低频复查 | 我们的关卡是真实日志重建的固定场景，不生成新关卡（没有 ACCEL 的编辑器，也没有 PAIRED 的对手）；结局是确定性的单次读数，不是回报分布 |
| 优先经验回放 | PER，Schaul et al., ICLR 2016, arXiv:1511.05952 [A]；RHO-LOSS，Mindermann et al., ICML 2022, arXiv:2206.07137 [A] | 按误差抽样 + 重要性修正；只选「可学、值得学、还没学会」的点 | 行权重 = 激活 × 可学性 × 新鲜度 | 我们的「可学性」是人工规则（t0 出栅格、人的路径也违反就判不可学），RHO-LOSS 用留出模型估；没有重要性修正时就是在改目标（s-rows 第 4 节已说明） |
| 带模仿锚的 RL 微调 | DAPG，Rajeswaran et al., RSS 2018, arXiv:1709.10087 [A]；AWAC，Nair et al., arXiv:2006.09359 [A]；IQL，Kostrikov, Nair, Levine, ICLR 2022, arXiv:2110.06169 [A]；TD3+BC，Fujimoto & Gu, NeurIPS 2021, arXiv:2106.06860 [A]；RLPD，Ball et al., ICML 2023, arXiv:2302.02948 [T]；Cal-QL，Nakamoto et al., NeurIPS 2023, arXiv:2303.05479 [A]；BC-SAC，Lu et al., arXiv:2212.11419 [T]（IROS 2023，出处未核实） | 演员目标 = RL 项 + λ·模仿项，或从离线数据起步的在线 RL | 模仿行（navtrain、PAI 人类）+ S hinge + DAgger | BC-SAC 的奖励是「到最近他车框的距离减 1 m 取负」和「到路沿距离减 1 m 截断」，**和我们的 agent / road hinge 几乎同形**，只是它进的是 Q 函数，我们直接对 plan 求导 |
| 自我改进 / 专家迭代 | ExIt，Anthony, Tian, Barber, NeurIPS 2017, arXiv:1705.08439 [A]；ReST，Gulcehre et al., arXiv:2308.08998 [A]；ReST-EM，Singh et al., TMLR, arXiv:2312.06585 [A] | 生成、按奖励筛选、在筛出的样本上监督训练，迭代 | 我们试过的 BEST-OF-N / S3 后果行（d379 / d422 / d423：不过） | 我们已经知道「自己的 rollout 里没有好起步」，所以专家迭代那条在我们这里断了，靠的是 hinge 和地图教师 |
| 持续学习 / 稳定性与可塑性 | EWC，Kirkpatrick et al., PNAS 2017, arXiv:1612.00796 [A]；LwF，Li & Hoiem, ECCV 2016, arXiv:1606.09282 [A]；Policy Distillation，Rusu et al., arXiv:1511.06295 [A]；Kickstarting，Schmitt et al., arXiv:1803.03835 [A]；Progress & Compress，Schwarz et al., ICML 2018, arXiv:1805.06370 [A]；Policy Consolidation，Kaplanis, Shanahan, Clopath, ICML 2019, arXiv:1902.00255 [A]；CLEAR，Rolnick et al., NeurIPS 2019, arXiv:1811.11682 [A]；Wołczyk et al., ICML 2024, arXiv:2402.02868 [T] | 回放旧数据、对旧策略做 KL 蒸馏、对重要参数加二次罚 | 「守住满分场景」+ 保留 navtrain 行 | Wołczyk 2024：RL 微调时模型在「暂时没访问到的状态」上会忘掉预训练能力；比较 BC（在旧状态上对旧策略的 KL）、episodic memory 和 EWC，**BC 最有效**，EWC 在大网络上常失效 |
| 在线混比 | DoReMi，Xie et al., NeurIPS 2023, arXiv:2305.10429 [A]；Online Data Mixing，Albalak et al., arXiv:2312.02406 [A]；Data Mixing Laws，Ye et al., ICLR 2025, arXiv:2403.16952 [A] | 按代理目标在线调各域权重（DRO 或多臂老虎机） | θ 中的 DAgger 份额、PAI 人类份额 | 它们的反馈是各域的训练损失；我们的反馈是闭环分，贵约 10³ 倍，所以只能用 PBT / 老虎机这类粗粒度控制，不能每步调 |

**我们的组合里，什么已知，什么没见过：**

- 已知：推演者 / 学习者分离、DAgger 聚合、按失败抽场景、PBT 调超参、模仿锚、回放旧数据防遗忘。都有现成默认值（第 4 节直接搬）。
- 驾驶里有先例但形态不同：在线状态 + 教师标签（OPTED、Gigapixel、RoaD 的专家引导）；按难度抽场景（Bronstein 2022、BC-SAC 的 Top10、CL4AD）；
  模仿 + 后果交替（RAD）。
- 没找到：
  1. 后果信号以**无标签的解析 hinge** 进入在线环路（S 行），推演者不需要奖励、价值函数，也不需要教师；
  2. PBT 的「超参」是**数据混比和 S 剂量**，适应度是真实日志重建场景上的配对闭环分，并在适应度里扣「新打坏的已会场景」；
  3. 用种群内 P 个快照在同一场景上的结局方差估「可学性」（SFL 的 p(1−p) 本来要对同一策略多次滚，我们的读数是确定性的，种群天然提供多次采样）。这一条是本页提出的建议，文献里没见过。
- 审稿风险：(2) 容易被看成工程调参；(1) 需要跟 OPTED 式「教师在线蒸馏」正面比，才能说明「没有教师也行」的价值。

---

## 2. 端到端驾驶里谁在做闭环训练环路

| 工作 | 仿真 / 规模 | 训练信号 | 模仿锚 | 结果 | 和我们的距离 |
|:--|:--|:--|:--|:--|:--|
| **OPTED**，Da Col, Igl, Karkus, Chitta, … Pavone, Schindler, Sakaridis, arXiv:2609.20756 [T] | AlpaSim + NuRec 3DGS，1 669 训练 / 441 评测场景，10 Hz、20 s，他车按日志回放 | 每个访问状态拷到无渲染仿真器里，由 PPO 训的特权教师（PufferDrive 2.0，WOMD 预训 + PAI 地图适配 5 亿步）往前开 K 步，得到 plan 目标；动作头用前向 KL，航点头用加权 L1 | **没有**混原始 BC 数据；冻结主干只训规划解码器；每轮收 32 场景、训 512 场景的 FIFO 缓冲，共 2 个 epoch | AlpaSim 场景分 LTFv6 26.5 → 41.8，VaVAM 3.9 → 37.1；教师 64.8。**副作用**：nuPlan 开环 ADE 0.74 → 1.22，LTFv6 原始 at-fault 碰撞 10.2 % → 23.4 %（作者说按里程归一后相当）；一轮 VaVAM 推演 524 s、标注 13.5 s、梯度 6 s（8×H100） | **最接近**。我们的 DAgger 支路 = 它的「地图教师版」。它的 6 s 梯度对 524 s 推演，印证「瓶颈在仿真」；开环 ADE 变差正是我们要防的模仿漂移 |
| **RoaD**，Garcia-Cobo, Igl, Karkus, …, arXiv:2512.01993 [T]（CVPR 2026 Findings，出处未核实） | AlpaSim，约 8 251 个 3DGS 场景生成数据，每场景 3 次；评 920 场景 × 3 | 策略自采 K = 64 条（温度 0.8），执行离专家最近的一条；偏离 > 3 m 就在 30 步内拉回专家；在执行的轨迹上做 BC | 没混原始数据，约 1 个 epoch，冻结编码器；作者自己推测 1 万步时过拟合是因为没和原始大数据共训 | 驾驶分 0.444 → 0.630，碰撞 0.0525 → 0.0239；**第二轮重采 + 再训：分 0.661，但碰撞升到 0.0420** | 一轮有效、第二轮伤安全，和我们 YS2 的轮次闸不过同形 |
| **RAD**，Gao et al., arXiv:2502.13144 [T] | 4 305 个 3DGS 场景（3 968 训练），32 个 RL worker 各占一张 4090 | PPO（γ 0.9、λ 0.95，横 / 纵向分开截断）；奖励 = 动 / 静态碰撞、位置偏离 > 2 m、航向偏离 > 40°，任一即终止 | **4 轮 RL 配 1 轮 IL 循环交替**（RL 批 32、IL 批 128），作者说 4:1 最好（正文没给比例消融表） | 碰撞率：只 IL 0.229，只 RL 0.143（偏离 0.345），RL + IL 0.089（偏离 0.257） | 交替日程 + 模仿锚的直接前例；只 RL 会偏离日志更多 |
| ReconDreamer-RL，Ni et al., arXiv:2508.08170 [A]（数字引自 [lit/2026-10-08-top-methods-alpasim-closed-loop.md](lit/2026-10-08-top-methods-alpasim-closed-loop.md)） | 视频扩散修 off-path 视图的重建环境 | RL | — | 碰撞 0.077 对 RAD 0.238 | 说明离日志越远渲染越差，S 状态的渲染可信度是隐患 |
| GSDrive，Guo et al., arXiv:2604.28111 [S] | 3DGS 环境，nuScenes | 多模态未来轨迹探测 + RL | 未读 | 未读 | 只知道存在 |
| **World Engine**，Li, Chen, Wang, …, arXiv:2606.19836 [A]（数字引自同上笔记） | 从日志重建并外推出安全关键变体；AlpaSim nuPlan 赛道的渲染来源 | RL 后训练 | — | rare 闭环 SR：底座 73.66 → **只在常见日志上 RL 69.63（变差）** → 在稀有日志上 RL 73.35 → 全流程 88.89 | 场景选择决定 RL 有没有用：在已会的场景上练会伤，正是 PLR 的出发点 |
| **Gigapixel / Scaling Self-Play for E2E Driving**，Rowe, Girgis, …, Vinitsky, Pal, Paull, arXiv:2606.19641 [S] | 批量化方框世界透视渲染，约 5 万智能体步 / 秒 | 「self-play DAgger」：像素策略在线蒸馏特权 RL 教师（作者说像素空间直接 RL 太不省样本） | 无人类轨迹监督 | HUGSIM、NAVSIM-v2 上有竞争力（摘要没给数） | 大规模版本的「在线状态 + 教师」，把真实感知换成轻量适配 |
| **GIGAFLOW**，Cusumano-Towner et al., ICML 2025（PMLR 267），arXiv:2502.03349 [S] | 16 亿 km 自博弈，单台 8 卡约 42 年驾驶 / 小时 | 自博弈 RL | 完全不用人类数据 | 三个基准 SOTA，平均 17.5 年一次事故（仿真内） | 特权输入；证明「后果能教会执行」，但不碰真实感知 |
| **CaRL**，Jaeger, Dauner, …, Chitta, Geiger, CoRL 2025, arXiv:2504.17838 [T] | CARLA 3 亿、nuPlan 5 亿样本，单台 8 卡 | PPO，奖励 ≈ 路线完成 × 违规惩罚因子 | 无模仿 | Longest6 v2 DS 64；nuPlan Val14 91.3 / 90.6 | **关键教训**：多项相加的复杂奖励在 mini-batch 变大时 PPO 卡进局部最优（批大 4 倍性能大跌），简单奖励才能放大批。对应我们的 hinge 项越加越多的风险 |
| Think2Drive，Li, Jia, … , ECCV 2024, arXiv:2402.16720 [A]；Roach，Zhang et al., ICCV 2021, arXiv:2108.08265 [A]；PlanT，Renz et al., CoRL 2022, arXiv:2210.14222 [A] | CARLA，特权 BEV 或对象级输入 | 世界模型里的 RL / PPO 教师 / 对象级模仿 | 学生通过模仿教师得到（Roach → TCP 等） | B2D 上的专家来源 | 「特权教师 + 蒸馏」这条线的源头；没有在线混比和场景优先 |
| CarPlanner，Zhang et al., CVPR 2025, arXiv:2502.19908 [A] | nuPlan | 自回归规划 + RL | IL 预训 | s-rows 笔记已引：可微可行驶面损失改善开环、伤闭环 | 直接警告我们的 road hinge |
| Hydra-MDP，arXiv:2406.06978 [A]；GTRS，arXiv:2506.06664 [A]；ZTRS，arXiv:2510.24108 [A] | NAVSIM | 把 PDM 子分数蒸成打分头，对词表打分 | 日志状态 | NAVSIM 冠军 | 打分器路线，没有在线状态（我们 d324 已关） |
| **TOAD**，Xu, Zablocki, Yin et al.（valeo.ai），"Test-Time Trajectory Optimization for Autonomous Driving"，arXiv:2606.07170 [A] | navhard | 测试时用打分器搜索 / 优化轨迹，不更新模型 | — | 已在 [lit/2026-10-08-top-methods-verification.md](lit/2026-10-08-top-methods-verification.md) 核过 | 草稿第 3 节的外挂，和环路正交 |
| BC-SAC，Lu, Fu, Tucker, …（Waymo），arXiv:2212.11419 [T] | 10 万英里以上城市日志，闭环模拟 | SAC + λ·BC；奖励 = 碰撞距离 hinge + 出路距离 hinge | 模仿项本身 | 最难场景失败降 38 % 以上；训在 Top10 难度最好，**只训 Top1 明显更差** | 按难度聚焦要留余量：只练最难的会坏 |
| Bronstein et al.（Waymo），CoRL 2022, arXiv:2212.01375 [S] | 车队数据 | 难度分类器 + 按难度出课程的模仿 | 模仿 | 只用 10 % 数据的规划器和用全量一样好；按难度优先：碰撞 −15 %，路线遵从 +14 % | 「均匀喂满数据浪费」的驾驶证据 |
| Hierarchical MBIL，Bronstein et al., IROS 2022, arXiv:2210.09539 [A] | 可微仿真里的闭环模仿 | 闭环模仿损失 | — | — | Waymo 的闭环模仿 |
| Rethinking Closed-loop Training，Zhang et al.（Waabi），ECCV 2022, arXiv:2306.15713 [A] | 闭环训练的场景分布研究 | RL | — | — | 结论方向：练什么场景比算法更重要（只看了摘要层面） |
| CAT-K，Zhang, Karkus, Igl et al., CVPR 2025, arXiv:2412.05334 [A] | WOSAC 交通模型 | 闭环 SFT：滚 K 条取最近 GT 的一条 | — | — | RoaD 的前身；AlpaGym 的反应式他车用它 |
| TakeAD，Liu et al., RA-L, arXiv:2512.17370 [A] | Bench2Drive | 迭代 DAgger（接管数据）+ DPO | IL 预训 | 闭环优于纯 IL | B2D 上的迭代 DAgger 先例 |
| CL4AD，Koprulu et al., arXiv:2608.22549 [T]（NeurIPS 2026 在审） | GPUDrive，最多 8 万场景 | PLR 系（AMGAE、PVL、MaxMC、learnability、learn-hard） | — | 99 % 成功率提前约 10 亿步，墙钟 −77 % | 驾驶里第一个规模化 PLR；超参在第 4 节 |
| AlpaGym（NVIDIA，HF 博客 "Taking Alpamayo to New Heights…"）[S] | AlpaSim 之上，单卡到多节点 | GRPO，默认奖励；他车 CAT-K | 文中没写 | 文中没写 | AlpaSim 官方的 RL 框架，没给架构细节和数字 |
| Learning to Drive from a World Model，Goff, Hogan, Hotz et al., arXiv:2504.19077 [S] | 重投影仿真 / 学到的世界模型，在策略 | 在策略训练，无手写规则 | 真实人类驾驶数据 | 部署到真实 ADAS（摘要没点名 openpilot；作者是 comma.ai，按作者推断） | 我们的编码器来自同一家 |
| ReCogDrive，arXiv:2506.08052 [A]；HAD，arXiv:2604.03581 [S]；WorldRFT，AAAI 2026, arXiv:2512.19133 [A]；SimScale，CVPR 2026, arXiv:2511.23369 [A]；R2SE，arXiv:2506.09800 [A] | NAVSIM 伪仿真 / 世界模型 | GRPO 类 RL 微调、仿真数据共训 | 有 | NAVSIM 分数 | 「RL 后训练」流行但多是**单步伪仿真**，不是多轮闭环环路 |

**Waymo / Wayve**：Waymo 公开的做法是 BC-SAC（模仿 + 带 hinge 形奖励的 SAC，聚焦难场景）、难度课程（Bronstein 2022）和可微仿真闭环模仿；
没找到他们公开的「持续在线环路」配方。Wayve 只找到 MILE（Hu et al., NeurIPS 2022, arXiv:2210.07729 [A]，离线、基于模型的模仿），
没找到公开的闭环训练实践，标为未知。

**为什么通用 e2e 配方里没有这种环路**（文献里自述的理由，加上我们的推断）：
1. **仿真贵**：OPTED 一轮推演 524 s、梯度 6 s；RAD 用 32 张卡跑推演。传感器级闭环比开环训练贵两到三个数量级，大部分团队只做一到两轮（RoaD、World Engine）。
2. **渲染偏离**：离日志越远，3DGS 越假（ReconDreamer-RL 的整篇动机；RoaD 自述「对渲染瑕疵敏感」）。在线状态本来就是偏离日志的状态，学到的可能是仿真器的毛病。
3. **奖励设计**：CaRL 说复杂奖励不能放大批；CarPlanner 说可微几何损失伤闭环；BC-SAC 用模仿项兜底奖励设计。
4. **多轮不稳**：RoaD 第二轮碰撞翻倍；World Engine 在常见日志上 RL 反而掉分；OPTED 开环 ADE 变差。这些都说明多轮环路需要模仿锚和场景选择，而多数论文只报一轮。
5. **sim2real**：Gigapixel、GIGAFLOW、CaRL 都要额外的感知适配或干脆不碰真实感知；我们的冻结 Cinque 主干绕开了这一条，是我们的结构性优势。

---

## 3. 踩雷清单

### 3.1 小评测集上的 PBT 适应度噪声（最大的雷）

- **证据**：PBT 原文：种群 ≤ 10 时「方差大、结果差」，因为 PBT 是贪心的；PB2 原文：B = 4 时 PBT **跑不过随机搜索**，并把原因归为依赖大种群做探索。
  FIRE PBT：PBT 偏好短期提升，典型表现是一上来就把学习率降掉，最终更差。我们这边：281 考场上同配方不同 seed 差约 0.03（S 的 4 个 seed 0.463–0.495），
  80 个场景的标准误约是 281 个的 √(281/80) ≈ 1.9 倍；而 1 000 步内两个学习者真实差多半 < 0.03。
  另一个常被忽略的数：在 281 上，即使是赢 +0.09 的 S 配方，每个 seed 也会**新打坏约 10 个场景**、修好约 44 个（d247 补记）。所以确定性读数下的「新坏」本身就带大量随机翻转。
- **修法**：
  1. 用 PBT 原文的 **t 检验选择**，不用截断选择：对两个学习者在练习切片上做逐场景配对差，配对 bootstrap 的 CI 不含 0（或差 > 1.5 SE）才复制；
     报告用 IQM / 分层 bootstrap（Agarwal et al., NeurIPS 2021, arXiv:2108.13264 [A]）。
  2. **适应度跨轮平滑**：用最近 2 轮的 EMA；ready 间隔 ≥ 2 轮。
  3. **防短视**：FIRE PBT 的思路是让一部分成员按「改进速率」而不是当前分被评；P = 3 时做不了子种群，最便宜的替代是**不允许 PBT 扰动 lr**（草稿 θ 里有 lr，建议拿掉）。
  4. **P = 3 时考虑 PB2**：它就是为小种群设计的（GP 老虎机选新 θ，不靠随机扰动）。但 θ 只有两维时，带配对检验的手工两步搜索也够用。
  5. **练习切片被选择过拟合**（赢家诅咒）：适应度反复用同一 80 场景选，最后选出的是「在这 80 个上运气好的」。281 整轮不碰是对的；再留第二个从不参与选择的训练外切片，每 3–4 轮读一次，看两者是否同向。
- **新坏护栏的写法**：不用「新坏数」的绝对值，用「新坏 − 同步数对照学习者的新坏」；「已会场景」定义为最近 3 个快照里至少 2 个满分的场景，单次满分不算。

### 3.2 在线微调的遗忘和模仿漂移

- **证据**：InstructGPT（Ouyang et al., arXiv:2203.02155 [T]）：PPO 后公开任务回退，**把 KL 系数加到默认的 100 倍（2.0）也修不好**，必须在每个 minibatch 混预训练梯度（γ = 27.8，预训练样本是 RL 回合数的 8 倍）。
  Wołczyk 2024：RL 微调忘的是「暂时没被访问的状态」上的能力，在旧状态上对旧策略做 BC 最有效。OPTED 不混原始数据，开环 ADE 0.74 → 1.22。
  RoaD 作者把过拟合归因于没和原始数据共训。RL's Razor（Shenfeld et al., arXiv:2509.04259 [A]）和 Retaining by Doing（Chen et al., arXiv:2510.18874 [A]）：
  在策略数据忘得少，遗忘量由新任务分布上与底座的 KL 决定。我们这边：续训本身伤底座（d259：ON-NONE −0.092 / −0.028）；S 剂量是倒 U（d311、d347、d348）。
- **修法**：
  1. **每批都保留 navtrain + PAI 人类模仿行**（按现在的份额），不要让 PBT 把模仿份额压到底。草稿 θ 里的「PAI 人类份额」要设下限。
  2. 可选的第二道锚：对冻结的 R6DAG（或上一轮最优快照）在 navtrain 状态上做 plan 蒸馏（LwF / kickstarting 式）。权重标定到它的梯度约是模仿梯度的 0.1–0.2。只在 3.2 的监测量变差时再开。
  3. **同步数对照**：P 个学习者里留一个 θ 固定为 R6DAG 配方、永不被复制的成员。它同时回答「是环路有用，还是多训了 N 步」（d264 的教训）。
  4. 每轮记 dev 开环 ADE 和 PAI 人类行 ADE，相对 0 步超过 +0.03 就报警（沿用 s-rows E1 的闸）。
  5. 可塑性丢失（Nikishin et al., ICML 2022, arXiv:2205.07802 [A]；Lyle et al., ICML 2023, arXiv:2303.01486 [A]；Dohare et al., arXiv:2306.13812 [A]）在几万步、小适配器的规模上不是首要问题，先不管。

### 3.3 hinge 形目标被「钻空子」

- **证据**：Gao, Schulman, Hilton, arXiv:2210.10760 [A]：代理目标优化过头后真实目标先升后降；Skalse et al., arXiv:2209.13085 [A]：形式化定义了可被钻空子的代理目标。
  CarPlanner：可微可行驶面损失改善开环、伤闭环。我们这边：静止行造成恒定右推（d377）；R6-REC 出路 26 / 22（d437）；几何目标都伤避让或出路。
  失分表里 61 % 是纵向（冲停车点 19.5、被追尾 / 慢 9.3）。footprint hinge 最便宜的满足方式是**缩短 plan（减速）或者横向躲开**，正好会被 PBT 的适应度放大：
  只要闭环分短期涨，PBT 就会把 S 压力推高。
- **修法**：
  1. 保持 shape-only 梯度（已有）；每轮监测 S 行上 plan 长度 / 末端速度相对人类日志的比值、横向偏移分布，以及「慢 / 被追尾」类零分数；任一单调漂移就冻结 S 压力的扰动方向。
  2. 适应度里分开记碰撞、出路、慢 / 被追尾三类零分，不只看均分；PBT 的复制条件加「三类零分都不比被替换者差 3 个以上」。
  3. 给 S 压力设上限 5×：倒 U 已经扫过，不让 PBT 再去探 8×。
  4. CaRL 的教训：hinge 项不要再加。每加一项，PBT 就多一个可以钻的方向，大批训练也更容易卡住。

### 3.4 回放行的过期

- **证据与判断**：RL 里的过期是「行为策略 ≠ 当前策略」，要靠 V-trace / 重要性比 / PPO 截断修。Fedus et al., ICML 2020, arXiv:2007.06700 [T]：
  回放效果取决于回放容量和「缓冲里最老策略的年龄」，这种依赖主要通过 n 步回报起作用；不用 n 步回报的组件对容量不敏感。
  AReaL（Fu et al., arXiv:2505.24298 [T]）在异步 LLM RL 里设最大陈旧度 η（数学 8、代码 4 个版本）并改用解耦 PPO 目标。
  **我们的 S 行没有回报，hinge 在当前 plan 上重算；DAgger 标签只取决于状态。** 所以只有两件事会过期：(a) 优先级（激活、可学性）；(b) 状态分布离当前策略多远。
  (b) 在 DAgger 理论里不算错，反而是要保留的；s-rows 第 1 节也测过：9 500 步内激活几乎不变。
- **修法**：
  1. 优先级每轮用当前最优快照前向刷新（便宜，约 0.2 卡时一次）；不必每 N 步刷。
  2. 新鲜度衰减用来调份额，不用来删行：半衰期 3–5 轮，下限 0.25，旧行永远有被抽到的机会（这同时是持续学习里的回放）。
  3. 推演快照最多落后 1 轮（类似 AReaL 的 η）。我们的瓶颈在推演，学习者等推演是常态，所以这条几乎自动满足。
  4. 每批里新一轮的行最多占 S 抽取的一半（借 RLPD 的对称采样：每批 50 % 回放、50 % 离线，加 LayerNorm 防值函数外推），防止最新一轮的状态分布独占梯度。

### 3.5 PLR 课程坍塌

- **证据**：SFL（Rutherford et al., arXiv:2408.15099 [A]）：PLR 的实际打分和遗憾不相关，反而和成功率相关，大量经验花在已经会的环境上；
  他们改为直接按可学性 p(1−p) 抽。CL4AD：批量规模下分数很快过期，要用更高温度（β = 2–4）缓解；小缓冲早期碰撞和出路更多；PLR 版更快到目标，但离人类风格更远。
  PLR 原文：只用分数（ρ = 0）或只用陈旧度（ρ = 1）都明显更差，两者混合才好。BC-SAC：只训 Top1 难度比 Top10 差。
  我们这边：零分行里 2/3 的 hinge 量学不动（s-rows 第 4 节）；聚焦抽样伤了 PAI（d386）。
- **坍塌的具体样子**：优先级集中到「渲染坏了 / 起点就不可救 / 地图教师错了」的场景，这些场景永远失败、永远高优先，抽走推演预算，产出的行又学不动。
- **修法**：
  1. 打分用可学性而不是失败：场景分 = s̄(1−s̄)，其中 s̄ 是**P 个快照**在该场景上的平均分（部分分直接算），再加「最近两轮有变化」的加成；P 个快照都 0 分且已持续 3 轮的，降为低频复查。
  2. 混入陈旧度（ρ = 0.3）和均匀抽样（重放比例 0.5，即一半推演场景从全训练集均匀抽）。
  3. 单场景份额上限（沿用 STOPPT 的 5 %），满分场景按陈旧度低频复查，保证「遗忘」能被看到。
  4. 不可救的起点（t0 已出栅格、渲染失败）在场景层面直接剔除，和行层面的可学性规则同口径。

### 3.6 新旧技能的配比

- **证据**：RAD 4 轮 RL 配 1 轮 IL；RLPD 每批 50 / 50；InstructGPT 模仿（预训练）样本是 RL 的 8 倍、梯度乘 27.8；BC-SAC 用 λ 调；
  我们的倒 U：S 压力 1.9–2.9× 最好。持续学习侧：CLEAR（回放 + 对旧行为的 KL）、Progress & Compress（新知识在活跃列学、再蒸进知识库并加 EWC）。
- **修法**：配比按「梯度份额」定义，不按行数（草稿里的 S / 模仿梯度比就是这个意思）；模仿梯度份额不低于约 0.6。
  「把尽量多的知识装进 planner」在文献里对应的是**最后一步的整合**：把种群里几个最好快照的知识用蒸馏或权重平均（Model soups，Wortsman et al., ICML 2022, arXiv:2203.05482 [A]）合成一个；
  草稿里还没有这一步，建议放到最后一轮再试（同起点、同架构的 PBT 成员满足 soups 的前提）。

### 3.7 渲染型仿真下的吞吐和 GPU 占用

- **证据**：我们的读数：PAI 约 200 场景 / 卡时（281 读约 1.4 卡时），nuPlan 约 2 600 场景 / 卡时；训练 1 000 步约 0.07 卡时。
  一轮里一个学习者的 80 场景 PAI 练习读（约 0.4 卡时）是它训练的约 6 倍。OPTED 是 524 s 推演对 6 s 梯度。
  Sample Factory（Petrenko et al., ICML 2020, arXiv:2006.11751 [A]）用双缓冲让推演和推理重叠；DD-PPO（Wijmans et al., ICLR 2020, arXiv:1911.00357 [A]）用「抢占阈值」截掉拖尾的 worker；
  SEED 把推理集中在学习者卡上；AReaL / Async RLHF（Noukhovitch et al., ICLR 2025, arXiv:2410.18252 [A]）用有界陈旧度的完全异步。
- **修法**：
  1. **推演即适应度**：每轮让 P 个快照在**同一批** PLR 选出的训练场景上开。这批读数既产新行、给 PLR 更新可学性，也是配对的适应度（再加固定 80 场景做锚）。不要为适应度再单独开一遍。
  2. **异步 PBT**：学习者不等读数，继续训下一轮；读数回来再决定复制，复制最多滞后 1 轮。
  3. **截尾**：推演作业按场景分块，轮次在 85–90 % 的块完成时就推进，剩下的块归下一轮（DD-PPO 的抢占思路），否则单个渲染器失败会让整轮等待。
  4. 每轮作业重启的固定开销（模型、数据加载）先量；超过一轮墙钟的 10 % 再考虑常驻学习者。默认仍用池子的 `--after` 链和 `cl fanout`，不另起调度器。

---

## 4. 最小可行版本的默认值（P = 3、θ = (S 压力, DAgger 份额)、约 1 000 步一轮、80 场景确定性练习切片）

| 项 | 建议默认值 | 出处 / 理由 |
|:--|:--|:--|
| 种群构成 | 学习者 A：θ 固定为 R6DAG（2.4×，DAgger 8 行 / 批），永不被复制；B、C 参与 exploit / explore | 同步数对照（d264），也是 PBT 的基线 |
| θ 初值与范围 | S 压力 2.4×，范围 [1.2, 5]；DAgger 8 行 / 批，范围 [4, 24]；对数空间 | 倒 U（d311 / d348）；上限防 3.3 的钻空子 |
| explore | 乘 0.8 / 1.2（PBT 原文）；以 0.2 概率在范围内重采；不扰动 lr | PBT 原文；FIRE PBT 的短视问题 |
| exploit | t 检验式：C 和 B 在「本轮推演场景 + 80 练习场景」上逐场景配对，bootstrap CI 不含 0 且点估计 ≥ 0.02 才复制；ready ≥ 2 轮；适应度用 2 轮 EMA | PBT 原文的 t 检验选择；3.1 的噪声估算 |
| 适应度 | 配对均分 − λ·(新坏 − A 的新坏)，λ 取 1 个场景 = 1/80 分；三类零分（碰撞 / 出路 / 慢）分开记，任一比被替换者多 3 个以上就不复制 | 3.1、3.3 |
| 模仿锚 | navtrain + PAI 人类行份额不动；模仿梯度份额 ≥ 约 0.6；蒸馏到 R6DAG 默认关，监测量报警时开（梯度约为模仿的 0.1–0.2） | InstructGPT、Wołczyk 2024 |
| 回放 | 所有轮的行都留；新鲜度半衰期 4 轮、下限 0.25；最新一轮的行最多占 S 抽取的 50 %；优先级每轮刷新一次 | DAgger 聚合；RLPD；3.4 |
| 陈旧度上限 | 推演快照落后 ≤ 1 轮；优先级 ≤ 1 轮 | AReaL η 的思路，我们的损失不需要重要性修正 |
| PLR | 排名优先；温度 β = 0.3；陈旧度 ρ = 0.3；重放比例 0.5（另一半均匀）；打分 = 种群均分 s̄(1−s̄) + 变化加成；单场景份额 ≤ 5 %；连续 3 轮全种群 0 分的降为低频 | PLR（β 0.1–0.3、ρ 0.1–0.3）；Robust PLR 表（p 0.5、β 0.3、ρ 0.3）；CL4AD（p 0.5、ρ 0.1–0.3，β 大规模才用 2–4）；SFL |
| 每轮推演量 | 每个快照 80 个 PLR 选出的训练场景 + 80 个固定练习场景 → 每轮约 3 × 160 = 480 场景，约 2.4 卡时；5 张卡里 3 张跑学习者时，推演决定墙钟 | 用 200 场景 / 卡时估 |
| 早停 | 3 轮后 B、C 都不比 A 好（配对 CI），或 A 本身在练习切片上掉 > 0.03，就停 PBT，转回固定 θ 的对照臂 | research-method 的早停闸 |

**先测什么（按顺序，前两项全是 CPU）**：

1. **适应度噪声与可分辨度**：用已存的 281 读数（S 的 4 个 seed、底座的 2 个 seed、R6DAG s0 / s1、NTV0）抽 80 场景子集，
   算「两个真实差 0.03 的 checkpoint 被排对的概率」，以及同配方不同 seed 的新坏 / 新修场景数分布。这一步决定 exploit 阈值和新坏护栏的容差（草稿风险 1 已列）。
2. **练习切片和 281 是否同向**：已有 checkpoint 在训练场景上的读数（DAgger / S 推演时存的逐场景结局）按 checkpoint 排序，和 281 的排序算秩相关。不同向的话，PBT 选出来的是练习切片上的赢家，不是考场上的赢家。
3. **冻结种群试跑（不 exploit）2–3 轮**：A / B / C 用三个不同 θ（如 2.4× / 4× / 1.5×）。量三件事：
   每轮墙钟拆成训练 / 推演 / 等待；学习者之间的配对差是否超过第 1 项的噪声；S 抽取里可学激活份额和开环 ADE 的轨迹。
   差不超过噪声，就说明 PBT 在 P = 3 时选不出东西，改为固定日程的对照臂（草稿里的爬升 / 自适应）。
4. 第 3 项通过再开 exploit；281 仍然只在最后读一次。

---

## 5. 参考文献速查（按出现顺序，标记同上）

RL 基础与架构：PPO 1707.06347 [A]；IMPALA 1802.01561 [A]；Ape-X 1803.00933 [A]；R2D2（ICLR 2019，无 arXiv，未核）；SEED RL 1910.06591 [A]；
Sample Factory 2006.11751 [A]；DD-PPO 1911.00357 [A]；PER 1511.05952 [A]；Fedus et al. 2007.06700 [T]；AReaL 2505.24298 [T]；Async RLHF 2410.18252 [A]。
模仿与在线模仿：DAgger 1011.0686 [A]；AggreVaTe 1406.5979 [A]；SMILe（AISTATS 2010，无 arXiv，未核）；HG-DAgger 1810.02890 [A]；RHO-LOSS 2206.07137 [A]。
PBT 系：PBT 1711.09846 [T]；PB2 2002.02518 [T]；FIRE PBT 2109.13800 [T]。
课程：PLR 2010.03934 [T]；Robust PLR 2110.02439 [T]；ACCEL 2203.01302 [T]；PAIRED 2012.02096 [A]；SFL 2408.15099 [A]；CL4AD 2608.22549 [T]；MATS-Gym 2403.17805 [A]（ICRA）。
模仿锚的 RL：DAPG 1709.10087 [A]；AWAC 2006.09359 [A]；IQL 2110.06169 [A]；TD3+BC 2106.06860 [A]；RLPD 2302.02948 [T]；Cal-QL 2303.05479 [A]；BC-SAC 2212.11419 [T]；InstructGPT 2203.02155 [T]；Ziegler et al. 1909.08593 [A]。
自我改进：ExIt 1705.08439 [A]；ReST 2308.08998 [A]；ReST-EM 2312.06585 [A]。
持续学习与遗忘：EWC 1612.00796 [A]；LwF 1606.09282 [A]；Policy Distillation 1511.06295 [A]；Kickstarting 1803.03835 [A]；Progress & Compress 1805.06370 [A]；
Policy Consolidation 1902.00255 [A]；CLEAR 1811.11682 [A]；Wołczyk et al. 2402.02868 [T]；RL's Razor 2509.04259 [A]；Retaining by Doing 2510.18874 [A]；
Primacy bias 2205.07802 [A]；Lyle et al. 2303.01486 [A]；Dohare et al. 2306.13812 [A]；Model soups 2203.05482 [A]。
混比：DoReMi 2305.10429 [A]；Online Data Mixing 2312.02406 [A]；Data Mixing Laws 2403.16952 [A]。
评测统计与钻空子：rliable 2108.13264 [A]；Henderson et al. 1709.06560 [A]；Gao et al. 2210.10760 [A]；Skalse et al. 2209.13085 [A]。
驾驶：OPTED 2609.20756 [T]；RoaD 2512.01993 [T]；RAD 2502.13144 [T]；ReconDreamer-RL 2508.08170 [A]；GSDrive 2604.28111 [S]；World Engine 2606.19836 [A]；
Gigapixel 2606.19641 [S]；GIGAFLOW 2502.03349 [S]（ICML 2025 已核）；CaRL 2504.17838 [T]；Think2Drive 2402.16720 [A]；Roach 2108.08265 [A]；PlanT 2210.14222 [A]；
CarPlanner 2502.19908 [A]；Hydra-MDP 2406.06978 [A]；GTRS 2506.06664 [A]；ZTRS 2510.24108 [A]；TOAD 2606.07170 [A]；Bronstein 2022 2212.01375 [S]；
Hierarchical MBIL 2210.09539 [A]；Rethinking Closed-loop Training 2306.15713 [A]；CAT-K 2412.05334 [A]；TakeAD 2512.17370 [A]；Learning to Drive from a World Model 2504.19077 [S]；
MILE 2210.07729 [A]；ReCogDrive 2506.08052 [A]；HAD 2604.03581 [S]；WorldRFT 2512.19133 [A]；SimScale 2511.23369 [A]；R2SE 2506.09800 [A]；
Waymax 2310.08710 [A]；GPUDrive 2408.01584 [A]；Cornelisse et al. 2502.14706 [A]；V-Max 2503.08388 [A]；Bench2Drive 2406.03877 [A]；HUGSIM 2412.01718 [A]；NAVSIM v2 2506.04218 [A]。
AlpaGym：NVIDIA HF 博客 "Taking Alpamayo to New Heights with Driving Foundation Models and Closed-Loop Training"，github.com/NVlabs/alpagym [S]，没有论文。

**未核实**：R2D2、SMILe 的会议出处（无 arXiv）；BC-SAC 的 IROS 2023、SFL 的 NeurIPS 2024、RoaD 的 CVPR 2026 Findings（搜索结果里有 CVPR 海报页，但没打开）；
GSDrive、HAD、Gigapixel 只读了摘要；AlpaGym 的架构和数字博客里没有；Wayve 的闭环训练实践没找到公开材料。
