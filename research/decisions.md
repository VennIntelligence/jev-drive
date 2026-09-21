# 决定记录

跨 session 的共同决定都记在这里，一条一个小节。多个 Claude session 同时在这个 repo 上工作，
口头达成的东西不落盘就会丢，或者被下一个 session 用不同的假设覆盖掉。

每条写四件事：**决定**、**理由**、**状态**、**怎么才能定下来**。

状态只有两种：

- **待定**：已经按它执行，但支撑它的证据还不够，随时可能翻案。所有下游结论都要跟着标注不确定。
- **已确认**：证据够了，可以写进论文。改它需要新的证据，并在这里记下改动。

## 怎么维护

- **有中间结果就落盘**，不要等实验全部做完。一条结果只要能改变某个决定的状态，就当场更新对应条目。
- **关键测量要预登记**：在跑之前写好条目，包括测量怎么做、限定条件，以及一张"什么结果怎么判"的表，
  结果一栏留空。这样就不能等看见数字再挑说法。第 3d 条是范例。
  预登记只用于会改变方向的那几次测量，不是每个实验都要。
- **写错了就地改掉**，不要只在后面追加一段。旧的错误说法留在文档里，下一个人会照着它做。
  改的时候写清楚：原来说了什么、现在的数字是什么、为什么变（新数据、更大样本、还是原来就算错了）。
  commit message 里也写明这是一次更正。
- **降级和升级都要记**：证据变弱就把「已确认」退回「待定」，不要只升不降。
- 每条决定被推翻时，检查引用它的文档（`research/`、`todos/`）有没有跟着改。

---

## 1. 论文框架：视觉是在 ego-state prior 之上的增量

> **2026-09-21：本条已被证伪，见下面的「定论」段。** 保留原文是为了让后面的人看到
> 我们当时相信什么、凭什么相信，以及是什么推翻了它。

**决定**：主张写成「在一个很强的 ego-state prior 之上，视觉带来了多少增量」，不写成「VLM 会开车」。
每张表里 ego-only 单独占一行；另外专门划一个 pre-maneuver-onset 子集（车还没开始转的时刻），
因为 prior 在那里最弱。

**理由**：nuScenes trainval 上，只用 11 维 ego 运动状态的 NLL 是 0.310，最好的冻结视觉 feature 是 0.400，
两者拼接 0.308，视觉在全体样本上几乎没有增量。planner 的 ADE 差距更大：ego-only 约 0.8 m，
视觉 feature 3.3–4.3 m。但在 hard subset 上，ego probe 的 turn recall 塌到 0.010，
Qwen 中间层是 0.246。详见 [qwen-latent-driving.md](qwen-latent-driving.md)。

**状态**：待定。

**怎么才能定下来**：需要 Waymo 上重复出同样的模式（ego-only 强、pre-maneuver-onset 子集上视觉有增量），
并且 nuScenes 那个 0.192 → 0.185 的增量要有 scene-level bootstrap CI 支撑。
现在的差距很小，还没做 CI，有可能不显著。

**2026-09-20 补充证据（planner v0，见 [todos/2026-09-20-planner-v0.md](../todos/2026-09-20-planner-v0.md)）**：
CI 做了，用的是 paired scene bootstrap。全体 val 样本上视觉**是显著的但很小**：late fusion 把 ADE
从 0.772 降到 0.743（ΔADE −0.029 m，CI [−0.040, −0.019]），trust-region miss 从 0.367 降到 0.338
（−0.029，CI [−0.052, −0.012]）。

pre-maneuver-onset 子集（n=135）上，**先按两行各自的 CI 读会读反**：1.331 [1.177,1.488] 对
1.297 [1.148,1.456] 大幅重叠，看上去没有增量；配对之后 ΔADE = **−0.033，CI [−0.065, −0.003]，不跨零**。
但这**不**说明增量集中在 onset——四个子集的 ΔADE 几乎一样（all −0.029、turning −0.026、
onset −0.033、straight −0.029），**增量是均匀的**，probe 那个"集中在 hard subset"的模式没有复现；
而且信任域口径下 onset 是反的（Δmiss **+0.030**，CI [+0.000, +0.065]）。
真正要问的"onset 的增量是否大于 straight 的"是 difference-of-differences，现在测了：
**DiD(ADE) = −0.004 m，CI [−0.038, +0.032]，半宽 0.035 m**——点估计基本是零，半宽是它的 9 倍。
**没有任何证据表明增量集中在 onset**，同时这个测量也分辨不了小于约 0.035 m 的集中效应，
**underpowered by construction**，两句话要一起说。信任域口径上 DiD = **+0.060 [+0.020, +0.105]**，
显著为正，即 onset 上加视觉反而更容易被判出界（nuScenes 的 miss 是 stand-in，不直接外推）。

**2026-09-20 再补：DiD 实测**（run `20260920-113447`，onset n=135 对 straight n=3479）

| 口径 | onset Δ | straight Δ | DiD [95% CI]（半宽） |
|---|--:|--:|---|
| ADE (m) | −0.033 | −0.029 | **−0.004 [−0.038, +0.032]（0.035）** |
| trust-region miss | +0.030 | −0.030 | **+0.060 [+0.020, +0.105]（0.043）** |

位移口径上 DiD 的点估计是 −0.004 m，**半宽是它的 9 倍**。两句话必须一起说：
没有任何证据表明增量集中在 onset；同时这个测量分辨不了小于约 0.035 m 的集中效应。

信任域口径上则是一个**方向明确的反向信号**：DiD = +0.060，显著为正，
也就是相对 straight，onset 上加视觉反而更容易被判出界。nuScenes 的 miss 是围着单条 logged future
建的 stand-in，不直接外推到 RFS，但 RFS 正是我们的主指标，**这一条必须在 Waymo 上专门盯**。

**2026-09-21 定论：这个框架被证伪了。** 完整 val（93/93 shard、106360 帧、479 rater 帧）上，
DiD 两个方向都是**正的**，方向 0 显著（**+0.098 [+0.041, +0.158]**）。正号的意思是
**视觉的增量在 pre-onset 上比在 straight 上更小**——不是「集中在 ego prior 最弱处」，
而是**恰恰在那里最小**。这是 3d 预登记表没有列出的第五种情况，细节见第 3d 条。

**但「视觉有用」本身没有被证伪**：全集 delta −0.078 [−0.101, −0.058]，straight、turning 也都显著。
所以要改写的是**「增量在哪里」**，不是「有没有增量」。
框架的新说法应该是：**视觉买到的是整体上的小幅改善，而在 ego prior 失效的时刻它同样帮不上忙。**
这句话本身是一个有价值的负面结论，但它撑不起原来那篇论文的主张。

**状态改为：已证伪（待 train split 复核）。** 复核的理由不是不甘心，而是 head 只在 239 个 sequence
上训练过，train split 有约 2037 个，多一个数量级。
如果 Waymo 上 DiD 在位移口径不显著、在 RFS 口径又是正的（更差），那就不是"故事变弱"，
而是**这个子集上视觉有害**，第 1 条要推翻重写。

---

## 2. 指标：RFS 优先，ADE 并列汇报

**决定**：Waymo 上的开发循环盯 RFS（Rater Feedback Score，官方指标，把预测和人工评分过的轨迹比），
ADE 只作为并列数字报出来。

**理由**：官方 ADE 是对着评分最高的 rater 轨迹算的。logged future 本身得分 2.63 m，
而公开最好的 test ADE 是 2.65 m，说明 ADE 已经饱和，只用 ego 的 baseline 也能逼到一米以内。
另外约一半的 ego-only 预测落在所有 rater trust region 之外（`cv` 上 in-trust-region 是 0.551），
RFS 那里才有空间。**"落在外面"不等于"被罚到 4.0"，这两个量不能混用**：
实现里是 `per = np.where(inside, per, np.maximum(per, floor))`，也就是**落在外面的候选取
「衰减后的分」和 4.0 的较大者——floor 是把分数抬上去的下界，不是惩罚**；
反过来，落在里面的候选如果 rater label 本身低于 4，分数也可以低于 4。
所以 floored fraction（分数恰好等于 4.0）必须单独算，不能用 1 − in_trust_region 代替。
*这一句改写过*：原来写的是"约一半……被罚到 4.0"，把 in-trust-region 的补集当成了 floored，方向也说反了。
社区三个独立实现报 constant-velocity 在 val 上 RFS 7.00/7.022/7.057 且 **floor rate 0/479**，
而我们的 stage-A 学出来的 ego-only head floor 率 23.6–25.6%。
`jevdrive/waymo.py` 里有官方 RFS 的 bit-exact 移植。

**2026-09-21 实测，这一列补完了（完整 val，479 个 rater 帧，`report/rater.csv`）**：

| 轨迹 | RFS (cluster) | in_trust_region | **floored** |
|:--|--:|--:|--:|
| rater_best | 9.587 | 1.000 | **0.000** |
| logged_future | 8.131 | 0.772 | **0.094** |
| rater_worst | 7.715 | 1.000 | 0.029 |
| cv_vel | 7.116 | 0.564 | 0.263 |
| **cv（零参数）** | **7.103** | 0.551 | **0.271** |
| ctrv | 7.018 | 0.524 | 0.259 |
| ca | 6.845 | 0.491 | 0.307 |
| ctra | 6.803 | 0.472 | 0.286 |
| zero（原地不动） | 5.383 | 0.280 | 0.618 |

**结论一：社区那个 0/479 在我们的口径下复现不出来，而且「学出来的 head 有 physics 没有的失效模式」
这个假设是错的——方向反了。** 我们的零参数 `cv` floor 率是 **27.1%**，而 stage-A 学出来的
ego-only head 是 23.6–25.6%，**学出来的比免费物理还略好一点**，不是更差。
两边用的是同一个定义（`sc <= RFS_FLOOR + 1e-9`，`waymo_stage_a.py` 和 `waymo.py` 共用）。
RFS 均值这一侧我们和社区是**吻合**的（7.103 对 7.00/7.022/7.057），所以分歧只在 floor rate 的定义上，
不在 RFS 实现上。**在弄清他们怎么数之前，不要引用那个 0/479，也不要拿它质疑我们的 ego baseline。**
*这一段推翻的是 2026-09-21 早些时候的说法*：当时把 24% 当成「bug 形状的异常、要先修」，
依据是社区的 0%；补完这一列之后没有异常可修。

**结论二：floored 按机动强烈分层，但分层的方向是「已经在转」，不是「转之前」。**
`cv` 的 floor 率：turn_yaw（已经在转）**86.3%**、right 62.1%、turn_intent 48.1%、
straight 24.6%、straight_yaw **17.9%**。这是第 3c 条「ego state 在转弯时才值钱」的镜像：
从一个正在转的状态做匀速外推，5 s 后偏到所有 trust region 之外。
pre_onset 是 50%，**但 n=6，不能用**（见第 3b 条）。

**结论三（新，值得单独报）：logged future 自己也被 floor 了 9.4%。**
车实际开出来的那条轨迹，在约十一分之一的 rater 帧上落在所有 rater 的 trust region 之外
且衰减到 4 以下。rater_best 是 0.0%、rater_worst 是 2.9%，作为内部一致性检查通过。
顺带坐实了社区那份 repo 存疑的一点：**RFS 确实可以低于 4**（`rfs_min` 多行是 3.0），
因为 rater label 本身可以低于 4，我们的实现早就是这样写的。

**状态**：待定。

**2026-09-21 定稿（完整 val，93/93 shard，479 个 rater-scored 帧——每 sequence 恰好一帧，样本到此为止）**：
logged future 对最高分 rater trajectory 的 ADE@5s = **2.699 m**，而公开 test ADE 是 RAP 2.6457、
Poutine 2.7419。**RAP 已经略微越过了「完美预测 log」这条线**，所以「ADE 没有 headroom」不是待定而是坐实。
RFS 那边：logged future 8.131（cluster mean，榜单口径）/ 8.175（frame mean），
ego-only 6.80–7.12，原地不动 5.38，最高分 rater 轨迹 9.59。公开最好 test RFS 是 8.043。
68 → 163 → 479 帧的过程里每一行变动都小于 0.5，排序稳定。
**一处随样本变化的修正**：163 帧时 `ctrv`/`ctra` 在 cluster mean 上低于 `cv`/`ca`，
完整 val 上两者持平（7.02 vs 7.10），所以那个差距是抽样噪声，不是信号。

**状态**：已确认。样本已经用尽（每 sequence 一帧是数据集构造），不会再变。

---

## 3. Waymo 上的 pre-maneuver-onset 子集

**决定**：Waymo 版本的 hard subset 用 past-state window 定义（当前 yaw rate 低，但未来要转），
和 nuScenes 的 `jevdrive/labels.py` 保持同一套定义，不用 nuScenes 的 CAN 数据，
这样两个数据集的表格能直接对齐。

**理由**：两边的子集定义不一致的话，跨数据集的对比就没有意义。

**状态**：**已确认**（2026-09-20，b5 核对完成）。

阈值和 nuScenes 撞上了，不用重新标定：`|yaw rate| < 1.0 °/s` 这一条在 Waymo 上留下 60% 的可用 val 帧，
nuScenes 是 64%，同一量级。用 intent 交叉验证：pre-onset 子集里 56% 是 GO_LEFT/GO_RIGHT，
straight 子集里只有 0.3%，`check` 里已经断言这个分离度。

有一处必须和 nuScenes 不同：**未来是否转弯不能用逐点差分求 heading**。Waymo 不存 future yaw，只有位置，
在 4 Hz 且经常接近静止的情况下差分出来的 heading change 纯是噪声（val 上 2 s 内的 99 分位是 358°）。
改成对 horizon 处的 waypoint 算一次弦角（chord bearing），等曲率圆弧下它正好等于 heading change 的一半，
低速时优雅退化。这个差异要写进论文的实验细节。

顺带修掉了原来 turning 子集缺 validity guard 的问题，样本数从虚高的 2657 降到真实的 1436。

---

## 3b. pre-maneuver-onset 子集上只能用 ADE，不能用 RFS

**决定**：论文的证据分两条腿：**整个 split 上用 RFS 报主结果，pre-maneuver-onset 子集上用 ADE 报机理**。
两边都必须并排给出 ego-only 基线。

**理由**：结构性限制，不是样本还不够的问题。Waymo 的 rater label 每个 sequence 只有一帧，
而且固定在 12 s 处，**不是**按「maneuver onset 时刻」挑的。所以落进 pre-onset 子集的 rater-scored 帧
本来就极少：现有 68 帧里只有 2 帧，外推到完整 val 也只有 14 帧（默认阈值）到 35 帧（最松阈值）。
**2026-09-20 更新**：样本涨到 163 个 rater-scored 帧之后，落在 pre-onset 的**仍然只有 2 个**，
满 val 的投影从 14 降到 **6**。这条结论从「大概率不行」变成「确定不行」。

**2026-09-21 实测（完整 val，479 个 rater 帧）**：默认阈值下落在 pre-onset 的是 **6 帧**，
和 29-shard 时的投影**完全一致**；最松的可用阈值也只有 29 帧。不再是外推，是测量。
ADE 那边没有这个问题：完整 val 上 pre-onset 有 **1510 帧**（当时投影约 1570）。

**状态**：**已确认**。这是数据集的构造决定的，再多下载也不会改变。

---

## 3c. 核心论据：ego state 在 onset 前是瞎的

**决定**：论文的核心论点用这个对比来立：**知道当前 yaw rate，在车已经在转的时候值约 1.2 m ADE@5s，
在转弯开始之前只值 0.03 m，差约 37 倍。**（2026-09-20 在 29 个 shard、489 个 pre-onset 帧上刷新；
12 个 shard、226 帧时是 1.03 m 对 0.02 m、40 倍。样本翻倍后结论稳住。）

**理由**：Waymo val（当前 shard）上，`ctrv` 相对 `cv`、`ctra` 相对 `ca` 的增益：

| 子集 | yaw rate 值多少 ADE@5s |
|---|---:|
| 全集 | −0.12 |
| 已经在转 | **−1.03** |
| pre-maneuver-onset | **−0.02** |
| 直行 | +0.01 |

和 nuScenes 上「ego probe 的 turn recall 在 hard subset 塌到 0.010、Qwen 中间层 0.246」是同一件事的两种说法。
这一格正是相机必须赢下来的地方，也是第 1 条框架的量化版本。

**状态**：待定。

**怎么才能定下来**：数字本身很稳（226 帧，且两组 baseline 给出一致的 40 倍差距），
但「相机能把这一格赢下来」还没有证据——那要等 Waymo 上的 frozen feature + 薄 head 跑出来。
如果视觉在 pre-onset 子集上同样只值 0.02 m，整个项目的前提就不成立，要立刻重新考虑。

**2026-09-20 的第一个反向信号（样本不足，不足以翻案，但必须盯住）**：planner v0 在 nuScenes 的
pre-onset 子集上，视觉**没有**增量（ADE 1.331 → 1.297，CI 大幅重叠；trust-region miss 反而从
0.889 升到 0.919）。这和 probe v1 在 hard subset 上的结论（turn recall 0.010 → 0.246）相反。
两者的差别是任务不同：probe 问「会不会转」，planner 问「轨迹长什么样」。
但 n 只有 135、CI 半宽 ±0.15 m，要判别的差距是 0.034 m——**nuScenes 在这个子集上是先天测不动的
（underpowered by construction）**。所以这个 null **不是**「视觉在 onset 前没用」的证据，
以后任何人引用它都必须带上这句话。

**Waymo 上约 1750 帧时必须重做，这是整个项目最关键的一次测量。** 做的时候有一个方法要求：
**要报 vision 相对 ego 的 paired delta 本身的 CI，不是两行各自的 CI**。
两行的 CI 重叠并不能判定 paired 比较的结果，planner v0 已经有的 paired scene-bootstrap 就是对的工具。

---

## 3d. 半 val 预演：pre-onset 上的 vision − ego 配对差（**预登记，数字待填**）

**这条在拿到结果之前写下来。** 它是全项目等的那个数，也因此是最容易被人脱离限定条件引用的数——
被以后的 session 引用，或者被我们自己写进 draft。所以警告和判据必须和数字在同一段里，不能隔一节。

**测量**：Waymo val 按 sequence 对半切，一半训 head 一半评估。在 pre-maneuver-onset 子集上，
报 vision 相对 ego-only 的 **paired delta 及其 scene-bootstrap CI**（不是两行各自的 CI，见第 3c 条）。

**限定条件（引用这个数时必须一起引用）**：head 在 val 的一半上训练、在另一半上评估，
val 总共只有 479 个 sequence。**这是预演，不是论文的数字。** 论文必须在 train split 上训练、
在完整 val 上评估。另外 pre-onset 子集上只能用 ADE，不能用 RFS（见第 3b 条）。

**预先定好怎么读结果**（避免事后按结果挑说法）：

| 结果 | 怎么判 |
|---|---|
| delta 显著为负（视觉有增量），且量级远大于 ego 在这里的 0.02 m | 第 1 条框架成立，按计划推进到 train split |
| delta 显著为负但很小（和 0.02 m 同量级） | 前提勉强成立，但要重新算「值不值得为此付 33 ms」 |
| delta 不显著，CI 半宽小于要判别的量级 | 真 null。第 3c 条的证伪条件触发，立刻重新考虑方向 |
| delta 不显著，但 CI 太宽 | **又一次测不动**，和 nuScenes 那次同性质，不能当证据。要么等 train split，要么换更有力的设计 |

**主量是 difference-of-differences，不是 onset 上的 delta**（2026-09-20 补充，在测量之前写下）：
planner v0 已经证明这两者会给出不同答案——onset 上 ΔADE 显著为负（−0.033，CI 不跨零），
但四个子集的 Δ 几乎一模一样，所以"增量集中在 ego prior 最弱处"这个主张并不成立。
上表的四行判据照旧报，但**框架成立与否由 DiD 决定**：

> DiD = (onset 上的 vision−ego delta) − (straight 上的 vision−ego delta)，带自己的 paired bootstrap CI。

DiD 显著为负才支持第 1 条的框架；DiD 不显著而两边 delta 都显著为负，则结论是
**"视觉带来一个均匀的小增量"**，那是一个弱得多、但仍然诚实的故事，论文要按那个写，
并且要重新评估这个项目值不值得继续。

**够不够测得动（预估，不是保证）**：nuScenes 上 onset 的 ΔADE CI 半宽是 0.031 m（n=135）。
Waymo 的 pre-onset 约 1750 帧，按 √n 缩放是 0.278 倍；Waymo 的 ADE 尺度又比 nuScenes 大约 3 倍，
两者相抵后 delta 的半宽约 0.026 m，DiD 大约再乘 √2，约 **0.037 m**。
所以只有当真实的 DiD 明显大于 0.04 m 时才判得出来。**跑完必须把实际半宽填进来，
和这个预估对照**；如果实际半宽也和效应同量级，那就是第四行，不是 null。

**标定（2026-09-20，测量之前）**：planner v0 现在实测了 nuScenes 上的 DiD，半宽 **0.035 m**
（n = 135 对 3479）。它只比 onset delta 的半宽 0.031 m 大 **1.13 倍**，不是上面假设的 √2——
因为对照臂 straight 有 3479 帧，几乎不贡献方差。用实测值重算：
0.035 × √(135/1750) × 3 ≈ **0.029 m**。所以上面那个 0.037 m 偏保守，判别门槛大约在 0.03 m，
不是 0.04 m。**这是对 power 预估的修正，不是对判据的修正**：上表四行怎么读一个字没改。

**结果**（2026-09-21 测得。run: box 上 `$DATA_DIR/runs/waymo_stage_a/half_val/20260921-081010/`，
完整 val 93/93 shard、106 360 帧、479 个 rater 帧；两个方向都跑）：

**主量 DiD（= pre-onset 的 delta − straight 的 delta），来自 `ridge_late`，不经过词表**：

| 方向 | DiD (m) [95% CI] | 半宽 | pre-onset delta (n) | straight delta (n) |
|:--|:--|--:|:--|:--|
| 0（fit 半 0 → eval 半 1） | **+0.098 [+0.041, +0.158]** | 0.059 | −0.027 [−0.073, +0.017]（791） | −0.125 [−0.162, −0.091]（23 612） |
| 1（fit 半 1 → eval 半 0） | +0.029 [−0.036, +0.091] | 0.063 | −0.043 [−0.090, +0.006]（719） | −0.072 [−0.112, −0.033]（22 968） |

**DiD 两个方向都为正，方向 0 显著为正。** 正号的意思是：**视觉的增量在 pre-onset 上比在 straight 上更小**，
而不是更大。这是上面那张预登记表没有列出的第五种情况——不是"均匀的小增量"，是**"增量真实存在，
但恰恰在 ego prior 最弱的地方最小"**。

**pre-onset 的 delta 本身**：两个方向都**不显著**（CI 跨零）。方向 0 的 CI [−0.073, +0.017]
**排除了 straight 那个量级的增益（−0.125）**，所以方向 0 落在上表**第三行（真 null，3c 的证伪条件触发）**；
方向 1 的 CI [−0.090, +0.006] 包含 −0.072，排除不掉，落在**第四行（测不动）**。

**视觉整体是有用的**，这一点两个方向都显著：全集 −0.078 [−0.101, −0.058] 和 −0.034 [−0.056, −0.012]，
straight 和 turning 子集也都显著。所以这不是"视觉没用"，是**"视觉有用，但用处不在我们押注的地方"**。

**RFS 上视觉买不到任何东西**（n=237 / 242 个 rater 帧）：ego-only 7.258 → +视觉 7.265（方向 0），
7.112 → 7.096（方向 1）；被压到下限的比例 23.6% → 24.9% 和 24.4% → 25.6%，**略微变差**。
RFS 的 CI 半宽约 ±0.3，所以这只能说"在这个样本量下测不到增益"，不能说增益是零。

**实际半宽 vs 预估**：delta 的半宽实测 **0.045–0.048**，预估是 0.029。差距不是模型错，是 n 假设错——
预估按"Waymo pre-onset 约 1750 帧"算，但那是**整个 val**，对半切之后评估半边只有 **719–791**。
把预估按实际 n 重标：0.029 × √(1750/791) = **0.043**，和实测 0.045 基本吻合，**power 模型是准的**。
DiD 的半宽 0.059–0.063 比预估大一倍，因为 nuScenes 那次 straight 臂有 3479 帧、几乎不贡献方差，
Waymo 这边两臂的 delta 本身都更嘈杂。

**词表 handicap 这个 confound 已排除**：第 11 条记录的"词表系统性亏待 pre-onset"（oracle minADE
0.905 对 0.467）只影响分类头。**主量 DiD 来自 `ridge_late`，是连续回归，完全不经过词表**，
所以那个 handicap 解释不了它。分类头的 DiD（+0.008 和 −0.035）CI 都很宽、都不显著，两边都不支持。
相对量级上也一样：方向 0 的相对增益 pre-onset −1.0%、straight −7.0%，方向 1 −1.9% 对 −3.9%，
**不是 pre-onset 基数大造成的假象**。

**状态**：**已确认**——DiD 显著为正（方向 0）、两个方向同号，第 1 条"增量集中在 ego prior 最弱处"的
框架**在 Waymo 半 val 上被证伪**。按 3d 事先写死的读法，论文要改成弱故事，并重新评估项目价值。
但注意限定条件：这是 **val-internal 预演**，head 只在 239 个 sequence 上训练。
train split 下完之后必须整套重做，那时训练数据多一个数量级，结论**有可能**变。

---

## 4. 分辨率默认用 800 px

**决定**：抽 frozen feature 默认用 800 px 宽的输入，不用 1600 px 原分辨率。

**理由**：nuScenes trainval 上两者最好的 NLL 差不到 0.006，hard subset 上完全相同（0.213），
而 800 px 抽 feature 快 4.6 倍（30.8 对 142.4 ms/frame）。

**状态**：待定。

**怎么才能定下来**：这个结论来自转向三分类。Waymo 的轨迹任务、以及需要看清远处小目标的场景，
可能对分辨率更敏感。在 Waymo 上做一次同样的两分辨率对比即可确认。

---

## 5. 取中间层

**决定**：frozen feature 取 Qwen3-VL 解码器的中间层（nuScenes 上最好的是 L20–L24，共 36 层），
pooling 用 image-token mean，不用 last token。

**理由**：转向 probe 的 layer 曲线是清楚的倒 U 形，L20–L24 最低，到 L36 退回早期层水平。
见 [qwen-latent-driving.md](qwen-latent-driving.md) 的图。

**状态**：待定。

**怎么才能定下来**：planner 的 layer curve 上，这个倒 U 形要弱得多。planner v0 跑完后更明确：
最好的是 `qwen_w800/L19–L22 mean`（ridge ADE 3.250–3.268 m），层位和 probe 的 L20–L24 重合，
中层也确实好过自家 ViT（3.438）和 DINOv2（3.592）；但**U 只有 0.2 m 深，而 CI 半宽是 ±0.27 m**，
这张表自己撑不住选层。层位的依据目前主要来自 probe。需要 Waymo 的 layer curve 确认。

pooling 的结论两个任务其实一致：**mean 都好于 last**。planner 上回归差 0.15 m、分类差 0.3 m；
probe v1 上 mean 在每一层都更低（全体最好 0.400 对 0.421，hard subset 0.213 对 0.224）。
（"probe 上 last 更好"是 v0 在 mini 上的结论，那时 val 不到 100 个样本，trainval 上被推翻了。）

**2026-09-20 补充证据（planner v0）**：轨迹回归上最好的是 L19 mean（ADE 3.250 m），L13–L22 都在
3.25–3.27，L01 是 3.445、L36 是 3.380，确实是同一个方向的浅 U，最好层和 probe 的 L20–L24 重合。
但最好和最差只差 0.2 m，而 bootstrap CI 半宽约 ±0.27 m，**单看轨迹任务不足以支撑选层**。
pooling 这一半倒是被加强了：mean pooling 在轨迹上稳定好于 last token（回归差 0.15 m、分类差 0.3 m），
**和 probe v1 在 trainval 上的结论方向一致**（mean 在每一层都优于 last）。
所以 mean pooling 这条可以当成跨任务成立的结论。

---

## 6. 下载顺序和带宽

**决定**（2026-09-20 11:20 修订）：Waymo val 和 navsim **并行**跑，HF 模型排在 navsim 之后。
Waymo train 要等 val 完成后再确认，test 最后（提交限制是每 30 天 6 次，不急）。

原来的决定是全部串行（navsim → HF → val → small）。改的原因：navsim 还剩 257 GB、约 10.5 小时，
串行的话 val 要到次日凌晨 4 点才完整，而**项目最关键的一次测量**（pre-onset 子集上视觉有没有增量，
见第 3c 条）正卡在 val 上。并行之后两边各拿约一半带宽，val 约 9 小时、navsim 约 21 小时。
这是用 navsim 的延迟换关键测量提前，由用户拍板。

**理由**：box 的总下行带宽只有约 12–18 MB/s，所有任务共享。并行跑的时候 Waymo 只有 1.3 MB/s，
ETA 336 小时；串行之后单个任务能拿到约 10–16 MB/s。

**2026-09-20 11:30 再修订：Waymo 改走 Clash 代理。** 并行之后实测发现「各拿一半」不成立：
navsim 走 hf-mirror 是国内线路，单连接就很稳；Waymo 走 direct 到 GCS 是丢包的跨境线路，抢不过，
实测 navsim 4.65 / Waymo 1.16 MB/s，val ETA 44 h，比串行还慢。把 navsim 并发从 8 降到 3 没用（1.31）。
改成 `--route proxy --streams 16` 后 Waymo 到 6.2 MB/s，val ETA 约 9 h。
**代价：val 的约 227 GB 要走代理订阅的流量**，用户明确拍板"时间最值钱"，接受这个代价。
注意这条推翻了 `docs/waymo-e2e.md` 里"bulk data 不走 Clash 以免烧流量"的默认做法，属于一次性破例。

**代理流量是这次破例的主要风险，而且会咬两次。** 订阅流量如果在传输中途用尽，丢的不只是下载：
gcloud 刷 OAuth token 也只能走代理，token 一过期，Waymo 下载连续传都做不了，必须先续费。
**从 box 上读不到剩余流量**：`~/data/clash/config.yaml` 里是静态节点列表，没有订阅 URL，
所以拿不到 `subscription-userinfo` 头。mihomo 自己的计数（`127.0.0.1:9090/connections`）只有本次启动以来的用量，
2026-09-20 11:40 时是 down 10.9 GB。**订阅总量是 1 TB，但要按节点倍率折算**（用户 2026-09-20 指出）。机场的节点名里带倍率，
30 个节点标着 **0.1 倍**（日本东京、新加坡、美国），也就是走这些节点时 1 TB 额度相当于 **10 TB 实际流量**。
原来 GLOBAL 走的是 `🇸🇬AWS新加坡04`，名字里没有倍率标记，即 **1 倍**，等于在十倍烧额度。

实测三个 0.1 倍节点，**不但省流量，速度还更快**（在跑的 Waymo 下载上直接测，每个测 60 s）：

| 节点 | 吞吐 |
|---|--:|
| 🇸🇬AWS新加坡04（1 倍，原来的） | 8.44 MB/s |
| 🇺🇸美国02-0.1 倍 | 11.79 MB/s |
| **🇯🇵日本东京01-0.1 倍** | **11.96 MB/s** |
| 🇯🇵日本东京06-0.1 倍（高速专线） | 9.73 MB/s |

已把 GLOBAL 固定到东京 01。val 的 227 GB 因此只折算约 **23 GB** 额度。
**以后所有大流量都要先确认走的是 0.1 倍节点**，操作步骤和这四个实测数字已经写进
[../docs/network-proxy.md](../docs/network-proxy.md) 的 "Picking a Clash node"，那里是操作手册，这里只留结论。
注意名字里的线路标签（电信联通推荐、高速专线推荐）不等于吞吐，这次「高速专线」反而最慢，要实测。

**Waymo train 的计划（2026-09-20 定，等 val 落地后再决定，三个选项并列，不要只在其中两个里挑）**：

| 选项 | 数据量 | 时间 | 代价 |
|---|---|---|---|
| A. 全量 train 走 direct | 原始 1.2 TB，slim 后约 520 GB | 独占带宽约 22 h | 不烧代理流量；但 direct 在有竞争时会被压到 1 MB/s 级别 |
| B. 全量 train 走代理（0.1 倍节点） | 同上 | 约 28 h（按 12 MB/s） | 折算约 **120 GB** 额度，占 1 TB 的 12%。倍率折算后这个选项重新可行，而且比 A 更快 |
| C. 部分 train（约 100 shard，原始 460 GB） | 460 GB | 代理约 11 h / direct 独占约 9 h | 折算约 46 GB 额度。数据少，但薄 head 大概率够用 |

选项 C 来自 b5：head 只是 frozen feature 上的线性层或浅 MLP，feature 在 2 Hz 上抽，
100 个 shard 约 15 万可用帧，远超这种 head 的需要。shard 是帧的随机抽样而不是连续区段，
所以子集**看起来**不是有偏抽样，但**依赖它之前要对着 scenario cluster 核对一遍**。
如果后来发现 head 确实被数据量限制住，再补下剩下的。

盘够：现在剩 2.0 TB，扣掉 navsim 剩余 257 GB 和 val slim 约 100 GB，还剩约 1.6 TB。
test 最后，不急（每 30 天 6 次提交）。

**状态**：已确认。串行仍然是默认做法（并行时每个任务只拿一半带宽，总时间不会变短）；
只有当某个下载卡住关键路径时才破例并行，并在这里记一笔。

队列在 box 上 `$DATA_DIR/tmp/dlq.sh`（窗口 `jev:dlq`，现在只负责 HF），Waymo 在 `jev:waymo`。

---

## 7. 谁改哪个文件

**决定**：`jevdrive/plots.py` 里 `layer_curve` 和 `k_sweep` 归 planner 那条线维护，
`probe_layer_curve` 归 probe 这条线。改别人的函数之前先打招呼。

**理由**：多个 session 同时编辑同一个文件会互相覆盖，git 层面看不出冲突。

**状态**：已确认。

---

## 8. 轨迹词表的 K 至少取 1024

**决定**：trajectory vocabulary 的 K 取 1024 起步，Waymo 上从 {1024, 4096, 8192} 里选，不再考虑 64/256。
并且每次 K sweep 都要同时报 **oracle minADE** 和 **trust-region 覆盖率**（整个词表里没有任何一条 anchor
能落进信任域的帧占比）。

**理由**：两个口径给出的结论不一样。nuScenes 上 K=64 的 oracle minADE 已经是 0.390 m，
远好于实际做到的 3.7 m，看起来 64 条就够；但按信任域看，K=64 有 **16.2%** 的帧无论怎么选都会出界，
K=256 是 5.5%，K=1024 才降到 1.0%，K=8192 是 0.2%。Waymo 的 RFS 会把出界的帧直接压到下限 4.0，
所以那 16% 是白送掉的分。ADE 口径完全看不见这件事。
另外：分类器自己的误差在 K=64…8192 之间基本不动（ADE 3.73→4.22，miss 0.776→0.772），
**误差全部来自 ranking，不是 coverage**，所以加大 K 的唯一理由就是覆盖率。
数据见 [todos/2026-09-20-planner-v0.md](../todos/2026-09-20-planner-v0.md) 的 K sweep 表。

**状态**：待定。

**怎么才能定下来**：nuScenes 的信任域是围着单条 logged future 建的，比 Waymo 的三条 rater 轨迹严格。
在 Waymo 上用真 RFS 重做一次 K sweep，看覆盖率的拐点落在哪里。

---

## 9. ego state 和图像特征一律用 late fusion 拼

**决定**：把 ego state 和 frozen 图像 feature 合起来的时候用 **late fusion**——先训一个 ego-only head，
把它的输出（回归的预测值 / 分类器的 logit）当成冻结的 offset，图像 head 只学残差。
不要把两块特征直接拼（哪怕图像那块先 PCA 降到 64 维）。

**理由**：probe v0 就发现 2560 维图像会把 11 维 ego 淹掉，当时提了 PCA 降维和 late fusion 两个修法。
planner v0 两个都试了：PCA 拼接在回归上没有收益（ADE 0.772 → 0.783，paired +0.012，不显著），
在分类上是灾难（miss 0.363 → 0.644，ADE 0.847 → 1.906）。原因是一个共用的 L2 强度没法同时正则
两块尺度完全不同的特征。late fusion 则是唯一显著为正的做法（ADE −0.029 m，miss −0.029）。

**状态**：已确认（在 nuScenes 上；实现是 `jevdrive/planner.py` 的 `ridge_late` 和 `cls_late`）。

---

## 10. 固定词表分类 vs 连续回归，取舍要按指标说

**决定**：不要笼统说「分类比回归差」。同 backbone、同特征下，分类在**位移口径**上输、
在**信任域口径**上赢，报的时候两个都给。

**理由**：nuScenes 上同一套 `L22_mean` 特征，分类比 ridge 回归的 ADE 差 0.512 m（约 16%，
CI [+0.254, +0.748]），但 trust-region miss 低 9.5 个百分点（CI [−0.136, −0.059]），两边都显著。
ADE 奖励条件均值，而回归 head 输出的就是均值；信任域是「要么进要么出」，
一条平均出来的折中轨迹既不像直行也不像转弯，两边都不沾。分类器输出的是真实存在过的一个 mode。
Waymo 上官方 ADE 已经饱和（logged future 自己 2.63，最好的 test 是 2.65）而 RFS 还有空间，
所以这个取舍对我们有利。

**状态**：待定。

**怎么才能定下来**：在 Waymo 上用真 RFS 做同一组对照。如果 RFS 上分类也赢，这就是论文里
「为什么用固定词表」最强的一条论据。

---

## 8. 固定词表分类 vs 连续回归

**决定**：主系统用固定词表分类（在 K 条 anchor 上打分选一条），不用连续回归，即使它在 ADE 上更差。

**理由**：planner v0 在 nuScenes 上做了同特征 paired 对比，两个指标给出相反的结论：
**ADE 上分类输 0.512 m（+16%，CI [+0.254, +0.748]）；trust-region miss 上分类赢 0.095
（CI [−0.136, −0.059]）**。原因是回归出来的平均轨迹既不像直行也不像转弯，两边都不沾，
而这种「哪边都不像」的轨迹正是 RFS 的 trust region 要罚的。
Waymo 上 ADE 已经饱和而 RFS 有空间（见第 2 条），所以这个取舍对我们有利。

soft target（对 top-m anchor 给软标签）能降 ADE 0.138 m，但不降 miss，而且 top-10 的多样性变差，
暂不采用。

**状态**：待定。

**怎么才能定下来**：nuScenes 上的 miss 是 RFS 的替身，不是 RFS——nuScenes 只有一条参考轨迹，
比 Waymo 的三条 rater 严格，绝对值不能跨数据集比。要在 Waymo 上用真 RFS 重做这个 paired 对比。

---

## 9. K 取多少：两个指标给出相反答案

**决定**：K ≥ 1024。

**理由**：这是 planner v0 最值得记的一条。**分类器自身的误差完全不随 K 改善**
（K 从 64 到 8192，cls ADE 3.73 → 4.22，不降反升），也就是说误差 100% 来自 ranking、0% 来自 coverage，
按 ADE 口径会得出「K=64 就够」的结论。但按 trust-region 覆盖率看，**K=64 时有 16.2% 的帧
无论怎么选都出界**（K=1024 时降到 1.0%，K=8192 时 0.2%），这部分损失 ADE 口径完全看不见。

**状态**：待定。

**2026-09-20 在 Waymo 上用真 RFS 几何复现了**（half-val 的 eval 半边，83 个 rater 帧）：

| K | oracle minADE | 没有任何 anchor 能进任一 rater trust region | pre-onset 的 oracle minADE | straight |
|--:|--:|--:|--:|--:|
| 64 | 1.115 | **0.096** | 1.766 | 1.015 |
| 256 | 0.748 | 0.036 | 1.160 | 0.658 |
| 1024 | 0.551 | **0.000** | 0.904 | 0.475 |
| 8192 | 0.407 | 0.000 | 0.689 | 0.328 |

K=64 的 oracle minADE 1.115 m 看上去绰绰有余，但 **9.6% 的 rater 帧无论怎么选都被压到 4.0**。
两个口径给出不同 K 结论这件事，现在有 Waymo 自己的证据，不再依赖 nuScenes 的替身阈值。
注意 **0/83 不是 0**：单侧 95% 上界是 3.5%，要压到 1% 需要完整 val 的约 240 个 rater 帧。

**一个对主张不利的结构性发现**：**词表系统性亏待 pre-onset**——每个 K 上它的 oracle minADE 都是
straight 的 **1.7–2.1 倍**。k-means 优化的是总体均方位移，会主动牺牲少数派，
而少数派正好是我们论文的主战场。也就是说**我们的 head 在最需要赢的地方被自己的词表拖了后腿**，
这部分误差和视觉有没有用无关，是设计造成的。

可用的杠杆（**都没试过，属于推测**）：按子集重采样后再聚类、给 pre-onset 留一批专用 anchor、
或者用对 minADE 之外的目标聚类。代价是整体 oracle 变差。
**在归因"视觉在 onset 上没用"之前，必须先排除这一条**，否则会把词表的锅算到视觉头上。

**怎么才能定下来**：完整 val 到位后重跑这张表（rater 帧从 83 涨到约 240），
并做一次「按子集重采样聚类」的对照，看 pre-onset 的 oracle 能不能拉上来、整体代价多大。

---

## 10. Ego 和视觉怎么融合：late fusion，不要拼接

**决定**：视觉 feature 和 ego state 用 late fusion（各自出 logit 再相加），不要降维后拼接。

**理由**：planner v0 上 PCA 拼接完全不行——回归上没有增益，分类上是灾难（miss 从 0.363 涨到 0.644）；
late fusion 才有效且显著（ADE −0.029 m [−0.040, −0.019]，miss −0.029 [−0.052, −0.012]）。
这印证了 probe v0 留下的警告：2560 维的视觉 feature 会淹掉 11 维的 ego 向量。

注意增量本身很小：**视觉叠在 ego 上只值 3.8%**。这正是第 1 条框架说的事，也是为什么
pre-onset 子集才是主战场。

**状态**：待定，同第 1 条。

---

## 11. 延迟：head 免费，成本全在抽特征

**决定**：效率叙事的主角是「冻结 backbone 抽一次特征」的成本，不是 head。
论文里必须同时给出两个口径，并写清区别。

**理由**：planner v0 实测 head 全程 **< 0.1 ms**（ridge 0.019 ms，K=1024 分类 0.040 ms，
K=8192 0.060 ms），K 涨 128 倍只多 0.02 ms。真截断（只跑到第 22 层，后 14 层不执行）
**batch 8 省 31%（13.3 → 9.2 ms/frame），batch 1 只省 7%（35.7 → 33.1 ms）**——
batch 1 下 launch overhead 占主导，截断省不下来。

两个口径：**端到端 batch 1 约 33.2 ms（约 30 Hz）**，这是部署口径；
按吞吐算是 21.2 ms/frame，这是离线抽特征的口径。不能混用。

**一个对我们有利的对比（2026-09-20，openjev 重测时发现）**：Jev-style 的延迟**依赖输入**。
把 demo 的 ego state 从占位值改成真实值（0.056 m/s、turn left）之后，openjev 从 333 ms 涨到 **471 ms**，
因为它对 longitudinal 只有 0.10 的 confidence、entropy 超阈值，于是**每个请求都触发 3 次并行 re-read**。
单次 read 只要 47 ms。也就是说「一次决策」这个卖点在不确定的场景下并不成立，而**不确定的场景正是
我们关心的那些**。我们的系统是一次 forward 加一个打分，**延迟与输入无关**，p95 和 p50 几乎相同
（planner v0 实测 head 的 p95 ≈ p50）。这一点值得在效率那张表里单独说，不要只比 mean。

**竞品实测汇总（2026-09-21，全部在同一张 RTX PRO 6000、batch 1、20 warmup + 200 timed）**：

| 系统 | 延迟 mean | 备注 |
|---|--:|---|
| Qwen-Drive-1.0-4B，直接规划 | 702 ms | |
| Qwen-Drive-1.0-4B，带 reasoning | 1258 ms | |
| AutoVLA（Qwen2.5-VL-3B） | 1362 ms | **它在 Waymo val 上从不走 CoT**（150/150，见第 15 条），所以 1362 ms 就是它的全部延迟，没有更慢的档 |
| openjev（DiffusionGemma-26B-A4B） | 471 ms | 输出不是轨迹，只是几个选项 |

那行备注原来写的是「这还是它自己选择『不思考』的场景，真正走 CoT 会到几秒」。
那是**猜测**，已被第 15 条的实测否掉：150 帧 think rate = 0，think block 逐字节相同，
延迟平坦、p99 只比中位数高 10%。**AutoVLA 没有「更慢的档」，只有这一档。**

**没有一个接近 10 Hz。** 而且开销全在通用部件上——预处理、vision tower、prefill、逐 token decode——
**不在 planning head 里**。我们的 head 不到 0.1 ms，所以这条线的论点不依赖精度结论，
在第 1 条被证伪之后它反而是最稳的那部分。

**状态**：已确认（数字来自实测，口径的定义是我们自己的选择）。

---

## 12. DINOv3 用不了，换成 DINOv2 + V-JEPA 2 + SigLIP2

**决定**：不再把 DINOv3 列为对照 backbone，也不复现 RAP 这个竞品；
对照换成 **DINOv2**（单帧自监督，已在盘上）、**V-JEPA 2**（视频自监督）、**SigLIP2**（图文预训练），
三个都没有门禁，token 直接能下。RAP 只引用它论文里的 test 分数，不在本地跑。

**理由**：2026-09-20，DINOv3 的访问申请**被作者拒绝**（`facebook/dinov3-*`，理由未给）。
HF 上被拒之后不能再次申请。社区记录显示 Meta 系门禁模型对中国大陆的申请普遍被拒，
背后是 Meta 自己要遵守的出口管制和许可条款，所以这多半不是我们表格填得不好。

**不去找第三方转存**：作者是明确拒绝而不是没看到，绕过许可在合规上站不住，论文里也没法写来源。

**这个替换不是降级**：V-JEPA 2 比 DINOv3 更贴我们的问题——单帧结构上估不出速度，
而全项目的争议正是「转弯开始之前视觉够不够」（见第 3d 条）。视频预训练的对照直接回答
「缺的是不是时间信息」，而且 Drive-JEPA（arXiv:2601.22032）用的就是它，有外部参照。
真正的损失只有 RAP 跑不了（它的 encoder 初始化就要加载 DINOv3），而我们本来也只打算引用它的数字。

**状态**：已确认（门禁是外部事实，不由我们决定）。

**如果还是想要 DINOv3**：两条路都要人来做——用机构邮箱和真实单位信息换账号重新申请，
或者在仓库讨论区联系作者说明学术用途。不阻塞任何实验，随时可以补上。

**V-JEPA 2 那一行不能读成"谁是更好的 backbone"**（2026-09-20 补）：它要 64 帧的 clip，
也就是比 Qwen 单帧**多拿了历史**，赢了也可能整个是 history 效应而不是 backbone 效应。
唯一能把两者分开的办法是给 Qwen 同样的窗口，那就是 Stage B。所以这一行的诚实读法是
**"在 Stage A 的代价下，喂时间信息值不值"**，它是 Stage B 立不立项的前置证据，不是 backbone 排名。
另外两个变量也要在表里写明：它只喂 FRONT（主线是 front3），
而且按我们的时间跨度采 64 帧会改变表观运动速度（和它预训练的帧率不一致，是已知的 distribution shift）。

---

## 13. 「历史窗口完整」必须是精确命中，不能是容差命中

**决定**：判断一个多帧窗口是否完整，判据是**每个请求的 slot 都精确命中**，不是「在容差内找到一个近邻」。
padding 保留（为了数组形状规整），但 padding 出来的窗口**不计入 completeness**，也不能进入任何
用到 history 的比较。凡是有「吃历史」的行出现的表，**整张表**都限制到严格完整的样本上（clip-set 纪律），
否则各行看到的不是同一批数据。

**理由**：两边各踩了一次，形态不同但后果一样。

nuScenes 是**结构性**的：按时间匹配不规则 timestamp，请求间隔（83.3 ms）和相机原生间隔几乎相等
但不完全相等，64 个 slot 累积下来相位漂移，半步容差**在结构上不可能一直成立**。
mini 上只有 152/404 通过，远低于 scene 长度允许的 ~290；改成实测步长 + 要求整个跨度落在 scene 内
之后是 294/404，和算出来的一致。

Waymo 没有相位问题（`frame` 是整数 index，`frame - k*stride` 是精确整数运算），
但**容差 + 复制上一帧的 padding 被当成 completeness 报了出去**，高估极其严重：
6 s 窗口容差口径 0.749、严格口径 **0.0227**，差 **33 倍**；0.5 s 窗口 0.811 对 0.295。
之前报过的那组 `within_tol` 数字**不能当 completeness 读**。

**归因**（顺带验证了方法本身）：Waymo 的 completeness 几乎全部由**缺 shard**决定，不是 clip 几何。
工作期间正好落了 17 个 shard，coverage 0.138 → 0.317，单 slot completeness 0.125 → 0.305 几乎精确跟随，
而 `span_inside`（几何天花板）纹丝不动（0.9998 → 0.9999）。

**2026-09-21 更正（完整 val，93/93 shard、106360 帧）**：这一条原本写的满 val 天花板是
「1.5 s 98%、3 s 92%、3.5 s 90%、6 s 78%」，**偏乐观 3–6 个点**，实测是
**1.5 s 93.2%、3 s 86.4%、3.5 s 84.2%、6 s 72.9%**。
原因是 `span_inside` 当时用**整个 split 的最小 index** 当 clip 起点，而 val 的 sequence 长度并不一致
（p10 199 帧、p90 229 帧），于是把短 sequence 的开头几帧算成了「够得着」。改成**按 sequence 取下界**之后，
完整 val 上 `complete` 和 `span_inside` 吻合到千分之一（`missing_shards` ≤ 0.001），
即**剩下的不完整窗口全部是跨度越过了自己 clip 的开头，没有一帧在等 shard**——归因本身反而被这次更正坐实了。
另外 completeness 必须**按 split 报**：val 已满而 train 才 89/263，混在一起的 0.618 只是两者的混合比例。

**顺带的实用结论**：完整 val 上 clip-set 不再是瓶颈——1×5 窗口 103953 帧、3×5 99141 帧、3×10 91941 帧。

**状态**：已确认（代码里有断言：严格集的窗口精确命中每个 index；tolerance 只能增不能减；complete ⊆ span_inside）。

---

## 14. Waymo 的帧间隔是测出来的 10.00 Hz，不是假设

**决定**：`FRAME_DT = 0.1 s`。

**理由**：WOD-E2E 的时间字段**全为零**（`timestamp_micros`、`pose_timestamp`、`camera_trigger_time`、
`camera_readout_done_time`），只剩 10 ms 的 shutter，所以间隔只能从数据里测。
方法用的是弧长与坐标系无关：对同一 sequence 相隔 d 个 index 的两帧 A、B，A 在「截止到 d·dt 的 3.75 s」
内走过的距离必须等于 B 自己 `past_states` 整个 3.75 s 窗口走过的距离，解出 d·dt。
**22464 对 val 帧上 dt = 0.1000 s，IQR ≈ 0，从 gap 2 到 gap 40 答案完全一致。**

**一个没对上的地方**（已在 docs/waymo-e2e.md 标为未解决）：按 10 Hz，val 的 index 0–238 是 23.8 s、
test 8–149 是 14.1 s，都和 challenge 页面「20 s clip、前 12 s 可见」对不上。
test 的提交帧在 index 148–150，正好是 20 s clip 结束前 5 s（一个 horizon）。
**10.00 Hz 这个数是硬的；index 到 clip 内绝对时刻的映射存疑**，但我们算的任何东西都不依赖它。

**查过的两件事**（2026-09-20，29 个 shard，33208 val 帧 / 738 test 帧）：
低 index 的帧**已经带有完整 4 s 历史**（index 8–12 的过去窗口中位位移 16.1 m，和中段帧的 19.6 m
没有区别，几乎没有 padding），所以 **index 0 不是历史起点**，`past_states` 来自 log 而不是存下来的帧范围。
另外两个 split **都从 index 8 开始**（val 33208 帧里只有 2 帧在 8 以下，test 一帧也没有），
test 停在 149，rater 帧在 147–150。

一个**纯推测的分解**（不作为结论）：test 149 + 隐藏的 5 s（50 帧）= 199，减去起始偏移 8 得 191 帧 = 19.1 s，
是 index 里最接近「20 s」的数；val 再往后多 39 帧（3.9 s），大小和 4 s 历史窗口相当。
自洽，但仍然是推断。

**论文里怎么写**：写「评分帧是可见范围的最后一帧，序列内 index 147–150」，
**不要写任何裸的「12 s」**——我们能看到的任何原点都推不出那个数。真要确定就去官方论坛问，别继续推。

**状态**：已确认（dt 本身）；index→绝对时刻的映射：存疑，不使用。

---

## 15. AutoVLA 的 adaptive gate 在分布外塌缩成「永不思考」

**观察**（2026-09-21，实测，不是文献）：在 WOD-E2E val 上按我们自己的三个 stratum
各抽 50 帧（straight / turning / pre-onset，每个 sequence 最多一帧，seed 固定），
AutoVLA 的 **think rate = 0/150**（95% 上界 2%）。150 次输出的 think block **逐字节相同**，
都是 "This is a straightforward scenario, and a direct decision can be made"。
延迟平坦在 **1.4 s**，p99 只比中位数高 10%，三个 stratum 之间的差异在噪声内。
有效性检查过：右转帧的轨迹终点确实在右侧 10–11 m，所以输入喂进去了，是模型自己的决定。

**为什么重要**：AutoVLA 是在 nuPlan 上、带 CoT 长度惩罚做 RFT 并在那里评测的。
换到分布外的 Waymo，那个本该 adaptive 的 gate 饱和在一端。
**把 gate 当成成本惩罚训出来，不等于得到一个可迁移的 when-to-think 策略。**
这条直接影响我们推给第二篇的 gate 计划（见第 1 条的分阶段表 D）：
在动手做 gate 之前，必须先证明它跨分布还成立，否则只是学到了训练分布上的一个常数。

**连带修正了「延迟依赖输入」这根柱子的归属**（第 11 条）：
openjev 的延迟随输入变化，是因为它的 re-read 策略在**运行时**对自身不确定性作出反应（333 → 471 ms）；
AutoVLA 没有尾巴，是因为它的决策在**训练时就冻死了**，而且这个 gate 在这里一分钱也没省下
（1.4 s 本来就是 prefill 加 44 个 token，跳过 reasoning 只是避开慢档，没有换来快档）。
三者对比其实更干净：**运行时反应的系统有尾巴，训练时冻死的 gate 既没有尾巴也没有收益，
而我们的系统没有尾巴也不需要尾巴。**

**状态**：已确认（在我们的抽样分布上）。**没测**它在 nuPlan / NAVSIM 上会不会思考——
要断言「gate 不可迁移」而不是「gate 在 Waymo 上不触发」，需要补这一条。

---

## 16. CARLA 能在这台无头机器上跑（闭环不再是不可能）

**事实**（2026-09-21 实测）：CARLA 0.9.15 在 RTX PRO 6000 Blackwell（sm_120、驱动 595.71.05）上
以 `-RenderOffScreen` 正常渲染，**画面是真的**（相机像素统计通过，不是全黑），
渲染确实在 NVIDIA GPU 上（5990 MiB、50% 利用率），不是退回 CPU 软件渲染；
Python 客户端连上后同步世界跑到 **1.69 倍实时**——但**这是 server 侧天花板，不是闭环速率**：
冒烟测试的相机回调不阻塞，量的是 server 能多快 step，而真正的 agent 要等这一 tick 的每一帧落地。
同配置的闭环口径慢将近 3 倍（1 相机 1600×900：92.4 ms/tick，0.54× 实时），见第 17 条。

**关键前置条件**：容器里原本没有可用的 Vulkan——NVIDIA 的 ICD 文件在，但初始化一律失败，
根因是 NVIDIA 的 Vulkan 驱动会 `dlopen` `libEGL.so.1` 而容器缺 GLVND 的 dispatch 库。
**装 `libegl1` 即可**，不需要自己编 Vulkan loader（试过，没用）。详见 [docs/carla.md](../docs/carla.md)。

**冒烟测试没有证明的事**（不要把绿灯读成可以跑 220 条路线）：
长时间稳定性完全没测。已知两个开放问题，都不是 Blackwell 特有、在原版 0.9.15 上也复现：
Town12 的 sensor-dormancy segfault（**220 条路线里有 104 条是 Town12**），
以及单卡长时间无人值守运行会 hang 的报告。**闭环评测的典型失败是第三小时卡死，不是起不来。**

**最优并发是 4 个实例，卡在 GPU 上**（2026-09-21 实测，单相机 1600×900）：
1 / 2 / 4 / 5 个实例的总吞吐是 28.2 / 46.8 / **86.5** / 82.0 FPS——**第 5 个是净亏的**；
GPU 在 5 个时已 99%，而显存只用 30/96 GB、CPU 只用 16.7/25 核；第 6 个实例在启动阶段
`Signal=11` segfault，不是变慢。启动必须错开——4 个同时启动会有一个 world-load 超时，
错开 20 s 之后 0 失败且总吞吐反而高 17%（74.2 → 86.5 FPS，3.07 倍单实例）。
**所以"大显存卡对闭环有用"是错的，一张小得多的卡也是同一个上限**；
之前按 25 核 / 单实例 4.5 核外推出"核心是瓶颈"，数字碰巧对，理由不对，已在
[carla-efficiency.md](carla-efficiency.md) 里改正。

**220 条路线一轮的成本（初步外推，不是实测）**：纯仿真器下限（假设 agent 免费、
典型路线 1000 tick、4 实例）单相机约 0.7 h、六相机约 2.5 h。但**模型推理和渲染抢同一张卡**，
不是加在旁边，所以真实成本要高得多：轻量 agent 约一天，六相机 BEV 模型几天。
外部对照：TF++ 用 8×2080Ti 约 4 h（≈32 GPU-hour），UniAD/VAD 用 4×A6000 报 "several days"。
**所以闭环是"天级"，不是"小时级"**——这正是效率专题存在的理由，见
[carla-efficiency.md](carla-efficiency.md)。

**真要跑 Bench2Drive 还需要**：AdditionalMaps 6.9 GB（165/220 条路线依赖它）。
另有一个环境冲突要解：CARLA 的 tarball 只带 cp27/cp37 的 Python 绑定，没有 cp38，
而 Bench2Drive 的文档要求 3.8——可行路径是用 PyPI 的 `carla==0.9.15` wheel。

**状态**：可行性已确认；**要不要做仍未决定**。决定要看单实例帧率和并发数外推出来的成本，
以及第 1 条被证伪之后论文的形态。闭环能提供开环给不了的证据，但工程量和风险都在上面这几条里。

## 17. 闭环评测是小时级，不是天级——但便宜是「接得对」换来的，不是卡换来的

**事实**（2026-09-21 实测，未修改的 Bench2Drive 0.0.4 leaderboard 跑真实路线，
完整数据见 [docs/bench2drive-cost.md](../docs/bench2drive-cost.md)）：

44 条 base-package 路线、4 个 worker、Qwen3-VL-4B 真的在环里，**0.71 小时**跑完。
外推 220 条：优化后 8 实例 **约 1.1 小时**，按 Bench2Drive 原样接则 **约 15 小时**，
同路线同卡差 **13.9×**。**结论：闭环评测是小时级，不需要换机器。**

**时间花在哪**：3 相机 1600×900 的一个 tick 是 158 ms，其中 **128 ms 是阻塞等相机帧**（79%），
scenario tree 25 ms，`world.tick()` 只有 8 ms。

**三条反直觉的、会改变做法的结论**：

1. **相机成本是 per-sensor 不是 per-pixel。** 像素砍到 1/16 只快 3%（在噪声带内），
   每多一个相机固定加 33 ms/tick。降分辨率没用，**减帧数（decimate）才有用**：2/4/10 → 1.76×/2.12×/3.36×。
   前提是先修一个 leaderboard 的 bug：`_preprocess_sensor_spec` 的白名单里没有 `sensor_tick`，
   agent 要 5 Hz 会被静默地给 20 Hz，所以 decimate 在修之前**一点用都没有**。
2. **按模型输入尺寸直接渲染是单笔最大收益，而且免费。** 仿真侧不要钱（与分辨率无关），
   policy 侧省掉 resize：Qwen 三相机 139 → 59.9 ms，DINOv2 21.9 → 5.3 ms。
   连带纠正：我们 [waymo-e2e.md](../docs/waymo-e2e.md) 的 127–216 ms/frame 是离线吞吐，
   **闭环里预处理占延迟的 70%**。论文里要写清 800×450 直出 ≠ 1600×900 降采样。
3. **不要写 Cython。** 热点是唯一一个函数（`RouteLightsBehavior._turn_close_lights_on`，占总时间 14.5%），
   而它花时间在「为不变的数据反复发 RPC」。语义等价地缓存掉之后 scenario tree 快 3.2×，
   **wall clock 一点没变好**——leaderboard 的 Python 本来就藏在 sensor 管线后面。**单实例下 H4 天花板是 0。**

**并发上限是配置的属性，不是机器的属性**：第 16 条之后记的「4 实例封顶、第 6 个 segfault」
是在饱和负载（1 相机每 tick 1600×900）下测的。换成优化配置，**6 个和 8 个实例都跑得干干净净**，
总吞吐 1.37×/1.62×，8 个时 CPU 才到顶（23.8/25 核），显存 59/96 GB。

**稳定性**：3.5 个 worker 小时里 **CARLA 本身一次都没崩**。12 次重启全部是我们自己的
traffic-manager 端口复用（TM 的 RPC server 在客户端进程里）。
**没有排除的事**：Town12 的 sensor-dormancy segfault 够不着（base package 只有
Town01–05 和 Town10HD，**Town06/07 也不在里面**，220 条里只有 44 条能跑）；
「长时间无人值守第三小时卡死」也没有被 3.5 小时排除。

**状态**：成本结论已确认（在 base-package 路线上）；**要不要做闭环仍未决定**。
要定下来还需要：AdditionalMaps 6.9 GB（Town12/13 是 151 条，地图大得多，按贵一倍算 3–4 小时），
以及一次真正跨夜的无人值守运行。

---

## 18. Bench2Drive 上「低延迟」不再是卖点，latency injection 已被做过两次

**决定**：**不要把「我们的 head 快」或「闭环里注入真实延迟」写成论文的 contribution。**
[frozen-vlm-planner.md](frozen-vlm-planner.md) 的 claim 4（效率）要按这一条重写或删掉。
闭环要不要做，从此是一个纯粹的「值不值」问题，不再有「我们有独家角度」这个理由撑着。

**理由**：2026-09-21 第三轮 deep research（原始报告在 `research/lit/2026-09-21-round3-*.md`，不进 git）。
下面每一条都有出处，出处写在报告里。

**(a) benchmark 很挤，而且顶部饱和。** 社区 leaderboard **101 条**（2025 年 49、2026 年 49），
近 12 个月 arXiv **114 篇**，近 3 个月 **28 篇**，其中 VLA 一线 34 篇。
HiDrive 明说 *"increasingly saturated, SOTA achieving near-perfect scores"*。
更难的派生榜已经在围它：Safe2Drive 把 LEAD 从 94.70 打到 39.95、SimLingo 85.07 打到 41.00；
Fail2Drive（Geiger 组）做 paired-route 泛化，SOTA 平均 SR 掉 22.8%，直指 *"success may reflect memorization"*。

**(b) 快 = 差 这条 trade-off 在 2026 年翻转了，Pareto front 已经在我们的目标区间。**

| 方法 | 延迟 | DS / SR |
|:--|--:|:--|
| ADT (2606.02105) | 19.2 ms | 77.90 / 55.00 |
| **FIVE-VLA (2609.18623，2026-09-16)** | **33 ms (A100)** | **90.95 / 77.27** |
| SimLingo-BASE | 41.1 ms | 85.94 / 66.82 |
| ORION (7B) | 806 ms | 77.74 / 54.62 |

我们实测的 batch 1 约 33 ms，**和 FIVE-VLA 同一格而分数差一大截**，不构成 contribution。
而且同一篇论文内部加算力买不到分数：ETA 从 102 ms 降到 50 ms 只掉 4.8 DS，
把 encoder 从 308M 放大到 1011M 反而 −1.1 DS；LinkVLA 从 361 ms 降到 48 ms 反而 +0.35 DS。

**(c) 我们「竞品都是 0.5–1.4 s」这个前提只对 7B 以上成立。** 1–4B 级别别人自己报的是
**150–300 ms**（AutoVLA 147 ms、DriveVLM-Dual 300 ms、Alpamayo+FlashDrive 151 ms）。
我们在 [docs/baselines.md](../docs/baselines.md) 测到 AutoVLA 1362 ms，那是 eager HF `generate`
的工程问题，不是路线差异——第 11 条已经写了「这些数字是 as released」，
但**不能据此说 VLA 这条路线本身慢**。这是对第 11 条叙事的限定，不是推翻它。

**(d) frozen backbone 和 anchor scoring 两件都已经被单独做过，而且分数很高。**

| 方法 | 冻什么 / 词表 | Bench2Drive DS / SR |
|:--|:--|:--|
| BLUE (2606.08684, EMNLP26) | 整个 SimLingo VLA 冻死，只训 **0.11M** | **90.58 / 76.18** |
| AnchorVLA (2607.03182) | stage-2 backbone 冻结，**K=100 k-means anchors** | **89.92 / 77.28** |
| SparseDriveV2 (2603.29163, ECCV26) | ResNet-34 + **262,144 anchors** | **89.15 / 70.00** |
| Orion-Lite (2604.08266) | EVA-02-L + QT-Former 冻结，0.1B head | 80.57 / 55.45 |

三项交叉（**通用** frozen encoder × k-means 词表 × 闭环）确实没人凑齐过，
但那是三个已知组件的拼装，不是新机制。

**(e) 「闭环注入真实延迟」被 scoop 两次，其中一次是 Bench2Drive 原班人马。**
- **RTS**（2601.07393，2026-01）改 CARLA 同步模式，*"dynamically records the forward inference time
  of the algorithm and returns it to the ScenarioManager"*，就在 Bench2Drive 220 条上跑。
  还给出了 fixed-delay 做不出的结论：*"models with long-tailed latency distributions exhibit
  noticeably lower Driving Scores"*——**这正是我们本来想讲的故事**。
- **Bench2Drive-Robust**（2605.18059，2026-05，Xiaosong Jia / Junchi Yan，代码公开）注入
  固定延迟：SimLingo 从 85.94 掉到 **28.45**（100 ms）。**它的 repo 里已经有 `INFERENCE_LATENCY_MODE`
  的 measured 模式**，论文原文说主表用 fixed delay 是 *"to ensure comparable severity across
  models and machines"*——而这正是审稿人会拿来打我们的那条理由。

**(f) 现实落点。** 榜首比的是 data pipeline 不是 architecture：官方板 top-2（TFv6、SimLingo）
**都不用 Bench2Drive 训练集**，用自采的 PDM-Lite / LEAD 数据（SimLingo 310 万样本、8×A100）。
用 B2D Base（1000 clips）训的方法全在中下段：Drive-π0 69.71、DriveTransformer 60.20、UniAD 38.69。
**最接近我们架构的已发表点是 Drive-JEPA：DS 64.52 / SR 36.82，而且它的 encoder 是微调的、不是冻结的。**
所以 frozen + 词表 + B2D Base 的预期区间是 **DS 55–70 / SR 30–45**，在 101 条里排中段，
**单凭这个数字不可发表**。

**状态**：**已确认**（文献事实，不是我们的测量）。由此产生的判断——claim 4 要重写——**已决定**。
**闭环要不要做仍未决定**，但理由变了：不再是第 16、17 条的「能不能跑得起」（能，且便宜），
而是「跑出来能说什么」。第 16 条那个 Town12 顾虑也不再是拦路虎（b5 2026-09-21 查明是首次
`ImportAssets.sh` 之后冷 shader 缓存导致的 300 s 超时，无 sensor 加载 36 s 成功）。

**怎么才能翻案**：如果我们能证明 measured-latency 模式下**榜单会重排**（SimLingo 100 ms 就塌到
28.45 这个 cliff 强烈暗示会），那仍是一篇论文——但那是把 (e) 的两篇合起来跑，
新意在**硬件归一化协议**（攻他们放弃 measured 模式的那个唯一理由），不在 latency injection 本身。

**若仍要做 Bench2Drive**：锁 CARLA 0.9.15；**至少 3 seeds 并报 per-seed std**
（官方是单次跑，issue #233 同 route 同 model 两次跑出 **21.14 对 100.0**，
NeurIPS checklist 自承 *"did not report error bars"*，官方数字本身还含人工 retry）；
ablation 用官方 Dev10（10 条，约 7 GPU·h，官方 README 推荐）；显式披露 crash/retry 不要静默记 0；
**v0.0.3 与 v0.0.4 的数字不可混排**（中段 ±10 DS 乱跳，TCP 与 UniAD 的相对次序会反转，
且外部团队至今无人在 v0.0.4 上发表）。

---

## 19. real↔sim 迁移已经有人做了，但做在 neural simulator 上；HUGSIM 是我们缺的那一列 control

**决定**：如果做「driving pretraining 能不能扛住 freezing 和 domain shift」这条线，
**评测必须是 NAVSIM + HUGSIM + Bench2Drive 三列，不是 NAVSIM + Bench2Drive 两列**。
同时，**不要再说「没人评测过 real-video pretraining 的闭环迁移」**——说过头了，会被一击打掉。

**理由**：2026-09-21 第三轮 deep research 的第四份报告
（`research/lit/2026-09-21-round3-sim-real-transfer.md`，不进 git）。
**覆盖度限定**：该 agent 的 WebSearch 配额在启动前已耗尽，全部结果来自 arXiv API + 定向 WebFetch，
**偏 arXiv**，workshop proceedings 和 project page 那一层没扫到。下面的「没人做过」都带这个限定。

**(a) 「没人做过」这个说法被证伪了，但有一个更窄的版本是真的。**

| 精确命题 | 裁决 |
|:--|:--|
| 没人把 real-video-pretrained encoder 放进 CARLA closed-loop | **假**——Drive-JEPA 就是（330 h real video → Bench2Drive DS 64.52） |
| 没人评测 real-video pretraining 的闭环迁移 | **假**——WA-JEPA 在 HUGSIM 上 zero-shot HD-Score 0.4462；DriveZero 在 HUGSIM；VaVAM 在 NeuroNCAP |
| **没人在 CARLA setting 下 ablate encoder 的 pretraining source** | **真** |
| **没人在冻结条件下做这件事** | **真**——Drive-JEPA 的 ViT 是 lr 1e-5 微调的 |
| **没人量化 real→CARLA 的 feature-space gap** | **真**——扫了约 24 篇相关工作，driving 场景下零 |

**(b) 两篇几乎就是我们的设计，novelty 必须相对它们重新定位。**
- **DriveZero**（[2609.06055](https://arxiv.org/abs/2609.06055)，2026-09-05）：DINOv3 + SigLIP2 + SAM +
  Depth-Anything-V2，**四个全部冻结**，接薄 head，在 **HUGSIM 闭环** + NAVSIM + nuPlan 上评测。
  和我们的差别只有两点：simulator 选了 HUGSIM 不是 CARLA，以及**它把多个 VFM 并起来用，
  没有做 per-encoder 的横向 ablation**。
- **WA-JEPA**（[2608.20974](https://arxiv.org/abs/2608.20974)）：nuPlan video 上的 JEPA 预训练，
  **HUGSIM 上 "without HUGSIM-specific fine-tuning"**，HD-Score 0.4462（best）。
  这是 real-video pretraining 零样本闭环迁移的直接证据。

**(c) sim→real 方向证据充分且基本为正，没有一篇报负。**
JiSAM（[2503.08422](https://arxiv.org/abs/2503.08422)）用 CARLA LiDAR 预训练 + **2.5% real labels**
就达到 full-real 水平；LEAD（[2512.20563](https://arxiv.org/abs/2512.20563)，CVPR 2026）一套 pipeline
在 Bench2Drive 95 DS 和 NAVSIM/Waymo 上同时 gain。
但 JiSAM 明说 **naive 迁移会失败**，要靠 domain-aware 设计救——说明 gap 确实大。

**(d) CARLA 不是「太 OOD 以至于结果没信息量」，这一点之前没验过。**
Bench2Drive 分数跨度 38.65（VAD）到 95.59（LEAD），**有分辨力**；而且 real internet VL 预训练的
SimLingo 能到 **86.55**，所以「real pretraining 在 CARLA 里天然废掉」是错的。
**但 SimLingo 是微调的**，而微调过的 Drive-JEPA（real driving video）只有 64.52。
这 31 分**不能归因于 domain gap**——LEAD 用 privileged expert 蒸馏 + 自采数据，架构和训练规模都不同。
**没有任何 controlled comparison 把 "pretraining domain" 这个变量隔离出来过，这正是空白所在。**

**(e) 为什么必须加 HUGSIM 这一列（本条最实用的部分）。**
只有 NAVSIM + Bench2Drive 两列时，**任何掉分都可以被 reviewer 解释成「闭环本来就更难」**，
而不是 domain shift——这个反驳我们答不了。**HUGSIM 是闭环但 render 自真实数据（3DGS 重建
KITTI-360 / Waymo / nuScenes / PandaSet），它正好把「闭环难」和「合成图像难」两个混淆因素分开。**
所以**最干净的对比对是 HUGSIM vs Bench2Drive，不是 NAVSIM vs Bench2Drive。**

可行性：MIT license、ungated、**重建好的 scenario 官方已放出，不用自己跑 3DGS 训练**；
3DGS 光栅化单卡 real-time，96 GB 远超需求；400+ scenario 跑一轮是**小时级**。
而且 DriveZero、WA-JEPA、Latent-WAM、MM-Future 都在 HD-Score 上有数，**我们的结果直接可比，
不用自建 baseline**。对照：NeuroNCAP 场景太窄（三类）且公开 baseline 只有 UniAD；
**AlpaSim 没有公开释出**；DriveArena 的 code 状态不明。

**(f) 一条必须提前准备的反向证据。**
VaViM/VaVAM（[2502.15672](https://arxiv.org/abs/2502.15672)）发现 **scaling real-video pretraining
会改善 open-loop，但 NeuroNCAP 上的 collision rate 反而上升**，作者归因于 overfit 到
trajectory-following 而非 adaptive decision-making。**我们很可能复现出同样的反转。**
那不是坏消息——它就是论文的 punchline——但要在开跑之前想好怎么解释，不要到 rebuttal 才第一次遇到。
建议按第 3d 条的办法**预登记**：反转出现怎么写、不出现怎么写，跑之前写死。

**状态**：文献事实**已确认**（带 (a) 的覆盖度限定）。
由此产生的实验设计要求——**三列而不是两列**——**已决定**。
**整条线做不做仍未决定**，那要和第 18 条一起给用户定。

**下一步的先决条件**：**先读 DriveZero 和 WA-JEPA 全文**。这两篇决定我们还剩多少 novelty，
在读完之前不要写主题文档，也不要开始抽特征。

---

## 20. L0：intervention 信息是「线性存在但被均匀目标低估」，还是「根本不在特征里」（**预登记，B/C/D 数字待填**）

**这条在看到任何 B/C/D 数字之前写下来。** 第 3d 条测出的 DiD 为正——视觉的增量在 pre-onset 上
**比在 straight 上更小**——有两种完全不同的解释，而它们指向的下一步实验是相反的：

1. **readout 层面**：判断「要不要转」的信息其实线性可读出（linear probe 意义上：冻结 backbone、
   只训一个线性 head，用来测信息在不在特征里），只是 ridge 的均匀 MSE 目标被 9 万多帧直行样本主导，
   1510 帧 pre-onset 在损失里几乎没有权重，所以 head 学不到它。那么改目标（加权、换更强的 head）就该有收益。
2. **representation 层面**：Qwen3-VL 的 frozen feature 里本来就没有这个信息的线性方向，
   换目标、换 head 都没用，下一步必须动表征（换层、换 encoder、finetune、时序输入），不是动 head。

**L0 就是用来把这两条分开的。** 它不新抽任何特征、不碰 RFS scorer、不动闭环，全部在第 3d 条那套
半 val 设置里跑：val 按 sequence 对半切，一半 fit 一半 eval，λ 用 fit 半内部的 sequence-grouped CV 选，
两个方向都报。base 统一是 `ridge ego`，所以 ΔADE = arm − ego，逐帧配对。

**kinematic surprise（本条的加权变量）**：`s_i = ADE(真实 future_i, CV 外推_i)`，5 s horizon，
CV 用 `jevdrive/waymo.py` 里 `baselines()["cv"]`，即「保持当前速度、yaw rate 取零」的直线外推，
和 3c 条那张 baseline 表用的是同一个。权重在 fit 半上归一化到均值 1。

### Arms

| arm | 内容 |
|:--|:--|
| A | **复现**：均匀 `ridge_late`，必须在 bootstrap 噪声内重现 3d 的数字，否则停下来报告，不往下做 |
| B | **surprise 加权**：WLS（把 X、Y 的行乘 √w_i），w_i ∈ {s/mean(s), (s/mean(s))², 1+α·s/mean(s)（α=1,4），以及硬筛：只留 fit 半 s 中位数以上 / 75 分位以上的帧} |
| C | **只在 pre-onset 上 fit**：`ridge_late` 只用 fit 半的 pre-onset 帧训练，在 eval 半的 pre-onset 上评估。这是「线性读出这批冻结特征」在这个子集上的**上界** |
| D | **更强的 head**：`jevdrive/planner.py` 里已有的 MLP head，均匀 vs B 里最好的那套加权 |

B 里「最好的那套」由 **fit 半内部的 grouped CV 上的 pre-onset ADE（不加权）** 选出，
**不允许**看 eval 半的结果来选。

**token attention head 这一支直接 deferred**：缓存下来的 feature 是 per-frame pooled 向量
（`L{09,18,27,36}_{mean,last}` 每个 2560 维、`vis_mean` 2560、`vit_mean` 1024，全是 (n, d) float16），
**没有存 spatial token**，所以 attention pooling 无从做起，要做得重新抽一遍特征，超出 L0 的范围。

### 判据（跑之前写死）

**实用效应门槛：pre-onset 的 ΔADE 要达到 −0.05 m 才算有实用增益。** 理由是 pre-onset 的 ADE 本身约 1.3 m，
0.05 m 约合 4%；而 3d 半 val 上 delta 的 CI 半宽实测 0.045 m，比这个门槛更小的效应这套设置根本判不动。
落在 [−0.05, +0.05] 之内的一律按「没有实用增益」读。

| 分支 | 触发条件 | 结论与下一步 |
|:--|:--|:--|
| **分支 1（readout 层面）** | 某个 B 方案或 C 在**两个方向上**都给出 pre-onset ΔADE ≤ −0.05 m 且 CI 不跨零，**并且** DiD 移动到 ≤ 0 | 信息线性存在于冻结特征里，均匀目标确实低估了它。下一步在 head/目标层面做 |
| **分支 2（representation 层面）** | 连 C（只在 pre-onset 上 fit）都把 pre-onset ΔADE 留在 [−0.05, +0.05] 之内 | 线性读出这批冻结特征在这个子集上已经**榨干**了，下一步必须动表征，不是动 head |

**零和检查（两个分支都要做）**：pre-onset 的增益必须和 straight_yaw 的 ΔADE 损失并排报。
如果 pre-onset 赚到的被 straight 亏掉的等量或更多地抵消，就要明确写一句
**「重新加权只是把误差在子集之间搬了个家」**，不能只报赚的那一半。

**这条不是押注**：它不预测哪个分支会中，它的作用是让下一次实验的方向在看到数字之前就被定死。

### 跑之前就能算出来的两件事（CPU，已测）

**一、s 的分布：s 主要衡量的是「刹车/起步」，不是「转不转」。** 完整 val 106 360 帧：

| 子集 | n | mean s | q10 | q50 | q90 | q99 | 纵向分量 | 横向分量 | 横向占比 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 106 360 | 2.81 | 0.21 | 1.95 | 6.64 | 11.71 | 2.33 | 0.82 | 0.29 |
| pre_onset | 1 510 | 4.25 | 1.65 | 3.84 | 7.57 | 10.94 | 2.80 | 2.67 | 0.63 |
| straight_yaw | 46 580 | 2.67 | 0.37 | 1.79 | 6.22 | 11.70 | 2.60 | 0.24 | 0.09 |
| turn_yaw | 11 060 | 5.37 | 1.67 | 4.80 | 9.94 | 13.91 | 2.48 | 4.09 | 0.76 |

s 的顶层 decile（阈值 6.64 m，10 637 帧）里 pre_onset 只占 **2.2%**（val 底噪 1.4%，富集仅 1.6 倍），
turn_yaw 占 33.0%（底噪 10.4%），straight_yaw 占 36.9%，还有 27.9% 三个子集都不属于。
`corr(s, |a0|) = 0.49`，`corr(s, v0) = 0.07`——**s 和纵向加速度强相关，和是否处在转弯决策点只是弱相关**。
所以 B 的加权与其说是「加权 intervention」，不如说是「加权刹车和起步」。这是 L0 设计本身的一个弱点，
先记在这里，结果怎样都不改判据。

**二、噪声检查：val 里 s 最高的 50 帧，没有一帧是数据 artifact。** 逐帧看了 ego history 的
速度跳变、重复点、future 的速度剖面和 rater 状态：`past_jump`（相邻 0.25 s 区间的速度差）最大 1.47 m/s，
val 的 99.9 分位是 1.84，全部在正常范围；重复点 0 帧；第 13 条那套「精确命中」的历史完整性问题在这里
不适用，因为 past_states 是每帧直接从 tfrecord 里读出来的 16×0.25 s，不是跨帧拼的窗口。
这 50 帧只来自 **6 个 sequence**（相邻帧高度自相关），平均 v0 = 11.2 m/s、a0 = −1.46 m/s²，
横向分量只占 5%——它们全是**急减速**，外加一个从静止起步冲到 17 m/s 的路口场景
（那条 sequence 的 yaw rate 估计是 ±290°/s 的噪声，但 `subsets()` 的 validity guard 已经把它挡在所有
yaw 子集之外，不影响任何子集数字）。**artifact 计数：0/50。** 真正的威胁不是脏数据，是上面第一点：
s 选出来的是纵向事件，不是决策点。

**状态**：预登记，**待定**。A 复现之后跑 B/C/D，结果填进本条。

