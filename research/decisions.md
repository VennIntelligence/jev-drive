# 决定记录

跨 session 的共同决定都记在这里，一条一个小节。多个 Claude session 同时在这个 repo 上工作，
口头达成的东西不落盘就会丢，或者被下一个 session 用不同的假设覆盖掉。

每条写四件事：**决定**、**理由**、**状态**、**怎么才能定下来**。

状态只有两种：

- **待定**：已经按它执行，但支撑它的证据还不够，随时可能翻案。所有下游结论都要跟着标注不确定。
- **已确认**：证据够了，可以写进论文。改它需要新的证据，并在这里记下改动。

---

## 1. 论文框架：视觉是在 ego-state prior 之上的增量

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
从 0.772 降到 0.743（−0.029 m，CI [−0.040, −0.019]），trust-region miss 从 0.367 降到 0.338
（−0.029，CI [−0.052, −0.012]）。但在 pre-maneuver-onset 子集上**看不到增量**：ADE 1.331 → 1.297，
两条 CI 大幅重叠，miss 反而从 0.889 变成 0.919。这个子集在 nuScenes 上只有 135 个样本，
CI 半宽 ±0.15 m，测不动 3% 量级的差异。所以框架本身不变，但**「onset 上视觉有增量」这一半仍然是未验证的**，
要靠 Waymo。

---

## 2. 指标：RFS 优先，ADE 并列汇报

**决定**：Waymo 上的开发循环盯 RFS（Rater Feedback Score，官方指标，把预测和人工评分过的轨迹比），
ADE 只作为并列数字报出来。

**理由**：官方 ADE 是对着评分最高的 rater 轨迹算的。logged future 本身得分 2.63 m，
而公开最好的 test ADE 是 2.65 m，说明 ADE 已经饱和，只用 ego 的 baseline 也能逼到一米以内。
另外约一半的 ego-only 预测落在所有 rater trust region 之外，被罚到 4.0，RFS 那里才有空间。
`jevdrive/waymo.py` 里有官方 RFS 的 bit-exact 移植。

**状态**：待定。

**怎么才能定下来**：2.63 m 这个数字目前是在**仅有的 68 帧 val** 上算的，样本很小。
b5 会在 val 下到 46 个 shard（约 240 帧）时给出更新的数字。如果那时 logged future 的得分明显偏离 2.65，
「ADE 已饱和」这个前提要重新判断，指标策略也要跟着改。

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
ADE 那边没有这个问题：现在 226 帧，完整 val 约 1750 帧，够用。

**状态**：**已确认**。这是数据集的构造决定的，再多下载也不会改变。

---

## 3c. 核心论据：ego state 在 onset 前是瞎的

**决定**：论文的核心论点用这个对比来立：**知道当前 yaw rate，在车已经在转的时候值 1.03 m ADE@5s，
在转弯开始之前只值 0.02 m，差 40 倍。**

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
但 n 只有 135、CI 半宽 ±0.15 m，nuScenes 上这个子集根本测不动。**Waymo 上约 1750 帧时必须重做，
这是整个项目最关键的一次测量。**

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

**决定**：box 的下载串行执行，顺序是 navsim → HF 模型（Qwen3-VL-32B、AutoVLA）→ Waymo val → Waymo small。
Waymo train 要等 val 完成后再确认，test 最后（提交限制是每 30 天 6 次，不急）。

**理由**：box 的总下行带宽只有约 12–18 MB/s，所有任务共享。并行跑的时候 Waymo 只有 1.3 MB/s，
ETA 336 小时；串行之后单个任务能拿到约 10–16 MB/s。

**状态**：已确认（队列在 box 上 `$DATA_DIR/tmp/dlq.sh`，窗口 `jev:dlq`）。

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

**怎么才能定下来**：在 Waymo 上用真 RFS 的 trust region 重做这个扫描（nuScenes 上用的是
从 `waymo.py` import 的官方几何，但阈值是 nuScenes 尺度的替身）。

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

**状态**：已确认（数字来自实测，口径的定义是我们自己的选择）。
