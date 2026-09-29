# navhard 上 openpilot Cinque 的 EPDMS 亏在哪：分项拆解与不训练 base model 的补救估计

2026-09-29。纯 CPU 分析，读的全是已有的逐 token 打分 CSV，没有新跑任何打分。数据与脚本：[scripts/navhard_deficit.py](../scripts/navhard_deficit.py)，输出表在 [results/navhard-deficit/](results/navhard-deficit/)。
Cinque `none` 的 navhard 结果来自 [decisions 第 66 条](decisions.md) 与 [op-leaderboard todo](../todos/2026-09-29-op-leaderboard.md)，逐 token CSV 在 box 的 `runs/navsim/eval/v2_navhard_two_stage_opi_lb_navhard_gimm-cinque{,.lc_m1.0}__base/`。

## 结论先行

1. **亏分几乎全在 DAC（drivable area compliance，本车框四角是否都在可行驶区内，0/1 乘性惩罚）**。对 ZTRS（HF 榜 48.1，我们在同一 devkit 上复现出 48.15）逐 token 换项：把 ZTRS 的 DAC 换给 Cinque，combined 涨 **+7.8**，而全部 9 项一起换只涨 +11.5。NC、TTC、EP、TLC、HC 换过来都是 0 上下（−0.35 到 +0.1）。所以「差在碰撞或速度」不成立，差在开出路面。
2. **stage 2 比 stage 1 更差，且差在同一项上**：Cinque 的 stage 2 / stage 1 EPDMS 比是 46.9 / 71.7 = 0.65，ZTRS 是 60.7 / 78.9 = 0.77，DriveZero 论文是 69.1 / 82.3 = 0.84。DAC 相对 ZTRS 差 8.3（stage 1）与 16.8（stage 2）分。
3. **没有便宜的规则能修 DAC**：按 plan 几何（4 s 横移、末端 yaw）判「危险」再回退到别的候选，一律变差（−0.2 到 −12）。DAC 失败不能从 plan 形状里读出来。
4. **不训练能拿的分是几个各 1–3 分的小项**：标准起步（EP）、帧间一致性（EC）、打分头切换。三项都是估计，合起来大约 +3 到 +7，到不了 mid-tier 的 48（需要 DAC 接近 ZTRS 才行，+7.8 的那一块）。
5. **op-adapt r2 的损失设计不针对这个缺口**：S_jev 的 NC、TTC、P 三项对应的正是 Cinque 没有缺口的项；DAC 项有，但 L_score 只用在有行人的 slot，普通驾驶帧被 L_distill 钉在原模型上；DDC、LK、EC 完全没有；读数里也没有 navhard。见第 5 节。

## 0. 口径

EPDMS（extended PDM score，NAVSIM v2 的规则打分）对一条 plan 的单 token 分数是

```
score = NC · DAC · DDC · TLC · (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC) / 16
```

乘性项四个：NC（no at-fault collision，本车过错碰撞，静物碰撞记 0.5）、DAC、DDC（driving direction compliance，逆行，部分逆行记 0.5）、TLC（traffic light compliance，闯灯）；加权项五个：EP（ego progress，进度相对 PDM-Closed 的比值，封顶 1）、TTC（time to collision within bound）、LK（lane keeping）、HC（history comfort）、EC（two-frame extended comfort，相邻两帧 plan 的加速度 / jerk / yaw rate / yaw accel 的 RMS 差不超阈值）。脚本对我们所有 CSV 逐 token 复算，与 devkit 的 `score` 列最大差 2e-16。

navhard two-stage 有 450 个 stage 1 真实场景（225 对相邻帧），每个 stage 1 场景对应约 12 个 stage 2 合成场景（3DGS 渲染的偏离后起点，起点在人类终点周围横向约 ±2 m 取）。官方 combined 不是逐 token 平均，是 devkit `calculate_individual_mapping_scores` 的组内乘积：每个 stage 1 场景的分数乘以它的 stage 2 场景的加权平均分，再对 450 个组取平均。所以**一个 stage 1 的零会把整组（含 stage 2 本来能拿的分）一起清掉**。

stage 2 的组内权重是以 stage 1 终点为中心的高斯核（σ² = 0.1，`scene_aggregator.py`）：终点离某个 stage 2 起点在约 1.9 m 以内，权重集中在最近的几个；再远就退化成均匀。本文用均匀权重复现 combined（Cinque 33.43 对官方 33.33，ZTRS 44.89 对 48.15，CV 11.08 对 11.48）。**ZTRS 的 +3.3 差额是这个权重给的**，Cinque 只有 −0.1，原因见 3.4。下面所有「逐 token 换项 / 反事实」的点数都在均匀权重口径，与官方值差不超过 0.1（Cinque），对 ZTRS 参照是偏低 3.3。

## 1. 分项分解

### 1.1 stage 1 / stage 2 原始子分（devkit 汇总行，与 HF 榜同一口径）

表内为 stage 内的加权平均子分（%），combined 为官方 EPDMS。ZTRS 复现（我们的 devkit）与 HF 榜几乎逐项一致（48.15 对 48.12，CV 11.48 对 11.48），所以榜上其他行可以直接对比。HF 数据来自 [榜单快照](../todos/2026-09-26-top10-intersection/raw_snapshots/agc2025-e2e-driving-navhard.public.md)（2026-09-26 抓取，修 bug 后口径），全表 [sub_scores.csv](results/navhard-deficit/sub_scores.csv)。

| 行 | EPDMS | NC 1 | DAC 1 | DDC 1 | EP 1 | TTC 1 | LK 1 | HC 1 | EC 1 | NC 2 | DAC 2 | DDC 2 | EP 2 | TTC 2 | LK 2 | HC 2 | EC 2 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Cinque `none`** | **33.3** | 98.0 | **89.3** | 98.3 | 75.0 | 98.2 | 94.7 | 97.8 | **37.8** | 90.9 | **73.6** | 90.2 | 66.1 | 87.0 | 48.1 | 97.4 | 45.7 |
| Cinque `lc@−1.0` | 33.0 | 98.2 | 89.1 | 99.0 | 75.1 | 98.0 | 95.8 | 97.8 | 36.9 | 90.2 | 72.0 | 90.6 | 68.0 | 86.8 | 48.3 | 97.3 | 50.1 |
| ZTRS（HF） | 48.1 | 98.9 | 97.6 | 100.0 | 66.7 | 98.9 | 96.2 | 96.7 | 44.0 | 91.1 | 90.4 | 95.8 | 63.6 | 89.8 | 60.4 | 97.6 | 66.1 |
| DrivoR（HF） | 54.6 | 99.1 | 98.2 | 99.3 | 75.4 | 98.7 | 94.9 | 97.6 | 70.2 | 92.3 | 91.6 | 97.3 | 75.7 | 90.6 | 56.1 | 98.4 | 44.7 |
| DriveZero（HF） | 56.8 | 98.7 | 99.3 | 100.0 | 76.7 | 98.4 | 95.3 | 96.9 | 44.4 | 94.4 | 95.4 | 97.7 | 76.1 | 92.5 | 60.0 | 95.7 | 44.0 |
| EABOT（HF 第 1） | 58.6 | 99.1 | 98.0 | 99.6 | 74.0 | 99.1 | 91.6 | 97.6 | 74.2 | 94.0 | 94.7 | 97.4 | 74.3 | 92.2 | 58.0 | 96.3 | 57.5 |
| LEAD-LTFv6（HF） | 31.9 | 96.6 | 86.7 | 99.2 | 84.5 | 95.1 | 94.4 | 97.8 | 76.4 | 79.9 | 75.6 | 86.3 | 89.6 | 76.1 | 50.0 | 95.2 | 66.7 |
| LTF（HF） | 25.1 | 96.2 | 79.6 | 99.1 | 84.1 | 95.1 | 94.2 | 97.6 | 79.1 | 77.8 | 70.2 | 84.3 | 85.1 | 75.7 | 45.4 | 95.8 | 76.0 |
| CV（我们 = HF） | 11.5 | 88.9 | 42.9 | 70.7 | 77.5 | 87.3 | 78.7 | 97.1 | 60.4 | 83.2 | 59.1 | 76.5 | 71.4 | 81.1 | 48.0 | 97.2 | 62.0 |

TLC 各行都在 96 以上（Cinque 100 / 99.2），不列。读法：

- **DAC 是唯一同时在两个 stage 都远低于所有 mid-tier 以上参照的项**：Cinque 89.3 / 73.6，ZTRS 97.6 / 90.4，DriveZero 99.3 / 95.4。LTF、LEAD-LTFv6 的 DAC 比 Cinque 还低（79.6 / 86.7），它们靠 EP 84–90 与 EC 76–79 拿分，NC / TTC 反而是短板（stage 2 NC 只有 78–80）。
- NC、TTC 在 stage 1 与 ZTRS 相当（98.0 对 98.9，98.2 对 98.9）；stage 2 差 0.2 与 2.8 分，绝对值不大。**reactive 场景里的碰撞并不是 Cinque 的缺口**：stage 2 的 NC 90.9 与 ZTRS 91.1 一样，DriveZero 也只有 94.4。
- EP：Cinque（75.0 / 66.1）比 ZTRS（66.7 / 63.6）高；与 DrivoR、DriveZero 相比，stage 1 同档（75–77），stage 2 低约 10（66 对 76）。ZTRS 是靠保守拿到 48 的。速度不是它与 ZTRS 的差距，对更高的 DrivoR 以上才有 1.5 到 2 分。
- EC 各家分散很大（37.8 到 79.1），不构成榜上排名的规律：DriveZero 只有 44 却拿 56.8。Cinque 的 EC 低是我们输入管线的特性，见 4.2。
- 两个 lc 臂在 stage 1 略涨、stage 2 略跌，combined −0.28：对 225 组做 bootstrap 的 lc − none 配对差 **−0.15 [−1.17, +0.91]**（[group_bootstrap_ci.csv](results/navhard-deficit/group_bootstrap_ci.csv)），是噪声。

### 1.2 与 navtest（PDMS 84.2）同口径比较

navtest 上我们只有 v1 打分（GIMM 输入的 Cinque 没有跑 v2 navtest），PDMS = NC · DAC · (5·EP + 5·TTC + 2·C) / 12，5 项。把每一项单独置 1（「这一项完美时能多拿几分」）：

| 项 | navtest 均值 | navtest 该项完美 +分 | navhard Cinque 均值（stage 1 / 2） | navhard 该项完美 +分（combined） | ZTRS 该项完美 +分（combined） |
|:--|--:|--:|--:|--:|--:|
| NC | 98.3 | +0.3 | 98.0 / 90.9 | +3.3 | +3.9 |
| DAC | 95.6 | +2.3 | 89.3 / 73.6 | **+14.3** | +7.1 |
| DDC | 97.8（v1 不进分） | — | 98.3 / 90.2 | +3.3 | +2.9 |
| EP | 73.1 | **+8.7** | 75.0 / 66.1 | +9.9 | +13.6 |
| TTC | 95.4 | +1.1 | 98.2 / 87.0 | +0.4 | +0.4 |
| LK | 不在 v1 | — | 94.7 / 48.1 | +2.4 | +3.1 |
| HC / comfort | 99.96 | 0.0 | 97.8 / 97.4 | +0.1 | +0.4 |
| EC | 不在 v1 | — | 37.8 / 45.7 | +5.8 | +6.5 |
| 全部乘性项一起完美 | | +2.7 | | **+26.5** | +16.4 |
| 全部加权项一起完美 | | +9.9 | | +20.1 | +25.7 |
| 零分 token 占比 | 5.9% | | 12.9%（s1）/ 34.3%（s2） | | 3.3% / 23.5% |

表：navtest 数来自 [navtest_v1_terms.csv](results/navhard-deficit/navtest_v1_terms.csv)（12 146 token，逐 token 复算），navhard 数来自 [marginal_if_perfect.csv](results/navhard-deficit/marginal_if_perfect.csv)（均匀权重）与 [slices.csv](results/navhard-deficit/slices.csv)。

读法：**navtest 上 Cinque 的损失是速度**（EP 一项 8.7 分，乘性项合计只有 2.7），这与已有结论一致（N0 的纵向拉长有效）。**navhard 上损失换成了乘性失败**（乘性项合计 26.5，其中 DAC 单项 14.3）。navtest 与 navhard 的 DAC 差异：navhard stage 1（也是真实场景，只是被选成 hard）的 DAC 就已经从 95.6 掉到 89.3，stage 2 再掉到 73.6。同一个模型换到偏离后的起点，先天不足的是路面约束。

## 2. 相对 mid-tier 的点数差

### 2.1 ZTRS：逐 token 换项（精确，均匀权重）

把 ZTRS 的某项逐 token 换给 Cinque、重算 combined（基线 33.43，ZTRS 44.89，差 11.46）：

| 项 | 全部 | 只换 stage 1 | 只换 stage 2 |
|:--|--:|--:|--:|
| **DAC** | **+7.78** | +3.93 | +3.55 |
| DDC | +1.19 | +0.30 | +0.88 |
| EC | +1.13 | +0.42 | +0.68 |
| LK | +0.10 | −0.02 | +0.12 |
| NC | −0.12 | +0.18 | −0.32 |
| TTC | −0.35 | −0.07 | −0.28 |
| EP | 0.00 | −0.72 | +0.71 |
| HC / TLC | −0.05 / −0.06 | | |
| 乘性四项一起 | +9.70 | +4.69 | +4.70 |
| 加权五项一起 | +0.80 | −0.44 | +1.23 |
| 全部九项 | +11.46 | +4.14 | +6.99 |

来源 [substitution_vs_ztrs.csv](results/navhard-deficit/substitution_vs_ztrs.csv)。DAC 占差额的 68%，DAC + DDC + EC 占 88%，其余六项之和是 −0.5，在噪声内。单项和（+9.6）小于九项同换（+11.46），差 1.8 分，是项间的交互（同一 token 上 DAC、DDC 同时失败，换掉一项后另一项才显出来）。stage 1 只占 4.1、stage 2 占 7.0，与 stage 2 更差一致；两个 stage 之和 11.1，对整体 11.46 余下 0.3 是组内乘积的交互项。

95% CI（225 组 bootstrap）：ZTRS − Cinque = +11.46 [+9.8, +13.2]，所以这个差额和它的构成不是抽样噪声。

### 2.2 更高的参照：按 log 份额分摊（近似）

其他 HF 参照没有逐 token 文件，只能用聚合子分：把每项的 log 比按各 stage 分摊官方差额（[logshare_vs_references.csv](results/navhard-deficit/logshare_vs_references.csv)）。这个近似在 ZTRS 上给出 DAC 10.4、DDC 2.7、EC 1.6，与逐 token 精确值（7.8 / 1.2 / 1.1）同序但偏大，因为它把 ZTRS 的 +3.3 权重差额也摊进了各项。**下表是估计，只看排序与量级。**

| 参照（官方差额） | DAC | DDC | EC | EP | NC | TTC | LK | 其余 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| ZTRS（+14.8） | 10.4（s1 3.1 / s2 7.3） | 2.7 | 1.6 | −1.5 | 0.4 | 0.5 | 0.8 | −0.1 |
| DrivoR（+21.2） | 12.3 | 3.4 | 1.8 | 1.7 | 1.1 | 0.7 | 0.5 | 0 |
| DriveZero（+23.5） | 14.6 | 3.9 | 0.3 | 1.9 | 1.8 | 1.0 | 0.8 | −0.8 |
| EABOT 第 1（+25.3） | 14.1 | 3.6 | 3.0 | 1.3 | 1.8 | 1.0 | 0.5 | 0 |

不管参照是 48、55 还是 59，DAC 都占差额的 55–70%，DDC 占 14–18%。**DAC + DDC 都是「开出该走的路」类失败**。

### 2.3 stage 1 与 stage 2 的分工

combined 的乘积结构下，stage 1 的失败代价被放大：stage 1 分数为 0 的组占 12.9%（450 组里的 58 个），这些组的 stage 2 平均分本来是 0.53，被一起清掉，**合计损失 6.9 分 combined**（[stage_coupling.csv](results/navhard-deficit/stage_coupling.csv)）。同期 ZTRS 的 stage 1 零分只有 3.3%。stage 1 的零几乎全是 DAC（Cinque 10.7%，ZTRS 2.4%），NC 只占 2.0%。

## 3. 失败在哪里

数据是 Cinque `none` 的 5 912 个 token（stage 1 450，stage 2 5 462），每个 token 有逐项 0/1 或分数、命令（左 / 直 / 右）、起始速度、地图。navhard 的 scenario type 标签不在我们的 slim pkl 里，所以只切了命令、速度、地图与 plan 几何。全部切片在 [slices.csv](results/navhard-deficit/slices.csv)。

### 3.1 零分（乘性失败）集中在 stage 2 的运动帧，以 DAC 为主

| stage | 命令 | n | 零分比例 | 其中 DAC=0 | NC<1 | ZTRS 零分比例 |
|:--|:--|--:|--:|--:|--:|--:|
| 1 | 左 | 123 | 20.3% | 14.6% | 4.9% | 3.3% |
| 1 | 直行 | 228 | 8.3% | 7.0% | 1.3% | 4.8% |
| 1 | 右 | 99 | 14.1% | 14.1% | 0% | 0% |
| 2 | 左 | 935 | 32.6% | 25.7% | 10.4% | 21.0% |
| 2 | 直行 | 3 358 | 33.7% | 19.2% | 12.0% | 24.4% |
| 2 | 右 | 1 155 | 37.8% | 32.7% | 7.4% | 22.9% |

命令的作用：转弯帧 DAC 失败比直行高（stage 1 14% 对 7%，stage 2 26–33% 对 19%），但 stage 2 有 61% 的 token 是直行，**stage 2 的 DAC 失败里 51% 来自直行帧**（645 / 1 263）。所以这不是「转弯没有 route」一个原因能解释的问题；`lc` 臂只能碰转弯，也解释了它在 navhard 上无增益。

| stage | 起始速度（m/s） | n | 得分 | 零分 | DAC=0 | EP | EC | ZTRS 零分 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 2 | ≤ 1 | 1 021 | 0.675 | 4.9% | 3.8% | **0.30** | 0.85 | 7.9% |
| 2 | 1–3 | 1 168 | 0.496 | 31.7% | 23.8% | 0.62 | 0.60 | 15.8% |
| 2 | 3–6 | 1 647 | 0.383 | 43.8% | 32.5% | 0.75 | 0.36 | 25.0% |
| 2 | 6–9 | 987 | 0.378 | 45.8% | 26.8% | 0.79 | 0.14 | 36.5% |
| 2 | 9–12 | 495 | 0.396 | 45.9% | 22.8% | 0.84 | 0.12 | 40.6% |
| 1 | 3–6 | 141 | 0.669 | 19.1% | 14.9% | 0.77 | 0.48 | 1.4% |

- 静止起点（≤ 1 m/s，占 stage 2 的 19%）的零分只有 5%，但 EP 只有 0.30：plan 起步太慢（这一档 72% 的 plan 4 s 前进不到 2 m）。这是 EP 上的损失，不是失败。
- 运动起点（> 1 m/s）零分 32–46%；**它相对 ZTRS 的额外零分在 1–6 m/s 最大**（31.7 对 15.8，43.8 对 25.0），高速档（> 6 m/s）ZTRS 自己也有 36–41% 的零分，所以高速的 stage 2 是所有方法都难的段。
- EC 随速度崩塌：≤ 1 m/s 时 0.85，> 6 m/s 时 0.12–0.14。stage 1 也是这个形状（6–9 m/s 只有 0.17）。这是相邻帧 plan 不一致（见 4.2）的直接读数。
- 地图：stage 2 匹兹堡 Hazelwood（2 184 个 token，占 40%）零分最高（38.4%），DDC 失败 26%；拉斯维加斯的 DDC 只有 2.8% 失败但 NC 失败 17.5%。

### 3.2 DAC 失败与 plan 形状：弱相关，不能当规则用

（起始速度 > 1 m/s 的 token）plan 4 s 末端的 |横移| 与 |yaw|：

| stage | 末端 \|横移\| | n | DAC=0 | 零分 |
|:--|:--|--:|--:|--:|
| 1 | ≤ 0.5 m | 114 | 5.3% | 7.0% |
| 1 | > 5 m | 94 | 22.3% | 29.8% |
| 2 | ≤ 0.5 m | 1 356 | **15.9%** | 29.0% |
| 2 | 1–2 m | 776 | 30.3% | 45.2% |
| 2 | > 5 m | 638 | 48.7% | 60.7% |

DAC 失败随横移与 yaw 单调上升，但**基线本身就高**：stage 2 里 plan 几乎直行（|横移| ≤ 0.5 m）的 token 仍有 15.9% DAC 失败，这批基本是「起点被偏移放在路边或斜向，plan 沿当前朝向直行而没有回到车道」，也就是 navhard stage 2 设计要测的 recovery。plan 只是在自车系里画一条线，它不知道车道在哪。

### 3.3 速度与纵向短缺

stage 1（有人类未来的 450 个 token）上，plan 4 s 前进距离比人类短 3.9 m（均值），随速度增大：≤ 1 m/s −1.1 m，3–6 −3.9，6–9 −5.0，> 9 −5.6（[stage1_lon_shortfall.csv](results/navhard-deficit/stage1_lon_shortfall.csv)）。plan 的前进距离约为 v₀ · 4 s 的 0.78。但这个短缺换算成分数只是 EP（marginal +9.9 全部置 1、拉到 ×1.1 只有 +1.0），并且 EP 不是与 mid-tier 的差距（2.1 节 EP 项换过来 0.00）。

### 3.4 为什么 ZTRS 有 +3.3 的权重加成而 Cinque 没有

stage 2 组内权重是 stage 1 终点的高斯核。Cinque 的 stage 1 终点与人类终点的距离中位数是 4.4 m，只有 20% 的 token 在 1.9 m 以内（[stage1_endpoint_vs_human.csv](results/navhard-deficit/stage1_endpoint_vs_human.csv)），所以绝大多数组的权重是均匀的，官方 33.33 与均匀 33.43 只差 −0.1。ZTRS 的官方值比均匀高 3.3，说明它的终点大概率落在 stage 2 起点附近，被加权到「它自己走到的那个起点」那一格，这是 navhard 「pseudo closed-loop」设计对终点精度的额外奖励。（这一段是从代码与数据推断，没有重算 ZTRS 的权重，因为 slim pkl 里没有绝对位姿。）

## 4. 不训练 base model 的补救：估计

**以下全是估计，不是测量。** 做法：在现有逐 token 分数上做替换或封顶，都是同一批 token 上的**样本内**估计，没有新跑打分、没有验证集，也没有代价项（例如提速会增加碰撞，没有建模）。数值来自 [fix_estimates.csv](results/navhard-deficit/fix_estimates.csv)。

### 4.1 DAC：几何规则无效，需要学习式或地图式信号

| 规则（样本内） | 触发比例 | Δ combined |
|:--|--:|--:|
| \|横移₄ₛ\| > 5 m 时回退到 CV（匀速直行） | 12.4% | −4.4 |
| \|横移₄ₛ\| > 5 m 或 \|yaw₄ₛ\| > 0.6 rad 时回退到 CV | 15.9% | −5.2 |
| 同上，回退到 Hydra 打分头选出的轨迹 | 15.9% | −0.8 |
| 同上，回退到 `cls_late` 头轨迹 | 15.9% | −1.5 |
| \|横移\| > 2 m 或 \|yaw\| > 0.4 rad 回退到 Hydra | 32.1% | −3.4 |

所有阈值与回退目标都变差（−0.2 到 −12.1，[表全部](results/navhard-deficit/fix_estimates.csv)）。大横移的 plan 在 navhard 里多数是对的（转弯、绕行、stage 2 的偏离恢复），一刀切回退损失比 DAC 收益大。

真正能针对 DAC 的是**打分头 / 切换器**（第 64 条的 N0，进行中的 N1）。navhard 上现有的候选（Cinque GIMM、`lc`、E6 Hydra 头、`cls_late` 头、CV）逐 token 取最好的（oracle，上界）：

| 候选池 | oracle combined | 相对 Cinque |
|:--|--:|--:|
| Cinque + `lc` | 36.0 | +2.6 |
| Cinque + Hydra 头（hold 输入） | 44.3 | **+10.9** |
| Cinque + `cls_late` 头 | 42.5 | +9.1 |
| 上面四个 | 48.5 | +15.1 |
| 加 CV | 50.9 | +17.5 |

N0 在 navtest 上拿到了 oracle 空间的大约 11%（+0.77 对约 7 分，第 64 条）。**如果 navhard 上兑现同样比例，Cinque + Hydra 池是 +1.2 分；乐观到 1/3 是 +3.6。** 估计区间 +1 到 +4，最大不确定性是打分头在 navhard 的合成场景上是否失效（它在 navtrain 上训，stage 2 是分布外）。这项需要 navhard 特征抽取（GPU），本次不做。

### 4.2 EC：帧间一致性，估计 +1 到 +3

Cinque 的 EC 是 37.8 / 45.7，在 v > 6 m/s 时 12–14%。输入管线是原因之一：用 ego-motion warp 补帧的 Cinque EC 是 64.0 / 56.6（GIMM-VFI 补帧是 37.8 / 45.7），代价是 warp 臂整体只有 29.7 分。EC 只要求相邻 0.5 s 的两条 plan 的加速度 / jerk / yaw 特征的 RMS 差在阈值内，plan 里的高频抖动与帧间不一致都会破坏它，对 plan 做时间方向平滑或低阶参数化（不改 base model）原则上可以改。

- 把 EC 换成 ZTRS 的逐 token 值：+1.1；换成 warp 臂：+2.0；换成 Hydra 头：+2.0。
- 假设平滑后 EC 达到 0.75：+3.0（EC 全置 1 的上界是 +5.8）。

这一项**能否被平滑真正修好没有数据**：GIMM 补帧本身带随机性，两帧 plan 的差可能是输入而不是输出噪声。验证只要一次 CPU 打分，不需要 GPU（在已有 plan 上对时间轴低通，用 devkit 重打）。

### 4.3 标准起步（EP）：估计 +1 到 +2.5

navhard 有 18% 的 token 起始速度 < 1 m/s，Cinque 在这批上的 EP 只有 0.31（Hydra 头 0.74，ZTRS 0.56，CV 0.27）。把这批 token 的 EP 至少抬到 0.8（假设不引入 DAC / NC 失败）：**+2.5**；这批 token 的 DAC 失败只有 3.8%，风险小。N0 在 navtest 上起步帧涨 +5.9，方向一致。整体 ×1.1 的速度拉伸：+1.0，×1.2：+2.0，但这一项在 navhard 里不是与 mid-tier 的差距，而且提速在直行帧上有代价（N0：navtest 直行 −0.8）。

### 4.4 合计与限度

| 项 | 估计 Δ | 依据 | 需要什么 |
|:--|--:|:--|:--|
| 打分头 / 切换器（针对 DAC） | +1 到 +4 | oracle 的 11%–33%（N0 兑现比例） | navhard 抽特征（GPU） |
| EC 平滑 | +1 到 +3 | 换项 +1.1 到 +2.0，假设 0.75 为 +3.0 | CPU 重打分 |
| 标准起步 | +1 到 +2.5 | EP ≥ 0.8 换项 +2.5 | 规则；导致的 DAC / NC 风险没建模 |
| 几何回退规则 | 负 | 样本内 −0.2 到 −12 | 不做 |
| 合计 | **约 +3 到 +9** | 简单相加，项间有重叠 | |

也就是 33.3 → 36 到 42。**到不了 mid-tier 的 48–50**：ZTRS 与 Cinque 的差额里 DAC 占 +7.8，而现有不改 base 的手段里，只有打分头能碰 DAC 且预期 ≤ 4。要过 48，DAC 必须由模型本身变好（即训练），或由一个真的能读出「哪里能开」的模块给出。

## 5. 对 op-adapt r2 的含义

读的是 [op-adapt r2 预登记](../todos/2026-09-29-op-adapt-r2-prereg.md) 第 2.1 节的损失（L_score 的 winner-take-all，L_dir 的配对方向约束，规则打分器 S_jev = NC · DAC · (5P + 5TTC + 2C) / 12）。

**对得上的部分**：S_jev 有 DAC 项（可行驶区判据与 NAVSIM 的 DAC 是同一类：所有行车道与路口，含对向车道），所以 DAC 信号在打分器里存在。

**对不上的部分（按重要性）**

1. **训练的 slot 不是 DAC 失败发生的地方。** L_score 只用在 sim 的 x⁺ slot（有行人要反应的）和 nuScenes 的 VRU 帧；正常帧和 x⁻ 帧交给 L_distill，明文写着「避免打分器在普通驾驶上与蒸馏拉扯」。navhard 的 DAC 失败 51%（stage 2）以上来自直行帧、且与行人无关。r2 的设计把普通驾驶帧钉在原模型上，正好保护了这个缺口。
2. **S_jev 的三个主项对应的是 Cinque 没有缺口的项。** NC、TTC、P 三项在 navhard 上与 ZTRS 逐项换值的收益是 −0.1、−0.35、0.0。reactive 场景里碰撞不是问题（stage 2 NC 90.9 对 ZTRS 91.1）。r2 想教的「见到行人减速」是一个真实的能力缺口（我们在别处测过），但它**不是 navhard 得分的缺口**，所以 r2 做成了也不该指望 navhard 涨分。
3. **缺 DDC、LK、EC。** S_jev 明文允许借对向车道（不扣 DAC）而没有 DDC 项；DDC 在 navhard 上差 +1.2（stage 2 上 90.2 对 95.8，逆行失败 12%）。LK（stage 2 只有 48.1 对 60.4）与 EC（37.8 / 45.7）也没有对应项。
4. **缺 stage 2 那类场景：偏离后的恢复。** stage 2 的 DAC 失败在「plan 直行、起点偏斜」的 token 上也有 16%（3.2 节）。r2 的训练分布是 CARLA / Cosmos 的正常起点（expert 轨迹上取的帧）加 nuScenes，没有「起点偏离车道中心 ±2 m / 朝向偏斜」的样本。WL-2 的候选里有 `shift_L/R`，但它们是**候选轨迹**，不是起点扰动。
5. **读数里没有 navhard，也没有 v2 项。** N-nav 只是 navtest PDMS ≥ O − 1（PDMS 没有 DDC、LK、EC，也没有两阶段）。所以就算 r2 改了 DAC 行为，登记的读数也看不出来。

**建议（按成本排）**

- **r2 的读数补一行 navhard EPDMS**（描述，不设门）：stage 1 / stage 2 的 DAC 失败率、DDC、EC，配对 Δ 对原模型。管线已在（navhard 的 GIMM 缓存、plan 与打分链路都有，[op-lb 状态](../tmp/2026-09-29-op-lb-state.md)），只是 GPU 侧抽特征一次，打分 CPU 约 15 分钟。这样 r2 是否顺带动了 DAC 有个真实读数，而不是靠推测。
- **S_jev 增加 DDC（对向车道逆行超过阈值记 0 或 0.5）**，代码上是 DAC 判据里加一行。
- **给 L_score 加一类「偏离起点」slot**：在 Cosmos / CARLA-rewind 的分支里横向 ±2 m、朝向 ±0.3 rad 的起点扰动，候选用 `op` 与 `shift_*`，S_jev 的 DAC 决定 Top 集。这是 navhard stage 2 恢复场景的直接对应，也是 r2 与 navhard 之间缺的那一环。是否值得纳入 r2 取决于用户是否要 r2 也管 navhard；如果 r2 只服务行人反应，这一条属于另一个 arm。
- 如果要在 navhard 上追 48，需要的是一条独立的路：**给 DAC 的读出**（N1 的打分头带 DAC 子头，且在 stage 2 样式的偏离起点上训练或校准），与 r2 的行人行为是两件事。

## 6. 局限

- 均匀 stage 2 权重复现 combined（Cinque 差 0.1，ZTRS 差 3.3），对 ZTRS 的比较偏低估 3.3，本文的点数差指均匀权重口径下的 11.46，不是官方 14.8。
- ZTRS 之外的参照只有聚合子分，2.2 节的份额分摊是近似（在 ZTRS 上已与精确值差约 2.6 分）。
- 4 节的所有 Δ 都是样本内估计，没有 held-out；打分头的估计沿用 N0 在 navtest 的兑现比例，navhard 上没有验证。
- navhard 的 scenario type 标签不在我们的索引里，所以「按场景类型切」只切了命令、速度、地图与 plan 几何。
- lc 臂与 none 的差在噪声内，导航输入在 navhard 的读数不作结论。
- 只做了 Cinque `none` 与 `lc@−1.0`，Lebowski 与 small 的 navhard 逐 token 没有拆（它们更低，第 37 条）。
