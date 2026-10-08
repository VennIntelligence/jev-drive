# op_parity turn-ceiling 去水分预登记：+12.20 里有多少是口径造成的，逐 token 的选择能不能学（2026-10-08，任何本线新读数之前写定）

## 问题

第 178 条：navtest |dyaw| ≥ 20° 的 3 154 个 token 上，SH30 plan 附近 F19 的 best-of-K 特权上限 +12.20 [+10.57, +13.76]（线 +4.0）。
同一条自己点了两个水分来源：(1) 分数只看 4 s，放慢的候选把出界 / 碰撞推到视野外（恒等 + 速度 × 0.6 一条就 +6.50）；
(2) EP 连续，oracle 在安全处一律加速（Shapley 里 EP +2.24）。另外它不说明这个选择可学。本线回答两件事，只用已有的逐 token 逐候选分数
（`$DATA_DIR/runs/op_parity/turn_ceiling/score_all.csv`，33 候选 × 2 seed × 3 154 token，8 个子分），不新打 simulator 分，不训基座，不碰 navtrain：

1. 去掉这两个来源之后上限还剩多少。
2. 一个不用特权输入的 selector，按 log cross-fit，out-of-fold 能拿回上限的多少（对照第 173 条 WOD 上 14%）。

## 已有的、不重测

- 第 178 条的全部读数（F19 / F27 / F33、单轴、小 K、固定变换、Shapley）不重算，只在本线的表里并排复现 F19 raw = +12.20 作为对账。
- 第 173 条的 selector 方法（多输出 ridge 回归各候选的增益、argmax、按 sequence 分折、内层 CV 选超参、置换对照、学习曲线）直接沿用，
  换成 navtest 的 log 分折和 no-EC EPDMS。它的结论（WOD 479 帧上视觉流不加分）不预设在 navtest 上成立。
- 已经看过、所以不算盲的数：V3 +7.72、O3 +5.95、K3 +7.79、F7 +11.02、固定变换全部不为正、oracle 在 F27 上 62.7% 选速度 > 1、2.5% 选速度 < 1、
  F19 的 EP Shapley +2.24、DAC +6.47。下面的族和线是在知道这些数的前提下定的；没看过的是任何去掉速度 < 1 / 固定 EP / 只横向的上限，以及任何 selector 读数。

## 一、去水分上限

对象、seed 处理、分桶、bootstrap 与第 178 条完全相同：3 154 token，两个 seed 各自在族内取最高分（并列取编号最小）后取 seed 均值，
桶 > 20° / 20–45° / > 45° / 左右，`jevdrive.stats.paired` 按 log 聚类配对 bootstrap（B 10 000），no-EC EPDMS × 100。

**分数口径**（逐 seed、候选、token，从 8 个子分用同一公式重算；恒等候选在所有口径下分数相同）：

| 口径 | 定义 | 去掉什么 |
|:--|:--|:--|
| `raw` | score-poses 原样 | 无（第 178 条的口径） |
| `epfix` | EP := 恒等候选的 EP | EP 的任何变化，包括放慢的 EP 损失 |
| `epcap` | EP := min(候选 EP, 恒等 EP) | 只去掉 EP 增益，放慢的 EP 损失照扣 |
| `pc`（path-consistent） | `epcap`，并且速度 < 1 的候选的地图类子分（DAC / DDC / TLC / LK）:= min(自己的, 同一 (偏移, 曲率) 在速度 = 1 时的) | EP 增益 + 放慢对地图类子分的任何修复 |

`pc` 的依据：速度缩放不改路径，速度 < 1 的候选走的是同一路径的前缀（弦线插值误差 ≤ 6 cm）。它在 4 s 内不出界而全速候选出界，只说明还没走到，
不是修好了；所以放慢不允许修 DAC / DDC / TLC / LK，只允许修 NC / TTC（对动态 agent 让行是真实的修法，也可能只是把撞静止物推后，这一点分不开，记为限定）。
取 min 是保守的：simulator 跟踪慢轨迹更准带来的真实地图类改善也被去掉。F19 / F27 里每条速度 0.8 的候选都有速度 1 的对应候选；F33 的 c31（速度 0.6）对应恒等。

**族**（都含恒等，候选编号沿用 `turn_ceiling.py candidates`）：

| 族 | K | 内容 |
|:--|--:|:--|
| F19 / F27 / F33 | 19 / 27 / 33 | 第 178 条原族 |
| F19ns / F27ns / F33ns | 13 / 18 / 23 | 去掉所有速度 < 1 的候选（ns = no slow） |
| L9 | 9 | 速度恒等：内层 偏移 × 曲率（只横向） |
| L13 | 13 | L9 + 外层偏移 ±1.0 m、曲率 0.7 / 1.4 |
| O3 / K3 / V3、Vup（恒等 + 速度 1.2）、Vdn（恒等 + 速度 0.8） | 3 / 3 / 3 / 2 / 2 | 单轴，各口径下 |

读数（每格：> 20°、20–45°、> 45°、左、右）：

1. 用户点名的三行：**F19ns × raw**（去掉速度 < 1）、**L9 × raw**（速度恒等）、**F19 × epfix**（EP 固定）。
2. 交叉：{F19, F19ns, L9, F27, F27ns, L13, F33, F33ns} × {raw, epfix, epcap}，{F19, F27, F33} × pc，单轴族 × 各口径。
3. 每行并列「按该口径选、按 raw 计分」的增益（去水分的选法在原口径下值多少）。
4. 主要行（F19 raw、F19ns raw、L9 epcap、F19 pc）的 oracle 增益按 Shapley 拆到 DAC / NC / TTC / EP / LK / 其余，以及各臂失败率。
5. 尺度参照（不含任何候选）：两个 seed 的恒等 plan 里逐 token 取高分，max(s0, s1) − 均值。它量的是「换一条同分布的 plan」值多少，
   用来判断小余量失败被任何扰动救回的基线量级。

**事先指定的可信单值**：**P19 = F19 × pc 在 > 20° 桶上的增益**，作为「闭环可信」上限。理由（现在写定）：它去掉了两个已点名的来源里可以按构造判定为假的全部
（EP 增益；放慢对地图类子分的修复），保留放慢 / 加速对动态 agent 的修复，这部分在 non-reactive 回放下可能为真。
**L9 × epcap** 是下界括号（完全不许动速度，EP 不计增益）。两者相差 > 2.0 时结论里两个数一起报，并说明差值全部来自速度对 NC / TTC 的修复。
看到数之后不换可信单值；如果我事后认为另一个变体更诚实，写进「偏离」并同时保留 P19 的判定。

**线**：
- **P19 在 > 20° 桶上的点估计 < +4.0 → 去水分后小族里没有足够的余量，按分数监督的 stage 2 不值得做，这条线结束。**（沿用第 178 条的 +4.0，约合 navtest 全量 +1.0。）
- P19 ≥ +4.0 而 L9 × epcap < +4.0：余量依赖速度轴对动态 agent 的修复，照实报告，stage 2 若做必须带速度候选，且结论降一档。
- 两者都 ≥ +4.0：横向族自己就够，stage 2 的候选族可以不含放慢。

## 二、selector：这个选择能不能学

**样本**：行 = (seed, token)，3 154 token × 2 seed；独立单位是 log（108 个）。外层 5 折 × 10 次重复，**按 log 分折**（同一 log 的所有 token、两个 seed 同折），
内层按 log 4 折选超参。所有拟合（标准化、ridge、树、超参、margin）只用训练折。视觉 / teacher 流的 PCA 白化在 navtest 的**非转弯** token（|dyaw| < 20°，约 9 k，不在评测集里）上拟合，不看任何分数。

**监督与读数**：目标 = 族内各候选在该族口径下的分数 − 恒等。selector 每行选一条候选；读数 = 被选候选在该口径下的分数 − 恒等，对 seed 取均值、对重复取均值，
按 log 聚类配对 bootstrap。回收率 = out-of-fold 增益 / 同族同口径上限（比值的 log bootstrap）。同时列被选候选按 raw 计的增益。

**输入流**（都是测试时可得的；不用 logged future、不用 dyaw / maneuver 标签、不用 score-poses 的任何输出列包括 `raw_depth` / `lqr_out`）：

| 流 | 维 | 内容 |
|:--|--:|:--|
| `ego` | 20 | `lb_navtest/tab.npz` 的 `ego`：指令 one-hot、4 帧位姿历史、速度、加速度（第 173 条的同一组） |
| `plan` | 15 | SH30 该 seed 自己的 plan：1 / 2 / 3 / 4 s 弧长，2 / 4 s 的横向位置与航向，首段与末段速度，最大 \|曲率\| 及其符号值，最大横向加速度，\|y(4 s)\|，\|航向(4 s)\| |
| `dis` | 4 | 两个 seed 的 plan 之间的分歧：终点距离、4 s 弧长差、横向差、航向差（跑两个 seed 即可得） |
| `T` | k | shipped Cinque 在同一行的蒸馏输出（`teacher.npz` 的 `out`，1 086 维，含其车道线 / 路沿 / 前车预测），PCA 白化取 k |
| `V` | k | 冻结的 openpilot 视觉 token（`lb_navtest@warp/front.npy`，SH30 的输入，取最后一帧 32 个 token 的均值，512 维），PCA 白化取 k |

**臂**：`E0` = ego；**`E` = ego + plan（主臂，与第 173 条的 floor 同构）**；`E+D` = ego + plan + dis；`T+E`；`V+E`；`V+T+E`。k ∈ {8, 32, 128} 由内层 CV 选。
视觉流用已缓存的冻结特征，不做新的特征提取，不用 GPU。

**head**：
- `L`（主）：多输出 ridge 回归各候选的增益，恒等的预测固定为 0，argmax；λ ∈ {3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e6}，内层 CV 按实现增益选（并列取大 λ）。
- `G`（次）：一个梯度提升树（sklearn `HistGradientBoostingRegressor`，max_iter 200、learning_rate 0.05、max_leaf_nodes 15、min_samples_leaf 50、l2 1.0，不调参），
  行 = (seed, token, 候选)，输入 = 该行特征 + 候选的 (偏移, 曲率, 速度)，回归增益，逐行 argmax。只跑 `E` 与 `V+E`（PCA k 固定 32）。
- `L+m`（次）：`L` 加 margin，预测增益 ≤ m 时保持恒等，m 由内层 CV 选。
- 对照：折内选出的最优固定候选（`const`）；全部输入按 token 置换的零分布（主臂，100 次）；in-sample 拟合；学习曲线（训练 log 的 25 / 50 / 75 / 100%）。

**族 × 口径**：F19 × raw（与 +12.20 对账）、F19ns × raw、L9 × epcap、**F19 × pc（可信族）**。

**判定**（主检验两个：`L` + `E` 在 F19 × raw 与 F19 × pc 的 > 20° 增益；其余都是探索性读数，报 95% CI 并注明）：
- **可学**：增益的 95% CI 下界 > 0，置换 p < 0.05，且回收率点估计 ≥ 14%（第 173 条 WOD 的回收率）。
- **弱可学**：CI 下界 > 0 但回收率 < 14%。
- **在这个标签规模和这些输入下不可学**：CI 含 0。
- 视觉是否加分：`V+E − E`、`T+E − E` 的配对差，CI 下界 > 0 才算加。

**对 stage 2 的读法**（现在写定，结论里按此给出，不是开关）：
- P19 < +4.0：不做。
- P19 ≥ +4.0 且 F19 × pc 上「可学」或「弱可学」且学习曲线在 100% 处仍上升：值得做，候选族用去水分的那一个。
- P19 ≥ +4.0 但所有臂 CI 含 0：不直接上全量（v2 navtrain cache + 19 core-h 打分），先在已有 navtest 标签上找到任何能 out-of-fold 为正的输入再说。
navtest 上 3 154 个带标签 token 是 navtrain 28 323 个的 11%，学习曲线的斜率是唯一能外推的量，外推照实标为推测。

## 不做

新 simulator 打分（包括把视野延长到 4 s 以外去直接量放慢的推后效应）、任何基座训练、navtrain、reactive 口径、逐 token BEV 复核、GPU 作业、新的视觉特征提取。

## 执行

`scripts/turn_dewater.py`（`ceiling` / `select` / `report`，各自在 `jevdrive.run.Run` 里，token 成员取自 `jevdrive.data.splits` 的 `navsim/navtest`，
CI 走 `jevdrive.stats`，并行走 `jevdrive.par.pmap`），`scripts/turn_dewater_chain.sh`（tmux `jev:turn-dewater`，
`$DATA_DIR/runs/op_parity/turn_dewater/chain/{STATUS, DONE, ERROR}`）。纯 CPU，估计 ≤ 15 min 墙钟；box 上另有 navtrain metric cache 在建，
核数取 `jevdrive.common.n_cpus()` 的一半以内，不碰它的文件和进程。结果 `results/turn_ceiling_dewater.md`，图 `figs/turn_ceiling_dewater/`。
