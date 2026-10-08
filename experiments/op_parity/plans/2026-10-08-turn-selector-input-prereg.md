# op_parity turn selector 输入预登记：转弯 token 上的「挑哪条」是缺边界几何，还是在 3 k 标签下本来就学不出来（2026-10-08，任何本线 selector 读数之前写定）

## 问题

第 186 条：navtest |dyaw| ≥ 20° 的 3 154 个 token 上，SH30 plan 附近 F19 的可信特权上限是 F19 × pc +9.55 [+8.14, +10.94]（L9 × epcap 下界 +9.20），
增益几乎全是 15% 的 token-seed 上少量大修（DAC +6.30）。按 log cross-fit 的 selector 在 ego + 指令 + plan 上是 −0.27 [−0.52, −0.06]，
加池化冻结视觉 + shipped Cinque 输出是 +0.53 [−0.28, +1.28]；「这个 token-seed 有修法」的 out-of-fold AUC 只有 0.69-0.70。第 179 条：模型自己的路沿输出
对 DAC 失败的 AUC 0.641，地图 SDF margin 0.921。本线回答：**为什么学不出来**——缺的是可行驶边界的几何，还是不管输入是什么这个选择在这个标签规模下都学不出来。
只用已有的逐 token 逐候选分数（`turn_ceiling/score_all.csv`），不新打 simulator 分，不训基座，不碰 navtrain。

## 已有的、不重测

- 第 186 条的上限、水分拆解、E 臂 / 池化视觉臂的 selector 读数、AUC 0.692（主臂）全部不重算；本线的对照行直接复用 `turn_dewater.py` 的同一套代码跑出的 E 臂（对账：F19 × pc 的 E 臂应复现 −0.27）。
- 第 179 条的 margin 定义与校准（`sc_analyze.py`：4 角点足迹、0.1 s 网格、navtrain 拟合的 (s, b)）不改。
- 已经看过、所以不算盲的数：第 186 条的全部表（含 AUC 0.69、场景臂 +0.53 / +0.61 / +0.69 的苗头）、第 179 条的 0.641 / 0.921。**没看过的**：任何带逐候选 margin 的 selector 读数、任何未池化视觉 / SH30 隐状态的读数、逐候选 margin 本身的数值。

## 协议（与 turn_dewater 的 selector 相同，不改族和口径）

行 = (seed, token)，3 154 token × 2 seed，独立单位 = log（108 个）；外层 5 折按 log 分折（同一 log 的所有 token、两个 seed 同折），内层按 log 选超参；
所有拟合（标准化、PCA 白化的拟合集除外，见下）只用训练折；读数 = 被选候选在族口径下的分数 − 恒等，对 seed 取均值、对重复取均值，按 log 聚类配对 bootstrap
（`jevdrive.stats` / `TC.ci`，B = 10 000），> 20° 桶为主，回收率 = 增益 / 同族同口径上限（`TC.ratio_ci`）。
**族 × 口径：F19 × pc（主，可信族）与 L9 × epcap（下界括号）。** 两者都报；判定只用 F19 × pc，L9 × epcap 作一致性检查。
对照臂 = E（ego + plan，ridge L，复用 turn_dewater 的代码和超参网格）。重复数：ridge / 规则头 10 次，树头 5 次，神经头 5 次（GPU）；学习曲线用训练 log 的 25 / 50 / 75 / 100%，树头 2 次、神经头 3 次重复。

## 臂

**逐候选 margin 的定义**（每个 seed 的每条候选路径各算一次）：4 角点足迹（`sc_analyze.footprint`，0.1 s 网格，41 步）在 SDF 栅格或模型路沿上的最小符号距离，
并按时间累积取最小到 1 / 2 / 3 / 4 s，得 4 个数，截到 [−2, +4] m；无法取值（路沿读数缺失）填 +4。
- `M`（特权）：SDF = navtest 的地图 SDF 栅格（第 148 条的 label，`runs/op_probe/labels/navtest.npz`），即 hinge 所用的量。
- `C`（非特权）：SH30 自己（该 seed）的 road_edges 输出，经 navtrain 拟合的 (s, b) 校准（第 179 条，`calibration.json`，不在 navtest 上重拟合）。**只用路沿；车道线输出不做**（见「不做」）。

**特权臂（上限，不是方法；表里所有用 `M` 的行都标 PRIV）**——ego + plan 加 `M`：
| 臂 | 头 | 内容 |
|:--|:--|:--|
| P1 | L：多输出 ridge，各候选的增益 | E + 族内全部候选的 `M`（K × 4 维），λ 同 E 臂网格，内层 CV 选 |
| P2 | G：树（HistGB，超参同 186 的 G 头：200 iter、lr 0.05、15 叶、min leaf 50、l2 1，不调参） | 行 = (seed, token, 候选)，输入 = E（35 维）+ 候选 (偏移, 曲率, 速度) + 该候选的 4 个 margin + 恒等的 4 个 margin + 族内最大的 4 s margin |
| P3 | R：一参数规则，不学别的 | 只考虑速度 ≥ 1 的候选。恒等 4 s margin ≥ τ 则保持恒等；否则在 4 s margin ≥ τ 的候选里取偏离最小者（偏离 = \|偏移\| / 0.5 + \|ln 曲率\| / ln 1.15 + \|ln 速度\| / ln 1.2），都不满足则取 margin 最大者。τ ∈ {−0.5, −0.25, 0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0} m 由内层 CV 按实现增益选 |

**非特权臂**（测试时可得的输入；不用 logged future、dyaw / 机动标签、任何 score-poses 输出列）：
| 臂 | 头 | 内容 |
|:--|:--|:--|
| N1 | L | E + 族内全部候选的 `C`（K × 4 维） |
| N2 | G | 同 P2，`M` 换成 `C` |
| N3 | R | 同 P3，`M` 换成 `C`，τ 网格同 P3 |
| N4 | L | E + SH30 自己的隐状态（该 seed 的 policy `select_4` 与 `mean`，各 512 维，拼成 1 024 维）PCA 白化取 k ∈ {8, 32, 128}（内层选），PCA 在 navtest 非转弯 token（\|dyaw\| < 20°，不在评测集内）上拟合，不看分数 |
| N5 | NN | 未池化冻结视觉 token（`lb_navtest@warp/front.npy` 最后一帧 32 × 512，SH30 的输入）+ E，小注意力头 |
| N6 | NN | 隐状态（1 024 维，训练折标准化）+ E，小 MLP 头 |
| N7 | NN | N5 的全部输入 + 隐状态 + 族内全部候选的 `C`（全部非特权信息合一） |

**神经头**（N5-N7，GPU 作业经 pool 提交）：token 流 LayerNorm → Linear 512→64 → 4 个可学习 query 的 softmax 注意力池化（256 维）；
E / 隐状态 / `C` 流 标准化 → Linear→64；拼接 → Linear→128 → GELU → dropout 0.1 → Linear→(K−1) 各候选的增益（恒等固定 0），MSE（目标除以训练集增益的标准差），
AdamW lr 1e-3、weight decay 0.05、batch 256、至多 40 epoch；每个外层训练折里按 log 切出 20% 作内层验证，**按验证集的实现增益选 epoch**（取最佳 epoch 的权重），
再在外层测试折上读数。参数量 < 0.5 M（N5 约 0.15 M）。不做任何别的超参搜索。

**多重性**：非特权臂 7 个（N1-N7），特权臂 3 个（P1-P3）。非特权臂的「显著为正」用 Bonferroni 校正的置信区间（m = 7，水平 1 − 0.05 / 7 ≈ 99.29%）；
特权臂用 m = 3（98.33%）。没有单独指定的主臂：判定以校正后区间为准；未校正 95% CI 单列，只作「提示」不作判定。
族和口径不再另外校正（判定只看 F19 × pc；L9 × epcap 是一致性检查）。

## 判定（现在写定）

记 R = F19 × pc 上的回收率点估计（增益 / +9.55），区间为上述校正区间。

- **特权「回收多」**：至少一个 P 臂校正区间下界 > 0 且 R ≥ 40%。**特权「回收少」**：所有 P 臂 R < 20%（或校正下界 ≤ 0 且 R < 40%）。其余 = 「部分」(20% ≤ R < 40%，或 R ≥ 40% 但下界 ≤ 0)。
- **非特权「为正」**：至少一个 N 臂在 F19 × pc 上校正区间下界 > 0，**且** 该臂在 L9 × epcap 上点估计 > 0。未校正下界 > 0 而校正后 ≤ 0 = 「提示，不命名」。

按此给三种结论之一（先判 iii）：
- **(iii)** 有非特权臂为正 → 命名该臂（并列时取点估计最高者，其余并报），作为 navtrain 规模阶段的输入；给出它的学习曲线。
- **(i)** 否则，特权「回收多」且没有非特权臂为正 → 瓶颈是边界感知；并列出最接近的非特权臂与特权臂的差。若特权只是「部分」：写成「(i) 偏弱」，不下强结论。
- **(ii)** 否则，特权「回收少」→ 即使地图 margin 精确已知也回收很少，这个选择在 3 k token 下不是靠边界几何能学的；剩下的唯一检验是 navtrain 规模（28 323 token、约 19 core-h 打分）。
- 以上都不满足 = 「部分，无命名」，照实描述。

**学习曲线**：所有臂在 F19 × pc 上做（见协议）。「到 100% 仍在上升」= 75% → 100% 点估计升 ≥ 0.3 EPDMS。曲线外推到 navtrain 规模一律标「推测」。
**AUC**：「这个 token-seed 在 F19 × pc 里有修法」（增益 > 0）的 out-of-fold AUC（5 折按 log，打分 = 头对非恒等候选预测增益的最大值；规则头用 −（恒等 4 s margin），另给 −（恒等 `M`）和 −（恒等 `C`）的单量 AUC），
按 log 聚类 bootstrap（B 1 000）报 CI，对照 E 臂 0.692；不设线。

## 作业前的闸门（不是 selector 读数，不满足则停下报告，不往下跑）

- G1：SH30 的恒等 `M`（4 s）对 DAC 失败（`DAC < 1`，分数 CSV 的恒等候选）的 AUC ≥ 0.85（第 179 条 0.921，那里 P2H10 + SH30 合并）。
- G2：SH30 的恒等 `C`（4 s）对同一标签的 AUC 在 [0.58, 0.72]（第 179 条 0.641）。
- G3：重新跑 SH30 抽隐状态时，plan 与存档 plan 的最大差 < 0.05 m（`sc_infer.py` 的 `--ref` 检查）。
- G4：E 臂 F19 × pc 复现 186 的 −0.27 附近（\|差\| < 0.1，同种子同代码；不满足 = 代码引入了偏差，停）。

## 不做

新 simulator 打分（包括更长视野、重打 margin 选中的候选）、任何基座训练、navtrain、reactive 口径、车道线输出的逐候选评估（路沿是 DAC 的直接对应，车道线要先定义一个对应量，不在本线里发明）、
HGB / NN 的超参搜索、闭环。若发现需要新 simulator 分，停下报告原因。

## 执行

`scripts/turn_selinput.py`（`feats` / `select` / `nn` / `report`，各在 `jevdrive.run.Run` 里，token 成员取 `jevdrive.data.splits` 的 `navsim/navtest`，CI 走 `jevdrive.stats` / `TC.ci`，
CPU 并行走 `jevdrive.par.pmap`，核数 ≤ `n_cpus() // 2` 且每 worker 线程 = 1，树头在主进程用有上限的 OpenMP 池）、`scripts/tsi_extract.py`（GPU：SH30 两个 seed 的 policy 隐状态，经 pool）、
`scripts/turn_selinput_chain.sh`（tmux `jev:turn-selinput`，`$DATA_DIR/runs/op_parity/turn_selinput/chain/{STATUS, DONE, ERROR}`）。
GPU 预算 ≤ 1 卡时，实际估计 < 0.4。结果 `results/turn_selector_input.md`，图 `figs/turn_selector_input/`。
