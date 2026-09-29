# NAVSIM 提分（N1 之后）：N1b 与后续各臂的预登记

状态: running（2026-09-30 开；本节「共同规则」与「N1b」写于任何 N1b 拟合与分数之前，后续各臂各自的登记块写于该臂的 navtest 分数之前，块首注明写入时间）
主题: [../research/leaderboard-skill-pack.md](../research/leaderboard-skill-pack.md) 第 6.1 节（N1）；N1 预登记 [2026-09-29-n1-scorer.md](2026-09-29-n1-scorer.md)，decisions 第 64、68 条；navhard 缺口 [../research/navhard-deficit-breakdown.md](../research/navhard-deficit-breakdown.md)
协调: op-adapt r2（[2026-09-29-op-adapt-r2-prereg.md](2026-09-29-op-adapt-r2-prereg.md)，改的是 openpilot 权重；本 lane 不改权重，只在冻结的 Cinque 之上做后端，不与 r2 争数据或读数）；
导航 lane 的 GIMM 缓存（`runs/op_lb/lb_{navtest,navhard,n1train}`）只读；Cosmos G4（GPU 0–5）不碰；GPU 6 与 WL-2 共卡，本 lane ≤ 25 GB。

## 问题

N1（navtest 87.32）是 1 024 个 anchor + 4 个原生槽上的五个线性子分头。全量拟合时十个头的 λ 全落在网格下沿 1e-5，held-out BCE 随 λ 单调下降，头可能正则过度。
这一轮问两件事：(1) 只把 λ 网格往下扩，按 navtrain held-out 选，N1 还能涨多少（N1b，用户已批准）；(2) 在同一个冻结 Cinque 上，哪些后端改动按「每 GPU·h 的期望增益」最值得做，便宜的就跑。

## 共同规则（写死，适用于本文件所有臂）

1. **选择只在 navtrain 上做**：N1 的 19 968 个 token（E6 的 navtrain 子分集，`lb_n1train` 固定顺序），按 log 切 5 份（`H._group_folds(logs, 5, seed=1)`），fold 0 是 held-out logs（4 037 个 token），与 N1 同一切法。
   超参数、权重、候选集、结构的比较都在 held-out 或 5 折 OOF（out-of-fold）上做。**navtest 不参与任何选择**，每个登记的配置只打一次分；navhard 同样每个配置最多一次，只作描述。
2. **每次 navtest 打分都记进下面的「navtest 打分记录」表**，包括不成立的臂。报告最好的一个时，同时报 Bonferroni 校正后的 CI（1 − 0.05 / m，m = 本 lane 在该对照上的 navtest 次数），并写明「最好者是在 m 次看过之后挑出来的」。
3. 配对 Δ 一律是 navtest 12 146 个 token 上逐 token 的 PDMS 差，10 000 次 token bootstrap（`np.random.default_rng(1)`），与 N0 / N1 的报告同一个函数。
4. 每个臂标注它是**能力（capability）**还是**对准 metric 的配方（trick）**：学的标签是 PDM scorer 的子分、只改候选或选择规则的，都记为 trick；改变基座看到的信息或基座自己产生的候选的，按具体情况标，写理由。
5. 不删任何数据，不提交任何榜单。

## N1b：扩大 λ 网格重拟合（写于任何 N1b 拟合之前）

- **数据、特征、候选、标签、held-out 切法、选择规则的权重网格（7 000 格）、并列取网格里靠前者**：全部与 N1 相同（`jevdrive/skill_pack_n1.py` 的 A0）。
- **唯一的改动**：λ 网格从 {1e-5, 1e-4, 1e-3, 1e-2} 扩成 {0, 1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2}，每个子分头按 held-out BCE 各自取最小（与 N1 同一判据）。
  L-BFGS 迭代数保持 N1 的 100。若选中 λ = 0（新网格下沿），记录，不再扩：此时起正则作用的是迭代数，迭代数不在 N1b 里动。
- 选完 λ 与权重后，在全部 19 968 行上重拟合，对 navtest 输出一个 pose 文件，**只打一次分**（v1.1 PDMS，`v1_navtest_sp_n1b_navtest`）。
- **navhard（描述）**：Cinque 在 `lb_navhard` 的 GIMM 缓存上抽 `temporal`（与 N1 同一抽取脚本），N1（按原网格重拟合复现）与 N1b 各在 navhard two-stage 上打一次 EPDMS。
  复现检查写死：原网格重拟合的 N1 在 navtest 上的选中候选与存档 `navtest_n1.npz` 的 pose 逐 token 相同（最大差 < 1e-4 m）的比例 ≥ 99%，否则先查原因，不打 navhard。
- **判读（写死）**：N1b − N1 的配对 Δ 95% CI 下界 > 0 → 「成立」；CI 跨零 → 「平，λ 触边不是 N1 的瓶颈」；上界 < 0 → 「差」。另报 N1b − N0、N1b − 原生、五个子分、按 command、held-out 上的 λ 曲线；navhard 报 N1、N1b 对 Cinque `none`（33.33）的 combined / stage 1 / stage 2 与 225 组 bootstrap 的配对差，只作描述。
- **预期（写于拟合之前）**：held-out +0.3 到 +1.5；navtest 对 N1 +0.0 到 +0.8，CI 下界 > 0 的把握约四成。navhard：N1 对 `none` +1 到 +4（缺口分解第 4.1 节的估计区间），N1b 与 N1 差在 ±1 内。
- **后续臂的基线**：不论 N1b 在 navtest 上的结果如何，后续臂都用 N1b 的 λ 选法（held-out 选 λ 是正确做法，不按 navtest 结果换基线），各自报对 N1 与对 N1b 的配对 Δ。

## 设计：held-out 上的估计（09-30 00:15，navtrain fold 0 held-out logs，4 037 token，navtest 未碰）

N1b 头在 held-out 上的损失：PDMS 89.52，oracle 99.83，差 10.3 分；其中 3.3 分来自选中的候选被 NC / DAC 判零（3.3% 的 token），7.1 分来自非零但不是最好（主要是 EP）。
按打分排序的 top-k 里的最好者：top-1 89.5、top-2 91.0、top-4 92.4、top-8 94.9，说明排序靠前的候选里就有更好的，两段重排有空间，但要有比第一段更好的信息。

| 改动（其余同 N1b） | held-out PDMS | Δ | 性质 |
|:--|--:|--:|:--|
| N1b 线性头（基线） | 89.52 | – | |
| 选择规则：sigmoid 乘积的期望 PDMS / 先门后 EP（τ 0.5–0.95） | 89.28 / 87.6–88.3 | −0.2 / −1.2 以下 | 不做 |
| fold bagging（4 个 3/4 子集头平均） | 89.54 | +0.03 | 不做 |
| L-BFGS 300 次迭代 | 89.75 | +0.23 | 小 |
| 加 E6 的 hold 输入特征（Cinque + Lebowski `temporal`，零 GPU） | 90.06 | +0.54 | pack（同一冻结基座的另一输入视图 + 第二个 openpilot） |
| 原生族 9 个额外槽（拉长到 1.40、横向缩放 0.8 / 1.2） | 90.99 | **+1.47** | trick（把原生 plan 开得更快，EP 对准）；×1.40 被选 26%，又触边 |
| 共享主干 MLP 头（2×1024，BCE，按 held-out BCE 选 epoch） | 91.00 | **+1.48** | trick（更好地蒸馏 PDM scorer） |
| 学习曲线：拟合行 1/8、1/4、1/2、全部 | 87.99 / 88.36 / 88.88 / 89.52 | 每翻倍 +0.4 到 +0.6 | 数据量；3× 预计 +0.9 |

## 臂 S：navtrain 扩到 6 万 token（写于任何 S 的数据之前，09-30 00:30）

- **数据**：`navsim_raise tokens2`：navtrain 中 stage one、有未来、`v1_navtrain` metric cache 里有、不在 E6 的 2 万里、不在 N0 的选参集 T 里、**不来自 N1 held-out fold 的 log** 的 token，按种子 20260930 打乱，取前 M = 40 000 个。
  管线与 N1 完全相同：关键帧 → GIMM 补帧（GPU 5 整卡，5 个 worker）→ Cinque `temporal` 与原生 plan → 1 024 anchor 的逐 anchor 子分（`elicit_e6_score.py`，E6 同一 anchor 文件）+ 4 个原生槽的子分（`n1_score_native.py`），两者都在 `v1_navtrain` cache 上。devkit 打不出分的 token 丢掉并记数。
- **拟合**：N1b 的配方（线性头、N1b 的 λ 网格按 held-out BCE 选、7 000 格权重），训练行 = N1 的 19 968 + 新行，held-out 行与 N1 完全相同（新行不来自 held-out log）。navtest 一次（`sp_scale_navtest`）。
- **pilot 清单（192 token，写死，任一不过就停，不起全量）**：C1 两类标签覆盖 ≥ 95%；C2 原生 s = 1.00 平均 PDMS ∈ [75, 92]；C3 `temporal` 无 NaN / Inf，范数中位数对 `lb_n1train` 之比 ∈ [0.8, 1.25]；C4 逐 token 平均 anchor PDMS 与 E6 标签的均值差 ≤ 5 分；C5 GIMM 期间峰值显存 ≤ 80 GB。
- **判读**：S − N1b 配对 Δ 的 CI 下界 > 0 → 「数据量有用」；跨零 → 「平」。另报 held-out 上的 S 与学习曲线外推（+0.9）的对照。**预期**：held-out +0.6 到 +1.2，navtest 对 N1b +0.3 到 +0.9。
- 性质：数据量（同一个 PDM 蒸馏，trick 的规模化），不是能力。

## 臂 N2：更强的打分头 + 更多原生槽 + 额外输入视图（写于任何 N2 组合的拟合之前，09-30 00:30）

- **候选配置（写死，12 个）**：头 ∈ {线性（N1b 的 λ，L-BFGS 300 次），MLP（2×1024 GELU，dropout 0.1，AdamW lr 1e-3 wd 1e-4，60 epoch OneCycle，batch 512，epoch 按 held-out BCE 选，5 个 seed 的 logit 平均）} ×
  输入 ∈ {GIMM `temporal`（N1），+ E6 的 hold 输入 Cinque 与 Lebowski `temporal`} × 原生槽 ∈ {4（N1），13（+ FAM），22（+ FAM + FAM2：拉长到 1.75、横向 0.6–0.9）}。
- **选择**：12 个配置都在 fold 0 held-out 上跑完，取 held-out PDMS 最高者（并列取表中靠前 = 更简单者）；权重网格与 N1 相同。**另报 fold 1 作 held-out 的重复**（训练用其余 4 折），只报选中配置与 N1b 线性基线，描述、不参与选择。
- **navtest**：选中配置在全部 19 968 行上重拟合（MLP 每个 seed 用自己选中的 epoch，同一 60 epoch 调度），navtest 一次（`sp_n2_navtest`），navhard 一次（描述）。
- **判读**：N2 − N1 配对 Δ 的 CI 下界 > 0 → 「成立」；跨零 → 「平」；上界 < 0 → 「差」。同时报 N2 − N1b 与 Bonferroni 校正（m = 本 lane 对 N1 的 navtest 次数）。另报 EP 超过 human 的 token 比例（N1 23.0%）、各槽份额、按 command。
- **预期**：held-out +1.5 到 +2.5（两项 +1.5 的增益不会完全相加）；navtest 对 N1 +0.8 到 +1.8，CI 下界 > 0 的把握约七成。
- 性质：**trick**。MLP 是更好地拟合 PDM scorer；拉长到 1.4 倍以上的原生槽是在规则允许的范围内开得比 human 快；hold 视图是同一冻结基座（和第二个 openpilot）的另一种输入，属于 pack。

## 臂 A（openpilot 自己的替代 plan 进候选集，只作 held-out 开发）与臂 N3（N2 配置 + 6 万行）（写于 A 的标签与 S 的数据完成之前，09-30 00:55）

- **A（held-out，不单独打 navtest）**：Cinque 在同一 GIMM 输入上强制 desire 从 −1.0 s 起持有（laneChangeLeft / laneChangeRight / turnLeft / turnRight，不看 NAVSIM command），得 4 条替代 plan，各取拉长 1.00 / 1.15，共 8 个新槽，devkit 子分标签同原生槽。
  在 N2 的选中配置上（MLP、hold 视图、22 个原生槽）比较有无这 8 个槽的 fold 0 held-out PDMS。**纳入规则（写死）**：有 − 无 ≥ +0.2 → N3 的网格里加入 alt 维度；否则 N3 不带 alt。
  性质：候选来自冻结基座自己在别的导航意图下的输出，是「基座的多模态」而不是人造几何，但选择仍靠 PDM 蒸馏，整体仍记 trick。
- **N3 配置（写死）**：N2 的选中配置（MLP 5 seed、hold ×2 视图、22 个原生槽、同一权重网格），训练行 = 19 968 + 臂 S 的新行（新行的 FAM + FAM2 原生族标签在 `v1_navtrain` cache 上补打）。
  网格：hidden ∈ {1024, 2048} ×（若 A 过纳入规则）alt ∈ {无, 有}。fold 0 held-out 上取最高（并列取靠前 = 更简单），fold 1 重复作描述。全部行重拟合，navtest 一次（`sp_n3_navtest`），navhard 一次（描述）。
- **判读**：N3 − N2 配对 Δ 的 CI 下界 > 0 → 「成立」；跨零 → 「平」；上界 < 0 → 「差」。同时报 N3 − N1 与 Bonferroni 校正（m = 本 lane 当时的 navtest 次数）。
- **预期**：held-out 对 N2 +0.3 到 +1.0（线性头的学习曲线每翻倍 +0.5，MLP 可能更陡）；navtest 对 N2 +0.2 到 +0.8，CI 下界 > 0 的把握约一半。
- 性质：trick（数据量 × 更强的 PDM 蒸馏）。

## 臂 S 结果（09-30 07:45 navtest 打分）

**判读：成立**（「数据量有用」）。40 000 个新 token 全部有 devkit 标签（0 个丢弃），训练行 59 968。pilot 清单全过（覆盖 100%，原生 s = 1.00 PDMS 84.9，范数比 1.005，anchor 均值 21.8 对 E6 23.3，峰值显存 62 GB）。
navtest S = **87.74**，S − N1b = **+0.48 [+0.23, +0.74]**，S − N1 +0.42 [+0.15, +0.70]（Bonferroni m = 5 [+0.06, +0.77]）。held-out 90.11（N1b 89.52，+0.59），与线性学习曲线外推的 +0.9 同向略低，在登记预期（held-out +0.6 到 +1.2，navtest +0.3 到 +0.9）之内。
选中 λ 比 N1b 再小一档（1e-7 到 1e-6）；按 command：直行 90.08、左 81.44、右 78.84、起步 94.29，都比 N1b 高 0.3–0.7。EP 超过 human 的 token 23.9%。
成本：GPU 5 整卡 00:29–06:50 GIMM（5 worker，稳态约 1.8 token/s，峰值 62 GB）+ 抽特征约 50 min，约 7.3 GPU·h；CPU 20 进程的 anchor 标签 4 h（约 80 core·h）。

## 臂 A 结果（held-out，09-30 05:2x）

**纳入规则不过**：N2 选中配置上加 8 个替代槽，fold 0 held-out 92.64 → 92.56（−0.08 < +0.2），N3 不带 alt。
oracle 从 96.01（22 个原生槽）升到 96.94（+ 8 个替代槽），替代槽单独的 oracle 93.66：openpilot 在强制 laneChange / turn desire 下给出的 plan 里确有更好的候选，但当前打分头挑不出来。
成本：替代 plan 抽取 train 2 h + navtest 1.5 h + navhard 0.7 h（GPU 6 小卡，4 条 rollout / token，约 2.3 s / token / shard），标签 CPU 5 min。navtest 未打分。

## N2 结果（09-30 00:30 navtest 打分）

**判读：成立。** navtest N2 = **90.60**，N2 − N1 = **+3.28 [+2.89, +3.66]**（Bonferroni m = 4 [+2.79, +3.76]），N2 − N1b +3.34，N2 − 原生 +6.43 [+6.00, +6.86]。高于登记预期上沿（+0.8 到 +1.8）。

held-out（fold 0，4 037 token）上的 12 个配置，按登记的表序：

| 头 | 视图 | 原生槽 | held-out PDMS |
|:--|:--|--:|--:|
| 线性 | GIMM | 4 | 89.75 |
| 线性 | + hold ×2 | 4 | 90.13 |
| MLP | GIMM | 4 | 91.88 |
| MLP | + hold ×2 | 4 | 92.07 |
| 线性 | GIMM | 13 | 90.93 |
| 线性 | + hold ×2 | 13 | 91.45 |
| MLP | GIMM | 13 | 92.24 |
| MLP | + hold ×2 | 13 | 92.44 |
| 线性 | GIMM | 22 | 91.12 |
| 线性 | + hold ×2 | 22 | 91.75 |
| MLP | GIMM | 22 | 92.56 |
| **MLP** | **+ hold ×2** | **22** | **92.64**（选中） |

- fold 1 重复（描述）：选中配置 92.69，N1b 式线性基线 90.51，+2.18（fold 0 上是 +3.12），方向与量级一致。
- 三项大致可加：MLP 在 4 槽上 +2.1（线性 89.75 → 91.88），槽 4 → 22 在 MLP 上 +0.7、在线性上 +1.4，hold 视图 +0.1 到 +0.6。MLP 5 个 seed 的最佳 epoch 在 20–34。
- 选中权重 (0, 1, 2, 2, 1)，β = 0。navtest 上 55.1% 的 token 选 anchor（N1 是 33.7%），原生族 22 个槽分散在 0.6%–4.5% 之间。

| 行 | NC | DAC | EP | TTC | C | PDMS |
|:--|--:|--:|--:|--:|--:|--:|
| N2 | 98.67 | 97.25 | 85.84 | 95.80 | 99.96 | **90.60** |
| N1 | 98.58 | 96.17 | 79.42 | 95.64 | 99.97 | 87.32 |
| human | 100 | 100 | 86.96 | 100 | 99.90 | 94.55 |

按 command：直行 92.29（N1 89.56），左 87.71（81.22），右 81.76（78.60），起步 95.04（93.86）。**EP 超过 human 的 token 占 40.8%**（N1 23.0%）。增益主要在 EP（+6.4）与 DAC（+1.1），NC、TTC 不变。
性质：trick。增益的形状是把车开得比 human 快、同时不多出界，完全是按 PDM scorer 的规则挑轨迹；EP 均值已接近 human（85.8 对 87.0），四成 token 超过 human。

navhard（描述）：N2 = 31.29，仍比原生低（均匀权重 −3.00 [−4.96, −1.01]）。stage 1 DAC 89.3 → 91.1、EP 75 → 84 涨了，但 stage 2 NC 90.9 → 86.3、DDC 90.2 → 85.4、EC 45.7 → 29.9 掉得更多。
更强的头让 navtest 涨 3.3 分，却没让 navhard 变好，与 N1 一样：打分头在偏离起点上没有读出能力，还会挑更快、更不一致的轨迹（EC 掉）。

held-out 上 N2 配置的学习曲线（fold 0，3 个点，每点 5 seed）：拟合行 1/4、1/2、全部 = 90.51 / 91.33 / 92.64，每翻倍 +0.8 到 +1.3，比线性头（+0.4 到 +0.6）陡，且没有变平。N3 的登记预期（held-out +0.3 到 +1.0）按这条曲线看偏保守，不改登记。

## navtest 打分记录（每次打分都记，含不成立的）

| # | 日期 | 配置 | navtest PDMS | 对照 | 配对 Δ [95% CI] | 登记位置 |
|--:|:--|:--|--:|:--|:--|:--|
| 1 | 09-29 | N0 切换器（本 pack 第一次） | 84.94 | 原生 84.17 | +0.77 [+0.28, +1.27] | [N0](2026-09-29-skill-pack-n0.md) |
| 2 | 09-29 | N1 | 87.32 | N0 | +2.38 [+1.94, +2.83] | [N1](2026-09-29-n1-scorer.md) |
| 3 | 09-30 | N1b（λ 网格下扩） | 87.26 | N1 | −0.06 [−0.26, +0.14] | 本文件 N1b 节 |
| 4 | 09-30 | N2（MLP 头 5 seed + hold 视图 + 22 个原生槽） | **90.60** | N1 | +3.28 [+2.89, +3.66]；Bonferroni m = 4：[+2.79, +3.76] | 本文件 N2 节 |
| 5 | 09-30 | S（N1b 线性配方，6 万行） | 87.74 | N1b | +0.48 [+0.23, +0.74]；对 N1 +0.42 [+0.15, +0.70]，Bonferroni m = 5 [+0.06, +0.77] | 本文件 S 节 |

（原生 84.17 与 `lc@−1.0` 84.90 是导航 lane 的读数，不计入本 pack 的次数。）

## navhard 打分记录（描述）

| # | 日期 | 配置 | EPDMS combined | stage 1 / stage 2 | 对 `none` 33.33 |
|--:|:--|:--|--:|:--|:--|
| 1 | 09-30 | N1（原网格重拟合，复现 100%） | 30.24 | 见 N1b 结果节 | 均匀权重配对 −3.05 [−4.55, −1.53] |
| 2 | 09-30 | N1b | 31.10 | 见 N1b 结果节 | 均匀权重配对 −3.48 [−5.00, −1.92] |
| 3 | 09-30 | N2 | 31.29 | 见 N2 结果节 | 均匀权重配对 −3.00 [−4.96, −1.01] |

## 执行记录

- 09-29 23:42 起 N1b 链（`scripts/navsim_raise_n1b.sh`）：navhard 特征 10 min；N1 按原网格重拟合复现存档 navtest 选择 **100%**（最大差 0），过复现门；navtest N1b 23:52 打分一次。
- 09-29 23:45 native family 9 个额外候选（拉长 0.90 / 1.20 / 1.25 / 1.30 / 1.40，横向缩放 0.8 / 1.2 × 拉长 1.00 / 1.15）在 19 968 个 token 上的 devkit 子分标签，14 进程 3 min。

## N1b 结果

**判读：平**（CI 跨零），「λ 触边不是 N1 的瓶颈」。navtest N1b = **87.26**，N1b − N1 = **−0.06 [−0.26, +0.14]**；N1b − N0 +2.32 [+1.87, +2.76]，N1b − 原生 +3.09 [+2.74, +3.44]。

| 行 | NC | DAC | EP | TTC | C | PDMS |
|:--|--:|--:|--:|--:|--:|--:|
| N1b | 98.55 | 96.31 | 79.15 | 95.63 | 99.93 | 87.26 |
| N1 | 98.58 | 96.17 | 79.42 | 95.64 | 99.97 | 87.32 |

- 选中的 λ 都在新网格内部：NC 3e-6、DAC 1e-6、EP 3e-6、TTC 3e-6、C 3e-7。held-out BCE 比 1e-5 时低 0.003–0.015（C 头最大，0.060 → 0.043）。
- held-out PDMS 89.52（N1 89.46，+0.06）；权重 (w_im, w_mul, w_ttc, w_ep, w_c) = (0, 5, 2, 5, 1)，β = 0.5（N1 是 (0, 1, 1, 2, 1)、β = 0）。navtest 上原生槽 76.7%（N1 66.3%），×1.15 占 47.6%。
- 按 command：直行 89.72（N1 89.56），左 80.75（81.22），右 78.31（78.60），起步 93.56（93.86），都在噪声内。
**navhard（描述）：pack 在 navhard 上是负的。** 预期（N1 对 `none` +1 到 +4）被推翻。225 × 2 组 bootstrap 用均匀 stage 2 权重（官方权重不能从 CSV 重算，见缺口分解第 0 节）。

| navhard two-stage | EPDMS | NC s1 / s2 | DAC s1 / s2 | DDC s1 / s2 | EP s1 / s2 | EC s1 / s2 | 均匀权重配对 Δ 对 `none` |
|:--|--:|:--|:--|:--|:--|:--|:--|
| Cinque `none`（GIMM） | 33.33 | 98.0 / 90.9 | 89.3 / 73.6 | 98.3 / 90.2 | 75.0 / 66.1 | 37.8 / 45.7 | – |
| N1 | 30.24 | 96.2 / 88.0 | 89.3 / 70.8 | 97.6 / 87.2 | 79.1 / 77.6 | 40.4 / 36.6 | −3.05 [−4.55, −1.53] |
| N1b | 31.10 | 95.3 / 88.0 | 88.7 / 73.6 | 97.3 / 88.8 | 78.5 / 76.5 | 39.1 / 37.2 | −3.48 [−5.00, −1.92] |
| E6 Hydra（hold 特征，存档） | 25.74 | 95.7 / 83.5 | 79.8 / 67.1 | 96.0 / 84.7 | 77.9 / 75.6 | 59.6 / 63.0 | −7.79 [−9.94, −5.63] |

读法：N1 在 navhard 上照样把车开快了（stage 2 EP 66 → 78），但 NC（−1.8 / −2.8）、DAC（stage 2 −2.8）、DDC（−0.7 / −3.0）和 EC（stage 2 −9）一起掉，乘性项的损失大于 EP 的收益。
打分头在 navtrain 的真实 stage-one 帧上训，stage 2 的 3DGS 合成偏离起点是分布外，DAC 头在那里没有读出能力。所以 N1 的 +3.15 是 navtest 特有的 metric 对准，不是可迁移的能力；这正是 decisions 第 68 条写的「会推翻本条的证据」之一（navhard 上原生槽不涨分）。

- 预期（+0.0 到 +0.8，四成把握）没兑现：头的 BCE 变好了，但排序没变好。线性头在这套特征上已经到头，下一步的增益要从别处来（见下面的设计节）。
