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

## navtest 打分记录（每次打分都记，含不成立的）

| # | 日期 | 配置 | navtest PDMS | 对照 | 配对 Δ [95% CI] | 登记位置 |
|--:|:--|:--|--:|:--|:--|:--|
| 1 | 09-29 | N0 切换器（本 pack 第一次） | 84.94 | 原生 84.17 | +0.77 [+0.28, +1.27] | [N0](2026-09-29-skill-pack-n0.md) |
| 2 | 09-29 | N1 | 87.32 | N0 | +2.38 [+1.94, +2.83] | [N1](2026-09-29-n1-scorer.md) |

（原生 84.17 与 `lc@−1.0` 84.90 是导航 lane 的读数，不计入本 pack 的次数。）

## navhard 打分记录（描述）

| # | 日期 | 配置 | EPDMS combined | stage 1 / stage 2 | 对 `none` 33.33 |
|--:|:--|:--|--:|:--|:--|

## 执行记录
