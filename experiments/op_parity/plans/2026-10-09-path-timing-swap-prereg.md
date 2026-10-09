# op_parity path-timing-swap 预登记：转弯 token 的 DAC 差距是路径形状错，还是路径与 timing 的耦合（lane SW1，2026-10-09，任何 swap 后的分数或几何读数之前写定并 push）

纯测量 lane：只用存档 plan、逐 token 分数与 token 表，CPU 重打分。日志路径与日志 timing 是特权输入，在这里只作分析用的 swap，
不构成方法，也不进任何可报告的推理路径。没有新的 WA-JEPA 推理，没有 GPU 作业。判定线（第 6 节）由 main 给定，写死不改。

## 1. 假设（main 提出，未验证）

navtest 上 SH30 对 WA-JEPA 的 −2.16 有四分之三在 > 20° token，主要是 DAC。已排除：缺道路几何（第 200 条）、缺「对变化的响应」（第 206 条）。
假设：转弯 token 的 DAC 差距有一大部分来自回归下路径形状与 timing 的耦合，而不是路径形状本身错。逐时刻回归的 plan 是对「转弯走多快」的
不确定性取平均；曲线上不同弧长的点的平均落在曲线**内侧**（chord effect），正好给出占主导的切内角（第 153 条：急弯 DAC 失败的 54%），
随转角而不是随 innovation 增长，hinge 只能修一部分，泄漏日志路径与 timing 时消失（第 200 条 JP 臂）。备择：plan 的几何路径本身错（半径 / 出口车道）。

## 2. 已有条目与本线的差别（不重测）

- 第 153 条：急弯 DAC 失败的侧别与类型（切内角 54%、转不过去 26%），`plan_kin` 的时间对齐横向偏移。没有把路径与 timing 拆开重打分。本线沿用其定义。
- 第 170 条：强 hinge 降的是原始 plan 出界与切内角，转不过去不动。本线问 λ 10 → λ 30 是否改变 2 × 2 的分解（RMH10 / P2 对 SH30）。
- 第 178 / 186 条：SH30 plan 邻域的逐 token best-of-K 上限；只动速度的固定缩放无净收益，放慢能把出界推到 4 s 视野外。那是按分数挑候选的 oracle，
  速度轴是弧长 × 常数。本线不挑候选：每个 token 恰好一条 swap 轨迹，timing 换成日志的整条弧长-时间曲线。视野效应在本线同样存在，见 E3。
- 第 196 条：同一路径、弧长 × a 的纵向 oracle 救回 NC 失败 74%。本线的（plan 路径，日志 timing）在 NC + TTC 上是它的非 oracle 版本（用日志 timing 而不是按分数选 a）。
- 第 200 条：泄漏日志路径距离场的 JP 臂。那是训练出来的臂，本线是存档 plan 上的后处理 swap。
- 第 206 条：等弧长下横向 innovation gain 相同（0.976 对 0.982），SH30 的 plan 沿同一路径走得短。那是斜率（比例响应），不是带符号的内外侧偏移，也没有重打分。
  本线量的是带符号偏移（内侧为正）及其与弧长差的关系，并用官方 scorer 给出每个格子的分数。
- `research/decisions.md` 里 grep 重定时 / 弧长 / 平均 / 路径-速度分解：除上述各条外，第 164 条是 WOD 上的弧长缩放（另一个榜），没有 2 × 2 swap。

写本稿前看过的数：plan.md 第 1 节、第 153 / 170 / 178 / 196 / 200 / 206 条与 `navtest_strata.md` 的已发表读数；各模型存档的 navtest 均分（`T.load` 打印的
89.47 / 89.63 等，均为已发表数）；`tab.npz`、导出位姿、WA-JEPA 轨迹字典的形状与前三行；nc_tax 的恒等闸门结果与打分耗时。没有算过任何 swap 轨迹、任何 swap 分数、
任何带符号的曲线偏移。

## 3. 轨迹与四个格子（全部写死）

**数据**：`runs/op_parity/cache/lb_navtest/tab.npz` 的 `fut`（日志未来，后轴，0.5–4 s 八个位姿）；模型 plan 为 bench 导出位姿 (n, 8, 3)。split `navsim/navtest`，12 146 token。
模型：SH30-F-s0 / s1（主）、RMH10-F-s0 / s1、P2-F-s0 / s1（无 hinge）、OT30-F-s0 / s1、WA-JEPA（存档轨迹 12 027 token，其余 119 个只有分数；WA-JEPA 是对照列）。

**曲线**（plan 与日志同一做法）：顶点 = 原点 + 8 个位姿；与上一个保留顶点距离 < 0.05 m 的顶点丢弃（静止抖动），其弧长取上一个保留顶点的值。
保留顶点 ≥ 3 个时，x、y 对累计弦长作三次样条（原点端切向固定为 (1, 0)，即后轴沿 t0 航向出发；末端 natural），否则折线。
弧长 s 在样条上按 0.05 m 的密采样累计。航向 ψ(s) = 顶点位姿航向（unwrap）对 s 的线性插值。用样条而不是折线的理由：scorer 把 8 个位姿按时间线性插值，
在折线的弦上重采样会把 swap 轨迹的顶点再往内侧移一个矢高（3 m 弦、R 10 m 约 0.1 m），与待测效应同号。
**timing** = 该轨迹自己的 s(t_k)，k = 1..8。

**四个格子**（缩写：第一个字母是路径，第二个是 timing；P = plan，L = log）：
- **PP** = 存档 plan 原数组（恒等闸门：8 个子分与存档逐 token 相同，EP 容差 1e-6，否则停）。
- **PL**（plan 路径，日志 timing）= plan 曲线在 s_log(t_k) 处的 (x, y, ψ)。
- **LP**（日志路径，plan timing）= 日志曲线在 s_plan(t_k) 处的 (x, y, ψ)。
- **LL** = 日志原数组（上限检查）。

**外推规则（主口径 ext-arc）**：所需弧长超过曲线总长时，从末位姿起沿恒曲率圆弧延伸；曲率 = 末段航向差 / max(末段弦长, 0.5 m)，截到 |κ| ≤ 0.3 m⁻¹，
起始切向 = 末位姿航向（`turn_ceiling.speed` 的规则，第 178 / 196 条用的同一条）。曲线总长 < 0.05 m 时从原点沿 t0 航向直线延伸。
**敏感性口径 ext-line**：沿末位姿航向直线延伸；只对 SH30 两个 seed 的 PL / LP 打分。
报告：每个格子需要外推的 token 数与外推长度分布（> 0.5 m、> 2 m）；主读数在「外推 ≤ 0.5 m」子集上重算一遍；两种外推口径的差。
若判定读数在全体与「外推 ≤ 0.5 m」子集、或两种外推口径之间落入不同判定档，按较不利于假设的一档下结论并写明。

**打分**：`python -m jevdrive.bench score-poses --traffic non_reactive`（v2 navtest metric cache，与存档 navtest 分数同口径），no-EC EPDMS（第 196 条口径）。
切内角 / 转不过去沿用第 153 条：对各格子的 DAC 失败行跑 `fd_navsim` 的带钩子重放，取 LQR 状态下第一个出界角的侧别；
切内角 = DAC 失败且侧别 = 转向侧（日志 4 s 航向变化的符号）；转不过去 = DAC 失败、非内侧、且该格子轨迹的 4 s 航向 / 日志 4 s 航向 < 0.9；其余为 other。
转角桶：bench strata 的 |日志 4 s 航向变化| < 5 / 5–20 / 20–45 / > 45°；方向 = 该航向变化的符号（左 / 右）。

## 4. 读数

1. 每个模型 × 格子：no-EC EPDMS 与 DAC、NC、TTC、EP、LK、DDC；全体与四个转角桶；> 45° 上切内角 / 转不过去 / other 的失败率。
   模型内 seed 取均值；CI 为按 log 的 cluster bootstrap（`jevdrive.stats`，B 10 000）。
2. 主读数（SH30，两 seed 的 token-seed 合并）：> 20° 与 > 45° 上 PP 的 DAC 失败在 PL 下、在 LP 下各消失多少；NC + TTC 失败（NC < 1 或 TTC < 1）同样。
   两个口径都报：**gross** = PP 失败的 token-seed 中该格子通过的比例；**net** = 1 − 该格子失败数 / PP 失败数（计入新增失败）。
3. 对 WA-JEPA 的 −2.16（全体 12 146 token，存档分数）每个 swap 关掉多少：(SH30 格子的 no-EC EPDMS − SH30 PP) / (WA-JEPA PP − SH30 PP)，两者都用 no-EC 口径并在表里写明；
   另给 WA-JEPA 自己的 2 × 2（12 027 token）。
4. **几何检查**（不经 scorer；> 20° 与 > 45°、左 / 右、四个桶；内侧为正，内侧 = 日志转向一侧；按 log 的 cluster bootstrap）。每个 token、每个视野 k：
   - **T_k（timed，时间对齐）** = (P(t_k) − L(t_k)) · n_L(t_k)：plan 的定时点相对同一时刻日志点、沿日志法向的偏移。这是逐时刻回归目标与 `plan_kin` 看到的量。
   - **C_k（curve，等弧长）** = (P̃(s*) − L̃(s*)) · n_L(s*)，s* = min(s_plan(t_k), s_log(t_k))：两条曲线在同一弧长处的偏移，只含形状。timing 相同时 T_k = C_k。
   - **X_k（cross-track）** = P(t_k) 到日志曲线（按 ext-arc 延伸）最近点的带符号距离：以日志曲线代表车道时，基于地图的 scorer 看到的量。
   汇总：token 内对 k = 1..8 取均值，以及 k = 8（4 s）；桶内对 token-seed 取均值。**判定用比值 ρ = mean(C) / mean(T)**（> 20°，k 均值，SH30 两 seed），CI 为 cluster bootstrap 的比值。
5. **与弧长差的关系**：Δs = s_log(4 s) − s_plan(4 s)，κ̄ = |日志 4 s 航向变化| / max(L_log, 1 m)，z = κ̄ Δs² / 2。> 20° token 上：
   (T_8 − C_8) 对 z 的斜率（纯几何恒等式的校验，应约为 1）；C_8 与 X_8 对 z 的 OLS 斜率与 Spearman（Δs > 0 与 Δs < 0 分开各报一次），以及 C_8 对带符号 Δs 的斜率。
   chord effect 预测内侧偏移随 κ̄ × 弧长差增长且对 Δs 的符号对称；半径错的路径与 Δs 无关。

## 5. 写稿时的事前分析与追加读数（不改第 6 节的线）

写定义时发现的一点，事前记下：单条 plan 的定时点全部落在它自己的曲线上，所以基于地图的 DAC 只取决于 plan 的曲线、4 s 内走到曲线的哪里、以及 LQR 跟踪，
不取决于时间对齐偏移 T。若「对 timing 不确定性取平均」是真实机制，被平均的是训练目标，结果是 **plan 的曲线本身**被压向内侧（弧长散布 σ、曲率 κ 时约 κσ²/2），
这会表现为 C > 0，也不会被 PL 修掉。因此第 6 节的两条线检验的是较窄的命题「路径形状对、错在 timing，且 DAC 失败经由 timing 发生」；
按这两条线判「证伪」并不等于否定「形状错的成因是对 timing 取平均」。为把两者分开，追加三个读数，判读规则同样事前写死：

- **E1（chord 量级，无自由参数的上界）**：残差比 r = s_log(t_k) / s_plan(t_k)，取自全部 navtest token（s_plan(4 s) ≥ 2 m），按转角桶 × t0 速度带 × k 分格，
  格内除以均值使 E[r] = 1（只留散布）。对每个 > 20° token，预测点 = 日志曲线上 s_log(t_k) × r_j 处各点的平均（r_j 遍历该格经验分布），
  预测内侧偏移 X̂_k = 该平均点到日志曲线的带符号距离。跨 token 的残差散布 = 条件散布 + 模型误差，所以 X̂ 是 chord 机制的**上界**。读数：桶内 mean(X̂) 对 mean(C)（k 均值与 4 s）。
- **E2**：第 4.5 节的关系（已列）。
- **E3（视野对照）**：PL 的 gross / net 按 L_log ≥ L_plan（重定时把轨迹拉长，不可能靠缩短视野获益）与 L_log < L_plan 分开报；LP 同样。
  PL 下消失的失败若集中在 L_log < L_plan，那是「少走一段」，不是耦合的证据。

## 6. 判定线（main 给定，不改）

- **SUPPORTED**：用日志 timing 重定时 plan 自己的路径（PL）去掉 SH30 在 > 20° token 上 ≥ 40% 的 DAC 失败（CI 下界 > 25%），**且** plan 的曲线在等弧长下
  偏向日志内侧的量小于定时点内侧偏移的一半（ρ < 0.5）。
- **FALSIFIED**：重定时去掉不到 15%，**或** 曲线自身的内侧偏移 ≥ 定时点偏移的 80%（ρ ≥ 0.8）。
- 其间为 **PARTIAL**。

操作化（事前写死）：「去掉的比例」判定用 gross（字面口径）；net 并列报出，若 net 落入比 gross 低的判定档，按较低档下结论并写明。
mean(T) ≤ 0（定时点平均不在内侧）时 ρ 无定义，几何一条记为「前提不成立」，单独说明，整体不判 SUPPORTED。

## 7. 各结果下对方法的结论（事前写死）

- **SUPPORTED**：杠杆是速度与路径的耦合。方法方向：把输出拆成一条路径曲线（弧长参数）加一个单独的速度剖面，或用不取平均的 head（多模态 / 分类 / 分位）；
  加强 hinge 或补道路几何不是主线。
- **FALSIFIED，且 E1 的上界 ≥ 观测 C 的一半、C 随 z 增长（E2 斜率 CI > 0）**：形状确实错，但量级与「对 timing 取平均」相容。杠杆仍在输出参数化
  （路径 / 速度分解或不取平均的 head），依据是 E1 / E2，不是 2 × 2；记为「按线证伪，机制未排除」，下一步需要一个训练臂来验证，swap 不能再给更多。
- **FALSIFIED，且 E1 上界 < 观测 C 的 25% 或 C 与 z 无关**：杠杆是转弯 token 上的路径形状本身（半径 / 出口车道的选择与精度），与 timing 无关；
  路径 / 速度分解不是答案，方向回到路径的监督信号（评价器后果、转弯目标）。
- **PARTIAL**：两种成分都有；按 PL 与 LP 各自关掉的分差份额排序，写明哪一半更大，不下单一机制的结论。
- 无论哪种：若 LP（日志路径，plan timing）在 > 20° 上去掉大部分 DAC 失败而 PL 不能，路径是主项；若 NC + TTC 主要被 PL 去掉，timing 是碰撞类的主项（与第 196 条一致）。

## 8. 执行

代码 `experiments/op_parity/scripts/pt_swap.py`（`jevdrive.run.Run` 内；打分是 score-poses 自己的并行，重放是 `fd_navsim` 的 worker 池，曲线构造单核 2 分钟内，故不用 `jevdrive.par`）
与 `pt_swap_chain.sh`（tmux `jev:pt-swap`，DONE / ERROR / STATUS；阶段：build → stage 0 少量 token 的恒等闸门 → 全量打分 + 闸门 → DAC 失败行重放 → report）。
打分并发 JOBS=3，不写死核数。输出 `runs/op_parity/pt_swap/`，完成后删掉 poses 与逐行中间件，只留小表。结果 `experiments/op_parity/results/path_timing_swap.md` 与
`results/pt_swap/`，图 `figs/pt_swap/`，决策第 207 条。
