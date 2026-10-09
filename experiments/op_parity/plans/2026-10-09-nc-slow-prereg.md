# op_parity nc-slow 预登记：「在哪里减速」能不能从推理时拿得到的输入里判出来（lane NC1，2026-10-09，任何本线读数之前写定）

接第 196 条（N1）。N1 给了上限：同一路径、只缩放纵向，逐 token 用 simulator 分数选档，navtest no-EC EPDMS +1.13、EP −0.05；全体一刀切没有净正的档。
本线问的是这 +1.13 里有多少能由**非特权输入**判出来，以及用哪种输入最便宜。两个阶段，阶段 1 是闸门。

写本稿之前已经看过的数：第 196 条全部（navtest 上 SH30 的逐类失败表、oracle 回收率、一刀切表，即 navtest 的标签层面读数），第 153、157、158、179、190–195、197、200 条。
没有看过：navtrain 上任何纵向缩放后的分数；任何预测器对这些标签的 AUC 或选档增益；任何输入与「该减速」标签的相关。

## 与已有条目的差别（不重测的部分）

- **第 190–195 条（F19 族与 selector）**：F19 是偏移 × 曲率 × 速度的 19 条候选，标签与训练只在转弯 token（|dyaw| ≥ 20°，28 323 个）上，selector 学出来的是 DAC 驱动的偏移 / 曲率选择，
  几乎不选降速（第 195 条：约 5% 的移动）；门 A（全 token）−1.32。本线的族**只有纵向**（同一路径，弧长 × a），标签覆盖**直行 token**（A 类失败 59% 在 < 5°），
  判读对象是 NC / TTC，不是 DAC。selector 的结构（非特权输入 → 候选分数回归 → 取最大）沿用，不算新机制；新的是族、标签覆盖面与逐输入的对比。
- **第 157 条（lead 停车余量规则）**：固定扣 2 m 的执行层规则，输入是模型自己的 lead_x，无训练。本线不测任何固定规则；openpilot 原生输出在这里只作为一个**学习的**读出头的输入（b 臂）。
- **第 158 条（agent hinge）**：plan 头上的 loss 项，标签是日志车辆框的几何重叠。本线阶段 1 不训练 driver；标签是 simulator 的分数（含 EP 代价与非反应式 TTC），不是几何 hinge。
- **第 179 条**：plan 对自身 lead 的 clearance 这一个**标量**对 NC / TTC 失败 AUC 0.52，无训练、无判线。本线的 b 臂是 lead / meta / action / plan 速度全部原生输出上的学习读出，目标换成「减速是否净赚」，
  有 navtrain 标签。若 b 臂的 AUC 仍在 0.5 附近，就是对第 179 条的同向复读，不另记新结论。
- **第 197 / 200 条**：真值 agent 占用经 memory 通道进 policy，NC 不动。那是「把真值喂给 driver」；本线的 c 臂是「真值 lead 间距喂给一个离线选档头」，只作特权上限，不进推理路径。

## 族、标签、集合

- **族**：`turn_ceiling.speed`（N1 用的同一个变换），a ∈ {0.7, 0.8, 0.9, 1.0, 1.1} 都打分；**选档集合 S = {0.7, 0.8, 0.9, 1.0}**（只许减速）。1.1 只为与 N1 的主族对齐，不进选档。
- **打分**：`python -m jevdrive.bench score-poses --traffic non_reactive`（navtrain 侧加 `--mcache v2_navtrain`），与 N1 相同；不写新 runner。no-EC EPDMS（缩放后的 plan 没有 EC）。
- **navtest 侧**（测试）：12 146 token × SH30-F-s0 / s1，分数直接用 N1 的 `nc_tax/score_all.csv`，不重打。
- **navtrain 侧**（训练）：plan 取第 190 条的 held-out 折模型（`CF5f{j}-F-s0`，j = token 所在 log 的折）在 12 个 `navtrain_full` shard 上已导出的位姿，每个 token 用没见过它的 log 的那个模型。
  token 集合 = v2 navtrain metric cache 里有的全部 token：28 323 个转弯 token + 非转弯（rest）token。rest 现有 3 个 shard（4 497 个），
  本线用 `nt_cache.py run --stage rest --limit 17` 把 rest 补到 17 个按日期均匀取的 shard（约 2.5 万个 token，新增约 10 GB，cache 合计约 23 GB，在 nt-cache 的 40 GB 上限内）。
  metadata 取自 shard 的 `tab.npz`（`navsim_zs/index/navtrain.pkl` 已不在 box 上）。
- **样本权重**：rest token 的权重 = navtrain rest 总数 / 入选的 rest 数，转弯 token 权重 1，使加权后的转弯占比等于 navtrain 的真实占比。拟合、OOF 选模、阈值都用这组权重。
- **逐 token 标签**（a ∈ {0.7, 0.8, 0.9}）：g_a = score(a) − score(1.0)；`y_slow` = max_a g_a > 1e-9（某个更慢的档严格高于 1.0）；
  `y_fail` = 1.0 档 NC < 1 或 TTC < 1，且存在更慢的档使其通过并且 g_a > 0（N1 的「干净救回」限制在减速档上）。

## 输入（臂）

每个臂 = 一组输入流 + 下节的学习器。训练侧特征来自 held-out 折模型的一次前向，测试侧来自 SH30-F-s0 / s1 自己（第 191 条的做法；折模型与 SH30 的隐状态逐坐标相关 0.9965）。

| 臂 | 输入流 | 特权 |
|:--|:--|:--|
| E | ego 向量（`tab.npz` 的 20 维：速度、加速度、指令、历史位姿）+ plan（8 个位姿、分段速度、4 s 弧长、末端转角） | 否 |
| H | E + policy 隐状态 `select_4`、`mean`（各 512） | 否 |
| V | E + 冻结 Cinque 视觉 token（最新一个 slot，32 × 512，warp 帧） | 否 |
| N | E + H + V | 否 |
| OP | E + openpilot 原生输出：`lead`（3 个假设 × 6 个时刻 × [x, y, v, a] 的均值与标准差）、`lead_prob`（3）、`meta`（55：engaged、各时域的刹车 / 急刹 / 油门概率等）、`action`（期望曲率与加速度）、plan 的速度与加速度通道 | 否 |
| OPH | E + H + V + OP（全部非特权输入） | 否 |
| G0 | E + 真值 lead 间距：t0 时沿 plan 路径的走廊内最近对象的净距、对象速度（0 → 0.5 s 位移）、接近速度、时距、TTC、横向偏置、类别，第二近对象，10 / 20 / 30 m 内走廊对象数 | **是** |
| G1 | E + G0 + 每个档 a ∈ S 下缩放后的 plan 与**日志未来**对象框的最小净距（0.1 s 插值，`lib/agent_hinge.py` 的距离） | **是** |

b 臂的可行性已核对：parity 路径的 policy 输出向量里有 `lead`、`lead_prob`、`meta`、`action`、`plan` 各 slice（`sc_infer.py` 已这样读 lead），不改第三方代码，只多读几段。
G0 / G1 用 `agent_labels/{navtrain_all,navtest}.npz`（第 158 条建的日志框），只作上限探针。不用 WA-JEPA 的任何权重或特征。

## 学习器与选档规则（全部在 navtrain 上定，navtest 分数只在 report 里读一次）

- **T**：每个档一个 `HistGradientBoostingRegressor` 回归 g_a（3 个），高维流先降维（H：训练侧白化 PCA 32 维；V：32 个 token 平均后 PCA 32 维），默认超参（max_iter 300、lr 0.05、叶 31、early stopping 关），带样本权重。
- **C**：一个 `HistGradientBoostingClassifier` 预测 `y_slow`，选档 = 概率过阈值时取固定档 a_fix；(τ, a_fix) 在 navtrain OOF 上定。输入同 T。
- **M**（只用于含 H 或 V 的臂）：第 191 条的头（各流线性 → GELU，视觉 token 4 个 query 的注意力池化，MLP 出 3 个 g_a），MSE，AdamW lr 1e-3、wd 1e-2、宽 256、dropout 0.1，
  内层按 log 留 20% 以加权 OOF 增益早停，5 个初始化取平均。
- **选档**：pick = argmax_{a ∈ {0.7, 0.8, 0.9}} ĝ_a，若 max ĝ_a > τ，否则 1.0。τ 在 navtrain OOF 预测上取使加权实得增益最大的值（候选 = max ĝ_a 的 200 个分位点与 +∞）。
- **OOF**：按 log 的 5 折，折 = 第 190 条的 cf5 折（特征本来就按这个折 held-out）。每个臂在 T / C / M 里取 navtrain OOF 加权实得增益最高的一个学习器；navtest 上每个臂只读被选中的那一个。
  最终模型在全部 navtrain 行上重拟合，τ 沿用 OOF 的值。

## 读数

对每个臂（navtest，两个 seed 的均值，按 log 配对 bootstrap，B 10 000，`jevdrive.stats`）：

1. **AUC**：max_a ĝ_a（或 C 的概率）对 `y_slow` 与对 `y_fail`，带 CI；navtrain OOF 的同一读数并列。
2. **工作点**：被移动的 token 比例；precision = 被移动的 token 里实得增益 > 0 的比例；recall = `y_fail` token 里被移动且救回的比例。
3. **换算成 EPDMS**：Δ = 100 × mean[score(pick) − score(1.0)]（no-EC），及其分解 ΔNC、ΔTTC、ΔDAC、ΔEP；NC 失败数与 TTC-only 失败数的变化（修好 / 新增）。
4. **回收比例** = Δ / 1.13（N1 的板级上限）。另报两个参照分母：只许减速档的同口径上限（O_slow），以及 S 上逐 token 取最大的全体上限（O_any，含 DAC 等非 NC 来源）。
5. **对照**：把每个 token 的 pick 换给随机别的 token（置换 200 次）的增益分布；全体固定档 0.9 / 0.8（N1 已有：−0.33 / −1.44）。
6. **次要读数（登记，不进判定）**：(i) 最好的非特权臂与 G0 在 navtest 内按 log 5 折 cross-fit（标签域内）的 Δ，用来区分「navtrain → navtest 的域差」与「输入里没有」；
   (ii) 最好的非特权臂的学习曲线（navtrain log 的 25% / 50% / 100%）；(iii) 按 N1 的失败类别（A1 / A2 / B / C / D / E）与速度段拆 recall。

## 闸门（读任何 navtest 增益之前必须过）

- **G-id**：navtrain 转弯 token 上 a = 1.0 行的 8 个子分与第 190 条 `sh30_crossfit/labels/score.csv` 的 `h` 行逐 token 相同（同一位姿、同一打分）。
- **G-leak**：每个 navtrain 行所用折模型的折 = `sha256("cf5|" + log) % 5`，该 log 不在这个折的训练 split 里；本线前向得到的 plan 与该折模型已存的 plan 在前 15 个点上最大差 < 0.03 m（速度 ≥ 0.5 m/s 的行，容差与第 191 条相同，超容差的行数按比例报）。
- **G-plan**：navtest 上本线前向的 SH30 plan 与 bench 存档的 plan 同口径一致。
- **G-oracle**：用 `score_all.csv` 按 N1 的规则重算板级上限，得 +1.13（s0 +1.10 / s1 +1.15）。
- navtest 与 navtrain 无共享 log。

## 阶段 1 的判定线（任务书给定）

**过线** = 至少一个非特权臂（E、H、V、N、OP、OPH）在 navtest 上 Δ ≥ 0.30 × 1.13 = **+0.339**，95% CI 下界 > 0，且 EP 损失（−ΔEP）< 0.3。
六个臂同时读，另报 Bonferroni（m = 6）下界；只在未校正的 95% 下过线时如实写「过线但经不起多重比较」，阶段 2 照登记进行，结论强度降一档。
**不过线**：lane 结束，报告哪个非特权输入最接近（Δ、回收比例、AUC）以及 G0 / G1 的上限是多少。几种读法事先写定：

| 非特权最好的臂 | G0 | 读法 |
|:--|:--|:--|
| 过线 | 任意 | 「在哪里减速」可判，进阶段 2 |
| 不过线 | 过线 | 信息在真值 lead 间距里、不在现有输入里：缺的是近距测距表征（与第 153、157 条一致），阶段 2 不做 |
| 不过线 | 不过线，G1 过线 | t0 的间距不够，要的是对方的未来运动；选档头这条路在 4 s 非反应式口径下需要预测而不是测距 |
| 不过线 | 都不过线 | 标签在 navtrain 规模下学不出来（看次要读数 i、ii 区分标签量与域差），族本身不是可学的目标 |

E 臂（只有 ego 与 plan）若单独过线，按过线处理，同时写明「不需要视觉」：这时最便宜的机制是 plan 头上的速度规则，阶段 2 的机制相应取最轻的那个。

## 阶段 2（只有阶段 1 过线才做；机制在阶段 1 读数之后、任何阶段 2 分数之前以本文件的修订节写定并提交）

候选从轻到重：(i) 阶段 1 的选档头直接作为 bench 模型后缀（零训练，第 193 条的接法）；(ii) 现有输出上的门；(iii) 速度剖面上的后果 loss（SH30 pilot 配方里加一项）。按每卡时的预期增益选一个。
协议：SH30 pilot 配方（3 000 步），从一开始就 2 个 seed，对同配方基线；navtest 全量（含 EC 的官方 EPDMS）+ N1 的失败分类拆分；navhard；HUGSIM 起步停滞护栏（`launch_stall`，第 143、146 条）。
**判定线**：navtest EPDMS ≥ +0.3 且 CI 下界 > 0；NC + TTC 失败数下降且 CI 不含 0；HUGSIM 起步停滞不增加。

## 成本与运行

- CPU：rest metric cache 约 2.1 万 token × 2.8 core-s ≈ 17 core-h；navtrain 打分约 5.4 万 token × 5 档，按 N1 实测 0.09 core-s / 行 ≈ 7 core-h。pool 作业数封顶 4 个（48 核），给 AlpaSim lane 留核。
- GPU：特征前向 12 个 shard + navtest 2 个 seed，各约 2–3 分钟；M 学习器 4 个臂。阶段 1 合计估 < 1.5 卡时；整条 lane 预算 6 卡时。
- 分级启动：先 shard 0 的 64 行前向 + 240 个 token 的打分冒烟（过 G-leak / G-id 的小样版），再全量。
- 一条自推进的链 `scripts/nc_slow_chain.sh`，tmux 窗口 `jev:nc-slow`，状态 `$DATA_DIR/runs/op_parity/nc_slow/chain/{STATUS, DONE, ERROR, log.txt}`；代码 `scripts/nc_slow.py`，在 `jevdrive.run.Run` 内，
  split 取自 `jevdrive.data.splits`（`navsim/navtrain`、`navsim/navtest`、`navsim/op-parity-cf5f{j}-{train,dev}`）。
- 结果 `experiments/op_parity/results/nc_slow.md` 与 `results/nc_slow/`，决策第 203 条。

## 限定（事先写明）

- non-reactive、4 s、no-EC：减速的回收在闭环里偏乐观（N1 的同一限定）；选档逐帧独立，没有时间一致性（第 193 / 194 条的 EC 与 jerk 代价在这里看不到）。
- navtrain 标签来自折模型的 plan，测试用 SH30 的 plan；转弯 token 在训练集合里过采样，靠权重校正。
- 正例少（预计 navtrain 约 1–2%，几百到一千个），学习曲线是判断「标签量不够」的唯一依据。
- G0 的对象速度用 0 → 0.5 s 的日志位移，严格说用了 0.5 s 的未来；G1 用了全部 4 s 的日志未来。两者都只是上限探针。
