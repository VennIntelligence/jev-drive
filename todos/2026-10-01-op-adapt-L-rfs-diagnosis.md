# op-adapt L：为什么 RFS 没动？（现有结果上的诊断，预登记）

状态: 预登记（写于任何数字之前；此前只读了 decisions 第 47、77、78 条、op-adapt L 预登记与后续、`log_expert_audit.py`、`op_adapt_l_readout.py`、`jevdrive/waymo.py` 里的 RFS 实现，没有算过任何切片计数、RFS 或距离）。
主题: [decisions 第 78 条](../research/decisions.md)、[op-adapt L 预登记](2026-10-01-op-adapt-L-prereg.md)、[后续检查](2026-10-01-op-adapt-L-followup.md)。
约束: 不训练、不动已登记的协议、不碰 test、不碰 r2 的 run 目录、不编辑 decisions。只读 `$DATA_DIR/runs/op_adapt_L/readout/{O,main-s0,main-s1,main-s2}/eval/rater.npz`（每个模型在 479 个 val rater 帧上的 plan，读数管线已存）与 WOD 的 index / past / future / rater 集合。纯 CPU，核 100–160，无 GPU，不登记调度表。

名词（首次使用处各给一句）：**RFS**（Rater Feedback Score）= WOD-E2E 榜单指标，每帧有 3 条 rater 轨迹与 0–10 的打分；plan 在 3 s 与 5 s 处各和每条 rater 轨迹比较，落进该 rater 的 **trust region**（信任区：沿该 rater 轨迹方向的纵向 ±4 m / 横向 ±1 m at 3 s，5 s 处 7.2 / 1.8 m，速度低于 1.4 m/s 时缩到一半）内就取该 rater 的分，区外每超一个阈值乘 0.1，取最好的 rater，两个时刻平均；不在任一 rater 的区内（两时刻同时）则不低于 floor 4。榜单口径 = 先按场景类别取均值再对类别等权平均。**capture**（捕获）= op-adapt L 预登记的定义（start = plan 4 s 位移 ≥ 人类的一半；stop = plan 最后 0.5 s 速度 ≤ 1 m/s；turn onset = plan 4 s 横向偏移与人类同号且 ≥ 人类的一半）。**rater 帧** = WOD val 里带 rater 打分的 479 帧（每段一帧，约第 149–150 帧）。`main` = 适配模型三个 seed（s0、s1、s2），`O` = 原模型。**×1.06** = plan 的 x 乘 1.06 的纵向校准（提交 7.92 用的约定）。

## 0. 公共定义

- **行集合**：479 个 val rater 帧，名字 `<sequence>-<frame>`；原模型与 `main` 在这些帧上的 plan 取 readout 已存的 `rater.npz`（同一个前向协议，O 的 RFS = 8.004）。RFS 用 `jevdrive/waymo.py::rater_feedback_score`（官方代码的移植，在本 lane 复现 8.004 / 官方 8.005）。
- **cluster**：段（sequence）。bootstrap = 段聚类，2 000 次，seed 0，同一组重抽权重用于被比较的两侧（`op_adapt_l.cluster_boot` / `boot_mean`）。
- **场景类别**：`sets["rater"]["cluster"]`（10 类，榜单口径）。帧 i 对榜单总分的**权重** w_i = 1 / (C · n_c(i))，C = 出现的类别数，n_c = 该类帧数，所以 Σ_i w_i x_i 就是榜单口径的「按类取均值再等权平均」。分项对榜单的贡献一律用 Σ w_i x_i 表示（单位 = RFS 点）。
- **切片**：用 `scripts/log_expert_audit.py::slices`（BASE 阈值）在 rater 帧的人类 logged future（0.5 s 格点 4 s 位移、后轴自车系）上打标签：start、stay、stop、turn_onset、in_turn、nudge、lane_change、control；另设 `other`（以上都不成立，有 future）和 `no_future`（该帧没有 logged future，切片不可判）。切片互相重叠，所以：**(a) 重叠计数**每个切片单独数；**(b) 互斥标签**按优先级 turn_onset > start > stop > stay > in_turn > nudge > lane_change > control > other（no_future 单独一类）给每帧一个标签，用于把总 Δ 精确拆成各项之和（分项加起来 = 总数，是自检）。**imitated 切片 S** = start ∪ stop ∪ turn_onset（重叠并集，op-adapt L 模仿过的三类）。
- **RFS 的 max**：每帧 RFS 的理论最大 = 该帧 3 位 rater 打分里的最大值（plan 取到该 rater 轨迹时 norm = 0）。frame gap_i = max_i − RFS_i（O 的）。
- 榜单顶 = 8.17（brief 给定）；O = 8.004，所以「到榜单顶的缺口」= 0.166。

## 1. Q1 覆盖与 headroom

表 1：每个切片（重叠计数与互斥标签各一列）的帧数、段数、类别分布；O 的切片平均 RFS；切片帧的满分 headroom H_slice = Σ_{i∈slice} w_i·gap_i（榜单单位）及占总 H_all = Σ_i w_i·gap_i 的份额、H_slice / 0.166（占到榜单顶缺口的倍数）；另列切片 rater 分的平均跨度（3 位 rater 打分的 max − min，衡量这一帧的「押注」大小）。

读法（先写死）：
- **R1a「imitated 切片 headroom 太小」**：H_S < 0.166（即使 S 帧全拿满分，也到不了榜单顶）**或** H_S / H_all < 0.15 时成立；否则否定。并列报 H_S / 0.166 的具体倍数。
- 哪个类别 headroom 最多：按类别 c 的 (1/C)·mean_i∈c gap_i（榜单单位），排序，并给每类里互斥标签的 headroom 分解（哪个切片占了类内 gap 的多数；占不到 50% 的部分归「无切片」）。

## 2. Q2 响应：RFS 随 capture 变了多少

对 S 里三个切片各自（重叠计数）与 `other`、`all` 给 `main` 对 O 的逐帧 ΔRFS（每帧每个 seed 一行，seed 三行各自对同一帧的 O；行的 bootstrap 按段聚类，所以段被抽到时三个 seed 一起进，CI 不含 seed 方差；seed 间散布另列），raw 与 ×1.06 各一份。再按**该切片自己的 capture 指示量**拆成：not→captured（O 没捕获 / main 捕获，「on」）、captured→not（「off」）、unchanged（两者同），各给行数、段数、ΔRFS 均值与 CI。另给互斥标签下各标签对总 Δ（榜单口径，Σ w_i Δ_i）的贡献；Σ 贡献 = 总 Δ 由计算核对。

读法（先写死）：
- **R2a「flip 带来了 RFS」**：一个切片的「on」行 ΔRFS 的 CI 下界 > 0，**并且**点估计 ≥ 0.3（RFS 是 0–10 量纲；0.3 相当于一帧的 3%，低于这个量我们不认为 capture 的翻转对一帧的 RFS 有实质效应）。CI 含 0 → 「无证据」；上界 < 0 → 「flip 伤 RFS」。
- **R2b「flip 相对 unchanged 的差」**：on 与 unchanged 的 ΔRFS 之差（同切片内）用同一重抽的配对 bootstrap 给 CI，读的是「flip 特有的」RFS 变化，排除适配模型整体漂移。
- **R2c「gains real but frames too few」**：S 的 on 行数 × 各切片 on 均值按 w_i 换成榜单单位，合成「on 翻转对榜单总分的预期贡献」E_on；若 R2a 成立而 E_on（以及 S 总贡献）< 0.07（= 现有 CI 半宽，低于它在 479 帧上看不见），则成立：增益可能真实，但 479 帧里 flip 的帧太少不足以显现。若 E_on ≥ 0.07 而总 Δ 仍是 +0.009 说明别的帧把它抵消了（读 off 行与 `other`）。

## 3. Q3 logged future 是不是 rater 偏好的目标

对 479 帧中有 logged future 的：rater 偏好轨迹 = 3 条 rater 轨迹里得分最高的一条（并列取第一条；报并列帧数）；与 logged future 的欧氏距离在 1、2、3、4 s（4 Hz 格点第 4、8、12、16 点）的中位数与 p90，另分纵向（沿 rater 偏好轨迹的 logged future 方向）/ 横向绝对差；「logged future 作 plan」的 RFS（`rater_feedback_score(future_xy, traj, scores, speed)`，5 s 全长；logged future 本身是 5 s 20 点）；它是否落在某 rater 的 trust region 内（`details` 的 inside 标志）；RFS ≤ 4.0（floor）的帧占比。按切片（重叠）报：RFS(logged)、RFS(O)、RFS(`main`)，以及 logged − O、`main` − logged 的配对 CI（段聚类）。

读法（先写死）：
- **R3a「logged future 是 RFS 好的目标」**：切片里 (logged − O) 的 CI 下界 > 0；**「RFS 中性」**：CI 含 0；**「RFS 差」**：上界 < 0。
- **R3b「logged future 不是 rater 偏好」成立**：S 上 (logged − O) 的 CI 下界 ≤ 0（即没有证据说模仿 logged future 能比 O 提高 RFS），或 S 上 logged future 的 RFS 均值低于 O 的。此时模仿 logged future 在机制上就不能指望抬 RFS（capture 与 RFS 目标不同），不必再问别的。
- 「`main` − logged」为正（适配后 plan 已比 logged future 的 RFS 还高）说明适配的 RFS 增益没有被 logged future 封顶。

## 4. Q4 RFS 的机制：纵向 / 横向分量

对每帧、每个模型（O 与 `main` 各 seed）按官方公式显式算出每位 rater、每个时刻（3 s、5 s）的 d_lng、d_lat 与阈值归一化后的 n_lng、n_lat 及 norm = max(n_lng, n_lat)，重建每帧 RFS（与 `rater_feedback_score` 逐位核对，差 < 1e-9，核对不过就停）。对 flip 行（on / off，切片按 S 的三个分开并合并）做反事实拆分：ΔRFS = Δ_lng-only + Δ_lat-only + 交互，其中 Δ_lng-only = S(d_lng 取 `main`、d_lat 取 O) − S(O)，Δ_lat-only = S(d_lng 取 O、d_lat 取 `main`) − S(O)，交互 = 总 Δ 减去前两项。另报：(i) trust region 状态的转移（inside: O → main 的 out→in、in→out、保持）；(ii) 在 O 与 `main` 各自得分最高的 rater 上，3 s、5 s 的 n_lng、n_lat 的均值与 binding 分量（norm 里取 max 的那个）是纵向还是横向的比例；(iii) 纵向 d_lng 的带符号均值（plan 相对 rater 轨迹在 rater 前进方向上是超前还是落后）。x×1.06 的情形同算一遍（只拆分，不汇报成两套表，放 CSV）。

读法（先写死）：
- **R4a「纵横向错位」**：on 行里 Δ_lng-only 与 Δ_lat-only **符号相反且各自 |均值| ≥ 0.1**，则成立（一个分量改善、另一个恶化、互相抵消）。
- **R4b「capture 翻转没让 plan 进 trust region」**：on 行里 out→in 的比例 < 50%，则成立（capture 用的是 4 s 位移的一半，trust region 是 3 s / 5 s 的 ±4 m / ±1 m，两者不是同一个口径，捕获可以发生在离 rater 轨迹仍很远时）。
- 以上两条的数字都以 S 内 on 行合并为主，切片分开的放 CSV。

## 5. Q5 决策输出

候选机制（先写死，各配一个「RFS 单位上界」）与判定，都由 §1–4 的数直接判定，不新增检验：

| 编号 | 机制 | 支持条件 | 上界（榜单单位） |
|:--|:--|:--|:--|
| H1 | imitated 切片 headroom 太小 | R1a 成立 | H_S |
| H2 | logged future 不是 rater 偏好 | R3b 成立 | Σ_S w_i·max(0, logged − O) |
| H3 | 增益真实但 flip 帧太少 | R2a 成立且 R2c 成立 | E_on |
| H4 | 纵向 / 横向错位 | R4a 成立 | Σ_on w_i·min(Δ_lng, Δ_lat) 的负部 |
| H5 | capture 口径与 trust region 错位 | R4b 成立 | Σ_on w_i·(n_out→in 之外的 on 行的 Δ) |

排序规则：成立的排在否定的前面；成立的之间按上界（榜单单位）从小到大排（上界最小的是对「RFS 没动」约束最紧的机制）；一个机制的上界 < 0.07 就意味着它单独足以让 +0.009 落在噪声里。同时给「headroom 最多的类别」与「要动它需要什么切片数据」：对 headroom 最多的前三个类别，给类内 gap ≥ 2 的帧数、它们的互斥标签分布、以及这些帧里有没有落在现有 audit 切片（若「无切片」占多数，则结论是：需要新定义这类帧的事件切片，而不是更多 start / stop / turn）。

## 6. 产物

- 脚本 `scripts/op_adapt_l_rfs_diagnosis.py`（子命令 `run`、`figure`）；在 box 上 `taskset -c 100-160` 运行（op-train 环境，`CUDA_VISIBLE_DEVICES=""`），总墙钟预计 < 5 min（479 帧 × 4 模型的 RFS 是毫秒级，bootstrap 2 000 次），所以不进 tmux。
- 小表 `research/results/op-adapt-L/rfs-diagnosis/*.csv`；图 `research/figs/op-adapt-L-rfs-diagnosis.png`（最多两张）；结果、偏离、已验证与推断都写在本文的「结果」节。
