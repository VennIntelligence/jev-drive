# HEAD1 预登记：冻结视觉 token 上的「转弯描述」head 能不能比 policy 自己的 plan 更准地给出过弯 heading，且误差不是 policy 误差的拷贝（2026-10-10，lane HEAD1）

本文在任何 head 读数之前提交并 push；偏离以带日期的补记写在受影响的读数之前。接第 204、207、240 条，邻居第 147、160、165、166、191、192、200、223 条。
代码 `experiments/corridor/scripts/head1_labels.py`（标签）、`head1_train.py`（head）、`head1_read.py`（读数与闸门）。
**地图、日志未来轨迹、日志实际走的车道序列只作训练标签与分析；推理时 head 只读冻结 Cinque 视觉 token（8 个 policy slot × 32 × 512，warp 协议）+ 20 维 ego 输入（含 command）。** 不用 WA-JEPA 的权重或特征；不解冻 encoder；不做闭环与全量训练。

## 问题
第 204 条：4 s heading 的 oracle 经 memory 通道值 +0.78，但同一份冻结特征上再预测一遍路径（QH）只值 +0.09；QH 的 head 不是 log 级 held-out、目标就是 policy 的目标。第 240 条：缺的是入弯在哪、指向哪（地图 centreline heading 在 plan 自己 4 s 弧长处 > 45° RMS 7.25° 对 SH30 的 11.04°），不是横向位置。第 223 / 191 条：同一份冻结 token 换一种标签形态、放到 navtrain 规模能读出此前读不出的量。
问：换成与速度无关的标签（heading 对弧长的曲线），冻结 token 上的 head (1) 是否比 policy 自己的 plan 更准，(2) 误差是否与 policy 的误差足够独立，(3) 若两者都成立，经 memory 通道喂回去动不动转弯桶。

## 不重测
grep 过 decisions（heading、弧长、入弯、thin head、QH、memory、fold）。不重测：第 204 条的 oracle 曲线与 QH（本线只重用其数与存档文件）；第 240 条的 N1–N4；第 200 / 197 条的几何 token；第 191 / 192 条的 selector 与 margin 头；第 160 / 145 条的解冻。与 QH 的差别：目标不是 8 个定时位姿而是弧长参数化的 heading 曲线（去掉速度），标签另有地图来源，训练是 navtrain 全量、log 级 fold，下游一切预测都是 held-out。

## 标签（`head1_labels.py`，CPU）
弧长网格 G：0, 2.5, …, 40 m，再加 45、50、60、70、80 m，共 22 点。值 = 相对 t0 ego heading 的 heading（rad）。
- **L（日志路径）**：t0 之后的日志帧（2 Hz，至多 60 帧 / 30 s，或走满 82 m）的后轴位姿，按日志路径自身的弧长重参数化（< 5 cm 的步长算静止，不计弧长），L(s) = 日志 yaw 在弧长 s 处的插值 − t0 yaw。日志没走到的网格点为 nan（不计 loss）。**训练时泄漏**：司机实际走的路径（形状、车道内摆位、变道）。时间轴被去掉：停车、跟慢车不改变标签。不需要地图。
- **M（地图车道序列）**：第 240 条的 lane graph Viterbi 匹配（代码原样，窗口取 L 的日志窗口，失败时退到 10 s、4 s 窗口），取日志 4 s 时所在的 lane run（第 240 条补记 B），其 centreline 从 ego 投影点起按 centreline 自身弧长取切向 heading。超过该 run 内最后一个匹配位姿的投影弧长之后为 nan（那之后只是「最直后继」，不是日志走的）。匹配失败的 token 全 nan。**训练时泄漏**：地图、t0 的定位与车道、覆盖范围内每个分叉处日志选的出口；不泄漏车道内横向位置与路径形状。
- **C（合并）**：同一个 head 两个输出通道同时回归 L 与 M（多任务）；C 的主输出是它的 L 通道。
推理时三者都不需要地图或未来。
**小规模核对（已做，无任何 head 读数）**：navtrain 10 个（左 / 右急弯、缓弯、静止、快速直行、慢速入弯）与 navtest 10 个（第 240 条的 10 个代表 token）画 BEV：车道序列、日志帧、centreline、两条曲线。20 / 20 匹配成功；标签坐标系与 token cache 的 `fut` 一致（最大差 2.3 cm）；L 与 M 形状一致，M 在 connector 交界处有折点、ego 已在弯中时 M(0) ≠ 0（−15° / +18°，即第 240 条的「日志走内侧」）。图在结果文档里给出。

## Head 与训练（`head1_train.py`）
- 结构：BODY1 S0 的 scene memory（ego MLP token + 256 个视觉 token，slot / 位置 embedding，3 层 transformer encoder，d 256）；22 个可学习的弧长 query 经 2 层 decoder（query 之间 self-attention + 对 memory 的 cross-attention）各出 2 个通道（L、M）。约 4 M 参数。**blind** = 同一结构不给视觉 token（只有 ego + command），是 floor。
- loss：Huber（δ = 0.1 rad）于有标签的网格点；arm L 只算 L 通道，arm M 只算 M 通道，arm C 与 blind 两个通道都算。采样概率 ∝ 1 + min(|日志 4 s heading 变化|, 90°) / 30°。8 000 步 × 256，AdamW lr 3e-4、wd 0.05、warmup 300、cosine。不调参：这一组超参在任何读数之前写定。
- **fold**：直接用第 191 条 SH30 cross-fit 的 5 个 log 级 fold（`navsim/op-parity-cf5f{j}-{train,dev}`，fold(log) = sha256("cf5|" + log) % 5）。fold j 的 head 只在 `cf5f{j}-train` 的 log 上训练，于是 head 与 fold policy `CF5f{j}-F-s0` 看过完全相同的 log，fold j 的 dev log 对两者都是 held-out。训练 log 里 sha256("head1val|" + log) % 10 == 0 的部分只用于选 step（验证 loss 最低的 step），不参与拟合。dev log 与 navtest 不参与拟合与选 step。
- stage 1 只训 fold 0：arm L、M、C 各 seed 0、1，blind（seed 0），学习曲线 arm C seed 0 取训练 log 的 12.5% / 25% / 50%（按 sha256("head1lc|" + log) 排序的嵌套子集；100% 即主 run）。共 10 个 run。其余 4 个 fold 只在 stage 2 需要（pilot 训练行的 OOF 预测），stage 1 过了才训。
- navtest 上的 head = fold 0 的单个模型（不做 fold ensemble：stage 2 里训练行读到的是「80% log 上训的单模型的 held-out 预测」，测试行要同分布；这正是 QH 的缺陷）。

## 读数（`head1_read.py`；全部按 log 的 cluster bootstrap，B 10 000，`default_rng(0)`，与 `jevdrive.stats` 同一抽样；统计量是 RMS / 相关 / R²，不是均值，故自己算统计量）
集合：(H) navtrain fold 0 的 dev log（head 与 `CF5f0-F-s0` 都 held-out，约 2 万 token）；(T) navtest 12 146 token（`SH30-F-s0/s1`、pilot `GH0-F-s0/s1` 的存档 plan，token-seed 合并）。桶 = |日志 4 s heading 变化|：全体 / > 20° / > 45°。
- **(a) heading 误差**。主量与第 240 条 N3 同口径：预测曲线在 **plan 自己的 4 s 弧长** s_p（第 207 条的 `Curve.sv[-1]`，非特权）处的值 − 日志 4 s heading，RMS。并列：policy 自己的 plan（4 s yaw − 日志 4 s yaw，SH30 > 45° 11.04°）、特权地图参照（M 标签在 s_p 处；navtest 上另列第 240 条 `feat.npz` 的 7.25°）、blind。另报：在日志自己的 4 s 弧长处（泄漏行程，只作分解）、固定弧长 5 / 10 / 15 / 20 / 30 / 40 m 处对 L 标签的误差（policy 的 plan 曲线同弧长处并列，只在 plan 与日志都覆盖到的 token 上）。
- **(b) 入弯点误差**。沿弧长平移拟合（第 240 条模型 A 的 heading 版）：δ = argmin_{|δ| ≤ 6 m, 步长 0.1} RMS_s[ψ_X(s) − ψ_L(s − δ)]，s 取网格上 ≤ min(s_p, 日志覆盖, 40 m) 的点（至少 3 点，ψ_L(u < 0) = 0），δ > 0 = 入弯晚。X = head 的 L 输出、policy 的 plan 曲线（同样采样到网格上）。报 > 45° 桶与 SH30「转不过去」token（第 207 / 240 条的定义与 replay）上的 δ 中位数、|δ| 均值及 head − policy 的配对差。
- **(c) 独立性**。e_h = head 在 s_p 处的误差，e_p = policy 的 4 s heading 误差（同 token-seed）。报 corr(e_h, e_p) 与 **R² = corr(e_p, e_p − e_h)²**：用 head 与 plan 的分歧 d = e_p − e_h（推理时可得）线性解释掉的 policy 误差方差份额，等价于把两者做最优线性融合后 plan heading 误差方差的降幅。
- **(d) floor**：blind 的 (a)，以及 head − blind 的配对差。
- **(e) 学习曲线**：arm C 在 12.5 / 25 / 50 / 100% 训练 log 下 (a) 的 > 45° 与全体 RMS（H 与 T）。

## 由第 204 条导出的独立性规则
第 204 条的独立噪声臂：给 memory 的纵向信号带独立噪声 σ_n，policy 自己的纵向误差 σ_p = 1.91 m，r = σ_n / σ_p = 0.39 / 0.79 / 1.57 时增益占 oracle（+1.38）的 86% / 62% / 22%。独立误差下 R² = 1 / (1 + r²) = 87% / 62% / 29%：**增益 ≈ oracle 增益 × R²**。
用同一条规则回算 QH（只用第 204 条已存的文件：thin head 的 navtest 预测与 `GH0-F-s0/s1` 的 plan，不是本线读数）：纵向 4 s 误差 corr 0.64，R² 0.084 → 预测 +1.38 × 0.084 = +0.12；heading R² 0.19 / 0.12（两个 seed）→ +0.78 × 0.15 = +0.12；实测 +0.09 [−0.11, +0.28]。按独立误差算（R² = 0.43）会预测 +0.59。规则同时复现独立噪声臂与 QH，所以用它。
stage 2 喂的是预测曲线积分出的路径形状（第 204 条 QS 的编码，oracle +0.93；4 s heading 单独 +0.78）。**预测增益 = 0.93 × R²**，R² 取 (T) 全体 token、对 pilot policy `GH0-F-s0/s1`（第 204 条曲线的那个 policy）、主量 (a) 的口径。

## 闸门（数字在读数之前写死）
head = 在 (H) 上按 (a) 的 > 45° RMS（s_p 取 `CF5f0-F-s0` 的 plan）选出的 arm（L / M / C 三者之一，seed 0；M 用 M 通道，L 与 C 用 L 通道）。选择只看 (H)，不看 navtest。对选出的 arm 的 seed 0、fold 0 模型在 (T) 上读三条线，全过才进 stage 2：
- **G1 准确**：> 45° 桶（1 517 token × 2 seed），RMS(head) − RMS(`SH30-F` 自己的 plan) 的 95% CI 上界 < 0。
- **G2 高于 floor**：同一桶，RMS(head) − RMS(blind) 的 95% CI 上界 < 0。
- **G3 独立**：0.93 × R² ≥ +0.5，即 R² ≥ 0.538（点估计；CI 照报）。0.93 × R² < +0.3（R² < 0.323）记为明确不过；之间记为不过但标「边缘」。
不过时报告是哪一条：G1 或 G2 不过 = 准确度（信息不在冻结 token 里，是 LoRA / 解冻臂的证据，本线不跑）；G1、G2 过而 G3 不过 = 独立性（head 的误差是 policy 误差的拷贝）。
另外两个 arm、seed 1、对 `GH0-F` 的 (a)、对 `SH30-F` 的 R²、(H) 上的全部读数照报，不进闸门。

## Stage 2（只在 stage 1 全过时；细节在开跑前以补记写定并 push）
第 204 条的配方：SH30 pilot（`navsim/op-parity-s234`，3 000 步 × 64），基线 H0 = `GH0-F-s0/s1` 复用。memory 输入 = head 的 L 输出积分成的路径（到 40 m，之后直线延长到 64 m）的距离场，无时间通道（QS 的编码）；训练行与 dev 行用各自 fold 的 OOF 预测（另训 fold 1–4），navtest 用 fold 0 模型。tokenizer 先带自己的 head 预训练再联合训练；臂：预测（2 seed）、打乱对照（2 seed）、屏蔽 memory 读数。读 EPDMS、转弯桶、> 45° 切内角与转不过去、直行桶（不得下降）、4 s 弧长比、dev ADE。线：增益 ≥ +0.5 且 CI 下界 > 0 记正；< +0.3 记负，且须先满足「通道被读」才算上限。

## 预算
stage 1：10 个 head run，每个约 10–15 分钟、约 34 GB VRAM（fold 的 token 全放在卡上），合计约 2 卡时；标签与读数是 CPU。stage 2 估计约 3–4 卡时（4 个 fold 的 head + 约 6 个 pilot 训练 + 2 个 tokenizer 预训练 + 导出）。合计远低于 25 卡时。全部 GPU job 走 pool（`cl submit` / `cl fanout`），与 S-DROP 共用，不选卡。

## 限定（开跑前已知）
- 一种 head 结构、一组超参、每 arm 2 seed；stage 1 的闸门只读 fold 0 的一个模型。
- (a) 的主量在 plan 自己的弧长处取值，head 的误差里含 policy 的行程误差，这会抬高 corr(e_h, e_p)；这是非特权口径的固有成分，日志弧长处的读数用来分开它。
- G3 的规则是从第 204 条两类臂（独立噪声、QH）归纳的线性换算，只有这两类校准点。
- L 标签把「另一条合法车道 / 另一种摆位」也算作误差（同第 207 条）；M 标签有 1–2% 的匹配失败，转弯 token 上更高。
- navtrain (H) 上的 policy 是 fold 模型 `CF5f0-F-s0`（1 seed），navtest 上是 `SH30-F` 2 seed。

## 补记 2026-10-10（HEAD1b）：越过 G3 的探索性第二轮（任何训练与读数之前提交并 push）

**性质。** stage 1 按登记规则在 G3 上不过（R² 0.513 对 0.538，第 243 条）。用户在看到 stage-1 结果之后（2026-10-10）决定越过这条闸门继续，并要求六张空闲卡并行跑、不再串行设闸。本补记以下全部内容因此是**探索性的**：闸门未过之后的第二次尝试，不是登记读数；所有文档、表、决策条目都照此标注。下面的线与数字仍在任何训练 / 读数之前写死，用途是约束本轮自己的读法，不改变其探索性。约束不变：不解冻 encoder / 不加 LoRA，不做全量训练与闭环，不用 WA-JEPA 权重或特征，推理时无特权输入，navtest 不参与任何拟合与选择。所有 GPU job 走 pool，N 个独立 run = N 个 job。

### A：把 head 训到收敛（arm L，输入不变）
stage-1 的 L 头在 8 000 步时 train loss 0.0010、val 0.0036，val 的最低点在最后一步：欠的不只是步数，还有泛化差距。候选（fold 0、seed 0，其余超参同 stage 1）：
`a16` 16 000 步；`a24` 24 000 步；`r16` 16 000 步 + dropout 0.2 + wd 0.1；`t16` 16 000 步 + 训练时每个 batch 随机只留一半视觉 token（256 → 128，推理用全部）；`rt24` 24 000 步 + 两者。
- 选择只看 fold 0 的 val log（训练 fold 内 sha256("head1val|" + log) % 10 == 0 的 89 个 log，trainer 已有的加权 Huber val loss，取各自最优 step）：val loss 最低者胜；差距 < 1% 时取更便宜的。stage-1 的 `L-f0-s0`（8 000 步，val 0.003612）作参照列，不参与选择。
- seed ensemble：选出的配置再训 seed 1；若 fold 0 val log 上两 seed 预测均值的 val loss 比 seed 0 低 ≥ 2%，最终 head = 两 seed 均值，否则 = seed 0。
- 最终配置训 fold 0–4（ensemble 成立则每 fold 两 seed）。每个 navtrain 行的预测来自没见过它所在 log 的那个 fold 的模型（OOF，覆盖全部 navtrain 行，断言检查）；navtest 用 fold 0 的模型（与训练行同分布：80% log 上训的模型的 held-out 预测）。
- 对最终 head 在 navtest 上把 G1 / G2 / G3 三条线各读**一次**，与 stage-1 的值并列（8.54° / −2.70；−6.59；R² 0.513）。blind floor 沿用 stage-1 的 blind（15.13°）。这是对收敛后的 head 的描述，不是新闸门；B、B2、C 不等它的结果。

### A2：拆开「目标形态 / 标签量与 fold / 结构」（只训 head，与 A 并行）
QH（第 204 条）= 8 个定时位姿目标 × 少量 log（非 log 级 held-out）× thin head（2 slot + MLP）；HEAD1 = 弧长 heading 曲线 × navtrain fold × 8 slot attention。补三个头，fold 0、stage-1 的 8 000 步与超参、seed 0 / 1：
`P-full` = HEAD1 结构 + QH 的目标与 loss（8 位姿）；`L-thin` = QH 的 thin head + HEAD1 的目标（22 点曲线）；`P-thin` = QH 的 thin head + QH 的目标。三者都用 HEAD1 的 fold 与标签量；连同已有的 stage-1 `L`（L-full）与 QH 原头（旧标签量）构成 2 × 2 + 1。
读数（held-out navtrain 与 navtest，stage-1 同口径）：> 45° 与全体的 heading 误差 RMS、对 `SH30-F` 的差（G1 口径）、对 `GH0-F` 的 R²（G3 口径）。位姿目标的头：由预测的 8 个位姿按 policy plan 同一套 feats 代码化成弧长 heading 曲线后在 plan 自己的 4 s 弧长处读，另列其定时 4 s yaw 的误差。读法：某因素的效应 = 沿该因素的两格之差（> 45° RMS 与 R²），两 seed 的范围并列；不设线。

### B：memory 通道 pilot（carrier 1）
A 的 fold 预测一齐就启动，不等 A 的 navtest 读数。配方 = 第 204 条：SH30 pilot（`navsim/op-parity-s234`，P2，hinge λ 30 / margin 0.5，3 000 步 × 64），基线 H0 = `GH0-F-s0/s1` 复用。
- memory 输入 `qp`：最终 head 的 L 曲线在 0–40 m 网格上积分成的折线，之后沿 40 m 处的 heading 直线延长到 64 m；QS 的距离场编码，无时间通道。训练行、dev 行、tokenizer 预训练行都用 OOF 预测，navtest 用 fold 0 的模型。
- tokenizer：先带自己的 thin plan head 在 `navsim/op-parity-geotok-train` 上以 `qp` 场预训练（第 204 条 `tok` 原样），再联合训练。
- 臂：`HP`（预测，seed 0 / 1）、`HPX`（同一 tokenizer 初值，读另一个 log 的行的场，seed 0 / 1）、`HP:noside`（测试时屏蔽 memory，两 seed）。
- fan-out 之前：一个 seed 的 smoke；恒等核对（memory 全屏蔽的 pilot 复现无 memory 基线，具体形式以实现时代码允许的最强形式为准，写进结果文档）。
- 读数（navtest 12 146 token，逐 token seed 均值，by-log 配对 bootstrap）：EPDMS；< 5 / 5–20 / 20–45 / > 45° 桶；> 45° 切内角与转不过去率；直行桶；4 s 弧长比；dev ADE；plan 自己在 4 s 弧长处的 heading 误差（前 = H0，后 = 臂）。
- 线（第 204 条惯例）：`HP − H0` ≥ +0.5 且 CI 下界 > 0 = 正；< +0.3 = 负，但只有「通道被读」成立才算（否则记「上限未量到」）；之间 = 部分。通道被读 = 第 204 条规则：每个 seed 的 dev 错配 ADE ≥ 1.05 × 自己的，或 `HP − HPX` 的 CI 下界 > 0。
- R² 换算的第一次直接检验：实测增益对 stage-1 的预测 +0.48，以及对收敛 head 重算的 0.93 × R²。判法：预测值落在实测 95% CI 内记「成立」，否则「不成立」，并给点估计之比。

### B2：辅助监督（carrier 2，与 B 并行）
理由：stage 1 不过的是独立性（head 与 policy 错在同一批弯上），这限制「预测再喂回」，不限制直接进 policy 的监督。做法：在 policy 的 plan 通路 hidden state 上加一个小头，回归 arm L 的标签（22 点弧长 heading 曲线，Huber δ 0.1 rad，只算有标签的点），与 SH30 pilot 配方联合训练，loss 权重 λ；推理不变，辅助头丢弃，不读任何 memory。
- λ ∈ {1, 3, 10}，另 λ = 0 作同一子集上的参照。选择只在训练 log 的验证部分上做：从 `s234-train` 中留出 sha256("head1val|" + log) % 10 == 0 的 log，四个选择 run（seed 0）只在其余 log 上训练，在留出 log 的行上读 plan 的 4 s heading 误差 RMS（> 20° 行）与 ADE；取 heading RMS 最低且 ADE 不比 λ = 0 差 2% 以上的 λ；若没有 λ 的 heading RMS 低于 λ = 0，仍取三者中最低者并照实写。
- 最终臂 `HA`：选出的 λ，完整的 `s234-train`（与 H0 相同的行），seed 0 / 1。读数与线同 B（没有「通道被读」一项；< +0.3 直接记负）。守护线：直行桶（< 5°）EPDMS 差的 CI 不整体在 0 以下；4 s 弧长比与 H0 之差的绝对值 ≤ 0.01；报 dev ADE。

### C：memory 信号放到 `P2H10S` pilot 配方上（与 B 并行，不再以 B 的结果为条件）
第 244 条：`P2H10S` 的 navhard 增益与 > 45° 的外移都在 hinge-only 离轨行上，所以 heading 信号必须在这些行上也在场。臂：`P2H10S` pilot 配方 + `qp` memory（离轨行也算 head 预测），seed 0 / 1，对同配方无 memory 的 pilot 基线（已有则复用，否则补训）。读数：B 的 navtest 各项，加第 244 条的外移量（> 45° 上超出日志路径 2 m 以上的 token 数 W2、4 s 横向外移对 base）。**前提**：离轨行上的 head 输入（冻结视觉 token + ego）已缓存或便宜可得；若离轨行与 on-log 行共用同一帧的视觉 token 而只是位姿扰动，head 的输入与 on-log 行相同，则照此说明并注明其含义；若需要未缓存的特征，只估算成本并报告，不启动。

### 预算与次序
估计（按 job 墙钟求和）：A 约 4.7 卡时（5 个选择 run + 至多 9 个 fold run，每个 14–20 分钟），A2 约 0.6，B 约 0.8，B2 约 0.9，C 约 1：合计约 9 卡时；上限 20 卡时，预计超出即停下报告。先小后大：每类一个 smoke，再 fan-out；派出后核对卡在干活。

### 限定（开跑前已知）
本轮是闸门未过之后的探索；候选配置只在 fold 0 的 89 个 val log 上选；navtest 的 head 仍是 fold 0 的模型；B / B2 / C 都是 pilot 规模、2 seed、开环；B2 的 λ 选择 run 的训练集比最终臂少约 10% 的 log；C 取决于离轨行的输入是否可得。

### 补记 2026-10-10（HEAD1b A / A2 的实现细则；任何 A / A2 训练与读数之前提交并 push）
补记正文没有写死、实现时必须定的几项（探索性，同上）：
- **A 的「更便宜」**：按名义计算量 = 步数 × 每步读的视觉 token 数（256 或 128）排序，不用实测墙钟（卡上并发会改变它）；实测训练时长并列。
- **A 的 ensemble 判据**：两 seed 在 fold 0 val 行上的预测（trainer 存下的 fp32 输出）在 CPU 上用同一加权 Huber 公式重算；seed 0 的值与均值的值用同一份重算结果比较（与 trainer 在 bf16 autocast 下记的 val loss 有末位差异，不混用）。
- **A 的非选中候选**不在 navtest 上读；它们只列 val loss 与 held-out navtrain（H）读数。
- **A2 的 thin head** = `path_req.py cmd_head` 的结构（slot 7 与 slot 0、LayerNorm + Linear(512, 64)、展平、拼 20 维 ego、2 × 1024 MLP、dropout 0.1），输出 22 × 2（L）或 8 × 3（P）；ego 用 HEAD1 trainer 的标准化输入，采样权重、lr 3e-4、wd 0.05、8 000 步、batch 256 同 stage 1（不是 QH 当时的 lr 1e-3 / wd 1e-2 / 均匀采样）。`P-full` = HEAD1 的 scene memory + 8 个 query × 3 输出。
- **位姿目标的 loss** = `path_req._imit`（xy Huber δ 1.0 m + 3 × yaw Huber δ 0.1 rad）；step 选择用同一 loss 在 fold 0 val 行上的加权值（与 L 头同一批 val 行、同一权重）。
- **位姿头的弧长读法**：8 个位姿经 `pt_swap.Curve` 成曲线，在 22 点网格上取 heading；policy plan 的 4 s 弧长超出该头自己的 4 s 弧长时，主读数沿末端 heading 直线延长（ext-line），并列 `pt_swap` 的常曲率延长（ext-arc）作敏感性，同时报需要延长的 token 比例。
- **QH 原头**只在 navtest 上读（它的拟合行 `geotok-train` 与 fold 0 的 held-out log 重叠，H 上不是 held-out）。

### 补记 2026-10-10（HEAD1b B2 实现说明，任何 B2 读数之前提交并 push；探索性）
- **挂载的 hidden state** = Cinque 图的 `select_4`（节点 638，512 维）：off-policy temporal summarizer 的 transformer 输出取最后一个 token，plan / lead / lead_prob / desire_state 四个头都从它解码（plan = final(select_4 + head_mlp.plan(LN(select_4)))）。它在 parity adapter 之后（adapter 把 bias 加在 summarizer 读的 9 帧上），可训练的 plan 通路除 plan 头自己的 MLP 外都在它上游；辅助梯度因此到达 adapter 与 summarizer，plan 的读出层不受辅助头直接约束。没有选 `add_54`（plan 头 MLP 之后）：那一层只差一个线性层就是 plan 输出，辅助目标会与 plan 读出争同一个 512 维。
- **辅助头** = LayerNorm → Linear(512, 512) → GELU → Linear(512, 22)（0.28 M），输出 rad；loss = 有标签点上的 Huber δ 0.1 的平均，只在 imitation 行（非 anchor 且有日志未来）；anchor 行不算（那里 adapter 输入被置零、plan 被蒸馏到原模型，标签对它没有意义）。辅助头自己一组参数，lr = `lr_new` 3e-4，单独 clip 1.0；policy 的 clip 仍是 base + adapter 的联合范数 1.0（λ > 0 时含辅助梯度）。采样仍是 pilot 配方的均匀行流（没有用 HEAD1 trainer 的转弯加权采样）。
- **λ = 0 的含义**：辅助头照样建、照样训，但读的是 detach 后的 `select_4`（权重 1）：policy 的梯度与权重与不带头的配方逐位相同，头是「原配方的 hidden state 里读得出多少曲线」的 probe。这是在补记之外多出的一个读数，不改 λ = 0 作为参照的定义。
- **恒等核对的形式**：同 seed、同行、短跑，`--aux-lam 0` 对不带任何新 flag 的原配方，逐 tensor 比较 `ckpt-final.pt`（期望逐位相同；不同则报最大差并说明）。
- **选择统计量的口径**：plan 的 4 s heading 误差 = plan 的第 8 个位姿的 yaw（unwrap）− 日志 4 s yaw（即 `head1_read.py` 里 policy 自己的误差的口径），在留出 log 的行中 |日志 4 s heading 变化| > 20° 的行上取 RMS；ADE = 8 个位姿的平均位移误差，全部留出行。若三个 λ 的 ADE 都超过 λ = 0 的 1.02 倍，取 heading RMS 最低者并照实写「ADE 条件无一满足」。
- **辅助头自身精度**：选择 run 在留出行上、最终臂在 `s234-dev` 行（token 级 dev，log 不是 held-out，只作参考）与 navtest 上读；navtest 上按 `head1_read.py` 的口径（曲线在 plan 自己的 4 s 弧长处的值 − 日志 4 s heading；另列日志 4 s 弧长处）与 stage-1 的 L 头并列。navtest 上的辅助头读数只是描述，不参与任何选择。

### 补记 2026-10-10（HEAD1b B / C 的实现细则；任何 B / C 训练与读数之前提交并 push）
补记正文没有写死、实现时必须定的几项（探索性，同上）。代码：`path_req.py`（kind `qp` / `qpx`、`ident`）、`bd4_train.py --mem-e2e`、`experiments/corridor/scripts/head1b_prof.py`、`head1_pilot_report.py`、`head1b_bc_chain.sh`。
- **`qp` 折线**：曲线取预测值原样（0 m 处不强制为 0）；0–40 m 的 17 个网格点，每个 2.5 m 步的方向 = 两端点 heading 的均值（中点法；曲率 0.1 /m 时 40 m 处的累积误差约 0.1 m），之后沿 40 m 处的 heading 直线延长 24 m 到 64 m；45–80 m 的 5 个点不用。场 = QS 的编码（到折线距离 clip 6 m / 3，时间通道为 0）。
- **`HPX`**：`qpx` = 第 197 条 `geo_x` 的固定错排（训练行与 navtest 各自的 perm 文件），与 QX 同；tokenizer 初值与 `HP` 同为 `tok/qp.pt`。
- **恒等核对的形式**：带 memory 的 adapter 多一个 side 分支，其参数在 ego MLP 与 query / decoder 之间创建，同一 seed 下 query / decoder 的初始化抽样与无 memory 的 adapter 不同，所以「memory 全屏蔽地训练一遍」不可能逐位复现 `GH0`。代码允许的最强形式是函数级：把 `GH0-F-s0` 的全部权重搬进 memory 臂的模型（side 分支保持自己的初始化），每一行的 memory 都屏蔽，输出必须等于 `GH0-F-s0` 自己的输出（`s234-dev` 的 408 行 + navtest 每 6 行取 1；线：plan 位置最大差 ≤ 1 mm，fp16 计算），同时不屏蔽时必须不同（屏蔽不是空操作）。`path_req.py ident`，fan-out 之前跑。训练后的屏蔽读数是 `HP:noside`（两 seed，bench）。
- **smoke**：输入放在 `$DATA_DIR/runs/corridor/head1b/smoke/prof`（navtrain = 日志路径标签 L 前向填充，特权，只测代码路径；navtest 与离轨行 = stage-1 的 `L-f0-p1000-s0` 头），tag `H1BSMOKE-*`，30 / 60 步，跑完删掉它的 bank 与 checkpoint；任何读数不碰。真实 job 用 `--when-exists $H1/final/DONE` 排在 head 之后，训练完核对 run summary 里的 profile 目录是 `final/prof`。
- **C 的前提核实**：`P2H10S` 的 hinge-only 离轨行（`ot1` / `yr1` / `bd4`）不与 on-log 行共用视觉 token：每一行是把 10 帧历史用平面引擎重投影到位移后的位姿、再过冻结 encoder 得到的自己的 token（`cache/<fam>_<data>@warp/front.npy`，8 slot × 32 × 512，已缓存），ego 输入也是在位移后的 t0 系里重新表达的历史位姿（速度 / 加速度 / command 同日志）。head 的两样输入都已缓存，成本是一次几秒的前向。
- **C 的决定：head 在离轨行自己的 token 与 ego 上算预测**，用没见过该行所在 log 的那个 fold 的模型（离轨行继承源 token 的 log），折线从位移后的自车位姿出发，不把该行的位姿偏移 (dy, dpsi) 施加到折线上。理由：(1) 补记正文写的是「离轨行也算 head 预测」；(2) 这正是推理时处在偏离状态（navhard stage 2、闭环）下会发生的事，head 只看得到当时的图像与 ego；(3) 另一种做法（on-log 行的预测按已知偏移搬到位移系）用了构造该行时才有的偏移量，等于在训练时给 memory 一个任何推理路径都拿不到的精度。含义与已知缺口：喂进去的曲线是相对位移后位姿的；head 只在 on-log 帧上训练过，标签在 0 m 处恒为 0，而日志路径在位移系里的起始 heading 是 −dpsi、横向偏 −dy，head 的参数化表达不了横向偏移，起始 heading 是否跟随位移要量。所以同时报告离轨行上的实际输入质量（`prof_ot/quality.json`）：曲线在日志 4 s 弧长处对位移系里日志 4 s heading 的误差（与同 token 的 on-log 行并列）、(离轨预测 − on-log 预测) 对 −dpsi 的回归斜率（1 = 完全按位移重表达，0 = 无视位移）。训练时 memory 在所有行（含 hinge-only 行）按 `MEM_DROP` 0.25 屏蔽；anchor 行的 bias 恒为 0。
- **C 的臂与基线**：臂 `P2H10S-HP-P-s{0,1}` = `P2H10S-P-s0` 的原命令（s2 + s3，3 000 步 × 128，其中 13 行 hinge-only，`--ho-w 3 --shape`）+ `--mem-e2e qp --mem-init tok/qp.pt`（B 的 tokenizer：只在 on-log 的 OOF 场上预训练）。基线 `P2H10S-P-s0` 复用，`P2H10S-P-s1` 按同一命令补训（现有 pilot 只有 seed 0）。C 不设打乱对照臂（补记正文没有列）：「通道被读」只看每个 seed 的 dev 错配 ADE，另报 `:noside`。
- **读数定义**（`head1_pilot_report.py`，B / B2 / C 共用）：分数差 = 逐 token seed 均值、by-log 配对 bootstrap（B 4 000，第 204 条的 `geo_oracle.paired`）；RMS 类读数 = token × seed 合并、`head1_read.CB`（B 10 000）。4 s 弧长比 = 臂的 plan 折线弧长均值 / 基线的均值（区间 = 配对差 / 基线均值），另列各自对日志弧长之比；B2 的守护线读 |该比 − 1| ≤ 0.01。plan 的 heading 误差 = plan 的 4 s heading − 日志 4 s heading（即 stage 1 的「plan 自己在 4 s 弧长处」）。喂入曲线的质量 = 曲线在日志 4 s 弧长处的值 − 日志 4 s heading 的 RMS、折线在该弧长处的位置误差、2.5–40 m 网格上对标签 L 的 RMS，pilot 训练行（OOF）对 navtest（fold 0 模型）。R² 换算用 stage-1 的 `feats.npz` 里 `GH0-F` 两 seed 的 plan 弧长，对喂入的 navtest 曲线重算。W2 与 4 s 横向外移 = 第 244 条的定义（`lib/route.py`，own plan 对日志路径，外侧为正，> 45° 的 1 517 个 token）。转弯失败率只对未屏蔽的 spec 读（replay `h1b`）。
- **恒等核对的修订（2026-10-10，第一次运行之后、任何 B / C 的真实训练与读数之前）**：按上面写的线（plan 位置最大差 ≤ 1 mm）第一次运行**不过**：屏蔽时 plan 位置最大差 dev 3.2 cm、navtest 抽样 5.5 cm（不屏蔽 9.7 / 10.9 m，屏蔽不是空操作）。这条线没有考虑 policy 通路的 fp16 计算：两种 adapter 的 bias 在 fp32 下只可能差在舍入位（attention 的求和顺序不同），但 bias 加到 fp16 的 hidden token 上后，个别元素的舍入翻转会被后面的 fp16 通路放大。改成两处读：(1) adapter 的 bias 本身（fp32，通道的直接输出）：屏蔽后与 `GH0` 的 bias 最大差 ≤ 1e-4 × bias 最大幅值；(2) plan：最大差与平均差都不超过参照的 3 倍，参照 = `GH0` 自己的 plan 在 bias 上加同样大小（实测 bias 差的 RMS）的随机扰动后的变化。两处都过记「恒等成立（到 fp16 舍入）」，原 1 mm 线的结果照实并列。
