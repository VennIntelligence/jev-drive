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
