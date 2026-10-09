# op_parity path-req 预登记：路径类 memory 信号要多准才值分，精度需求曲线（2026-10-09，任何本线分数读数之前写定）

lane P1，接 d200（[geo-e2e 预登记](2026-10-09-geo-e2e-prereg.md)）。决定编号 204。
代码 `scripts/path_req.py`（退化族、场、tokenizer 预训练、thin head 参照、闸门、报告、图）、`scripts/path_req_chain.sh`；`pp_train.py` 的 `--mem-e2e` 接受 `q<kind>`，加 `--mem-lr`。

## 问题

d200：真值几何经 adapter memory 通道被读了也只值 +0.13（上界 +0.3）；泄漏日志未来路径的距离场（JP）在学会读的那个 seed 上值 +1.24，并且是唯一动了 NC + TTC 与 > 45°「转不过去」的输入。
所以支路的 token 必须对 plan 有预测力。在建支路（plan.md L1）之前要知道它的输出要多准：把路径场 oracle 按可解释的方式退化，量每个退化版本经同一通道在 pilot 规模值多少，
得到「带性质 X 的预测信号值 Y」；再把今天不用特权输入的预测器（冻结 Cinque 特征上的 thin head、policy 自己的 plan）的误差放到同一条曲线上。
如果曲线说信号必须比 policy 自己的 plan 更准才值 +0.5，本线的回答就是负：把路径信息喂回去的支路帮不上忙。

**全部路径场臂都读日志未来，是泄漏标签的 oracle 探针，不是方法，不进任何可报告的推理路径。** 唯一例外是 QH（读 thin head 的预测路径，不含特权输入），它是参照点，也不是方法。
不用 WA-JEPA 的任何权重或特征。本线不训练前视支路，不做 L1。

## 与既有条目的差别

grep 过 `research/decisions.md`（路径场、memory、oracle、thin head、自己的 plan、专家路径）。相邻：d147、d160、d165、d166、d190–d195、d197、d200。

- **d200 的 JP 臂**：1 通道「到日志未来折线的距离场」，随机初始化联合训练，2 seed，1 个学会读。它回答「装置能不能吸收」，没有退化、没有 3 seed 的可靠读法，
  2-seed 均值混了一个读了的和一个没读的 seed。本线：先把「被读」做成 3 / 3，再沿四条退化轴量曲线。JP 的折线止于 4 s 位置，所以它其实带 4 s 路程（平均速度）；
  已打分的 `GEP-F-s1` 的 plan 在 navtest 上沿迹误差（4 s，RMS）1.11 m 对基线 1.91 m，横向 0.58 对 0.79 m：JP 的增益多半来自纵向信息。本线因此把时序与形状拆开量。
- **d197 / d200 的几何臂**：输入是场景几何，不是路径；不重测。
- **d160**：WA-Cf token（+0.95）只作参照行，不重跑，不用它的 token。
- **d166 / d165 / d192 / d147**：冻结特征读几何 / margin 的精度与其对 planner 的价值，是几何侧的量；本线量的是路径信号的精度需求，另给一个 thin head 读路径的非特权参照点。
- **d190–d195**：selector 在输出端从候选里挑；本线是输入端的路径信息经 memory 通道，机制不同，不重测。
- 不重测：任何几何 token、解冻 Cinque（d145）、route waypoint 进 adapter（d189 的 AR 臂，那是导航路线不是未来轨迹）。

## 设计

### 共同配方（= d197 / d200）

`pp_train --arm P2 --frames warp --host --hinge-lam 30 --hinge-margin 0.5`，`navsim/op-parity-s234`（25 415 train / 408 dev），3 000 步 × 64，warmup 100，d_frac 0.25，
memory 每行以 p = 0.25 屏蔽；行序、anchor 行、屏蔽行逐 seed 与 d197 / d200 各臂相同。基线 H0 = `GH0-F-s0` / `s1`（复用，不重训，88.48）。
tokenizer = `geo_oracle.build_net` 的卷积部分（约 1.1 M 参数，32 × 512 token），单独参数组、lr 3e-4、同一 schedule、梯度单独 clip 1.0（同 d200）。
训练后同一 job 出 navtest bank（`mem/ge_<tag>/lb_navtest.npy`），`jevdrive.bench` 原样读，不写新 runner。

### 场（全部臂 2 通道，128 × 96，0.5 m，x −8..56 m，y ±24 m）

折线 = 原点 + 若干顶点。通道 0：格心到折线的距离，clip 6 m / 3（同 JP）。通道 1：最近折线点的到达时间（占 4 s 的比例）× max(0, 1 − 距离 / 3 m)；不带时序的 kind 这一通道全零。
未退化的 oracle **QF** = 原点 + 日志 8 个位姿（0.5 … 4 s），两通道齐全，即完整的未来轨迹（路径 + 时序）。

### 退化族

误差是逐行固定的（同一行在每个 epoch、每个 seed 读到同一份退化输入，navtest 行另抽一份），像一个确定的预测器，不是每步重抽的增广。记 ε ~ N(0, σ²) 逐行独立。

| 轴 | 臂（seed） | 定义 |
|:--|:--|:--|
| 横向误差 | QN025 / 050 / 100 / 200（各 0、1） | 第 k 个位姿沿日志 heading 的法向平移 ε · (t_k / 4 s)^1.5；σ = 4 s 处横向误差的标准差 0.25 / 0.5 / 1.0 / 2.0 m（相当于 8 点 ADE 0.09 / 0.19 / 0.37 / 0.74 m） |
| 纵向误差 | QL075 / 150 / 300（各 0、1） | 位姿沿日志路径滑动，弧长 s_k → max(0, s_k + ε · (t_k / 4 s)^1.5) 再取单调（超出日志终点沿末端 heading 直线外推）；σ = 0.75 / 1.5 / 3.0 m（ADE 0.27 / 0.54 / 1.08 m）。形状不变，时序变 |
| horizon 截断 | QT2、QT1（各 0、1） | 只保留前 2 s / 1 s 的轨迹（4 / 2 个位姿，时序保留）；4 s = QF |
| 内容 | QS（0、1） | 只有路径形状：日志路径沿末端 heading 直线延长到 64 m，时间通道全零。没有速度、没有路程 |
| | QV（0、1） | 只有速度曲线：8 个弧长摆在正前方直线上，带时间通道。没有横向几何 |
| | QCH（0、1） | 只有 4 s 的 heading：一段 20 m 的等曲率弧，总转角 = 日志 4 s 的 heading 变化；无速度、无形状 |
| | QC7（0、1） | 只有出口类别：同样的弧，转角取带符号转角桶（直行、±5–20°、±20–45°、> 45°，共 7 类）的桶中心 0 / 12.5 / 32.5 / 70° |
| 对照 | QX（0、1） | QF，但每行读别的 log 的场（d197 `geo_x` 的固定置换，navtest 同样置换）；tokenizer 从 QF 的预训练权重起步 |
| 非特权参照 | QH（0、1） | QF 的场，但位姿换成 thin head 的预测（见下）；tokenizer 从 QF 的预训练权重起步。不泄漏标签 |

指数 1.5 与档位的来历：已打分的基线 `GH0-F-s0/s1` 在 navtest 上的 plan 误差（这是已有读数的再分解，不是本线臂的分数）横向 RMS 随时间按 t^1.6、纵向按 t^1.45 增长；
4 s 处横向 RMS 0.78 m（稳健 σ = 1.4826 × 中位绝对误差 = 0.33 m），纵向 RMS 1.91 m（稳健 1.50 m）；full 的 SH30 是 0.67 / 0.30 与 1.72 / 1.36 m。
两条噪声轴的档位把 policy 自己的误差夹在中间（横向 0.25–2.0 m，纵向 0.75–3.0 m）。

**没有单独的「偏置」臂，理由**：逐行相同的确定性偏置（整体平移、固定比例的转弯不足）是可逆变换，联合训练的读取端会把它学掉，量到的不是退化；
对一个训练过的读取端，有意义的横向退化只有逐行随机的那一部分，即上面的 ε。训练时无偏、测试时才有偏置的部署漂移是另一个问题，本线不测。
**噪声与 policy 自身误差独立，这一点是偏乐观的**：真实支路与 policy 共用冻结特征，误差与 policy 的误差相关，同样大小的误差带来的新信息更少。
所以曲线给的是「误差为 X 的信号最多值 Y」；QH 这一对臂直接量一个真实的、误差相关的预测器，不靠换算。

### 可靠性：先把抽签修掉（stage 0）

候选从便宜到贵，全部在未退化的 QF 上、3 seed（0、1、2）：

- **QFC**：随机初始化联合训练（d200 JP 的做法）；
- **QFL**：随机初始化，tokenizer lr 1e-3（× 3.3）；
- **QF**：tokenizer 先带自己的 thin head 预训练（`path_req.py tok`：`navsim/op-parity-geotok-train` 的 76 k 行，pilot 的 shard 之外，只有模仿项，3 000 步 × 256，lr 1e-3），再联合训练。
  这就是 d200 JW 的做法换到路径场上；不直接借 d200 的几何 tokenizer（通道数与输入都不同）。

**逐 seed 的「读了」（dev 408 行，训练 DONE 里的三个读数）**：dev ADE ≤ 0.8 × 基线的 0.6305 m（= 0.504 m，d200 的吸收阈值），且换成别的 log 的场时 dev ADE ≥ 1.5 × 自己的。
**dev 闸门**：QF 3 / 3 个 seed 读了。不满足就停，修好再继续（链退出，不往下跑）。QFC、QFL 的 3-seed 读取率照报，用来回答「最便宜的可靠做法」；它们不打 navtest 分。
**曲线各臂一律用 QF 的做法**：每个退化臂的 tokenizer 在自己的退化输入上带自己的 head 预训练（seed 0 一次，两个训练 seed 共用），再联合训练；即使 QFC / QFL 也 3 / 3，曲线也不改用它们（3 seed 证明不了抽签消失）。
**navtest 闸门**（stage 0 打分后）：QF 每个 seed 对 H0 的 EPDMS 差 CI 整体高于 0；QF − QX 的 CI 高于 0；QF 屏蔽 memory − QF 的 CI 低于 0。三条都满足才读任何退化臂的分数。
**提前停**：QF − H0 的 CI 上界 < +0.5 → 未退化的 oracle 自己就不到线，曲线无从谈起，停，回答为负。

### 读数（同 d197 / d200）

navtest 12 146 token，`python -m jevdrive.bench`，逐 token 取 seed 均值，按 log 聚类的配对 bootstrap（`jevdrive.stats.paired`，B 4 000，95% CI），全部臂对 H0（2 seed）。

- 主量：EPDMS 增益（臂 − H0）。分桶 EPDMS（< 5°、5–20°、20–45°、> 45°）；DAC / NC / TTC / EP / LK 子分；> 20° 与 > 45° DAC 失败率；NC + TTC 失败率；
  > 45° 切内角率与转不过去率（four_dirs 回放）。
- **每个臂的「通道被读」三项**：臂 − QX 的 EPDMS；屏蔽 memory − 臂的 EPDMS（QF 三个 seed 都打，退化臂打 seed 0；dev 上的屏蔽读数每个 seed 都有）；dev ADE 的 on / 屏蔽 / 错配。
  退化臂记为「读了」iff 每个 seed 的 dev ADE 错配 ≥ 1.05 × on（d200 没读的臂是 1.005–1.014，读了的 JW 是 1.06 / 1.09），或臂 − QX 的 CI 高于 0。
  **没读的臂不进曲线**：它的点只说明这次没读出来，报告里单列并标出，判读时把它当作「未量到」而不是「值 0」，除非它的内容本来就在 QX 的水平（σ 很大的档）。
- 每个噪声臂在 navtest 上实际加进去的误差（4 s 横向 / 纵向 RMS、ADE）、tokenizer 预训练 head 的 dev ADE（自己的场 / 别的 log 的场）。
- 各臂 plan 自身对日志的误差分解（看读了之后 plan 哪个方向变准）。
- 参照行：d200 的 JP（+1.24 / +0.01）、JW（+0.13）、d160 的 WA-Cf（+0.95），不重跑。

### 非特权预测器的位置

在 navtest（全是训练没见过的 log）上量，同一套统计量（ADE、≤ 1 s / ≤ 2 s 的 ADE、FDE、4 s 横向与纵向的 RMS 与稳健 σ、4 s heading 误差、7 类出口准确率）：

- **policy 自己的 plan**：pilot 的 H0（`GH0-F-s0/s1`，dev ADE 0.63 m）与 full 的 SH30（`SH30-F-s0/s1`），用 bench 已存的 plan 文件。
- **冻结 Cinque 特征上的 thin head**（`path_req.py head`）：8 个缓存 policy slot 里最新与最旧两个（各 32 × 512，W 协议）逐 token 线性降到 64 维，加 P2 的 ego 输入（20 维），
  2 层 1024 的 MLP 出 8 个位姿；在 `navsim/op-parity-geotok-train` 上训（6 000 步 × 256），所以它在 pilot 训练行、dev、navtest 上都是 held-out 的预测。
- **QH**：把这个 thin head 的预测路径当场喂进通道（训练行与 navtest 行都是 held-out 预测，train / test 同分布）。它是「把路径预测喂回去」不带标签泄漏的直接读数。
  policy 自己的 plan 不做同样的直接臂：pilot 训练行上的 H0 plan 是 in-sample 的，要 cross-fit 才干净，超出本线预算；它的位置只按误差放。

## 判读规则（写死）

1. **每条轴的容许档**：按退化由轻到重排，对 +0.5 与 +0.3 两条线各报两个数：增益的 **CI 下界** 仍 ≥ 线的最后一档，和第一次 < 线的那一档（不插值；点估计的插值交点只作描述）。
   噪声轴的单位是 4 s 处误差的标准差（米），horizon 轴是保留的秒数，内容轴逐臂报是否过线。
2. **值不值得建**：支路「值得建」iff 在横向与纵向两条噪声轴上，非特权预测器在 held-out log 上的 4 s 误差（主统计量 RMS，稳健 σ 作敏感性一并报）都不大于该轴 +0.5 线的容许档。
   误差落在「最后过线档」与「第一个不过线档」之间记「未定（档位之间）」；RMS 与稳健 σ 给出不同结论时照实写两个，按 RMS 判。两条轴是逐条加噪的，都过只是必要条件。
3. **是否要比 policy 自己的 plan 更准**：若某条轴 +0.5 线的第一个不过线档 ≤ H0 自己在该轴的误差（RMS），写明「信号必须比 policy 自己的 plan 更准」，本线回答为负。
4. **QH 的直接读数**优先于按误差换算的位置：QH − H0 ≥ +0.5 且 CI 下界 > 0（L1 的 pilot 线）→ 现成的非特权路径预测就够，支路值得建；
   QH − H0 < +0.3 且与 QX 分不开 → 这个精度的路径预测喂回去没有用。两者之间照实报。
5. 内容轴与 horizon 轴回答支路该预测什么：哪一类内容（形状 / 速度曲线 / heading / 出口类别）单独过 +0.5、+0.3，增益需要多长的 horizon。
6. 没读的臂按「未量到」处理（见上）。第一次读数就是全部 seed；不加训、不换档、不放大。

## 步骤与启动

一条自推进的链（`path_req_chain.sh`，tmux `jev:path-req`，状态 `$DATA_DIR/runs/op_parity/path_req/chain/{STATUS, DONE, ERROR}`），所有 job 走 pool，不选卡，每个训练一个 job。

0. 启动前：`path_req.py selftest`（CPU：各 kind 的折线 / 场有限，噪声臂在 navtest 上的实际误差与名义值相符）；pool 上 30 步 × 64 的训练 smoke（qf、ql1.5、qs）、
   tokenizer 与 thin head 的 smoke，量真实 VRAM。smoke 不产生 navtest 分数。
1. **stage 0**：QF 的 tokenizer 预训练 + 11 个训练（QF / QFC / QFL × 3，QX × 2）→ `pp_full_check.py train` → dev 闸门 → navtest（QF × 3、QF 屏蔽 × 3、QX × 2）→ 回放 → navtest 闸门。
2. **stage 1（一条轴：横向误差）**：dev 闸门过后即提交 4 个预训练 + 8 个训练（不读分数，stage 0 打分时卡是空的）；navtest 闸门过后才打分（8 + 4 个屏蔽读数）→ 回放 → 阶段报告。
3. **stage 2（其余）**：thin head；纵向 3 档、horizon 2 档、内容 4 臂、QH，共 20 个训练 + 9 个预训练 → navtest（20 + 10）→ 回放 → 报告。
4. 每阶段打分后删掉本线自己写的 navtest bank（`mem/ge_Q*`，每个 380 MB）；别的不删。
5. 分级启动检查单（每阶段，看分数之前）：训练 loss 有限、`pp_full_check`（dev ADE ≤ 1.2 m、inputs-off drift ≤ 0.30 m）、bank 无 NaN、QX 的置换无一行来自本 log。

## 预算

42 个训练（各约 8 分钟，3 卡并行、每卡 3 个）+ 14 个预训练 + thin head + 50 次 plan 导出：按 job 墙钟求和约 7 卡时，按卡占用（卡数 × 墙钟）约 3 卡时；上限 8 卡时。
CPU：50 次 navtest 打分与 3 次回放，经 pool 排队，约 1.5–2 小时墙钟；不超 pool 的 CPU 预算。

## 限定（开跑前已知）

- pilot 规模（25 k token、3 000 步）、一种 tokenizer 结构与场编码、退化臂 2 seed；曲线针对这条 memory 通道与 pilot 配方。
- 噪声与 policy 自身误差独立、高斯、单参数形状（t^1.5）；真实预测器的误差重尾且与 policy 相关。曲线是给定误差大小时价值的上界，QH 是对这一点的直接检查。
- 逐轴退化，横向与纵向误差没有同时加。
- QS 的「无时序」不完全：慢速转弯时日志路径在 4 s 处中断、之后是直线延长，弯在哪里停住仍带一点路程信息。
- 基线已有 λ 30 hinge；量的是路径信息作输入的增量。
- QF 与 d200 的 JP 编码不同（多一个时间通道、tokenizer 预训练起步），两者的数不直接可比；JP 只作参照行。
- thin head 只是一种非特权预测器（2 个 slot、MLP）；更强的头或支路的误差会更小，这正是 L1 要回答的，本线只给今天够得到的参照点。

## 开跑前的声明（2026-10-09，任何分数读数之前）

- `selftest` 通过：18 个臂的折线与场有限；噪声臂在 navtest 上实际加进去的 4 s 误差 RMS 为横向 0.249 / 0.498 / 0.997 / 1.993 m（纵向分量 0），
  纵向 0.735 / 1.465 / 2.906 m（静止与慢速行在 0 处截断，略低于名义值；带出的横向分量 0.02 / 0.05 / 0.15 m，来自沿弯道滑动）。
- pool 上的 smoke 通过（不产生 navtest 分数，bank 已删）：训练 30 步 × 64（qf、ql1.5、qs）loss 有限，dev 诊断与 navtest bank 导出跑通，bank (12 146, 32, 512)，token RMS 0.48，
  30 步时 on / 屏蔽 / 错配的 dev ADE 相同（1.456 m，零初始化输出层）；tokenizer 预训练 16.5 it/s、峰值 1.2 GB；thin head 36.8 it/s。
- VRAM 实测：训练 smoke 含 dev eval 与 navtest 导出，pool 记录的峰值 18.5 GB（d200 的路径场臂 3 000 步 18.2 GB）；训练 job 按 24 GB 申报，4 核、RAM 40 GB。
  今天 AlpaSim lane 占着 pool 的大半 CPU 预算（85 / 105 核），本线的训练 CPU 申报从 d200 的 6 核降到 4 核（两个取数线程加主线程），排队由 pool 决定。
