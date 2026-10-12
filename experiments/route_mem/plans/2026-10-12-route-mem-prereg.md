# ROUTE-MEM 预登记：driver 记住自己收到过的 route，补上路口前 0–40 m 的车道几何（2026-10-12，任何训练与闭环读数之前写定）

代码 `experiments/route_mem/`：`lib/rmem.py`（记忆的构造与 adapter 支路，离线行与 serving 共用）、`scripts/rm_build.py`（训练行的记忆、served 检查）、
`scripts/rm_train.py`（训练 + 依赖读数）、`scripts/rm_chain.sh`（pool 链）；serving 改动在 `experiments/alpasim/lib/sh30_core.py` / `sh30_driver.py`，
只对带 `route_mem.json` 的 checkpoint 生效（其余 checkpoint 逐位不变）。box 输出 `$DATA_DIR/runs/route_mem/`。

## 问题

第 275 条：冻结 Cinque token 在直路上把车道放到 0.25 m（20 m 处），但在路口内与路口前 20 m 只有 1.1–1.7 m，我们所有出路零分都落在这一段。
第 279 条：driver **早先收到过**的 route 路点（日志路径投到车道中心，AlpaSim 每个决策发一批，从约 42 m 到 80 m）用 egomotion 带到当前帧，在 98 个出路零分窗口里把车道中心
放到 0.23–0.35 m；限制是覆盖：0.42 的窗口决策没有任何一批够到 20 m（批从 40 m 外开始，路口里车慢）。第 278 条：横向控制在所有决策上已在参照水平，缺的是纵向，所以本线只针对
路口里的出路 / corridor 零分，不追求横向精修。

**本线问的是**：把这份「收到过的 route 的记忆」作为 adapter 的一个输入支路（Cinque 冻结、指令保留），在 `YS2G` 的配方上训练，闭环里路口零分会不会在记忆够得着的地方下降、
够不着的地方不动？

## 与已有条目的区别（读过 decisions.md 127–134、189、204、264–279）

- **d127–134（op_route_ft，CARLA B2D）**：openpilot 上的 bearing / 多项式 route adapter，25 个路口只走对 4–7（线 13）；剩下的是「转得太晚」与低速蠕行，进弯状态与训练位姿不重合（d129、d134）。
  不同处：(1) 这里的 driver 在 nuPlan 上已经会转（公开 1491 均分 0.95，d269），缺口只是路口里的车道几何，不是「肯不肯转」；(2) 输入是稠密车道中心折线（40 点、2 m 间隔、从车后 6 m 到前方 72 m）
  带每点有效位，不是一个方位角或拟合多项式；(3) 训练行带闭环自己访问的状态（`on1_L18YS` 的 on-policy 行）与 AlpaSim 同一 RouteGenerator 算出的记忆，不是 CARLA 的 expert 位姿。
- **d189 的 AR 臂（AP2P-AR，pilot 排除）**：20 个 route 路点（只有当前这一批，从 40 m 外开始）进 adapter 的 ego 向量，ADE 每个 m 都更差、EPDMS 在噪声内。不同处：那一批不覆盖 0–40 m，
  正是我们缺的那段；这里是整个 episode 收到的批合并后带到当前帧，近场在车走过约 40 m 之后就有覆盖，且以 token 化的点集 + 掩码喂进一个独立的零初始化支路。
- **d204（path-req）**：经同一 memory 通道，只给日志未来路径的「形状」（QS）navtest +0.93，非特权的 thin head 预测路径 +0.09（同一冻结特征再预测没有新信息）。
  这里的 route 是 AlpaSim 在比赛里发给 driver 的合法输入（日志路径投到车道中心并延长），不是从冻结特征再预测的，所以属于「带来 policy 没有的信息」的一类；它没有速度 / 时序。
- **d279（local_map）**：只是离线几何读数，没有训练、没有闭环；本线是它的「下一步」。

## 设计

### 记忆（`lib/rmem.py`，离线行与 serving 同一份代码）

- **批**：episode 里到当前决策为止收到的每条 route 消息（rig 帧路点，去掉 NaN 填充），用消息时刻的 ego 位姿转到 local 帧，再转到当前后轴帧（第 279 条 G-carry 验证过的刚体变换）。
- **合并**：所有批都是同一条 route 折线上的采样（AlpaSim 每个 rollout 建一次、只会延长）；按时间顺序，最旧的一批之后接上每个新批里超出当前末端（沿末段方向 > 1 m）的点。
- **窗口**：ego 在合并折线上的投影弧长（在折线之前 / 之后时沿首段 / 末段外推），取弧长偏移 −6, −4, …, +72 m 共 40 点；折线覆盖处有效。每点
  [有效, x/20, y/20, cos 航向, sin 航向]（后轴帧，x 前 y 左），无效处全零，共 200 维。episode 刚开始只有当前一批（约 42 m 起），近处的点无效，由掩码表达。
  （任务书里「只保留起点在车后或 5 m 以内的段」在这里等价于：近场点只有在某一批的起点已经在车后时才有效；远场由当前批覆盖。）

### 支路（`rmem.RouteParityAdapter`）

原 P2 adapter 原样（权重、对前 20 维 ego 的 forward 都不变）+ `RouteHead`：2 层 transformer decoder（d = 128，4 头），32 个 slot query 读 [null, ego token, 有效 route 点 token]，
点 token = MLP(5 维) + 40 个位置嵌入，输出层 d → 512 **零初始化**，乘 `present`，与原 adapter 的偏置相加。参数约 0.53 M。所以从任何 P2 checkpoint 起步时第 0 步就是那个 checkpoint。

### 训练行的记忆（`rm_build.py`，在训练之前算好，已完成）

- **NAVSIM 行**（12 个 navtrain shard，103 291 行，含 dev 行）：把 token 当成一个跟着日志走的 18 决策 rollout 的第 k 个决策，k = crc32("rm|" + token) mod 18（不超过 token 之前的帧数）；
  scene 的 route = AlpaSim 的 `RouteGeneratorMap` 在从决策 0 开始的 9.5 s 日志轨迹上（20 帧 2 Hz 三次样条到 0.1 s，同 `ap2_route.py` 的做法，日志不够时取到日志末端），
  决策 j = 0..k 各在日志位姿处发一条消息，再用日志位姿带到 token 帧。
- **yaw-rate 行**（`yr1_*`）：取源 token 的记忆，按行的 `off`（dy, dψ）变到被扰动的帧（`ot_rows.off_hinge` 的约定）。
- **on-policy 行**（`on1_L18YS`，34 788 行，`YS-L18` s0 / s1 的 18 决策 train rollout）：每个 rollout 在 drive.jsonl 记录的 driver 锚点上依决策顺序重跑 AlpaSim 的
  `RouteGenerator.create`（MAP，40 m，after160 的 data source，第 266 条的做法），再按上面的合并 / 窗口算每个决策的记忆。
- 构建时的闸（读数前已得）：G-nav：NAVSIM 行可建 0.978–0.984；记忆覆盖处日志 4 s 未来位姿到合并折线的横距中位 0.24–0.26 m（12 个 shard）；+20 m 有覆盖的行 0.42–0.44。
  G-pool：重算的首个路点与 driver 收到的 `route0` 在 0.05 m 内 0.993（线 ≥ 0.99），锚点与池的行帧逐行一致（< 1e-3），+20 m 有覆盖 0.45。

### 训练（arm RM，标签 `YRM-L18-s{0,1}`）

`YS2G` 的配方原样（第 2b 轮：从 `YS-L18-s1`，池 `on1_L18YS` 的 13 行 / batch、S 两项、`--yr-mass 0.1`、lr × 0.1、`--warmup 1`、2 500 步、seed 0 / 1），加支路：
- 支路是第三个参数组，lr 3e-4（adapter 原始的 lr_new，**不乘 0.1**：它是新参数，温和 schedule 的理由只对已训练的权重成立），自己的 100 步线性 warm-up，同一 cosine；
- 每个 batch 的非 anchor 行以 p = 0.25 把记忆置空（pp_train 的 MEM_DROP，独立随机流 [seed, 0, 89]，原有的行流不变），使「没有记忆」在分布内（依赖读数需要）；
- anchor 行 `present` = 0，两个偏置都为 0（仍是出厂模型）。
对照：`YS2G-L18`（已存，同池、同 schedule、同起点，只差支路）与 `YS-L18`（seed 均值）。

## 闸（按顺序；任何一条不过就停，写原因，不上闭环）

- **G-init**：训练日志里第 0 步的 dev ADE 与 `YS-L18-s1` 的（0.5284 m，`YS2G` 训练记录的 dev0）相同到 1e-4。
- **G-serve**（训练前跑，零初始化 checkpoint `RMI-L18` = `YS-L18-s1` + 零支路，5 个 held scene，卡 1–5）：(a) 与同 5 个 scene 上 `YS-L18-s1` 的 plan，4 s 内各点差 ≤ 0.03 m 的决策 ≥ 0.99
  （docs/long-runs.md 的 fp16 规则）；(b) served 记忆（drive.jsonl `rmem`）对同一锚点上离线重建的记忆：有效掩码相同的决策 ≥ 0.99，且两边都有效的点的最大位置差 ≤ 0.05 m 的决策 ≥ 0.99。
- **G-dev**：每个 seed 的 dev ADE（记忆开）≤ `YS-L18` seed 均值 0.5286 + 0.02 = 0.5486 m。
- **G-rely（依赖）**：dev 行上 Δ = ADE(记忆置空) − ADE(记忆开)，逐行。路口行 = 车前 route 在 40 m 内进入路口（ev.Geo 的规则：两条车道多边形中心线夹角 > 30°），
  直路行 = 60 m 内无路口且 route 40 m 内转角 < 5°。每个 seed：路口行平均 Δ ≥ 0.01 m，且「路口行平均 Δ − 直路行平均 Δ」的按 log 聚类 bootstrap（B 10 000，rng 0）95% CI 下界 > 0。
  两个 seed 都不过 = 支路被忽略，停；只有一个过 = 照报，两个 seed 照常读（读数里写明）。

## 闭环读数

held 460（452 个有 rollout）、18 决策、每个 checkpoint 6 块（`rm_chain.sh read`，与 `ol_fast.sh read` 同一 driver 路径与 override，`--gpus 1,2,3,4,5 --pin-strict`，每卡两个 stack），
`ol_report.py --family Y --arm RM --ctl S2G --base-tags YS-L18-s0,YS-L18-s1 --ref-mean`：对 `YS2G` seed 均值（对照）与 `YS-L18` seed 均值配对（log 聚类 CI）。
机制量用 zeros_mech 的 `zm_read.py collect / report`（经 `ol_zm.py`，原样，只加 label；写在 `runs/onpol_long/zm/`），与 `ol_dirraster.py lines` 同一口径的早-内 / 晚-外 / 日志终点后直路 corridor 计数。
served 记忆的覆盖（每个决策的有效点数、最近有效偏移、+20 m 是否有效）由 `rm_build.py cover` 从读数的 drive.jsonl 汇总。

**98 窗口的分组（读数前从已存数据定死）**：第 272 / 275 条的 98 个出路 + corridor 零分窗口（`YR10m10-F-s0` 64、`YS-L18-s1` 34），落在 72 个 held scene 上。
local_map 的覆盖旗（`est_zero.parquet` 的 `L2_b_c20`：起点在横线之前、覆盖 20 m 的最新 route）在窗口决策里的占比 ≥ 0.5 记为「记忆够得着」：**38 个 scene 够得着，34 个够不着**
（窗口决策整体 0.42 没有覆盖，同第 279 条）。`YS2G` 在这些 scene 上的出路 ∪ corridor 零分：够得着 16 / 16（s0 / s1），够不着 17 / 16。

## 登记的预期 / 线（可证伪；seed 均值点估计，CI 一并报）

1. **够得着的 38 个 scene**：RM 的出路 ∪ corridor 零分 ≤ `YS2G` 的 2/3（16.0 → ≤ 10.7）。
2. **够不着的 34 个 scene**：RM 与 `YS2G` 之差的绝对值 ≤ 5.5（16.5 的 1/3；预期不动）。1 过而 2 不过（够不着处也大幅下降）说明增益不是来自记忆的几何，而是别的（例如支路学到的偏置）。
3. 零分总数 ≤ 41.5（`YS2G`）。
4. **GT distance ≤ 1.276 m（`YS2G`；选择量）**。
5. 横穿对向转弯的早-内零分不多于 `YS2G`（5.5）。
6. at-fault 零分不多于 `YS2G`（22.0）（ol_report 的硬闸）。
日志终点之后的直路 corridor 零分单列（评估器的人为量），不进判词。

判词：3、4、6 都满足 = 采用（成为比赛最后一个名额的候选之一，仍须过 1491 / 10 决策的公开护栏，只有这时才跑）；1 与 2 回答机制（「增益在记忆够得着的地方」）；
4 不满足 = 不采用；其余照实写哪几条过。

**对论文主张的含义（读数前写定）**：「自我不迁移：自己收到过什么的记忆」——如果 1 过、2 过（只在记忆够得着处降），说明把 driver 自己收到过的信息存下来、在自己的帧里带着走，
能补上冻结眼睛看不见的路口几何，支持「记忆是可迁移的那部分自我」；如果 G-rely 不过或 1 不过，说明在这个配方与数据量下，记忆进不了策略，主张在这条线上没有得到支持。

## 代价（实测 / 估计）

特征构建：12 个 shard 各 12 核约 1 分钟，on-policy 池 32 核约 1 分钟（已完成）；训练每个 seed 约 10–11 分钟（`YS2G` 实测 610 s，加支路与每 500 步多一次置空 dev 读数）；
读数每块约 8 分钟，2 × 6 块在卡 1–5 上约 20–25 分钟；报告与 zm collect 约 2 分钟。总计约 1 小时墙钟，约 2.5 卡时。

## 限定（开跑前已知）

- NAVSIM 行的记忆按「跟着日志走」造，闭环里 driver 偏离日志时 route 的投影会变；on-policy 行覆盖了这部分，但只有 13 行 / batch，且只带 S 两个 hinge（没有模仿目标）。
- 模仿信号来自 NAVSIM 行：route 就是日志路径投到车道中心，模仿目标是日志未来，所以支路可以学「沿 route 走」；在 AlpaSim 里 route 也来自日志路径（GT distance 的参照），这是比赛给的合法输入。
- 2 500 步温和续训对一个新支路可能太短；G-rely 就是为此设的停止点，不加步、不换 lr。
- 一种记忆编码（40 点、2 m）、一种支路结构、2 seed、452 scene。

## 补记 1（2026-10-12 19:15，训练已提交、任何闭环读数之前）：G-serve (a) 的口径

G-serve 的 5-scene 读数已出：(b) 过（served 记忆对离线重建，掩码相同 90 / 90 个决策，最大位置差 1.5e-4 m）。(a) 按原写法不过：全部 90 个决策里 plan 差 ≤ 0.03 m 的只有 0.17
（中位 0.16 m，最大 5.7 m）。逐决策看，输入完全相同的决策（k = 0 / 1：锚点与 ego 向量差为 0）plan 差 0.002–0.029 m，全部在 0.03 m 内；从 k = 2–3 起锚点开始分开（1e-4 → 1e-2 m），
差随决策放大。原因是 serving 默认的 `torch.compile`：带支路的模型是另一张图，fp16 的融合与舍入不同，闭环把首个决策 1 cm 级的差放大成轨迹分叉；支路本身输出恒为 0（训练里第 0 步 dev ADE
与 init 逐位相同，0.5284）。原写法把闭环分叉当成了接线错误，口径错了。
改为：(a') 同 5 个 scene、`SH30_COMPILE=0`（eager）下 `RMI-L18` 与 `YS-L18-s1` 全部决策的 plan 差 ≤ 0.03 m 的占比 ≥ 0.99（eager 下零支路逐位不改输出，期望逐位相同）；
compiled 下输入相同的决策（k ≤ 1）全部 ≤ 0.03 m 作为描述照报。(b) 不变。G-serve = (a') 与 (b) 都过。这条只决定能否上闭环，不碰任何登记线。
另记：compiled 的这种分叉在 `YRM` 对 `YS2G` 的比较里同样存在（不同 checkpoint 本来就不同），与以往各臂的读数同性质，不另加处理。
