# LON-ANCHOR 预登记：按站位对齐的人速度剖面，作为 S 行上的双边纵向弧长项（2026-10-20，写于任何标签 / 训练输出之前）

## 为什么

第 404 条（INERTIA）：plan 的纵向剖面跟着自车当前速度走（停车 / 转弯窗的响应是空路的 0.89 / 0.92，约 90 % 走帧通道，RB0 里就有）；
PAI 上模型以人的 1.41 倍速度到达停止线、从不 plan 停车，入弯过快。此前的单边上限都失败：STOPPT（第 386 条，触发 0.43 %，聚焦剂量只让整体变慢）、
TURN-CAP（第 396 条，好窗误触发）、JT 的双边弧长项（第 352 / 353 条，按**同一日志时刻**的人速度、2 s 从自车速度过渡，学成「开快点」，前撞 12 → 51）、
JOINT-DAGGER（第 321 条，规则路径目标打掉「别撞」）。

## 设计（用户 10-20 改定：信号尽量简单，手工细节教出 hack）

原派活里的 STOP / TURN 窗口、可行过渡、只设上限的变体**全部去掉**，换成一个项、所有 S 行：

- **标签**（S 行 = 模型自己的状态）：自车在人日志路径上的站位 s_e（nuPlan：on_rows 的 `s_log(t) + lead`；PAI：onp_rows 的 `s_ego`）；
  t_h = 人第一次到达 s_e 的日志时刻；目标弧长 s_t,j = S_h(t_h + 0.5 j s) − s_e，j = 1..8（plan 的 8 个时间点）。即「人从自车现在所在的位置开始的自己的剖面」，
  不是 JT 那样按同一时刻对齐，也没有 2 s 过渡。实现：`lib/nr1_anchor.py target`。
- **项**：d_j = plan 第 j 个 pose 的累计折线弧长 − s_t,j，smooth L1（beta 0.5）对 8 个 pose 求和（= `nr1_lon.terms`，JT 弧长项的同一函数，只换目标）；
  双边；只沿各 pose 自身航向回传梯度（`nr1_lon.lon_only`），路径形状不动。经 nr1_lat 进训练器：`lon_anchor` = 10 × W × term，再乘 ho_w、对池行求和 / n_imit。
  开关 `rm_train --on-anchor W`，默认 0：不加载、不加任何运算。
- **唯一排除**：前车。前车 = 中心落在人路径（从 s_e 起 2–60 m）1.8 m 内的物体（第 362 条的横向线），间距 = 其最近路径点的弧长。
  自车此刻的前车间距 gap_e 比人在 t_h 时的前车间距 gap_h 近 0.5 m 以上 → 此行不进项（不把车往前车推）。人的前车量不到（nuPlan：t_h 附近 0.6 s 内无日志帧；
  PAI：同场景各行的盒子里没有 t_h ± 0.15 s 的时刻）而自车有前车 → 也不进项（`lead_unknown`，按「人的前车更远」的反面处理，保守）。自车 2–60 m 内无前车 → 不排除。
  nuPlan 人的前车取日志标注（`on_rows._ann`，两帧间线性插值），自车的取池 `agent.npz` 第 0 步；PAI 两者都取 rollout 的回放物体（同一时钟，非反应式回放）。
- **标签无定义的行**（不是设计上的排除，只是算不出来）：自车状态不在日志时间范围内或在人起点之后 1 m 以外（`no_path`）、人从未到达 s_e（`no_reach`）、
  日志不覆盖 t_h + 4 s（`log_end`；不外推，避免在日志末尾造出假停车）。
- 不加：窗口、可行性整形、只设上限的变体、对横向偏离的门限（偏离很远的行照样进项；横向偏离分布报告）。

**已知风险（写在读数之前）**：站位对齐忽略时钟。自车晚到人等过红灯的位置时，目标说「站住」，而模拟里此刻的灯（回放帧）可能已绿；早到时反之。
这正是 STAND-R（第 399 / 400 条）的起步泄漏类型。闸 G-report 报告「目标在 4 s 末站住、而人在同一时钟在动（≥ 2 m/s）」的行数（clock_conflict），不设线。
前车排除只挡住「自车比人离前车近」，「人当时在车队里、自车现在前面空」的行照样进项（目标慢）。

## 臂（同一项的三个剂量 + 同步对照，1 seed）

- **配方**：NTV0-L18-s0 的命令（`jt.sh train` 的 DRY 打印，ONK=32、ARCROWS=brake：JT00S 抽样、on-k 32、无 STAND / FREE-M / 弧长项），
  池 = 第 402 条的合池 `on1_JTCP`（on1_JT 五部分 + 569 个 curated-train PAI 场景的 84 042 行 NTV0 自身 S 行），按 1K pilot 惯例：
  `--init NTV0-L18-s0`、`--steps 1000`、`--warmup 50`（= 第 402 条 pilot 的命令，去掉 `--on-stoppt`）。
- **抽样 LONP**：JT00S 的 nuPlan S 份额减半，另一半均匀给 cp 行（份额 0.5，= 第 402 条 pilot）；cp 行中 held 15 %（crc32(scene) % 100 < 15）、
  PAI 281、practice-pai（= val160）以及 `experiments/pai_fix/scenes/` 下任何文件里出现的场景 uuid **一律权重 0**（`lon_anchor.py weights --require-fix`：
  目录不存在就不建抽样、不训练）。四个训练用同一个抽样文件。
- 臂：**LONC1K-L18-s0**（对照，无项）、**LONAH1K-L18-s0**（0.5 × W1）、**LONA1K-L18-s0**（W1）、**LONAD1K-L18-s0**（2 × W1）；命令只差 `--tag` 与 `--on-anchor`（`lon_anchor.sh dry` 打印并 diff）。
- **剂量规则（JT / JTB 的 0.1 梯度规则）**：校准跑 LONACAL（LONA 的命令、W_PROBE = 0.004、100 步、`--diag-every 10`），读训练器 `--diag` 的
  grad/imit 与 grad/lon_anchor（主干梯度范数，乘过权重），W1 = W_PROBE × 0.1 / median(g_anchor / g_imit)，两位有效数字（`lon_anchor.py calib`）。
  即 ×1 剂量时项对主干的梯度约为模仿损失的 10 %。校准跑的权重文件删除，不作 pilot。

## 闸（运行前写定）

- **G1（人路径当 plan → 项 ≈ 0）**：每个有标签的行，plan = 人日志路径上 s_e + s_t,j 处的点（行坐标系；nuPlan `gpath`、PAI `path`，1 m 网格插值，超出端点沿端点切向直线外推），
  用训练器的同一函数计算。行「激活」= 8 个 pose 里 max |d_j| > 0.5 m。**线：激活份额 ≤ 1 %**（全部有标签行；nuPlan / PAI 分开报告）。不过 → 停，不训练。
  （自车横向偏离使第一段弦长略长、弯里弦短于弧，这两项是 G1 真正检验的东西。）
- **G4（关掉 → 逐位相同）**：对照的命令 60 步 eager（`JEV_RM_EAGER=1`），旧代码（本次改动前的提交）对新代码、不带 `--on-anchor`，`df.py bitcmp`
  （每个标量、每个权重、dev ADE）。不相同 → 停。
- **只报告、不设线**（`results/lon_anchor/gate.md`）：存档 plan（nuPlan 行的 served plan、cp 行 NTV0 的 `poses_model`）对目标的激活份额、偏快 / 偏慢份额、
  4 s 弧差分布（按来源与 cp 类：stop_nolead / turn_sharp / launch / bend）；目标在 4 s 末站住的行数与其中 clock_conflict 行数；t_h − t 的分布（自车领先 / 落后人的时间）；
  标签结果计数（ok / lead / lead_unknown / no_path / no_reach / log_end）。

## 读数（每臂对对照；训练后在队列里自动跑）

- `inertia.py cf / report`（**代码不改**）在 P160 行上，tags = NTV0-L18-s0（编码帧，与第 404 条同）、LONC1K、LONAH1K、LONA1K、LONAD1K、RSG-L18-s0（G0 的 gate-tag），
  输出 `results/inertia/report_lon_anchor.{md,json}`。
- **通过（每臂）**：STOP 比（G_B STOP / NS）≤ **0.6** 且 TURN 比（G_B TURN / OPEN）≤ **0.6**，两者都比对照低至少 **0.15**；且 OPEN 的 v24 斜率（B）≥ **0.7**。
  说明：OPEN 斜率线来自原派活（窗口设计）。这一项全部 S 行都打，若真起作用，空路斜率也可能降到 0.7 以下（人在给定站位的速度与自车速度无关）；
  线照登记执行，读数时另报各窗的绝对 ΔS_B、plan 含停车份额（has_F）与 v24 − 人。
- 闭环：本 lane 不读 PAI 281 也不读 held；PAI-FIX 筛选由主会话做。

## 花费估计

标签 CPU 约 0.5–1 h（32 核）；G1 CPU 数分钟；同一性 2 × 60 步、校准 100 步、四个 1 000 步训练各约 5 min（≈ 0.5 卡时）；inertia cf 6 个模型约 10 min（单卡）。
合计 GPU < 1 卡时。

## 链

`lon_anchor.sh pre`（labels → gate），`lon_anchor.sh chain <旧提交>`（weights → 同一性 → 校准 → 4 训练 → inertia cf → report → 读数），作业 id 在
`$DATA_DIR/runs/defect_fix/lon_anchor/jobs.txt`，阶段在 `stages.txt`。结果在 `experiments/defect_fix/results/lon_anchor/`。
