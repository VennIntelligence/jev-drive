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
这正是 STAND-R（第 399 / 400 条）的起步泄漏类型。闸 G-report 报告「目标在 4 s 末站住、而人在同一时钟在动（≥ 2 m/s）」的行数（clock_conflict），以及反方向「目标 4 s 走 ≥ 2 m、而人在同一时钟 4 s 内走 < 0.5 m」
的行数（go_conflict：站着时站位差几厘米就对应到人起步之后的时刻，标签冒烟里在场景开头等灯的行上见到），都不设线。
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

## 补记 1（2026-10-20 晚，主会话决定；写于 v2 标签 / 闸 / 任何训练输出之前）

**起因（登记的 G1 读数，`results/lon_anchor/gate.md`）**：G1 按登记的构造 FAIL，激活 4.65 %（线 1 %；nuPlan 4.82 %、PAI 4.43 %）。读数后的诊断：
激活行里 99.9 % 的 d_j 在 8 个 pose 上是同一个常数，等于「从自车原点跳到人线上」第一段弦长的多余部分（激活行 |e0| 中位 2.5 m；|e0| > 0.5 m 的行激活 22 %，
≤ 0.5 m 的 0.1 %）；去掉这段后只剩 8 行（0.01 %）。即登记的测试 plan 在头 0.5 s 横跳上人线，这段横向位移被算成纵向弧长；项本身对约 20 % 偏离人线 > 0.5 m 的行
也有同样的毛病（回线的横向运动多出弧长）。存档 plan 上另报告了站位对齐与时钟不一致的行：目标在 4 s 末站住而同一时钟的人在动 1 591 行，
目标走 ≥ 2 m 而同一时钟的人站着 4 120 行（PAI 3 203）。

三项改动（v1 标签文件 `lon_anchor.npz` 与 v1 闸文件保留不动；v2 写 `lon_anchor_v2.npz`、`labels_v2.json`、`gate_v2.{json,md}`）：

1. **G1 改为从自车出发的人路径**：测试 plan = 人日志路径上 s_e + s_t,j 处的点，再沿路径左法向平移 e(s_t,j) = e0 × max(0, 1 − s_t,j / D)，D = max(8 m, 2 s × v0)
   （= onp_geom.imit 的横向偏移淡出）。线不变：激活（max |d| > 0.5 m）≤ 1 %。不过 → 停。
2. **项里的「进度」改为投影到人路径上的站位增量**（闸与训练同一函数）：p_j = plan 第 j 个 pose 在人路径（标签文件里的行坐标系折线，s_e − 5 … s_e + 95 m，
   1 m 顶点，超出日志两端沿端点方向直线延伸）上的投影站位 − 原点的投影站位；d_j = p_j − s_t,j；其余不变（smooth L1 beta 0.5、8 个 pose 求和、双边）。
   投影取在 `lon_only` 之后的 pose 上，所以梯度仍只沿各 pose 自身航向、不跨航向、不到 yaw；最近线段在 detach 的 pose 上选。横向回线 / 离线不再产生进度。
   实现：`lib/nr1_anchor.py progress / station / terms`。
3. **时钟不一致排除（唯一新增的排除）**：c_j = 人在同一日志时刻起的行程 S_h(t + 0.5 j) − S_h(t)。跳过：(a) clock_go：目标 4 s 走 ≥ 2 m 而 c_8 < 2 m
   （同一时钟的人 4 s 平均 < 0.5 m/s，站着）；(b) clock_stop：目标最后 0.5 s 一步 < 0.5 m/s（站住）而同一时钟的人最后 0.5 s 一步 ≥ 0.5 m/s（在动）。
   顺序：标签定义 → 前车排除 → 时钟排除。按池（nuPlan / PAI）与类报告去掉的行数：PAI 按场景类（stop_nolead / turn_sharp / launch / bend），
   两池都按行类（launch：自车 t0 < 0.5 m/s；stop：目标最后一步 < 0.5 m/s；turn：人路径从 s_e 到 s_e + s_t,8 航向变化 ≥ 30°；其余 move）。

另报告（不设线，`gate_v2.md`）：每池存档 plan 对目标「目标更长（项推快）/ 更短（推慢）/ 在 ±0.5 m 内」的份额；按自车前车间距分箱（0–15 / 15–30 / 30–60 m / 无前车）
的同样份额（与 AF-front 相关）。主会话已接受 nuPlan 上「推快」的倾向（模型本来就偏慢）。

臂、抽样（LONP，PAI-FIX 125 个场景、PAI 281、practice-pai 排除）、剂量规则、G4、inertia 读数与通过线都不变。

### 补记 1 追加（10-20 18:20，抽样修正；任何训练完成之前）

第一次链（18:10 入队）的 `weights` 把 84 042 行 cp 行**全部**判成 PAI-FIX 场景：`experiments/pai_fix/scenes/` 里此时多了
`onbox_train_20261020.txt`（盒上 854 个 curated-train 场景的清单，包含全部 570 个 cp 场景），它是库存清单，不是考题。于是 LONP 里 cp 份额为 0，
校准跑（W1 = 0.00091）与已开始的四个训练都是在纯 nuPlan 抽样上。四个训练、cf、report 在完成前取消，校准作废（结果文件改名为 `*_nocp.json` 留档）。
修正：排除来源 = PAI 281、practice-pai，以及 `pai_fix/scenes/` 下除库存清单（文件名以 `onbox_train_` 开头）外的所有文件
（考题清单、chunks、`train_exclude.csv`、`drives_val160.csv`）；新增断言：份额 > 0 时排除后剩下的 cp 行须超过 held 外的一半。
按此，cp 场景里被排除 5 个（都来自 drives_val160.csv），PAI 281 / val160 命中 0。整条 A 链（weights → 同一性 → 校准 → 4 训练 → inertia）用新的同一性 / 校准 tag
（LONGG2-old / new、LONACAL2）原样重跑；臂、剂量规则、闸、读数都不变。

## 补记 2（10-20 18:25，用户批准的 B 块 SIMP；写于任何 B 输出之前）

**设计**（主会话转达）：从 RB0-F-s0 起的一个简化联合配方，其余同 NTV0 配方（每批 83 navtrain + 13 yaw-rate + 32 S，1 000 步 pilot，warm-up 50，1 seed）。
- 保留：模仿、distill（30）、模仿行上的 drivable hinge（10）、cons（3）。
- S 行上三项：
  1. `region`：道路 hinge（Losses43 road_ho：road-and-lane 栅格，Pacifica 足迹四角，margin 0.3）与 lat_onc（对向车道栅格，后轴与车身中心两点，depth 0）
     合成一个 hinge：每个 0.1 s 稠密步 j = 1..40，v_j = max(道路部分, 对向部分)，道路部分 = 四角 relu(0.3 − sdf) 的均值，对向部分 = 两点 relu(−sdf) 的均值
     （各栅格在各自的点上读，与原两项一致）；项 = v_j 对步取均值；shape-only；权重 10（`--simp-region 1`，= road_lam 10）。只有一种栅格的行读那一种
     （on1_JTCP 的对向栅格只覆盖 rb0 / rtb1 行；PAI 行两种栅格都没有，region 为 0）。
  2. `clear`：agent hinge（车身前缘外推 0.3 m、前方物体、走廊内 margin 0.3、走廊外 0）与 lat_side（原车身、走廊外且不在后方的物体、0.4）合成一个净空 hinge：
     每步每个物体，走廊内且中心在原车身中心之前的「前车」取 relu(0.3 − 外推车身距离)，走廊外且不在后方的「侧方」取 relu(0.4 − 原车身距离)，对物体取最大、
     对 41 步取均值；权重 10（`--simp-clear 1`，= agent_lam 10）。梯度：侧方物体 shape-only；前车部分保留沿航向的分量，但**只允许缩短方向**
     （每个 pose 沿航向的位移只回传「让它后退」的梯度，`nr1_simp.lead_poses`）：前车只能让 plan 刹，永远推不长（深度重叠时 SDF 想往前穿，这一方向也被挡住）。
  3. LON-ANCHOR（补记 1 的项），剂量 = 主会话按 A 块 PAI-FIX 筛选选定的那一档。
- 去掉：lat_soft、STAND、FREE-M、弧长项（后三者 NTV0 配方本来就没有）。
- 实现：`lib/nr1_simp.py`；nr1_lat 的 `attach()`（on_train.setup 在 load 之后调用）把训练器的道路栅格交给 region，并把被替换的 Losses43 道路 / agent hinge
  在 S 行上关掉（只动 S 行的 ok 掩码）；两个开关默认 0 时 attach 什么都不做。
- **臂**：SIMP 在 PAI curated 池份额 0 / 0.5 / 1.0（占 32 行 S 抽取；`SIMPA1K / SIMPB1K / SIMPC1K-L18-s0`，抽样 SIMPW00 / 05 / 10，排除规则同补记 1 追加），
  + 同步对照 `SIMPN1K-L18-s0` = NTV0 配方从 RB0-F-s0 起 1 000 步（份额 0，同一池 on1_JTCP、同一抽样 SIMPW00；lat_onc / lat_soft / lat_side 照原配方）。
  臂命令 = 对照命令去掉 `--lat-onc 0.5 --lat-soft 0.1 --lat-side 0.5`、加 `--simp-region 1 --simp-clear 1 --on-anchor <剂量>`（`simp.sh dry` 打印并 diff）。
  去掉 cons 的臂只在 SIMP 赢之后再说。

**闸（训练前）**：
- 同一性：对照命令 60 步 eager，B 代码之前的提交 vs B 代码，不带 SIMP 开关，`df.py bitcmp` 逐位相同。
- 人路径当 plan：S 行的 `fut`（从访问状态出发的人的模仿目标）过 region 与 clear，激活（每步违反的最大值 > 0.05 m）份额 **≤ 2 %**（各自），不过 → 停。
  同表报告被替换的四个原项（道路 hinge、lat_onc、agent hinge、lat_side）在同一批行上的激活份额，nuPlan / PAI 分开。LON 的人路径闸即补记 1 的 G1（已过，0.10 %）。
- 只报告：存档 plan 上 region / clear / 前车部分（「clear 刹车梯度」）的触发份额，nuPlan / PAI 分开（`simp.py gate` → `results/lon_anchor/simp_gate.{json,md}`）。

**读数**：训练后 `inertia.py`（不改）在 P160 上跑 NTV0-L18-s0（编码帧）、四个 B 检查点、RB0-F-s0（起点）、RSG-L18-s0（G0），按 A 块同一条通过线报告每臂对对照；
闭环筛选由主会话在 PAI-FIX 上做。

## 补记 3（10-20 晚，主会话转达用户的 B 块加臂；写于任何 SIMP-F 输出之前）

**起因**：第 407 条（LEAD-BREAK，`results/lead_break/`）：RB0-F 的 ego 状态输入覆盖了前车刹车；navtrain 上需刹车的行 84 % 历史里已在减速，
模型学成「已经在刹才刹」。

**臂 SIMP-F（`SIMPF1K-L18-s0`）**：= SIMPB1K（SIMP，PAI 份额 0.5，同一 LON-ANCHOR 剂量）+ navtrain 模仿行的历史拉平：`lead_break.py nav` 判为 BRK
（需刹车：按原速保持会在 4 s 内越过前车尾部、接近速度 ≥ 0.5 m/s，v0 ≥ 2 m/s）的 navtrain 训练行（`simp.sh flat` → `$DATA_DIR/runs/defect_fix/simp/flat_brk.txt`），
每次被抽到时喂一个不减速的历史：ax = 0，3 个过去 pose 沿自己的折线按当前速度重新排布（`parity_adapter.cv_history`，与 TR1 hist_cv 同一函数），标签不变。
开关 `rm_train --flat-brk FILE`，默认空：不加任何运算；关掉时的逐位同一性另做一次（`simp.sh ident <B 代码前的提交> F` → `simp_identity_F.json`），训练前检查。
锚行（cons）不动；只动训练划分里的 navtrain 行（dev 行不动）。报告拉平的行数（文件里的 token 数、匹配到的训练行数、每步抽到的行数 `flat/rows`）。

**pilot 后的离线闸（只对 SIMP-F）**：`lead_break.py cf`（代码只加 `--tags / --name`，规则不变）在 RB0-F-s0 自己的 PAI 撞前车状态上，
tags = RB0-F-s0（编码帧、G0）、SIMPN1K（对照）、SIMPB1K（不拉平的同配方）、SIMPF1K。读数「原样 serve 时 4 s 内留在前车后面的份额」（F 臂）：
**过** = BRK 上 SIMPF − SIMPN ≥ **+0.10** 且 CI 下界 > 0，**并且** FOL（稳定跟车）上 |SIMPF − SIMPN| ≤ **0.05**；另报 SIMPF − SIMPB、
以及朝 RB0-F 关输入（OFF，第 407 条 0.61）缩小的差距份额（`simp.py leadread` → `results/lon_anchor/simp_leadbreak.{json,md}`）。inertia 读数照补记 2 也加上这一臂。

**补记 2 闸的读数（写在这里备查，不改闸）**：人路径当 plan，region 激活 7.62 %、clear 3.94 %（线各 2 %），**FAIL**；同一批行上被替换的原项：
道路 hinge 11.69 %、lat_onc 1.21 %、agent hinge 2.21 %、lat_side 2.41 %（`results/lon_anchor/simp_gate.md`）。即合并项在人路径上的触发不比它们替换的原项多
（region 低于原道路 hinge；clear ≈ agent + side 之和），2 % 的线定在了现配方自身的水平以下。按登记，B 臂在主会话决定之前不训练（train 的前置检查读 simp_gate.json）。
