# CURATED-PAI 预登记：盘上 570 个 curated-train NuRec 场景的定向 PAI 行，1 000 步 pilot + 闭环闸（2026-10-20，写于任何输出之前）

用户原话：用已经下好的试一下，很可能不行，所以早设闸。第 367 条（PRACTICE-PAI，160 个旧场景、RSG 起）是反证一；这次换成**定向挑选的新场景**
（停车无前车 / 急弯 / 起步），从现候选 **NTV0-L18-s0** 起，并让停车类状态上真有会触发的项。

## 问题

NTV0-L18-s0 在 curated-train 定向场景上、按 PAI 提交 serve 滚出来的状态做成 S 行（+ 这些行上的 STOPPT 项），混进 NTV0 自己的配方短训 1 000 步，
PAI 281（ACC + GAP）是否**好过**同命令、不抽这批行的对照，并且不差于存档的 NTV0 + ACC + GAP（0.744）？

## 数据（第 1 步）

- 场景：box `datasets/nurec/all-usdzs/` 上全部 **570 个 curated-train 场景**（`results/curated/scenes570.tsv`，登记为
  `physical_ai_av/curated-pilot@v1:3fc11cacdbdd`；在 curated-train 内，与 curated-val 441（= test281 + practice-pai）、test281 都不相交，`scripts/cp.py split`）。
  文件大小与清单逐个一致（0 个截断），570 个都在模拟器场景表里、可直接跑。
- **与派活描述不同的一点**：下载器用的是 box 上 01:21 的旧版 `nurec_select.csv`（无 cls / tranche 列，排序不同），盘上的是**旧版**的前 570 名；
  Mac 上 02:24 的新版前 570 名里只有 355 个在盘上。570 个盘上场景全部在新版的 1 554 行里（即都过了新版的考题排除），类别取新版：
  **stop_nolead 220、turn_sharp 216、launch 82、bend 52**（无 rest），新版 tranche 1 259 个、tranche 2 311 个，432 个 drive group。
  新版 csv 不提交（box 上同名未跟踪文件是下载器的清单，提交会让 box 的 `git pull` 冲突）；本 pilot 只读 scenes570.tsv。
- 考题排除：pav.py select 的规则（与 test281 / val160 同一 drive 且时钟差 < 2 min，或同一路段 10 m 内）已在新版里；570 个的 `exam_gap_s` 全是 inf
  （没有 600 s 内同 drive 的考题场景），所以「去掉 exam_gap_s < 600 s」一行也不去。
- **残余泄漏风险**：PAI-AV drive group 排除没做（Tokyo 标签未到，Tokyo 不通）。时钟与路段规则之外，同一地点不同时间的 clip 仍可能和考题共享路口外观 /
  停止线位置；方向上会让 PAI 281 偏乐观。这一条写进结论，不因此改闸。
- 只用 NuRec 渲染帧（第 394 条：真实帧与渲染帧对策略不同，这里不混）。

## 臂（一个联合臂 + 同配方对照，不做析因）

- **起点**：NTV0-L18-s0。配方 = NTV0 的命令（`jt.sh train` 的 DRY 打印，ONK=32、ARCROWS=brake、池 on1_JT 的五部分 + 新行、抽样由 JT00S 推出），
  只改 `--init NTV0-L18-s0`、`--steps 1000`、`--warmup 50`（1K pilot 的惯例），加 `--on-stoppt 0.004`（第 386 条的登记剂量）。
- **新行**：第 367 条的管线原样：NTV0-L18-s0 按 PAI 探针 serve（`JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7`、g10、CONC 4、ALPASIM_FAST=1、
  `PAI_DUMP=100000`、`KEEP_ASL=1`、无视频）滚 → `onp_rows rows / cold` → `onp_cache build` → `rf_pai_pool --name CPAI570`（S 行，车 hinge；PAI 无道路栅格、
  无横向栅格）→ R1 输入路径检查（第 333 条补记 1 的线，NTV0 对 RB0-F-s1）→ `rf_pait merge` 进 on1_JT 的五部分之后（`on1_JTCP`，前缀 `cp`，L18NR1v2 仍是前缀栅格）。
- **为什么加 STOPPT**：这个配方里 PAI 的 S 行只带车 hinge（道路 hinge / 横向项在 PAI 行上都关），停车无前车的状态上**没有任何项会触发**。
  第 386 条 STOPPT 线关闭时写明没检验的条件是「先有在 PAI 测试类场景上冲过停车点的自身滚动」——这批定向场景正是它。标签只打在 cp 行上
  （nuPlan 行 ok = False，所以项只在 PAI 行上起作用；与第 386 条 A / B / C 臂不同，那三臂 nuPlan 行也带项），规则与 `stoppt.py` 的 PAI 行规则相同。
- **pilot**：抽样 = JT00S，nuPlan S 份额减半，另一半均匀给 held 之外的 cp 行（第 367 / 386 B 的「一半」剂量；held = crc32(scene) % 100 < 15，与
  `--on-hold-pct 15` 一致）。**对照**：同一条命令，抽样 = JT00S、cp 行权重 0（nuPlan 部分逐行等于 JT00S；cp 行从不被抽到，STOPPT 项因此为 0）。
  两臂命令只差 `--tag` 与 `--on-wfile`（`cp.sh dry` 打印并 diff）。
- 剂量：1 000 步 × 32 = 3.2 万次池抽取，cp 约 1.6 万次；行数待定（第 367 条 160 场景 2.6 万行，按比例约 9 万行、held 外约 7.7 万）→ 约 0.2 遍。小剂量，
  闸不过只说明这个剂量下无可见好处。
- 不加：人轨迹渲染模仿行（force-GT / log 回放）。工具有一部分（`onp_log_driver.py` + `rf_pait.py pack`），但只有转弯 / 直路窗口，停车接近窗口和 box 上的
  路线重生成（alpasim-base docker）要新写，估计 > 2 h，今晚不做。

## 闸（读之前写定）

- **滚动分阶段**（docs/long-runs.md）：c000 = 1 个场景 → 检查（打分、帧 dump、rollout.asl 都在）→ c001 = 10 个 → 检查（≥ 9 / 10 打分；量每场景磁盘）→
  其余 559 个（每 chunk 32 个）。阶段 2 检查按量得的每场景大小定第 3 阶段的场景数与「同时持有 rollout 的 chunk 数」W，使本 pilot 的净增长
  （保留的 rows / cache / 池 每场景约 0.094 GB + 合池约 12 GB + W 个 chunk 的 rollout.asl / dump）≤ 100 GB，且剩余空间 ≥ 450 GB；第 3 阶段不足 150 个场景 → 停。
  每个 chunk：滚 → 建行 → 删本 pilot 的 rollout.asl 与 dump（只删本 pilot 自己生成的文件；不渲染视频）。
- **R1**：不过 → 停（不训练）。
- **G0（训练前的触发计数，第 386 条的教训）**：NTV0 存档 plan（`poses_model`）上，按类报行数、STOPPT 合格 / 触发、车 hinge 触发、转弯行（turn8 > 30°）。
  线：stop_nolead 类 STOPPT 触发份额 **≥ 6.6 %**（第 386 条练习池 PAI 行 848 / 25 634 = 3.3 % 的两倍）**且** held 外 STOPPT 触发行 **≥ 2 000**（练习池 848 的 2.4 倍）。
  不过 → 停（0 训练卡时），结论写「定向场景也没给出足够的过点状态」。转弯类只报告：这个配方里没有读 PAI 道路的项，转弯行只可能触发车 hinge。
- **闭环（两臂各读 PAI 281 全部 8 个 chunk，NTV0 + ACC + GAP 存档读数的 chunk 表与 serve 原样：`$DATA_DIR/runs/defect_fix/pai_accgap/NTV0-L18-s0`）**；
  场景集合 = 两臂都打出分、无 driver RPC 错误；场景聚类 bootstrap 95 % CI（10 000 次，seed 0）。PASS 要三条都过：
  - L1 配对均值 pilot − 对照 **≥ +0.010**；
  - L2 pilot − 存档 NTV0 + ACC + GAP **≥ 0**（「改进现候选」）；
  - L3 零分：pilot ≤ 对照 + 2，且 ≤ 存档 + 2。
  - 只报告：对照 − 存档（1 000 步续训本身的漂移）、`pai281.py` 分类表（被追尾部分分、slow、corridor、offroad、AF）、过点零分（`stoppt.py overrun`，
    对 A0 的 rf_tax 提取）、**本地 PCS 尺子**（第 392 条 `pai_pcs_local.py`，Mac 上由主会话把两臂作为池外主体打分；不判）。
- 闸在队列里：`cp-gate` 作业写 `$DATA_DIR/runs/practice_pai/curated/gate/{gate.json, gate.md, VERDICT, pai281_full.md, overrun.json}`，并把结论写到
  `$DATA_DIR/runs/practice_pai/curated/VERDICT`；任何 STOP（阶段检查 / R1 / G0）也写这个文件。
- PASS → 下一步：全长或第二 seed + 重读，再考虑提交；FAIL → 停这条线（第 333 / 367 条之后的第三个 PAI 行反证），GPU 不再为 PAI 行下载场景。

## 花费估计

- 滚动：570 个场景 ≈ 18 个 32 场景 chunk + 2 个小 chunk，第 333 条同类 chunk 19.5 min → 约 6.5 作业小时（每卡 3 个，≈ 2.2 卡时）；W 小时墙钟受限
  （W = 2 时约 4–5 h，W ≥ 8 时约 1 h）。建行每 chunk 约 10 min（CPU 8 核）。
- 训练 2 × 约 5 min；读数 16 个 chunk × 约 0.4 h ≈ 6.4 作业小时（≈ 2 卡时）；其余 CPU。
- 合计 GPU 约 4–5 卡时，墙钟约 4–8 h。磁盘净增长 ≤ 100 GB（阶段 2 检查强制）。

## 链（`scripts/cp.sh chain`，一次入队，优先级 11；阶段 2 检查之后由队内作业 `cp-rest` 提交第 3 阶段与尾部）

c000 滚 → chk1 → (c000 建行 → 删)；c001 滚 → chk2 → (c001 建行 → 删)；cp-rest：c002.. 每个 chunk 滚 → 建行 → 删（第 k + W 个等第 k 个删完）→ link → cache →
pool → fwd × 2 → r1 → merge → labels（G0）→ 权重 × 2 → 训练 pilot / 对照 → 各 8 个 PAI 281 读数 → rf_tax × 2 → gate。作业 id 记在
`$DATA_DIR/runs/practice_pai/curated/jobs.txt` 与 `stages.txt`。
