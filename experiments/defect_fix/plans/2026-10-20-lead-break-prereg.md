# LEAD-BREAK 预登记（2026-10-20，任何模型输出之前 push）

诊断，不训练，不跑闭环读数。问题：第 405 条说计划自身的跟车坏在 P0 → RB0-F 这一步（PAI 281 原生 AF-front 16 对 50；
RB0-F 的 50 个里 P0 原生只撞 6 个；加 LEAD MPC 后 26 个得 1 → 前车头还在看，是计划不用它）。基座这一步里是什么让计划不再为前车刹？

## 已定的，不重做
- 第 218 条（P2 时代）：关掉 adapter bias 后 plan = shipped；adapter 的速度剖面是 ego 状态的函数；closing-lead 帧上
  「shipped 减速、adapted 不减」PAI 33 %；adapted 已在刹时照刹，起刹被换掉；navtest closing-lead 帧 76 % 喂入 ego 已在减速。
- 第 331 条：P2-F / RB0-F 在平历史（F）下停在静止前车后 0.85 → 0.39，人历史（H）下不差 → 「读历史不读画面」。开环、navtest。
- 第 404 条：惯性（抄速度）是 P0 自带的，不解释 16 对 50。
- 第 252b / 376 条：PAI 上 keywarp 服务闭环变差（AF-front 3 → 6），所以 PAI 用 10 Hz 真帧。
本条新做的：在 RB0-F 自己撞前车的 PAI 状态上，把三个候选机制分开量；以及 navtrain 里「需要起刹」行的含量。

## 数据
- 状态：RB0-F-s0 原生 PAI 281 读（`$DATA_DIR/runs/defect_fix/lineage/read/RB0-NAT`）里 AF-front 零分的 50 个场景，
  从保存的 rollout.asl 用 `onpol_pai/scripts/onp_rows.py`（rows + cold，不改）重建行文件（10 Hz 渲染帧、喂入行、行动者框）。
  route memory 从 drive.jsonl 取（inertia.py `rmem_of`）。
- navtrain：`navtrain_full.s*of12` 的 tab.npz（ego、speed、fut）+ `agent_labels/navtrain_all-k32.npz`（9 个时刻的框）。

## 决策选取（登记定义）
每 2 个决策取 1 个（5 Hz）；t0 早于该场景第一个事件时刻、不在 after_end 之后；v0 ≥ 2 m/s；8 个槽都是真帧；
自车离日志路径 |e0| ≤ 1.5 m（路径可当车道用）。
前车 = t0 时刻中心离日志路径（`path`，1 m 网格）横向 < 1.75 m、弧长在自车前方的最近行动者框；
间距 gap = 前车弧长 − 前车长度/2 −（自车 rig 到车头：veh_sim 偏移 + 长度/2）；前车速度 = t0 → t0+1 s 的路径弧长位移。
- **BRK（需要刹车）**：有前车、0 < gap ≤ 80 m、closing = v0 − v_lead ≥ 0.5 m/s，且匀速自车（s = v0 t）在 4 s 内某个
  0.5 s 时刻越过前车实际未来位置（该 rollout 里的框，前车尾）。
- **FOL（对照，跟车不需刹）**：有前车、5 ≤ gap ≤ 40 m、|closing| < 1 m/s、匀速 4 s 不越过。
指标（每个决策、每个模型 / 臂）：plan 8 个点投到同一路径弧长，margin = min_k [前车尾弧长(k) − 车头弧长(k)]；
**clear = margin ≥ 0**（主指标，BRK 上的份额）；辅：a02 = plan 0–2 s 平均加速度、ΔS4 = v0·4 − plan 4 s 弧长。
CI：场景聚类 bootstrap 2 000 次，seed 0；配对差在同一组决策上。

## 臂（同一状态的开环前向；`inertia.py` 的编码 / 前向路径）
模型：P0（无 adapter）、RB0-F-s0、NTV0-L18-s0（探索）。
- F：原样（10 Hz 真槽、喂入行、route memory）。
- OFF：present = 0（adapter 与 route bias 都乘 0 = 训练里 anchor / cons 行的条件，按训练应约等于 P0）。
- ZR：route memory 置空（present 保留）。ZE：ego 状态 16 项置 0（present、指令、route 保留；探索，速度 0 不在分布内）。
- AXN：喂入 ax = −2 m/s²（其余原样；探索：减速历史能不能让计划刹）。
- KW：按训练协议造槽（`pai_core.key_slots`：t0 − 1.5 / 1.0 / 0.5 / 0 s 四个关键帧 + 沿自车轨迹 warp），行原样。
- KWOFF：KW 槽 + present = 0（训练里 cons 的原样条件）。
P0 只有 F 与 KW 有意义（不读行）。

## 闸与判词（登记）
- **G0**：RB0-F 的 F 臂对存档 `poses_model`：≥ 99 % 行在 2 个 fp16 步以内且最大 ≤ 0.15 m（inertia 补记 2 的形式）。不过则停。
- **G1**：BRK 上 clear(P0 F) − clear(RB0-F F) ≥ +0.15 且 CI 下界 > 0（开环复现闭环差）。不过则只报描述，不判机制。
- gap = clear(P0 F) − clear(RB0-F F)。
- **(b) adapter 输入盖过前车线索**：R_b = [clear(RB0 OFF) − clear(RB0 F)] / gap。R_b ≥ 0.5 且差值 CI 下界 > 0 → 支持；
  0.2–0.5 → 部分；< 0.2 → 不支持（那就是策略权重本身的帧响应变了）。
- **(a1) 槽协议错配**：R_a = [clear(RB0 KW) − clear(RB0 F)] / gap，同样的 0.5 / 0.2 线。
- **(a2) warp 槽里没有前车线索**：[clear(P0 KW) − clear(P0 F)] / gap ≤ −0.5 → warp 抹掉了线索（训练槽里线索缺失）；> −0.2 → 线索在 warp 槽里还在。
- **(c) navtrain 含量**：同样的 BRK 定义（路径 = 人的未来 8 点 + 沿末端航向直线外推 80 m，自车车头 4.049 m）用在 navtrain 上。
  (c1) 分布外：每个 PAI BRK 决策在 navtrain 的近邻数（|Δv0| ≤ 2 m/s、|Δclosing| ≤ 1.5 m/s、|ΔTTC| ≤ 1 s，TTC = gap / closing 截到 10 s），
  中位数 < 20 → 分布外。(c2) 起刹行稀少：navtrain BRK 行里 t0 喂入 ax > −0.5 m/s²（还没开始刹）的份额 < 25 % → 「起刹行稀少，
  刹车信息多数已在历史里」。
- 探索（不判）：ZR、ZE、AXN、KWOFF、NTV0 各臂；FOL 上各臂的 clear 与 ΔS4。

## 预测（读数之前写）
- (b) 支持：anchor 行把 OFF 训成 P0，第 218 条 P2 关 bias = shipped；预测 R_b ≥ 0.5。
- (a1) 不支持（第 376 条 keywarp 闭环 AF-front 反而多）；(a2) 线索在 warp 槽里还在（关键帧之间是真运动）。
- (c1) 不是分布外；(c2) 起刹行占少数（第 218 条 navtest 76 % 已在减速），即训练里 adapter 能从历史拿到刹车，学不到从画面起刹。

## 成本
行重建：CPU 作业约 5 分钟；cf：一个 GPU 作业约 10 分钟；navtrain 统计与报告：CPU。全部经 `python -m jevdrive.cl submit`，`--after` 串链。
