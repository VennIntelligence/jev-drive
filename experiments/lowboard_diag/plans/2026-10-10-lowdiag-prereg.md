# LOWDIAG：两块低分闭环榜的失分诊断，一套共用分类（预登记，2026-10-10）

lane LOWDIAG。本文件在本 lane 的任何逐 scene / 逐 rollout 读数之前写定并 push。写之前读过的只有已有决策的汇总数字
（第 138、153、157、159、170、194、218、219、220、226、227、235 条及其结果文档）与各 run 目录里存了哪些文件；没有读过本文件定义的
任何一个分类量。不训练；仿真器的物体框、地图、日志路径、录制路线只作标签与 oracle，用到它们的反事实都是上限。
WA-JEPA 只作参考行。模板是第 227 条（`experiments/body1/results/zeros_diagnosis.md`）。

与已有读数的区别（不重测它们的机制）：
- 227：AlpaSim nuPlan track 的零分。这里是 PAI track 与 HUGSIM，且两榜用同一套类。
- 219 / 220：PAI **plain** driver 40 scene 单 seed 的 at-fault 碰撞。这里是 **served**（`JEV_VCONT=1.0 JEV_LEAD=1`，第 226 条）
  60 scene 2 seed 的全部零分，碰撞只是其中一类；219 的「command 强制直行后终点只动 0.02 m」不重测，只在 corridor 零分上逐例读
  「强制成日志方向的 command」的效果（219 没有做过这个方向）。
- 153：P2H10 的 HUGSIM 四方向（按用户的四个方向分桶）。这里是 SH30（当前配方，第 170 条），类的定义换成本文件的共用分类，
  并加上底模自身输出一栏。fd_hugsim 的几何量定义原样沿用。
- 235：PAI 的零分只按 scorer flag 计数。这里给每个零分一个原因类。

## 1. 单位与失分

- **PAI**：base `P2H10-F`，served，PAI2 的 60 scene。主集 = s0、s1（FIX1 的 `ab_*` stack，零分 rollout 的 `rollout.asl` 保留）。
  单位 = rollout；失分 = 1 − score。失败单位 = 零分 rollout。非零但 < 1 的 rollout（progress 失分）单列一行；它们的 `rollout.asl`
  已删，只用保留的 `drive.jsonl`（served plan、适配前 plan、lead 记录）与 controller trace 分类，不够用时才把 s0 / s1 的 12 个
  stack 按原命令、原 chunk 列表重跑并保留全部日志（pool，约 2.5 card-h）。s2、s3 只给 scorer flag 计数作核对。
- **HUGSIM**：`SH30-F-s0 / s1`，`spec_plan_smooth`，all64，已存的 bench run。单位 = run；失分 = 1 − HD。失败单位 = end ∈
  {fg_collision, bg_collision, off_route, max_steps}。end = complete 的 run 的失分（扣分项）单列一行，按子分数拆，不强行归类。
- **navhard stage 1**（可选，便宜才做）：开环，不在本分类里；若做，只报按子分数的失分份额。

## 2. 每个失败单位的真值量

- 事件时刻 E：PAI = 置零 flag 的第一个计分步（at-fault 碰撞 / offroad / left corridor；多个 flag 取最早的，另记全部）；
  HUGSIM = 结束步。事件类型 K ∈ {agent（碰撞物体）, boundary（PAI offroad，HUGSIM bg_collision）, corridor（PAI left corridor，
  HUGSIM off_route）, none（HUGSIM max_steps；PAI 无 flag 的零分）}。
- 参考线 ref：PAI = 日志 ego 路径（scorer 的 corridor 参照）；HUGSIM = 录制路线（fd_hugsim 的 `Route`）。lat(t) = ego 到 ref 的
  有符号横向距离。**onset** t_on = E 之前最后一个 |lat| < 1.5 m 的时刻（fd_hugsim 的规则；一直 < 1.5 m 则 t_on = E）。
- ref 在 onset 附近的几何：窗口 [s_on − 10 m, s_on + 20 m] 上 ref 的航向变化 Δψ 与最小半径 R（曲率取 ±4 m 差分）。
  **turn** = R < 15 m 或 |Δψ| ≥ 30°；**bend** = 15 ≤ R < 50 m 或 10° ≤ |Δψ| < 30°；**straight** = 其余。
- 被撞物体：与 ego 框相交（否则最近）的 simulator 物体框；相对航向 dh、在 ego 系的位置、速度。
- 路面：PAI = simulator 地图的 road area 与 lane 的并（227 的 `road_union`；PAI 地图取不出来时本量缺失并写明）；
  HUGSIM = `ground.ply`（fd_hugsim 的覆盖率规则）。

## 3. 原始 flag（可多选，全部报计数）

- **O1 exception**：driver 异常 / rollout 中断（PAI 无 flag 的零分）。
- **O2 sim / scorer**：(a) PAI hand-over spin：k = 0 时车速 > 23 m/s 且前 2 s 内 |yaw rate| > 0.3 rad/s（第 219 条的接口缺陷形态）；
  (b) HUGSIM 脚本对向 / 横穿车撞上来：被撞物体 |dh| ≥ 60° 且「ego 从 E − 3 s 起沿已行驶路径以 4 m/s² 刹停」的反事实仍接触
  （停下也被撞）；(c) 被追尾：物体中心在 ego 中心之后且 |dh| < 45°；(d) PAI offroad 置位而 ego 框在 E 时刻仍被路面并集覆盖。
- **L1 no-stop**：K = agent；物体在 E 之前 ≥ 2 s 已在 ego 已行驶路径的扫掠带内（路径宽 = ego 宽 + 0.5 m）；接触在 ego 前部
  （物体中心在 ego 中心之前）；且「沿已行驶路径、从 E − 3 s 起 4 m/s² 减速」的反事实不再接触（物体按记录运动）。
- **L2 fast entry**：K ∈ {boundary, corridor}，ref 为 turn 或 bend，onset 车速 v 满足 v² / R > 4 m/s²（fd_hugsim 的线）。
- **L3 over-speed**（只 PAI，只作 flag、不参与互斥判定）：onset 车速 > 同一弧长处日志车速的 1.2 倍。
- **L4 no-go**：K = none 的无进展：HUGSIM max_steps / launch stall；PAI 非零 rollout 的 progress 失分。子类：`follow` = 静止或
  低于日志车速 0.5 倍的时间里 ≥ 50% 有 lead 限速在起作用（`fix.lead.src` 不是 e2e）或走廊内 20 m 内有物体；`free` = 其余。
- **R1 branch**：K ∈ {boundary, corridor, agent}，E 时 |lat| ≥ 1.5 m，且 onset 之后 ego 与 ref 的航向差最大值 ≥ 20°
  （走了另一个出口 / ref 转弯而 ego 直行 / ref 直行而 ego 转弯）。
- **R2 turn**：K ∈ {boundary, corridor}，ref 为 turn。记内侧 / 外侧（lat 的符号对转向符号）。
- **D drift**：ref 为 straight，E 时 |lat| ≥ 1.5 m，且不满足 R1（航向差 < 20° 的横移）。
- **C1 agent clearance**：K = agent 且不满足 L1、O2。
- **C2 boundary clearance**：K = boundary 且 ref 不是 turn。

## 4. 互斥类与优先级

一个单位一个类，按下面顺序取第一个命中的；原始 flag 的多选计数另表给出。

1. **other**：O1 或 O2。
2. **longitudinal（证据明确的）**：L1（碰撞而刹停可免）；L2（入弯过快）；L4（无事件的不走 / 走得慢）。
3. **route**：R1 或 R2；以及 K = corridor 的全部剩余单位（在路上而离开 ref：按任务书「没有 / 没有跟随 route 而离开 corridor」
   归 route，子类记为 drift 或 bend）。
4. **clearance**：C1、C2，以及 K = boundary 的 D（直路横移出界 / 擦边）。

优先级的理由与已知偏向：other 先行是为了不把仿真 / scorer 的问题算进驾驶类；longitudinal 排在 route 之前只针对三种证据明确的
情形（L3 不参与），因此「先离开日志车道、再撞上该车道里的车」会归 longitudinal（若刹停可免），这一格同时带 D / R1 flag，
从多选表里可以读出来；corridor 一律归 route 是任务书的定义，其中 drift 子类是否算「route 工作线」由第 6 节单独回答。

## 5. 是否已在 served plan 里、lead time

每个决策的 served plan（PAI：`drive.jsonl` 的 `poses`，4 s；HUGSIM：发给控制器的 plan，3 s）按 ego 框扫掠：
agent = 与被撞物体同时刻的框相交；boundary = 框离开路面并集（PAI 深度 > 0.2 m；HUGSIM 覆盖率低于 fd_hugsim 的线）；
corridor = plan 上的点离 ref ≥ 4 m（PAI）/ plan 终点离 ref 的横向距离 ≥ 1.5 m 且在增大（HUGSIM off_route）。
**in-plan** = E 之前 3 s 内至少一个决策的 plan 命中与 K 相同的事件；**lead time** = E − 到 E 为止最后一段连续命中的第一个决策时刻
（封顶 8 s）。另报 ego 与上一个 plan 0.5 s 处的偏差（跟踪误差），中位与最大。K = none 不适用。

## 6. 底模自身输出是否正确（「base right」）

底模 = shipped 权重、不带 adapter（`pp_train.load_pmodel("P0")`）。

**PAI（同一时刻、同一 vision token 的离线 replay，开环）**。窗口 W = now ∈ [E − 4 s, E − 0.5 s] 的决策；对 K = corridor 与
boundary 用 [t_on − 2 s, t_on + 2 s] ∩ (−∞, E)。对每个决策构造三条候选并做第 5 节同样的扫掠：
(i) P0 的 plan 原样；(ii) served 的 path 配 P0 的速度剖面（按弧长重定时）；(iii) P0 的 path 配 served 的速度剖面。
- **base-right（speed）** = served plan 命中事件的决策里，(ii) 不命中的份额 ≥ 50%。
- **base-right（path）** = 同上，(iii) 不命中的份额 ≥ 50%；对 K = corridor 另加一条：W 内 P0 plan 在共同弧长处到 ref 的横向误差
  比 served 小 ≥ 1 m 的决策份额 ≥ 50%。
- **base-right（any）** = (i) 不命中的份额 ≥ 50%。
- L4（不走）：P0 的 2 s 弧长 ≥ served 的 1.5 倍的决策份额 ≥ 50% 记为 base-right（speed）。
- lead 头：served driver 已经在环里用 lead 输出限速（第 226 条），所以 lead 头「看见了」不再单独算 base-right，只报
  E 之前 ≥ 1.5 s 是否有 lead_prob ≥ 0.5（P0 与 served checkpoint 各一列）。
限定写在前面：开环，P0 看到的是 adapted driver 开出来的状态；(ii)(iii) 是运动学拼接。

**HUGSIM**。已存 run 没有存同一时刻的 P0 plan，也没有存帧，同一时刻的 P0 plan 读不到（写入「不能确定」）。两个替代读数分开报：
- **base-right（lead，同一时刻）**：run 自己每步记录的 lead 头（served ONNX 同一次前向；第 219 条在 AlpaSim 上量过 served 与 P0 的
  lead 读数相同，这里作为假设并写明）。E 之前 ≥ 1.5 s 存在一步 lead_prob ≥ 0.5 且所需减速度 a_need ≥ 1.5 m/s²
  （第 219 条 L2 的主工作点，`col1_pai.a_need` 的式子）记为「lead 头要求减速」；只对 K = agent 定义。
- **base-right（闭环 P0）**：新跑一条 `P0` 的 `spec_plan_smooth` all64（`python -m jevdrive.bench run`，pool）。对 SH30 的每个失败
  单位：P0 在同一 scenario 里通过了 SH30 事件处的 route 弧长 s_E + 5 m 而没有同类事件，记为 base-right（闭环）；另报 P0 在
  s_on 处的车速对 SH30 的车速（L2）。P0 自己在 s_E 之前就结束的 scenario 记为「P0 未到达」，不计入分母。
  限定：闭环下两条轨迹的状态不同，这是「底模在同一地点的行为」，不是同一时刻的输出。

## 7. PAI corridor 零分：缺 route 信息、drift、还是转不过去

对象 = 互斥类 route 中 K = corridor 的单位（以及 offroad + corridor 同时置位的单位，单列）。在 onset 处量：
- **ref 类型**：第 2 节的 turn / bend / straight；另加 fork = ref 在 s_on + 30 m 处相对 ego 在 onset 的位姿的直线延长横移 ≥ 3 m
  而 |Δψ| < 10°（变道 / 岔口）。
- **喂入的 command**：W′ = [t_on − 4 s, t_on] 内喂入 command 的众数 cmd_fed；**cmd_true** = driver 自己的规则
  （`command_from_route`）套在日志路径上（以各决策时刻的 ego 位姿为原点）的众数；**route 偏差** = 发给 driver 的 route waypoint
  在 s_on + 30 m 处到日志路径的横向距离。
- **强制 command 的 replay**：同一 token 上把 command 强制成 左 / 直 / 右 的三条 plan。effect = 强制成 cmd_true 与强制成直行
  （cmd_true 是直行时取与 cmd_fed 的对比）的 plan 在 4 s 终点的横向差，W′ 内的中位。

子类（按顺序取第一个命中的）：
1. **route 信息缺失或错误**：ref 非 straight 且（cmd_fed ≠ cmd_true 或 route 偏差 ≥ 3 m）；或 ref 为 straight 而 cmd_fed 不是
   直行且 ego 朝 cmd_fed 的方向离开。
2. **信息正确但没有用上**：ref 非 straight，cmd_fed = cmd_true，route 偏差 < 3 m，effect < 1 m（command 对 plan 没有作用）。
3. **转不过去**：ref 非 straight，cmd_fed = cmd_true，effect ≥ 1 m，plan 朝正确方向转但不够或太晚（served plan 在共同弧长处的
   航向变化 < ref 的 0.8 倍，或切内角）。
4. **drift**：ref 为 straight，其余。

「route 工作线是否存在」的读法（事先写定）：子类 1 + 2 占 corridor 零分的份额 ≥ 50% → 存在，且瓶颈是 route 信息 / command 通道；
子类 3 ≥ 50% → 是转弯能力（表征 / 目标侧），不是 route 信息；子类 4 ≥ 50% → 是第 205 条的横向漂移，不是 route；
都不到 50% → 报份额，不下单一结论。另报每个子类里 base-right（path）的份额。

## 8. 分阶段与核对

1. 每榜先取 5–10 个有代表性的失败单位（每个 scorer flag / end 类型至少 2 个，另含 1–2 个两 seed 结果不同的 scene），出 BEV
   （ego 框与已行驶路径、ref、served plan、P0 plan 或 lead 读数、物体、路面）加模型输入帧（PAI），逐个用眼核对分类器的每个 flag。
   这一步允许改的只有实现错误与几何量的取法；阈值或类定义若要改，写成本文件的补记并注明是在看过哪几个单位之后改的。
2. 通过后跑全部单位，出表：每榜每类的失分份额、每类里 base-right 的份额、原始 flag 多选计数、in-plan 与 lead time。
3. CI：按 scene / scenario 重采样（10 000 次）。n < 8 的格只作描述。

## 9. 事先写下的预期（不是判据）

PAI served：corridor 类最大（约一半零分），其中 drift 与 route 信息两类都不小；碰撞类里 L1 仍占多数。HUGSIM：other（脚本对向车）
是最大的一类，其次 longitudinal（静止前车停得太近、入弯过快）。base-right 在 longitudinal 类里应高于其余类（第 218 条）。
