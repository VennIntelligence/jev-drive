# openpilot 进 Bench2Drive 闭环：base 路线跟随器 + openpilot modifier（op-arb）

状态: done（pilot，2026-09-28 18:54；计划写于 15:40 CST，任何 op-arb 闭环数字之前）
主题: [research/openpilot-closedloop-integration.md](../research/openpilot-closedloop-integration.md)（结果与设计讨论）
相关: [decisions.md](../research/decisions.md) 第 31、33、38、41、49、51 条；[openpilot 迁移](2026-09-24-zeroshot-exam/openpilot-migration.md) D1–D5；
[night-queue-3](2026-09-26-night-queue-3.md) CL 节与 [OPL] 条目
代码: `scripts/op_arb_agent.py`（agent，各仲裁模式）、`scripts/op_arb_server.py`（openpilot server，多返回 lead / meta / desire 头和无 desire 的 twin）、
`scripts/op_arb.sh`（启动）、`jevdrive/op_arb_report.py`（读数）
box run dir: `$DATA_DIR/runs/op_arb/`

## 目标

用户提议：一个很小的 base 只管起步和按路线过弯，openpilot 作为 modifier 叠在上面，需要让行、减速、停车、绕行时由 openpilot 接管或混入。
这里先诊断 openpilot 在闭环里「起不了步」「不转弯」的原因（带逐 tick 数字），再在少量 dev 路线上比较 2–3 种仲裁方式，并做 base-only 消融：
openpilot 必须证明自己加了东西，否则分数是 base 挣的。论文立场（trick or trade）要求 openpilot 外面加的东西通用、小。

## Setup（所有臂相同，只有交给 P7 的轨迹不同）

- 模型：openpilot Cinque（op-adapt 与 WL 两条线改的就是它），20 Hz 每 tick 一步，原生 road + wide rig 每 tick 渲染、同帧配对，5 s 预热刹车保持，
  后轴变换，路线 desire（路口前 20 m turnLeft / Right，变道段 laneChange），即 night-queue-3 CL2 的 openpilot 路径原样；P7 控制器（`P7.json`，5 Hz 交 plan）。
- base：路线几何（Bench2Drive 发给每个 agent 的 dense route，`RouteAdapter` 的 rejoin path，只用传感器位姿）+ 速度 governor
  （设定速度 8 m/s、路径曲率上的横向加速度 ≤ 2 m/s²、加速度 ≤ 1.5 m/s²、路线终点停车）。base 没有任何感知。
- 臂（`op_arb_agent.py` 的 mode）：

| 臂 | 横向 | 纵向 | 对应 openpilot 真车用法 |
|:--|:--|:--|:--|
| native | openpilot plan | openpilot plan | CL2（第 33 条的「openpilot 自己开」） |
| base | base | base | openpilot 关掉（消融） |
| oshadow（只诊断，不作候选） | base | base + 真值 governor（路径内前车 / 行人 IDM，路线上红黄灯停） | — |
| acc | base | min(base, openpilot lead 头上的 IDM) | chill 模式（ACC + 驾驶员管路口） |
| e2e | base | acc + 行驶中 openpilot plan 作速度约束，plan 造成的停车锁存到 openpilot 放行 | experimental 模式（端到端纵向） |
| switch | 行驶中、不在路口转弯区时 openpilot plan；否则 base | 同左；起步与转弯区用 e2e | D3（TCP 伙伴）换成 base |

- IDM 参数（写死）：s0 2.5 m、T 1.2 s、a 1.5 m/s²、b 2.0 m/s²；lead 头用第 0 个 lead、lead_prob > 0.5，x 从相机量起，减去 2.06 m 到前保险杠。
- 路口转弯区：路线 LEFT / RIGHT 命令段前 15 m 到后 5 m（与 D3 相同）。

## 路线（写死，`scripts/op_arb.sh routes`）

全部取自 `bench2drive_0.0.4_val.xml`（Bench2Drive 0.0.4 的 val 路线，与 220 考卷只重 2 条，所以不是 G / K 的考卷路线）：
每个 scenario 类型取 XML 顺序里第一条不在 Town12/13 的路线（没有就取第一条）。
- phase 1（诊断，6 条）：SignalizedJunctionLeftTurn、VanillaSignalizedTurnEncounterRedLight、HardBreakRoute、NonSignalizedJunctionLeftTurn、ParkingExit、DynamicObjectCrossing。
- phase 2（评测，10 条，与 phase 1 不重）：SignalizedJunctionRightTurn、VanillaSignalizedTurnEncounterGreenLight、T_Junction、VanillaNonSignalizedTurn、StaticCutIn、
  MergerIntoSlowTrafficV2、ConstructionObstacle、VehicleTurningRoutePedestrian、OppositeVehicleTakingPriority、SignalizedJunctionLeftTurnEnterFlow。
- TM seed 0，单次。

## 步骤

1. phase 1：native、oshadow 两臂 × 6 条（server 上开 twin：同帧再步进一个 desire = 0 的 session）。读数（`jevdrive/op_arb_report.py diag`）：
   - 起步：静止 tick 按真值分三类（前方 30 m 内无车无人且无红黄灯 / 路径内前车 ≤ 15 m / 红黄灯 ≤ 30 m），报 plan 速度（1、3、5 s）、plan 5 s 位移、action accel、
     lead_prob 与 lead x、meta 的 gas_press，以及各信号区分「该走」与「该等」的 AUC。
   - 行驶中：自由路段 plan 速度相对当前速度的偏差（有没有「一直推迟」的低速偏置）；接近真值 hazard（前车 TTC、红灯、行人）时 plan 减速与 lead 头的 AUC。
   - 转弯：路线 LEFT / RIGHT 段上，openpilot plan（有 desire）和 twin（无 desire）在 1、2、3、5 s 处相对路线的横向偏差；native 臂的路口通过率。
2. 根据 phase 1 定 e2e 的放行信号（release：gas / planx / none）和阈值，**在 phase 2 任何数字之前写进本文执行日志**。
3. phase 2：base、acc、e2e、switch 四臂 × 10 条。读数：DS、RC、SR、碰撞、闯灯、blocked，逐臂对 base 的配对差（10 条，只作方向），openpilot 约束起作用的 tick 比例与距离比例，
   按真值 hazard 分的违规。
4. 写 research/openpilot-closedloop-integration.md。

## 判据（写在任何数字之前）

- 「openpilot 加了东西」= 某个候选臂对 base 的配对 DS 差为正，且碰撞 + 闯灯数少于 base，并且这些改进发生在 openpilot 约束起作用的 tick 附近（违规按时刻归因）。
  10 条路线、单 seed，只能给方向，不给 CI 结论。
- 「openpilot 是 modifier 不是装饰」= 候选臂里 openpilot 约束（lead / plan）是 binding 的 tick 占行驶 tick 的比例 ≥ 5%，且去掉它（= base）结果变差。
- trick 风险按「为 B2D 加了什么 openpilot 真车部署里没有的东西」逐项列：base 的路线跟随（真车上是驾驶员 / 导航）、设定速度（驾驶员）、起步放行（驾驶员按 resume）。

## 资源与预算

- CARLA 只在 GPU 6（测试卡，与 G 的 pilot 共卡，按每卡 ≤ 6 个 server 数实际个数），本 lane 2 个 server（index 160–161），核 144–167；SCH 行 `op-arb`。
- 预算 ≤ 20 worker·h；估计 phase 1 12 次路线运行、phase 2 40 次，每次约 6–10 min，共约 6–9 worker·h。
- 停进程只按记录的 PID。

## 执行日志
- 2026-09-28 13:53 CST smoke（1 条路线 24721，oshadow，不计入任何表）：管线通，每 tick 约 176 ms（agent 167 ms，其中 openpilot + twin 约 92 ms），Completed。
- 2026-09-28 13:58–14:38 CST phase 1（native、oshadow × 6 条）跑完，读数在 `research/results/op_arb/p1/`。要点（详见 research 文档第 2 节）：native 6 / 6 从头到尾车速 0，被判 blocked；
  静止时 plan 5 s 位移中位数：无障碍 0.97 m、红灯 0.18 m、前车 ≤ 15 m 0.08 m；一旦在走（oshadow 带着走），1–4 m/s 上 plan 的 5 s 速度比当前高 3.5–6 m/s（会加速），所以起步失败只发生在静止；
  路口前 0–20 m 带 turn desire 的 plan 在 10 / 15 m 处只跟路线横移的 −0.06 / −0.05（等于直行），无 desire 的 twin 一样，进了弯以后才跟（1.04–1.12）。
- 2026-09-28 14:5x CST **phase 2 的 e2e / switch / oplat 参数（写于任何 phase 2 数字之前；只用 phase 1 的 6 条诊断路线定）**：
  1. plan 约束：`plan_gate` always（行驶中 plan 没有低速偏置，不需要门）、`plan_form` abs（openpilot 自己的位置）、`plan_vmin` 1.0 m/s；plan 只在车速 ≥ 1 m/s 时算「binding」，binding 之后 1.5 s 内继续约束到停下（修掉了一个 bug：静止时的 plan 永远比 base 小，原代码会让它一直 binding，等于回到「起不了步」）。
  2. 停车锁存的放行：`release` planx，openpilot plan 的 5 s 位移 > 2.0 m 持续 1.0 s（`release_th` 2.0、`release_s` 1.0）。依据：phase 1 静止片段按真值分成「该等」（红灯、前车）8 段、「该走」13 段，
     这个规则在 8 段「该等」里 0 段误放、在 13 段「该走」里 4 段放行（最快 0.95 s）；1.5 m 阈值会在 1 段 50 s 红灯里误放，gas_press 系列会在前车等待里误放。
  3. 兜底：锁存 20 s 后 base 自己放行（`latch_max_s` 20，真车上相当于驾驶员按 resume），每次放行记原因（signal / timeout），兜底的比例单独报。
  4. 先按分级规则在 2 条诊断路线（27787 红灯起点、24721 前车急刹）上各跑 e2e、switch 一次看状态机，不计分；然后 phase 2 五臂（base、acc、e2e、switch、oplat）× 10 条。
     oplat = switch 去掉路口转弯区（openpilot 自己过路口，只有起步和锁存归 base），用来在闭环里直接量「带 desire 的 openpilot 会不会转弯」。
- 2026-09-28 14:42–14:58 CST 状态机 smoke（诊断路线 27787、24721，不计分）：e2e 两条都 Completed，锁存按 openpilot 信号放行 8 次、兜底 1 次；switch 在 27787 上 agent 崩溃，
  原因是 `governor` 在不到 3 m 的路径上 `np.convolve(..., "same")` 多返回一个点（代码 bug，已修，行为不变）；switch 在 24721 上 openpilot 开车跟停前车后车头偏出路线，base 接手后顶住不动被判 blocked（驾驶结果，不是接线问题）。
  15:0x 起 phase 2（`jev:op-arb-p2`）：base、acc、e2e、switch、oplat × 10 条，参数按上一条。
- 2026-09-28 16:0x CST **bug：phase 2 的 base 臂不是纯 base**。`op_arb_agent.py` 只在 oshadow 里去掉 lead 约束，base 模式也把 openpilot lead 头的 IDM 算进了 min，所以已跑完的 `p2-base` 实际上是 acc 的第二次运行
  （10 条里 9 条 DS 与 acc 相同）。处理：box 上改名为 `p2-acc2`，作 acc 的重复运行（同 seed 的运行噪声参照）报；代码改为 lead 约束只进 acc / e2e / switch；phase 2 其余臂跑完后重跑 base（10 条），
  base 的数字只用重跑版。e2e 臂正在跑，这个改动只影响 base 模式，不影响它。
- 2026-09-28 18:54 CST phase 2 与 base 重跑完成，SCH 行 `op-arb` 标 done，box 上没有本 lane 的进程。结果（10 条，DS）：base 56.6、acc 58.2、acc2 62.9、e2e 66.2（对 base +9.7 [−5.5, +27.0]）、switch 26.1、oplat 8.5。
  表在 `research/results/op_arb/p2/`，读法与推荐在 research 文档第 3–6 节，决策日志第 57 条。

## 结果

见 [research/openpilot-closedloop-integration.md](../research/openpilot-closedloop-integration.md) 与 [decisions.md](../research/decisions.md) 第 57 条。状态: done（pilot），220 条 × 3 seed 的正式配对待用户与 main 讨论后再登记。

## 后续：两个诊断（用户 2026-09-28 ~19:00 决定后；登记写于 19:1x CST，任何诊断数字之前）

用户判定：五种仲裁都不满意，集成设计搁置，以后再好好调。现在只做两个诊断，同样的 10 条 dev 路线，GPU 6、≤ 2 server、≤ 15 worker·h（SCH 行 `op-arb` 重开）。

**D1 幽灵停车：闭环自我强化，还是只看画面也想停？**
- 事件：e2e 臂（`p2-e2e`）里停车锁存的起点，真值情境为「无障碍」的那些（上一节数到 30 / 42，脚本重算）。每个事件取锁存前 10 s 内最后一段连续「plan 是最紧约束」的起点作为 binding 起点，记它的路线进度 ri_b、车速。
- 闭环读数（binding 起点那一步）：plan 的减速 v(0) − v(3 s)、v(5 s)、meta brake@0 s、lead_prob、lead x。
- shadow 读数：在「别人开、openpilot 只看」的运行里，取同一路线进度（|ri − ri_b| ≤ 2，第一次经过）那一步的同样读数。两个 shadow：`p2-base`（8 m/s 巡航，已跑）和下面 D2 的 `p2-baseslow`（与 e2e 同均速）。
- 「想停」的判据（写死）：plan 的 v(5 s) < 1 m/s 或 v(5 s) < 0.3 · v(0)。基线率 = 同一 shadow 运行里真值无障碍、离任何幽灵停车位置 > 10 m 的所有行驶步上的同一比例。
- 读法（写死）：两个 shadow 里至少一个在幽灵停车位置的「想停」比例 ≥ 50% 且 ≥ 2 × 基线率 →「openpilot 只看画面也想停」（画面里有东西让它停）；两个 shadow 都 ≤ 20% 或 ≤ 1.5 × 基线率 →「闭环自我强化」；其余写「混合」。
  另报闭环里 binding 起点之前 3 s 的车速轨迹（是不是已经被 plan 压慢了才停）。

**D2 同均速对照：「看见了」还是「慢所以撞不上」**
- 新臂 `baseslow`：base 单独开（openpilot 只记录），每条路线的设定速度 c_r = 8 m/s × (e2e 在该路线的平均车速 / base 在该路线的平均车速)，截在 [0.5, 8]：
  15102 1.28、22535 1.69、24497 7.62、24944 0.61、27043 2.25、27297 2.51、27870 6.51、28147 0.92、37969 2.91、9196 2.53（平均车速取非预热步的时间平均，e2e 与 base 取 phase 2 的运行）。
  限定：只匹配平均车速，不匹配速度剖面（e2e 是「正常开 + 长时间停」，baseslow 是匀速慢开）。
- 新臂 `e2enofb`：e2e 去掉 20 s 兜底放行（其余参数不变），只靠 openpilot 自己的放行信号。
- 读数：每臂 DS、RC、车辆 / 行人 / 静物碰撞、闯红灯、blocked；e2e、e2enofb 分别对 baseslow 的逐路线违规数差与 DS 差（10 条，只作方向）。
- 读法（写死）：e2e 的碰撞 + 闯灯总数比 baseslow 少 ≥ 3 次，且少的发生在 e2e 的 openpilot 约束（lead / plan / 锁存）当时 binding 的路线上 →「看见了」有贡献；
  baseslow 的违规数 ≤ e2e →「e2e 对 base 的 +9.7 可以用开得慢解释」；其余写「分不开」。
- 有兜底 / 无兜底两组数字都报：e2enofb 的 DS、blocked 数、锁存靠 openpilot 信号放行的次数与比例。
