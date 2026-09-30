# op-adapt L 进 Bench2Drive 闭环（预登记）

状态: 预登记，写于任何适配模型的闭环数据之前（此前只做了工程：把 checkpoint 导成可服务的 ONNX 并做数值等价检查、agent / server / 脚本接线，见「工程与等价检查」；没有一次 CARLA 运行）。2026-10-01 夜间跑。
主题: [research/decisions.md](../research/decisions.md) 第 57、74、77、78 条；[op-drive](2026-09-29-op-drive.md)（本 lane 直接复用它的管线、R1 resume 策略、配速对照与 dev 集）；[op-adapt L 预登记](2026-10-01-op-adapt-L-prereg.md) 与 [后续检查](2026-10-01-op-adapt-L-followup.md)。
决策记录 `research/decisions.md` 不在本 lane 里改（由用户写）。op-adapt r2 的 run 目录不动。

名词（首次使用处各给一句）：**B2D** = Bench2Drive，CARLA 里的闭环评测，按路线（route）跑；**DS** = driving score（完成度乘各类违规的罚分倍数），**RC** = route completion（完成路线的比例）；**infraction** = 违规事件（碰撞、闯红灯、出车道、卡死等）；
**op-drive** = 用 openpilot Cinque 横纵向都开的 B2D 管线（路线几何在路口区内接管横向，纵向 = 设定速度、lead 头上的 IDM、行驶中的 plan 位置剖面三者取最紧）；**latch** = plan 自己要求停车后的「停车锁存」，放行有三条路：plan 说走（`signal`）、前车离开（`lead_go`）、驾驶员 resume（静止 `latch_max_s` 秒后的定时放行，`timeout`）；
**R1** = 第 74 条采用的 resume 策略（红 / 黄灯前 40 m 内不按 resume，其余照旧 5 s）；**pace-matched control（配速对照）** = 把无感知的路线 base 开慢到与被测臂相同的平均车速（op-drive 配速登记 P），用来扣掉「开得慢本身」的收益；
**intent adapter** = op-adapt L 里给每个 WOD 路线意图（直行 / 左 / 右）一个可学的 32×512 embedding，加到 hidden token 上；**ghost stop（幽灵停车）** = 真值上没有任何理由（前车、行人、红灯、路线终点）的停车；
**TM seed** = CARLA traffic manager 的随机种子；**训练 seed** = op-adapt L 的 checkpoint seed（`main` 有 0 / 1 / 2）；**route-clustered bootstrap** = 对路线（不是对单次运行）重采样，每条路线先对 2 个 TM seed 取平均。

## 1. 要回答的问题

op-adapt L 在 WOD val 的开环读数上把起步 / 停车 / 起转的捕获率各抬高了 +0.10 / +0.31 / +0.12（第 78 条，`main` 三个 seed，假起步 +2.7 pp，RFS 不变），后续置换检验说明 adapter 读的是路线意图。
op-drive 的结论是：原生 Cinque 的全部收益可以由「开得慢」解释（drive 对配速对照 −6.4 DS，第 74 条），失败形态是 plan 在静止时不放行（靠 5 s timer resume）、幽灵停车、路口不转。
问题：**开环的捕获增益在闭环里变成驾驶行为了吗**（起步由 plan 自己放行、该停时停、路线说转时 plan 跟着转），**它让 DS / RC 超过「开得慢」能给的吗**。登记的判定对象是对配速对照的配对差，不是裸 DS。

## 2. 管线（与 op-drive 逐项相同，只换模型和导航输入）

全部沿用 `scripts/op_arb_agent.py` 的 `drive` 模式与 `scripts/op_arb.sh set 2 <tag>`：curv 执行（openpilot 的 desired curvature 20 Hz 转方向盘）、区（route steering zone）与分歧兜底、纵向 = min(设定速度 8 m/s 与曲率限速、lead 头 IDM、行驶中 plan)、`hold: intent` 锁存、放行规则、低速滑行 `coast_v` 2.5、R1（`resume: nored`，`latch_max_s` 5，注意用户提到的 20 s 是旧默认，op-drive 与本 lane 都用 5 s）。路线、TM seed、相机 rig、controller P7 都不变。

### 2.1 模型怎么换进去

checkpoint（`$DATA_DIR/runs/op_adapt_L/runs/<name>-s<seed>/ckpt-final.pt`）导成 `cinque.ort.onnx` 的副本（`scripts/op_l_onnx.py build`）：训练过的 initializer 换成 checkpoint 的权重（训练用 fp32 主权重，这里转回图的 fp16），
另加一个输入 `intent_bias`（1×32×512 fp16 = E[intent]，全零 = 无意图），加到当前帧的 hidden token（policy 的输入与它的池化 `mean`）和 policy 33 槽队列里的 8 个真实过去槽，**不加到写回模型自己状态队列的那一份**，
于是每一步都把「当前 intent」加到全部 9 个 context 帧上，与训练里 `H' = H + E[intent]` 对全部 9 帧的写法一致。服务器（`scripts/op_arb_server.py --onnx`）按每次请求里的 `intent` 选 E 的一行。没有 adapter 的 checkpoint（`noint`）只换权重。

### 2.2 路线指令到 intent 输入的映射（因果，只用 op-drive 已在用的路线）

WOD intent 是 1 直行 / 2 左 / 3 右（0 未知，闭环里不用）。agent 已经有的路线是带命令的 waypoint 序列，intent 在每个 plan tick 由自车的路线进度 s 和路线命令 run 决定：

- 自车在某个 LEFT（RIGHT）命令 run 的 [起点 − 20 m, 终点 + 5 m] 内 → intent 2（3）；其余（直行命令、变道命令、车道保持）→ intent 1。
- 20 m 与 op-drive 原生 turn desire 的提前量（`DESIRE_RANGE_M` 20）相同；+5 m 与路线区的后缘相同。依据：WOD 的转向 intent 帧是低速的转弯前后帧（左 / 右 intent 帧的 v0 中位 1.2 / 2.4 m/s，55% / 77% 的帧 4 s 内横向位移 ≥ 2 m，`prep/wod.npz`），不是整段提前很久的标签，所以取「即将进入转弯」的近提前量，而不是 40–60 m。
- 因果：只读路线与自车在路线上的投影，不读任何真值（不读其他车、红绿灯状态、地图拓扑）。变道命令不映射成 intent（WOD intent 没有变道）。

**native desire 的处理**：主臂（`lmain` 及其消融）**关掉**原生 desire 脉冲（`desire: false`，policy 的 desire 输入恒零），与训练条件一致（op-adapt L 训练与读数里 desire 输入恒零，导航只经 adapter）。
原因与代价：训练分布里没有「adapter + 原生 desire」的联合输入；关掉 desire 后与 `drive`（原生 desire 开）比，差别同时含「模型」与「导航通道」两项，所以另设两个拆分臂：`dnod`（原 Cinque，desire 关，无任何导航输入）与 `lkd`（`main`，adapter 与原生 desire 同时开）。

## 3. 臂

全部在 dev 10 条路线 × TM seed 0、1 上，`drive` 与 `dbase` **本 lane 并发重跑**（不复用 op-drive 的旧结果，让 CARLA 与系统的非确定性对所有臂同样作用）。每个 `set` 调用 = 一个 unit（一个臂 × 一个 TM seed × 10 条路线）。

| 臂 | 模型 | 导航输入 | 用途 |
|:--|:--|:--|:--|
| `dbase` | 无（路线 base，openpilot 只记录） | – | 配速所需的参照（c0 = 8 × v(臂) / v(dbase)）与 base 参照 |
| **`drive`** | 原 Cinque | 原生 desire | 对照 A（第 74 条的 drive，R1） |
| **`lmain`** | op-adapt L `main`（= `sel_s4ia_dw3`）训练 seed 0 | intent adapter，desire 关 | **主臂** |
| **`lnoint`** | `noint` seed 0（同一训练，无 adapter） | 无 | 归因：intent 的贡献（与 `lmain` 配对）；与 `dnod` 配对 = 适配本身（不含 intent）的贡献 |
| `dnod` | 原 Cinque | 无（desire 关） | 拆分：`lmain − dnod` = 模型 + adapter，排除掉「desire 被关」 |
| `lkd` | `main` seed 0 | adapter + 原生 desire | 拆分：原生 desire 保留是否伤 / 帮 |
| `ldw10` | `dw10` seed 0 | adapter，desire 关 | 可选：蒸馏权重 10（过了开环 L2 线，增益缩到约三分之一） |
| `lmain1` / `lmain2` | `main` 训练 seed 1 / 2 | 同 `lmain` | 训练 seed 散布（`lmain` 跑完 dev 后跑，不以 pilot 结果为条件） |
| `dtz` / `ltz` | 原 Cinque（desire 开）/ `main` seed 0 | 同 `drive` / `lmain` | **探索臂**：区里去掉 LEFT / RIGHT（`zone_m` 只留 STRAIGHT 与变道），转弯的横向由 openpilot 开，分歧兜底（plan 与路线在 15 m 处相距 > 1 m 交回路线）保留；直接读「plan 能不能转」，不进判定 |
| `dbaseslow*` | 无 | – | 每个被判定的臂（`drive`、`lmain`、`lnoint`，以及跑了的 `ldw10`、`lmain1` / `lmain2`）各一条配速链，见 §5 |

配速链按 op-drive 的登记 P 原样：c0 = 8 × v̄(臂) / v̄(dbase)（逐路线，截在 [0.5, 8]），再按 c_{k+1} = clip(c_k × v̄(臂) / v̄(slow_k), 0.5, 8) 迭代（即至少一次 P 校正），终止于 10 条路线 × 该 TM seed 的均速比 v̄(臂) / v̄(slow) ∈ [0.9, 1.1]，最多到 `dbaseslow4`；对照取终止的那一版，不按 DS 挑；每一版的 DS 都报。

## 4. 路线与 seed

- **dev**（唯一判定集）：op-drive 的 10 条 `27043,15102,24944,27870,22535,37969,24497,27297,9196,28147`，TM seed 0、1；每条路线对两个 seed 取平均再做配对。dev 是看过的集（第 74 条），不是干净的测试。
- **held-out**（op-drive 登记的 19 条，从未运行过）：**只在 §8 的 dev 条件满足时运行一次**；否则保持未看，结果里如实写。
- `tuning` 6 条不用。

## 5. 读数与定义（写于任何读数之前）

所有读数来自 b2d 的 `results.json`（DS、RC、各类 infraction）与每步日志 `plans.jsonl`（`scripts/op_arb_agent.py`；本 lane 新增字段 `intent`）。报告代码 `jevdrive/op_l_b2d_report.py`（复用 `op_arb_report.drive_row` 的口径）。
统计：route-clustered bootstrap，10 条路线（各 2 seed 平均）重采样 10 000 次，percentile 95% CI；违规数与计数类读数在 20 次运行上汇总，CI 同样按路线重采样。

**主判定读数**（S2 风格，同第 74 条，对象换成被测臂 X）：

| 线 | 定义 | 过 |
|:--|:--|:--|
| **L-S2** | ΔDS(X − slow_X) 的 10 路线均值 > 0，且 X 的违规（车 / 行人 / 静物碰撞 + 闯红灯 + 停车标志，两个 seed 合计）少于 slow_X | 两条同时成立 |
| **L-S1** | ΔDS(X − drive) 均值 > 0，且更好的路线数 ≥ 更差的路线数（\|ΔDS\| ≤ 0.5 算同） | 同左 |
| L-DiD（描述为主） | [DS(X) − DS(slow_X)] − [DS(drive) − DS(slow_drive)]，CI 下界 > 0 才读成「适配的闭环收益大于原生自己的」 | CI 下界 > 0 = 强读法；点估计 > 0 但 CI 跨 0 = 弱读法 |
| L-Safe（门，不是收益） | X 的 openpilot 横向占比 ≥ drive 的 − 0.10；xt_op_med ≤ 0.5 m；（off_lane + route_dev）≤ drive + 2；blocked ≤ drive + 2（两个 seed 合计） | 全过才算「模型没有把驾驶弄坏」；不过时 L-S1 / L-S2 的通过不作数 |

**行为读数**（与开环切片对应，全部对 `drive` 同批读数配对，报 20 次运行的汇总与 CI；这些线只说明「行为有没有接上」，不改 DS 的判定）：

- **standstill episode**：非 warm 步里 v < 0.2 m/s 连续 ≥ 1 s 的段，且车之前动过（v > 1 m/s）；onset 的真值情境沿用 `op_arb_report.context`（red = 红 / 黄灯 ≤ 30 m、lead = 路线上前车 ≤ 15 m、ped、free = 30 m / 40 m 内无前车 / 行人 / 红灯、其余 other）；路线终点前 15 m 内的 episode 不计（终点停车是规则内的）。
- **B1 起步由 plan 自己放行**：按 latch 的放行原因统计（`rel` ∈ signal / lead_go / timeout）。(i) **定时 resume 占比** T = timeout / (signal + lead_go + timeout)，(ii) **plan 放行数** = signal 次数 / 运行，(iii) free 情境 standstill 的中位时长（描述）。
  **B1 过** = T 比 `drive` 低 ≥ 0.15（点估计）且 signal 次数 / 运行不低于 `drive`。
- **B2 该停时停、不多停**：(i) **幽灵停车率** = onset 情境为 free 的 standstill 数 / km（行驶里程），(ii) 红灯前 plan 停车（`stops_red`）不少于 `drive` 的 0.8 倍，(iii) 闯红灯 infraction 数 ≤ `drive`。
  **B2 过** = (i) ≤ `drive` 的 1.25 倍 且 (iii)。（这是开环里「stop 捕获 +0.31」与「不变保守」两条在闭环的对应：更多该停的停车不应以更多幽灵停车为代价。）
- **B3 路线说转时 plan 跟着转**：在 LEFT / RIGHT 区内、v > 1 m/s 的步上，`div`（plan 与路线在 min(15 m, plan 弧长) 处的距离，已按弧长缩放）≤ 1.0 m 的步占比 = `turn_agree`。**B3 过** = `turn_agree` 比 `drive` 高 ≥ 0.10。
  另报 intent 的正确性（日志里 `intent` 与路线 LEFT / RIGHT 区的一致率，必须 100%，由构造保证，是管线自检）。
- 探索臂（`dtz`、`ltz`）：每个 LEFT / RIGHT 命令 run 的 `turn_ok` = 自车路线进度越过该 run 的终点，且区内真值到路线线距离的最大值 ≤ 1.5 m，同期没有 off_lane / route_dev 位置落在该 run 内；`turn_ok` 率与 openpilot 横向占比（区内 lat == op 的步占比）描述性报告。
- 其余描述性读数：DS、RC、各类 infraction、blocked / timeout、平均车速、锁存次数、openpilot 横向 / 纵向占比、xt_op_med、`plan` 计算延迟（`ms` 字段）。

**读法（写死）**：
1. 「闭环收益成立」= L-S2 且 L-S1 且 L-Safe；强读法再加 L-DiD 的 CI 下界 > 0。
2. 「行为接上了」= B1 ∧ B2 ∧ B3（逐条报）。单独的 B 通过而 L-S2 不过 = 行为变了但没有换成分数；B 都不过而 L-S2 过 = 分数不是靠这三件事换来的（写明是别的机制）。
3. `lmain − lnoint`（配对，同 §5 的统计）回答 adapter 的闭环贡献，`lnoint − dnod` 回答适配权重本身的，`lkd − lmain` 回答原生 desire 保留的影响；都只是描述。
4. 计数类读数的 20 次运行很小（drive 上 timer resume 约 0.8 / 运行，signal 约 0），「没有差别」不能读成「等价」：CI 太宽时写「无法区分」。
5. 训练 seed 散布：`lmain` / `lmain1` / `lmain2` 三个的 L-S2、L-S1 各自报；只有同号才把「成立」写成跨 seed 的。

## 6. sanity checklist（写死；每个阶段读数前先过；不过就停，修真 bug，绝不放宽登记线）

**阶段 1（1 路线 × 1 臂，`lmain` TM seed 0 在路线 24944 上，1 个 worker，另有同路线的 `drive` 作延迟对照）**：
C1 服务器日志显示加载的是 `lmain-s0.onnx` 与 `E.npy`（形状 4×32×512），起服务耗时与 session 数已记；数值等价检查已记录（§7）。
C2 日志里 `intent` 与路线命令一致：LEFT / RIGHT 区内为 2 / 3，区外恒 1，一致率 100%，并至少见到一次 2 或 3（路线有转弯）。
C3 控制回路延迟：`ms` 中位数 ≤ 同路线 `drive` 的 1.2 倍，p99 ≤ 1.5 倍（不慢于基础模型：图上只多 4 个算子）。
C4 无崩溃：results.json 存在，`status` 不含 Failed / crash，attempt 数 ≤ 3（CARLA 服务端 rc 139 重试算基础设施，记录次数）。
C5 plan 合理：自由路况（真值无前车 / 行人 / 红灯）行驶步（v > 2 m/s）上 plan v(5 s) 的中位数与 `drive` 的比 ∈ [0.8, 1.25]，\|y@2 s\| 中位 ≤ 0.6 m，无 NaN，两条内侧 lane 线概率（取小）的中位数 ≥ `drive` 的 0.5 倍（见偏离 D1），幻觉 lead 步占比 ≤ `drive` 的 2 倍。
C6 字段齐全：`plans.jsonl` 里 `lat / lat_why / div / go / intent` 齐。

**阶段 2（~10 个 unit = 10 条 dev 路线 × TM seed 0：`lmain`、`drive`、`dbase` 各一张卡同时跑，这也是 CARLA 打包的 profile 点）**：
- 每次运行有 results.json；agent 崩溃（status 含 Failed / crash）为 0；CARLA 服务端 rc 139 重试 ≤ 10% 的运行且最终都有结果。
- 数值范围：`dbase` 的 DS 在 op-drive 的 57.7 ± 15 内，`drive` 在 63.2 ± 15 内（并发重跑应复现 op-drive 的 drive；不在就先查环境与非确定性）。
- C2（intent 一致率 100%）、C5 在全部 10 条路线上重查；`lmain` 的 L-Safe 四项（阶段 2 的 seed 0 上）不违反；平均车速比 v(lmain) / v(drive) ∈ [0.5, 2.0]（否则配速对照的意义要另议）。
- 利用率与延迟如实记录（§7）。

阶段 1、2 不过的处理：停 chain，写 ERROR，修真 bug（只改 agent / server / 脚本，不改任何登记线与参数）后从阶段 1 重来；修复内容写进「执行日志」。

## 7. 工程与等价检查（已完成，无 CARLA 运行）

- `scripts/op_l_onnx.py`：`build`（checkpoint → ONNX + `E.npy`）、`ref`（训练用 PyTorch port 在 `$DATA_DIR/runs/op_adapt/ref/frames_{0,1}.npz` 的 WOD 帧流上算输出，intent 按 [1,1,2,2,3,3,0,1] 每 3 帧轮换）、`check`（onnxruntime 按 20 Hz 状态队列逐步跑同一流，取每帧第二步）。
- 结果（`lmain-s0`，2 条流各 134 个可比帧，前 16 帧因 context 未满不比）：plan 的 xy 与 port 的平均距离 0.011 m（最大 0.125 m，fp16 量化一格），与原模型 ONNX 对 port 的本底（0.010 m）相同；全部输出列最大差 0.25（logit 量级），p99 0.031。
  intent 的效应量是本底的 20–40 倍：同一帧流在 intent 置零与按模式喂入之间 plan 平均距离 intent 1 / 2 / 3 各 0.25 / 0.42 / 0.36 m，intent 0 为 0.011 m（= 本底），所以 adapter 确实在起作用，而且与 port 对得上。
- 服务器改动：`--onnx`（服务适配 ONNX）、`--no-twin`（不建 desire-free 双胞胎 session，session 数减半；所有臂的 config 里 `twin` 都是 false，对行为无影响）。`jevdrive/openpilot/model.py` 的 `OPModel` 接受 ONNX 路径并在队列模型上传 `intent_bias`。

## 8. held-out 的 dev 条件

仅当以下条件全部成立才运行 held-out（19 条 × TM seed 0、1，`drive`、`lmain` 与两者各自的配速链，只跑一次）：`lmain`（训练 seed 0）在 dev 上 L-S1 ∧ L-S2 ∧ L-Safe 成立，且 `lmain1`、`lmain2` 中至少一个的 ΔDS(X − slow_X) 同号（> 0）。
条件不满足：held-out 不碰，结果里写「未运行」。运行时：用同一 `set h <tag>` 与同一配速规则，判定线同 §5，CI 一并报，但不要求 CI 下界 > 0（19 条）。

## 9. 资源、排程与 wall 估计

- 三张卡 0、1、2；每卡 1 个 openpilot server（`--no-twin`，pool = worker 数）+ 6 个 CARLA worker（首选；阶段 2 的 profile 决定是否 5 / 7）；CARLA index 块 200–235（每卡 12 个，SCH 行 `op-l-b2d`）；核数在调度表里登记，不用 `pkill -f`，只按记录的 PID 停；每个进程 `OPENBLAS_CORETYPE=Haswell`；agent 进程 PID 控制在 17.5k 内（`B2D_PIDS_WAIT` 17000）。
- 一个 unit 的 wall：op-drive 的 dev f 同规格（6 worker / 卡，32 核 / 卡）实测 10–13 min / unit（10 条路线，drive / dlon / dbaseslow 各 10–12 min）。
- unit 数：dev 主线（`dbase` 2、`drive` 2、`lmain` 2、`lnoint` 2、三条配速链各 ≈ 4）= 20；`dnod` 2、`lkd` 2、`ldw10` 2 + 链 ≈ 4、`lmain1` / `lmain2` 各 2 + 链 ≈ 4 = 28；探索 `dtz` / `ltz` 各 2 = 4。共约 52 unit。
- 估计：52 × 12 min ÷ 3 卡 ≈ 3.5 h（卡之间无 CPU 争用时）；核数只有 cgroup 75 核（比 op-drive 时少），若 CPU 成为瓶颈按 1.3 倍算约 4.5 h；加阶段 1、2（≈ 40 min）与服务器换模型的启动开销。**总预算 ≤ 8 h wall**；held-out（条件满足时）另加约 2.5 h。
- profile：阶段 2 用 3 卡 × 6 worker 的初始打包，记录每卡的 CPU 核数使用、GPU 利用率与显存、每步 `ms`、unit wall（采样脚本 `scripts/op_l_b2d_util.sh`，写 `util.csv`）；利用率不足（GPU < 80% 且有空核）就调 worker 数。before = op-drive f 的数字（6 worker / 32 核 / 卡，twin 开，10–13 min / unit），after = 本 lane 阶段 2 与全批的实测，写进「执行日志」。
- 链脚本 `scripts/op_l_b2d_chain.py`（自推进，三张卡各一个 worker，按依赖与优先级取 unit，优先连续使用同一模型以省服务器重启）；hand-off 文件在 `$DATA_DIR/runs/op_l_b2d/`：`STATUS`、`DONE`、`ERROR`、`log.txt`、`events.jsonl`；每个 unit 结束自动写该 unit 的结果行。

## 10. 这次登记没有做的

- 不在 held-out 上调任何东西；不改判定线；不做 HUGSIM。
- 不把 dev 读成干净测试：dev 已在 op-drive 里看过；本 lane 的结论是「在 dev 上」的。
- op-adapt 的训练没有见过 CARLA 帧（训练域是 WOD 与 nuScenes）；闭环读数同时含域迁移，无法拆开；如实写。

## 偏离

- **D1（阶段 1 checklist C5 的 lane 条款，阶段 1 读数后、阶段 2 数据之前改）**：原文「lane 概率不退化（中位 > 0.3）」是我把 op-drive 诊断里「内侧车道线概率均值 0.54」（另一个量、另一批运行）当成了绝对门槛，而且代码里误对 4 条线取小。CARLA 场景里该概率本来就低：同一路线上 `drive` 自己的内侧两线取小的中位数只有 0.048，`lmain` 0.066，字面规则对基础模型同样判 FAIL，所以它区分不了适配模型是否退化。
  改成相对规则「`lmain` 的中位数 ≥ `drive` 的 0.5 倍」。`lmain` 0.066 ≥ 0.024，按新旧两种读法实质结论一致（没有退化）。这是 checklist 的措辞错误，不是放宽任何判定线。

- **D2（阶段 2 checklist 的两处措辞，阶段 2 读数出来后、任何判定读数之前改）**：(a)「agent 崩溃（status 含 Failed / crash）为 0」照抄自 op-drive，字面会把 B2D 的正常驾驶结局 `Failed - TickRuntime`（被施工障碍卡到超时，24497 / 37969，`dbase`、`drive`、`lmain` 三臂同值，与 op-drive 里四臂同被卡一致）与 `Failed - Agent got blocked` 当成崩溃，op-drive 自己的 dev 也会被判 FAIL。
  改成「崩溃 = status 为 missing / Failed / Simulation crashed / Agent crashed / Agent couldn't be set up」，结局类 status 另列。阶段 2 三个 unit 共 30 次运行，无崩溃，无 CARLA rc 139 重试。
  (b) intent 一致性检查用日志里的 `ri` 重建期望 intent，而 agent 在 `_plan` 开头算 intent、`ri` 在末尾记，两者之间路线进度索引可能前进 1 格，区边界附近 9 / 12 669 步（27043、27297、27870 各在区边界 0.05 m 内，`ri` 差 1）不一致。
  改成「日志里的 intent 等于 ri − 1、ri、ri + 1 三者任一处的期望值」；改后 0 / 12 669。这是检查的时序容差，agent 里的映射没有变。
- **D3（打包，阶段 2 的 profile 之后）**：见下面执行日志。unit 的 wall 与 worker 数、每卡 slot 数是调度参数，不影响任何登记读数。

## 执行日志

（运行开始后在此追加：时间、做了什么、读数、偏离。）

- 2026-10-01 07:22–07:26 **阶段 1**（tag `s1`，不进 dev 判定；路线 24944 seed 0，`lmain` 在卡 1、`drive` 在卡 0，各 1 个 worker，chain `--plan stage1`）。服务器起服务 3 s（pool 1，ORT cuda-iob，没有 TRT 引擎构建），单 unit wall 2.4 / 3.8 min。
  checklist：C1 通过（日志里 `lmain-s0.onnx`、E 载入）；C2 intent 与路线 0 / 1034 步不一致，339 步为左转 intent；C3 延迟 `lmain` 38.0 / 46.2 ms（中位 / p99）对 `drive` 37.4 / 44.3；C4 Completed，无重试；
  C5 自由路况 plan v(5 s) 中位 5.21 对 4.30（比 1.21，过）、\|y@2 s\| 中位 0.135 m、幻觉 lead 步占比 0.017 对 0.041、无 NaN；C6 字段齐。lane 条款的字面规则对 `drive` 自己也 FAIL，按偏离 D1 改成相对规则后通过。**阶段 1 通过，进阶段 2。**
  一条路线上的旁证（不是读数，不作判断）：`lmain` DS 100 / RC 100，锁存 1 次且由 plan 的 `signal` 放行；`drive` DS 70，锁存 1 次且由 timer resume 放行、1 次违规。

- 2026-10-01 07:27–07:41 **阶段 2**（tag `ld` / `lm`，dev 10 条 × TM seed 0，`dbase` 卡 0、`drive` 卡 1、`lmain` 卡 2，每卡 6 worker，no-twin；这三个 unit 就是 dev 的正式 unit，后面不重跑）。
  unit wall：`dbase` 10.9 min、`lmain` 11.5 min、`drive` 12.9 min（op-drive f：10–13 min，同规格，wall 的预估成立）。三张卡的 before（6 worker / 卡，22 核 / 卡；采样 15 s，阶段 2 稳态段）：

  | 卡（arm） | GPU 利用率均值 | 显存峰值 | 用核均值 / 峰值（共 22 核） |
  |:--|--:|--:|--:|
  | 0（`dbase`） | 29% | 47.1 GB | 6.3 / 12.1 |
  | 1（`drive`） | 49% | 45.1 GB | 10.0 / 12.7 |
  | 2（`lmain`） | 41% | 46.2 GB | 9.0 / 13.3 |

  利用率不足：每卡 GPU 只用了三到五成、核只用了一半，显存 45–47 GB（每个 CARLA 约 6.5–7 GB）是装载上限的主因（83.6 GB 卡上 6 个以上只能再加 2–3 个）。unit 的 wall 被被障碍卡住的长路线（24497、37969、9196 被卡到超时）拖长，后半段 worker 空闲。
  **打包改动（after）**：每卡 2 个 slot（各自的 openpilot server + 4 个 CARLA worker，核与 CARLA index 各分一半），一张卡同时装 8 个 worker（显存约 62 GB），一个 slot 被卡住的长路线占着时另一个 slot 继续推进；全批用 `--slots 2 --workers 4`，after 的利用率与 unit wall 在全批日志里记录。
  checklist（`op_l_b2d_report check stage2`，按偏离 D2 改后）：全过：每 unit 10 行；无崩溃；`dbase` DS 56.6（线 57.7 ± 15）、`drive` 62.5（线 63.2 ± 15）；intent 一致率 100%（0 / 12 669 步）；
  L-Safe：`lmain` 横向占比 0.486 对 `drive` 0.485、xt_op_med 0.067 m、off_lane + route_dev 2 对 2、blocked 0 对 1；速度比 `lmain` / `drive` = 2.56 / 2.05；C5：自由路况 plan v(5 s) 中位 6.22 对 5.68（比 1.10）、\|y@2 s\| 中位 0.091 m、内侧 lane 概率中位 0.243 对 0.128、幻觉 lead 步占比 0.205 对 0.317、无 NaN。**阶段 2 通过，启动全批。**
