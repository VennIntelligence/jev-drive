# op_parity / s2-gohold 预登记：System 1（openpilot 轨迹）+ System 2（Qwen3-VL-4B 离散决策）late fusion 的能力边界（2026-10-08，任何 oracle 臂、任何 Qwen3 决策、任何融合臂打分之前写定）

## 问题

第 164 条：WOD val 479 rater 帧上 WP2 距 top-rated 轨迹 1.476，一半是速度剖面（WP2 自己的路径配 top 速度 +0.722），停车 / 起步是最大一块；按速度档 / 前车的离线缩放 out-of-fold 拿不回，「哪些帧该走更远」是场景决策。wod-launch Step 1：红灯停着不丢分，最大的停车块是 stop sign 路口（45 帧），绿灯起步是 WP2 落后 shipped 的地方，停车缺口一半在底模。

**问的是**（在做任何联合训练之前）：
1. 一个离散的 go / hold 决策在这个榜上值多少 RFS（特权上限，只作 oracle 探针）；词表要多粗才够（2 / 3 / 5 类）。
2. zero-shot Qwen3-VL-4B 能交付其中多少。
3. 每次决策的延迟。

System 1 是唯一的轨迹生成器；System 2 只输出一个离散符号，融合 = 对 System 1 自己的路径做纵向重定时（`pp_wod_diag.retime`）。

## 不重测的既有结论（只引用）

第 84 / 85 / 86 / 91 条（Qwen3-VL-4B zero-shot 读灯 93–94%，68–125 ms / forward，分解问题优于单条指令）、第 89 条、第 43 / 102 条、第 163 / 164 条的存档预测与口径、wod-launch Step 1 的场景标签（`results/wod_launch/frames.csv` 的 `context` 列，原样使用作分层；lead 一题 52.5% 不可靠、cross 召回 4 / 11 的限定照搬）。

## 口径

479 rater 帧，cluster 平均 RFS，按序列配对 bootstrap（B 4 000，`wod_launch_report.Ctx`）；WP2 = 两 seed 逐帧得分平均；System 1 另报 shipped。分层：`stopped`（v0 < 0.5 m/s，120 帧）、`SL` / `SS`、`moving`（v0 ≥ 0.5）、全部；stopped 内再按 wod-launch 的 context（red / green / stop_sign / lead / cross / open）。

## 词表与码本（只用 WOD train 日志拟合，不看 val）

对任一 5 s 轨迹：s(t) = 沿轨迹弧长（0.25 s 一步，20 步），v0 = 该帧的 metric 速度。

- **二元基类**（两个 regime 语义相同：hold = 5 s 末处于静止）：
  - 停车 regime（v0 < 0.5）：`hold` = s(5) < 2.0 m，否则 `go`。2 m 的依据：RFS 在 v0 < 1.4 m/s 时纵向信任域半宽 3 s 处 2.0 m、5 s 处 3.6 m，两条 hold 类轨迹必然互相落在信任域内。
  - 行进 regime（v0 ≥ 0.5；v0 分档 0.5–2 / 2–5 / 5–8 / 8–12 / ≥ 12 m/s）：`hold`（停下 / 让行）= s(5) − s(4) < 0.5 m（最后 1 s 平均速度 < 0.5 m/s），否则 `go`。
- **K 类词表**：V2 = {hold, go}；V3、V5 把 `go` 再按 train 日志的分位数切成 2 / 4 段（V3 中位数，V5 四分位）。切分键：停车 regime 用 s(5)，行进 regime 用等效平均加速度 a5 = 2 (s(5) − 5 v0) / 25（每个 v0 档各自的分位数）。V3 在停车 regime 读作 hold / creep / go。
- **原型剖面**（每个 regime 档 × 类一条，来自 `wod/r2-train` 的日志未来）：停车 regime = 类内 s(t) 的均值；行进 regime = 类内 Δ(t) = s(t) − v0 t 的均值，解码 s(t) = cummax(clip(v0 t + Δ(t), 0))。V2 停车 regime 的 `go` 原型就是「train 日志上拟合的固定起步剖面」。
- 分位数边界、原型全部只由 train 行得到；val 的 rater 标签只用于 oracle 决策与评测。

## 融合规则

给定 System 1 的计划 p 与决策类 c（词表 K）：
- **gate（主）**：p 自己按同一规则落在类 c → 原样输出 p；否则把 p 的路径重定时到类 c 的原型剖面（路径不够长时沿末端方向外推，同第 164 条 O1）。
- **replace（次）**：不论 p 自己的类，一律重定时到原型。
- 作用范围：**standstill-only**（只在 v0 < 0.5 的帧上用决策，行进帧原样；主）与 **all**（行进帧也用）。

## 1. Oracle 上限（特权：决策来自 val rater 轨迹；只是上限，不是方法）

- **top-rated 决策（主）**：c = top-rated rater 轨迹按上面规则所属的类。报 K = 2 / 3 / 5 × {gate, replace} × {standstill-only, all} × System 1 ∈ {WP2, shipped}，在 stopped / moving / all 上的 d RFS（对 System 1）与 CI。
- **best-of-K（次，码本上界）**：每帧取使 RFS 最大的类。
- 参照行：连续上限 = System 1 路径配 top-rated 速度（O1，第 164 条 +0.722 / stopped +1.068）；System 1 自己的类与 oracle 类的一致率（System 1 已经隐含做对了多少决策）；日志的类与 oracle 类的一致率。
- 读法：V2 gate 在 stopped 上的上限 ÷ 连续上限 = 二元决策能拿到的份额；V3 / V5 的增量 = 更细的词表多拿多少。

## 2. zero-shot Qwen3-VL-4B（不训练；提示词与本文件一起提交，之后不改）

- 模型与打分：box 上 HF 缓存的 Qwen3-VL-4B-Instruct，强制 `ANSWER:` 之后第一个 token 在选项上打分取 argmax（第 84 条 / wod-launch 的做法，`vlm_arb` 的同一套）。输入图是 Alpamayo JPEG 导出（`t2_jpg`，FRONT_LEFT / FRONT / FRONT_RIGHT，即 openpilot 三前视的来源相机），处理器默认分辨率。
- **分解问题**（479 帧全问；提示词见 `scripts/s2_gohold.py` 的 `QS`）：`light`（FRONT；wod-launch 原题）、`stop_ctrl`（FRONT + FRONT_RIGHT；wod-launch 原题）、`row`（三前视；谁先行：others_first / ego_may_go，新题）、`lead`（FRONT 的 t0 − 1 s 与 t0 两帧；wod-launch 原题，已知不可靠，照问）、`cross`（FRONT；wod-launch 原题）。
- **决策**（四张图：FRONT t0 − 1 s，FRONT_LEFT / FRONT / FRONT_RIGHT t0；文字里给 ego 当前是静止还是约多少 km/h）：
  - `D-decomp`（**主**）：把五个分解答案写成文字观察附在提示里，再问 hold / go。
  - `D-direct`（次）：同样四张图，单条指令直接问 hold / go。
  - `D-rule`（次，无最终 VLM 步）：light = red_or_yellow，或 lead = stopped_lead，或 cross = crossing，或 row = others_first → hold；否则 go。
  - `D-decomp3`（次，只在 stopped 帧）：hold / creep / go 三选一，对 V3。
- **一致率**：stopped 帧上各决策对 oracle V2 类的准确率、go 召回、hold 召回，按 context 分；同表给 WP2 / shipped / 日志自己的类。moving 帧只报总数。
- **融合 RFS**：Qwen3 决策按上面的 gate / replace、V2 码本作用于 WP2（及 shipped）；hold → hold 原型（原地），go → train 拟合的起步剖面。对 WP2 单独、对同一融合规则下的 oracle V2，报 stopped / moving / all 的 d RFS 与 CI，stopped 内按 context 分。
- **阈值变体（次）**：`D-decomp` 的 p(go) 阈值在 {0.1 … 0.9} 上按序列 5 折 × 20 次 out-of-fold 选（目标 = stopped 帧的融合 RFS），报 out-of-fold 与 in-sample。除此之外没有任何参数在 val 上拟合。

## 判读（打分前写定）

- **System 2 delivers**：主臂（`D-decomp`，V2，gate，standstill-only，System 1 = WP2）在 stopped 上 fused − WP2 的 CI 下界 > 0；并且在 all 范围版本里 moving 上不丢（CI 含 0 或在 0 以上，且点估计 ≥ −0.05）。standstill-only 版本在行进帧上按构造与 WP2 相同，所以「行进帧不丢」由 all 范围版本检验；若 stopped 达标而 all 范围在 moving 上丢，写成「只在按 ego 速度门控到停车帧时 delivers」。
- stopped 上点估计 > 0 但 CI 含 0 → 弱 / 未定；CI 上界 < 0 → 有害。
- **上限判读**：oracle V2 gate 在 stopped 上的 CI 下界 > 0 → 二元决策有值；oracle 上限本身 CI 含 0 → 这个榜上 go / hold 决策不值分，Qwen3 的结果无论如何都读作 not。
- 多重比较：3 个词表 × 2 融合 × 2 范围 × 2 System 1 的 oracle 表是描述性的；Qwen3 只有一个主臂，其余决策变体与阈值变体都标次。

## 3. 成本

每次 forward 的预处理 ms 与模型 ms（cuda synchronize，去掉前 10 帧），按问题报中位数；`D-direct` 一次决策 = 1 次 forward，`D-decomp` = 5 + 1 次。

## 图（选帧规则打分前写定）

stopped 帧 8 张，每张：Qwen3 看到的 FRONT 图、openpilot 实际输入的 road / wide 帧、Qwen3 各题答案、BEV（WP2 / 融合后 WP2 / oracle V2 融合 / 三条 rater 轨迹 / 日志）。选帧（主臂 fused − WP2 的逐帧差）：stop_sign 与 green 各取增益最大与损失最大的一帧；red 取损失最大的一帧（都无损失则取 p(go) 最高的一帧）；lead 取损失最大；open 取增益最大与损失最大。

## 预算与实现

- Qwen3 一个 pool 作业（约 8 次 forward / 帧，估计 20–30 min 一卡，`--vram 14`）；码本拟合、oracle、报告、图都是 CPU。合计 < 1 卡时。
- 代码：`scripts/s2_gohold.py`（fit / qwen / report / figs）；复用 `pp_wod_diag.retime`、`wod_gap.arc`、`wod_launch_report.Ctx`、`wod_launch_figs.rgb`、`wod_launch.py` 的提示词与打分方式。不训练任何模型，不提交 test，不改存档预测，不动其他线的文件。

## 限定（预先写明）

开环；479 帧，停车分层 120 帧，context 格子 11–45 帧；oracle 决策是特权信息；融合在静止计划上的「go」是沿正前方直线外推，测的是剖面而不是 WP2 生成的路径；top-rated 轨迹里 39 条在 5 s 前截断（metric 用末点 pad），行进 regime 下它们按规则落入 hold，属标签产物；Qwen3 的图是 JPEG 导出而非 openpilot 的 YUV 模型帧；场景 context 标签来自同一个 VLM 的 zero-shot 答案。

## 补记 B（2026-10-08，协调者转达的范围变更；写于任何打分之前：上文的纵向部分此时也还没有打过分、没有问过 Qwen3）

范围变更：System 2 的目标不是纯纵向的 go / hold，而是横纵联合的机动及其时机（何时绕障、何时在灯前起步）。上文的纵向词表保留为答案里的一行；本线扩展如下，预算提到约 2 卡时。上文的口径、约束（不训练、不提交 test、特权只作上限）不变。分层在原有基础上加 `turn`（intent 左 / 右，52 帧）与 10 个场景 cluster（Cut_ins、Foreign Object Debris、Construction、Special Vehicles、Cyclist、Pedestrian、Interections、Multi-Lane / Single-Lane Maneuvers、Others）。

### B1. Oracle 上限：System 2 从 K 条候选里选一条（路径 + 速度联合）

候选集全部不用 val rater 标签构造：

- **(i) System 1 自己的输出**：`S1x2` = {WP2, shipped}；`S1x5` = {WP2 s0, WP2 s1, shipped, WP1 s0, WP1 s1}。存档预测里 plan 头只有一条假设（`plan_pos` 单条 + 方差），没有其他假设可用，照此写明。
- **(ii) train 日志的轨迹词表**：`wod/r2-train` 日志未来（ego 系，5 s，20 × 2）上的 k-means（sklearn，seed 0，30 万行子样本）锚点，K = 8 / 16 / 32 / 64。两种：`KM{K}` 全局一套；`KMv{K}` 按 v0 档（上文 6 档）各一套，选择时只看本帧所在档的 K 条。
- **(iii) 因子集**：WP2 的路径族 × 速度剖面。路径 5 条：`keep`（WP2 自己的路径，不够长时沿末端方向外推）、`nudge` 左 / 右（横移 1.2 m，在弧长 max(10 m, 2 s × v0) 内 smoothstep 过渡）、`lane` 左 / 右（横移 3.5 m，过渡弧长 max(20 m, 4 s × v0)）。速度：`follow`（WP2 自己的剖面）+ 上文码本的原型：`F20` = 5 × (follow + V3 的 hold / creep / go)，`F30` = 5 × (follow + V5 的 5 类)。

每个集合，oracle 每帧取 RFS 最高的候选（特权上限）。报：集合单独的 best-of-K RFS；与 System 1 = WP2 并集后的增益（d RFS 对 WP2，CI），在 all / stopped / turn / 各 cluster 上。

**纵向 / 横向 / 联合分解**（每个集合、同一批候选 c，基准计划 p = WP2）：纵向选择 = p 的路径配 c 的弧长剖面；横向选择 = c 的路径配 p 的弧长剖面；联合 = c 本身。各自取每帧最优（都含 p 自己）。报三个增益、`either` = 逐帧 max(纵向, 横向)、`needs joint` = 联合 − either（CI）。参照行：top-rated 轨迹本身、第 164 条的连续 O1（速度）/ O2（路径）。读法：联合上限里纵向单独拿到的份额、横向单独拿到的份额、必须联合的份额，对照第 164 条的 49% / 20% / 其余。

上限判读：某集合「值得做选择器」= 并集增益在 all 上 CI 下界 > 0；集合之间只作描述性比较。

### B2. zero-shot Qwen3-VL-4B 作选择器

- **候选集固定为 `F20`**（现在就定，不看 oracle 表再挑）：它是唯一既能画（路径）又能说（速度）的集合。若 B1 显示某个 K ≤ 16 的 k-means 集合在 all 上的上限比 `F20` 高 0.2 以上，只在报告里记下，不换集合。
- **路径选择（set-of-mark）**：把 5 条路径投影画在 t0 的 FRONT 图上（相机内外参 `op_calib.json`；从车前 6 m 画到 clip(max(WP2 的 5 s 弧长, go 原型的 5 s 距离), 15, 50) m；细线、不填充、不遮挡；颜色避开路面标线色：A 品红 = lane 左，B 青 = nudge 左，C 绿 = keep，D 橙 = nudge 右，E 蓝 = lane 右；字母标在线的远端）。画的是 WP2 seed 0 的路径族，选择同样作用于 seed 1。提示词 `PATH_Q`：附分解观察，问选哪条。叠加只用 System 1 自己的计划与固定偏移，不用地图、不用 rater 标签。
- **速度选择（文字）**：四张图 + 分解观察，`SPEED_Q` 四选一 hold / creep / follow / go，映射到 V3 原型（follow = WP2 自己的剖面）。
- 分解问题在上文五题之外加一题 `block`（三前视：ego 车道前方是否被障碍 / 停着或慢的车 / 施工占用，需要向左或向右绕）。新加的观察只进入 `PATH_Q` / `SPEED_Q`，上文 `D-decomp` 的提示不变。
- **读数**：(a) top-1 一致 = Qwen3 选的候选达到该帧 `F20` 的 oracle 最高分（并列都算），同表给「什么都不改」（WP2 自己）的一致率；路径、速度各自的一致率；(b) 所选候选的 RFS 对 WP2 的 d 与 CI：联合、只用路径选择（速度 follow）、只用速度选择（路径 keep），在 all / stopped / moving / turn / 各 cluster 上；(c) 对 `F20` oracle 的差距。
- **判读（主臂 B）**：Qwen3 联合选择在 all 上对 WP2 的 CI 下界 > 0 → selector delivers；点估计 > 0 且 CI 含 0 → 弱 / 未定；CI 上界 < 0 → 有害。各 cluster 的 CI 只作「哪里帮、哪里伤」的描述。没有任何参数在 val 上拟合。
- 上文的主臂（`D-decomp` 的 standstill go / hold）保留为主臂 A，判读规则不变。

### B3. 时机

WOD 每个序列只评一帧，时机只能表现为所选的速度剖面（何时起步 = 剖面的形状）；本线不测时机本身，报告里写明，并指出哪种读数能测（HUGSIM 闭环 / navtest 的多帧序列上逐帧决策的切换时刻对日志或对碰撞 / 进度的差），不跑。

### 图（补充）

上文的 8 张 stopped 帧图里 Qwen3 的图改为带 5 条路径叠加的 FRONT 图，BEV 加 Qwen3 的联合选择与 `F20` 的 oracle 最优。另加行进帧 6 张：Qwen3 选了非 keep 路径的帧里，联合选择对 WP2 增益最大的 3 帧与损失最大的 3 帧（不足 6 帧时用速度选择非 follow 的帧补）。

### 预算

Qwen3 一个作业约 11 次 forward / 帧（估计 30–40 min 一卡）；k-means、oracle 选择表（约 300 条候选 × 3 种分解 × 2 seed × 479 帧的 RFS）、报告都是 CPU。
