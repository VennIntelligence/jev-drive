# NR1 预登记补记 21（TURN-CAP：急弯里「别比人的够得着弧长快太多」的单边上限；从 B2 出发）

**状态：已登记，2026-10-20 01:40 CST（box 时间）。** 登记时还没有任何 TURN-CAP 标签（`turn_cap.npz`）、闸输出、定标、训练或臂读数；盘上只有对照（pilot B2
`STANDFMXH1K-L18-s0`）的存档读数、B2 全长两个 seed 的存档大考场读数（第 391 条）与第 395 条的诊断。主会话转达用户夜间授权：写登记 → 实现与核对 → 离线闸
→ 最多两条 1 000 步剂量 pilot 与读数（池里排队，不指定卡）；第 1 段只读，第 2 段只给过线的臂；**不跑全长**：pilot 过了就报告，早上由用户定是否把 TURN-CAP 与
STAND-R（补记 20，另一个代理在跑，本线不碰它的文件与作业）合成一条联合全长臂（21.6 先把命令写好）。
**结果（10-20 01:24 box 时间，登记之后）：离线闸不过，本线停。** G1 0 / 0、G2 36 / 41 = 88 %（9 / 9 窗）过；**G3 不过**：B2 s1 得 1 分的 held 窗里
5.9 % 的决策、11.7 % 的窗触发（线 1 % / 5 %）。没有训练、没有读数，排队的同一性 / 定标 / 对照转弯集读数已取消，GPU 0 卡时。详见
[results/turn_cap/](../results/turn_cap/README.md) 与第 396 条；21.6 的联合臂草稿因此不适用（TURN-CAP 没有过线的臂）。

代码：`lib/nr1_free.py`（`turn_range` / `turncap_rows` / `turncap_cap` / `turncap_load` / `turncap_term`）、`lib/nr1_lat.py`、`rm_train --on-turncap W`、
`scripts/turn_cap.{py,sh}`；结果 `results/turn_cap/`，盒上 `$DATA_DIR/runs/defect_fix/turn_cap/`。

## 21.0 为什么是这一类（第 395 条）

- B2（`STANDFMXH-L18-s1`）剩下的易题侧向零分里，**a2 急右转过速 9 / 9 窗**（s1：098f5c53、202f583c、3f805a18、476b54f9、6f27563e、87705868、9e7c5a46、
  efe356db、f2305669；s0 另有 41a358be）：28–99° 右转、R 约 7–10 m、人 2–4 m/s，自车同站点 1.1–1.65 倍人速、转弯中途提前加速，8 / 9 从外侧出。
- 执行不是原因（1 s 横向误差 2–3 cm）；**plan 形状随速度变宽**（等弧长 8 m 处外偏 0.11 → 0.3 → 0.7 → 1.2 m，速度比 ≤ 1.05 → > 1.4）；
  急转窗侧向零分率随速度比 7 % → 45 %。不是 FREE-M 造成的：JTN-s1（无 FREE-M）在同一批窗速度比同为约 1.2、9 窗挂 7 窗，是底座的习惯。
- 杠杆的估值：封顶 +27 [+6, +70] board-mix / +49 g15（s1），按零分率比的期望约 5.9 窗 +17 / +30。估值是观察性的（急弯可能同时带来高速比与零分）。
- 风险（第 352 / 353 / 365 条）：每一个单向纵向信号都沿共享的低速轴漏到起步、等待与慢窗。本项只在「自车 ≥ 2 m/s、人在动、人的路径急弯」的行上，
  按构造碰不到站立 / 起步状态，但会不会泛化成「整体变慢」（第 386 条 STOPPT 臂 C：k = 9 速度比 0.841 → 0.770）由第 1 段护栏判。

## 21.1 项 TURN-CAP（规则写在任何标签统计之前，读数后不改）

- **行**（load 时并进 `ok`，无梯度）：池的 S 行（`mode == self`），`fut` / `fut_reg` 有限，且全部满足：
  (a) 自车 t0 速度（池 tab `speed`）≥ V_TC = 2.0 m/s（在动；站着 / 起步 / 蠕行一律为 0）；
  (b) 人在动：人的 4 s 记录弧长 arc(`fut_reg`)_8 ≥ H_TC = 4.0 m（平均 ≥ 1 m/s；人停下 / 站着的行归 STAND / STOPPT，不在这里）；
  (c) **急弯**：人的日志路径在自车站位 s_e 的 [s_e − 10 m, s_e + 20 m] 上（rows meta `gpath`，1 m 采样）各段航向（展开后）的极差 ≥ TURN_TC = 45°。
  左右转都算（第 395 条左转在速度比 > 1.2 时零分率也到 26–50 %），右 / 左分开描述。
  为什么往回看 10 m：任务书的「人路径 20 m 内转 ≥ 45°」只看前方时，自车已在弯中（出界前 3 s 的决策正是这样）剩下的转角会小于 45°，项在最该起作用的地方失效；
  90° 弯 R 8 m 的弧长 12.6 m，30 m 的窗盖住整段弯。这是登记前的选择，不是读数后的调整。
- **上限**：c_j = K_TC × max(arc(`fut`)_j, arc(`fut_reg`)_j)，K_TC = 1.05，j = 0.5 … 4 s。`fut` = 人的速度曲线从访问状态的车速在 2 s 内过渡过去（nr1_lon 的目标），
  所以自车比人快时上限给它 2 s 慢下来；取两条人路径的大者，**永远不要求比人的弧长短**（两条人路径当 plan 按构造都不触发）。
  例：自车 4 m/s、人 3 m/s 匀速：arc(`fut`)_8 = 13 m，c_8 = 13.65 m；保持 4 m/s 的 plan 16 m 触发；同站点速度比 ≥ 约 1.07 的匀速 plan 才触发，
  提前加速的 plan（第 395 条的样子）更早触发。
- **触发与值**：s_plan_8 > c_8 时，逐步 Σ_j smooth L1(β 0.5) of relu(s_plan_j − c_j)（= `stand_terms` 换这个上限）；s_plan 按 `nr1_lon.arc(lon_only(...))`，
  只沿各点自身航向移动，航向与横向无梯度（只改快慢，不直接改形状；形状随速度变宽是模型自己的耦合，第 395 条）。
  经 nr1_lat 记 `on/lon_turncap_ho`（10 × W × 项，乘 ho_w、按 n_imit 归一，同其他 NR1 项）与 `on/lon_turncap_pos`；`--diag` 记 `grad/lon_turncap`。
- **开关**：`rm_train --on-turncap W`，默认 0 = 不读 `turn_cap.npz`、不加任何运算；nr1_lat 只在 W > 0 时多一个键。
- **标签**：`turn_cap.py label --pool on1_JTX`（CPU）对 S 行写 `$CR/on1_JTX/turn_cap.npz`（names 与 tab 逐行相同、`turn` 极差、`net` 净转角，左正；
  trainval 行 NaN）；`gpath` 来自各部分 rows meta（L18RB0 / L18RTB1 / L18YS / L18Y / PN0–PN6），按 meta 速度与 tab 速度逐行对齐核对。
  同时记录（只描述）：域内行数（右 / 左）、占 S 行与 JTX00 抽样的份额、域内存档 served plan 的触发份额与 plan / 人弧长分位。

### 代码（实际实现，`lib/nr1_free.py`）

```python
V_TC, H_TC, TURN_TC, K_TC, TC_BACK, TC_AHEAD = 2.0, 4.0, 45.0, 1.05, 10, 20     # supplement 21 (TURN-CAP)


def turn_range(path):
    """path (n, m, 2) at 1 m over [s_e - TC_BACK, s_e + TC_AHEAD] -> heading range (deg), net change (deg, left +)."""
    P = np.asarray(path, np.float64)
    d = np.diff(P, axis=1)
    sl = np.hypot(d[..., 0], d[..., 1])
    h = np.where(sl > 0.2, np.arctan2(d[..., 1], d[..., 0]), np.nan)
    for j in range(1, h.shape[1]):
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j - 1], h[:, j])
    for j in range(h.shape[1] - 2, -1, -1):                            # a clamped start: take the first real heading
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j + 1], h[:, j])
    bad = np.isnan(h).any(1) | ~np.isfinite(P).all((1, 2))
    h = np.degrees(np.unwrap(np.nan_to_num(h), axis=1))
    rng, net = h.max(1) - h.min(1), h[:, -1] - h[:, 0]
    return np.where(bad, np.nan, rng), np.where(bad, np.nan, net)


def turncap_rows(v0, hum4, turn):
    hum4, turn = np.nan_to_num(np.asarray(hum4, np.float64), nan=-1.0), np.nan_to_num(np.asarray(turn, np.float64), nan=-1.0)
    return (np.asarray(v0) >= V_TC) & (hum4 >= H_TC) & (turn >= TURN_TC)


def turncap_cap(fut, fut_reg):
    a, b = cum_arc(fut), cum_arc(fut_reg)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return (K_TC * np.nan_to_num(np.maximum(a, b))).astype(np.float32), ok


def turncap_term(poses, rows, pack):                 # pack = turncap_load(...): cap (N, 8), ok = fin & S & turncap_rows(...)
    return stand_terms(poses, pack["cap"][rows], pack["ok"][rows])
```

- **自检**（`python nr1_free.py`，追加，已过）：直路极差 0、R 8 m 90° 右转极差 ≈ 90°、净转角右负左正、路径起点被截时不出假转角；行限制（自车 1.5 m/s、人 3 m、
  转 40°、NaN 各为否）；上限 = 1.05 × 两条人弧长的大者（自车 4 → 人 3 m/s：13.65 m）；16 m 的 plan 触发、13 m（人的够得着弧长）不触发；两条人路径不触发；
  非行为 0；梯度只沿航向、使 plan 变短、航向无梯度。

## 21.2 剂量（两档，定标规则写在定标之前）

- 稀有行的项按固定 W 往往不起作用（第 386 条 STOPPT 臂 A：触发只占池抽取 0.43 %，过点零分 11 → 11）。所以按「总拉力」定标，与第 388 条 STAND 的做法相同：
- **定标作业**（同时是冒烟）：pilot B2 的命令（`cmd_B2.txt`，tag 换成 `TCDOSE`）+ `--on-turncap 0.004 --steps 100 --diag-every 1 --diag-freeze`，eager：
  权重停在初始化，每步量各加权项对主干的梯度范数。W_eq = 0.004 × mean(`grad/lon_free`) / mean(`grad/lon_turncap`)（FREE-M 在命令里本来就是 B2 的 0.001）：
  TURN-CAP 在急弯行上的总拉力 = B2 的 FREE-M 在在动行上的总拉力。理由：B2 的 FREE-M 剂量已证明 1 000 步内能明显改闭环速度，而 TURN-CAP 是它在转弯里的反向镜像。
- **两臂**：T1 = W_eq（两位有效数字，夹到 [0.002, 0.04]）；T3 = 3 × T1（剂量响应；稀有行上 1 倍可能不动，3 倍看漏不漏）。不做因子设计。
- 冒烟判据：所有 `loss/*` 有限、`on/lon_turncap_pos` 某步 > 0、W_eq 有限；不过则不训练。夹到边界时照记（`dose.json` 的 `clipped`）。

### 命令 diff（`turn_cap.sh check` 核对）

```
T1  TCAPA1K-L18-s0 : --tag STANDFMXH1K-L18-s0 -> TCAPA1K-L18-s0 ; + --on-turncap W_T1
T3  TCAPB1K-L18-s0 : --tag STANDFMXH1K-L18-s0 -> TCAPB1K-L18-s0 ; + --on-turncap W_T3 (= 3 x W_T1)
```
其余逐词等于 pilot B2（STAND 0.0068、FREE-M 0.001 / move / 0.95、on1_JTX、JTX00、1 000 步、warmup 50、seed 0）。

## 21.3 离线闸（CPU，先跑；参数读数后不改；G1–G3 全过、G4 相同才训练）

闭环状态按 `free_gate.py` 的约定（场景时刻 t ↔ 日志帧 start_idx + round(20 t)；控制器轨迹刚体对齐到 start_idx 处人的位姿；人的 4 s 原始行程与按控制器速度
2 s 过渡的够得着版本；上限取两者之大 × 1.05，同训练器）；转弯标签 = 日志折线在 [s_e − 10, s_e + 20] m 的 1 m 采样航向极差，s_e = 自车位置在日志折线
（帧 f − 200 … f + 400）上的投影；plan = drive.jsonl 的 `poses`。读数 = B2 全长两个 seed 的存档大考场 held（`jt/arena/STANDFMXH-L18-s{1,0}@off`，18 块）。

1. **G1 人路径误触发 = 0**：on1_JTX 的域内行，以人的两条 4 s 路径（`fut`、`fut_reg`）当规划，触发数为 0（按构造，抓代码错）。
2. **G2 覆盖目标**：B2 s1 的 9 个目标窗，出界前 3 s 内的决策（k ≥ 2、t ∈ [t_ev − 3 s, t_ev)，t_ev 取 `lat_zero/windows.csv`），用存档 plan 打分：
   合并触发率 ≥ 50 %，**且** 至少 6 / 9 窗有 ≥ 1 次触发。如实先写下：202f583c（34°）与 87705868（38°）的人路径转角在 45° 以下，按定义大概率不在域内，
   所以线放在 6 / 9。s0 的 a2 窗（含 41a358be）同法只描述。
3. **G3 不碰成功的驾驶**：B2 s1 得 1 分的 held 窗，全部决策（k ≥ 2）里触发的份额 ≤ 1 %，且有任一触发的窗 ≤ 5 %。按构造，转弯域以外（直路、站着、起步、
   自车 < 2 m/s）触发为 0；另报：触发窗按层、域内决策的触发份额、第 395 条急转窗集里通过 / 失败窗的触发窗数、不在急转窗集的触发窗数。
4. **G4 开关关掉逐位相同**：pilot B2 的命令 60 步 eager，本补记之前的 HEAD（`3a0537ad`）对新代码（不开 `--on-turncap`）→ `results/turn_cap/identity.json`
   （`df.py bitcmp`）；不相同则 pilot 不起。

闸不过：停，不训练，结论写进结果与 decisions.md。

## 21.4 1 000 步 pilot（1 seed，对盘上的 `STANDFMXH1K-L18-s0`）的两段规则

**读数集**：第 395 条的急转窗只有 1 个（098f5c53）落在 STAND-R 用的 6 块子集里，6 块上对照的「转弯窗易题零分」是 0，量不出目标。所以第 1 段读
**子集 6 块（chunk 0 / 3 / 6 / 9 / 12 / 15）+ 转弯集**：转弯集 = 第 395 条的 169 个急转 held 窗（`tmp/lat_zero/sharp_windows.csv`）∪ 9 个目标窗 = 170 窗
（`results/turn_cap/turn_set.csv`），其中不在子集块里的 113 窗按 cache shard 分 6 个小读数（`turn_cap.sh tread`，同 `jt.sh aread` 的 serve，只换窗表）。
对照的 6 块用存档，转弯集在本线补读（同一 serve、原生）。另加：held 460 @18、等待集 v2 全集（`wait_set.sh readc`），全部原生（不加 hold、不用 stop3）。

**第 1 段**（`turn_cap.py stage1`，Mac；易题 = garage = 1，成因按 `easy_items.classify`，第 390 条同一套先匹配规则）：

| # | 读数 | 线 |
|:--|:--|:--|
| 主 1 | **目标类易题零分**：转弯集里的易题侧向零分（成因为「转弯窗 offroad / 走廊」「直路 offroad」「在动走廊」三类之一） | ≤ 对照 − max(2, ⌈0.25 × 对照⌉) |
| 主 2 | 读数集（子集 ∪ 转弯集，各读数都有的冻结 board-mix 题）PCS（`pcs_local` 冻结题，未读的窗不进似然） | ≥ 对照 + 10 |
| 护 1 | **其余每一类易题失分**（包括转弯集以外的侧向零分、红灯线、起步、差一点、停住、棘轮、追尾） | 每类 ≤ 对照 + 2 |
| 护 2 | held 460 零分 / held 460 前向碰撞 / 读数集 AF-front 窗 | 各 ≤ 对照 + 2 |
| 护 3 | 读数集慢部分分窗（type `slow`）/ k = 9 同站点速度比 | ≤ 对照 + 2 / ≥ 对照 − 0.03 |
| 护 4 | 等待集 v2 a 变体有前车两类等住率 | 各 ≥ 对照 − 0.03 |
| 护 5 | 等待集 v2 b 变体有前车两类：中位起步延迟 / 起步数 | 各 ≤ 对照 + 0.15 s / ≥ 0.95 × 对照 |

「降低」用 max(2, 25 %) 而不是 1：第 393 条里类间 1–2 窗在 seed 噪声内。对照的目标类数若 ≤ 1，主 1 不可能过，按 FAIL 报（写清「对照上目标类太少，量不出」）。
只描述：9 个目标窗逐窗分数、修好 / 新增的目标窗、6 块子集上的 PCS（与 STAND-R 可比）、读数集均分与零分、held 460 均分配对、闭环右漂、k 分速度比、分层、
训练中的 `on/lon_turncap_pos`、dev ADE。第 1 段任一条不过 → 停，写清漏在哪一类，第 2 段不读。

**第 2 段**（第 1 段过了才提交）：PAI 281 半读（c000 / c002 / c004 / c006，第 366 条 serve、ACC + GAP、g10）：臂对对照的配对均值 ≥ −0.01，零分 ≤ 对照 + 2。
对照的 PAI 半读用 STAND-R 已提交的同一读数（`runs/defect_fix/stand_r/pai/STANDFMXH1K-L18-s0`，同一组 chunk、同一 serve；本线只读，不碰它的作业）。

**判读**：一臂两段全过 = PASS，报告并停（不跑全长）。两臂都过时选 T1（剂量小、漏的风险小），除非 T3 的主 1 多修 ≥ 2 窗而护栏同样过。两臂都不过 → 写清哪条轴漏了。

## 21.5 成本（同类作业的实测 wall 时间估）

| 段 | 作业 | job 小时 | 卡小时（nuPlan 读数 4 个 / 卡；PAI 1 个 / 卡，上界） |
|:--|:--|--:|--:|
| 标签 / 闸 / 同一性 / 定标 | CPU + 2 × 60 步 + 100 步冻结 | 0.5 | 约 0.2 |
| 对照转弯集 | 6 个小读数（113 窗） | 约 0.7 | 约 0.2 |
| 第 1 段 × 2 臂 | 训练 0.05 + 子集 6 × 0.25 + 转弯集 0.7 + held 6 × 0.20 + 等待集 8 × 0.105 | 2 × 4.3 = 8.6 | 约 2.3 |
| 第 2 段（每个过线的臂） | PAI 4 × 0.39 | 1.6 | 约 1.6 |
| **合计** | | **约 10–13** | **约 3–4.5** |

## 21.6 联合全长臂（草稿；只作记录，不启动；早上由用户定）

若 TURN-CAP 的一臂 PASS、且 STAND-R（补记 20）的一臂 PASS，联合臂 = B2 s1 的全长命令（`runs/defect_fix/stand_b/cmd_FHS.txt`：9 500 步、warmup 500、`--seed 1`、
STAND 0.0068、FREE-M 0.001 / move / 0.95、on1_JTX、JTX00）只改这些词：

```
STANDRTC-L18-s1 : --tag STANDFMXH-L18-s1 -> STANDRTC-L18-s1
                  + --on-stand-roll 3.0                         (STAND-R; R2 选中时再加 --on-stand 0.0068 -> 0.0136)
                  + --on-turncap <W of the passing TURN-CAP arm>  (results/turn_cap/dose.json W.T1 或 W.T3)
```
- 两项的行不相交（STAND-R：人整段站着、自车 < 3 m/s；TURN-CAP：人 4 s ≥ 4 m、自车 ≥ 2 m/s、急弯），梯度只沿航向，可以直接相加；但两项没有一起 pilot 过，
  联合臂第一次读数就是它们的交互。
- 读数与 B2 全长相同（原生）：大考场 held 18 块、held 460 @18、PAI 281 ACC + GAP 8 块、等待集 v2；对 B2 s1 存档配对。
- 描述性读法（写在读数前）：冻结池 PCS ≥ 1398 + 20；易题失分合计 ≤ 47 − 5，红灯线类 ≤ 5、转弯集易题侧向零分 ≤ B2 s1 − 3，其余每类 ≤ B2 s1 + 2；
  held 460 零分 ≤ 41 + 2、前撞 ≤ 4 + 2；PAI 对 B2 s1 + ACC + GAP 配对 ≥ −0.005、零分 ≤ +2。
- 成本约 10 job 小时（约 5 卡小时，池空时 2–2.5 h wall）。

## 21.7 调度
- `turn_cap.sh pre 3a0537ad`：转弯集分片表（`lists`）→ 标签（CPU）→ 离线闸（CPU）；同一性（旧 rev 对新代码，60 步 eager）；定标（`--after` 标签）；
  对照的转弯集读数（无依赖），一次提交。`turn_cap.sh pilot <闸,同一性,定标>`：两臂训练（`--after` 三者；训练体再查三个 json，W 从 `dose.json` 读）
  → 第 1 段读数（等待集、held 460、子集 6 块、转弯集 6 个）→ `psum`（`--after` 全部读数与对照的转弯集）→ `STAGE1_DONE`。
- 第 1 段判读要冻结池 PCS 与易题分类，只能在 Mac 上算，所以第 2 段（`turn_cap.sh pai <臂>`）由代理在判读后对过线的臂提交，不进同一条链。
- 训练优先级 12，读数 13。id 记 `$DATA_DIR/runs/defect_fix/turn_cap/{jobs,stages}.txt`。不轮询：每段一个后台阻塞命令等完成标记（或 ERROR），只唤醒一次。

## 21.8 局限（先写下）
- 1 seed、1 000 步、两个剂量；pilot 的底是 1 000 步的 B2（k = 9 速度比 0.907，比全长 B2 的 1.04 慢），它在急弯里的过速可能比全长 B2 少，目标类的对照数
  可能偏小；pilot 量的是机制与漏，不是全长增益。
- 转弯标签与上限都按人的日志定，serve 时模型看不到标签，学到的是「这类输入（急弯、比人快）上别再快」；会不会泛化成直路 / 起步上的变慢由护 3–5 判。
- 上限按时钟对齐的人速度（`fut` / `fut_reg`），不是同站点的人速度；自车领先于人很多时，时钟上的人可能还在弯前（更快或更慢）。不另处理，G2 量的就是这个规则在目标决策上的覆盖。
- 只罚快慢、不罚形状：如果宽出的 plan 在人速下仍宽（第 395 条的 b 类，正常速度下的路径错），本项管不到。
- 第 395 条的期望增益（约 5.9 窗）来自观察性的零分率比，可能被「急弯同时带来高速比与零分」放大。
