# ROUND2 预登记：NTV0 续训的联合臂 J（历史拉平 p = 0.5 + PAI 人类状态模仿行含时间膨胀变体）+ 半数据臂 + 同步数对照（2026-10-20 晚，写于任何训练输出之前）

## 为什么

- 第 409 条：SIMP 简化配方输约 0.08，不采用；**历史抹平保留**（lead_break +0.15，修 3 个停车零分），下一轮以 NTV0 配方为底、不换项。
- 第 404 条：计划抄自身速度、约 90 % 走帧通道；第 407 条：起刹缺失，捷径在标签（需刹行 84 % 历史已在减速）。
- PAI-ROWS（`experiments/pai_rows/`）建了 853 个场景的人类状态模仿行（标签 = 人的 log 未来）及其帧间隔拉伸变体（补记 2：一致的时间膨胀，帧、历史、标签同按 f 播放），
  对着帧通道的速度复制给出「同一画面、不同速度」的模仿信号。
- 用户决定（10-20，有约束力）：底 = NTV0 配方，**从 NTV0-L18-s0 续训**；拉平 = SIMP-G 的广义起刹行表，每次抽到以概率 0.5 拉平；J 同时混入 PAI 人类行（含拉伸）
  作模仿行；步数随数据走；三个臂 CTRL / J / J-half，一个 seed。

## 臂（同一命令，只差下表列出的项）

底命令 = NTV0-L18-s0 的命令（`jt.sh train`，`ONK` / `ARCROWS=brake`，warm-up 500，lr-scale 0.1，cosine 到 0.1，compile 默认开）把 `--init RB0-F-s0`
换成 `--init NTV0-L18-s0`（`scripts/r2.sh dry` 打印三条命令及与 CTRL 的 diff）。

| 臂 | tag | 池 / 抽样 | on-k | 每批（期望） | 拉平 | 步数 |
|---|---|---|--:|---|---|---|
| CTRL | R2C-L18-s0 | on1_JT / JT00S | 32 | 83 navtrain + 13 yaw-rate + 32 S | 无 | = J |
| J | R2J-L18-s0 | on1_JTPH / R2JF | 64 | 51 navtrain + 13 yaw-rate + 32 S + 32 PAI 人类行（plain 16 / f0.5 8 / f1.5 8） | `--flat-brk flat_onset.txt --flat-p 0.5` | N_J |
| J-half | R2JH-L18-s0 | on1_JTPH / R2JA | 64 | 同 J，PAI 人类行只取半 A 的场景 | 同 J | N_H |

- **PAI 人类行 = 模仿行**：合池 `on1_JTPH` = on1_JT 的五个部分（顺序不变，L18NR1v2 仍是前缀栅格）+ `on1_PAIH:ph` + `on1_PAIH05:ph05` + `on1_PAIH15:ph15`
  （`rf_pait merge`，mode `both`，即 trainval 行进批的同一路径）。`r2.py fix` 把这三部分的 `agent.npz` ok 置 False（PAI 本无道路栅格，横向栅格只覆盖前缀），
  所以它们在 mixed 模式里只有模仿项（+ 非计划头 distill），没有任何 hinge；`--on-normal` 下与 navtrain 行一样受路线记忆 drop 与历史速度噪声。合池前 169 606 行
  须与 on1_JT 逐位相等（names / ego / fut / cam / mode / 路线特征 / teacher / front 抽样）。
- **抽样**（`r2.py weights`）：S 行沿用 JT00S 的权重（缩到抽样的 0.5）；三个 PAI 部分各自在「可抽行」上均匀，份额 0.25 / 0.125 / 0.125。可抽 = 场景在
  `physical_ai_av/pai-rows@v1`（已排除考场与 val160 驾驶）、不在训练器的 held（crc32(log) % 100 < 15，与 scenes.tsv 的 held 列核对相等），J-half 另限
  `physical_ai_av/pai-rows-half-a@v1`。抽样是 64 行的多项抽样（同 JT 里 trainval 行与 S 行的抽法），S 行与 PAI 行的每批数是期望值 32 / 32。
- **拉平**：SIMP-G 的 navtrain 起刹行表（`runs/defect_fix/simp/flat_onset.txt`，38 509 token，训练行 37 847 / 101 499 = 37 %），每次抽到时以 0.5 的概率拉平
  （ax 0、过去位姿按当前速度等距，标签不变；`rm_train --flat-p`，独立随机流 [seed, 0, 179]）。J 每批 51 个 navtrain 行，非 anchor 约 38 行 → 期望每批拉平约 7.1 行
  （SIMP-G 在 83 行上每批约 23 行）。PAI 行不拉平。
- 不变：yaw-rate 13、32 个 S 行（期望）、所有 hinge / 横向项、`--on-arc 0 --on-arc-rows brake`、held 15 %、一个 seed（0）。

## 步数（规则写定，行数在池建成后由 `r2.py steps` 按规则算出，写 `results/steps.{json,md}`）

- N = round(R / 32, 到 100)，R = 该臂可抽的 PAI 人类行总数（plain + 两个拉伸），32 = 每批期望的 PAI 人类行数 → **每个 PAI 人类行平均被看到约 1 次**。
  J 用全部 853 场景，J-half 用半 A（428 场景），所以 J-half 每行遍数与 J 相同、步数约一半。CTRL 取 N_J。
- 按 probe（11 场景：plain 160、f0.5 146、f1.5 117 行 / 场景；补记 2 的 6 s 规则会让 f1.5 再少一些）估：

  | 臂 | 可抽场景 | R（估）plain / f0.5 / f1.5 / 合计 | 步数（估） | 每行遍数 plain / f0.5 / f1.5 / 全部 |
  |---|--:|--:|--:|--:|
  | J | 约 719 | 11.5 万 / 10.5 万 / 8.4 万 / 30.4 万 | 约 9 500 | 1.32 / 0.72 / 0.90 / 1.00 |
  | J-half | 约 361 | 5.8 万 / 5.3 万 / 4.2 万 / 15.3 万 | 约 4 800 | 1.32 / 0.72 / 0.90 / 1.00 |
  | CTRL | 0 | — | = J | — |

  2 : 1 : 1 的份额让 plain 行遍数高于拉伸行（拉伸行每个变体一份）。实际数以 `results/steps.md` 为准，不再改规则。

## 闸（训练前，读之前写定）

1. **关掉即逐位相等**（`r2.sh pre`，60 步 eager，CTRL 的命令）：(a) 旧代码（`git worktree` 在旧 rev，**从旧工作树运行**）对新代码不加新 flag；(b) 旧代码对新代码
   `--flat-brk <onset 表> --flat-p 0`；两项 `df.py bitcmp` 全部标量、权重、dev ADE、抽样 hash 逐位相等，且两次运行的 meta.json 显示不同的 git sha 与 cwd
   → `results/identity.json`。
2. **smoke**（probe 池 on1_PAIHT*，在 PAI-ROWS reprobe 过后；合池 `on1_JTPHT` = rb0 + rtb1 + 三个 probe 池，J 的命令 60 步）：标量全有限；
   `flat/rows` 均值在期望（0.5 × 51 × 0.75 × 37 %）的 0.6–1.4 倍；池模仿行 `on/n_imit` 在期望 32 的 0.75–1.25 倍；这些行 `on/n_hinge` 恒为 0、
   `on/agent_ho` 恒为 0；记录的 p = 0.5 → `results/smoke.json`。
3. **池**：PAI-ROWS 的 H 通道写出 `H/DONE`、无 `H/STOP`、全量 plain 池 R1 过、reprobe 的 f = 1 位等闸过（`r2.sh merge` 自查）。合池 fix 的前缀逐位相等过。
   训练作业在 identity 与 smoke 都过时才运行。

## 读数（训练后，一次提交）

- **离线闸**（J、J-half 各自对 CTRL）：
  - inertia.py（P160，工具不改；`lon_anchor.py inertia` 的登记线）：STOP 比与 TURN 比都 ≤ 0.6，且各比 CTRL 低 ≥ 0.15；空路 v24 斜率 ≥ 0.7。
  - lead_break.py（RB0-F 自身的 PAI 撞车状态；`simp.py leadread` 的登记线）：BRK「4 s 内留在前车后」份额对 CTRL 升 ≥ 0.10 且 CI 下界 > 0，FOL 变化在 ±0.05 内。
- **PAI-FIX 原生**（VC + LEAD，`pai_fix.sh read <tag> vl "v t"`，三个臂同时提交）：`pai_fix.py report` 对 CTRL、对 NTV0-L18-s0-vl 各一份：TARGET 均分与按类零分、
  STOP / STOP-L / TURN / GUARD / T281，按场景配对差与 CI。**不读 ACC + GAP、不读 PAI 281 / held**（主会话之后再选）。
- 主读数（只读，不在本 lane 做采用）：J − CTRL 的 PAI-FIX 原生 TARGET 配对差；数据规模读数 J − J-half。预期方向（写在读之前）：拉平 → lead_break BRK 升、STOP 零分降；
  拉伸行 → inertia 的 STOP / TURN 比降；PAI 人类行 → TURN（路线转 / 出路零分）改善。哪个都不过只记录，不追加臂。
- 种子差（第 406 条）：TARGET ±0.04、几乎全在 TURN；单 seed 下 |差| < 0.04 记为「分不开」。

## 花费与磁盘（估）

- 训练：NTV0 9 500 步 compile 1 610 s → CTRL / J 各约 0.5 卡时，J-half 约 0.25（大池的 page cache 可能稍慢）；PAI-FIX vl 三臂 × 约 0.44 ≈ 1.3 卡时；
  inertia / lead_break cf 约 0.5 卡时；identity + smoke 约 0.2 → 合计约 **3.3 卡时**。
- 磁盘：on1_JTPH ≈ 170 GB（front 约 140 GB + road / drivable 约 26 GB），合池前检查空余 ≥ 250 GB；smoke 池用完即删。轮次结束后 on1_JTPH 的去留逐项请用户定。

## 实现

- `experiments/route_mem/scripts/rm_train.py`：`--flat-p P`（默认 1 = 原行为，不加任何操作；0 = 关，不载入列表；0 < P < 1 用独立流 [seed, 0, 179] 每批每行一个均匀数）。
- `experiments/round2/scripts/r2.sh`（lane，全部经 GPU 池，`--after` / `--when-exists` 串起）与 `r2.py`（fix / weights / steps / smoke / costs）。
- 附带发现（不改）：`simp.sh` 的 identity 作业把旧 worktree 的 jt.sh 只用来打印命令，作业在主工作树里运行，所以 SIMPGG*-old 与 -new 的 meta.json
  同为新代码（sha 22898015、同一 cwd），那几次「旧对新」比较实为新对新。本 lane 的 identity 从旧工作树运行并核对两边 sha。
