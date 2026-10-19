# NR1 预登记补记 20（STAND-R：人站着、自车已在滚动时的 STAND；从 B2 出发）

**状态：已登记，2026-10-20 00:45 CST（box 时间）。** 登记时还没有任何标签（stand_gap.npz）、闸输出、训练或臂读数；盘上只有对照（pilot B2）与 B2 全长的存档读数，
对照的第 1 段锚点（20.3 表）就是从这些存档按登记代码算的。草稿由前一个代理写于 2026-10-20（依据 [easy_items_B2.md](../../alpasim_proxy/results/easy_items_B2.md)，第 393 条），
定稿代理审过后改了下面「定稿修订」各条，其余原样。主会话已批准 pilot 阶段；**范围（主会话 10-20 转达、用户同意）：只跑离线闸与两条 1 000 步 pilot（加对照的
PAI 半读），不论 pilot 结果都不启动全长臂**（理由：这一类大部分已由 serve 端 stop3 保住，期望增益小），20.4 只作记录。另加一个诊断读数：B2 s1 + stop3 的大考场 held。
代码：`lib/nr1_free.py`（`rolling` / `roll_cap`）、`lib/nr1_lat.py`、`rm_train --on-stand-roll`、`scripts/stand_r.{py,sh}`；结果 `results/stand_r/`。

### 定稿修订（登记前，读数前）
1. **g 的对齐核对过，成立。** `stop_label` 的 s_e = s_log(t_log) + lead，而 rows meta 的 lead = s0 − s_log（`on_rows`），所以 s_e 就是访问状态在日志路径上的站位 s0，
   g = h4c = s_h(t + 4 s) − s0；人 4 s 站着时等于他此刻的站位差。**改正**：草稿 20.7 说「横向偏离远的行 `path_ok` 为假」不对，`path_ok` 只查时间范围与 s_e ≥ S[0] − 1 m；
   横向由池的构建保证（`on_rows` 排除 |e0| > 4 m 的行，池里 |e0| 最大 4.0 m）。标签脚本改为 `scripts/stand_r.py gap`（复用 `stoppt.py` 的 rows meta 与 `stop_label`），
   on1_JTX 里 pn0–pn5 部分（PN0–PN5 rows）也有标签；对 `stoppt/labels.npz` 的 h4c 在 on1_JT 的 169 606 行上逐行比对（差 < 1 mm，否则作业失败）。
2. **开关**：V_ROLL = 0 时不读 `stand_gap.npz`，pack、rec、损失键与旧代码完全相同（逐位同一性仍按 20.2 第 4 条实测）。开时 STAND 项拆成两个键：站着行 `lon_stand`、
   滚动行 `lon_standr`（两者之和 = 原项，同权 10 × W_S）；草稿的 `on/lon_stand_roll_pos` 改为 trainer 自带的 `on/lon_standr_pos`（本批池行里滚动部分触发的份额），
   `--diag` 记 `grad/lon_standr`。
3. **起步泄漏**：g < 0（自车已越过人站的地方）的行保留，上限 = b_j（按 `fut` 的 2 s 过渡刹停）。这正是失败窗的样子：从线前站着蠕行过人站处 1–3 m 后才过 0.5 m/s。
   草稿离线量用的近似（`tmp/easy_items_B2/scr/gate.py`）把 g < −1 m 的决策排除了、上限也是近似，这里改为按登记规则逐决策精确计算（人的日志轨迹、`fut` 的过渡、
   `stand_terms` 打存档 plan），闸线不变。
4. **闸与读数的分段**：等待集 v2 与 held 460 右漂的读数本来就和第 1 段一起跑（便宜），所以它们的护栏**提前到第 1 段判**（只会更早停，不放松）；第 2 段只剩 PAI 半读。
   第 1 段的易题按 `easy_items.classify`（第 390 条同一套先匹配规则）在子集上分类：子集 = 6 块里冻结 board-mix 题、各读数都有的窗（对照 716 窗）；
   「红灯线易题零分」= 类 `zero: static, red line, no lead` 且 garage = 1。对照锚点按此代码从存档重算，与草稿一致：子集均分 0.9127、零分 55、子集 PCS 1246、
   易题 525 窗里失分 53（红灯线 30、其余 23）。held 460、慢窗、k = 9、等待集、右漂由 `stand.py pilot` 的表给出（对照数为 pilot_B2.md 的同一读数：held 零分 42、前撞 3、
   子集 AF-front 1、慢 36、k = 9 速度比 0.9068）。草稿写的「慢 35、AF-front 0」是另一种计法，以表为准。
5. **G1 也查站着行**（人的两条路径当 plan，站着 + 滚动行都不得触发）。

## 20.0 为什么是这一类

- B2（`STANDFMXH-L18-s1`，第 391 条 nuPlan 原生候选）的易题失分 47 窗（s0 54 窗）。按第 390 条的单类规则，最大的一类是**静止窗红灯线前无前车的滚出**：
  s1 9 窗，单类修 +29 [+6, +60] board-mix / +54 g15；s0 10 窗，+25 / +41。FREE-M 剂量减半后这一类不动（B 10 / 10 → B2 10 / 9），4 个读数里有 6 窗相同，
  stop3 能保住 9–10 窗。按组算更大的是侧向零分组（19–20 窗，+54 到 +80）和易题部分分组（16–24 窗）。但这两组各含好几种机制：侧向零分里只剩 11 窗还过速，
  部分分组是半剂量换回来的棘轮、差一点和停住。
- **失败机制（存档决策表）**：
  - k = 0 时自车站着、人站着，plan 4 s 弧长已有 2–4 m（STAND 的上限 b_8 约 1 m，所以 STAND 在这些决策上触发，但剂量不够）；
  - 自车开始蠕行，约在 k = 4–5 过 0.5 m/s，过了以后就**不在 STAND 的行里**（STAND 要求自车 t0 速度 < 0.5 m/s），于是一路加速（k = 7 时 plan 弧长 13–20 m）；
  - 零分之前的有效决策里，人站着、自车在 0.5–3 m/s 的约占一半（s1 40 / 79，s0 41 / 92）。
- 所以缺的有两样：自车在滚动时的行（域），和站着行上的剂量。这是第 388 条 STAND 的同一方向（只由标签驱动，不按灯设条件，第 380 条）。

## 20.1 项 STAND-R（规则写在任何标签统计之前，读数后不改）

- **行**：池的 S 行（`mode == self`），`fut` / `fut_reg` 有限，人的 4 s 记录弧长 arc(`fut_reg`)_8 < 0.5 m（人整段 4 s 站着，同 STAND），且
  (a) **站着行**：自车 t0 速度 < 0.5 m/s，与 STAND 完全相同（上限、值都不改）；
  (b) **滚动行（新）**：0.5 ≤ 自车 t0 速度 < V_R，且有站位差标签 g（有限）。
- **站位差 g**：g = s_h(t) − s_e，即人此刻（站着）的日志站位减去自车站位。复用 `nr1_free.stop_label` 的 `h4c` = s_h(t + 4 s) − s_e：人 4 s 都站着，所以它就等于 g。
  g 可以为负，表示自车已越过人站的地方。为什么必须要 g：池里滚动行有 74 % 自车**在人后面**，而且落后的距离超过 STAND 的上限
  （on1_JT 的 976 行里 rest > b_8 的占 74 %，> b_8 + 2 m 的占 59 %，2026-10-20 在盒上数的）。不按站位对齐的话，这一项会主要在教「停在人前面老远」，
  也就是第 386 条 STOPPT 那种整体变慢、起步变差的漏法。
- **上限**：c_j = max(b_j, max(g, 0) + M)，b_j = max(arc(fut)_j, arc(fut_reg)_j) + M 与 STAND 相同，M = 1.0 m。`fut` 是从访问状态的车速过渡到人的速度曲线
  （前 2 s 过渡），所以滚动行的 b_j 已含按 2 s 刹停的距离（约 v0 × 1 s），不要求做不到的 plan；落后的自车可以开到人站的地方 + 1 m。
- **触发与值**：与 STAND 相同：s_plan_8 > c_8 时，逐步 Σ_j smooth L1(β 0.5) of relu(s_plan_j − c_j)；只沿各点自身航向移动，航向无梯度；10 × W_S × 项，记 `on/lon_stand_ho`。
  另加只读计数 `on/lon_stand_roll_pos`（滚动行的触发份额）。
- **剂量**：W_S 一个数同时作用于 (a)、(b) 两类行（站着行上的剂量不够也是失败原因之一）。

### 代码改动（草稿原文；实际实现见 `lib/nr1_free.py`，按定稿修订 1–2 有小改：`roll_cap`、`stand_term_split`、标签由 `stand_r.py gap` 写）

```python
V_ROLL = 0.0                     # supplement 20: upper ego speed of the rolling STAND rows (0 = off, STAND as in supplement 17)


def configure_stand(margin=1.0, roll=0.0):
    global MARGIN, V_ROLL
    assert margin > 0 and (roll == 0 or roll > V_STAND), (margin, roll)
    MARGIN, V_ROLL = float(margin), float(roll)
    return dict(stand_margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND, v_roll=V_ROLL)


def rolling(v0, hum4, gap):
    """supplement 20 rows: the human stands all 4 s, the ego rolls at V_STAND <= v0 < V_ROLL, the station gap is labelled."""
    v0 = np.asarray(v0)
    return (np.isfinite(hum4) & (np.nan_to_num(hum4, nan=1e9) < H_STAND) & (v0 >= V_STAND) & (v0 < V_ROLL)
            & np.isfinite(gap))


def stand_load(pool_dir, names, dev):
    """-> pack: cap (N, 8), ok (N,) on dev for the pool's standing S rows (+ rolling rows when V_ROLL > 0)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: tab rows != the pool's"
    cap, fin = stand_cap(z["fut"], z["fut_reg"])
    S = z["mode"].astype(str) == "self" if "mode" in z.files else np.ones(len(fin), bool)
    hum4 = cum_arc(z["fut_reg"])[:, -1]
    st = standing(z["speed"], hum4)
    roll = np.zeros_like(st)
    if V_ROLL > 0:
        g = np.load(Path(pool_dir) / "stand_gap.npz")             # scripts/stoppt.py gap: names, gap = s_h(t) - s_e (h4c)
        assert np.array_equal(g["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: stand_gap rows != the pool's"
        roll = rolling(z["speed"], hum4, g["gap"]) & ~st
        cap[roll] = np.maximum(cap[roll], (np.maximum(g["gap"][roll], 0.0) + MARGIN)[:, None]).astype(np.float32)
    ok = fin & S & (st | roll)
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), standing=int((ok & st).sum()), rolling=int((ok & roll).sum()),
               ego_slow=int((S & fin & (z["speed"] < V_STAND)).sum()), margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND, v_roll=V_ROLL)
    return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok, device=dev),
                roll=torch.as_tensor(ok & roll, device=dev), rec=rec)
```

- `stand_terms` 不改（它只吃 cap 与 ok）。`rm_train` 加 `--on-stand-roll V`（默认 0），转给 `nr1_lat.configure(..., stand_roll=)` → `configure_stand(margin, roll)`。
  `--diag` 记 `on/lon_stand_roll_pos` = 本批滚动行里触发的份额。
- **标签**：`stoppt.py gap --pool on1_JTX`（CPU）复用 `nu_rows` / `_nu_job` / `stop_label`，对 S 行写 `$CR/on1_JTX/stand_gap.npz`（names 与 tab 逐行相同，
  gap = `h4c`；trainval 行、路径不可用（`path_ok` 假）的行为 NaN）。on1_JTX 的前 169 606 行与 on1_JT 逐行相同（已核对），这部分可以和
  `stoppt/labels.npz` 的 `h4c` 逐行比对，当作代码检查。
- **自检**（`python nr1_free.py`，追加）：(1) V_ROLL = 0 时 pack 与旧代码逐元相同；(2) 滚动行 v0 = 2、g = 0：plan 4 s 弧长 2.5 m 不触发（c_8 ≈ 3 m），8 m 触发；
  (3) g = 10：plan 9 m 不触发、12 m 触发（落后的自车可以开到人站的地方 + 1 m）；(4) g = −3（已越过）：c_j = b_j。

### 命令 diff

- 1 000 步 pilot 的底：盘上的 pilot B2 `STANDFMXH1K-L18-s0`（`runs/defect_fix/stand_b/cmd_B2.txt`），它的存档读数当对照。两臂只改 tag 和下面这些词：

```
R1  STANDRH1K-L18-s0 : --tag STANDFMXH1K-L18-s0 -> STANDRH1K-L18-s0   ; + --on-stand-roll 3.0                (W_S 0.0068 unchanged)
R2  STANDR2H1K-L18-s0: --tag STANDFMXH1K-L18-s0 -> STANDR2H1K-L18-s0  ; + --on-stand-roll 3.0 ; --on-stand 0.0068 -> 0.0136
```

- 全长（只在 pilot 通过、用户批准后）：B2 s1 的命令（`runs/defect_fix/stand_b/cmd_FHS.txt`，9 500 步、warmup 500、`--seed 1`）上做同样的改动：
  `--tag STANDFMXH-L18-s1 -> STANDR[2]H-L18-s1`，加 `+ --on-stand-roll 3.0`，若选 R2 再加 `--on-stand 0.0068 -> 0.0136`。`stand_b.sh check` 式的 DRY diff
  只能出现这几个词。

### 剂量范围（理由）
- **域**：V_R = 3.0 m/s。红灯线失败窗里，零分之前人站着时的决策速度在 3 m/s 以下的占 77 / 79（s1）和 88 / 92（s0）。
  JTX00 抽样里滚动行（< 3 m/s，3 568 行）占总抽样 1.12 %，1 000 步约抽到 790 次；站着行 19 324 行，占 6.6 %。
  V_R = 5 只多 466 行，而且那些已是真正在开的车，不加。
- **W_S**：R1 用 1 倍（0.0068，只扩域）、R2 用 2 倍（0.0136，扩域并在站着行上加量）。不超过 2 倍的理由：STAND 单独用时（第 388 条 A 臂）a 变体
  other_lead 等住率 0.944 → 0.889，已压在护栏线上，再往上加起步风险最大。两臂并行（剂量响应），不做因子设计。

## 20.2 离线闸（CPU，先跑；参数读数后不改；四条全过才训练）

1. **人路径误触发 = 0**：on1_JTX 的滚动行上，以人的两条 4 s 路径当规划，触发数为 0（按构造成立，用来抓代码错）。
2. **覆盖**：B2 s0 / s1 存档的易题红灯线零分窗（共 19 窗次），零分前、人站着、自车 0.5–3 m/s 的有效决策，用存档 plan（`dec.parquet` 的 poses）按
   `stand_terms` 打分，触发率 ≥ 80 %。2026-10-20 在 Mac 上按近似上限（b_8 ≈ v + 1、g = s_h_now − s_e）量过：s1 14 / 14、s0 10 / 10。
3. **不碰成功的驾驶**：B2 s0 / s1 得 1 分的窗里，落在滚动行域（人站着、自车 0.5–3 m/s）的决策数 ≤ 1 %，且这些决策里触发的大多在 static 层（只描述）。
   近似量：s1 25 个决策 / 20 窗、s0 26 / 22 窗（全部 2 106 窗里），其中触发 8 / 10 窗，几乎都是 static 层的小幅蠕行。
4. **开关关掉逐位相同**：pilot B2 的命令 60 步 eager，旧 HEAD 对新代码（不开 `--on-stand-roll`）→ `results/stand_r/identity.json`（`df.py bitcmp`）。
   另做 60 步冒烟：R2 的命令，损失有限，`on/lon_stand_roll_pos` 在某步 > 0。

## 20.3 1 000 步 pilot（1 seed，对盘上的 `STANDFMXH1K-L18-s0`）的早停闸

**第 1 段**（先读：大考场子集 chunk 0 / 3 / 6 / 9 / 12 / 15 + held 460 @18 + 等待集 v2，全部原生）。对照的存档数字（定稿修订 4 重算）：子集 716 窗均分 0.9127、
零分 55、子集易题失分 53（其中红灯线易题零分 30）、子集 PCS 1246；表的计法：慢部分分 36、AF-front 1；held 460 零分 42、前撞 3；k = 9 速度比 0.907。
判读由 `stand_r.py stage1`（Mac）按下表做，`results/stand_r/stage1.{md,json}`。

| # | 读数 | 线 |
|:--|:--|:--|
| 主 1 | 子集易题红灯线零分 | ≤ 对照 − 6（≤ 24） |
| 主 2 | 子集 PCS（`pcs_local.py score --subset`，Mac） | ≥ 对照 + 15（≥ 1261） |
| 护 1 | held 460 零分 | ≤ 对照 + 2（≤ 44） |
| 护 2 | held 460 前向碰撞 / 子集 AF-front 窗 | ≤ 对照 + 2（≤ 5 / ≤ 2） |
| 护 3 | **子集易题失分，红灯线以外**（`easy_items.classify` 同一套规则）：合计 | ≤ 对照 + 2（≤ 25） |
| 护 4 | 同上，**每一类**（起步、差一点、停住、棘轮、追尾、转弯、在动走廊、出路） | 每类 ≤ 对照 + 2 |
| 护 5 | 子集慢部分分窗（type `slow`）/ k = 9 速度比 | ≤ 对照 + 3（≤ 39）/ ≥ 对照 − 0.03（≥ 0.877） |
| 护 6 | 等待集 v2（由第 2 段提前）：a 变体两类等住率 / b 变体中位起步延迟 / 起步数 | ≥ 对照 − 0.03 / ≤ 对照 + 0.15 s / ≥ 0.95 × 对照 |
| 护 7 | held 460 右漂（非转弯 k 11–17，臂 − 对照；由第 2 段提前） | > −0.05 m |

第 1 段任一条不过，就停：第 2 段不读，写清是哪一类漏了。

**第 2 段**（第 1 段过了才提交）：
- （等待集与右漂已提前到第 1 段：护 6、护 7。）
- PAI 281 半读（c000 / c002 / c004 / c006，第 366 条 serve、ACC + GAP、g10）：臂对对照的配对均值 ≥ −0.01，零分 ≤ 对照 + 2。
  对照的 PAI 半读盘上没有，和闸一起先提交（同一组 chunk、同一 serve）；配对与零分由 `pp.py gate` 算，线按本条判（不用 pp.py 自己的线）。

**判读**：一臂全过 = PASS，报告后停（本轮不启动全长）。两臂都过时选 R1（剂量小、漏的风险小），除非 R2 的主 1 多修 ≥ 3 窗而护栏同样过。

## 20.4 全长（只作记录：主会话 10-20 定的范围下不启动）

- 从 B2 s1 的命令出发（20.1 的 diff），1 seed。读数与 B2 相同：大考场 held 18 块、held 460、PAI 281 ACC + GAP 8 块、等待集 v2。
- 对 B2 s1 存档（不是闸外的新对照）：
  - 易题红灯线零分 ≤ 9 − 4 = 5；
  - 易题失分合计 ≤ 47 − 3 = 44，且红灯线以外每类 ≤ B2 s1 + 2；
  - held 460 零分 ≤ 41 + 2、前撞 ≤ 4 + 2；
  - PAI 对 B2 s1 + ACC + GAP 配对 ≥ −0.005、零分 ≤ +2；
  - 冻结池 PCS ≥ 1398 + 20。
- 期望：B2 的能力处每修一个易题窗约值 +4–5 PCS（按 a 排序的曲线）。修 4–5 窗约 +15–25 board-mix；整类（9 窗）+29 / +54 g15。
  这是第 390 条按 a 的曲线外推，真实增益要看有没有别的类漏。

## 20.5 成本（盒上 2026-10-19 同类作业的实测 wall 时间）

| 段 | 作业（每臂） | job 小时 | 卡小时（nuPlan read 4 个 / 卡；PAI 按 1 个 / 卡计，上界） |
|:--|:--|--:|--:|
| 闸 / 标签 / 同一性 / 冒烟 | CPU + 2 × 60 步 | 0.3 | 约 0.1 |
| pilot 第 1 段 × 2 臂 | 训练 0.05 + 子集 6 × 0.25 + held 6 × 0.20 | 2 × 2.75 = 5.5 | 2 × 0.73 ≈ 1.5 |
| pilot 第 2 段 × 2 臂 + 对照 PAI | 等待集 8 × 0.105，PAI 4 × 0.39（臂 2 + 对照 1） | 2 × 0.85 + 3 × 1.55 = 6.4 | 0.4 + 4.7 ≈ 5.1 |
| **pilot 合计** | | **约 12** | **约 6.5**（池空时约 1.5–2 h wall） |
| 全长 1 臂（批准后） | 训练 0.5 + 大考场 18 × 0.24 + held 1.2 + 等待集 0.85 + PAI 8 × 0.39 | 约 10 | 约 5（约 2–2.5 h wall） |

第 1 段不过的话，第 2 段的 6.4 job 小时不花。

## 20.6 调度
- `stand_r.sh pre <old rev>`：标签 → 闸（CPU）、同一性（旧 rev 对新代码，B2 pilot 命令 60 步 eager）、冒烟（R2 命令 60 步）、对照 PAI 半读，一次提交；
  `stand_r.sh pilot <闸,同一性,冒烟>`：两臂训练（`--after` 这三个；池的语义是父作业失败则子作业失败，训练体还会再查三个 json）→ 第 1 段读数 → `psum`。
  第 1 段判读要冻结池 PCS 与易题分类，只能在 Mac 上算（`stand_r.py stage1`），所以第 2 段（`stand_r.sh pai <臂>`）由代理在第 1 段判读后对过线的臂提交，不进同一条链。
- 训练优先级 12，读数 13。id 记在 `$DATA_DIR/runs/defect_fix/stand_r/{jobs,stages}.txt`，结果放 `experiments/defect_fix/results/stand_r/`。不轮询：
  每段一个后台阻塞命令等汇总标记（或 ERROR）。

## 20.7 局限（先写下）
- 1 seed、1 000 步。pilot 的底是 1 000 步的 B2，子集里红灯线零分 37，比全长的 9–10 窗宽得多。全长剩下的是更难的核心，pilot 量的是机制和漏，不是全长增益。
- g 来自日志站位。自车偏离日志路径很远时站位投影不可靠；池在构建时已排除 |e0| > 4 m 的行（定稿修订 1），4 m 以内的投影误差不另处理。
- 有前车的站着 / 滚动行照旧在内（与 STAND 一致），所以这一项也会在排队时压短 plan，按方向应减少前撞，但可能加重排队里的晚起步，由第 2 段的 b 变体护栏判。
- 扩域后，人站着时这一项要求「不越过人站的地方 + 1 m」。人站着但信号已变绿、人反应慢的行，会教出同样的慢。这类行在池里的份额只描述，不进闸。
