# Longitudinal operating point of the current driver, and the collision class to target (LONG-ORACLE)

Written 2026-10-11. Drivers `P2H10S-F-s0 / s1` (primary, decisions 232 / 244 / 247) and base `P2H10-F-s0 / s1`, navtest 12 146 tokens (bench
warp-frame plans, the plans the benchmark scores), navhard stage 1 (450 tokens, G frames, `P2H10S` only: the base has no s0 / s1 navhard harness).
Pre-registration [plans/2026-10-11-long-oracle-prereg.md](../plans/2026-10-11-long-oracle-prereg.md), pushed (5d9618fa) before any per-token read.
No model was run, nothing trained: stored bench plans and per-token scores, decision 153's instrumented devkit replay, and
`python -m jevdrive.bench score-poses --traffic non_reactive` for the scaled plans (11 scales x 2 seeds x 2 arms, about 15 min of box CPU).
Code: `experiments/op_parity/scripts/nc_tax.py` (decision 196's script, now taking `NC_TAX_*` overrides, defaults unchanged),
`scripts/lo_chain.sh`, `scripts/lo_report.py`. Tables: [long_oracle/tables.md](long_oracle/tables.md) (readings 3-5) and
`long_oracle/<arm>/tables.md` (taxonomy and privileged oracle per arm).

**Verdict (registered rule): no room.** No uniform scale and no conditioned operating point beats the served plan on navtest in either seed of
`P2H10S-F`; the served speed sits at the top of the curve (0.95 ties, 1.05 is already worse). By the stop rule there is no PAI sweep.
The class to target is **A, a stopped or slow vehicle ahead in the ego lane** (41% [32, 51] of the NC failures are A failures WA-JEPA passes;
the next class is 16%).

## Checks

- Replay: the 8 sub-scores equal the bench archive on every failing token (max |diff| 1e-16 navtest, 1.4e-12 on one base seed; navhard 0);
  0 rows dropped.
- Identity gate: the a = 1.0 rows of both scoring runs (failing tokens, all tokens) equal the archive on all 12 146 tokens, both arms, both seeds.

## 1. Taxonomy (navtest; seed means, share CIs are log-cluster bootstraps)

| | `P2H10S-F` | base `P2H10-F` | SH30 (decision 196) |
|:--|--:|--:|--:|
| NC failures (s0 / s1) | 174 (181 / 167) | 182 (188 / 176) | 176.5 |
| TTC-only failures | 91 (90 / 92) | 83.5 (86 / 81) | 87.5 |
| WA-JEPA passes the NC-failed token | 77% | 78% | 76% |
| NC event at ego < 3 m/s | 41% | 39% | 42% |
| plan > 1.1 x logged 4 s arc | 59% | 60% | 63% |

**NC failures by class, `P2H10S-F`** (base in brackets):

| class | n | share % [95% CI] | WA-JEPA passes % | arm-specific share of all NC % [95% CI] | event < 3 m/s % | plan > 1.1 x log % | privileged recovery 0.7-1.1 % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **A stopped / slow vehicle ahead** | **90.5** (93.5) | **52 [42, 62]** | 78 | **41 [32, 51]** (41) | 51 | 71 | 75 [61, 88] |
| - A1 stopped vehicle | 58 (59) | 33 | | | | | 66 |
| - A2 moving lead | 32.5 (34.5) | 19 | | | | | 92 |
| B cut-in | 21 (21) | 12 [6, 20] | 60 | 7 [3, 13] | 76 | 83 | 88 [75, 100] |
| C crossing at junction | 14.5 (14.5) | 8 [0, 17] | 66 | 5 [0, 11] | 34 | 72 | 72 [38, 100] |
| D side contact | 14 (18) | 8 [4, 14] | 100 | 8 [4, 14] | 0 | 0 | 68 [52, 86] |
| E other (static 18.5, oncoming 9, VRU 6.5) | 34 (35) | 20 [13, 27] | 81 | 16 [11, 22] | 13 | 29 | 56 [38, 74] |

**Class x turn x speed** (`P2H10S-F`, seed-mean counts): the NC failure count by turn bucket is 95.5 / 31.5 / 20.5 / 26.5 (< 5 / 5-20 / 20-45 /
> 45 deg) and by t0 speed band 34 / 76.5 / 60 / 3.5 (< 2 / 2-5 / 5-10 / >= 10 m/s). The largest cells are all class A: moving lead, straight,
5-10 m/s (17; WA-JEPA passes 76%, recovery 100%); stopped vehicle, > 45 deg, 2-5 m/s (12.5); stopped vehicle, straight, t0 < 2 m/s
(11; **WA-JEPA passes all**, recovery only 41%: the plan should not move, and a slower launch still reaches the vehicle); stopped vehicle,
20-45 deg, 2-5 m/s (8.5). Full cross tables in `long_oracle/<arm>/tables.md`.

**TTC-only** (`P2H10S-F`, 91): E other 44% (static 22.5, VRU 9, oncoming 8.5), A 32% (A1 26.5), B 10%, D 8%, C 5%. WA-JEPA passes 67%.

**`P2H10S` against base.** The class profile is the same; the shape-only hinge removed side contacts (D 14 against 18) and a few A
(90.5 against 93.5), and added TTC-only A1 failures (26.5 against 18). Nothing longitudinal moved, as decision 232 expects (speed profile back to
base). Against SH30 (decision 196) every class count is within 3 tokens.

**navhard stage 1** (`P2H10S-F`, counts only): NC 18 / 15, TTC-only 1 / 3; NC by class: stopped vehicle ahead 7, oncoming 4, VRU 2, the rest 1
or less. Again the stopped vehicle ahead is the largest class.

## 2. Privileged oracle (decision 196's rule: per token, the best recovering scale picked with the simulator's score)

| family | `P2H10S-F` NC recovered % [95% CI] | clean % | base NC recovered % | TTC-only recovered % (S) |
|:--|--:|--:|--:|--:|
| 0.95-1.05 (new) | 18 [13, 24] | 17 | 21 | 33 |
| 0.9-1.1 | 33 [26, 40] | 31 | 35 | 51 |
| 0.85-1.15 (new) | 49 [42, 57] | 45 | 49 | 62 |
| 0.7-1.1 (main) | 72 [64, 82] | 67 | 74 | 73 |
| 0.5-1.15 (all scales) | 94 [92, 97] | 92 | 95 | 86 |
| faster only (1.1) | 3 | 3 | 3 | 13 |

Board-level upper bound (scale changed only on the arm's failing tokens with a clean recovery, all else unchanged): `P2H10S-F` NC +0.92,
TTC +1.25, EP -0.05, **no-EC EPDMS +1.07** (s0 +1.14 / s1 +1.01); base +1.13 (SH30 +1.13). The privileged ceiling of a directed slowdown
did not change with the hinge.

## 3. Uniform scaling of the whole board (non-reactive, no-EC EPDMS points against a = 1.0, log-cluster paired CI)

| a | `P2H10S-F-s0` | `P2H10S-F-s1` | base s0 | base s1 | dNC | dTTC | dDAC | dEP | NC fixed / new | split (i) / (ii) / (iii) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.70 | -3.24 [-3.83, -2.66] | -3.32 [-3.91, -2.72] | -3.11 | -3.17 | +0.63 | +0.14 | +0.83 | -13.15 | 121 / 40.5 | +0.26 / +0.55 / -4.09 |
| 0.80 | -1.26 [-1.67, -0.84] | -1.41 [-1.84, -0.97] | -1.11 | -1.24 | +0.64 | +0.68 | +0.84 | -8.37 | 96.5 / 15.5 | +0.57 / +0.66 / -2.56 |
| 0.85 | -0.56 [-0.89, -0.22] | -0.76 [-1.10, -0.43] | -0.54 | -0.58 | +0.55 | +0.65 | +0.79 | -6.12 | 78 / 9.5 | +0.52 / +0.65 / -1.83 |
| 0.90 | -0.19 [-0.44, +0.06] | -0.36 [-0.60, -0.13] | -0.16 | -0.30 | +0.36 | +0.55 | +0.61 | -3.96 | 51.5 / 6.5 | +0.40 / +0.52 / -1.19 |
| **0.95** | **+0.02 [-0.13, +0.18]** | **-0.04 [-0.21, +0.12]** | -0.06 | -0.07 | +0.20 | +0.28 | +0.37 | -1.91 | 27.5 / 2.5 | +0.23 / +0.33 / -0.57 |
| 1.00 | 0 | 0 | 0 | 0 | | | | | 174 failures | |
| 1.05 | -0.30 [-0.53, -0.09] | -0.37 [-0.61, -0.15] | -0.36 | -0.27 | -0.30 | -0.37 | -0.45 | +1.77 | 4 / 41 | -0.31 / -0.39 / +0.37 |
| 1.10 | -0.89 [-1.28, -0.55] | -0.93 [-1.33, -0.55] | -0.98 | -0.91 | -0.67 | -0.82 | -0.96 | +3.38 | 6 / 89 | -0.70 / -0.85 / +0.64 |
| 1.15 | -1.87 [-2.44, -1.34] | -2.05 [-2.59, -1.54] | -1.95 | -1.99 | -1.32 | -1.54 | -1.58 | +4.83 | 6.5 / 171 | -1.35 / -1.39 / +0.78 |

Sub-score columns, counts and the split are `P2H10S-F` seed means. 0.5 / 0.6 are in the tables (-9.5 / -6.1).

- **The served speed is the optimum of the family.** Slowing by 5% is a tie (+0.02 / -0.04), every slower level loses (EP outweighs NC + TTC + DAC),
  and every faster level loses with a CI excluding 0. The curve is asymmetric: 5% faster costs -0.33 (41 new NC failures for 4 fixed), 5% slower
  costs nothing. The knife edge of decisions 249-251 is visible open loop: faster is the costly side.
- **Half of the slowdown's credit is not collisions.** At 0.9-0.95 the DAC-only tokens (ii) carry more of the gain than the NC / TTC tokens (i):
  decision 203's horizon effect (a slower plan leaves the drivable area later, or beyond 4 s).
- **By class** (`long_oracle/tables.md`, failures of a = 1.0 that pass at uniform a): at 0.9 the moving lead A2 40%, cut-in 48%, stopped vehicle A1
  only 23%; at 0.8 A2 83% but A1 49%. A stopped vehicle needs a large, local slowdown (or no launch), which a global knob cannot give without the
  EP bill. Faster levels fix only side contacts (D 21-25% at 1.05-1.15) and a few static objects.

## 4. Conditioned operating point (non-privileged: scale per t0 speed band, 2-fold log cross-fit)

| arm / seed | speed band (4 cells, primary) | speed band x plan turn (16 cells) | tokens moved | split (i) / (ii) / (iii) (primary) |
|:--|--:|--:|--:|--:|
| `P2H10S-F-s0` | +0.14 [-0.07, +0.34] | +0.25 [-0.04, +0.54] | 9 601 | +0.23 / +0.22 / -0.31 |
| `P2H10S-F-s1` | +0.10 [-0.06, +0.25] | +0.07 [-0.22, +0.35] | 6 551 | +0.10 / +0.12 / -0.11 |
| base s0 | +0.07 [-0.17, +0.31] | +0.11 [-0.18, +0.43] | 7 839 | +0.21 / +0.22 / -0.35 |
| base s1 | +0.19 [-0.02, +0.39] | +0.30 [+0.02, +0.59] | 9 187 | +0.23 / +0.33 / -0.37 |

All primary CIs include 0 (the 0.5-1.15 grid picks the same cells). The cell picks are consistent across folds, seeds and arms, and point the
same way as the knife edge: **faster at launch** (t0 < 2 m/s: 1.05-1.10 in 7 of 8 fold-seed pairs), **slower at 2-5 m/s** (0.85-0.95, all 8),
0.95-1.00 at 5-10 m/s, mixed at >= 10. The 2-5 m/s band is where the NC failure rate peaks (76.5 of 174 failures). The training-fold gains
(+0.3 to +1.4 per cell) shrink out of fold to +0.1 overall, and the gain is half DAC-only tokens. The one CI above 0 (base s1, 16 cells,
+0.30 [+0.02, +0.59]) is a secondary reading on the control arm; its counterpart on base s0 and both primary-arm seeds include 0.

## 5. Room rule and PAI

Registered rule: both `P2H10S-F` seeds must have a uniform a != 1 with CI lo > 0, or the primary conditioned point with CI lo > 0, with the
NC / TTC tokens carrying at least half of a slowdown's gain. Result: no uniform level (best 0.95: lo -0.13 / -0.21), conditioned lo -0.07 / -0.06.
**No room; clear negative. The PAI sweep was not run** (no Tokyo card time used). Note for any later PAI work: `JEV_VCONT` is not a speed
multiplier (it is the 1.0 s blend from the ego speed into the plan's profile, decision 226), and serving has no speed-scale knob today.

## Limits

- Non-reactive traffic: slowdown recoveries are optimistic (no rear-ending charged) and new failures of a slowdown pessimistic; closed loop
  (decisions 249-251) is where the knife edge was seen, and it is not re-measured here.
- No-EC EPDMS throughout (EC is undefined for a rescaled plan); the board's EPDMS includes EC.
- The family is "same path, arc length x a" (`turn_ceiling.speed`); it cannot express "do not move" or "brake late", so the A1 launch cell is
  outside it by construction.
- Class labels are decision 153's heuristics; navhard is stage 1 counts on G frames, `P2H10S` only.
- The conditioned point has 4 / 16 cells and one fold rule; it answers "is there a servable operating point", not "is a learned selector good"
  (decision 203 closed that).
- PAI's served path carries the openpilot lead limit (`JEV_LEAD=1`), navtest plans do not; the navtest negative covers the plan's own speed only.

## Draft decision entry (for main to file; not appended to research/decisions.md)

```
# 2xx（编号由 main 定）. 当前 driver 没有比 served 更好的纵向工作点：`P2H10S-F` 在 navtest 上同路径速度缩放 0.7–1.15 的一刀切没有一档净正（0.95 打平 +0.02 [−0.13, +0.18] / −0.04 [−0.21, +0.12]，1.05 已经 −0.30 / −0.37 且 CI 不含 0），按 t0 速度段交叉拟合的条件化工作点 +0.14 [−0.07, +0.34] / +0.10 [−0.06, +0.25] 也不过线，按登记停止规则不跑 PAI；NC 失败的类型分布与第 196 条（SH30）相同，下一步的纵向修法打 A 类（本车道前方静止 / 慢车，arm 特有 41% [32, 51]）（**中**，开环 non-reactive，2 seed × 2 臂，只读存档 + CPU 重打分）

2026-10-11。lane LONG-ORACLE，任务书 overnight.md N1 改到当前 checkpoint，接第 196、203、226、232、241、247、249–251 条。预登记 [experiments/long_oracle/plans/2026-10-11-long-oracle-prereg.md](../../experiments/long_oracle/plans/2026-10-11-long-oracle-prereg.md)（任何逐 token 读数之前 push，5d9618fa），结果 [results/long_oracle.md](../../experiments/long_oracle/results/long_oracle.md)。执行者完成并起草本条，main 没有复算。不训练、不跑模型，0 卡时；box CPU 约 15 分钟。

1. **分类（navtest，seed 均值）**：`P2H10S-F` NC 失败 174（181 / 167），TTC-only 91；base `P2H10-F` 182 / 83.5；SH30 是 176.5 / 87.5。WA-JEPA 在同 token 通过 77%。A 前方静止或慢车 90.5，52% [42, 62]，arm 特有（失败而 WA-JEPA 通过）占全部 NC 失败 41% [32, 51]，下一类 E 其他 16%；A1 静止车 58、A2 同向前车 32.5、B 切入 21、C 横穿 14.5、D 侧碰 14（base 18）、E 34。最大的格子：直行 5–10 m/s 追上行驶前车 17；> 45° 2–5 m/s 撞静止车 12.5；直行 t0 < 2 m/s 撞静止车 11（WA-JEPA 全过，缩放族只救 41%：该不走而不是走慢）。`P2H10S` 对 base 只少了侧碰与几个 A，纵向块没动（第 232 条预期）。navhard stage 1：NC 18 / 15，静止车 7 为最大类。
2. **特权 oracle**（第 196 条口径）：0.7–1.1 救回 NC 72% [64, 82]，0.95–1.05 只 18%，0.85–1.15 49%；板级上限 no-EC EPDMS +1.07（base +1.13，SH30 +1.13）。
3. **一刀切**（对 a = 1.0 的 no-EC EPDMS，log 簇配对 CI）：0.95 +0.02 / −0.04（CI 含 0），0.9 −0.19 / −0.36，0.85 −0.56 / −0.76，1.05 −0.30 / −0.37，1.1 −0.89 / −0.93，1.15 −1.87 / −2.05；base 同形。served 在曲线顶上，左侧平、右侧陡：快 5% 新增 41 个 NC 失败只修 4 个。0.9–0.95 的增益里只有 DAC 改变的 token 比 NC / TTC 改变的 token 贡献更多（第 203 条的视野效应）。按类：0.9 时同向前车回收 40%、切入 48%、静止车只 23%。
4. **条件化工作点**（按 t0 速度段选档，log 两折交叉拟合）：+0.14 [−0.07, +0.34] / +0.10 [−0.06, +0.25]；16 格版 +0.25 / +0.07，CI 都含 0。选档方向一致：起步（< 2 m/s）更快 1.05–1.10，2–5 m/s 更慢 0.85–0.95，5–10 m/s 0.95–1.0。
5. **登记判定**：两 seed 都没有 CI 下界 > 0 的档或条件化点，「没有余地」，明确为负；PAI 扫描未跑。`JEV_VCONT` 不是速度倍率（第 226 条），serving 也没有速度倍率旋钮。

**对方向的含义**：(1) 纵向的失分不能靠调一个工作点拿回：全局速度是最优的，价值全在逐 token 的「哪里减、哪里不走」（定向 oracle +1.07 只花 0.05 EP），与第 196 / 203 条一致，现在对当前 checkpoint 成立。(2) 修法目标：A 类，其中两种机制分开——A2 同向前车与切入是「接近过快」，缩放 0.8 就救回 83–87%，需要的是近距对象间距 / 相对速度信号（第 203 条：openpilot lead 输出 AUC 0.71 最接近）；A1 静止车尤其直行 t0 < 2 m/s 是「不该起步」，缩放族救不回，需要一个 hold 决策。(3) 速度段的选档方向（起步更快、2–5 m/s 更慢）与闭环里的刀刃一致，但开环拿不到显著增益，不值得作为 serving 规则上 PAI。
**状态**：**成立为负**（中）。限定：non-reactive、no-EC、开环；缩放族表达不了「不走」；条件化点只有 4 / 16 格；PAI served 带 LEAD 而 navtest plan 不带。
**会推翻或推进本条的证据**：PAI 或 HUGSIM 闭环里一个速度倍率（尤其 2–5 m/s 段 0.9）对 served 的配对读数为正；一个非特权的 hold 门在 A1 起步格上的命中率与误报率。
```

## Files

- `long_oracle/tables.md`, `uniform.csv`, `uniform_seed_mean.csv`, `uniform_by_class.csv`, `uniform_new_fail_cells.csv`, `conditioned.csv`,
  `conditioned_picks.csv`, `summary.json` (room rule).
- `long_oracle/P2H10S-F/`, `long_oracle/P2H10-F/`: nc_tax reports (class tables, per-token failing rows `navtest_fail_tokens.csv`, oracle tables,
  `uniform_scaling.csv`, navhard stage 1 for `P2H10S`).
- Box: `$DATA_DIR/runs/long_oracle/{P2H10S-F,P2H10-F}/` (replay parquet, `poses.npz`, `score_fail.csv`, `score_all.csv`, gates), chain log in `chain/`.
