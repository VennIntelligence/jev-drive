# Turning school arms: selection ladder result, 2026-10-13

Prereg: [../../plans/2026-10-13-turning-school-prereg.md](../../plans/2026-10-13-turning-school-prereg.md) (Amendments 1-3). Pools: on1_PAIT
5 160 rows / 53 strict turns (gates [../pait_gates_PAIT80.json](../pait_gates_PAIT80.json), [../pait_gates_PAITn.json](../pait_gates_PAITn.json):
route, road, speed and R1 pass), on1_PAITC = + on1_CFTv, on1_RSGPAIT = on1_L18RB0 + on1_PAIT (PAI rows 30 % of draws, W 1.0306).
All arms: init RB0-F-s0, gentle schedule (2 500 steps, lr x 0.1). Box chain rf_tschool.sh; RSGboth-L18 from [../rsgboth/summary.md](../rsgboth/summary.md).

## Ladder step 2 (s0 must pass both): held 460 @18 mean >= RSG-L18 - 0.01 (0.8566) and zeros <= RSG-L18-s0 + 3 (37); public 1479 @10 zeros <= RSG-L18 + 5 (50)

| arm | held 460 mean s0 / s1 | held zeros s0 / s1 | public mean s0 / s1 | public zeros s0 / s1 | step 2 (s0) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18 (reference) | 0.8656 / 0.8676 | 34 / 34 | 0.9473 / 0.9487 | 46 / 44 | - |
| **RSGPAIT-L18** (primary) | 0.9045 / 0.9057 | 37 / 36 | 0.9500 / 0.9495 | 56 / 54 | held PASS (37 = line), guard **FAIL** (+10 paired [+2.0, +18.6] vs RSG-L18-s0) |
| RSGboth-L18 (other agent) | 0.9020 / 0.8922 | 41 / 46 | 0.9522 / 0.9539 | 58 / 56 | held zeros FAIL, guard FAIL |
| RPAIT-L18 | 0.8147 / 0.8081 | 47 / 48 | 0.9113 / 0.9104 | 70 / 70 | FAIL both |
| RPAITC-L18 | 0.8029 / 0.8175 | 48 / 42 | 0.9115 / 0.9122 | 65 / 67 | FAIL both |
| RNONEG-L18 (no rows) | 0.8156 / 0.8123 | 60 / 61 | - | - | - |

**No arm survives step 2, so per the ladder nothing goes to the offline ranking or to the Tokyo exam; no EXAM_READ_REQUEST was written.**

Reading: RSGPAIT is the best nuPlan arm of the batch on the held mean (+0.040 [+0.016, +0.064] vs RSG-L18-s0 on held 452, GT distance
1.75 m vs 1.14 m: it drives much farther) and ties RSG's public mean (+0.003 [-0.004, +0.009]), but it pays about +10 public zeros (offroad 36
vs 31, corridor 11 vs 8, at-fault +6). Against RSGboth (same `both` mode, no PAI rows) it has fewer held zeros (37 / 36 vs 41 / 46) and a
similar public count (56 / 54 vs 58 / 56): the extra zeros come with the `both` mode on the S rows, not with the PAI rows; the PAI rows do not
hurt nuPlan when they are 30 % of the pool draws inside the S recipe. PAI-only arms (RPAIT, RPAITC) leave the nuPlan domain (held -0.056,
public +20 to +25 zeros vs RSG), as the 80-scene pilots showed.

## Offline PAI turn read (diagnostic only, not a ladder step here: no survivor)

[../pai_openloop/p1_diag.md](../pai_openloop/p1_diag.md): P1 log-following rows (48 route-follow turn scenes, RSG-L18-s0 driver) minus
hard_exam = 19 turns, all in hard_train (in-sample for the PAIT-trained arms, out-of-sample for RSG / RSGboth / RB0). The read reproduces the
driver's served plan (RSG-L18-s0: median 0.010 m, p95 0.028 m; [../pai_openloop/p1check.json](../pai_openloop/p1check.json)).
Median plan speed excess over the human at the turn entry (0 m): RB0 0.38, RSG 0.34, RSGboth 0.35, **RSGPAIT 0.23**, RPAIT 0.12, RPAITC
0.14 m/s (straight-ahead turns: 0.57 / 0.50 / 0.45 / 0.18 / 0.07 / 0.09); at 20 m before: human 5.93, RSG 6.40, RSGboth 6.81, RSGPAIT 6.20 m/s.
lat2 0.96-1.00 and side4 1.00 for all: at the human's own states every checkpoint turns as needed. Open loop at human states the speed
excess is small for every arm (the plan follows the ego's speed history): the closed-loop failure is entering at the speed the policy itself
built up, which this read cannot see.
