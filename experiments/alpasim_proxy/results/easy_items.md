# EASY-ITEMS: what we lose on the windows the garage aces (2026-10-19)

Diagnostic, CPU only, no training and no new rollouts. Decision 382 found that on the private PCS we lose mainly on easy items: windows
that strong (garage-type) policies finish with exactly 1, where we score a partial or a zero. This audit lists those windows on the arena
held split, says why JTN-L18-s1 native (the nuPlan candidate) misses each one, and prices each cause on the frozen board-mix PCS of
d382 / d384.

Code: [scripts/easy_items.py](../scripts/easy_items.py).
- `extract` (box, about 2 min of CPU) pulls the per-step scorer metrics of garage 0014, LTF, JTN-s1 + stop3 and RSG + stop3.
- `report` and `pai` (Mac, `tmp/pcs_local/.venv`) produce everything below.

Outputs: [easy_items.json](easy_items.json) and [easy_items/](easy_items/), which holds one row per missed easy window
(`misses_JTN1.csv.gz` and `misses_JTN1S3.csv.gz`).

Inputs:
- the stored arena reads (pcs_local `scores.csv.gz`, `rows_*`);
- native_gap's per-decision table and per-step metrics for RSG / JTN1;
- the frozen items `boardmix.npz` / `boardmix-g15.npz`;
- d381's per-scene PAI table (`route_follow/pai_gap/scenes.csv`).

## The scorer has no sub-scores besides progress

Source: alpasim eval 0bb4c4b (`scene_score.py`, `scorers/ground_truth.py`, `aggregation/processing.py`).

- **The score is 0 on three hard failures:** an at-fault collision (front or lateral), offroad, or leaving the 4 m corridor sideways.
- **Otherwise the score is `min(progress_clipped_rel / 0.8, 1)`**, and it is 1 outright when the logged path is under 5 m.
- **`progress_clipped_rel`** is the projection of the ego polygon's centroid onto the whole logged path at the last kept step. It is
  rescaled to the part of the path reachable in the simulated time.
- **Truncation:** steps after any collision (rear included) or offroad are dropped.
- **There is no comfort, TTC, lane-margin or drivable-margin term.**

Two things follow:
- **Every partial is a progress shortfall.** A near-1 score of 0.98-0.999 means the car ended at 0.784-0.799 of the reachable human arc.
- **A lateral offset cannot cost progress**, because progress is a projection onto the path. Lateral error only matters once it becomes a
  zero (offroad or corridor).

The board's "GT distance" is something else: the per-scene maximum of `dist_to_gt_trajectory`, the distance to the whole logged path.
That includes overrunning the end of the path. A car that lags the human scores 0 on it.

## 1. The easy items

**Definition: the 2 106 frozen board-mix items (held windows the garage completed), restricted to those where garage 0014 scores exactly 1.
That gives 1 564 easy windows.**

| subset | definition | count |
|---|---|--:|
| easy | garage 0014 = 1 | **1 564** |
| easy, strict | garage, LTF and DD all = 1 | 923 |
| other | garage < 1 | 542 |

- The garage 0014 checkpoint is the only local garage-type read. No py123d-garage host variant has a local read (d382).
- **Discrimination does not work as a second definition.**
  - Median board-mix a is 2.22 on easy items and 0.10 on the other 542, so a and easy select the same windows.
  - But a is fitted with JTN-s1 in the pool, so the definition is circular.
  - **On every easy item with a >= 2 (848), JTN-s1 already scores 1.** The 181 easy windows it misses have a of 0.59-1.68, median 1.25.
  - So a high-a set picks out exactly the windows we do not lose, and garage = 1 is the definition used.

Scores on the easy windows:

| subject | mean | exact 1 | partial | of which near-1 | zero |
|---|--:|--:|--:|--:|--:|
| **JTN-s1 native** | 0.927 | 1 383 | 76 | 13 | 105 |
| JTN-s1 + stop3 | 0.983 | 1 467 | 85 | 11 | 12 |
| RSG native | 0.798 | 1 148 | 109 | 11 | 307 |
| RSG + stop3 | 0.977 | 1 423 | 122 | 12 | 19 |
| NTV0-s0 native | 0.783 | 1 143 | 90 | 21 | 331 |
| LTF | 0.676 | 1 008 | 59 | 7 | 497 |
| garage 0014 | 1.000 | 1 564 | 0 | 0 | 0 |

Where JTN-s1 native's 181 misses fall:
- static-free 83, other 42, braking 22, cruise 16, turn 10, launch 6, static-lead 2;
- with stop3: 0 static, 41 other, 19 braking, 16 cruise, 12 turn, 9 launch.

## 2. Why JTN-s1 native misses them, ranked by PCS

Each class is set to exactly 1 and the read is re-scored against the frozen items.

- The base is 1247 on board-mix and 1317 on g15.
- The CIs are log-clustered bootstraps (2 000 draws) of the paired difference.
- Causes are assigned first-match: zeros by the arena failure type, then rear-hit, then near-1, then the d384 decision-table rule.
- **The near-1 windows are the same ratchet:** 11 of the 13 have sub-cause "free-road ratchet", at a same-station speed ratio below 0.9 by
  k = 9.

| cause (JTN-s1 native, easy windows) | windows | mean points | PCS board-mix [CI] | PCS g15 [CI] |
|---|--:|--:|--:|--:|
| **zero: static window, rolls out of the corridor** (red line, no lead; d372) | 82 | 82.0 | **+79 [+37, +126]** | **+112 [+56, +171]** |
| **partial: free-road ratchet** (no lead, ego decays to about 0.4 x the human) | 47 | 5.2 | **+29 [+15, +54]** | **+30 [+15, +59]** |
| zero: offroad | 13 | 13.0 | +10 [+4, +19] | +13 [+5, +24] |
| partial: near-1 (0.98-0.999; 11 / 13 ratchet) | 13 | 0.1 | +5 [+2, +10] | +6 [+2, +11] |
| zero: corridor, moving window | 6 | 6.0 | +5 [+2, +8] | +7 [+3, +11] |
| partial: stop / lock | 4 | 0.6 | +3 [0, +7] | +3 [0, +8] |
| partial: struck from behind | 5 | 1.2 | +2 [0, +6] | +3 [0, +7] |
| zero: at-fault collision | 3 | 3.0 | +2 [0, +5] | +3 [0, +7] |
| partial: late launch / lead / other | 7 | 2.6 | +1 to +2 each | +1 each |
| union: ratchet incl. its near-1s | 58 | 5.0 | +36 [+19, +67] | +38 [+19, +75] |
| union: every moving progress partial (no rear-hit) | 71 | 9.0 | +49 [+27, +97] | +49 [+26, +105] |
| union: lateral zeros (offroad, moving corridor) | 20 | 20.0 | +16 [+9, +28] | +21 [+12, +37] |
| all 105 zeros | 105 | 105.0 | +113 [+63, +177] | +166 [+96, +258] |
| all 181 misses | 181 | 114.7 | +1480 (prior-bounded) | +1256 |

The same table for **JTN-s1 + stop3** (base 1324 / 1433). Its moving-window partials are the same windows as native's: 47 of its 48
ratchet windows are native ratchet windows. Their sub-causes are taken from native's decision table, since the hold only acts once the car
stops.

| cause (JTN-s1 + stop3) | windows | mean points | PCS board-mix [CI] | PCS g15 [CI] |
|---|--:|--:|--:|--:|
| **partial: free-road ratchet** | 48 | 5.6 | **+77 [+52, +120]** | **+104 [+69, +171]** |
| union: ratchet incl. its near-1s | 58 | 6.0 | **+101 [+69, +162]** | **+139 [+93, +232]** |
| zero: offroad | 11 | 11.0 | +15 [+7, +25] | +23 [+11, +40] |
| partial: struck from behind | 7 | 2.5 | +12 [+2, +25] | +14 [+2, +28] |
| partial: near-1 | 11 | 0.1 | +10 [+4, +19] | +15 [+5, +27] |
| partial: stop3 hold costs (native = 1) | 8 | 3.0 | +10 [0, +27] | +15 [0, +40] |
| partial: stop / lock | 4 | 0.6 | +6 [+1, +15] | +8 [+1, +21] |
| union: every moving progress partial (no rear-hit) | 78 | 13.0 | +253 [+149, +666] | +362 [+200, +723] |
| all 12 zeros | 12 | 12.0 | +16 [+7, +26] | +24 [+12, +42] |

How to read it:
- **Natively, the one big item is standing at the red line** (+79 / +112). The ratchet comes second (+36 / +38 including its near-1s).
  Everything else is at most +16. This agrees with d384, where the 98 slow windows on all held windows were worth +39 / +41; the easy
  ones carry almost all of that.
- **Once the car stands (stop3, or STAND if it ever works), the ratchet becomes the lever, at about 2.5 times the value:** +101 / +139.
  Each fix raises theta, which makes the remaining partials more expensive (d382), so gains grow superadditively. All 78 moving
  partials together are worth +253 / +362.
- **Rear-hit, near-1 and launch are small.** Near-1 windows are not a different defect: they are the ratchet stopping 0.1-1.6 % short
  of the 0.8 saturation.

## 3. Against the garage on the same windows: longitudinal, not lateral

Per-step scorer metrics on kept steps. The garage has no decision dumps, so plan-level comparison is not possible. Values are medians over
windows. Speed is ego speed from `dist_traveled_m` at 0.25 / 1.25 / 2.25 / 3.25 s after the first scored step.

| set (n) | progress ours / garage | arc / human arc ours / garage | lateral, median over time (m) ours / garage | lateral, max (m) ours / garage | speed ours (m/s) | speed garage (m/s) |
|---|---|---|---|---|---|---|
| JTN-s1 partial misses (76) | 0.74 / 0.94 | **0.68 / 0.87** | 0.09 / 0.11 | 0.20 / 0.27 | 5.7, 4.5, 3.2, 2.4 | 5.9, 5.5, 5.2, 5.2 |
| JTN-s1 + stop3 partial misses (85) | 0.72 / 0.95 | 0.67 / 0.87 | 0.09 / 0.12 | 0.15 / 0.28 | 4.7, 3.8, 2.7, 1.8 | 5.0, 5.2, 5.0, 5.1 |
| JTN-s1 = 1 (1 383) | 0.88 / 0.86 | 0.93 / 0.88 | 0.06 / 0.04 | 0.16 / 0.08 | | |

- **On the partials the shortfall is entirely longitudinal.** Both cars start at about the human's speed. The garage holds it; ours decays
  from 5.7 to 2.4 m/s within 3 s, which is the d372 / d375 ratchet. On those same windows we are laterally *closer* to the human path than
  the garage. Only 7 % of them pass 1 m lateral at any step, against 20 % for the garage.
- **On the zeros, 82 of 105 are longitudinal too:** rolling forward out of the 4 m disc at a red line. On a static window the logged path
  has almost no length, so the distance is Euclidean. 20 zeros are lateral (offroad 13, moving corridor 6, both 1) and are worth +16 / +21.
- **GT distance**, as the board defines it, on the 2 106 items:

  | subject | GT distance (m) |
  |---|--:|
  | JTN-s1 | 1.13 |
  | JTN-s1 + stop3 | 0.79 |
  | garage | 0.93 |
  | RSG | 2.43 |

  The board has 0.85 / 0.685 / 0.59. The ordering of our two entries matches, though the garage is higher locally.
- **Most of the native excess over the garage comes from the static roll-forward.** JTN-s1 minus garage is +0.20 m per window, and the
  82 static corridor zeros alone contribute +0.244 of it (with stop3, +0.10). On easy windows where we score 1 the excess is +0.03. On
  the moving partials it is -0.01. On the 542 windows the garage misses it is -0.08.

**Answer: the shortfall is longitudinal.** Natively, the higher GT distance is the static creep, and on moving windows it is planning short.
Path quality is not where we lose.

## 4. PAI 281

- **No strong-model local PAI read exists.**
  - No garage, LTF, DD, Alpamayo or host read covers the PAI 281 scenes. The only third-party PAI numbers are board aggregates (d312,
    d356).
  - So easy PAI items and their discrimination cannot be identified, and nothing on PAI can be priced in PCS.
- **Weaker proxy:** call a scene easy when the two other family candidates (RSG-s0 and JTN-s1, both + ACC + GAP) score exactly 1.

  | subset | scenes | NTV0 + ACC + GAP points lost |
  |---|--:|--:|
  | easy: RSG and JTN-s1 both = 1 | 156 | **1.0** (one drift corridor zero) |
  | 155 of those, where NTV0 is also 1 | 155 | - |
  | one of RSG / JTN-s1 = 1 | 26 | 3.0 |
  | neither = 1 | 98 | **67.6** |

  Inside "neither", all of d381's classes appear:
  - passing the human's rest point, 21;
  - turn / junction, 14 corridor + 12 offroad;
  - rear-hit partials, 10.4;
  - drift / lane, 5;
  - AF-front, 3;
  - slow, 2.1.
- **Reading:** on PAI our family fails together. 94 % of A0's lost points sit on scenes that all three of our candidates fail. The human
  scores 1.000 on 277 / 277 (d381 H), so all of them are attainable.
- Whether they are easy items in the PCS sense depends on whether the host-like garage variants ace them. That is unknown locally. If they
  do, the nuPlan pattern applies: rear-hit and slow partials on scenes strong models ace would cost far more than their 12.5 points
  suggest.
- The proxy cannot separate easy from hard here. It only shows there is no "near-easy" PAI loss that one of our own models already solves.

## 5. Levers

1. **Stand at a red / stop line with no lead (native: +79 / +112 PCS; 82 easy windows).**
   - This maps to d372's red-line static windows. Serve-side, stop3 already buys it: +82 / +124 on the full held (d382), +138 on the
     board.
   - Model-side it is the STAND line (NR1 supplement 17; no decision row yet). Pilot B, `STANDFMX1K` (STAND + FREE-M) against
     `JTX1K`, moved both easy-item causes the right way on its 753-window subset: static zeros 74 -> 44, slow 48 -> 25, near-1
     12 -> 6, k = 9 speed ratio 0.85 -> 0.95. It failed the held 460 guards (zeros 37 -> 45, front collisions 2 -> 7). See
     [stand/jtx/pilot_B.md](../../defect_fix/results/stand/jtx/pilot_B.md). Its frozen PCS has not been scored here.
   - Not to repeat: BN control continuation (d379 / d384) got its mean gain here but re-grew slow partials and was PCS-negative on s1.
     STOPPT (d386) is the moving-approach version; it did not fire in the pool and was not adopted.
   - **The bar for any STAND arm:** static zeros on garage-1 windows go down, and the 58 ratchet windows do not grow. Score it with this
     table (`easy_items.py report` on its rows).
2. **Free-road speed ratchet on moving windows (native +36 / +38; on a standing base +101 / +139; 58 easy windows, 47 + 11 near-1).**
   - This maps to d372 / d375 / d384: the plan asks for 7-17 % less than the current speed at every step, and the frame channel locks
     each shortfall in.
   - Closed or failed:
     - FREE-M (d378 / d384): it fixed 30 slow windows but added 29 static zeros, net +5 / +11;
     - history-only fixes (d320 / d343 / JEV_VHIST): they cannot reach the frame channel (d375);
     - `SH30_WARP=const` (d375): -0.009;
     - best-of-N / continuation (d379): it re-grows the ratchet;
     - PNX's new motion rows (d385): JTXS slow 105 against 98 for JTN-s1, PCS -9.
   - The one arm that moved it is STAND + FREE-M (pilot B above), which paired the ratchet fix with a standing term.
   - **What is left is the d375 recommendation**, a training-side fix: the policy's own slow-trajectory frames with the human's speed
     labels, applied only to moving free-road rows. It has to be built on, or jointly with, a parent that stands, because at native
     ability the ratchet is worth a third as much.
   - Its guard is the 82 static windows: FREE-M's failure mode was exactly trading ratchet fixes for static zeros. That trade costs about
     0.7 PCS per static zero against 0.4-1.7 per ratchet fix, so the arm has to hold static zeros flat.

Not levers:
- lateral path quality (20 zeros, +16 / +21);
- rear-hit (+2 / +12);
- launch (+1 / +2);
- comfort or margin terms, which the scorer does not have.

## Limits

- **One garage checkpoint** stands for the strong pool. The host-like variants may ace a different set, and the garage weight in
  board-mix is not identified (±100 PCS, d382).
- Single classes are priced one at a time, and gains are superadditive, so the unions are not sums.
- The "all misses" CIs are prior-bounded.
- The cause rules are d384's heuristic thresholds.
- Stop3 sub-causes are inherited from native's decision table. Only 8 of stop3's windows are its own (native = 1).
- 1 seed per read.
- The local GT distance is 0.15-0.34 m above the board's, and the board's suite composition differs from the arena's (d325).
