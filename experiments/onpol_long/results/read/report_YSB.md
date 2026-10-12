**VOID (10-12 evening; decision 274 voided in place).** This read used an oncoming removal set (lane heading at the point nearest the row, whole roadblocks) that marks 9.9 % of the human's own positions as oncoming (zeros_mech's lane rule: 0.4 %), so it measured the wrong raster. Arm B is redone on the corrected region in Amendment 15; its result is added below this header when it lands. The original read follows unchanged.

# ONPOL-LONG Amendment 13: arm B, directional road raster (`YSB-L18`, round 2b's recipe with the directional S road raster; held 452, 18 decisions, cards 1-5)

Registered lines (seed-mean point estimates; control `YS2G-L18`, same pool, schedule, start; only the road raster differs):

| line | YSB | YS2G | YS-L18 | base YR10m10-F-s0 | met |
|:--|--:|--:|--:|--:|:--|
| 1. early-inside zeros on across-traffic turns < YS2G | 3 | 5.5 | 10.5 | 6 | True |
| 2. late-outside zeros on kerb-side turns <= YS2G | 10.5 | 7.5 | 6 | 17 | False |
| 3. share of corridor + collision zeros with the rear axle in an oncoming lane in the 2 s before < 0.17 | 0.133 (8 / 60) | 0.226 (12 / 53) | 0.305 (18 / 59) | 0.171 (7 / 41) | True |
| 4. zeros <= 41.5 | 53 | 41.5 | 44 | 74 | False |
| 5. GT distance <= YS2G (selection quantity) | 1.400 | 1.276 | 1.357 | 1.495 | False |
| separate (evaluator artefact): straight-scene corridor zeros past the log end / all straight corridor zeros | 3 / 4 | 4.5 / 6 | 4.5 / 6 | 2 / 7 | - |

**Verdict: not adopted** (note added after Amendment 14's label build: the removal set's geometry also marks part of the human's own path as oncoming, see report_YSB2.md, so this is not a clean negative for a directional raster as such) (lines 1 and 3 met, 2, 4 and 5 not). Mechanism lines from zeros_mech's `zm_read.py collect / report` run unedited on these reads (collect first written to `runs/zeros_mech/collect/20261012-175711`, moved on 10-12 evening to `runs/onpol_long/zm/zeros_mech/collect/20261012-175711` so that it does not shadow the zeros_mech lane's latest collect directory, gates G-act / G-dir pass, 0 scene errors; its YS-L18 / YS2G / base numbers reproduce decision 268: 18 / 59, 12 / 53, 7 / 41); `YSB` and the off-card-0 re-read of `YS-L18-s0` enter it through merged views of their 6 read chunks. The line computation is `ol_dirraster.py lines`.

Labels (`ol_dirraster.py build`, pool `on1_L18YS`, 34 788 rows): gate (i) the undirected rebuild equals the pool raster on 200 rows (max difference 0.0); every row has a lane or roadblock removed (mean 14.4 lanes within the 80 m query), 0.091 of the raster pixels change sign; the served plan's road hinge (any footprint corner within 0.3 m) is active on 0.122 of rows before and 0.285 after (6718 newly active rows; 24 % of a 400-row sample of those have the t0 footprint already across the new edge, where training falls back to the undirected drivable raster). The centre line between directions is now a road edge, so the hinge pushes plans toward the kerb all the time; figure `figs/armb_dir_raster_turns.png` (three early-inside turn zeros of YS-L18-s1, two decisions before the event).

Stage timings (pool, real): label build 0.7 min (64 cores); training 6.7 min per seed (cards 3 / 4, 2 500 steps); held read 7.3-9.1 min per 77-scene chunk, two stacks per card on cards 1-5 (12 chunks, ~23 min wall including queueing behind each other); report + zm_read 0.7 min. Submit to report: about 27 min.

---

# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSB-L18 (SB) | 0.8290 | 53.0 | 8.5 / 23.0 / 23.0 / 0.0 | 31.5 | 1.400 | 0.1141 | YSB-L18-s0 0.8346 / 51 / 31 / 1.396 ; YSB-L18-s1 0.8235 / 55 / 32 / 1.403 |
| control YS2G-L18 | 0.8474 | 41.5 | 7.0 / 15.0 / 19.5 / 0.0 | 22.0 | 1.276 | 0.0903 | YS2G-L18-s0 0.8434 / 43 / 24 / 1.272 ; YS2G-L18-s1 0.8514 / 40 / 20 / 1.279 |
| reference YS-L18-s0 | 0.8345 | 45.0 | 9.0 / 14.0 / 23.0 / 0.0 | 23.0 | 1.367 | 0.0962 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 |
| reference YS-L18-s0, YS-L18-s1 | 0.8376 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0960 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | -0.0184 [-0.0417, +0.0015] | +11.5 [+1.8, +22.7] | +9.5 [-0.5, +19.5] | +1.5 / +8.0 / +3.5 / +0.0 | +0.124 [+0.013, +0.232] (452 scenes) | 9 / 18 |
| arm - reference YS-L18-s0+YS-L18-s1 | -0.0086 [-0.0338, +0.0138] | +9.0 [-1.6, +20.8] | +8.0 [-1.2, +17.3] | -0.5 / +8.5 / +1.0 / +0.0 | +0.043 [-0.099, +0.183] (452 scenes) | 13 / 22 |
| control - reference YS-L18-s0+YS-L18-s1 | +0.0098 [-0.0003, +0.0205] | -2.5 [-7.3, +1.8] | -1.5 [-6.1, +3.3] | -2.0 / +0.5 / -2.5 / +0.0 | -0.081 [-0.141, -0.024] (452 scenes) | 5 / 2 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 3.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (53.0 vs 41.5); at-fault zeros not above the control's (hard gate) False (31.5 vs 22.0); GT distance not larger False (1.400 vs 1.276 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s0+YS-L18-s1: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": false, "passed": false, "score_not_lower": false} (zeros 53.0 vs 44.0, at-fault 31.5 vs 23.5, GT 1.400 vs 1.357).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
