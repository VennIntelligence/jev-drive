# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS2-L18 (round 2: S rows of YS-L18 rollouts, from YS-L18-s1) | 0.8326 | 50.5 | 7.5 / 23.0 / 22.0 / 0.0 | 30.5 | 1.396 | 0.1220 | YS2-L18-s0 0.8383 / 48 / 30 / 1.396 ; YS2-L18-s1 0.8268 / 53 / 31 / 1.395 |
| control YNONE-L18 (same continuation, no rows) | 0.8239 | 51.5 | 6.0 / 23.5 / 25.5 / 0.0 | 29.5 | 1.370 | 0.0976 | YS2NONE-L18-s0 0.8248 / 50 / 27 / 1.351 ; YS2NONE-L18-s1 0.8231 / 53 / 32 / 1.389 |
| reference YS-L18-s1 | 0.8407 | 43.0 | 9.0 / 15.0 / 21.0 / 0.0 | 24.0 | 1.346 | 0.0959 | YS-L18-s1 0.8407 / 43 / 24 / 1.346 |
| reference YS-L18-s1, YS-L18-s0 | 0.8378 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0961 | YS-L18-s1 0.8407 / 43 / 24 / 1.346 ; YS-L18-s0 0.8349 / 45 / 23 / 1.368 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0086 [-0.0159, +0.0331] | -1.0 [-13.0, +11.1] | +1.0 [-8.8, +10.3] | +1.5 / -0.5 / -3.5 / +0.0 | +0.025 [-0.082, +0.125] (452 scenes) | 13 / 8 |
| arm - reference YS-L18-s1 | -0.0082 [-0.0354, +0.0156] | +7.5 [-3.8, +19.7] | +6.5 [-2.0, +14.7] | -1.5 / +8.0 / +1.0 / +0.0 | +0.050 [-0.059, +0.162] (452 scenes) | 19 / 20 |
| control - reference YS-L18-s1 | -0.0168 [-0.0426, +0.0073] | +8.5 [-3.2, +21.3] | +5.5 [-4.8, +15.1] | -3.0 / +8.5 / +4.5 / +0.0 | +0.024 [-0.067, +0.114] (452 scenes) | 16 / 15 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 3.

**Adoption line (registered): NOT adopted** - fewer zeros than the control True (50.5 vs 51.5); at-fault zeros not above the control's (hard gate) False (30.5 vs 29.5); GT distance not larger False (1.396 vs 1.370 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s1: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": false, "passed": false, "score_not_lower": false} (zeros 50.5 vs 43.0, at-fault 30.5 vs 24.0, GT 1.396 vs 1.346).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": false, "fixed_minus_broken_over_null": true, "met": false}.
