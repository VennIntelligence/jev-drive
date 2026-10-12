# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS2-L18 (round 2: S rows of YS-L18 rollouts, from YS-L18-s1) | 0.8326 | 50.5 | 7.5 / 23.0 / 22.0 / 0.0 | 30.5 | 1.396 | 0.1220 | YS2-L18-s0 0.8383 / 48 / 30 / 1.396 ; YS2-L18-s1 0.8268 / 53 / 31 / 1.395 |
| control YNONE-L18 (same continuation, no rows) | 0.8239 | 51.5 | 6.0 / 23.5 / 25.5 / 0.0 | 29.5 | 1.370 | 0.0976 | YS2NONE-L18-s0 0.8248 / 50 / 27 / 1.351 ; YS2NONE-L18-s1 0.8231 / 53 / 32 / 1.389 |
| reference YR10m10-F-s0 | 0.7849 | 74.0 | 10.0 / 33.0 / 35.0 / 0.0 | 43.0 | 1.495 | 0.1420 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 |
| reference YR10m10-F-s0, YR10m10-F-s1 | 0.7901 | 72.5 | 7.0 / 35.0 / 34.0 / 0.0 | 42.0 | 1.449 | 0.1342 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0086 [-0.0159, +0.0331] | -1.0 [-13.0, +11.1] | +1.0 [-8.8, +10.3] | +1.5 / -0.5 / -3.5 / +0.0 | +0.025 [-0.082, +0.125] (452 scenes) | 13 / 8 |
| arm - reference YR10m10-F-s0 | +0.0477 [+0.0179, +0.0776] | -23.5 [-36.9, -10.3] | -12.5 [-22.8, -1.2] | -2.5 / -10.0 / -13.0 / +0.0 | -0.099 [-0.226, +0.022] (452 scenes) | 37 / 11 |
| control - reference YR10m10-F-s0 | +0.0390 [+0.0070, +0.0714] | -22.5 [-37.5, -7.4] | -13.5 [-25.5, -1.5] | -4.0 / -9.5 / -9.5 / +0.0 | -0.125 [-0.273, +0.021] (452 scenes) | 37 / 10 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 3.

**Adoption line (registered): NOT adopted** - fewer zeros than the control True (50.5 vs 51.5); at-fault zeros not above the control's (hard gate) False (30.5 vs 29.5); GT distance not larger False (1.396 vs 1.370 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 50.5 vs 74.0, at-fault 30.5 vs 43.0, GT 1.396 vs 1.495).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": true, "met": false}.
