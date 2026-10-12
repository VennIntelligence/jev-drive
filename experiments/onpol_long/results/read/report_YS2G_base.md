# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS2G-L18 (round 2b: round-2 rows, gentle, from YS-L18-s1 (Amendment 8)) | 0.8474 | 41.5 | 7.0 / 15.0 / 19.5 / 0.0 | 22.0 | 1.276 | 0.0903 | YS2G-L18-s0 0.8434 / 43 / 24 / 1.272 ; YS2G-L18-s1 0.8514 / 40 / 20 / 1.279 |
| control YNONE-L18 (same continuation, no rows) | 0.8261 | 48.5 | 12.5 / 16.5 / 21.0 / 0.0 | 29.0 | 1.310 | 0.0943 | YS2GNONE-L18-s0 0.8195 / 50 / 31 / 1.320 ; YS2GNONE-L18-s1 0.8328 / 47 / 27 / 1.300 |
| reference YR10m10-F-s0 | 0.7849 | 74.0 | 10.0 / 33.0 / 35.0 / 0.0 | 43.0 | 1.495 | 0.1420 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 |
| reference YR10m10-F-s0, YR10m10-F-s1 | 0.7901 | 72.5 | 7.0 / 35.0 / 34.0 / 0.0 | 42.0 | 1.449 | 0.1342 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0212 [+0.0073, +0.0351] | -7.0 [-14.3, +0.0] | -7.0 [-13.8, +0.0] | -5.5 / -1.5 / -1.5 / +0.0 | -0.035 [-0.083, +0.015] (452 scenes) | 7 / 3 |
| arm - reference YR10m10-F-s0 | +0.0625 [+0.0353, +0.0905] | -32.5 [-45.3, -19.8] | -21.0 [-30.5, -10.5] | -3.0 / -18.0 / -15.5 / +0.0 | -0.219 [-0.339, -0.106] (452 scenes) | 40 / 6 |
| control - reference YR10m10-F-s0 | +0.0412 [+0.0127, +0.0705] | -25.5 [-39.1, -12.2] | -14.0 [-24.2, -3.4] | +2.5 / -16.5 / -14.0 / +0.0 | -0.185 [-0.325, -0.054] (452 scenes) | 37 / 11 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 3.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (41.5 vs 48.5); at-fault zeros not above the control's (hard gate) True (22.0 vs 29.0); GT distance not larger True (1.276 vs 1.310 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 41.5 vs 74.0, at-fault 22.0 vs 43.0, GT 1.276 vs 1.495).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": true, "met": false}.
