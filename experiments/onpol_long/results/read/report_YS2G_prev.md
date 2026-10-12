# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS2G-L18 (round 2b: round-2 rows, gentle, from YS-L18-s1 (Amendment 8)) | 0.8474 | 41.5 | 7.0 / 15.0 / 19.5 / 0.0 | 22.0 | 1.276 | 0.0903 | YS2G-L18-s0 0.8434 / 43 / 24 / 1.272 ; YS2G-L18-s1 0.8514 / 40 / 20 / 1.279 |
| control YNONE-L18 (same continuation, no rows) | 0.8261 | 48.5 | 12.5 / 16.5 / 21.0 / 0.0 | 29.0 | 1.310 | 0.0943 | YS2GNONE-L18-s0 0.8195 / 50 / 31 / 1.320 ; YS2GNONE-L18-s1 0.8328 / 47 / 27 / 1.300 |
| reference YS-L18-s1 | 0.8407 | 43.0 | 9.0 / 15.0 / 21.0 / 0.0 | 24.0 | 1.346 | 0.0959 | YS-L18-s1 0.8407 / 43 / 24 / 1.346 |
| reference YS-L18-s1, YS-L18-s0 | 0.8378 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0961 | YS-L18-s1 0.8407 / 43 / 24 / 1.346 ; YS-L18-s0 0.8349 / 45 / 23 / 1.368 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0212 [+0.0073, +0.0351] | -7.0 [-14.3, +0.0] | -7.0 [-13.8, +0.0] | -5.5 / -1.5 / -1.5 / +0.0 | -0.035 [-0.083, +0.015] (452 scenes) | 7 / 3 |
| arm - reference YS-L18-s1 | +0.0067 [-0.0065, +0.0195] | -1.5 [-7.4, +4.4] | -2.0 [-8.7, +4.5] | -2.0 / +0.0 / -1.5 / +0.0 | -0.070 [-0.136, -0.006] (452 scenes) | 10 / 7 |
| control - reference YS-L18-s1 | -0.0146 [-0.0261, -0.0024] | +5.5 [-1.0, +11.7] | +5.0 [+0.0, +10.1] | +3.5 / +1.5 / +0.0 / +0.0 | -0.036 [-0.092, +0.020] (452 scenes) | 4 / 5 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 3.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (41.5 vs 48.5); at-fault zeros not above the control's (hard gate) True (22.0 vs 29.0); GT distance not larger True (1.276 vs 1.310 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s1: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 41.5 vs 43.0, at-fault 22.0 vs 24.0, GT 1.276 vs 1.346).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": true, "met": false}.
