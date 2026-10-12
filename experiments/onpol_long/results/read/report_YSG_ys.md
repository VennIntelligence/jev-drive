# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSG-L18 (S rows, gentle continuation (Amendment 6)) | 0.8285 | 51.5 | 8.5 / 22.0 / 23.5 / 0.0 | 30.5 | 1.300 | 0.1063 | YSG-L18-s0 0.8287 / 51 / 30 / 1.300 ; YSG-L18-s1 0.8283 / 52 / 31 / 1.301 |
| control YNONE-L18 (same continuation, no rows) | 0.7743 | 78.0 | 9.0 / 37.0 / 35.5 / 0.0 | 46.0 | 1.454 | 0.1422 | YNONEG-L18-s0 0.7664 / 81 / 48 / 1.453 ; YNONEG-L18-s1 0.7823 / 75 / 44 / 1.455 |
| reference YS-L18-s0 | 0.8349 | 45.0 | 9.0 / 14.0 / 23.0 / 0.0 | 23.0 | 1.368 | 0.0963 | YS-L18-s0 0.8349 / 45 / 23 / 1.368 |
| reference YS-L18-s0, YS-L18-s1 | 0.8378 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0961 | YS-L18-s0 0.8349 / 45 / 23 / 1.368 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0542 [+0.0280, +0.0807] | -26.5 [-38.6, -15.2] | -15.5 [-24.3, -7.8] | -0.5 / -15.0 / -12.0 / +0.0 | -0.153 [-0.254, -0.063] (452 scenes) | 29 / 6 |
| arm - reference YS-L18-s0+YS-L18-s1 | -0.0093 [-0.0296, +0.0114] | +7.5 [-2.4, +17.4] | +7.0 [-2.1, +15.8] | -0.5 / +7.5 / +1.5 / +0.0 | -0.057 [-0.132, +0.024] (452 scenes) | 13 / 17 |
| control - reference YS-L18-s0+YS-L18-s1 | -0.0635 [-0.0967, -0.0309] | +34.0 [+18.4, +50.0] | +22.5 [+12.1, +33.1] | +0.0 / +22.5 / +13.5 / +0.0 | +0.097 [-0.018, +0.223] (452 scenes) | 10 / 35 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 6.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (51.5 vs 78.0); at-fault zeros not above the control's (hard gate) True (30.5 vs 46.0); GT distance not larger True (1.300 vs 1.454 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s0+YS-L18-s1: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": true, "passed": false, "score_not_lower": false} (zeros 51.5 vs 44.0, at-fault 30.5 vs 23.5, GT 1.300 vs 1.357).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": false}.
