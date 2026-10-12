# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSG-L18 (S rows, gentle continuation (Amendment 6)) | 0.8285 | 51.5 | 8.5 / 22.0 / 23.5 / 0.0 | 30.5 | 1.300 | 0.1063 | YSG-L18-s0 0.8287 / 51 / 30 / 1.300 ; YSG-L18-s1 0.8283 / 52 / 31 / 1.301 |
| control YNONE-L18 (same continuation, no rows) | 0.7743 | 78.0 | 9.0 / 37.0 / 35.5 / 0.0 | 46.0 | 1.454 | 0.1422 | YNONEG-L18-s0 0.7664 / 81 / 48 / 1.453 ; YNONEG-L18-s1 0.7823 / 75 / 44 / 1.455 |
| reference YR10m10-F-s0 | 0.7849 | 74.0 | 10.0 / 33.0 / 35.0 / 0.0 | 43.0 | 1.495 | 0.1420 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 |
| reference YR10m10-F-s0, YR10m10-F-s1 | 0.7901 | 72.5 | 7.0 / 35.0 / 34.0 / 0.0 | 42.0 | 1.449 | 0.1342 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0542 [+0.0280, +0.0807] | -26.5 [-38.6, -15.2] | -15.5 [-24.3, -7.8] | -0.5 / -15.0 / -12.0 / +0.0 | -0.153 [-0.254, -0.063] (452 scenes) | 29 / 6 |
| arm - reference YR10m10-F-s0 | +0.0436 [+0.0194, +0.0693] | -22.5 [-34.0, -11.6] | -12.5 [-21.4, -4.0] | -1.5 / -11.0 / -11.5 / +0.0 | -0.194 [-0.304, -0.089] (452 scenes) | 31 / 7 |
| control - reference YR10m10-F-s0 | -0.0106 [-0.0266, +0.0053] | +4.0 [-3.8, +12.1] | +3.0 [-5.0, +11.6] | -1.0 / +4.0 / +0.5 / +0.0 | -0.041 [-0.075, -0.004] (452 scenes) | 4 / 6 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 6.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (51.5 vs 78.0); at-fault zeros not above the control's (hard gate) True (30.5 vs 46.0); GT distance not larger True (1.300 vs 1.454 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 51.5 vs 74.0, at-fault 30.5 vs 43.0, GT 1.300 vs 1.495).

CL-1 as written in closed-loop.md (reported beside it): met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": true}.
