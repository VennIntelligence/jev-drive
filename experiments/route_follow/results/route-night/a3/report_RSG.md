# ONPOL-LONG round 1, family R (base RB0-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm RSG-L18 (S rows, gentle continuation (Amendment 6)) | 0.8819 | 26.0 | 5.0 / 15.0 / 7.0 / 0.0 | 20.0 | 1.140 | 0.0809 | RSG-L18-s0 0.8809 / 26 / 21 / 1.135 ; RSG-L18-s1 0.8830 / 26 / 19 / 1.144 |
| control RNONE-L18 (same continuation, no rows) | 0.7974 | 69.5 | 16.5 / 39.5 / 15.5 / 0.0 | 56.0 | 1.489 | 0.1222 | RNONE-L18-s0 0.7946 / 72 / 55 / 1.526 ; RNONE-L18-s1 0.8002 / 67 / 57 / 1.452 |
| reference RB0-F-s0 | 0.8366 | 51.0 | 7.0 / 29.0 / 17.0 / 0.0 | 36.0 | 1.314 | 0.1033 | RB0-F-s0 0.8366 / 51 / 36 / 1.314 |
| reference RB0-F-s0, RB0-F-s1 | 0.8382 | 51.0 | 6.0 / 33.0 / 13.5 / 0.0 | 39.0 | 1.276 | 0.1022 | RB0-F-s0 0.8366 / 51 / 36 / 1.314 ; RB0-F-s1 0.8399 / 51 / 42 / 1.238 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0845 [+0.0630, +0.1071] | -43.5 [-55.0, -32.4] | -36.0 [-45.4, -26.6] | -11.5 / -24.5 / -8.5 / +0.0 | -0.349 [-0.434, -0.275] (452 scenes) | 31 / 3 |
| arm - reference RB0-F-s0 | +0.0454 [+0.0215, +0.0719] | -25.0 [-36.9, -14.2] | -16.0 [-26.4, -6.7] | -2.0 / -14.0 / -10.0 / +0.0 | -0.174 [-0.252, -0.105] (452 scenes) | 27 / 4 |
| control - reference RB0-F-s0 | -0.0392 [-0.0603, -0.0177] | +18.5 [+6.9, +30.2] | +20.0 [+9.9, +30.7] | +9.5 / +10.5 / -1.5 / +0.0 | +0.175 [+0.092, +0.261] (452 scenes) | 14 / 21 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons RNONE-L18 s0 / s1 and RB0-F s0 / s1 = 5.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (26.0 vs 69.5); at-fault zeros not above the control's (hard gate) True (20.0 vs 56.0); GT distance not larger True (1.140 vs 1.489 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against RB0-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 26.0 vs 51.0, at-fault 20.0 vs 36.0, GT 1.140 vs 1.314).

CL-1 as written in closed-loop.md (reported beside it): met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": true}.
