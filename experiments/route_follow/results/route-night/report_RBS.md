# ONPOL-LONG round 1, family R (base RB0-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm RBS-L18 (BS) | 0.7051 | 102.5 | 22.0 / 67.5 / 15.0 / 0.0 | 89.5 | 1.342 | 0.0912 | RBS-L18-s0 0.6832 / 114 / 98 / 1.391 ; RBS-L18-s1 0.7270 / 91 / 81 / 1.294 |
| control RNONE-L18 (same continuation, no rows) | 0.7974 | 69.5 | 16.5 / 39.5 / 15.5 / 0.0 | 56.0 | 1.489 | 0.1222 | RNONE-L18-s0 0.7946 / 72 / 55 / 1.526 ; RNONE-L18-s1 0.8002 / 67 / 57 / 1.452 |
| reference RB0-F-s0 | 0.8366 | 51.0 | 7.0 / 29.0 / 17.0 / 0.0 | 36.0 | 1.314 | 0.1033 | RB0-F-s0 0.8366 / 51 / 36 / 1.314 |
| reference RB0-F-s0, RB0-F-s1 | 0.8382 | 51.0 | 6.0 / 33.0 / 13.5 / 0.0 | 39.0 | 1.276 | 0.1022 | RB0-F-s0 0.8366 / 51 / 36 / 1.314 ; RB0-F-s1 0.8399 / 51 / 42 / 1.238 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | -0.0923 [-0.1269, -0.0523] | +33.0 [+13.6, +49.4] | +33.5 [+15.8, +49.2] | +5.5 / +28.0 / -0.5 / +0.0 | -0.147 [-0.302, +0.016] (452 scenes) | 25 / 44 |
| arm - reference RB0-F-s0 | -0.1314 [-0.1673, -0.0914] | +51.5 [+33.2, +66.8] | +53.5 [+37.0, +67.5] | +15.0 / +38.5 / -2.0 / +0.0 | +0.028 [-0.123, +0.191] (452 scenes) | 20 / 48 |
| control - reference RB0-F-s0 | -0.0392 [-0.0603, -0.0177] | +18.5 [+6.9, +30.2] | +20.0 [+9.9, +30.7] | +9.5 / +10.5 / -1.5 / +0.0 | +0.175 [+0.092, +0.261] (452 scenes) | 14 / 21 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons RNONE-L18 s0 / s1 and RB0-F s0 / s1 = 5.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (102.5 vs 69.5); at-fault zeros not above the control's (hard gate) False (89.5 vs 56.0); GT distance not larger True (1.342 vs 1.489 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against RB0-F-s0: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": false, "passed": false, "score_not_lower": false} (zeros 102.5 vs 51.0, at-fault 89.5 vs 36.0, GT 1.342 vs 1.314).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": false, "met": false}.
