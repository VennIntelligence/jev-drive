# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS-L18 (S hinges only on the same rows (Amendment 3)) | 0.8378 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0961 | YS-L18-s0 0.8349 / 45 / 23 / 1.368 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |
| control YNONE-L18 (same continuation, no rows) | 0.6932 | 120.0 | 23.5 / 56.0 / 51.5 / 0.0 | 79.5 | 1.746 | 0.1910 | YNONE-L18-s0 0.6840 / 124 / 81 / 1.727 ; YNONE-L18-s1 0.7024 / 116 / 78 / 1.766 |
| base YR10m10-F-s0 | 0.7849 | 74.0 | 10.0 / 33.0 / 35.0 / 0.0 | 43.0 | 1.495 | 0.1420 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 |
| base YR10m10-F seeds 0, 1 | 0.7901 | 72.5 | 7.0 / 35.0 / 34.0 / 0.0 | 42.0 | 1.449 | 0.1342 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.1446 [+0.1187, +0.1730] | -76.0 [-89.9, -63.5] | -56.0 [-69.8, -43.1] | -14.5 / -41.5 / -29.5 / +0.0 | -0.389 [-0.509, -0.272] (452 scenes) | 52 / 5 |
| arm - base s0 | +0.0529 [+0.0241, +0.0823] | -30.0 [-43.5, -16.7] | -19.5 [-28.8, -9.3] | -1.0 / -18.5 / -13.0 / +0.0 | -0.138 [-0.266, -0.014] (452 scenes) | 38 / 10 |
| control - base s0 | -0.0917 [-0.1268, -0.0596] | +46.0 [+30.8, +62.6] | +36.5 [+21.4, +53.1] | +13.5 / +23.0 / +16.5 / +0.0 | +0.252 [+0.116, +0.392] (452 scenes) | 15 / 38 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 8.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (44.0 vs 120.0); at-fault zeros not above the control's (hard gate) True (23.5 vs 79.5); GT distance not larger True (1.357 vs 1.746 m).

CL-1 as written in closed-loop.md (reported beside it): met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": true}.
