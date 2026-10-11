# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSKC-L18 (S + K + C on-policy rows) | 0.6944 | 96.0 | 18.0 / 62.0 / 17.0 / 0.0 | 80.0 | 1.243 | 0.0806 | YSKC-L18-s0 0.6938 / 94 / 76 / 1.225 ; YSKC-L18-s1 0.6951 / 98 / 84 / 1.261 |
| control YNONE-L18 (same continuation, no rows) | 0.6932 | 120.0 | 23.5 / 56.0 / 51.5 / 0.0 | 79.5 | 1.746 | 0.1910 | YNONE-L18-s0 0.6840 / 124 / 81 / 1.727 ; YNONE-L18-s1 0.7024 / 116 / 78 / 1.766 |
| base YR10m10-F-s0 | 0.7849 | 74.0 | 10.0 / 33.0 / 35.0 / 0.0 | 43.0 | 1.495 | 0.1420 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 |
| base YR10m10-F seeds 0, 1 | 0.7901 | 72.5 | 7.0 / 35.0 / 34.0 / 0.0 | 42.0 | 1.449 | 0.1342 | YR10m10-F-s0 0.7849 / 74 / 43 / 1.495 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0013 [-0.0437, +0.0515] | -24.0 [-47.9, -2.5] | +0.5 [-21.5, +21.4] | -5.5 / +6.0 / -34.5 / +0.0 | -0.503 [-0.660, -0.358] (452 scenes) | 41 / 33 |
| arm - base s0 | -0.0905 [-0.1313, -0.0456] | +22.0 [+1.5, +40.4] | +37.0 [+20.8, +52.7] | +8.0 / +29.0 / -18.0 / +0.0 | -0.252 [-0.407, -0.109] (452 scenes) | 27 / 36 |
| control - base s0 | -0.0917 [-0.1268, -0.0596] | +46.0 [+30.8, +62.6] | +36.5 [+21.4, +53.1] | +13.5 / +23.0 / +16.5 / +0.0 | +0.252 [+0.116, +0.392] (452 scenes) | 15 / 38 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 8.

**Adoption line (registered): NOT adopted** - fewer zeros than the control True (96.0 vs 120.0); at-fault zeros not above the control's (hard gate) False (80.0 vs 79.5); GT distance not larger True (1.243 vs 1.746 m).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": false, "met": false}.
