# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSEX-L18 (YS-L18 with the ego status removed, also when served (Amendment 11)) | 0.8424 | 60.0 | 9.5 / 17.5 / 35.0 / 0.0 | 27.0 | 1.562 | 0.1193 | YSEX-L18-s0 0.8463 / 58 / 24 / 1.552 ; YSEX-L18-s1 0.8384 / 62 / 30 / 1.572 |
| control YS-L18 | 0.8376 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0960 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |
| reference YR10m10-F-s0 | 0.7866 | 73.0 | 8.0 / 33.0 / 36.0 / 0.0 | 41.0 | 1.492 | 0.1415 | YR10m10-F-s0 0.7866 / 73 / 41 / 1.492 |
| reference YR10m10-F-s0, YR10m10-F-s1 | 0.7909 | 72.0 | 6.0 / 35.0 / 34.5 / 0.0 | 41.0 | 1.448 | 0.1339 | YR10m10-F-s0 0.7866 / 73 / 41 / 1.492 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0048 [-0.0179, +0.0266] | +16.0 [+7.2, +26.1] | +3.5 [-1.7, +8.2] | +0.5 / +3.0 / +13.0 / +0.0 | +0.205 [+0.114, +0.307] (452 scenes) | 7 / 15 |
| arm - reference YR10m10-F-s0 | +0.0558 [+0.0209, +0.0883] | -13.0 [-27.9, +3.0] | -14.0 [-23.0, -4.1] | +1.5 / -15.5 / -1.0 / +0.0 | +0.070 [-0.097, +0.246] (452 scenes) | 35 / 21 |
| control - reference YR10m10-F-s0 | +0.0510 [+0.0223, +0.0804] | -29.0 [-42.6, -15.7] | -17.5 [-27.1, -6.7] | +1.0 / -18.5 / -14.0 / +0.0 | -0.135 [-0.262, -0.010] (452 scenes) | 37 / 10 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 2.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (60.0 vs 44.0); at-fault zeros not above the control's (hard gate) False (27.0 vs 23.5); GT distance not larger False (1.562 vs 1.357 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: not passed - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": false, "passed": false, "score_not_lower": true} (zeros 60.0 vs 73.0, at-fault 27.0 vs 41.0, GT 1.562 vs 1.492).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
