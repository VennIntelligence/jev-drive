# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSED-L18 (YS-L18 + ego-status dropout p 0.5 (Amendment 11)) | 0.8410 | 54.5 | 10.5 / 14.0 / 31.0 / 0.0 | 24.5 | 1.576 | 0.1190 | YSED-L18-s0 0.8316 / 59 / 27 / 1.552 ; YSED-L18-s1 0.8505 / 50 / 22 / 1.600 |
| control YS-L18 | 0.8376 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0960 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |
| reference YR10m10-F-s0 | 0.7866 | 73.0 | 8.0 / 33.0 / 36.0 / 0.0 | 41.0 | 1.492 | 0.1415 | YR10m10-F-s0 0.7866 / 73 / 41 / 1.492 |
| reference YR10m10-F-s0, YR10m10-F-s1 | 0.7909 | 72.0 | 6.0 / 35.0 / 34.5 / 0.0 | 41.0 | 1.448 | 0.1339 | YR10m10-F-s0 0.7866 / 73 / 41 / 1.492 ; YR10m10-F-s1 0.7953 / 71 / 41 / 1.404 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0034 [-0.0133, +0.0195] | +10.5 [+3.8, +18.4] | +1.0 [-3.1, +5.3] | +1.5 / -0.5 / +9.0 / +0.0 | +0.220 [+0.135, +0.309] (452 scenes) | 2 / 8 |
| arm - reference YR10m10-F-s0 | +0.0544 [+0.0215, +0.0850] | -18.5 [-32.9, -3.0] | -16.5 [-25.6, -6.3] | +2.5 / -19.0 / -5.0 / +0.0 | +0.084 [-0.083, +0.249] (452 scenes) | 34 / 15 |
| control - reference YR10m10-F-s0 | +0.0510 [+0.0223, +0.0804] | -29.0 [-42.6, -15.7] | -17.5 [-27.1, -6.7] | +1.0 / -18.5 / -14.0 / +0.0 | -0.135 [-0.262, -0.010] (452 scenes) | 37 / 10 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YR10m10-F s0 / s1 = 2.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (54.5 vs 44.0); at-fault zeros not above the control's (hard gate) False (24.5 vs 23.5); GT distance not larger False (1.576 vs 1.357 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: not passed - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": false, "passed": false, "score_not_lower": true} (zeros 54.5 vs 73.0, at-fault 24.5 vs 41.0, GT 1.576 vs 1.492).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
