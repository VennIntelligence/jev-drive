# ONPOL-LONG round 1, family R (base RSG-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm RS2G-L18 (round 2b: round-2 rows, gentle, from YS-L18-s1 (Amendment 8)) | 0.8793 | 28.5 | 4.0 / 17.0 / 8.5 / 0.0 | 21.0 | 1.158 | 0.0845 | RS2G-L18-s0 0.8801 / 28 / 21 / 1.158 ; RS2G-L18-s1 0.8785 / 29 / 21 / 1.159 |
| control RNONE-L18 (same continuation, no rows) | 0.7974 | 69.5 | 16.5 / 39.5 / 15.5 / 0.0 | 56.0 | 1.489 | 0.1222 | RNONE-L18-s0 0.7946 / 72 / 55 / 1.526 ; RNONE-L18-s1 0.8002 / 67 / 57 / 1.452 |
| reference RSG-L18-s0 | 0.8809 | 26.0 | 4.0 / 17.0 / 6.0 / 0.0 | 21.0 | 1.135 | 0.0824 | RSG-L18-s0 0.8809 / 26 / 21 / 1.135 |
| reference RSG-L18-s0, RSG-L18-s1 | 0.8819 | 26.0 | 5.0 / 15.0 / 7.0 / 0.0 | 20.0 | 1.140 | 0.0809 | RSG-L18-s0 0.8809 / 26 / 21 / 1.135 ; RSG-L18-s1 0.8830 / 26 / 19 / 1.144 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0819 [+0.0615, +0.1035] | -41.0 [-52.0, -30.3] | -35.0 [-44.7, -25.2] | -12.5 / -22.5 / -7.0 / +0.0 | -0.331 [-0.416, -0.255] (452 scenes) | 27 / 4 |
| arm - reference RSG-L18-s0 | -0.0016 [-0.0123, +0.0085] | +2.5 [-1.1, +6.9] | +0.0 [-2.6, +3.0] | +0.0 / +0.0 / +2.5 / +0.0 | +0.023 [-0.004, +0.050] (452 scenes) | 1 / 3 |
| control - reference RSG-L18-s0 | -0.0835 [-0.1055, -0.0625] | +43.5 [+33.2, +54.3] | +35.0 [+25.8, +44.3] | +12.5 / +22.5 / +9.5 / +0.0 | +0.354 [+0.277, +0.442] (452 scenes) | 4 / 32 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons RNONE-L18 s0 / s1 and RSG-L18 s0 / s1 = 5.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (28.5 vs 69.5); at-fault zeros not above the control's (hard gate) True (21.0 vs 56.0); GT distance not larger True (1.158 vs 1.489 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against RSG-L18-s0: not passed - {"fewer_zeros": false, "atfault_not_above": true, "gt_not_larger": false, "passed": false, "score_not_lower": false} (zeros 28.5 vs 26.0, at-fault 21.0 vs 21.0, GT 1.158 vs 1.135).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": false}.
