# ONPOL-LONG round 1, family R (base RSG-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm RBSG2-L18 (BSG2) | 0.8872 | 29.0 | 4.5 / 21.0 / 4.0 / 0.0 | 25.5 | 1.200 | 0.0884 | RBSG2-L18-s0 0.8857 / 29 / 26 / 1.215 ; RBSG2-L18-s1 0.8887 / 29 / 25 / 1.185 |
| control RNONE-L18 (same continuation, no rows) | 0.7974 | 69.5 | 16.5 / 39.5 / 15.5 / 0.0 | 56.0 | 1.489 | 0.1222 | RNONE-L18-s0 0.7946 / 72 / 55 / 1.526 ; RNONE-L18-s1 0.8002 / 67 / 57 / 1.452 |
| reference RSG-L18-s0 | 0.8809 | 26.0 | 4.0 / 17.0 / 6.0 / 0.0 | 21.0 | 1.135 | 0.0824 | RSG-L18-s0 0.8809 / 26 / 21 / 1.135 |
| reference RSG-L18-s0, RSG-L18-s1 | 0.8819 | 26.0 | 5.0 / 15.0 / 7.0 / 0.0 | 20.0 | 1.140 | 0.0809 | RSG-L18-s0 0.8809 / 26 / 21 / 1.135 ; RSG-L18-s1 0.8830 / 26 / 19 / 1.144 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0898 [+0.0663, +0.1167] | -40.5 [-54.0, -28.6] | -30.5 [-42.9, -19.1] | -12.0 / -18.5 / -11.5 / +0.0 | -0.289 [-0.425, -0.166] (452 scenes) | 32 / 5 |
| arm - reference RSG-L18-s0 | +0.0063 [-0.0116, +0.0278] | +3.0 [-4.8, +9.9] | +4.5 [-3.3, +11.4] | +0.5 / +4.0 / -2.0 / +0.0 | +0.065 [-0.023, +0.160] (452 scenes) | 7 / 9 |
| control - reference RSG-L18-s0 | -0.0835 [-0.1055, -0.0625] | +43.5 [+33.2, +54.3] | +35.0 [+25.8, +44.3] | +12.5 / +22.5 / +9.5 / +0.0 | +0.354 [+0.277, +0.442] (452 scenes) | 4 / 32 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons RNONE-L18 s0 / s1 and RSG-L18 s0 / s1 = 5.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (29.0 vs 69.5); at-fault zeros not above the control's (hard gate) True (25.5 vs 56.0); GT distance not larger True (1.200 vs 1.489 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against RSG-L18-s0: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": false, "passed": false, "score_not_lower": true} (zeros 29.0 vs 26.0, at-fault 25.5 vs 21.0, GT 1.200 vs 1.135).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": false}.
