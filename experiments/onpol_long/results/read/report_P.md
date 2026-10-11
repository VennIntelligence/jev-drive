# ONPOL-LONG round 1, family P (base P2H10S-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm PSKC-L18 (S + K + C on-policy rows) | 0.5901 | 118.0 | 18.5 / 79.0 / 22.0 / 0.0 | 97.5 | 1.292 | 0.0934 | PSKC-L18-s0 0.5956 / 114 / 97 / 1.264 ; PSKC-L18-s1 0.5845 / 122 / 98 / 1.319 |
| control PNONE-L18 (same continuation, no rows) | 0.7706 | 81.0 | 11.5 / 24.0 / 47.0 / 0.0 | 35.5 | 1.674 | 0.1771 | PNONE-L18-s0 0.7741 / 81 / 36 / 1.703 ; PNONE-L18-s1 0.7670 / 81 / 35 / 1.646 |
| base P2H10S-F-s0 | 0.7986 | 68.0 | 8.0 / 24.0 / 39.0 / 0.0 | 32.0 | 1.588 | 0.1500 | P2H10S-F-s0 0.7986 / 68 / 32 / 1.588 |
| base P2H10S-F seeds 0, 1 | 0.8005 | 67.5 | 8.0 / 23.5 / 38.0 / 0.0 | 31.5 | 1.576 | 0.1496 | P2H10S-F-s0 0.7986 / 68 / 32 / 1.588 ; P2H10S-F-s1 0.8024 / 67 / 31 / 1.563 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | -0.1805 [-0.2174, -0.1366] | +37.0 [+16.0, +54.5] | +62.0 [+38.7, +86.7] | +7.0 / +55.0 / -25.0 / +0.0 | -0.383 [-0.590, -0.190] (452 scenes) | 24 / 51 |
| arm - base s0 | -0.2085 [-0.2510, -0.1633] | +50.0 [+27.8, +70.4] | +65.5 [+42.9, +90.9] | +10.5 / +55.0 / -17.0 / +0.0 | -0.296 [-0.491, -0.120] (452 scenes) | 23 / 55 |
| control - base s0 | -0.0280 [-0.0497, -0.0106] | +13.0 [+4.8, +22.9] | +3.5 [-3.2, +11.6] | +3.5 / +0.0 / +8.0 / +0.0 | +0.087 [+0.012, +0.163] (452 scenes) | 11 / 18 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons PNONE-L18 s0 / s1 and P2H10S-F s0 / s1 = 1.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (118.0 vs 81.0); at-fault zeros not above the control's (hard gate) False (97.5 vs 35.5); GT distance not larger True (1.292 vs 1.674 m).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": false, "met": false}.
