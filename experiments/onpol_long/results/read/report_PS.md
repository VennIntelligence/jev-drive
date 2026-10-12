# ONPOL-LONG round 1, family P (base P2H10S-F-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm PS-L18 (S hinges only on the same rows (Amendment 3)) | 0.7911 | 66.0 | 15.0 / 21.5 / 31.0 / 0.0 | 36.0 | 1.430 | 0.1291 | PS-L18-s0 0.7925 / 68 / 38 / 1.431 ; PS-L18-s1 0.7897 / 64 / 34 / 1.428 |
| control PNONE-L18 (same continuation, no rows) | 0.7706 | 81.0 | 11.5 / 24.0 / 47.0 / 0.0 | 35.5 | 1.674 | 0.1771 | PNONE-L18-s0 0.7741 / 81 / 36 / 1.703 ; PNONE-L18-s1 0.7670 / 81 / 35 / 1.646 |
| base P2H10S-F-s0 | 0.7986 | 68.0 | 8.0 / 24.0 / 39.0 / 0.0 | 32.0 | 1.588 | 0.1500 | P2H10S-F-s0 0.7986 / 68 / 32 / 1.588 |
| base P2H10S-F seeds 0, 1 | 0.8005 | 67.5 | 8.0 / 23.5 / 38.0 / 0.0 | 31.5 | 1.576 | 0.1496 | P2H10S-F-s0 0.7986 / 68 / 32 / 1.588 ; P2H10S-F-s1 0.8024 / 67 / 31 / 1.563 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0205 [-0.0101, +0.0528] | -15.0 [-30.2, -0.9] | +0.5 [-11.5, +10.9] | +3.5 / -2.5 / -16.0 / +0.0 | -0.245 [-0.370, -0.111] (452 scenes) | 28 / 19 |
| arm - base s0 | -0.0075 [-0.0437, +0.0254] | -2.0 [-17.6, +14.5] | +4.0 [-4.8, +12.3] | +7.0 / -2.5 / -8.0 / +0.0 | -0.158 [-0.274, -0.020] (452 scenes) | 25 / 25 |
| control - base s0 | -0.0280 [-0.0497, -0.0106] | +13.0 [+4.8, +22.9] | +3.5 [-3.2, +11.6] | +3.5 / +0.0 / +8.0 / +0.0 | +0.087 [+0.012, +0.163] (452 scenes) | 11 / 18 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons PNONE-L18 s0 / s1 and P2H10S-F s0 / s1 = 1.

**Adoption line (registered): NOT adopted** - fewer zeros than the control True (66.0 vs 81.0); at-fault zeros not above the control's (hard gate) False (36.0 vs 35.5); GT distance not larger True (1.430 vs 1.674 m).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": false}.
