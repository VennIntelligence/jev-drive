Scenes common to the 10 drivers: 700 from 27 logs, 7 shards. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10000 draws, seed 0. Missing from the manifests: none.

- SH30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- SH30-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- AP2-AB-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- AP2-AB-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- OT30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- OT30-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- APO-a05m10-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- APO-a05m10-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- APO-a05m25-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- APO-a05m25-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0

## Scores

| driver | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 0.9140 [0.8845, 0.9366] | 555 | 50 | 19 | 13 | 18 | 95 | 32 | 0.948 |
| SH30-F-s1 | 0.9219 [0.8928, 0.9446] | 560 | 43 | 17 | 13 | 13 | 97 | 30 | 0.948 |
| AP2-AB-s0 | 0.9223 [0.9003, 0.9404] | 598 | 49 | 21 | 12 | 16 | 53 | 33 | 0.987 |
| AP2-AB-s1 | 0.9240 [0.9009, 0.9436] | 598 | 48 | 20 | 11 | 17 | 54 | 31 | 0.987 |
| OT30-F-s0 | 0.9258 [0.9029, 0.9447] | 565 | 42 | 11 | 12 | 19 | 93 | 23 | 0.950 |
| OT30-F-s1 | 0.9335 [0.9050, 0.9558] | 580 | 38 | 11 | 10 | 17 | 82 | 21 | 0.959 |
| APO-a05m10-s0 | 0.9423 [0.9203, 0.9595] | 605 | 35 | 9 | 7 | 19 | 60 | 16 | 0.992 |
| APO-a05m10-s1 | 0.9394 [0.9154, 0.9599] | 610 | 37 | 8 | 9 | 20 | 53 | 17 | 0.990 |
| APO-a05m25-s0 | 0.9179 [0.8939, 0.9378] | 596 | 52 | 13 | 14 | 25 | 52 | 27 | 0.988 |
| APO-a05m25-s1 | 0.9065 [0.8773, 0.9299] | 591 | 61 | 15 | 14 | 32 | 48 | 29 | 0.989 |

## Seed means (per-scene mean over the training seeds of a recipe; counts are seed means)

| recipe | seeds | mean scene score [95% CI] | zeros | at-fault collision | offroad + corridor | slow | at-fault events | seed difference [95% CI] | scenes where the seeds differ in zero / non-zero |
|:--|:--|:--|--:|--:|--:|--:|--:|:--|--:|
| SH30 | 0.9140 / 0.9219 | 0.9180 [0.8896, 0.9394] | 46.5 | 18 | 28.5 | 96 | 31 | -0.0079 [-0.0221, +0.0089] | 27 |
| AP2 | 0.9223 / 0.9240 | 0.9232 [0.9008, 0.9416] | 48.5 | 20.5 | 28 | 53.5 | 32 | -0.0017 [-0.0101, +0.0071] | 15 |
| OT30 | 0.9258 / 0.9335 | 0.9296 [0.9046, 0.9494] | 40 | 11 | 29 | 87.5 | 22 | -0.0077 [-0.0196, +0.0058] | 20 |
| APO-a05m10 | 0.9423 / 0.9394 | 0.9409 [0.9187, 0.9590] | 36 | 8.5 | 27.5 | 56.5 | 16.5 | +0.0029 [-0.0082, +0.0133] | 14 |
| APO-a05m25 | 0.9179 / 0.9065 | 0.9122 [0.8867, 0.9331] | 56.5 | 14 | 42.5 | 50 | 28 | +0.0114 [-0.0022, +0.0274] | 33 |

## Piece B lines (pre-registered: two-seed mean minus the baseline's two-seed mean >= +0.015 with CI lower bound > 0; at-fault events not higher; slow scenes not more than the baseline's + 10)

| recipe - baseline | difference [95% CI] | score line | at-fault events (recipe / baseline) | events line | slow (recipe / baseline) | slow line | candidate |
|:--|:--|:--|--:|:--|--:|:--|:--|
| APO-a05m10 - AP2 | +0.0177 [+0.0077, +0.0277] | met | 16.5 / 32 | met | 56.5 / 53.5 | met | **yes** |
| APO-a05m25 - AP2 | -0.0109 [-0.0244, +0.0014] | not met | 28 / 32 | met | 50 / 53.5 | met | **no** |

## Paired differences

| A - B | mean scene score difference [95% CI] |
|:--|:--|
| APO-a05m25 - APO-a05m10 | -0.0286 [-0.0412, -0.0177] |
| AP2 - SH30 | +0.0052 [-0.0065, +0.0196] |
| OT30 - SH30 | +0.0116 [-0.0012, +0.0264] |
| APO-a05m10 - OT30 | +0.0113 [-0.0021, +0.0239] |
| APO-a05m10 - SH30 | +0.0229 [+0.0079, +0.0394] |
| APO-a05m25 - SH30 | -0.0057 [-0.0246, +0.0137] |
| AP2-AB-s0 - AP2-AB-s1 | -0.0017 [-0.0101, +0.0071] |
| SH30-F-s0 - SH30-F-s1 | -0.0079 [-0.0221, +0.0089] |
