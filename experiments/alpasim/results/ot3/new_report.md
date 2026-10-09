Scenes common to the 6 drivers: 400 from 17 logs, 4 shards. Bootstrap: 10000 draws, seed 0; `scenes` resamples scenes, `logs` resamples whole nuPlan logs (`date_vehicle`). Missing from the manifests: none.

- P2H10-F-s0@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0
- P2H10-F-s1@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0
- P2-F-s0@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0
- P2-F-s1@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0
- YR10m10-F-s0@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0
- YR10m10-F-s1@new: 400 scored scenes, 1 run dirs, disagreeing duplicate scenes 0

## Drivers

| driver | mean scene score [95% CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0@new | 0.8892 [0.8535, 0.9291] | 297 | 35 | 3 | 13 | 9 | 68 | 16 | 0.898 |
| P2H10-F-s1@new | 0.8960 [0.8660, 0.9303] | 297 | 31 | 2 | 13 | 6 | 72 | 15 | 0.891 |
| P2-F-s0@new | 0.8842 [0.8510, 0.9226] | 287 | 36 | 3 | 15 | 6 | 77 | 18 | 0.889 |
| P2-F-s1@new | 0.8830 [0.8518, 0.9185] | 284 | 35 | 3 | 15 | 5 | 81 | 18 | 0.881 |
| YR10m10-F-s0@new | 0.9080 [0.8791, 0.9425] | 303 | 28 | 2 | 12 | 4 | 69 | 14 | 0.899 |
| YR10m10-F-s1@new | 0.9078 [0.8792, 0.9414] | 300 | 27 | 2 | 9 | 5 | 73 | 11 | 0.895 |

## Recipes (per-scene mean over the training seeds; counts are seed means)

| recipe | seeds | mean [95% CI, logs] | zeros | at-fault collision | offroad | left corridor | slow | at-fault events | seed 0 - seed 1 | 95% CI scenes | 95% CI logs | scenes where the seeds differ in zero / non-zero |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| P2H10 | 0.8892 / 0.8960 | 0.8926 [0.8598, 0.9297] | 33 | 2.5 | 13 | 7.5 | 70 | 15.5 | -0.0068 | [-0.0214, +0.0065] | [-0.0150, +0.0008] | 8 |
| P2 | 0.8842 / 0.8830 | 0.8836 [0.8525, 0.9195] | 35.5 | 3 | 15 | 5.5 | 79 | 18 | +0.0013 | [-0.0132, +0.0160] | [-0.0152, +0.0178] | 9 |
| YR10m10 | 0.9080 / 0.9078 | 0.9079 [0.8800, 0.9414] | 27.5 | 2 | 10.5 | 4.5 | 71 | 12.5 | +0.0003 | [-0.0157, +0.0164] | [-0.0129, +0.0155] | 11 |

## Against P2H10 (pre-registered line for the candidates: difference >= +0.01 with the log-clustered CI lower bound > 0; at-fault events not higher; slow scenes not more than P2H10's + 10)

| recipe - P2H10 | difference | 95% CI scenes | 95% CI logs (decides) | score line | at-fault events (recipe / base) | events line | slow (recipe / base) | slow line | zero -> non-zero / non-zero -> zero (seed-mean scores) | candidate |
|:--|--:|:--|:--|:--|--:|:--|--:|:--|:--|:--|
| P2 - P2H10 | -0.0090 | [-0.0215, +0.0028] | [-0.0239, +0.0025] | not met | 18 / 15.5 | not met | 79 / 70 | met | 2 / 4 | **no** |
| YR10m10 - P2H10 | +0.0153 | [-0.0003, +0.0322] | [+0.0043, +0.0267] | met | 12.5 / 15.5 | met | 71 / 70 | met | 10 / 3 | **yes** |

Single checkpoints against the baseline recipe:

| driver - P2H10 | difference | 95% CI scenes | 95% CI logs |
|:--|--:|:--|:--|
| P2-F-s0@new - P2H10 | -0.0083 | [-0.0225, +0.0044] | [-0.0254, +0.0046] |
| P2-F-s1@new - P2H10 | -0.0096 | [-0.0250, +0.0051] | [-0.0265, +0.0059] |
| YR10m10-F-s0@new - P2H10 | +0.0154 | [-0.0037, +0.0355] | [+0.0015, +0.0312] |
| YR10m10-F-s1@new - P2H10 | +0.0152 | [-0.0006, +0.0320] | [+0.0032, +0.0255] |
