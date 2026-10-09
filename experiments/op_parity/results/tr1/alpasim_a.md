Scenes common to the 4 drivers: 700 from 27 logs, 7 shards. Bootstrap: 10000 draws, seed 0; `scenes` resamples scenes, `logs` resamples whole nuPlan logs (`date_vehicle`). Missing from the manifests: none.

- P2H10-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- P2H10-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- T1LG-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- T1LG-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0

## Drivers

| driver | mean scene score [95% CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0 | 0.9468 [0.9252, 0.9652] | 562 | 26 | 8 | 10 | 8 | 112 | 18 | 0.942 |
| P2H10-F-s1 | 0.9495 [0.9300, 0.9669] | 572 | 23 | 6 | 10 | 7 | 105 | 16 | 0.936 |
| T1LG-F-s0 | 0.8885 [0.8642, 0.9120] | 481 | 30 | 6 | 14 | 10 | 189 | 20 | 0.859 |
| T1LG-F-s1 | 0.8945 [0.8718, 0.9157] | 484 | 25 | 5 | 9 | 11 | 191 | 14 | 0.861 |

## Recipes (per-scene mean over the training seeds; counts are seed means)

| recipe | seeds | mean [95% CI, logs] | zeros | at-fault collision | offroad | left corridor | slow | at-fault events | seed 0 - seed 1 | 95% CI scenes | 95% CI logs | scenes where the seeds differ in zero / non-zero |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| P2H10 | 0.9468 / 0.9495 | 0.9481 [0.9277, 0.9657] | 24.5 | 7 | 10 | 7.5 | 108.5 | 17 | -0.0028 | [-0.0098, +0.0033] | [-0.0098, +0.0037] | 5 |
| T1LG | 0.8885 / 0.8945 | 0.8915 [0.8683, 0.9136] | 27.5 | 5.5 | 11.5 | 10.5 | 190 | 17 | -0.0059 | [-0.0144, +0.0016] | [-0.0132, +0.0020] | 9 |

## Against P2H10 (pre-registered line for the candidates: difference >= +0.01 with the log-clustered CI lower bound > 0; at-fault events not higher; slow scenes not more than P2H10's + 10)

| recipe - P2H10 | difference | 95% CI scenes | 95% CI logs (decides) | score line | at-fault events (recipe / base) | events line | slow (recipe / base) | slow line | zero -> non-zero / non-zero -> zero (seed-mean scores) | candidate |
|:--|--:|:--|:--|:--|--:|:--|--:|:--|:--|:--|
| T1LG - P2H10 | -0.0566 | [-0.0718, -0.0412] | [-0.0804, -0.0315] | not met | 17 / 17 | met | 190 / 108.5 | not met | 8 / 9 | **no** |

Single checkpoints against the baseline recipe:

| driver - P2H10 | difference | 95% CI scenes | 95% CI logs |
|:--|--:|:--|:--|
| T1LG-F-s0 - P2H10 | -0.0596 | [-0.0761, -0.0430] | [-0.0852, -0.0320] |
| T1LG-F-s1 - P2H10 | -0.0537 | [-0.0689, -0.0382] | [-0.0757, -0.0304] |
