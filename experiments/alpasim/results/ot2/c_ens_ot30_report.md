Scenes common to the 4 drivers: 700 from 27 logs, 7 shards. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10000 draws, seed 0. Missing from the manifests: none.

- SH30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- ENS-OT30: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- OT30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0
- OT30-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes 0

## Scores

| driver | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | slow (0 < score < 1) | at-fault events | mean progress |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 0.9140 [0.8845, 0.9366] | 555 | 50 | 19 | 13 | 18 | 95 | 32 | 0.948 |
| ENS-OT30 | 0.9312 [0.9017, 0.9532] | 573 | 39 | 12 | 11 | 16 | 88 | 23 | 0.955 |
| OT30-F-s0 | 0.9258 [0.9029, 0.9447] | 565 | 42 | 11 | 12 | 19 | 93 | 23 | 0.950 |
| OT30-F-s1 | 0.9335 [0.9050, 0.9558] | 580 | 38 | 11 | 10 | 17 | 82 | 21 | 0.959 |

## Piece C line: ENS-OT30 (pre-registered: ensemble minus its better member >= +0.008 with CI lower bound > 0; at-fault events not higher)

- members OT30-F-s0 0.9258, OT30-F-s1 0.9335; better member OT30-F-s1; ensemble 0.9312; per-scene best-of-members 0.9468 (+0.0133 [+0.0056, +0.0221] over the better member)
- ensemble minus better member: -0.0022 [-0.0097, +0.0045] -> score line **not met**; at-fault events 23 vs 21 -> **not met**; verdict: **not adopted**
- ensemble minus the mean of its members: +0.0016 [-0.0055, +0.0071]
- scenes where exactly some members are zero (20): ensemble zero in 9; all members zero (30): ensemble zero in 30; no member zero (650): ensemble zero in 0
- zero classes of the ensemble: collision 12, offroad 11, corridor 16; slow 88 (members 93, 82)
