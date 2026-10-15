# v8 vs v9 references at the logged handoff decisions (k 17 .. 26), sv9_check.py replay

Per rollout the worst decision of the window: drop = v0 - reference speed at 1.5 s (the middle of the MPC's tracked 1 .. 2 s), deficit = 2 x v0 - reference arc at 2 s. Quantiles 10 / 50 / 90 %.

| label | group | n | served plan drop m/s | v8 drop | v9 drop | served deficit m | v8 deficit | v9 deficit | v9 guard calls (lead) |
|:--|:--|--:|:--|:--|:--|:--|:--|:--|:--|
| app1x | handoff spin | 29 | [14.3, 17.2, 26.8] | [7.4, 7.4, 7.4] | [3.7, 3.7, 3.7] | [21.6, 28.6, 40.2] | [10.3, 10.3, 10.3] | [5.1, 5.1, 5.1] | 290 (2) |
| app1x | no handoff spin | 252 | [-0.4, 1.0, 5.6] | [-0.4, 1.1, 5.8] | [-0.4, 1.1, 4.7] | [-0.6, 1.6, 8.5] | [-0.6, 1.5, 7.5] | [-0.6, 1.5, 6.1] | 1750 (976) |
| rsg | handoff spin | 2 | [24.0, 27.3, 30.7] | [7.4, 7.4, 7.4] | [3.7, 3.7, 3.7] | [35.9, 41.3, 46.7] | [10.3, 10.3, 10.3] | [5.1, 5.1, 5.1] | 20 (0) |
| rsg | no handoff spin | 279 | [-0.7, 0.4, 3.2] | [-0.8, 0.4, 3.7] | [-0.8, 0.4, 3.4] | [-1.1, 0.6, 4.8] | [-1.0, 0.5, 4.9] | [-1.0, 0.5, 4.3] | 2019 (726) |
