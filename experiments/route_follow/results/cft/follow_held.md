# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RCFT-L18-s0@18 (served) | 452 / 24 | 62.5 % (24) [45, 78] | 81.2 % (16) [62, 100] | 25.0 % (8) [0, 57] | 100.0 % (1) [100, 100] | 60.9 % (23) [44, 77] | 15 / 2 / 3 / 4 | 0.51 / 1.97 (347) | -0.24 (+0.51) | 0.999 |
| RCFT-L18-s1@18 (served) | 452 / 24 | 62.5 % (24) [40, 82] | 75.0 % (16) [50, 93] | 37.5 % (8) [0, 78] | 100.0 % (1) [100, 100] | 60.9 % (23) [39, 81] | 15 / 2 / 3 / 4 | 0.51 / 1.84 (344) | -0.26 (+0.50) | 0.999 |
| RNONEG-L18-s0@18 (served) | 452 / 24 | 41.7 % (24) [20, 65] | 56.2 % (16) [33, 75] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 39.1 % (23) [20, 62] | 10 / 3 / 4 / 7 | 0.54 / 2.05 (348) | -0.26 (+0.46) | 0.999 |
| RNONEG-L18-s1@18 (served) | 452 / 24 | 50.0 % (24) [30, 71] | 56.2 % (16) [33, 75] | 37.5 % (8) [10, 71] | 100.0 % (1) [100, 100] | 47.8 % (23) [30, 68] | 12 / 5 / 2 / 5 | 0.55 / 1.97 (347) | -0.25 (+0.57) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RB0-F-s1@18 (served) | 452 / 24 | 62.5 % (24) [50, 78] | 75.0 % (16) [58, 100] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [51, 80] | 15 / 6 / 1 / 2 | 0.52 / 2.01 (333) | -0.21 (+0.77) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RCFT-L18-s0@18:RSG-L18-s0@18 | all | -4.2 [-30.4, +20.7] | 24 / 24 |
| RCFT-L18-s0@18:RSG-L18-s0@18 | left | +12.5 [-11.8, +41.7] | 16 / 16 |
| RCFT-L18-s0@18:RSG-L18-s0@18 | right | -37.5 [-88.9, +20.0] | 8 / 8 |
| RCFT-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s0@18:RSG-L18-s0@18 | uncovered | -4.3 [-31.6, +20.8] | 23 / 23 |
| RCFT-L18-s1@18:RSG-L18-s0@18 | all | -4.2 [-31.8, +22.7] | 24 / 24 |
| RCFT-L18-s1@18:RSG-L18-s0@18 | left | +6.2 [+0.0, +23.1] | 16 / 16 |
| RCFT-L18-s1@18:RSG-L18-s0@18 | right | -25.0 [-88.9, +50.0] | 8 / 8 |
| RCFT-L18-s1@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s1@18:RSG-L18-s0@18 | uncovered | -4.3 [-33.3, +23.1] | 23 / 23 |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | all | +20.8 [-4.8, +42.3] | 24 / 24 |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | left | +25.0 [+0.0, +54.5] | 16 / 16 |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | right | +12.5 [-33.3, +50.0] | 8 / 8 |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +21.7 [-5.0, +42.9] | 23 / 23 |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | all | +12.5 [-11.1, +35.0] | 24 / 24 |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | left | +18.8 [+0.0, +38.9] | 16 / 16 |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | right | +0.0 [-55.6, +55.6] | 8 / 8 |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | uncovered | +13.0 [-11.8, +36.0] | 23 / 23 |
| RCFT-L18-s0@18:RB0-F-s0@18 | all | +8.3 [-14.3, +26.1] | 24 / 24 |
| RCFT-L18-s0@18:RB0-F-s0@18 | left | +25.0 [+0.0, +54.5] | 16 / 16 |
| RCFT-L18-s0@18:RB0-F-s0@18 | right | -25.0 [-57.1, +0.0] | 8 / 8 |
| RCFT-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s0@18:RB0-F-s0@18 | uncovered | +8.7 [-15.0, +26.3] | 23 / 23 |
| RCFT-L18-s1@18:RB0-F-s0@18 | all | +8.3 [-12.5, +27.3] | 24 / 24 |
| RCFT-L18-s1@18:RB0-F-s0@18 | left | +18.8 [+0.0, +38.9] | 16 / 16 |
| RCFT-L18-s1@18:RB0-F-s0@18 | right | -12.5 [-55.6, +25.0] | 8 / 8 |
| RCFT-L18-s1@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RCFT-L18-s1@18:RB0-F-s0@18 | uncovered | +8.7 [-12.5, +27.8] | 23 / 23 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | all | +25.0 [+8.0, +44.4] | 24 / 24 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | left | +12.5 [+0.0, +26.7] | 16 / 16 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | right | +50.0 [+11.1, +88.9] | 8 / 8 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +26.1 [+8.7, +45.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RCFT-L18-s0@18:RSG-L18-s0@18 | covered | 85 (20 / 11) | 0.71 / 0.97 | -0.257 [-0.507, +0.218] | +0.000 [-0.176, +0.219] |
| RCFT-L18-s0@18:RSG-L18-s0@18 | uncovered | 242 (23 / 14) | 0.46 / 0.47 | -0.010 [-0.071, +0.037] | -0.000 [-0.003, +0.000] |
| RCFT-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 15 of 342 | | | |
| RCFT-L18-s1@18:RSG-L18-s0@18 | covered | 89 (21 / 12) | 0.80 / 0.84 | -0.037 [-0.375, +0.444] | +0.002 [-0.140, +0.326] |
| RCFT-L18-s1@18:RSG-L18-s0@18 | uncovered | 241 (23 / 14) | 0.43 / 0.47 | -0.042 [-0.074, +0.027] | +0.000 [-0.010, +0.000] |
| RCFT-L18-s1@18:RSG-L18-s0@18 | discordant (left out) | 11 of 341 | | | |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | covered | 84 (18 / 11) | 0.72 / 1.10 | -0.382 [-0.484, +0.153] | -0.032 [-0.455, +0.242] |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 243 (23 / 14) | 0.47 / 0.47 | -0.006 [-0.074, +0.037] | +0.000 [-0.001, +0.000] |
| RCFT-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 12 of 339 | | | |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | covered | 89 (21 / 12) | 1.01 / 1.07 | -0.059 [-0.392, +0.381] | -0.001 [-0.148, +0.297] |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | uncovered | 241 (23 / 14) | 0.43 / 0.49 | -0.056 [-0.099, -0.008] | -0.000 [-0.010, +0.000] |
| RCFT-L18-s1@18:RNONEG-L18-s1@18 | discordant (left out) | 8 of 338 | | | |
| RCFT-L18-s0@18:RB0-F-s0@18 | covered | 83 (19 / 11) | 0.71 / 1.05 | -0.338 [-0.537, +0.279] | +0.000 [-0.367, +0.148] |
| RCFT-L18-s0@18:RB0-F-s0@18 | uncovered | 244 (23 / 14) | 0.46 / 0.49 | -0.028 [-0.089, +0.011] | +0.000 [-0.002, +0.000] |
| RCFT-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 11 of 338 | | | |
| RCFT-L18-s1@18:RB0-F-s0@18 | covered | 88 (20 / 12) | 1.01 / 1.05 | -0.041 [-0.406, +0.388] | +0.018 [-0.165, +0.167] |
| RCFT-L18-s1@18:RB0-F-s0@18 | uncovered | 242 (23 / 14) | 0.43 / 0.49 | -0.053 [-0.092, +0.006] | +0.000 [-0.001, +0.000] |
| RCFT-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 9 of 339 | | | |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | covered | 90 (19 / 12) | 0.92 / 0.94 | -0.021 [-0.377, +0.345] | -0.106 [-0.436, +0.058] |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 246 (23 / 14) | 0.48 / 0.47 | +0.010 [-0.080, +0.088] | +0.000 [-0.001, +0.001] |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 9 of 345 | | | |
