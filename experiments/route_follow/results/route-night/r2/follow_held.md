# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RBSG2-L18-s0@18 (served) | 452 / 24 | 70.8 % (24) [57, 89] | 75.0 % (16) [55, 100] | 62.5 % (8) [38, 86] | 0.0 % (1) [0, 0] | 73.9 % (23) [60, 89] | 17 / 1 / 1 / 5 | 0.51 / 1.66 (345) | -0.27 (+0.37) | 0.999 |
| RBSG2-L18-s1@18 (served) | 452 / 24 | 75.0 % (24) [52, 95] | 81.2 % (16) [62, 100] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 73.9 % (23) [50, 95] | 18 / 1 / 0 / 5 | 0.48 / 1.52 (349) | -0.28 (+0.32) | 0.999 |
| RBSG-L18-s0@18 (served) | 452 / 24 | 54.2 % (24) [30, 79] | 75.0 % (16) [50, 93] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 52.2 % (23) [29, 78] | 13 / 3 / 3 / 5 | 0.60 / 2.23 (337) | -0.36 (+0.21) | 0.999 |
| RBSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [39, 81] | 68.8 % (16) [46, 92] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 60.9 % (23) [40, 83] | 14 / 4 / 1 / 5 | 0.65 / 2.31 (335) | -0.35 (+0.27) | 0.999 |
| RNONEG-L18-s0@18 (served) | 452 / 24 | 41.7 % (24) [20, 65] | 56.2 % (16) [33, 75] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 39.1 % (23) [20, 62] | 10 / 3 / 4 / 7 | 0.54 / 2.05 (348) | -0.26 (+0.46) | 0.999 |
| RNONE-L18-s0@18 (served) | 452 / 24 | 62.5 % (24) [47, 76] | 75.0 % (16) [55, 93] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [48, 82] | 15 / 5 / 2 / 2 | 0.48 / 1.73 (334) | -0.24 (+0.63) | 0.999 |
| RNONE-L18-s1@18 (served) | 452 / 24 | 62.5 % (24) [47, 76] | 75.0 % (16) [55, 93] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [48, 82] | 15 / 5 / 1 / 3 | 0.50 / 1.84 (339) | -0.23 (+0.60) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RB0-F-s1@18 (served) | 452 / 24 | 62.5 % (24) [50, 78] | 75.0 % (16) [58, 100] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [51, 80] | 15 / 6 / 1 / 2 | 0.52 / 2.01 (333) | -0.21 (+0.77) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RBSG2-L18-s0@18:RB0-F-s0@18 | all | +16.7 [+0.0, +36.8] | 24 / 24 |
| RBSG2-L18-s0@18:RB0-F-s0@18 | left | +18.8 [-5.6, +50.0] | 16 / 16 |
| RBSG2-L18-s0@18:RB0-F-s0@18 | right | +12.5 [+0.0, +42.9] | 8 / 8 |
| RBSG2-L18-s0@18:RB0-F-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RBSG2-L18-s0@18:RB0-F-s0@18 | uncovered | +21.7 [+7.7, +38.1] | 23 / 23 |
| RBSG2-L18-s1@18:RB0-F-s0@18 | all | +20.8 [+0.0, +43.5] | 24 / 24 |
| RBSG2-L18-s1@18:RB0-F-s0@18 | left | +25.0 [+6.7, +50.0] | 16 / 16 |
| RBSG2-L18-s1@18:RB0-F-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RBSG2-L18-s1@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RBSG2-L18-s1@18:RB0-F-s0@18 | uncovered | +21.7 [+0.0, +44.0] | 23 / 23 |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | all | +29.2 [+6.9, +52.2] | 24 / 24 |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | left | +18.8 [-5.6, +50.0] | 16 / 16 |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | right | +50.0 [+25.0, +75.0] | 8 / 8 |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +34.8 [+17.4, +52.4] | 23 / 23 |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | all | +16.7 [-3.6, +40.0] | 24 / 24 |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | left | +0.0 [-13.0, +23.1] | 16 / 16 |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | right | +50.0 [+25.0, +75.0] | 8 / 8 |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | uncovered | +21.7 [+4.5, +40.9] | 23 / 23 |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | all | +16.7 [-10.5, +45.8] | 24 / 24 |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | left | +12.5 [-13.3, +35.3] | 16 / 16 |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | right | +25.0 [-25.0, +77.8] | 8 / 8 |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | uncovered | +13.0 [-13.6, +44.8] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RBSG2-L18-s0@18:RB0-F-s0@18 | covered | 85 (16 / 11) | 0.62 / 1.01 | -0.389 [-0.711, +0.124] | -0.135 [-0.551, +0.211] |
| RBSG2-L18-s0@18:RB0-F-s0@18 | uncovered | 234 (23 / 14) | 0.48 / 0.47 | +0.011 [-0.104, +0.110] | +0.000 [+0.000, +0.020] |
| RBSG2-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 13 of 332 | | | |
| RBSG2-L18-s1@18:RB0-F-s0@18 | covered | 87 (17 / 11) | 0.46 / 1.02 | -0.561 [-0.814, -0.129] | -0.252 [-0.666, -0.135] |
| RBSG2-L18-s1@18:RB0-F-s0@18 | uncovered | 239 (23 / 14) | 0.49 / 0.48 | +0.008 [-0.067, +0.102] | +0.000 [-0.001, +0.001] |
| RBSG2-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 12 of 338 | | | |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | covered | 86 (16 / 11) | 0.63 / 0.91 | -0.272 [-0.834, +0.094] | -0.157 [-0.372, +0.083] |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 233 (23 / 14) | 0.48 / 0.45 | +0.031 [-0.092, +0.130] | +0.000 [+0.000, +0.016] |
| RBSG2-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 13 of 332 | | | |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | covered | 87 (19 / 11) | 0.64 / 0.76 | -0.112 [-0.621, +0.244] | -0.067 [-0.412, +0.000] |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | uncovered | 229 (23 / 14) | 0.48 / 0.52 | -0.034 [-0.199, +0.021] | +0.000 [-0.011, +0.000] |
| RBSG2-L18-s0@18:RBSG-L18-s0@18 | discordant (left out) | 13 of 329 | | | |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | covered | 87 (17 / 10) | 0.45 / 0.96 | -0.518 [-0.843, +0.028] | -0.233 [-0.749, -0.004] |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | uncovered | 227 (23 / 14) | 0.48 / 0.52 | -0.038 [-0.167, -0.010] | -0.009 [-0.045, -0.001] |
| RBSG2-L18-s1@18:RBSG-L18-s1@18 | discordant (left out) | 11 of 325 | | | |
