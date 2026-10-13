# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RPAIT80-L18-s0@18 (served) | 452 / 24 | 54.2 % (24) [33, 74] | 56.2 % (16) [33, 75] | 50.0 % (8) [11, 89] | 100.0 % (1) [100, 100] | 52.2 % (23) [33, 72] | 13 / 6 / 2 / 3 | 0.64 / 2.44 (336) | -0.28 (+0.36) | 0.999 |
| RPAITC80-L18-s0@18 (served) | 452 / 24 | 58.3 % (24) [40, 76] | 68.8 % (16) [46, 88] | 37.5 % (8) [14, 62] | 100.0 % (1) [100, 100] | 56.5 % (23) [39, 75] | 14 / 5 / 0 / 5 | 0.56 / 1.89 (344) | -0.31 (+0.50) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RNONEG-L18-s0@18 (served) | 452 / 24 | 41.7 % (24) [20, 65] | 56.2 % (16) [33, 75] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 39.1 % (23) [20, 62] | 10 / 3 / 4 / 7 | 0.54 / 2.05 (348) | -0.26 (+0.46) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RNONEG-L18-s1@18 (served) | 452 / 24 | 50.0 % (24) [30, 71] | 56.2 % (16) [33, 75] | 37.5 % (8) [10, 71] | 100.0 % (1) [100, 100] | 47.8 % (23) [30, 68] | 12 / 5 / 2 / 5 | 0.55 / 1.97 (347) | -0.25 (+0.57) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | all | -12.5 [-31.6, +4.5] | 24 / 24 |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | left | -12.5 [-35.7, +12.5] | 16 / 16 |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | right | -12.5 [-42.9, +0.0] | 8 / 8 |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | uncovered | -13.0 [-31.8, +5.0] | 23 / 23 |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | all | +12.5 [-5.9, +34.5] | 24 / 24 |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | left | +0.0 [-18.8, +18.8] | 16 / 16 |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | right | +37.5 [+0.0, +77.8] | 8 / 8 |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +13.0 [-5.9, +34.8] | 23 / 23 |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | all | +0.0 [-16.7, +16.7] | 24 / 24 |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | left | +0.0 [-18.8, +18.8] | 16 / 16 |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | right | +0.0 [-33.3, +33.3] | 8 / 8 |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | uncovered | +0.0 [-17.2, +17.4] | 23 / 23 |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | all | -8.3 [-31.0, +14.8] | 24 / 24 |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | left | +0.0 [-16.7, +20.0] | 16 / 16 |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | right | -25.0 [-71.4, +20.0] | 8 / 8 |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | uncovered | -8.7 [-31.8, +15.4] | 23 / 23 |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | all | +16.7 [+0.0, +33.3] | 24 / 24 |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | left | +12.5 [+0.0, +31.2] | 16 / 16 |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | right | +25.0 [+0.0, +40.0] | 8 / 8 |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +17.4 [+0.0, +33.3] | 23 / 23 |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | all | +4.2 [-11.1, +17.4] | 24 / 24 |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | left | +12.5 [+0.0, +31.2] | 16 / 16 |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | right | -12.5 [-42.9, +0.0] | 8 / 8 |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | uncovered | +4.3 [-11.1, +17.4] | 23 / 23 |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | all | +4.2 [-20.0, +28.0] | 24 / 24 |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | left | +12.5 [-11.8, +38.5] | 16 / 16 |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | right | -12.5 [-55.6, +25.0] | 8 / 8 |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | uncovered | +4.3 [-21.1, +28.6] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | covered | 72 (18 / 12) | 1.60 / 1.18 | +0.426 [-0.147, +0.833] | +0.502 [+0.013, +0.637] |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | uncovered | 241 (23 / 14) | 0.52 / 0.47 | +0.050 [+0.006, +0.143] | +0.004 [+0.000, +0.016] |
| RPAIT80-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 12 of 325 | | | |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | covered | 72 (16 / 12) | 1.63 / 1.21 | +0.421 [-0.368, +0.984] | +0.563 [-0.685, +1.187] |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 239 (23 / 14) | 0.53 / 0.47 | +0.060 [-0.019, +0.164] | +0.003 [+0.000, +0.028] |
| RPAIT80-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 9 of 320 | | | |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | covered | 70 (16 / 12) | 1.63 / 1.09 | +0.547 [-0.370, +1.024] | +0.442 [-0.642, +1.129] |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | uncovered | 239 (23 / 14) | 0.52 / 0.49 | +0.034 [-0.020, +0.135] | +0.002 [+0.000, +0.026] |
| RPAIT80-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 10 of 319 | | | |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | covered | 73 (18 / 11) | 1.19 / 1.04 | +0.153 [-0.227, +0.546] | +0.194 [-0.301, +0.248] |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | uncovered | 242 (23 / 14) | 0.49 / 0.47 | +0.018 [-0.030, +0.098] | +0.000 [+0.000, +0.005] |
| RPAITC80-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 17 of 332 | | | |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | covered | 72 (15 / 11) | 1.24 / 1.10 | +0.134 [-0.406, +0.788] | +0.220 [-0.659, +0.456] |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 241 (23 / 14) | 0.49 / 0.47 | +0.020 [-0.044, +0.125] | +0.000 [+0.000, +0.001] |
| RPAITC80-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 13 of 326 | | | |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | covered | 70 (16 / 11) | 1.24 / 1.05 | +0.191 [-0.280, +0.811] | +0.077 [-0.348, +0.580] |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | uncovered | 241 (23 / 14) | 0.49 / 0.48 | +0.008 [-0.040, +0.093] | +0.000 [+0.000, +0.003] |
| RPAITC80-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 14 of 325 | | | |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | covered | 76 (21 / 13) | 1.38 / 1.56 | -0.179 [-0.447, +0.160] | -0.310 [-0.534, +0.159] |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | uncovered | 245 (23 / 14) | 0.49 / 0.53 | -0.044 [-0.093, +0.000] | -0.003 [-0.017, -0.000] |
| RPAITC80-L18-s0@18:RPAIT80-L18-s0@18 | discordant (left out) | 9 of 330 | | | |
