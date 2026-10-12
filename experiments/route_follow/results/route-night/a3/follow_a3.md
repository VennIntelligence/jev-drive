# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RBSGc-L18-s0@18 (served) | 452 / 24 | 79.2 % (24) [62, 95] | 81.2 % (16) [54, 100] | 75.0 % (8) [43, 100] | 100.0 % (1) [100, 100] | 78.3 % (23) [62, 95] | 19 / 1 / 0 / 4 | 0.54 / 1.99 (348) | -0.31 (+0.45) | 0.999 |
| RBSGc-L18-s1@18 (served) | 452 / 24 | 70.8 % (24) [52, 88] | 81.2 % (16) [54, 100] | 50.0 % (8) [14, 80] | 100.0 % (1) [100, 100] | 69.6 % (23) [50, 88] | 17 / 2 / 0 / 5 | 0.68 / 2.16 (350) | -0.33 (+0.39) | 0.999 |
| RBSG-L18-s0@18 (served) | 452 / 24 | 54.2 % (24) [30, 79] | 75.0 % (16) [50, 93] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 52.2 % (23) [29, 78] | 13 / 3 / 3 / 5 | 0.60 / 2.23 (337) | -0.36 (+0.21) | 0.999 |
| RBSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [39, 81] | 68.8 % (16) [46, 92] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 60.9 % (23) [40, 83] | 14 / 4 / 1 / 5 | 0.65 / 2.31 (335) | -0.35 (+0.27) | 0.999 |
| RNONEG-L18-s0@18 (served) | 452 / 24 | 41.7 % (24) [20, 65] | 56.2 % (16) [33, 75] | 12.5 % (8) [0, 43] | 100.0 % (1) [100, 100] | 39.1 % (23) [20, 62] | 10 / 3 / 4 / 7 | 0.54 / 2.05 (348) | -0.26 (+0.46) | 0.999 |
| RNONE-L18-s0@18 (served) | 452 / 24 | 62.5 % (24) [47, 76] | 75.0 % (16) [55, 93] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [48, 82] | 15 / 5 / 2 / 2 | 0.48 / 1.73 (334) | -0.24 (+0.63) | 0.999 |
| RNONE-L18-s1@18 (served) | 452 / 24 | 62.5 % (24) [47, 76] | 75.0 % (16) [55, 93] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [48, 82] | 15 / 5 / 1 / 3 | 0.50 / 1.84 (339) | -0.23 (+0.60) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RB0-F-s1@18 (served) | 452 / 24 | 62.5 % (24) [50, 78] | 75.0 % (16) [58, 100] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [51, 80] | 15 / 6 / 1 / 2 | 0.52 / 2.01 (333) | -0.21 (+0.77) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSG-L18-s0@18:RB0-F-s0@18 | all | +12.5 [-4.2, +29.6] | 24 / 24 |
| RSG-L18-s0@18:RB0-F-s0@18 | left | +12.5 [+0.0, +26.7] | 16 / 16 |
| RSG-L18-s0@18:RB0-F-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSG-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0@18:RB0-F-s0@18 | uncovered | +13.0 [-4.5, +30.0] | 23 / 23 |
| RSG-L18-s1@18:RB0-F-s0@18 | all | +4.2 [-13.3, +22.7] | 24 / 24 |
| RSG-L18-s1@18:RB0-F-s0@18 | left | +6.2 [-11.8, +26.7] | 16 / 16 |
| RSG-L18-s1@18:RB0-F-s0@18 | right | +0.0 [-33.3, +33.3] | 8 / 8 |
| RSG-L18-s1@18:RB0-F-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RSG-L18-s1@18:RB0-F-s0@18 | uncovered | +8.7 [-7.7, +23.3] | 23 / 23 |
| RBSGc-L18-s0@18:RB0-F-s0@18 | all | +25.0 [+12.0, +39.1] | 24 / 24 |
| RBSGc-L18-s0@18:RB0-F-s0@18 | left | +25.0 [+8.3, +41.2] | 16 / 16 |
| RBSGc-L18-s0@18:RB0-F-s0@18 | right | +25.0 [+0.0, +55.6] | 8 / 8 |
| RBSGc-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RBSGc-L18-s0@18:RB0-F-s0@18 | uncovered | +26.1 [+12.5, +40.7] | 23 / 23 |
| RBSGc-L18-s1@18:RB0-F-s0@18 | all | +16.7 [+0.0, +30.8] | 24 / 24 |
| RBSGc-L18-s1@18:RB0-F-s0@18 | left | +25.0 [+8.3, +41.2] | 16 / 16 |
| RBSGc-L18-s1@18:RB0-F-s0@18 | right | +0.0 [-42.9, +30.0] | 8 / 8 |
| RBSGc-L18-s1@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RBSGc-L18-s1@18:RB0-F-s0@18 | uncovered | +17.4 [+0.0, +32.0] | 23 / 23 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | all | +25.0 [+8.0, +44.4] | 24 / 24 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | left | +12.5 [+0.0, +26.7] | 16 / 16 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | right | +50.0 [+11.1, +88.9] | 8 / 8 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +26.1 [+8.7, +45.0] | 23 / 23 |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | all | +37.5 [+22.7, +52.6] | 24 / 24 |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | left | +25.0 [+8.3, +41.2] | 16 / 16 |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | right | +62.5 [+28.6, +88.9] | 8 / 8 |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | uncovered | +39.1 [+23.8, +54.5] | 23 / 23 |
| RBSG-L18-s0@18:RB0-F-s0@18 | all | +0.0 [-23.1, +24.0] | 24 / 24 |
| RBSG-L18-s0@18:RB0-F-s0@18 | left | +18.8 [+0.0, +38.9] | 16 / 16 |
| RBSG-L18-s0@18:RB0-F-s0@18 | right | -37.5 [-75.0, +14.3] | 8 / 8 |
| RBSG-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RBSG-L18-s0@18:RB0-F-s0@18 | uncovered | +0.0 [-23.8, +25.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0@18:RB0-F-s0@18 | covered | 92 (19 / 12) | 0.92 / 1.04 | -0.118 [-0.517, +0.390] | -0.011 [-0.545, +0.077] |
| RSG-L18-s0@18:RB0-F-s0@18 | uncovered | 249 (23 / 14) | 0.48 / 0.50 | -0.012 [-0.091, +0.056] | +0.000 [-0.002, +0.000] |
| RSG-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 4 of 345 | | | |
| RSG-L18-s1@18:RB0-F-s0@18 | covered | 91 (19 / 12) | 0.84 / 1.04 | -0.196 [-0.627, +0.332] | -0.010 [-0.481, +0.143] |
| RSG-L18-s1@18:RB0-F-s0@18 | uncovered | 249 (23 / 14) | 0.51 / 0.49 | +0.013 [-0.072, +0.085] | +0.000 [-0.000, +0.000] |
| RSG-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 5 of 345 | | | |
| RBSGc-L18-s0@18:RB0-F-s0@18 | covered | 86 (19 / 12) | 0.57 / 1.04 | -0.464 [-0.848, +0.142] | -0.016 [-0.358, +0.324] |
| RBSGc-L18-s0@18:RB0-F-s0@18 | uncovered | 242 (23 / 14) | 0.50 / 0.49 | +0.014 [-0.051, +0.092] | +0.000 [+0.000, +0.006] |
| RBSGc-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 11 of 339 | | | |
| RBSGc-L18-s1@18:RB0-F-s0@18 | covered | 87 (19 / 12) | 0.89 / 1.04 | -0.147 [-0.457, +0.613] | +0.000 [-0.215, +0.649] |
| RBSGc-L18-s1@18:RB0-F-s0@18 | uncovered | 240 (23 / 14) | 0.55 / 0.49 | +0.060 [+0.004, +0.191] | +0.001 [+0.000, +0.012] |
| RBSGc-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 12 of 339 | | | |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | covered | 90 (19 / 12) | 0.92 / 0.94 | -0.021 [-0.377, +0.345] | -0.106 [-0.436, +0.058] |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 246 (23 / 14) | 0.48 / 0.47 | +0.010 [-0.080, +0.088] | +0.000 [-0.001, +0.001] |
| RSG-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 9 of 345 | | | |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | covered | 87 (18 / 12) | 0.59 / 0.93 | -0.340 [-0.869, +0.103] | +0.000 [-0.248, +0.284] |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | uncovered | 242 (23 / 14) | 0.50 / 0.47 | +0.033 [-0.045, +0.116] | +0.000 [+0.000, +0.008] |
| RBSGc-L18-s0@18:RNONEG-L18-s0@18 | discordant (left out) | 10 of 339 | | | |
| RBSG-L18-s0@18:RB0-F-s0@18 | covered | 83 (16 / 11) | 0.84 / 0.95 | -0.108 [-0.713, +0.557] | +0.005 [-0.312, +0.466] |
| RBSG-L18-s0@18:RB0-F-s0@18 | uncovered | 232 (23 / 14) | 0.52 / 0.48 | +0.038 [-0.024, +0.197] | +0.000 [+0.000, +0.045] |
| RBSG-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 9 of 324 | | | |
