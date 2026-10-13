# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGmixPAIT-L18-s0@18 (served) | 452 / 24 | 70.8 % (24) [50, 90] | 75.0 % (16) [50, 93] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 69.6 % (23) [48, 89] | 17 / 0 / 2 / 5 | 0.59 / 2.20 (346) | -0.33 (+0.62) | 0.999 |
| RSGmixPAIT-L18-s1@18 (served) | 452 / 24 | 62.5 % (24) [40, 82] | 68.8 % (16) [42, 88] | 50.0 % (8) [11, 89] | 100.0 % (1) [100, 100] | 60.9 % (23) [39, 82] | 15 / 3 / 2 / 4 | 0.62 / 2.24 (344) | -0.31 (+0.60) | 0.999 |
| RSGPAIT-L18-s0@18 (served) | 452 / 24 | 87.5 % (24) [72, 100] | 87.5 % (16) [67, 100] | 87.5 % (8) [57, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [71, 100] | 21 / 0 / 2 / 1 | 0.59 / 2.15 (328) | +0.05 (+0.98) | 0.998 |
| RSGPAIT-L18-s1@18 (served) | 452 / 24 | 87.5 % (24) [75, 100] | 93.8 % (16) [77, 100] | 75.0 % (8) [43, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [74, 100] | 21 / 0 / 2 / 1 | 0.62 / 2.23 (331) | +0.06 (+1.02) | 0.998 |
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSGboth-L18-s1@18 (served) | 452 / 24 | 87.5 % (24) [71, 100] | 81.2 % (16) [54, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [71, 100] | 21 / 1 / 1 / 1 | 0.54 / 2.02 (330) | +0.11 (+0.94) | 0.998 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RB0-F-s1@18 (served) | 452 / 24 | 62.5 % (24) [50, 78] | 75.0 % (16) [58, 100] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [51, 80] | 15 / 6 / 1 / 2 | 0.52 / 2.01 (333) | -0.21 (+0.77) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | all | +4.2 [+0.0, +15.0] | 24 / 24 |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | left | +6.2 [+0.0, +23.1] | 16 / 16 |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | uncovered | +4.3 [+0.0, +15.4] | 23 / 23 |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | all | +4.2 [-11.1, +17.4] | 24 / 24 |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | left | +6.2 [-16.7, +25.0] | 16 / 16 |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | uncovered | +0.0 [-13.0, +13.0] | 23 / 23 |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | all | +16.7 [+0.0, +36.4] | 24 / 24 |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | left | +18.8 [+0.0, +38.9] | 16 / 16 |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | uncovered | +17.4 [+0.0, +36.8] | 23 / 23 |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | all | +8.3 [-12.0, +27.3] | 24 / 24 |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | left | +12.5 [-13.3, +35.7] | 16 / 16 |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | right | +0.0 [-33.3, +33.3] | 8 / 8 |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | uncovered | +8.7 [-12.5, +27.8] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | covered | 83 (20 / 11) | 1.02 / 0.84 | +0.185 [-0.262, +0.429] | +0.124 [-0.103, +0.174] |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | uncovered | 244 (23 / 14) | 0.48 / 0.48 | +0.006 [-0.057, +0.073] | +0.001 [+0.000, +0.006] |
| RSGmixPAIT-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 14 of 341 | | | |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | covered | 86 (20 / 12) | 1.23 / 0.87 | +0.358 [-0.051, +0.633] | +0.200 [+0.023, +0.401] |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | uncovered | 243 (23 / 14) | 0.50 / 0.49 | +0.003 [-0.034, +0.077] | +0.000 [+0.000, +0.012] |
| RSGmixPAIT-L18-s1@18:RSG-L18-s1@18 | discordant (left out) | 15 of 344 | | | |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | covered | 78 (18 / 11) | 1.00 / 0.99 | +0.013 [-0.255, +0.478] | +0.073 [-0.349, +0.361] |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | uncovered | 242 (23 / 14) | 0.48 / 0.49 | -0.003 [-0.116, +0.103] | +0.000 [+0.000, +0.007] |
| RSGmixPAIT-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 14 of 334 | | | |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | covered | 79 (19 / 11) | 1.17 / 1.01 | +0.161 [-0.325, +0.695] | +0.154 [-0.481, +0.452] |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | uncovered | 241 (23 / 14) | 0.50 / 0.48 | +0.012 [-0.069, +0.111] | +0.000 [-0.000, +0.005] |
| RSGmixPAIT-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 16 of 336 | | | |
