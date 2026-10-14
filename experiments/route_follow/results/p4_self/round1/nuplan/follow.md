# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGSELF-L18-s0@18 (served) | 452 / 24 | 62.5 % (24) [38, 84] | 75.0 % (16) [50, 93] | 37.5 % (8) [0, 78] | 100.0 % (1) [100, 100] | 60.9 % (23) [36, 83] | 15 / 0 / 5 / 4 | 0.70 / 2.62 (361) | -0.27 (+0.19) | 0.999 |
| RSGSELF-L18-s1@18 (served) | 452 / 24 | 54.2 % (24) [33, 75] | 68.8 % (16) [47, 93] | 25.0 % (8) [0, 56] | 0.0 % (1) [0, 0] | 56.5 % (23) [35, 77] | 13 / 1 / 4 / 6 | 0.71 / 2.71 (359) | -0.25 (+0.15) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | all | -4.2 [-26.3, +13.0] | 24 / 24 |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | left | +6.2 [-15.4, +31.2] | 16 / 16 |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | right | -25.0 [-66.7, +0.0] | 8 / 8 |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | uncovered | -4.3 [-27.8, +13.6] | 23 / 23 |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | all | -4.2 [-21.1, +10.0] | 24 / 24 |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | left | +6.2 [-15.4, +31.2] | 16 / 16 |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | right | -25.0 [-57.1, +0.0] | 8 / 8 |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | uncovered | -4.3 [-22.2, +10.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | covered | 91 (20 / 11) | 1.03 / 0.97 | +0.062 [-0.241, +0.496] | +0.357 [+0.087, +0.539] |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | uncovered | 245 (23 / 14) | 0.53 / 0.48 | +0.052 [-0.000, +0.124] | +0.005 [+0.000, +0.018] |
| RSGSELF-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 10 of 346 | | | |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | covered | 93 (19 / 12) | 1.32 / 0.71 | +0.607 [+0.112, +1.207] | +0.510 [+0.211, +0.907] |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | uncovered | 244 (23 / 14) | 0.52 / 0.50 | +0.019 [-0.047, +0.125] | +0.003 [+0.000, +0.023] |
| RSGSELF-L18-s1@18:RSG-L18-s1@18 | discordant (left out) | 8 of 345 | | | |
