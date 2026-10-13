# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@train (served) | 104 / 64 | 32.8 % (64) [22, 43] | 40.0 % (40) [26, 51] | 20.8 % (24) [7, 38] | 33.3 % (3) [0, 100] | 32.8 % (61) [22, 43] | 21 / 23 / 9 / 11 | 0.74 / 2.56 (918) | -0.25 (-0.09) | 1.000 |
| RSG-L18-s0@held (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSGboth-L18-s0@train (served) | 104 / 64 | 57.8 % (64) [46, 69] | 62.5 % (40) [47, 76] | 50.0 % (24) [29, 71] | 66.7 % (3) [50, 100] | 57.4 % (61) [45, 69] | 37 / 18 / 4 / 5 | 0.70 / 2.17 (843) | +0.21 (+0.52) | 0.997 |
| RSGboth-L18-s0@held (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RB0-F-s0@train (served) | 104 / 64 | 28.1 % (64) [17, 40] | 30.0 % (40) [17, 45] | 25.0 % (24) [9, 42] | 0.0 % (3) [0, 0] | 29.5 % (61) [18, 41] | 18 / 31 / 9 / 6 | 0.71 / 2.62 (881) | -0.30 (-0.10) | 1.000 |
| RB0-F-s0@held (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGboth-L18-s0@train:RSG-L18-s0@train | all | +25.0 [+12.5, +38.6] | 64 / 64 |
| RSGboth-L18-s0@train:RSG-L18-s0@train | left | +22.5 [+9.4, +37.8] | 40 / 40 |
| RSGboth-L18-s0@train:RSG-L18-s0@train | right | +29.2 [+4.3, +52.2] | 24 / 24 |
| RSGboth-L18-s0@train:RSG-L18-s0@train | covered | +33.3 [+0.0, +50.0] | 3 / 3 |
| RSGboth-L18-s0@train:RSG-L18-s0@train | uncovered | +24.6 [+11.7, +38.6] | 61 / 61 |
| RSG-L18-s0@train:RB0-F-s0@train | all | +4.7 [-3.4, +14.8] | 64 / 64 |
| RSG-L18-s0@train:RB0-F-s0@train | left | +10.0 [+0.0, +22.9] | 40 / 40 |
| RSG-L18-s0@train:RB0-F-s0@train | right | -4.2 [-12.5, +0.0] | 24 / 24 |
| RSG-L18-s0@train:RB0-F-s0@train | covered | +33.3 [+0.0, +100.0] | 3 / 3 |
| RSG-L18-s0@train:RB0-F-s0@train | uncovered | +3.3 [-3.6, +11.0] | 61 / 61 |
| RSGboth-L18-s0@train:RB0-F-s0@train | all | +29.7 [+16.4, +42.9] | 64 / 64 |
| RSGboth-L18-s0@train:RB0-F-s0@train | left | +32.5 [+17.1, +46.2] | 40 / 40 |
| RSGboth-L18-s0@train:RB0-F-s0@train | right | +25.0 [+0.0, +48.0] | 24 / 24 |
| RSGboth-L18-s0@train:RB0-F-s0@train | covered | +66.7 [+50.0, +100.0] | 3 / 3 |
| RSGboth-L18-s0@train:RB0-F-s0@train | uncovered | +27.9 [+14.5, +41.4] | 61 / 61 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGboth-L18-s0@train:RSG-L18-s0@train | covered | 290 (51 / 32) | 1.07 / 1.22 | -0.151 [-0.466, +0.298] | -0.055 [-0.201, -0.001] |
| RSGboth-L18-s0@train:RSG-L18-s0@train | uncovered | 487 (59 / 39) | 0.50 / 0.50 | +0.002 [-0.039, +0.057] | +0.000 [+0.000, +0.000] |
| RSGboth-L18-s0@train:RSG-L18-s0@train | discordant (left out) | 39 of 816 | | | |
| RSG-L18-s0@train:RB0-F-s0@train | covered | 337 (50 / 32) | 1.20 / 1.21 | -0.007 [-0.337, +0.245] | -0.016 [-0.142, +0.003] |
| RSG-L18-s0@train:RB0-F-s0@train | uncovered | 524 (59 / 39) | 0.52 / 0.55 | -0.027 [-0.072, +0.020] | +0.000 [+0.000, +0.000] |
| RSG-L18-s0@train:RB0-F-s0@train | discordant (left out) | 5 of 866 | | | |
| RSGboth-L18-s0@train:RB0-F-s0@train | covered | 281 (47 / 30) | 1.03 / 1.09 | -0.061 [-0.626, +0.431] | -0.015 [-0.213, +0.055] |
| RSGboth-L18-s0@train:RB0-F-s0@train | uncovered | 485 (59 / 39) | 0.50 / 0.52 | -0.012 [-0.067, +0.038] | +0.000 [+0.000, +0.000] |
| RSGboth-L18-s0@train:RB0-F-s0@train | discordant (left out) | 38 of 804 | | | |
