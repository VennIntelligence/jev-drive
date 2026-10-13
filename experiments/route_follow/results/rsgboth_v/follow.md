# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGbA30-L18-s0@18 (served) | 452 / 24 | 75.0 % (24) [58, 91] | 75.0 % (16) [50, 93] | 75.0 % (8) [60, 100] | 100.0 % (1) [100, 100] | 73.9 % (23) [57, 91] | 18 / 2 / 2 / 2 | 0.53 / 2.06 (329) | +0.11 (+0.86) | 0.998 |
| RSGbW-L18-s0@18 (served) | 452 / 24 | 83.3 % (24) [72, 95] | 87.5 % (16) [73, 100] | 75.0 % (8) [60, 100] | 100.0 % (1) [100, 100] | 82.6 % (23) [71, 95] | 20 / 2 / 2 / 0 | 0.62 / 2.16 (312) | +0.11 (+0.95) | 0.998 |
| RSGbC2-L18-s0@18 (served) | 452 / 24 | 83.3 % (24) [72, 95] | 87.5 % (16) [73, 100] | 75.0 % (8) [60, 100] | 100.0 % (1) [100, 100] | 82.6 % (23) [71, 95] | 20 / 2 / 1 / 1 | 0.55 / 2.03 (335) | -0.07 (+0.62) | 0.999 |
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | all | -16.7 [-34.6, +0.0] | 24 / 24 |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | left | -12.5 [-35.7, +0.0] | 16 / 16 |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | right | -25.0 [-40.0, +0.0] | 8 / 8 |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | uncovered | -17.4 [-34.8, +0.0] | 23 / 23 |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | all | +8.3 [-9.1, +27.3] | 24 / 24 |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | left | +6.2 [+0.0, +23.1] | 16 / 16 |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | uncovered | +8.7 [-9.5, +27.6] | 23 / 23 |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | all | -8.3 [-17.9, +0.0] | 24 / 24 |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | right | -25.0 [-40.0, +0.0] | 8 / 8 |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | uncovered | -8.7 [-18.2, +0.0] | 23 / 23 |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | all | +16.7 [+0.0, +34.6] | 24 / 24 |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | uncovered | +17.4 [+0.0, +35.0] | 23 / 23 |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | all | -8.3 [-17.9, +0.0] | 24 / 24 |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | right | -25.0 [-40.0, +0.0] | 8 / 8 |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | uncovered | -8.7 [-18.2, +0.0] | 23 / 23 |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | all | +16.7 [+0.0, +34.6] | 24 / 24 |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | uncovered | +17.4 [+0.0, +35.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | covered | 85 (21 / 12) | 0.74 / 0.59 | +0.150 [-0.189, +0.221] | +0.000 [-0.170, +0.092] |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | uncovered | 237 (23 / 14) | 0.50 / 0.50 | -0.004 [-0.056, +0.025] | +0.000 [-0.003, +0.000] |
| RSGbA30-L18-s0@18:RSGboth-L18-s0@18 | discordant (left out) | 2 of 324 | | | |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | covered | 74 (21 / 13) | 0.75 / 0.92 | -0.170 [-0.606, +0.533] | -0.162 [-0.364, +0.163] |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | uncovered | 236 (23 / 14) | 0.50 / 0.48 | +0.025 [-0.028, +0.090] | +0.000 [+0.000, +0.002] |
| RSGbA30-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 17 of 327 | | | |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | covered | 73 (19 / 12) | 0.96 / 0.52 | +0.438 [-0.011, +0.538] | +0.100 [-0.045, +0.436] |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | uncovered | 225 (23 / 14) | 0.50 / 0.48 | +0.020 [-0.026, +0.117] | +0.000 [+0.000, +0.001] |
| RSGbW-L18-s0@18:RSGboth-L18-s0@18 | discordant (left out) | 10 of 308 | | | |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | covered | 63 (17 / 10) | 0.85 / 0.82 | +0.026 [-0.641, +0.624] | -0.003 [-0.207, +0.346] |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | uncovered | 223 (23 / 14) | 0.50 / 0.47 | +0.028 [-0.025, +0.150] | +0.000 [+0.000, +0.004] |
| RSGbW-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 23 of 309 | | | |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | covered | 84 (22 / 13) | 0.74 / 0.61 | +0.122 [-0.199, +0.233] | +0.088 [-0.101, +0.126] |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | uncovered | 235 (23 / 14) | 0.50 / 0.49 | +0.011 [-0.027, +0.025] | +0.000 [-0.000, +0.000] |
| RSGbC2-L18-s0@18:RSGboth-L18-s0@18 | discordant (left out) | 5 of 324 | | | |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | covered | 77 (20 / 12) | 0.70 / 0.84 | -0.140 [-0.531, +0.316] | -0.015 [-0.186, +0.131] |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | uncovered | 234 (23 / 14) | 0.51 / 0.48 | +0.031 [-0.024, +0.098] | +0.000 [+0.000, +0.002] |
| RSGbC2-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 20 of 331 | | | |
