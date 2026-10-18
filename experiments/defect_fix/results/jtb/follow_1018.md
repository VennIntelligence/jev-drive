# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-on@18 (none) | 24 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | - | - | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | +0.24 (+0.48) | 1.000 |
| RSG-off@18 (none) | 24 / 24 | 54.2 % (24) [31, 79] | 56.2 % (16) [25, 80] | 50.0 % (8) [20, 86] | - | - | 13 / 4 / 1 / 6 | 0.47 / 1.72 (345) | +0.23 (+0.41) | 1.000 |
| JTN-on@18 (none) | 24 / 24 | 83.3 % (24) [72, 95] | 81.2 % (16) [62, 100] | 87.5 % (8) [57, 100] | - | - | 20 / 1 / 2 / 1 | 0.49 / 1.67 (327) | +0.66 (+1.06) | 1.000 |
| JTN-off@18 (none) | 24 / 24 | 45.8 % (24) [28, 62] | 25.0 % (16) [0, 42] | 87.5 % (8) [57, 100] | - | - | 11 / 6 / 4 / 3 | 0.51 / 2.00 (315) | +0.59 (+0.86) | 1.000 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSG-off@18:RSG-on@18 | all | -12.5 [-30.4, +4.2] | 24 / 24 |
| RSG-off@18:RSG-on@18 | left | -12.5 [-38.5, +9.5] | 16 / 16 |
| RSG-off@18:RSG-on@18 | right | -12.5 [-33.3, +0.0] | 8 / 8 |
| JTN-off@18:JTN-on@18 | all | -37.5 [-60.0, -20.0] | 24 / 24 |
| JTN-off@18:JTN-on@18 | left | -56.2 [-84.6, -33.3] | 16 / 16 |
| JTN-off@18:JTN-on@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| JTN-off@18:RSG-off@18 | all | -8.3 [-36.0, +18.2] | 24 / 24 |
| JTN-off@18:RSG-off@18 | left | -31.2 [-50.0, -7.7] | 16 / 16 |
| JTN-off@18:RSG-off@18 | right | +37.5 [+0.0, +70.0] | 8 / 8 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSG-off@18:RSG-on@18 | covered | 0 (0 / 0) | - | - | - |
| RSG-off@18:RSG-on@18 | uncovered | 332 (24 / 14) | 0.46 / 0.51 | -0.050 [-0.269, +0.054] | -0.000 [-0.031, +0.000] |
| RSG-off@18:RSG-on@18 | discordant (left out) | 0 of 332 | | | |
| JTN-off@18:JTN-on@18 | covered | 0 (0 / 0) | - | - | - |
| JTN-off@18:JTN-on@18 | uncovered | 300 (24 / 14) | 0.48 / 0.46 | +0.026 [-0.064, +0.078] | +0.000 [+0.000, +0.000] |
| JTN-off@18:JTN-on@18 | discordant (left out) | 0 of 300 | | | |
| JTN-off@18:RSG-off@18 | covered | 0 (0 / 0) | - | - | - |
| JTN-off@18:RSG-off@18 | uncovered | 312 (24 / 14) | 0.51 / 0.47 | +0.038 [-0.049, +0.122] | +0.000 [+0.000, +0.001] |
| JTN-off@18:RSG-off@18 | discordant (left out) | 0 of 312 | | | |
