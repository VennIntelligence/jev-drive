# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSGk26-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 1 / 6 | 0.53 / 1.94 (350) | -0.27 (+0.40) | 0.999 |
| RSGk51-L18-s0@18 (served) | 452 / 24 | 58.3 % (24) [40, 80] | 62.5 % (16) [42, 85] | 50.0 % (8) [20, 86] | 0.0 % (1) [0, 0] | 60.9 % (23) [42, 83] | 14 / 2 / 2 / 6 | 0.53 / 2.16 (344) | -0.24 (+0.48) | 0.999 |
| RSGk51w1-L18-s0@18 (served) | 452 / 24 | 70.8 % (24) [56, 86] | 75.0 % (16) [50, 93] | 62.5 % (8) [29, 90] | 100.0 % (1) [100, 100] | 69.6 % (23) [54, 86] | 17 / 0 / 1 / 6 | 0.56 / 2.03 (354) | -0.24 (+0.44) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGk26-L18-s0@18:RSG-L18-s0@18 | all | +0.0 [+0.0, +0.0] | 24 / 24 |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | uncovered | +0.0 [+0.0, +0.0] | 23 / 23 |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | all | -8.3 [-20.8, +9.1] | 24 / 24 |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | left | -6.2 [-21.7, +18.2] | 16 / 16 |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | right | -12.5 [-33.3, +0.0] | 8 / 8 |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | uncovered | -4.3 [-17.4, +11.1] | 23 / 23 |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | all | +4.2 [-10.5, +18.2] | 24 / 24 |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | left | +6.2 [+0.0, +23.1] | 16 / 16 |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | right | +0.0 [-42.9, +30.0] | 8 / 8 |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | uncovered | +4.3 [-11.1, +19.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGk26-L18-s0@18:RSG-L18-s0@18 | covered | 94 (22 / 13) | 0.85 / 0.83 | +0.018 [-0.400, +0.321] | +0.001 [-0.161, +0.106] |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | uncovered | 249 (23 / 14) | 0.49 / 0.48 | +0.007 [-0.041, +0.044] | +0.000 [+0.000, +0.000] |
| RSGk26-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 6 of 349 | | | |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | covered | 86 (18 / 11) | 0.77 / 0.92 | -0.153 [-0.550, +0.334] | -0.000 [-0.292, +0.381] |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | uncovered | 240 (23 / 14) | 0.48 / 0.48 | +0.005 [-0.079, +0.060] | +0.000 [-0.002, +0.000] |
| RSGk51-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 13 of 339 | | | |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | covered | 91 (21 / 12) | 0.88 / 0.84 | +0.038 [-0.403, +0.331] | +0.015 [-0.169, +0.107] |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | uncovered | 245 (23 / 14) | 0.48 / 0.48 | +0.005 [-0.029, +0.050] | +0.000 [+0.000, +0.000] |
| RSGk51w1-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 11 of 347 | | | |
