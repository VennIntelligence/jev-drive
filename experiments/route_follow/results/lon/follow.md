# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGbothLon-L18-s0@18 (served) | 452 / 24 | 79.2 % (24) [65, 92] | 81.2 % (16) [61, 100] | 75.0 % (8) [60, 100] | 100.0 % (1) [100, 100] | 78.3 % (23) [64, 92] | 19 / 2 / 1 / 2 | 0.55 / 2.01 (336) | +0.08 (+0.76) | 0.999 |
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSGbC2-L18-s0@18 (served) | 452 / 24 | 83.3 % (24) [72, 95] | 87.5 % (16) [73, 100] | 75.0 % (8) [60, 100] | 100.0 % (1) [100, 100] | 82.6 % (23) [71, 95] | 20 / 2 / 1 / 1 | 0.55 / 2.03 (335) | -0.07 (+0.62) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | all | -12.5 [-25.0, +0.0] | 24 / 24 |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | left | -6.2 [-23.1, +0.0] | 16 / 16 |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | right | -25.0 [-40.0, +0.0] | 8 / 8 |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | uncovered | -13.0 [-25.0, +0.0] | 23 / 23 |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | all | +12.5 [+0.0, +28.1] | 24 / 24 |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | left | +12.5 [+0.0, +35.7] | 16 / 16 |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | right | +12.5 [-25.0, +55.6] | 8 / 8 |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | uncovered | +13.0 [+0.0, +29.2] | 23 / 23 |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | all | -4.2 [-15.0, +0.0] | 24 / 24 |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | left | -6.2 [-23.1, +0.0] | 16 / 16 |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | uncovered | -4.3 [-15.4, +0.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | covered | 83 (21 / 12) | 0.74 / 0.59 | +0.151 [-0.113, +0.223] | +0.014 [-0.068, +0.122] |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | uncovered | 238 (23 / 14) | 0.50 / 0.51 | -0.007 [-0.036, +0.020] | +0.000 [-0.000, +0.000] |
| RSGbothLon-L18-s0@18:RSGboth-L18-s0@18 | discordant (left out) | 3 of 324 | | | |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | covered | 79 (21 / 13) | 0.74 / 0.82 | -0.079 [-0.501, +0.459] | -0.029 [-0.155, +0.192] |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | uncovered | 238 (23 / 14) | 0.52 / 0.48 | +0.037 [-0.024, +0.126] | +0.000 [+0.000, +0.003] |
| RSGbothLon-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 16 of 333 | | | |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | covered | 93 (21 / 12) | 0.76 / 0.72 | +0.041 [-0.103, +0.204] | +0.010 [-0.049, +0.096] |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | uncovered | 237 (23 / 14) | 0.49 / 0.50 | -0.017 [-0.039, +0.026] | +0.000 [+0.000, +0.000] |
| RSGbothLon-L18-s0@18:RSGbC2-L18-s0@18 | discordant (left out) | 4 of 334 | | | |
