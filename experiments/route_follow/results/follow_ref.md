# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| YS-L18-s0@18 | 452 / 24 | 54.2 % (24) [38, 72] | 62.5 % (16) [39, 86] | 37.5 % (8) [10, 71] | - | - | 13 / 4 / 3 / 4 | 0.49 / 2.23 (353) | -0.26 (+0.22) | 0.999 |
| YS-L18-s1@18 | 452 / 24 | 54.2 % (24) [38, 72] | 62.5 % (16) [39, 86] | 37.5 % (8) [10, 71] | - | - | 13 / 3 / 4 / 4 | 0.49 / 2.31 (349) | -0.23 (+0.22) | 0.999 |
| YS2G-L18-s0@18 | 452 / 24 | 58.3 % (24) [40, 76] | 68.8 % (16) [42, 88] | 37.5 % (8) [10, 71] | - | - | 14 / 3 / 2 / 5 | 0.51 / 2.01 (349) | -0.25 (+0.33) | 0.999 |
| YS2G-L18-s1@18 | 452 / 24 | 58.3 % (24) [40, 76] | 68.8 % (16) [42, 88] | 37.5 % (8) [10, 71] | - | - | 14 / 3 / 2 / 5 | 0.49 / 2.00 (345) | -0.23 (+0.38) | 0.999 |
| YR10m10-F-s0@18 | 452 / 24 | 37.5 % (24) [12, 65] | 50.0 % (16) [15, 78] | 12.5 % (8) [0, 43] | - | - | 9 / 6 / 5 / 4 | 0.51 / 1.92 (333) | -0.29 (+0.45) | 0.999 |
| YR10m10-F-s1@18 | 452 / 24 | 37.5 % (24) [13, 63] | 43.8 % (16) [8, 72] | 25.0 % (8) [0, 56] | - | - | 9 / 8 / 3 / 4 | 0.51 / 1.88 (322) | -0.25 (+0.59) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| YS-L18-s1@18:YR10m10-F-s0@18 | all | +16.7 [-12.5, +45.5] | 24 / 24 |
| YS-L18-s1@18:YR10m10-F-s0@18 | left | +12.5 [-23.5, +53.3] | 16 / 16 |
| YS-L18-s1@18:YR10m10-F-s0@18 | right | +25.0 [-25.0, +71.4] | 8 / 8 |
| YS2G-L18-s1@18:YR10m10-F-s0@18 | all | +20.8 [-4.2, +45.8] | 24 / 24 |
| YS2G-L18-s1@18:YR10m10-F-s0@18 | left | +18.8 [-7.1, +53.3] | 16 / 16 |
| YS2G-L18-s1@18:YR10m10-F-s0@18 | right | +25.0 [-25.0, +71.4] | 8 / 8 |
