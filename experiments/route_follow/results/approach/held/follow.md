# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| APP1X-L18-s0@18 (served) | 452 / 24 | 33.3 % (24) [17, 56] | 31.2 % (16) [11, 58] | 37.5 % (8) [11, 71] | 0.0 % (1) [0, 0] | 34.8 % (23) [18, 57] | 8 / 8 / 4 / 4 | 0.68 / 2.64 (330) | -0.48 (+0.12) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| APP1X-L18-s0@18:RSG-L18-s0@18 | all | -33.3 [-54.2, -9.5] | 24 / 24 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | left | -37.5 [-60.9, +0.0] | 16 / 16 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | right | -25.0 [-57.1, +0.0] | 8 / 8 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| APP1X-L18-s0@18:RSG-L18-s0@18 | uncovered | -30.4 [-50.0, -9.1] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| APP1X-L18-s0@18:RSG-L18-s0@18 | covered | 72 (15 / 9) | 1.19 / 0.83 | +0.363 [-0.813, +1.032] | +0.258 [-0.877, +0.657] |
| APP1X-L18-s0@18:RSG-L18-s0@18 | uncovered | 235 (23 / 14) | 0.56 / 0.49 | +0.074 [-0.025, +0.188] | +0.003 [+0.000, +0.033] |
| APP1X-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 10 of 317 | | | |
