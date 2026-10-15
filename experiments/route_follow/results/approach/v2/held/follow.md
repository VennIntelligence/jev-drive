# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| APP2W05-L18-s0@18 (served) | 452 / 24 | 58.3 % (24) [41, 81] | 56.2 % (16) [30, 86] | 62.5 % (8) [38, 86] | 0.0 % (1) [0, 0] | 60.9 % (23) [45, 82] | 14 / 4 / 4 / 2 | 0.73 / 2.55 (322) | -0.07 (+0.52) | 0.998 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| APP2W05-L18-s0@18:RSG-L18-s0@18 | all | -8.3 [-28.0, +14.3] | 24 / 24 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | left | -12.5 [-35.0, +16.7] | 16 / 16 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | right | +0.0 [-33.3, +33.3] | 8 / 8 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | uncovered | -4.3 [-22.2, +15.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| APP2W05-L18-s0@18:RSG-L18-s0@18 | covered | 68 (16 / 9) | 1.27 / 0.83 | +0.437 [-0.364, +1.396] | +0.305 [+0.000, +1.133] |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | uncovered | 232 (23 / 14) | 0.59 / 0.49 | +0.105 [-0.004, +0.229] | +0.001 [+0.000, +0.024] |
| APP2W05-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 15 of 315 | | | |
