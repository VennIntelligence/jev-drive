# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@18 (none) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | - | - | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| XP-L12-E46@18 (none) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | - | - | 22 / 0 / 0 / 2 | 0.22 / 0.56 (368) | -0.01 (+0.54) | 0.999 |
| WAJEPA@18 (none) | 24 / 24 | 29.2 % (24) [12, 50] | 25.0 % (16) [6, 53] | 37.5 % (8) [14, 62] | - | - | 7 / 14 / 1 / 2 | 0.51 / 1.69 (283) | +0.35 (+0.36) | 0.000 |
| LTF@18 (none) | 24 / 24 | 0.0 % (24) [0, 0] | 0.0 % (16) [0, 0] | 0.0 % (8) [0, 0] | - | - | 0 / 23 / 0 / 1 | 0.49 / 2.49 (270) | +0.06 (+0.06) | 0.000 |
| DD@18 (none) | 24 / 24 | 0.0 % (24) [0, 0] | 0.0 % (16) [0, 0] | 0.0 % (8) [0, 0] | - | - | 0 / 24 / 0 / 0 | 0.49 / 2.39 (250) | +0.13 (+0.13) | 0.000 |
| G0014@18 (none) | 24 / 24 | 4.2 % (24) [0, 12] | 6.2 % (16) [0, 20] | 0.0 % (8) [0, 0] | - | - | 1 / 18 / 4 / 1 | 0.85 / 3.74 (199) | +0.71 (+0.95) | 0.000 |
| RSGr@18 (none) | 24 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | - | - | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | +0.24 (+0.48) | 1.000 |
| XPr@18 (none) | 24 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | - | - | 22 / 0 / 0 / 2 | 0.22 / 0.56 (368) | +0.56 (+0.54) | 1.000 |
