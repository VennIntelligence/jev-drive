# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0-XP-L12-E60@18 (served) | 452 / 24 | 95.8 % (24) [89, 100] | 93.8 % (16) [86, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 95.7 % (23) [87, 100] | 23 / 0 / 0 / 1 | 0.25 / 0.53 (348) | +0.00 (+0.80) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | all | +29.2 [+8.7, +52.6] | 24 / 24 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | left | +25.0 [+5.3, +57.1] | 16 / 16 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | right | +37.5 [+0.0, +77.8] | 8 / 8 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | uncovered | +30.4 [+9.5, +54.2] | 23 / 23 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | all | +37.5 [+18.2, +60.0] | 24 / 24 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | left | +31.2 [+12.5, +58.3] | 16 / 16 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | right | +50.0 [+11.1, +88.9] | 8 / 8 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | uncovered | +34.8 [+13.6, +59.1] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | covered | 72 (19 / 11) | 0.22 / 0.75 | -0.536 [-1.085, -0.248] | -0.433 [-1.060, -0.272] |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | uncovered | 239 (23 / 14) | 0.25 / 0.47 | -0.224 [-0.282, -0.146] | -0.107 [-0.156, -0.067] |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s0@18 | discordant (left out) | 24 of 335 | | | |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | covered | 75 (21 / 12) | 0.24 / 0.64 | -0.398 [-0.854, -0.065] | -0.341 [-0.745, -0.102] |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | uncovered | 241 (23 / 14) | 0.25 / 0.48 | -0.232 [-0.302, -0.156] | -0.085 [-0.163, -0.037] |
| RSG-L18-s0-XP-L12-E60@18:RSG-L18-s1@18 | discordant (left out) | 22 of 338 | | | |
