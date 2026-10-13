# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSGboth-L18-s1@18 (served) | 452 / 24 | 87.5 % (24) [71, 100] | 81.2 % (16) [54, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [71, 100] | 21 / 1 / 1 / 1 | 0.54 / 2.02 (330) | +0.11 (+0.94) | 0.998 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |
| RB0-F-s0@18 (served) | 452 / 24 | 54.2 % (24) [38, 72] | 56.2 % (16) [33, 75] | 50.0 % (8) [25, 75] | 100.0 % (1) [100, 100] | 52.2 % (23) [38, 70] | 13 / 4 / 2 / 5 | 0.54 / 1.98 (347) | -0.25 (+0.56) | 0.999 |
| RB0-F-s1@18 (served) | 452 / 24 | 62.5 % (24) [50, 78] | 75.0 % (16) [58, 100] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 65.2 % (23) [51, 80] | 15 / 6 / 1 / 2 | 0.52 / 2.01 (333) | -0.21 (+0.77) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGboth-L18-s0@18:RSG-L18-s0@18 | all | +25.0 [+4.8, +48.0] | 24 / 24 |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | right | +37.5 [+0.0, +77.8] | 8 / 8 |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | uncovered | +26.1 [+5.3, +50.0] | 23 / 23 |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | all | +29.2 [+5.9, +51.6] | 24 / 24 |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | left | +18.8 [+0.0, +36.4] | 16 / 16 |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | right | +50.0 [+11.1, +88.9] | 8 / 8 |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | uncovered | +26.1 [+5.0, +50.0] | 23 / 23 |
| RSGboth-L18-s0@18:RB0-F-s0@18 | all | +37.5 [+16.0, +56.5] | 24 / 24 |
| RSGboth-L18-s0@18:RB0-F-s0@18 | left | +31.2 [+10.5, +58.3] | 16 / 16 |
| RSGboth-L18-s0@18:RB0-F-s0@18 | right | +50.0 [+25.0, +75.0] | 8 / 8 |
| RSGboth-L18-s0@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0@18:RB0-F-s0@18 | uncovered | +39.1 [+17.6, +56.5] | 23 / 23 |
| RSGboth-L18-s1@18:RB0-F-s0@18 | all | +33.3 [+15.8, +48.0] | 24 / 24 |
| RSGboth-L18-s1@18:RB0-F-s0@18 | left | +25.0 [+8.3, +40.9] | 16 / 16 |
| RSGboth-L18-s1@18:RB0-F-s0@18 | right | +50.0 [+25.0, +75.0] | 8 / 8 |
| RSGboth-L18-s1@18:RB0-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s1@18:RB0-F-s0@18 | uncovered | +34.8 [+16.7, +50.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGboth-L18-s0@18:RSG-L18-s0@18 | covered | 72 (21 / 12) | 0.51 / 0.97 | -0.460 [-0.732, +0.375] | -0.069 [-0.215, +0.107] |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | uncovered | 237 (23 / 14) | 0.50 / 0.47 | +0.029 [-0.022, +0.120] | +0.000 [+0.000, +0.003] |
| RSGboth-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 18 of 327 | | | |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | covered | 75 (21 / 12) | 0.63 / 0.69 | -0.054 [-0.478, +0.551] | +0.000 [-0.186, +0.335] |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | uncovered | 239 (23 / 14) | 0.51 / 0.49 | +0.016 [-0.032, +0.138] | +0.000 [+0.000, +0.002] |
| RSGboth-L18-s1@18:RSG-L18-s1@18 | discordant (left out) | 16 of 330 | | | |
| RSGboth-L18-s0@18:RB0-F-s0@18 | covered | 69 (18 / 12) | 0.52 / 1.08 | -0.556 [-0.819, +0.382] | -0.001 [-0.686, +0.599] |
| RSGboth-L18-s0@18:RB0-F-s0@18 | uncovered | 235 (23 / 14) | 0.49 / 0.48 | +0.012 [-0.052, +0.114] | +0.000 [+0.000, +0.003] |
| RSGboth-L18-s0@18:RB0-F-s0@18 | discordant (left out) | 18 of 322 | | | |
| RSGboth-L18-s1@18:RB0-F-s0@18 | covered | 70 (16 / 11) | 0.64 / 1.04 | -0.399 [-0.683, +0.410] | -0.028 [-0.398, +0.482] |
| RSGboth-L18-s1@18:RB0-F-s0@18 | uncovered | 236 (23 / 14) | 0.51 / 0.48 | +0.028 [-0.048, +0.154] | +0.000 [+0.000, +0.002] |
| RSGboth-L18-s1@18:RB0-F-s0@18 | discordant (left out) | 18 of 324 | | | |
