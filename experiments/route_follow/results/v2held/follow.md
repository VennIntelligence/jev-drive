# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RV2a-L18-s0@18 (served) | 452 / 24 | 75.0 % (24) [61, 90] | 75.0 % (16) [50, 93] | 75.0 % (8) [43, 100] | - | 75.0 % (24) [61, 90] | 18 / 1 / 1 / 4 | 0.53 / 1.94 (348) | -0.27 (+0.44) | 0.999 |
| RV2a-L18-s1@18 (served) | 452 / 24 | 62.5 % (24) [40, 82] | 68.8 % (16) [42, 88] | 50.0 % (8) [11, 89] | - | 62.5 % (24) [40, 82] | 15 / 2 / 0 / 7 | 0.53 / 1.88 (353) | -0.26 (+0.49) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RV2a-L18-s0@18:RSG-L18-s0@18 | all | +8.3 [+0.0, +20.8] | 24 / 24 |
| RV2a-L18-s0@18:RSG-L18-s0@18 | left | +6.2 [+0.0, +23.1] | 16 / 16 |
| RV2a-L18-s0@18:RSG-L18-s0@18 | right | +12.5 [+0.0, +33.3] | 8 / 8 |
| RV2a-L18-s0@18:RSG-L18-s0@18 | covered | +nan [+nan, +nan] | 0 / 1 |
| RV2a-L18-s0@18:RSG-L18-s0@18 | uncovered | +9.8 [+0.0, +22.2] | 24 / 23 |
| RV2a-L18-s1@18:RSG-L18-s1@18 | all | +4.2 [+0.0, +11.1] | 24 / 24 |
| RV2a-L18-s1@18:RSG-L18-s1@18 | left | +6.2 [+0.0, +14.3] | 16 / 16 |
| RV2a-L18-s1@18:RSG-L18-s1@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RV2a-L18-s1@18:RSG-L18-s1@18 | covered | +nan [+nan, +nan] | 0 / 1 |
| RV2a-L18-s1@18:RSG-L18-s1@18 | uncovered | +1.6 [+0.0, +4.5] | 24 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RV2a-L18-s0@18:RSG-L18-s0@18 | covered | 79 (21 / 12) | 0.99 / 1.11 | -0.117 [-0.261, +0.265] | +0.034 [-0.157, +0.199] |
| RV2a-L18-s0@18:RSG-L18-s0@18 | uncovered | 247 (23 / 14) | 0.48 / 0.48 | +0.001 [-0.027, +0.037] | +0.000 [+0.000, +0.000] |
| RV2a-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 22 of 348 | | | |
| RV2a-L18-s1@18:RSG-L18-s1@18 | covered | 81 (21 / 12) | 0.96 / 1.02 | -0.052 [-0.460, +0.166] | -0.057 [-0.248, +0.070] |
| RV2a-L18-s1@18:RSG-L18-s1@18 | uncovered | 254 (23 / 14) | 0.51 / 0.51 | -0.003 [-0.033, +0.051] | +0.000 [-0.000, +0.000] |
| RV2a-L18-s1@18:RSG-L18-s1@18 | discordant (left out) | 18 of 353 | | | |
