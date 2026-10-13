# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| R-own@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.27 (+0.48) | 0.999 |
| R-log-priv@18 (served) | 452 / 24 | 91.7 % (24) [77, 100] | 87.5 % (16) [67, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [76, 100] | 22 / 1 / 1 / 0 | 0.54 / 1.99 (333) | +0.03 (+1.03) | 0.999 |
| R-bridge@18 (served) | 452 / 24 | 45.8 % (24) [27, 68] | 50.0 % (16) [27, 71] | 37.5 % (8) [10, 71] | 0.0 % (1) [0, 0] | 47.8 % (23) [27, 71] | 11 / 11 / 0 / 2 | 0.84 / 2.67 (247) | -0.37 (+0.18) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| R-own@18:RSG-L18-s0@18 | all | +0.0 [+0.0, +0.0] | 24 / 24 |
| R-own@18:RSG-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| R-own@18:RSG-L18-s0@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| R-own@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| R-own@18:RSG-L18-s0@18 | uncovered | +0.0 [+0.0, +0.0] | 23 / 23 |
| R-bridge@18:RSG-L18-s0@18 | all | -20.8 [-44.4, +4.5] | 24 / 24 |
| R-bridge@18:RSG-L18-s0@18 | left | -18.8 [-41.7, +7.7] | 16 / 16 |
| R-bridge@18:RSG-L18-s0@18 | right | -25.0 [-77.8, +25.0] | 8 / 8 |
| R-bridge@18:RSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| R-bridge@18:RSG-L18-s0@18 | uncovered | -17.4 [-43.5, +8.3] | 23 / 23 |
| R-log-priv@18:RSG-L18-s0@18 | all | +25.0 [+0.0, +47.8] | 24 / 24 |
| R-log-priv@18:RSG-L18-s0@18 | left | +18.8 [-7.1, +45.5] | 16 / 16 |
| R-log-priv@18:RSG-L18-s0@18 | right | +37.5 [+0.0, +77.8] | 8 / 8 |
| R-log-priv@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| R-log-priv@18:RSG-L18-s0@18 | uncovered | +26.1 [+0.0, +50.0] | 23 / 23 |
| R-bridge@18:R-own@18 | all | -20.8 [-44.4, +4.5] | 24 / 24 |
| R-bridge@18:R-own@18 | left | -18.8 [-41.7, +7.7] | 16 / 16 |
| R-bridge@18:R-own@18 | right | -25.0 [-77.8, +25.0] | 8 / 8 |
| R-bridge@18:R-own@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| R-bridge@18:R-own@18 | uncovered | -17.4 [-43.5, +8.3] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| R-own@18:RSG-L18-s0@18 | covered | 100 (23 / 13) | 0.92 / 0.92 | +0.005 [-0.000, +0.015] | +0.000 [+0.000, +0.000] |
| R-own@18:RSG-L18-s0@18 | uncovered | 253 (23 / 14) | 0.48 / 0.48 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| R-own@18:RSG-L18-s0@18 | discordant (left out) | 0 of 353 | | | |
| R-bridge@18:RSG-L18-s0@18 | covered | 43 (9 / 6) | 1.01 / 1.05 | -0.044 [-1.400, +0.786] | +0.000 [-1.123, +0.693] |
| R-bridge@18:RSG-L18-s0@18 | uncovered | 182 (23 / 14) | 0.72 / 0.44 | +0.283 [+0.119, +0.510] | +0.070 [+0.001, +0.232] |
| R-bridge@18:RSG-L18-s0@18 | discordant (left out) | 18 of 243 | | | |
| R-log-priv@18:RSG-L18-s0@18 | covered | 71 (19 / 11) | 0.78 / 0.82 | -0.038 [-0.431, +0.247] | +0.078 [-0.082, +0.235] |
| R-log-priv@18:RSG-L18-s0@18 | uncovered | 237 (23 / 14) | 0.48 / 0.47 | +0.012 [-0.008, +0.090] | +0.008 [+0.001, +0.025] |
| R-log-priv@18:RSG-L18-s0@18 | discordant (left out) | 17 of 325 | | | |
| R-bridge@18:R-own@18 | covered | 43 (9 / 6) | 1.01 / 1.05 | -0.044 [-1.400, +0.786] | +0.000 [-1.123, +0.693] |
| R-bridge@18:R-own@18 | uncovered | 182 (23 / 14) | 0.72 / 0.44 | +0.283 [+0.119, +0.510] | +0.070 [+0.001, +0.232] |
| R-bridge@18:R-own@18 | discordant (left out) | 18 of 243 | | | |
