# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RB0-F-s0@18 (served) | 1021 / 64 | 28.1 % (64) [17, 40] | 30.0 % (40) [17, 45] | 25.0 % (24) [9, 42] | 0.0 % (3) [0, 0] | 29.5 % (61) [18, 41] | 18 / 31 / 9 / 6 | 0.71 / 2.62 (881) | -0.32 (-0.10) | 0.993 |
| YR10m10-F-s0@18 (offline) | 1021 / 64 | 25.0 % (64) [15, 35] | 32.5 % (40) [19, 47] | 12.5 % (24) [0, 26] | 0.0 % (3) [0, 0] | 26.2 % (61) [16, 37] | 16 / 26 / 7 / 15 | 0.63 / 2.40 (912) | -0.32 (-0.10) | 0.993 |
| YS-L18-s0@18 (offline) | 1021 / 64 | 31.2 % (64) [20, 43] | 37.5 % (40) [24, 53] | 20.8 % (24) [5, 39] | 0.0 % (3) [0, 0] | 32.8 % (61) [21, 44] | 20 / 20 / 9 / 15 | 0.65 / 2.68 (950) | -0.33 (-0.12) | 0.993 |
| YS-L18-s1@18 (offline) | 1021 / 64 | 26.6 % (64) [16, 38] | 37.5 % (40) [23, 54] | 8.3 % (24) [0, 22] | 0.0 % (3) [0, 0] | 27.9 % (61) [17, 39] | 17 / 26 / 7 / 14 | 0.69 / 2.75 (929) | -0.31 (-0.13) | 0.993 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RB0-F-s0@18:YR10m10-F-s0@18 | all | +3.1 [-6.2, +13.3] | 64 / 64 |
| RB0-F-s0@18:YR10m10-F-s0@18 | left | -2.5 [-13.5, +8.3] | 40 / 40 |
| RB0-F-s0@18:YR10m10-F-s0@18 | right | +12.5 [+0.0, +26.1] | 24 / 24 |
| RB0-F-s0@18:YR10m10-F-s0@18 | covered | +0.0 [+0.0, +0.0] | 3 / 3 |
| RB0-F-s0@18:YR10m10-F-s0@18 | uncovered | +3.3 [-6.6, +13.8] | 61 / 61 |
| RB0-F-s0@18:YS-L18-s0@18 | all | -3.1 [-15.1, +8.1] | 64 / 64 |
| RB0-F-s0@18:YS-L18-s0@18 | left | -7.5 [-19.4, +5.4] | 40 / 40 |
| RB0-F-s0@18:YS-L18-s0@18 | right | +4.2 [-18.2, +24.0] | 24 / 24 |
| RB0-F-s0@18:YS-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 3 / 3 |
| RB0-F-s0@18:YS-L18-s0@18 | uncovered | -3.3 [-15.6, +8.6] | 61 / 61 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RB0-F-s0@18:YR10m10-F-s0@18 | covered | 309 (45 / 28) | 1.07 / 0.78 | +0.287 [-0.189, +0.705] | +0.090 [-0.001, +0.248] |
| RB0-F-s0@18:YR10m10-F-s0@18 | uncovered | 520 (59 / 39) | 0.54 / 0.54 | -0.000 [-0.059, +0.058] | +0.000 [+0.000, +0.000] |
| RB0-F-s0@18:YR10m10-F-s0@18 | discordant (left out) | 15 of 844 | | | |
| RB0-F-s0@18:YS-L18-s0@18 | covered | 309 (45 / 29) | 0.97 / 1.12 | -0.153 [-0.568, +0.377] | -0.005 [-0.173, +0.150] |
| RB0-F-s0@18:YS-L18-s0@18 | uncovered | 510 (59 / 39) | 0.54 / 0.51 | +0.034 [-0.027, +0.107] | +0.000 [+0.000, +0.000] |
| RB0-F-s0@18:YS-L18-s0@18 | discordant (left out) | 24 of 843 | | | |
