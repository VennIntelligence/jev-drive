# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSG-L18-s0@18 (none) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | - | - | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| YR10m10-F-s0@18 (none) | 452 / 24 | 37.5 % (24) [12, 65] | 50.0 % (16) [15, 78] | 12.5 % (8) [0, 43] | - | - | 9 / 6 / 5 / 4 | 0.51 / 1.92 (333) | -0.29 (+0.45) | 0.999 |
| WAJEPA@18 (none) | 452 / 24 | 29.2 % (24) [11, 50] | 25.0 % (16) [6, 54] | 37.5 % (8) [14, 62] | - | - | 7 / 14 / 1 / 2 | 0.51 / 1.69 (283) | -0.38 (+0.36) | 0.000 |
| LTF@18 (none) | 24 / 24 | 0.0 % (24) [0, 0] | 0.0 % (16) [0, 0] | 0.0 % (8) [0, 0] | - | - | 0 / 23 / 0 / 1 | 0.49 / 2.49 (270) | +0.06 (+0.06) | 0.000 |
| DD@18 (none) | 24 / 24 | 0.0 % (24) [0, 0] | 0.0 % (16) [0, 0] | 0.0 % (8) [0, 0] | - | - | 0 / 24 / 0 / 0 | 0.49 / 2.39 (250) | +0.13 (+0.13) | 0.000 |
| G0014@18 (none) | 24 / 24 | 4.2 % (24) [0, 12] | 6.2 % (16) [0, 20] | 0.0 % (8) [0, 0] | - | - | 1 / 18 / 4 / 1 | 0.85 / 3.74 (199) | +0.71 (+0.95) | 0.000 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| WAJEPA@18:RSG-L18-s0@18 | all | -37.5 [-68.8, +0.0] | 24 / 24 |
| WAJEPA@18:RSG-L18-s0@18 | left | -43.8 [-73.3, +0.0] | 16 / 16 |
| WAJEPA@18:RSG-L18-s0@18 | right | -25.0 [-71.4, +22.2] | 8 / 8 |
| LTF@18:RSG-L18-s0@18 | all | -66.7 [-85.7, -45.5] | 24 / 24 |
| LTF@18:RSG-L18-s0@18 | left | -68.8 [-87.5, -41.7] | 16 / 16 |
| LTF@18:RSG-L18-s0@18 | right | -62.5 [-100.0, -22.2] | 8 / 8 |
| DD@18:RSG-L18-s0@18 | all | -66.7 [-85.7, -45.5] | 24 / 24 |
| DD@18:RSG-L18-s0@18 | left | -68.8 [-87.5, -41.7] | 16 / 16 |
| DD@18:RSG-L18-s0@18 | right | -62.5 [-100.0, -22.2] | 8 / 8 |
| G0014@18:RSG-L18-s0@18 | all | -62.5 [-83.3, -40.9] | 24 / 24 |
| G0014@18:RSG-L18-s0@18 | left | -62.5 [-82.4, -36.4] | 16 / 16 |
| G0014@18:RSG-L18-s0@18 | right | -62.5 [-100.0, -22.2] | 8 / 8 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| WAJEPA@18:RSG-L18-s0@18 | covered | 0 (0 / 0) | - | - | - |
| WAJEPA@18:RSG-L18-s0@18 | uncovered | 266 (24 / 14) | 0.48 / 0.48 | +0.004 [-0.147, +0.126] | +0.000 [-0.007, +0.000] |
| WAJEPA@18:RSG-L18-s0@18 | discordant (left out) | 0 of 266 | | | |
| LTF@18:RSG-L18-s0@18 | covered | 0 (0 / 0) | - | - | - |
| LTF@18:RSG-L18-s0@18 | uncovered | 264 (24 / 14) | 0.48 / 0.49 | -0.011 [-0.215, +0.184] | +0.000 [-0.016, +0.003] |
| LTF@18:RSG-L18-s0@18 | discordant (left out) | 0 of 264 | | | |
| DD@18:RSG-L18-s0@18 | covered | 0 (0 / 0) | - | - | - |
| DD@18:RSG-L18-s0@18 | uncovered | 244 (24 / 14) | 0.48 / 0.46 | +0.024 [-0.131, +0.173] | +0.000 [-0.009, +0.001] |
| DD@18:RSG-L18-s0@18 | discordant (left out) | 0 of 244 | | | |
| G0014@18:RSG-L18-s0@18 | covered | 0 (0 / 0) | - | - | - |
| G0014@18:RSG-L18-s0@18 | uncovered | 198 (24 / 14) | 0.84 / 0.46 | +0.380 [+0.126, +0.707] | +0.124 [+0.000, +0.310] |
| G0014@18:RSG-L18-s0@18 | discordant (left out) | 0 of 198 | | | |
