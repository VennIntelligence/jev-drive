# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGbothSB-L18-s0@18 (served) | 452 / 24 | 75.0 % (24) [58, 91] | 87.5 % (16) [73, 100] | 50.0 % (8) [11, 89] | 100.0 % (1) [100, 100] | 73.9 % (23) [57, 91] | 18 / 3 / 1 / 2 | 0.51 / 1.90 (335) | +0.11 (+0.84) | 0.998 |
| RSGbothSB-L18-s1@18 (served) | 452 / 24 | 87.5 % (24) [71, 100] | 81.2 % (16) [54, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [71, 100] | 21 / 1 / 1 / 1 | 0.51 / 1.89 (337) | +0.11 (+0.85) | 0.999 |
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSGboth-L18-s1@18 (served) | 452 / 24 | 87.5 % (24) [71, 100] | 81.2 % (16) [54, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [71, 100] | 21 / 1 / 1 / 1 | 0.54 / 2.02 (330) | +0.11 (+0.94) | 0.998 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |
| RSG-L18-s1@18 (served) | 452 / 24 | 58.3 % (24) [38, 79] | 62.5 % (16) [38, 86] | 50.0 % (8) [11, 89] | 0.0 % (1) [0, 0] | 60.9 % (23) [39, 81] | 14 / 1 / 0 / 9 | 0.54 / 2.07 (361) | -0.28 (+0.39) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | all | +8.3 [-8.7, +22.7] | 24 / 24 |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | right | -12.5 [-42.9, +0.0] | 8 / 8 |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | uncovered | +8.7 [-9.1, +23.1] | 23 / 23 |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | all | -16.7 [-34.6, +0.0] | 24 / 24 |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | right | -50.0 [-88.9, -11.1] | 8 / 8 |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | uncovered | -17.4 [-34.8, +0.0] | 23 / 23 |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | all | +29.2 [+5.9, +51.6] | 24 / 24 |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | left | +18.8 [+0.0, +36.4] | 16 / 16 |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | right | +50.0 [+11.1, +88.9] | 8 / 8 |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | uncovered | +26.1 [+5.0, +50.0] | 23 / 23 |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | all | +0.0 [+0.0, +0.0] | 24 / 24 |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | uncovered | +0.0 [+0.0, +0.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | covered | 77 (21 / 12) | 0.88 / 0.97 | -0.092 [-0.582, +0.447] | -0.071 [-0.202, +0.184] |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | uncovered | 238 (23 / 14) | 0.48 / 0.47 | +0.006 [-0.043, +0.049] | +0.000 [-0.000, +0.000] |
| RSGbothSB-L18-s0@18:RSG-L18-s0@18 | discordant (left out) | 17 of 332 | | | |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | covered | 82 (21 / 12) | 0.84 / 0.56 | +0.280 [-0.105, +0.481] | -0.008 [-0.134, +0.107] |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | uncovered | 236 (23 / 14) | 0.47 / 0.49 | -0.015 [-0.097, +0.000] | -0.000 [-0.004, +0.000] |
| RSGbothSB-L18-s0@18:RSGboth-L18-s0@18 | discordant (left out) | 7 of 325 | | | |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | covered | 77 (20 / 12) | 0.63 / 0.75 | -0.121 [-0.540, +0.647] | -0.149 [-0.242, +0.106] |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | uncovered | 242 (23 / 14) | 0.48 / 0.49 | -0.007 [-0.055, +0.021] | +0.000 [-0.001, +0.000] |
| RSGbothSB-L18-s1@18:RSG-L18-s1@18 | discordant (left out) | 16 of 335 | | | |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | covered | 84 (22 / 13) | 0.67 / 0.63 | +0.038 [-0.190, +0.267] | -0.007 [-0.189, +0.029] |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | uncovered | 239 (23 / 14) | 0.48 / 0.51 | -0.022 [-0.121, +0.005] | +0.000 [-0.002, +0.000] |
| RSGbothSB-L18-s1@18:RSGboth-L18-s1@18 | discordant (left out) | 5 of 328 | | | |
