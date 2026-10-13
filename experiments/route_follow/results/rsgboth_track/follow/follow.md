# ROUTE-FOLLOW closed-loop read (rf_follow.py)

Turn = route net heading change >= 60 deg within 30 m inside the log's own travel to decision 17; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. Covered = the memory's +20 m point valid at the decision 20 m before the entry. Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).

| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|
| RSGboth-L18-s0-rtgate@18 (served) | 452 / 24 | 79.2 % (24) [65, 95] | 81.2 % (16) [63, 100] | 75.0 % (8) [43, 100] | 0.0 % (1) [0, 0] | 82.6 % (23) [70, 95] | 19 / 2 / 1 / 2 | 0.50 / 1.93 (336) | +0.13 (+0.88) | 0.998 |
| RSGboth-L18-s0-rtgatep@18 (served) | 452 / 24 | 87.5 % (24) [76, 100] | 87.5 % (16) [73, 100] | 87.5 % (8) [67, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [75, 100] | 21 / 1 / 1 / 1 | 0.50 / 2.10 (335) | +0.13 (+0.89) | 0.998 |
| RSGboth-L18-s0-rttimep@18 (served) | 452 / 24 | 87.5 % (24) [76, 100] | 87.5 % (16) [73, 100] | 87.5 % (8) [67, 100] | 100.0 % (1) [100, 100] | 87.0 % (23) [75, 100] | 21 / 1 / 1 / 1 | 0.52 / 2.03 (334) | +0.13 (+0.87) | 0.998 |
| RSG-L18-s0-rtgatep@18 (served) | 452 / 24 | 58.3 % (24) [36, 78] | 62.5 % (16) [33, 83] | 50.0 % (8) [11, 89] | 100.0 % (1) [100, 100] | 56.5 % (23) [35, 77] | 14 / 1 / 1 / 8 | 0.51 / 1.83 (364) | -0.27 (+0.48) | 0.999 |
| RSGboth-L18-s0@18 (served) | 452 / 24 | 91.7 % (24) [80, 100] | 87.5 % (16) [73, 100] | 100.0 % (8) [100, 100] | 100.0 % (1) [100, 100] | 91.3 % (23) [79, 100] | 22 / 1 / 0 / 1 | 0.52 / 1.99 (327) | +0.12 (+0.86) | 0.999 |
| RSG-L18-s0@18 (served) | 452 / 24 | 66.7 % (24) [45, 86] | 68.8 % (16) [42, 88] | 62.5 % (8) [22, 100] | 100.0 % (1) [100, 100] | 65.2 % (23) [44, 86] | 16 / 1 / 0 / 7 | 0.52 / 1.90 (353) | -0.26 (+0.48) | 0.999 |

| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |
|:--|:--|:--|:--|
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | all | -12.5 [-20.7, +0.0] | 24 / 24 |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | left | -6.2 [-14.3, +0.0] | 16 / 16 |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | right | -25.0 [-57.1, +0.0] | 8 / 8 |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | uncovered | -8.7 [-18.2, +0.0] | 23 / 23 |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | all | +12.5 [-5.6, +33.3] | 24 / 24 |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | left | +12.5 [-8.7, +46.2] | 16 / 16 |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | right | +12.5 [+0.0, +33.3] | 8 / 8 |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | covered | -100.0 [-100.0, -100.0] | 1 / 1 |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | uncovered | +17.4 [+0.0, +34.8] | 23 / 23 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | all | -4.2 [-12.5, +0.0] | 24 / 24 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | right | -12.5 [-33.3, +0.0] | 8 / 8 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | uncovered | -4.3 [-12.5, +0.0] | 23 / 23 |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | all | +20.8 [+4.8, +37.0] | 24 / 24 |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | right | +25.0 [+0.0, +57.1] | 8 / 8 |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | uncovered | +21.7 [+5.3, +37.5] | 23 / 23 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | all | -4.2 [-12.5, +0.0] | 24 / 24 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | right | -12.5 [-33.3, +0.0] | 8 / 8 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | uncovered | -4.3 [-12.5, +0.0] | 23 / 23 |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | all | +20.8 [+4.8, +37.0] | 24 / 24 |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | left | +18.8 [+0.0, +46.2] | 16 / 16 |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | right | +25.0 [+0.0, +57.1] | 8 / 8 |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | uncovered | +21.7 [+5.3, +37.5] | 23 / 23 |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | all | -33.3 [-58.3, -12.5] | 24 / 24 |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | left | -25.0 [-54.5, -5.3] | 16 / 16 |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | right | -50.0 [-88.9, -11.1] | 8 / 8 |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | uncovered | -34.8 [-59.1, -13.6] | 23 / 23 |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | all | -8.3 [-23.8, +0.0] | 24 / 24 |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | left | -6.2 [-23.1, +0.0] | 16 / 16 |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | right | -12.5 [-42.9, +0.0] | 8 / 8 |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | uncovered | -8.7 [-25.0, +0.0] | 23 / 23 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | all | +8.3 [+0.0, +16.7] | 24 / 24 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | left | +6.2 [+0.0, +14.3] | 16 / 16 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | right | +12.5 [+0.0, +42.9] | 8 / 8 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | covered | +100.0 [+100.0, +100.0] | 1 / 1 |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | uncovered | +4.3 [+0.0, +13.0] | 23 / 23 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | all | +0.0 [+0.0, +0.0] | 24 / 24 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | left | +0.0 [+0.0, +0.0] | 16 / 16 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | right | +0.0 [+0.0, +0.0] | 8 / 8 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | covered | +0.0 [+0.0, +0.0] | 1 / 1 |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | uncovered | +0.0 [+0.0, +0.0] | 23 / 23 |

Per-decision L5 (Amendment 1): junction-stretch decisions paired on (scene, k), |lateral to the route| (m), A - B difference of medians [log-clustered paired 95 % CI], median of paired differences beside it:

| pair A - B | half | pairs (scenes / logs) | median A / B | A - B medians [CI] | paired median [CI] |
|:--|:--|:--|:--|:--|:--|
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | covered | 78 (23 / 13) | 0.55 / 0.61 | -0.063 [-0.418, +0.116] | -0.000 [-0.026, +0.058] |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | uncovered | 239 (23 / 14) | 0.48 / 0.50 | -0.021 [-0.121, +0.012] | +0.000 [-0.002, +0.000] |
| RSGboth-L18-s0-rtgate@18:RSGboth-L18-s0@18 | discordant (left out) | 6 of 323 | | | |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | covered | 69 (21 / 12) | 0.49 / 0.82 | -0.330 [-0.723, +0.221] | -0.152 [-0.542, +0.233] |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | uncovered | 238 (23 / 14) | 0.48 / 0.48 | +0.008 [-0.069, +0.077] | +0.000 [-0.000, +0.002] |
| RSGboth-L18-s0-rtgate@18:RSG-L18-s0@18 | discordant (left out) | 22 of 329 | | | |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | covered | 79 (23 / 13) | 0.59 / 0.57 | +0.026 [-0.493, +0.684] | +0.007 [-0.081, +0.080] |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | uncovered | 239 (23 / 14) | 0.48 / 0.50 | -0.021 [-0.121, +0.019] | +0.000 [-0.001, +0.000] |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | discordant (left out) | 6 of 324 | | | |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | covered | 69 (21 / 12) | 0.50 / 0.87 | -0.372 [-0.795, +0.720] | -0.223 [-0.480, +0.242] |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | uncovered | 238 (23 / 14) | 0.48 / 0.48 | +0.008 [-0.068, +0.080] | +0.000 [-0.000, +0.002] |
| RSGboth-L18-s0-rtgatep@18:RSG-L18-s0@18 | discordant (left out) | 22 of 329 | | | |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | covered | 84 (23 / 13) | 0.71 / 0.61 | +0.096 [-0.012, +0.257] | -0.001 [-0.009, +0.034] |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | uncovered | 238 (23 / 14) | 0.48 / 0.50 | -0.014 [-0.071, +0.016] | +0.000 [-0.000, +0.000] |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0@18 | discordant (left out) | 5 of 327 | | | |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | covered | 72 (21 / 12) | 0.66 / 0.85 | -0.193 [-0.701, +0.567] | -0.060 [-0.268, +0.161] |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | uncovered | 237 (23 / 14) | 0.48 / 0.47 | +0.012 [-0.040, +0.093] | +0.000 [+0.000, +0.003] |
| RSGboth-L18-s0-rttimep@18:RSG-L18-s0@18 | discordant (left out) | 21 of 330 | | | |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | covered | 68 (20 / 12) | 0.74 / 0.50 | +0.244 [-0.628, +0.728] | +0.061 [-0.378, +0.401] |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | uncovered | 239 (23 / 14) | 0.47 / 0.50 | -0.030 [-0.139, +0.034] | +0.000 [-0.004, +0.000] |
| RSG-L18-s0-rtgatep@18:RSGboth-L18-s0@18 | discordant (left out) | 20 of 327 | | | |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | covered | 93 (21 / 12) | 0.75 / 0.97 | -0.221 [-0.533, +0.163] | -0.003 [-0.136, +0.056] |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | uncovered | 253 (23 / 14) | 0.47 / 0.48 | -0.011 [-0.055, +0.028] | +0.000 [-0.000, +0.000] |
| RSG-L18-s0-rtgatep@18:RSG-L18-s0@18 | discordant (left out) | 7 of 353 | | | |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | covered | 87 (23 / 13) | 0.59 / 0.60 | -0.004 [-0.129, +0.470] | +0.000 [-0.014, +0.000] |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | uncovered | 246 (23 / 14) | 0.48 / 0.48 | +0.002 [-0.009, +0.032] | +0.000 [+0.000, +0.000] |
| RSGboth-L18-s0-rtgatep@18:RSGboth-L18-s0-rtgate@18 | discordant (left out) | 0 of 333 | | | |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | covered | 85 (23 / 13) | 0.70 / 0.59 | +0.110 [-0.445, +0.571] | +0.000 [+0.000, +0.003] |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | uncovered | 243 (23 / 14) | 0.48 / 0.48 | +0.003 [-0.024, +0.073] | +0.000 [+0.000, +0.000] |
| RSGboth-L18-s0-rttimep@18:RSGboth-L18-s0-rtgatep@18 | discordant (left out) | 3 of 331 | | | |
