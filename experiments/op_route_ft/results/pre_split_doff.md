# Failure split of the 25 B2D junction turns (rft_split.py)

Window = entry .. exit of the turn (<= 15 s). s = desired curvature act_k signed to the commanded side (+ = towards the exit side; act_k is positive to the right: shipped took the left turn 28008 0 with k -0.16, right turns lost with k +0.1..+0.18). Causes, first match: not entered; `wrong side` (peak s < 0.5 / R_min and the opposite sign reaches -0.5 / R_min); `not chosen` (s never reaches 0.5 / R_min between 6 s before entry and the end of the span); `late` (it does, but the car is already > 3 m of arc past the turn start); `in time, crawl / stop` (steered by the turn start, but stopped > 25% of the speed window or median speed < 1.5 m/s there); `in time, moving` (steered by the turn start, moving, still lost). Speed window = first 15 s after entry; steering is read over the whole span (to the exit or 25 m off the route).

## Cause counts (turns)

| arm | group | n | took | not chosen | wrong side | late | in time, crawl / stop | in time, moving | never entered |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped@doff | choice | 13 | 1 | 7 | 0 | 3 | 0 | 0 | 2 |
| shipped@doff | forced | 11 | 2 | 1 | 3 | 1 | 3 | 0 | 1 |
| shipped@doff | all | 24 | 3 | 8 | 3 | 4 | 3 | 0 | 3 |
| rc-ctl-s0@doff | choice | 13 | 3 | 0 | 0 | 7 | 1 | 0 | 2 |
| rc-ctl-s0@doff | forced | 12 | 2 | 0 | 1 | 1 | 6 | 0 | 2 |
| rc-ctl-s0@doff | all | 25 | 5 | 0 | 1 | 8 | 7 | 0 | 4 |
| rc-bear-s0@doff | choice | 13 | 1 | 0 | 0 | 9 | 2 | 0 | 1 |
| rc-bear-s0@doff | forced | 12 | 5 | 0 | 0 | 2 | 2 | 1 | 2 |
| rc-bear-s0@doff | all | 25 | 6 | 0 | 0 | 11 | 4 | 1 | 3 |
| rc-ctl-pre-s0@doff | choice | 13 | 2 | 6 | 0 | 2 | 0 | 0 | 3 |
| rc-ctl-pre-s0@doff | forced | 12 | 1 | 5 | 1 | 1 | 2 | 0 | 2 |
| rc-ctl-pre-s0@doff | all | 25 | 3 | 11 | 1 | 3 | 2 | 0 | 5 |
| rc-bear-pre-s0@doff | choice | 13 | 1 | 2 | 2 | 3 | 2 | 0 | 3 |
| rc-bear-pre-s0@doff | forced | 12 | 0 | 4 | 1 | 1 | 3 | 0 | 3 |
| rc-bear-pre-s0@doff | all | 25 | 1 | 6 | 3 | 4 | 5 | 0 | 6 |

## Entered turns: longitudinal and steering medians

| arm | entered | v at entry (m/s) | median v in window | min v | stopped share of window | stopped s | peak s over span (1/m) | peak s / needed | turn-in arc m after turn start (median; n steered) | peak |k| towards the commanded side (choice / forced) | desire pulse matches side |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped@doff | 21 | 2.0 | 0.9 | 0.0 | 0.42 | 6.3 | 0.038 | 0.48 | 3.0 (10) | 6 / 11 / 7 / 10 | 0.00 |
| rc-ctl-s0@doff | 21 | 2.0 | 1.1 | 0.0 | 0.40 | 6.0 | 0.151 | 1.53 | 2.9 (20) | 9 / 11 / 9 / 10 | 0.00 |
| rc-bear-s0@doff | 22 | 3.5 | 1.6 | 0.0 | 0.37 | 5.6 | 0.168 | 2.27 | 4.5 (22) | 9 / 12 / 9 / 10 | 0.00 |
| rc-ctl-pre-s0@doff | 20 | 2.3 | 0.5 | 0.0 | 0.48 | 7.3 | 0.035 | 0.40 | 4.0 (7) | 5 / 10 / 6 / 10 | 0.00 |
| rc-bear-pre-s0@doff | 19 | 3.2 | 0.4 | 0.0 | 0.48 | 7.3 | 0.048 | 0.70 | 1.7 (10) | 7 / 10 / 7 / 9 | 0.00 |

## By R_min (entered turns; share lost / steered, median v in window)

| arm | R_min bin | n | took | steered (peak s >= 0.5 need) | median v in window | stopped share |
|---|---|--:|--:|--:|--:|--:|
| shipped@doff | < 7 m | 6 | 1 | 2 | 1.1 | 0.39 |
| shipped@doff | 7-10 m | 4 | 0 | 2 | 1.1 | 0.40 |
| shipped@doff | >= 10 m | 11 | 2 | 6 | 0.9 | 0.48 |
| rc-ctl-s0@doff | < 7 m | 6 | 2 | 6 | 1.7 | 0.27 |
| rc-ctl-s0@doff | 7-10 m | 4 | 1 | 3 | 0.2 | 0.48 |
| rc-ctl-s0@doff | >= 10 m | 11 | 2 | 11 | 0.7 | 0.45 |
| rc-bear-s0@doff | < 7 m | 6 | 0 | 6 | 1.4 | 0.38 |
| rc-bear-s0@doff | 7-10 m | 4 | 1 | 4 | 1.6 | 0.35 |
| rc-bear-s0@doff | >= 10 m | 12 | 5 | 12 | 1.7 | 0.39 |
| rc-ctl-pre-s0@doff | < 7 m | 6 | 1 | 1 | 0.6 | 0.45 |
| rc-ctl-pre-s0@doff | 7-10 m | 3 | 0 | 0 | 0.4 | 0.42 |
| rc-ctl-pre-s0@doff | >= 10 m | 11 | 2 | 6 | 0.6 | 0.47 |
| rc-bear-pre-s0@doff | < 7 m | 6 | 0 | 2 | 0.6 | 0.54 |
| rc-bear-pre-s0@doff | 7-10 m | 3 | 1 | 1 | 0.9 | 0.36 |
| rc-bear-pre-s0@doff | >= 10 m | 10 | 0 | 7 | 0.2 | 0.45 |

## Per turn

| route | turn | forced | kind | side | angle | R_min | shipped@doff: cause, s peak, v med, stop s, turn-in arc m | rc-ctl-s0@doff: cause, s peak, v med, stop s, turn-in arc m | rc-bear-s0@doff: cause, s peak, v med, stop s, turn-in arc m | rc-ctl-pre-s0@doff: cause, s peak, v med, stop s, turn-in arc m | rc-bear-pre-s0@doff: cause, s peak, v med, stop s, turn-in arc m |
|---|--:|--:|---|---|--:|--:|---|---|---|---|---|
| 10255 | 0 | 0 | junction | R | 89 | 6.4 | not chosen, 0.02, 1.8, 4.8, None | late, 0.15, 1.9, 4.5, 9.8 | late, 0.14, 1.6, 3.3, 3.8 | not chosen, 0.01, 0.4, 7.5, None | wrong side, 0.03, 0.4, 7.1, None |
| 15102 | 0 | 0 | junction | L | -80 | 11.5 | not chosen, 0.02, 2.4, 3.4, None | late, 0.16, 2.5, 2.2, 17.2 | late, 0.10, 2.4, 5.4, 8.5 | not chosen, 0.03, 2.2, 3.9, None | late, 0.19, 1.6, 4.7, 18.5 |
| 24416 | 0 | 1 | curve | R | 51 | 36.6 | in time, crawl / stop, 0.09, 0.6, 7.1, -2.5 | took, 0.10, 1.2, 6.1, -2.5 | took, 0.17, 1.7, 5.7, -2.5 | in time, crawl / stop, 0.04, 0.2, 7.7, -2.5 | in time, crawl / stop, 0.05, 0.0, 8.8, -2.5 |
| 24758 | 0 | 1 | curve | L | -79 | 33.3 | wrong side, 0.00, 0.0, 10.0, None | late, 0.05, 0.0, 10.3, 30.2 | took, 0.08, 0.0, 10.1, 31.2 | late, 0.03, 0.0, 9.4, 13.8 | wrong side, 0.01, 0.0, 8.1, None |
| 24758 | 1 | 0 | curve | R | 35 | 31.2 | never entered | never entered | took, 0.05, 4.3, 0.0, 6.2 | never entered | never entered |
| 24944 | 0 | 1 | curve | L | -89 | 10.7 | not chosen, 0.02, 0.0, 9.6, None | in time, crawl / stop, 0.06, 0.1, 8.8, 1.5 | in time, moving, 0.17, 1.7, 3.4, 1.2 | not chosen, 0.04, 0.1, 9.2, None | not chosen, 0.03, 0.0, 8.6, None |
| 24944 | 1 | 0 | junction | L | -89 | 8.7 | never entered | never entered | never entered | never entered | never entered |
| 25051 | 0 | 1 | junction | R | 90 | 6.2 | wrong side, 0.08, 0.6, 5.8, None | in time, crawl / stop, 0.20, 1.1, 5.0, -0.2 | late, 0.69, 1.1, 7.0, 4.8 | not chosen, 0.01, 0.7, 5.1, None | not chosen, 0.03, 1.1, 4.0, None |
| 26153 | 0 | 1 | curve | R | 37 | 41.0 | took, 0.07, 1.6, 5.8, -6.5 | took, 0.06, 0.0, 7.9, -3.8 | took, 0.05, 0.0, 7.8, -4.2 | took, 0.05, 1.9, 4.5, -2.2 | late, 0.02, 0.3, 7.5, 11.2 |
| 26153 | 1 | 1 | curve | L | -112 | 15.4 | in time, crawl / stop, 0.05, 0.0, 13.2, -2.2 | in time, crawl / stop, 0.19, 2.0, 4.1, 2.5 | in time, crawl / stop, 0.47, 0.0, 14.8, -4.0 | not chosen, 0.03, 0.3, 7.6, None | never entered |
| 26365 | 0 | 1 | curve | R | 85 | 42.5 | - | never entered | never entered | never entered | never entered |
| 26723 | 0 | 1 | curve | R | 69 | 12.0 | late, 0.04, 0.0, 9.5, 3.8 | in time, crawl / stop, 0.14, 0.0, 11.4, 1.2 | late, 0.29, 0.1, 7.9, 3.2 | not chosen, 0.02, 0.0, 11.5, None | not chosen, 0.03, 0.2, 7.6, None |
| 26723 | 1 | 1 | curve | R | 82 | 11.3 | never entered | never entered | never entered | never entered | never entered |
| 26872 | 0 | 0 | junction | L | -61 | 12.6 | not chosen, 0.01, 1.0, 3.1, None | late, 0.04, 1.0, 4.2, 3.2 | late, 0.14, 1.1, 1.8, 10.5 | took, 0.04, 0.8, 5.3, None | in time, crawl / stop, 0.14, 0.0, 8.5, 2.2 |
| 27297 | 0 | 0 | junction | R | 88 | 6.6 | not chosen, 0.03, 1.3, 6.5, None | took, 0.11, 2.2, 3.9, -2.0 | in time, crawl / stop, 0.20, 2.0, 4.0, 2.8 | not chosen, 0.04, 1.3, 5.6, None | wrong side, 0.04, 2.0, 3.5, None |
| 27994 | 0 | 1 | junction | L | -101 | 7.0 | wrong side, 0.02, 0.9, 6.9, None | wrong side, 0.00, 0.2, 8.3, None | took, 1.23, 1.5, 5.7, 5.8 | not chosen, 0.03, 1.9, 4.1, None | not chosen, 0.02, 2.1, 1.3, None |
| 28008 | 0 | 1 | curve | L | -53 | 33.7 | took, 0.09, 3.0, 5.7, -2.5 | in time, crawl / stop, 0.20, 0.7, 6.6, -2.5 | took, 0.12, 1.8, 6.5, -2.5 | in time, crawl / stop, 0.04, 0.7, 7.1, -2.5 | in time, crawl / stop, 0.10, 1.9, 5.7, -2.5 |
| 28008 | 1 | 0 | junction | L | -90 | 8.1 | not chosen, 0.00, 2.0, 3.4, None | took, 0.23, 0.1, 8.7, 5.5 | late, 0.14, 1.6, 4.8, 4.5 | never entered | never entered |
| 28147 | 0 | 0 | junction | L | -88 | 15.2 | not chosen, 0.02, 1.3, 5.9, None | late, 0.05, 1.5, 6.0, 4.5 | late, 0.61, 2.1, 3.4, 20.5 | late, 0.04, 1.4, 5.7, 6.8 | late, 0.05, 2.4, 2.3, 27.2 |
| 28180 | 0 | 1 | junction | R | 87 | 4.7 | in time, crawl / stop, 0.80, 0.9, 6.3, 2.5 | in time, crawl / stop, 0.12, 1.9, 4.1, -1.0 | in time, crawl / stop, 0.80, 0.3, 7.5, 2.0 | wrong side, 0.03, 0.1, 8.9, None | in time, crawl / stop, 0.22, 0.0, 13.5, -0.5 |
| 334 | 0 | 0 | junction | L | -88 | 7.4 | late, 0.24, 0.5, 7.4, 4.2 | in time, crawl / stop, 0.98, 0.2, 7.8, -2.2 | in time, crawl / stop, 0.31, 1.1, 6.7, 1.0 | not chosen, 0.05, 0.4, 7.5, None | took, 0.46, 0.2, 8.0, 1.2 |
| 34183 | 0 | 0 | junction | L | -89 | 13.4 | late, 0.04, 0.9, 6.6, 12.5 | late, 0.86, 0.6, 6.6, 3.8 | late, 0.41, 2.0, 4.1, 4.8 | late, 0.04, 0.6, 6.9, 5.5 | late, 0.06, 1.6, 6.6, 11.8 |
| 5423 | 0 | 0 | junction | R | 90 | 6.2 | took, 0.89, 1.6, 5.6, 3.5 | took, 1.13, 1.3, 3.2, 3.2 | late, 0.13, 1.2, 6.9, 5.8 | took, 0.18, 0.9, 5.6, 4.0 | not chosen, 0.03, 0.8, 7.3, None |
| 6999 | 0 | 0 | junction | L | -89 | 6.1 | not chosen, 0.02, 0.6, 6.3, None | late, 0.18, 1.5, 3.4, 7.0 | late, 0.17, 1.7, 5.4, 5.2 | not chosen, 0.02, 0.3, 7.6, None | in time, crawl / stop, 0.21, 0.0, 13.3, -0.5 |
| 9196 | 0 | 0 | junction | L | -89 | 8.1 | late, 0.39, 1.3, 6.0, 6.8 | late, 0.16, 1.6, 4.5, 3.2 | late, 0.17, 1.9, 3.7, 4.5 | not chosen, 0.03, 0.2, 7.6, None | not chosen, 0.05, 0.9, 7.2, None |
