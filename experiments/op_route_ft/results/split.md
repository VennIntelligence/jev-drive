# Failure split of the 25 B2D junction turns (rft_split.py)

Window = entry .. exit of the turn (<= 15 s). s = desired curvature act_k signed to the commanded side (+ = towards the exit side; act_k is positive to the right: shipped took the left turn 28008 0 with k -0.16, right turns lost with k +0.1..+0.18). Causes, first match: not entered; `wrong side` (peak s < 0.5 / R_min and the opposite sign reaches -0.5 / R_min); `not chosen` (s never reaches 0.5 / R_min between 6 s before entry and the end of the span); `late` (it does, but the car is already > 3 m of arc past the turn start); `in time, crawl / stop` (steered by the turn start, but stopped > 25% of the speed window or median speed < 1.5 m/s there); `in time, moving` (steered by the turn start, moving, still lost). Speed window = first 15 s after entry; steering is read over the whole span (to the exit or 25 m off the route).

## Cause counts (turns)

| arm | group | n | took | not chosen | wrong side | late | in time, crawl / stop | in time, moving | never entered |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped | choice | 13 | 0 | 7 | 1 | 2 | 1 | 0 | 2 |
| shipped | forced | 12 | 1 | 3 | 1 | 2 | 2 | 0 | 3 |
| shipped | all | 25 | 1 | 10 | 2 | 4 | 3 | 0 | 5 |
| rc-ctl-s0 | choice | 13 | 2 | 0 | 1 | 8 | 0 | 0 | 2 |
| rc-ctl-s0 | forced | 12 | 4 | 0 | 1 | 1 | 4 | 0 | 2 |
| rc-ctl-s0 | all | 25 | 6 | 0 | 2 | 9 | 4 | 0 | 4 |
| rc-bear-s0 | choice | 13 | 1 | 4 | 0 | 4 | 3 | 0 | 1 |
| rc-bear-s0 | forced | 12 | 3 | 0 | 0 | 2 | 4 | 0 | 3 |
| rc-bear-s0 | all | 25 | 4 | 4 | 0 | 6 | 7 | 0 | 4 |
| rc-poly-s0 | choice | 13 | 2 | 4 | 1 | 4 | 0 | 1 | 1 |
| rc-poly-s0 | forced | 12 | 5 | 2 | 0 | 2 | 1 | 0 | 2 |
| rc-poly-s0 | all | 25 | 7 | 6 | 1 | 6 | 1 | 1 | 3 |
| rc-all-s0 | choice | 13 | 3 | 1 | 0 | 7 | 1 | 0 | 1 |
| rc-all-s0 | forced | 12 | 3 | 0 | 0 | 1 | 5 | 0 | 3 |
| rc-all-s0 | all | 25 | 6 | 1 | 0 | 8 | 6 | 0 | 4 |

## Entered turns: longitudinal and steering medians

| arm | entered | v at entry (m/s) | median v in window | min v | stopped share of window | stopped s | peak s over span (1/m) | peak s / needed | turn-in arc m after turn start (median; n steered) | peak |k| towards the commanded side (choice / forced) | desire pulse matches side |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped | 20 | 2.2 | 1.0 | 0.0 | 0.41 | 6.2 | 0.024 | 0.31 | 3.1 (8) | 4 / 11 / 6 / 9 | 0.94 |
| rc-ctl-s0 | 21 | 1.9 | 0.9 | 0.0 | 0.42 | 6.3 | 0.182 | 1.91 | 3.5 (19) | 9 / 11 / 9 / 10 | 0.58 |
| rc-bear-s0 | 21 | 2.8 | 1.2 | 0.0 | 0.40 | 6.0 | 0.130 | 1.34 | 2.5 (17) | 8 / 12 / 8 / 9 | 0.95 |
| rc-poly-s0 | 22 | 2.5 | 2.0 | 0.0 | 0.34 | 5.2 | 0.072 | 1.23 | 4.2 (15) | 8 / 12 / 8 / 10 | 0.54 |
| rc-all-s0 | 21 | 3.0 | 1.7 | 0.0 | 0.31 | 4.7 | 0.165 | 3.39 | 3.5 (20) | 11 / 12 / 9 / 9 | 0.70 |

## By R_min (entered turns; share lost / steered, median v in window)

| arm | R_min bin | n | took | steered (peak s >= 0.5 need) | median v in window | stopped share |
|---|---|--:|--:|--:|--:|--:|
| shipped | < 7 m | 6 | 0 | 3 | 0.8 | 0.47 |
| shipped | 7-10 m | 4 | 0 | 0 | 1.5 | 0.32 |
| shipped | >= 10 m | 10 | 1 | 5 | 0.5 | 0.50 |
| rc-ctl-s0 | < 7 m | 6 | 2 | 6 | 0.3 | 0.55 |
| rc-ctl-s0 | 7-10 m | 4 | 1 | 4 | 1.8 | 0.34 |
| rc-ctl-s0 | >= 10 m | 11 | 3 | 9 | 1.0 | 0.43 |
| rc-bear-s0 | < 7 m | 6 | 0 | 6 | 0.0 | 0.56 |
| rc-bear-s0 | 7-10 m | 4 | 0 | 3 | 1.1 | 0.40 |
| rc-bear-s0 | >= 10 m | 11 | 4 | 8 | 1.7 | 0.36 |
| rc-poly-s0 | < 7 m | 6 | 2 | 5 | 1.5 | 0.36 |
| rc-poly-s0 | 7-10 m | 4 | 0 | 2 | 2.1 | 0.21 |
| rc-poly-s0 | >= 10 m | 12 | 5 | 8 | 2.2 | 0.35 |
| rc-all-s0 | < 7 m | 6 | 0 | 6 | 0.6 | 0.48 |
| rc-all-s0 | 7-10 m | 4 | 1 | 4 | 2.0 | 0.24 |
| rc-all-s0 | >= 10 m | 11 | 5 | 10 | 1.7 | 0.34 |

## Per turn

| route | turn | forced | kind | side | angle | R_min | shipped: cause, s peak, v med, stop s, turn-in arc m | rc-ctl-s0: cause, s peak, v med, stop s, turn-in arc m | rc-bear-s0: cause, s peak, v med, stop s, turn-in arc m | rc-poly-s0: cause, s peak, v med, stop s, turn-in arc m | rc-all-s0: cause, s peak, v med, stop s, turn-in arc m |
|---|--:|--:|---|---|--:|--:|---|---|---|---|---|
| 10255 | 0 | 0 | junction | R | 89 | 6.4 | not chosen, 0.01, 1.2, 6.4, None | late, 0.18, 0.0, 11.5, 5.2 | late, 0.30, 1.8, 4.7, 10.5 | not chosen, 0.07, 0.1, 7.9, None | late, 0.12, 1.7, 4.6, 8.2 |
| 15102 | 0 | 0 | junction | L | -80 | 11.5 | wrong side, 0.01, 0.0, 10.7, None | late, 0.17, 2.3, 2.1, 17.2 | late, 0.10, 2.4, 5.5, 8.8 | late, 0.17, 2.2, 1.9, 13.8 | late, 0.08, 2.3, 3.6, 6.5 |
| 24416 | 0 | 1 | curve | R | 51 | 36.6 | in time, crawl / stop, 0.08, 0.9, 6.1, -2.5 | took, 0.10, 1.0, 6.0, -2.5 | took, 0.14, 1.7, 5.7, -2.5 | took, 0.13, 2.8, 5.7, -2.5 | took, 0.20, 0.8, 6.6, -2.5 |
| 24758 | 0 | 1 | curve | L | -79 | 33.3 | late, 0.02, 0.0, 10.1, 31.0 | wrong side, 0.01, 0.0, 10.3, None | took, 0.09, 0.0, 9.8, 32.5 | took, 0.07, 1.6, 5.7, 32.0 | took, 0.06, 0.0, 9.5, 31.5 |
| 24758 | 1 | 0 | curve | R | 35 | 31.2 | never entered | never entered | took, 0.04, 4.1, 0.0, 3.0 | took, 0.04, 4.5, 0.0, 5.8 | took, 0.03, 3.8, 0.0, 2.5 |
| 24944 | 0 | 1 | curve | L | -89 | 10.7 | not chosen, 0.01, 0.1, 9.6, None | in time, crawl / stop, 0.05, 0.1, 9.2, 2.0 | in time, crawl / stop, 0.14, 1.3, 4.1, 1.5 | in time, crawl / stop, 0.21, 1.0, 5.7, 2.2 | in time, crawl / stop, 0.17, 1.5, 6.7, 1.2 |
| 24944 | 1 | 0 | junction | L | -89 | 8.7 | never entered | never entered | never entered | never entered | never entered |
| 25051 | 0 | 1 | junction | R | 90 | 6.2 | not chosen, 0.05, 0.3, 7.6, None | late, 0.21, 0.4, 7.4, 3.8 | in time, crawl / stop, 0.29, 1.2, 5.8, 2.5 | late, 0.28, 0.1, 8.7, 5.0 | late, 0.62, 0.4, 7.4, 5.5 |
| 26153 | 0 | 1 | curve | R | 37 | 41.0 | in time, crawl / stop, 0.27, 0.0, 8.5, -1.5 | took, 0.09, 0.0, 8.1, -4.5 | in time, crawl / stop, 0.04, 0.0, 7.8, -4.2 | took, 0.05, 2.2, 4.3, 0.5 | in time, crawl / stop, 0.15, 1.0, 6.4, -4.0 |
| 26153 | 1 | 1 | curve | L | -112 | 15.4 | never entered | in time, crawl / stop, 0.19, 1.4, 5.8, 2.2 | never entered | not chosen, 0.03, 0.0, 12.7, None | never entered |
| 26365 | 0 | 1 | curve | R | 85 | 42.5 | never entered | never entered | never entered | never entered | never entered |
| 26723 | 0 | 1 | curve | R | 69 | 12.0 | not chosen, 0.03, 0.0, 9.7, None | in time, crawl / stop, 0.47, 0.0, 8.0, 1.2 | late, 0.38, 0.0, 9.5, 3.2 | late, 0.07, 2.1, 5.7, 4.2 | in time, crawl / stop, 0.38, 0.0, 9.1, 3.0 |
| 26723 | 1 | 1 | curve | R | 82 | 11.3 | never entered | never entered | never entered | never entered | never entered |
| 26872 | 0 | 0 | junction | L | -61 | 12.6 | not chosen, 0.01, 1.4, 2.5, None | late, 0.10, 1.0, 5.2, 14.5 | not chosen, 0.02, 2.4, 0.7, None | not chosen, 0.01, 2.2, 0.0, None | not chosen, 0.02, 2.4, 2.0, None |
| 27297 | 0 | 0 | junction | R | 88 | 6.6 | in time, crawl / stop, 0.10, 0.0, 8.6, 1.2 | took, 0.93, 0.2, 12.2, -0.2 | in time, crawl / stop, 0.17, 0.0, 8.8, 0.2 | took, 1.31, 1.8, 4.0, -0.8 | late, 0.13, 1.8, 4.5, 10.2 |
| 27994 | 0 | 1 | junction | L | -101 | 7.0 | wrong side, 0.02, 1.2, 6.1, None | in time, crawl / stop, 0.13, 2.0, 4.5, -3.0 | late, 0.19, 0.1, 10.1, 11.0 | not chosen, 0.07, 2.0, 4.3, None | in time, crawl / stop, 0.89, 1.5, 4.7, 1.8 |
| 28008 | 0 | 1 | curve | L | -53 | 33.7 | took, 0.16, 3.2, 5.7, -2.5 | took, 0.23, 0.9, 6.3, -2.5 | took, 0.07, 2.0, 6.0, -2.5 | took, 0.05, 2.5, 5.7, -2.5 | took, 0.11, 2.0, 6.3, -2.5 |
| 28008 | 1 | 0 | junction | L | -90 | 8.1 | not chosen, 0.01, 2.0, 2.9, None | took, 0.27, 1.5, 3.8, 6.2 | not chosen, 0.01, 2.0, 2.5, None | not chosen, 0.00, 2.2, 0.0, None | late, 0.11, 2.1, 2.5, 13.2 |
| 28147 | 0 | 0 | junction | L | -88 | 15.2 | late, 0.04, 1.2, 6.0, 23.2 | wrong side, 0.02, 1.7, 5.3, None | not chosen, 0.02, 3.1, 0.0, None | wrong side, 0.00, 2.2, 4.9, None | late, 0.12, 1.7, 3.0, 20.2 |
| 28180 | 0 | 1 | junction | R | 87 | 4.7 | late, 0.16, 0.0, 9.2, 5.0 | took, 0.51, 0.4, 6.3, -0.2 | in time, crawl / stop, 0.55, 0.0, 13.3, -0.5 | took, 0.24, 2.2, 2.5, 11.5 | in time, crawl / stop, 0.97, 0.0, 13.6, -0.5 |
| 334 | 0 | 0 | junction | L | -88 | 7.4 | not chosen, 0.01, 0.6, 5.9, None | late, 0.14, 2.1, 3.8, 8.2 | in time, crawl / stop, 1.23, 0.2, 7.7, 0.8 | late, 0.15, 2.2, 2.0, 10.8 | took, 1.21, 4.0, 1.8, 1.5 |
| 34183 | 0 | 0 | junction | L | -89 | 13.4 | not chosen, 0.02, 1.1, 6.3, None | late, 0.20, 1.3, 5.2, 9.5 | not chosen, 0.00, 0.0, 10.5, None | not chosen, 0.00, -0.0, 11.5, None | took, 1.14, 2.0, 3.2, 4.0 |
| 5423 | 0 | 0 | junction | R | 90 | 6.2 | late, 0.18, 1.2, 6.2, 5.5 | late, 0.32, 0.0, 8.5, 3.5 | in time, crawl / stop, 0.16, 0.0, 9.8, -0.2 | in time, moving, 0.38, 1.5, 3.8, -3.0 | in time, crawl / stop, 0.55, 0.4, 7.4, 0.8 |
| 6999 | 0 | 0 | junction | L | -89 | 6.1 | not chosen, 0.02, 1.8, 4.8, None | late, 0.18, 1.4, 4.2, 6.8 | late, 0.11, 0.0, 8.9, 4.5 | late, 0.38, 1.5, 5.4, 5.0 | late, 0.60, 0.7, 6.0, 7.0 |
| 9196 | 0 | 0 | junction | L | -89 | 8.1 | not chosen, 0.00, 1.8, 4.5, None | late, 0.12, 0.0, 8.2, 15.0 | late, 0.13, 1.9, 3.9, 14.5 | late, 0.13, 1.1, 6.4, 3.2 | late, 0.17, 1.9, 4.6, 7.8 |
