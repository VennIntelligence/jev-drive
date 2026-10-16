# DESIRE-CAP open loop

Responses are paired against the same row's none arm, signed toward the desire side; stage = mean of its seeds row by row. CI = log / scene cluster bootstrap (B 10 000, seed 0).

## 1. Lane change (primary): C_fast (v0 >= 8.9 m/s, human keeps lane), LCL / LCR pooled, pulse age 0

rows 543, logs 89

| stage | seeds | R_y 1 s | 2 s | 4 s mean [CI] | 4 s median | 6 s | 8 s | R_psi 4 s (deg) | dir ok | full / partial / flat | LCL 4 s | LCR 4 s | ratio vs P0 [CI] | class |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|:--|
| P0 | 1 | 0.05 | 0.30 | 1.04 [0.92, 1.19] | 0.90 | 1.43 | 1.54 | 1.2 | 1.000 | 0.02 / 0.81 / 0.16 | 1.09 | 0.98 | - | partial |
| P2-F | 2 | 0.01 | 0.09 | 0.32 [0.26, 0.40] | 0.18 | 0.45 | 0.45 | 0.5 | 0.648 | 0.00 / 0.25 / 0.75 | 0.03 | 0.62 | 0.31 [0.28, 0.34] | weakened |
| P2H10-F | 2 | 0.01 | 0.09 | 0.30 [0.24, 0.37] | 0.17 | 0.41 | 0.39 | 0.5 | 0.635 | 0.00 / 0.21 / 0.79 | 0.03 | 0.56 | 0.28 [0.25, 0.31] | killed |
| YR10m10-F | 2 | 0.08 | 0.27 | 0.68 [0.59, 0.78] | 0.55 | 0.91 | 0.90 | 0.9 | 0.912 | 0.00 / 0.55 / 0.45 | 0.20 | 1.15 | 0.65 [0.61, 0.69] | weakened |
| YS-L18 | 2 | 0.13 | 0.28 | 0.61 [0.53, 0.71] | 0.46 | 0.85 | 0.84 | 0.9 | 0.873 | 0.00 / 0.48 / 0.52 | 0.12 | 1.10 | 0.59 [0.54, 0.63] | weakened |
| RB0-F | 2 | 0.07 | 0.24 | 0.57 [0.49, 0.66] | 0.47 | 0.78 | 0.81 | 0.7 | 0.856 | 0.00 / 0.48 / 0.52 | 0.09 | 1.06 | 0.55 [0.51, 0.58] | weakened |
| RSG-L18 | 2 | 0.09 | 0.29 | 0.69 [0.60, 0.78] | 0.64 | 0.94 | 1.00 | 0.9 | 0.878 | 0.00 / 0.59 / 0.41 | 0.10 | 1.27 | 0.66 [0.62, 0.71] | weakened |
| RBSG2-L18 | 2 | 0.07 | 0.22 | 0.47 [0.41, 0.54] | 0.41 | 0.58 | 0.56 | 0.6 | 0.818 | 0.00 / 0.42 / 0.58 | 0.11 | 0.83 | 0.45 [0.42, 0.48] | weakened |

## 2. Turn (primary): T_L + T_R (human turning in a junction), turn desire on the human's side, pulse age 0

rows 1036, logs 110

| stage | R_psi 4 s mean [CI] (deg) | median | R_y 4 s (m) | R_psi 2 s | opposite-side desire R_psi 4 s | ratio vs P0 [CI] | class |
|:--|:--|--:|--:|--:|--:|:--|:--|
| P0 | -3.3 [-4.3, -2.3] | -1.9 | -0.61 | -2.6 | 19.5 | - | absent |
| P2-F | -7.0 [-8.1, -5.9] | -4.2 | -0.51 | -3.0 | 16.8 | 2.14 [1.69, 2.88] | kept |
| P2H10-F | -6.4 [-7.4, -5.4] | -4.0 | -0.47 | -2.8 | 17.1 | 1.96 [1.55, 2.65] | kept |
| YR10m10-F | -7.0 [-8.1, -5.9] | -4.1 | -0.57 | -3.0 | 18.1 | 2.15 [1.71, 2.86] | kept |
| YS-L18 | -5.8 [-6.9, -4.6] | -2.9 | -0.42 | -2.5 | 17.9 | 1.77 [1.41, 2.33] | kept |
| RB0-F | -7.9 [-8.8, -7.0] | -4.9 | -0.67 | -3.3 | 13.6 | 2.42 [1.91, 3.35] | kept |
| RSG-L18 | -8.1 [-9.0, -7.2] | -5.1 | -0.64 | -3.3 | 13.6 | 2.49 [1.95, 3.47] | kept |
| RBSG2-L18 | -8.5 [-9.3, -7.6] | -5.5 | -0.56 | -3.2 | 14.3 | 2.60 [2.01, 3.64] | kept |

## 3. Pulse age (open loop: frames do not move after the pulse)

| stage | LC R_y 4 s @ 0 / 0.4 / 1.0 / 2.0 s | LC R_y 8 s @ same | turn R_psi 4 s @ same |
|:--|:--|:--|:--|
| P0 | 1.04 / 0.81 / 0.61 / 0.43 | 1.54 / 1.19 / 0.89 / 0.63 | -3.3 / -2.0 / 0.7 / 5.0 |
| P2-F | 0.32 / 0.23 / 0.12 / 0.07 | 0.45 / 0.37 / 0.22 / 0.10 | -7.0 / -7.2 / -5.2 / -0.5 |
| P2H10-F | 0.30 / 0.21 / 0.11 / 0.06 | 0.39 / 0.32 / 0.17 / 0.07 | -6.4 / -6.4 / -4.4 / 0.2 |
| YR10m10-F | 0.68 / 0.58 / 0.41 / 0.25 | 0.90 / 0.79 / 0.52 / 0.27 | -7.0 / -6.4 / -4.0 / 0.9 |
| YS-L18 | 0.61 / 0.62 / 0.47 / 0.30 | 0.84 / 0.88 / 0.64 / 0.36 | -5.8 / -5.0 / -3.3 / 0.9 |
| RB0-F | 0.57 / 0.53 / 0.38 / 0.23 | 0.81 / 0.74 / 0.49 / 0.25 | -7.9 / -7.8 / -5.6 / -0.4 |
| RSG-L18 | 0.69 / 0.64 / 0.46 / 0.29 | 1.00 / 0.92 / 0.63 / 0.34 | -8.1 / -8.0 / -5.8 / -0.4 |
| RBSG2-L18 | 0.47 / 0.45 / 0.33 / 0.21 | 0.56 / 0.52 / 0.36 / 0.20 | -8.5 / -8.2 / -6.0 / -0.5 |

## 4. Companion readings (report only)

| stage | C_slow LC R_y 4 s | C_slow turn R_psi 4 s (TL/TR pooled) | LC_L+LC_R success none -> same-side LC@0 | LC_L+LC_R R_y 4 s | P_keep LC R_y 4 s | P_lca route-side LC R_y 4 s | P_turn route-side turn R_psi 4 s | desire_state p(own) LC@0 C_fast none -> LC |
|:--|--:|--:|:--|--:|--:|--:|--:|:--|
| P0 | 0.83 | 10.6 | 0.730 -> 0.972 | 0.88 | 1.60 | 1.52 | -3.7 | 0.000 -> 1.000 |
| P2-F | 0.13 | 6.6 | 0.747 -> 0.919 | 0.46 | 0.63 | 0.72 | -3.4 | 0.000 -> 0.591 |
| P2H10-F | 0.13 | 6.4 | 0.744 -> 0.919 | 0.48 | 0.61 | 0.70 | -3.1 | 0.000 -> 0.593 |
| YR10m10-F | 0.39 | 6.6 | 0.663 -> 0.961 | 0.88 | 1.11 | 1.29 | -4.7 | 0.000 -> 0.710 |
| YS-L18 | 0.36 | 6.0 | 0.681 -> 0.926 | 0.80 | 1.07 | 1.26 | -2.7 | 0.000 -> 0.494 |
| RB0-F | 0.34 | 2.2 | 0.681 -> 0.944 | 0.82 | 0.94 | 1.34 | -5.0 | 0.000 -> 0.589 |
| RSG-L18 | 0.40 | 1.1 | 0.674 -> 0.947 | 0.88 | 1.07 | 1.50 | -4.8 | 0.000 -> 0.631 |
| RBSG2-L18 | 0.23 | 0.7 | 0.642 -> 0.888 | 0.69 | 0.89 | 1.86 | -5.5 | 0.000 -> 0.592 |

## 5. Route-memory stages with the memory cleared (served empty memory), pulse age 0

| stage | C_fast LC R_y 4 s (memory fed) | (memory cleared) | T turn R_psi 4 s (fed) | (cleared) |
|:--|--:|--:|--:|--:|
| RB0-F | 0.57 | 0.55 | -7.9 | -8.0 |
| RSG-L18 | 0.69 | 0.69 | -8.1 | -7.6 |
| RBSG2-L18 | 0.47 | 0.48 | -8.5 | -6.9 |

## 6. Registered lines

- Line 1 (base lane change): P0 1.04 m at 4 s, direction ok 1.000 -> **partial**; 8 s 1.54 m
- Line 2 (base turn): P0 -3.3 deg at 4 s -> **absent**
- Line 3 (lc): P2-F 0.31 weakened, P2H10-F 0.28 killed, YR10m10-F 0.65 weakened, YS-L18 0.59 weakened, RB0-F 0.55 weakened, RSG-L18 0.66 weakened, RBSG2-L18 0.45 weakened; first killing stage per branch: ['P2H10-F']
- Line 3 (turn): not judged (P0 absent)
- Line 4 (shape, lane change, median at 4 s): P0 graded, P2-F flat, P2H10-F flat, YR10m10-F graded, YS-L18 flat, RB0-F flat, RSG-L18 graded, RBSG2-L18 flat
- G-rep: P0 none vs cap_ledger plans/H/P0 at 4 s: median 0.00e+00 m, max 0.00e+00 m
