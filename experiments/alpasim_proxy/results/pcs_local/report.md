## 1. Does local PCS reproduce the board?

Arena held windows in the fit: 2106 (garage's complete set). Board-mapped PCS = least squares of ability on the 6 anchors' private PCS; JTN-s1 native / + stop3 are out of sample. Kendall tau over the 8 subjects with a private PCS.

| variant | rows (subjects) | tau, 8 known | tau, 6 anchors | RMSE on 6 anchors | JTN-s1 | JTN-s1 + stop3 | garage | go-straight | stop3 - garage (board -80) | native - go-straight (board +25) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| official | 33 (33) | +0.29 | -0.07 | 235 | 1617 | 1874 | 1174 | 1077 | +700 | +540 |
| seed1 | 33 (33) | +0.29 | -0.07 | 235 | 1623 | 1848 | 1178 | 1076 | +670 | +547 |
| seed2 | 33 (33) | +0.29 | -0.07 | 234 | 1655 | 1913 | 1181 | 1074 | +732 | +582 |
| seed3 | 33 (33) | +0.29 | -0.07 | 234 | 1675 | 1926 | 1185 | 1072 | +741 | +603 |
| 1pl | 33 (33) | +0.36 | +0.07 | 219 | 1587 | 1683 | 1275 | 1035 | +409 | +552 |
| no-third | 29 (29) | +0.29 | -0.07 | 236 | 1645 | 1964 | 1168 | 1083 | +796 | +563 |
| dedup | 12 (12) | +0.29 | -0.07 | 240 | 1020 | 1005 | 1117 | 1143 | -112 | -123 |
| garage10 | 42 (33) | +0.29 | -0.07 | 219 | 1899 | 2108 | 1264 | 1042 | +845 | +857 |
| garage30 | 62 (33) | +0.57 | +0.47 | 162 | 1965 | 2198 | 1502 | 1026 | +696 | +939 |
| boardmix | 51 (8) | +0.71 | +0.47 | 145 | 1251 | 1333 | 1553 | 1035 | -219 | +216 |
| boardmix-s1 | 51 (8) | +0.86 | +0.73 | 142 | 1249 | 1335 | 1556 | 1044 | -221 | +205 |
| boardmix-s2 | 51 (8) | +0.71 | +0.47 | 144 | 1253 | 1336 | 1553 | 1034 | -217 | +218 |
| boardmix-1pl | 51 (8) | +0.36 | +0.07 | 208 | 1635 | 1703 | 1328 | 1012 | +375 | +623 |
| boardmix-g15 | 36 (8) | +0.71 | +0.47 | 143 | 1322 | 1446 | 1554 | 1030 | -108 | +292 |
| boardmix-g60 | 81 (8) | +0.86 | +0.73 | 142 | 1198 | 1261 | 1558 | 1050 | -297 | +148 |

Board: JTN-s1 1325, JTN-s1 + stop3 1463, garage 1543, go-straight 1300. The arena held mean orders the 8 with tau +0.71 (board mean scene score: +0.71).
Out-of-pool scoring check: MAP ability under fixed items minus the fitted posterior mean, pool subjects, all variants: max |diff| 3.30.

### Every subject: arena held outcomes and local PCS (official pool and board-mix pool)

| subject | family | held mean | exact 1 | exact 0 | partial | PCS official | PCS board-mix | board PCS (mean) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| garage | third | 0.803 | 0.743 | 0.164 | 0.094 | 1174 | 1553 | 1543 (0.848) |
| JTB-s0+stop3 | ours | 0.943 | 0.916 | 0.037 | 0.047 | 1874 | 1387 |  |
| JTN-s1+stop3 | ours | 0.944 | 0.876 | 0.027 | 0.097 | 1874 | 1333 | 1463 (0.950) |
| JTN-s0+stop3 | ours | 0.944 | 0.866 | 0.027 | 0.107 | 1864 | 1272 |  |
| JT00-s0+leash3 | ours | 0.905 | 0.885 | 0.077 | 0.038 | 1872 | 1270 |  |
| JT10-s0+leash3 | ours | 0.906 | 0.882 | 0.076 | 0.041 | 1868 | 1263 |  |
| NTV0-s0+stop3 | ours | 0.937 | 0.855 | 0.038 | 0.107 | 1847 | 1253 |  |
| NTVX-s0+stop3 | ours | 0.941 | 0.858 | 0.034 | 0.108 | 1865 | 1251 |  |
| JTN-s1 | ours | 0.893 | 0.830 | 0.092 | 0.078 | 1617 | 1251 | 1325 (0.915) |
| JTN-s0+stop3+ACC | ours | 0.928 | 0.879 | 0.042 | 0.079 | 1757 | 1250 |  |
| RSG+stop3+ACC | ours | 0.929 | 0.877 | 0.044 | 0.079 | 1739 | 1240 |  |
| RSG+stop3 | ours | 0.945 | 0.851 | 0.029 | 0.120 | 1819 | 1233 |  |
| BNC2K-s1 | ours | 0.906 | 0.835 | 0.077 | 0.088 | 1647 | 1226 |  |
| BNX2K-s1 | ours | 0.888 | 0.826 | 0.098 | 0.076 | 1571 | 1226 |  |
| RSG+leash3 | ours | 0.935 | 0.844 | 0.041 | 0.114 | 1789 | 1224 |  |
| JTN-s1+warpconst | ours | 0.885 | 0.819 | 0.099 | 0.082 | 1559 | 1219 |  |
| RSG+lead | ours | 0.925 | 0.850 | 0.035 | 0.115 | 1721 | 1218 |  |
| RSG+full | ours | 0.925 | 0.850 | 0.035 | 0.115 | 1722 | 1218 |  |
| JTN-s0 | ours | 0.880 | 0.807 | 0.104 | 0.088 | 1533 | 1198 |  |
| JTB-s0 | ours | 0.827 | 0.811 | 0.169 | 0.021 | 1388 | 1167 |  |
| NR1F | ours | 0.805 | 0.750 | 0.188 | 0.063 | 1299 | 1131 |  |
| NR1HV | ours | 0.804 | 0.745 | 0.188 | 0.067 | 1297 | 1130 |  |
| NR1HV2 | ours | 0.796 | 0.737 | 0.195 | 0.068 | 1280 | 1125 |  |
| RSG | ours | 0.788 | 0.707 | 0.203 | 0.090 | 1274 | 1106 |  |
| NTV0-s0 | ours | 0.772 | 0.704 | 0.219 | 0.077 | 1263 | 1104 |  |
| JT10-s0 | ours | 0.764 | 0.751 | 0.234 | 0.015 | 1273 | 1104 |  |
| NTVX-s0 | ours | 0.769 | 0.699 | 0.223 | 0.078 | 1256 | 1100 |  |
| JT00-s0 | ours | 0.755 | 0.745 | 0.243 | 0.012 | 1257 | 1096 |  |
| YR | ours | 0.751 | 0.675 | 0.240 | 0.085 | 1214 | 1076 | 1030 (0.800) |
| DD | third | 0.664 | 0.612 | 0.323 | 0.065 | 1088 | 1041 | 1080 (0.772) |
| LTF | third | 0.663 | 0.606 | 0.323 | 0.070 | 1086 | 1037 | 1014 (0.738) |
| go-straight | third | 0.688 | 0.645 | 0.262 | 0.093 | 1077 | 1035 | 1300 (0.799) |
| APY | ours | 0.647 | 0.598 | 0.349 | 0.054 | 1108 | 1006 | 779 (0.640) |

## 2a. Item parameters by cell, official pool

Cells: arena strata; static split by a lead within 8 m at the window start. a = discrimination; g0 / g1 = exact-0 / exact-1 thresholds (P(0) = sigmoid(g0 - a theta), P(1) = sigmoid(a theta - g1)).

| cell | windows | a median [q25, q75] | share a < 0.5 | b median | g0 median | g1 median | third-party mean | garage | ours mean | JTN-s1 | JTN-s1 + stop3 |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| braking | 346 | 2.38 [0.47, 2.97] | 0.26 | -0.62 | -1.19 | -0.59 | 0.706 | 0.815 | 0.917 | 0.925 | 0.948 |
| cruise | 113 | 2.88 [0.78, 2.95] | 0.20 | -0.62 | -1.57 | -0.88 | 0.914 | 0.967 | 0.985 | 0.983 | 0.981 |
| launch | 181 | 0.45 [0.15, 2.84] | 0.54 | -0.12 | -1.56 | -0.22 | 0.688 | 0.526 | 0.822 | 0.853 | 0.746 |
| other | 430 | 2.84 [0.46, 3.03] | 0.26 | -0.62 | -1.35 | -0.79 | 0.792 | 0.850 | 0.937 | 0.936 | 0.941 |
| static-free | 376 | 0.94 [0.76, 1.14] | 0.03 | -0.61 | +0.80 | +1.01 | 0.642 | 0.835 | 0.632 | 0.686 | 0.995 |
| static-lead | 457 | 2.54 [2.43, 2.76] | 0.10 | -0.61 | -1.21 | -0.85 | 0.793 | 0.937 | 0.924 | 0.991 | 0.998 |
| turn | 203 | 2.11 [0.42, 3.05] | 0.29 | -0.62 | -0.53 | +0.01 | 0.329 | 0.476 | 0.875 | 0.898 | 0.889 |

Spearman over items of a against (third-party mean - ours mean): -0.21; against the garage score: +0.42.

## 2b. Item parameters by cell, boardmix pool

Cells: arena strata; static split by a lead within 8 m at the window start. a = discrimination; g0 / g1 = exact-0 / exact-1 thresholds (P(0) = sigmoid(g0 - a theta), P(1) = sigmoid(a theta - g1)).

| cell | windows | a median [q25, q75] | share a < 0.5 | b median | g0 median | g1 median | third-party mean | garage | ours mean | JTN-s1 | JTN-s1 + stop3 |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| braking | 346 | 1.85 [0.14, 3.06] | 0.33 | -0.29 | -1.42 | -0.65 | 0.706 | 0.815 | 0.917 | 0.925 | 0.948 |
| cruise | 113 | 2.57 [1.56, 3.27] | 0.10 | -0.29 | -1.90 | -1.42 | 0.914 | 0.967 | 0.985 | 0.983 | 0.981 |
| launch | 181 | 0.13 [0.08, 3.20] | 0.60 | -0.22 | -1.67 | +0.74 | 0.688 | 0.526 | 0.822 | 0.853 | 0.746 |
| other | 430 | 2.42 [0.64, 3.21] | 0.24 | -0.29 | -1.43 | -0.99 | 0.792 | 0.850 | 0.937 | 0.936 | 0.941 |
| static-free | 376 | 1.73 [1.54, 1.82] | 0.16 | -0.29 | +0.99 | +1.22 | 0.642 | 0.835 | 0.632 | 0.686 | 0.995 |
| static-lead | 457 | 2.02 [1.73, 3.24] | 0.06 | -0.29 | -1.50 | -1.22 | 0.793 | 0.937 | 0.924 | 0.991 | 0.998 |
| turn | 203 | 0.15 [0.10, 1.81] | 0.54 | -0.30 | +1.11 | +1.60 | 0.329 | 0.476 | 0.875 | 0.898 | 0.889 |

Spearman over items of a against (third-party mean - ours mean): +0.35; against the garage score: +0.75.

## 3. What costs us PCS (boardmix pool, items fixed, board-mapped PCS)


### JTN-s1+stop3: PCS 1324 (garage 1553, gap -229)

PCS change when this subject's scores on one cell are replaced (all other cells unchanged):

| cell | windows | ours 0 / partial / 1 | garage 0 / partial / 1 | take garage's scores | max(ours, garage) | zeros -> 1 | partials -> 1 | all -> 1 |
|---|--:|---|---|--:|--:|--:|--:|--:|
| braking | 346 | 16 / 28 / 302 | 58 / 62 / 226 | +15 | +25 | +4 | +21 | +26 |
| cruise | 113 | 0 / 18 / 95 | 3 / 8 / 102 | +15 | +16 | +0 | +16 | +16 |
| launch | 181 | 5 / 74 / 102 | 35 / 74 / 72 | +10 | +18 | +1 | +18 | +19 |
| other | 430 | 15 / 62 / 353 | 52 / 50 / 328 | +54 | +66 | +5 | +59 | +68 |
| static-free | 376 | 2 / 0 / 374 | 62 / 0 / 314 | -6 | +0 | +0 | +0 | +0 |
| static-lead | 457 | 1 / 0 / 456 | 29 / 0 / 428 | -3 | +0 | +0 | +0 | +0 |
| turn | 203 | 18 / 22 / 163 | 106 / 3 / 94 | +11 | +22 | +11 | +11 | +24 |
| all | 2106 | 57 / 204 / 1845 | 345 / 197 / 1564 | +201 | +942 | +23 | +942 | +942 |

JTN-s1+stop3 by outcome class (those windows set to 1):

| outcome | windows | mean-score points lost | PCS if fixed | PCS per window | garage scores 1 on |
|---|--:|--:|--:|--:|--:|
| 0 at-fault collision | 1 | 1.0 | +1 | +0.65 | 1 |
| 0 corridor, moving window | 6 | 6.0 | +1 | +0.11 | 0 |
| 0 offroad | 50 | 50.0 | +21 | +0.43 | 11 |
| partial >= 0.98 | 17 | 0.2 | +10 | +0.59 | 11 |
| partial, launch window (late / no launch, creep, rear-hit) | 74 | 40.9 | +18 | +0.25 | 9 |
| partial, slow progress | 105 | 16.8 | +145 | +1.38 | 60 |
| partial, struck from behind | 8 | 2.1 | +6 | +0.72 | 5 |

JTN-s1+stop3 combined counterfactuals:

| change | windows | mean score after | PCS after | gain |
|---|--:|--:|--:|--:|
| slow-progress partials -> 1 (progress >= 0.8 of the human) | 105 | 0.952 | 1469 | +145 |
| + partials >= 0.98 and struck-from-behind -> 1 | 130 | 0.954 | 1575 | +251 |
| + offroad zeros -> 1 | 180 | 0.977 | 1742 | +418 |
| static-window corridor zeros -> 1 (native: stop at rest) | 0 | 0.944 | 1324 | +0 |
| launch-window partials -> 1 | 74 | 0.964 | 1342 | +18 |
| every partial -> 1 (zeros kept) | 204 | 0.973 | 2266 | +942 |
| every zero -> 1 (partials kept) | 57 | 0.972 | 1347 | +23 |

Marginal PCS of fixing one window (JTN-s1 + stop3):

| our outcome | garage on the window | windows | median PCS gain | mean | median a |
|---|---|--:|--:|--:|--:|
| exact 0 | garage 0 | 41 | +0.1 | +0.1 | 0.14 |
| exact 0 | garage 1 | 12 | +1.3 | +1.2 | 1.49 |
| exact 0 | garage partial | 4 | +0.5 | +0.5 | 0.38 |
| partial < 0.9 | garage 0 | 31 | +0.0 | +0.0 | 0.08 |
| partial < 0.9 | garage 1 | 41 | +0.9 | +1.3 | 0.75 |
| partial < 0.9 | garage partial | 59 | +0.0 | +0.1 | 0.08 |
| partial >= 0.9 | garage 0 | 6 | -0.0 | -0.0 | 0.11 |
| partial >= 0.9 | garage 1 | 44 | +0.8 | +1.0 | 0.84 |
| partial >= 0.9 | garage partial | 23 | -0.0 | -0.1 | 0.09 |

The three-way likelihood on one typical window (median-a launch-cell item): ability change per outcome, holding the rest:
- launch item (a 3.18): 0 -> -2.7, 0.5 -> -25.0, 0.9 -> -8.1, 0.95 -> -6.0, 0.99 -> -4.4, 1 -> +0.0
- other item (a 2.50): 0 -> -2.1, 0.5 -> -13.1, 0.9 -> -4.8, 0.95 -> -3.9, 0.99 -> -3.2, 1 -> +0.0

### JTN-s1: PCS 1247 (garage 1553, gap -306)

PCS change when this subject's scores on one cell are replaced (all other cells unchanged):

| cell | windows | ours 0 / partial / 1 | garage 0 / partial / 1 | take garage's scores | max(ours, garage) | zeros -> 1 | partials -> 1 | all -> 1 |
|---|--:|---|---|--:|--:|--:|--:|--:|
| braking | 346 | 24 / 28 / 294 | 58 / 62 / 226 | +11 | +14 | +5 | +10 | +15 |
| cruise | 113 | 0 / 18 / 95 | 3 / 8 / 102 | +7 | +7 | +0 | +7 | +7 |
| launch | 181 | 10 / 43 / 128 | 35 / 74 / 72 | -1 | +3 | +1 | +2 | +3 |
| other | 430 | 19 / 58 / 353 | 52 / 50 / 328 | +24 | +29 | +6 | +22 | +30 |
| static-free | 376 | 118 / 0 / 258 | 62 / 0 / 314 | +77 | +81 | +84 | +0 | +84 |
| static-lead | 457 | 4 / 0 / 453 | 29 / 0 / 428 | +0 | +2 | +2 | +0 | +2 |
| turn | 203 | 18 / 18 / 167 | 106 / 3 / 94 | +3 | +9 | +7 | +3 | +9 |
| all | 2106 | 193 / 165 / 1748 | 345 / 197 / 1564 | +278 | +1020 | +129 | +57 | +1020 |

JTN-s1 by outcome class (those windows set to 1):

| outcome | windows | mean-score points lost | PCS if fixed | PCS per window | garage scores 1 on |
|---|--:|--:|--:|--:|--:|
| 0 at-fault collision | 4 | 4.0 | +2 | +0.52 | 3 |
| 0 corridor, moving window | 18 | 18.0 | +6 | +0.31 | 6 |
| 0 corridor, static window | 115 | 115.0 | +82 | +0.72 | 82 |
| 0 offroad | 56 | 56.0 | +15 | +0.26 | 14 |
| partial >= 0.98 | 18 | 0.2 | +5 | +0.28 | 13 |
| partial, launch window (late / no launch, creep, rear-hit) | 43 | 16.7 | +2 | +0.04 | 4 |
| partial, slow progress | 96 | 12.8 | +39 | +0.40 | 54 |
| partial, struck from behind | 8 | 2.1 | +3 | +0.34 | 5 |

JTN-s1 combined counterfactuals:

| change | windows | mean score after | PCS after | gain |
|---|--:|--:|--:|--:|
| slow-progress partials -> 1 (progress >= 0.8 of the human) | 96 | 0.899 | 1285 | +39 |
| + partials >= 0.98 and struck-from-behind -> 1 | 122 | 0.900 | 1299 | +53 |
| + offroad zeros -> 1 | 178 | 0.927 | 1326 | +79 |
| static-window corridor zeros -> 1 (native: stop at rest) | 115 | 0.948 | 1329 | +82 |
| launch-window partials -> 1 | 43 | 0.901 | 1248 | +2 |
| every partial -> 1 (zeros kept) | 165 | 0.908 | 1303 | +57 |
| every zero -> 1 (partials kept) | 193 | 0.985 | 1375 | +129 |

### Our stored candidates scored as one more submission against the boardmix pool

| subject | held mean | PCS | zeros | partials |
|---|--:|--:|--:|--:|
| JTB-s0+stop3 | 0.943 | 1387 | 77 | 100 |
| JTN-s1+stop3 | 0.944 | 1333 | 57 | 204 |
| JTN-s0+stop3 | 0.944 | 1272 | 57 | 226 |
| JT00-s0+leash3 | 0.905 | 1270 | 163 | 80 |
| JT10-s0+leash3 | 0.906 | 1263 | 161 | 87 |
| NTV0-s0+stop3 | 0.937 | 1253 | 79 | 226 |
| NTVX-s0+stop3 | 0.941 | 1251 | 71 | 228 |
| JTN-s1 | 0.893 | 1251 | 193 | 165 |
| JTN-s0+stop3+ACC | 0.928 | 1250 | 88 | 167 |
| RSG+stop3+ACC | 0.929 | 1240 | 92 | 166 |
| RSG+stop3 | 0.945 | 1233 | 61 | 253 |
| BNC2K-s1 | 0.906 | 1226 | 162 | 185 |
| BNX2K-s1 | 0.888 | 1226 | 206 | 161 |
| RSG+leash3 | 0.935 | 1224 | 87 | 241 |
| JTN-s1+warpconst | 0.885 | 1219 | 209 | 172 |
| RSG+lead | 0.925 | 1218 | 74 | 242 |
| RSG+full | 0.925 | 1218 | 74 | 242 |
| JTN-s0 | 0.880 | 1198 | 220 | 186 |
| JTB-s0 | 0.827 | 1167 | 355 | 44 |
| NR1F | 0.805 | 1131 | 395 | 132 |
| NR1HV | 0.804 | 1130 | 396 | 142 |
| NR1HV2 | 0.796 | 1125 | 410 | 143 |
| RSG | 0.788 | 1106 | 428 | 190 |
| NTV0-s0 | 0.772 | 1104 | 462 | 162 |
| JT10-s0 | 0.764 | 1104 | 493 | 32 |
| NTVX-s0 | 0.769 | 1100 | 469 | 164 |
| JT00-s0 | 0.755 | 1096 | 511 | 26 |
| YR | 0.751 | 1076 | 506 | 179 |
| APY | 0.647 | 1006 | 734 | 113 |

## 4. How many subjects a local PCS needs

|   garage_rows | our_other_subjects   |   rows |   tau8 |   native |   stop3 |   garage |   go_straight |
|--------------:|:---------------------|-------:|-------:|---------:|--------:|---------:|--------------:|
|             1 | False                |     22 |   0.21 |   928.19 |  916.14 |  1133.53 |       1168.61 |
|             1 | True                 |     47 |   0.29 |  1321.95 | 1365.16 |  1145.02 |       1095.54 |
|             3 | False                |     24 |   0.43 |  1715.12 | 1734.92 |  1249.05 |        984.89 |
|             3 | True                 |     49 |   0.29 |  1592.11 | 1705.92 |  1217.53 |       1049.34 |
|            10 | False                |     31 |   0.71 |  1382.64 | 1531.16 |  1556.18 |       1031.13 |
|            10 | True                 |     56 |   0.36 |  1771.04 | 1914.11 |  1334.71 |       1017.54 |
|            30 | False                |     51 |   0.71 |  1250.96 | 1333.37 |  1552.58 |       1034.89 |
|            30 | True                 |     76 |   0.64 |  1438.73 | 1732.18 |  1521.57 |       1028.20 |
