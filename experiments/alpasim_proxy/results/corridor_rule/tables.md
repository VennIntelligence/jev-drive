# corridor rule: per-step scorer metrics of stored reads

## Arena, static stratum (human max displacement < 1 m), by the scorer's GT length

| tag | split | GT length (m) | windows | corridor zeros | tripped (d >= 4 m) | corridor / tripped | median lateral / d at the trip |
|---|---|---|--:|--:|--:|--:|--:|
| APY10m10-AB-s0 | held | [0.0, 0.05) | 783 | 606 | 627 | 0.97 | 1.00 |
| APY10m10-AB-s0 | held | [0.05, 0.25) | 35 | 15 | 22 | 0.68 | 1.00 |
| APY10m10-AB-s0 | held | [0.25, 0.5) | 31 | 10 | 15 | 0.67 | 1.00 |
| APY10m10-AB-s0 | held | [0.5, 1.0) | 28 | 7 | 17 | 0.41 | 0.49 |
| APY10m10-AB-s0 | held | [1.0, 2.0) | 1 | 1 | 1 | 1.00 | 1.00 |
| APY10m10-AB-s0 | train | [0.0, 0.05) | 1971 | 1560 | 1650 | 0.95 | 1.00 |
| APY10m10-AB-s0 | train | [0.05, 0.25) | 120 | 65 | 97 | 0.67 | 1.00 |
| APY10m10-AB-s0 | train | [0.25, 0.5) | 72 | 31 | 52 | 0.60 | 1.00 |
| APY10m10-AB-s0 | train | [0.5, 1.0) | 99 | 37 | 66 | 0.56 | 1.00 |
| APY10m10-AB-s0 | train | [1.0, 2.0) | 4 | 0 | 2 | 0.00 | 0.01 |
| G0014 | train | [0.0, 0.05) | 4 | 0 | 0 | 0.00 | nan |
| G0014 | train | [0.05, 0.25) | 2 | 0 | 1 | 0.00 | 0.02 |
| NR1F-L18-s0 | held | [0.0, 0.05) | 783 | 323 | 336 | 0.96 | 1.00 |
| NR1F-L18-s0 | held | [0.05, 0.25) | 35 | 11 | 14 | 0.79 | 1.00 |
| NR1F-L18-s0 | held | [0.25, 0.5) | 31 | 5 | 8 | 0.62 | 1.00 |
| NR1F-L18-s0 | held | [0.5, 1.0) | 28 | 4 | 9 | 0.44 | 0.39 |
| NR1F-L18-s0 | held | [1.0, 2.0) | 1 | 0 | 0 | 0.00 | nan |
| NR1F-L18-s0 | train | [0.0, 0.05) | 1967 | 730 | 768 | 0.95 | 1.00 |
| NR1F-L18-s0 | train | [0.05, 0.25) | 118 | 39 | 51 | 0.76 | 1.00 |
| NR1F-L18-s0 | train | [0.25, 0.5) | 72 | 17 | 25 | 0.68 | 1.00 |
| NR1F-L18-s0 | train | [0.5, 1.0) | 99 | 17 | 29 | 0.59 | 1.00 |
| NR1F-L18-s0 | train | [1.0, 2.0) | 4 | 0 | 1 | 0.00 | 0.01 |
| P0 | train | [0.0, 0.05) | 4 | 0 | 1 | 0.00 | 0.10 |
| P0 | train | [0.05, 0.25) | 2 | 0 | 1 | 0.00 | 0.01 |
| RSG-L18-s0 | held | [0.0, 0.05) | 783 | 356 | 366 | 0.97 | 1.00 |
| RSG-L18-s0 | held | [0.05, 0.25) | 35 | 11 | 15 | 0.73 | 1.00 |
| RSG-L18-s0 | held | [0.25, 0.5) | 31 | 7 | 10 | 0.70 | 1.00 |
| RSG-L18-s0 | held | [0.5, 1.0) | 28 | 6 | 11 | 0.55 | 1.00 |
| RSG-L18-s0 | held | [1.0, 2.0) | 1 | 0 | 0 | 0.00 | nan |
| RSG-L18-s0 | train | [0.0, 0.05) | 1971 | 889 | 931 | 0.95 | 1.00 |
| RSG-L18-s0 | train | [0.05, 0.25) | 120 | 42 | 57 | 0.74 | 1.00 |
| RSG-L18-s0 | train | [0.25, 0.5) | 72 | 22 | 30 | 0.73 | 1.00 |
| RSG-L18-s0 | train | [0.5, 1.0) | 99 | 20 | 33 | 0.61 | 1.00 |
| RSG-L18-s0 | train | [1.0, 2.0) | 4 | 0 | 1 | 0.00 | 0.01 |
| YR10m10-F-s0 | held | [0.0, 0.05) | 783 | 364 | 376 | 0.97 | 1.00 |
| YR10m10-F-s0 | held | [0.05, 0.25) | 35 | 12 | 16 | 0.75 | 1.00 |
| YR10m10-F-s0 | held | [0.25, 0.5) | 31 | 7 | 10 | 0.70 | 1.00 |
| YR10m10-F-s0 | held | [0.5, 1.0) | 28 | 6 | 12 | 0.50 | 0.70 |
| YR10m10-F-s0 | held | [1.0, 2.0) | 1 | 1 | 1 | 1.00 | 1.00 |
| YR10m10-F-s0 | train | [0.0, 0.05) | 1971 | 956 | 1010 | 0.95 | 1.00 |
| YR10m10-F-s0 | train | [0.05, 0.25) | 120 | 50 | 64 | 0.78 | 1.00 |
| YR10m10-F-s0 | train | [0.25, 0.5) | 72 | 22 | 31 | 0.71 | 1.00 |
| YR10m10-F-s0 | train | [0.5, 1.0) | 99 | 25 | 40 | 0.62 | 1.00 |
| YR10m10-F-s0 | train | [1.0, 2.0) | 4 | 0 | 1 | 0.00 | 0.00 |
| starter | train | [0.0, 0.05) | 4 | 0 | 0 | 0.00 | nan |
| starter | train | [0.05, 0.25) | 2 | 0 | 0 | 0.00 | nan |

## Arena, every stratum: corridor zeros given a trip, by GT length (all tags pooled)

| GT length (m) | windows | tripped | corridor | corridor / tripped | median lat/d at trip |
|---|--:|--:|--:|--:|--:|
| [0.0, 0.05) | 11024 | 6065 | 5784 | 0.95 | 1.00 |
| [0.05, 0.25) | 624 | 338 | 245 | 0.72 | 1.00 |
| [0.25, 0.5) | 412 | 181 | 121 | 0.67 | 1.00 |
| [0.5, 1.0) | 508 | 217 | 122 | 0.56 | 1.00 |
| [1.0, 2.0) | 716 | 239 | 92 | 0.38 | 0.18 |
| [2.0, 5.0) | 1330 | 486 | 150 | 0.31 | 0.19 |
| [5.0, 1000000000.0) | 16256 | 1001 | 295 | 0.29 | 0.42 |

## End state of the human's GT track: corridor zeros given a trip (the first metric step >= 4 m from the GT)

| tag | split | stratum | end | windows | tripped | corridor | corridor / tripped | median d_max kept (m), no corridor | min d at a corridor zero (m) |
|---|---|---|---|--:|--:|--:|--:|--:|--:|
| APY10m10-AB-s0 | held | all other | moving | 1228 | 143 | 23 | 0.16 | 0.45 | 4.03 |
| APY10m10-AB-s0 | held | all other | off-heading | 66 | 12 | 10 | 0.83 | 0.24 | 4.05 |
| APY10m10-AB-s0 | held | all other | still | 56 | 24 | 24 | 1.00 | 1.12 | 4.03 |
| APY10m10-AB-s0 | held | static | moving | 48 | 33 | 1 | 0.03 | 4.30 | 4.26 |
| APY10m10-AB-s0 | held | static | off-heading | 2 | 2 | 0 | 0.00 | 5.62 | nan |
| APY10m10-AB-s0 | held | static | still | 828 | 647 | 638 | 0.99 | 2.82 | 4.00 |
| APY10m10-AB-s0 | train | all other | moving | 2916 | 338 | 52 | 0.15 | 0.45 | 4.01 |
| APY10m10-AB-s0 | train | all other | off-heading | 174 | 25 | 22 | 0.88 | 0.28 | 4.02 |
| APY10m10-AB-s0 | train | all other | still | 127 | 47 | 47 | 1.00 | 1.85 | 4.07 |
| APY10m10-AB-s0 | train | static | moving | 176 | 135 | 13 | 0.10 | 4.43 | 4.10 |
| APY10m10-AB-s0 | train | static | off-heading | 4 | 3 | 2 | 0.67 | 3.32 | 4.23 |
| APY10m10-AB-s0 | train | static | still | 2080 | 1723 | 1678 | 0.97 | 3.17 | 4.00 |
| APY10m10-AB-s0 | wait-set a | static | moving | 146 | 122 | 3 | 0.02 | 4.65 | 4.08 |
| G0014 | train | all other | moving | 7 | 2 | 0 | 0.00 | 0.01 | nan |
| G0014 | wait-set a | static | moving | 146 | 23 | 1 | 0.04 | 0.00 | 4.84 |
| NR1F-L18-s0 | held | all other | moving | 1228 | 87 | 8 | 0.09 | 0.31 | 4.19 |
| NR1F-L18-s0 | held | all other | off-heading | 66 | 11 | 11 | 1.00 | 0.11 | 4.04 |
| NR1F-L18-s0 | held | all other | still | 56 | 17 | 17 | 1.00 | 0.08 | 4.08 |
| NR1F-L18-s0 | held | static | moving | 48 | 20 | 1 | 0.05 | 0.71 | 4.44 |
| NR1F-L18-s0 | held | static | off-heading | 2 | 1 | 0 | 0.00 | 2.30 | nan |
| NR1F-L18-s0 | held | static | still | 828 | 346 | 342 | 0.99 | 0.40 | 4.02 |
| NR1F-L18-s0 | train | all other | moving | 2909 | 193 | 22 | 0.11 | 0.30 | 4.04 |
| NR1F-L18-s0 | train | all other | off-heading | 174 | 20 | 18 | 0.90 | 0.14 | 4.21 |
| NR1F-L18-s0 | train | all other | still | 127 | 22 | 22 | 1.00 | 0.13 | 4.20 |
| NR1F-L18-s0 | train | static | moving | 176 | 58 | 8 | 0.14 | 0.71 | 4.00 |
| NR1F-L18-s0 | train | static | off-heading | 4 | 3 | 2 | 0.67 | 2.63 | 4.44 |
| NR1F-L18-s0 | train | static | still | 2080 | 813 | 793 | 0.98 | 0.51 | 4.00 |
| P0 | train | all other | moving | 7 | 0 | 0 | 0.00 | 0.01 | nan |
| P0 | wait-set a | static | moving | 146 | 19 | 1 | 0.05 | 0.00 | 5.23 |
| RSG-L18-s0 | held | all other | moving | 1228 | 87 | 8 | 0.09 | 0.30 | 4.08 |
| RSG-L18-s0 | held | all other | off-heading | 66 | 8 | 7 | 0.88 | 0.13 | 4.00 |
| RSG-L18-s0 | held | all other | still | 56 | 17 | 17 | 1.00 | 0.24 | 4.04 |
| RSG-L18-s0 | held | static | moving | 48 | 19 | 1 | 0.05 | 1.88 | 4.50 |
| RSG-L18-s0 | held | static | off-heading | 2 | 1 | 0 | 0.00 | 3.67 | nan |
| RSG-L18-s0 | held | static | still | 828 | 382 | 379 | 0.99 | 0.87 | 4.01 |
| RSG-L18-s0 | train | all other | moving | 2916 | 187 | 25 | 0.13 | 0.31 | 4.02 |
| RSG-L18-s0 | train | all other | off-heading | 174 | 20 | 18 | 0.90 | 0.12 | 4.04 |
| RSG-L18-s0 | train | all other | still | 127 | 22 | 22 | 1.00 | 0.43 | 4.08 |
| RSG-L18-s0 | train | static | moving | 176 | 60 | 9 | 0.15 | 1.77 | 4.15 |
| RSG-L18-s0 | train | static | off-heading | 4 | 3 | 2 | 0.67 | 2.32 | 4.21 |
| RSG-L18-s0 | train | static | still | 2080 | 987 | 962 | 0.97 | 0.92 | 4.00 |
| RSG-L18-s0 | wait-set a | static | moving | 146 | 73 | 3 | 0.04 | 3.67 | 4.05 |
| YR10m10-F-s0 | held | all other | moving | 1228 | 110 | 24 | 0.22 | 0.40 | 4.01 |
| YR10m10-F-s0 | held | all other | off-heading | 66 | 12 | 11 | 0.92 | 0.22 | 4.01 |
| YR10m10-F-s0 | held | all other | still | 56 | 21 | 21 | 1.00 | 0.39 | 4.05 |
| YR10m10-F-s0 | held | static | moving | 48 | 20 | 1 | 0.05 | 1.85 | 4.98 |
| YR10m10-F-s0 | held | static | off-heading | 2 | 1 | 0 | 0.00 | 3.39 | nan |
| YR10m10-F-s0 | held | static | still | 828 | 394 | 389 | 0.99 | 1.08 | 4.01 |
| YR10m10-F-s0 | train | all other | moving | 2916 | 239 | 52 | 0.22 | 0.40 | 4.00 |
| YR10m10-F-s0 | train | all other | off-heading | 174 | 26 | 25 | 0.96 | 0.18 | 4.06 |
| YR10m10-F-s0 | train | all other | still | 127 | 29 | 29 | 1.00 | 0.65 | 4.05 |
| YR10m10-F-s0 | train | static | moving | 176 | 67 | 10 | 0.15 | 1.97 | 4.00 |
| YR10m10-F-s0 | train | static | off-heading | 4 | 3 | 2 | 0.67 | 3.25 | 4.03 |
| YR10m10-F-s0 | train | static | still | 2080 | 1074 | 1041 | 0.97 | 1.15 | 4.00 |
| YR10m10-F-s0 | wait-set a | static | moving | 146 | 74 | 3 | 0.04 | 3.97 | 4.09 |
| starter | train | all other | moving | 7 | 0 | 0 | 0.00 | 0.00 | nan |
| starter | wait-set a | static | moving | 146 | 0 | 0 | 0.00 | 0.00 | nan |

## Wait set (decision 344), variant a (human never moves), by GT length

| tag | GT length (m) | windows | corridor zeros | tripped | median d at the trip |
|---|---|--:|--:|--:|--:|
| APY10m10-AB-s0 | [0.0, 0.05) | 102 | 0 | 90 | 4.98 |
| APY10m10-AB-s0 | [0.05, 0.25) | 43 | 3 | 31 | 4.84 |
| APY10m10-AB-s0 | [0.25, 0.5) | 1 | 0 | 1 | 4.09 |
| G0014 | [0.0, 0.05) | 102 | 0 | 15 | 4.41 |
| G0014 | [0.05, 0.25) | 43 | 1 | 7 | 4.59 |
| G0014 | [0.25, 0.5) | 1 | 0 | 1 | 6.11 |
| P0 | [0.0, 0.05) | 102 | 0 | 11 | 4.35 |
| P0 | [0.05, 0.25) | 43 | 1 | 7 | 5.13 |
| P0 | [0.25, 0.5) | 1 | 0 | 1 | 4.36 |
| RSG-L18-s0 | [0.0, 0.05) | 102 | 0 | 53 | 4.80 |
| RSG-L18-s0 | [0.05, 0.25) | 43 | 3 | 19 | 5.16 |
| RSG-L18-s0 | [0.25, 0.5) | 1 | 0 | 1 | 4.44 |
| YR10m10-F-s0 | [0.0, 0.05) | 102 | 0 | 54 | 5.22 |
| YR10m10-F-s0 | [0.05, 0.25) | 43 | 3 | 19 | 4.78 |
| YR10m10-F-s0 | [0.25, 0.5) | 1 | 0 | 1 | 5.23 |
| starter | [0.0, 0.05) | 102 | 0 | 0 | nan |
| starter | [0.05, 0.25) | 43 | 0 | 0 | nan |
| starter | [0.25, 0.5) | 1 | 0 | 0 | nan |

