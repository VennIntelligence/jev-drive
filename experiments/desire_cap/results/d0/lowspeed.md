# DESIRE-D0 low-speed read (open loop, dc.py protocol)

Response R_y(4 s) = the plan's lateral at 4 s with the lane-change desire (pulse age 0) minus without, toward the desire side; mean [log / scene cluster bootstrap 95 % CI], dir = share > 0.1 m. Stage = row mean of its seeds (RSG-L18: s0, s1). Line: mean >= 0.5 m and dir >= 0.95 (n >= 10 rows) = the pathway exists in that bin and side.

## NAV (C + LC rows)

| stage | bin (m/s) | rows / groups | left: mean [CI] | dir | right: mean [CI] | dir | left exists | right exists | ages 0.4 / 1 / 2 s (pooled) |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|:--|
| P0 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| P0 | 3-6 | 128 / 40 | 1.03 [0.82, 1.25] | 0.96 | 0.85 [0.71, 1.00] | 0.91 | True | False | 0.70 / 0.54 / 0.41 |
| P0 | 6-8.9 | 503 / 83 | 1.09 [0.98, 1.22] | 0.98 | 0.66 [0.57, 0.77] | 0.90 | True | False | 0.67 / 0.50 / 0.39 |
| P0 | 8.9-99 | 635 / 93 | 1.16 [1.02, 1.34] | 0.99 | 1.03 [0.88, 1.20] | 0.96 | True | True | 0.88 / 0.69 / 0.52 |
| RSG-L18 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| RSG-L18 | 3-6 | 128 / 40 | 0.19 [0.12, 0.26] | 0.57 | 0.72 [0.56, 0.86] | 0.82 | False | False | 0.45 / 0.34 / 0.22 |
| RSG-L18 | 6-8.9 | 503 / 83 | 0.15 [0.12, 0.19] | 0.47 | 0.72 [0.63, 0.82] | 0.78 | False | False | 0.43 / 0.32 / 0.22 |
| RSG-L18 | 8.9-99 | 635 / 93 | 0.15 [0.11, 0.20] | 0.35 | 1.32 [1.16, 1.50] | 0.88 | False | False | 0.71 / 0.54 / 0.37 |

## NAV C (cruise, human keeps lane)

| stage | bin (m/s) | rows / groups | left: mean [CI] | dir | right: mean [CI] | dir | left exists | right exists | ages 0.4 / 1 / 2 s (pooled) |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|:--|
| P0 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| P0 | 3-6 | 79 / 33 | 0.93 [0.71, 1.20] | 0.97 | 0.76 [0.63, 0.89] | 0.95 | True | False | 0.59 / 0.42 / 0.29 |
| P0 | 6-8.9 | 395 / 78 | 1.02 [0.92, 1.14] | 0.99 | 0.63 [0.54, 0.73] | 0.95 | True | False | 0.60 / 0.43 / 0.32 |
| P0 | 8.9-99 | 543 / 89 | 1.09 [0.95, 1.27] | 0.99 | 0.98 [0.83, 1.16] | 0.97 | True | True | 0.81 / 0.61 / 0.43 |
| RSG-L18 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| RSG-L18 | 3-6 | 79 / 33 | 0.12 [0.06, 0.19] | 0.51 | 0.59 [0.45, 0.73] | 0.81 | False | False | 0.33 / 0.22 / 0.11 |
| RSG-L18 | 6-8.9 | 395 / 78 | 0.11 [0.09, 0.14] | 0.41 | 0.70 [0.60, 0.80] | 0.78 | False | False | 0.38 / 0.26 / 0.15 |
| RSG-L18 | 8.9-99 | 543 / 89 | 0.10 [0.08, 0.14] | 0.29 | 1.27 [1.11, 1.45] | 0.87 | False | False | 0.64 / 0.46 / 0.29 |

## NAV LC (human changes lane)

| stage | bin (m/s) | rows / groups | left: mean [CI] | dir | right: mean [CI] | dir | left exists | right exists | ages 0.4 / 1 / 2 s (pooled) |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|:--|
| P0 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| P0 | 3-6 | 56 / 16 | 1.21 [0.93, 1.48] | 0.93 | 0.96 [0.67, 1.24] | 0.82 | False | False | 0.85 / 0.72 / 0.60 |
| P0 | 6-8.9 | 124 / 28 | 1.28 [1.06, 1.54] | 0.95 | 0.80 [0.60, 1.04] | 0.76 | True | False | 0.88 / 0.76 / 0.65 |
| P0 | 8.9-99 | 105 / 21 | 1.59 [1.30, 1.87] | 0.98 | 1.24 [1.01, 1.48] | 0.90 | True | False | 1.26 / 1.12 / 0.97 |
| RSG-L18 | 0-3 | 0 / 0 | nan [nan, nan] | nan | nan [nan, nan] | nan | False | False | - / - / - |
| RSG-L18 | 3-6 | 56 / 16 | 0.30 [0.20, 0.42] | 0.70 | 0.88 [0.55, 1.13] | 0.80 | False | False | 0.63 / 0.53 / 0.40 |
| RSG-L18 | 6-8.9 | 124 / 28 | 0.27 [0.18, 0.36] | 0.66 | 0.84 [0.68, 1.03] | 0.77 | False | False | 0.62 / 0.55 / 0.46 |
| RSG-L18 | 8.9-99 | 105 / 21 | 0.45 [0.34, 0.56] | 0.76 | 1.54 [1.29, 1.84] | 0.89 | False | False | 1.08 / 0.99 / 0.85 |

## PAI rule rows (route_desire fires; desire on the route side)

| stage | bin (m/s) | rows / groups | left: mean [CI] | dir | right: mean [CI] | dir | left exists | right exists | ages 0.4 / 1 / 2 s (pooled) |
|:--|:--|--:|:--|--:|:--|--:|:--|:--|:--|
| P0 | 0-3 | 279 / 12 | 1.10 [0.82, 1.40] | 1.00 | 0.43 [0.15, 0.57] | 0.92 | True | False | 0.66 / 0.58 / 0.55 |
| P0 | 3-6 | 285 / 13 | 1.27 [1.00, 1.48] | 1.00 | 0.74 [0.27, 1.19] | 0.93 | True | False | 0.79 / 0.67 / 0.58 |
| P0 | 6-8.9 | 134 / 10 | 1.43 [0.77, 2.27] | 1.00 | 1.32 [0.59, 1.62] | 1.00 | True | True | 1.20 / 1.00 / 0.87 |
| P0 | 8.9-99 | 1418 / 36 | 1.62 [1.22, 1.92] | 0.99 | 0.95 [0.61, 1.35] | 0.89 | True | False | 1.01 / 0.88 / 0.73 |
| RSG-L18 | 0-3 | 279 / 12 | 0.39 [0.27, 0.57] | 0.95 | 0.54 [-0.15, 1.00] | 0.74 | False | False | 0.38 / 0.33 / 0.32 |
| RSG-L18 | 3-6 | 285 / 13 | 0.47 [0.13, 0.91] | 0.83 | 0.59 [-0.08, 1.48] | 0.59 | False | False | 0.59 / 0.49 / 0.35 |
| RSG-L18 | 6-8.9 | 134 / 10 | 0.51 [0.10, 1.29] | 0.78 | 1.33 [-0.02, 1.89] | 0.71 | False | False | 0.79 / 0.67 / 0.58 |
| RSG-L18 | 8.9-99 | 1418 / 36 | 0.97 [0.55, 1.24] | 0.96 | 1.30 [0.87, 1.83] | 0.88 | True | False | 1.22 / 0.96 / 0.78 |

## Registered lines

- P0 low-speed pathway (both families): 3-6: left True, right False, 6-8.9: left True, right False
- RSG-L18 low-speed pathway (both families): 3-6: left False, right False, 6-8.9: left False, right False
- **Speed gate for D0: v_min = 8.9 m/s** (RSG-L18 right side, NAV (C + LC rows) and PAI rule rows both exist from this bin up; else 8.9)
