Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `SH30` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| SH30 | NAVSIM standard, m = 4 | 0.570 |  | -0.03 | | |
| SH30 | AlpaSim standard, m = 4 | 0.585 |  | -0.10 | 90.14 |  |
| SH30 | AlpaSim standard, m = 3 | 0.601 |  | -0.04 | 89.59 |  |
| SH30 | AlpaSim standard, m = 2 | 0.677 |  | -0.48 | 88.38 |  |
| SH30 | AlpaSim standard, m = 1 | 0.992 |  | -1.10 | 85.14 |  |
| OT30 | NAVSIM standard, m = 4 | 0.576 | 0.005 [0.003, 0.008] | -0.03 | | |
| OT30 | AlpaSim standard, m = 4 | 0.589 | 0.005 [0.002, 0.008] | -0.09 | 90.58 | 0.44 [-0.04, 0.90] |
| OT30 | AlpaSim standard, m = 3 | 0.605 | 0.004 [0.001, 0.007] | -0.07 | 89.78 | 0.19 [-0.29, 0.65] |
| OT30 | AlpaSim standard, m = 2 | 0.677 | -0.000 [-0.004, 0.004] | -0.45 | 88.33 | -0.06 [-0.56, 0.49] |
| OT30 | AlpaSim standard, m = 1 | 0.995 | 0.002 [-0.003, 0.008] | -1.05 | 83.65 | -1.49 [-2.69, -0.46] |
| OT30s1 | NAVSIM standard, m = 4 | 0.575 | 0.004 [0.002, 0.007] | -0.04 | | |
| OT30s1 | AlpaSim standard, m = 4 | 0.588 | 0.003 [0.000, 0.006] | -0.09 | | |
| OT30s1 | AlpaSim standard, m = 3 | 0.604 | 0.003 [0.000, 0.006] | -0.06 | | |
| OT30s1 | AlpaSim standard, m = 2 | 0.682 | 0.005 [0.001, 0.008] | -0.48 | | |
| OT30s1 | AlpaSim standard, m = 1 | 0.989 | -0.003 [-0.008, 0.002] | -1.07 | | |
