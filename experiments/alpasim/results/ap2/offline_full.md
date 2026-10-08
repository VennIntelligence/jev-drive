Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `SH30` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| SH30 | NAVSIM standard, m = 4 | 0.570 |  | -0.03 | 90.43 |  |
| SH30 | AlpaSim standard, m = 4 | 0.585 |  | -0.10 | 90.14 |  |
| SH30 | AlpaSim standard, m = 3 | 0.601 |  | -0.04 | | |
| SH30 | AlpaSim standard, m = 2 | 0.677 |  | -0.48 | | |
| SH30 | AlpaSim standard, m = 1 | 0.992 |  | -1.10 | 85.14 |  |
| AP2 | NAVSIM standard, m = 4 | 0.585 | 0.014 [0.010, 0.019] | +0.02 | 89.98 | -0.45 [-0.91, -0.02] |
| AP2 | AlpaSim standard, m = 4 | 0.583 | -0.002 [-0.008, 0.003] | -0.01 | 90.28 | 0.14 [-0.38, 0.66] |
| AP2 | AlpaSim standard, m = 3 | 0.586 | -0.015 [-0.022, -0.008] | +0.01 | | |
| AP2 | AlpaSim standard, m = 2 | 0.600 | -0.077 [-0.091, -0.065] | +0.04 | | |
| AP2 | AlpaSim standard, m = 1 | 0.666 | -0.326 [-0.366, -0.292] | -0.14 | 88.33 | 3.19 [2.07, 4.37] |
| P | NAVSIM standard, m = 4 | 0.633 | 0.063 [0.054, 0.072] | -0.01 | 89.38 | -1.06 [-1.95, -0.27] |
| P | AlpaSim standard, m = 4 | 0.634 | 0.049 [0.040, 0.059] | -0.03 | 89.41 | -0.72 [-1.50, 0.01] |
| P | AlpaSim standard, m = 3 | 0.640 | 0.039 [0.030, 0.049] | -0.02 | | |
| P | AlpaSim standard, m = 2 | 0.662 | -0.015 [-0.030, -0.001] | -0.13 | | |
| P | AlpaSim standard, m = 1 | 0.749 | -0.243 [-0.280, -0.211] | -0.38 | 87.66 | 2.52 [1.31, 3.75] |
