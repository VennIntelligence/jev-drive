Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `SHP` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| SH30 | NAVSIM standard, m = 4 | 0.570 | -0.060 [-0.068, -0.052] | -0.03 | 90.43 | 1.18 [0.42, 1.98] |
| SH30 | AlpaSim standard, m = 4 | 0.585 | -0.049 [-0.057, -0.041] | -0.10 | 90.14 | 0.88 [0.13, 1.64] |
| SH30 | AlpaSim standard, m = 3 | 0.601 | -0.054 [-0.063, -0.046] | -0.04 | | |
| SH30 | AlpaSim standard, m = 2 | 0.677 | -0.061 [-0.072, -0.050] | -0.48 | | |
| SH30 | AlpaSim standard, m = 1 | 0.992 | -0.031 [-0.043, -0.019] | -1.10 | | |
| SHP | NAVSIM standard, m = 4 | 0.630 |  | +0.02 | 89.25 |  |
| SHP | AlpaSim standard, m = 4 | 0.634 |  | +0.01 | 89.26 |  |
| SHP | AlpaSim standard, m = 3 | 0.655 |  | -0.05 | | |
| SHP | AlpaSim standard, m = 2 | 0.738 |  | -0.57 | | |
| SHP | AlpaSim standard, m = 1 | 1.024 |  | -0.80 | | |
| N0 | NAVSIM standard, m = 4 | 0.630 | 0.000 [0.000, 0.000] | +0.02 | | |
| N0 | AlpaSim standard, m = 4 | 0.634 | 0.000 [0.000, 0.000] | +0.01 | | |
| N0 | AlpaSim standard, m = 3 | 0.658 | 0.003 [-0.004, 0.011] | +0.03 | | |
| N0 | AlpaSim standard, m = 2 | 0.874 | 0.136 [0.116, 0.157] | -0.01 | | |
| N0 | AlpaSim standard, m = 1 | 5.010 | 3.987 [3.731, 4.240] | +4.29 | | |
| A | NAVSIM standard, m = 4 | 0.667 | 0.037 [0.030, 0.043] | -0.02 | | |
| A | AlpaSim standard, m = 4 | 0.649 | 0.016 [0.011, 0.020] | -0.06 | 89.49 | 0.23 [-0.37, 0.79] |
| A | AlpaSim standard, m = 3 | 0.649 | -0.006 [-0.014, 0.002] | -0.01 | | |
| A | AlpaSim standard, m = 2 | 0.687 | -0.051 [-0.067, -0.034] | -0.11 | | |
| A | AlpaSim standard, m = 1 | 0.943 | -0.080 [-0.130, -0.035] | -0.27 | | |
| AR | AlpaSim standard, m = 4 | 0.663 | 0.029 [0.022, 0.036] | -0.08 | 89.60 | 0.34 [-0.42, 1.01] |
| AR | AlpaSim standard, m = 3 | 0.664 | 0.009 [-0.001, 0.019] | -0.03 | | |
| AR | AlpaSim standard, m = 2 | 0.717 | -0.021 [-0.038, -0.004] | -0.17 | | |
| AR | AlpaSim standard, m = 1 | 1.094 | 0.071 [0.028, 0.110] | -0.37 | | |
| AB | NAVSIM standard, m = 4 | 0.633 | 0.003 [-0.000, 0.006] | -0.01 | | |
| AB | AlpaSim standard, m = 4 | 0.634 | 0.001 [-0.003, 0.004] | -0.03 | 89.41 | 0.15 [-0.36, 0.65] |
| AB | AlpaSim standard, m = 3 | 0.640 | -0.015 [-0.020, -0.010] | -0.02 | | |
| AB | AlpaSim standard, m = 2 | 0.662 | -0.076 [-0.086, -0.065] | -0.13 | | |
| AB | AlpaSim standard, m = 1 | 0.749 | -0.274 [-0.306, -0.247] | -0.38 | | |
