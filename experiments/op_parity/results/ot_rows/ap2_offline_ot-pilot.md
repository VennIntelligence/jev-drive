Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `SHP` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| SHP | NAVSIM standard, m = 4 | 0.630 |  | +0.02 | | |
| SHP | AlpaSim standard, m = 4 | 0.634 |  | +0.01 | 89.26 |  |
| SHP | AlpaSim standard, m = 3 | 0.655 |  | -0.05 | | |
| SHP | AlpaSim standard, m = 2 | 0.738 |  | -0.57 | | |
| SHP | AlpaSim standard, m = 1 | 1.024 |  | -0.80 | 85.51 |  |
| OTP | NAVSIM standard, m = 4 | 0.636 | 0.005 [0.003, 0.008] | -0.03 | | |
| OTP | AlpaSim standard, m = 4 | 0.640 | 0.006 [0.003, 0.009] | -0.02 | 89.63 | 0.37 [-0.17, 0.89] |
| OTP | AlpaSim standard, m = 3 | 0.661 | 0.006 [0.003, 0.009] | -0.12 | | |
| OTP | AlpaSim standard, m = 2 | 0.753 | 0.015 [0.011, 0.019] | -0.67 | | |
| OTP | AlpaSim standard, m = 1 | 1.020 | -0.004 [-0.008, 0.001] | -0.82 | 85.05 | -0.47 [-1.11, 0.12] |
