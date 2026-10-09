Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `P2H10` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| P2H10 | NAVSIM standard, m = 4 | 0.553 |  | -0.03 | | |
| P2H10 | AlpaSim standard, m = 4 | 0.568 |  | -0.10 | 89.43 |  |
| P2H10 | AlpaSim standard, m = 3 | 0.584 |  | -0.04 | | |
| P2H10 | AlpaSim standard, m = 2 | 0.659 |  | -0.47 | | |
| P2H10 | AlpaSim standard, m = 1 | 0.975 |  | -1.11 | 85.00 |  |
| YR10m10-F | NAVSIM standard, m = 4 | 0.558 | 0.005 [0.003, 0.008] | -0.03 | | |
| YR10m10-F | AlpaSim standard, m = 4 | 0.578 | 0.010 [0.007, 0.013] | -0.09 | 89.96 | 0.53 [-0.07, 1.19] |
| YR10m10-F | AlpaSim standard, m = 3 | 0.593 | 0.009 [0.005, 0.013] | -0.08 | | |
| YR10m10-F | AlpaSim standard, m = 2 | 0.661 | 0.002 [-0.003, 0.007] | -0.49 | | |
| YR10m10-F | AlpaSim standard, m = 1 | 0.959 | -0.016 [-0.022, -0.011] | -1.04 | 83.94 | -1.06 [-2.11, -0.08] |
| YR10m25-F | NAVSIM standard, m = 4 | 0.568 | 0.015 [0.012, 0.018] | -0.03 | | |
| YR10m25-F | AlpaSim standard, m = 4 | 0.597 | 0.030 [0.025, 0.035] | -0.07 | 89.96 | 0.52 [-0.25, 1.33] |
| YR10m25-F | AlpaSim standard, m = 3 | 0.610 | 0.026 [0.020, 0.031] | -0.05 | | |
| YR10m25-F | AlpaSim standard, m = 2 | 0.669 | 0.011 [0.005, 0.017] | -0.51 | | |
| YR10m25-F | AlpaSim standard, m = 1 | 0.950 | -0.025 [-0.036, -0.015] | -1.10 | 83.44 | -1.56 [-2.83, -0.41] |
