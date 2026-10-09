Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `P2H10` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| P2H10 | NAVSIM standard, m = 4 | 0.553 |  | -0.03 | | |
| P2H10 | AlpaSim standard, m = 4 | 0.568 |  | -0.10 | 89.43 |  |
| P2H10 | AlpaSim standard, m = 3 | 0.584 |  | -0.04 | | |
| P2H10 | AlpaSim standard, m = 2 | 0.659 |  | -0.47 | | |
| P2H10 | AlpaSim standard, m = 1 | 0.975 |  | -1.11 | 85.00 |  |
| APY10m10-AB | NAVSIM standard, m = 4 | 0.574 | 0.021 [0.016, 0.025] | -0.01 | | |
| APY10m10-AB | AlpaSim standard, m = 4 | 0.571 | 0.004 [-0.001, 0.008] | -0.03 | 89.80 | 0.37 [-0.19, 0.96] |
| APY10m10-AB | AlpaSim standard, m = 3 | 0.574 | -0.010 [-0.017, -0.004] | -0.01 | | |
| APY10m10-AB | AlpaSim standard, m = 2 | 0.586 | -0.073 [-0.087, -0.061] | +0.03 | | |
| APY10m10-AB | AlpaSim standard, m = 1 | 0.654 | -0.321 [-0.363, -0.285] | -0.13 | 86.96 | 1.97 [0.83, 3.11] |
