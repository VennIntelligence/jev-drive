Tokens: 11972 of 12146 navtest tokens with a logged future and a rebuilt route for every m; EPDMS (no EC, non-reactive traffic) on 2035 of them. Paired differences vs `P2H10` at the same m, bootstrap over logs.

| arm | inputs | ADE vs log (m) | vs ref [95% CI] | x error at 4 s (m) | EPDMS subset | vs ref [95% CI] |
|:--|:--|--:|:--|--:|--:|:--|
| P2H10 | NAVSIM standard, m = 4 | 0.553 |  | -0.03 | | |
| P2H10 | AlpaSim standard, m = 4 | 0.568 |  | -0.10 | 89.43 |  |
| P2H10 | AlpaSim standard, m = 3 | 0.584 |  | -0.04 | | |
| P2H10 | AlpaSim standard, m = 2 | 0.659 |  | -0.47 | | |
| P2H10 | AlpaSim standard, m = 1 | 0.975 |  | -1.11 | 85.00 |  |
| AP2H10-AB | NAVSIM standard, m = 4 | 0.567 | 0.014 [0.010, 0.018] | +0.03 | | |
| AP2H10-AB | AlpaSim standard, m = 4 | 0.565 | -0.003 [-0.008, 0.002] | -0.01 | 89.52 | 0.08 [-0.39, 0.56] |
| AP2H10-AB | AlpaSim standard, m = 3 | 0.567 | -0.017 [-0.023, -0.011] | +0.02 | | |
| AP2H10-AB | AlpaSim standard, m = 2 | 0.583 | -0.076 [-0.089, -0.064] | +0.04 | | |
| AP2H10-AB | AlpaSim standard, m = 1 | 0.649 | -0.327 [-0.368, -0.292] | -0.10 | 87.25 | 2.25 [1.16, 3.38] |
| OT10a05-F | NAVSIM standard, m = 4 | 0.556 | 0.003 [0.001, 0.005] | -0.03 | | |
| OT10a05-F | AlpaSim standard, m = 4 | 0.572 | 0.004 [0.002, 0.007] | -0.11 | 89.68 | 0.25 [-0.22, 0.79] |
| OT10a05-F | AlpaSim standard, m = 3 | 0.589 | 0.005 [0.002, 0.008] | -0.10 | | |
| OT10a05-F | AlpaSim standard, m = 2 | 0.664 | 0.005 [0.002, 0.010] | -0.48 | | |
| OT10a05-F | AlpaSim standard, m = 1 | 0.977 | 0.001 [-0.004, 0.007] | -1.05 | 83.71 | -1.28 [-2.23, -0.45] |
| OT10a15-F | NAVSIM standard, m = 4 | 0.562 | 0.009 [0.006, 0.012] | -0.02 | | |
| OT10a15-F | AlpaSim standard, m = 4 | 0.582 | 0.014 [0.011, 0.018] | -0.10 | 89.98 | 0.54 [-0.05, 1.24] |
| OT10a15-F | AlpaSim standard, m = 3 | 0.600 | 0.016 [0.012, 0.020] | -0.08 | | |
| OT10a15-F | AlpaSim standard, m = 2 | 0.670 | 0.011 [0.006, 0.016] | -0.47 | | |
| OT10a15-F | AlpaSim standard, m = 1 | 0.982 | 0.007 [-0.005, 0.018] | -1.02 | 82.22 | -2.78 [-4.15, -1.55] |
