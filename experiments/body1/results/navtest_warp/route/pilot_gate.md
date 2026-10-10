| Pilot gate (hold logs and navtest, shards s2 + s3) | `P2H10R-Pb25-s0` against `P2H10-P-s0` | line | verdict |
|:--|:--|:--|:-:|
| own-plan agent-contact rate | 0.0156 against 0.0261: 40.2 %, -0.0105 [-0.0142, -0.0071] | >= 30 %, interval excluding 0 | met |
| own-plan boundary rate (NAVSIM raster) | 0.0182 against 0.0275: 33.8 %, -0.0093 [-0.0142, -0.0044] | >= 25 %, interval excluding 0 | met |
| dev ADE at step 3 000 (m) | 0.5949 against 0.5913 | <= + 0.01 | met |
| continuation slope alpha_05 | 0.851 against 1.079 | <= + 0.05 | met |
| 4 s arc ratio, hold states pooled (n = 5062) | 1.0001 [0.9991, 1.0013] | >= 0.995 | met |
| 4 s arc ratio, hold open states (n = 2263) | 1.0006 [0.9994, 1.0019] | >= 0.995 | met |
| (C-a) W2, navtest turn tokens > 45 deg (n = 1517) | 138 against 110 | <= reference + 5 | **NOT met** |
| - control `P2H10S-P-s0` (must fail C-a, C-b) / "A on on-log rows only" (must pass) | 141 against 110 (NOT met) / 115 against 110 (met) | | |
| (C-b) W2, on-log hold turn rows (n = 207) | 12 against 10 | <= reference + 1 | **NOT met** |
| - control `P2H10S-P-s0` (must fail C-a, C-b) / "A on on-log rows only" (must pass) | 15 against 10 (NOT met) / 9 against 10 (met) | | |
| (C-c) leaves the 4 m tube, navtest turn tokens | 19 against 25 | <= reference + 5 | met |
| - control `P2H10S-P-s0` (must fail C-a, C-b) / "A on on-log rows only" (must pass) | 26 against 25 (met) / 24 against 25 (met) | | |
| (C-c) leaves the 4 m tube, hold turn rows, four families (n = 686) | 10 against 27 | <= reference + 1 | met |
| - control `P2H10S-P-s0` (must fail C-a, C-b) / "A on on-log rows only" (must pass) | 25 against 27 (met) / 28 against 27 (met) | | |
| (reported) mean lateral at 4 s on turn rows, + = outside (m): navtest / hold | +0.432 against +0.308 / +0.285 against +0.088 | | |
| (reported) 4 s arc ratio, navtest pooled / open | 0.9993 [0.9987, 1.0000] / 1.0000 [0.9991, 1.0007] | | |

Pilot gate: NOT met
