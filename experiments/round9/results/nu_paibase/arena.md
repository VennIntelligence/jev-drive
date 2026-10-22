# nuPlan arena held, native: NPB (PAI-mixed base + B2 recipe) vs B2 / JTN-L18-s1: score-only (arena, facts cache)

Arena held windows read by both, route-fail windows dropped (free_m arena's rule); log-clustered bootstrap. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| NPB70-L18-s0 | STANDFMXH-L18-s0 | 2228 | 0.9419 / 0.9373 | +0.0046 [-0.0054, +0.0154] | 80 / 92 | 80 / 89 |
| NPB70-L18-s0 | STANDFMXH-L18-s1 | 2228 | 0.9419 / 0.9358 | +0.0061 [-0.0058, +0.0190] | 80 / 101 | 80 / 99 |
| NPB70-L18-s0 | JTN-L18-s1 | 2228 | 0.9419 / 0.8909 | +0.0509 [+0.0247, +0.0790] | 80 / 210 | 80 / 206 |
| NPB70-L18-s1 | STANDFMXH-L18-s0 | 2228 | 0.9307 / 0.9373 | -0.0065 [-0.0192, +0.0056] | 110 / 92 | 108 / 89 |
| NPB70-L18-s1 | STANDFMXH-L18-s1 | 2228 | 0.9307 / 0.9358 | -0.0050 [-0.0178, +0.0066] | 110 / 101 | 108 / 99 |
| NPB70-L18-s1 | JTN-L18-s1 | 2228 | 0.9307 / 0.8909 | +0.0398 [+0.0157, +0.0663] | 110 / 210 | 108 / 206 |
| NPB30-L18-s0 | STANDFMXH-L18-s0 | 2228 | 0.9351 / 0.9373 | -0.0021 [-0.0121, +0.0087] | 96 / 92 | 94 / 89 |
| NPB30-L18-s0 | STANDFMXH-L18-s1 | 2228 | 0.9351 / 0.9358 | -0.0006 [-0.0107, +0.0098] | 96 / 101 | 94 / 99 |
| NPB30-L18-s0 | JTN-L18-s1 | 2228 | 0.9351 / 0.8909 | +0.0442 [+0.0207, +0.0690] | 96 / 210 | 94 / 206 |
| NPB30-L18-s1 | STANDFMXH-L18-s0 | 2228 | 0.9309 / 0.9373 | -0.0063 [-0.0176, +0.0044] | 111 / 92 | 107 / 89 |
| NPB30-L18-s1 | STANDFMXH-L18-s1 | 2228 | 0.9309 / 0.9358 | -0.0048 [-0.0161, +0.0054] | 111 / 101 | 107 / 99 |
| NPB30-L18-s1 | JTN-L18-s1 | 2228 | 0.9309 / 0.8909 | +0.0400 [+0.0195, +0.0623] | 111 / 210 | 107 / 206 |

Zero classes (held windows):

| class | NPB70-L18-s0 | NPB70-L18-s1 | NPB30-L18-s0 | NPB30-L18-s1 | STANDFMXH-L18-s0 | STANDFMXH-L18-s1 | JTN-L18-s1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| corridor | 32 | 51 | 44 | 55 | 35 | 43 | 145 |
| offroad | 46 | 56 | 48 | 51 | 51 | 53 | 55 |
| route-fail | 18 | 18 | 18 | 18 | 18 | 18 | 18 |
| AF-front | 0 | 2 | 2 | 4 | 3 | 2 | 4 |
| offroad+corridor | 1 | 0 | 1 | 0 | 2 | 2 | 3 |
| AF-side | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| AF-front+corridor | 1 | 0 | 0 | 0 | 0 | 0 | 3 |
| AF-side+corridor | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
