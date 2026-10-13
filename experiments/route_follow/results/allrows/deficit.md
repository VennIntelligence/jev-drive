# Plan 2 s deficit (m): plan arc to 2 s - fed speed x 2 s, decisions at >= 1 m/s up to the first zero-score event

| label | scenes | decisions | median (decisions) | mean of rollout means |
|:--|--:|--:|--:|--:|
| RSG-L18-s0 | 1473 | 13255 | -0.433 | -0.190 |
| RSG-L18-s1 | 1473 | 13289 | -0.434 | -0.191 |
| RSGall-L18-s0 | 1472 | 13232 | -0.441 | -0.200 |
| RSGall-L18-s1 | 1473 | 13258 | -0.417 | -0.183 |
| RSGallV-L18-s0 | 1472 | 13232 | -0.260 | -0.062 |
| RSGallV-L18-s1 | 1473 | 13248 | -0.235 | -0.036 |
| RSG-L18 | 1473 | 26544 | -0.433 | -0.191 |
| RSGall-L18 | 1471 | 26490 | -0.428 | -0.193 |
| RSGallV-L18 | 1472 | 26480 | -0.247 | -0.049 |

| A - B (rollout means, common scenes) | scenes / logs | difference [log-clustered 95 % CI] |
|:--|--:|:--|
| RSGall-L18:RSG-L18 | 1471 / 136 | +0.001 [-0.004, +0.006] |
| RSGallV-L18:RSG-L18 | 1471 / 136 | +0.143 [+0.125, +0.162] |
| RSGallV-L18:RSGall-L18 | 1469 / 136 | +0.142 [+0.124, +0.162] |
