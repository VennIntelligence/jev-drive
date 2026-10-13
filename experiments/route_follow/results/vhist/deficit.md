# Plan 2 s deficit (m): plan arc to 2 s - fed speed x 2 s, decisions at >= 1 m/s up to the first zero-score event

| label | scenes | decisions | median (decisions) | mean of rollout means |
|:--|--:|--:|--:|--:|
| RSG | 450 | 7217 | -0.258 | -0.257 |
| RSG+hold | 450 | 7214 | -0.068 | -0.036 |
| RSG+holdlead | 450 | 7205 | -0.069 | -0.138 |
| allV | 450 | 7225 | -0.106 | -0.080 |
| allV+holdlead | 450 | 7238 | -0.021 | -0.015 |
| allV+zero | 450 | 7361 | -0.176 | -0.161 |

| A - B (rollout means, common scenes) | scenes / logs | difference [log-clustered 95 % CI] |
|:--|--:|:--|
| RSG+holdlead:RSG | 450 / 39 | +0.119 [+0.079, +0.162] |
| RSG+holdlead:RSG+hold | 450 / 39 | -0.102 [-0.151, -0.062] |
| allV+holdlead:allV | 450 / 39 | +0.064 [+0.036, +0.098] |
| allV+holdlead:RSG | 450 / 39 | +0.241 [+0.200, +0.288] |
