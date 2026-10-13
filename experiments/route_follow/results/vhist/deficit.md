# Plan 2 s deficit (m): plan arc to 2 s - fed speed x 2 s, decisions at >= 1 m/s up to the first zero-score event

| label | scenes | decisions | median (decisions) | mean of rollout means |
|:--|--:|--:|--:|--:|
| RSG | 450 | 7217 | -0.258 | -0.257 |
| RSG+hold | 450 | 7214 | -0.068 | -0.036 |
| allV | 450 | 7225 | -0.106 | -0.080 |
| allV+zero | 450 | 7361 | -0.176 | -0.161 |

| A - B (rollout means, common scenes) | scenes / logs | difference [log-clustered 95 % CI] |
|:--|--:|:--|
| RSG+hold:RSG | 450 / 39 | +0.221 [+0.168, +0.284] |
| allV+zero:allV | 450 / 39 | -0.081 [-0.126, -0.040] |
| allV:RSG | 450 / 39 | +0.177 [+0.146, +0.211] |
| allV+zero:RSG | 450 / 39 | +0.096 [+0.055, +0.139] |
