# ROUND3 PAI-FIX native (VC + LEAD) lines against CTRL R2C-L18-s0

| arm | G-LAUNCH arm / CTRL (diff, CI) | T281 route-turn zeros arm / CTRL | TARGET arm / CTRL (diff, CI) | launched G-LAUNCH | pass |
|:--|:--|:--|:--|:--|:--|
| R5A-L18-s0 | 0.328 / 0.865 (-0.537 [-0.848, -0.215]) FAIL | 2 / 12 ok | 0.546 / 0.523 (+0.023 [-0.093, +0.138]) ok | 2 / 5 of 6 | fail |
| R5A-L18-s1 | 0.319 / 0.865 (-0.546 [-0.847, -0.233]) FAIL | 2 / 12 ok | 0.466 / 0.523 (-0.057 [-0.171, +0.055]) FAIL | 2 / 5 of 6 | fail |
| R3MV-L18-s0 | 0.569 / 0.865 (-0.295 [-0.585, -0.035]) FAIL | 4 / 12 ok | 0.574 / 0.523 (+0.051 [-0.044, +0.149]) ok | 3 / 5 of 6 | fail |

Class guard: zeros per class over STOP + STOP-L + TURN + G-LAUNCH + G-LEAD (scenes both reads scored without RPC), arm <= CTRL + 3; G-LAUNCH rear-hit partials.

| arm | overshoot | route-turn | offroad | AF-front | side | zero-other | G-LAUNCH rear-hit arm / CTRL | guard | all four |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|
| R5A-L18-s0 | 9 / 13 | 1 / 5 | 4 / 3 | 5 / 4 | 2 / 0 | 1 / 0 | 5 / 1 | ok | fail |
| R5A-L18-s1 | 13 / 13 | 2 / 5 | 6 / 3 | 4 / 4 | 0 / 0 | 1 / 0 | 5 / 1 | ok | fail |
| R3MV-L18-s0 | 10 / 13 | 1 / 5 | 4 / 3 | 2 / 4 | 1 / 0 | 1 / 0 | 3 / 1 | ok | fail |
