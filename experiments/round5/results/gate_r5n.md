# ROUND3 PAI-FIX native (VC + LEAD) lines against CTRL R2C-L18-s0

| arm | G-LAUNCH arm / CTRL (diff, CI) | T281 route-turn zeros arm / CTRL | TARGET arm / CTRL (diff, CI) | launched G-LAUNCH | pass |
|:--|:--|:--|:--|:--|:--|
| R5N-L18-s0 | 0.670 / 0.865 (-0.195 [-0.497, +0.005]) FAIL | 3 / 12 ok | 0.532 / 0.523 (+0.008 [-0.113, +0.131]) ok | 4 / 5 of 6 | fail |
| R3MV-L18-s0 | 0.569 / 0.865 (-0.295 [-0.585, -0.035]) FAIL | 4 / 12 ok | 0.574 / 0.523 (+0.051 [-0.044, +0.149]) ok | 3 / 5 of 6 | fail |

Class guard: zeros per class over STOP + STOP-L + TURN + G-LAUNCH + G-LEAD (scenes both reads scored without RPC), arm <= CTRL + 3; G-LAUNCH rear-hit partials.

| arm | overshoot | route-turn | offroad | AF-front | side | zero-other | G-LAUNCH rear-hit arm / CTRL | guard | all four |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|
| R5N-L18-s0 | 8 / 13 | 1 / 5 | 7 / 3 FAIL | 4 / 4 | 0 / 0 | 2 / 0 | 2 / 1 | FAIL | fail |
| R3MV-L18-s0 | 10 / 13 | 1 / 5 | 4 / 3 | 2 / 4 | 1 / 0 | 1 / 0 | 3 / 1 | ok | fail |
