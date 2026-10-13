# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## all10: 1479 scenes, 136 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.9473 | 46 | 8 / 31 / 8 | 39 | 0.792 | 1.011 [0.724, 1.568] |
| RSG-L18-s1 | 0.9487 | 44 | 8 / 30 / 7 | 38 | 0.799 | 1.040 [0.731, 1.659] |
| full6 | 0.9398 | 52 | 10 / 36 / 7 | 46 | 0.650 | 0.856 [0.591, 1.402] |
| gate | 0.9436 | 47 | 8 / 33 / 6 | 41 | 0.708 | 0.961 [0.702, 1.463] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s1 | +0.0014 [-0.0029, +0.0052] | -2.0 [-7.3, +3.9] | -1.0 [-6.0, +4.4] | +0.007 [-0.006, +0.021] | +0.029 [-0.127, +0.229] |
| full6 | -0.0075 [-0.0159, +0.0012] | +6.0 [-6.8, +18.1] | +7.0 [-6.1, +19.2] | -0.142 [-0.171, -0.116] | -0.156 [-0.476, +0.169] |
| gate | -0.0037 [-0.0079, +0.0001] | +1.0 [-4.1, +6.5] | +2.0 [-3.8, +7.4] | -0.084 [-0.109, -0.060] | -0.050 [-0.234, +0.094] |
| gate | -0.0037 [-0.0079, +0.0001] | +1.0 [-4.1, +6.5] | +2.0 [-3.8, +7.4] | -0.084 [-0.109, -0.060] | -0.050 [-0.234, +0.094] |
| full6 | -0.0075 [-0.0159, +0.0012] | +6.0 [-6.8, +18.1] | +7.0 [-6.1, +19.2] | -0.142 [-0.171, -0.116] | -0.156 [-0.476, +0.169] |
| RSG-L18-s1 | +0.0014 [-0.0029, +0.0052] | -2.0 [-7.3, +3.9] | -1.0 [-6.0, +4.4] | +0.007 [-0.006, +0.021] | +0.029 [-0.127, +0.229] |

Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below RSG-L18-s0 on the public mean: {'RSG-L18-s1': True, 'full6': True, 'gate': True}.

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
