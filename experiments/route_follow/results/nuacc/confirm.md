# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0-re1016

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0-re1016 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSG-L18-s0-acc | 0.8452 | 43 | 7 / 22 / 16 | 29 | 2.465 | 0.827 [0.592, 1.313] |
| RSG-L18-s0-vl | 0.8364 | 25 | 5 / 13 / 8 | 18 | 0.974 | 1.108 [0.667, 2.483] |
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |

| a - RSG-L18-s0-re1016 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0-acc | -0.0357 [-0.0654, -0.0079] | +17.0 [+8.0, +25.9] | +8.0 [+0.0, +15.1] | +1.330 [+1.056, +1.577] | -0.161 [-0.606, +0.111] |
| RSG-L18-s0-vl | -0.0445 [-0.0724, -0.0193] | -1.0 [-7.3, +4.6] | -3.0 [-8.7, +2.0] | -0.161 [-0.248, -0.077] | +0.120 [-0.131, +0.953] |
| RSG-L18-s0 | +0.0000 [+0.0000, +0.0000] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| RSG-L18-s0-acc - RSG-L18-s0-vl | +0.0088 [-0.0137, +0.0319] | +18.0 [+9.0, +27.2] | +11.0 [+3.3, +17.7] | +1.491 [+1.233, +1.719] | -0.281 [-1.283, +0.040] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
