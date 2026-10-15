# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSGk26-L18-s0 | 0.8910 | 22 | 7 / 11 / 5 | 18 | 1.094 | 1.159 [0.816, 1.877] |
| RSGk51-L18-s0 | 0.8890 | 25 | 7 / 13 / 5 | 20 | 1.184 | 1.062 [0.705, 1.919] |
| RSGk51w1-L18-s0 | 0.8939 | 23 | 6 / 11 / 6 | 17 | 1.182 | 1.256 [0.841, 2.262] |
| RSG-L18-s1 | 0.8830 | 26 | 6 / 13 / 8 | 19 | 1.144 | 1.095 [0.712, 2.034] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSGk26-L18-s0 | +0.0101 [-0.0008, +0.0231] | -4.0 [-9.6, +1.2] | -3.0 [-8.8, +2.7] | -0.041 [-0.101, +0.030] | +0.171 [-0.210, +0.552] |
| RSGk51-L18-s0 | +0.0081 [-0.0044, +0.0215] | -1.0 [-6.4, +4.1] | -1.0 [-7.1, +5.9] | +0.049 [-0.050, +0.147] | +0.075 [-0.316, +0.522] |
| RSGk51w1-L18-s0 | +0.0130 [+0.0037, +0.0240] | -3.0 [-7.7, +1.1] | -4.0 [-8.6, +0.0] | +0.046 [-0.011, +0.105] | +0.268 [+0.026, +0.751] |
| RSG-L18-s1 | +0.0021 [-0.0097, +0.0137] | +0.0 [-4.7, +4.9] | -2.0 [-7.0, +2.6] | +0.009 [-0.022, +0.041] | +0.107 [-0.174, +0.590] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
