# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| R-own | 0.8781 | 27 | 5 / 17 / 6 | 22 | 1.148 | 0.940 [0.630, 1.636] |
| R-log-priv | 0.9328 | 30 | 8 / 18 / 6 | 25 | 0.916 | 0.979 [0.638, 1.688] |
| R-bridge | 0.7404 | 90 | 12 / 74 / 7 | 86 | 1.128 | 0.218 [0.160, 0.315] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| R-own | -0.0028 [-0.0127, +0.0058] | +1.0 [-3.2, +5.7] | +1.0 [-3.2, +5.7] | +0.013 [-0.018, +0.043] | -0.048 [-0.340, +0.177] |
| R-log-priv | +0.0520 [+0.0298, +0.0737] | +4.0 [-4.7, +12.8] | +4.0 [-3.5, +12.0] | -0.220 [-0.322, -0.111] | -0.009 [-0.433, +0.421] |
| R-bridge | -0.1405 [-0.1800, -0.0983] | +64.0 [+45.0, +82.0] | +65.0 [+45.5, +84.6] | -0.007 [-0.181, +0.148] | -0.770 [-1.459, -0.478] |
| R-bridge - R-own | -0.1376 [-0.1804, -0.0922] | +63.0 [+42.8, +81.9] | +64.0 [+43.4, +84.2] | -0.020 [-0.184, +0.130] | -0.722 [-1.369, -0.434] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
