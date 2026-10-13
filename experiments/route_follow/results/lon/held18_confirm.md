# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSGbothLon-L18-s0 | 0.9194 | 31 | 5 / 18 / 8 | 23 | 1.646 | 1.061 [0.669, 2.044] |
| RSGboth-L18-s0 | 0.9179 | 33 | 7 / 17 / 9 | 24 | 1.776 | 1.033 [0.705, 1.807] |
| RSGbC2-L18-s0 | 0.9132 | 32 | 6 / 18 / 8 | 24 | 1.294 | 0.974 [0.609, 1.947] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSGbothLon-L18-s0 | +0.0385 [+0.0161, +0.0627] | +5.0 [-2.3, +11.7] | +2.0 [-5.3, +9.1] | +0.511 [+0.371, +0.666] | +0.073 [-0.274, +0.646] |
| RSGboth-L18-s0 | +0.0370 [+0.0163, +0.0622] | +7.0 [+0.0, +13.5] | +3.0 [-4.1, +9.6] | +0.641 [+0.479, +0.815] | +0.045 [-0.357, +0.505] |
| RSGbC2-L18-s0 | +0.0323 [+0.0105, +0.0565] | +6.0 [-2.8, +14.0] | +3.0 [-5.3, +10.7] | +0.158 [+0.071, +0.248] | -0.014 [-0.364, +0.597] |
| RSGbothLon-L18-s0 - RSGboth-L18-s0 | +0.0014 [-0.0119, +0.0130] | -2.0 [-7.1, +3.8] | -1.0 [-5.9, +4.5] | -0.130 [-0.184, -0.077] | +0.028 [-0.213, +0.460] |
| RSGbothLon-L18-s0 | +0.0385 [+0.0161, +0.0627] | +5.0 [-2.3, +11.7] | +2.0 [-5.3, +9.1] | +0.511 [+0.371, +0.666] | +0.073 [-0.274, +0.646] |
| RSGbothLon-L18-s0 - RSGbC2-L18-s0 | +0.0062 [-0.0088, +0.0179] | -1.0 [-6.2, +5.7] | -1.0 [-5.7, +4.8] | +0.352 [+0.268, +0.446] | +0.086 [-0.269, +0.413] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
