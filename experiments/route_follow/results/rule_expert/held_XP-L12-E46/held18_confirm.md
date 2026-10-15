# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSG-L18-s0-XP-L12-E46 | 0.9178 | 22 | 4 / 9 / 9 | 13 | 1.388 | 1.775 [1.115, 3.690] |
| RSG-L18-s1 | 0.8830 | 26 | 6 / 13 / 8 | 19 | 1.144 | 1.095 [0.712, 2.034] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0-XP-L12-E46 | +0.0369 [+0.0010, +0.0712] | -4.0 [-16.2, +8.2] | -8.0 [-18.4, +2.7] | +0.252 [+0.079, +0.419] | +0.787 [-0.100, +2.613] |
| RSG-L18-s1 | +0.0021 [-0.0097, +0.0137] | +0.0 [-4.7, +4.9] | -2.0 [-7.0, +2.6] | +0.009 [-0.022, +0.041] | +0.107 [-0.174, +0.590] |
| RSG-L18-s0-XP-L12-E46 | +0.0369 [+0.0010, +0.0712] | -4.0 [-16.2, +8.2] | -8.0 [-18.4, +2.7] | +0.252 [+0.079, +0.419] | +0.787 [-0.100, +2.613] |
| RSG-L18-s0-XP-L12-E46 - RSG-L18-s1 | +0.0349 [-0.0022, +0.0705] | -4.0 [-17.1, +9.0] | -6.0 [-15.6, +4.2] | +0.244 [+0.068, +0.409] | +0.680 [-0.295, +2.399] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
