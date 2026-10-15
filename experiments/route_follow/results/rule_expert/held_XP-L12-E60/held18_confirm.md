# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSG-L18-s0-XP-L12-E60 | 0.9252 | 21 | 3 / 9 / 9 | 12 | 1.438 | 1.959 [1.181, 4.673] |
| RSG-L18-s1 | 0.8830 | 26 | 6 / 13 / 8 | 19 | 1.144 | 1.095 [0.712, 2.034] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0-XP-L12-E60 | +0.0443 [+0.0075, +0.0794] | -5.0 [-17.9, +8.3] | -9.0 [-20.1, +2.4] | +0.302 [+0.120, +0.481] | +0.971 [-0.065, +3.685] |
| RSG-L18-s1 | +0.0021 [-0.0097, +0.0137] | +0.0 [-4.7, +4.9] | -2.0 [-7.0, +2.6] | +0.009 [-0.022, +0.041] | +0.107 [-0.174, +0.590] |
| RSG-L18-s0-XP-L12-E60 | +0.0443 [+0.0075, +0.0794] | -5.0 [-17.9, +8.3] | -9.0 [-20.1, +2.4] | +0.302 [+0.120, +0.481] | +0.971 [-0.065, +3.685] |
| RSG-L18-s0-XP-L12-E60 - RSG-L18-s1 | +0.0423 [+0.0049, +0.0780] | -5.0 [-18.8, +9.0] | -7.0 [-17.1, +3.6] | +0.294 [+0.111, +0.467] | +0.864 [-0.212, +3.476] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
