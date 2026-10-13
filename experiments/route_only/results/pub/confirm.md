# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## all10: 1479 scenes, 136 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.9473 | 46 | 8 / 31 / 8 | 39 | 0.792 | 1.011 [0.724, 1.568] |
| R-own | 0.9406 | 56 | 10 / 38 / 9 | 48 | 0.800 | 0.820 [0.609, 1.180] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| R-own | -0.0068 [-0.0105, -0.0031] | +10.0 [+4.6, +15.6] | +9.0 [+3.1, +15.0] | +0.008 [+0.001, +0.017] | -0.191 [-0.462, -0.060] |

Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below RSG-L18-s0 on the public mean: {'R-own': True}.

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
