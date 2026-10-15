# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| APP1X-L18-s0 | 0.7551 | 70 | 6 / 41 / 25 | 47 | 1.252 | 0.393 [0.299, 0.537] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| APP1X-L18-s0 | -0.1258 [-0.1558, -0.0979] | +44.0 [+30.4, +58.3] | +26.0 [+15.5, +36.9] | +0.116 [-0.006, +0.244] | -0.595 [-1.277, -0.318] |
| APP1X-L18-s0 | -0.1258 [-0.1558, -0.0979] | +44.0 [+30.4, +58.3] | +26.0 [+15.5, +36.9] | +0.116 [-0.006, +0.244] | -0.595 [-1.277, -0.318] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
