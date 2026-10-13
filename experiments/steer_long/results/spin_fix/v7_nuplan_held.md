# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| RSG-L18-s1 | 0.8830 | 26 | 6 / 13 / 8 | 19 | 1.144 | 1.095 [0.712, 2.034] |
| time | 0.8730 | 29 | 3 / 17 / 11 | 20 | 1.073 | 1.037 [0.679, 2.001] |
| curv | 0.8730 | 29 | 3 / 17 / 11 | 20 | 1.079 | 1.036 [0.678, 2.002] |
| gate | 0.8753 | 28 | 2 / 18 / 10 | 20 | 1.074 | 1.037 [0.679, 1.995] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s1 | +0.0021 [-0.0097, +0.0137] | +0.0 [-4.7, +4.9] | -2.0 [-7.0, +2.6] | +0.009 [-0.022, +0.041] | +0.107 [-0.174, +0.590] |
| time | -0.0079 [-0.0231, +0.0069] | +3.0 [-3.9, +10.3] | -1.0 [-6.5, +4.0] | -0.062 [-0.132, +0.014] | +0.049 [-0.214, +0.517] |
| curv | -0.0079 [-0.0230, +0.0069] | +3.0 [-3.9, +10.3] | -1.0 [-6.5, +4.0] | -0.057 [-0.128, +0.018] | +0.048 [-0.214, +0.515] |
| gate | -0.0056 [-0.0201, +0.0098] | +2.0 [-4.7, +8.5] | -1.0 [-7.2, +4.8] | -0.061 [-0.129, +0.010] | +0.049 [-0.256, +0.565] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
