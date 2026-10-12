# ONPOL-LONG Amendment 5: confirmation reads against YR10m10-F-s0

## all10: 1479 scenes, 136 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |
|:--|--:|--:|:--|--:|--:|
| YR10m10-F-s0 | 0.9323 | 68 | 7 / 43 / 20 | 50 | 0.941 |
| YS-L18-s0 | 0.9448 | 47 | 10 / 20 / 17 | 30 | 0.904 |
| YS-L18-s1 | 0.9453 | 48 | 11 / 25 / 12 | 36 | 0.903 |

| a - YR10m10-F-s0 | mean score | zeros | at-fault zeros | GT distance (m) |
|:--|:--|:--|:--|:--|
| YS-L18-s0 | +0.0125 [+0.0033, +0.0226] | -21.0 [-35.7, -7.3] | -20.0 [-32.0, -8.5] | -0.037 [-0.082, +0.005] |
| YS-L18-s1 | +0.0131 [+0.0036, +0.0227] | -20.0 [-34.4, -5.9] | -14.0 [-26.2, -1.9] | -0.038 [-0.088, +0.008] |

## train18: 1021 scenes, 97 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) |
|:--|--:|--:|:--|--:|--:|
| YR10m10-F-s0 | 0.7649 | 168 | 26 / 77 / 74 | 102 | 1.526 |
| YS-L18-s0 | 0.8241 | 99 | 35 / 23 / 43 | 58 | 1.439 |
| YS-L18-s1 | 0.8144 | 114 | 32 / 31 / 53 | 63 | 1.431 |

| a - YR10m10-F-s0 | mean score | zeros | at-fault zeros | GT distance (m) |
|:--|:--|:--|:--|:--|
| YS-L18-s0 | +0.0591 [+0.0381, +0.0819] | -69.0 [-93.4, -47.4] | -44.0 [-66.2, -24.6] | -0.087 [-0.161, -0.016] |
| YS-L18-s1 | +0.0495 [+0.0298, +0.0714] | -54.0 [-77.3, -32.5] | -39.0 [-60.7, -19.5] | -0.095 [-0.172, -0.015] |

Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below YR10m10-F-s0 on the public mean: {'YS-L18-s0': True, 'YS-L18-s1': True}.

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
