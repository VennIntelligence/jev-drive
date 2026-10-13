# ONPOL-LONG Amendment 5: confirmation reads against YR10m10-F-s0

## all10: 1479 scenes, 136 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| YR10m10-F-s0 | 0.9323 | 68 | 7 / 43 / 20 | 50 | 0.941 | 0.789 [0.576, 1.162] |
| RSG-L18-s0 | 0.9473 | 46 | 8 / 31 / 8 | 39 | 0.792 | 1.011 [0.724, 1.568] |
| RSG-L18-s1 | 0.9487 | 44 | 8 / 30 / 7 | 38 | 0.799 | 1.040 [0.731, 1.659] |
| RRED-L18-s0 | 0.9255 | 70 | 10 / 50 / 11 | 59 | 0.830 | 0.666 [0.499, 0.959] |
| RSG-L18 | 0.9480 | 45 | 8 / 30.5 / 7.5 | 38.5 | 0.796 | 1.026 [0.729, 1.600] |

| a - YR10m10-F-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSG-L18-s0 | +0.0151 [+0.0078, +0.0232] | -22.0 [-34.1, -10.8] | -11.0 [-21.6, -1.0] | -0.149 [-0.186, -0.113] | +0.222 [+0.016, +0.553] |
| RSG-L18-s1 | +0.0165 [+0.0089, +0.0247] | -24.0 [-36.1, -12.7] | -12.0 [-23.2, -1.8] | -0.141 [-0.180, -0.104] | +0.251 [+0.032, +0.648] |
| RRED-L18-s0 | -0.0067 [-0.0146, +0.0009] | +2.0 [-8.8, +12.9] | +9.0 [+0.0, +18.9] | -0.111 [-0.144, -0.079] | -0.124 [-0.319, -0.001] |
| RSG-L18 | +0.0158 [+0.0085, +0.0235] | -23.0 [-34.7, -11.9] | -11.5 [-22.1, -1.4] | -0.145 [-0.183, -0.109] | +0.236 [+0.030, +0.578] |
| RRED-L18-s0 - RSG-L18 | -0.0225 [-0.0298, -0.0155] | +25.0 [+15.2, +35.1] | +20.5 [+11.6, +30.2] | +0.034 [+0.011, +0.056] | -0.360 [-0.717, -0.189] |
| RRED-L18-s0 - RSG-L18-s0 | -0.0218 [-0.0295, -0.0147] | +24.0 [+13.8, +34.7] | +20.0 [+10.6, +30.5] | +0.038 [+0.015, +0.058] | -0.346 [-0.690, -0.172] |

Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below YR10m10-F-s0 on the public mean: {'RSG-L18-s0': True, 'RSG-L18-s1': True, 'RRED-L18-s0': True, 'RSG-L18': True}.

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
