# ONPOL-LONG Amendment 5: confirmation reads against YR10m10-F-s0

## all10: 1479 scenes, 136 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| YR10m10-F-s0 | 0.9323 | 68 | 7 / 43 / 20 | 50 | 0.941 | 0.789 [0.576, 1.162] |
| RSGbothLon-L18-s0 | 0.9504 | 58 | 8 / 39 / 11 | 47 | 0.924 | 0.922 [0.671, 1.380] |
| RSGboth-L18-s0 | 0.9522 | 58 | 9 / 36 / 13 | 45 | 0.965 | 0.971 [0.703, 1.462] |
| RSG-L18-s0 | 0.9473 | 46 | 8 / 31 / 8 | 39 | 0.792 | 1.011 [0.724, 1.568] |
| RSGbC2-L18-s0 | 0.9520 | 56 | 9 / 33 / 14 | 42 | 0.843 | 1.008 [0.731, 1.530] |

| a - YR10m10-F-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| RSGbothLon-L18-s0 | +0.0182 [+0.0099, +0.0264] | -10.0 [-21.4, +1.2] | -3.0 [-13.9, +8.4] | -0.017 [-0.072, +0.036] | +0.132 [-0.077, +0.392] |
| RSGboth-L18-s0 | +0.0200 [+0.0107, +0.0290] | -10.0 [-22.8, +3.2] | -5.0 [-16.2, +6.7] | +0.025 [-0.029, +0.078] | +0.182 [-0.044, +0.468] |
| RSG-L18-s0 | +0.0151 [+0.0078, +0.0232] | -22.0 [-34.1, -10.8] | -11.0 [-21.6, -1.0] | -0.149 [-0.186, -0.113] | +0.222 [+0.016, +0.553] |
| RSGbC2-L18-s0 | +0.0198 [+0.0118, +0.0277] | -12.0 [-23.2, -0.9] | -8.0 [-18.1, +2.2] | -0.098 [-0.145, -0.050] | +0.219 [+0.015, +0.506] |
| RSGbothLon-L18-s0 - RSGboth-L18-s0 | -0.0018 [-0.0056, +0.0021] | +0.0 [-5.6, +5.4] | +2.0 [-2.9, +6.9] | -0.042 [-0.053, -0.030] | -0.049 [-0.183, +0.059] |
| RSGbothLon-L18-s0 - RSG-L18-s0 | +0.0031 [-0.0037, +0.0097] | +12.0 [+3.8, +20.5] | +8.0 [+0.0, +15.9] | +0.132 [+0.089, +0.175] | -0.090 [-0.338, +0.087] |
| RSGbothLon-L18-s0 - RSGbC2-L18-s0 | -0.0016 [-0.0059, +0.0027] | +2.0 [-4.2, +8.2] | +5.0 [+0.0, +10.4] | +0.081 [+0.059, +0.103] | -0.087 [-0.237, +0.024] |

Guardrail (Amendment 5): all 1491 at 10 decisions not more than 0.02 below YR10m10-F-s0 on the public mean: {'RSGbothLon-L18-s0': True, 'RSGboth-L18-s0': True, 'RSG-L18-s0': True, 'RSGbC2-L18-s0': True}.

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
