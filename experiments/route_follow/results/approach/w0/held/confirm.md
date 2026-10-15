# ONPOL-LONG Amendment 5: confirmation reads against RSG-L18-s0

## held18: 452 scenes, 39 logs

| checkpoint | mean score | zeros | collision / offroad / corridor zeros | at-fault zeros | GT distance (m) | at-fault distance (km, total km / at-fault incidents) [log-clustered 95 % CI] |
|:--|--:|--:|:--|--:|--:|:--|
| RSG-L18-s0 | 0.8809 | 26 | 4 / 17 / 6 | 21 | 1.135 | 0.988 [0.664, 1.735] |
| APP0C-L18-s0 | 0.8958 | 24 | 4 / 16 / 5 | 20 | 1.165 | 1.064 [0.664, 2.075] |
| APP2W05-L18-s0 | 0.8388 | 70 | 11 / 34 / 27 | 45 | 1.808 | 0.519 [0.399, 0.732] |

| a - RSG-L18-s0 | mean score | zeros | at-fault zeros | GT distance (m) | at-fault distance (km) |
|:--|:--|:--|:--|:--|:--|
| APP0C-L18-s0 | +0.0149 [+0.0027, +0.0271] | -2.0 [-7.0, +2.7] | -1.0 [-4.6, +2.2] | +0.030 [-0.015, +0.073] | +0.076 [-0.064, +0.479] |
| APP2W05-L18-s0 | -0.0420 [-0.0789, -0.0063] | +44.0 [+29.4, +59.5] | +24.0 [+12.5, +35.4] | +0.673 [+0.526, +0.833] | -0.469 [-1.144, -0.178] |
| APP0C-L18-s0 | +0.0149 [+0.0027, +0.0271] | -2.0 [-7.0, +2.7] | -1.0 [-4.6, +2.2] | +0.030 [-0.015, +0.073] | +0.076 [-0.064, +0.479] |
| APP0C-L18-s0 - APP2W05-L18-s0 | +0.0569 [+0.0188, +0.0979] | -46.0 [-63.7, -29.5] | -25.0 [-36.8, -13.3] | -0.643 [-0.798, -0.504] | +0.545 [+0.196, +1.487] |

train18 = the scenes whose rollouts supplied the rows: a guardrail, not clean evidence. all10: the base side is the stored 2026-10-09 rollouts (ot3 / cf1 manifests); 12 route-fold scenes have no rollout.
