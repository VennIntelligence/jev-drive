# NTV1 step 1 verdict (PAI 281, ACC + GAP, Harmonizer OFF; arms served with JEV_CMD=heading)

278 scenes scored by all six reads (both scorers) with no driver RPC error in any of them; RPC scenes per read {'NT1N1-F-s0': 0, 'NT1N2A-F-s0': 0, 'NT1N2B-F-s0': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0}.

## Reads

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | new zeros on NTV0 full-score (old / new) | dev ADE 10k / end |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| N1 (NT1N1-F-s0) | 0.797 | 0.721 | 46 / 67 | 25 | 16 | 7 | 2 | 13 | 24 | 5 | 24 | 10 | 34/37 | 15 / 12 | 0.5467 / 0.5331 |
| N2a (NT1N2A-F-s0) | 0.722 | 0.642 | 67 / 89 | 53 | 23 | 24 | 6 | 5 | 23 | 8 | 21 | 2 | 32/37 | 40 / 35 | 0.5775 / 0.5573 |
| N2b (NT1N2B-F-s0) | 0.779 | 0.700 | 54 / 76 | 38 | 19 | 16 | 3 | 7 | 24 | 7 | 18 | 6 | 34/37 | 27 / 24 | 0.5467 / 0.5522 |
| P1L (PCL1L-F-s0) | 0.813 | 0.734 | 39 / 61 | 23 | 14 | 6 | 3 | 10 | 24 | 4 | 27 | 11 | 32/37 | 15 / 12 | 0.5440 / 0.5250 |
| NTV0-s0 (NTV0-L18-s0) | 0.818 | 0.746 | 38 / 58 | 18 | 12 | 6 | 0 | 14 | 23 | 3 | 26 | 11 | 33/37 | 0 / 0 | - / 0.5362 |
| R8K-s2 (R8K-L18-s2) | 0.872 | 0.804 | 28 / 47 | 20 | 14 | 4 | 2 | 2 | 20 | 5 | 18 | 4 | 35/37 | 10 / 9 | - / 0.5578 |

NTV0-s0 full-score scenes (old score >= 0.98) among the common set: 183.

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn |
|:--|--:|--:|--:|--:|--:|
| N1 - P1L | -0.016 [-0.054, +0.022] | -0.012 [-0.051, +0.026] | +6 [-5, +17] | +2 [-7, +11] | +3 [-3, +9] |
| N1 - NTV0-s0 | -0.021 [-0.058, +0.015] | -0.025 [-0.061, +0.012] | +9 [-1, +20] | +7 [+0, +15] | -1 [-8, +6] |
| N1 - R8K-s2 | -0.076 [-0.121, -0.030] | -0.083 [-0.130, -0.036] | +20 [+7, +33] | +5 [-5, +15] | +11 [+5, +18] |
| N2a - P1L | -0.091 [-0.145, -0.038] | -0.091 [-0.145, -0.037] | +28 [+13, +43] | +30 [+16, +44] | -5 [-11, +1] |
| N2a - NTV0-s0 | -0.096 [-0.148, -0.045] | -0.104 [-0.158, -0.051] | +31 [+17, +46] | +35 [+23, +47] | -9 [-16, -3] |
| N2a - R8K-s2 | -0.151 [-0.198, -0.104] | -0.162 [-0.212, -0.112] | +42 [+28, +56] | +33 [+20, +46] | +3 [+0, +7] |
| N2b - P1L | -0.034 [-0.085, +0.018] | -0.034 [-0.085, +0.019] | +15 [+0, +29] | +15 [+2, +27] | -3 [-8, +2] |
| N2b - NTV0-s0 | -0.039 [-0.088, +0.008] | -0.046 [-0.095, +0.002] | +18 [+5, +31] | +20 [+9, +31] | -7 [-14, -1] |
| N2b - R8K-s2 | -0.094 [-0.138, -0.051] | -0.104 [-0.149, -0.060] | +29 [+16, +42] | +18 [+7, +29] | +5 [+1, +10] |
| N2a - N1 | -0.075 [-0.129, -0.021] | -0.079 [-0.135, -0.024] | +22 [+6, +38] | +28 [+15, +41] | -8 [-15, -2] |
| N2b - N1 | -0.018 [-0.068, +0.032] | -0.022 [-0.073, +0.029] | +9 [-5, +23] | +13 [+0, +26] | -6 [-12, -1] |
| N2a - N2b | -0.057 [-0.109, -0.007] | -0.057 [-0.112, -0.005] | +13 [-2, +29] | +15 [+1, +29] | -2 [-6, +2] |

Zero / partial classes (count difference, 95 % CI; old-scorer classes):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| N1 - P1L | +0 [-4, +4] | +3 [-3, +9] | +2 [-5, +9] | +1 [-4, +6] | -1 [-5, +2] | +1 [-3, +5] | -3 [-10, +4] | -1 [-6, +3] | +0 [+0, +0] |
| N1 - NTV0-s0 | +1 [-3, +6] | -1 [-8, +6] | +4 [-2, +10] | +2 [-3, +7] | +2 [+0, +5] | +1 [-2, +4] | -2 [-8, +4] | -1 [-3, +0] | +0 [+0, +0] |
| N1 - R8K-s2 | +4 [-2, +10] | +11 [+5, +18] | +2 [-6, +10] | +0 [-6, +6] | +0 [-4, +4] | +3 [-2, +8] | +6 [-2, +14] | +6 [+0, +12] | +0 [+0, +0] |
| N2a - P1L | -1 [-8, +6] | -5 [-11, +1] | +9 [-1, +19] | +4 [+0, +9] | +3 [-3, +9] | +18 [+9, +28] | -6 [-16, +4] | -9 [-15, -4] | +0 [+0, +0] |
| N2a - NTV0-s0 | +0 [-6, +6] | -9 [-16, -3] | +11 [+3, +20] | +5 [-1, +11] | +6 [+2, +11] | +18 [+10, +26] | -5 [-15, +5] | -9 [-15, -4] | +0 [+0, +0] |
| N2a - R8K-s2 | +3 [-2, +8] | +3 [+0, +7] | +9 [+1, +18] | +3 [-3, +10] | +4 [-1, +9] | +20 [+12, +29] | +3 [-5, +11] | -2 [-6, +2] | +0 [+0, +0] |
| N2b - P1L | +0 [-5, +5] | -3 [-8, +2] | +5 [-3, +13] | +3 [-3, +9] | +0 [-5, +5] | +10 [+2, +18] | -9 [-18, +0] | -5 [-11, +1] | +0 [+0, +0] |
| N2b - NTV0-s0 | +1 [-3, +5] | -7 [-14, -1] | +7 [+0, +15] | +4 [-1, +10] | +3 [+0, +7] | +10 [+2, +18] | -8 [-17, +1] | -5 [-12, +1] | +0 [+0, +0] |
| N2b - R8K-s2 | +4 [+0, +9] | +5 [+1, +10] | +5 [-2, +12] | +2 [-4, +8] | +1 [-3, +5] | +12 [+5, +19] | +0 [-7, +7] | +2 [-3, +7] | +0 [+0, +0] |
| N2a - N1 | -1 [-8, +5] | -8 [-15, -2] | +7 [-2, +16] | +3 [-3, +10] | +4 [-1, +10] | +17 [+9, +26] | -3 [-12, +6] | -8 [-14, -3] | +0 [+0, +0] |
| N2b - N1 | +0 [-5, +5] | -6 [-12, -1] | +3 [-6, +12] | +2 [-3, +8] | +1 [-3, +5] | +9 [+1, +17] | -6 [-14, +2] | -4 [-10, +2] | +0 [+0, +0] |
| N2a - N2b | -1 [-7, +5] | -2 [-6, +2] | +4 [-5, +14] | +1 [-6, +8] | +3 [-2, +8] | +8 [-1, +17] | +3 [-6, +12] | -4 [-8, -1] | +0 [+0, +0] |

## Registered lines (new scorer registered, old reported)

- new scorer: N1 - P1L -0.016 [-0.054, +0.022] (line >= -0.015) -> **below the line: the heading command costs in the S recipe (report)**
  - N2A - N1 -0.075, route-turn zeros 5 vs N1 13 -> **PAI human rows hurt in this placement**
  - N2B - N1 -0.018, route-turn zeros 7 vs N1 13 -> **PAI human rows hurt in this placement**
- old scorer: N1 - P1L -0.012 [-0.051, +0.026] (line >= -0.015) -> **the heading command costs nothing in the S recipe (adopt)** (reported)
  - N2A - N1 -0.079, route-turn zeros 5 vs N1 13 -> **PAI human rows hurt in this placement**
  - N2B - N1 -0.022, route-turn zeros 7 vs N1 13 -> **PAI human rows hurt in this placement**

## Gates

- cmdagree_N1: True
- cmdagree_N2A: True
- cmdagree_N2B: True
- g0_flagoff_P1L: True
- g0_flagoff_RB0F: True
- lr_full_N1: True
- lr_full_N2A: True
- lr_full_N2B: True
- pai281: None
- post_N1: True
- post_N2A: True
- post_N2B: True
- ratio_N1: True
- ratio_N2A: True
- ratio_N2B: True
- same_N2B_N1_smoke: True
- smoke_N1: True
- smoke_N2A: True
- smoke_N2B: True
- smoke_cmd_N1: True
- smoke_cmd_N2A: True
- smoke_cmd_N2B: True
- smoke_lr_N1: True
- smoke_lr_N2A: True
- smoke_lr_N2B: True
- wfile: True
- S / imit gradient ratio N1: 500 steps after T 0.642 (band 0.21-1.04, pass); whole post-T segment 0.444
- S / imit gradient ratio N2A: 500 steps after T 0.241 (band 0.21-1.04, pass); whole post-T segment 0.418
- S / imit gradient ratio N2B: 500 steps after T 0.361 (band 0.21-1.04, pass); whole post-T segment 0.426
- served command vs rule N1: 55693 inferences, agree 1.00000 -> pass
- served command vs rule N2A: 55693 inferences, agree 1.00000 -> pass
- served command vs rule N2B: 55494 inferences, agree 1.00000 -> pass

## Cost

GPU jobs 13.0 job-hours (training 3.64, reads 9.19, gates / smokes 0.13); wall 2.1 h from the first start to the last end.
