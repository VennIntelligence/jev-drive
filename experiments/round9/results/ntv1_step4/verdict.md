# NTV1 step 4 verdict: D3 = D1's recipe on the aggregated DAgger pool (PAI 281, ACC + GAP, Harmonizer OFF; P1L's serve)

## Reads

276 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NT4D3-F-s0': 0, 'NT3D1-F-s0': 0, 'NT3D1-F-s1': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0, 'INTP75-NR': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D3 | 0.879 | 0.799 | 23 / 45 | 15 | 13 | 2 | 0 | 3 | 23 | 4 | 23 | 6 | 36/37 | 0.5304 |
| D1-s0 | 0.867 | 0.798 | 27 / 46 | 18 | 15 | 3 | 0 | 3 | 20 | 5 | 22 | 7 | 36/37 | 0.5297 |
| D1-s1 | 0.837 | 0.761 | 31 / 52 | 20 | 15 | 5 | 0 | 3 | 22 | 7 | 31 | 6 | 34/37 | 0.5370 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 14 | 6 | 3 | 10 | 24 | 4 | 27 | 11 | 32/37 | 0.5250 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 | 0.5362 |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 4 | 2 | 2 | 20 | 5 | 18 | 4 | 35/37 | 0.5578 |
| INTP75-NR (ref.) | 0.878 | 0.809 | 26 / 45 | 17 | 10 | 5 | 2 | 3 | 20 | 5 | 18 | 4 | 36/37 | 0.5578 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| D3 - D1-s0 | +0.012 [-0.014, +0.039] | +0.001 [-0.030, +0.032] | -1 [-10, +8] | -3 [-10, +3] | +0 [-3, +3] | -1 [-4, +2] |
| D3 - D1-s1 | +0.042 [+0.012, +0.072] | +0.038 [+0.008, +0.070] | -7 [-15, +1] | -5 [-12, +2] | +0 [-3, +3] | -3 [-8, +1] |
| D3 - P1L | +0.067 [+0.030, +0.106] | +0.067 [+0.028, +0.107] | -16 [-27, -6] | -8 [-16, +0] | -7 [-13, -2] | +0 [-5, +5] |
| D3 - NTV0-s0 | +0.066 [+0.027, +0.104] | +0.059 [+0.017, +0.100] | -14 [-25, -3] | -3 [-11, +4] | -12 [-19, -6] | +1 [-3, +5] |
| D3 - R8K-s2 | +0.007 [-0.023, +0.039] | -0.004 [-0.037, +0.030] | -2 [-11, +7] | -5 [-12, +2] | +1 [+0, +3] | -1 [-6, +4] |
| D3 - INTP75-NR (ref.) | +0.001 [-0.029, +0.031] | -0.010 [-0.043, +0.022] | +0 [-9, +9] | -2 [-9, +5] | +0 [-3, +3] | -1 [-6, +4] |
| D1-s0 - P1L | +0.055 [+0.016, +0.096] | +0.066 [+0.025, +0.108] | -15 [-27, -4] | -5 [-14, +4] | -7 [-12, -2] | +1 [-3, +5] |
| D1-s1 - D1-s0 | -0.030 [-0.057, -0.004] | -0.037 [-0.067, -0.008] | +6 [-2, +15] | +2 [-4, +8] | +0 [+0, +0] | +2 [-2, +6] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D3 - D1-s0 | +3 [-1, +8] | +0 [-3, +3] | -2 [-8, +4] | -1 [-4, +2] | +0 [+0, +0] | -1 [-3, +0] | +1 [-5, +7] | -1 [-5, +2] | +0 [+0, +0] |
| D3 - D1-s1 | +1 [+0, +3] | +0 [-3, +3] | -2 [-8, +4] | -3 [-8, +1] | +0 [+0, +0] | -3 [-7, +0] | -8 [-14, -2] | +0 [-4, +4] | +0 [+0, +0] |
| D3 - P1L | -1 [-6, +4] | -7 [-13, -2] | -1 [-7, +5] | +0 [-5, +5] | -3 [-7, +0] | -4 [-9, +0] | -4 [-13, +4] | -5 [-10, -1] | +0 [+0, +0] |
| D3 - NTV0-s0 | +0 [-5, +5] | -12 [-19, -6] | +1 [-5, +7] | +1 [-3, +5] | +0 [+0, +0] | -4 [-9, +0] | -3 [-12, +6] | -5 [-11, +0] | +0 [+0, +0] |
| D3 - R8K-s2 | +3 [+0, +7] | +1 [+0, +3] | -1 [-7, +5] | -1 [-6, +4] | -2 [-5, +0] | -2 [-5, +0] | +5 [-2, +13] | +2 [-2, +6] | +0 [+0, +0] |
| D3 - INTP75-NR (ref.) | +3 [+0, +7] | +0 [-3, +3] | +3 [-2, +8] | -1 [-6, +4] | -2 [-5, +0] | -3 [-7, +0] | +5 [-3, +13] | +2 [+0, +5] | +0 [+0, +0] |
| D1-s0 - P1L | -4 [-9, +0] | -7 [-12, -2] | +1 [-6, +8] | +1 [-3, +5] | -3 [-7, +0] | -3 [-8, +2] | -5 [-13, +3] | -4 [-9, +0] | +0 [+0, +0] |
| D1-s1 - D1-s0 | +2 [-2, +6] | +0 [+0, +0] | +0 [-6, +6] | +2 [-2, +6] | +0 [+0, +0] | +2 [+0, +5] | +9 [+3, +16] | -1 [-3, +0] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (200 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (180 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| D3 | 12 | 5 | 7.82 | 5 | 36 | 30 / 40 |
| D1-s0 | 14 | 8 | 10.40 | 8 | 34 | 28 / 40 |
| D1-s1 | 21 | 11 | 15.10 | 11 | 29 | 24 / 40 |
| P1L | 16 | 13 | 13.94 | 12 | 15 | 10 / 40 |
| R8K-s2 | 14 | 9 | 10.53 | 9 | 40 | 40 / 40 |
| INTP75-NR (ref.) | 8 | 6 | 6.73 | 6 | 36 | 33 / 40 |

D1 seed mean (s0 + s1) / 2 on these scenes: new 0.852, old 0.779; D3 - D1 seed mean: new +0.027 [+0.002, +0.052], old +0.020 [-0.008, +0.047].

## Registered line (new scorer registered, old reported)

- new scorer: D3 - D1-s0 +0.012 [-0.014, +0.039] (helps >= +0.015, hurts <= -0.015) -> **neutral**
- old scorer: D3 - D1-s0 +0.001 [-0.030, +0.032] (helps >= +0.015, hurts <= -0.015) -> **neutral** (reported)

Candidate rule (D3 new mean >= 0.875): D3 0.879 -> **triggered**: seed 1 + val160 reads of both seeds submitted

## Highway (behavior_stats Q1: free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | scenes < 0.90 | reading |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| D3 | 711 | 1.01 | 0.91 | 8% | 29% | +0.90 | 0.98 | 1 / 32 | not slow |
| D1-s0 | 736 | 1.00 | 0.89 | 8% | 22% | +0.34 | 2.20 | 2 / 33 | slow on highway |
| P1L | 820 | 1.00 | 0.85 | 10% | 6% | -0.63 | 1.10 | 2 / 33 | not slow |
| NTV0-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | 3 / 32 | not slow |
| R8K-s2 | 767 | 1.00 | 0.83 | 11% | 21% | -0.11 | 0.20 | 2 / 33 | not slow |

## Gates

- build: None
- diff_D3: True
- g0b_D3_preT: True
- gw_draw_files: True
- lr_full_D3: True
- pai281: None
- post_D3: True
- ratio_D3: True
- smoke_D3: True
- smoke_lr_D3: True
- t12: True
- t3a_D1_preT: True
- t3b_w0: True
- w_NTV1D2: None
- w_NTV1D2Z: None
- S / imit gradient ratio D3: 500 steps after T 0.318 (band 0.21-1.04, pass); whole post-T segment 0.419 (D1 0.421)

| navtrain dev ADE | step 10 000 | end |
|:--|--:|--:|
| D3 | 0.5440 | 0.5304 |
| D1-s0 | 0.5440 | 0.5297 |
| P1L | 0.5440 | 0.5250 |

## Cost

GPU job-hours: roll 8.10, build 0.39, gates 0.14, train 1.13, read281 3.11; total 12.9; wall 2.9 h.
