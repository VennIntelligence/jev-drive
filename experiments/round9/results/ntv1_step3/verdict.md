# NTV1 step 3 verdict (PAI 281, ACC + GAP, Harmonizer OFF; P1L's serve)

## Reads

276 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NT3D1-F-s0': 0, 'NT3D2-F-s0': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0, 'INTP75-NR': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1 | 0.867 | 0.798 | 27 / 46 | 18 | 15 | 3 | 0 | 3 | 20 | 5 | 22 | 7 | 36/37 | 0.5297 |
| D2 | 0.829 | 0.746 | 36 / 59 | 24 | 19 | 1 | 4 | 3 | 24 | 8 | 25 | 6 | 34/37 | 0.5286 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 14 | 6 | 3 | 10 | 24 | 4 | 27 | 11 | 32/37 | 0.5250 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 | 0.5362 |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 4 | 2 | 2 | 20 | 5 | 18 | 4 | 35/37 | 0.5578 |
| INTP75-NR (ref.) | 0.878 | 0.809 | 26 / 45 | 17 | 10 | 5 | 2 | 3 | 20 | 5 | 18 | 4 | 36/37 | 0.5578 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| D1 - P1L | +0.055 [+0.016, +0.096] | +0.066 [+0.025, +0.108] | -15 [-27, -4] | -5 [-14, +4] | -7 [-12, -2] | +1 [-3, +5] |
| D1 - NTV0-s0 | +0.054 [+0.012, +0.095] | +0.057 [+0.014, +0.101] | -13 [-25, -2] | +0 [-8, +8] | -12 [-19, -5] | +2 [-3, +7] |
| D1 - R8K-s2 | -0.005 [-0.039, +0.030] | -0.005 [-0.042, +0.032] | -1 [-12, +9] | -2 [-10, +6] | +1 [+0, +3] | +0 [-5, +5] |
| D1 - INTP75-NR (ref.) | -0.012 [-0.044, +0.021] | -0.012 [-0.046, +0.023] | +1 [-9, +11] | +1 [-7, +9] | +0 [-3, +3] | +0 [-5, +5] |
| D2 - P1L | +0.017 [-0.023, +0.059] | +0.014 [-0.027, +0.056] | -2 [-13, +9] | +1 [-9, +11] | -7 [-12, -2] | +4 [-1, +10] |
| D2 - NTV0-s0 | +0.016 [-0.029, +0.059] | +0.005 [-0.042, +0.052] | +0 [-13, +13] | +6 [-4, +16] | -12 [-19, -5] | +5 [-1, +11] |
| D2 - R8K-s2 | -0.043 [-0.080, -0.005] | -0.057 [-0.096, -0.018] | +12 [+1, +23] | +4 [-5, +13] | +1 [+0, +3] | +3 [-3, +9] |
| D2 - INTP75-NR (ref.) | -0.049 [-0.084, -0.015] | -0.064 [-0.101, -0.027] | +14 [+4, +24] | +7 [-1, +16] | +0 [-3, +3] | +3 [-3, +10] |
| D2 - D1 | -0.038 [-0.065, -0.013] | -0.052 [-0.083, -0.024] | +13 [+5, +22] | +6 [+0, +12] | +0 [+0, +0] | +3 [-1, +8] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1 - P1L | -4 [-9, +0] | -7 [-12, -2] | +1 [-6, +8] | +1 [-3, +5] | -3 [-7, +0] | -3 [-8, +2] | -5 [-13, +3] | -4 [-9, +0] | +0 [+0, +0] |
| D1 - NTV0-s0 | -3 [-8, +1] | -12 [-19, -5] | +3 [-3, +10] | +2 [-3, +7] | +0 [+0, +0] | -3 [-8, +2] | -4 [-12, +4] | -4 [-10, +1] | +0 [+0, +0] |
| D1 - R8K-s2 | +0 [-4, +4] | +1 [+0, +3] | +1 [-6, +8] | +0 [-5, +5] | -2 [-5, +0] | -1 [-5, +2] | +4 [-4, +12] | +3 [-2, +8] | +0 [+0, +0] |
| D1 - INTP75-NR (ref.) | +0 [-3, +3] | +0 [-3, +3] | +5 [-1, +12] | +0 [-5, +5] | -2 [-5, +0] | -2 [-6, +2] | +4 [-4, +12] | +3 [-1, +8] | +0 [+0, +0] |
| D2 - P1L | +0 [-6, +6] | -7 [-12, -2] | +5 [-2, +13] | +4 [-1, +10] | +1 [-3, +5] | -5 [-10, +0] | -2 [-11, +7] | -5 [-10, -1] | +0 [+0, +0] |
| D2 - NTV0-s0 | +1 [-5, +7] | -12 [-19, -5] | +7 [+0, +15] | +5 [-1, +11] | +4 [+1, +8] | -5 [-10, +0] | -1 [-10, +8] | -5 [-11, +0] | +0 [+0, +0] |
| D2 - R8K-s2 | +4 [+0, +9] | +1 [+0, +3] | +5 [-3, +13] | +3 [-3, +9] | +2 [-3, +7] | -3 [-7, +0] | +7 [-2, +16] | +2 [-3, +7] | +0 [+0, +0] |
| D2 - INTP75-NR (ref.) | +4 [-1, +9] | +0 [-3, +3] | +9 [+2, +17] | +3 [-3, +10] | +2 [-2, +6] | -4 [-8, -1] | +7 [-1, +15] | +2 [-2, +6] | +0 [+0, +0] |
| D2 - D1 | +4 [+1, +8] | +0 [+0, +0] | +4 [+0, +9] | +3 [-1, +8] | +4 [+1, +8] | -2 [-5, +0] | +3 [-3, +9] | -1 [-3, +0] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (200 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (180 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| D1 | 14 | 8 | 10.40 | 8 | 34 | 28 / 40 |
| D2 | 22 | 14 | 17.51 | 13 | 31 | 26 / 40 |
| P1L | 16 | 13 | 13.94 | 12 | 15 | 10 / 40 |
| R8K-s2 | 14 | 9 | 10.53 | 9 | 40 | 40 / 40 |
| INTP75-NR (ref.) | 8 | 6 | 6.73 | 6 | 36 | 33 / 40 |

## Registered lines (new scorer registered, old reported)

- new scorer: D1 - P1L +0.055 [+0.016, +0.096] (help >= +0.02, hurt <= -0.01) -> **own-state DAgger helps**; D2 - D1 -0.038 [-0.065, -0.013] (line >= -0.01) -> **distill_ho still needed in the DAgger recipe**
- old scorer: D1 - P1L +0.066 [+0.025, +0.108] (help >= +0.02, hurt <= -0.01) -> **own-state DAgger helps**; D2 - D1 -0.052 [-0.083, -0.024] (line >= -0.01) -> **distill_ho still needed in the DAgger recipe** (reported)

Candidate line (new mean >= 0.86, Harmonizer off): **D1 (NT3D1-F-s0) 0.867** -> seed 1 + Harmonizer-ON reads submitted

## Highway (behavior_stats Q1: free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | scenes < 0.90 | reading |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| D1 | 736 | 1.00 | 0.89 | 8% | 22% | +0.34 | 2.20 | 2 / 33 | slow on highway |
| D2 | 742 | 1.00 | 0.86 | 10% | 16% | +0.01 | 2.27 | 2 / 33 | slow on highway |
| P1L | 820 | 1.00 | 0.85 | 10% | 6% | -0.63 | 1.10 | 2 / 33 | not slow |
| NTV0-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | 3 / 32 | not slow |
| R8K-s2 | 767 | 1.00 | 0.83 | 11% | 21% | -0.11 | 0.20 | 2 / 33 | not slow |

## Gates

- dsw_D2_switch: True
- g0a_flagoff_P1L: True
- g0b_preT_D1: True
- g0c_w0: True
- gr_row_treatment: True
- gw_draw_file: True
- lr_full_D1: True
- lr_full_D2: True
- pai281: None
- post_D1: True
- post_D2: True
- ratio_D1: True
- ratio_D2: True
- rep_preT_D2: True
- smoke_D1: True
- smoke_D2: True
- smoke_lr_D1: True
- smoke_lr_D2: True
- S / imit gradient ratio D1: 500 steps after T 0.310 (band 0.21-1.04, pass); whole post-T segment 0.421
- S / imit gradient ratio D2: 500 steps after T 0.351 (band 0.21-1.04, pass); whole post-T segment 0.416

| navtrain dev ADE | step 10 000 | end |
|:--|--:|--:|
| D1 | 0.5440 | 0.5297 |
| D2 | 0.5440 | 0.5286 |
| P1L | 0.5440 | 0.5250 |

## Cost

GPU jobs 10.8 job-hours (training 4.34, reads 6.26, gates / smokes 0.24); wall 2.7 h from the first start to the last end.
