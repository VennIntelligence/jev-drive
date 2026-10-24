# NTV1 S-NU verdict: D3's recipe with the S part from D3's own nuPlan rollouts (PAI 281, ACC + GAP, Harmonizer OFF; D3's serve)

## Reads

276 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NT5SNU-F-s0': 0, 'NT4D3-F-s0': 0, 'NT4D3-F-s1': 0, 'NT3D1-F-s0': 0, 'NT3D1-F-s1': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0, 'INTP75-NR': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-NU | 0.842 | 0.763 | 34 / 56 | 17 | 11 | 4 | 2 | 9 | 23 | 7 | 20 | 5 | 32/37 | 0.5293 |
| D3-s0 | 0.879 | 0.799 | 23 / 45 | 15 | 13 | 2 | 0 | 3 | 23 | 4 | 23 | 6 | 36/37 | 0.5304 |
| D3-s1 | 0.840 | 0.767 | 32 / 52 | 22 | 17 | 3 | 2 | 4 | 21 | 5 | 26 | 6 | 33/37 | 0.5359 |
| D1-s0 | 0.867 | 0.798 | 27 / 46 | 18 | 15 | 3 | 0 | 3 | 20 | 5 | 22 | 7 | 36/37 | 0.5297 |
| D1-s1 | 0.837 | 0.761 | 31 / 52 | 20 | 15 | 5 | 0 | 3 | 22 | 7 | 31 | 6 | 34/37 | 0.5370 |
| P1L | 0.811 | 0.732 | 39 / 61 | 23 | 14 | 6 | 3 | 10 | 24 | 4 | 27 | 11 | 32/37 | 0.5250 |
| NTV0-s0 | 0.813 | 0.741 | 39 / 59 | 18 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 | 0.5362 |
| R8K-s2 | 0.872 | 0.803 | 28 / 47 | 20 | 14 | 4 | 2 | 2 | 20 | 5 | 18 | 4 | 35/37 | 0.5578 |
| INTP75-NR (ref.) | 0.878 | 0.809 | 26 / 45 | 17 | 10 | 5 | 2 | 3 | 20 | 5 | 18 | 4 | 36/37 | 0.5578 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| S-NU - D3-s0 | -0.037 [-0.071, -0.004] | -0.037 [-0.074, -0.001] | +11 [+1, +21] | +2 [-5, +9] | +6 [+2, +11] | +3 [-2, +8] |
| S-NU - D3-s1 | +0.002 [-0.036, +0.042] | -0.005 [-0.045, +0.035] | +4 [-7, +15] | -5 [-14, +4] | +5 [+1, +10] | +2 [-3, +8] |
| S-NU - D1-s0 | -0.024 [-0.060, +0.010] | -0.035 [-0.073, +0.001] | +10 [+0, +20] | -1 [-8, +6] | +6 [+2, +11] | +2 [-3, +8] |
| S-NU - R8K-s2 | -0.029 [-0.066, +0.007] | -0.040 [-0.081, -0.000] | +9 [-2, +20] | -3 [-11, +5] | +7 [+2, +12] | +2 [-3, +7] |
| S-NU - P1L | +0.031 [-0.012, +0.074] | +0.031 [-0.011, +0.074] | -5 [-17, +6] | -6 [-16, +4] | -1 [-7, +5] | +3 [-3, +9] |
| S-NU - NTV0-s0 | +0.029 [-0.012, +0.071] | +0.022 [-0.021, +0.065] | -3 [-15, +9] | -1 [-10, +7] | -6 [-14, +2] | +4 [-1, +10] |
| S-NU - INTP75-NR (ref.) | -0.036 [-0.073, +0.001] | -0.047 [-0.085, -0.009] | +11 [+0, +22] | +0 [-7, +7] | +6 [+1, +12] | +2 [-3, +8] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-NU - D3-s0 | +0 [-4, +4] | +6 [+2, +11] | -2 [-8, +4] | +3 [-2, +8] | +2 [+0, +5] | +2 [+0, +5] | -3 [-8, +2] | -1 [-3, +0] | +0 [+0, +0] |
| S-NU - D3-s1 | +2 [+0, +5] | +5 [+1, +10] | -6 [-14, +2] | +2 [-3, +8] | +0 [-4, +4] | +1 [+0, +3] | -6 [-13, +1] | -1 [-5, +3] | +0 [+0, +0] |
| S-NU - D1-s0 | +3 [+0, +7] | +6 [+2, +11] | -4 [-10, +2] | +2 [-3, +8] | +2 [+0, +5] | +1 [+0, +3] | -2 [-9, +5] | -2 [-6, +2] | +0 [+0, +0] |
| S-NU - R8K-s2 | +3 [-1, +8] | +7 [+2, +12] | -3 [-10, +3] | +2 [-3, +7] | +0 [-4, +4] | +0 [-3, +3] | +2 [-5, +10] | +1 [-2, +5] | +0 [+0, +0] |
| S-NU - P1L | -1 [-6, +3] | -1 [-7, +5] | -3 [-10, +4] | +3 [-3, +9] | -1 [-5, +3] | -2 [-8, +3] | -7 [-15, +1] | -6 [-11, -2] | +0 [+0, +0] |
| S-NU - NTV0-s0 | +0 [-4, +4] | -6 [-14, +2] | -1 [-8, +5] | +4 [-1, +10] | +2 [+0, +5] | -2 [-7, +3] | -6 [-14, +2] | -6 [-12, -1] | +0 [+0, +0] |
| S-NU - INTP75-NR (ref.) | +3 [+0, +7] | +6 [+1, +12] | +1 [-5, +7] | +2 [-3, +8] | +0 [-3, +3] | -1 [-5, +2] | +2 [-4, +8] | +1 [-2, +5] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (200 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (180 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| S-NU | 16 | 11 | 12.66 | 11 | 33 | 28 / 40 |
| D3-s0 | 12 | 5 | 7.82 | 5 | 36 | 30 / 40 |
| D3-s1 | 16 | 8 | 11.36 | 8 | 28 | 24 / 40 |
| D1-s0 | 14 | 8 | 10.40 | 8 | 34 | 28 / 40 |
| D1-s1 | 21 | 11 | 15.10 | 11 | 29 | 24 / 40 |
| P1L | 16 | 13 | 13.94 | 12 | 15 | 10 / 40 |
| R8K-s2 | 14 | 9 | 10.53 | 9 | 40 | 40 / 40 |
| INTP75-NR (ref.) | 8 | 6 | 6.73 | 6 | 36 | 33 / 40 |

D3 2-seed mean (s0 + s1) / 2 on these scenes: new 0.859, old 0.783; S-NU - D3 2-seed mean: new -0.017 [-0.050, +0.015], old -0.021 [-0.056, +0.013].

## Registered line (new scorer registered, old reported)

- new scorer: S-NU - D3-s0 -0.037 [-0.071, -0.004] (helps >= +0.015, hurts <= -0.015) -> **fresh own S hurts**
- old scorer: S-NU - D3-s0 -0.037 [-0.074, -0.001] (helps >= +0.015, hurts <= -0.015) -> **fresh own S hurts** (reported)

## Highway (behavior_stats Q1: free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | scenes < 0.90 | reading |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| S-NU | 672 | 1.02 | 0.93 | 8% | 30% | +0.96 | 0.97 | 1 / 34 | not slow |
| D3-s0 | 711 | 1.01 | 0.91 | 8% | 29% | +0.90 | 0.98 | 1 / 32 | not slow |
| NTV0-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | 3 / 32 | not slow |
| R8K-s2 | 767 | 1.00 | 0.83 | 11% | 21% | -0.11 | 0.20 | 2 / 33 | not slow |

## Gates

- afix: True
- c0chk: True
- diff_arm: True
- g0b_arm_preT: True
- gw_draw_files: True
- latchk: True
- latdiag_0: None
- latdiag_1: None
- latdiag_None: None
- latgate: True
- lr_full_arm: True
- pai281: None
- pgate: True
- post_arm: True
- ratio_arm: True
- repro_rb0: True
- rollstats: None
- rowstats: None
- smoke_arm: True
- smoke_lr_arm: True
- t3_w0: True
- w_NSNU: None
- w_NSNUZ: None
- zfix: True
- S / imit gradient ratio: 500 steps after T 0.438 (band 0.21-1.04); whole post-T 0.334 (D3 0.419)

| navtrain dev ADE | step 10 000 | end |
|:--|--:|--:|
| S-NU | 0.5440 | 0.5293 |
| D3-s0 | 0.5440 | 0.5304 |

## Cost

GPU job-hours: roll 2.60, build 0.06, gates 0.13, train 1.32, read281 3.09, val160 1.74; total 8.9; wall 3.6 h.
