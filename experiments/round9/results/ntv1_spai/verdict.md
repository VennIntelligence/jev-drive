# NTV1 S-PAI verdict: D3's recipe with half the S slots on D3's own PAI states (PAI 281 + val160, ACC + GAP, Harmonizer OFF)

## PAI 281

276 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NT5SP-F-s0': 0, 'NT4D3-F-s0': 0, 'NT4D3-F-s1': 0, 'NT3D1-F-s0': 0, 'NT3D1-F-s1': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0, 'INTP75-NR': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-PAI | 0.883 | 0.807 | 22 / 43 | 10 | 6 | 4 | 0 | 6 | 22 | 5 | 22 | 8 | 35/37 | 0.5305 |
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
| S-PAI - D3-s0 | +0.004 [-0.024, +0.032] | +0.008 [-0.021, +0.037] | -2 [-10, +6] | -5 [-11, +1] | +3 [+0, +7] | +1 [-3, +5] |
| S-PAI - D3-s1 | +0.043 [+0.011, +0.076] | +0.039 [+0.007, +0.074] | -9 [-18, +0] | -12 [-20, -5] | +2 [+0, +5] | +0 [-5, +5] |
| S-PAI - D1-s0 | +0.016 [-0.015, +0.049] | +0.009 [-0.026, +0.044] | -3 [-13, +7] | -8 [-15, -2] | +3 [+0, +7] | +0 [-5, +5] |
| S-PAI - R8K-s2 | +0.011 [-0.021, +0.044] | +0.004 [-0.030, +0.039] | -4 [-13, +5] | -10 [-18, -3] | +4 [+1, +8] | +0 [-4, +4] |
| S-PAI - INTP75-NR (ref.) | +0.005 [-0.026, +0.036] | -0.003 [-0.036, +0.030] | -2 [-11, +7] | -7 [-14, -1] | +3 [-1, +8] | +0 [-5, +5] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-PAI - D3-s0 | -1 [-3, +0] | +3 [+0, +7] | -7 [-13, -2] | +1 [-3, +5] | +0 [+0, +0] | +2 [+0, +5] | -1 [-6, +3] | +2 [+0, +5] | +0 [+0, +0] |
| S-PAI - D3-s1 | +1 [+0, +3] | +2 [+0, +5] | -11 [-18, -4] | +0 [-5, +5] | -2 [-5, +0] | +1 [+0, +3] | -4 [-10, +2] | +2 [-2, +6] | +0 [+0, +0] |
| S-PAI - D1-s0 | +2 [-2, +6] | +3 [+0, +7] | -9 [-16, -3] | +0 [-5, +5] | +0 [+0, +0] | +1 [-2, +5] | +0 [-6, +6] | +1 [-3, +5] | +0 [+0, +0] |
| S-PAI - R8K-s2 | +2 [+0, +5] | +4 [+1, +8] | -8 [-14, -2] | +0 [-4, +4] | -2 [-5, +0] | +0 [-3, +3] | +4 [-3, +12] | +4 [-1, +9] | +0 [+0, +0] |
| S-PAI - INTP75-NR (ref.) | +2 [+0, +5] | +3 [-1, +8] | -4 [-9, +0] | +0 [-5, +5] | -2 [-5, +0] | -1 [-5, +2] | +4 [-2, +10] | +4 [+1, +8] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (200 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (180 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| S-PAI | 12 | 6 | 8.14 | 6 | 36 | 30 / 40 |
| D3-s0 | 12 | 5 | 7.82 | 5 | 36 | 30 / 40 |
| D3-s1 | 16 | 8 | 11.36 | 8 | 28 | 24 / 40 |
| D1-s0 | 14 | 8 | 10.40 | 8 | 34 | 28 / 40 |
| D1-s1 | 21 | 11 | 15.10 | 11 | 29 | 24 / 40 |
| P1L | 16 | 13 | 13.94 | 12 | 15 | 10 / 40 |
| R8K-s2 | 14 | 9 | 10.53 | 9 | 40 | 40 / 40 |
| INTP75-NR (ref.) | 8 | 6 | 6.73 | 6 | 36 | 33 / 40 |

S-PAI - D3 2-seed mean: new +0.024 [-0.002, +0.049], old +0.024 [-0.003, +0.050].

## Registered line (new scorer registered, old reported)

- new scorer: S-PAI - D3-s0 +0.004 [-0.024, +0.032] (helps >= +0.015, hurts <= -0.015) -> **neutral**
- old scorer: S-PAI - D3-s0 +0.008 [-0.021, +0.037] (helps >= +0.015, hurts <= -0.015) -> **neutral** (reported)

## Highway (behavior_stats Q1: free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | scenes < 0.90 | reading |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| S-PAI | 727 | 1.01 | 0.91 | 8% | 24% | +0.57 | 0.94 | 1 / 34 | not slow |
| D3-s0 | 711 | 1.01 | 0.91 | 8% | 29% | +0.90 | 0.98 | 1 / 32 | not slow |
| NTV0-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | 3 / 32 | not slow |
| R8K-s2 | 767 | 1.00 | 0.83 | 11% | 21% | -0.11 | 0.20 | 2 / 33 | not slow |

## val160

151 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NT5SP-F-s0': 0, 'NT4D3-F-s0': 0, 'NT4D3-F-s1': 0, 'R8K-L18-s2': 0, 'NT3D1-F-s0': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-PAI | 0.867 | 0.781 | 15 / 28 | 2 | 1 | 1 | 0 | 2 | 13 | 11 | 14 | 3 | 16/20 | 0.5305 |
| D3-s0 | 0.842 | 0.756 | 19 / 32 | 7 | 6 | 1 | 0 | 3 | 13 | 9 | 14 | 3 | 16/20 | 0.5304 |
| D3-s1 | 0.864 | 0.777 | 15 / 28 | 5 | 4 | 1 | 0 | 1 | 13 | 9 | 16 | 1 | 17/20 | 0.5359 |
| R8K-s2 | 0.853 | 0.767 | 18 / 31 | 9 | 7 | 0 | 2 | 2 | 13 | 7 | 13 | 1 | 16/20 | 0.5578 |
| D1-s0 | 0.827 | 0.748 | 21 / 33 | 11 | 7 | 1 | 3 | 1 | 12 | 9 | 16 | 1 | 16/20 | 0.5297 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| S-PAI - D3-s0 | +0.025 [-0.016, +0.067] | +0.025 [-0.016, +0.067] | -4 [-10, +2] | -5 [-11, +1] | -1 [-3, +0] | +2 [+0, +5] |
| S-PAI - D3-s1 | +0.003 [-0.034, +0.041] | +0.003 [-0.034, +0.041] | +0 [-6, +6] | -3 [-8, +1] | +1 [+0, +3] | +2 [+0, +5] |
| S-PAI - R8K-s2 | +0.014 [-0.025, +0.053] | +0.014 [-0.025, +0.053] | -3 [-9, +3] | -7 [-13, -2] | +0 [+0, +0] | +4 [+0, +9] |
| S-PAI - D1-s0 | +0.039 [+0.000, +0.083] | +0.033 [-0.005, +0.074] | -5 [-11, +1] | -9 [-16, -3] | +1 [+0, +3] | +2 [-3, +7] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S-PAI - D3-s0 | +0 [+0, +0] | -1 [-3, +0] | -5 [-10, +0] | +2 [+0, +5] | +0 [+0, +0] | +0 [-3, +3] | +0 [-4, +4] | +0 [-3, +3] | +0 [+0, +0] |
| S-PAI - D3-s1 | +0 [+0, +0] | +1 [+0, +3] | -3 [-8, +1] | +2 [+0, +5] | +0 [+0, +0] | +0 [+0, +0] | -2 [-6, +2] | +2 [+0, +5] | +0 [+0, +0] |
| S-PAI - R8K-s2 | +0 [+0, +0] | +0 [+0, +0] | -6 [-11, -2] | +4 [+0, +9] | -2 [-5, +0] | +1 [+0, +3] | +1 [-4, +6] | +2 [-2, +6] | +0 [+0, +0] |
| S-PAI - D1-s0 | +1 [+0, +3] | +1 [+0, +3] | -6 [-12, -1] | +2 [-3, +7] | -3 [-7, +0] | +0 [+0, +0] | -2 [-6, +2] | +2 [+0, +5] | +0 [+0, +0] |

val160 S-PAI - D3-s0: new +0.025 [-0.016, +0.067] (helps), old +0.025 [-0.016, +0.067]; S-PAI - D3 2-seed mean: new +0.014 [-0.021, +0.050], old +0.014 [-0.021, +0.050].

## Gates

- build: None
- diff_SP: True
- g13: True
- g4: True
- gcode: True
- gpre_SP: True
- gw0: True
- gw_draw_files: True
- lr_full_SP: True
- post_SP: True
- ratio_SP: True
- smoke_SP: True
- smoke_SP.json.dgc: True
- smoke_lr_SP: True
- w_NTV1D2S: None
- w_NTV1D2SZ: None
- S / imit gradient ratio: 500 steps after T 0.474 (band 0.21-1.04, pass); whole post-T segment 0.389 (D3 0.419)

| navtrain dev ADE | step 10 000 | end |
|:--|--:|--:|
| S-PAI | 0.5440 | 0.5305 |
| D3-s0 | 0.5440 | 0.5304 |

## Cost

GPU job-hours: gates 0.63, roll 8.10, train 1.08, read281 3.07, v160 1.73; total 14.6; wall 2.9 h.
