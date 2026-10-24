# NTV1 K / H verdict (PAI 281, ACC + GAP; D1's serve)

## Harmonizer OFF

278 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NTK1-F-s0': 0, 'NT3D1-F-s0': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1-K OFF | 0.868 | 0.792 | 28 / 49 | 21 | 19 | 2 | 0 | 2 | 22 | 4 | 18 | 3 | 34/37 | 0.5336 |
| D1-s0 OFF | 0.868 | 0.799 | 27 / 46 | 18 | 15 | 3 | 0 | 3 | 20 | 5 | 22 | 7 | 36/37 | 0.5297 |
| P1L OFF | 0.813 | 0.734 | 39 / 61 | 23 | 14 | 6 | 3 | 10 | 24 | 4 | 27 | 11 | 32/37 | 0.5250 |
| NTV0-s0 OFF | 0.814 | 0.742 | 39 / 59 | 18 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 | 0.5362 |
| R8K-s2 OFF | 0.872 | 0.804 | 28 / 47 | 20 | 14 | 4 | 2 | 2 | 20 | 5 | 18 | 4 | 35/37 | 0.5578 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| D1-K OFF - D1-s0 OFF | +0.000 [-0.022, +0.022] | -0.007 [-0.031, +0.016] | +3 [-3, +10] | +3 [-2, +8] | -1 [-3, +0] | -1 [-5, +2] |
| D1-K OFF - P1L OFF | +0.055 [+0.018, +0.094] | +0.059 [+0.019, +0.099] | -12 [-23, -1] | -2 [-12, +7] | -8 [-14, -3] | +0 [-5, +5] |
| D1-K OFF - NTV0-s0 OFF | +0.053 [+0.012, +0.096] | +0.050 [+0.004, +0.095] | -10 [-22, +2] | +3 [-5, +12] | -13 [-20, -6] | +1 [-3, +5] |
| D1-K OFF - R8K-s2 OFF | -0.005 [-0.039, +0.030] | -0.012 [-0.049, +0.025] | +2 [-8, +12] | +1 [-7, +9] | +0 [+0, +0] | -1 [-6, +4] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1-K OFF - D1-s0 OFF | +2 [+0, +5] | -1 [-3, +0] | +4 [+0, +9] | -1 [-5, +2] | +0 [+0, +0] | -1 [-3, +0] | -4 [-9, +0] | -4 [-8, -1] | +0 [+0, +0] |
| D1-K OFF - P1L OFF | -2 [-8, +3] | -8 [-14, -3] | +5 [-3, +13] | +0 [-5, +5] | -3 [-7, +0] | -4 [-9, +0] | -9 [-18, +0] | -8 [-14, -3] | +0 [+0, +0] |
| D1-K OFF - NTV0-s0 OFF | -1 [-6, +4] | -13 [-20, -6] | +7 [+0, +14] | +1 [-3, +5] | +0 [+0, +0] | -4 [-9, +0] | -8 [-17, +1] | -8 [-14, -3] | +0 [+0, +0] |
| D1-K OFF - R8K-s2 OFF | +2 [-2, +6] | +0 [+0, +0] | +5 [-2, +12] | -1 [-6, +4] | -2 [-5, +0] | -2 [-5, +0] | +0 [-9, +9] | -1 [-5, +2] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (202 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (182 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| D1-K OFF | 14 | 9 | 11.77 | 9 | 41 | 33 / 40 |
| D1-s0 OFF | 14 | 8 | 10.40 | 8 | 34 | 28 / 40 |
| P1L OFF | 16 | 13 | 13.94 | 12 | 15 | 10 / 40 |
| R8K-s2 OFF | 14 | 9 | 10.53 | 9 | 40 | 40 / 40 |

## Harmonizer ON

279 scenes scored by every read below (both scorers) with no driver RPC error in any; RPC scenes per read {'NTK1-F-s0': 0, 'NT3D1-F-s0': 0, 'PCL1L-F-s0': 0, 'NTV0-L18-s0': 0, 'R8K-L18-s2': 0}.

| read | new mean | old mean | zeros new / old | lateral | offroad | zero-other | side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1-K ON | 0.846 | 0.771 | 34 / 55 | 24 | 19 | 3 | 2 | 3 | 22 | 6 | 22 | 4 | 33/37 | 0.5336 |
| D1-s0 ON | 0.851 | 0.772 | 30 / 52 | 23 | 18 | 2 | 3 | 3 | 23 | 3 | 28 | 5 | 33/37 | 0.5297 |
| P1L ON | 0.804 | 0.733 | 41 / 61 | 26 | 18 | 7 | 1 | 6 | 22 | 7 | 27 | 10 | 31/37 | 0.5250 |
| NTV0-s0 ON | 0.809 | 0.745 | 40 / 58 | 23 | 17 | 6 | 0 | 9 | 21 | 5 | 27 | 9 | 33/37 | 0.5362 |
| R8K-s2 ON | 0.866 | 0.794 | 27 / 47 | 20 | 15 | 4 | 1 | 2 | 21 | 4 | 20 | 7 | 32/37 | 0.5578 |

Paired differences (95 % bootstrap CI, scenes as units; class counts from old-scorer facts):

| pair | new scorer | old scorer | zeros (old) | lateral zeros | route-turn | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| D1-K ON - D1-s0 ON | -0.005 [-0.028, +0.018] | -0.001 [-0.028, +0.024] | +3 [-4, +10] | +1 [-6, +7] | +0 [+0, +0] | +3 [+0, +7] |
| D1-K ON - P1L ON | +0.042 [+0.002, +0.081] | +0.038 [-0.002, +0.078] | -6 [-17, +5] | -2 [-11, +7] | -3 [-8, +1] | -1 [-5, +3] |
| D1-K ON - NTV0-s0 ON | +0.037 [-0.004, +0.076] | +0.026 [-0.017, +0.068] | -3 [-14, +8] | +1 [-8, +10] | -6 [-12, -1] | +1 [-4, +6] |
| D1-K ON - R8K-s2 ON | -0.020 [-0.055, +0.015] | -0.024 [-0.061, +0.013] | +8 [-2, +18] | +4 [-5, +13] | +1 [+0, +3] | +2 [-3, +7] |

All zero / partial classes (count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| D1-K ON - D1-s0 ON | -1 [-5, +2] | +0 [+0, +0] | +1 [-4, +6] | +3 [+0, +7] | -1 [-4, +2] | +1 [+0, +3] | -6 [-12, -1] | -1 [-5, +2] | +0 [+0, +0] |
| D1-K ON - P1L ON | +0 [-5, +5] | -3 [-8, +1] | +1 [-7, +9] | -1 [-5, +3] | +1 [-2, +4] | -4 [-9, +0] | -5 [-13, +3] | -6 [-12, -1] | +0 [+0, +0] |
| D1-K ON - NTV0-s0 ON | +1 [-4, +6] | -6 [-12, -1] | +2 [-5, +9] | +1 [-4, +6] | +2 [+0, +5] | -3 [-8, +1] | -5 [-14, +4] | -5 [-11, +1] | +0 [+0, +0] |
| D1-K ON - R8K-s2 ON | +1 [-2, +5] | +1 [+0, +3] | +4 [-4, +12] | +2 [-3, +7] | +1 [-2, +4] | -1 [-5, +2] | +2 [-6, +10] | -3 [-8, +1] | +0 [+0, +0] |

Negative flips on NTV0-s0's full-score scenes. Count A (INTERP's): NTV0 new >= 0.98 (203 scenes), read new < 0.98. Count B (step 1's): NTV0 old >= 0.98 (185 scenes), read new = 0.

| read | A: negative flips | of them to 0 | points lost | B: new zeros | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2's fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| D1-K ON | 18 | 11 | 13.24 | 11 | 34 | 24 / 33 |
| D1-s0 ON | 18 | 10 | 12.52 | 10 | 31 | 24 / 33 |
| P1L ON | 17 | 14 | 14.76 | 13 | 15 | 12 / 33 |
| R8K-s2 ON | 11 | 8 | 9.52 | 8 | 33 | 33 / 33 |

## Registered lines (D1-K - D1-s0; new scorer OFF registered, the rest reported)

- Harmonizer OFF, new scorer: +0.000 [-0.022, +0.022] -> **neutral** (registered)
- Harmonizer OFF, old scorer: -0.007 [-0.031, +0.016] -> **neutral**
- Harmonizer ON, new scorer: -0.005 [-0.028, +0.018] -> **neutral**
- Harmonizer ON, old scorer: -0.001 [-0.028, +0.024] -> **neutral**

## ON - OFF gap (277 scenes common to both blocks)

| scorer | D1-K | D1-s0 | D1-K gap - D1-s0 gap | other reads |
|:--|--:|--:|--:|:--|
| new | -0.022 | -0.017 | -0.005 [-0.035, +0.024] | P1L -0.009, NTV0-s0 -0.006, R8K-s2 -0.007 |
| old | -0.022 | -0.028 | +0.006 [-0.028, +0.039] | P1L -0.002, NTV0-s0 +0.002, R8K-s2 -0.010 |
(D1-s0's gap on its own common set, decision 471: -0.016.)

## What K changed in the labels

- Drawable DAgger rows 92561; floored (b5 faster than b3) 17515 (18.9%); 4 s arc increase p10 / p50 / p90 [0.78, 5.64, 18.42] m; straight rows floored 11372 / 63599 (increase [0.77, 5.64, 20.51] m).
- Not floored because: no human window 12541, human slows 7742, clock lag 9421, student not slower 33183, lead 16829, T4 revert 1235.
- By class (floored / drawable) {"launch": [561, 6547], "other": [10213, 46702], "stop_nolead": [1941, 18703], "turn_sharp": [4800, 20609]}; by speed {"2-5": [1661, 14347], "5-10": [6024, 33538], "10-15": [4618, 21947], "15-20": [2360, 8965], "20-25": [1008, 5320], ">=25": [1844, 8444]}.
- F2 on the b5 labels: 6060 drawable rows to weight 0 (on b3 labels the rule would take 9862); drawable 92561 -> 86501; floored rows still drawn 17233.
- G1: {"G1a": true, "G1b": true, "G1c": true, "T4": true}; G1a max |diff| 0.00e+00 m over 100.0%; G1b {"min_arc_change": -0.0008720667474122479, "max_change_over_floor": 0.004212534538964974, "max_floor_over_human": 0.0, "b3_start_offset_rows": 997}; G1c median |lag| 0.000 s; T4 p99 3.72 (b3 3.57).

## H (not run)

PAIHTW rows matching D1's pool by name: 0 of 282226 (by part {"pd": 0, "rb0": 0, "rtb1": 0, "tv": 0, "y": 0, "ys": 0}); store prefixes ['paih', 'paih05', 'paih15'], pool prefixes {'pd': ['paidc'], 'rb0': ['2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78', '2021.05.25.14.16.10_veh-35_00083_00485-24567f5ad57455c4', '2021.05.25.14.16.10_veh-35_00083_00485-34f30283d3bd53ec', '2021.05.25.14.16.10_veh-35_00083_00485-431ae29947e95c26', '2021.05.25.14.16.10_veh-35_00083_00485-45e8b10f95925c47'], 'rtb1': ['2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78', '2021.05.25.14.16.10_veh-35_00083_00485-24567f5ad57455c4', '2021.05.25.14.16.10_veh-35_00083_00485-34f30283d3bd53ec', '2021.05.25.14.16.10_veh-35_00083_00485-431ae29947e95c26', '2021.05.25.14.16.10_veh-35_00083_00485-45e8b10f95925c47'], 'tv': [], 'y': ['2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78', '2021.05.25.14.16.10_veh-35_00083_00485-24567f5ad57455c4', '2021.05.25.14.16.10_veh-35_00083_00485-34f30283d3bd53ec', '2021.05.25.14.16.10_veh-35_00083_00485-431ae29947e95c26', '2021.05.25.14.16.10_veh-35_00083_00485-45e8b10f95925c47'], 'ys': ['2021.05.25.14.16.10_veh-35_00083_00485-0c318d7923d15b78', '2021.05.25.14.16.10_veh-35_00083_00485-24567f5ad57455c4', '2021.05.25.14.16.10_veh-35_00083_00485-34f30283d3bd53ec', '2021.05.25.14.16.10_veh-35_00083_00485-431ae29947e95c26', '2021.05.25.14.16.10_veh-35_00083_00485-45e8b10f95925c47']}. Cost estimate: see the plan, section 2.

## Gates

- d1_argv_vs_stored_D1: True
- gw_draw_file: True
- k_preT_vs_stored_D1: True
- ksw_K_switch: True
- lr_full_K: True
- post_K: True
- ratio_K: True
- rows: True
- smoke_D1: True
- smoke_K: True
- smoke_lr_D1: True
- smoke_lr_K: True
- t3_switch_off: True
- t3_weight0: True
- teach: None
- twin_cover: None
- verdict: None
- w_NTV1D: None
- w_NTV1K: None
- S / imit gradient ratio, 500 steps after T: 0.352 (band 0.21-1.04); whole post-T segment 0.423

| navtrain dev ADE | step 10 000 | end |
|:--|--:|--:|
| D1-K | 0.5440 | 0.5336 |
| D1-s0 | 0.5440 | 0.5297 |

## Cost

GPU jobs 14.2 job-hours (training 1.38, reads 12.66, gates / smokes 0.20); wall 3.1 h.
