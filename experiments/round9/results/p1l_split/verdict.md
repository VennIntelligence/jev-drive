# P1L-SPLIT verdict (PAI 281, ACC + GAP, Harmonizer OFF)

278 scenes scored by all six reads (both scorers) with no driver RPC error in any of them.

## Reads

| read | new mean | old mean | zeros new / old | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P1La (PCL1LA-F-s0) | 0.795 | 0.719 | 44 / 65 | 18 | 5 + 1 | 14 | 24 | 3 | 27 | 9 | 32/37 | 0.5229 |
| P1Lb (PCL1LB-F-s0) | 0.792 | 0.705 | 43 / 67 | 17 | 6 + 2 | 11 | 25 | 6 | 32 | 11 | 33/37 | 0.5245 |
| P1L (PCL1L-F-s0) | 0.816 | 0.737 | 38 / 60 | 14 | 6 + 3 | 10 | 23 | 4 | 27 | 11 | 32/37 | 0.5250 |
| P1 (PCL1-F-s0) | 0.793 | 0.718 | 44 / 65 | 16 | 6 + 1 | 14 | 23 | 5 | 28 | 10 | 32/37 | 0.5227 |
| NTV0-s0 (NTV0-L18-s0) | 0.818 | 0.746 | 38 / 58 | 12 | 6 + 0 | 15 | 22 | 3 | 26 | 11 | 33/37 | 0.5362 |
| RB0-F (RB0-F-s0) | 0.718 | 0.649 | 68 / 87 | 19 | 14 + 4 | 21 | 21 | 8 | 23 | 8 | 32/37 | 0.5276 |

## Registered lines (within 0.01 of P1L -> carries P1L's gain; within 0.01 of P1 -> alone does nothing; else partial)

- new scorer: P1L 0.816, P1 0.793
  - P1LA: 0.795 (vs P1L -0.021, vs P1 +0.002, 8% of P1L - P1; g 77%) -> **alone does nothing**
  - P1LB: 0.792 (vs P1L -0.025, vs P1 -0.002, -7% of P1L - P1; g 74%) -> **alone does nothing**
- old scorer (reported only): P1L 0.737, P1 0.718
  - P1LA: 0.719 (vs P1L -0.018, vs P1 +0.002, 9% of P1L - P1; g 72%) -> **alone does nothing**
  - P1LB: 0.705 (vs P1L -0.032, vs P1 -0.012, -63% of P1L - P1; g 58%) -> **partial**

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | new scorer | old scorer | zeros (old) |
|:--|--:|--:|--:|
| P1La - P1L | -0.021 [-0.052, +0.009] | -0.018 [-0.049, +0.014] | +5 [-4, +14] |
| P1La - P1 | +0.002 [-0.028, +0.032] | +0.002 [-0.029, +0.033] | +0 [-9, +9] |
| P1La - NTV0-s0 | -0.023 [-0.058, +0.012] | -0.027 [-0.064, +0.010] | +7 [-3, +17] |
| P1La - RB0-F | +0.077 [+0.037, +0.120] | +0.070 [+0.028, +0.114] | -22 [-35, -10] |
| P1Lb - P1L | -0.025 [-0.053, +0.002] | -0.032 [-0.063, -0.001] | +7 [-2, +16] |
| P1Lb - P1 | -0.002 [-0.028, +0.024] | -0.012 [-0.043, +0.018] | +2 [-7, +11] |
| P1Lb - NTV0-s0 | -0.026 [-0.063, +0.009] | -0.041 [-0.080, -0.003] | +9 [-2, +20] |
| P1Lb - RB0-F | +0.074 [+0.035, +0.114] | +0.056 [+0.013, +0.099] | -20 [-33, -8] |
| P1La - P1Lb | +0.003 [-0.031, +0.038] | +0.014 [-0.022, +0.051] | -2 [-12, +8] |
| P1L - P1 | +0.023 [-0.010, +0.058] | +0.020 [-0.016, +0.056] | -5 [-15, +5] |

Zero / partial classes (old scorer classes; count difference, 95 % CI):

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P1La - P1L | +1 [-2, +5] | +4 [+0, +9] | +4 [+0, +9] | -1 [-5, +3] | -2 [-5, +0] | -1 [-5, +2] | +0 [-4, +4] | -2 [-5, +0] | +0 [+0, +0] |
| P1La - P1 | +1 [-2, +4] | +0 [-3, +3] | +2 [-3, +7] | -2 [-6, +2] | +0 [-3, +3] | -1 [-6, +4] | -1 [-6, +4] | -1 [-5, +2] | +0 [+0, +0] |
| P1Lb - P1L | +2 [-2, +6] | +1 [-2, +5] | +3 [-1, +8] | +2 [-3, +7] | -1 [-4, +2] | +0 [-6, +5] | +5 [+0, +11] | +0 [-4, +4] | +0 [+0, +0] |
| P1Lb - P1 | +2 [-3, +7] | -3 [-8, +1] | +1 [-4, +6] | +1 [-2, +5] | +1 [+0, +3] | +0 [-3, +3] | +4 [-1, +10] | +1 [+0, +3] | +0 [+0, +0] |
| P1L - P1 | +0 [-4, +4] | -4 [-9, +0] | -2 [-8, +3] | -1 [-5, +3] | +2 [-2, +6] | +0 [-5, +5] | -1 [-5, +3] | +1 [-2, +5] | +0 [+0, +0] |

## Gates and schedule

- P1LA: S / imit ratio 500 steps after T 0.653 (pass, band 0.21-1.04), whole post-T 0.431; final dev ADE 0.5229; pre-T vs P1L 7671 / 7671 scalars bit-identical; lr 780 points, max rel. error 0.0e+00
- P1LB: S / imit ratio 500 steps after T 0.596 (pass, band 0.21-1.04), whole post-T 0.433; final dev ADE 0.5245; pre-T vs P1L 7671 / 7671 scalars bit-identical; lr 780 points, max rel. error 0.0e+00
- reference ratios: P1L 0.755 / 0.462, P1 0.584 / 0.408 (500 steps / whole post-T)

| step | P1La g0 / g1 / g2 | P1Lb g0 / g1 / g2 |
|--:|:--|:--|
| 10025 | 1.50e-07 / 1.50e-06 / 1.43e-04 | 1.43e-05 / 1.43e-04 / 7.50e-05 |
| 10100 | 6.00e-07 / 6.00e-06 / 1.42e-04 | 1.42e-05 / 1.42e-04 / 3.00e-04 |
| 10500 | 2.98e-06 / 2.98e-05 / 1.32e-04 | 1.32e-05 / 1.32e-04 / 2.98e-04 |
| 15000 | 1.54e-06 / 1.54e-05 / 3.77e-05 | 3.77e-06 / 3.77e-05 / 1.54e-04 |
| 19500 | 3.00e-07 / 3.00e-06 / 1.95e-12 | 1.95e-13 / 1.95e-12 / 3.00e-05 |

| navtrain dev ADE | m |
|:--|--:|
| PCL1LA-F-s0 @ 10000 | 0.5440 |
| PCL1LA-F-s0 @ 19500 | 0.5229 |
| PCL1LB-F-s0 @ 10000 | 0.5440 |
| PCL1LB-F-s0 @ 19500 | 0.5245 |
| PCL1L-F-s0 @ 10000 | 0.5440 |
| PCL1L-F-s0 @ 19500 | 0.5250 |
| PCL1-F-s0 @ 10000 | 0.5440 |
| PCL1-F-s0 @ 19500 | 0.5227 |

## Cost

GPU jobs 10.5 job-hours (training 3.19, reads 7.12, gates / smokes 0.20); wall 2.5 h from the first start to the last end.
