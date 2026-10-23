# P1L verdict (PAI 281, ACC + GAP, Harmonizer OFF)

278 scenes scored by all five reads (both scorers) with no driver RPC error in any of them.

## Reads

| read | old mean | new mean | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL1L-F-s0 | 0.737 | 0.816 | 60 / 38 | 14 | 6 + 3 | 10 | 23 | 4 | 27 | 11 | 32/37 | 0.5250 |
| PCL1-F-s0 | 0.718 | 0.793 | 65 / 44 | 16 | 6 + 1 | 14 | 23 | 5 | 28 | 10 | 32/37 | 0.5227 |
| PCL0-L18-s0 | 0.738 | 0.806 | 60 / 41 | 14 | 7 + 0 | 16 | 21 | 2 | 27 | 9 | 32/37 | 0.5366 |
| NTV0-L18-s0 | 0.746 | 0.818 | 58 / 38 | 12 | 6 + 0 | 15 | 22 | 3 | 26 | 11 | 33/37 | 0.5362 |
| RB0-F-s0 | 0.649 | 0.718 | 87 / 68 | 19 | 14 + 4 | 21 | 21 | 8 | 23 | 8 | 32/37 | 0.5276 |

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | new scorer | old scorer | zeros (old) |
|:--|--:|--:|--:|
| PCL1L-F-s0 - PCL1-F-s0 | +0.023 [-0.010, +0.058] | +0.020 [-0.016, +0.056] | -5 [-15, +5] |
| PCL1L-F-s0 - PCL0-L18-s0 | +0.010 [-0.025, +0.045] | -0.001 [-0.037, +0.035] | +0 [-10, +10] |
| PCL1L-F-s0 - NTV0-L18-s0 | -0.002 [-0.039, +0.036] | -0.009 [-0.048, +0.029] | +2 [-9, +13] |
| PCL1L-F-s0 - RB0-F-s0 | +0.099 [+0.057, +0.143] | +0.088 [+0.045, +0.133] | -27 [-40, -15] |

Zero / partial classes, P1L minus each read (count difference, 95 % CI; old scorer classes):

| P1L - ref | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| - PCL1-F-s0 | +0 [-4, +4] | -4 [-9, +0] | -2 [-8, +3] | -1 [-5, +3] | +2 [-2, +6] | +0 [-5, +5] | -1 [-5, +3] | +1 [-2, +5] | +0 [+0, +0] |
| - PCL0-L18-s0 | +2 [-2, +6] | -6 [-13, +0] | +0 [-6, +6] | +2 [-2, +6] | +3 [+0, +7] | -1 [-6, +4] | +0 [-6, +6] | +2 [-2, +6] | +0 [+0, +0] |
| - NTV0-L18-s0 | +1 [-3, +6] | -5 [-12, +2] | +2 [-3, +8] | +1 [-3, +5] | +3 [+0, +7] | +0 [-5, +5] | +1 [-2, +5] | +0 [-4, +4] | +0 [+0, +0] |
| - RB0-F-s0 | +2 [-3, +7] | -11 [-18, -4] | -5 [-13, +2] | -4 [-10, +1] | -1 [-4, +2] | -8 [-15, -1] | +4 [-2, +10] | +3 [-1, +8] | +0 [+0, +0] |

## Registered lines (g = (arm - RB0-F) / (NTV0-s0 - RB0-F); >= 88 % and P1L - P1 > 0 / <= 75 %; new scorer registered, old reported)

- new scorer: RB0-F 0.718, NTV0-s0 0.818 -> lines 0.806 / 0.793; g P1L **98%**, P1 75%, P0 88%; P1L - P1 +0.023 -> **lr at S entry was the cause; use it in NTV1**
- old scorer: RB0-F 0.649, NTV0-s0 0.746 -> lines 0.734 / 0.722; g P1L **91%**, P1 71%, P0 92%; P1L - P1 +0.020 -> **lr at S entry was the cause; use it in NTV1** (reported only)

## Highway (behavior_stats Q1 rules)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | line |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| PCL1L-F-s0 | 820 | 1.00 | 0.85 | 10% | 6% | -0.63 | 1.10 | not slow on highway |
| NTV0-L18-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | not slow on highway |
| RB0-F-s0 | 827 | 1.00 | 0.87 | 9% | 8% | -0.49 | 0.86 | not slow on highway |
| PCL1-F-s0 | 804 | 1.00 | 0.84 | 10% | 6% | -0.73 | 1.16 | not slow on highway |

P1L highway-majority scenes with a free-road median < 0.90: 2: fd01e9ab 0.74, ffe5b17d 0.81

## Gates and schedule

- S / imit gradient ratio, 500 steps after T: P1L 0.755 (band 0.21-1.04, pass), P1 0.584; whole post-T segment: P1L 0.462, P1 0.408
- post: final dev ADE 0.5250; pre-T segment vs PCL1-F-s0: 6471 / 6471 scalars bit-identical (steps <= 10 000)
- lr: 780 logged points, max relative error vs the formula 0.0e+00

| step | g0 net | g1 adapter | g2 route branch |
|--:|--:|--:|--:|
| 10025 | 1.500e-07 | 1.500e-06 | 7.500e-05 |
| 10100 | 5.999e-07 | 5.999e-06 | 2.999e-04 |
| 10500 | 2.982e-06 | 2.982e-05 | 2.982e-04 |
| 15000 | 1.539e-06 | 1.539e-05 | 1.539e-04 |
| 19500 | 3.000e-07 | 3.000e-06 | 3.000e-05 |

| navtrain dev ADE | m |
|:--|--:|
| PCL1L-F-s0 @ 10000 | 0.5440 |
| PCL1L-F-s0 @ 19500 | 0.5250 |
| PCL1-F-s0 @ 10000 | 0.5440 |
| PCL1-F-s0 @ 19500 | 0.5227 |
| PCL0-L18-s0 | 0.5366 |
| NTV0-L18-s0 | 0.5362 |
| RB0-F-s0 | 0.5276 |

## Cost

GPU jobs 4.2 job-hours (training 1.01, reads 3.06, gates / smokes 0.10); wall 1.5 h from the first start to the last end.
