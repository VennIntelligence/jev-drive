# POOL-PILOT verdict (PAI 281, ACC + GAP, Harmonizer OFF)

277 scenes scored by all seven reads with no driver RPC error in any of them (both scorers on the same set).

## Reads

| read | old mean | new mean | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit p | slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL1-F-s0 | 0.717 | 0.793 | 65 / 44 | 16 | 6 + 1 | 14 | 23 | 5 | 27 | 10 | 32/37 | 0.5227 |
| PCL1B-F-s0 | 0.703 | 0.768 | 72 / 54 | 21 | 11 + 2 | 13 | 20 | 5 | 23 | 6 | 32/37 | 0.5375 |
| PCL0-L18-s0 | 0.737 | 0.806 | 60 / 41 | 14 | 7 + 0 | 16 | 21 | 2 | 27 | 9 | 32/37 | 0.5366 |
| NTV0-L18-s0 | 0.745 | 0.817 | 58 / 38 | 12 | 6 + 0 | 15 | 22 | 3 | 25 | 11 | 33/37 | 0.5362 |
| RB0-F-s0 | 0.648 | 0.717 | 87 / 68 | 19 | 14 + 4 | 21 | 21 | 8 | 22 | 8 | 32/37 | 0.5276 |
| RB0L-F-s0 | 0.600 | 0.683 | 101 / 78 | 37 | 11 + 1 | 18 | 24 | 10 | 22 | 9 | 33/37 | 0.5110 |
| NTV0NS-L18-s0 | 0.626 | 0.698 | 92 / 72 | 21 | 13 + 2 | 20 | 22 | 14 | 24 | 6 | 32/37 | 0.5287 |

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | old scorer | new scorer | zeros (old) |
|:--|--:|--:|--:|
| PCL1-F-s0 - NTV0-L18-s0 | -0.029 [-0.067, +0.009] | -0.025 [-0.062, +0.010] | +7 [-3, +18] |
| PCL1-F-s0 - RB0-F-s0 | +0.068 [+0.026, +0.112] | +0.076 [+0.035, +0.118] | -22 [-34, -10] |
| PCL1-F-s0 - RB0L-F-s0 | +0.117 [+0.072, +0.163] | +0.110 [+0.067, +0.153] | -36 [-49, -23] |
| PCL1-F-s0 - NTV0NS-L18-s0 | +0.091 [+0.045, +0.136] | +0.094 [+0.051, +0.138] | -27 [-40, -14] |
| PCL1B-F-s0 - NTV0-L18-s0 | -0.043 [-0.075, -0.012] | -0.050 [-0.082, -0.020] | +14 [+5, +23] |
| PCL1B-F-s0 - RB0-F-s0 | +0.054 [+0.020, +0.090] | +0.051 [+0.015, +0.087] | -15 [-26, -4] |
| PCL1B-F-s0 - RB0L-F-s0 | +0.103 [+0.053, +0.153] | +0.085 [+0.037, +0.135] | -29 [-44, -14] |
| PCL1B-F-s0 - NTV0NS-L18-s0 | +0.077 [+0.038, +0.117] | +0.069 [+0.030, +0.110] | -20 [-32, -8] |
| PCL0-L18-s0 - NTV0-L18-s0 | -0.008 [-0.040, +0.023] | -0.012 [-0.044, +0.018] | +2 [-7, +11] |
| PCL0-L18-s0 - RB0-F-s0 | +0.089 [+0.051, +0.130] | +0.089 [+0.051, +0.128] | -27 [-39, -16] |
| PCL0-L18-s0 - RB0L-F-s0 | +0.137 [+0.088, +0.188] | +0.123 [+0.075, +0.173] | -41 [-55, -27] |
| PCL0-L18-s0 - NTV0NS-L18-s0 | +0.111 [+0.070, +0.154] | +0.107 [+0.067, +0.150] | -32 [-44, -21] |
| PCL1-F-s0 - PCL1B-F-s0 | +0.014 [-0.026, +0.055] | +0.025 [-0.013, +0.064] | -7 [-19, +4] |
| PCL1-F-s0 - PCL0-L18-s0 | -0.020 [-0.060, +0.019] | -0.013 [-0.051, +0.024] | +5 [-6, +16] |

Zero / partial classes vs NTV0-L18-s0 and NTV0NS-L18-s0 (count difference, 95 % CI):

| arm - ref | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL1-F-s0 - NTV0-L18-s0 | +1 [-4, +6] | -1 [-6, +4] | +4 [-3, +11] | +2 [-2, +6] | +1 [+0, +3] | +0 [-5, +5] | +2 [+0, +5] | -1 [-5, +2] | +0 [+0, +0] |
| PCL1-F-s0 - NTV0NS-L18-s0 | +1 [-3, +5] | -6 [-13, +1] | -5 [-14, +4] | -9 [-16, -3] | -1 [-3, +0] | -7 [-14, +0] | +3 [-3, +9] | +4 [+0, +9] | +0 [+0, +0] |
| PCL1-F-s0 - RB0-F-s0 | +2 [-3, +8] | -7 [-13, -2] | -3 [-11, +5] | -3 [-9, +3] | -3 [-8, +1] | -8 [-15, -2] | +5 [-1, +11] | +2 [-3, +7] | +0 [+0, +0] |
| PCL1B-F-s0 - NTV0-L18-s0 | -2 [-6, +2] | -2 [-7, +3] | +9 [+3, +16] | +2 [+0, +5] | +2 [+0, +5] | +5 [+1, +10] | -2 [-8, +4] | -5 [-10, -1] | +0 [+0, +0] |
| PCL1B-F-s0 - NTV0NS-L18-s0 | -2 [-6, +2] | -7 [-13, -2] | +0 [-8, +8] | -9 [-17, -2] | +0 [-4, +4] | -2 [-8, +3] | -1 [-8, +5] | +0 [-5, +5] | +0 [+0, +0] |
| PCL1B-F-s0 - RB0-F-s0 | -1 [-6, +4] | -8 [-14, -3] | +2 [-4, +8] | -3 [-10, +3] | -2 [-6, +2] | -3 [-8, +2] | +1 [-6, +8] | -2 [-7, +3] | +0 [+0, +0] |
| PCL0-L18-s0 - NTV0-L18-s0 | -1 [-6, +4] | +1 [-5, +7] | +2 [-3, +7] | -1 [-5, +2] | +0 [+0, +0] | +1 [-2, +5] | +2 [-3, +7] | -2 [-5, +0] | +0 [+0, +0] |
| PCL0-L18-s0 - NTV0NS-L18-s0 | -1 [-5, +3] | -4 [-12, +4] | -7 [-14, +0] | -12 [-20, -5] | -2 [-5, +0] | -6 [-11, -2] | +3 [-3, +9] | +3 [-1, +8] | +0 [+0, +0] |
| PCL0-L18-s0 - RB0-F-s0 | +0 [-5, +5] | -5 [-12, +1] | -5 [-11, +1] | -6 [-11, -2] | -4 [-8, -1] | -7 [-13, -2] | +5 [-1, +11] | +1 [-3, +6] | +0 [+0, +0] |

## Registered lines (gap share g = (arm - RB0-F) / (NTV0-s0 - RB0-F); >= 74 % with the CI vs the offline counterpart above 0 / <= 21 %)

- new scorer: RB0-F 0.717, NTV0-s0 0.817 -> lines 0.791 / 0.738
  - P1 PCL1-F-s0: 0.793, g = **75%**, vs RB0L-F-s0 +0.110 [+0.067, +0.153] -> **carries the gain**
  - P1B PCL1B-F-s0: 0.768, g = **50%**, vs RB0-F-s0 +0.051 [+0.015, +0.087] -> **undecided**
  - P0 PCL0-L18-s0: 0.806, g = **88%** -> **at NTV0 (>= 74 %)**
- old scorer: RB0-F 0.648, NTV0-s0 0.745 -> lines 0.720 / 0.669
  - P1 PCL1-F-s0: 0.717, g = **71%**, vs RB0L-F-s0 +0.117 [+0.072, +0.163] -> **undecided**
  - P1B PCL1B-F-s0: 0.703, g = **56%**, vs RB0-F-s0 +0.054 [+0.020, +0.090] -> **undecided**
  - P0 PCL0-L18-s0: 0.737, g = **91%** -> **at NTV0 (>= 74 %)**

## Highway (behavior_stats Q1 rules; free-road v_ego / v_human on highway stations; slow + rear-hit points by end class, new scorer)

| read | steps | median | p10 | < 0.85 | > 1.10 | mean dv m/s | slow + rear pts | registered line |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| PCL1-F-s0 | 804 | 1.00 | 0.84 | 10% | 6% | -0.73 | 1.16 | not slow on highway |
| PCL1B-F-s0 | 786 | 1.00 | 0.85 | 10% | 8% | -0.57 | 1.29 | not slow on highway |
| PCL0-L18-s0 | 824 | 1.00 | 0.86 | 9% | 6% | -0.60 | 1.05 | not slow on highway |
| NTV0-L18-s0 | 803 | 1.00 | 0.87 | 9% | 6% | -0.62 | 1.06 | not slow on highway |
| RB0-F-s0 | 827 | 1.00 | 0.87 | 9% | 8% | -0.49 | 0.86 | not slow on highway |

Highway-majority scenes with a free-road median ratio < 0.90 (scene: median / stations):

- PCL1-F-s0: 3 of 32 scenes with free-road steps: 1a050ead 0.87 / 36, 4e454838 0.90 / 36, ffe5b17d 0.81 / 36
- PCL1B-F-s0: 1 of 32 scenes with free-road steps: ffe5b17d 0.81 / 36
- PCL0-L18-s0: 4 of 32 scenes with free-road steps: 1a050ead 0.83 / 36, 4b5a7c8b 0.88 / 15, 4e454838 0.90 / 36, ffe5b17d 0.81 / 36
- NTV0-L18-s0: 3 of 32 scenes with free-road steps: 1a050ead 0.86 / 36, 4e454838 0.90 / 36, ffe5b17d 0.81 / 36
- RB0-F-s0: 1 of 32 scenes with free-road steps: ffe5b17d 0.81 / 36
- ramp (listed only): PCL1-F-s0 0 steps, median None; PCL1B-F-s0 0 steps, median None; PCL0-L18-s0 0 steps, median None; NTV0-L18-s0 0 steps, median None; RB0-F-s0 0 steps, median None

## Gates in the run

- S / imit gradient ratio, 500 steps after T (P1): 0.584 (band 0.21-1.04; design's literal 0.224-0.598: in) -> pass
- S / imit gradient ratio, 500 steps after T (P1B): 0.389 (band 0.21-1.04; design's literal 0.224-0.598: in) -> pass
- S / imit gradient ratio, 500 steps after T (P0): 0.440 (band 0.21-1.04; design's literal 0.224-0.598: in) -> pass
- post (P1): final dev ADE 0.5227 (<= 0.566); whole pre-T segment vs RB0L-F-s0: 2077 / 2077 scalars bit-identical, steps <= 10000
- post (P1B): final dev ADE 0.5375 (<= 0.566); whole pre-T segment vs RB0-F-s0: 1042 / 1042 scalars bit-identical, steps <= 5000
- post (P0): final dev ADE 0.5366 (<= 0.566)

| navtrain dev ADE | m |
|:--|--:|
| PCL1-F-s0 @ 10000 | 0.5440 |
| PCL1-F-s0 @ 19500 | 0.5227 |
| PCL1B-F-s0 @ 5000 | 0.5599 |
| PCL1B-F-s0 @ 10000 | 0.5375 |
| PCL0-L18-s0 (RB0-F + 9 500) | 0.5366 |
| RB0L-F-s0 @ 10000 | 0.5440 |
| RB0L-F-s0 @ 19500 | 0.5110 |
| RB0-F-s0 | 0.5276 |
| NTV0-L18-s0 | 0.5362 |
| NTV0NS-L18-s0 | 0.5287 |

## Cost

GPU jobs 12.0 job-hours (training 2.45, reads 9.33, gates / smokes 0.19); wall 1.9 h from the first start to the last end.
