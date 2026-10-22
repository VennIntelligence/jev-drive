# PAI-BASE base gate (plan section 4): RB0-P (PAI share 30 / 50 / 70 % of the imitation rows) vs RB0-F-s0, PAI 281 ACC + GAP, Harmonizer OFF

Identity (RB0-F-s0's argv rerun on HEAD, eager, 10 000 steps vs the stored RB0-F-s0): identical.

RB0-F-s0: mean 0.650, zeros 88, offroad + zero-other 19 + 14, launched 32/37.

| PAI share | mean | - RB0-F (paired, 95 % CI) | zeros | offroad + zero-other | launched | gate |
|:--|--:|--:|--:|--:|--:|:--|
| 30 % | 0.593 | -0.057 [-0.110, -0.004] | 105 | 30 + 27 | 29/37 | ci_lo_gt0 launch |
| 50 % | 0.605 | -0.045 [-0.097, +0.008] | 102 | 28 + 24 | 27/37 | ci_lo_gt0 launch |
| 70 % | 0.564 | -0.085 [-0.140, -0.029] | 114 | 30 + 28 | 29/37 | ci_lo_gt0 launch |

Rule: paired 95 % CI lower bound > 0, launched >= RB0-F's - 2, the replayed draws pass; among passing shares the highest paired mean (tie: the lower share). Identity must hold.

**Carried: none (the line stops).**
