# RB0-LONG verdict (PAI 281, ACC + GAP, Harmonizer OFF)

Old scorer (the read's own, as registered), 280 scenes common to RB0L / RB0-F-s0 / NTV0-L18-s0 without RPC error.

- RB0L-F-s0 mean **0.600**; registered line: >= 0.72 / <= 0.67 -> **more offline steps do not give NTV0's gain**
- RB0L - RB0-F-s0: -0.048 [-0.093, -0.002]
- RB0L - NTV0-L18-s0: -0.144 [-0.191, -0.099]
- NTV0-L18-s0 - RB0-F-s0: +0.096 [+0.060, +0.135]
- RB0L - NTV0NS-L18-s0 (278 scenes common to the four reads): -0.026 [-0.070, +0.017]; NTV0NS mean 0.624, RB0L 0.597 on that set

| read | mean | zeros | z_offroad | z_zero-other | z_side | z_route-turn | z_overshoot | z_AF-front | p_rear-hit | p_slow | launch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0L-F-s0 | 0.600 | 102 | 37 | 11 | 1 | 18 | 25 | 10 | 23 | 9 | 33/37 |
| RB0-F-s0 | 0.648 | 88 | 19 | 14 | 4 | 21 | 22 | 8 | 23 | 8 | 32/37 |
| NTV0-L18-s0 | 0.744 | 59 | 12 | 6 | 0 | 15 | 23 | 3 | 26 | 11 | 33/37 |
| NTV0NS-L18-s0 (own common set) | 0.624 | 93 | 21 | 13 | 2 | 20 | 23 | 14 | 24 | 6 | 32/37 |

drift / lane change = zero-other + side; rear-ended / slow = partials (0 < score < 0.98); launch = launched / launch scenes.

| navtrain dev ADE | m |
|:--|--:|
| RB0L-F-s0 @ 10000 | 0.5440 |
| RB0L-F-s0 @ 19500 | 0.5110 |
| RB0-F-s0 @ 10000 | 0.5276 |
| NTV0-L18-s0 (10 000 + 9 500) | 0.5362 |

## New scorer (scorer_v2, decision 453; reported next to the registered old-scorer line)

280 scenes common to the three new-scorer reads without RPC error (RB0L rescored mean over all its scenes 0.680). RB0L 0.682, RB0-F-s0 0.716, NTV0-L18-s0 0.816; zeros 79 / 69 / 39.

- RB0L - RB0-F-s0: -0.034 [-0.079, +0.012]; RB0L - NTV0-L18-s0: -0.133 [-0.179, -0.089]
- share of the RB0-F -> NTV0 gap: **-34%**; the registered lines as shares: >= 74% / <= 21% -> **more offline steps do not give NTV0's gain**
- old and new scorer land on the same line
