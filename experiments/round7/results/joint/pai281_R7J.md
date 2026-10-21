# PAI 281, ACC + GAP (+ JEV_CMD=heading for R7J): round7 joint arm R7J s0 / s1 vs stored R6DAG s0 / s1 and NTV0; Harmonizer OFF and ON

Scenes every read scored without a driver RPC error: 274 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 274 | 0.750 | 56 | 22 | 14 | 11 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| R6DAG-s0-ag | 274 | 0.779 | 51 | 19 | 2 | 19 | 4 | 2 | 5 | 22 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 274 | 0.816 | 42 | 18 | 2 | 12 | 3 | 3 | 4 | 20 | 6 | 0 | 35/37 |
| R7J-s0-ag | 274 | 0.787 | 51 | 21 | 3 | 16 | 5 | 3 | 3 | 17 | 6 | 0 | 33/37 |
| R7J-s1-ag | 274 | 0.786 | 52 | 19 | 2 | 21 | 3 | 2 | 5 | 17 | 3 | 0 | 35/37 |
| NTV0-agH | 274 | 0.748 | 56 | 20 | 9 | 16 | 5 | 0 | 6 | 26 | 9 | 0 | 33/37 |
| R6DAG-s1-agH | 274 | 0.780 | 48 | 20 | 2 | 15 | 3 | 3 | 5 | 25 | 8 | 0 | 32/37 |
| R7J-s0-agH | 274 | 0.774 | 51 | 20 | 3 | 18 | 6 | 1 | 3 | 21 | 8 | 0 | 32/37 |
| R7J-s1-agH | 274 | 0.765 | 53 | 19 | 3 | 22 | 3 | 2 | 4 | 23 | 6 | 0 | 32/37 |
| R6DAG-s0-agH | 274 | 0.796 | 44 | 21 | 2 | 14 | 3 | 2 | 2 | 27 | 8 | 0 | 31/37 |

## R7J-s0-ag - R6DAG-s0-ag

- mean +0.008 [-0.017, +0.033]; zeros +0 (new 7, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 51 | +0 [-7, +7] | 7 | 7 |
| overshoot | 21 | 19 | +2 [+0, +5] | 2 | 0 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 16 | 19 | -3 [-8, +2] | 2 | 5 |
| AF-front | 5 | 4 | +1 [-3, +6] | 3 | 2 |
| side | 3 | 2 | +1 [-3, +5] | 3 | 2 |
| zero-other | 3 | 5 | -2 [-5, +0] | 0 | 2 |
| rear-hit | 17 | 22 | -5 [-10, +0] | 1 | 6 |
| slow | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 35 | -2 [-5, +0] | 0 | 2 |

## R7J-s1-ag - R6DAG-s1-ag

- mean -0.030 [-0.062, +0.001]; zeros +10 (new 15, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 42 | +10 [+2, +19] | 15 | 5 |
| overshoot | 19 | 18 | +1 [+0, +3] | 1 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 21 | 12 | +9 [+3, +16] | 10 | 1 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 2 | 3 | -1 [-6, +3] | 2 | 3 |
| zero-other | 5 | 4 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 17 | 20 | -3 [-8, +2] | 2 | 5 |
| slow | 3 | 6 | -3 [-8, +2] | 2 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R7J-s0-ag - NTV0-ag

- mean +0.037 [-0.007, +0.081]; zeros -5 (new 15, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 56 | -5 [-17, +6] | 15 | 20 |
| overshoot | 21 | 22 | -1 [-6, +4] | 3 | 4 |
| route-turn | 3 | 14 | -11 [-18, -5] | 0 | 11 |
| offroad | 16 | 11 | +5 [-2, +12] | 9 | 4 |
| AF-front | 5 | 3 | +2 [-3, +7] | 4 | 2 |
| side | 3 | 0 | +3 [+0, +7] | 3 | 0 |
| zero-other | 3 | 6 | -3 [-7, +1] | 1 | 4 |
| rear-hit | 17 | 25 | -8 [-16, +0] | 5 | 13 |
| slow | 6 | 11 | -5 [-10, +0] | 1 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 33 | +0 [-5, +5] | 3 | 3 |

## R7J-s1-ag - NTV0-ag

- mean +0.036 [-0.007, +0.080]; zeros -4 (new 16, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 56 | -4 [-16, +8] | 16 | 20 |
| overshoot | 19 | 22 | -3 [-8, +1] | 1 | 4 |
| route-turn | 2 | 14 | -12 [-19, -6] | 0 | 12 |
| offroad | 21 | 11 | +10 [+2, +18] | 13 | 3 |
| AF-front | 3 | 3 | +0 [-4, +4] | 2 | 2 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 5 | 6 | -1 [-6, +4] | 3 | 4 |
| rear-hit | 17 | 25 | -8 [-16, +0] | 5 | 13 |
| slow | 3 | 11 | -8 [-14, -3] | 0 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R7J-s0-agH - R6DAG-s1-agH

- mean -0.006 [-0.033, +0.021]; zeros +3 (new 8, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 48 | +3 [-4, +10] | 8 | 5 |
| overshoot | 20 | 20 | +0 [-3, +3] | 1 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 18 | 15 | +3 [-2, +8] | 5 | 2 |
| AF-front | 6 | 3 | +3 [+0, +7] | 3 | 0 |
| side | 1 | 3 | -2 [-5, +0] | 0 | 2 |
| zero-other | 3 | 5 | -2 [-5, +0] | 0 | 2 |
| rear-hit | 21 | 25 | -4 [-10, +2] | 3 | 7 |
| slow | 8 | 8 | +0 [-3, +3] | 1 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R7J-s1-agH - R6DAG-s1-agH

- mean -0.015 [-0.041, +0.009]; zeros +5 (new 8, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 48 | +5 [-1, +12] | 8 | 3 |
| overshoot | 19 | 20 | -1 [-3, +0] | 0 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 22 | 15 | +7 [+2, +12] | 7 | 0 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 2 | 3 | -1 [-5, +2] | 1 | 2 |
| zero-other | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 23 | 25 | -2 [-8, +4] | 4 | 6 |
| slow | 6 | 8 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R7J-s0-agH - NTV0-agH

- mean +0.026 [-0.015, +0.068]; zeros -5 (new 12, fixed 17)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 56 | -5 [-16, +6] | 12 | 17 |
| overshoot | 20 | 20 | +0 [-5, +5] | 3 | 3 |
| route-turn | 3 | 9 | -6 [-12, -1] | 1 | 7 |
| offroad | 18 | 16 | +2 [-5, +9] | 7 | 5 |
| AF-front | 6 | 5 | +1 [-5, +7] | 5 | 4 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 21 | 26 | -5 [-11, +1] | 3 | 8 |
| slow | 8 | 9 | -1 [-6, +4] | 3 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R7J-s1-agH - NTV0-agH

- mean +0.017 [-0.023, +0.058]; zeros -3 (new 12, fixed 15)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 56 | -3 [-13, +7] | 12 | 15 |
| overshoot | 19 | 20 | -1 [-5, +3] | 2 | 3 |
| route-turn | 3 | 9 | -6 [-12, -1] | 1 | 7 |
| offroad | 22 | 16 | +6 [-1, +14] | 10 | 4 |
| AF-front | 3 | 5 | -2 [-7, +3] | 2 | 4 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 23 | 26 | -3 [-11, +5] | 6 | 9 |
| slow | 6 | 9 | -3 [-8, +1] | 1 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R7J-s0-agH - R7J-s0-ag

- mean -0.012 [-0.040, +0.015]; zeros +0 (new 7, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 51 | +0 [-7, +7] | 7 | 7 |
| overshoot | 20 | 21 | -1 [-5, +2] | 1 | 2 |
| route-turn | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| offroad | 18 | 16 | +2 [-5, +9] | 7 | 5 |
| AF-front | 6 | 5 | +1 [-3, +6] | 3 | 2 |
| side | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| zero-other | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| rear-hit | 21 | 17 | +4 [-2, +10] | 7 | 3 |
| slow | 8 | 6 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R7J-s1-agH - R7J-s1-ag

- mean -0.021 [-0.052, +0.009]; zeros +1 (new 9, fixed 8)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 52 | +1 [-7, +9] | 9 | 8 |
| overshoot | 19 | 19 | +0 [+0, +0] | 0 | 0 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 22 | 21 | +1 [-5, +7] | 5 | 4 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 2 | 2 | +0 [-4, +4] | 2 | 2 |
| zero-other | 4 | 5 | -1 [-5, +3] | 2 | 3 |
| rear-hit | 23 | 17 | +6 [-0, +13] | 9 | 3 |
| slow | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 35 | -3 [-7, +0] | 0 | 3 |

## R7J-s1-ag - R7J-s0-ag

- mean -0.000 [-0.025, +0.025]; zeros +1 (new 7, fixed 6)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 51 | +1 [-6, +8] | 7 | 6 |
| overshoot | 19 | 21 | -2 [-5, +0] | 0 | 2 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 21 | 16 | +5 [-1, +11] | 7 | 2 |
| AF-front | 3 | 5 | -2 [-6, +2] | 1 | 3 |
| side | 2 | 3 | -1 [-5, +2] | 1 | 2 |
| zero-other | 5 | 3 | +2 [+0, +5] | 2 | 0 |
| rear-hit | 17 | 17 | +0 [-4, +4] | 2 | 2 |
| slow | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [+0, +5] | 2 | 0 |

## R7J-s0-agH - R6DAG-s0-agH

- mean -0.022 [-0.050, +0.005]; zeros +7 (new 10, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 44 | +7 [+0, +14] | 10 | 3 |
| overshoot | 20 | 21 | -1 [-6, +3] | 2 | 3 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 18 | 14 | +4 [+0, +9] | 5 | 1 |
| AF-front | 6 | 3 | +3 [-1, +7] | 4 | 1 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| rear-hit | 21 | 27 | -6 [-13, +1] | 3 | 9 |
| slow | 8 | 8 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 31 | +1 [+0, +3] | 1 | 0 |

## R6DAG-s0-agH - R6DAG-s0-ag

- mean +0.017 [-0.014, +0.049]; zeros -7 (new 7, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 44 | 51 | -7 [-16, +2] | 7 | 14 |
| overshoot | 21 | 19 | +2 [+0, +5] | 2 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 14 | 19 | -5 [-11, +1] | 2 | 7 |
| AF-front | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| side | 2 | 2 | +0 [-4, +4] | 2 | 2 |
| zero-other | 2 | 5 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 27 | 22 | +5 [-2, +12] | 9 | 4 |
| slow | 8 | 6 | +2 [-3, +7] | 4 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 35 | -4 [-8, -1] | 0 | 4 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## Registered adoption rule (plan section 6)

Both seeds: OFF 281 mean - same-seed R6DAG >= 0; - NTV0 95 % CI > 0; offroad + zero-other <= same-seed R6DAG's; launches >= 34 / 37; local PCS ruler - NTV0 >= same-seed R6DAG's (d434: -213 / -155); Harmonizer-ON 281 - R6DAG-s1 ON (0.780) >= 0 (paired).

| seed | OFF mean | - R6DAG same seed | - NTV0 | offroad + zero-other | launched | ruler - NTV0 | ON mean, - R6DAG-s1 ON | ON - same-seed R6DAG ON | failing |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| s0 | 0.787 | +0.008 [-0.017, +0.033] | +0.037 [-0.007, +0.081] | 19 vs 24 | 33/37 | -153 [-265, -35] vs -213 | 0.774, -0.006 [-0.033, +0.021] | -0.022 [-0.050, +0.005] | ntv0_ci launch harm_on |
| s1 | 0.786 | -0.030 [-0.062, +0.001] | +0.036 [-0.007, +0.080] | 26 vs 16 | 35/37 | -145 [-246, -6] vs -155 | 0.765, -0.015 [-0.041, +0.009] | -0.015 [-0.041, +0.009] | mean_vs_r6 ntv0_ci lat harm_on |

**Adopt: no.**

### New zeros / partials on the 182 scenes NTV0 scores 1.0

| read | new zeros | new partials | classes |
|:--|--:|--:|:--|
| R6DAG-s0-ag | 12 | 11 | rear-hit 6, offroad 6, slow 3, ok 2, zero-other 2, side 2, AF-front 1, overshoot 1 |
| R6DAG-s1-ag | 5 | 7 | offroad 3, rear-hit 3, ok 2, slow 2, side 1, zero-other 1 |
| R7J-s0-ag | 13 | 5 | AF-front 4, offroad 4, rear-hit 3, overshoot 2, side 2, ok 1, zero-other 1, slow 1 |
| R7J-s1-ag | 13 | 5 | offroad 7, rear-hit 4, zero-other 2, side 2, overshoot 1, ok 1, AF-front 1 |
| NTV0-agH | 10 | 2 | offroad 5, zero-other 2, AF-front 2, rear-hit 2, route-turn 1 |
| R6DAG-s1-agH | 14 | 12 | rear-hit 8, offroad 7, slow 3, overshoot 2, side 2, zero-other 2, ok 1, AF-front 1 |
| R7J-s0-agH | 15 | 9 | offroad 8, rear-hit 4, AF-front 3, slow 3, overshoot 2, ok 2, side 1, zero-other 1 |
| R7J-s1-agH | 15 | 9 | offroad 9, rear-hit 7, side 2, zero-other 2, overshoot 1, ok 1, slow 1, AF-front 1 |
| R6DAG-s0-agH | 13 | 12 | rear-hit 8, offroad 7, slow 3, overshoot 2, side 2, AF-front 1, zero-other 1, ok 1 |

### The six ruler-gap partial scenes (d436): score / class

| scene | NTV0-ag | R6DAG-s0-ag | R6DAG-s1-ag | R7J-s0-ag | R7J-s1-ag | NTV0-agH | R6DAG-s1-agH | R7J-s0-agH | R7J-s1-agH | R6DAG-s0-agH |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bbc53aa8 | 1.00 ok | 0.74 slow | 0.74 slow | 0.93 slow | 1.00 ok | 1.00 ok | 0.76 slow | 0.86 slow | 1.00 ok | 0.75 slow |
| d90fe123 | 1.00 ok | 0.65 rear-hit | 0.65 rear-hit | 0.71 rear-hit | 0.66 rear-hit | 1.00 ok | 0.68 rear-hit | 1.00 ok | 1.00 ok | 0.69 rear-hit |
| a1b166cc | 1.00 ok | 0.00 offroad | 0.90 slow | 0.13 rear-hit | 0.00 offroad | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |
| 94c1d722 | 0.96 slow | 0.00 offroad | 0.98 slow | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.97 slow | 0.00 offroad | 0.00 offroad | 0.94 slow |
| 2a0efada | 1.00 ok | 0.81 slow | 0.99 ok | 1.00 ok | 1.00 ok | 1.00 ok | 0.88 slow | 0.97 slow | 1.00 ok | 1.00 ok |
| 37d660b1 | 1.00 ok | 0.99 ok | 0.99 ok | 0.99 ok | 0.99 ok | 1.00 ok | 0.99 ok | 1.00 ok | 1.00 ok | 0.94 slow |

### The two stuck curve cut-ins: score / class

| scene | NTV0-ag | R6DAG-s0-ag | R6DAG-s1-ag | R7J-s0-ag | R7J-s1-ag | NTV0-agH | R6DAG-s1-agH | R7J-s0-agH | R7J-s1-agH | R6DAG-s0-agH |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 596af0c7 | 0.61 rear-hit | 0.00 offroad | 0.62 rear-hit | 0.00 offroad | 0.00 offroad | 0.61 rear-hit | 0.62 rear-hit | 0.62 rear-hit | 0.62 rear-hit | 0.62 rear-hit |
| ea468790 | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |

### Straight-road slow / rear-hit partials (score < 1, no zero) per read

| read | partial scenes by class |
|:--|:--|
| NTV0-ag | rear-hit 25, slow 11 |
| R6DAG-s0-ag | rear-hit 22, slow 6, ok 3 |
| R6DAG-s1-ag | rear-hit 20, slow 6, ok 2 |
| R7J-s0-ag | rear-hit 17, slow 6, ok 1 |
| R7J-s1-ag | rear-hit 17, slow 3, ok 1 |
| NTV0-agH | rear-hit 26, slow 9, ok 1 |
| R6DAG-s1-agH | rear-hit 25, slow 8, ok 2 |
| R7J-s0-agH | rear-hit 21, slow 8, ok 3 |
| R7J-s1-agH | rear-hit 23, slow 6, ok 1 |
| R6DAG-s0-agH | rear-hit 27, slow 8, ok 3 |
