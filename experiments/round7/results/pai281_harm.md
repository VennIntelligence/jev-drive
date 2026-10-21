# PAI 281, ACC + GAP, Harmonizer ON vs stored OFF: NTV0-L18-s0 and R6DAG-L18-s1

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-off | 279 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R6DAG-off | 279 | 0.815 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |
| NTV0-on | 279 | 0.745 | 58 | 21 | 9 | 17 | 5 | 0 | 6 | 27 | 9 | 0 | 33/37 |
| R6DAG-on | 279 | 0.780 | 49 | 21 | 2 | 15 | 3 | 3 | 5 | 26 | 8 | 0 | 32/37 |

## NTV0-on - NTV0-off

- mean +0.002 [-0.031, +0.035]; zeros -1 (new 11, fixed 12)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 58 | 59 | -1 [-10, +8] | 11 | 12 |
| overshoot | 21 | 23 | -2 [-6, +2] | 1 | 3 |
| route-turn | 9 | 15 | -6 [-12, -1] | 1 | 7 |
| offroad | 17 | 12 | +5 [-1, +11] | 7 | 2 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 6 | 6 | +0 [-5, +5] | 3 | 3 |
| rear-hit | 27 | 26 | +1 [-3, +6] | 3 | 2 |
| slow | 9 | 11 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 33 | +0 [-3, +3] | 1 | 1 |

## R6DAG-on - R6DAG-off

- mean -0.035 [-0.066, -0.005]; zeros +6 (new 12, fixed 6)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 49 | 43 | +6 [-2, +14] | 12 | 6 |
| overshoot | 21 | 19 | +2 [+0, +5] | 2 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 15 | 12 | +3 [-3, +9] | 6 | 3 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| zero-other | 5 | 4 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 26 | 21 | +5 [-1, +11] | 7 | 2 |
| slow | 8 | 6 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 35 | -3 [-7, +0] | 0 | 3 |

## R6DAG-on - NTV0-on

- mean +0.035 [-0.006, +0.078]; zeros -9 (new 12, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 49 | 58 | -9 [-20, +2] | 12 | 21 |
| overshoot | 21 | 21 | +0 [-5, +5] | 3 | 3 |
| route-turn | 2 | 9 | -7 [-13, -2] | 0 | 7 |
| offroad | 15 | 17 | -2 [-9, +5] | 6 | 8 |
| AF-front | 3 | 5 | -2 [-7, +3] | 2 | 4 |
| side | 3 | 0 | +3 [+0, +7] | 3 | 0 |
| zero-other | 5 | 6 | -1 [-5, +3] | 2 | 3 |
| rear-hit | 26 | 27 | -1 [-8, +6] | 6 | 7 |
| slow | 8 | 9 | -1 [-7, +5] | 4 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R6DAG-off - NTV0-off

- mean +0.072 [+0.035, +0.111]; zeros -16 (new 6, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 43 | 59 | -16 [-27, -6] | 6 | 22 |
| overshoot | 19 | 23 | -4 [-8, -1] | 0 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 12 | 12 | +0 [-6, +6] | 5 | 5 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 3 | 0 | +3 [+0, +7] | 3 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 21 | 26 | -5 [-13, +2] | 5 | 10 |
| slow | 6 | 11 | -5 [-11, +1] | 2 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
