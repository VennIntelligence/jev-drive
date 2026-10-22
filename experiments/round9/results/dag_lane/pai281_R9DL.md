# PAI 281, ACC + GAP (+ JEV_CMD=heading for R9DL): round9 DAG-LANE R9DL s0 / s1 vs R6DAG s0 / s1 / s2 and NTV0; Harmonizer OFF and ON

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 279 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R6DAG-s0-ag | 279 | 0.775 | 53 | 20 | 2 | 20 | 4 | 2 | 5 | 23 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 279 | 0.815 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |
| R6DAG-s2-ag | 279 | 0.791 | 51 | 20 | 3 | 18 | 4 | 2 | 4 | 18 | 3 | 0 | 35/37 |
| R9DL-s0-ag | 279 | 0.786 | 51 | 21 | 2 | 20 | 4 | 1 | 3 | 19 | 7 | 0 | 35/37 |
| R9DL-s1-ag | 279 | 0.800 | 47 | 20 | 2 | 15 | 5 | 2 | 3 | 21 | 6 | 0 | 35/37 |
| NTV0-agH | 279 | 0.745 | 58 | 21 | 9 | 17 | 5 | 0 | 6 | 27 | 9 | 0 | 33/37 |
| R6DAG-s0-agH | 279 | 0.792 | 46 | 22 | 2 | 15 | 3 | 2 | 2 | 27 | 8 | 0 | 31/37 |
| R6DAG-s1-agH | 279 | 0.780 | 49 | 21 | 2 | 15 | 3 | 3 | 5 | 26 | 8 | 0 | 32/37 |
| R6DAG-s2-agH | 279 | 0.780 | 48 | 21 | 2 | 15 | 5 | 2 | 3 | 26 | 9 | 0 | 32/37 |
| R9DL-s0-agH | 279 | 0.773 | 53 | 21 | 3 | 20 | 5 | 1 | 3 | 25 | 10 | 0 | 33/37 |
| R9DL-s1-agH | 279 | 0.765 | 54 | 21 | 2 | 20 | 3 | 4 | 4 | 27 | 8 | 0 | 32/37 |

## R9DL-s0-ag - R6DAG-s0-ag

- mean +0.011 [-0.016, +0.039]; zeros -2 (new 7, fixed 9)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 53 | -2 [-10, +6] | 7 | 9 |
| overshoot | 21 | 20 | +1 [-2, +5] | 2 | 1 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 20 | 20 | +0 [-5, +5] | 3 | 3 |
| AF-front | 4 | 4 | +0 [-3, +3] | 1 | 1 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 3 | 5 | -2 [-5, +0] | 0 | 2 |
| rear-hit | 19 | 23 | -4 [-8, -1] | 0 | 4 |
| slow | 7 | 6 | +1 [-2, +5] | 2 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R9DL-s0-agH - R6DAG-s0-agH

- mean -0.019 [-0.045, +0.004]; zeros +7 (new 9, fixed 2)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 46 | +7 [+1, +14] | 9 | 2 |
| overshoot | 21 | 22 | -1 [-5, +2] | 1 | 2 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 20 | 15 | +5 [+1, +10] | 5 | 0 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| rear-hit | 25 | 27 | -2 [-8, +3] | 3 | 5 |
| slow | 10 | 8 | +2 [+0, +5] | 2 | 0 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 31 | +2 [+0, +5] | 2 | 0 |

## R9DL-s0-ag - R6DAG-s2-ag

- mean -0.005 [-0.036, +0.026]; zeros +0 (new 10, fixed 10)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 51 | +0 [-9, +9] | 10 | 10 |
| overshoot | 21 | 20 | +1 [-2, +5] | 2 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 20 | 18 | +2 [-4, +8] | 6 | 4 |
| AF-front | 4 | 4 | +0 [-4, +4] | 2 | 2 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 3 | 4 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 19 | 18 | +1 [-4, +6] | 4 | 3 |
| slow | 7 | 3 | +4 [+1, +8] | 4 | 0 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R9DL-s0-agH - R6DAG-s2-agH

- mean -0.007 [-0.037, +0.022]; zeros +5 (new 11, fixed 6)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 48 | +5 [-3, +13] | 11 | 6 |
| overshoot | 21 | 21 | +0 [-4, +4] | 2 | 2 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 20 | 15 | +5 [-1, +11] | 7 | 2 |
| AF-front | 5 | 5 | +0 [-5, +5] | 3 | 3 |
| side | 1 | 2 | -1 [-4, +2] | 1 | 2 |
| zero-other | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| rear-hit | 25 | 26 | -1 [-6, +4] | 3 | 4 |
| slow | 10 | 9 | +1 [-3, +6] | 3 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 32 | +1 [+0, +3] | 1 | 0 |

## R9DL-s0-ag - NTV0-ag

- mean +0.043 [+0.003, +0.084]; zeros -8 (new 11, fixed 19)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 59 | -8 [-19, +3] | 11 | 19 |
| overshoot | 21 | 23 | -2 [-6, +2] | 1 | 3 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 20 | 12 | +8 [+1, +15] | 11 | 3 |
| AF-front | 4 | 3 | +1 [-2, +5] | 2 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 3 | 6 | -3 [-8, +1] | 1 | 4 |
| rear-hit | 19 | 26 | -7 [-15, +1] | 5 | 12 |
| slow | 7 | 11 | -4 [-10, +1] | 2 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R9DL-s0-agH - NTV0-agH

- mean +0.028 [-0.010, +0.066]; zeros -5 (new 11, fixed 16)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 58 | -5 [-15, +5] | 11 | 16 |
| overshoot | 21 | 21 | +0 [-4, +4] | 2 | 2 |
| route-turn | 3 | 9 | -6 [-11, -2] | 0 | 6 |
| offroad | 20 | 17 | +3 [-5, +11] | 10 | 7 |
| AF-front | 5 | 5 | +0 [-5, +6] | 4 | 4 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 3 | 6 | -3 [-8, +1] | 1 | 4 |
| rear-hit | 25 | 27 | -2 [-10, +6] | 7 | 9 |
| slow | 10 | 9 | +1 [-5, +7] | 5 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 33 | +0 [-3, +3] | 1 | 1 |

## R9DL-s0-agH - R9DL-s0-ag

- mean -0.013 [-0.041, +0.015]; zeros +2 (new 9, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 51 | +2 [-6, +10] | 9 | 7 |
| overshoot | 21 | 21 | +0 [-3, +3] | 1 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 20 | 20 | +0 [-6, +6] | 5 | 5 |
| AF-front | 5 | 4 | +1 [-2, +4] | 2 | 1 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| rear-hit | 25 | 19 | +6 [-1, +13] | 9 | 3 |
| slow | 10 | 7 | +3 [-2, +8] | 5 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 35 | -2 [-5, +0] | 0 | 2 |

## R9DL-s1-ag - R6DAG-s1-ag

- mean -0.015 [-0.041, +0.010]; zeros +4 (new 9, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 43 | +4 [-3, +11] | 9 | 5 |
| overshoot | 20 | 19 | +1 [+0, +3] | 1 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 15 | 12 | +3 [-3, +9] | 6 | 3 |
| AF-front | 5 | 3 | +2 [+0, +5] | 2 | 0 |
| side | 2 | 3 | -1 [-6, +3] | 2 | 3 |
| zero-other | 3 | 4 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 21 | 21 | +0 [-5, +5] | 3 | 3 |
| slow | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R9DL-s1-agH - R6DAG-s1-agH

- mean -0.015 [-0.041, +0.010]; zeros +5 (new 9, fixed 4)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 54 | 49 | +5 [-2, +12] | 9 | 4 |
| overshoot | 21 | 21 | +0 [-4, +4] | 2 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 20 | 15 | +5 [-1, +11] | 7 | 2 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 4 | 3 | +1 [-2, +5] | 2 | 1 |
| zero-other | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 27 | 26 | +1 [-3, +5] | 3 | 2 |
| slow | 8 | 8 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R9DL-s1-ag - R6DAG-s2-ag

- mean +0.009 [-0.025, +0.044]; zeros -4 (new 11, fixed 15)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 51 | -4 [-14, +6] | 11 | 15 |
| overshoot | 20 | 20 | +0 [-3, +3] | 1 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 15 | 18 | -3 [-11, +5] | 6 | 9 |
| AF-front | 5 | 4 | +1 [-3, +6] | 3 | 2 |
| side | 2 | 2 | +0 [-4, +4] | 2 | 2 |
| zero-other | 3 | 4 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 21 | 18 | +3 [-3, +9] | 6 | 3 |
| slow | 6 | 3 | +3 [+0, +7] | 3 | 0 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R9DL-s1-agH - R6DAG-s2-agH

- mean -0.015 [-0.040, +0.009]; zeros +6 (new 9, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 54 | 48 | +6 [+0, +13] | 9 | 3 |
| overshoot | 21 | 21 | +0 [-4, +4] | 2 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 20 | 15 | +5 [+1, +10] | 5 | 0 |
| AF-front | 3 | 5 | -2 [-6, +2] | 1 | 3 |
| side | 4 | 2 | +2 [-2, +6] | 3 | 1 |
| zero-other | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| rear-hit | 27 | 26 | +1 [-4, +6] | 4 | 3 |
| slow | 8 | 9 | -1 [-5, +3] | 2 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R9DL-s1-ag - NTV0-ag

- mean +0.057 [+0.015, +0.100]; zeros -12 (new 11, fixed 23)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 59 | -12 [-23, -1] | 11 | 23 |
| overshoot | 20 | 23 | -3 [-8, +1] | 1 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 15 | 12 | +3 [-4, +10] | 8 | 5 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 3 | 6 | -3 [-8, +1] | 1 | 4 |
| rear-hit | 21 | 26 | -5 [-13, +3] | 6 | 11 |
| slow | 6 | 11 | -5 [-11, +0] | 1 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R9DL-s1-agH - NTV0-agH

- mean +0.020 [-0.018, +0.060]; zeros -4 (new 12, fixed 16)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 54 | 58 | -4 [-15, +6] | 12 | 16 |
| overshoot | 21 | 21 | +0 [-4, +4] | 2 | 2 |
| route-turn | 2 | 9 | -7 [-13, -2] | 0 | 7 |
| offroad | 20 | 17 | +3 [-4, +11] | 9 | 6 |
| AF-front | 3 | 5 | -2 [-7, +3] | 2 | 4 |
| side | 4 | 0 | +4 [+1, +8] | 4 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 27 | 27 | +0 [-7, +7] | 6 | 6 |
| slow | 8 | 9 | -1 [-6, +4] | 3 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R9DL-s1-agH - R9DL-s1-ag

- mean -0.035 [-0.068, -0.000]; zeros +7 (new 15, fixed 8)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 54 | 47 | +7 [-2, +16] | 15 | 8 |
| overshoot | 21 | 20 | +1 [-2, +5] | 2 | 1 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 20 | 15 | +5 [-1, +12] | 8 | 3 |
| AF-front | 3 | 5 | -2 [-6, +2] | 1 | 3 |
| side | 4 | 2 | +2 [-3, +7] | 4 | 2 |
| zero-other | 4 | 3 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 27 | 21 | +6 [-1, +13] | 10 | 4 |
| slow | 8 | 6 | +2 [-3, +7] | 4 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 35 | -3 [-7, +0] | 0 | 3 |

## R9DL-s1-ag - R9DL-s0-ag

- mean +0.014 [-0.006, +0.036]; zeros -4 (new 3, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 51 | -4 [-10, +2] | 3 | 7 |
| overshoot | 20 | 21 | -1 [-5, +2] | 1 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 15 | 20 | -5 [-10, +0] | 1 | 6 |
| AF-front | 5 | 4 | +1 [+0, +3] | 1 | 0 |
| side | 2 | 1 | +1 [-2, +5] | 2 | 1 |
| zero-other | 3 | 3 | +0 [+0, +0] | 0 | 0 |
| rear-hit | 21 | 19 | +2 [-3, +7] | 4 | 2 |
| slow | 6 | 7 | -1 [-3, +0] | 0 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
