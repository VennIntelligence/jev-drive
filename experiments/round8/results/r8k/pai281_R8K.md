# PAI 281, ACC + GAP: round8 R8K s0 / s1 (R6DAG + keep-pace floor / F2 + Harmonizer twins) vs R6DAG, R7J and NTV0; Harmonizer OFF and ON

Scenes every read scored without a driver RPC error: 273 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 273 | 0.752 | 55 | 21 | 14 | 11 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| R6DAG-s0-ag | 273 | 0.782 | 50 | 18 | 2 | 19 | 4 | 2 | 5 | 22 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 273 | 0.819 | 41 | 17 | 2 | 12 | 3 | 3 | 4 | 20 | 6 | 0 | 35/37 |
| R7J-s0-ag | 273 | 0.789 | 50 | 20 | 3 | 16 | 5 | 3 | 3 | 17 | 6 | 0 | 33/37 |
| R7J-s1-ag | 273 | 0.789 | 51 | 18 | 2 | 21 | 3 | 2 | 5 | 17 | 3 | 0 | 35/37 |
| R8K-s0-ag | 273 | 0.799 | 47 | 17 | 3 | 16 | 6 | 1 | 4 | 17 | 4 | 0 | 35/37 |
| R8K-s1-ag | 273 | 0.808 | 44 | 17 | 3 | 15 | 5 | 0 | 4 | 20 | 3 | 0 | 35/37 |
| NTV0-agH | 273 | 0.751 | 55 | 19 | 9 | 16 | 5 | 0 | 6 | 26 | 9 | 0 | 33/37 |
| R6DAG-s0-agH | 273 | 0.799 | 43 | 20 | 2 | 14 | 3 | 2 | 2 | 27 | 8 | 0 | 31/37 |
| R6DAG-s1-agH | 273 | 0.783 | 47 | 19 | 2 | 15 | 3 | 3 | 5 | 25 | 8 | 0 | 32/37 |
| R7J-s0-agH | 273 | 0.777 | 50 | 19 | 3 | 18 | 6 | 1 | 3 | 21 | 8 | 0 | 32/37 |
| R7J-s1-agH | 273 | 0.768 | 52 | 18 | 3 | 22 | 3 | 2 | 4 | 23 | 6 | 0 | 32/37 |
| R8K-s0-agH | 273 | 0.776 | 51 | 19 | 2 | 19 | 5 | 2 | 4 | 20 | 6 | 0 | 31/37 |
| R8K-s1-agH | 273 | 0.770 | 52 | 18 | 2 | 20 | 7 | 2 | 3 | 22 | 4 | 0 | 32/37 |

## R8K-s0-ag - R6DAG-s0-ag

- mean +0.017 [-0.013, +0.049]; zeros -3 (new 8, fixed 11)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 50 | -3 [-12, +6] | 8 | 11 |
| overshoot | 17 | 18 | -1 [-3, +0] | 0 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 16 | 19 | -3 [-9, +3] | 3 | 6 |
| AF-front | 6 | 4 | +2 [-2, +6] | 3 | 1 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 17 | 22 | -5 [-10, +0] | 1 | 6 |
| slow | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R8K-s0-agH - R6DAG-s0-agH

- mean -0.023 [-0.051, +0.003]; zeros +8 (new 11, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 43 | +8 [+1, +16] | 11 | 3 |
| overshoot | 19 | 20 | -1 [-5, +2] | 1 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 19 | 14 | +5 [+1, +10] | 5 | 0 |
| AF-front | 5 | 3 | +2 [+0, +5] | 2 | 0 |
| side | 2 | 2 | +0 [-4, +4] | 2 | 2 |
| zero-other | 4 | 2 | +2 [+0, +5] | 2 | 0 |
| rear-hit | 20 | 27 | -7 [-13, -2] | 1 | 8 |
| slow | 6 | 8 | -2 [-5, +0] | 0 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 31 | +0 [+0, +0] | 0 | 0 |

## R8K-s0-ag - R7J-s0-ag

- mean +0.010 [-0.021, +0.040]; zeros -3 (new 9, fixed 12)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 50 | -3 [-12, +6] | 9 | 12 |
| overshoot | 17 | 20 | -3 [-7, +0] | 0 | 3 |
| route-turn | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| offroad | 16 | 16 | +0 [-7, +7] | 6 | 6 |
| AF-front | 6 | 5 | +1 [-3, +5] | 3 | 2 |
| side | 1 | 3 | -2 [-5, +0] | 0 | 2 |
| zero-other | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| rear-hit | 17 | 17 | +0 [-5, +5] | 3 | 3 |
| slow | 4 | 6 | -2 [-5, +0] | 0 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [+0, +5] | 2 | 0 |

## R8K-s0-agH - R7J-s0-agH

- mean -0.001 [-0.024, +0.023]; zeros +1 (new 6, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 50 | +1 [-6, +8] | 6 | 5 |
| overshoot | 19 | 19 | +0 [-3, +3] | 1 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 19 | 18 | +1 [-2, +5] | 2 | 1 |
| AF-front | 5 | 6 | -1 [-5, +3] | 2 | 3 |
| side | 2 | 1 | +1 [-2, +5] | 2 | 1 |
| zero-other | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| rear-hit | 20 | 21 | -1 [-6, +4] | 3 | 4 |
| slow | 6 | 8 | -2 [-5, +0] | 0 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 32 | -1 [-3, +0] | 0 | 1 |

## R8K-s0-ag - NTV0-ag

- mean +0.047 [+0.003, +0.090]; zeros -8 (new 13, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 55 | -8 [-19, +3] | 13 | 21 |
| overshoot | 17 | 21 | -4 [-8, -1] | 0 | 4 |
| route-turn | 3 | 14 | -11 [-18, -4] | 1 | 12 |
| offroad | 16 | 11 | +5 [-2, +12] | 9 | 4 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 17 | 25 | -8 [-16, +0] | 5 | 13 |
| slow | 4 | 11 | -7 [-13, -2] | 1 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R8K-s0-agH - NTV0-agH

- mean +0.026 [-0.014, +0.067]; zeros -4 (new 12, fixed 16)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 55 | -4 [-14, +6] | 12 | 16 |
| overshoot | 19 | 19 | +0 [-4, +4] | 2 | 2 |
| route-turn | 2 | 9 | -7 [-12, -2] | 0 | 7 |
| offroad | 19 | 16 | +3 [-3, +10] | 7 | 4 |
| AF-front | 5 | 5 | +0 [-5, +5] | 3 | 3 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 20 | 26 | -6 [-13, +1] | 3 | 9 |
| slow | 6 | 9 | -3 [-8, +2] | 2 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 33 | -2 [-6, +2] | 1 | 3 |

## R8K-s0-agH - R8K-s0-ag

- mean -0.023 [-0.059, +0.013]; zeros +4 (new 14, fixed 10)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 51 | 47 | +4 [-6, +14] | 14 | 10 |
| overshoot | 19 | 17 | +2 [+0, +5] | 2 | 0 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 19 | 16 | +3 [-3, +9] | 6 | 3 |
| AF-front | 5 | 6 | -1 [-6, +4] | 3 | 4 |
| side | 2 | 1 | +1 [-2, +5] | 2 | 1 |
| zero-other | 4 | 4 | +0 [-4, +4] | 2 | 2 |
| rear-hit | 20 | 17 | +3 [-3, +9] | 6 | 3 |
| slow | 6 | 4 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 35 | -4 [-8, -1] | 0 | 4 |

## R6DAG-s0-agH - R6DAG-s0-ag

- mean +0.017 [-0.014, +0.049]; zeros -7 (new 7, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 43 | 50 | -7 [-16, +2] | 7 | 14 |
| overshoot | 20 | 18 | +2 [+0, +5] | 2 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 14 | 19 | -5 [-11, +1] | 2 | 7 |
| AF-front | 3 | 4 | -1 [-4, +2] | 1 | 2 |
| side | 2 | 2 | +0 [-4, +4] | 2 | 2 |
| zero-other | 2 | 5 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 27 | 22 | +5 [-2, +12] | 9 | 4 |
| slow | 8 | 6 | +2 [-3, +7] | 4 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 31 | 35 | -4 [-8, -1] | 0 | 4 |

## R8K-s1-ag - R6DAG-s1-ag

- mean -0.011 [-0.037, +0.014]; zeros +3 (new 8, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 44 | 41 | +3 [-4, +10] | 8 | 5 |
| overshoot | 17 | 17 | +0 [+0, +0] | 0 | 0 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 15 | 12 | +3 [-2, +8] | 5 | 2 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 0 | 3 | -3 [-7, +0] | 0 | 3 |
| zero-other | 4 | 4 | +0 [+0, +0] | 0 | 0 |
| rear-hit | 20 | 20 | +0 [-5, +5] | 3 | 3 |
| slow | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R8K-s1-agH - R6DAG-s1-agH

- mean -0.013 [-0.042, +0.016]; zeros +5 (new 10, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 47 | +5 [-2, +13] | 10 | 5 |
| overshoot | 18 | 19 | -1 [-4, +2] | 1 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 20 | 15 | +5 [+0, +10] | 6 | 1 |
| AF-front | 7 | 3 | +4 [+1, +8] | 4 | 0 |
| side | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| zero-other | 3 | 5 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 22 | 25 | -3 [-9, +3] | 3 | 6 |
| slow | 4 | 8 | -4 [-8, -1] | 0 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R8K-s1-ag - R7J-s1-ag

- mean +0.019 [-0.012, +0.052]; zeros -7 (new 7, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 44 | 51 | -7 [-16, +2] | 7 | 14 |
| overshoot | 17 | 18 | -1 [-3, +0] | 0 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 15 | 21 | -6 [-13, +0] | 3 | 9 |
| AF-front | 5 | 3 | +2 [+0, +5] | 2 | 0 |
| side | 0 | 2 | -2 [-5, +0] | 0 | 2 |
| zero-other | 4 | 5 | -1 [-4, +2] | 1 | 2 |
| rear-hit | 20 | 17 | +3 [-1, +8] | 4 | 1 |
| slow | 3 | 3 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## R8K-s1-agH - R7J-s1-agH

- mean +0.003 [-0.021, +0.026]; zeros +0 (new 5, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 52 | +0 [-6, +6] | 5 | 5 |
| overshoot | 18 | 18 | +0 [-3, +3] | 1 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 20 | 22 | -2 [-7, +3] | 2 | 4 |
| AF-front | 7 | 3 | +4 [+0, +9] | 5 | 1 |
| side | 2 | 2 | +0 [-3, +3] | 1 | 1 |
| zero-other | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| rear-hit | 22 | 23 | -1 [-5, +2] | 1 | 2 |
| slow | 4 | 6 | -2 [-5, +0] | 0 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 32 | +0 [+0, +0] | 0 | 0 |

## R8K-s1-ag - NTV0-ag

- mean +0.056 [+0.012, +0.099]; zeros -11 (new 12, fixed 23)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 44 | 55 | -11 [-22, +0] | 12 | 23 |
| overshoot | 17 | 21 | -4 [-8, -1] | 0 | 4 |
| route-turn | 3 | 14 | -11 [-18, -4] | 1 | 12 |
| offroad | 15 | 11 | +4 [-2, +11] | 8 | 4 |
| AF-front | 5 | 3 | +2 [-3, +7] | 4 | 2 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 20 | 25 | -5 [-14, +3] | 7 | 12 |
| slow | 3 | 11 | -8 [-14, -2] | 1 | 9 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R8K-s1-agH - NTV0-agH

- mean +0.020 [-0.021, +0.061]; zeros -3 (new 13, fixed 16)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 55 | -3 [-14, +8] | 13 | 16 |
| overshoot | 18 | 19 | -1 [-5, +2] | 1 | 2 |
| route-turn | 2 | 9 | -7 [-12, -2] | 0 | 7 |
| offroad | 20 | 16 | +4 [-3, +11] | 8 | 4 |
| AF-front | 7 | 5 | +2 [-3, +8] | 5 | 3 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 3 | 6 | -3 [-8, +1] | 1 | 4 |
| rear-hit | 22 | 26 | -4 [-11, +3] | 5 | 9 |
| slow | 4 | 9 | -5 [-10, +0] | 1 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 33 | -1 [-5, +2] | 1 | 2 |

## R8K-s1-agH - R8K-s1-ag

- mean -0.038 [-0.069, -0.008]; zeros +8 (new 12, fixed 4)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 52 | 44 | +8 [+0, +16] | 12 | 4 |
| overshoot | 18 | 17 | +1 [+0, +3] | 1 | 0 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 20 | 15 | +5 [+0, +11] | 7 | 2 |
| AF-front | 7 | 5 | +2 [+0, +5] | 2 | 0 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| rear-hit | 22 | 20 | +2 [-3, +8] | 5 | 3 |
| slow | 4 | 3 | +1 [-2, +4] | 2 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 35 | -3 [-7, +0] | 0 | 3 |

## R6DAG-s1-agH - R6DAG-s1-ag

- mean -0.036 [-0.069, -0.003]; zeros +6 (new 12, fixed 6)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 41 | +6 [-2, +14] | 12 | 6 |
| overshoot | 19 | 17 | +2 [+0, +5] | 2 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 15 | 12 | +3 [-3, +9] | 6 | 3 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| zero-other | 5 | 4 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 25 | 20 | +5 [-1, +11] | 7 | 2 |
| slow | 8 | 6 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 32 | 35 | -3 [-7, +0] | 0 | 3 |

## R8K-s1-ag - R8K-s0-ag

- mean +0.009 [-0.015, +0.034]; zeros -3 (new 5, fixed 8)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 44 | 47 | -3 [-10, +4] | 5 | 8 |
| overshoot | 17 | 17 | +0 [+0, +0] | 0 | 0 |
| route-turn | 3 | 3 | +0 [+0, +0] | 0 | 0 |
| offroad | 15 | 16 | -1 [-7, +5] | 4 | 5 |
| AF-front | 5 | 6 | -1 [-5, +2] | 1 | 2 |
| side | 0 | 1 | -1 [-3, +0] | 0 | 1 |
| zero-other | 4 | 4 | +0 [+0, +0] | 0 | 0 |
| rear-hit | 20 | 17 | +3 [-1, +8] | 4 | 1 |
| slow | 3 | 4 | -1 [-3, +0] | 0 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## Registered adoption rule (plan section 5)

Both seeds: OFF 281 - same-seed R6DAG (paired point) >= 0; offroad + zero-other <= same-seed R6DAG + 2; launches >= 34 / 37; ruler - NTV0 >= same-seed R6DAG's in the same report; (OFF + ON) / 2 - same-seed R6DAG's (OFF + ON) / 2 (paired points) >= 0.

| seed | OFF mean | OFF - R6DAG same seed | offroad + zero-other | launched | ruler - NTV0 | ON mean, - R6DAG same seed ON | (OFF + ON) / 2 - R6DAG | - R7J same seed OFF / ON | failing |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| s0 | 0.799 | +0.017 [-0.013, +0.049] | 20 vs 24 + 2 | 35/37 | -181 [-275, -24] vs -213 | 0.776, -0.023 [-0.051, +0.003] | -0.0026 | +0.010 [-0.021, +0.040] / -0.001 [-0.024, +0.023] | offon_avg |
| s1 | 0.808 | -0.011 [-0.037, +0.014] | 19 vs 16 + 2 | 35/37 | -156 [-250, +32] vs -155 | 0.770, -0.013 [-0.042, +0.016] | -0.0117 | +0.019 [-0.012, +0.052] / +0.003 [-0.021, +0.026] | mean_vs_r6 lat ruler offon_avg |

**Adopt: no.**

## Side by side (common scenes; ruler = local PCS - NTV0-s0+AG, same bootstrap draws)

| | R8K-s0 | R7J-s0 | R6DAG-s0 | R8K-s1 | R7J-s1 | R6DAG-s1 |
|:--|--:|--:|--:|--:|--:|--:|
| OFF mean | 0.799 | 0.789 | 0.782 | 0.808 | 0.789 | 0.819 |
| ON mean | 0.776 | 0.777 | 0.799 | 0.770 | 0.768 | 0.783 |
| (OFF + ON) / 2 | 0.788 | 0.783 | 0.790 | 0.789 | 0.778 | 0.801 |
| ruler - NTV0 | -181 [-275, -24] | -153 [-265, -35] | -213 [-305, -57] | -156 [-250, +32] | -145 [-246, -6] | -155 [-260, +36] |
| offroad + zero-other OFF / ON | 20 / 23 | 19 / 21 | 24 / 16 | 19 / 23 | 26 / 26 | 16 / 20 |
| rear-hit / slow partials OFF | 17 / 4 | 17 / 6 | 22 / 6 | 20 / 3 | 17 / 3 | 20 / 6 |
| rear-hit / slow partials ON | 20 / 6 | 21 / 8 | 27 / 8 | 22 / 4 | 23 / 6 | 25 / 8 |
| launched OFF / ON | 35 / 31 | 33 / 32 | 35 / 31 | 35 / 32 | 35 / 32 | 35 / 32 |
| zeros OFF / ON | 47 / 51 | 50 / 50 | 50 / 43 | 44 / 52 | 51 / 52 | 41 / 47 |

### New zeros / partials on the 182 scenes NTV0 (OFF) scores 1.0

| read | new zeros | new partials | classes |
|:--|--:|--:|:--|
| R6DAG-s0-ag | 12 | 11 | rear-hit 6, offroad 6, slow 3, ok 2, zero-other 2, side 2, AF-front 1, overshoot 1 |
| R6DAG-s1-ag | 5 | 7 | offroad 3, rear-hit 3, ok 2, slow 2, side 1, zero-other 1 |
| R7J-s0-ag | 13 | 5 | AF-front 4, offroad 4, rear-hit 3, overshoot 2, side 2, ok 1, zero-other 1, slow 1 |
| R7J-s1-ag | 13 | 5 | offroad 7, rear-hit 4, zero-other 2, side 2, overshoot 1, ok 1, AF-front 1 |
| R8K-s0-ag | 11 | 6 | offroad 5, rear-hit 4, AF-front 3, ok 1, zero-other 1, side 1, slow 1, route-turn 1 |
| R8K-s1-ag | 9 | 7 | rear-hit 6, offroad 5, AF-front 2, zero-other 1, slow 1, route-turn 1 |
| NTV0-agH | 10 | 2 | offroad 5, zero-other 2, AF-front 2, rear-hit 2, route-turn 1 |
| R6DAG-s0-agH | 13 | 12 | rear-hit 8, offroad 7, slow 3, overshoot 2, side 2, AF-front 1, zero-other 1, ok 1 |
| R6DAG-s1-agH | 14 | 12 | rear-hit 8, offroad 7, slow 3, overshoot 2, side 2, zero-other 2, ok 1, AF-front 1 |
| R7J-s0-agH | 15 | 9 | offroad 8, rear-hit 4, AF-front 3, slow 3, overshoot 2, ok 2, side 1, zero-other 1 |
| R7J-s1-agH | 15 | 9 | offroad 9, rear-hit 7, side 2, zero-other 2, overshoot 1, ok 1, slow 1, AF-front 1 |
| R8K-s0-agH | 16 | 7 | offroad 9, rear-hit 4, AF-front 3, zero-other 2, slow 2, overshoot 1, ok 1, side 1 |
| R8K-s1-agH | 16 | 8 | offroad 10, rear-hit 6, AF-front 2, side 2, zero-other 2, slow 1, ok 1 |

### The six ruler-gap partial scenes (d436): score / class

| scene | NTV0-ag | R6DAG-s0-ag | R6DAG-s1-ag | R7J-s0-ag | R7J-s1-ag | R8K-s0-ag | R8K-s1-ag | NTV0-agH | R6DAG-s0-agH | R6DAG-s1-agH | R7J-s0-agH | R7J-s1-agH | R8K-s0-agH | R8K-s1-agH |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bbc53aa8 | 1.00 ok | 0.74 slow | 0.74 slow | 0.93 slow | 1.00 ok | 0.75 slow | 0.76 slow | 1.00 ok | 0.75 slow | 0.76 slow | 0.86 slow | 1.00 ok | 0.78 slow | 0.00 offroad |
| d90fe123 | 1.00 ok | 0.65 rear-hit | 0.65 rear-hit | 0.71 rear-hit | 0.66 rear-hit | 0.69 rear-hit | 0.71 rear-hit | 1.00 ok | 0.69 rear-hit | 0.68 rear-hit | 1.00 ok | 1.00 ok | 1.00 ok | 0.65 rear-hit |
| a1b166cc | 1.00 ok | 0.00 offroad | 0.90 slow | 0.13 rear-hit | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |
| 94c1d722 | 0.96 slow | 0.00 offroad | 0.98 slow | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.94 slow | 0.97 slow | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |
| 2a0efada | 1.00 ok | 0.81 slow | 0.99 ok | 1.00 ok | 1.00 ok | 1.00 ok | 1.00 ok | 1.00 ok | 1.00 ok | 0.88 slow | 0.97 slow | 1.00 ok | 0.00 zero-other | 0.00 zero-other |
| 37d660b1 | 1.00 ok | 0.99 ok | 0.99 ok | 0.99 ok | 0.99 ok | 1.00 ok | 1.00 ok | 1.00 ok | 0.94 slow | 0.99 ok | 1.00 ok | 1.00 ok | 1.00 ok | 1.00 ok |

### The two stuck curve cut-ins: score / class

| scene | NTV0-ag | R6DAG-s0-ag | R6DAG-s1-ag | R7J-s0-ag | R7J-s1-ag | R8K-s0-ag | R8K-s1-ag | NTV0-agH | R6DAG-s0-agH | R6DAG-s1-agH | R7J-s0-agH | R7J-s1-agH | R8K-s0-agH | R8K-s1-agH |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 596af0c7 | 0.61 rear-hit | 0.00 offroad | 0.62 rear-hit | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.62 rear-hit | 0.61 rear-hit | 0.62 rear-hit | 0.62 rear-hit | 0.62 rear-hit | 0.62 rear-hit | 0.99 ok | 0.62 rear-hit |
| ea468790 | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |
