# interim

Scenes every read scored without a driver RPC error: 273 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 273 | 0.752 | 55 | 21 | 14 | 11 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| NTV0-agH | 273 | 0.751 | 55 | 19 | 9 | 16 | 5 | 0 | 6 | 26 | 9 | 0 | 33/37 |
| R6DAG-s0-ag | 273 | 0.782 | 50 | 18 | 2 | 19 | 4 | 2 | 5 | 22 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 273 | 0.819 | 41 | 17 | 2 | 12 | 3 | 3 | 4 | 20 | 6 | 0 | 35/37 |
| R7J-s0-ag | 273 | 0.789 | 50 | 20 | 3 | 16 | 5 | 3 | 3 | 17 | 6 | 0 | 33/37 |
| R7J-s1-ag | 273 | 0.789 | 51 | 18 | 2 | 21 | 3 | 2 | 5 | 17 | 3 | 0 | 35/37 |
| R6DAG-s0-agH | 273 | 0.799 | 43 | 20 | 2 | 14 | 3 | 2 | 2 | 27 | 8 | 0 | 31/37 |
| R6DAG-s1-agH | 273 | 0.783 | 47 | 19 | 2 | 15 | 3 | 3 | 5 | 25 | 8 | 0 | 32/37 |
| R7J-s0-agH | 273 | 0.777 | 50 | 19 | 3 | 18 | 6 | 1 | 3 | 21 | 8 | 0 | 32/37 |
| R7J-s1-agH | 273 | 0.768 | 52 | 18 | 3 | 22 | 3 | 2 | 4 | 23 | 6 | 0 | 32/37 |
| R8K-s0-ag | 273 | 0.799 | 47 | 17 | 3 | 16 | 6 | 1 | 4 | 17 | 4 | 0 | 35/37 |
| R8K-s1-ag | 273 | 0.808 | 44 | 17 | 3 | 15 | 5 | 0 | 4 | 20 | 3 | 0 | 35/37 |
| R8K-s0-agH | 273 | 0.776 | 51 | 19 | 2 | 19 | 5 | 2 | 4 | 20 | 6 | 0 | 31/37 |

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

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
