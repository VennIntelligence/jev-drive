# PAI 281, Harmonizer OFF: py123d_garage v0.2.0 (ltf-base-joint64-68M, conditional, posttrain seed 0, as shipped) vs our ACC + GAP reads

Scenes every read scored without a driver RPC error: 277 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GARAGE020 | 277 | 0.750 | 47 | 6 | 1 | 18 | 20 | 1 | 1 | 30 | 23 | 0 | 29/37 |
| NTV0-ag | 277 | 0.745 | 58 | 23 | 15 | 11 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R6DAG-s0-ag | 277 | 0.773 | 53 | 20 | 2 | 20 | 4 | 2 | 5 | 23 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 277 | 0.814 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |
| R6DAG-s2-ag | 277 | 0.789 | 51 | 20 | 3 | 18 | 4 | 2 | 4 | 18 | 3 | 0 | 35/37 |
| R8K-s0-ag | 277 | 0.791 | 50 | 19 | 3 | 17 | 6 | 1 | 4 | 18 | 4 | 0 | 35/37 |
| R8K-s1-ag | 277 | 0.800 | 47 | 19 | 3 | 16 | 5 | 0 | 4 | 21 | 3 | 0 | 35/37 |
| R8K-s2-ag | 277 | 0.803 | 47 | 20 | 2 | 14 | 5 | 2 | 4 | 18 | 4 | 0 | 35/37 |

## GARAGE020 - NTV0-ag

- mean +0.005 [-0.055, +0.064]; zeros -11 (new 27, fixed 38)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 58 | -11 [-26, +5] | 27 | 38 |
| overshoot | 6 | 23 | -17 [-26, -9] | 1 | 18 |
| route-turn | 1 | 15 | -14 [-21, -7] | 0 | 14 |
| offroad | 18 | 11 | +7 [-2, +17] | 15 | 8 |
| AF-front | 20 | 3 | +17 [+8, +26] | 20 | 3 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 1 | 6 | -5 [-10, -1] | 0 | 5 |
| rear-hit | 30 | 26 | +4 [-9, +17] | 23 | 19 |
| slow | 23 | 11 | +12 [+1, +23] | 21 | 9 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 33 | -4 [-11, +3] | 4 | 8 |

## GARAGE020 - R6DAG-s0-ag

- mean -0.023 [-0.082, +0.033]; zeros -6 (new 28, fixed 34)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 53 | -6 [-21, +10] | 28 | 34 |
| overshoot | 6 | 20 | -14 [-22, -7] | 1 | 15 |
| route-turn | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| offroad | 18 | 20 | -2 [-12, +8] | 12 | 14 |
| AF-front | 20 | 4 | +16 [+8, +25] | 18 | 2 |
| side | 1 | 2 | -1 [-4, +2] | 1 | 2 |
| zero-other | 1 | 5 | -4 [-8, -1] | 0 | 4 |
| rear-hit | 30 | 23 | +7 [-4, +19] | 21 | 14 |
| slow | 23 | 6 | +17 [+8, +27] | 21 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

## GARAGE020 - R6DAG-s1-ag

- mean -0.064 [-0.121, -0.007]; zeros +4 (new 31, fixed 27)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 43 | +4 [-11, +19] | 31 | 27 |
| overshoot | 6 | 19 | -13 [-21, -6] | 1 | 14 |
| route-turn | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| offroad | 18 | 12 | +6 [-3, +15] | 14 | 8 |
| AF-front | 20 | 3 | +17 [+8, +26] | 20 | 3 |
| side | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| zero-other | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 30 | 21 | +9 [-2, +21] | 22 | 13 |
| slow | 23 | 6 | +17 [+7, +27] | 22 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

## GARAGE020 - R6DAG-s2-ag

- mean -0.040 [-0.097, +0.018]; zeros -4 (new 28, fixed 32)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 51 | -4 [-19, +12] | 28 | 32 |
| overshoot | 6 | 20 | -14 [-22, -7] | 0 | 14 |
| route-turn | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| offroad | 18 | 18 | +0 [-10, +10] | 12 | 12 |
| AF-front | 20 | 4 | +16 [+7, +25] | 19 | 3 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 30 | 18 | +12 [+1, +24] | 23 | 11 |
| slow | 23 | 3 | +20 [+11, +30] | 23 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

## GARAGE020 - R8K-s0-ag

- mean -0.041 [-0.100, +0.017]; zeros -3 (new 30, fixed 33)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 50 | -3 [-18, +13] | 30 | 33 |
| overshoot | 6 | 19 | -13 [-21, -6] | 1 | 14 |
| route-turn | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| offroad | 18 | 17 | +1 [-9, +11] | 14 | 13 |
| AF-front | 20 | 6 | +14 [+6, +23] | 17 | 3 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 30 | 18 | +12 [+1, +24] | 23 | 11 |
| slow | 23 | 4 | +19 [+9, +29] | 23 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

## GARAGE020 - R8K-s1-ag

- mean -0.050 [-0.109, +0.009]; zeros +0 (new 31, fixed 31)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 47 | +0 [-15, +16] | 31 | 31 |
| overshoot | 6 | 19 | -13 [-21, -6] | 1 | 14 |
| route-turn | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| offroad | 18 | 16 | +2 [-8, +12] | 14 | 12 |
| AF-front | 20 | 5 | +15 [+7, +23] | 17 | 2 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 30 | 21 | +9 [-3, +21] | 23 | 14 |
| slow | 23 | 3 | +20 [+11, +30] | 23 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

## GARAGE020 - R8K-s2-ag

- mean -0.053 [-0.113, +0.007]; zeros +0 (new 33, fixed 33)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 47 | 47 | +0 [-16, +16] | 33 | 33 |
| overshoot | 6 | 20 | -14 [-22, -7] | 0 | 14 |
| route-turn | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| offroad | 18 | 14 | +4 [-6, +14] | 15 | 11 |
| AF-front | 20 | 5 | +15 [+6, +24] | 18 | 3 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| rear-hit | 30 | 18 | +12 [+1, +24] | 23 | 11 |
| slow | 23 | 4 | +19 [+9, +29] | 23 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 35 | -6 [-12, +0] | 2 | 8 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
