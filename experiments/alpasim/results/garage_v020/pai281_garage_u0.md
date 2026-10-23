# PAI 281, Harmonizer OFF: py123d_garage v0.2.0 (ltf-base-joint64-68M, unconditional, midtrain seed 0, as shipped) vs our ACC + GAP reads

Scenes every read scored without a driver RPC error: 280 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GARAGE020 | 280 | 0.645 | 79 | 7 | 21 | 14 | 19 | 1 | 17 | 28 | 17 | 0 | 27/37 |
| NTV0-ag | 280 | 0.744 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R6DAG-s0-ag | 280 | 0.776 | 53 | 20 | 2 | 20 | 4 | 2 | 5 | 23 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 280 | 0.816 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |
| R6DAG-s2-ag | 280 | 0.792 | 51 | 20 | 3 | 18 | 4 | 2 | 4 | 18 | 3 | 0 | 35/37 |
| R8K-s0-ag | 280 | 0.793 | 50 | 19 | 3 | 17 | 6 | 1 | 4 | 18 | 4 | 0 | 35/37 |
| R8K-s1-ag | 280 | 0.802 | 47 | 19 | 3 | 16 | 5 | 0 | 4 | 21 | 3 | 0 | 35/37 |
| R8K-s2-ag | 280 | 0.806 | 47 | 20 | 2 | 14 | 5 | 2 | 4 | 18 | 4 | 0 | 35/37 |

## GARAGE020 - NTV0-ag

- mean -0.099 [-0.162, -0.037]; zeros +20 (new 51, fixed 31)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 59 | +20 [+3, +37] | 51 | 31 |
| overshoot | 7 | 23 | -16 [-25, -8] | 2 | 18 |
| route-turn | 21 | 15 | +6 [-3, +16] | 15 | 9 |
| offroad | 14 | 12 | +2 [-7, +11] | 12 | 10 |
| AF-front | 19 | 3 | +16 [+7, +25] | 19 | 3 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 17 | 6 | +11 [+3, +19] | 14 | 3 |
| rear-hit | 28 | 26 | +2 [-11, +15] | 23 | 21 |
| slow | 17 | 11 | +6 [-4, +16] | 16 | 10 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 33 | -6 [-13, +1] | 4 | 10 |

## GARAGE020 - R6DAG-s0-ag

- mean -0.131 [-0.192, -0.070]; zeros +26 (new 54, fixed 28)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 53 | +26 [+9, +43] | 54 | 28 |
| overshoot | 7 | 20 | -13 [-21, -6] | 1 | 14 |
| route-turn | 21 | 2 | +19 [+11, +28] | 20 | 1 |
| offroad | 14 | 20 | -6 [-18, +6] | 14 | 20 |
| AF-front | 19 | 4 | +15 [+6, +25] | 19 | 4 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 17 | 5 | +12 [+4, +20] | 15 | 3 |
| rear-hit | 28 | 23 | +5 [-8, +18] | 23 | 18 |
| slow | 17 | 6 | +11 [+2, +20] | 16 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

## GARAGE020 - R6DAG-s1-ag

- mean -0.171 [-0.232, -0.111]; zeros +36 (new 57, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 43 | +36 [+20, +53] | 57 | 21 |
| overshoot | 7 | 19 | -12 [-20, -5] | 2 | 14 |
| route-turn | 21 | 2 | +19 [+11, +28] | 20 | 1 |
| offroad | 14 | 12 | +2 [-8, +12] | 13 | 11 |
| AF-front | 19 | 3 | +16 [+7, +26] | 19 | 3 |
| side | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| zero-other | 17 | 4 | +13 [+5, +21] | 15 | 2 |
| rear-hit | 28 | 21 | +7 [-5, +19] | 23 | 16 |
| slow | 17 | 6 | +11 [+2, +20] | 17 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

## GARAGE020 - R6DAG-s2-ag

- mean -0.146 [-0.210, -0.084]; zeros +28 (new 57, fixed 29)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 51 | +28 [+10, +46] | 57 | 29 |
| overshoot | 7 | 20 | -13 [-21, -5] | 2 | 15 |
| route-turn | 21 | 3 | +18 [+10, +27] | 19 | 1 |
| offroad | 14 | 18 | -4 [-15, +7] | 13 | 17 |
| AF-front | 19 | 4 | +15 [+7, +24] | 18 | 3 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 17 | 4 | +13 [+5, +21] | 15 | 2 |
| rear-hit | 28 | 18 | +10 [-2, +22] | 24 | 14 |
| slow | 17 | 3 | +14 [+6, +23] | 17 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

## GARAGE020 - R8K-s0-ag

- mean -0.148 [-0.209, -0.088]; zeros +29 (new 54, fixed 25)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 50 | +29 [+12, +46] | 54 | 25 |
| overshoot | 7 | 19 | -12 [-20, -5] | 2 | 14 |
| route-turn | 21 | 3 | +18 [+10, +27] | 19 | 1 |
| offroad | 14 | 17 | -3 [-14, +8] | 14 | 17 |
| AF-front | 19 | 6 | +13 [+4, +22] | 17 | 4 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 17 | 4 | +13 [+5, +21] | 15 | 2 |
| rear-hit | 28 | 18 | +10 [-2, +22] | 24 | 14 |
| slow | 17 | 4 | +13 [+5, +22] | 17 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

## GARAGE020 - R8K-s1-ag

- mean -0.156 [-0.218, -0.097]; zeros +32 (new 54, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 47 | +32 [+16, +49] | 54 | 22 |
| overshoot | 7 | 19 | -12 [-20, -5] | 2 | 14 |
| route-turn | 21 | 3 | +18 [+10, +27] | 19 | 1 |
| offroad | 14 | 16 | -2 [-12, +9] | 13 | 15 |
| AF-front | 19 | 5 | +14 [+5, +23] | 18 | 4 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 17 | 4 | +13 [+5, +21] | 15 | 2 |
| rear-hit | 28 | 21 | +7 [-6, +20] | 24 | 17 |
| slow | 17 | 3 | +14 [+6, +23] | 17 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

## GARAGE020 - R8K-s2-ag

- mean -0.160 [-0.224, -0.096]; zeros +32 (new 59, fixed 27)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 79 | 47 | +32 [+14, +50] | 59 | 27 |
| overshoot | 7 | 20 | -13 [-21, -5] | 2 | 15 |
| route-turn | 21 | 2 | +19 [+11, +28] | 20 | 1 |
| offroad | 14 | 14 | +0 [-10, +11] | 14 | 14 |
| AF-front | 19 | 5 | +14 [+5, +24] | 18 | 4 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 17 | 4 | +13 [+5, +21] | 15 | 2 |
| rear-hit | 28 | 18 | +10 [-2, +22] | 24 | 14 |
| slow | 17 | 4 | +13 [+5, +22] | 17 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 35 | -8 [-15, -1] | 2 | 10 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
