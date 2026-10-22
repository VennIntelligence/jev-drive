# PAI 281, ACC + GAP, Harmonizer OFF: bases RB0-P30 / P50 / P70-s0 (PAI rows from the start) vs RB0-F-s0

Scenes every read scored without a driver RPC error: 281 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0F-ag | 281 | 0.650 | 88 | 22 | 21 | 19 | 8 | 4 | 14 | 23 | 8 | 0 | 32/37 |
| RB0P30-ag | 281 | 0.593 | 105 | 21 | 9 | 30 | 10 | 8 | 27 | 20 | 5 | 0 | 29/37 |
| RB0P50-ag | 281 | 0.605 | 102 | 21 | 11 | 28 | 10 | 8 | 24 | 18 | 3 | 0 | 27/37 |
| RB0P70-ag | 281 | 0.564 | 114 | 26 | 11 | 30 | 14 | 5 | 28 | 16 | 5 | 0 | 29/37 |

## RB0P30-ag - RB0F-ag

- mean -0.057 [-0.110, -0.004]; zeros +17 (new 39, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 105 | 88 | +17 [+2, +32] | 39 | 22 |
| overshoot | 21 | 22 | -1 [-7, +5] | 4 | 5 |
| route-turn | 9 | 21 | -12 [-21, -3] | 4 | 16 |
| offroad | 30 | 19 | +11 [+1, +21] | 19 | 8 |
| AF-front | 10 | 8 | +2 [-5, +9] | 8 | 6 |
| side | 8 | 4 | +4 [-2, +10] | 7 | 3 |
| zero-other | 27 | 14 | +13 [+4, +22] | 18 | 5 |
| rear-hit | 20 | 23 | -3 [-12, +6] | 9 | 12 |
| slow | 5 | 8 | -3 [-8, +2] | 2 | 5 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 32 | -3 [-8, +1] | 1 | 4 |

## RB0P50-ag - RB0F-ag

- mean -0.045 [-0.097, +0.008]; zeros +14 (new 36, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 102 | 88 | +14 [-1, +29] | 36 | 22 |
| overshoot | 21 | 22 | -1 [-6, +4] | 3 | 4 |
| route-turn | 11 | 21 | -10 [-19, -1] | 5 | 15 |
| offroad | 28 | 19 | +9 [-1, +19] | 17 | 8 |
| AF-front | 10 | 8 | +2 [-5, +9] | 7 | 5 |
| side | 8 | 4 | +4 [-2, +10] | 7 | 3 |
| zero-other | 24 | 14 | +10 [+1, +19] | 16 | 6 |
| rear-hit | 18 | 23 | -5 [-14, +3] | 7 | 12 |
| slow | 3 | 8 | -5 [-10, +0] | 1 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 27 | 32 | -5 [-10, -1] | 0 | 5 |

## RB0P70-ag - RB0F-ag

- mean -0.085 [-0.140, -0.029]; zeros +26 (new 47, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 114 | 88 | +26 [+10, +42] | 47 | 21 |
| overshoot | 26 | 22 | +4 [-3, +11] | 9 | 5 |
| route-turn | 11 | 21 | -10 [-19, -1] | 6 | 16 |
| offroad | 30 | 19 | +11 [+2, +21] | 18 | 7 |
| AF-front | 14 | 8 | +6 [-3, +15] | 13 | 7 |
| side | 5 | 4 | +1 [-4, +6] | 4 | 3 |
| zero-other | 28 | 14 | +14 [+4, +24] | 20 | 6 |
| rear-hit | 16 | 23 | -7 [-16, +2] | 8 | 15 |
| slow | 5 | 8 | -3 [-8, +1] | 1 | 4 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 29 | 32 | -3 [-8, +2] | 2 | 5 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
