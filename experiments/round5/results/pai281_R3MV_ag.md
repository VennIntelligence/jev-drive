# PAI 281, ACC + GAP: R3MV vs stored NTV0-L18-s0 and R2C-L18-s0

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 279 | 0.745 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| R2C-ag | 279 | 0.742 | 60 | 22 | 15 | 11 | 4 | 1 | 7 | 24 | 10 | 0 | 33/37 |
| R3MV-ag | 279 | 0.742 | 62 | 22 | 2 | 20 | 3 | 1 | 14 | 24 | 5 | 0 | 35/37 |

## R3MV-ag - NTV0-ag

- mean -0.004 [-0.052, +0.044]; zeros +3 (new 24, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 62 | 59 | +3 [-10, +16] | 24 | 21 |
| overshoot | 22 | 23 | -1 [-6, +4] | 3 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 20 | 12 | +8 [+1, +16] | 12 | 4 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 14 | 6 | +8 [+1, +15] | 11 | 3 |
| rear-hit | 24 | 25 | -1 [-9, +8] | 9 | 10 |
| slow | 5 | 11 | -6 [-13, +0] | 2 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R3MV-ag - R2C-ag

- mean -0.001 [-0.049, +0.049]; zeros +2 (new 24, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 62 | 60 | +2 [-12, +15] | 24 | 22 |
| overshoot | 22 | 22 | +0 [-6, +6] | 4 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 20 | 11 | +9 [+1, +18] | 13 | 4 |
| AF-front | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 14 | 7 | +7 [+0, +14] | 10 | 3 |
| rear-hit | 24 | 24 | +0 [-7, +7] | 7 | 7 |
| slow | 5 | 10 | -5 [-11, +1] | 2 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
