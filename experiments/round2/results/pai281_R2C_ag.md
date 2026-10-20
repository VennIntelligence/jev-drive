# PAI 281, ACC + GAP: R2C vs stored NTV0-L18-s0

Scenes every read scored without a driver RPC error: 280 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 280 | 0.744 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R2C-ag | 280 | 0.741 | 60 | 22 | 15 | 11 | 4 | 1 | 7 | 25 | 10 | 0 | 33/37 |

## R2C-ag - NTV0-ag

- mean -0.003 [-0.030, +0.023]; zeros +1 (new 8, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 60 | 59 | +1 [-6, +8] | 8 | 7 |
| overshoot | 22 | 23 | -1 [-3, +0] | 0 | 1 |
| route-turn | 15 | 15 | +0 [-5, +5] | 4 | 4 |
| offroad | 11 | 12 | -1 [-6, +3] | 2 | 3 |
| AF-front | 4 | 3 | +1 [-2, +5] | 2 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 7 | 6 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 25 | 26 | -1 [-6, +3] | 2 | 3 |
| slow | 10 | 11 | -1 [-5, +2] | 1 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 33 | +0 [-3, +3] | 1 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
