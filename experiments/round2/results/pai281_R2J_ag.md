# PAI 281, ACC + GAP: R2J vs stored NTV0-L18-s0

Scenes every read scored without a driver RPC error: 280 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 280 | 0.744 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R2J-ag | 280 | 0.689 | 64 | 21 | 2 | 20 | 8 | 0 | 13 | 33 | 12 | 0 | 21/37 |

## R2J-ag - NTV0-ag

- mean -0.056 [-0.107, -0.005]; zeros +5 (new 25, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 64 | 59 | +5 [-8, +19] | 25 | 20 |
| overshoot | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 20 | 12 | +8 [+0, +16] | 12 | 4 |
| AF-front | 8 | 3 | +5 [+0, +10] | 6 | 1 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 13 | 6 | +7 [+1, +14] | 9 | 2 |
| rear-hit | 33 | 26 | +7 [-4, +18] | 19 | 12 |
| slow | 12 | 11 | +1 [-7, +9] | 8 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 21 | 33 | -12 [-20, -5] | 2 | 14 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
