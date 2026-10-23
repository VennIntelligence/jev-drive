# PAI 281, ACC + GAP, Harmonizer OFF: RB0L-F-s0 vs NTV0NS-L18-s0 (NTV0 without S rows)

Scenes every read scored without a driver RPC error: 278 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0L-ag | 278 | 0.597 | 102 | 25 | 18 | 37 | 10 | 1 | 11 | 22 | 9 | 0 | 33/37 |
| NTV0NS-ag | 278 | 0.624 | 93 | 23 | 20 | 21 | 14 | 2 | 13 | 24 | 6 | 0 | 32/37 |
| RB0F-ag | 278 | 0.646 | 88 | 22 | 21 | 19 | 8 | 4 | 14 | 22 | 8 | 0 | 32/37 |
| NTV0s0-ag | 278 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |

## RB0L-ag - NTV0NS-ag

- mean -0.026 [-0.070, +0.017]; zeros +9 (new 27, fixed 18)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 102 | 93 | +9 [-4, +22] | 27 | 18 |
| overshoot | 25 | 23 | +2 [-3, +8] | 5 | 3 |
| route-turn | 18 | 20 | -2 [-9, +5] | 5 | 7 |
| offroad | 37 | 21 | +16 [+7, +25] | 19 | 3 |
| AF-front | 10 | 14 | -4 [-12, +3] | 5 | 9 |
| side | 1 | 2 | -1 [-3, +0] | 0 | 1 |
| zero-other | 11 | 13 | -2 [-10, +6] | 8 | 10 |
| rear-hit | 22 | 24 | -2 [-8, +4] | 4 | 6 |
| slow | 9 | 6 | +3 [-1, +8] | 4 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 32 | +1 [+0, +3] | 1 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
