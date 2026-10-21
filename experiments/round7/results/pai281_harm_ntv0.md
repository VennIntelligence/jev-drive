# PAI 281, ACC + GAP, Harmonizer ON vs stored OFF: NTV0-L18-s0, all finished scenes (partial read, plan amendment 1)

Scenes every read scored without a driver RPC error: 280 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-off | 280 | 0.744 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| NTV0-on | 280 | 0.746 | 58 | 21 | 9 | 17 | 5 | 0 | 6 | 27 | 9 | 0 | 33/37 |

## NTV0-on - NTV0-off

- mean +0.002 [-0.032, +0.035]; zeros -1 (new 11, fixed 12)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 58 | 59 | -1 [-10, +8] | 11 | 12 |
| overshoot | 21 | 23 | -2 [-6, +2] | 1 | 3 |
| route-turn | 9 | 15 | -6 [-12, -1] | 1 | 7 |
| offroad | 17 | 12 | +5 [-1, +11] | 7 | 2 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 6 | 6 | +0 [-5, +5] | 3 | 3 |
| rear-hit | 27 | 26 | +1 [-3, +5] | 3 | 2 |
| slow | 9 | 11 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 33 | 33 | +0 [-3, +3] | 1 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
