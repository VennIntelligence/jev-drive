# PAI 281, ACC + GAP, Harmonizer ON vs stored OFF on chunks c000-c002 (partial read, plan amendment 1)

Scenes every read scored without a driver RPC error: 108 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-off | 108 | 0.754 | 22 | 10 | 4 | 6 | 0 | 0 | 2 | 10 | 5 | 0 | 13/13 |
| R6DAG-off | 108 | 0.815 | 17 | 8 | 1 | 6 | 0 | 1 | 1 | 8 | 4 | 0 | 13/13 |
| NTV0-on | 108 | 0.734 | 24 | 8 | 3 | 10 | 0 | 0 | 3 | 10 | 4 | 0 | 13/13 |
| R6DAG-on | 108 | 0.800 | 18 | 8 | 1 | 6 | 0 | 2 | 1 | 9 | 5 | 0 | 13/13 |

## NTV0-on - NTV0-off

- mean -0.020 [-0.084, +0.044]; zeros +2 (new 7, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 24 | 22 | +2 [-5, +9] | 7 | 5 |
| overshoot | 8 | 10 | -2 [-5, +0] | 0 | 2 |
| route-turn | 3 | 4 | -1 [-4, +2] | 1 | 2 |
| offroad | 10 | 6 | +4 [-1, +10] | 6 | 2 |
| AF-front | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 3 | 2 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 10 | 10 | +0 [+0, +0] | 0 | 0 |
| slow | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 13 | 13 | +0 [+0, +0] | 0 | 0 |

## R6DAG-on - R6DAG-off

- mean -0.014 [-0.063, +0.034]; zeros +1 (new 4, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 18 | 17 | +1 [-4, +6] | 4 | 3 |
| overshoot | 8 | 8 | +0 [+0, +0] | 0 | 0 |
| route-turn | 1 | 1 | +0 [+0, +0] | 0 | 0 |
| offroad | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| AF-front | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| side | 2 | 1 | +1 [+0, +3] | 1 | 0 |
| zero-other | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| rear-hit | 9 | 8 | +1 [+0, +3] | 1 | 0 |
| slow | 5 | 4 | +1 [-2, +5] | 2 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 13 | 13 | +0 [+0, +0] | 0 | 0 |

## R6DAG-on - NTV0-on

- mean +0.066 [-0.002, +0.138]; zeros -6 (new 4, fixed 10)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 18 | 24 | -6 [-13, +1] | 4 | 10 |
| overshoot | 8 | 8 | +0 [-3, +3] | 1 | 1 |
| route-turn | 1 | 3 | -2 [-5, +0] | 0 | 2 |
| offroad | 6 | 10 | -4 [-9, +0] | 1 | 5 |
| AF-front | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 1 | 3 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 9 | 10 | -1 [-5, +3] | 2 | 3 |
| slow | 5 | 4 | +1 [-3, +5] | 3 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 13 | 13 | +0 [+0, +0] | 0 | 0 |

## R6DAG-off - NTV0-off

- mean +0.061 [+0.001, +0.124]; zeros -5 (new 3, fixed 8)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 17 | 22 | -5 [-12, +1] | 3 | 8 |
| overshoot | 8 | 10 | -2 [-5, +0] | 0 | 2 |
| route-turn | 1 | 4 | -3 [-7, +0] | 0 | 3 |
| offroad | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| AF-front | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 1 | 2 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 8 | 10 | -2 [-6, +2] | 1 | 3 |
| slow | 4 | 5 | -1 [-5, +3] | 2 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 13 | 13 | +0 [+0, +0] | 0 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).
