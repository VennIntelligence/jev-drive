# PAI 281, ACC + GAP: R5A (S3 outcome rows) s0 vs stored R3MV-L18-s0 and NTV0-L18-s0

Scenes every read scored without a driver RPC error: 278 (of 280 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 278 | 0.744 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| R3MV-s0-ag | 278 | 0.741 | 62 | 22 | 2 | 20 | 3 | 1 | 14 | 24 | 5 | 0 | 35/37 |
| R5A-s0-ag | 278 | 0.673 | 70 | 21 | 6 | 20 | 7 | 1 | 15 | 27 | 8 | 0 | 20/37 |

## R5A-s0-ag - R3MV-s0-ag

- mean -0.068 [-0.113, -0.023]; zeros +8 (new 20, fixed 12)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 62 | +8 [-3, +19] | 20 | 12 |
| overshoot | 21 | 22 | -1 [-6, +4] | 3 | 4 |
| route-turn | 6 | 2 | +4 [+1, +8] | 4 | 0 |
| offroad | 20 | 20 | +0 [-7, +7] | 7 | 7 |
| AF-front | 7 | 3 | +4 [+1, +8] | 4 | 0 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 15 | 14 | +1 [-5, +7] | 5 | 4 |
| rear-hit | 27 | 24 | +3 [-5, +11] | 10 | 7 |
| slow | 8 | 5 | +3 [-2, +8] | 5 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 20 | 35 | -15 [-23, -8] | 0 | 15 |

## R5A-s0-ag - NTV0-ag

- mean -0.071 [-0.120, -0.024]; zeros +11 (new 25, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 59 | +11 [-1, +23] | 25 | 14 |
| overshoot | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| route-turn | 6 | 15 | -9 [-16, -3] | 1 | 10 |
| offroad | 20 | 12 | +8 [+0, +16] | 12 | 4 |
| AF-front | 7 | 3 | +4 [+0, +9] | 5 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 15 | 6 | +9 [+2, +16] | 11 | 2 |
| rear-hit | 27 | 25 | +2 [-7, +11] | 12 | 10 |
| slow | 8 | 11 | -3 [-10, +4] | 5 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 20 | 33 | -13 [-21, -6] | 1 | 14 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## New zeros / partials on the 183 of 278 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R3MV-s0-ag | 21 | 9 | zero-other 10, offroad 7, rear-hit 6, overshoot 3, slow 2, ok 1, side 1 |
| R5A-s0-ag | 21 | 16 | rear-hit 10, zero-other 9, offroad 7, slow 4, AF-front 3, ok 2, overshoot 1, route-turn 1 |

## Local PAI PCS ruler (d392 pool / anchors, out-of-pool read, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| R3MV-s0+AG | 1918 [1849, 1960] | -276 (-277) [-375, -217] | 0.00 |
| R5A-s0+AG | 1925 [1847, 1969] | -269 (-270) [-358, -210] | 0.00 |

Reference points (d429): R2C-s0+AG -29 [-95, +19], R3MV-s0+AG -276 [-375, -217]. Full ruler report: pcs_r5a.md.
