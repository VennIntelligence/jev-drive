# PAI 281, ACC + GAP: REC (MV-LC + recovery rows) s0 / s1 vs stored R5LC s0 / s1 (MV-LC) and NTV0-L18-s0

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 279 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R5LC-s0-ag | 279 | 0.746 | 62 | 21 | 3 | 21 | 6 | 1 | 10 | 18 | 7 | 0 | 34/37 |
| R5LC-s1-ag | 279 | 0.725 | 68 | 22 | 4 | 24 | 6 | 1 | 11 | 22 | 6 | 0 | 35/37 |
| REC-s0-ag | 279 | 0.735 | 65 | 22 | 2 | 26 | 6 | 1 | 8 | 21 | 5 | 0 | 35/37 |
| REC-s1-ag | 279 | 0.727 | 67 | 22 | 2 | 22 | 7 | 1 | 13 | 19 | 3 | 0 | 34/37 |

## REC-s0-ag - R5LC-s0-ag

- mean -0.011 [-0.050, +0.029]; zeros +3 (new 17, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 65 | 62 | +3 [-8, +14] | 17 | 14 |
| overshoot | 22 | 21 | +1 [-2, +5] | 2 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 26 | 21 | +5 [-3, +13] | 10 | 5 |
| AF-front | 6 | 6 | +0 [-5, +5] | 3 | 3 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 8 | 10 | -2 [-8, +4] | 4 | 6 |
| rear-hit | 21 | 18 | +3 [-3, +9] | 7 | 4 |
| slow | 5 | 7 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

## REC-s1-ag - R5LC-s1-ag

- mean +0.002 [-0.035, +0.039]; zeros -1 (new 14, fixed 15)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 68 | -1 [-12, +9] | 14 | 15 |
| overshoot | 22 | 22 | +0 [-3, +3] | 1 | 1 |
| route-turn | 2 | 4 | -2 [-5, +0] | 0 | 2 |
| offroad | 22 | 24 | -2 [-9, +4] | 5 | 7 |
| AF-front | 7 | 6 | +1 [-5, +7] | 5 | 4 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 13 | 11 | +2 [-5, +9] | 7 | 5 |
| rear-hit | 19 | 22 | -3 [-9, +3] | 3 | 6 |
| slow | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## REC-s0-ag - NTV0-ag

- mean -0.008 [-0.056, +0.040]; zeros +6 (new 26, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 65 | 59 | +6 [-7, +19] | 26 | 20 |
| overshoot | 22 | 23 | -1 [-6, +4] | 3 | 4 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 26 | 12 | +14 [+5, +23] | 18 | 4 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 8 | 6 | +2 [-3, +8] | 5 | 3 |
| rear-hit | 21 | 26 | -5 [-14, +4] | 8 | 13 |
| slow | 5 | 11 | -6 [-12, +0] | 2 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## REC-s1-ag - NTV0-ag

- mean -0.017 [-0.068, +0.034]; zeros +8 (new 29, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 59 | +8 [-6, +22] | 29 | 21 |
| overshoot | 22 | 23 | -1 [-6, +4] | 3 | 4 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 22 | 12 | +10 [+2, +18] | 14 | 4 |
| AF-front | 7 | 3 | +4 [-1, +10] | 6 | 2 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 13 | 6 | +7 [+0, +14] | 10 | 3 |
| rear-hit | 19 | 26 | -7 [-16, +2] | 8 | 15 |
| slow | 3 | 11 | -8 [-14, -3] | 0 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-3, +6] | 3 | 2 |

## R5LC-s0-ag - NTV0-ag

- mean +0.003 [-0.043, +0.049]; zeros +3 (new 22, fixed 19)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 62 | 59 | +3 [-10, +16] | 22 | 19 |
| overshoot | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 21 | 12 | +9 [+2, +17] | 12 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 10 | 6 | +4 [-1, +10] | 6 | 2 |
| rear-hit | 18 | 26 | -8 [-17, +0] | 5 | 13 |
| slow | 7 | 11 | -4 [-11, +3] | 4 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-2, +5] | 2 | 1 |

## R5LC-s1-ag - NTV0-ag

- mean -0.019 [-0.068, +0.029]; zeros +9 (new 27, fixed 18)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 68 | 59 | +9 [-4, +22] | 27 | 18 |
| overshoot | 22 | 23 | -1 [-5, +3] | 2 | 3 |
| route-turn | 4 | 15 | -11 [-18, -4] | 1 | 12 |
| offroad | 24 | 12 | +12 [+4, +21] | 15 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 11 | 6 | +5 [-1, +11] | 7 | 2 |
| rear-hit | 22 | 26 | -4 [-13, +5] | 9 | 13 |
| slow | 6 | 11 | -5 [-12, +1] | 3 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## REC-s1-ag - REC-s0-ag

- mean -0.009 [-0.043, +0.025]; zeros +2 (new 13, fixed 11)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 65 | +2 [-8, +12] | 13 | 11 |
| overshoot | 22 | 22 | +0 [-4, +4] | 2 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 22 | 26 | -4 [-10, +1] | 2 | 6 |
| AF-front | 7 | 6 | +1 [-4, +6] | 4 | 3 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 13 | 8 | +5 [-1, +12] | 8 | 3 |
| rear-hit | 19 | 21 | -2 [-8, +4] | 4 | 6 |
| slow | 3 | 5 | -2 [-5, +0] | 0 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## New zeros / partials on the 183 of 279 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R5LC-s0-ag | 19 | 8 | offroad 8, zero-other 5, slow 4, AF-front 3, rear-hit 3, overshoot 2, ok 1, route-turn 1 |
| R5LC-s1-ag | 23 | 10 | offroad 11, zero-other 6, rear-hit 6, slow 3, overshoot 2, AF-front 2, ok 1, route-turn 1, side 1 |
| REC-s0-ag | 20 | 9 | offroad 10, rear-hit 6, zero-other 4, overshoot 3, AF-front 3, slow 2, ok 1 |
| REC-s1-ag | 23 | 7 | offroad 8, zero-other 6, rear-hit 6, AF-front 5, overshoot 3, ok 1, side 1 |

## Local PAI PCS ruler (d392 pool / anchors, out-of-pool read, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| R5LC-s0+AG | 1931 [1859, 1992] | -266 (-264) [-366, -189] | 0.00 |
| R5LC-s1+AG | 1905 [1823, 1959] | -285 (-291) [-394, -208] | 0.00 |
| R6REC-s0+AG | 1908 [1837, 1959] | -288 (-287) [-377, -205] | 0.00 |
| R6REC-s1+AG | 1952 [1893, 2005] | -242 (-243) [-336, -177] | 0.00 |

Reference points (d429): R2C-s0+AG -29 [-95, +19], R3MV-s0+AG -276 [-375, -217]. Full ruler report: pcs_rec.md.

## Registered rule (plans/2026-10-21-recovery-rows.md)

adopt-candidate = both REC seeds: mean - NTV0 >= 0 (paired point estimate), offroad + zero-other zeros <= NTV0's + 3 = 21, local PCS ruler difference to NTV0 with a 95 % CI upper bound >= 0.

| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | ruler - NTV0 [95 % CI] | CI upper >= 0 |
|:--|--:|:-:|--:|:-:|--:|:-:|
| REC-s0-ag | -0.008 [-0.056, +0.040] | no | 34 (offroad 26, zero-other 8) | no | -288 [-377, -205] | no |
| REC-s1-ag | -0.017 [-0.068, +0.034] | no | 35 (offroad 22, zero-other 13) | no | -242 [-336, -177] | no |

**REC: not an adopt-candidate.**
