# PAI 281, ACC + GAP: R5-LC (MV-LC) s0 / s1 vs R3MV s0 / s1 and stored NTV0-L18-s0 / R2C-L18-s0

Scenes every read scored without a driver RPC error: 275 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 275 | 0.745 | 58 | 22 | 15 | 12 | 3 | 0 | 6 | 25 | 11 | 0 | 33/37 |
| R2C-ag | 275 | 0.742 | 59 | 21 | 15 | 11 | 4 | 1 | 7 | 24 | 10 | 0 | 33/37 |
| R3MV-s0-ag | 275 | 0.741 | 61 | 21 | 2 | 20 | 3 | 1 | 14 | 24 | 5 | 0 | 35/37 |
| R3MV-s1-ag | 275 | 0.712 | 70 | 20 | 3 | 21 | 3 | 2 | 21 | 21 | 4 | 0 | 34/37 |
| R5LC-s0-ag | 275 | 0.746 | 61 | 20 | 3 | 21 | 6 | 1 | 10 | 18 | 7 | 0 | 34/37 |
| R5LC-s1-ag | 275 | 0.724 | 67 | 21 | 4 | 24 | 6 | 1 | 11 | 22 | 6 | 0 | 35/37 |

## R5LC-s0-ag - R3MV-s0-ag

- mean +0.005 [-0.025, +0.035]; zeros +0 (new 9, fixed 9)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 61 | 61 | +0 [-8, +8] | 9 | 9 |
| overshoot | 20 | 21 | -1 [-3, +0] | 0 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 21 | 20 | +1 [-3, +5] | 3 | 2 |
| AF-front | 6 | 3 | +3 [+0, +7] | 3 | 0 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 10 | 14 | -4 [-10, +1] | 2 | 6 |
| rear-hit | 18 | 24 | -6 [-12, +0] | 2 | 8 |
| slow | 7 | 5 | +2 [+0, +5] | 2 | 0 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## R5LC-s1-ag - R3MV-s1-ag

- mean +0.012 [-0.023, +0.046]; zeros -3 (new 11, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 70 | -3 [-13, +7] | 11 | 14 |
| overshoot | 21 | 20 | +1 [+0, +3] | 1 | 0 |
| route-turn | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| offroad | 24 | 21 | +3 [-3, +9] | 6 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 2 | -1 [-5, +2] | 1 | 2 |
| zero-other | 11 | 21 | -10 [-17, -4] | 1 | 11 |
| rear-hit | 22 | 21 | +1 [-4, +6] | 4 | 3 |
| slow | 6 | 4 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

## R5LC-s0-ag - NTV0-ag

- mean +0.001 [-0.045, +0.047]; zeros +3 (new 22, fixed 19)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 61 | 58 | +3 [-9, +16] | 22 | 19 |
| overshoot | 20 | 22 | -2 [-7, +3] | 2 | 4 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 21 | 12 | +9 [+2, +17] | 12 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 10 | 6 | +4 [-1, +10] | 6 | 2 |
| rear-hit | 18 | 25 | -7 [-15, +1] | 5 | 12 |
| slow | 7 | 11 | -4 [-11, +3] | 4 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-2, +4] | 2 | 1 |

## R5LC-s1-ag - NTV0-ag

- mean -0.021 [-0.069, +0.026]; zeros +9 (new 27, fixed 18)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 58 | +9 [-4, +22] | 27 | 18 |
| overshoot | 21 | 22 | -1 [-5, +3] | 2 | 3 |
| route-turn | 4 | 15 | -11 [-18, -4] | 1 | 12 |
| offroad | 24 | 12 | +12 [+4, +20] | 15 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 11 | 6 | +5 [-1, +11] | 7 | 2 |
| rear-hit | 22 | 25 | -3 [-12, +6] | 9 | 12 |
| slow | 6 | 11 | -5 [-12, +1] | 3 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R3MV-s1-ag - R3MV-s0-ag

- mean -0.029 [-0.062, +0.003]; zeros +9 (new 15, fixed 6)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 61 | +9 [+0, +18] | 15 | 6 |
| overshoot | 20 | 21 | -1 [-3, +0] | 0 | 1 |
| route-turn | 3 | 2 | +1 [+0, +3] | 1 | 0 |
| offroad | 21 | 20 | +1 [-5, +7] | 5 | 4 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 2 | 1 | +1 [-2, +5] | 2 | 1 |
| zero-other | 21 | 14 | +7 [+2, +13] | 8 | 1 |
| rear-hit | 21 | 24 | -3 [-8, +2] | 2 | 5 |
| slow | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## R3MV-s0-ag - NTV0-ag

- mean -0.004 [-0.053, +0.045]; zeros +3 (new 24, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 61 | 58 | +3 [-10, +16] | 24 | 21 |
| overshoot | 21 | 22 | -1 [-6, +4] | 3 | 4 |
| route-turn | 2 | 15 | -13 [-20, -6] | 0 | 13 |
| offroad | 20 | 12 | +8 [+1, +16] | 12 | 4 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 14 | 6 | +8 [+1, +16] | 11 | 3 |
| rear-hit | 24 | 25 | -1 [-9, +7] | 9 | 10 |
| slow | 5 | 11 | -6 [-12, +0] | 2 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R3MV-s1-ag - NTV0-ag

- mean -0.033 [-0.085, +0.019]; zeros +12 (new 32, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 58 | +12 [-2, +26] | 32 | 20 |
| overshoot | 20 | 22 | -2 [-7, +3] | 2 | 4 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 21 | 12 | +9 [+2, +17] | 12 | 3 |
| AF-front | 3 | 3 | +0 [-4, +4] | 2 | 2 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 21 | 6 | +15 [+7, +24] | 17 | 2 |
| rear-hit | 21 | 25 | -4 [-13, +5] | 8 | 12 |
| slow | 4 | 11 | -7 [-14, -1] | 2 | 9 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-3, +5] | 3 | 2 |

## R5LC-s1-ag - R5LC-s0-ag

- mean -0.022 [-0.046, +0.001]; zeros +6 (new 9, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 61 | +6 [-1, +13] | 9 | 3 |
| overshoot | 21 | 20 | +1 [+0, +3] | 1 | 0 |
| route-turn | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| offroad | 24 | 21 | +3 [-3, +9] | 6 | 3 |
| AF-front | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 11 | 10 | +1 [-2, +5] | 2 | 1 |
| rear-hit | 22 | 18 | +4 [-1, +10] | 6 | 2 |
| slow | 6 | 7 | -1 [-5, +3] | 2 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

## R5LC-s0-ag - R2C-ag

- mean +0.004 [-0.043, +0.050]; zeros +2 (new 22, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 61 | 59 | +2 [-11, +15] | 22 | 20 |
| overshoot | 20 | 21 | -1 [-6, +4] | 3 | 4 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 21 | 11 | +10 [+2, +18] | 14 | 4 |
| AF-front | 6 | 4 | +2 [-3, +7] | 4 | 2 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 10 | 7 | +3 [-2, +8] | 5 | 2 |
| rear-hit | 18 | 24 | -6 [-13, +1] | 4 | 10 |
| slow | 7 | 10 | -3 [-10, +3] | 4 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-2, +4] | 2 | 1 |

## R5LC-s1-ag - R2C-ag

- mean -0.018 [-0.067, +0.031]; zeros +8 (new 28, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 67 | 59 | +8 [-5, +22] | 28 | 20 |
| overshoot | 21 | 21 | +0 [-5, +5] | 3 | 3 |
| route-turn | 4 | 15 | -11 [-18, -4] | 1 | 12 |
| offroad | 24 | 11 | +13 [+4, +22] | 17 | 4 |
| AF-front | 6 | 4 | +2 [-3, +7] | 4 | 2 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 11 | 7 | +4 [-1, +10] | 6 | 2 |
| rear-hit | 22 | 24 | -2 [-10, +6] | 8 | 10 |
| slow | 6 | 10 | -4 [-10, +2] | 3 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## Registered rule (prereg addendum 5)

adopt-candidate = both MV-LC seeds beat NTV0 on the mean (paired point estimate > 0) and each seed's offroad + zero-other zeros <= NTV0's + 3 = 21.

| arm | - NTV0 mean [95 % CI] | beats NTV0 | offroad + zero-other | <= cap |
|:--|--:|:-:|--:|:-:|
| R5LC-s0-ag | +0.001 [-0.045, +0.047] | yes | 31 (offroad 21, zero-other 10) | no |
| R5LC-s1-ag | -0.021 [-0.069, +0.026] | no | 35 (offroad 24, zero-other 11) | no |

**MV-LC: not an adopt-candidate** (primary rule). Seed noise (R3MV s1 - s0, same argv): -0.029 [-0.062, +0.003].

### Secondary column (addendum 5, d429): new zeros / partials on the 181 of 275 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R2C-ag | 6 | 2 | AF-front 2, side 1, route-turn 1, offroad 1, zero-other 1, rear-hit 1, slow 1 |
| R3MV-s0-ag | 21 | 9 | zero-other 10, offroad 7, rear-hit 6, overshoot 3, slow 2, ok 1, side 1 |
| R3MV-s1-ag | 28 | 8 | zero-other 15, offroad 8, rear-hit 5, slow 2, side 2, overshoot 2, ok 1, route-turn 1 |
| R5LC-s0-ag | 19 | 8 | offroad 8, zero-other 5, slow 4, AF-front 3, rear-hit 3, overshoot 2, ok 1, route-turn 1 |
| R5LC-s1-ag | 23 | 10 | offroad 11, zero-other 6, rear-hit 6, slow 3, overshoot 2, AF-front 2, ok 1, route-turn 1, side 1 |

The local PAI PCS ruler (the second secondary column) is appended by `r5.py lcpcs` after its CPU jobs.

### Secondary column (addendum 5, d429): local PAI PCS ruler (d392 pool / anchors, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| R2C-s0+AG | 2161 [2089, 2239] | -29 (-34) [-95, +19] | 0.17 |
| R3MV-s0+AG | 1918 [1849, 1960] | -276 (-277) [-375, -217] | 0.00 |
| R3MV-s1+AG | 1915 [1828, 1967] | -282 (-280) [-384, -225] | 0.00 |
| R5LC-s0+AG | 1931 [1859, 1992] | -266 (-264) [-366, -189] | 0.00 |
| R5LC-s1+AG | 1905 [1823, 1959] | -285 (-291) [-394, -208] | 0.00 |

Registered secondary rule: each MV-LC seed's ruler difference to NTV0 has a 95 % CI upper bound >= 0 (not credibly below NTV0; R2C passed at -29 [-95, +19], R3MV failed at -276 [-375, -217], d429): R5LC-s0+AG fail, R5LC-s1+AG fail -> **not both**. Full ruler report: pcs_r5l.md.
