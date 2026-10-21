# PAI 281, ACC + GAP: R6DAG (DAgger map teacher) s0 / s1 vs stored R5LC (MV-LC) s0 / s1 and NTV0-L18-s0

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 279 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R5LC-s0-ag | 279 | 0.746 | 62 | 21 | 3 | 21 | 6 | 1 | 10 | 18 | 7 | 0 | 34/37 |
| R5LC-s1-ag | 279 | 0.725 | 68 | 22 | 4 | 24 | 6 | 1 | 11 | 22 | 6 | 0 | 35/37 |
| DAG-s0-ag | 279 | 0.775 | 53 | 20 | 2 | 20 | 4 | 2 | 5 | 23 | 6 | 0 | 35/37 |
| DAG-s1-ag | 279 | 0.815 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |

## DAG-s0-ag - NTV0-ag

- mean +0.032 [-0.011, +0.076]; zeros -6 (new 15, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 59 | -6 [-18, +6] | 15 | 21 |
| overshoot | 20 | 23 | -3 [-8, +1] | 1 | 4 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 20 | 12 | +8 [+0, +16] | 13 | 5 |
| AF-front | 4 | 3 | +1 [-3, +5] | 3 | 2 |
| side | 2 | 0 | +2 [+0, +5] | 2 | 0 |
| zero-other | 5 | 6 | -1 [-5, +3] | 2 | 3 |
| rear-hit | 23 | 26 | -3 [-12, +6] | 9 | 12 |
| slow | 6 | 11 | -5 [-12, +1] | 3 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## DAG-s1-ag - NTV0-ag

- mean +0.072 [+0.034, +0.111]; zeros -16 (new 6, fixed 22)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 43 | 59 | -16 [-26, -6] | 6 | 22 |
| overshoot | 19 | 23 | -4 [-8, -1] | 0 | 4 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 12 | 12 | +0 [-6, +6] | 5 | 5 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 3 | 0 | +3 [+0, +7] | 3 | 0 |
| zero-other | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| rear-hit | 21 | 26 | -5 [-13, +2] | 5 | 10 |
| slow | 6 | 11 | -5 [-11, +1] | 2 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## DAG-s0-ag - R5LC-s0-ag

- mean +0.029 [-0.009, +0.068]; zeros -9 (new 11, fixed 20)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 53 | 62 | -9 [-20, +2] | 11 | 20 |
| overshoot | 20 | 21 | -1 [-5, +2] | 1 | 2 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 20 | 21 | -1 [-8, +6] | 6 | 7 |
| AF-front | 4 | 6 | -2 [-6, +2] | 1 | 3 |
| side | 2 | 1 | +1 [-2, +5] | 2 | 1 |
| zero-other | 5 | 10 | -5 [-12, +1] | 3 | 8 |
| rear-hit | 23 | 18 | +5 [-1, +11] | 7 | 2 |
| slow | 6 | 7 | -1 [-4, +2] | 1 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

## DAG-s1-ag - R5LC-s1-ag

- mean +0.091 [+0.053, +0.130]; zeros -25 (new 5, fixed 30)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 43 | 68 | -25 [-37, -14] | 5 | 30 |
| overshoot | 19 | 22 | -3 [-7, +0] | 0 | 3 |
| route-turn | 2 | 4 | -2 [-5, +0] | 0 | 2 |
| offroad | 12 | 24 | -12 [-20, -5] | 2 | 14 |
| AF-front | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| side | 3 | 1 | +2 [-2, +6] | 3 | 1 |
| zero-other | 4 | 11 | -7 [-14, -1] | 2 | 9 |
| rear-hit | 21 | 22 | -1 [-8, +6] | 6 | 7 |
| slow | 6 | 6 | +0 [-5, +5] | 3 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## DAG-s1-ag - DAG-s0-ag

- mean +0.040 [+0.014, +0.068]; zeros -10 (new 3, fixed 13)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 43 | 53 | -10 [-18, -2] | 3 | 13 |
| overshoot | 19 | 20 | -1 [-3, +0] | 0 | 1 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 12 | 20 | -8 [-14, -2] | 1 | 9 |
| AF-front | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| side | 3 | 2 | +1 [-3, +5] | 3 | 2 |
| zero-other | 4 | 5 | -1 [-3, +0] | 0 | 1 |
| rear-hit | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| slow | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

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

## R5LC-s1-ag - R5LC-s0-ag

- mean -0.022 [-0.045, +0.001]; zeros +6 (new 9, fixed 3)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 68 | 62 | +6 [-1, +13] | 9 | 3 |
| overshoot | 22 | 21 | +1 [+0, +3] | 1 | 0 |
| route-turn | 4 | 3 | +1 [+0, +3] | 1 | 0 |
| offroad | 24 | 21 | +3 [-3, +9] | 6 | 3 |
| AF-front | 6 | 6 | +0 [-4, +4] | 2 | 2 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 11 | 10 | +1 [-2, +4] | 2 | 1 |
| rear-hit | 22 | 18 | +4 [-1, +10] | 6 | 2 |
| slow | 6 | 7 | -1 [-5, +3] | 2 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## Registered rule (plan section 6)

adopt-candidate (primary) = both DAG seeds: mean - NTV0 >= 0 (paired point estimate) and offroad + zero-other <= NTV0's + 3 = 21; the ruler rule (CI upper bound of PCS - NTV0 >= 0) is appended by `dag.py pcs`.

| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | - MV-LC same seed | offroad + zero-other - MV-LC | new zeros / partials on NTV0 1.0 |
|:--|--:|:-:|--:|:-:|--:|--:|--:|
| DAG-s0-ag | +0.032 [-0.011, +0.076] | yes | 25 (offroad 20, zero-other 5) | no | +0.029 [-0.009, +0.068] | -6 | 12 / 11 |
| DAG-s1-ag | +0.072 [+0.034, +0.111] | yes | 16 (offroad 12, zero-other 4) | yes | +0.091 [+0.053, +0.130] | -19 | 5 / 7 |

**Primary rule: not both seeds.**

### New zeros / partials on the 183 of 279 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R5LC-s0-ag | 19 | 8 | offroad 8, zero-other 5, slow 4, AF-front 3, rear-hit 3, overshoot 2, ok 1, route-turn 1 |
| R5LC-s1-ag | 23 | 10 | offroad 11, zero-other 6, rear-hit 6, slow 3, overshoot 2, AF-front 2, ok 1, route-turn 1, side 1 |
| DAG-s0-ag | 12 | 11 | rear-hit 6, offroad 6, slow 3, ok 2, zero-other 2, side 2, AF-front 1, overshoot 1 |
| DAG-s1-ag | 5 | 7 | offroad 3, rear-hit 3, ok 2, slow 2, side 1, zero-other 1 |

### Local PAI PCS ruler (d392 pool / anchors, out of pool, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| R5LC-s0+AG | 1931 [1859, 1992] | -266 (-264) [-366, -189] | 0.00 |
| R5LC-s1+AG | 1905 [1823, 1959] | -285 (-291) [-394, -208] | 0.00 |
| DAG-s0+AG | 1983 [1907, 2122] | -210 (-213) [-305, -57] | 0.00 |
| DAG-s1+AG | 2040 [1965, 2224] | -153 (-155) [-260, +36] | 0.06 |

Ruler rule: each DAG seed's ruler difference to NTV0 has a 95 % CI upper bound >= 0: DAG-s0+AG fail, DAG-s1+AG pass

**DAG: not an adopt-candidate** (primary rule fail; ruler rule fail). Clearly better than MV-LC (both seeds: offroad + zero-other <= MV-LC - 5 and ruler point >= MV-LC + 100): {'s0': False, 's1': True} -> no.
