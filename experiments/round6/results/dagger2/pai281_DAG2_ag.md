# PAI 281, ACC + GAP: R6DAG2 (DAgger iteration 2) s0 / s1 vs stored R6DAG s0 / s1, R5LC (MV-LC) s0 / s1 and NTV0-L18-s0

Scenes every read scored without a driver RPC error: 279 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 279 | 0.743 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R5LC-s0-ag | 279 | 0.746 | 62 | 21 | 3 | 21 | 6 | 1 | 10 | 18 | 7 | 0 | 34/37 |
| R5LC-s1-ag | 279 | 0.725 | 68 | 22 | 4 | 24 | 6 | 1 | 11 | 22 | 6 | 0 | 35/37 |
| R6DAG-s0-ag | 279 | 0.775 | 53 | 20 | 2 | 20 | 4 | 2 | 5 | 23 | 6 | 0 | 35/37 |
| R6DAG-s1-ag | 279 | 0.815 | 43 | 19 | 2 | 12 | 3 | 3 | 4 | 21 | 6 | 0 | 35/37 |
| DAG2-s0-ag | 279 | 0.789 | 50 | 21 | 2 | 18 | 4 | 0 | 5 | 21 | 8 | 0 | 34/37 |
| DAG2-s1-ag | 279 | 0.794 | 50 | 20 | 2 | 18 | 3 | 0 | 7 | 17 | 7 | 0 | 34/37 |

## DAG2-s0-ag - NTV0-ag

- mean +0.045 [+0.005, +0.088]; zeros -9 (new 12, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 59 | -9 [-20, +2] | 12 | 21 |
| overshoot | 21 | 23 | -2 [-6, +2] | 1 | 3 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 18 | 12 | +6 [-2, +14] | 11 | 5 |
| AF-front | 4 | 3 | +1 [-3, +5] | 3 | 2 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 5 | 6 | -1 [-6, +4] | 3 | 4 |
| rear-hit | 21 | 26 | -5 [-13, +3] | 6 | 11 |
| slow | 8 | 11 | -3 [-9, +3] | 3 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-3, +6] | 3 | 2 |

## DAG2-s1-ag - NTV0-ag

- mean +0.051 [+0.007, +0.095]; zeros -9 (new 14, fixed 23)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 59 | -9 [-21, +3] | 14 | 23 |
| overshoot | 20 | 23 | -3 [-8, +1] | 1 | 4 |
| route-turn | 2 | 15 | -13 [-20, -7] | 0 | 13 |
| offroad | 18 | 12 | +6 [-1, +13] | 10 | 4 |
| AF-front | 3 | 3 | +0 [-4, +4] | 2 | 2 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 7 | 6 | +1 [-3, +5] | 3 | 2 |
| rear-hit | 17 | 26 | -9 [-17, -2] | 3 | 12 |
| slow | 7 | 11 | -4 [-10, +1] | 2 | 6 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-3, +6] | 3 | 2 |

## DAG2-s0-ag - R6DAG-s0-ag

- mean +0.014 [-0.012, +0.041]; zeros -3 (new 6, fixed 9)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 53 | -3 [-11, +4] | 6 | 9 |
| overshoot | 21 | 20 | +1 [-2, +4] | 2 | 1 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 18 | 20 | -2 [-8, +3] | 3 | 5 |
| AF-front | 4 | 4 | +0 [-4, +4] | 2 | 2 |
| side | 0 | 2 | -2 [-5, +0] | 0 | 2 |
| zero-other | 5 | 5 | +0 [-4, +4] | 2 | 2 |
| rear-hit | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| slow | 8 | 6 | +2 [-2, +6] | 3 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## DAG2-s1-ag - R6DAG-s1-ag

- mean -0.021 [-0.049, +0.007]; zeros +7 (new 12, fixed 5)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 43 | +7 [-1, +15] | 12 | 5 |
| overshoot | 20 | 19 | +1 [+0, +3] | 1 | 0 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 18 | 12 | +6 [+0, +12] | 8 | 2 |
| AF-front | 3 | 3 | +0 [-3, +3] | 1 | 1 |
| side | 0 | 3 | -3 [-7, +0] | 0 | 3 |
| zero-other | 7 | 4 | +3 [+0, +7] | 3 | 0 |
| rear-hit | 17 | 21 | -4 [-9, +1] | 1 | 5 |
| slow | 7 | 6 | +1 [-2, +5] | 2 | 1 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## DAG2-s0-ag - R5LC-s0-ag

- mean +0.042 [+0.005, +0.081]; zeros -12 (new 9, fixed 21)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 62 | -12 [-23, -1] | 9 | 21 |
| overshoot | 21 | 21 | +0 [-3, +3] | 1 | 1 |
| route-turn | 2 | 3 | -1 [-3, +0] | 0 | 1 |
| offroad | 18 | 21 | -3 [-10, +4] | 5 | 8 |
| AF-front | 4 | 6 | -2 [-7, +3] | 2 | 4 |
| side | 0 | 1 | -1 [-3, +0] | 0 | 1 |
| zero-other | 5 | 10 | -5 [-12, +2] | 4 | 9 |
| rear-hit | 21 | 18 | +3 [-3, +9] | 6 | 3 |
| slow | 8 | 7 | +1 [-4, +6] | 4 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 34 | +0 [-3, +3] | 1 | 1 |

## DAG2-s1-ag - R5LC-s1-ag

- mean +0.070 [+0.033, +0.108]; zeros -18 (new 7, fixed 25)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 68 | -18 [-29, -7] | 7 | 25 |
| overshoot | 20 | 22 | -2 [-6, +2] | 1 | 3 |
| route-turn | 2 | 4 | -2 [-5, +0] | 0 | 2 |
| offroad | 18 | 24 | -6 [-13, +1] | 4 | 10 |
| AF-front | 3 | 6 | -3 [-7, +0] | 0 | 3 |
| side | 0 | 1 | -1 [-3, +0] | 0 | 1 |
| zero-other | 7 | 11 | -4 [-10, +2] | 3 | 7 |
| rear-hit | 17 | 22 | -5 [-12, +1] | 3 | 8 |
| slow | 7 | 6 | +1 [-3, +5] | 3 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 35 | -1 [-3, +0] | 0 | 1 |

## DAG2-s1-ag - DAG2-s0-ag

- mean +0.005 [-0.024, +0.035]; zeros +0 (new 9, fixed 9)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 50 | 50 | +0 [-8, +8] | 9 | 9 |
| overshoot | 20 | 21 | -1 [-4, +2] | 1 | 2 |
| route-turn | 2 | 2 | +0 [+0, +0] | 0 | 0 |
| offroad | 18 | 18 | +0 [-6, +6] | 4 | 4 |
| AF-front | 3 | 4 | -1 [-5, +2] | 1 | 2 |
| side | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| zero-other | 7 | 5 | +2 [-2, +6] | 3 | 1 |
| rear-hit | 17 | 21 | -4 [-10, +1] | 2 | 6 |
| slow | 7 | 8 | -1 [-5, +2] | 1 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 34 | +0 [+0, +0] | 0 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## Registered rule

adopt-candidate (primary) = both DAG2 seeds: mean - NTV0 >= 0 (paired point estimate) and offroad + zero-other <= NTV0's + 3 = 21; the ruler rule (CI upper bound of PCS - NTV0 >= 0) is appended by `dag.py pcs`.

| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | - R6DAG same seed: mean / offroad + zero-other | - R5LC same seed: mean / offroad + zero-other | new zeros / partials on NTV0 1.0 | launched |
|:--|--:|:-:|--:|:-:|--:|--:|--:|--:|
| DAG2-s0-ag | +0.045 [+0.005, +0.088] | yes | 23 (offroad 18, zero-other 5) | no | +0.014 [-0.012, +0.041] / -2 | +0.042 [+0.005, +0.081] / -8 | 11 / 7 | 34/37 |
| DAG2-s1-ag | +0.051 [+0.007, +0.095] | yes | 25 (offroad 18, zero-other 7) | no | -0.021 [-0.049, +0.007] / +9 | +0.070 [+0.033, +0.108] / -10 | 13 / 6 | 34/37 |

**Primary rule: not both seeds.**

### New zeros / partials on the 183 of 279 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R5LC-s0-ag | 19 | 8 | offroad 8, zero-other 5, slow 4, AF-front 3, rear-hit 3, overshoot 2, ok 1, route-turn 1 |
| R5LC-s1-ag | 23 | 10 | offroad 11, zero-other 6, rear-hit 6, slow 3, overshoot 2, AF-front 2, ok 1, route-turn 1, side 1 |
| R6DAG-s0-ag | 12 | 11 | rear-hit 6, offroad 6, slow 3, ok 2, zero-other 2, side 2, AF-front 1, overshoot 1 |
| R6DAG-s1-ag | 5 | 7 | offroad 3, rear-hit 3, ok 2, slow 2, side 1, zero-other 1 |
| DAG2-s0-ag | 11 | 7 | offroad 6, zero-other 3, rear-hit 3, slow 3, overshoot 1, ok 1, AF-front 1 |
| DAG2-s1-ag | 13 | 6 | offroad 8, zero-other 3, slow 2, ok 2, rear-hit 2, overshoot 1, AF-front 1 |

### The 9 stable OFFROAD-MV scenes (fix.py STABLE): score / fclass

| scene | mechanism | NTV0-ag | R5LC-s0-ag | R5LC-s1-ag | R6DAG-s0-ag | R6DAG-s1-ag | DAG2-s0-ag | DAG2-s1-ag |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 6e7b42a3 | over-speed drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 1.00 ok | 0.00 offroad |
| dc401449 | slow-down drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 1.00 ok | 0.00 offroad |
| a2b415d5 | straight: lateral offset | 1.00 ok | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 0.00 offroad | 0.00 offroad |
| 596af0c7 | curve: cut inside | 0.61 rear-hit | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.62 rear-hit | 0.00 offroad | 0.00 offroad |
| 94c1d722 | turn: cut inside | 0.96 slow | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.98 slow | 0.94 slow | 0.97 slow |
| e1c6ae9f | curve (fork): cut inside | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 1.00 ok |
| ea468790 | turn: cut inside | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad |
| 0f4b8753 | slow-down drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 1.00 ok | 1.00 ok | 1.00 ok | 1.00 ok |
| 7750265d | straight: lateral offset | 0.00 overshoot | 0.00 offroad | 0.00 overshoot | 0.00 offroad | 1.00 ok | 0.00 offroad | 1.00 ok |
| solved (>= 0.98) | | 6 | 0 | 0 | 4 | 6 | 4 | 3 |

### Local PAI PCS ruler (d392 pool / anchors, out of pool, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| DAG-s0+AG | 1983 [1907, 2122] | -210 (-213) [-305, -57] | 0.00 |
| DAG-s1+AG | 2040 [1965, 2224] | -153 (-155) [-260, +36] | 0.06 |
| R5LC-s0+AG | 1931 [1859, 1992] | -266 (-264) [-366, -189] | 0.00 |
| R5LC-s1+AG | 1905 [1823, 1959] | -285 (-291) [-394, -208] | 0.00 |
| DAG2-s0+AG | 2008 [1952, 2109] | -183 (-187) [-277, -83] | 0.00 |
| DAG2-s1+AG | 1971 [1908, 2066] | -225 (-224) [-326, -105] | 0.00 |

Ruler rule: each seed's ruler difference to NTV0 has a 95 % CI upper bound >= 0: DAG2-s0+AG fail, DAG2-s1+AG fail

**DAG2: not an adopt-candidate** (primary rule fail; ruler rule fail).
