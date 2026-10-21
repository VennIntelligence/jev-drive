# PAI 281, ACC + GAP: FIX (MV-LC without F1 / F2 rows) s0 / s1 vs stored R5LC s0 / s1 (MV-LC) and NTV0-L18-s0

Scenes every read scored without a driver RPC error: 278 (of 281 per read). Zero classes (pai_fix.py fclass): overshoot / route-turn / offroad / AF-front / side / zero-other; partials (0 < score < 0.98): rear-hit / slow (creep + no-launch + slow) / hit; launch = launched / launch scenes (handoff speed < 1 m/s, log moves > 5 m).

| read | n | mean | zeros | z_overshoot | z_route-turn | z_offroad | z_AF-front | z_side | z_zero-other | p_rear-hit | p_slow | p_hit | launch |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 278 | 0.742 | 59 | 23 | 15 | 12 | 3 | 0 | 6 | 26 | 11 | 0 | 33/37 |
| R5LC-s0-ag | 278 | 0.745 | 62 | 21 | 3 | 21 | 6 | 1 | 10 | 18 | 7 | 0 | 34/37 |
| R5LC-s1-ag | 278 | 0.724 | 68 | 22 | 4 | 24 | 6 | 1 | 11 | 22 | 6 | 0 | 35/37 |
| FIX-s0-ag | 278 | 0.712 | 70 | 23 | 4 | 21 | 3 | 1 | 18 | 23 | 5 | 0 | 34/37 |
| FIX-s1-ag | 278 | 0.738 | 64 | 24 | 3 | 20 | 5 | 1 | 11 | 20 | 5 | 0 | 35/37 |

## FIX-s0-ag - R5LC-s0-ag

- mean -0.034 [-0.068, -0.001]; zeros +8 (new 15, fixed 7)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 62 | +8 [-1, +17] | 15 | 7 |
| overshoot | 23 | 21 | +2 [+0, +5] | 2 | 0 |
| route-turn | 4 | 3 | +1 [-2, +4] | 2 | 1 |
| offroad | 21 | 21 | +0 [-6, +6] | 5 | 5 |
| AF-front | 3 | 6 | -3 [-8, +1] | 1 | 4 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 18 | 10 | +8 [+1, +15] | 10 | 2 |
| rear-hit | 23 | 18 | +5 [-1, +12] | 8 | 3 |
| slow | 5 | 7 | -2 [-6, +2] | 1 | 3 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 34 | +0 [-3, +3] | 1 | 1 |

## FIX-s1-ag - R5LC-s1-ag

- mean +0.014 [-0.019, +0.048]; zeros -4 (new 10, fixed 14)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 64 | 68 | -4 [-14, +6] | 10 | 14 |
| overshoot | 24 | 22 | +2 [+0, +5] | 2 | 0 |
| route-turn | 3 | 4 | -1 [-3, +0] | 0 | 1 |
| offroad | 20 | 24 | -4 [-11, +3] | 5 | 9 |
| AF-front | 5 | 6 | -1 [-5, +3] | 2 | 3 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 11 | 11 | +0 [-6, +6] | 4 | 4 |
| rear-hit | 20 | 22 | -2 [-7, +3] | 2 | 4 |
| slow | 5 | 6 | -1 [-5, +2] | 1 | 2 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 35 | +0 [+0, +0] | 0 | 0 |

## FIX-s0-ag - NTV0-ag

- mean -0.031 [-0.078, +0.018]; zeros +11 (new 27, fixed 16)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 70 | 59 | +11 [-2, +24] | 27 | 16 |
| overshoot | 23 | 23 | +0 [-4, +4] | 2 | 2 |
| route-turn | 4 | 15 | -11 [-18, -4] | 1 | 12 |
| offroad | 21 | 12 | +9 [+2, +17] | 12 | 3 |
| AF-front | 3 | 3 | +0 [-4, +4] | 2 | 2 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 18 | 6 | +12 [+4, +20] | 14 | 2 |
| rear-hit | 23 | 26 | -3 [-12, +6] | 9 | 12 |
| slow | 5 | 11 | -6 [-12, -1] | 1 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-3, +5] | 3 | 2 |

## FIX-s1-ag - NTV0-ag

- mean -0.005 [-0.053, +0.042]; zeros +5 (new 24, fixed 19)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 64 | 59 | +5 [-8, +18] | 24 | 19 |
| overshoot | 24 | 23 | +1 [-3, +5] | 3 | 2 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 20 | 12 | +8 [+1, +16] | 11 | 3 |
| AF-front | 5 | 3 | +2 [-2, +6] | 3 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 11 | 6 | +5 [-1, +11] | 7 | 2 |
| rear-hit | 20 | 26 | -6 [-15, +3] | 8 | 14 |
| slow | 5 | 11 | -6 [-12, -1] | 1 | 7 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## R5LC-s0-ag - NTV0-ag

- mean +0.003 [-0.044, +0.048]; zeros +3 (new 22, fixed 19)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 62 | 59 | +3 [-9, +16] | 22 | 19 |
| overshoot | 21 | 23 | -2 [-7, +3] | 2 | 4 |
| route-turn | 3 | 15 | -12 [-20, -5] | 1 | 13 |
| offroad | 21 | 12 | +9 [+2, +17] | 12 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 10 | 6 | +4 [-1, +10] | 6 | 2 |
| rear-hit | 18 | 26 | -8 [-16, +0] | 5 | 13 |
| slow | 7 | 11 | -4 [-11, +3] | 4 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 34 | 33 | +1 [-2, +4] | 2 | 1 |

## R5LC-s1-ag - NTV0-ag

- mean -0.019 [-0.067, +0.029]; zeros +9 (new 27, fixed 18)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 68 | 59 | +9 [-4, +22] | 27 | 18 |
| overshoot | 22 | 23 | -1 [-5, +3] | 2 | 3 |
| route-turn | 4 | 15 | -11 [-18, -4] | 1 | 12 |
| offroad | 24 | 12 | +12 [+4, +21] | 15 | 3 |
| AF-front | 6 | 3 | +3 [-1, +8] | 4 | 1 |
| side | 1 | 0 | +1 [+0, +3] | 1 | 0 |
| zero-other | 11 | 6 | +5 [+0, +11] | 7 | 2 |
| rear-hit | 22 | 26 | -4 [-13, +5] | 9 | 13 |
| slow | 6 | 11 | -5 [-12, +1] | 3 | 8 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 33 | +2 [-2, +6] | 3 | 1 |

## FIX-s1-ag - FIX-s0-ag

- mean +0.026 [-0.003, +0.056]; zeros -6 (new 6, fixed 12)

| class | A | B | A - B (count, 95 % CI) | new | fixed |
|:--|--:|--:|--:|--:|--:|
| zeros | 64 | 70 | -6 [-14, +2] | 6 | 12 |
| overshoot | 24 | 23 | +1 [+0, +3] | 1 | 0 |
| route-turn | 3 | 4 | -1 [-4, +2] | 1 | 2 |
| offroad | 20 | 21 | -1 [-6, +4] | 3 | 4 |
| AF-front | 5 | 3 | +2 [+0, +5] | 2 | 0 |
| side | 1 | 1 | +0 [-3, +3] | 1 | 1 |
| zero-other | 11 | 18 | -7 [-13, -2] | 1 | 8 |
| rear-hit | 20 | 23 | -3 [-8, +1] | 1 | 4 |
| slow | 5 | 5 | +0 [+0, +0] | 0 | 0 |
| hit | 0 | 0 | +0 [+0, +0] | 0 | 0 |
| launched (of launch scenes) | 35 | 34 | +1 [+0, +3] | 1 | 0 |

Paired bootstrap: jevdrive.stats.paired (10 000 draws, seed 0, scenes as units; each PAI scene is its own log).

## New zeros / partials on the 182 of 278 common scenes NTV0 scores 1.0

| read | new zeros | new partials | by class (zeros + partials) |
|:--|--:|--:|:--|
| R5LC-s0-ag | 19 | 8 | offroad 8, zero-other 5, slow 4, AF-front 3, rear-hit 3, overshoot 2, ok 1, route-turn 1 |
| R5LC-s1-ag | 23 | 10 | offroad 11, zero-other 6, rear-hit 6, slow 3, overshoot 2, AF-front 2, ok 1, route-turn 1, side 1 |
| FIX-s0-ag | 24 | 8 | zero-other 12, offroad 8, rear-hit 6, overshoot 2, slow 1, ok 1, side 1, AF-front 1 |
| FIX-s1-ag | 22 | 6 | offroad 9, zero-other 6, rear-hit 5, overshoot 3, AF-front 2, slow 1, route-turn 1, side 1 |

## Local PAI PCS ruler (d392 pool / anchors, out-of-pool read, paired vs NTV0-s0+AG on the same bootstrap draws)

NTV0-s0+AG 2195 [2138, 2299].

| read | PCS [95 % CI] | - NTV0 median (point) [95 % CI] | P(> 0) |
|:--|--:|--:|--:|
| R5LC-s0+AG | 1931 [1859, 1992] | -266 (-264) [-366, -189] | 0.00 |
| R5LC-s1+AG | 1905 [1823, 1959] | -285 (-291) [-394, -208] | 0.00 |
| R6FIX-s0+AG | 1916 [1825, 1964] | -280 (-279) [-369, -213] | 0.00 |
| R6FIX-s1+AG | 1926 [1838, 1985] | -265 (-270) [-357, -182] | 0.00 |

Reference points (d429): R2C-s0+AG -29 [-95, +19], R3MV-s0+AG -276 [-375, -217]. Full ruler report: pcs_fix.md.

## Registered rule (plans/2026-10-21-fix-rows.md)

adopt-candidate = both FIX seeds: mean - NTV0 >= 0 (paired point estimate), offroad + zero-other zeros <= NTV0's + 3 = 21, local PCS ruler difference to NTV0 with a 95 % CI upper bound >= 0.

| arm | - NTV0 mean [95 % CI] | >= NTV0 | offroad + zero-other | <= cap | ruler - NTV0 [95 % CI] | CI upper >= 0 |
|:--|--:|:-:|--:|:-:|--:|:-:|
| FIX-s0-ag | -0.031 [-0.078, +0.018] | no | 39 (offroad 21, zero-other 18) | no | -280 [-369, -213] | no |
| FIX-s1-ag | -0.005 [-0.053, +0.042] | no | 31 (offroad 20, zero-other 11) | no | -265 [-357, -182] | no |

**FIX: not an adopt-candidate.**

## Summary (278-scene common set; new = on the scenes NTV0 scores 1.0)

| read | mean | - R5LC same seed [95 % CI] | offroad | zero-other | route-turn | rear-hit | slow | launch | new zeros / partials | ruler PCS - NTV0 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-ag | 0.742 |  | 12 | 6 | 15 | 26 | 11 | 33/37 | 0 / 0 | ref |
| R5LC-s0-ag | 0.745 |  | 21 | 10 | 3 | 18 | 7 | 34/37 | 19 / 8 | -266 [-366, -189] |
| R5LC-s1-ag | 0.724 |  | 24 | 11 | 4 | 22 | 6 | 35/37 | 23 / 10 | -285 [-394, -208] |
| FIX-s0-ag | 0.712 | -0.034 [-0.068, -0.001] | 21 | 18 | 4 | 23 | 5 | 34/37 | 24 / 8 | -280 [-369, -213] |
| FIX-s1-ag | 0.738 | +0.014 [-0.019, +0.048] | 20 | 11 | 3 | 20 | 5 | 35/37 | 22 / 6 | -265 [-357, -182] |

## The 9 stable OFFROAD-MV scenes (offroad_mv.md §1): score / fclass per read

| scene | mechanism | NTV0-ag | R5LC-s0-ag | R5LC-s1-ag | FIX-s0-ag | FIX-s1-ag | FIX - R5LC s0 | FIX - R5LC s1 |
|:--|:--|:-:|:-:|:-:|:-:|:-:|--:|--:|
| 6e7b42a3 | over-speed drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| dc401449 | slow-down drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 AF-front | 0.00 AF-front | +0.00 | +0.00 |
| a2b415d5 | straight: lateral offset | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| 596af0c7 | curve: cut inside | 0.61 rear-hit | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| 94c1d722 | turn: cut inside | 0.96 slow | 0.00 offroad | 0.00 offroad | 0.90 slow | 0.91 slow | +0.90 | +0.91 |
| e1c6ae9f | curve (fork): cut inside | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| ea468790 | turn: cut inside | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| 0f4b8753 | slow-down drift | 1.00 ok | 0.00 offroad | 0.00 offroad | 0.00 offroad | 0.00 offroad | +0.00 | +0.00 |
| 7750265d | straight: lateral offset | 0.00 overshoot | 0.00 offroad | 0.00 overshoot | 0.00 overshoot | 0.00 overshoot | +0.00 | +0.00 |

Offroad among the 9: NTV0-ag 0, R5LC-s0-ag 9, R5LC-s1-ag 8, FIX-s0-ag 6, FIX-s1-ag 6. Score sum FIX - R5LC: s0 +0.90, s1 +0.91.
