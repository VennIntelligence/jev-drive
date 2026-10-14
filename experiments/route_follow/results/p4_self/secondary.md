## (a) identity (the policy alone)

completes 9 / 27: 96eb4700, a4be61d6, c6c01d25, 268949d2, 77fdd75c, 96da4f1a, b40f4729, dc1966f4, ea468790

## (b) identity rollout, first failure per scene

`end rel entry` = route station of the ego at the end / zero event minus the strict entry (m): < 0 = before the turn entry.

| scene | side | topo | done | first failure | t (s) | end rel entry (m) | before entry | v_entry | min v inside |
|:--|:--|:--|:--|:--|--:|--:|:--|--:|--:|
| 0d10d2e3 | L | straight_road | no | collision | 8.799999999999999 | 8.59 | - | 3.515 | 1.899 |
| 213dfdac | L | curve_single | no | offroad | 14.5 | 12.15 | - | 3.59 | 3.351 |
| 33c8976b | L | curve_single | no | offroad | 5.0 | 4.42 | - | 2.549 | 2.567 |
| 8a365ac2 | L | no_straight_exit | no | offroad | 7.3 | 9.07 | - | 3.166 | 3.211 |
| 96eb4700 | L | curve_single | yes | - | - | 127.0 | - | 6.387 | 6.397 |
| a4be61d6 | L | straight_exit | yes | - | - | 38.84 | - | 3.678 | 3.695 |
| a4d2f7cf | L | straight_exit | no | end | - | 16.87 | - | 3.837 | 3.828 |
| c6c01d25 | L | curve_single | yes | - | - | 66.19 | - | 3.765 | 3.857 |
| e5e47fd2 | L | straight_road | no | corridor | 5.2 | 5.17 | - | 13.784 | 13.798 |
| 1594e89e | R | straight_road | no | collision | 6.1 | 9.89 | - | 1.621 | 1.584 |
| 1e8e939e | R | curve_single | no | collision | 6.3999999999999995 | -9.11 | **yes** | None | None |
| 232d240e | R | curve_single | no | offroad | 6.699999999999999 | 13.39 | - | 4.234 | 3.484 |
| 268949d2 | R | no_straight_exit | yes | - | - | 98.18 | - | 2.632 | 2.65 |
| 3232fdb7 | R | straight_exit | no | corridor | 17.5 | -4.35 | **yes** | None | None |
| 37d660b1 | R | straight_exit | no | corridor | 7.1 | 8.2 | - | 4.553 | 4.601 |
| 403d558a | R | straight_exit | no | corridor | 4.0 | 4.66 | - | 11.789 | 11.928 |
| 77fdd75c | R | curve_single | yes | - | - | 143.0 | - | 10.38 | 9.647 |
| 96da4f1a | R | curve_single | yes | - | - | 123.0 | - | 4.839 | 4.857 |
| 98694f91 | R | straight_exit | no | corridor | 17.4 | 12.0 | - | 3.379 | 2.846 |
| a174faed | R | straight_exit | no | corridor | 10.6 | 4.08 | - | 6.156 | 5.886 |
| b40f4729 | R | straight_exit | yes | - | - | 121.0 | - | 2.97 | 3.056 |
| d0b9a872 | R | curve_single | no | offroad | 4.2 | 6.37 | - | 3.676 | 3.599 |
| dc1966f4 | R | straight_exit | yes | - | - | 32.04 | - | 3.582 | 2.058 |
| ddc3e8df | R | curve_single | no | corridor | 6.0 | 18.1 | - | 2.578 | 2.617 |
| ea468790 | R | curve_single | yes | - | - | 37.81 | - | 3.216 | 2.88 |
| eaba3f6a | R | straight_exit | no | corridor | 5.5 | 2.77 | - | 12.505 | 12.543 |
| f9e361aa | R | straight_exit | no | corridor | 6.5 | 2.83 | - | 11.356 | 11.38 |

Identity fails before the turn entry: 2 / 27 (not a turn failure). Failing identity by kind: collision before entry: 1; collision in / after turn: 2; corridor before entry: 1; corridor in / after turn: 8; end in / after turn: 1; offroad in / after turn: 5

## (c) completions per cell, per profile, per D (distinct rollouts; a duplicated distance is run once and counted under the largest D)

| cell | rollouts | done | done + shape gate |
|:--|--:|--:|--:|
| g1-raw | 2 | 1 | 1 |
| g1-ve | 83 | 8 | 6 |
| g1-ve08 | 83 | 8 | 6 |
| g1.25-raw | 83 | 22 | 14 |
| g1.25-ve | 83 | 23 | 13 |
| g1.25-ve08 | 83 | 20 | 9 |
| g1.5-raw | 83 | 10 | 3 |
| g1.5-ve | 83 | 6 | 4 |
| g1.5-ve08 | 83 | 8 | 5 |

| D | profile raw (done / n) | ve | ve08 | all |
|--:|--:|--:|--:|--:|
| 40 | 16 / 54 | 17 / 81 | 17 / 81 | 50 / 216 |
| 30 | 2 / 20 | 1 / 30 | 1 / 30 | 4 / 80 |
| 20 | 1 / 20 | 1 / 30 | 1 / 30 | 3 / 80 |
| 10 | 2 / 26 | 1 / 39 | 2 / 39 | 5 / 104 |
| 0 | 12 / 48 | 17 / 69 | 15 / 69 | 44 / 186 |

| profile | rollouts | done | done + gate |
|:--|--:|--:|--:|
| raw | 168 | 33 | 18 |
| ve | 249 | 37 | 23 |
| ve08 | 249 | 36 | 20 |

| gain | rollouts | done | done + gate |
|:--|--:|--:|--:|
| g1 | 168 | 17 | 13 |
| g1.25 | 249 | 65 | 36 |
| g1.5 | 249 | 24 | 12 |

## (d) fallback, rear-ended, crawl, shape gate

- src=log rollouts: 18 (2 distance(s): 1e8e939e D0, 3232fdb7 D0), done 8
- rear-ended (collision_rear): 36 rollouts, of which counted as done 17
- crawl (min speed h..entry < 0.5 v_e): approach 219 / 666 branch rollouts (D0 rollouts start at the entry: 0); among done 32 / 106
- crawl inside [entry, exit]: 57 / 666; among done 6 / 106
- completing branch cells: 106; pass the full shape gate: 61; gate parts failing among the 106: kmax 34, dk 26, alat 9, v_entry 10
- completing cells with realized entry speed <= class v_e + 1: 96
- determinism before h (pre_h_diff > 0): 0 of 648 self rollouts

## (e) distinct handovers per scene

distinct handover decisions per scene: 1: 4 scenes, 2: 10 scenes, 3: 3 scenes, 5: 10 scenes; total 83
scenes with one or two distinct handovers: 14 (a scene with one cannot reach the >= 2 line)
| scene | distinct handovers | h per D |
|:--|--:|:--|
| 0d10d2e3 | 3 | D40:17, D10:29, D0:68 |
| 213dfdac | 5 | D40:19, D30:29, D20:42, D10:77, D0:111 |
| 33c8976b | 2 | D40:17, D0:34 |
| 8a365ac2 | 2 | D40:17, D0:52 |
| 96eb4700 | 1 | D40:17 |
| a4be61d6 | 2 | D40:17, D0:49 |
| a4d2f7cf | 5 | D40:42, D30:62, D20:108, D10:139, D0:165 |
| c6c01d25 | 1 | D40:17 |
| e5e47fd2 | 5 | D40:17, D30:25, D20:33, D10:41, D0:48 |
| 1594e89e | 2 | D40:17, D0:22 |
| 1e8e939e | 5 | D40:43, D30:50, D20:55, D10:64, D0:121(log) |
| 232d240e | 2 | D40:17, D0:26 |
| 268949d2 | 2 | D40:17, D0:62 |
| 3232fdb7 | 5 | D40:64, D30:87, D20:115, D10:149, D0:148(log) |
| 37d660b1 | 3 | D40:17, D10:30, D0:54 |
| 403d558a | 3 | D40:17, D10:26, D0:36 |
| 77fdd75c | 1 | D40:17 |
| 96da4f1a | 2 | D40:17, D0:47 |
| 98694f91 | 5 | D40:17, D30:27, D20:43, D10:90, D0:128 |
| a174faed | 5 | D40:17, D30:44, D20:63, D10:80, D0:97 |
| b40f4729 | 2 | D40:17, D0:34 |
| d0b9a872 | 2 | D40:17, D0:24 |
| dc1966f4 | 5 | D40:19, D30:30, D20:46, D10:101, D0:131 |
| ddc3e8df | 1 | D40:17 |
| ea468790 | 2 | D40:17, D0:96 |
| eaba3f6a | 5 | D40:17, D30:25, D20:36, D10:45, D0:53 |
| f9e361aa | 5 | D40:23, D30:34, D20:44, D10:54, D0:63 |

### scenes counted by the line (>= 2 distinct handovers with a completing, entry-speed-ok cell)

10: 8a365ac2, a4be61d6, e5e47fd2, 232d240e, 268949d2, 37d660b1, 96da4f1a, b40f4729, dc1966f4, ea468790
