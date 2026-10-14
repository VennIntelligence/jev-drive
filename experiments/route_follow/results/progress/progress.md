# PROGRESS reading A: near-route coverage before the failure (rf_progress.py)

Prereg `plans/2026-10-15-progress-lane-readings-prereg.md`. Turn zeros of RSG-L18 (PAI 281 s0, held 460 s0 / s1); covered = >= 50 % of the memory's 0-20 m offsets valid at t_f - 3 s (primary) / - 1 s. T = over-turn class or shortcut flag. RR = P(T | uncovered) / P(T | covered), scene-clustered bootstrap 95 % CI (B = 10 000).

Gates: G-A1 heading rule = taxonomy corridor class 40 / 40; G-A2 served memory points to the rebuilt route median 0.00 m (p90 0.02); t_f vs taxonomy max |diff| 0.00 s; pai rmem missing 0 / 281; held_s0 rmem missing 0 / 460; held_s1 rmem missing 0 / 460

**Claim A (pooled, t_f - 3 s): RR 0.72 [0.22, 2.29] (inf 0 %) -> unsupported** (T 12 / 53 uncovered vs 6 / 19 covered).

Amendment 2 (geometric coverage: route stations s_e + 0..18 m inside the span of the valid memory points projected on the route): **RR 0.67 [0.20, 2.09] (inf 0 %) -> unsupported** (T 12 / 54 uncovered vs 6 / 18 covered).

| set | turn zeros | RR t_f - 3 s (T unc / n vs T cov / n) | verdict | RR t_f - 1 s | geometric RR t_f - 3 s | verdict | geometric t_f - 1 s | shortcut only (- 3 s) | over-turn only (- 3 s) |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| pooled | 72 | 0.72 [0.22, 2.29] (inf 0 %) (12/53 vs 6/19) | unsupported | 1.27 [0.39, 4.29] (inf 0 %) (12/44 vs 6/28) | 0.67 [0.20, 2.09] (inf 0 %) (12/54 vs 6/18) | unsupported | 1.00 [0.31, 3.34] (inf 0 %) (12/48 vs 6/24) | 0.36 [0.00, nan] (inf 4 %) | 1.08 [0.24, nan] (inf 5 %) |
| pai | 40 | 0.00 [0.00, 0.00] (0/21 vs 6/19) | unsupported | 0.00 [0.00, 0.00] (0/14 vs 6/26) | 0.00 [0.00, 0.00] (0/22 vs 6/18) | unsupported | 0.00 [0.00, 0.00] (0/16 vs 6/24) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| held_s0 | 16 | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | inconclusive (a bucket < 5) | inf [nan, nan] (inf 100 %) (6/15 vs 0/1) | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | inconclusive (a bucket < 5) | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | nan [nan, nan] (inf nan %) | nan [nan, nan] (inf nan %) |
| held_s1 | 16 | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | inconclusive (a bucket < 5) | inf [nan, nan] (inf 100 %) (6/15 vs 0/1) | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | inconclusive (a bucket < 5) | nan [nan, nan] (inf nan %) (6/16 vs 0/0) | nan [nan, nan] (inf nan %) | nan [nan, nan] (inf nan %) |
| held | 32 | nan [nan, nan] (inf nan %) (12/32 vs 0/0) | inconclusive (a bucket < 5) | inf [nan, nan] (inf 100 %) (12/30 vs 0/2) | nan [nan, nan] (inf nan %) (12/32 vs 0/0) | inconclusive (a bucket < 5) | nan [nan, nan] (inf nan %) (12/32 vs 0/0) | nan [nan, nan] (inf nan %) | nan [nan, nan] (inf nan %) |

Counts per class x bucket (t_f - 3 s; pooled), shortcut flags inside each class, and the medians of the memory read at t_f - 3 s:

| class | n | covered | uncovered | geo covered / uncovered | shortcut | cov 0-20 | cov 20-40 | geo cov 0-20 / 20-40 | geo nearest covered (m) | first valid offset (m) | first lateral (m) | first bearing (deg) | first >= 10 m: offset / bearing | driven since start (m) | v at t_f |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| lane_change | 1 | 1 | 0 | 1 / 0 | 1 | 1.00 | 1.00 | 1.00 / 1.00 | 0 | 0 | -1.8 | -90 | 10 / -11 | 50 | 11.6 |
| late_wide | 34 | 8 | 26 | 8 / 26 | 2 | 0.00 | 0.65 | 0.00 / 0.30 | 34 | 26 | +12.4 | +39 | 26 / +23 | 10 | 4.7 |
| other | 3 | 2 | 1 | 2 / 1 | 1 | 1.00 | 1.00 | 1.00 / 1.00 | 0 | 0 | -0.2 | -23 | 10 / -17 | 125 | 4.0 |
| over_turn | 12 | 3 | 9 | 3 / 9 | 0 | 0.00 | 0.55 | 0.00 / 0.00 | 40 | 29 | +15.2 | +30 | 29 / +24 | 5 | 6.2 |
| route_end | 2 | 0 | 2 | 0 / 2 | 0 | 0.00 | 0.30 | 0.00 / 0.10 | 37 | 34 | +24.5 | +45 | 34 / +45 | 9 | 5.1 |
| straight | 19 | 5 | 14 | 4 / 15 | 2 | 0.00 | 0.90 | 0.00 / 0.40 | 31 | 20 | -0.4 | -1 | 20 / -1 | 14 | 5.8 |
| wrong_exit | 1 | 0 | 1 | 0 / 1 | 0 | 0.00 | 0.00 | 0.00 / 0.00 | 40 | 40 | +0.3 | +1 | 40 / +1 | 0 | 34.4 |

Completed strict turns: 35 (33 with the entry crossed): uncovered at the entry 97 %, 3 s before it 97 %; turn zeros at t_f - 3 s 74 % (zeros - completed -23 pp [-36, -11]); geometric: completed at the entry 97 %, 3 s before 97 %, turn zeros 75 %. Per set (slot): pai n 5, entry 80 %, - 3 s 80 %; held_s0 n 15, entry 100 %, - 3 s 100 %; held_s1 n 13, entry 100 %, - 3 s 100 %.

Slot labelling (Amendment 2): where the near route is missing, the first served slot says the route starts 4.5 m (median; p90 15.7) nearer than it does along the route (26 / 53 turn zeros >= 5 m); completed turns at the entry 4.5 m (12 / 32 >= 5 m).

Road legs at t_f - 3 s (72 turn zeros): far segment (s + 60 m) on another leg (>= 45 deg) 57, of the 18 T zeros 15; two opposite turns within 80 m 3; a turn entry inside the first 40 m (the message gap) 40. Zeros without a turn (not in the read): 23.

Per turn zero (t_f - 3 s):

| set | scene | primary | class | shortcut (ego / log dev deg) | cov 0-20 / 20-40 (geo) | first valid (m, lat, deg) | driven (m) | far leg dpsi | double | gap turn |
|:--|:--|:--|:--|:--|:--|:--|--:|--:|:--|:--|
| held_s0 | 5b61b9072cad5dbc | off_curve_outside | late_wide | False (+18 / +2) | 0.0 / 0.6 (0.0 / 0.6) | 28, -13.0, -28 | 15 | +26 |  |  |
| held_s0 | e6056b57c0515735 | cor_turn_under | late_wide | False (+14 / +8) | 0.2 / 1.0 (0.0 / 0.2) | 16, +11.4, +23 | 7 | +88 |  | yes |
| held_s0 | a3c4e4bf10d65ddb | off_junc_outside | late_wide | False (+2 / +3) | 0.0 / 0.2 (0.0 / 0.0) | 36, +36.1, +77 | 3 | +81 |  | yes |
| held_s0 | 4faa4706a50958e2 | cor_turn_under | late_wide | False (+19 / +1) | 0.0 / 0.4 (0.0 / 0.3) | 32, +28.7, +61 | 11 | +109 |  |  |
| held_s0 | d7b1349fb4775cc1 | cor_turn_under | late_wide | False (+23 / +16) | 0.0 / 0.1 (0.0 / 0.0) | 38, +32.3, +55 | 3 | +62 |  | yes |
| held_s0 | e14711ef4d2f5000 | off_junc_outside | late_wide | False (+19 / +0) | 0.0 / 0.2 (0.0 / 0.0) | 36, +35.3, +74 | 4 | +92 |  |  |
| held_s0 | 3b4031def0f45d96 | off_junc_outside | late_wide | None (+nan / +nan) | 0.0 / 0.0 (0.0 / 0.0) | nan, +nan, +nan | 1 | -187 |  | yes |
| held_s0 | b5155b952c645ce0 | off_junc_inside | over_turn | False (+15 / +3) | 0.0 / 0.3 (0.0 / 0.0) | 34, +22.1, +35 | 4 | +67 |  | yes |
| held_s0 | 02015675e4585611 | off_junc_inside | over_turn | False (-17 / -4) | 0.0 / 0.4 (0.0 / 0.0) | 32, -33.1, -70 | 0 | -92 |  | yes |
| held_s0 | c687799dac8d5aa2 | off_junc_inside | over_turn | False (+7 / +5) | 0.0 / 0.7 (0.0 / 0.0) | 26, +28.6, +70 | 5 | +108 |  | yes |
| held_s0 | a3dd6d539acc5c9b | off_curve_inside | over_turn | False (+10 / +0) | 0.0 / 0.7 (0.0 / 0.3) | 26, +11.5, +22 | 10 | +64 |  | yes |
| held_s0 | be6f90f06bc35be3 | off_curve_inside | over_turn | False (+1 / +5) | 0.0 / 0.0 (0.0 / 0.0) | 42, +19.0, +27 | 0 | +50 |  |  |
| held_s0 | 47e0ff3bc36a5f69 | coll_veh_oncoming | route_end | False (-4 / +1) | 0.0 / 0.3 (0.0 / 0.1) | 34, +24.4, +44 | 9 | +67 |  |  |
| held_s0 | f9ed38d9ddfa531e | off_junc_straight | straight | False (-1 / -9) | 0.0 / 0.8 (0.0 / 0.0) | 24, -0.8, -1 | 0 | -22 | yes | yes |
| held_s0 | b628faa4178d5eaf | cor_straight_at_turn | straight | False (+33 / +0) | 0.0 / 0.5 (0.0 / 0.3) | 30, +25.2, +54 | 9 | +58 |  | yes |
| held_s0 | 20728d3e677b593d | off_junc_straight | straight | True (-40 / -7) | 0.0 / 0.5 (0.0 / 0.2) | 30, -1.9, -3 | 8 | -51 |  |  |
| held_s1 | ae878998fecb50a5 | cor_turn_under | late_wide | False (+10 / +4) | 0.0 / 0.7 (0.0 / 0.2) | 26, +29.4, +82 | 7 | +100 |  |  |
| held_s1 | e6056b57c0515735 | coll_veh_oncoming | late_wide | False (+3 / +20) | 0.2 / 1.0 (0.0 / 0.1) | 16, +14.3, +26 | 3 | +88 |  | yes |
| held_s1 | a3dd6d539acc5c9b | cor_turn_under | late_wide | True (+25 / +8) | 0.0 / 0.6 (0.0 / 0.0) | 28, +13.4, +23 | 4 | +67 |  | yes |
| held_s1 | 4faa4706a50958e2 | cor_turn_under | late_wide | False (+24 / +1) | 0.0 / 0.4 (0.0 / 0.2) | 32, +28.8, +60 | 10 | +101 |  |  |
| held_s1 | e14711ef4d2f5000 | off_junc_outside | late_wide | False (+18 / +0) | 0.0 / 0.2 (0.0 / 0.0) | 36, +35.4, +75 | 4 | +92 |  |  |
| held_s1 | 0fa603b5789956f1 | cor_turn_under | late_wide | False (+9 / +1) | 0.0 / 0.4 (0.0 / 0.2) | 32, +29.9, +63 | 7 | +70 |  |  |
| held_s1 | 3b4031def0f45d96 | off_junc_outside | late_wide | None (+nan / +nan) | 0.0 / 0.0 (0.0 / 0.0) | nan, +nan, +nan | 1 | -187 |  | yes |
| held_s1 | b5155b952c645ce0 | off_junc_inside | over_turn | False (+13 / +3) | 0.0 / 0.3 (0.0 / 0.0) | 34, +21.7, +35 | 5 | +67 |  | yes |
| held_s1 | 02015675e4585611 | off_junc_inside | over_turn | False (-12 / -1) | 0.0 / 0.4 (0.0 / 0.0) | 32, -32.6, -73 | 2 | -91 |  | yes |
| held_s1 | c687799dac8d5aa2 | off_junc_inside | over_turn | False (+6 / +5) | 0.0 / 0.7 (0.0 / 0.0) | 26, +28.8, +71 | 5 | +108 |  | yes |
| held_s1 | be6f90f06bc35be3 | off_junc_inside | over_turn | False (+3 / +0) | 0.0 / 0.0 (0.0 / 0.0) | 42, +23.2, +33 | 1 | +50 |  |  |
| held_s1 | 47e0ff3bc36a5f69 | coll_veh_oncoming | route_end | False (-4 / +1) | 0.0 / 0.3 (0.0 / 0.1) | 34, +24.6, +45 | 9 | +67 |  |  |
| held_s1 | d7b1349fb4775cc1 | cor_straight_at_turn | straight | False (+25 / +13) | 0.0 / 0.1 (0.0 / 0.0) | 38, +31.7, +53 | 2 | +62 |  | yes |
| held_s1 | f9ed38d9ddfa531e | off_junc_straight | straight | False (-1 / -9) | 0.0 / 0.8 (0.0 / 0.0) | 24, -0.8, -1 | 0 | -22 | yes | yes |
| held_s1 | b628faa4178d5eaf | cor_straight_at_turn | straight | False (+31 / +0) | 0.0 / 0.5 (0.0 / 0.3) | 30, +25.7, +56 | 10 | +56 |  | yes |
| held_s1 | 20728d3e677b593d | off_junc_straight | straight | True (-30 / -7) | 0.0 / 0.5 (0.0 / 0.2) | 30, -2.5, -4 | 8 | -51 |  |  |
| pai | 1e8e939e | coll_cyclist | lane_change | True (-28 / -15) | 1.0 / 1.0 (1.0 / 1.0) | 0, -1.8, -90 | 50 | -40 |  |  |
| pai | 00fabf60 | cor_turn_under | late_wide | False (+24 / +1) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.6, -99 | 92 | +96 |  |  |
| pai | 1a69f757 | coll_veh_oncoming | late_wide | False (-4 / -3) | 0.3 / 1.0 (0.3 / 1.0) | 14, -4.5, -18 | 28 | -45 |  |  |
| pai | 22fa5c92 | coll_veh_oncoming | late_wide | False (+3 / +9) | 0.0 / 0.5 (0.0 / 0.0) | 30, +28.7, +60 | 4 | +89 |  | yes |
| pai | 2a882d89 | off_junc_outside | late_wide | False (+1 / -1) | 1.0 / 1.0 (1.0 / 1.0) | 0, +0.0, +92 | 75 | +34 |  |  |
| pai | 33c8976b | off_junc_outside | late_wide | False (+13 / +13) | 0.0 / 0.4 (0.0 / 0.0) | 32, +34.6, +79 | 3 | +101 |  | yes |
| pai | 37d660b1 | cor_turn_under | late_wide | False (-22 / -7) | 0.1 / 1.0 (0.0 / 0.7) | 18, -17.3, -52 | 17 | -83 |  | yes |
| pai | 3ce13eed | cor_turn_under | late_wide | False (-17 / -6) | 0.0 / 0.8 (0.0 / 0.6) | 24, -25.4, -86 | 16 | -91 |  | yes |
| pai | 68e53160 | cor_turn_under | late_wide | False (+18 / +3) | 0.0 / 1.0 (0.0 / 0.6) | 20, +19.0, +51 | 16 | +90 |  |  |
| pai | 7381c9fa | off_junc_outside | late_wide | False (+25 / +4) | 0.0 / 0.2 (0.0 / 0.0) | 36, +34.0, +68 | 1 | +31 |  |  |
| pai | 94c1d722 | off_junc_outside | late_wide | False (-15 / -6) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.4, -93 | 81 | -68 |  |  |
| pai | 98f88a4e | off_curve_outside | late_wide | False (-21 / -1) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.9, -105 | 60 | -22 |  |  |
| pai | 9f393c8a | coll_veh_side | late_wide | False (-18 / -1) | 0.0 / 0.6 (0.0 / 0.4) | 28, -19.3, -40 | 11 | -31 | yes |  |
| pai | a1b166cc | cor_turn_under | late_wide | False (+7 / +4) | 0.0 / 0.7 (0.0 / 0.5) | 26, +24.7, +60 | 14 | +73 |  |  |
| pai | b680a689 | cor_turn_under | late_wide | True (+24 / +11) | 1.0 / 1.0 (1.0 / 1.0) | 0, -1.9, -93 | 53 | +91 |  | yes |
| pai | cdf95fd1 | cor_turn_under | late_wide | False (+7 / +16) | 1.0 / 1.0 (1.0 / 1.0) | 0, +0.7, +89 | 184 | +98 |  | yes |
| pai | ce161092 | off_curve_outside | late_wide | False (+9 / +0) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.2, -93 | 52 | +82 |  | yes |
| pai | ddc3e8df | cor_turn_under | late_wide | False (-11 / -6) | 0.0 / 0.3 (0.0 / 0.3) | 34, -22.4, -41 | 9 | -32 |  |  |
| pai | e37aa159 | cor_turn_under | late_wide | False (+18 / +0) | 0.5 / 1.0 (0.5 / 1.0) | 10, +1.9, +11 | 31 | +87 |  | yes |
| pai | fd01e9ab | cor_turn_under | late_wide | False (+30 / +17) | 0.0 / 0.5 (0.0 / 0.1) | 30, +29.3, +67 | 5 | +95 |  | yes |
| pai | fda39805 | cor_turn_under | late_wide | False (-5 / -14) | 0.0 / 1.0 (0.0 / 1.0) | 20, -15.7, -51 | 25 | -37 |  |  |
| pai | 382a7247 | coll_veh_oncoming | other | False (-2 / -0) | 1.0 / 1.0 (1.0 / 1.0) | 0, +0.4, +91 | 125 | -44 |  |  |
| pai | ba62e10d | cor_drift | other | True (+35 / +1) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.2, -96 | 158 | +82 |  | yes |
| pai | de9db5dc | coll_veh_side | other | False (-6 / -1) | 0.3 / 1.0 (0.0 / 0.0) | 14, -14.0, -23 | 3 | -94 |  | yes |
| pai | 2466266e | off_curve_inside | over_turn | False (+12 / +2) | 1.0 / 1.0 (1.0 / 1.0) | 0, +2.7, +103 | 67 | +34 |  |  |
| pai | 300465f7 | off_junc_inside | over_turn | False (-14 / -0) | 1.0 / 1.0 (1.0 / 1.0) | 0, -2.4, -90 | 45 | +108 |  |  |
| pai | d8bfaaf2 | off_curve_inside | over_turn | False (-2 / -0) | 0.9 / 1.0 (0.9 / 1.0) | 2, -0.9, -24 | 39 | -31 |  |  |
| pai | 0a228e44 | cor_straight_at_turn | straight | False (-24 / -6) | 1.0 / 1.0 (1.0 / 1.0) | 0, +0.1, +91 | 54 | -72 |  |  |
| pai | 2f227f2c | cor_straight_at_turn | straight | False (-18 / -20) | 0.0 / 1.0 (0.0 / 0.2) | 20, -9.8, -17 | 7 | -67 |  | yes |
| pai | 302c5c99 | cor_straight_at_turn | straight | False (+32 / +43) | 1.0 / 0.9 (0.0 / 0.2) | 0, +10.7, +92 | 20 | +180 |  | yes |
| pai | 3232fdb7 | cor_straight_at_turn | straight | False (-7 / -6) | 1.0 / 1.0 (1.0 / 1.0) | 0, -1.9, -90 | 80 | -90 |  | yes |
| pai | 403d558a | cor_straight_at_turn | straight | False (-37 / -0) | 0.0 / 0.9 (0.0 / 0.4) | 22, -3.9, -7 | 11 | -101 |  | yes |
| pai | 52003cee | cor_straight_at_turn | straight | False (-40 / -4) | 1.0 / 1.0 (1.0 / 1.0) | 0, -0.3, -90 | 74 | -90 |  | yes |
| pai | a174faed | cor_straight_at_turn | straight | False (-41 / -4) | 0.4 / 1.0 (0.4 / 1.0) | 12, -1.3, -6 | 30 | -88 |  | yes |
| pai | a5599793 | off_junc_straight | straight | False (+10 / +2) | 0.0 / 0.6 (0.0 / 0.5) | 28, -0.5, -1 | 14 | +1 |  |  |
| pai | bd97de34 | cor_straight_at_turn | straight | False (+20 / +1) | 1.0 / 1.0 (1.0 / 1.0) | 0, +3.1, +89 | 147 | +93 |  | yes |
| pai | e5e47fd2 | cor_straight_at_turn | straight | False (+27 / +5) | 0.0 / 1.0 (0.0 / 1.0) | 20, +0.2, +1 | 22 | +103 |  | yes |
| pai | eaba3f6a | cor_straight_at_turn | straight | False (-30 / -10) | 0.1 / 1.0 (0.1 / 1.0) | 18, -0.4, -1 | 24 | -90 |  | yes |
| pai | f9e361aa | cor_straight_at_turn | straight | False (-30 / -1) | 0.3 / 1.0 (0.3 / 1.0) | 14, +0.1, +0 | 29 | -81 |  | yes |
| pai | 32be03e4 | cor_wrong_exit | wrong_exit | False (+57 / +0) | 0.0 / 0.0 (0.0 / 0.0) | 40, +0.3, +1 | 0 | +1 |  |  |
