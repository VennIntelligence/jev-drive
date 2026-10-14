# LANEOFF reading B: lateral offset before the turn, and required lane changes (rf_laneoff.py)

Prereg `plans/2026-10-15-progress-lane-readings-prereg.md`. Strict turns of RSG-L18 (PAI 281 s0: 36; held 460 s0 / s1: 24 each), completed vs failed (any other stored outcome). Offsets in m (left +); differences failed - completed with scene-clustered bootstrap 95 % CI (B = 10 000).

**Claim B (PAI strict, entry - 20 m): median |lat_route| failed 0.48 vs completed 0.06 m, D_med +0.42 [+0.08, +0.81]; out-of-lane share D_out +0.16 [+0.00, +0.33] (n failed 19, completed 3) -> inconclusive**

| group / point | failed / completed | median abs lat_route failed / completed | D_med | D_out (share) | D abs lat_log | D abs heading err (deg) | D speed (m/s) | D abs lat_mem (n defined) | oncoming lane failed / completed | rule |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_strict|m20 | 19 / 3 | 0.48 / 0.06 | +0.42 [+0.08, +0.81] | +0.16 [+0.00, +0.33] | -0.01 [-0.11, +0.15] | +0.1 [-1.6, +1.9] | +0.93 [-3.25, +4.53] | +0.91 [+0.11, +2.43] (8) | 5 / 0 | inconclusive |
| pai_strict|e0 | 26 / 5 | 0.69 / 0.53 | +0.15 [-1.72, +0.76] | -0.16 [-0.63, +0.10] | +0.87 [-0.36, +1.01] | +2.9 [-3.1, +10.2] | -2.85 [-5.18, +1.94] | +0.39 [+0.14, +1.36] (9) | 7 / 0 | inconclusive |
| held_strict|e0 | 18 / 28 | 0.36 / 0.60 | -0.25 [-0.80, +0.19] | -0.07 [-0.23, +0.00] | +0.05 [-0.10, +0.17] | -2.6 [-4.8, +1.7] | -1.00 [-1.80, -0.06] | - (0) | 4 / 7 | unsupported |
| pooled_strict|m20 | 19 / 3 | 0.48 / 0.06 | +0.42 [+0.08, +0.81] | +0.16 [+0.00, +0.33] | -0.01 [-0.11, +0.15] | +0.1 [-1.6, +1.9] | +0.93 [-3.25, +4.53] | +0.91 [+0.11, +2.43] (8) | 5 / 0 | inconclusive |
| pooled_strict|e0 | 44 / 33 | 0.50 / 0.59 | -0.08 [-0.64, +0.31] | -0.07 [-0.23, +0.05] | +0.19 [+0.02, +0.85] | -0.4 [-3.2, +4.8] | -1.00 [-1.77, -0.03] | +0.39 [+0.14, +1.36] (9) | 11 / 7 | unsupported |
| pai_bend|m20 | 25 / 3 | 0.48 / 0.06 | +0.42 [+0.06, +0.81] | +0.12 [+0.00, +0.26] | +0.11 [-0.10, +0.19] | +1.0 [-0.3, +1.8] | -2.72 [-5.09, +2.51] | +0.90 [+0.11, +1.80] (12) | 7 / 0 | inconclusive |
| pai_bend|e0 | 30 / 9 | 0.57 / 0.75 | -0.17 [-0.69, +0.48] | -0.04 [-0.31, +0.14] | +0.88 [+0.09, +1.00] | +4.8 [-2.5, +9.8] | -3.91 [-5.31, -0.41] | +0.39 [+0.14, +1.41] (12) | 9 / 0 | inconclusive |
| pai_strict_no_short|m20 | 16 / 3 | 0.54 / 0.06 | +0.48 [+0.12, +0.83] | +0.19 [+0.00, +0.40] | -0.01 [-0.14, +0.16] | -0.3 [-1.7, +1.8] | +2.00 [-3.03, +5.25] | +1.87 [+0.86, +10.50] (6) | 4 / 0 | inconclusive |
| pai_strict_no_short|e0 | 24 / 5 | 0.69 / 0.53 | +0.15 [-1.72, +0.73] | -0.16 [-0.63, +0.11] | +0.87 [-0.37, +1.01] | +2.9 [-3.1, +10.2] | -2.41 [-5.13, +2.35] | +0.36 [+0.14, +1.04] (8) | 6 / 0 | inconclusive |
| pooled_strict_no_short|m20 | 16 / 3 | 0.54 / 0.06 | +0.48 [+0.12, +0.83] | +0.19 [+0.00, +0.40] | -0.01 [-0.14, +0.16] | -0.3 [-1.7, +1.8] | +2.00 [-3.03, +5.25] | +1.87 [+0.86, +10.50] (6) | 4 / 0 | inconclusive |
| pooled_strict_no_short|e0 | 26 / 33 | 0.66 / 0.59 | +0.07 [-0.51, +0.59] | -0.05 [-0.22, +0.08] | +0.84 [+0.13, +0.98] | +3.1 [-0.9, +7.7] | -0.93 [-1.87, +1.77] | +0.36 [+0.14, +1.04] (8) | 6 / 7 | inconclusive |

Exploratory (not registered): the offset signed toward the turn's inside (+ = inside, - = the side away from the turn):

| group / point | median inward failed / completed | D (failed - completed) | >= 0.75 m outside failed / completed | >= 1.75 m outside failed / completed |
|:--|:--|:--|:--|:--|
| pai_strict|m20 | -0.48 / +0.06 | -0.55 [-0.94, -0.17] | 7 / 0 | 3 / 0 |
| pai_strict|e0 | -0.16 / +0.53 | -0.70 [-2.57, +0.02] | 8 / 0 | 1 / 0 |
| held_strict|e0 | +0.26 / +0.56 | -0.30 [-0.80, +0.24] | 0 / 2 | 0 / 0 |
| pooled_strict|m20 | -0.48 / +0.06 | -0.55 [-0.94, -0.17] | 7 / 0 | 3 / 0 |
| pooled_strict|e0 | +0.16 / +0.53 | -0.37 [-0.95, +0.03] | 8 / 2 | 1 / 0 |
| pai_bend|m20 | -0.48 / +0.02 | -0.50 [-0.89, +0.11] | 9 / 0 | 3 / 0 |
| pai_bend|e0 | -0.25 / +0.75 | -1.00 [-1.58, -0.23] | 9 / 0 | 2 / 0 |
| pai_strict_no_short|m20 | -0.54 / +0.06 | -0.61 [-0.96, -0.21] | 6 / 0 | 3 / 0 |
| pai_strict_no_short|e0 | -0.16 / +0.53 | -0.70 [-2.57, +0.02] | 8 / 0 | 1 / 0 |
| pooled_strict_no_short|m20 | -0.54 / +0.06 | -0.61 [-0.96, -0.21] | 6 / 0 | 3 / 0 |
| pooled_strict_no_short|e0 | -0.05 / +0.53 | -0.58 [-1.38, +0.08] | 8 / 2 | 1 / 0 |

Where the points are undefined (counts by reason x group):

- pai_strict|m20: not_reached|failed 1, ok|completed 3, ok|failed 19, started_past|completed 2, started_past|failed 11
- pai_strict|e0: not_reached|failed 5, ok|completed 5, ok|failed 26
- held_strict|m20: started_past|completed 30, started_past|failed 18
- held_strict|e0: ok|completed 28, ok|failed 18, started_past|completed 2
- pooled_strict|m20: not_reached|failed 1, ok|completed 3, ok|failed 19, started_past|completed 32, started_past|failed 29
- pooled_strict|e0: not_reached|failed 5, ok|completed 33, ok|failed 44, started_past|completed 2
- pai_bend|m20: not_reached|failed 2, ok|completed 3, ok|failed 25, started_past|completed 6, started_past|failed 10
- pai_bend|e0: not_reached|failed 7, ok|completed 9, ok|failed 30
- pai_strict_no_short|m20: ok|completed 3, ok|failed 16, started_past|completed 2, started_past|failed 11
- pai_strict_no_short|e0: not_reached|failed 3, ok|completed 5, ok|failed 24
- pooled_strict_no_short|m20: ok|completed 3, ok|failed 16, started_past|completed 32, started_past|failed 13
- pooled_strict_no_short|e0: not_reached|failed 3, ok|completed 33, ok|failed 26, started_past|completed 2

PAI strict turns (lat_route / lat_log / heading error / speed / lat_mem at entry - 20 m | at entry; blank = undefined):

| scene | outcome | entry - 20 m | entry |
|:--|:--|:--|:--|
| 09528162 | off | +1.9 / +0.0 / -2.0 / +8.3 /  | +0.3 / +1.4 / +10.7 / +2.6 /  |
| 132ae6cb | completed | +0.1 / -0.1 / -2.4 / +3.6 /  | +0.4 / +0.0 / +6.0 / +6.7 /  |
| 1e8e939e | event | +2.5 / +1.0 / -8.5 / +13.7 / +2.5 |  /  /  /  /  |
| 22fa5c92 | event |  /  /  /  /  | +1.6 / +0.3 / +10.8 / +2.6 /  |
| 268949d2 | off |  /  /  /  /  | -0.6 / -0.1 / +1.4 / +2.8 /  |
| 2f227f2c | event | +0.4 / +0.2 / +2.0 / +6.0 /  |  /  /  /  /  |
| 302c5c99 | event | -0.2 / +0.0 / -0.8 / +5.0 / +10.5 | -1.6 / -2.6 / -25.4 / +6.7 /  |
| 3232fdb7 | event | +1.9 / +2.8 / +1.1 / +3.7 / +1.9 |  /  /  /  /  |
| 33c8976b | event |  /  /  /  /  | -1.1 / -0.3 / +2.2 / +2.6 /  |
| 37d660b1 | event | -0.2 / -0.0 / -2.3 / +3.6 /  | -0.1 / +0.9 / +8.5 / +4.7 /  |
| 382a7247 | completed | +0.1 / +0.0 / +1.3 / +5.0 /  | +0.5 / -1.3 / -0.5 / +9.0 /  |
| 3c2822d9 | completed |  /  /  /  /  | +1.1 / -0.0 / +11.0 / +6.5 /  |
| 3ce13eed | event |  /  /  /  /  | +2.1 / +1.4 / +14.3 / +3.7 /  |
| 403d558a | event | +0.8 / -0.0 / -0.2 / +9.7 /  | +1.1 / +1.2 / +15.2 / +11.9 /  |
| 68e53160 | event |  /  /  /  /  | +0.7 / +1.0 / +3.8 / +5.1 /  |
| 6fd8cf30 | short_rear | +1.0 / +0.1 / +1.4 / +2.8 /  | +0.4 / +0.9 / +2.8 / +2.7 /  |
| 8a143cb2 | completed |  /  /  /  /  | -2.4 / +0.0 / -7.1 / +1.9 /  |
| 94c1d722 | event | +0.9 / -0.1 / -1.9 / +6.8 / +0.9 | +0.5 / +0.0 / +0.9 / +3.2 / +0.5 |
| 9f393c8a | event |  /  /  /  /  | -0.4 / +0.0 / -1.7 / +5.8 /  |
| a174faed | event | +0.5 / +0.1 / +0.7 / +5.8 /  | +0.9 / +1.0 / +17.0 / +6.2 /  |
| a1b166cc | event |  /  /  /  /  | -1.0 / -0.6 / -9.2 / +3.8 /  |
| a4d2f7cf | short_slow | +0.1 / -0.1 / +2.2 / +2.8 / +0.1 | +1.5 / +1.2 / +15.8 / +3.9 / +1.5 |
| abd45a30 | short_rear | -0.1 / -0.4 / -2.4 / +3.2 / -0.1 |  /  /  /  /  |
| b40f4729 | off |  /  /  /  /  | -1.6 / +0.2 / -6.6 / +3.1 /  |
| ce161092 | event | +0.0 / -0.1 / -0.3 / +8.2 /  | +0.3 / +0.0 / +2.1 / +8.1 / +0.2 |
| d90fe123 | completed | +0.0 / +0.2 / +0.1 / +8.8 / +0.0 | +0.1 / -0.2 / +0.6 / +8.3 / +0.1 |
| ddc3e8df | event |  /  /  /  /  | -0.3 / -0.0 / +13.7 / +2.6 /  |
| e0f6ebe1 | off | +0.9 / +1.0 / +2.4 / +7.2 / +0.9 | -0.5 / -1.0 / -10.3 / +7.7 / -0.4 |
| e37aa159 | event | -0.7 / -0.2 / -1.9 / +5.6 /  | -0.7 / -1.4 / -1.2 / +7.4 /  |
| e5e47fd2 | event | -0.1 / +0.0 / -0.7 / +12.7 /  | -1.1 / -1.0 / -12.1 / +13.8 / -1.1 |
| e848c843 | off | +0.6 / -0.1 / -0.5 / +3.5 /  | -0.4 / -0.2 / -7.9 / +3.4 / -0.4 |
| ea468790 | off |  /  /  /  /  | +0.2 / +0.9 / +3.2 / +3.3 /  |
| eaba3f6a | event | +0.5 / +0.0 / +0.9 / +10.6 /  | +1.5 / +1.8 / +16.0 / +12.6 / +1.5 |
| f9e361aa | event | -0.0 / +0.2 / -0.1 / +10.3 /  | +0.2 / +1.8 / +14.0 / +11.4 / +0.2 |
| fd01e9ab | event |  /  /  /  /  | +1.1 / -0.9 / +5.5 / +3.7 /  |

Lane changes required by the route (proxy: route at s + 40 m >= 2 m to the side of the tangent at s, heading change < 15 deg; strict < 5 deg); shifted = the ego moved >= 2 m toward the route's side within [s, s + 60] before its zero event; log = the same on the human's path:

| set | proxy | rollouts with a site | ego reached the site | ego shifted >= 2 m | log shifted >= 2 m | zeros among reached (shifted) |
|:--|:--|--:|--:|--:|--:|--:|
| pai | < 15 deg | 116 | 71 | 47 (66 %) | 63 (89 %) | 21 (10) |
| pai | < 5 deg | 98 | 71 | 50 (70 %) | 62 (87 %) | 18 (10) |
| held | < 15 deg | 242 | 142 | 119 (84 %) | 132 (93 %) | 2 (0) |
| held | < 5 deg | 152 | 126 | 89 (71 %) | 114 (90 %) | 0 (0) |

Amendment 3, map lane changes (the route point 40 m ahead lies >= 2 m from every successor chain of the lane the route is in, and on another lane; first site per rollout; validated = the other lane is adjacent (gap <= 6 m) and the human's log itself moved 2-6 m to that side). Shifted = the ego moved >= 2 m toward that side within [s, s + 60] before its zero event; in lane = |lat_route| <= 1.75 m somewhere in [s + 30, s + 60]; stayed = still within 1.75 m of its own lane's chain at s + 40; log = the human's shift (gate):

| set | rollouts scanned | rollouts with a site (scenes) | ego reached the site | ego shifted >= 2 m | ego in the route's lane after | ego stayed in its lane | log shifted >= 2 m | zeros among reached (of them shifted) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| pai|map site | 281 | 31 (31) | 21 | 7 (33 %) | 11 (52 %) | 3 (14 %) | 8 (38 %) | 6 (3) |
| pai|validated | 281 | 6 (6) | 4 | 1 (25 %) | 2 (50 %) | 0 (0 %) | 4 (100 %) | 1 (0) |
| held|map site | 904 | 52 (26) | 18 | 8 (44 %) | 14 (78 %) | 2 (11 %) | 12 (67 %) | 0 (0) |
| held|validated | 904 | 6 (3) | 4 | 0 (0 %) | 2 (50 %) | 0 (0 %) | 4 (100 %) | 0 (0) |

Per validated map site:

| set | scene | score | primary | side | gap (m) | reached | ego shift | in lane after | stayed | log shift |
|:--|:--|--:|:--|:--|--:|:--|--:|:--|:--|--:|
| held_s0 | d91f0c28683e52da | 0.61 | low_slow_free | R | 3.3 | False | +nan | False | nan | +3.6 |
| held_s0 | 3c7c1b09e9625732 | 0.63 | low_slow_free | R | 3.7 | True | +0.0 | False | nan | +4.2 |
| held_s0 | 0481cf2b75f1532f | 1.00 | ok | R | 4.9 | True | +2.0 | True | False | +2.4 |
| held_s1 | d91f0c28683e52da | 0.60 | low_unexplained | R | 3.3 | False | +nan | False | nan | +3.6 |
| held_s1 | 3c7c1b09e9625732 | 0.65 | low_slow_free | R | 3.7 | True | -0.0 | False | nan | +4.2 |
| held_s1 | 0481cf2b75f1532f | 1.00 | ok | R | 4.9 | True | +1.9 | True | False | +2.4 |
| pai | 1e8e939e | 0.00 | coll_cyclist | L | 5.7 | True | +1.3 | True | False | +2.9 |
| pai | 415e950b | 0.39 | low_slow_free | L | 4.3 | True | +0.3 | False | nan | +4.7 |
| pai | 51242b4d | 0.66 | low_slow_free | L | 3.8 | True | +1.1 | False | nan | +2.5 |
| pai | 8a143cb2 | 1.00 | ok | L | 2.6 | True | +4.0 | True | False | +2.6 |
| pai | e904e9c0 | 0.00 | cor_lane_change | R | 4.1 | False | +nan | False | nan | +2.4 |
| pai | f8ee31cb | 1.00 | ok | L | 3.1 | False | +nan | False | nan | +3.0 |

Rollouts scanned: PAI 281, held 904 (both seeds).

Per site (PAI, strict proxy < 5 deg first, then < 15 deg):

| scene | score | primary | y at +40 m | dpsi | reached | ego shift (m) | log shift (m) |
|:--|--:|:--|--:|--:|:--|--:|--:|
| 000525f6 (lc5) | 0.66 | low_slow_free | -3.6 | 2 | True | -0.0 | +0.5 |
| 00fabf60 (lc5) | 0.00 | cor_turn_under | -12.5 | 3 | True | +13.9 | +11.1 |
| 032b6f21 (lc5) | 1.00 | ok | -2.1 | 4 | True | +2.8 | +2.9 |
| 054b5901 (lc5) | 0.00 | cor_route_end | +2.0 | 5 | True | +4.4 | +4.1 |
| 074e5ee3 (lc5) | 0.40 | low_slow_free | -2.5 | 5 | True | +4.2 | +4.0 |
| 0786d456 (lc5) | 1.00 | ok | +2.3 | 5 | False | +nan | +5.0 |
| 09528162 (lc5) | 0.28 | low_rear_ended | -4.2 | 2 | True | -0.0 | +5.1 |
| 095cf563 (lc5) | 0.00 | cor_route_end | -2.9 | 2 | False | +nan | +2.3 |
| 0a228e44 (lc5) | 0.00 | cor_straight_at_turn | -2.3 | 5 | True | +1.2 | +6.1 |
| 1089758a (lc5) | 1.00 | ok | +3.0 | 5 | True | +6.2 | +5.6 |
| 1316f5fa (lc5) | 1.00 | ok | +4.6 | 4 | True | +5.0 | +4.8 |
| 132ae6cb (lc5) | 0.04 | low_slow_behind_lead | +2.9 | 4 | True | +0.0 | +4.1 |
| 1bfad87c (lc5) | 0.00 | off_straight | +2.7 | 5 | True | +3.3 | +4.7 |
| 1e8e939e (lc5) | 0.00 | coll_cyclist | -2.1 | 5 | True | +1.3 | +2.2 |
| 2466266e (lc5) | 0.00 | off_curve_inside | +6.7 | 5 | True | +4.7 | +7.1 |
| 26004ca8 (lc5) | 0.32 | low_slow_free | -2.2 | 5 | True | +2.9 | +3.0 |
| 2a0efada (lc5) | 0.47 | low_slow_free | +2.4 | 2 | False | +nan | +0.0 |
| 2a882d89 (lc5) | 0.00 | off_junc_outside | +2.7 | 4 | False | +nan | +2.4 |
| 300465f7 (lc5) | 0.00 | off_junc_inside | -2.7 | 4 | True | +1.8 | +4.2 |
| 302c5c99 (lc5) | 0.00 | cor_straight_at_turn | +2.9 | 4 | False | +nan | +0.0 |
| 3232fdb7 (lc5) | 0.00 | cor_straight_at_turn | -2.4 | 4 | False | +nan | +4.7 |
| 32be03e4 (lc5) | 0.00 | cor_wrong_exit | -7.2 | 1 | False | +nan | +0.1 |
| 33c8976b (lc5) | 0.00 | off_junc_outside | +2.8 | 4 | False | +nan | +4.5 |
| 37d660b1 (lc5) | 0.00 | cor_turn_under | -3.4 | 4 | False | +nan | +4.0 |
| 382a7247 (lc5) | 0.00 | coll_veh_oncoming | +3.9 | 5 | True | +6.1 | +4.8 |
| 3c2822d9 (lc5) | 0.88 | low_slow_free | +3.0 | 5 | True | +5.7 | +4.9 |
| 3ce13eed (lc5) | 0.00 | cor_turn_under | -2.3 | 3 | False | +nan | +2.4 |
| 3ffad395 (lc5) | 1.00 | ok | -2.7 | 5 | True | +1.4 | +3.3 |
| 404541da (lc5) | 0.39 | low_slow_behind_lead | +3.2 | 5 | True | +4.5 | +4.6 |
| 415e950b (lc5) | 0.39 | low_slow_free | +2.0 | 3 | True | +3.2 | +4.2 |
| 478d6ec4 (lc5) | 0.87 | low_slow_free | -3.0 | 5 | True | +0.3 | +0.0 |
| 49bcab6f (lc5) | 1.00 | ok | +2.8 | 5 | True | +4.7 | +4.6 |
| 4a863a41 (lc5) | 0.45 | low_slow_free | +2.0 | 5 | True | +4.0 | +4.2 |
| 4b5a7c8b (lc5) | 0.31 | low_rear_ended | -3.6 | 5 | True | +2.2 | +2.6 |
| 4cb7f7ac (lc5) | 0.00 | cor_route_end | -2.0 | 4 | True | +2.4 | +2.8 |
| 4e4a2a51 (lc5) | 0.58 | low_slow_free | +3.0 | 5 | True | +4.3 | +4.0 |
| 4f7cb072 (lc5) | 0.43 | low_slow_free | +2.6 | 5 | True | +4.2 | +4.3 |
| 51400458 (lc5) | 1.00 | ok | -2.3 | 4 | True | +4.2 | +4.6 |
| 56bf7a4b (lc5) | 0.53 | low_slow_free | -13.5 | 3 | False | +nan | +11.4 |
| 596af0c7 (lc5) | 0.41 | low_slow_free | -2.1 | 5 | False | +nan | +3.3 |
| 599c94eb (lc5) | 1.00 | ok | -2.3 | 2 | True | +2.9 | +2.6 |
| 5ddb96bd (lc5) | 0.48 | low_rear_ended | -3.4 | 3 | True | +1.1 | +2.5 |
| 60cc35db (lc5) | 1.00 | ok | -2.4 | 4 | True | +8.8 | +7.0 |
| 6250146a (lc5) | 1.00 | ok | -2.6 | 0 | True | -0.0 | +0.0 |
| 62cfe9d3 (lc5) | 0.88 | low_slow_free | +5.7 | 4 | True | +7.3 | +5.6 |
| 6e6cbe67 (lc5) | 0.00 | cor_drift | -4.7 | 5 | True | +0.7 | +1.4 |
| 6fb661c5 (lc5) | 1.00 | ok | -2.5 | 5 | True | +4.1 | +3.7 |
| 6fd8cf30 (lc5) | 0.30 | low_slow_behind_lead | +3.0 | 1 | False | +nan | +2.6 |
| 7381c9fa (lc5) | 0.00 | off_junc_outside | +5.6 | 2 | True | +0.0 | +4.7 |
| 76f7e025 (lc5) | 1.00 | ok | +3.0 | 5 | True | +3.7 | +4.0 |
| 7750265d (lc5) | 1.00 | ok | +3.4 | 5 | True | +1.7 | +2.9 |
| 788c4ad9 (lc5) | 1.00 | ok | +2.1 | 3 | True | +3.6 | +3.7 |
| 7d994ed3 (lc5) | 0.00 | cor_route_end | -2.1 | 5 | True | -0.0 | +1.0 |
| 87a3b6d4 (lc5) | 1.00 | ok | -2.5 | 5 | True | +3.7 | +3.7 |
| 8a143cb2 (lc5) | 1.00 | ok | -3.4 | 4 | True | +0.6 | +0.4 |
| 8b25a6ac (lc5) | 0.70 | low_slow_free | -2.8 | 5 | False | +nan | +4.2 |
| 901c2cff (lc5) | 0.53 | low_slow_free | +2.3 | 5 | False | +nan | +3.4 |
| 94c1d722 (lc5) | 0.00 | off_junc_outside | -3.1 | 5 | False | +nan | +5.6 |
| 984b2242 (lc5) | 1.00 | ok | -2.1 | 4 | True | +2.4 | +2.8 |
| 98f88a4e (lc5) | 0.00 | off_curve_outside | -2.7 | 5 | False | +nan | +4.7 |
| 9d9e7f39 (lc5) | 0.44 | low_slow_free | -3.1 | 4 | True | +6.2 | +5.4 |
| 9f393c8a (lc5) | 0.00 | coll_veh_side | -9.9 | 4 | False | +nan | +9.5 |
| a174faed (lc5) | 0.00 | cor_straight_at_turn | -4.8 | 3 | False | +nan | +5.7 |
| a1f07c94 (lc5) | 0.78 | low_slow_free | +2.0 | 4 | True | +2.3 | +3.3 |
| a2c5ae46 (lc5) | 0.00 | cor_drift | +3.5 | 3 | True | +4.3 | +1.2 |
| a5599793 (lc5) | 0.00 | off_junc_straight | +2.7 | 4 | True | +0.0 | +5.2 |
| a68ad579 (lc5) | 0.44 | low_unexplained | +2.5 | 5 | True | +5.4 | +5.5 |
| af85ffdb (lc5) | 1.00 | ok | -2.8 | 5 | True | +2.9 | +4.6 |
| b44a4a44 (lc5) | 0.00 | coll_ped | -2.9 | 4 | False | +nan | +3.9 |
| b727530b (lc5) | 0.63 | low_slow_free | -2.8 | 5 | True | -0.0 | +3.7 |
| ba62e10d (lc5) | 0.00 | cor_drift | -7.6 | 4 | True | -0.0 | +6.9 |
| baabc57d (lc5) | 0.85 | low_slow_free | +2.2 | 1 | True | +2.2 | +3.0 |
| bbc53aa8 (lc5) | 0.53 | low_slow_free | -7.2 | 4 | True | -0.0 | +0.0 |
| bd97de34 (lc5) | 0.00 | cor_straight_at_turn | +2.1 | 4 | True | +3.0 | +2.2 |
| bf1847a9 (lc5) | 1.00 | ok | +2.9 | 4 | True | +4.9 | +4.2 |
| bf56fd00 (lc5) | 0.47 | low_slow_behind_lead | -4.5 | 4 | True | +7.1 | +7.0 |
| c4a58f83 (lc5) | 0.68 | low_slow_free | -2.6 | 5 | True | +4.3 | +4.3 |
| c5da4975 (lc5) | 0.55 | low_slow_free | -2.1 | 4 | True | +2.0 | +3.3 |
| ca173de1 (lc5) | 1.00 | ok | +2.5 | 5 | True | +0.6 | +1.0 |
| ca481583 (lc5) | 1.00 | ok | -3.6 | 2 | False | +nan | +0.0 |
| cdf95fd1 (lc5) | 0.00 | cor_turn_under | -2.4 | 5 | True | +3.5 | +3.2 |
| d8bfaaf2 (lc5) | 0.00 | off_curve_inside | +6.9 | 5 | True | +5.8 | +6.4 |
| d90fe123 (lc5) | 1.00 | ok | +2.7 | 4 | True | +5.5 | +3.0 |
| db8693db (lc5) | 0.58 | low_slow_behind_lead | +2.6 | 5 | True | +3.3 | +4.6 |
| dc401449 (lc5) | 1.00 | ok | +2.5 | 5 | True | +3.7 | +4.5 |
| ddb6a43a (lc5) | 0.36 | low_slow_free | +2.6 | 4 | True | +3.6 | +4.1 |
| ddc3e8df (lc5) | 0.00 | cor_turn_under | -2.7 | 4 | False | +nan | +4.4 |
| e1c6ae9f (lc5) | 1.00 | ok | +2.4 | 5 | True | +3.2 | +3.5 |
| e904e9c0 (lc5) | 0.00 | cor_lane_change | -4.0 | 5 | False | +nan | +2.2 |
| ea468790 (lc5) | 0.55 | low_slow_free | -2.6 | 4 | True | +1.0 | +2.6 |
| eaba3f6a (lc5) | 0.00 | cor_straight_at_turn | -2.8 | 4 | False | +nan | +1.9 |
| ec1d0b2b (lc5) | 0.23 | low_queue | -2.6 | 5 | True | +2.8 | +4.0 |
| f3a87e7a (lc5) | 1.00 | ok | -3.8 | 5 | True | +3.6 | +5.5 |
| f65aa80b (lc5) | 1.00 | ok | +2.2 | 5 | True | +2.5 | +2.7 |
| f8ee31cb (lc5) | 1.00 | ok | +3.2 | 0 | False | +nan | +3.0 |
| f9e361aa (lc5) | 0.00 | cor_straight_at_turn | -2.6 | 4 | False | +nan | +2.6 |
| fbf97438 (lc5) | 0.31 | low_slow_free | -2.3 | 4 | True | +2.6 | +2.5 |
| fd01e9ab (lc5) | 0.00 | cor_turn_under | +2.0 | 5 | False | +nan | +4.6 |
| 000525f6 (lc) | 0.66 | low_slow_free | -3.6 | 11 | True | -0.0 | +0.6 |
| 00fabf60 (lc) | 0.00 | cor_turn_under | -2.9 | 12 | False | +nan | +9.0 |
| 032b6f21 (lc) | 1.00 | ok | -2.1 | 4 | True | +2.8 | +2.9 |
| 054b5901 (lc) | 0.00 | cor_route_end | +2.0 | 6 | True | +4.6 | +4.3 |
| 074e5ee3 (lc) | 0.40 | low_slow_free | -6.4 | 15 | True | +11.3 | +11.3 |
| 0786d456 (lc) | 1.00 | ok | +2.3 | 5 | False | +nan | +5.0 |
| 09528162 (lc) | 0.28 | low_rear_ended | -10.0 | 11 | True | +1.1 | +13.0 |
| 095cf563 (lc) | 0.00 | cor_route_end | -2.9 | 2 | False | +nan | +2.3 |
| 0a228e44 (lc) | 0.00 | cor_straight_at_turn | -2.0 | 6 | True | +1.3 | +4.7 |
| 1089758a (lc) | 1.00 | ok | +2.0 | 8 | True | +6.7 | +6.3 |
| 1316f5fa (lc) | 1.00 | ok | +2.0 | 12 | True | +8.0 | +7.0 |
| 132ae6cb (lc) | 0.04 | low_slow_behind_lead | +8.7 | 13 | True | +1.0 | +12.4 |
| 13d646af (lc) | 0.69 | low_slow_free | +2.0 | 7 | True | +0.0 | +5.7 |
| 1a69f757 (lc) | 0.00 | coll_veh_oncoming | -6.2 | 15 | True | -0.0 | +11.5 |
| 1bfad87c (lc) | 0.00 | off_straight | +7.2 | 15 | True | +12.6 | +12.4 |
| 1e8e939e (lc) | 0.00 | coll_cyclist | -2.5 | 7 | False | +nan | +5.6 |
| 2466266e (lc) | 0.00 | off_curve_inside | +6.2 | 14 | True | +4.5 | +7.1 |
| 26004ca8 (lc) | 0.32 | low_slow_free | -3.4 | 11 | False | +nan | +7.1 |
| 268949d2 (lc) | 1.00 | ok | -7.8 | 14 | True | +12.0 | +13.3 |
| 288829f8 (lc) | 1.00 | ok | -3.4 | 13 | False | +nan | +10.0 |
| 2a0efada (lc) | 0.47 | low_slow_free | +2.4 | 9 | False | +nan | +0.0 |
| 2a882d89 (lc) | 0.00 | off_junc_outside | +8.6 | 15 | True | +0.4 | +12.1 |
| 2c263e19 (lc) | 1.00 | ok | -2.1 | 10 | True | +7.3 | +7.3 |
| 300465f7 (lc) | 0.00 | off_junc_inside | -2.5 | 9 | True | +1.8 | +4.2 |
| 302c5c99 (lc) | 0.00 | cor_straight_at_turn | +7.7 | 11 | False | +nan | +0.0 |
| 3232fdb7 (lc) | 0.00 | cor_straight_at_turn | -2.4 | 4 | False | +nan | +4.7 |
| 32be03e4 (lc) | 0.00 | cor_wrong_exit | -7.2 | 12 | False | +nan | +0.1 |
| 33c8976b (lc) | 0.00 | off_junc_outside | +8.7 | 14 | False | +nan | +13.2 |
| 37d660b1 (lc) | 0.00 | cor_turn_under | -9.3 | 13 | False | +nan | +12.2 |
| 382a7247 (lc) | 0.00 | coll_veh_oncoming | +10.1 | 15 | True | +15.9 | +13.9 |
| 3c2822d9 (lc) | 0.88 | low_slow_free | +7.7 | 12 | True | +12.7 | +11.6 |
| 3ce13eed (lc) | 0.00 | cor_turn_under | -9.0 | 13 | False | +nan | +11.8 |
| 3ffad395 (lc) | 1.00 | ok | -4.1 | 14 | False | +nan | +11.0 |
| 404541da (lc) | 0.39 | low_slow_behind_lead | +2.1 | 7 | True | +4.8 | +4.6 |
| 415e950b (lc) | 0.39 | low_slow_free | +2.0 | 3 | True | +3.2 | +4.2 |
| 478d6ec4 (lc) | 0.87 | low_slow_free | -2.1 | 10 | True | +0.1 | +0.0 |
| 49bcab6f (lc) | 1.00 | ok | +2.2 | 7 | False | +nan | +4.8 |
| 4a863a41 (lc) | 0.45 | low_slow_free | +2.4 | 5 | False | +nan | +4.7 |
| 4b5a7c8b (lc) | 0.31 | low_rear_ended | -7.0 | 12 | False | +nan | +8.2 |
| 4cb7f7ac (lc) | 0.00 | cor_route_end | -2.0 | 4 | True | +2.4 | +2.8 |
| 4e4a2a51 (lc) | 0.58 | low_slow_free | -3.0 | 8 | False | +nan | +6.4 |
| 4f7cb072 (lc) | 0.43 | low_slow_free | +2.6 | 8 | False | +nan | +5.7 |
| 51242b4d (lc) | 0.66 | low_slow_free | +3.8 | 14 | True | +1.0 | +2.3 |
| 51400458 (lc) | 1.00 | ok | -2.3 | 4 | True | +4.2 | +4.6 |
| 51aee21d (lc) | 1.00 | ok | -2.3 | 7 | True | +6.7 | +5.8 |
| 56bf7a4b (lc) | 0.53 | low_slow_free | -2.1 | 9 | True | +1.2 | +7.0 |
| 596af0c7 (lc) | 0.41 | low_slow_free | -2.1 | 8 | True | +6.3 | +5.6 |
| 599c94eb (lc) | 1.00 | ok | -2.3 | 2 | True | +2.9 | +2.6 |
| 5ddb96bd (lc) | 0.48 | low_rear_ended | -6.0 | 15 | True | +4.1 | +6.4 |
| 60cc35db (lc) | 1.00 | ok | -9.1 | 14 | True | +16.2 | +14.6 |
| 6250146a (lc) | 1.00 | ok | -2.6 | 12 | True | -0.0 | +0.0 |
| 62cfe9d3 (lc) | 0.88 | low_slow_free | +7.2 | 14 | True | +10.7 | +8.6 |
| 6e6cbe67 (lc) | 0.00 | cor_drift | -4.7 | 14 | True | +0.5 | +1.3 |
| 6ee9ee78 (lc) | 0.00 | cor_drift | -2.0 | 5 | True | +2.3 | +4.8 |
| 6fb661c5 (lc) | 1.00 | ok | -2.0 | 5 | True | +3.9 | +3.8 |
| 6fd8cf30 (lc) | 0.30 | low_slow_behind_lead | +3.0 | 1 | False | +nan | +2.6 |
| 7381c9fa (lc) | 0.00 | off_junc_outside | +11.1 | 14 | True | +0.0 | +11.5 |
| 76f7e025 (lc) | 1.00 | ok | +2.3 | 5 | True | +4.4 | +4.2 |
| 7750265d (lc) | 1.00 | ok | +2.1 | 13 | True | +1.8 | +3.1 |
| 788c4ad9 (lc) | 1.00 | ok | +2.2 | 6 | True | +6.8 | +6.4 |
| 7c346fd5 (lc) | 0.34 | low_slow_free | -2.0 | 6 | False | +nan | +4.0 |
| 7d994ed3 (lc) | 0.00 | cor_route_end | -2.1 | 5 | True | -0.0 | +1.0 |
| 875e79d9 (lc) | 1.00 | ok | +4.7 | 14 | False | +nan | +10.8 |
| 87a3b6d4 (lc) | 1.00 | ok | -2.0 | 7 | True | +5.5 | +5.1 |
| 8a143cb2 (lc) | 1.00 | ok | -8.9 | 12 | True | +7.6 | +7.5 |
| 8b25a6ac (lc) | 0.70 | low_slow_free | -2.7 | 9 | False | +nan | +6.9 |
| 901c2cff (lc) | 0.53 | low_slow_free | +2.1 | 7 | False | +nan | +5.1 |
| 94c1d722 (lc) | 0.00 | off_junc_outside | -7.5 | 13 | False | +nan | +12.3 |
| 984b2242 (lc) | 1.00 | ok | -2.1 | 4 | True | +2.4 | +2.8 |
| 98f88a4e (lc) | 0.00 | off_curve_outside | -7.8 | 15 | False | +nan | +12.0 |
| 9d9052bd (lc) | 1.00 | ok | +2.0 | 6 | True | +4.3 | +4.9 |
| 9d9e7f39 (lc) | 0.44 | low_slow_free | -5.6 | 9 | False | +nan | +9.0 |
| 9f393c8a (lc) | 0.00 | coll_veh_side | -13.6 | 14 | True | -0.0 | +13.8 |
| a174faed (lc) | 0.00 | cor_straight_at_turn | -9.7 | 12 | False | +nan | +11.6 |
| a1f07c94 (lc) | 0.78 | low_slow_free | +2.0 | 4 | True | +2.3 | +3.3 |
| a2c5ae46 (lc) | 0.00 | cor_drift | +3.5 | 13 | True | +4.0 | +1.1 |
| a5599793 (lc) | 0.00 | off_junc_straight | +2.7 | 4 | True | +0.0 | +5.2 |
| a68ad579 (lc) | 0.44 | low_unexplained | +7.0 | 15 | True | +12.4 | +11.9 |
| af85ffdb (lc) | 1.00 | ok | -2.0 | 6 | True | +2.7 | +4.1 |
| b40f4729 (lc) | 1.00 | ok | -8.8 | 13 | True | +9.3 | +11.4 |
| b44a4a44 (lc) | 0.00 | coll_ped | -8.4 | 14 | False | +nan | +12.6 |
| b65e55e9 (lc) | 0.40 | low_slow_free | +2.0 | 6 | False | +nan | +3.6 |
| b727530b (lc) | 0.63 | low_slow_free | -2.1 | 10 | True | +2.0 | +5.3 |
| ba62e10d (lc) | 0.00 | cor_drift | -7.0 | 15 | True | -0.0 | +6.8 |
| baabc57d (lc) | 0.85 | low_slow_free | +2.2 | 1 | True | +2.2 | +3.0 |
| bbc53aa8 (lc) | 0.53 | low_slow_free | -7.2 | 10 | True | -0.0 | +0.0 |
| bd97de34 (lc) | 0.00 | cor_straight_at_turn | +2.6 | 6 | True | +4.9 | +4.2 |
| bf1847a9 (lc) | 1.00 | ok | +8.4 | 13 | True | +13.5 | +12.8 |
| bf56fd00 (lc) | 0.47 | low_slow_behind_lead | -6.0 | 7 | False | +nan | +10.6 |
| c4a58f83 (lc) | 0.68 | low_slow_free | -2.9 | 9 | False | +nan | +6.0 |
| c5da4975 (lc) | 0.55 | low_slow_free | -2.1 | 4 | True | +2.0 | +3.3 |
| ca173de1 (lc) | 1.00 | ok | +2.1 | 7 | True | +3.0 | +3.2 |
| ca481583 (lc) | 1.00 | ok | -3.6 | 12 | False | +nan | +0.0 |
| cdf95fd1 (lc) | 0.00 | cor_turn_under | -2.8 | 8 | False | +nan | +5.0 |
| d8bfaaf2 (lc) | 0.00 | off_curve_inside | +10.0 | 14 | True | +10.6 | +11.6 |
| d90fe123 (lc) | 1.00 | ok | +8.6 | 14 | True | +14.6 | +11.4 |
| db8693db (lc) | 0.58 | low_slow_behind_lead | +6.3 | 14 | True | +11.4 | +11.1 |
| dc401449 (lc) | 1.00 | ok | +2.1 | 11 | True | +9.4 | +9.1 |
| ddb6a43a (lc) | 0.36 | low_slow_free | +9.6 | 15 | True | +13.9 | +14.1 |
| ddc3e8df (lc) | 0.00 | cor_turn_under | -8.7 | 15 | True | -0.0 | +14.7 |
| df99cdc4 (lc) | 0.70 | low_slow_behind_lead | +2.0 | 14 | True | +0.2 | +0.2 |
| e1c6ae9f (lc) | 1.00 | ok | +4.5 | 11 | False | +nan | +8.7 |
| e37aa159 (lc) | 0.00 | cor_turn_under | +8.3 | 13 | False | +nan | +8.6 |
| e7ddb7f0 (lc) | 1.00 | ok | -2.5 | 8 | False | +nan | +5.9 |
| e848c843 (lc) | 0.79 | low_slow_free | -10.4 | 15 | True | +1.8 | +14.5 |
| e904e9c0 (lc) | 0.00 | cor_lane_change | +6.0 | 15 | True | +3.8 | +12.6 |
| ea468790 (lc) | 0.55 | low_slow_free | -8.7 | 13 | True | +3.6 | +11.2 |
| eaba3f6a (lc) | 0.00 | cor_straight_at_turn | -9.7 | 14 | False | +nan | +11.5 |
| ec1d0b2b (lc) | 0.23 | low_queue | -3.5 | 7 | False | +nan | +6.7 |
| f3a87e7a (lc) | 1.00 | ok | -4.8 | 14 | False | +nan | +10.2 |
| f65aa80b (lc) | 1.00 | ok | +2.1 | 6 | False | +nan | +2.6 |
| f8ee31cb (lc) | 1.00 | ok | +3.2 | 0 | False | +nan | +3.0 |
| f9121c36 (lc) | 0.60 | low_slow_free | +2.0 | 6 | True | +4.4 | +4.2 |
| f9e361aa (lc) | 0.00 | cor_straight_at_turn | -7.6 | 11 | False | +nan | +9.6 |
| fbf97438 (lc) | 0.31 | low_slow_free | -4.3 | 12 | False | +nan | +8.7 |
| fd01e9ab (lc) | 0.00 | cor_turn_under | +7.2 | 14 | False | +nan | +10.5 |
