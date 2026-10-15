# MODE AVERAGING, Reading 1 (rf_modeavg.py read)

Prereg `plans/2026-10-15-mode-averaging-prereg.md`. delta = 1 - plan 4 s heading change / human 4 s heading change at the same route station (decisions with human >= 15 deg, winsorised to [-1, 2]); window = approach decisions 0-30 m before the entry; D = per-turn median delta. Populations: pai = PAI 281 RSG-L18-s0 (strict 36 + 2 over-turn zeros), held = held 460 @18 RSG-L18-s0 / -s1 (24 + 24 strict, 5 + 4 over-turn zeros). CIs: scene-clustered bootstrap (held: log), B 10 000.

Gates and coverage: pai: 38 turns, lane matched 1.00, route direction among the map exits 0.97 (G-map pass), D defined 36, window decisions 1980 (1246 with a deficit), command joined 1.00; held: 57 turns, lane matched 0.96, route direction among the map exits 1.00 (G-map pass), D defined 55, window decisions 290 (244 with a deficit), command joined 1.00; pooled: 95 turns, lane matched 0.98, route direction among the map exits 0.99 (G-map pass), D defined 91, window decisions 2270 (1490 with a deficit), command joined 1.00

## M1: deficit by the number of exits (turn level, n_x >= 3 minus n_x = 2)

| pop | turns 3+ / 2 | median D 3+ / 2 | difference [CI] | verdict |
|:--|:--|:--|:--|:--|
| pai | 15 / 16 | +0.38 / +0.37 | +0.01 [-0.49, +0.54] | inconclusive |
| held | 24 / 25 | +0.06 / -0.01 | +0.08 [-0.03, +0.20] | inconclusive |
| pooled | 39 / 41 | +0.08 / +0.03 | +0.05 [-0.06, +0.23] | inconclusive |

Per exit count (4 = 4 or more): turns (failed), median D (failed / completed):

| pop | n_x | turns (failed) | D | D failed | D completed |
|:--|:--|:--|:--|:--|:--|
| pai | 1 | 5 (5) | -0.34 | -0.34 | - |
| pai | 2 | 18 (14) | +0.37 | +0.55 | +0.13 |
| pai | 3 | 11 (11) | +0.20 | +0.20 | - |
| pai | 4 | 4 (3) | +0.55 | +0.70 | +0.40 |
| held | 0 | 2 (0) | - | - | - |
| held | 1 | 6 (1) | +0.05 | +0.04 | +0.05 |
| held | 2 | 25 (14) | -0.01 | +0.01 | -0.05 |
| held | 3 | 22 (10) | +0.05 | +0.05 | +0.05 |
| held | 4 | 2 (2) | +0.15 | +0.15 | - |
| pooled | 0 | 2 (0) | - | - | - |
| pooled | 1 | 11 (6) | +0.04 | -0.15 | +0.05 |
| pooled | 2 | 43 (28) | +0.03 | +0.05 | -0.01 |
| pooled | 3 | 33 (21) | +0.07 | +0.17 | +0.05 |
| pooled | 4 | 6 (5) | +0.31 | +0.22 | +0.40 |

## M2: deficit by the served command (decision level, S minus the turn's own direction)

| pop | decisions S / agree (scenes) | median delta S / agree | difference [CI] | verdict | distance-matched | opposite command (n, delta) |
|:--|:--|:--|:--|:--|:--|:--|
| pai | 0 / 1246 (0 / 36) | - / - | - [-, -] | inconclusive (group too small) | - | 0, - |
| held | 0 / 244 (0 / 14) | - / - | - [-, -] | inconclusive (group too small) | - | 0, - |
| pooled | 0 / 1490 (0 / 50) | - / - | - [-, -] | inconclusive (group too small) | - | 0, - |

Per distance bin (S - agree; counts S / agree): pai 0-10 - (0 / 735), 10-20 - (0 / 402), 20-30 - (0 / 109); held 0-10 - (0 / 228), 10-20 - (0 / 16), 20-30 - (0 / 0); pooled 0-10 - (0 / 963), 10-20 - (0 / 418), 20-30 - (0 / 109)

## M3 (descriptive): deficit by route-memory coverage of the junction

| pop | measure | bin | decisions | median delta | failed | completed |
|:--|:--|:--|:--|:--|:--|:--|
| pai | cov | 0 | 519 | +0.17 | +0.18 | +0.11 |
| pai | cov | (0,0.5) | 100 | +0.50 | +0.97 | +0.49 |
| pai | cov | [0.5,1) | 218 | +0.48 | +0.48 | - |
| pai | cov | 1 | 409 | +0.55 | +0.63 | -0.13 |
| pai | n_past | 0 | 0 | - | - | - |
| pai | n_past | 1-5 | 0 | - | - | - |
| pai | n_past | 6-15 | 7 | +0.68 | +0.68 | - |
| pai | n_past | >15 | 1239 | +0.39 | +0.39 | +0.38 |
| held | cov | 0 | 244 | +0.01 | +0.01 | +0.01 |
| held | cov | (0,0.5) | 0 | - | - | - |
| held | cov | [0.5,1) | 0 | - | - | - |
| held | cov | 1 | 0 | - | - | - |
| held | n_past | 0 | 0 | - | - | - |
| held | n_past | 1-5 | 0 | - | - | - |
| held | n_past | 6-15 | 0 | - | - | - |
| held | n_past | >15 | 244 | +0.01 | +0.01 | +0.01 |
| pooled | cov | 0 | 763 | +0.08 | +0.12 | +0.03 |
| pooled | cov | (0,0.5) | 100 | +0.50 | +0.97 | +0.49 |
| pooled | cov | [0.5,1) | 218 | +0.48 | +0.48 | - |
| pooled | cov | 1 | 409 | +0.55 | +0.63 | -0.13 |
| pooled | n_past | 0 | 0 | - | - | - |
| pooled | n_past | 1-5 | 0 | - | - | - |
| pooled | n_past | 6-15 | 7 | +0.68 | +0.68 | - |
| pooled | n_past | >15 | 1483 | +0.22 | +0.27 | +0.08 |

## M4: the plan's heading near the entry (0-10 m, r = plan / human 4 s heading change)

| pop | turns | decisions | straight r < 0.25 | partial | full r >= 0.6 | median r | modal bin | reading |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai | failed | 661 | 0.13 | 0.18 | 0.69 | +0.85 | [0.8, 0.9] | mixed |
| pai | completed | 74 | 0.00 | 0.32 | 0.68 | +0.84 | [0.9, 1.0] | mixed |
| pai | over_turn | 1 | 1.00 | 0.00 | 0.00 | +0.03 | [-0.0, 0.1] | straight-dominant |
| held | failed | 120 | 0.00 | 0.08 | 0.92 | +0.97 | [0.9, 1.0] | mixed |
| held | completed | 108 | 0.00 | 0.02 | 0.98 | +0.99 | [0.9, 1.0] | mixed |
| held | over_turn | 34 | 0.00 | 0.00 | 1.00 | +1.07 | [0.7, 0.8] | mixed |
| pooled | failed | 781 | 0.11 | 0.17 | 0.72 | +0.88 | [0.9, 1.0] | mixed |
| pooled | completed | 182 | 0.00 | 0.14 | 0.86 | +0.96 | [0.9, 1.0] | mixed |
| pooled | over_turn | 35 | 0.03 | 0.00 | 0.97 | +1.07 | [0.7, 0.8] | mixed |

Deficit by distance to the entry and class (median delta, n):

| pop | d_e | completed | late_wide | no_zero | other | over_turn | straight |
|:--|:--|:--|:--|:--|:--|:--|:--|
| pai | 0-10 | +0.16 (74) | +0.13 (347) | -0.00 (209) | -0.43 (1) | +0.97 (1) | +0.89 (103) |
| pai | 10-20 | +0.45 (52) | +0.73 (123) | +0.77 (135) | -1.00 (3) | +0.92 (6) | +0.85 (83) |
| pai | 20-30 | +0.63 (27) | +0.14 (28) | +0.89 (15) | - (0) | +0.57 (1) | +0.92 (38) |
| held | 0-10 | +0.01 (108) | +0.08 (8) | +0.03 (78) | - | -0.07 (34) | - |
| held | 10-20 | -0.03 (7) | - (0) | -0.06 (9) | - | - (0) | - |
| held | 20-30 | - (0) | - (0) | - (0) | - | - (0) | - |
| pooled | 0-10 | +0.04 (182) | +0.12 (355) | +0.01 (287) | -0.43 (1) | -0.07 (35) | +0.89 (103) |
| pooled | 10-20 | +0.41 (59) | +0.73 (123) | +0.76 (144) | -1.00 (3) | +0.92 (6) | +0.85 (83) |
| pooled | 20-30 | +0.63 (27) | +0.14 (28) | +0.89 (15) | - (0) | +0.57 (1) | +0.92 (38) |

## Amendment 2 (post-hoc, descriptive)

Served command: in the 0-30 m window it names the turn's own direction on every decision (no S, no opposite). Turns whose rollout starts more than 45 m before the entry (12): the command switches from S to the turn direction at +36.6 m (median; +28.7 to +40.6) before the entry and stays there.

M4 shares at 10-20 m before the entry (where the late-wide deficit sits):

| pop | turns | decisions | straight | partial | full | median r | modal bin | reading |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai | failed | 350 | 0.53 | 0.29 | 0.18 | +0.22 | [-0.0, 0.1] | mixed |
| pai | completed | 52 | 0.00 | 0.58 | 0.42 | +0.55 | [0.5, 0.6] | averaging |
| held | failed | 9 | 0.00 | 0.00 | 1.00 | +1.06 | [1.0, 1.1] | mixed |
| held | completed | 7 | 0.00 | 0.00 | 1.00 | +1.03 | [1.0, 1.1] | mixed |

M3 by distance (median delta (n); memory coverage of the junction 0 / (0, 1) / 1): pai 0-10: +0.10 (403), +0.38 (132), +0.07 (200); pai 10-20: +0.69 (93), +0.53 (130), +0.83 (179); pai 20-30: +0.69 (23), +0.61 (56), +0.61 (30); held 0-10: +0.02 (228), - (0), - (0); held 10-20: -0.05 (16), - (0), - (0); held 20-30: - (0), - (0), - (0)

## Amendment 3 (post-hoc, descriptive): speed against geometry

The human's 4 s covers more road than a slower plan's 4 s. r_arc = plan heading change / the route's own heading change over the plan's 4 s arc a4 (from the ego's station); decisions with that route change >= 15 deg. Medians.

| pop | d_e | turns | plan deg | human 4 s deg | route over plan arc deg | plan arc a4 m | ego v m/s | delta (human) | delta_arc | straight / partial / full (r_arc) | median r_arc | reading |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai | 0-10 | failed | +42.1 | +56.8 | +47.9 | +12.7 | +2.5 | +0.15 | +0.09 | 0.10 / 0.04 / 0.86 | +0.91 | mixed |
| pai | 0-10 | completed | +53.8 | +71.3 | +74.6 | +26.4 | +6.1 | +0.16 | +0.21 | 0.00 / 0.26 / 0.74 | +0.79 | mixed |
| pai | 0-10 | cls late_wide | +51.0 | +56.2 | +44.6 | +12.3 | +2.4 | +0.13 | +0.01 | 0.00 / 0.01 / 0.99 | +0.99 | mixed |
| pai | 0-10 | cls straight | +5.6 | +74.0 | +80.6 | +22.6 | +5.9 | +0.89 | +0.91 | 0.60 / 0.09 / 0.31 | +0.09 | straight-dominant |
| pai | 0-10 | cls no_zero | +42.8 | +43.8 | +41.8 | +11.3 | +2.2 | -0.00 | +0.06 | 0.00 / 0.06 / 0.93 | +0.94 | mixed |
| pai | 10-20 | failed | +5.7 | +29.5 | +6.5 | +13.1 | +3.5 | +0.78 | +0.43 | 0.29 / 0.26 / 0.45 | +0.57 | mixed |
| pai | 10-20 | completed | +33.4 | +50.4 | +37.1 | +26.4 | +5.3 | +0.45 | +0.23 | 0.00 / 0.12 / 0.88 | +0.77 | mixed |
| pai | 10-20 | cls late_wide | +5.3 | +27.6 | +3.5 | +9.8 | +2.6 | +0.73 | +0.17 | 0.00 / 0.24 / 0.76 | +0.83 | mixed |
| pai | 10-20 | cls straight | +1.9 | +44.2 | +68.5 | +22.5 | +5.9 | +0.85 | +0.92 | 0.61 / 0.38 / 0.01 | +0.08 | straight-dominant |
| pai | 10-20 | cls no_zero | +6.1 | +27.6 | +3.3 | +10.0 | +3.2 | +0.77 | +0.20 | 0.00 / 0.05 / 0.95 | +0.80 | mixed |
| pai | 20-30 | failed | +1.3 | +3.4 | +0.3 | +14.8 | +3.8 | +0.67 | +0.88 | 0.63 / 0.13 / 0.24 | +0.12 | straight-dominant |
| pai | 20-30 | completed | +0.0 | +5.5 | +0.0 | +13.4 | +1.8 | +0.63 | +0.18 | 0.00 / 0.20 / 0.80 | +0.82 | mixed |
| pai | 20-30 | cls late_wide | +14.8 | +15.7 | +5.2 | +22.7 | +5.3 | +0.14 | +0.19 | 0.00 / 0.00 / 1.00 | +0.81 | mixed |
| pai | 20-30 | cls straight | +0.4 | +1.9 | +1.1 | +21.2 | +4.6 | +0.92 | +1.00 | 0.92 / 0.08 / 0.00 | -0.00 | straight-dominant |
| pai | 20-30 | cls no_zero | +1.0 | +3.1 | +0.0 | +11.9 | +2.7 | +0.89 | +0.76 | 0.57 / 0.14 / 0.29 | +0.24 | mode selection |
| held | 0-10 | failed | +53.3 | +53.2 | +50.4 | +13.9 | +2.3 | +0.03 | -0.06 | 0.03 / 0.04 / 0.93 | +1.06 | mixed |
| held | 0-10 | completed | +72.3 | +72.1 | +73.1 | +17.7 | +3.5 | +0.01 | -0.05 | 0.00 / 0.00 / 1.00 | +1.05 | mixed |
| held | 10-20 | failed | +44.7 | +36.8 | +54.9 | +22.6 | +7.5 | -0.06 | +0.05 | 0.00 / 0.00 / 1.00 | +0.95 | mixed |
| held | 10-20 | completed | +55.4 | +53.4 | +69.5 | +21.8 | +6.7 | -0.03 | +0.17 | 0.00 / 0.00 / 1.00 | +0.83 | mixed |

Per turn (pai):

| scene | side deg | class | n_x (lane) | exit directions deg | route deg | D | window (deficit) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 22fa5c92 | L +91 | late_wide | 3 (3) | -90, -90, -1, -1, +89, +89, +89, +89 | +89 | -0.03 | 72 (52) |
| 33c8976b | L +107 | late_wide | 1 (1) | +107 | +107 | -0.34 | 34 (34) |
| 37d660b1 | R -85 | late_wide | 2 (2) | -84, -84, +5, +5, +6, +6 | -84 | +0.57 | 53 (44) |
| 3ce13eed | R -100 | late_wide | 3 (1) | -98, -98, -97, -97, +6, +6, +12, +12, +45, +45, +57, +57, +65, +65, +74, +74 | -97 | +0.38 | 59 (42) |
| 68e53160 | L +108 | late_wide | 2 (1) | +0, +0, +112, +112, +114, +115 | +112 | +0.23 | 74 (45) |
| 94c1d722 | R -86 | late_wide | 2 (2) | -86, +88 | -86 | +0.90 | 104 (92) |
| 9f393c8a | R -84 | late_wide | 1 (1) | -83, -83 | -83 | +0.31 | 12 (12) |
| a1b166cc | L +91 | late_wide | 2 (1) | -7, -6, -5, -5, +88, +88, +89, +89 | +88 | +0.54 | 76 (47) |
| ce161092 | L +85 | late_wide | 2 (1) | +1, +2, +2, +3, +3, +3, +87, +87, +87, +87, +87, +87 | +87 | -0.33 | 37 (30) |
| ddc3e8df | R -66 | late_wide | 2 (1) | -66, -66, +113, +113 | -67 | +0.09 | 12 (12) |
| e37aa159 | L +82 | late_wide | 1 (1) | +79, +84, +85, +87 | +87 | +0.33 | 55 (55) |
| fd01e9ab | L +98 | late_wide | 2 (3) | +95, +96, +96, +98, +99, +99, +156, +175, +177, +178, +178, +178 | +96 | +0.25 | 53 (33) |
| 09528162 | R -100 | no_zero | 3 (3) | -100, -100, +3, +3, +88, +91 | -100 | +0.20 | 46 (46) |
| 268949d2 | R -88 | no_zero | 2 (2) | -88, -88, -88, +93, +93, +94 | -88 | -0.03 | 63 (14) |
| 6fd8cf30 | R -88 | no_zero | 2 (2) | -89, -89, +1, +1 | -89 | +0.75 | 137 (99) |
| a4d2f7cf | L +110 | no_zero | 3 (2) | -66, -66, -65, -4, -4, -4, -2, -2, -1, +108, +109, +109, +112, +113, +114 | +71 | +0.17 | 103 (54) |
| abd45a30 | L +90 | no_zero | 2 (2) | -2, -2, +88, +88 | +88 | +0.71 | 75 (31) |
| b40f4729 | R -89 | no_zero | 3 (3) | -89, -89, -88, -23, -10, +0, +88, +90, +91 | -88 | -0.17 | 34 (10) |
| c4a58f83 | R -85 | no_zero | 2 (1) | -88, -88, -0, +0, +0, +0, +0 | -87 | - | 0 (0) |
| e0f6ebe1 | R -90 | no_zero | 3 (3) | -88, -88, +2, +2, +110, +110 | -88 | -0.39 | 42 (24) |
| e848c843 | R -90 | no_zero | 3 (3) | -88, -88, -88, +1, +2, +2, +92, +92, +92 | -88 | -0.04 | 91 (50) |
| ea468790 | R -83 | no_zero | 1 (1) | -80, -80, -80 | -80 | -0.48 | 96 (31) |
| 1e8e939e | R -49 | other | 1 (1) | -70, -70, -70 | -70 | -0.83 | 15 (4) |
| 2466266e | L +96 | over_turn | 2 (1) | +1, +94, +112, +112, +113 | +112 | - | 16 (0) |
| 300465f7 | L +169 | over_turn | 4 (4) | -168, -149, -93, -85, -78, +12, +12, +102, +102 | +192 | +0.92 | 14 (8) |
| 2f227f2c | R -66 | straight | 2 (1) | -68, +1 | -68 | +0.93 | 67 (29) |
| 302c5c99 | L +163 | straight | 3 (2) | -3, -2, +91, +94, +179, +179 | +179 | +0.77 | 62 (46) |
| 3232fdb7 | R -87 | straight | 4 (4) | -144, -144, -143, -88, -88, -88, +0, +0, +0, +49, +53, +56, +60, +60, +64, +66, +67, +68, +71, +71, +85 | -88 | +0.70 | 94 (22) |
| 403d558a | R -93 | straight | 2 (2) | -104, -104, -5, -4, -1, -1, -0, +0 | -104 | +0.88 | 30 (30) |
| a174faed | R -90 | straight | 4 (2) | -180, -180, -179, -179, -90, -90, -90, -90, -0, -0, -0, -0, +85, +85 | -90 | +0.22 | 53 (34) |
| e5e47fd2 | L +95 | straight | 3 (1) | -83, -83, -6, -5, -5, -4, +122, +124, +130, +136 | +124 | +1.01 | 23 (23) |
| eaba3f6a | R -90 | straight | 3 (3) | -89, -88, +2, +2, +110, +110 | -88 | +0.98 | 28 (12) |
| f9e361aa | R -81 | straight | 3 (3) | -80, -80, +1, +1, +2, +2, +100, +100 | -80 | +1.00 | 29 (28) |
| 132ae6cb | L +86 | completed | 2 (1) | -4, -4, -4, +85, +85, +85, +86, +86, +86 | +86 | +0.50 | 105 (50) |
| 382a7247 | L +83 | completed | 4 (1) | -180, -180, -11, -11, -11, -11, -11, -11, +84, +85, +85, +179 | +84 | +0.40 | 51 (42) |
| 3c2822d9 | L +81 | completed | 2 (1) | -9, -9, -9, -9, +81, +81 | +81 | +0.03 | 10 (10) |
| 8a143cb2 | R -76 | completed | 2 (1) | -78, -77, +47, +50, +60, +62 | -77 | +0.24 | 20 (20) |
| d90fe123 | L +85 | completed | 2 (1) | +1, +2, +2, +3, +3, +3, +87, +87, +87, +87, +87, +87 | +87 | -0.13 | 35 (31) |
