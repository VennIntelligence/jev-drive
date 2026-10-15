# RECEDING-TURN fingerprint (rf_receding.py)

Prereg `plans/2026-10-15-receding-turn-prereg.md`. Per strict turn: slope = Theil-Sen of the served plan's 15 deg onset arc s_on on the distance to the entry d_e over the approach (1 = onset fixed in the world, 0 = fixed ahead of the car: receding); drift = forward movement of the world-frame onset along the route over the approach (m). Medians failed / completed, D = failed - completed with a scene-clustered bootstrap 95 % CI (B 10 000).

Gates: G-frame median |anchor - controller| per set pai_RSG-L18-s0 0.000, pai_RV2a-L18-s0 0.000, pai_RV2b-L18-s1 0.000, pai_APP1X-L18-s0 0.000, held_RSG-L18-s0 0.000, held_RSG-L18-s1 0.000, held_RV2a-L18-s0 0.000, held_RV2a-L18-s1 0.000, held_RV2b-L18-s0 0.000, held_RV2b-L18-s1 0.000 m; G-entry (held) 1.00 of 144 within 1 m (pass); G-human (pai_RSG) median slope 0.92, median |drift| 1.80 m (n 27, pass); class rule vs taxonomy 1.00 (n 24).

| set | turns f / c | C1 slope med f / c | D slope [CI] | C1 | C2 drift med f / c (m) | D drift [CI] | C2 | human slope f / c | model-plan slope f / c |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | 31 / 5 (slope 21 / 3) | +0.67 / +0.79 | -0.13 [-0.41, +0.33] | inconclusive | +3.3 / +4.0 | -0.7 [-9.3, +3.7] | refuted | +0.93 / +0.89 | +0.66 / +0.83 |
| pai_RV2a-L18-s0 | 31 / 5 (slope 22 / 3) | +0.62 / +0.86 | -0.24 [-0.39, +0.10] | refuted | +3.7 / +1.1 | +2.6 [-1.7, +4.5] | refuted | +0.93 / +0.87 | +0.68 / +0.63 |
| pai_RV2b-L18-s1 | 29 / 7 (slope 20 / 5) | +0.66 / +0.72 | -0.06 [-0.36, +0.23] | refuted | +2.7 / +5.6 | -3.0 [-4.4, +3.3] | refuted | +0.93 / +0.88 | +0.69 / +0.64 |
| pai_APP1X-L18-s0 | 25 / 11 (slope 18 / 8) | +0.53 / +0.96 | -0.42 [-0.72, -0.19] | inconclusive | +5.1 / +0.3 | +4.8 [+1.0, +8.0] | inconclusive | +0.94 / +0.91 | +0.60 / +0.95 |
| held_RSG-L18-s0 | 8 / 16 (slope 2 / 4) | +0.67 / +0.65 | +0.02 [-0.33, +0.74] | inconclusive (group < 3) | +2.9 / +3.6 | -0.7 [-2.4, +1.1] | inconclusive (group < 3) | +0.90 / +0.71 | - |
| held_RSG-L18-s1 | 10 / 14 (slope 3 / 3) | +0.76 / +0.56 | +0.20 [-0.29, +0.70] | refuted | +2.9 / +2.6 | +0.3 [-2.1, +1.7] | refuted | +0.91 / +0.54 | - |
| held_RV2a-L18-s0 | 6 / 18 (slope 2 / 4) | +0.65 / +0.67 | -0.01 [-0.36, +0.69] | inconclusive (group < 3) | +3.0 / +3.4 | -0.4 [-2.1, +1.2] | inconclusive (group < 3) | +0.90 / +0.71 | - |
| held_RV2a-L18-s1 | 9 / 15 (slope 2 / 4) | +0.79 / +0.52 | +0.27 [-0.17, +0.64] | inconclusive (group < 3) | +4.2 / +2.7 | +1.5 [-1.3, +2.6] | inconclusive (group < 3) | +0.91 / +0.49 | - |
| held_RV2b-L18-s0 | 8 / 16 (slope 2 / 4) | +0.63 / +0.67 | -0.04 [-0.39, +0.69] | inconclusive (group < 3) | +3.9 / +3.4 | +0.5 [-0.8, +2.8] | inconclusive (group < 3) | +0.89 / +0.71 | - |
| held_RV2b-L18-s1 | 7 / 17 (slope 1 / 5) | +0.82 / +0.61 | +0.21 [-0.19, +0.63] | inconclusive (group < 3) | +4.8 / +3.1 | +1.7 [+0.1, +3.0] | inconclusive (group < 3) | +0.92 / +0.54 | - |
| held_pooled | 48 / 96 (slope 12 / 24) | +0.79 / +0.57 | +0.23 [-0.28, +0.67] | refuted | +3.4 / +2.9 | +0.5 [-1.6, +1.6] | refuted | +0.90 / +0.54 | - |

Descriptives (median failed / completed): onset visible share of approach decisions, d_e at first visibility (m), lag = s_on - route onset over d_e <= 15 m (m), unseen = share of decisions whose 4 s horizon reaches the route onset but the plan's onset is absent or > 5 m beyond it, approach speed (m/s), approach decisions.

| set | visible | d_e first visible | lag15 | unseen | v | n_app |
|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | +0.36 / +0.82 | +10.16 / +19.17 | -1.14 / -0.18 | +0.08 / +0.00 | +3.26 / +5.35 | +63.00 / +45.00 |
| pai_RV2a-L18-s0 | +0.37 / +0.82 | +11.56 / +6.97 | -0.64 / -2.00 | +0.05 / +0.00 | +3.24 / +3.35 | +60.00 / +45.00 |
| pai_RV2b-L18-s1 | +0.39 / +0.84 | +11.86 / +8.87 | -0.84 / -0.07 | +0.01 / +0.00 | +3.26 / +3.15 | +60.00 / +44.00 |
| pai_APP1X-L18-s0 | +0.37 / +0.63 | +11.70 / +8.87 | -1.31 / -0.53 | +0.00 / +0.00 | +2.50 / +2.87 | +73.00 / +53.00 |
| held_RSG-L18-s0 | +1.00 / +1.00 | +5.29 / +5.00 | -1.11 / -1.64 | +0.00 / +0.00 | +2.27 / +4.84 | +5.00 / +3.50 |
| held_RSG-L18-s1 | +1.00 / +1.00 | +5.29 / +5.00 | -1.07 / -1.63 | +0.00 / +0.00 | +3.35 / +3.97 | +5.00 / +3.50 |
| held_RV2a-L18-s0 | +1.00 / +1.00 | +5.75 / +5.00 | -1.15 / -1.68 | +0.00 / +0.00 | +3.34 / +3.97 | +5.00 / +4.50 |
| held_RV2a-L18-s1 | +1.00 / +1.00 | +5.00 / +5.00 | -1.06 / -1.71 | +0.00 / +0.00 | +4.37 / +3.57 | +5.00 / +4.00 |
| held_RV2b-L18-s0 | +1.00 / +1.00 | +5.29 / +5.00 | -1.07 / -1.64 | +0.00 / +0.00 | +2.27 / +4.84 | +5.50 / +3.50 |
| held_RV2b-L18-s1 | +1.00 / +1.00 | +5.00 / +5.00 | -0.98 / -1.72 | +0.00 / +0.00 | +2.23 / +4.41 | +5.00 / +4.00 |
| held_pooled | +1.00 / +1.00 | +5.29 / +5.00 | -1.10 / -1.67 | +0.00 / +0.00 | +2.32 / +3.97 | +5.00 / +4.00 |

Fingerprint by class (turns, turns with a slope; medians):

| set | class | n (slope) | slope | drift m | lag15 m | unseen | visible |
|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | completed | 5 (3) | +0.79 | +4.0 | -0.2 | +0.00 | +0.82 |
| pai_RSG-L18-s0 | late_wide | 12 (10) | +0.70 | +2.2 | -1.4 | +0.00 | +0.61 |
| pai_RSG-L18-s0 | straight | 8 (2) | +0.70 | +2.4 | +2.9 | +1.00 | +0.12 |
| pai_RSG-L18-s0 | other | 1 (1) | +1.25 | -6.0 | +2.7 | +0.18 | +0.82 |
| pai_RSG-L18-s0 | no_zero | 10 (8) | +0.53 | +6.1 | -2.0 | +0.03 | +0.33 |
| pai_all4 | completed | 28 (19) | +0.87 | +1.6 | -0.5 | +0.00 | +0.77 |
| pai_all4 | late_wide | 42 (37) | +0.61 | +3.0 | -1.4 | +0.00 | +0.59 |
| pai_all4 | straight | 26 (9) | +0.67 | +2.6 | +2.5 | +1.00 | +0.12 |
| pai_all4 | other | 5 (4) | +1.22 | -5.5 | +2.1 | +0.18 | +0.82 |
| pai_all4 | no_zero | 43 (31) | +0.54 | +5.6 | -1.8 | +0.00 | +0.34 |
| held_pooled | completed | 96 (24) | +0.57 | +2.9 | -1.7 | +0.00 | +1.00 |
| held_pooled | late_wide | 6 (0) | - | - | -1.0 | +0.00 | +1.00 |
| held_pooled | no_zero | 42 (12) | +0.79 | +3.4 | -1.1 | +0.00 | +1.00 |
| over0_d299 | over_turn | 11 (7) | +0.72 | +2.1 | -1.9 | +0.00 | +0.50 |

C3: plan heading change over 4 s toward the turn (deg, median) by distance to the entry; human log in brackets:

| set | class | 30-40 m | 20-30 m | 10-20 m | 0-10 m |
|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | completed | +9.0 (+5.0, n 10) | +0.0 (+5.5, n 95) | +33.4 (+50.4, n 52) | +53.8 (+71.3, n 74) |
| pai_RSG-L18-s0 | late_wide | +2.7 (+3.3, n 95) | +14.8 (+15.7, n 52) | +5.3 (+27.6, n 173) | +51.0 (+56.2, n 416) |
| pai_RSG-L18-s0 | straight | +0.3 (+1.6, n 109) | +0.4 (+1.9, n 151) | +1.9 (+44.2, n 118) | +5.6 (+74.0, n 117) |
| pai_RSG-L18-s0 | other | +10.0 (-20.9, n 7) | +41.7 (-19.9, n 5) | +62.7 (+1.4, n 9) | +65.7 (+45.8, n 1) |
| pai_RSG-L18-s0 | no_zero | +1.1 (+0.2, n 131) | +1.0 (+3.1, n 176) | +6.1 (+27.6, n 172) | +42.8 (+43.8, n 339) |
| pai_all4 | completed | +2.5 (+3.4, n 114) | +0.6 (+6.7, n 253) | +15.1 (+28.1, n 279) | +46.2 (+46.7, n 788) |
| pai_all4 | late_wide | +1.8 (+3.3, n 307) | +11.5 (+14.1, n 236) | +7.1 (+27.6, n 559) | +51.6 (+57.4, n 1427) |
| pai_all4 | straight | +0.4 (+1.6, n 293) | +0.1 (+1.9, n 494) | +3.0 (+48.1, n 363) | +8.8 (+75.6, n 383) |
| pai_all4 | other | +2.6 (+3.3, n 97) | +21.4 (+16.1, n 40) | +59.4 (+27.9, n 44) | +61.9 (+69.1, n 20) |
| pai_all4 | no_zero | +0.5 (+0.5, n 619) | +0.4 (+3.2, n 1057) | +2.5 (+28.4, n 961) | +38.8 (+60.6, n 1177) |
| held_all6 | completed | - (-, n 0) | - (-, n 0) | +56.6 (+53.4, n 24) | +72.2 (+72.0, n 352) |
| held_all6 | late_wide | - (-, n 0) | - (-, n 0) | - (-, n 0) | +75.4 (+81.9, n 24) |
| held_all6 | no_zero | - (-, n 0) | - (-, n 0) | +45.3 (+42.0, n 30) | +61.9 (+62.9, n 206) |

C4: inside the turn (entry .. exit + 10 m). past = decisions with the ego at / past the exit heading; more = share of those whose plan still asks > +5 deg into the turn; excess = median (plan dpsi - true remaining heading), human in brackets; OLS slope of plan on true remaining.

| set | class | decisions (turns) | past (turns) | more | plan dpsi past | excess (human) | OLS |
|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | completed | 214 (5) | 15 (2) | +0.00 | -4.4 | +0.9 (-1.2) | +0.93 |
| pai_RSG-L18-s0 | late_wide | 355 (12) | 0 (0) | - | - | -1.6 (-9.0) | +0.49 |
| pai_RSG-L18-s0 | straight | 28 (6) | 2 (1) | +1.00 | +7.1 | -81.1 (-26.8) | +0.06 |
| pai_RSG-L18-s0 | no_zero | 384 (8) | 0 (0) | - | - | +0.2 (-5.6) | +0.84 |
| pai_all4 | completed | 1341 (14) | 53 (3) | +0.00 | -1.5 | +0.3 (-0.9) | +0.99 |
| pai_all4 | late_wide | 1063 (16) | 0 (0) | - | - | -3.0 (-9.9) | +0.42 |
| pai_all4 | straight | 224 (8) | 6 (1) | +1.00 | +7.2 | -60.2 (-6.8) | +0.03 |
| pai_all4 | other | 68 (1) | 0 (0) | - | - | -0.0 (-10.0) | +1.03 |
| pai_all4 | no_zero | 1519 (16) | 24 (2) | +0.54 | +5.3 | +0.7 (-4.6) | +0.70 |
| held_all6 | completed | 921 (19) | 23 (4) | +0.09 | -3.9 | -0.8 (-0.4) | +0.86 |
| held_all6 | late_wide | 36 (1) | 0 (0) | - | - | +5.8 (-16.1) | +0.76 |
| held_all6 | no_zero | 518 (10) | 44 (7) | +0.02 | -9.0 | +0.1 (-3.1) | +0.93 |
| over0_d299 | over_turn | 22 (5) | 0 (0) | - | - | -5.4 (+3.7) | +1.13 |

APP1X vs RSG (PAI): median slope, paired median difference APP1X - RSG [CI], drift medians, completed:

- runway22 (n 22): slope +0.72 vs +0.74, paired +0.04 [-0.17, +0.16] (n 12); drift +3.0 vs +3.3 m; completed 4 vs 3
- strict36 (n 36): slope +0.72 vs +0.67, paired +0.05 [-0.06, +0.18] (n 21); drift +3.1 vs +3.5 m; completed 11 vs 5

Per turn (pai_RSG-L18-s0):

| scene | side deg | outcome | class (src) | start to entry m | n app | slope | drift | human slope | visible | lag15 | unseen |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 22fa5c92 | L +91 | event | late_wide (rule) | 9.0 | 72 | +0.31 | +6.2 | +0.66 | +0.74 | -3.2 | +0.00 |
| 33c8976b | L +107 | event | late_wide (tax) | 5.0 | 34 | - | - | - | +1.00 | -3.3 | +0.00 |
| 37d660b1 | R -85 | event | late_wide (tax) | 21.9 | 53 | +1.05 | -0.7 | +0.95 | +0.57 | +0.9 | +0.12 |
| 3ce13eed | R -100 | event | late_wide (tax) | 16.9 | 59 | +0.79 | +1.6 | +0.98 | +0.53 | +0.4 | +0.23 |
| 68e53160 | L +108 | event | late_wide (tax) | 12.0 | 74 | +0.48 | +5.8 | +1.05 | +0.51 | -2.3 | +0.00 |
| 94c1d722 | R -86 | event | late_wide (tax) | 77.3 | 112 | +0.99 | +0.1 | +0.93 | +0.29 | -1.3 | +0.00 |
| 9f393c8a | R -84 | event | late_wide (rule) | 6.8 | 12 | +1.14 | -0.8 | +0.42 | +1.00 | +0.6 | +0.00 |
| a1b166cc | L +91 | event | late_wide (tax) | 10.0 | 76 | +0.61 | +2.7 | +0.81 | +0.33 | -1.9 | +0.07 |
| ce161092 | L +85 | event | late_wide (tax) | 53.0 | 50 | +1.00 | -0.1 | +0.92 | +0.66 | -1.4 | +0.03 |
| ddc3e8df | R -66 | event | late_wide (tax) | 2.9 | 12 | - | - | - | +1.00 | -0.7 | +0.00 |
| e37aa159 | L +82 | event | late_wide (tax) | 37.2 | 129 | +0.49 | +13.4 | +0.61 | +0.38 | +1.8 | +0.00 |
| fd01e9ab | L +98 | event | late_wide (tax) | 7.0 | 53 | +0.36 | +4.7 | +0.14 | +0.79 | -2.0 | +0.00 |
| 09528162 | R -100 | off | no_zero (-) | 28.0 | 46 | +0.54 | +11.6 | +0.77 | +0.83 | -1.1 | +0.14 |
| 268949d2 | R -88 | off | no_zero (-) | 6.9 | 63 | +0.45 | +3.6 | - | +0.38 | -0.3 | +0.09 |
| 6fd8cf30 | R -88 | short_rear | no_zero (-) | 42.0 | 176 | +0.19 | +8.0 | +0.91 | +0.29 | -2.0 | +0.00 |
| a4d2f7cf | L +110 | short_slow | no_zero (-) | 73.8 | 123 | +0.76 | +3.3 | +0.97 | +0.32 | -2.9 | +0.00 |
| abd45a30 | L +90 | short_rear | no_zero (-) | 119.6 | 106 | - | - | +1.00 | +0.07 | -0.2 | +0.22 |
| b40f4729 | R -89 | off | no_zero (-) | 8.9 | 34 | +0.53 | +6.3 | - | +1.00 | +0.7 | +0.03 |
| c4a58f83 | R -85 | short_slow | no_zero (-) | 230.3 | 0 | - | - | - | - | - | - |
| e0f6ebe1 | R -90 | off | no_zero (-) | 70.4 | 64 | +0.67 | +6.9 | +1.01 | +0.44 | -2.7 | +0.20 |
| e848c843 | R -90 | off | no_zero (-) | 56.6 | 110 | +0.47 | +5.9 | +0.87 | +0.33 | -2.6 | +0.00 |
| ea468790 | R -83 | off | no_zero (-) | 11.0 | 96 | +0.88 | +0.8 | +0.23 | +0.32 | -2.1 | +0.00 |
| 1e8e939e | R -49 | event | other (rule) | 101.4 | 22 | +1.25 | -6.0 | - | +0.82 | +2.7 | +0.18 |
| 2f227f2c | R -66 | event | straight (tax) | 30.0 | 67 | - | - | +0.92 | +0.00 | - | +1.00 |
| 302c5c99 | L +163 | event | straight (tax) | 33.9 | 75 | +0.67 | +2.9 | +0.95 | +0.35 | +2.0 | +0.46 |
| 3232fdb7 | R -87 | event | straight (tax) | 89.1 | 125 | - | - | - | +0.14 | +2.9 | +0.11 |
| 403d558a | R -93 | event | straight (tax) | 36.5 | 36 | - | - | +0.98 | +0.11 | - | +1.00 |
| a174faed | R -90 | event | straight (tax) | 41.9 | 81 | +0.74 | +1.8 | +0.99 | +0.35 | +3.4 | +0.42 |
| e5e47fd2 | L +95 | event | straight (tax) | 54.6 | 32 | - | - | +0.98 | +0.12 | - | +1.00 |
| eaba3f6a | R -90 | event | straight (tax) | 53.6 | 39 | - | - | +0.73 | +0.00 | - | +1.00 |
| f9e361aa | R -81 | event | straight (tax) | 57.7 | 40 | - | - | +0.97 | +0.00 | - | +1.00 |
| 132ae6cb | L +86 | completed | completed (-) | 26.0 | 105 | +0.42 | +11.5 | +0.87 | +0.39 | -3.7 | +0.00 |
| 382a7247 | L +83 | completed | completed (-) | 27.9 | 51 | +0.79 | +4.0 | +0.93 | +0.57 | +1.8 | +0.12 |
| 3c2822d9 | L +81 | completed | completed (-) | 4.0 | 10 | - | - | - | +1.00 | -2.1 | +0.00 |
| 8a143cb2 | R -76 | completed | completed (-) | 2.7 | 20 | - | - | - | +1.00 | +0.6 | +0.00 |
| d90fe123 | L +85 | completed | completed (-) | 94.8 | 45 | +0.94 | +1.6 | +0.89 | +0.82 | -0.2 | +0.00 |
