# Corridor check: registered read and the Amendment 4 sweep (exploratory)

Hard exit = a footprint point of a 1-4 s pose in the oncoming lane or off-road outside the route-lane corridor. G-H = the human's own 1-4 s footprint in the strict-turn windows (line <= 0.05); C1 = share of inside / oncoming / over-turn turn zeros with a hard-exit plan at t_f - 1 / 2 / 3 s (line >= 0.70); C2 = share of completed-strict-turn window decisions with a hard-exit plan (line <= 0.10). Held = the six held 460 sets pooled (scene clusters); PAI = RSG-L18-s0 on PAI 281. C1 held is the pooled hit count (CI in each check.md).

| variant | G-H held | G-H PAI | C1 held | C1 PAI | C2 held | C2 PAI |
|:--|:--|:--|:--|:--|:--|:--|
| registered (margin 0.3 m, box) | 0.13 [0.04, 0.22] | 0.33 [0.22, 0.45] | 0.84 (54 / 64) | 1.00 (6 / 6) | 0.32 [0.19, 0.46] | 0.56 [0.38, 0.74] |
| Amendment 4: m0.6 | 0.06 [0.00, 0.13] | 0.25 [0.16, 0.35] | 0.78 (50 / 64) | 1.00 (6 / 6) | 0.25 [0.13, 0.38] | 0.52 [0.33, 0.71] |
| Amendment 4: m1.0 | 0.01 [0.00, 0.05] | 0.20 [0.11, 0.29] | 0.69 (44 / 64) | 0.83 (5 / 6) | 0.18 [0.08, 0.30] | 0.46 [0.26, 0.68] |
| Amendment 4: m1.5 | 0.01 [0.00, 0.03] | 0.13 [0.07, 0.20] | 0.50 (32 / 64) | 0.83 (5 / 6) | 0.12 [0.04, 0.21] | 0.38 [0.19, 0.58] |
| Amendment 4: axle0.3 | 0.01 [0.00, 0.03] | 0.12 [0.06, 0.19] | 0.48 (31 / 64) | 0.83 (5 / 6) | 0.12 [0.04, 0.22] | 0.36 [0.17, 0.56] |
