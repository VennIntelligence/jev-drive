# Route-lane corridor check (rf_corridor.py check)

Prereg `plans/2026-10-15-corridor-check-prereg.md`. Plan hard exit = a footprint point of a served 1-4 s pose in the oncoming lane or off the road (outside the route-lane corridor + 0.3 m); probes = the decisions at t_f - 1 / 2 / 3 s. Shares with scene-clustered bootstrap 95 % CI (B = 10 000).

**C1 (>= 0.70 of inside / oncoming / over-turn turn zeros have a hard-exit plan at t_f - 1 s or earlier): 0.48 [0.32, 0.64], 31 / 64 defined (0 without a probe) -> fails (robust).**

**C2 (<= 0.10 of the completed strict turns' window decisions have a hard-exit plan): 0.12 [0.04, 0.22] over 1160 decisions of 86 turns -> fails.**

**G-H (label sanity: the human's own 1-4 s footprint in the strict-turn windows, hard exit <= 0.05): held 0.01 [0.00, 0.03] (334 decisions) -> pass; pai 0.12 [0.06, 0.19] (3364 decisions) -> FAIL, labels void**. Void sources: ['pai'].

## G-H detail (human)

| source / population | hard exit | oncoming | off-road | soft only | rear axle inside the corridor |
|:--|:--|:--|:--|:--|:--|
| held | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.14] | 0.96 [0.93, 0.99] |
| pai | 0.12 [0.06, 0.19] | 0.03 [0.00, 0.06] | 0.10 [0.04, 0.16] | 0.08 [0.02, 0.15] | 0.92 [0.86, 0.96] |
| held (all decisions) | 0.01 [0.00, 0.01] | - | - | - | - |
| pai (all decisions) | 0.03 [0.02, 0.04] | - | - | - | - |

## C1 / C3: turn zeros, plan hard exit at a probe

| population | hit share [CI] | hits / defined | no probe |
|:--|:--|--:|--:|
| C1 pooled | 0.48 [0.32, 0.64] | 31 / 64 | 0 |
| C1 held_APP1X_s0 | 0.52 [0.33, 0.71] | 11 / 21 | 0 |
| C1 held_RBSG2_s0 | 0.11 [0.00, 0.33] | 1 / 9 | 0 |
| C1 held_RSG_s0 | 1.00 [1.00, 1.00] | 6 / 6 | 0 |
| C1 held_RSG_s1 | 1.00 [1.00, 1.00] | 4 / 4 | 0 |
| C1 held_RV2c_s0 | 0.36 [0.09, 0.64] | 4 / 11 | 0 |
| C1 held_RV2c_s1 | 0.38 [0.15, 0.62] | 5 / 13 | 0 |
| C1 pai | 0.83 [0.50, 1.00] | 5 / 6 | 0 |
| C1 class over_turn | 0.44 [0.26, 0.60] | 27 / 61 | 0 |
| C1 class corridor + oncoming touch (not over_turn) | 1.00 [1.00, 1.00] | 9 / 9 | 0 |
| C1 all sources (descriptive, ignores G-H) | 0.51 [0.36, 0.66] | 36 / 70 | 0 |
| C3 class lane_change | 1.00 [1.00, 1.00] | 1 / 1 | 0 |
| C3 class lane_change|pai | 1.00 [1.00, 1.00] | 1 / 1 | 0 |
| C3 class late_wide | 0.66 [0.52, 0.78] | 54 / 82 | 0 |
| C3 class late_wide|held_APP1X_s0 | 0.42 [0.26, 0.58] | 13 / 31 | 0 |
| C3 class late_wide|held_RBSG2_s0 | 0.75 [0.25, 1.00] | 3 / 4 | 0 |
| C3 class late_wide|held_RSG_s0 | 0.83 [0.50, 1.00] | 5 / 6 | 0 |
| C3 class late_wide|held_RSG_s1 | 1.00 [1.00, 1.00] | 7 / 7 | 0 |
| C3 class late_wide|held_RV2c_s0 | 0.88 [0.62, 1.00] | 7 / 8 | 0 |
| C3 class late_wide|held_RV2c_s1 | 0.75 [0.38, 1.00] | 6 / 8 | 0 |
| C3 class late_wide|pai | 0.72 [0.50, 0.89] | 13 / 18 | 0 |
| C3 class other | 0.67 [0.33, 1.00] | 4 / 6 | 0 |
| C3 class other|held_APP1X_s0 | 0.00 [0.00, 0.00] | 0 / 1 | 0 |
| C3 class other|held_RBSG2_s0 | 1.00 [1.00, 1.00] | 1 / 1 | 0 |
| C3 class other|held_RV2c_s1 | 1.00 [1.00, 1.00] | 2 / 2 | 0 |
| C3 class other|pai | 0.50 [0.00, 1.00] | 1 / 2 | 0 |
| C3 class route_end | 0.50 [0.00, 1.00] | 2 / 4 | 0 |
| C3 class route_end|held_RBSG2_s0 | 0.00 [0.00, 0.00] | 0 / 2 | 0 |
| C3 class route_end|held_RSG_s0 | 1.00 [1.00, 1.00] | 1 / 1 | 0 |
| C3 class route_end|held_RSG_s1 | 1.00 [1.00, 1.00] | 1 / 1 | 0 |
| C3 class straight | 0.61 [0.42, 0.76] | 19 / 31 | 0 |
| C3 class straight|held_APP1X_s0 | 0.50 [0.00, 1.00] | 2 / 4 | 0 |
| C3 class straight|held_RBSG2_s0 | 0.75 [0.25, 1.00] | 3 / 4 | 0 |
| C3 class straight|held_RSG_s0 | 0.67 [0.00, 1.00] | 2 / 3 | 0 |
| C3 class straight|held_RSG_s1 | 0.75 [0.25, 1.00] | 3 / 4 | 0 |
| C3 class straight|held_RV2c_s0 | 1.00 [1.00, 1.00] | 2 / 2 | 0 |
| C3 class straight|held_RV2c_s1 | 1.00 [1.00, 1.00] | 2 / 2 | 0 |
| C3 class straight|pai | 0.42 [0.17, 0.67] | 5 / 12 | 0 |
| C3 class wrong_exit | 0.00 [0.00, 0.00] | 0 / 1 | 0 |
| C3 class wrong_exit|pai | 0.00 [0.00, 0.00] | 0 / 1 | 0 |

Secondary (not registered):

- c1 hit_win: 0.51 [0.36, 0.66] (36 / 70)
- c1 hard_1: 0.41 [0.30, 0.53] (29 / 70)
- c1 hard_2: 0.37 [0.22, 0.53] (26 / 70)
- c1 hard_3: 0.40 [0.21, 0.59] (24 / 60)
- c2 completed turns with any hard decision: {"n": 91, "k": 35}

## C2: completed strict turns, window decisions

| set | hard exit [CI] | decisions | turns | oncoming | off-road | soft only |
|:--|:--|--:|--:|--:|--:|--:|
| pooled | 0.12 [0.04, 0.22] | 1160 | 86 | | | |
| held_APP1X_s0 | 0.06 [0.00, 0.16] | 107 | 8 | 0.06 | 0.00 | 0.26 |
| held_RBSG2_s0 | 0.07 [0.00, 0.15] | 230 | 17 | 0.07 | 0.00 | 0.04 |
| held_RSG_s0 | 0.16 [0.03, 0.32] | 216 | 16 | 0.16 | 0.00 | 0.16 |
| held_RSG_s1 | 0.13 [0.01, 0.27] | 188 | 14 | 0.12 | 0.01 | 0.09 |
| held_RV2c_s0 | 0.15 [0.05, 0.26] | 198 | 15 | 0.15 | 0.00 | 0.09 |
| held_RV2c_s1 | 0.15 [0.06, 0.26] | 221 | 16 | 0.15 | 0.00 | 0.10 |
| pai | 0.36 [0.17, 0.56] | 340 | 5 | 0.02 | 0.34 | 0.27 |
| all sources (descriptive, ignores G-H) | 0.18 [0.09, 0.27] | 1500 | 91 | | | |
| failed strict turns (all window decisions)|held_APP1X_s0 | 0.18 [0.07, 0.30] | 254 | 16 | | | |
| failed strict turns (all window decisions)|held_RBSG2_s0 | 0.01 [0.00, 0.03] | 117 | 7 | | | |
| failed strict turns (all window decisions)|held_RSG_s0 | 0.06 [0.00, 0.17] | 144 | 8 | | | |
| failed strict turns (all window decisions)|held_RSG_s1 | 0.13 [0.03, 0.26] | 179 | 10 | | | |
| failed strict turns (all window decisions)|held_RV2c_s0 | 0.17 [0.03, 0.38] | 139 | 9 | | | |
| failed strict turns (all window decisions)|held_RV2c_s1 | 0.20 [0.03, 0.44] | 115 | 8 | | | |
| failed strict turns (all window decisions)|pai | 0.39 [0.28, 0.51] | 3415 | 30 | | | |

## Turn zeros (per zero)

| set | scene | class | zero type | C1 | onc touch | hit | t_f-1 | t_f-2 | t_f-3 | first hard pose (s) at t_f-1 / -2 / -3 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| held_APP1X_s0 | c1e1661e600c570e | late_wide | corridor | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.0 |
| held_APP1X_s0 | 62c5045db3c159aa | late_wide | corridor | C1 | True | H | H | H | H | 1.0 / 1.5 / 3.5 |
| held_APP1X_s0 | ed8294474bc35b6c | late_wide | corridor | C1 | True | H | . | H | . | - / 1.0 / - |
| held_APP1X_s0 | f3512ae175b45844 | late_wide | corridor | C1 | True | H | H | H | H | 3.5 / 1.0 / 1.0 |
| held_APP1X_s0 | fe4e5fdf88195cfa | over_turn | offroad | C1 | False | H | H | . | - | 2.0 / - / - |
| held_APP1X_s0 | 8d90099a801d5682 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 53d3787367c75240 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 41da8ea7c14754d2 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | e14f1ffbe6f856e8 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 83e895f722575452 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 47da19c4edbe5b95 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 365c0937de9d5885 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 6f393be3019c508a | over_turn | offroad | C1 | True | H | . | . | H | - / - / 1.0 |
| held_APP1X_s0 | 02015675e4585611 | over_turn | offroad | C1 | False | H | H | H | - | 3.0 / 4.0 / - |
| held_APP1X_s0 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | H | H | H | 2.0 / 1.0 / 1.5 |
| held_APP1X_s0 | fc0a60a9d8245aea | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 78c26c7e63c3534c | over_turn | offroad | C1 | False | H | H | . | - | 3.5 / - / - |
| held_APP1X_s0 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | H | H | . | . | 2.5 / - / - |
| held_APP1X_s0 | f082cf21642c5cbc | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 8c9e3fd944c051e3 | over_turn | offroad | C1 | False | H | H | H | . | 1.0 / 2.5 / - |
| held_APP1X_s0 | 8d3ac506e17050ba | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | fe4e5fdf88195cfa | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | 5b61b9072cad5dbc | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | e14f1ffbe6f856e8 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | 06fbdeb141965cca | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | . | H | H | - / 1.0 / 1.0 |
| held_RBSG2_s0 | 78c26c7e63c3534c | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | . | . | . | - | - / - / - |
| held_RBSG2_s0 | 8c9e3fd944c051e3 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | b628faa4178d5eaf | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RSG_s0 | e6056b57c0515735 | late_wide | corridor | C1 | True | H | H | H | H | 2.0 / 1.0 / 1.0 |
| held_RSG_s0 | b5155b952c645ce0 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.0 |
| held_RSG_s0 | 02015675e4585611 | over_turn | offroad | C1 | False | H | . | . | H | - / - / 3.5 |
| held_RSG_s0 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | . | H | H | - / 1.0 / 2.0 |
| held_RSG_s0 | a3dd6d539acc5c9b | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.0 |
| held_RSG_s0 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | H | H | . | . | 3.0 / - / - |
| held_RSG_s1 | b5155b952c645ce0 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 1.5 |
| held_RSG_s1 | 02015675e4585611 | over_turn | offroad | C1 | False | H | . | H | H | - / 4.0 / 3.5 |
| held_RSG_s1 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | . | H | H | - / 1.0 / 2.0 |
| held_RSG_s1 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | H | H | H | . | 2.5 / 3.0 / - |
| held_RV2c_s0 | 02015675e4585611 | over_turn | offroad | C1 | False | H | H | . | - | 4.0 / - / - |
| held_RV2c_s0 | bd70654aee0f54a6 | over_turn | offroad | C1 | False | H | H | H | H | 4.0 / 3.0 / 1.0 |
| held_RV2c_s0 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 1.5 |
| held_RV2c_s0 | fde7abb615895be2 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.5 |
| held_RV2c_s0 | b9c124bdb19956b7 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s0 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | . | . | . | - | - / - / - |
| held_RV2c_s0 | e14711ef4d2f5000 | over_turn | offroad | C1 | False | . | . | . | - | - / - / - |
| held_RV2c_s0 | f082cf21642c5cbc | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s0 | 8c9e3fd944c051e3 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s0 | b628faa4178d5eaf | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s0 | e8850900bade58f6 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | a3dd6d539acc5c9b | late_wide | corridor | C1 | True | H | H | H | H | 1.0 / 1.0 / 3.5 |
| held_RV2c_s1 | fe4e5fdf88195cfa | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | c2ae8e85ce2052ff | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | 02015675e4585611 | over_turn | offroad | C1 | False | . | . | . | - | - / - / - |
| held_RV2c_s1 | bd70654aee0f54a6 | over_turn | offroad | C1 | True | H | H | H | H | 3.5 / 3.0 / 1.0 |
| held_RV2c_s1 | c687799dac8d5aa2 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.0 |
| held_RV2c_s1 | fde7abb615895be2 | over_turn | offroad | C1 | True | H | H | H | H | 1.0 / 1.0 / 2.5 |
| held_RV2c_s1 | b9c124bdb19956b7 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | be6f90f06bc35be3 | over_turn | offroad | C1 | False | . | . | . | - | - / - / - |
| held_RV2c_s1 | e14711ef4d2f5000 | over_turn | offroad | C1 | False | H | H | . | - | 4.0 / - / - |
| held_RV2c_s1 | f082cf21642c5cbc | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | 8c9e3fd944c051e3 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | b628faa4178d5eaf | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| pai | b680a689 | late_wide | corridor | C1 | True | H | H | H | H | 1.0 / 1.0 / 1.0 |
| pai | ddc3e8df | late_wide | corridor | C1 | True | H | H | H | H | 1.0 / 1.0 / 1.0 |
| pai | ba62e10d | other | corridor | C1 | True | H | H | H | H | 1.0 / 1.0 / 1.0 |
| pai | 2466266e | over_turn | offroad | C1 | False | H | H | . | . | 3.5 / - / - |
| pai | 300465f7 | over_turn | offroad | C1 | False | . | . | . | . | - / - / - |
| pai | d8bfaaf2 | over_turn | offroad | C1 | False | H | H | . | . | 3.0 / - / - |
| held_APP1X_s0 | 8eda03e314a456a4 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 013d05a439b95210 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 4495218e41b35f25 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | fcb45b2aa29356d9 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 65729e98f81a526e | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 9bfa838de21f5d25 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | d484aff53cd1589f | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | ba6136583b6254e2 | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 0c38a8ace1f5548e | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 5816a23cfee25d4e | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | daf38cecd5045cf6 | late_wide | offroad |  | True | H | H | H | H | 1.0 / 2.5 / 1.0 |
| held_APP1X_s0 | ae878998fecb50a5 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 13cd1cf21a4f5acb | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | c7a44a2e52bc5e22 | late_wide | offroad |  | True | H | . | . | H | - / - / 4.0 |
| held_APP1X_s0 | e9cfc98fdc09590a | late_wide | offroad |  | True | H | H | . | . | 2.0 / - / - |
| held_APP1X_s0 | a3c4e4bf10d65ddb | late_wide | collision |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | b5155b952c645ce0 | late_wide | corridor |  | False | H | H | . | . | 3.0 / - / - |
| held_APP1X_s0 | 7900996a42f35f67 | late_wide | offroad |  | True | H | H | H | . | 2.0 / 1.5 / - |
| held_APP1X_s0 | 4faa4706a50958e2 | late_wide | corridor |  | False | H | H | H | H | 1.5 / 2.5 / 3.5 |
| held_APP1X_s0 | a0be9e4c6cc15ec7 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 5a3a4277dc785511 | late_wide | corridor |  | False | H | H | . | . | 1.5 / - / - |
| held_APP1X_s0 | d7b1349fb4775cc1 | late_wide | corridor |  | False | H | H | H | H | 1.5 / 2.5 / 3.5 |
| held_APP1X_s0 | ddc1271ea57154bc | late_wide | offroad |  | False | H | H | . | . | 4.0 / - / - |
| held_APP1X_s0 | e14711ef4d2f5000 | late_wide | offroad |  | False | H | H | . | . | 2.0 / - / - |
| held_APP1X_s0 | 409711b03072566a | late_wide | offroad |  | False | . | . | . | - | - / - / - |
| held_APP1X_s0 | 7ae0a03be0c357d2 | late_wide | corridor |  | False | H | H | H | H | 1.0 / 1.0 / 1.5 |
| held_APP1X_s0 | b628faa4178d5eaf | late_wide | corridor |  | False | H | . | H | H | - / 3.5 / 4.0 |
| held_APP1X_s0 | 0fa603b5789956f1 | late_wide | corridor |  | False | H | H | H | H | 1.0 / 1.5 / 3.0 |
| held_APP1X_s0 | 69e8a01f073d5080 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 77424ed925b45c77 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | e072351fbbfd5765 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 330166def5a35f4d | other | collision |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | bf00c76f25185b83 | straight | offroad |  | False | . | . | . | . | - / - / - |
| held_APP1X_s0 | 7272f8ee7e64591e | straight | offroad |  | False | H | H | . | . | 2.0 / - / - |
| held_APP1X_s0 | f9ed38d9ddfa531e | straight | offroad |  | False | H | H | . | - | 1.0 / - / - |
| held_APP1X_s0 | 20728d3e677b593d | straight | offroad |  | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | 661ce644db6d5546 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | ddc1271ea57154bc | late_wide | offroad |  | False | H | H | H | - | 1.0 / 2.5 / - |
| held_RBSG2_s0 | 078bc1027dde5d1a | late_wide | offroad |  | False | H | H | H | . | 1.0 / 2.5 / - |
| held_RBSG2_s0 | 3b4031def0f45d96 | late_wide | offroad |  | False | H | H | . | . | 1.5 / - / - |
| held_RBSG2_s0 | bbb7e4525bd05d3d | other | offroad |  | False | H | H | . | - | 3.5 / - / - |
| held_RBSG2_s0 | 94652fa0c64d5846 | route_end | collision |  | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | e511376fdf3250e5 | route_end | collision |  | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | bd70654aee0f54a6 | straight | offroad |  | False | H | H | H | H | 3.0 / 4.0 / 1.0 |
| held_RBSG2_s0 | f9ed38d9ddfa531e | straight | offroad |  | False | H | H | H | - | 1.0 / 2.0 / - |
| held_RBSG2_s0 | 02b68b9cc51f506a | straight | offroad |  | False | . | . | . | . | - / - / - |
| held_RBSG2_s0 | 20728d3e677b593d | straight | offroad |  | False | H | . | H | H | - / 2.5 / 3.0 |
| held_RSG_s0 | 5b61b9072cad5dbc | late_wide | offroad |  | False | H | . | H | . | - / 2.5 / - |
| held_RSG_s0 | a3c4e4bf10d65ddb | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_RSG_s0 | 4faa4706a50958e2 | late_wide | corridor |  | False | H | H | H | H | 1.5 / 2.0 / 3.0 |
| held_RSG_s0 | d7b1349fb4775cc1 | late_wide | corridor |  | False | H | H | H | . | 2.5 / 4.0 / - |
| held_RSG_s0 | e14711ef4d2f5000 | late_wide | offroad |  | False | H | H | H | . | 1.0 / 2.0 / - |
| held_RSG_s0 | 3b4031def0f45d96 | late_wide | offroad |  | False | H | H | H | . | 1.0 / 2.0 / - |
| held_RSG_s0 | 47e0ff3bc36a5f69 | route_end | collision |  | True | H | . | H | H | - / 1.0 / 1.5 |
| held_RSG_s0 | f9ed38d9ddfa531e | straight | offroad |  | False | . | . | . | . | - / - / - |
| held_RSG_s0 | b628faa4178d5eaf | straight | corridor |  | False | H | H | . | . | 4.0 / - / - |
| held_RSG_s0 | 20728d3e677b593d | straight | offroad |  | False | H | H | H | H | 1.0 / 1.5 / 3.0 |
| held_RSG_s1 | ae878998fecb50a5 | late_wide | corridor |  | False | H | H | H | . | 2.5 / 3.5 / - |
| held_RSG_s1 | e6056b57c0515735 | late_wide | collision |  | False | H | H | H | H | 1.0 / 1.5 / 3.0 |
| held_RSG_s1 | a3dd6d539acc5c9b | late_wide | corridor |  | False | H | H | H | H | 1.5 / 2.0 / 3.0 |
| held_RSG_s1 | 4faa4706a50958e2 | late_wide | corridor |  | False | H | H | H | H | 2.0 / 3.0 / 3.5 |
| held_RSG_s1 | e14711ef4d2f5000 | late_wide | offroad |  | False | H | H | H | . | 1.0 / 2.0 / - |
| held_RSG_s1 | 0fa603b5789956f1 | late_wide | corridor |  | False | H | H | H | H | 1.0 / 2.0 / 3.5 |
| held_RSG_s1 | 3b4031def0f45d96 | late_wide | offroad |  | False | H | H | H | . | 1.0 / 2.0 / - |
| held_RSG_s1 | 47e0ff3bc36a5f69 | route_end | collision |  | True | H | . | H | H | - / 1.0 / 1.5 |
| held_RSG_s1 | d7b1349fb4775cc1 | straight | corridor |  | False | H | H | H | . | 3.0 / 4.0 / - |
| held_RSG_s1 | f9ed38d9ddfa531e | straight | offroad |  | False | . | . | . | . | - / - / - |
| held_RSG_s1 | b628faa4178d5eaf | straight | corridor |  | False | H | H | H | . | 3.0 / 4.0 / - |
| held_RSG_s1 | 20728d3e677b593d | straight | offroad |  | False | H | . | H | H | - / 2.5 / 3.0 |
| held_RV2c_s0 | 5fe06174763b5c36 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_RV2c_s0 | cebd7d74024b5cce | late_wide | collision |  | True | H | . | H | H | - / 1.0 / 2.0 |
| held_RV2c_s0 | 47e0ff3bc36a5f69 | late_wide | collision |  | False | H | H | . | . | 1.5 / - / - |
| held_RV2c_s0 | 9a675656c3c85f4c | late_wide | collision |  | False | H | H | . | H | 2.5 / - / 3.0 |
| held_RV2c_s0 | a3dd6d539acc5c9b | late_wide | corridor |  | False | H | H | H | H | 1.0 / 1.0 / 3.5 |
| held_RV2c_s0 | ddc1271ea57154bc | late_wide | offroad |  | False | H | H | H | H | 3.5 / 4.0 / 4.0 |
| held_RV2c_s0 | 1f54430427975391 | late_wide | offroad |  | False | H | H | H | H | 2.0 / 3.0 / 3.5 |
| held_RV2c_s0 | 3b4031def0f45d96 | late_wide | offroad |  | False | H | H | H | . | 1.5 / 2.5 / - |
| held_RV2c_s0 | f9ed38d9ddfa531e | straight | offroad |  | False | H | . | H | - | - / 2.0 / - |
| held_RV2c_s0 | 20728d3e677b593d | straight | offroad |  | False | H | H | H | . | 1.0 / 2.0 / - |
| held_RV2c_s1 | 5fe06174763b5c36 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | a3c4e4bf10d65ddb | late_wide | collision |  | False | . | . | . | . | - / - / - |
| held_RV2c_s1 | cebd7d74024b5cce | late_wide | collision |  | True | H | . | H | H | - / 1.0 / 2.0 |
| held_RV2c_s1 | 47e0ff3bc36a5f69 | late_wide | collision |  | False | H | H | . | . | 1.5 / - / - |
| held_RV2c_s1 | 9a675656c3c85f4c | late_wide | collision |  | False | H | H | H | H | 2.5 / 2.0 / 3.0 |
| held_RV2c_s1 | ddc1271ea57154bc | late_wide | offroad |  | False | H | H | H | H | 3.0 / 3.5 / 3.0 |
| held_RV2c_s1 | 1f54430427975391 | late_wide | offroad |  | False | H | H | H | H | 1.5 / 2.5 / 3.5 |
| held_RV2c_s1 | 3b4031def0f45d96 | late_wide | offroad |  | False | H | H | H | . | 1.5 / 2.5 / - |
| held_RV2c_s1 | 4b3a1e85619e557f | other | collision |  | False | H | H | . | . | 3.0 / - / - |
| held_RV2c_s1 | 20728d3e677b593d | other | corridor |  | False | H | . | H | . | - / 2.5 / - |
| held_RV2c_s1 | f9ed38d9ddfa531e | straight | offroad |  | False | H | . | H | - | - / 2.5 / - |
| held_RV2c_s1 | a67f68a6c9e85668 | straight | offroad |  | False | H | . | H | . | - / 3.5 / - |
| pai | 1e8e939e | lane_change | collision |  | False | H | H | H | H | 1.0 / 2.0 / 3.5 |
| pai | 00fabf60 | late_wide | corridor |  | False | H | H | H | . | 2.5 / 3.0 / - |
| pai | 1a69f757 | late_wide | collision |  | False | H | H | H | H | 1.0 / 1.0 / 3.0 |
| pai | 22fa5c92 | late_wide | collision |  | False | . | . | . | . | - / - / - |
| pai | 2a882d89 | late_wide | offroad |  | False | . | . | . | . | - / - / - |
| pai | 33c8976b | late_wide | offroad |  | False | H | H | . | . | 1.5 / - / - |
| pai | 37d660b1 | late_wide | corridor |  | False | H | H | H | . | 1.5 / 1.5 / - |
| pai | 3ce13eed | late_wide | corridor |  | False | H | H | H | H | 1.5 / 2.0 / 4.0 |
| pai | 68e53160 | late_wide | corridor |  | False | H | H | H | H | 2.5 / 3.0 / 4.0 |
| pai | 7381c9fa | late_wide | offroad |  | False | H | H | H | H | 1.0 / 3.0 / 1.0 |
| pai | 94c1d722 | late_wide | offroad |  | True | H | H | H | H | 1.0 / 1.0 / 1.5 |
| pai | 98f88a4e | late_wide | offroad |  | False | H | H | H | H | 1.0 / 1.5 / 2.5 |
| pai | 9f393c8a | late_wide | collision |  | False | . | . | . | . | - / - / - |
| pai | a1b166cc | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| pai | cdf95fd1 | late_wide | corridor |  | False | H | H | H | H | 2.0 / 3.0 / 3.0 |
| pai | ce161092 | late_wide | offroad |  | False | H | H | H | H | 1.0 / 2.0 / 3.0 |
| pai | e37aa159 | late_wide | corridor |  | False | H | . | H | . | - / 3.0 / - |
| pai | fd01e9ab | late_wide | corridor |  | False | . | . | . | . | - / - / - |
| pai | fda39805 | late_wide | corridor |  | False | H | H | H | H | 1.0 / 1.0 / 1.0 |
| pai | 382a7247 | other | collision |  | False | . | . | . | . | - / - / - |
| pai | de9db5dc | other | collision |  | False | H | H | . | . | 4.0 / - / - |
| pai | 0a228e44 | straight | corridor |  | False | H | H | H | H | 1.5 / 1.5 / 4.0 |
| pai | 2f227f2c | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | 302c5c99 | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | 3232fdb7 | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | 403d558a | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | 52003cee | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | a174faed | straight | corridor |  | False | H | H | H | H | 1.0 / 2.0 / 3.0 |
| pai | a5599793 | straight | offroad |  | False | H | H | H | H | 1.5 / 2.5 / 3.5 |
| pai | bd97de34 | straight | corridor |  | False | H | . | H | . | - / 1.5 / - |
| pai | e5e47fd2 | straight | corridor |  | False | H | H | H | H | 1.0 / 2.0 / 3.5 |
| pai | eaba3f6a | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | f9e361aa | straight | corridor |  | False | . | . | . | . | - / - / - |
| pai | 32be03e4 | wrong_exit | corridor |  | False | . | . | . | - | - / - / - |
