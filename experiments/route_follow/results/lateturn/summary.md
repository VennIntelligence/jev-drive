# LATE-TURNS: completion by class (rf_lateturn.py report)

Class from the policy-free scan (scan.csv; rules in the script docstring): late = turn (strict or half-way) with entry >= 40 m after the clip start, early = entry < 40 m (delayed = 20-40 m), fork = the route leaves the ego lane's straight successor (branch or lane change then turn / branch) inside the log's travel. Reads @18 decisions, rm_chain readlist serving; train (onpol-train) and held 460 pooled unless split. Completion: the ego crosses the target station (strict turn: exit + 10 m; half-way turn: s_end - 5 m; fork only: min(s_sep + 10, s_end - 5) m) <= 2.5 m from the route before any zero event (= rf_follow's outcome on strict turns). CIs: log-clustered bootstrap 10 000; model pairs paired on the same scenes.

## Completion by class x model

| class | scenes (train / held) | RSG-L18-s0 | RSGboth-L18-s0 | RB0-F-s0 |
|:--|:--|:--|:--|:--|
| late (all, entry >= 40 m) | 4 (3 / 1) | 0 % [0, 0] (4) | 100 % [100, 100] (4) | 0 % [0, 0] (4) |
| delayed (entry 20-40 m) | 18 (15 / 3) | 22 % [0, 39] (18) | 61 % [43, 85] (18) | 17 % [0, 33] (18) |
| late or delayed (entry >= 20 m) | 22 (18 / 4) | 18 % [0, 32] (22) | 68 % [52, 88] (22) | 14 % [0, 27] (22) |
| fork (all) | 97 (63 / 34) | 57 % [46, 67] (97) | 74 % [65, 82] (97) | 48 % [38, 59] (97) |
| fork, no turn | 30 (16 / 14) | 57 % [38, 78] (30) | 73 % [59, 87] (30) | 50 % [33, 69] (30) |
| turn + fork | 67 (47 / 20) | 57 % [45, 67] (67) | 75 % [63, 84] (67) | 48 % [34, 60] (67) |
| early (all, entry < 40 m) | 165 (99 / 66) | 49 % [41, 57] (165) | 67 % [59, 74] (165) | 42 % [33, 51] (165) |
| early, entry < 20 m | 147 (84 / 63) | 52 % [43, 61] (147) | 68 % [59, 75] (147) | 45 % [36, 54] (147) |
| early < 20 m, no fork | 87 (43 / 44) | 46 % [36, 56] (87) | 64 % [52, 74] (87) | 40 % [30, 51] (87) |
| turn, no fork (all) | 102 (55 / 47) | 42 % [33, 52] (102) | 64 % [53, 73] (102) | 36 % [27, 46] (102) |
| turn + fork, entry < 20 m | 60 (41 / 19) | 62 % [49, 73] (60) | 73 % [60, 84] (60) | 52 % [38, 65] (60) |

## Paired completion differences (pp, same scenes)

| class | RSGboth - RSG | RSG - RB0 | RSGboth - RB0 |
|:--|:--|:--|:--|
| late (all, entry >= 40 m) | +100 [+100, +100] | +0 [+0, +0] | +100 [+100, +100] |
| delayed (entry 20-40 m) | +39 [+21, +69] | +6 [+0, +11] | +44 [+23, +69] |
| late or delayed (entry >= 20 m) | +50 [+30, +76] | +5 [+0, +9] | +55 [+36, +77] |
| fork (all) | +18 [+8, +27] | +8 [+2, +14] | +26 [+15, +37] |
| fork, no turn | +17 [-4, +34] | +7 [-6, +21] | +23 [+3, +42] |
| turn + fork | +18 [+7, +29] | +9 [+3, +16] | +27 [+13, +41] |
| early (all, entry < 40 m) | +18 [+11, +26] | +7 [+3, +12] | +25 [+17, +34] |
| early, entry < 20 m | +16 [+8, +24] | +7 [+3, +13] | +23 [+14, +32] |
| early < 20 m, no fork | +18 [+7, +29] | +6 [+0, +12] | +24 [+13, +34] |
| turn, no fork (all) | +22 [+11, +32] | +6 [+0, +11] | +27 [+17, +37] |
| turn + fork, entry < 20 m | +12 [+0, +23] | +10 [+4, +18] | +22 [+7, +37] |

## Contrasts within model (unpaired, different scenes): completion difference (pp)

| contrast | RSG-L18-s0 | RSGboth-L18-s0 | RB0-F-s0 |
|:--|:--|:--|:--|
| entry >= 20 m - entry < 20 m | -34 [-52, -18] | +0 [-18, +23] | -31 [-46, -15] |
| turn + fork - turn, no fork | +15 [-1, +29] | +11 [-4, +26] | +11 [-5, +27] |
| fork, no turn - early < 20 m, no fork | +11 [-11, +35] | +9 [-9, +28] | +10 [-10, +31] |

## Outcomes, zeros, too slow, inside cut

| class | model | mean score | completed / event / off / short | zeros (col / off / cor / other) | zero % | short % | inside-cut % (turns) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| late (all, entry >= 40 m) | RSG-L18-s0 | 0.74 [0.59, 0.86] | 0 / 1 / 0 / 3 | 0 (0 / 0 / 0 / 0) | 0 % [0, 0] (4) | 75 % [25, 100] (4) | - |
| late (all, entry >= 40 m) | RSGboth-L18-s0 | 1.00 [1.00, 1.00] | 4 / 0 / 0 / 0 | 0 (0 / 0 / 0 / 0) | 0 % [0, 0] (4) | 0 % [0, 0] (4) | - |
| late (all, entry >= 40 m) | RB0-F-s0 | 0.72 [0.59, 0.85] | 0 / 1 / 0 / 3 | 0 (0 / 0 / 0 / 0) | 0 % [0, 0] (4) | 75 % [25, 100] (4) | - |
| delayed (entry 20-40 m) | RSG-L18-s0 | 0.67 [0.52, 0.81] | 4 / 5 / 2 / 7 | 4 (0 / 3 / 1 / 0) | 22 % [0, 38] (18) | 39 % [21, 69] (18) | 67 % [20, 100] (6) |
| delayed (entry 20-40 m) | RSGboth-L18-s0 | 0.72 [0.55, 0.99] | 11 / 5 / 0 / 2 | 5 (0 / 5 / 0 / 0) | 28 % [0, 44] (18) | 11 % [0, 31] (18) | 67 % [20, 100] (6) |
| delayed (entry 20-40 m) | RB0-F-s0 | 0.53 [0.36, 0.71] | 3 / 8 / 3 / 4 | 7 (1 / 4 / 2 / 0) | 39 % [15, 56] (18) | 22 % [8, 46] (18) | 67 % [20, 100] (6) |
| late or delayed (entry >= 20 m) | RSG-L18-s0 | 0.68 [0.56, 0.80] | 4 / 6 / 2 / 10 | 4 (0 / 3 / 1 / 0) | 18 % [0, 32] (22) | 45 % [26, 71] (22) | 67 % [20, 100] (6) |
| late or delayed (entry >= 20 m) | RSGboth-L18-s0 | 0.77 [0.62, 0.99] | 15 / 5 / 0 / 2 | 5 (0 / 5 / 0 / 0) | 23 % [0, 38] (22) | 9 % [0, 24] (22) | 67 % [20, 100] (6) |
| late or delayed (entry >= 20 m) | RB0-F-s0 | 0.56 [0.42, 0.72] | 3 / 9 / 3 / 7 | 7 (1 / 4 / 2 / 0) | 32 % [12, 48] (22) | 32 % [15, 55] (22) | 67 % [20, 100] (6) |
| fork (all) | RSG-L18-s0 | 0.77 [0.70, 0.84] | 55 / 15 / 10 / 17 | 15 (2 / 9 / 4 / 0) | 15 % [9, 23] (97) | 18 % [10, 27] (97) | 57 % [36, 76] (23) |
| fork (all) | RSGboth-L18-s0 | 0.82 [0.74, 0.89] | 72 / 15 / 7 / 3 | 16 (4 / 10 / 2 / 0) | 16 % [10, 24] (97) | 3 % [0, 7] (97) | 61 % [43, 78] (23) |
| fork (all) | RB0-F-s0 | 0.67 [0.59, 0.75] | 47 / 23 / 10 / 17 | 26 (4 / 14 / 8 / 0) | 27 % [18, 35] (97) | 18 % [10, 25] (97) | 52 % [32, 72] (23) |
| fork, no turn | RSG-L18-s0 | 0.84 [0.77, 0.92] | 17 / 2 / 4 / 7 | 1 (0 / 1 / 0 / 0) | 3 % [0, 9] (30) | 23 % [7, 38] (30) | - |
| fork, no turn | RSGboth-L18-s0 | 0.87 [0.76, 0.97] | 22 / 3 / 5 / 0 | 3 (1 / 1 / 1 / 0) | 10 % [0, 21] (30) | 0 % [0, 0] (30) | - |
| fork, no turn | RB0-F-s0 | 0.72 [0.59, 0.84] | 15 / 5 / 5 / 5 | 5 (0 / 5 / 0 / 0) | 17 % [4, 30] (30) | 17 % [4, 29] (30) | - |
| turn + fork | RSG-L18-s0 | 0.74 [0.64, 0.82] | 38 / 13 / 6 / 10 | 14 (2 / 8 / 4 / 0) | 21 % [12, 31] (67) | 15 % [6, 26] (67) | 57 % [36, 76] (23) |
| turn + fork | RSGboth-L18-s0 | 0.79 [0.69, 0.88] | 50 / 12 / 2 / 3 | 13 (3 / 9 / 1 / 0) | 19 % [11, 30] (67) | 4 % [0, 10] (67) | 61 % [43, 78] (23) |
| turn + fork | RB0-F-s0 | 0.64 [0.54, 0.74] | 32 / 18 / 5 / 12 | 21 (4 / 9 / 8 / 0) | 31 % [21, 42] (67) | 18 % [9, 28] (67) | 52 % [32, 72] (23) |
| early (all, entry < 40 m) | RSG-L18-s0 | 0.73 [0.66, 0.79] | 81 / 38 / 13 / 33 | 35 (6 / 20 / 9 / 0) | 21 % [15, 28] (165) | 20 % [13, 28] (165) | 55 % [44, 65] (88) |
| early (all, entry < 40 m) | RSGboth-L18-s0 | 0.77 [0.71, 0.83] | 111 / 35 / 5 / 14 | 34 (6 / 24 / 4 / 0) | 21 % [14, 27] (165) | 8 % [5, 13] (165) | 56 % [46, 65] (88) |
| early (all, entry < 40 m) | RB0-F-s0 | 0.62 [0.54, 0.69] | 69 / 54 / 15 / 27 | 56 (11 / 26 / 19 / 0) | 34 % [27, 42] (165) | 16 % [11, 22] (165) | 50 % [39, 61] (88) |
| early, entry < 20 m | RSG-L18-s0 | 0.74 [0.67, 0.80] | 77 / 33 / 11 / 26 | 31 (6 / 17 / 8 / 0) | 21 % [15, 29] (147) | 18 % [11, 26] (147) | 54 % [43, 64] (82) |
| early, entry < 20 m | RSGboth-L18-s0 | 0.78 [0.70, 0.84] | 100 / 30 / 5 / 12 | 29 (6 / 19 / 4 / 0) | 20 % [13, 27] (147) | 8 % [4, 13] (147) | 55 % [45, 64] (82) |
| early, entry < 20 m | RB0-F-s0 | 0.63 [0.54, 0.70] | 66 / 46 / 12 / 23 | 49 (10 / 22 / 17 / 0) | 33 % [26, 42] (147) | 16 % [10, 22] (147) | 49 % [38, 59] (82) |
| early < 20 m, no fork | RSG-L18-s0 | 0.73 [0.63, 0.81] | 40 / 22 / 6 / 19 | 18 (4 / 10 / 4 / 0) | 21 % [13, 30] (87) | 22 % [12, 31] (87) | 51 % [39, 63] (61) |
| early < 20 m, no fork | RSGboth-L18-s0 | 0.78 [0.68, 0.86] | 56 / 19 / 3 / 9 | 17 (3 / 11 / 3 / 0) | 20 % [12, 29] (87) | 10 % [4, 17] (87) | 51 % [40, 61] (61) |
| early < 20 m, no fork | RB0-F-s0 | 0.61 [0.50, 0.71] | 35 / 31 / 8 / 13 | 30 (6 / 14 / 10 / 0) | 34 % [24, 46] (87) | 15 % [7, 23] (87) | 46 % [33, 58] (61) |
| turn, no fork (all) | RSG-L18-s0 | 0.72 [0.64, 0.80] | 43 / 26 / 7 / 26 | 21 (4 / 12 / 5 / 0) | 21 % [13, 29] (102) | 25 % [17, 35] (102) | 54 % [42, 65] (65) |
| turn, no fork (all) | RSGboth-L18-s0 | 0.77 [0.68, 0.85] | 65 / 23 / 3 / 11 | 21 (3 / 15 / 3 / 0) | 21 % [12, 30] (102) | 11 % [5, 18] (102) | 54 % [43, 64] (65) |
| turn, no fork (all) | RB0-F-s0 | 0.61 [0.51, 0.70] | 37 / 37 / 10 / 18 | 35 (7 / 17 / 11 / 0) | 34 % [25, 44] (102) | 18 % [10, 25] (102) | 49 % [37, 61] (65) |
| turn + fork, entry < 20 m | RSG-L18-s0 | 0.75 [0.64, 0.84] | 37 / 11 / 5 / 7 | 13 (2 / 7 / 4 / 0) | 22 % [12, 33] (60) | 12 % [4, 23] (60) | 62 % [40, 82] (21) |
| turn + fork, entry < 20 m | RSGboth-L18-s0 | 0.78 [0.67, 0.88] | 44 / 11 / 2 / 3 | 12 (3 / 8 / 1 / 0) | 20 % [11, 32] (60) | 5 % [0, 12] (60) | 67 % [48, 85] (21) |
| turn + fork, entry < 20 m | RB0-F-s0 | 0.65 [0.54, 0.76] | 31 / 15 / 4 / 10 | 19 (4 / 8 / 7 / 0) | 32 % [21, 43] (60) | 17 % [8, 27] (60) | 57 % [35, 77] (21) |

## Turn shape vs the human (rf_ledger.turn_rows on the human path; turn classes only)

Speed ego - human (m/s) when the ego first crosses entry - 40 / - 20 / 0 m of the human-path turn; onset - human (m, - = early); peak inside / outside offset to the human path (m). Means [log-clustered CI] (n).

| class | model | ego v @-40 | ego - human @-40 | ego - human @-20 | ego - human @0 | onset - human | peak in | peak out |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| late or delayed (entry >= 20 m) | RSG-L18-s0 | 8.56 [7.90, 9.58] (3) | -0.01 [-0.06, 0.03] (3) | -0.51 [-1.15, -0.12] (15) | -2.07 [-2.75, -1.42] (18) | -0.62 [-2.33, 1.26] (16) | -0.82 [-1.30, -0.37] (21) | 0.47 [0.12, 0.92] (21) |
| late or delayed (entry >= 20 m) | RSGboth-L18-s0 | 8.56 [7.90, 9.58] (3) | -0.01 [-0.06, 0.03] (3) | -0.06 [-0.25, 0.09] (15) | -0.20 [-0.44, 0.14] (21) | -0.85 [-1.62, -0.02] (20) | -0.83 [-1.17, -0.49] (21) | 0.62 [0.35, 0.99] (21) |
| late or delayed (entry >= 20 m) | RB0-F-s0 | 8.56 [7.90, 9.58] (3) | -0.01 [-0.06, 0.03] (3) | -0.50 [-1.13, -0.11] (15) | -2.05 [-2.82, -1.34] (16) | -1.59 [-3.39, -0.35] (13) | -1.05 [-1.58, -0.55] (20) | 0.41 [0.04, 0.84] (20) |
| late or delayed (entry >= 20 m) | human | 8.57 [7.87, 9.60] | | | | | | |
| early, entry < 20 m | RSG-L18-s0 | - (0) | - (0) | -0.04 [-0.18, 0.10] (2) | -0.14 [-0.24, -0.04] (121) | -0.05 [-0.41, 0.32] (118) | -0.64 [-0.80, -0.49] (121) | 1.01 [0.79, 1.24] (121) |
| early, entry < 20 m | RSGboth-L18-s0 | - (0) | - (0) | -0.10 [-0.30, 0.10] (2) | 0.03 [-0.06, 0.12] (120) | -0.22 [-0.54, 0.11] (119) | -0.62 [-0.74, -0.50] (121) | 0.74 [0.56, 0.94] (121) |
| early, entry < 20 m | RB0-F-s0 | - (0) | - (0) | -0.03 [-0.17, 0.10] (2) | -0.13 [-0.24, -0.03] (120) | -0.01 [-0.37, 0.35] (116) | -0.62 [-0.78, -0.46] (120) | 1.33 [1.08, 1.59] (120) |
| early, entry < 20 m | human | - | | | | | | |
| fork (all) | RSG-L18-s0 | 8.83 [7.90, 9.77] (2) | -0.11 [-0.25, 0.03] (2) | -0.91 [-1.92, -0.07] (6) | -0.29 [-0.58, -0.07] (54) | 0.23 [-0.44, 0.96] (55) | -0.59 [-0.84, -0.37] (57) | 0.99 [0.66, 1.36] (57) |
| fork (all) | RSGboth-L18-s0 | 8.98 [7.90, 10.07] (2) | 0.04 [0.03, 0.05] (2) | -0.27 [-0.57, -0.05] (6) | 0.03 [-0.13, 0.17] (57) | -0.20 [-0.58, 0.22] (57) | -0.64 [-0.78, -0.49] (57) | 0.68 [0.44, 0.96] (57) |
| fork (all) | RB0-F-s0 | 8.86 [7.90, 9.82] (2) | -0.09 [-0.20, 0.03] (2) | -0.87 [-1.83, -0.07] (6) | -0.22 [-0.47, -0.03] (53) | 0.19 [-0.30, 0.73] (52) | -0.55 [-0.76, -0.35] (56) | 1.36 [0.96, 1.79] (56) |
| fork (all) | human | 8.94 [7.87, 10.02] | | | | | | |
| turn, no fork (all) | RSG-L18-s0 | 8.89 [8.19, 9.58] (2) | -0.04 [-0.06, -0.01] (2) | -0.36 [-0.93, -0.04] (12) | -0.46 [-0.70, -0.24] (89) | -0.31 [-0.72, 0.06] (83) | -0.69 [-0.88, -0.52] (90) | 0.86 [0.62, 1.09] (90) |
| turn, no fork (all) | RSGboth-L18-s0 | 8.89 [8.19, 9.58] (2) | -0.04 [-0.06, -0.01] (2) | 0.02 [-0.09, 0.17] (12) | -0.04 [-0.16, 0.08] (89) | -0.36 [-0.74, 0.00] (87) | -0.63 [-0.80, -0.48] (90) | 0.72 [0.52, 0.93] (90) |
| turn, no fork (all) | RB0-F-s0 | 8.89 [8.19, 9.58] (2) | -0.04 [-0.06, -0.01] (2) | -0.35 [-0.91, -0.03] (12) | -0.45 [-0.70, -0.22] (87) | -0.36 [-0.84, 0.05] (81) | -0.75 [-0.95, -0.57] (89) | 1.04 [0.80, 1.29] (89) |
| turn, no fork (all) | human | 8.92 [8.25, 9.60] | | | | | | |

## Notes

- Scan (scenes.csv = rf_lateturn.py scan): 1491 scenes, 1473 with a route. Classes (train / held): early 140 / 66 (of them delayed
  20-40 m 15 / 3), late (>= 40 m) 3 / 1, fork 63 / 34 (with a turn 47 / 20, fork only 16 / 14; kinds branch 74, other 12, lc 11).
  86 more scenes have a first turn the log does not get half way round inside decision 17 (22 with entry >= 40 m): untestable @18.
  The @18 window is short: median log travel 47 m, so late turns barely exist on nuPlan at 18 decisions (the reconstructions cap reads at 18).
- Reads: held = the stored held 460 reads; train = the 104 trainread scenes + 52 new (every late / delayed / fork train scene not yet read,
  under the cap of 80, so no sampling; rf_lateturn.sh). Early train half-way turns (41) were not read, so early n = 165 of 206.
- Outcome check: on strict turns the target-station outcome equals rf_follow's own outcome on every rollout (127 / 78 / 24 / 35, 3 rollouts
  where rf_follow found no turn on the rollout's route are events).
- Late / delayed failures of RSG and RB0 are mostly "short" (10 / 22 and 7 / 22): the target is s_end - 5 m for half-way turns, and they reach
  the entry 2.1 m/s slower than the human (RSGboth -0.2). Counting only event + off: RSG 8 / 22, RSGboth 5 / 22, RB0 12 / 22, against early
  < 20 m 44 / 147, 35 / 147, 58 / 147: no sign that a turn later in the clip is turned worse once the ego gets there.
- PAI (pai.md, pai_classes.csv): 20 s clips, so late turns exist there (15 of the 36 strict turns).
- Review page: tmp/lateturn/late-turns/ (Tokyo ~/Desktop/late-turns/): 22 late / delayed + the 38 worst RSGboth forks, BEV + camera rerun on card 0.
