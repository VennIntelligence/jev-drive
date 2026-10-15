# SPEED-GEOMETRY reading (rf_speedgeo.py)

Prereg `plans/2026-10-16-speed-geometry-prereg.md`. Part A: approach + turn decisions of the strict turns; r_T = plan peak curvature over 0-T s / human peak curvature over the same stretch of road (arc-matched), q = ego speed / human speed at the same route station. Turn-weighted WLS of log r on log q, turn-cluster bootstrap 95 % CI (B 10 000). Part B: plan 2 s speed drop dP = v - v_plan(2 s) vs the human's dH = v_h(t_s) - v_h(t_s + 2 s) at the same log station; R = mean dP / mean dH over scenes, scene bootstrap CI.

## Part A

A1 (slope of log r on log q; A1c = coefficient on log v with log v_h and log kappa_h as covariates; within = turn-demeaned):

| set | variant | turns / decisions | slope [CI] | A1c log v [CI] | within [CI] | verdict |
|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG | T2 | 34 / 1429 | -1.68 [-2.48, -0.71] | -1.77 [-2.59, -0.74] | -0.14 [-0.70, +0.31] | supported |
| pai_RSG | T4 | 35 / 1761 | -1.49 [-2.30, -0.66] | -1.70 [-2.53, -0.77] | -0.30 [-0.85, +0.25] | supported |
| pai_RSG | T2_time | 34 / 1427 | -1.52 [-2.32, -0.55] | -1.67 [-2.47, -0.67] | -0.11 [-0.68, +0.43] | supported |
| held_RSG | T2 | 48 / 465 | -0.88 [-1.64, -0.24] | -0.82 [-1.50, -0.26] | -0.87 [-1.74, +0.12] | supported |
| held_RSG | T4 | 48 / 481 | -0.89 [-1.66, -0.23] | -0.87 [-1.52, -0.34] | -0.90 [-1.74, +0.04] | supported |
| held_RSG | T2_time | 48 / 460 | -0.63 [-0.88, -0.33] | -0.60 [-0.80, -0.31] | -0.73 [-1.22, -0.19] | supported |
| pai_APP1X | T2 | 31 / 1549 | -0.64 [-1.20, -0.20] | -0.62 [-1.29, -0.01] | -0.17 [-0.60, +0.23] | supported |
| pai_APP1X | T4 | 32 / 1834 | -0.35 [-0.90, +0.12] | -0.54 [-1.10, +0.01] | -0.13 [-0.55, +0.23] | inconclusive |
| pai_APP1X | T2_time | 32 / 1557 | -0.49 [-1.01, -0.10] | -0.44 [-1.09, +0.05] | -0.19 [-0.56, +0.14] | inconclusive |

A2 (peak lateral acceleration over the plan's 0-4 s, m/s^2; counterfactual = human curvature on the same stretch x ego speed^2, decisions with v > v_h; human = human curvature x human speed^2):

| set | plan p90 [CI] (n) | counterfactual p90 [CI] (n) | human p90 [CI] | verdict |
|:--|:--|:--|:--|:--|
| pai_RSG | 2.89 [2.26, 3.18] (2574) | 8.94 [5.11, 11.13] (841) | 3.91 [3.10, 5.06] (2594) | supported |
| held_RSG | 2.46 [2.18, 2.67] (649) | 2.71 [2.33, 2.92] (184) | 2.64 [2.11, 3.06] (555) | refuted |
| pai_APP1X | 2.46 [2.19, 2.74] (2704) | 3.76 [2.68, 5.90] (709) | 3.47 [2.83, 4.19] (2980) | inconclusive |

Plan a_lat by ego speed (pai_RSG, m/s: n, median, p90): 2-4 1203, 0.73, 1.97; 4-6 612, 0.96, 2.55; 6-8 382, 2.59, 3.32; 8-12 311, 0.90, 3.51; 12-40 66, 0.33, 3.94

A3 (APP1X vs RSG, 31 shared turns with eligible decisions): median r_2 0.88 vs 0.77, median |log r_2| 0.17 vs 0.29, median q 0.94 vs 0.99; paired log r_2 difference +0.09 [+0.01, +0.21], paired q difference -0.09 [-0.15, +0.00]; completed 11 vs 4.

Per class (turn medians of the eligible T2 decisions; slope with a 2 000-resample CI when >= 5 turns):

| set | class | turns (eligible) | decisions | median r_2 | median q | v / v_h median | plan a_lat p90 | slope [CI] |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG | completed | 5 (5) | 246 | 0.97 | 1.01 | 6.1 / 6.8 | 3.34 | +0.19 [-0.11, +1.09] |
| pai_RSG | late_wide | 12 (12) | 539 | 0.92 | 0.89 | 3.4 / 4.4 | 2.94 | -0.33 [-0.89, +0.01] |
| pai_RSG | straight | 8 (8) | 197 | 0.05 | 1.36 | 5.1 / 5.3 | 1.35 | -1.78 [-2.61, -0.94] |
| pai_RSG | over_turn | 2 (0) | 0 | - | - | 14.1 / 6.2 | 4.91 | - |
| pai_RSG | no_zero | 9 (8) | 425 | 0.84 | 0.82 | 3.2 / 4.8 | 2.17 | -0.28 [-0.64, +0.04] |
| pai_RSG | other | 1 (1) | 22 | 0.34 | 2.00 | 13.9 / 6.9 | 4.21 | - |
| held_RSG | completed | 30 (30) | 268 | 0.95 | 1.00 | 5.0 / 5.3 | 2.64 | -0.35 [-0.62, -0.07] |
| held_RSG | late_wide | 2 (2) | 18 | 0.64 | 1.27 | 3.2 / 2.7 | 2.12 | - |
| held_RSG | over_turn | 9 (5) | 25 | 1.09 | 0.99 | 2.3 / 1.9 | 2.66 | -0.38 [-1.45, +7.32] |
| held_RSG | no_zero | 16 (16) | 179 | 1.05 | 0.84 | 3.9 / 4.7 | 2.14 | -0.98 [-2.01, +0.28] |
| pai_APP1X | completed | 11 (11) | 545 | 0.88 | 1.02 | 4.1 / 4.7 | 2.76 | -0.17 [-0.34, +0.41] |
| pai_APP1X | late_wide | 8 (8) | 327 | 0.87 | 0.98 | 3.5 / 4.2 | 2.42 | -0.44 [-1.48, +0.42] |
| pai_APP1X | straight | 2 (2) | 77 | 0.60 | 0.94 | 1.8 / 3.7 | 1.31 | - |
| pai_APP1X | no_zero | 13 (9) | 595 | 0.93 | 0.69 | 2.7 / 6.4 | 1.98 | -0.77 [-1.38, -0.20] |
| pai_APP1X | other | 1 (1) | 5 | 1.72 | 0.74 | 2.1 / 2.0 | 2.07 | - |

## Part B

B1 (collision zeros, window [t_ev - 2 s, t_ev)): R1 = +0.01 [-7.03, +8.95] (n 9; dP -0.00, dH -0.33 m/s) -> **untestable (human did not brake: mean dH < 0.5)**.
B2 (human braked for a lead, no collision; ego lead TTC <= 3 s): 83 matched scenes, 14 with eligible decisions (242): R2 = +0.81 [+0.15, +1.64] (n 11; dP +1.37, dH +1.69 m/s) -> **refuted**.
- both facing a lead (human TTC <= 3 s at t_s too, 79 decisions): +1.11 [+0.92, +1.60] (n 5; dP +2.51, dH +2.26 m/s)
- own times (plan at ego TTC <= 3 s vs human at human TTC <= 3 s): +1.20 [+0.96, +1.64] (n 6; dP +2.43, dH +2.02 m/s)
- taxonomy lead / queue scenes (18, 7 in the matched set): -24.42 [-24.42, +5.37] (n 2; dP +1.07, dH -0.04 m/s)
- plan baseline drop on lead-free decisions (v >= 2): mean over 273 rollouts +0.73 m/s (human there +0.10); matched scenes +0.99
B3 (d299 oncoming-lane events, n 7): before [t_e - 2 s, t_e) -3.44 [-4.89, +10.27] (n 7; dP -0.06, dH +0.02 m/s); in the lane [t_e, t_e + 2 s) +0.54 [+0.22, +1.76] (n 7; dP -0.76, dH -1.42 m/s).

B1 per scene (m/s; executed drop = ego v(t_ev - 2 s) - v(t_ev); hit range / TTC at t_ev - 2 s):

| scene | primary | dP | dH | v / v_h | v_plan(2 s) min | executed drop | v at t_ev | human v at the event station | hit range m | hit TTC s | ego lead TTC min |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 1a69f757 | coll_veh_oncoming | -2.22 | -1.72 | 6.3 / 4.0 | 6.9 | -3.00 | 8.0 | 5.1 | 34.6 | 2.4 | - |
| 1e8e939e | coll_cyclist | +3.63 | +2.92 | 13.5 / 6.6 | 8.6 | +2.91 | 11.6 | 2.5 | 26.9 | 2.1 | 0.0 |
| 229791a5 | coll_veh_crossing | -1.12 | -1.77 | 4.4 / 4.1 | 5.0 | -1.34 | 5.0 | 5.1 | 14.4 | 4.7 | 6.2 |
| 22fa5c92 | coll_veh_oncoming | -0.51 | -0.05 | 2.6 / 3.4 | 2.4 | -0.32 | 3.0 | 3.6 | 16.2 | 5.4 | 5.7 |
| 382a7247 | coll_veh_oncoming | -0.71 | +0.20 | 8.8 / 10.0 | 8.8 | -0.66 | 9.4 | 9.8 | 23.0 | 2.7 | - |
| 9f393c8a | coll_veh_side | +1.35 | -1.07 | 4.2 / 5.7 | 1.8 | +1.84 | 3.2 | 5.9 | 5.7 | 11.0 | - |
| b44a4a44 | coll_ped | -0.02 | -1.55 | 0.0 / 0.0 | 0.0 | +0.00 | 0.0 | 0.0 | 4.4 | 4.3 | - |
| de9db5dc | coll_veh_side | -0.53 | -0.93 | 2.1 / 3.8 | 2.3 | -0.75 | 2.5 | 4.0 | 5.5 | 6.4 | - |
| fce18b62 | coll_veh_side | +0.09 | +1.02 | 1.5 / 2.6 | 1.3 | +0.25 | 1.3 | 2.5 | 3.9 | 5.9 | - |

B3 per event:

| scene | primary | dP / dH before | dP / dH in lane | v / v_h | ego lead TTC min |
|:--|:--|:--|:--|:--|--:|
| 09528162 | low_rear_ended | +2.60 / +2.02 | -0.54 / +0.81 | 4.8 / 5.7 | 8.7 |
| 268949d2 | ok | -1.36 / -1.43 | -2.05 / -3.04 | 1.6 / 0.9 | 30.4 |
| 94c1d722 | off_junc_outside | -0.67 / +0.09 | -0.36 / -2.72 | 2.9 / 4.1 | - |
| a4d2f7cf | ok | -0.51 / +1.11 | -0.22 / -0.82 | 3.7 / 4.6 | 4.1 |
| ddc3e8df | cor_turn_under | -0.71 / -1.56 | -1.46 / -2.64 | 2.5 / 2.5 | - |
| e0f6ebe1 | ok | +0.53 / +0.88 | -0.54 / +0.51 | 7.6 / 5.4 | - |
| e848c843 | low_slow_free | -0.32 / -0.98 | -0.13 / -2.01 | 3.3 / 2.7 | - |

## Amendment 1 (EXPLORATORY, post hoc)

- A1 without the straight-through (and other) turns: pai_RSG -0.24 [-0.48, -0.02] (turns 25, within +0.01 [-0.55, +0.48]); held_RSG -0.88 [-1.64, -0.24] (turns 48, within -0.87 [-1.74, +0.12]); pai_APP1X -0.38 [-0.65, -0.03] (turns 28, within -0.14 [-0.59, +0.29])
- share of eligible decisions from straight-through turns, by speed ratio: pai_RSG q>1.0 0.28, q>1.25 0.39, q>1.5 0.50; pai_APP1X q>1.0 0.07, q>1.25 0.12, q>1.5 0.04
- B2, plan at a lead within 3 s TTC vs its own lead-free decisions (matched scenes, pooled decisions; v, dP, dP / v, executed 2 s drop, human dH at those stations): lead n 242 (14 scenes) v 7.2, dP +2.36, dP/v +0.495, executed +3.43, dH +1.55; free n 4384 (79 scenes) v 10.0, dP +0.57, dP/v +0.045, executed +0.47, dH +0.60
- B2 per scene (scenes where the human dropped >= 0.5 m/s there): the plan drops at least half the human's in 6 / 8
- B1, hit object closing at range-rate TTC <= 3 s at t_ev - 2 s: 3 scenes, plan dP > 0.5 m/s in 1: 1a69f757 coll_veh_oncoming TTC 2.4 s at 35 m, dP -2.22, v 6.3 -> 8.0 (human at the event station 5.1); 1e8e939e coll_cyclist TTC 2.1 s at 27 m, dP +3.63, v 13.5 -> 11.6 (human at the event station 2.5); 382a7247 coll_veh_oncoming TTC 2.7 s at 23 m, dP -0.71, v 8.8 -> 9.4 (human at the event station 9.8)
- B1 speed at the event station: ego median 3.2, human 4.0 m/s; ego faster by > 0.5 m/s in 2 / 9
