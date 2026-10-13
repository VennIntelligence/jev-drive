# RSGboth vs RSG: solved / unsolved ledger on held 460 @18 (rf_ledger.py)

460 scenes common to all labels. Points = sum(1 - score) / 460 x 100 in the class; pair = RSGboth - RSG per scene (log-clustered 95 % CI). Group pairs (seeds averaged): n = scenes summed over both seeds, points = seed mean. Rules in the script docstring.

## RSGboth-L18-s0:RSG-L18-s0: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 78 / 10 | 4.17 / 0.59 | -3.57 [-4.79, -2.49] | -68.0 | improved |
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 8 / 8 | 1.74 / 1.74 | 0.00 [-0.86, 0.87] | +0.0 | unchanged |
| offroad: curve / straight | 5 / 5 | 1.09 / 1.09 | 0.00 [-1.10, 1.00] | +0.0 | unchanged |
| stop: unexplained / late / stopped lead | 16 / 0 | 1.06 / 0.00 | -1.06 [-1.69, -0.54] | -16.0 | solved |
| collision: vehicle | 4 / 7 | 0.87 / 1.52 | 0.65 [-0.51, 1.91] | +3.0 | unchanged (trend worse) |
| corridor: route following | 4 / 5 | 0.87 / 1.09 | 0.22 [-0.73, 1.15] | +1.0 | unchanged (trend worse) |
| ok (>= 0.9) | 326 / 406 | 0.47 / 0.09 | -0.39 [-0.57, -0.22] | +80.0 | improved (more scenes) |
| stop / rear / other low | 6 / 3 | 0.35 / 0.21 | -0.14 [-0.43, 0.03] | -3.0 | unchanged |
| corridor: past the log end | 1 / 4 | 0.22 / 0.87 | 0.65 [0.00, 1.35] | +3.0 | unchanged (trend worse) |

## RSGboth-L18-s0:RSG-L18-s0: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 7 / 10 | 1.52 / 2.17 | 0.65 [-0.70, 1.98] | +3.0 | unchanged (trend worse) |
| turn zero: inside (cut) | 5 / 3 | 1.09 / 0.65 | -0.43 [-1.14, 0.00] | -2.0 | unchanged (trend better) |
| junction zero: straight through (no turn) | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.75] | +1.0 | unchanged (trend worse) |
| turn zero: went straight at the turn | 1 / 1 | 0.22 / 0.22 | 0.00 [-0.63, 0.63] | +0.0 | unchanged |

## RSGboth-L18-s0:RSG-L18-s0: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn_completed | 16 / 22 | 0.12 / 0.01 | -0.12 [-0.29, 0.00] | +6.0 | improved (more scenes) |
| turn_short | 7 / 1 | 0.07 / 0.01 | -0.06 [-0.15, 0.00] | -6.0 | unchanged |

## RSGboth-L18-s0:RSG-L18-s0: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 62 / 8 | 3.17 / 0.49 | -2.68 [-3.56, -1.87] | -54.0 | improved |
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_slow_behind_lead | 16 / 2 | 0.99 / 0.10 | -0.89 [-1.59, -0.32] | -14.0 | improved |
| low_unexplained | 15 / 0 | 0.90 / 0.00 | -0.90 [-1.47, -0.43] | -15.0 | solved |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_outside | 3 / 3 | 0.65 / 0.65 | 0.00 [-0.64, 0.61] | +0.0 | unchanged |
| off_junc_inside | 3 / 2 | 0.65 / 0.43 | -0.22 [-0.74, 0.00] | -1.0 | unchanged (trend better) |
| cor_turn_under | 3 / 4 | 0.65 / 0.87 | 0.22 [-0.72, 1.12] | +1.0 | unchanged (trend worse) |
| ok | 326 / 406 | 0.47 / 0.09 | -0.39 [-0.57, -0.22] | +80.0 | improved (more scenes) |
| off_curve_inside | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.73, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_side | 2 / 1 | 0.43 / 0.22 | -0.22 [-1.01, 0.50] | -1.0 | unchanged (trend better) |
| off_junc_straight | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.75] | +1.0 | unchanged (trend worse) |
| off_straight | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.75, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_ahead | 1 / 2 | 0.22 / 0.43 | 0.22 [0.00, 0.73] | +1.0 | unchanged (trend worse) |
| cor_straight_at_turn | 1 / 1 | 0.22 / 0.22 | 0.00 [-0.63, 0.63] | +0.0 | unchanged |
| cor_route_end | 1 / 4 | 0.22 / 0.87 | 0.65 [0.00, 1.35] | +3.0 | unchanged (trend worse) |
| off_curve_outside | 1 / 3 | 0.22 / 0.65 | 0.43 [-0.43, 1.24] | +2.0 | unchanged (trend worse) |
| coll_veh_oncoming | 1 / 2 | 0.22 / 0.43 | 0.22 [0.00, 0.66] | +1.0 | unchanged (trend worse) |
| low_rear_ended | 3 / 1 | 0.19 / 0.05 | -0.14 [-0.43, 0.00] | -2.0 | unchanged |
| low_stopped_lead | 1 / 0 | 0.16 / 0.00 | -0.16 [-0.55, 0.00] | -1.0 | unchanged |
| low_queue | 2 / 2 | 0.14 / 0.17 | 0.03 [-0.00, 0.09] | +0.0 | unchanged |
| low_other | 1 / 0 | 0.02 / 0.00 | -0.02 [-0.07, 0.00] | -1.0 | unchanged |
| coll_stationary | 0 / 2 | 0.00 / 0.43 | 0.43 [0.00, 1.04] | +2.0 | unchanged (trend worse) |

## RSGboth-L18-s1:RSG-L18-s1: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 75 / 9 | 3.79 / 0.44 | -3.35 [-4.40, -2.34] | -66.0 | improved |
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 8 / 9 | 1.74 / 1.96 | 0.22 [-0.83, 1.50] | +1.0 | unchanged (trend worse) |
| stop: unexplained / late / stopped lead | 19 / 1 | 1.34 / 0.06 | -1.28 [-2.10, -0.62] | -18.0 | improved |
| collision: vehicle | 6 / 11 | 1.30 / 2.39 | 1.09 [0.00, 2.31] | +5.0 | unchanged (trend worse) |
| corridor: route following | 6 / 3 | 1.30 / 0.65 | -0.65 [-1.52, 0.00] | -3.0 | unchanged (trend better) |
| ok (>= 0.9) | 326 / 400 | 0.50 / 0.03 | -0.47 [-0.65, -0.31] | +74.0 | improved (more scenes) |
| stop / rear / other low | 6 / 4 | 0.23 / 0.25 | 0.02 [-0.22, 0.34] | -2.0 | unchanged |
| corridor: past the log end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.21, 1.65] | +4.0 | worse |
| offroad: curve / straight | 1 / 6 | 0.22 / 1.30 | 1.09 [0.00, 2.33] | +5.0 | unchanged (trend worse) |

## RSGboth-L18-s1:RSG-L18-s1: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 6 / 6 | 1.30 / 1.30 | 0.00 [-0.92, 0.83] | +0.0 | unchanged |
| turn zero: inside (cut) | 4 / 5 | 0.87 / 1.09 | 0.22 [0.00, 0.73] | +1.0 | unchanged (trend worse) |
| junction zero: straight through (no turn) | 2 / 4 | 0.43 / 0.87 | 0.43 [0.00, 1.49] | +2.0 | unchanged (trend worse) |
| turn zero: went straight at the turn | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |

## RSGboth-L18-s1:RSG-L18-s1: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn_short | 9 / 1 | 0.17 / 0.03 | -0.14 [-0.28, -0.03] | -8.0 | improved |
| turn_completed | 14 / 21 | 0.09 / 0.00 | -0.09 [-0.23, 0.00] | +7.0 | improved (more scenes) |
| turn_off | 0 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +1.0 | unchanged |

## RSGboth-L18-s1:RSG-L18-s1: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 59 / 6 | 2.98 / 0.32 | -2.66 [-3.51, -1.80] | -53.0 | improved |
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_unexplained | 19 / 1 | 1.34 / 0.06 | -1.28 [-2.10, -0.62] | -18.0 | improved |
| cor_turn_under | 4 / 2 | 0.87 / 0.43 | -0.43 [-1.18, 0.00] | -2.0 | unchanged (trend better) |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_inside | 4 / 2 | 0.87 / 0.43 | -0.43 [-1.09, 0.00] | -2.0 | unchanged (trend better) |
| low_slow_behind_lead | 16 / 3 | 0.81 / 0.12 | -0.69 [-1.22, -0.24] | -13.0 | improved |
| ok | 326 / 400 | 0.50 / 0.03 | -0.47 [-0.65, -0.31] | +74.0 | improved (more scenes) |
| cor_straight_at_turn | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| off_junc_straight | 2 / 4 | 0.43 / 0.87 | 0.43 [0.00, 1.49] | +2.0 | unchanged (trend worse) |
| coll_veh_ahead | 2 / 4 | 0.43 / 0.87 | 0.43 [-0.39, 1.34] | +2.0 | unchanged (trend worse) |
| coll_veh_oncoming | 2 / 4 | 0.43 / 0.87 | 0.43 [0.00, 1.11] | +2.0 | unchanged (trend worse) |
| off_junc_outside | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.71] | +1.0 | unchanged (trend worse) |
| off_straight | 1 / 2 | 0.22 / 0.43 | 0.22 [-0.52, 0.93] | +1.0 | unchanged (trend worse) |
| cor_route_end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.21, 1.65] | +4.0 | worse |
| coll_veh_side | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_veh_crossing | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.75, 0.00] | -1.0 | unchanged (trend better) |
| low_rear_ended | 3 / 2 | 0.14 / 0.14 | -0.00 [-0.17, 0.23] | -1.0 | unchanged |
| low_queue | 2 / 2 | 0.06 / 0.11 | 0.05 [-0.10, 0.26] | +0.0 | unchanged |
| low_other | 1 / 0 | 0.02 / 0.00 | -0.02 [-0.08, 0.00] | -1.0 | unchanged |
| off_curve_outside | 0 / 1 | 0.00 / 0.22 | 0.22 [0.00, 0.71] | +1.0 | unchanged (trend worse) |
| off_curve_inside | 0 / 3 | 0.00 / 0.65 | 0.65 [0.00, 1.47] | +3.0 | unchanged (trend worse) |
| coll_stationary | 0 / 2 | 0.00 / 0.43 | 0.43 [0.00, 1.08] | +2.0 | unchanged (trend worse) |

## RSGboth-RSG: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 153 / 19 | 3.98 / 0.52 | -3.46 [-4.54, -2.44] | -67.0 | improved |
| infra (no rollout / start pose) | 24 / 24 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 16 / 17 | 1.74 / 1.85 | 0.11 [-0.77, 1.14] | +0.5 | unchanged |
| stop: unexplained / late / stopped lead | 35 / 1 | 1.20 / 0.03 | -1.17 [-1.82, -0.64] | -17.0 | solved |
| collision: vehicle | 10 / 18 | 1.09 / 1.96 | 0.87 [-0.13, 1.89] | +4.0 | unchanged (trend worse) |
| corridor: route following | 10 / 8 | 1.09 / 0.87 | -0.22 [-0.89, 0.32] | -1.0 | unchanged (trend better) |
| offroad: curve / straight | 6 / 11 | 0.65 / 1.20 | 0.54 [-0.43, 1.53] | +2.5 | unchanged (trend worse) |
| ok (>= 0.9) | 652 / 806 | 0.48 / 0.06 | -0.43 [-0.60, -0.28] | +77.0 | improved (more scenes) |
| stop / rear / other low | 12 / 7 | 0.29 / 0.23 | -0.06 [-0.27, 0.16] | -2.5 | unchanged |
| corridor: past the log end | 2 / 9 | 0.22 / 0.98 | 0.76 [0.13, 1.46] | +3.5 | worse |

## RSGboth-RSG: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 13 / 16 | 1.41 / 1.74 | 0.33 [-0.48, 1.08] | +1.5 | unchanged (trend worse) |
| turn zero: inside (cut) | 9 / 8 | 0.98 / 0.87 | -0.11 [-0.37, 0.00] | -0.5 | unchanged |
| junction zero: straight through (no turn) | 4 / 7 | 0.43 / 0.76 | 0.33 [0.00, 1.12] | +1.5 | unchanged (trend worse) |
| turn zero: went straight at the turn | 3 / 2 | 0.33 / 0.22 | -0.11 [-0.66, 0.30] | -0.5 | unchanged |

## RSGboth-RSG: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 2 / 2 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn_short | 16 / 2 | 0.12 / 0.02 | -0.10 [-0.19, -0.03] | -7.0 | improved |
| turn_completed | 30 / 43 | 0.10 / 0.00 | -0.10 [-0.25, -0.01] | +6.5 | improved (more scenes) |
| turn_off | 0 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +0.5 | unchanged |

## RSGboth-RSG: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 121 / 14 | 3.08 / 0.41 | -2.67 [-3.46, -1.88] | -53.5 | improved |
| infra | 16 / 16 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_unexplained | 34 / 1 | 1.12 / 0.03 | -1.09 [-1.75, -0.57] | -16.5 | solved |
| low_slow_behind_lead | 32 / 5 | 0.90 / 0.11 | -0.79 [-1.38, -0.31] | -13.5 | improved |
| off_at_start | 8 / 8 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_inside | 7 / 4 | 0.76 / 0.43 | -0.33 [-0.86, 0.00] | -1.5 | unchanged (trend better) |
| cor_turn_under | 7 / 6 | 0.76 / 0.65 | -0.11 [-0.62, 0.33] | -0.5 | unchanged |
| off_junc_outside | 5 / 6 | 0.54 / 0.65 | 0.11 [-0.30, 0.65] | +0.5 | unchanged |
| ok | 652 / 806 | 0.48 / 0.06 | -0.43 [-0.60, -0.28] | +77.0 | improved (more scenes) |
| off_junc_straight | 4 / 7 | 0.43 / 0.76 | 0.33 [0.00, 1.12] | +1.5 | unchanged (trend worse) |
| coll_veh_ahead | 3 / 6 | 0.33 / 0.65 | 0.33 [-0.13, 0.94] | +1.5 | unchanged (trend worse) |
| cor_straight_at_turn | 3 / 2 | 0.33 / 0.22 | -0.11 [-0.66, 0.30] | -0.5 | unchanged |
| coll_veh_side | 3 / 2 | 0.33 / 0.22 | -0.11 [-0.51, 0.25] | -0.5 | unchanged |
| coll_veh_oncoming | 3 / 6 | 0.33 / 0.65 | 0.33 [0.00, 0.68] | +1.5 | unchanged (trend worse) |
| off_straight | 3 / 3 | 0.33 / 0.33 | 0.00 [-0.62, 0.45] | +0.0 | unchanged |
| cor_route_end | 2 / 9 | 0.22 / 0.98 | 0.76 [0.13, 1.46] | +3.5 | worse |
| off_curve_inside | 2 / 4 | 0.22 / 0.43 | 0.22 [0.00, 0.55] | +1.0 | unchanged (trend worse) |
| low_rear_ended | 6 / 3 | 0.16 / 0.09 | -0.07 [-0.26, 0.09] | -1.5 | unchanged |
| off_curve_outside | 1 / 4 | 0.11 / 0.43 | 0.33 [-0.15, 0.89] | +1.5 | unchanged (trend worse) |
| coll_veh_crossing | 1 / 0 | 0.11 / 0.00 | -0.11 [-0.38, 0.00] | -0.5 | unchanged |
| low_queue | 4 / 4 | 0.10 / 0.14 | 0.04 [-0.05, 0.17] | +0.0 | unchanged |
| low_stopped_lead | 1 / 0 | 0.08 / 0.00 | -0.08 [-0.27, 0.00] | -0.5 | unchanged |
| low_other | 2 / 0 | 0.02 / 0.00 | -0.02 [-0.06, 0.00] | -1.0 | unchanged |
| coll_stationary | 0 / 4 | 0.00 / 0.43 | 0.43 [0.00, 0.97] | +2.0 | unchanged (trend worse) |

