# RSGboth vs RSG: solved / unsolved ledger on held 460 @18 (rf_ledger.py)

460 scenes common to all labels. Points = sum(1 - score) / 460 x 100 in the class; pair = RSGboth - RSG per scene (log-clustered 95 % CI). Group pairs (seeds averaged): n = scenes summed over both seeds, points = seed mean. Rules in the script docstring.

## RSGbothSB-L18-s0:RSG-L18-s0: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 78 / 11 | 4.17 / 0.64 | -3.53 [-4.72, -2.46] | -67.0 | improved |
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 8 / 7 | 1.74 / 1.52 | -0.22 [-1.16, 0.71] | -1.0 | unchanged (trend better) |
| offroad: curve / straight | 5 / 9 | 1.09 / 1.96 | 0.87 [-0.53, 2.12] | +4.0 | unchanged (trend worse) |
| stop: unexplained / late / stopped lead | 16 / 0 | 1.06 / 0.00 | -1.06 [-1.69, -0.54] | -16.0 | solved |
| collision: vehicle | 4 / 8 | 0.87 / 1.74 | 0.87 [-0.25, 2.02] | +4.0 | unchanged (trend worse) |
| corridor: route following | 4 / 3 | 0.87 / 0.65 | -0.22 [-0.94, 0.51] | -1.0 | unchanged (trend better) |
| ok (>= 0.9) | 326 / 401 | 0.47 / 0.06 | -0.41 [-0.58, -0.27] | +75.0 | improved (more scenes) |
| stop / rear / other low | 6 / 4 | 0.35 / 0.26 | -0.09 [-0.40, 0.09] | -2.0 | unchanged |
| corridor: past the log end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |

## RSGbothSB-L18-s0:RSG-L18-s0: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 7 / 13 | 1.52 / 2.83 | 1.30 [-0.22, 2.81] | +6.0 | unchanged (trend worse) |
| turn zero: inside (cut) | 5 / 2 | 1.09 / 0.43 | -0.65 [-1.42, 0.00] | -3.0 | unchanged (trend better) |
| junction zero: straight through (no turn) | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.75] | +1.0 | unchanged (trend worse) |
| turn zero: went straight at the turn | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |

## RSGbothSB-L18-s0:RSG-L18-s0: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 3 | 0.22 / 0.65 | 0.43 [0.00, 0.96] | +2.0 | unchanged (trend worse) |
| turn_completed | 16 / 18 | 0.12 / 0.00 | -0.12 [-0.30, -0.00] | +2.0 | unchanged |
| turn_short | 7 / 2 | 0.07 / 0.01 | -0.07 [-0.15, 0.00] | -5.0 | unchanged |
| turn_off | 0 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +1.0 | unchanged |

## RSGbothSB-L18-s0:RSG-L18-s0: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 62 / 8 | 3.17 / 0.52 | -2.66 [-3.53, -1.84] | -54.0 | improved |
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_slow_behind_lead | 16 / 3 | 0.99 / 0.12 | -0.87 [-1.57, -0.31] | -13.0 | improved |
| low_unexplained | 15 / 0 | 0.90 / 0.00 | -0.90 [-1.47, -0.43] | -15.0 | solved |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_outside | 3 / 3 | 0.65 / 0.65 | 0.00 [-0.66, 0.58] | +0.0 | unchanged |
| off_junc_inside | 3 / 1 | 0.65 / 0.22 | -0.43 [-1.06, 0.00] | -2.0 | unchanged (trend better) |
| cor_turn_under | 3 / 3 | 0.65 / 0.65 | 0.00 [-0.58, 0.66] | +0.0 | unchanged |
| ok | 326 / 401 | 0.47 / 0.06 | -0.41 [-0.58, -0.27] | +75.0 | improved (more scenes) |
| off_curve_inside | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.73, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_side | 2 / 0 | 0.43 / 0.00 | -0.43 [-1.16, 0.00] | -2.0 | unchanged (trend better) |
| off_junc_straight | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.75] | +1.0 | unchanged (trend worse) |
| off_straight | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.75, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_ahead | 1 / 3 | 0.22 / 0.65 | 0.43 [0.00, 1.09] | +2.0 | unchanged (trend worse) |
| cor_straight_at_turn | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| cor_route_end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |
| off_curve_outside | 1 / 7 | 0.22 / 1.52 | 1.30 [0.22, 2.35] | +6.0 | worse |
| coll_veh_oncoming | 1 / 3 | 0.22 / 0.65 | 0.43 [0.00, 1.06] | +2.0 | unchanged (trend worse) |
| low_rear_ended | 3 / 1 | 0.19 / 0.05 | -0.14 [-0.43, 0.00] | -2.0 | unchanged |
| low_stopped_lead | 1 / 0 | 0.16 / 0.00 | -0.16 [-0.55, 0.00] | -1.0 | unchanged |
| low_queue | 2 / 2 | 0.14 / 0.16 | 0.02 [-0.01, 0.08] | +0.0 | unchanged |
| low_other | 1 / 0 | 0.02 / 0.00 | -0.02 [-0.07, 0.00] | -1.0 | unchanged |
| low_with_log | 0 / 1 | 0.00 / 0.05 | 0.05 [0.00, 0.15] | +1.0 | unchanged |
| coll_stationary | 0 / 2 | 0.00 / 0.43 | 0.43 [0.00, 1.16] | +2.0 | unchanged (trend worse) |

## RSGbothSB-L18-s1:RSG-L18-s1: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 75 / 7 | 3.79 / 0.34 | -3.45 [-4.55, -2.40] | -68.0 | improved |
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 8 / 8 | 1.74 / 1.74 | 0.00 [-1.12, 1.35] | +0.0 | unchanged |
| stop: unexplained / late / stopped lead | 19 / 2 | 1.34 / 0.13 | -1.21 [-2.05, -0.52] | -17.0 | improved |
| collision: vehicle | 6 / 6 | 1.30 / 1.30 | 0.00 [-1.05, 1.03] | +0.0 | unchanged |
| corridor: route following | 6 / 4 | 1.30 / 0.87 | -0.43 [-1.16, 0.00] | -2.0 | unchanged (trend better) |
| ok (>= 0.9) | 326 / 409 | 0.50 / 0.10 | -0.39 [-0.55, -0.25] | +83.0 | improved (more scenes) |
| stop / rear / other low | 6 / 3 | 0.23 / 0.19 | -0.04 [-0.27, 0.24] | -3.0 | unchanged |
| corridor: past the log end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |
| offroad: curve / straight | 1 / 4 | 0.22 / 0.87 | 0.65 [-0.36, 1.87] | +3.0 | unchanged (trend worse) |

## RSGbothSB-L18-s1:RSG-L18-s1: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 6 / 7 | 1.30 / 1.52 | 0.22 [-0.51, 0.97] | +1.0 | unchanged (trend worse) |
| turn zero: inside (cut) | 4 / 3 | 0.87 / 0.65 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| junction zero: straight through (no turn) | 2 / 4 | 0.43 / 0.87 | 0.43 [0.00, 1.49] | +2.0 | unchanged (trend worse) |
| turn zero: went straight at the turn | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |

## RSGbothSB-L18-s1:RSG-L18-s1: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn_short | 9 / 1 | 0.17 / 0.02 | -0.14 [-0.29, -0.03] | -8.0 | improved |
| turn_completed | 14 / 21 | 0.09 / 0.01 | -0.08 [-0.23, 0.01] | +7.0 | improved (more scenes) |
| turn_off | 0 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +1.0 | unchanged |

## RSGbothSB-L18-s1:RSG-L18-s1: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 59 / 5 | 2.98 / 0.25 | -2.73 [-3.63, -1.85] | -54.0 | improved |
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_unexplained | 19 / 1 | 1.34 / 0.06 | -1.28 [-2.10, -0.63] | -18.0 | improved |
| cor_turn_under | 4 / 3 | 0.87 / 0.65 | -0.22 [-0.77, 0.00] | -1.0 | unchanged (trend better) |
| off_junc_inside | 4 / 1 | 0.87 / 0.22 | -0.65 [-1.39, 0.00] | -3.0 | unchanged (trend better) |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_slow_behind_lead | 16 / 2 | 0.81 / 0.09 | -0.72 [-1.28, -0.25] | -14.0 | improved |
| ok | 326 / 409 | 0.50 / 0.10 | -0.39 [-0.55, -0.25] | +83.0 | improved (more scenes) |
| cor_straight_at_turn | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_ahead | 2 / 3 | 0.43 / 0.65 | 0.22 [-0.49, 1.01] | +1.0 | unchanged (trend worse) |
| coll_veh_oncoming | 2 / 2 | 0.43 / 0.43 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_outside | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.71] | +1.0 | unchanged (trend worse) |
| off_junc_straight | 2 / 4 | 0.43 / 0.87 | 0.43 [0.00, 1.49] | +2.0 | unchanged (trend worse) |
| cor_route_end | 1 / 5 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |
| coll_veh_side | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_crossing | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.75, 0.00] | -1.0 | unchanged (trend better) |
| off_straight | 1 / 1 | 0.22 / 0.22 | 0.00 [-0.66, 0.62] | +0.0 | unchanged |
| low_rear_ended | 3 / 2 | 0.14 / 0.10 | -0.04 [-0.18, 0.12] | -1.0 | unchanged |
| low_queue | 2 / 1 | 0.06 / 0.08 | 0.02 [-0.14, 0.24] | -1.0 | unchanged |
| low_other | 1 / 0 | 0.02 / 0.00 | -0.02 [-0.08, 0.00] | -1.0 | unchanged |
| low_stopped_lead | 0 / 1 | 0.00 / 0.07 | 0.07 [0.00, 0.25] | +1.0 | unchanged |
| off_curve_inside | 0 / 2 | 0.00 / 0.43 | 0.43 [0.00, 1.09] | +2.0 | unchanged (trend worse) |
| off_curve_outside | 0 / 1 | 0.00 / 0.22 | 0.22 [0.00, 0.71] | +1.0 | unchanged (trend worse) |
| coll_stationary | 0 / 1 | 0.00 / 0.22 | 0.22 [0.00, 0.72] | +1.0 | unchanged (trend worse) |

## RSGbothSB-L18-s0:RSGboth-L18-s0: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 8 / 7 | 1.74 / 1.52 | -0.22 [-0.95, 0.46] | -1.0 | unchanged (trend better) |
| collision: vehicle | 7 / 8 | 1.52 / 1.74 | 0.22 [-0.45, 1.01] | +1.0 | unchanged (trend worse) |
| corridor: route following | 5 / 3 | 1.09 / 0.65 | -0.43 [-0.99, 0.00] | -2.0 | unchanged (trend better) |
| offroad: curve / straight | 5 / 9 | 1.09 / 1.96 | 0.87 [0.22, 1.55] | +4.0 | worse |
| corridor: past the log end | 4 / 5 | 0.87 / 1.09 | 0.22 [0.00, 0.67] | +1.0 | unchanged (trend worse) |
| slow | 10 / 11 | 0.59 / 0.64 | 0.04 [-0.01, 0.11] | +1.0 | unchanged |
| stop / rear / other low | 3 / 4 | 0.21 / 0.26 | 0.05 [-0.01, 0.15] | +1.0 | unchanged |
| ok (>= 0.9) | 406 / 401 | 0.09 / 0.06 | -0.02 [-0.07, 0.02] | -5.0 | unchanged |

## RSGbothSB-L18-s0:RSGboth-L18-s0: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 10 / 13 | 2.17 / 2.83 | 0.65 [-0.24, 1.48] | +3.0 | unchanged (trend worse) |
| junction zero: straight through (no turn) | 3 / 3 | 0.65 / 0.65 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn zero: inside (cut) | 3 / 2 | 0.65 / 0.43 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| turn zero: went straight at the turn | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |

## RSGbothSB-L18-s0:RSGboth-L18-s0: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 3 | 0.22 / 0.65 | 0.43 [0.00, 0.96] | +2.0 | unchanged (trend worse) |
| turn_short | 1 / 2 | 0.01 / 0.01 | -0.01 [-0.04, 0.02] | +1.0 | unchanged |
| turn_completed | 22 / 18 | 0.01 / 0.00 | -0.01 [-0.02, 0.00] | -4.0 | unchanged |
| turn_off | 0 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +1.0 | unchanged |

## RSGbothSB-L18-s0:RSGboth-L18-s0: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| cor_route_end | 4 / 5 | 0.87 / 1.09 | 0.22 [0.00, 0.67] | +1.0 | unchanged (trend worse) |
| cor_turn_under | 4 / 3 | 0.87 / 0.65 | -0.22 [-0.88, 0.51] | -1.0 | unchanged (trend better) |
| off_curve_outside | 3 / 7 | 0.65 / 1.52 | 0.87 [0.22, 1.55] | +4.0 | worse |
| off_junc_straight | 3 / 3 | 0.65 / 0.65 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_outside | 3 / 3 | 0.65 / 0.65 | 0.00 [-0.63, 0.58] | +0.0 | unchanged |
| low_slow_free | 8 / 8 | 0.49 / 0.52 | 0.02 [-0.01, 0.06] | +0.0 | unchanged |
| coll_stationary | 2 / 2 | 0.43 / 0.43 | 0.00 [-0.57, 0.68] | +0.0 | unchanged |
| coll_veh_ahead | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.71] | +1.0 | unchanged (trend worse) |
| off_junc_inside | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_oncoming | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.72] | +1.0 | unchanged (trend worse) |
| off_straight | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| cor_straight_at_turn | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_side | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.76, 0.00] | -1.0 | unchanged (trend better) |
| off_curve_inside | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_queue | 2 / 2 | 0.17 / 0.16 | -0.00 [-0.01, 0.00] | +0.0 | unchanged |
| low_slow_behind_lead | 2 / 3 | 0.10 / 0.12 | 0.02 [-0.01, 0.07] | +1.0 | unchanged |
| ok | 406 / 401 | 0.09 / 0.06 | -0.02 [-0.07, 0.02] | -5.0 | unchanged |
| low_rear_ended | 1 / 1 | 0.05 / 0.05 | -0.00 [-0.00, 0.00] | +0.0 | unchanged |
| low_with_log | 0 / 1 | 0.00 / 0.05 | 0.05 [0.00, 0.15] | +1.0 | unchanged |

## RSGbothSB-L18-s1:RSGboth-L18-s1: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra (no rollout / start pose) | 12 / 12 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| collision: vehicle | 11 / 6 | 2.39 / 1.30 | -1.09 [-2.00, -0.25] | -5.0 | improved |
| offroad: junction | 9 / 8 | 1.96 / 1.74 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| offroad: curve / straight | 6 / 4 | 1.30 / 0.87 | -0.43 [-1.04, 0.00] | -2.0 | unchanged (trend better) |
| corridor: past the log end | 5 / 5 | 1.09 / 1.09 | 0.00 [-0.66, 0.59] | +0.0 | unchanged |
| corridor: route following | 3 / 4 | 0.65 / 0.87 | 0.22 [0.00, 0.73] | +1.0 | unchanged (trend worse) |
| slow | 9 / 7 | 0.44 / 0.34 | -0.10 [-0.26, -0.00] | -2.0 | improved |
| stop / rear / other low | 4 / 3 | 0.25 / 0.19 | -0.06 [-0.15, -0.00] | -1.0 | improved |
| stop: unexplained / late / stopped lead | 1 / 2 | 0.06 / 0.13 | 0.07 [-0.02, 0.25] | +1.0 | unchanged |
| ok (>= 0.9) | 400 / 409 | 0.03 / 0.10 | 0.07 [0.02, 0.14] | +9.0 | improved (more scenes) |

## RSGbothSB-L18-s1:RSGboth-L18-s1: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 6 / 7 | 1.30 / 1.52 | 0.22 [0.00, 0.73] | +1.0 | unchanged (trend worse) |
| turn zero: inside (cut) | 5 / 3 | 1.09 / 0.65 | -0.43 [-1.07, 0.00] | -2.0 | unchanged (trend better) |
| junction zero: straight through (no turn) | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn zero: went straight at the turn | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |

## RSGbothSB-L18-s1:RSGboth-L18-s1: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn_short | 1 / 1 | 0.03 / 0.02 | -0.01 [-0.02, 0.00] | +0.0 | unchanged |
| turn_completed | 21 / 21 | 0.00 / 0.01 | 0.01 [0.00, 0.02] | +0.0 | unchanged |
| turn_off | 1 / 1 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +0.0 | unchanged |

## RSGbothSB-L18-s1:RSGboth-L18-s1: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra | 8 / 8 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| cor_route_end | 5 / 5 | 1.09 / 1.09 | 0.00 [-0.66, 0.59] | +0.0 | unchanged |
| off_at_start | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_veh_oncoming | 4 / 2 | 0.87 / 0.43 | -0.43 [-1.11, 0.00] | -2.0 | unchanged (trend better) |
| off_junc_straight | 4 / 4 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_veh_ahead | 4 / 3 | 0.87 / 0.65 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| off_junc_outside | 3 / 3 | 0.65 / 0.65 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_curve_inside | 3 / 2 | 0.65 / 0.43 | -0.22 [-0.73, 0.00] | -1.0 | unchanged (trend better) |
| coll_stationary | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| off_straight | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.64, 0.00] | -1.0 | unchanged (trend better) |
| cor_turn_under | 2 / 3 | 0.43 / 0.65 | 0.22 [0.00, 0.73] | +1.0 | unchanged (trend worse) |
| off_junc_inside | 2 / 1 | 0.43 / 0.22 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| low_slow_free | 6 / 5 | 0.32 / 0.25 | -0.08 [-0.23, 0.01] | -1.0 | unchanged |
| off_curve_outside | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| cor_straight_at_turn | 1 / 1 | 0.22 / 0.22 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_veh_side | 1 / 0 | 0.22 / 0.00 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| low_rear_ended | 2 / 2 | 0.14 / 0.10 | -0.03 [-0.11, 0.00] | +0.0 | unchanged |
| low_slow_behind_lead | 3 / 2 | 0.12 / 0.09 | -0.03 [-0.08, 0.00] | -1.0 | unchanged |
| low_queue | 2 / 1 | 0.11 / 0.08 | -0.03 [-0.08, 0.00] | -1.0 | unchanged |
| low_unexplained | 1 / 1 | 0.06 / 0.06 | -0.01 [-0.02, 0.00] | +0.0 | unchanged |
| ok | 400 / 409 | 0.03 / 0.10 | 0.07 [0.02, 0.14] | +9.0 | improved (more scenes) |
| low_stopped_lead | 0 / 1 | 0.00 / 0.07 | 0.07 [0.00, 0.25] | +1.0 | unchanged |

## SB-RSG: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| slow | 153 / 18 | 3.98 / 0.49 | -3.49 [-4.58, -2.46] | -67.5 | improved |
| infra (no rollout / start pose) | 24 / 24 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| offroad: junction | 16 / 15 | 1.74 / 1.63 | -0.11 [-1.04, 0.92] | -0.5 | unchanged |
| stop: unexplained / late / stopped lead | 35 / 2 | 1.20 / 0.07 | -1.13 [-1.80, -0.59] | -16.5 | improved |
| collision: vehicle | 10 / 14 | 1.09 / 1.52 | 0.43 [-0.60, 1.45] | +2.0 | unchanged (trend worse) |
| corridor: route following | 10 / 7 | 1.09 / 0.76 | -0.33 [-0.92, 0.20] | -1.5 | unchanged (trend better) |
| offroad: curve / straight | 6 / 13 | 0.65 / 1.41 | 0.76 [-0.28, 1.77] | +3.5 | unchanged (trend worse) |
| ok (>= 0.9) | 652 / 810 | 0.48 / 0.08 | -0.40 [-0.55, -0.27] | +79.0 | improved (more scenes) |
| stop / rear / other low | 12 / 7 | 0.29 / 0.22 | -0.07 [-0.29, 0.13] | -2.5 | unchanged |
| corridor: past the log end | 2 / 10 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |

## SB-RSG: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 13 / 20 | 1.41 / 2.17 | 0.76 [-0.14, 1.65] | +3.5 | unchanged (trend worse) |
| turn zero: inside (cut) | 9 / 5 | 0.98 / 0.54 | -0.43 [-0.97, 0.00] | -2.0 | unchanged (trend better) |
| junction zero: straight through (no turn) | 4 / 7 | 0.43 / 0.76 | 0.33 [0.00, 1.12] | +1.5 | unchanged (trend worse) |
| turn zero: went straight at the turn | 3 / 1 | 0.33 / 0.11 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |

## SB-RSG: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 2 / 4 | 0.22 / 0.43 | 0.22 [0.00, 0.48] | +1.0 | unchanged (trend worse) |
| turn_short | 16 / 3 | 0.12 / 0.02 | -0.11 [-0.20, -0.03] | -6.5 | improved |
| turn_completed | 30 / 39 | 0.10 / 0.00 | -0.10 [-0.25, -0.01] | +4.5 | improved (more scenes) |
| turn_off | 0 / 2 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +1.0 | unchanged |

## SB-RSG: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| low_slow_free | 121 / 13 | 3.08 / 0.38 | -2.70 [-3.50, -1.90] | -54.0 | improved |
| infra | 16 / 16 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| low_unexplained | 34 / 1 | 1.12 / 0.03 | -1.09 [-1.76, -0.57] | -16.5 | solved |
| low_slow_behind_lead | 32 / 5 | 0.90 / 0.10 | -0.79 [-1.39, -0.31] | -13.5 | improved |
| off_at_start | 8 / 8 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_inside | 7 / 2 | 0.76 / 0.22 | -0.54 [-1.19, 0.00] | -2.5 | unchanged (trend better) |
| cor_turn_under | 7 / 6 | 0.76 / 0.65 | -0.11 [-0.49, 0.25] | -0.5 | unchanged |
| off_junc_outside | 5 / 6 | 0.54 / 0.65 | 0.11 [-0.25, 0.45] | +0.5 | unchanged |
| ok | 652 / 810 | 0.48 / 0.08 | -0.40 [-0.55, -0.27] | +79.0 | improved (more scenes) |
| off_junc_straight | 4 / 7 | 0.43 / 0.76 | 0.33 [0.00, 1.12] | +1.5 | unchanged (trend worse) |
| coll_veh_ahead | 3 / 6 | 0.33 / 0.65 | 0.33 [-0.13, 0.94] | +1.5 | unchanged (trend worse) |
| cor_straight_at_turn | 3 / 1 | 0.33 / 0.11 | -0.22 [-0.71, 0.00] | -1.0 | unchanged (trend better) |
| coll_veh_side | 3 / 0 | 0.33 / 0.00 | -0.33 [-0.88, 0.00] | -1.5 | unchanged (trend better) |
| coll_veh_oncoming | 3 / 5 | 0.33 / 0.54 | 0.22 [0.00, 0.53] | +1.0 | unchanged (trend worse) |
| off_straight | 3 / 2 | 0.33 / 0.22 | -0.11 [-0.69, 0.29] | -0.5 | unchanged |
| cor_route_end | 2 / 10 | 0.22 / 1.09 | 0.87 [0.22, 1.60] | +4.0 | worse |
| off_curve_inside | 2 / 3 | 0.22 / 0.33 | 0.11 [-0.25, 0.48] | +0.5 | unchanged |
| low_rear_ended | 6 / 3 | 0.16 / 0.07 | -0.09 [-0.27, 0.04] | -1.5 | unchanged |
| off_curve_outside | 1 / 8 | 0.11 / 0.87 | 0.76 [0.12, 1.39] | +3.5 | worse |
| coll_veh_crossing | 1 / 0 | 0.11 / 0.00 | -0.11 [-0.38, 0.00] | -0.5 | unchanged |
| low_queue | 4 / 3 | 0.10 / 0.12 | 0.02 [-0.07, 0.16] | -0.5 | unchanged |
| low_stopped_lead | 1 / 1 | 0.08 / 0.04 | -0.04 [-0.24, 0.10] | +0.0 | unchanged |
| low_other | 2 / 0 | 0.02 / 0.00 | -0.02 [-0.06, 0.00] | -1.0 | unchanged |
| low_with_log | 0 / 1 | 0.00 / 0.03 | 0.03 [0.00, 0.08] | +0.5 | unchanged |
| coll_stationary | 0 / 3 | 0.00 / 0.33 | 0.33 [0.00, 0.88] | +1.5 | unchanged (trend worse) |

## SB-RSGboth: by taxonomy bucket

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra (no rollout / start pose) | 24 / 24 | 2.61 / 2.61 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| collision: vehicle | 18 / 14 | 1.96 / 1.52 | -0.43 [-0.83, -0.11] | -2.0 | improved |
| offroad: junction | 17 / 15 | 1.85 / 1.63 | -0.22 [-0.76, 0.23] | -1.0 | unchanged (trend better) |
| offroad: curve / straight | 11 / 13 | 1.20 / 1.41 | 0.22 [-0.22, 0.62] | +1.0 | unchanged (trend worse) |
| corridor: past the log end | 9 / 10 | 0.98 / 1.09 | 0.11 [-0.32, 0.61] | +0.5 | unchanged |
| corridor: route following | 8 / 7 | 0.87 / 0.76 | -0.11 [-0.44, 0.26] | -0.5 | unchanged |
| slow | 19 / 18 | 0.52 / 0.49 | -0.03 [-0.10, 0.02] | -0.5 | unchanged |
| stop / rear / other low | 7 / 7 | 0.23 / 0.22 | -0.01 [-0.07, 0.06] | +0.0 | unchanged |
| ok (>= 0.9) | 806 / 810 | 0.06 / 0.08 | 0.03 [-0.01, 0.07] | +2.0 | unchanged |
| stop: unexplained / late / stopped lead | 1 / 2 | 0.03 / 0.07 | 0.03 [-0.01, 0.12] | +0.5 | unchanged |

## SB-RSGboth: turn-related zeros

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn zero: outside (wide / under-turned) | 16 / 20 | 1.74 / 2.17 | 0.43 [0.00, 0.89] | +2.0 | unchanged (trend worse) |
| turn zero: inside (cut) | 8 / 5 | 0.87 / 0.54 | -0.33 [-0.82, 0.00] | -1.5 | unchanged (trend better) |
| junction zero: straight through (no turn) | 7 / 7 | 0.76 / 0.76 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| turn zero: went straight at the turn | 2 / 1 | 0.22 / 0.11 | -0.11 [-0.36, 0.00] | -0.5 | unchanged |

## SB-RSGboth: 24 strict turns by rf_follow outcome

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| turn_event | 2 / 4 | 0.22 / 0.43 | 0.22 [0.00, 0.48] | +1.0 | unchanged (trend worse) |
| turn_short | 2 / 3 | 0.02 / 0.02 | -0.01 [-0.02, 0.01] | +0.5 | unchanged |
| turn_completed | 43 / 39 | 0.00 / 0.00 | -0.00 [-0.01, 0.01] | -2.0 | unchanged |
| turn_off | 1 / 2 | 0.00 / 0.00 | 0.00 [0.00, 0.00] | +0.5 | unchanged |

## SB-RSGboth: by rf_tax primary class

| class | n RSG / RSGboth | points RSG / RSGboth | diff points [CI] | diff count | verdict |
|:--|:--|:--|:--|--:|:--|
| infra | 16 / 16 | 1.74 / 1.74 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| cor_route_end | 9 / 10 | 0.98 / 1.09 | 0.11 [-0.32, 0.61] | +0.5 | unchanged |
| off_at_start | 8 / 8 | 0.87 / 0.87 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| off_junc_straight | 7 / 7 | 0.76 / 0.76 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_veh_oncoming | 6 / 5 | 0.65 / 0.54 | -0.11 [-0.36, 0.00] | -0.5 | unchanged |
| off_junc_outside | 6 / 6 | 0.65 / 0.65 | 0.00 [-0.32, 0.29] | +0.0 | unchanged |
| cor_turn_under | 6 / 6 | 0.65 / 0.65 | 0.00 [-0.39, 0.47] | +0.0 | unchanged |
| coll_veh_ahead | 6 / 6 | 0.65 / 0.65 | 0.00 [0.00, 0.00] | +0.0 | unchanged |
| coll_stationary | 4 / 3 | 0.43 / 0.33 | -0.11 [-0.46, 0.26] | -0.5 | unchanged |
| off_junc_inside | 4 / 2 | 0.43 / 0.22 | -0.22 [-0.67, 0.00] | -1.0 | unchanged (trend better) |
| off_curve_outside | 4 / 8 | 0.43 / 0.87 | 0.43 [0.11, 0.78] | +2.0 | worse |
| off_curve_inside | 4 / 3 | 0.43 / 0.33 | -0.11 [-0.36, 0.00] | -0.5 | unchanged |
| low_slow_free | 14 / 13 | 0.41 / 0.38 | -0.03 [-0.09, 0.01] | -0.5 | unchanged |
| off_straight | 3 / 2 | 0.33 / 0.22 | -0.11 [-0.32, 0.00] | -0.5 | unchanged |
| cor_straight_at_turn | 2 / 1 | 0.22 / 0.11 | -0.11 [-0.36, 0.00] | -0.5 | unchanged |
| coll_veh_side | 2 / 0 | 0.22 / 0.00 | -0.22 [-0.58, 0.00] | -1.0 | unchanged (trend better) |
| low_queue | 4 / 3 | 0.14 / 0.12 | -0.01 [-0.04, 0.00] | -0.5 | unchanged |
| low_slow_behind_lead | 5 / 5 | 0.11 / 0.10 | -0.00 [-0.04, 0.03] | +0.0 | unchanged |
| low_rear_ended | 3 / 3 | 0.09 / 0.07 | -0.02 [-0.06, 0.00] | +0.0 | unchanged |
| ok | 806 / 810 | 0.06 / 0.08 | 0.03 [-0.01, 0.07] | +2.0 | unchanged |
| low_unexplained | 1 / 1 | 0.03 / 0.03 | -0.00 [-0.01, 0.00] | +0.0 | unchanged |
| low_with_log | 0 / 1 | 0.00 / 0.03 | 0.03 [0.00, 0.08] | +0.5 | unchanged |
| low_stopped_lead | 0 / 1 | 0.00 / 0.04 | 0.04 [0.00, 0.13] | +0.5 | unchanged |

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

