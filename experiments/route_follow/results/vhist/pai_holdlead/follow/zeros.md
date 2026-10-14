# Corridor zeros by map topology at the failure (rf_pai_follow.py)

Class rules in the script docstring. Route zeros = the RSG taxonomy's cor_straight_at_turn + cor_turn_under (22).

| class | rsg all corridor (47) | holdlead all corridor (49) | rsg route zeros (22) |
|:--|--:|--:|--:|
| straight exit drivable: the ego lane's own successor | 14 | 14 | 11 |
| straight exit drivable: another lane runs straight ahead (ego lane turn-only) | 3 | 3 | 3 |
| T / no straight exit (branches, all turn) | 1 | 2 | 1 |
| no straight exit: turn-only lane, one chain, no straight lane ahead | 7 | 7 | 6 |
| lane change needed | 1 | 2 | 1 |
| no route turn: beyond the log's end | 12 | 12 | 0 |
| no route turn: other | 8 | 9 | 0 |
| other (turn) | 0 | 0 | 0 |
| other (no lane) | 1 | 0 | 0 |

rsg, route22 zeros: speed at the junction (median m/s, ego / log, median ratio, ego faster than log by > 1 m/s):

| station | ego | log | ratio | faster / n |
|:--|--:|--:|--:|--:|
| entry - 40 m | 4.9 | 4.0 | 1.09 | 8 / 22 |
| entry - 20 m | 5.7 | 6.2 | 1.02 | 8 / 22 |
| entry - 0 m | 9.0 | 5.1 | 1.41 | 12 / 20 |
| slowed into the entry (v at 0 < max(v at -40, -20) - 1 m/s; median change) | 0 / 20 (+1.7) | 10 / 20 (-0.9) | | |

rsg, turn zeros: speed at the junction (median m/s, ego / log, median ratio, ego faster than log by > 1 m/s):

| station | ego | log | ratio | faster / n |
|:--|--:|--:|--:|--:|
| entry - 40 m | 5.5 | 6.8 | 1.10 | 11 / 27 |
| entry - 20 m | 5.9 | 6.2 | 1.13 | 12 / 27 |
| entry - 0 m | 10.7 | 5.0 | 1.47 | 14 / 23 |
| slowed into the entry (v at 0 < max(v at -40, -20) - 1 m/s; median change) | 0 / 23 (+1.2) | 12 / 23 (-1.0) | | |

holdlead, turn zeros: speed at the junction (median m/s, ego / log, median ratio, ego faster than log by > 1 m/s):

| station | ego | log | ratio | faster / n |
|:--|--:|--:|--:|--:|
| entry - 40 m | 7.0 | 7.1 | 1.07 | 11 / 28 |
| entry - 20 m | 7.0 | 6.2 | 1.09 | 12 / 28 |
| entry - 0 m | 9.5 | 5.2 | 1.43 | 14 / 24 |
| slowed into the entry (v at 0 < max(v at -40, -20) - 1 m/s; median change) | 1 / 24 (+1.1) | 13 / 24 (-1.1) | | |

## rsg: 47 corridor zeros

| scene | taxonomy | class | turn deg | ego on route lane | ego paths (deg) | route LC | v ego / log at -40 | -20 | 0 m |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| 3ce13eed | cor_turn_under **R** | curve_single | -100 | True | [-96.9] | False | 1.9 / 1.1 | 1.9 / 1.1 | 3.7 / 3.3 |
| 68e53160 | cor_turn_under **R** | curve_single | +108 | True | [112.4] | False | 0.0 / 0.0 | 0.0 / 0.0 | 5.1 / 4.5 |
| a1b166cc | cor_turn_under **R** | curve_single | +91 | True | [87.5] | False | 0.0 / 0.0 | 0.0 / 0.0 | 3.8 / 4.9 |
| cdf95fd1 | cor_turn_under **R** | curve_single | +93 | True | [98.8] | False | 8.3 / 11.0 | 8.1 / 10.2 | 10.7 / 7.9 |
| ddc3e8df | cor_turn_under **R** | curve_single | -66 | True | [-50.0] | False | 2.5 / 1.5 | 2.5 / 1.5 | 2.6 / 2.5 |
| e37aa159 | cor_turn_under **R** | curve_single | +82 | True | [86.6] | False | 4.6 / 3.1 | 5.6 / 6.1 | 7.4 / 8.8 |
| bd97de34 | cor_straight_at_turn **R** | lane_change | +97 | False | [1.7] | True | 12.1 / 9.5 | 12.6 / 8.9 | 12.7 / 6.8 |
| fd01e9ab | cor_turn_under **R** | no_straight_exit | +97 | True | [96.3, 93.3, 173.4, 173.8, 172.8] | False | 0.0 / 0.0 | 0.0 / 0.0 | 3.7 / 5.7 |
| 00fabf60 | cor_turn_under **R** | straight_exit | +85 | True | [102.6, 107.1, -19.7] | False | 5.5 / 4.8 | 8.6 / 8.3 | 11.1 / 7.6 |
| 0a228e44 | cor_straight_at_turn **R** | straight_exit | -73 | True | [-71.0, 104.5, 103.6, 14.1] | False | 15.5 / 5.9 | 16.4 / 6.7 | 17.2 / 5.7 |
| 3232fdb7 | cor_straight_at_turn **R** | straight_exit | -87 | True | [64.4, 72.8, 79.7, 86.6, -1.4, -188.7, -88.5] | False | 5.1 / 9.1 | 3.7 / 9.1 | nan / 5.3 |
| 37d660b1 | cor_turn_under **R** | straight_exit | -85 | True | [5.7, -83.5] | False | 3.4 / 2.0 | 3.6 / 3.6 | 4.7 / 5.5 |
| 403d558a | cor_straight_at_turn **R** | straight_exit | -93 | True | [-99.9, -7.6] | False | 11.3 / 6.8 | 9.8 / 9.8 | 11.8 / 6.2 |
| 52003cee | cor_straight_at_turn **R** | straight_exit | -90 | True | [-0.0, -88.7] | False | 13.3 / 8.0 | 13.8 / 7.1 | 14.8 / 3.2 |
| a174faed | cor_straight_at_turn **R** | straight_exit | -90 | True | [-89.7, -0.6] | False | 2.0 / 2.0 | 5.8 / 6.2 | 6.2 / 4.6 |
| b680a689 | cor_turn_under **R** | straight_exit | +90 | True | [-0.3, 88.9, -87.2] | False | 5.8 / 7.3 | 3.0 / 6.5 | nan / 0.8 |
| eaba3f6a | cor_straight_at_turn **R** | straight_exit | -90 | True | [-0.3, 103.9, -87.0] | False | 9.5 / 9.6 | 10.4 / 5.3 | 12.5 / 4.3 |
| f9e361aa | cor_straight_at_turn **R** | straight_exit | -81 | True | [-0.1, 97.5, -79.8] | False | 8.6 / 8.5 | 10.3 / 8.0 | 11.4 / 5.3 |
| fda39805 | cor_turn_under **R** | straight_exit | -85 | True | [0.3, -71.7, -89.1, -95.6] | False | 1.4 / 0.9 | 1.4 / 0.9 | 4.1 / 3.6 |
| 2f227f2c | cor_straight_at_turn **R** | straight_road | -66 | True | [-66.8] | False | 0.1 / 0.0 | 5.9 / 4.8 | 10.7 / 4.4 |
| 302c5c99 | cor_straight_at_turn **R** | straight_road | +163 | True | [178.2, 98.9] | False | 2.4 / 1.3 | 5.0 / 5.0 | 6.6 / 3.7 |
| e5e47fd2 | cor_straight_at_turn **R** | straight_road | +95 | True | [144.4, 136.1] | False | 10.5 / 10.4 | 12.6 / 11.2 | 13.8 / 6.5 |
| 054b5901 | cor_route_end | beyond_log_end | - |  |  |  | 15.8 / 11.1 | 16.1 / 7.3 | 15.7 / nan |
| 127c4f85 | cor_route_end | beyond_log_end | - |  |  |  | 12.0 / 9.8 | 9.3 / 6.9 | 1.4 / nan |
| 2d10a08d | cor_route_end | beyond_log_end | - |  |  |  | 16.0 / 14.8 | 16.3 / 9.3 | 16.6 / nan |
| 35e082a4 | cor_route_end | beyond_log_end | - |  |  |  | 9.9 / 6.1 | 6.9 / 7.0 | 8.8 / nan |
| 3fe2bac8 | cor_route_end | beyond_log_end | - |  |  |  | 0.4 / 0.3 | 0.4 / 0.3 | 0.4 / 0.3 |
| 44df49fe | cor_route_end | beyond_log_end | - |  |  |  | 12.7 / 9.9 | 13.4 / 7.1 | 14.5 / nan |
| 4cb7f7ac | cor_route_end | beyond_log_end | - |  |  |  | 13.2 / 10.0 | 13.3 / 6.7 | 12.8 / nan |
| 4ecc114d | cor_route_end | beyond_log_end | - |  |  |  | 13.3 / 11.9 | 14.3 / 9.9 | 14.6 / nan |
| 559d4ad6 | cor_route_end | beyond_log_end | - |  |  |  | 11.8 / 10.8 | 11.3 / 7.4 | 12.8 / nan |
| 68c3be44 | cor_route_end | beyond_log_end | - |  |  |  | 12.0 / 10.5 | 13.8 / 7.4 | 15.1 / nan |
| 9f8acd00 | cor_route_end | beyond_log_end | - |  |  |  | 15.9 / 10.6 | 16.2 / 7.3 | 16.4 / nan |
| bc309fd6 | cor_route_end | beyond_log_end | - |  |  |  | 11.3 / 11.4 | 10.5 / 8.0 | 13.6 / nan |
| ba40be62 | cor_route_end | curve_single | +86 | True | [87.7] | False | 11.6 / 10.5 | 12.3 / 5.6 | nan / nan |
| ba62e10d | cor_drift | other_no_lane | +74 | False | [] | False | 4.0 / 11.0 | 3.4 / 6.5 | nan / 4.6 |
| 32be03e4 | cor_wrong_exit | other_no_turn | - |  |  |  | 38.1 / 38.2 | 38.0 / 38.0 | 35.1 / 37.9 |
| 6e6cbe67 | cor_drift | other_no_turn | - |  |  |  | 20.8 / 22.4 | 19.8 / 22.9 | 19.0 / 23.3 |
| 6ee9ee78 | cor_drift | other_no_turn | - |  |  |  | 8.8 / 9.2 | 3.4 / 9.2 | 3.1 / 9.1 |
| 7cfa81de | cor_drift | other_no_turn | - |  |  |  | 14.1 / 11.1 | 14.3 / 11.3 | 14.1 / 11.5 |
| 7d994ed3 | cor_route_end | other_no_turn | - |  |  |  | 11.0 / 10.8 | 12.8 / 10.4 | 14.0 / 9.4 |
| a2c5ae46 | cor_drift | other_no_turn | - |  |  |  | 15.7 / 12.5 | 15.8 / 12.6 | 15.9 / 12.5 |
| e904e9c0 | cor_lane_change | other_no_turn | - |  |  |  | 12.7 / 12.2 | 13.2 / 12.8 | 14.2 / 13.0 |
| eeba2aa9 | cor_drift | other_no_turn | - |  |  |  | 17.3 / 18.9 | 17.7 / 18.9 | 17.8 / 19.0 |
| 095cf563 | cor_route_end | straight_exit | -90 | True | [2.9, -90.0] | False | 6.7 / 7.8 | 7.9 / 2.6 | 7.8 / 7.4 |
| cbd87234 | cor_route_end | straight_exit | -90 | True | [-0.3, 105.7, -88.0] | False | 14.0 / 9.8 | 14.0 / 8.6 | 14.3 / 4.4 |
| fc9fde87 | cor_route_end | straight_exit | -88 | True | [-86.5, 91.5, 2.2] | False | 15.2 / 8.8 | 14.9 / 8.0 | 15.4 / 5.1 |

## holdlead: 49 corridor zeros

| scene | taxonomy | class | turn deg | ego on route lane | ego paths (deg) | route LC | v ego / log at -40 | -20 | 0 m |
|:--|:--|:--|--:|:--|:--|:--|:--|:--|:--|
| 054b5901 | - | beyond_log_end | - |  |  |  | 15.6 / 11.0 | 16.0 / 7.1 | 15.7 / nan |
| 127c4f85 | - | beyond_log_end | - |  |  |  | 11.9 / 9.8 | 9.1 / 6.9 | 1.6 / nan |
| 2d10a08d | - | beyond_log_end | - |  |  |  | 16.1 / 14.8 | 16.3 / 9.3 | 16.6 / nan |
| 35e082a4 | - | beyond_log_end | - |  |  |  | 9.9 / 6.1 | 6.9 / 7.0 | 8.6 / nan |
| 3fe2bac8 | - | beyond_log_end | - |  |  |  | 0.4 / 0.3 | 0.4 / 0.3 | 0.4 / 0.3 |
| 44df49fe | - | beyond_log_end | - |  |  |  | 12.7 / 9.9 | 13.4 / 7.1 | 14.6 / nan |
| 4cb7f7ac | - | beyond_log_end | - |  |  |  | 12.9 / 10.0 | 13.1 / 6.7 | 12.9 / nan |
| 4ecc114d | - | beyond_log_end | - |  |  |  | 12.9 / 11.8 | 13.7 / 9.8 | 14.1 / nan |
| 559d4ad6 | - | beyond_log_end | - |  |  |  | 12.0 / 10.9 | 11.5 / 7.6 | 12.4 / nan |
| 68c3be44 | - | beyond_log_end | - |  |  |  | 11.6 / 10.5 | 13.6 / 7.5 | 15.1 / nan |
| 9f8acd00 | - | beyond_log_end | - |  |  |  | 15.8 / 10.5 | 16.2 / 7.1 | 16.4 / nan |
| bc309fd6 | - | beyond_log_end | - |  |  |  | 11.4 / 11.5 | 10.4 / 8.1 | 13.0 / nan |
| 3ce13eed | - | curve_single | -100 | True | [-96.9] | False | 1.9 / 1.1 | 1.9 / 1.1 | 3.7 / 3.3 |
| 68e53160 | - | curve_single | +108 | True | [112.4] | False | 0.0 / 0.0 | 0.0 / 0.0 | 5.1 / 4.5 |
| a1b166cc | - | curve_single | +91 | True | [87.5] | False | 0.0 / 0.0 | 0.0 / 0.0 | 3.8 / 4.9 |
| ba40be62 | - | curve_single | +86 | True | [87.7] | False | 11.6 / 10.5 | 12.3 / 5.6 | nan / nan |
| cdf95fd1 | - | curve_single | +93 | True | [98.8] | False | 8.2 / 11.0 | 8.6 / 10.2 | 10.9 / 7.9 |
| ddc3e8df | - | curve_single | -66 | True | [-50.0] | False | 2.5 / 1.5 | 2.5 / 1.5 | 2.6 / 2.5 |
| e37aa159 | - | curve_single | +82 | True | [86.6] | False | 4.6 / 3.1 | 5.6 / 6.1 | 7.4 / 8.8 |
| bd97de34 | - | lane_change | +97 | False | [1.7] | True | 12.1 / 9.5 | 12.6 / 8.9 | 12.7 / 6.8 |
| c4a58f83 | - | lane_change | -85 | True | [0.2] | False | 12.2 / 16.1 | 10.3 / 12.7 | 4.9 / 7.8 |
| 444fca69 | - | no_straight_exit | +95 | True | [95.0, 96.1, 92.9] | False | 7.5 / 8.1 | 2.4 / 4.2 | nan / nan |
| fd01e9ab | - | no_straight_exit | +97 | True | [96.3, 93.3, 173.4, 173.8, 172.8] | False | 0.0 / 0.0 | 0.0 / 0.0 | 3.7 / 5.7 |
| 32be03e4 | - | other_no_turn | - |  |  |  | 38.1 / 38.2 | 38.0 / 38.0 | 35.1 / 37.9 |
| 6e6cbe67 | - | other_no_turn | - |  |  |  | 20.8 / 22.4 | 19.8 / 22.9 | 19.0 / 23.3 |
| 6ee9ee78 | - | other_no_turn | - |  |  |  | 8.8 / 9.2 | 3.4 / 9.2 | 3.1 / 9.1 |
| 7750265d | - | other_no_turn | - |  |  |  | 12.0 / 14.7 | 9.0 / 11.2 | 6.8 / 6.0 |
| 7cfa81de | - | other_no_turn | - |  |  |  | 14.1 / 11.1 | 14.3 / 11.3 | 14.2 / 11.5 |
| 7d994ed3 | - | other_no_turn | - |  |  |  | 11.0 / 10.8 | 12.8 / 10.4 | 14.0 / 9.4 |
| a2c5ae46 | - | other_no_turn | - |  |  |  | 15.7 / 12.4 | 15.8 / 12.6 | 15.9 / 12.5 |
| e904e9c0 | - | other_no_turn | - |  |  |  | 12.7 / 12.2 | 13.2 / 12.8 | 14.2 / 13.0 |
| eeba2aa9 | - | other_no_turn | - |  |  |  | 17.3 / 18.9 | 17.7 / 18.9 | 17.8 / 19.0 |
| 00fabf60 | - | straight_exit | +85 | True | [103.6, 107.1, -18.4] | False | 5.5 / 4.8 | 8.7 / 8.3 | 11.2 / 7.6 |
| 095cf563 | - | straight_exit | -90 | True | [2.9, -89.9] | False | 6.5 / 7.8 | 7.9 / 2.6 | 8.2 / 7.4 |
| 0a228e44 | - | straight_exit | -73 | True | [-71.0, 104.5, 103.6, 14.1] | False | 15.5 / 5.9 | 16.4 / 6.7 | 17.2 / 5.7 |
| 3232fdb7 | - | straight_exit | -87 | True | [64.7, 73.2, 80.0, 86.8, -1.4, -189.3, -88.4] | False | 7.8 / 9.1 | 6.0 / 9.1 | nan / 5.3 |
| 37d660b1 | - | straight_exit | -85 | True | [5.7, -83.5] | False | 3.4 / 2.0 | 3.6 / 3.6 | 4.7 / 5.5 |
| 403d558a | - | straight_exit | -93 | True | [-99.9, -7.6] | False | 11.3 / 6.8 | 9.8 / 9.8 | 11.8 / 6.2 |
| 52003cee | - | straight_exit | -90 | True | [-0.0, -88.7] | False | 13.3 / 8.0 | 13.8 / 7.1 | 14.8 / 3.2 |
| a174faed | - | straight_exit | -90 | True | [-89.7, -0.6] | False | 2.0 / 2.0 | 5.8 / 6.2 | 6.2 / 4.6 |
| b680a689 | - | straight_exit | +90 | True | [-0.3, 88.8, -87.0] | False | 5.8 / 7.3 | 3.3 / 6.5 | nan / 0.8 |
| cbd87234 | - | straight_exit | -90 | True | [-0.3, 105.7, -88.0] | False | 14.0 / 9.8 | 14.1 / 8.6 | 14.4 / 4.4 |
| eaba3f6a | - | straight_exit | -90 | True | [-0.3, 103.8, -87.0] | False | 9.5 / 9.6 | 10.5 / 5.3 | 12.5 / 4.3 |
| f9e361aa | - | straight_exit | -81 | True | [-0.1, 97.5, -79.8] | False | 8.6 / 8.5 | 10.3 / 8.0 | 11.4 / 5.3 |
| fc9fde87 | - | straight_exit | -88 | True | [-86.5, 91.5, 2.2] | False | 15.2 / 8.8 | 14.9 / 8.0 | 15.4 / 5.1 |
| fda39805 | - | straight_exit | -85 | True | [0.3, -71.7, -89.1, -95.6] | False | 1.4 / 0.9 | 1.4 / 0.9 | 4.1 / 3.6 |
| 2f227f2c | - | straight_road | -66 | True | [-66.8] | False | 0.1 / 0.0 | 5.9 / 4.8 | 10.7 / 4.4 |
| 302c5c99 | - | straight_road | +163 | True | [178.2, 98.9] | False | 2.4 / 1.3 | 5.0 / 5.0 | 6.6 / 3.7 |
| e5e47fd2 | - | straight_road | +95 | True | [144.4, 136.1] | False | 10.5 / 10.4 | 12.6 / 11.2 | 13.8 / 6.5 |
