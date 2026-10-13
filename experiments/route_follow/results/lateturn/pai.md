# PAI 36 strict turns by class (rf_lateturn.py pai; rf_pai_follow.py's turns and outcomes, no new rollouts)

PAI clips are 20 s: window = the log's whole travel. late = entry >= 40 m after the ego's start station; fork rule as nuPlan (route vs the ego lane's straight chain on the PAI map). Scene bootstrap 95 % CI.

| class | label | turns | completion | completed / event / off / short |
|:--|:--|--:|:--|:--|
| late (all) | rsg | 15 | 7 % [0, 20] | 1 / 8 / 2 / 4 |
| early (all) | rsg | 21 | 19 % [5, 38] | 4 / 13 / 4 / 0 |
| fork (all) | rsg | 17 | 6 % [0, 18] | 1 / 8 / 4 / 4 |
| late, no fork | rsg | 5 | 20 % [0, 60] | 1 / 4 / 0 / 0 |
| early, no fork | rsg | 14 | 21 % [0, 43] | 3 / 9 / 2 / 0 |
| all | rsg | 36 | 14 % [3, 25] | 5 / 21 / 6 / 4 |
| late (all) | rbsg2 | 15 | 7 % [0, 20] | 1 / 9 / 0 / 5 |
| early (all) | rbsg2 | 21 | 38 % [19, 57] | 8 / 12 / 1 / 0 |
| fork (all) | rbsg2 | 17 | 6 % [0, 18] | 1 / 11 / 0 / 5 |
| late, no fork | rbsg2 | 5 | 20 % [0, 60] | 1 / 4 / 0 / 0 |
| early, no fork | rbsg2 | 14 | 50 % [21, 79] | 7 / 6 / 1 / 0 |
| all | rbsg2 | 36 | 25 % [11, 39] | 9 / 21 / 1 / 5 |

Fork kinds (rsg turns): branch 16, none 14, lane_offset 4, chain_end 1, lc 1

Entry station after the start (rsg turns, m): 3, 3, 4, 5, 7, 7, 7, 9, 9, 10, 11, 12, 17, 22, 26, 28, 28, 30, 34, 37, 38, 42, 42, 53, 55, 55, 57, 58, 72, 74, 78, 90, 96, 102, 120, 231
