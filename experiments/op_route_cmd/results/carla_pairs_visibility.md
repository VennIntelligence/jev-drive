# Exit visibility from the open-loop-rig camera (geometry only)

Source: 21148 of 21151 (pose, exit) rows (the rest have a polyline shorter than the window). Camera level at x 1.59 m, y 0.0, z 1.86 m
(`interface.B2D_SPEC_MOUNT`). Each cell: **mean fraction of the exit's first 20 m after the junction mouth that lies inside the FOV** (share of rows whose
whole 20 m window is inside). Window = polyline arc length d ... d + 20 m seen from the pose at distance d before the junction (script docstring has the rest).
FOV half-angles: road 15.7 deg, wide 29.4 deg (the model frames), hypothetical 120 / 180 deg.

| exit turn angle (abs, connector heading change) | rows | road 31 deg | wide 59 deg | 120 deg | 180 deg |
|---|--:|--:|--:|--:|--:|
| straight (< 25 deg) | 7220 | 100% (99%) | 100% (100%) | 100% (100%) | 100% (100%) |
| 25 - 60 deg | 207 | 81% (67%) | 93% (85%) | 98% (98%) | 100% (100%) |
| 60 - 120 deg (right-angle) | 13612 | 60% (4%) | 89% (54%) | 100% (100%) | 100% (100%) |
| >= 120 deg (u-turn) | 109 | 52% (5%) | 82% (36%) | 98% (91%) | 100% (100%) |

| min turn radius R_min (turning rows) | rows | road 31 deg | wide 59 deg | 120 deg | 180 deg |
|---|--:|--:|--:|--:|--:|
| tight: R_min < 7 m | 5137 | 52% (2%) | 84% (40%) | 100% (99%) | 100% (100%) |
| mid: 7 - 10 m | 4959 | 62% (4%) | 91% (55%) | 100% (100%) | 100% (100%) |
| wide: R_min >= 10 m | 3593 | 73% (9%) | 96% (76%) | 100% (100%) | 100% (100%) |

| pose distance d (rows with turn angle >= 60 deg) | rows | road 31 deg | wide 59 deg | 120 deg | 180 deg |
|---|--:|--:|--:|--:|--:|
| d = 10 m | 4225 | 43% (1%) | 74% (11%) | 100% (99%) | 100% (100%) |
| d = 20 m | 5447 | 62% (3%) | 94% (57%) | 100% (100%) | 100% (100%) |
| d = 30 m | 4049 | 76% (8%) | 99% (95%) | 100% (100%) | 100% (100%) |

| d x R_min (turn angle >= 60 deg) | rows | road 31 deg | wide 59 deg | 120 deg | 180 deg |
|---|--:|--:|--:|--:|--:|
| d = 10 m, tight (< 7 m) | 1486 | 35% (0%) | 64% (3%) | 100% (98%) | 100% (100%) |
| d = 10 m, wide (>= 10 m) | 1091 | 56% (2%) | 88% (24%) | 100% (100%) | 100% (100%) |
| d = 20 m, tight (< 7 m) | 2105 | 53% (1%) | 88% (31%) | 100% (100%) | 100% (100%) |
| d = 20 m, wide (>= 10 m) | 1378 | 73% (5%) | 100% (99%) | 100% (100%) | 100% (100%) |
| d = 30 m, tight (< 7 m) | 1534 | 67% (3%) | 99% (89%) | 100% (100%) | 100% (100%) |
| d = 30 m, wide (>= 10 m) | 1018 | 87% (15%) | 100% (99%) | 100% (100%) | 100% (100%) |
