# How far is the official route's command change from the true junction entrance?

Question: the agent finds the junction entrance and the stop line through the CARLA map API (not an allowed input). The official input is the route given by the leaderboard. How good is it as a replacement? Setting: the 19 evaluation routes, run logs only, no CARLA started. Definitions: [../plans/2026-10-02-offline-analyses-definitions.md](../plans/2026-10-02-offline-analyses-definitions.md) section B. Script: [route_junction_error.py](../scripts/route_junction_error.py) (with [route_common.py](../scripts/route_common.py)); per-junction CSV `route_junction_error.csv`.

**What was used.** The dense plan is the list the agent logged as `route.json` in every attempt (`xy` and `cmd` of `global_plan_world_coord`, identical in all attempts of a route; points 1 to 2 m apart: 1.3 m median and at most 1.9 m next to the command changes). It is what `AutonomousAgent.set_global_plan` receives; the leaderboard keeps it in `_plan_gps_HACK` and the agent's own copy in `_dense_plan`. The downsampled plan is `route_manipulation.downsample_route(plan, 50)` re-implemented branch by branch (the code was read on the box in `Bench2Drive/leaderboard/leaderboard/utils/route_manipulation.py`); the 2-D distances differ from the original 3-D ones by the elevation only (no z in the logs). The true junction entrance is the first map-flagged route point, reconstructed exactly from the logged `ego_s + junc_dist` of the arbitration steps (1 m route resolution); the stop line is `ego_s + REAR_TO_BUMPER + tl_dist` of the plan-step context, median per light. All positions are arc length in metres along the dense route. Error = position of the command change minus the truth (negative: the command starts before the truth). Nothing needed the CARLA map on the box.

## 1. Error distribution

Signed error of the first point of the turn / straight command run against the true junction entrance and against the light's stop line, for the dense plan and the downsampled plan (arc length along the downsampled polyline, i.e. what an agent counting distance on that plan would get). The downsample rule always keeps the first point of a new command (except right after a lane change), so the two plans place the command change at the same route point; the difference is the chord shortcut between kept points.

| quantity (signed, command position minus truth) | n | mean | median | min | max | mean |e| | p90 |e| | n within 1 m | n within 3 m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dense plan vs entrance | 13 | -0.53 | -0.83 | -2.45 | +3.05 | 1.00 | 2.40 | 9 / 13 | 12 / 13 |
| downsampled plan vs entrance | 13 | -1.41 | -0.83 | -11.33 | +3.05 | 1.88 | 2.93 | 8 / 13 | 11 / 13 |
| dense plan vs stop line | 12 | +4.05 | +3.26 | +2.52 | +10.23 | 4.05 | 5.50 | 0 / 12 | 3 / 12 |
| downsampled plan vs stop line | 12 | +3.10 | +3.18 | -1.10 | +5.51 | 3.28 | 5.20 | 0 / 12 | 4 / 12 |
| stop line vs entrance (S - E) | 12 | -4.62 | -4.18 | -10.23 | -0.00 | 4.62 | 5.74 | 1 / 12 | 1 / 12 |

Reading it: against the junction entrance the dense plan is within 1 m on 9 of 13 junctions and within 3 m on 12 (median -0.8 m, worst -2.5 and +3.0 m); the command starts, if anything, slightly before the entrance. Against the stop line it is 2.5 to 10 m too late (median +3.3 m): the route carries no information about the stop line, which lies a median 4.2 m (up to 10.2 m) before the entrance. The downsampled plan has one outlier (route 24944, -11.3 m) from the chord of a curved 80 m approach; read as remaining distance from the car's projection the chord error vanishes close to the junction:

| car before the entrance | n | mean | median | min | max | mean |e| | p90 |e| | n within 1 m | n within 3 m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 m | 13 | -0.53 | -0.83 | -2.45 | +3.05 | 1.00 | 2.40 | 9 / 13 | 12 / 13 |
| 10 m | 9 | -0.80 | -0.82 | -2.45 | +0.01 | 0.80 | 2.25 | 7 / 9 | 9 / 9 |
| 20 m | 6 | -0.65 | -0.41 | -2.20 | +0.01 | 0.65 | 1.54 | 5 / 6 | 6 / 6 |
| 30 m | 3 | -0.44 | -0.52 | -0.82 | +0.01 | 0.45 | 0.76 | 3 / 3 | 3 / 3 |
| 50 m | 1 | -5.30 | -5.30 | -5.30 | -5.30 | 5.30 | 5.30 | 0 / 1 | 0 / 1 |

## 2. Per junction

| route | junction id | true entrance s m | stop line s m | command | dense: command start s m | dense error vs entrance m | dense error vs stop line m | downsampled: arc of first new-command sample m | downsampled error vs entrance m | gap to previous sample m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 9196 | 1190 | 27.4 | 22.0 | LEFT | 25.2 | -2.2 | +3.2 | 25.2 | -2.2 | 25.2 |
| 15102 | 2458 | 6.0 | 6.0 | LEFT | 9.0 | +3.0 | +3.1 | 9.0 | +3.0 | 9.0 |
| 15483 | 12988 | 9.7 | 6.0 | RIGHT | 8.9 | -0.8 | +2.9 | 8.9 | -0.8 | 8.9 |
| 15612 | 4837 | 10.1 | 6.0 | RIGHT | 9.3 | -0.8 | +3.3 | 9.3 | -0.8 | 9.3 |
| 16390 | 11047 | 9.5 | 6.0 | RIGHT | 8.6 | -0.8 | +2.6 | 8.6 | -0.8 | 8.6 |
| 16508 | 6090 | 11.8 | 6.0 | RIGHT | 9.3 | -2.5 | +3.3 | 9.3 | -2.5 | 9.3 |
| 16529 | 9170 | 9.7 | 6.0 | LEFT | 8.5 | -1.2 | +2.5 | 8.5 | -1.2 | 8.5 |
| 24944 | 20 | 79.9 | 69.7 | LEFT | 79.9 | -0.0 | +10.2 | 68.6 | -11.3 | 28.2 |
| 27043 | 433 | 30.9 | 27.0 | RIGHT | 30.1 | -0.8 | +3.1 | 30.1 | -0.8 | 30.1 |
| 27297 | 4101 | 26.7 | 22.4 | RIGHT | 25.8 | -0.8 | +3.4 | 25.7 | -0.9 | 25.8 |
| 27870 | 621 | 10.4 | 5.0 | RIGHT | 10.4 | +0.0 | +5.4 | 10.4 | +0.0 | 10.4 |
| 28147 | 189 | 25.7 | 20.2 | LEFT | 25.7 | -0.0 | +5.5 | 25.7 | -0.0 | 25.7 |
| 37969 | 8775 | 34.1 | - | STRAIGHT | 34.1 | +0.0 | - | 34.1 | +0.0 | 1.4 |

Reading it: columns 7 and 8 are the dense-plan errors that matter (command start against entrance and against stop line); column 10 is the same for the downsampled plan. On 15102 the command starts 3.0 m after the entrance, on 16508 and 9196 2.5 and 2.2 m before it; elsewhere within 1.2 m. Route 37969 has a STRAIGHT command from 34.1 m to 149.7 m (the run does not end at the junction exit), and its junction is unsignalised; the lane-change run before it (24.1 to 32.7 m) is not a junction command.

## 3. Routes, commands and junctions without a match

| route | plan length m | dense pts | downsampled pts | junctions (map flag) | turn / straight runs (cmd, start..end m) | lane-change runs | turn runs with no flagged junction |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 2520 | 133 | 102 | 4 | 0 | - | - | - |
| 9196 | 74 | 74 | 4 | 1 | LEFT 25.2..41.8 | - | - |
| 15102 | 63 | 64 | 4 | 1 | LEFT 9.0..31.4 | - | - |
| 15483 | 52 | 46 | 4 | 1 | RIGHT 8.9..19.2 | - | - |
| 15612 | 52 | 56 | 4 | 1 | RIGHT 9.3..19.6 | - | - |
| 16390 | 51 | 52 | 4 | 1 | RIGHT 8.6..18.9 | - | - |
| 16508 | 49 | 40 | 4 | 1 | RIGHT 9.3..19.6 | - | - |
| 16529 | 54 | 38 | 4 | 1 | LEFT 8.5..22.9 | - | - |
| 17280 | 58 | 58 | 4 | 0 | LEFT 9.4..27.7 | - | LEFT 9.4..27.7 |
| 19324 | 132 | 67 | 4 | 0 | - | - | - |
| 19832 | 133 | 101 | 4 | 0 | - | - | - |
| 22535 | 133 | 134 | 4 | 0 | - | - | - |
| 24497 | 132 | 67 | 4 | 0 | - | - | - |
| 24944 | 133 | 124 | 5 | 1 | LEFT 79.9..98.6 | - | - |
| 27043 | 74 | 79 | 4 | 1 | RIGHT 30.1..41.9 | - | - |
| 27297 | 67 | 71 | 4 | 1 | RIGHT 25.8..36.6 | - | - |
| 27870 | 61 | 61 | 4 | 1 | RIGHT 10.4..25.9 | - | - |
| 28147 | 96 | 80 | 4 | 1 | LEFT 25.7..63.4 | - | - |
| 37969 | 203 | 193 | 10 | 1 | STRAIGHT 34.1..149.7 | CHANGELANELEFT 24.1..32.7 | - |

Reading it: the 19 plans are short (49 to 203 m); the downsampled plan has 4 points on 17 routes (start, command start, command end, end), 5 on 24944 and 10 on 37969, so it gives the command change and the turn extent but almost no geometry. 6 routes have no map-flagged junction: five straight routes (2520, 19324, 19832, 22535, 24497) and 17280, which has a turn (see below).

- **Junctions without a command change: 0 of 13.** Every map-flagged junction on the 19 routes has a LEFT / RIGHT / STRAIGHT run starting within 3.1 m of its entrance. The 13 junctions are one per route on 13 routes; 12 have a traffic light, 37969 has none. (Verified for the whole plan length on every route: the logged cars saw the end of every plan within their 100 m look-ahead.)
- **Turn commands without a flagged junction: 1.** Route 17280 has a LEFT run from 9.4 m to 27.7 m, but the map-based junction test never fires on it (the arbitration steps log `junc_dist` 999 and `jid` -1 for the whole route; scenario: stop-sign turn). The map-based entrance therefore does not exist there, while the route command does.
- A second light on 24944 (stop line at 126.6 m, 6 m before the plan end) has no junction entrance in the plan; its junction lies beyond the end.

## Is it good enough

- **Opening a junction window (R1 slow-down, 25 m look-ahead, stop at the entrance):** yes. The command start is within 3.1 m of the true entrance on all 13 junctions (p90 of |error| 2.4 m), never later than +3.0 m, and it exists for every junction on these routes, including the one the map flag misses (17280). A rule that opens its window at the command change loses about 1 m on average against the map-based entrance; a 25 m window absorbs that. The scorer's line is the junction entrance (decision 87), so placing a stop at the command start is within 1 m of that line on 9 of 13 junctions.
- **Placing a stop at the light's stop line:** no, not from the route alone. The stop line lies 0 to 10.2 m (median 4.2 m) before the entrance and the route does not contain it; command start minus stop line is +2.5 to +10.2 m. Using the command start as target puts the car 2.5 to 10 m past the line, the `vred` behaviour. A fixed offset (command start - 3.3 m) would be within 1 m of the stop line on 9 of 12 lights, within 2.3 m on 11, and off by 6.9 m on one (24944). Whether a junction has a light at all is not in the route either (12 of 13 have one here).
- **Verified from logs:** all numbers above for these 19 routes and 13 junctions (one junction per route, so n is small). **Inferred:** that the official plan equals the logged plan (the agent logs its own copy of the leaderboard's dense list; `set_global_plan` was read in the source but not run); the downsampled plan from the re-implemented rule.
