# bypass_misfire: where does the privileged bypass misfire (pbyp, 19 routes x 2 traffic seeds)

Offline analysis of finished logs in `$DATA_DIR/runs/vlm_arb/arms/` (units `v2-pbyp-*` against the paired `drive` runs). No CARLA, no GPU, no
new driving. Definitions (activation, classes, replay) are in [plans/2026-10-02-bypass-offline.md](../plans/2026-10-02-bypass-offline.md);
code: `scripts/bypass_extract.py`, `bypass_common.py`, `bypass_misfire.py`; per-activation rows in `bypass_activations.csv`, the full rule grid
in `bypass_rule_grid.csv`. Every number below is computed from the logs (verified) unless a sentence says inferred.

**Answer in one paragraph.** The guess "it pulls out around red-light queues" is mostly wrong. Of 26 activations on the 15 non-target routes,
22 are triggered by a vehicle that is not on the route ahead of the ego: 8 sit about 10 m *behind* the ego's spawn point and 14 sit *beyond the
last route point*, and the geometry projects both onto the first or last route point, where the blocker test (static 2 s, within -5 to 50 m,
lateral offset below one lane) passes. Only 3 are the head of a red-light queue (fired at the instant the light turns green), 1 is a short stop of
moving traffic; no activation was triggered by a parked or broken-down vehicle. All 26 blockers drive off again within a median of 1.8 s.
The collisions follow: 21 of the 27 pbyp collisions on non-target routes are inside such an episode and absent from the paired drive run.

## 1. Activations by class

Non-target routes (15 routes x 2 traffic seeds, 30 pbyp runs):

| class | activations | routes | t0 <= 7 s | blocker resumes (n) | median resume after t0 [s] | collision within 15 s | red light within 15 s | blocked later in episode |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| behind_route_start | 8 | 4 | 8 | 8/8 | 1.7 | 4 | 1 | 2 |
| beyond_route_end | 14 | 8 | 2 | 14/14 | 4.9 | 10 | 1 | 6 |
| red_light_queue | 3 | 2 | 2 | 3/3 | 0.2 | 1 | 0 | 0 |
| moving_traffic_pause | 1 | 1 | 0 | 1/1 | 0.0 | 1 | 0 | 0 |
| all | 26 | 12 | 12 | 26/26 | 1.8 | 16 | 2 | 8 |

Obstacle routes (4 routes x 2 seeds, 8 pbyp runs):

| class | activations | routes | t0 <= 7 s | blocker resumes (n) | median resume after t0 [s] | collision within 15 s | red light within 15 s | blocked later in episode |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| scenario_obstacle | 8 | 4 | 4 | 0/8 | - | 3 | 0 | 0 |
| behind_route_start | 4 | 2 | 4 | 4/4 | 1.4 | 4 | 0 | 0 |
| beyond_route_end | 3 | 3 | 0 | 2/3 | 0.4 | 2 | 0 | 0 |
| moving_traffic_pause | 1 | 1 | 0 | 1/1 | 0.8 | 0 | 0 | 0 |

How to read: of the 26 activations on non-target routes, 22 fire on a vehicle that is not on the route ahead of the ego at all: 8 on a vehicle behind the ego's spawn point (about 10 m behind, projection clamped to the route start) and 14 on a vehicle beyond the last route point (clamped to the route end). Only 3 are a vehicle waiting at a red light (all three at the instant the light turns green, before the first car moves) and 1 is a short stop of moving traffic. Every one of the 26 blockers moved again (median 1.8 s after the activation); none was parked or broken down (verified from the logs). 20 of the 26 activations pull the path into the oncoming lane (`borrow`).

Every activation on the non-target routes (ego state at the activation, blocker, what followed within 15 s; `ext` is the distance of the blocker beyond the route end (+) or behind the route start (-)):

| route | seed | t0 [s] | class | ego s | ego v | borrowed lane | blocker | ext [m] | static for [s] | resumes after [s] | ego light | ego to junction [m] | outcome within 15 s |
|:--|--:|--:|:--|--:|--:|:--|:--|--:|--:|--:|--:|--:|:--|
| 15483 | 0 | 13.1 | beyond_route_end | 4 | 0.0 | oncoming | chevrolet.impala | 2.0 | 10.2 | 10.0 | G @ -2 m | 5 | none_in_15s |
| 15612 | 0 | 12.7 | beyond_route_end | 10 | 2.6 | oncoming | chevrolet.impala | 3.1 | 9.8 | 3.2 | - | 0 | collision (lincoln.mkz_2017 at 19.4 s) |
| 16390 | 0 | 5.0 | beyond_route_end | 0 | 0.0 | oncoming | chevrolet.impala | 3.0 | 2.2 | 5.6 | G @ 2 m | 9 | collision (ford.mustang at 14.5 s) |
| 16508 | 0 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | ford.mustang | -9.6 | 2.4 | 1.6 | G @ 2 m | 12 | none_in_15s |
| 16529 | 0 | 9.2 | beyond_route_end | 11 | 4.2 | oncoming | chevrolet.impala | 4.0 | 6.4 | 7.8 | - | 6 | collision (dodge.charger_2020 at 20.2 s) |
| 17280 | 0 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | ford.mustang | -9.9 | 2.4 | 1.2 | - | - | collision (traffic.speed_limit.30 at 7.9 s) |
| 22535 | 0 | 31.2 | beyond_route_end | 79 | 5.4 | same dir | ford.mustang | 34.3 | 21.6 | 16.0 | - | - | collision (mini.cooper_s_2021 at 42.9 s) |
| 27297 | 0 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | chevrolet.impala | -9.7 | 2.0 | 1.8 | G @ 19 m | 27 | collision (lincoln.mkz_2017 at 11.5 s) |
| 27297 | 0 | 17.6 | beyond_route_end | 26 | 4.8 | oncoming | lincoln.mkz_2017 | 4.8 | 14.8 | 1.6 | - | 0 | collision (walker.pedestrian.0001 at 20.6 s) |
| 27870 | 0 | 9.2 | beyond_route_end | 9 | 3.8 | oncoming | dodge.charger_2020 | 12.8 | 2.4 | 0.6 | - | 2 | collision (lincoln.mkz_2017 at 16.9 s) |
| 28147 | 0 | 5.2 | red_light_queue | 0 | 0.0 | same dir | audi.tt | - | 3.2 | 0.2 | G @ 16 m | 26 | none_in_15s |
| 37969 | 0 | 5.0 | behind_route_start | 0 | 0.0 | same dir | dodge.charger_2020 | -10.0 | 2.0 | 1.8 | - | 34 | none_in_15s |
| 15483 | 1 | 13.1 | beyond_route_end | 4 | 0.0 | oncoming | chevrolet.impala | 2.0 | 10.2 | 10.2 | G @ -2 m | 5 | none_in_15s |
| 15612 | 1 | 12.8 | beyond_route_end | 10 | 2.7 | oncoming | chevrolet.impala | 3.1 | 10.0 | 3.4 | - | 0 | collision (lincoln.mkz_2020 at 23.3 s) |
| 16390 | 1 | 5.0 | beyond_route_end | 0 | 0.0 | oncoming | chevrolet.impala | 3.0 | 2.2 | 5.6 | G @ 2 m | 9 | collision (ford.mustang at 14.5 s) |
| 16508 | 1 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | ford.mustang | -9.6 | 2.4 | 1.6 | G @ 2 m | 12 | red_light |
| 16508 | 1 | 25.9 | beyond_route_end | 39 | 0.1 | oncoming | chevrolet.impala | 35.3 | 2.0 | 0.8 | - | - | none_in_15s |
| 16529 | 1 | 9.2 | beyond_route_end | 11 | 4.4 | oncoming | chevrolet.impala | 4.0 | 6.4 | 4.6 | - | 6 | red_light |
| 17280 | 1 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | ford.mustang | -9.9 | 2.4 | 1.2 | - | - | collision (traffic.speed_limit.30 at 7.9 s) |
| 22535 | 1 | 28.4 | moving_traffic_pause | 63 | 5.0 | same dir | dodge.charger_2020 | - | 0.0 | 0.0 | - | - | collision (nissan.patrol_2021 at 38.2 s) |
| 24944 | 1 | 89.8 | red_light_queue | 116 | 0.0 | oncoming | mercedes.coupe_2020 | - | 28.8 | 0.4 | G @ 7 m | - | collision (audi.tt at 99.7 s) |
| 27297 | 1 | 5.0 | behind_route_start | 0 | 0.0 | oncoming | chevrolet.impala | -9.7 | 2.0 | 1.8 | G @ 19 m | 27 | none_in_15s |
| 27297 | 1 | 18.1 | beyond_route_end | 28 | 4.2 | oncoming | lincoln.mkz_2020 | 21.1 | 2.0 | 2.4 | - | 9 | collision (lincoln.mkz_2020 at 22.3 s) |
| 27870 | 1 | 14.4 | beyond_route_end | 8 | 0.0 | oncoming | dodge.charger_2020 | 12.8 | 7.6 | 5.2 | G @ -7 m | 3 | collision (lincoln.mkz_2017 at 23.1 s) |
| 28147 | 1 | 5.2 | red_light_queue | 0 | 0.0 | same dir | audi.tt | - | 3.2 | 0.2 | G @ 16 m | 26 | none_in_15s |
| 37969 | 1 | 5.0 | behind_route_start | 0 | 0.0 | same dir | dodge.charger_2020 | -10.0 | 2.0 | 1.8 | - | 34 | collision (dodge.charger_2020 at 10.5 s) |

## 2. Per route: DS against the paired drive run

| route | scenario | DS drive s0 / s1 | DS pbyp s0 / s1 | mean DS diff | activations by class (both seeds) | collisions drive / pbyp | blocked drive / pbyp |  |
|:--|:--|--:|--:|--:|:--|--:|--:|:--|
| 15102 | VanillaSignalizedTurnEncounterGree | 100 / 100 | 100 / 100 | +0.0 | none | 0 / 0 | 0 / 0 |  |
| 15483 | VanillaSignalizedTurnEncounterRedL | 100 / 100 | 71 / 71 | -28.7 | beyond route end 2 | 0 / 0 | 0 / 0 |  |
| 15612 | VanillaSignalizedTurnEncounterRedL | 70 / 25 | 29 / 30 | -18.0 | beyond route end 2 | 0 / 2 | 1 / 1 |  |
| 16390 | VanillaSignalizedTurnEncounterRedL | 70 / 70 | 30 / 30 | -40.4 | beyond route end 2 | 0 / 2 | 0 / 0 |  |
| 16508 | VanillaNonSignalizedTurn | 100 / 100 | 86 / 48 | -33.0 | behind route start 2, beyond route end 1 | 0 / 0 | 0 / 0 |  |
| 16529 | VanillaNonSignalizedTurn | 70 / 70 | 30 / 49 | -30.6 | beyond route end 2 | 0 / 1 | 0 / 1 |  |
| 17280 | VanillaNonSignalizedTurnEncounterS | 80 / 80 | 1 / 1 | -78.7 | behind route start 2 | 0 / 2 | 0 / 2 |  |
| 22535 | StaticCutIn | 100 / 100 | 60 / 60 | -40.0 | beyond route end 1, moving traffic pause 1 | 0 / 2 | 0 / 0 |  |
| 24944 | T_Junction | 70 / 70 | 70 / 39 | -15.6 | red light queue 1 | 0 / 1 | 0 / 0 |  |
| 27043 | SignalizedJunctionRightTurn | 60 / 60 | 60 / 60 | +0.0 | none | 2 / 2 | 0 / 0 |  |
| 27297 | VehicleTurningRoutePedestrian | 70 / 70 | 1 / 13 | -62.9 | behind route start 2, beyond route end 2 | 0 / 8 | 0 / 2 |  |
| 27870 | VanillaNonSignalizedTurn | 70 / 100 | 32 / 45 | -46.6 | beyond route end 2 | 0 / 2 | 0 / 2 |  |
| 28147 | SignalizedJunctionLeftTurnEnterFlo | 100 / 100 | 100 / 100 | +0.0 | red light queue 2 | 0 / 0 | 0 / 0 |  |
| 37969 | MergerIntoSlowTrafficV2 | 60 / 60 | 11 / 7 | -51.3 | behind route start 2 | 2 / 3 | 0 / 0 |  |
| 9196 | OppositeVehicleTakingPriority | 42 / 15 | 42 / 42 | +13.6 | none | 3 / 2 | 1 / 0 |  |
| 19324 | Accident | 33 / 33 | 60 / 100 | +46.6 | beyond route end 1, moving traffic pause 1, scenario obstacle 2 | 0 / 1 | 0 / 0 | obstacle route |
| 19832 | ParkedObstacle | 33 / 33 | 60 / 22 | +7.7 | beyond route end 1, scenario obstacle 2 | 0 / 4 | 0 / 0 | obstacle route |
| 24497 | ConstructionObstacle | 21 / 22 | 36 / 36 | +14.8 | behind route start 2, scenario obstacle 2 | 2 / 4 | 0 / 0 | obstacle route |
| 2520 | ConstructionObstacle | 23 / 23 | 35 / 59 | +23.4 | behind route start 2, beyond route end 1, scenario obstacle 2 | 2 / 3 | 0 / 0 | obstacle route |

Mean over routes: non-target -28.8 DS (15 routes), obstacle routes +23.1 DS (4 routes); reproduces report.md (-28.8 / +23.1).

How to read: the four obstacle routes gain DS because the real obstacle is detected, but each of them also has at least one misfire activation (8 of the 16 activations on these four routes). On non-target routes the largest losses (17280 -79, 27297 -63, 37969 -51, 27870 -47, 16390 -40, 22535 -40) are runs where the activation is followed by a collision and an ego that stays in the shifted lane; three routes have no activation and no change (15102, 27043, 9196), and 24944 has one in seed 1 only.

## 3. Which pbyp events follow a misfire (30 non-target runs)

| event | drive (official) | pbyp (official) | in misfire episode, not in drive | in misfire episode, also in drive (same place) | outside episodes, also in drive | outside episodes, not in drive | no time or place in record |
|:--|--:|--:|--:|--:|--:|--:|--:|
| collision | 7 | 27 | 21 | 0 | 4 | 2 | 0 |
| vehicle_blocked | 2 | 8 | 8 | 0 | 0 | 0 | 0 |
| red_light | 13 | 14 | 1 | 9 | 4 | 0 | 0 |
| outside_route_lanes | 1 | 16 | 0 | 0 | 0 | 0 | 16 |
| stop_infraction | 2 | 0 | 0 | 0 | 0 | 0 | 0 |

Events inside the episode of each class (an episode runs from the activation to the end of its bypass state, or to the next activation):

| class | activations | collisions | blocked | red light |
|:--|--:|--:|--:|--:|
| behind_route_start | 8 | 5 | 2 | 1 |
| beyond_route_end | 14 | 14 | 6 | 9 |
| red_light_queue | 3 | 1 | 0 | 0 |
| moving_traffic_pause | 1 | 1 | 0 | 0 |

How to read: of the 27 pbyp collisions on non-target routes, 21 fall in a misfire episode and are not reproduced by the paired drive run; 4 are the same collisions as drive (27043 and 9196 have no activation); the red-light infractions are nearly all the same as drive. All 8 blocked events are in a misfire episode (the ego ends in a lane it cannot leave). 16 of the 30 pbyp runs also log `outside_route_lanes`, which has no position or time in the official record.

## 4. Log replay of suppression rules

Replay on the logs only: an activation counts as prevented when, from its logged time on, the rule never lets it fire before its blocker moves off again or the episode ends; otherwise it fires late by the stated delay. Ego-side signals after t0 come from the paired drive run (what the ego sees if it does not bypass). The closed loop is not replayed, so what the car would have done afterwards is unknown.

| rule | non-target misfires prevented | collisions + blocked in those episodes | median delay of the rest [s] | obstacle-route misfires prevented | correct activations blocked | correct: median delay [s] | correct: max delay [s] |
|:--|--:|--:|--:|--:|--:|--:|--:|
| baseline | 0 / 26 | 0 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| X15 | 0 / 26 | 0 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| X30 | 0 / 26 | 0 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| X50 | 0 / 26 | 0 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| N10 | 8 / 26 | 14 | 6.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| N25 | 10 / 26 | 14 | 6.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| N40 | 16 / 26 | 17 | 6.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| Nb25 | 4 / 26 | 0 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| Nb40 | 8 / 26 | 3 | - | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| T5 | 14 / 26 | 13 | 2.8 | 8 / 8 | 0 / 8 | 0.0 | 0.0 |
| T10 | 16 / 26 | 15 | 3.0 | 8 / 8 | 0 / 8 | 5.0 | 5.0 |
| T20 | 22 / 26 | 27 | 9.8 | 8 / 8 | 0 / 8 | 15.0 | 15.0 |
| tmin7 | 10 / 26 | 7 | 2.0 | 4 / 8 | 0 / 8 | 1.0 | 2.0 |
| valid | 22 / 26 | 27 | - | 7 / 8 | 0 / 8 | 0.0 | 0.0 |
| X30+N25 | 10 / 26 | 14 | 6.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| N25+T5 | 21 / 26 | 25 | 7.0 | 8 / 8 | 0 / 8 | 0.0 | 0.0 |
| N25+T10 | 21 / 26 | 25 | 7.0 | 8 / 8 | 0 / 8 | 5.0 | 5.0 |
| N40+T10 | 21 / 26 | 25 | 7.0 | 8 / 8 | 0 / 8 | 5.0 | 5.0 |
| X30+N25+T10 | 21 / 26 | 25 | 7.0 | 8 / 8 | 0 / 8 | 5.0 | 5.0 |
| G5 | 8 / 26 | 9 | 4.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| G10 | 11 / 26 | 18 | 9.8 | 0 / 8 | 0 / 8 | 0.0 | 0.0 |
| valid+T5 | 25 / 26 | 28 | - | 8 / 8 | 0 / 8 | 0.0 | 0.0 |
| valid+T10 | 25 / 26 | 28 | - | 8 / 8 | 0 / 8 | 5.0 | 5.0 |
| valid+G5 | 25 / 26 | 28 | - | 7 / 8 | 0 / 8 | 0.0 | 0.0 |
| valid+T5+G5 | 26 / 26 | 29 | - | 8 / 8 | 0 / 8 | 0.0 | 0.0 |
| valid+N25+T5+G5 | 26 / 26 | 29 | - | 8 / 8 | 0 / 8 | 0.0 | 0.0 |
| valid+X30+N25+T10 | 25 / 26 | 28 | - | 8 / 8 | 0 / 8 | 5.0 | 5.0 |

X: no activation while the ego light is red or yellow and closer than X m. N: none while the ego is within N m before a junction entrance or inside one. Nb: same test on the blocker. T: blocker static for at least T s (the original rule is 2 s). valid: blocker projects strictly inside the route polyline. tmin: no activation before tmin s. G: none while the ego light was red or yellow within 50 m at any time in the last G s (the queue's first car has not moved yet when the light turns green).

N x T grid, no X, no valid (cells: non-target prevented of 26 / correct activations delayed, median s; 8 correct, none blocked):

|  | T=2 (original) | T=5 | T=10 | T=20 |
|:--|--:|--:|--:|--:|
| N=off | 0 / 0.0 | 14 / 0.0 | 16 / 5.0 | 22 / 15.0 |
| N=10 | 8 / 0.0 | 21 / 0.0 | 21 / 5.0 | 22 / 15.0 |
| N=25 | 10 / 0.0 | 21 / 0.0 | 21 / 5.0 | 22 / 15.0 |
| N=40 | 16 / 0.0 | 21 / 0.0 | 21 / 5.0 | 22 / 15.0 |

How to read: a red-light test on the ego light (X) prevents nothing, because no misfire happens while the ego's light is red (the privileged rule already has that test at 50 m); the three red-queue activations happen when it turns green (G). A junction test on the ego (N) removes the activations at the spawn point and some beyond-end ones, because the ego stands in or next to a junction there, and delays the rest by about 7 s. A longer static time T removes most misfires because the blockers move again within seconds, and costs the correct activations T minus their current static time (0 s at T=5, 5 s at T=10, 15 s at T=20). The projection test (valid) removes 22 of 26 non-target and 7 of 8 obstacle-route misfires at no delay. These are fits on the same 38 runs, so the combinations are a ranking, not a confirmation.

## 5. Collisions on the obstacle routes

| route | seed | t [s] | actor | episode class at t | ego s | phase vs obstacle | bypass shifted | ego lat [m] | actor lat [m] | actor direction | ego v | actor v | actor ahead of ego [m] | closing speed (+ = ego approaching) | rel speed | state offset [m] |
|:--|--:|--:|:--|:--|--:|:--|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|
| 19324 | 0 | 53.3 | nissan.patrol_2021 11394 | beyond_route_end | 113 | entering (ramp) | yes | -0.8 | -3.2 | same direction | 3.8 | 7.8 | 2.0 | -1.7 | 4.2 | -3.2 |
| 19832 | 0 | 21.5 | chevrolet.impala 7421 | scenario_obstacle | 33 | entering (ramp) | yes | -0.9 | -3.1 | same direction | 0.6 | 7.1 | 1.4 | -2.2 | 6.5 | -3.2 |
| 24497 | 0 | 7.1 | ford.mustang 297 | behind_route_start | 1 | abeam obstacle | yes | -1.3 | -3.7 | same direction | 2.0 | 6.1 | 2.9 | -2.5 | 4.3 | -3.5 |
| 24497 | 0 | 20.5 | ford.mustang 313 | scenario_obstacle | 31 | entering (ramp) | yes | -1.3 | -3.5 | same direction | 1.6 | 8.9 | 2.5 | -4.8 | 7.3 | -3.5 |
| 2520 | 0 | 7.1 | nissan.patrol_2021 11240 | behind_route_start | 1 | abeam obstacle | yes | -1.0 | -3.5 | same direction | 2.3 | 6.8 | 3.8 | -3.3 | 4.7 | -3.2 |
| 2520 | 0 | 16.0 | nissan.patrol_2021 11243 | behind_route_start | 20 | returning | yes | -0.7 | -3.5 | same direction | 1.7 | 0.5 | -4.4 | -1.1 | 1.1 | -3.2 |
| 19832 | 1 | 19.0 | lincoln.mkz_2017 11230 | scenario_obstacle | 34 | entering (ramp) | yes | -0.9 | -3.2 | same direction | 0.0 | 8.0 | 0.9 | -1.4 | 8.0 | -3.2 |
| 19832 | 1 | 21.4 | chevrolet.impala 11234 | scenario_obstacle | 34 | entering (ramp) | yes | -1.0 | -3.2 | same direction | 1.8 | 6.5 | -0.4 | 1.8 | 4.8 | -3.2 |
| 19832 | 1 | 65.6 | dodge.charger_2020 11258 | beyond_route_end | 115 | entering (ramp) | yes | -1.0 | -3.2 | same direction | 5.8 | 7.3 | 0.2 | 1.0 | 1.8 | -3.2 |
| 24497 | 1 | 7.1 | ford.mustang 291 | behind_route_start | 0 | abeam obstacle | yes | -1.2 | -3.7 | same direction | 1.6 | 6.5 | 3.0 | -3.2 | 5.0 | -3.5 |
| 24497 | 1 | 28.0 | chevrolet.impala 315 | scenario_obstacle | 32 | entering (ramp) | yes | -1.2 | -3.4 | same direction | 0.7 | 12.7 | 6.9 | -11.3 | 12.0 | -3.5 |
| 2520 | 1 | 7.1 | nissan.patrol_2021 11234 | behind_route_start | 1 | abeam obstacle | yes | -1.0 | -3.4 | same direction | 2.1 | 6.9 | 3.7 | -3.6 | 5.0 | -3.2 |

How to read: of the 12 collisions on the obstacle routes, 5 belong to the artefact at the spawn point (at 7.1, 7.1, 7.1, 7.1, 16.0 s, before the real obstacle is within 50 m; one of them, 2520 seed 0 at 16.0 s, is with a nearly stationary vehicle (0.5 m/s) in the borrowed lane while the ego 'returns' from the spurious shift), 2 to the artefact near the route end, and 5 to the real bypass. 12 of 12 are with a vehicle of the ego's own driving direction, 12 of 12 with an actor within 0.5 m of the lateral offset the path borrows, 7 of 12 while the ego is in the entry ramp 12 to 20 m before the obstacle, 7 of 12 with the other vehicle alongside the ego (centre offset along the ego heading within 3 m: a faster vehicle in the adjacent lane meets the ego while it moves over). Ego speed at the collision is 0.0 to 5.8 m/s, the other vehicle's 0.5 to 12.7 m/s (the 0.2 s snapshot nearest to the collision; collision times are the first contact minus a 1.05 s clock offset).

## 6. Verified and inferred

Verified from the logs: activation times, blocker identity, speed history, projection onto the route (clamped to the route ends: 8 + 14 cases,
`ext` column), resume times, official events and their times (collisions from the contact sensor, red-light and blocked events located on the ego
trajectory), the paired DS differences (they reproduce report.md: -28.8 and +23.1), the replay counts under the stated rules.

Inferred: why the 14 vehicles beyond the route end were stopped (a queue or held traffic beyond the final junction; the light state there is not
logged); that the 3 red-queue activations are a red-light queue (the light state is reconstructed from the ego's own light at the same stop line,
the transition times agree across runs to 0.1 s); that the collisions would not have happened without the activation (the paired drive run has
none at the same place, but the ego trajectories differ); the 1.05 s offset between the contact sensor's clock and the plan clock (measured on the 38 collisions
from the ego's closest approach to the recorded location, attributed to 20 dropped ticks); every statement about what a suppression rule would have done after the replayed time.

Data limits: the 13 seed-0 `drive` runs in `eval-drive-s0` have no `privileged.jsonl`, so only their ego-side signals (plans, junction distance
from the 's' lines) were used; junction entrances come from the `drive` runs' own junction distance and are exact, exits are sampled at the
ego's step. The traffic frozen until about 6 s (all vehicles stand from about 2.8 s, the ego itself is held until 6 s) is what makes standing
vehicles pass the 2 s static test at the first allowed moment, 5.0 s (inferred from the speed histories, the harness code was not read).
