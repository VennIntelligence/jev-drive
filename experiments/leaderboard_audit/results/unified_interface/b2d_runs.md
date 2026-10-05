# Unified interface on B2D: camera height (A) and the nored resume (B)

Lane `uni-b2d`, 1 card (card 2 held a foreign CARLA server, so the lease took card 1 only), repo abab703, shipped Cinque, one run per cell.
Lane file [`experiments/leaderboard_audit/scripts/unified_b2d_lane.py`](https://github.com/VennIntelligence/jev-drive/blob/abab70314a9915f21135840f9f8cdd31792a12aa/experiments/leaderboard_audit/scripts/unified_b2d_lane.py). Units: `$DATA_DIR/runs/unified/b2d/arms/<arm>-s<seed>`
(`s122dbg-s2` smoke, `legacy / s143 / s122 / s143nz / s122nz -s2`, `drivetimer-s0 / -s1`). All 7 + 1 units finished all routes
(`routes_never_finished` empty in every summary.json, 0 restarts, 0 failed jobs; 38 + 1 route-runs, ~1 h wall).
Readout: `drive_row` (op_arb_report) + the junction turn geometry of `junction_cl_report.py` (branch = car came within 5 m of the dense exit point);
helper script was run from a scratch copy, nothing committed. Arms: seed 2, 6 val routes (28180, 24944, 27297, 9196, 6999, 34183), 7 turns in total.

## Smoke (s122, route 28180)
Finished, DS 100 / RC 100, 1050 ticks, no agent crash. interface.json: preset spec, rig.height_m 1.22, lateral.exec op-path, light.source none.
Lane probabilities 0.5-0.97 on moving frames, speed up to 6.1 m/s, mean 2.3 (it stops at lights and restarts). 1.22 m at x 3.8 m works.

## A. Camera height

Per route DS / RC (status: C = Completed, B = blocked, D = deviated, T = TickRuntime failure). Zones on = dense-route zones steer in junctions, zones off = action head steers everywhere.

| route | legacy | s143 | s122 | s143nz | s122nz |
|---|---|---|---|---|---|
| 28180 | 70 / 100 | 70 / 100 | 100 / 100 | 50.2 / 78.6 D | 20.3 / 77.0 D |
| 24944 | 70 / 100 | 42 / 100 | 100 / 100 | 16.9 / 19.6 B | 92.0 / 100 |
| 27297 | 70 / 100 | 70 / 100 | 70 / 100 | 36.4 / 52.0 D | 29.6 / 100 |
| 9196 | 13.7 / 58.0 B | 40.5 / 100 | 15.5 / 60.8 B | 22.2 / 52.9 B | 14.9 / 67.5 B |
| 6999 | 21.7 / 45.3 T | 48.0 / 100 | 60.0 / 100 | 24.2 / 50.3 T | 38.1 / 100 |
| 34183 | 100 / 100 | 60 / 100 | 60 / 100 | 32.4 / 55.2 T | 25.4 / 100 |
| **mean** | **57.6 / 83.9** | **55.1 / 100** | **67.6 / 93.5** | **30.4 / 51.4** | **36.7 / 90.8** |

Infractions summed over the 6 routes (collisions = veh + ped + layout; "completed" = status Completed):

| arm | completed | collisions | red light | off-lane | route dev | blocked | TickRuntime | turns taken (of 7) | xt median m | xt p95 mean m | mean v m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 4 | 3 | 4 | 1 | 0 | 1 | 1 | 5 | 0.07 | 0.69 | 1.87 |
| s143 | 6 | 4 | 4 | 1 | 0 | 0 | 0 | 7 | 0.08 | 0.62 | 2.46 |
| s122 | 5 | 4 | 2 | 1 | 0 | 1 | 0 | 6 | 0.13 | 1.18 | 2.24 |
| s143nz | 0 | 3 | 3 | 3 | 2 | 2 | 2 | 0 | 0.35 | 14.0 | 0.76 |
| s122nz | 4 | 10 | 3 | 4 | 1 | 1 | 0 | 5 | 0.77 | 9.0 | 1.78 |

Turn-correct per turn (took the dense exit): s143 7/7; s122 6/7 (misses 9196, which goes wrong in both s122 variants and in legacy / s143nz / s122nz);
legacy 5/7 (misses 6999 and 9196); s143nz 0/7 (every turn "lost", one never entered); s122nz 5/7 (misses 28180 and 9196). xt = ground-truth
distance to the route polyline while moving (lane-centre proxy; with zones off it is dominated by cars that leave the lane, medians are not comparable across
the zones axis).

Paired per-route DS differences (6 routes, seed 2; sd / se are context only):

| pair | 28180 | 24944 | 27297 | 9196 | 6999 | 34183 | mean | sd / se |
|---|---|---|---|---|---|---|---|---|
| s122 - s143 (zones on) | +30 | +58 | 0 | -25 | +12 | 0 | **+12.5** | 28.6 / 11.7 |
| s122nz - s143nz (zones off) | -29.9 | +75.1 | -6.8 | -7.3 | +13.9 | -7.0 | **+6.3** | 36.4 / 14.9 |
| s143 - legacy (clip + delay) | 0 | -28 | 0 | +26.8 | +26.3 | -40 | **-2.5** | 27.4 / 11.2 |

RC paired means: s122 - s143 -6.5, s122nz - s143nz +39.3, s143 - legacy +16.1.

Action-head curvature in the junction turns (|act_k| over the in-turn window, entered turns, 1/m; mean over turns of the per-turn max / of the per-turn mean):

| arm | n turns | max |act_k| | mean |act_k| |
|---|---|---|---|
| s143nz | 6 | 0.133 | 0.0070 |
| s122nz | 7 | 0.869 | 0.0426 |
| s143 (zones on) | 7 | 0.240 | 0.0261 |
| s122 (zones on) | 7 | 0.269 | 0.0290 |

Reading A: the 1.22 m rig asks for several times more curvature in the turns than 1.433 m when only the action head steers (mean 0.043 vs 0.007; the 1.433 cars
mostly never start the turn: 0/7 branches, stuck or stray, v 0.76 m/s) and it takes 5 of 7 turns with the action head alone against 0 of 7, at higher
infraction cost (10 collisions vs 3, p95 cross-track 9 m). With zones on the two heights are indistinguishable on turns (7/7 vs 6/7, same
0.03 mean curvature) and the DS gap (+12.5) is carried by 24944 (+58) and 28180 (+30), against 9196 (-25); 27297 and 34183 equal or tied in DS. With 6 routes,
one seed and the large run-to-run noise (see B: identical on unaffected routes, but 24944 s143 42 vs s122 100 differ by a single red light, 30 DS points)
this is a direction, not a result: 1.22 does not break driving, and it is not worse on average than 1.433 in either regime. Zones-off is a bad reading in
absolute terms for both heights (s143nz 0 of 6 routes completed). Legacy vs s143: no mean change (-2.5), but legacy lost 2 turns / had a TickRuntime failure
and RC fell to 83.9 against 100 for s143, so clip + delay mainly protects route completion.

## B. nored vs timer on the decision-102 drive arm

Old = `$DATA_DIR/runs/op_img_cmd/cl/arms/drive-s0 / -s1` (nored), new = `unified/b2d/arms/drivetimer-s0 / -s1` (`"resume": "timer"`).

| route | seed | DS old | DS new | RC old | RC new | red light old / new | blocked old / new |
|---|---|---|---|---|---|---|---|
| 24944 | 0 | 70.0 | 70.0 | 100 | 100 | 1 / 1 | 0 / 0 |
| 27043 | 0 | 60.0 | 60.0 | 100 | 100 | 0 / 0 | 0 / 0 |
| 27297 | 0 | 70.0 | 70.0 | 100 | 100 | 1 / 1 | 0 / 0 |
| 9196 | 0 | 19.6 | 22.3 | 58.0 | 100 | 0 / 1 | 1 / 0 |
| 24944 | 1 | 100.0 | 70.0 | 100 | 100 | 0 / 1 | 0 / 0 |
| 27043 | 1 | 60.0 | 60.0 | 100 | 100 | 0 / 0 | 0 / 0 |
| 27297 | 1 | 70.0 | 70.0 | 100 | 100 | 1 / 1 | 0 / 0 |
| 9196 | 1 | 31.3 | 42.0 | 100 | 100 | 0 / 1 | 0 / 0 |

| set | n | DS old | DS new | DS diff (new - old) | RC diff | red lights old / new |
|---|---|---|---|---|---|---|
| all | 8 | 60.1 | 58.0 | -2.1 | +5.2 | 3 / 6 |
| nored acted (9196 s0, s1; 24944 s1) | 3 | 50.3 | 44.8 | -5.6 | +14.0 | 0 / 3 |
| nored never acted (27297, 27043 both seeds, 24944 s0) | 5 | 66.0 | 66.0 | 0.0 | 0.0 | 3 / 3 |

Reading B: the five routes where nored cannot act reproduce exactly (DS and infractions identical to the decimal), so run-to-run noise is zero there and
every difference is attributable to the resume. Removing the ground-truth light costs the 3 affected runs exactly the 3 red-light infractions they had avoided
(-30 DS on 24944 s1) and partly buys back route completion (9196 s0 RC 58 -> 100, no blocked; DS gains +2.7 and +10.7 on 9196 offset by -30 on 24944 s1). Net: -2.1 DS on all 8, -5.6 on the 3 acted
runs; decision 102's drive-arm numbers were inflated by about 2 DS (all) because of the privileged resume, in direction only, n = 8.

## Odd
- Three TickRuntime failures (legacy 6999, s143nz 34183, s143nz 6999) are CARLA/agent runtime failures on cars with v ~0.2-0.3 m/s (stuck cars), not infrastructure retries; no unit needed attempt 2.
- 9196 fails or scores low in every arm except s143 (40.5 / 100): the arm that completed it is not the one expected to; single seed.
- Everything above is seed 2 for A, so differences smaller than ~30 DS on one route are inside the one-red-light granularity (10-30 points per infraction).
