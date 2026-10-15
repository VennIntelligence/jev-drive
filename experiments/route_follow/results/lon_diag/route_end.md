# LON-DIAG reading 1: the 17 `cor_route_end` zeros of RSG-L18-s0 on PAI 281

Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md` (reading 1). Script: `experiments/route_follow/scripts/rf_londiag.py route_end`. Contact sheet: `figs/lon_diag_route_end.png`.

Gate c (this script's arc alignment vs the taxonomy CSV's `ahead_of_log_s`): max |difference| = 0.005 s (line: < 0.2 s).

Human stop point = start of the log's final run below 0.5 m/s. Cause (exclusive, in order): E log end speed >= 1 m/s or log < 2 m; Q stationary (< 0.5 m/s) vehicle in the route corridor (AlpaSim route, centre within hw + W / 2 + 0.3 m) 0-15 m ahead of the human's front bumper at the human's stop time (asl actors; the time is clipped to the rollout, `q_dt_s`; `queue_veh_logpath` in the CSV = the same on the log path extended straight); L-STOP a STOP wait line 0-12 m ahead of the stop point (centre within 4 m of the log path); L-LIGHT only UNKNOWN wait lines there; N otherwise. Intent = any decision in the 3 s before the ego reaches the stop point (arc on the extended log path) with v_p <= 0.7 v_f (served plan; `intent_model` = the same on `poses_model`). `lead_present` = the serve-side lead filter saw a lead (fix.lead.d set); `lead_limit` = the lead MPC cut the plan.

| scene | exam | red24 | cause | wait lines 0-12 m | queue vehicle | intent | min v_p/v_f (3 s) | model plan intent | v arrive (m/s) | human v 40 m before | ego early at stop (s) | ahead_of_log_s | lead seen / limit |
|:--|:-:|:-:|:--|:--|:--|:-:|--:|:-:|--:|--:|--:|--:|:-:|
| 054b5901 | x |  | L-STOP | STOP+UNKNOWN | - | no | 0.94 | no | 15.8 | 11.6 | 5.0 | 5.37 | n / n |
| 095cf563 |  |  | E | - | automobile@9.4m | no | 0.92 | no | 7.7 | 7.0 | 2.1 | 2.00 | y / y |
| 127c4f85 | x | x | L-STOP | STOP+UNKNOWN | - | yes | 0.65 | yes | 3.9 | 10.0 | 2.1 | 0.58 | y / y |
| 2d10a08d |  |  | N | - | - | no | 0.99 | no | 16.6 | 15.0 | 3.6 | 3.48 | n / n |
| 35e082a4 |  |  | N | - | - | no | 0.92 | no | 8.2 | 10.3 | 2.6 | 12.58 | y / y |
| 3fe2bac8 |  |  | E | - | - | no | nan | no | 0.4 | 0.6 | -0.0 | -5.02 | n / n |
| 44df49fe | x |  | L-STOP | STOP+UNKNOWN | - | no | 0.98 | no | 14.3 | 10.2 | 4.3 | 4.47 | n / n |
| 4cb7f7ac | x |  | L-STOP | STOP | - | no | 0.97 | no | 13.0 | 10.5 | 6.3 | 9.29 | y / y |
| 4ecc114d |  |  | N | - | - | no | 1.00 | no | 14.6 | 12.0 | 2.2 | 2.39 | y / n |
| 559d4ad6 | x | x | L-STOP | STOP | - | no | 0.93 | no | 12.3 | 11.3 | 3.8 | 4.47 | n / n |
| 68c3be44 |  |  | L-LIGHT | UNKNOWN | - | no | 1.03 | no | 15.0 | 10.8 | 3.7 | 3.78 | y / y |
| 7d994ed3 | x |  | L-STOP | STOP+UNKNOWN | - | no | 1.02 | no | 13.5 | 10.0 | 7.7 | 5.49 | n / n |
| 9f8acd00 | x |  | L-STOP | STOP+UNKNOWN | - | no | 0.97 | no | 16.4 | 10.8 | 5.2 | 5.58 | y / y |
| ba40be62 | x |  | L-STOP | STOP | - | no | 0.93 | no | 12.4 | 10.5 | 2.4 | 2.29 | n / n |
| bc309fd6 | x |  | L-STOP | STOP | - | no | 0.92 | no | 12.9 | 12.0 | 2.4 | 2.44 | n / y |
| cbd87234 | x | x | E | STOP+UNKNOWN | - | no | 0.95 | no | 14.1 | 9.7 | 9.9 | 9.77 | n / n |
| fc9fde87 | x |  | Q | STOP | automobile@10.9m | no | 0.93 | no | 15.6 | 8.5 | 7.2 | 7.49 | n / n |

| cause | n | no intent | red24 | exam |
|:--|--:|--:|--:|--:|
| E | 3 | 3 | 1 | 1 |
| L-LIGHT | 1 | 1 | 0 | 0 |
| L-STOP | 9 | 8 | 2 | 9 |
| N | 3 | 3 | 0 | 0 |
| Q | 1 | 1 | 0 | 1 |

## Registered lines

- **L1** (does the car read 'stop'): 14 non-E scenes, 13 without deceleration intent -> **does not read the stop (signal problem)** (lines: >= 10 'does not read', <= 5 'late / weak', else mixed).
- **L2** (where the signal is): L-LIGHT + Q = 2 (line >= 8: labels exist in nuPlan) -> **fail**; N = 3 (line >= 5: needs the videos) -> **fail**.

## Not registered: what the frames show (one look at the camera half of each frame, 2026-10-15; for the user to confirm)

The cause classes come from the map's wait-line classes. In the frames at the ego's arrival, signal heads are visible over the ego's approach
in most L-STOP scenes (054b5901 and 35e082a4 show a red; 44df49fe, 4cb7f7ac, 9f8acd00, ba40be62, bc309fd6 show signal heads), and also at
68c3be44 (L-LIGHT) and 2d10a08d (N). 7d994ed3 is a light-rail crossing (a train passes on the left). 127c4f85 (the one scene with intent) shows no
signal head. 4ecc114d (N) has stopped cars ahead in the next lane. So the PAI `STOP` wait-line class does not mean a stop sign, and the L2
count (L-LIGHT + Q = 2) understates the signalised stops. The rule split stands as registered; read the contact sheet before acting on L2.

![route-end contact sheet](../../figs/lon_diag_route_end.png)
