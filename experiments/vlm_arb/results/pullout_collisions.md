# Did the vehicle that hit us react? Pull-out collisions of pbyp, pbyp2 and pbyp2ng

Offline read of the privileged logs of the runs in which the car pulled out around an obstacle: arms `pbyp` (old), `pbyp2`, `pbyp2ng` on the four obstacle routes (24497, 2520, 19324, 19832), 2 traffic seeds each, 24 runs (8 per arm; the chase-camera reruns `v2-gif*` are not used; `drive` runs do not pull out and only feed the headway series). Definitions: [../plans/2026-10-02-offline-analyses-definitions.md](../plans/2026-10-02-offline-analyses-definitions.md) section C. Script: [pullout_collisions.py](../scripts/pullout_collisions.py) (uses [bypass_extract.py](../scripts/bypass_extract.py) / [bypass_common.py](../scripts/bypass_common.py) for the compact extracts). Row-level CSVs: `pullout_collisions.csv`, `pullout_events.csv`, `pullout_headways.csv`.

**Frame and definitions.** Everything is in the route frame: arc position along the official route polyline and lateral offset from it (positive here = towards the adjacent same-direction lane, which is on the right on all four routes; the lane separation is the bypass offset, 3.25 m on three routes and 3.5 m on 24497). *Pull-out start* = last time before the entry at which the ego centre was within 0.15 m of its lane centre. *Lane entry* = the first time the outermost body corner on the lane side (centre offset + 0.92 m x cos(yaw error) + 2.45 m x sin(yaw error) when turned towards the lane) crosses half the lane separation, coming from inside the own lane for at least 1 s. *Same-direction vehicle of the adjacent lane* = a vehicle whose lateral offset is within 1.75 m of the lane centre and whose heading is within 45 degrees of the route. Gaps are bumper to bumper along the route (vehicle half lengths 2.45 m for the ego and the logged extent for the other). *Braked* = the other vehicle's acceleration over a 0.6 s window reached -3 m/s2 or lower between the pull-out start and 0.5 s before the contact. Contact times in `contacts.jsonl` run on the world clock, 1.0 to 1.4 s ahead of the scenario clock of `privileged.jsonl` (constant per route: 19324 1.1 s, 19832 1.0 s, 24497 1.35 s, 2520 1.4 s, measured from the frame numbers of both logs); all times below are on the scenario clock.

## 1. Every collision

19 contacts with a moving same-direction vehicle of the adjacent lane (12 in `pbyp`, 3 in `pbyp2`, 4 in `pbyp2ng`; one row per other vehicle, first contact). Contacts with static props, with standing vehicles and with vehicles outside the adjacent-lane band are not in the table.

| arm | seed | route | contact t s | situation | gap at pull-out start m | time gap s | gap at lane entry m | time gap s | ego v at contact m/s | ego lateral into the lane m | pull start -> contact s | lane entry -> contact s | other v at pull start m/s | other v at contact m/s | other peak decel after pull start m/s2 | first decel >= 1 m/s2, s after pull start | other speed change pull start -> contact m/s | braked (>= 3 m/s2) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pbyp | 0 | 19324 | 53.25 | pulled out alongside / within 6.5 m of the vehicle | -0.2 | - | -4.5 | - | 3.5 | 0.62 | 1.0 | 0.4 | 8.9 | 8.9 | 1.4 | 0.4 | -0.0 | no |
| pbyp | 0 | 19832 | 21.5 | pulled out with the vehicle >= 15 m behind | 23.4 | 2.27 | 19.0 | 1.93 | 0.1 | 0.9 | 3.65 | 3.05 | 10.3 | 7.6 | 4.5 | 1.4 | -2.6 | yes |
| pbyp | 1 | 19832 | 19.0 | pulled out alongside / within 6.5 m of the vehicle | 6.3 | 0.8 | -0.5 | - | 1.8 | 0.86 | 1.75 | 0.75 | 7.9 | 7.7 | 5.3 | 0.6 | -0.7 | yes |
| pbyp | 1 | 19832 | 21.4 | pulled out with the vehicle >= 15 m behind | 25.6 | 2.99 | 22.0 | 2.27 | 1.0 | 0.99 | 4.15 | 3.15 | 8.5 | 7.5 | 4.9 | 0.8 | 0.2 | yes |
| pbyp | 1 | 19832 | 65.6 | pulled out alongside / within 6.5 m of the vehicle | 0.0 | 0.0 | -3.5 | - | 4.8 | 0.78 | 0.95 | 0.35 | 11.5 | 8.1 | 5.6 | 0.0 | -3.4 | yes |
| pbyp2ng | 0 | 19832 | 22.1 | pulled out with the vehicle >= 15 m behind | 25.2 | 2.47 | 18.9 | 2.04 | 1.9 | 0.87 | 4.25 | 3.45 | 10.2 | 8.9 | 5.6 | 0.4 | -1.3 | yes |
| pbyp2ng | 1 | 19832 | 21.7 | pulled out with the vehicle >= 15 m behind | 25.5 | 2.53 | 20.9 | 2.14 | 0.9 | 0.86 | 3.85 | 3.25 | 10.1 | 9.1 | 6.2 | 0.4 | -1.0 | yes |
| pbyp | 0 | 24497 | 6.75 | old pbyp misfire at the route start, vehicle alongside | -4.6 | - | -4.1 | - | 2.3 | 0.77 | 0.7 | 0.3 | 8.2 | 10.1 | 0.0 | - | 1.2 | no |
| pbyp | 0 | 24497 | 20.15 | pulled out with the vehicle >= 15 m behind | 24.2 | 2.75 | 8.3 | 0.75 | 1.5 | 1.11 | 3.1 | 1.3 | 8.8 | 11.8 | 2.5 | 1.2 | 3.1 | no |
| pbyp | 1 | 24497 | 6.75 | old pbyp misfire at the route start, vehicle alongside | -4.6 | - | -4.1 | - | 2.2 | 0.77 | 0.7 | 0.3 | 8.2 | 10.1 | 0.0 | - | 1.2 | no |
| pbyp | 1 | 24497 | 27.7 | pulled out with the vehicle >= 15 m behind | 23.4 | 5.61 | 25.0 | 2.52 | 1.0 | 1.09 | 4.45 | 2.85 | 4.2 | 12.9 | 0.3 | - | 8.7 | no |
| pbyp2 | 1 | 24497 | 54.3 | ego standing partly in the lane for > 8 s | 34.2 | - | 35.0 | - | 0.0 | 1.59 | 26.85 | 26.25 | 0.0 | 12.7 | 0.8 | - | 12.6 | no |
| pbyp2 | 1 | 24497 | 58.45 | ego standing partly in the lane for > 8 s | 33.6 | - | 34.4 | - | 0.0 | 1.67 | 31.0 | 30.4 | 0.0 | 13.1 | 0.1 | - | 13.1 | no |
| pbyp2 | 1 | 24497 | 61.9 | ego standing partly in the lane for > 8 s | 33.9 | - | 34.7 | - | 0.0 | 1.72 | 34.45 | 33.85 | 0.0 | 13.1 | 0.1 | - | 13.1 | no |
| pbyp | 0 | 2520 | 6.7 | old pbyp misfire at the route start, vehicle alongside | -5.0 | - | -4.9 | - | 2.0 | 0.49 | 0.65 | 0.25 | 10.5 | 8.9 | 1.3 | 0.0 | -1.6 | no |
| pbyp | 0 | 2520 | 15.6 | old pbyp misfire at the route start, vehicle alongside | -5.0 | - | -4.5 | - | 0.0 | 0.77 | 9.55 | 9.15 | 2.0 | 10.3 | 12.4 | 4.4 | 7.7 | yes |
| pbyp | 1 | 2520 | 6.7 | old pbyp misfire at the route start, vehicle alongside | -5.0 | - | -4.9 | - | 1.9 | 0.48 | 0.65 | 0.25 | 10.5 | 8.9 | 1.3 | 0.0 | -1.6 | no |
| pbyp2ng | 0 | 2520 | 21.25 | pulled out alongside / within 6.5 m of the vehicle | -0.4 | - | -2.3 | - | 6.3 | 0.84 | 1.0 | 0.4 | 9.6 | 9.5 | 2.8 | 0.0 | -0.0 | no |
| pbyp2ng | 1 | 2520 | 21.15 | pulled out alongside / within 6.5 m of the vehicle | -1.6 | - | -4.0 | - | 6.2 | 0.81 | 0.9 | 0.3 | 9.3 | 8.0 | 3.1 | 0.4 | -0.4 | yes |

Reading it: gap and time gap are the nearest vehicle approaching from behind in the adjacent lane at the pull-out start and at the lane entry (negative or empty = the vehicle was already alongside). The next-to-last block of columns is the other vehicle's behaviour after the pull-out started. The "situation" label is computed from the pull-out start: misfire at the route start (old `pbyp`, t = 6 s, the vehicle was standing alongside and drove off), pulled out alongside or within 6.5 m of a vehicle, pulled out with the vehicle at least 15 m behind, and the ego standing partly in the lane for more than 8 s (`pbyp2` 24497 seed 1, hit three times at 0 m/s).

| situation | contacts | other vehicle braked >= 3 m/s2 after the pull-out started | median peak decel m/s2 | median other v at contact m/s | time gap at pull start, min / max s |
|:--|--:|--:|--:|--:|--:|
| ego standing partly in the lane for > 8 s | 3 | 0 | 0.1 | 13.1 | - |
| old pbyp misfire at the route start, vehicle alongside | 5 | 1 | 1.3 | 10.1 | - |
| pulled out alongside / within 6.5 m of the vehicle | 5 | 3 | 3.1 | 8.1 | 0.80 / 0.80 |
| pulled out with the vehicle >= 15 m behind | 6 | 4 | 4.7 | 9.0 | 2.27 / 5.61 |
| all | 19 | 8 | 2.5 | 9.1 | - |

## 2. Did the other vehicle brake

In plain terms: **sometimes, and never enough.** In 8 of the 19 contacts the other vehicle braked at 3 m/s2 or harder after the pull-out had started (all six 19832 contacts at 4.5 to 6.3 m/s2, one at 2520 `pbyp2ng` seed 1 at 3.1 m/s2, one 2520 `pbyp` contact at 12.4 m/s2 after a stop-and-go); it had started to decelerate (1 m/s2) 0.0 to 1.4 s after the pull-out began (4.4 s in the stop-and-go case) and had shed 0 to 3.4 m/s of about 10 m/s when it hit at 7.5 to 9.1 m/s. In the other 11 contacts it did not brake: the two 24497 `pbyp` contacts that began with a 24 m gap (315 even accelerated from 4.2 to 12.9 m/s into the car), the standing-in-lane case (three vehicles at a steady 13.1 m/s, ego at 0 m/s for 26 to 34 s), the old misfires at the route start (the vehicle was driving off alongside), 19324 `pbyp` (alongside, 1.4 m/s2), and `pbyp2ng` 2520 seed 0 (alongside, ego at 6.3 m/s, peak 2.8 m/s2).

**Gap when the ego started to pull out** (from the table): in the 6 contacts where the vehicle came from behind, 23.4 to 25.6 m, i.e. 2.3 to 3.0 s at 8.5 to 10.3 m/s (one 5.6 s, a vehicle that then accelerated to 12.9 m/s); at the lane entry 18.9 to 25.0 m, 1.9 to 2.5 s (24497 `pbyp` seed 0: 8.3 m, 0.75 s, the vehicle accelerated). In the 5 alongside cases the nearest follower was at most 6.3 m (0.8 s) behind, for four of them at 0 m or already beside the ego. The ego was slow or stopped at the line of the lane: at contact its speed was 0.1 to 1.9 m/s in the four approach-from-behind cases of 19832 (it slowed from about 5 m/s to 0 while its centre stood 0.9 m into the lane, i.e. the body half a car-width over the lane line).

**Time from lane entry to contact:** 0.25 to 0.75 s in the alongside and misfire cases, 1.3 to 3.45 s in the approach cases (3.05 to 3.45 s in the four 19832 ones), 9.2 s in one stop-and-go case and 26 to 34 s for the standing ego. From the pull-out start: 0.65 to 1.75 s alongside and misfire, 3.1 to 4.5 s approaching.

## 3. All pull-outs, collided or not

The same measures for all 28 pull-outs of these runs, with the follower's braking in the 5 s after the lane entry (acceleration over 0.6 s windows).

| arm | seed | route | lane entry t s | contact within 12 s | start-of-route misfire | ego v at pull start | ego v at entry | nearest follower gap at pull start m | time gap s | gap at entry m | time gap s | follower v at entry m/s | follower peak decel in 5 s after entry m/s2 | follower speed change in 5 s after entry m/s | ego time in the adjacent lane s | vehicle alongside at pull start |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pbyp | 0 | 19324 | 17.45 | no |  | 6.0 | 6.1 | 11.8 | 0.93 | 8.2 | 0.73 | 11.3 | 14.4 | -11.0 | 18.0 | no |
| pbyp | 0 | 19324 | 52.85 | yes |  | 2.2 | 3.2 | 50.5 | 9.71 | 48.4 | 6.53 | 7.4 | - | - | - | yes |
| pbyp | 1 | 19324 | 17.25 | no |  | 6.4 | 6.5 | 13.5 | 1.08 | 11.1 | 0.94 | 11.8 | 14.6 | -11.8 | 17.8 | no |
| pbyp | 1 | 19324 | 46.25 | no |  | 8.0 | 7.4 | 6.4 | 0.54 | 4.7 | 0.39 | 12.1 | 15.0 | -12.1 | - | no |
| pbyp2ng | 0 | 19324 | 17.45 | no |  | 6.1 | 6.4 | 12.7 | 1.01 | 9.2 | 0.79 | 11.6 | 16.0 | -11.6 | 18.2 | no |
| pbyp2ng | 1 | 19324 | 17.05 | no |  | 6.5 | 6.7 | 15.9 | 1.32 | 13.5 | 1.08 | 12.5 | 16.1 | -10.4 | 17.4 | no |
| pbyp | 0 | 19832 | 18.45 | yes |  | 3.3 | 2.4 | 0.2 | 0.02 | 19.0 | 1.93 | 9.8 | 4.5 | 0.2 | 18.4 | no |
| pbyp | 1 | 19832 | 18.25 | yes |  | 3.6 | 2.3 | 6.3 | 0.8 | 22.0 | 2.27 | 9.7 | 4.9 | -0.9 | 21.0 | no |
| pbyp | 1 | 19832 | 65.25 | yes |  | 3.4 | 4.5 | 0.0 | 0.0 | 60.1 | 7.85 | 7.7 | - | - | - | no |
| pbyp2ng | 0 | 19832 | 18.65 | yes |  | 2.7 | 1.9 | 2.0 | 0.19 | 18.9 | 2.04 | 9.2 | 5.6 | 0.8 | 22.0 | no |
| pbyp2ng | 1 | 19832 | 18.45 | yes |  | 3.0 | 2.2 | 2.2 | 0.2 | 20.9 | 2.14 | 9.7 | 6.2 | -2.5 | 18.4 | no |
| pbyp | 0 | 24497 | 6.45 | yes | yes | 1.2 | 1.9 | - | - | - | - | - | - | - | 6.6 | yes |
| pbyp | 0 | 24497 | 18.85 | yes |  | 0.5 | 1.6 | 24.2 | 2.75 | 8.3 | 0.75 | 11.1 | 0.2 | 1.2 | 22.8 | yes |
| pbyp | 1 | 24497 | 6.45 | yes | yes | 1.2 | 1.9 | - | - | - | - | - | - | - | 7.2 | yes |
| pbyp | 1 | 24497 | 24.85 | yes |  | 2.1 | 0.5 | 2.7 | 0.17 | 25.0 | 2.52 | 9.9 | 0.3 | 3.2 | 23.4 | no |
| pbyp2 | 0 | 24497 | 18.65 | no |  | 3.2 | 3.2 | 18.6 | 1.47 | 16.6 | 1.26 | 13.1 | 13.5 | -7.3 | 8.6 | no |
| pbyp2 | 1 | 24497 | 28.05 | no |  | 1.1 | 2.0 | 18.0 | 1.36 | 10.9 | 0.83 | 13.1 | 5.9 | -6.1 | 51.2 | no |
| pbyp2ng | 0 | 24497 | 17.85 | no |  | 2.3 | 2.4 | 23.1 | 3.12 | 23.1 | 2.17 | 10.6 | 12.7 | -10.6 | 13.0 | yes |
| pbyp2ng | 1 | 24497 | 21.45 | no |  | 2.3 | 2.0 | 22.9 | 2.09 | 14.3 | 1.09 | 13.1 | 5.5 | -5.8 | 21.8 | no |
| pbyp | 0 | 2520 | 6.45 | yes | yes | 1.2 | 2.0 | - | - | - | - | - | - | - | 9.8 | yes |
| pbyp | 0 | 2520 | 19.45 | no |  | 4.1 | 3.5 | 0.0 | 0.0 | 17.2 | 2.2 | 7.8 | 13.0 | -6.3 | 30.0 | no |
| pbyp | 1 | 2520 | 6.45 | yes | yes | 1.2 | 1.9 | - | - | - | - | - | - | - | 7.0 | yes |
| pbyp | 1 | 2520 | 23.45 | no |  | 5.1 | 5.1 | 24.0 | 1.85 | 18.7 | 1.3 | 14.4 | 10.8 | -10.3 | 15.8 | yes |
| pbyp2 | 0 | 2520 | 29.45 | no |  | 1.1 | 1.9 | 18.7 | 1.3 | 13.4 | 0.92 | 14.6 | 5.2 | -6.5 | 2.4 | no |
| pbyp2 | 0 | 2520 | 128.25 | no |  | 1.1 | 0.0 | 18.7 | 1.3 | 25.8 | 1.72 | 14.9 | 7.3 | -9.5 | - | no |
| pbyp2 | 1 | 2520 | 39.85 | no |  | 0.8 | 1.0 | 19.1 | 1.56 | 13.0 | 0.9 | 14.4 | 7.2 | -6.7 | - | no |
| pbyp2ng | 0 | 2520 | 20.85 | yes |  | 5.2 | 6.3 | 26.1 | 3.03 | 29.5 | 2.76 | 10.7 | - | - | 18.6 | yes |
| pbyp2ng | 1 | 2520 | 20.85 | yes |  | 5.2 | 6.2 | 26.1 | 3.03 | 29.5 | 2.75 | 10.7 | - | - | 20.2 | yes |

Reading it: the pull-outs that ended without a contact are not the ones with larger gaps. On 19324 the car pulled out at 6.1 to 7.4 m/s with the follower only 4.7 to 13.5 m (0.4 to 1.1 s) behind and the follower stopped completely (peak decel 14 to 16 m/s2, speed change -10 to -12 m/s), five times out of five; the 19832 pull-outs started at 19 to 22 m / 2 s behind with the ego at 2.3 m/s and ended in contact because the follower braked at only 4.5 to 6.3 m/s2 (inferred: the vehicle reacts much more strongly to an ego that is ahead of it and moving than to one that is slow and half in the lane; the data has 28 events on 4 routes, so this is a pattern, not a measurement of the traffic manager). The ego stayed in the adjacent lane 2.4 to 51 s per pull-out that ended without a contact (median 18 s, n = 11), because the obstacles are 9 to 26 m long and the car is slow past them.

## 4. Gaps between same-direction vehicles in that lane

Time headways of the vehicles passing the obstacle's start line in the adjacent lane, from the longest-observed run (at least 100 s) of each route and traffic seed (the arms of one seed share the traffic stream, so one run per seed is used to avoid counting the same vehicles twice; 24497 seed 0 has no run observed for 100 s and is left out, 7 series); bumper gap = headway x speed - lengths.

| route | traffic seed | run used | lane observed s | vehicles | headway p10 s | median s | p90 s | min s | headways < 3 s | median bumper gap m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 19324 | 0 | pbyp2 | 200.0 | 58 | 2.96 | 3.25 | 3.89 | 2.37 | 8 | 19.6 |
| 19324 | 1 | drive | 200.0 | 58 | 2.89 | 3.19 | 4.00 | 2.6 | 9 | 19.8 |
| 19832 | 0 | pbyp2 | 200.0 | 59 | 2.84 | 3.13 | 3.74 | 2.41 | 16 | 19.9 |
| 19832 | 1 | drive | 200.0 | 60 | 2.85 | 3.09 | 3.52 | 2.11 | 13 | 19.2 |
| 24497 | 1 | drive | 200.0 | 60 | 2.75 | 3.15 | 3.78 | 1.87 | 18 | 31.5 |
| 2520 | 0 | pbyp2 | 200.0 | 58 | 2.87 | 3.26 | 3.86 | 2.39 | 7 | 24.7 |
| 2520 | 1 | drive | 200.0 | 59 | 2.92 | 3.19 | 3.88 | 2.39 | 7 | 25.8 |
| all |  |  |  | 405 | 2.84 | 3.18 | 3.89 | 1.87 | 78 | 21.9 |

headway percentiles over the 405 headways of the series above: p5 2.53, p25 3.04, p75 3.45, p95 4.00, max 4.32 s; share < 2 s 0.2%, < 3 s 19.3%, < 4 s 94.6%, > 5 s 0.0%
ego time in the adjacent lane for the pull-outs that left it without a contact: n 11, min 2.4, median 17.8, max 51.2 s

Reading it: the stream is dense and regular: median time headway 3.18 s, 90% between 2.8 and 3.9 s, 0.2% under 2 s, none over 4.4 s (405 headways); the median bumper gap is 21.9 m (about 20 m on 19324 and 19832, 25 to 31 m on 2520 and 24497) at about 10 to 13 m/s. Since the ego stays in the adjacent lane for a median of 18 s (minimum 2.4 s), no gap of any realistic size ever opens in this lane: a rule that waits for a gap that covers the whole manoeuvre cannot fire (which is what the frozen gap rule of decision 87 did: 170 to 185 s of waiting on three routes). Any gap policy has to accept a gap of 2 to 3 s at the start (every gap in this stream is shorter than 4.4 s) and rely on the follower yielding to an ego that commits; the logs above say that worked when the ego kept moving ahead of the follower (19324: 5 pull-outs, no contact; the one 19324 contact was a pull-out alongside a passing vehicle) and failed when the ego stalled with its body half in the lane (19832) or pulled out alongside a vehicle (2520).

## Verified and inferred

- Verified from logs: contact times (clock offset measured), positions, speeds and extents of the ego and all vehicles within 70 m at 0.2 s, the lateral offsets, the gaps and accelerations computed from them, the headway series.
- Inferred: that the lane band of 3.25 / 3.5 m separation is the adjacent lane (from the bypass offset and the lateral position of every hitting vehicle, -3.1 to -3.4 m); the pull-out start and lane entry from the lateral offset of the ego; that deceleration after the pull-out started is a reaction to the ego (the other vehicles also brake for their own leaders, and the traffic manager's rule for yielding is not in the logs); the 3 m/s2 braking threshold (a choice).
- Not in the logs: the other vehicles' control inputs (throttle / brake), the traffic manager's decision state, a lane-level map of the adjacent lane.
- Limit: 4 routes x 2 seeds, one obstacle per route; 19 contacts and 28 pull-outs of one policy family; two thirds of the contacts are from the old `pbyp` arm (5 of 19 are start-of-route misfires or their sequel, already fixed in `pbyp2`).
