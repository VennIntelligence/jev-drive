# AUD1: does the served nuPlan driver assume things the public logs satisfy and other logs do not? (2026-10-11)

Question: the official submission (`teams/lao-siji:apy10m10-ab-s0-v1` = image `jev-alpasim:apy10m10-ab-s0-2d7d7fdf`, checkpoint
`APY10m10-AB-s0`, `ap2` driver) scored 0.640 on the private 1000-rollout suite (at-fault distance 0.132, GT distance 2.29 m) against 0.935 on
the 1491 public scenes. Is part of that a driver input path (route, egomotion, cold start, calibration, state, serving config) that breaks
outside the public logs?

**Answer in short.** No input path of the driver breaks under the inputs the deployed evaluator (AlpaSim 0bb4c4b, the commit every public
and private nuPlan preset runs) can produce. Several paths are fragile by construction and are only safe because the evaluator is
deterministic about them: exact timestamp equality between CAM_F0 and the ego pose, the 17 ms companion pose of the first decision, the
1920 x 1080 image size. Two assumptions do hold on the public logs and plausibly fail more often on other logs: the driver always serves
right-hand traffic although the checkpoint was trained with the left-hand flag on Singapore rows, and the plan barely reads the route
command, so it cannot follow a route that turns where the image does not. Neither is a driver bug that a fix in serving code removes for
certain; both are listed with their measured size below.

Method. `scripts/aud1_driver_tests.py` builds the served Driver exactly as the image does (`ap2_driver.Driver` around
`ap2_core.Core("APY10m10-AB-s0", synth="gpu")`, interpreted passes as with `SH30_COMPILE=0`, serving switches off), replays the logged
driver-side requests of public rollouts (`c1_extract.py msgs`: `lat1/msgs400`, 400 scenes, plus 29 scenes, 21 of them Singapore, extracted
from the `APY10m10-AB-s2` run under `runs/onpol_nuplan/seeds4`) with ONE input changed, and reports the plan at 1 / 2 / 3 / 4 s against the
unchanged replay at the same decision. Open loop: the logged state of every decision is kept. Replaying the nominal input twice gives
identical plans (0.000 m). Scenes by role (k = 0 speed, heading in the local frame): `cruise` veh-35 00083-0c318d79 (8.9 m/s, 87 deg),
`stop` veh-35 00083-431ae299 (0 m/s, launches), `slow` veh-35 04097-d90cd23a (1.7 m/s), `turn` veh-35 00083-83fd1c19 (4.9 m/s, LEFT
command), `sg` veh-52 00422-fb1d534c (Singapore, 3.9 m/s). Pool job `1011-220425-cd42` (one card, 63 s); statistics over 429 scenes of
messages and the served runs' `drive.jsonl` (1558 rollouts of APY10m10-AB-s0) in tmux on the box. Raw numbers:
[aud1/driver_tests.json](aud1/driver_tests.json), [aud1/driver_stats.json](aud1/driver_stats.json). Box artefacts: `$DATA_DIR/runs/alpasim/aud1_driver/`.
Notation below: dy4 / dyaw4 / dx4 = test minus nominal lateral offset / heading / longitudinal position of the plan at 4 s (largest
absolute value over the 10 decisions unless a decision is named).

| item | verdict | one line |
|---|---|---|
| 1. route waypoints | safe | NaN / short / far / behind / beside routes never raise; only the 4-way command reaches the model, and it moves the plan by at most 0.11 m (straight road) to 0.39 m (turn) at 4 s |
| 2. egomotion | safe in this evaluator, suspicious by construction | the start-state rotation guard works on all 1558 public rollouts; it fails (8.9 m/s fed as lateral velocity, plan heading off 47 deg) if the first message lacks the 17 ms pose, and from 10 cm position noise at 1.7 m/s |
| 3. cold start | safe | backfill at 0 / 8.9 / 25 m/s gives sane plans; a grid other than 0.5 s silently serves every decision cold; a frame time off by even 1 ms aborts every `drive` |
| 4. camera calibration | safe | K, R (pitch, roll, yaw), distortion and t (height, x) all reach the frames or the output; no fixed rig; the only fixed camera property is the 1920 x 1080 image size |
| 5. scene length and state | safe | no state accumulates beyond the 4-frame window; 24 extra decisions give a steady plan (1.4 cm) at unchanged latency; 5 concurrent sessions in one process are bit-identical to sequential replays |
| 6. what the image serves | safe (as intended) | image = the pushed digest; APY10m10-AB-s0 (sha matches the box), ap2 driver, backwarp, switches off, interpreted passes; the 48-scene local smoke ran that exact image with no overrides |
| extra: traffic side | suspicious | trained with the left-hand flag on Singapore rows, always served right-hand: 0.94 m / 5.6 deg at 4 s on a Singapore scene; public Singapore 0.903 against 0.935 overall |

## 1. Route waypoints

Code: `ap2_driver.py:39-46` (`submit_route`: command by the shipped sample's `command_from_route`, waypoints kept), `sh30_driver.py:60`
(the shipped rule, imported unchanged: first waypoint with hypot >= 5 m, y > 2 m LEFT, < -2 m RIGHT, none STRAIGHT, no waypoints UNKNOWN),
`ap2_core.py:97-98` (waypoints reach the model only for a route arm), `ap2_inputs.py:45-50` (`route_feat`, the (x - 60) / 20, y / 20 encoding),
`parity_adapter.py:37-44` (`ego_features` reads `cmd[:3]`: UNKNOWN becomes an all-zero command).

What the evaluator sends (source `route_generator.py:110-152, 201-235`, `events/policy.py:107-113`; statistics over 4290 decisions of 429
scenes): always 20 entries, NaN-padded; 10 valid in the median, 0 valid in 9 decisions, fewer than 5 in 59; first valid x 42 m in the
median, -20.9 m minimum (ego past the recorded route), valid x from -54.9 to 83.8 m. Commands: LEFT 35 %, STRAIGHT 46 %, RIGHT 19 %; the
shipped rule and `ap2_inputs.route_cmd` (the training rule) agree on 4290 / 4290. The command flips between k = 0 and k = 1 in 26 / 429 scenes;
at standstill (< 0.5 m/s, 286 decisions) it is STRAIGHT 201, LEFT 64, RIGHT 21.

| test (every decision) | cruise: command, dy4 / dyaw4 / dx4 | turn: command, dy4 / dyaw4 / dx4 |
|---|---|---|
| 20 NaN waypoints | S, 0 / 0 / 0 | S (was L), 0.39 m / 1.2 deg / 0.6 m |
| 0 waypoints (UNKNOWN = all-zero command) | U, 0.03 / 0.07 / 2.3 | U, 0.37 / 0.7 / 1.6 |
| no route message at all | U, same as 0 waypoints | U, same |
| first 5 valid only, no padding | S, 0 | L, 0 |
| first valid waypoint NaN | S, 0 | L, 0 |
| all waypoints 40 m further (80-120 m) | S, 0 | L, 0 |
| shifted 50 m back (starts ~8 m behind) | S, 0 | L, 0 |
| shifted 3 m left / right | L / R, 0.08 / 0.11 m | L / S-then-L, 0 / 0.39 m |
| shifted 5 m left / right (forced LEFT / RIGHT) | L / R, 0.08 m, 0.16 deg / 0.11 m, 0.27 deg | L / S-then-L, 0 / 0.39 m |
| 20 NaN at the standing start (`stop`) | S, 0.59 m / 1.7 deg / 0.4 m | |

Verdict: **safe**. Nothing raises; NaN padding, short routes, routes far ahead or behind the ego all reduce to one of the four commands as
the training rule does (no NaN ever reaches the model). The served checkpoint is not a route arm (`ap2` metadata `route: False`, checked
in the checkpoint), so the (x - 60) / 20 encoding is not served; it would be unclipped in x (route x up to 84 m and down to -55 m gives
[-5.7, 1.2]) while y is clipped to +-3, which matters only for a route arm. The finding that matters is the size of the command's effect:
forcing LEFT or RIGHT on a straight road moves the 4 s point by 8-11 cm, and taking the LEFT away on the turn scene by 39 cm. The plan
follows the image, not the route. On a private scene whose log turns where the road ahead does not look like a turn, the driver cannot be
told to turn; that is a model property, not an input bug (see "Most likely contributors").

## 2. Egomotion

Code: `sh30_driver.py:124-137` (`Session.velocity`: a state is rotated by the pose yaw when that matches the pose-difference velocity better
by 0.5 m/s), `ap2_driver.py:67-75` (start states with time <= the first `drive` and speed < `SLOW` = 1 m/s, `ap2_driver.py:35`, are rotated
unconditionally: the source's rule), `sh30_driver.py:242-246` (history poses in the t0 rig frame from positions and quaternion yaw),
`sh30_core.py:114-127` (`fill_history`), `op_interp.py:85-113` (`EgoTrack`: yaw from the quaternions, velocities from the states).

Source facts: AlpaSim seeds the start states from the recording and rotates them by -yaw (`event_loop.py:183-209`), the first decision is at
render start + 17 ms (`unbound_rollout.py:105-143`: the synthetic first frame ends one shutter time after the scene start) and its egomotion
message holds two poses (t = 0 and 0.017 s) in all 429 scenes; there is no egomotion noise model at 0bb4c4b (`events/policy.py:191-210`
says so; no noise option in the runtime config). Heading is never inferred from positions in the served path: yaw comes from the
quaternions everywhere; positions enter only through the history poses and through the rotation guard's pose-difference velocity
(`velocity`, a 17 ms baseline at k = 0, 0.5 s later). `damp_history` (motion 1, a no-op), `cv_history` (adapter `in_tf` is None for this
checkpoint) and `track_wod` (velocity heading) are not on the served path.

| test | role | observed |
|---|---|---|
| start states delivered in the rig frame (an evaluator without the -yaw quirk) | cruise / slow / stop | dy4 <= 0.003 m, dx4 <= 0.17 m: the guard leaves correct states alone above 1 m/s; below 1 m/s the unconditional rotation turns a correct vector, but the vector is < 1 m/s (stop: 0.00 m/s fed either way) |
| first egomotion message with one pose (no 17 ms companion) | cruise | k = 0 fed (vx, vy) = (0.39, -8.90) instead of (8.91, -0.18): dy4 1.45 m, dyaw4 47.2 deg, dx4 -23.1 m; k >= 1 unchanged |
| same | slow (1.7 m/s) | fed (0.73, -1.49): dx4 -8.2 m, dyaw4 1.5 deg |
| N(0, 3 cm) on every pose x, y | cruise / slow / stop | dy4 0.12 / 0.09 / 0.38 m, dyaw4 0.2 / 1.1 / 3.4 deg; the k = 0 rotation still right |
| N(0, 10 cm) on every pose x, y | slow | the guard flips at k = 0 (fed (0.73, -1.49)): dx4 -8.2 m; later decisions dy4 up to 0.66 m |
| N(0, 1 deg) on every pose yaw | cruise / slow | dy4 0.87 / 1.46 m, dyaw4 1.8 / 17.6 deg (slow, k = 9, a decision already at a turn boundary) |
| acceleration missing (0) | cruise / slow | dy4 0.08 / 2.47 m (slow k = 9; 0.33 m elsewhere), dx4 up to 6.9 / 9.2 m: the speed profile reads ax |
| served logs, 1558 public rollouts | | k = 0 state rotated in 1510; fed \|vy\| > 1 m/s at k = 0: 0; fed \|vy\| > 0.5 m/s at k >= 1: 8 of 14 022 |

Verdict: **safe in the deployed evaluator, suspicious by construction.** The guard that undoes the -yaw quirk depends on a second pose 17 ms
after the first and on position noise well below speed x 17 ms; both hold in every public rollout and are fixed by the runtime's code, not
by the log. If a decision ever starts with one pose, a moving ego is fed its speed as lateral velocity and the k = 0 plan swings by
tens of degrees. The model is sensitive to yaw noise in the history (1 deg per pose gives up to 0.9 m at 4 s while cruising) but the
evaluator's poses are noise-free.

## 3. Cold start

Code: `ap2_driver.py:59-62` (keys: frames 0.5 s +- `TOL_US` = 2 ms before t0 that also have a pose, `sh30_driver.py:72`), `ap2_driver.py:54-58`
(t0 = the newest time with BOTH a frame and a pose, exact equality, and `now - t0` <= 2 ms, else FAILED_PRECONDITION),
`sh30_core.py:114-127` (constant-velocity, constant-yaw-rate backfill of missing history), `sh30_core.py:159-177` + `op_interp.py:200-237`
(backwarp: slots before the first keyframe are that keyframe re-projected to the back-extrapolated poses), `ap2_inputs.py:20-21`
(T_KEY / N_SLOT: the 0.5 s grid is built in).

| test | role | observed |
|---|---|---|
| nominal ladder | all | n_keys 1, 2, 3, 4, 4, ... (served logs: the same at every k for 1558 rollouts) |
| k = 0 speed 0.05 m/s | cruise | a launch plan (x at 4 s 4.6 m instead of 30.3 m), dy4 0.26 m, dyaw4 4.1 deg |
| k = 0 speed 25 m/s (pose step scaled too) | cruise | x at 4 s 82 m (20.5 m/s mean), dy4 0.69 m, dyaw4 0.1 deg: backfill 37.5 m behind, sane |
| standing start, nominal | stop | x at 4 s 5.3 m, y -0.03 m, yaw 0 deg |
| decision / frame / pose grid 0.4 s or 0.6 s | cruise / stop | no key matches: n_keys = 1 at EVERY decision, no error; dy4 0.08 / 0.48 m, dx4 5.3 / 2.0 m |
| CAM_F0 `frame_end_us` 1 ms or 3 ms after the pose time from decision 1 on | cruise / stop | 9 of 10 `drive` calls abort FAILED_PRECONDITION (a failed rollout in the evaluator) |
| first decision at t = 0 (no 17 ms offset, one start pose) | cruise | k = 0 as "one pose" above (47 deg); k = 1 has one key (the t = 0 frame is 17 ms off the 0.5 s step); ladder one step late |
| same | stop | dy4 0.01 m: harmless at rest |

Verdict: **safe** in this evaluator (control and frame interval 0.5 s, frame end = decision time exactly in all 4290 decisions,
`frame_end_minus_now_max_us` 0). Two silent or total failure modes exist: another step length makes every decision cold without a log line,
and any offset between frame and pose timestamps (1 ms is enough, because t0 needs exact equality; the 2 ms tolerance only applies to
`now - t0`) aborts every decision after the first.

## 4. Camera calibration

Code: `sh30_driver.py:104-116` (Session: K, distortion, `rig_to_camera` rotation and translation from `available_cameras`; rational or
thin-prism distortion and non-pinhole models are rejected with FAILED_PRECONDITION), `sh30_core.py:57-84` + `navsim_zs.py:169-205`
(OpenpilotMaps: model rays through the full rotation R and K / distortion into CAM_F0; translation not used, pure rotation remap),
`navsim_zs.py:115-117` (`calib_key` = R, K, D rounded to 1e-4; t omitted, which is correct because the map does not use t),
`op_interp.py:146-163, 200-237` (slot warp through the road plane z = 0 at camera height t.z; `_plane_gpu` cache keyed by all three t),
`ap2_core.py:111` + `op_interp.py:427` (plan to rear axle through the camera lever arm t.x, t.y).

Which components reach the model frames (cruise, share of model-frame pixels changed; newest slot = the keyframe, oldest = a warped slot):

| change | newest slot k0 / k5 | oldest slot k0 / k5 | plan dy4 / dyaw4 / dx4, cruise | turn |
|---|---|---|---|---|
| R: +1 deg about camera x (pitch) | 84 % / 84 % | 86 % / 84 % | 0.11 m / 0.3 deg / 1.7 m | 0.59 m / 2.3 deg |
| R: +1 deg about camera z (roll) | 71 % / 72 % | 82 % / 70 % | 0.01 m / 0.1 deg / 0.4 m | 0.12 m / 0.6 deg |
| R: +1 deg about camera y (yaw) | | | 0.28 m / 0.8 deg / 3.4 m | 0.45 m / 1.2 deg |
| t.z + 0.3 m (height) | 0 % | 22 % / 30 % | 0.05 m / 0.1 deg / 4.8 m | 0.24 m / 1.3 deg |
| t.x + 0.5 m | 0 % | 14 % / 4 % | 0.01 m / 0.0 deg / 0.1 m (lever arm) | 0.09 m |
| f x 1.02 | 71 % / 73 % | 80 % / 70 % | 0.03 m / 0.1 deg / 0.3 m | 0.14 m |
| CAM_F0 at 960 x 540, K scaled | every image rejected: IndexError in the flat-index sampler, INVALID_ARGUMENT, no `drive` succeeds | | | |

Constants that encode a nuPlan vehicle or sensor property, all of them on the served path:

| constant | where | role |
|---|---|---|
| `NUPLAN_WH = (1920, 1080)` | `navsim_zs.py:27` | CAM_F0 size for the flat-index sampling maps; a smaller image aborts, a larger one would be sampled silently wrong |
| road plane z = 0 in the rig frame | `op_interp.py:148-152, 212` | the warp assumes the rig origin is on a flat road (nuPlan rig origin: rear axle at ground, camera t.z 1.49-1.54 m) |
| `D_FAR = 60` m | `op_interp.py:33` | rays above the horizon put at 60 m in the warp |
| `OP_K`, openpilot model K | `op_interp.py:30-31`, `openpilot/frames.py` | the model's own virtual cameras (not nuPlan) |
| `STEP_US = 500_000`, `TOL_US = 2_000` | `sh30_driver.py:72` | the 0.5 s key grid and its tolerance |
| `T_KEY`, `SLOT_T`, `N_SLOT` | `ap2_inputs.py:20-21`, `sh30_core.py:44` | 2 Hz history, 0.2 s policy slots |
| `SLOW = 1.0` m/s | `ap2_driver.py:35` | below it the start state is rotated unconditionally |
| cam `[1.7, 0.0, 1.5]` | `ap2_driver.py:122, 126` | warm-up decisions only, never a served plan |
| `cam_to_front = 2.26` default | `serve_fix.py:102` | only with a serving switch on (off in the image); otherwise from the session's ego box |
| ego box 5.176 x 2.297 m, `CENTER` 1.461 m | `scripts/c1_lib.py` | analysis only, not served |

No wheelbase, camera height, pitch or ego dimension is hard-coded in the served path. Per-scene values seen: camera x 1.64-1.81 m,
height 1.49-1.54 m, fx 1526-1623, fy 1362-1633 over 429 scenes; distortion delivered as zeros (the renderer's pinhole).

Verdict: **safe.** Every calibration component the evaluator sends is used where it physically belongs; the caches are keyed by the
components they use. The only fixed sensor property is the image size, which the shared nuPlan config fixes at 1920 x 1080 for every preset.

## 5. Scene length and state

Code: `sh30_driver.py:191-194` (frames pruned to the newest 4), `sh30_driver.py:205-210` (poses / states kept for the whole session,
looked up by time), `ap2_driver.py:53` (`t_first` per session), `sh30_driver.py:149, 156-178` (sessions keyed by `session_uuid` under a lock,
removed on close; the scene id is not used for any state: it is empty in benchmark runs), `sh30_driver.py:150, 252` + `ap2_driver.py:81`
(one GPU lock around `Core.plan`), `sh30_core.py:54-73` (`_MAPS`, `_IDX`: per-calibration sampling maps, process-global, never evicted,
~14 MB each), `op_interp.py:206` (`_GPU` plane cache, cleared at 32 entries).

| test | observed |
|---|---|
| 24 decisions after the logged 10 (same image, constant velocity and heading, same route): 34 decisions, 17 s | no error; from the 4th extra decision on the plan is constant to 1.4 cm (float noise of the growing local coordinates); `drive` 28.9 ms over the first 10, 29.7 ms over the last 10 |
| 5 sessions replayed at once in 5 threads through one Driver (4 scenes, one of them twice), twice | every plan identical to its sequential replay (max difference 0.000 m) |

Verdict: **safe.** Nothing but the 4-frame window and the per-session pose history carries over between decisions, and nothing is keyed by
scene; two rollouts of the same scene at once do not interfere. Memory grows with the number of distinct calibrations a process sees
(about 30 per replica on the private suite, ~0.4 GB) and with session length (one pose per 0.5 s), both negligible.

## 6. What the pushed image serves

Checked on the Tokyo box: `jev-alpasim:apy10m10-ab-s0-2d7d7fdf` and `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:apy10m10-ab-s0-v1`
are the same image id `dcf5738172fb` (repo digest `abdbd7dc...`). Its environment: `JEV_TAG=APY10m10-AB-s0`, `JEV_DRIVER=auto`
(`tags.json` maps it to `ap2`), `JEV_VCONT=0`, `JEV_LEAD=0` (JEV_BASE / JEV_STOP / JEV_REPLAN unset = off), `SH30_COMPILE=0` (interpreted
passes; `Dockerfile:67-68`), `SH30_SYNTH` unset = gpu, `AP2_COLD` unset = the checkpoint's `backwarp`, `SH30_LHT` unset = right-hand traffic,
`ALPASIM_DRIVER_GRPC_WORKERS=8`, `SH30_GIT_HASH=2d7d7fdf`, `CMD python /app/serve.py`, user 10001. Checkpoint sha256 `0a0b8678...` in the image
equals the box's `APY10m10-AB-s0/ckpt-final.pt`; the nine driver-path files in the image hash equal to the repo at `HEAD` (no change to the
closure since 2d7d7fdf). The local test (`tokyo_image/APY10m10-AB-s0-20261011-165113/test`) and the 48-scene containerised smoke (`.../smoke`)
both ran `jev-alpasim:apy10m10-ab-s0-2d7d7fdf` with no `-e` override but the replica index / count; the driver log reads `serving: JEV_VCONT 0,
JEV_LEAD 0, JEV_BASE 0` and `APY10m10-AB-s0 (backwarp, route arm False)`. Containerised 0.9708 against native 0.9701 on those 48 scenes.

Differences between the local evidence and the submission, all known: the native 1491-scene runs (0.935) used compiled passes
(`SH30_COMPILE` default 1, inductor), the image interpreted ones (fp16-level plan differences, results/lat1_frame_synthesis.md); the smoke
ran 8 concurrent rollouts on one replica, the official run 2 per replica on 16 replicas (concurrency is shown harmless in item 5); the 48
smoke scenes are all from one Las Vegas log set.

Verdict: **safe**: the image serves what was intended and what was tested.

## Extra: traffic side

`ap2_driver.py:84` passes `lht=self.lht`, which is `SH30_LHT=1` only (`ap2_driver.py:124`); the image leaves it unset. Training sets the
flag per row from the map (`experiments/op_parity/scripts/pp_prep.py:87`, `pp_train.py:477`: `sg-one-north` rows are left-hand). Serving
the Singapore scene with the flag on moves the plan by up to 0.94 m and 5.6 deg at 4 s; the US cruise scene moves by 0.02 m. The driver
cannot read the map, but the vehicle is a proxy (Singapore logs are veh-52 here). Public Singapore scenes: 220 of 1491 (2 logs), APY10m10-AB-s0
0.903 against Las Vegas 0.977, Boston 0.926, Pittsburgh 0.907. Verdict: **suspicious**: a train / serve mismatch that costs on every
Singapore scene; its closed-loop size is not measured.

## Most likely contributors to the private-set drop (ranked; none is proven by this audit)

1. **The plan does not follow the route command.** Forcing LEFT or RIGHT moves the 4 s point by 0.08-0.11 m on a straight road; removing
   LEFT on a turn by 0.39 m. Private GT distance 2.29 m against 1.13 m mean on the public scenes points at the ego leaving the log's path,
   which is what a policy does when the log turns and the image does not demand it. Not a serving bug; it is how APY10m10-AB uses
   its inputs (`parity_adapter.py:37-44`: three command bits among 19 ego numbers).
2. **Left-hand traffic is never signalled** (`ap2_driver.py:84, 124`; image env). If the private suite holds more Singapore than the public
   15 %, every one of those scenes runs with a flag the checkpoint never saw on Singapore rows.
3. **Longer scenes, if the private window is longer.** The driver is fine with long scenes (item 5), but `n_sim_steps` is 200 capped by the
   recording, and the public scenes are 10 decisions only because of their 8 s trajdata window. The closed-loop heading drift of this model
   grows with decisions (decision 205); a longer private window would turn that drift into offroad / corridor zeros and a larger GT distance.
   Not checkable from the driver.

Ruled out as explanations: route NaN padding / short routes (item 1), the start-state rotation and cold start (items 2-3, all 1558 public
rollouts correct and the evaluator's timing is structural), calibration handling (item 4), concurrency and state (item 5), the image's
configuration (item 6). The fragile spots found (one-pose start, exact frame / pose timestamp equality, fixed image size) cannot be
triggered by AlpaSim 0bb4c4b with the nuPlan presets; they would turn into immediate failures or 47-degree first plans under any other
runtime, so they are worth a guard before any other track or a runtime update.

Code: `scripts/aud1_driver_tests.py` (`tests` = the pool job, `stats` = the message and served-log statistics).
