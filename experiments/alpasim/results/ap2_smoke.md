# AP2: the openpilot adapter model aligned to AlpaSim's inputs, as a nuPlan-track driver (2026-10-08)

Pre-registration (written before any AP2 score): [../plans/2026-10-08-alpasim-aligned-prereg.md](../plans/2026-10-08-alpasim-aligned-prereg.md). Code:
`lib/ap2_inputs.py` (the input standard, numpy), `lib/ap2_core.py` (one decision), `lib/ap2_driver.py` (EgodriverService on top of `sh30_driver.py`),
`scripts/ap2_route.py` (AlpaSim's route on NAVSIM tokens), `scripts/ap2_prep.py` (cold-start slot tokens), `scripts/ap2_train.py` (trainer; pp_train's
Store / PModel / Losses unchanged), `scripts/ap2_offline.py` (navtest read, closed-loop cross-check), `scripts/ap2_chain.sh` (stages), `scripts/drivers/ap2.sh`
(`run.sh <dir> ap2`). Execution evidence, not a score claim: 48 public navtest scenes of one Las Vegas drive, one run, one seed; nothing was tuned on them.

@@RESULT@@

## What AlpaSim hands a driver, and what AP2 does with it

Measured on 48 scenes with the shipped LTF sample behind the tap (`ltf_full48_tap/20261008-160440`, mean scene score 0.8735 as the untapped run) and read from
the source at 0bb4c4b.

| AlpaSim input | as AlpaSim defines it | AP2 | training rows (navtrain) |
|:--|:--|:--|:--|
| CAM_F0 JPEG | 1920 x 1080 pinhole, no distortion, 2 Hz, MTGS render; intrinsics and mounting differ per scene (fx 1542-1582, fy 1466-1570, x 1.67-1.80 m, z 1.52 m) | **used**: packed into openpilot's road + wide frames with the session's own calibration (same virtual views as training) | real nuPlan JPEGs, same packing. **Not alignable: the rendered-image domain** (there are no MTGS renders of navtrain) |
| 7 other cameras | same | **not used**, dropped undecoded: side / rear cameras add nothing through Cinque's encoder (decision 144) | - |
| frame rate | 2 Hz | **used** through protocol W: 8 policy slots 0.2 s apart, the frames between keyframes re-projected along the ego track (native 2 Hz slots lose 4 EPDMS, decision 142) | the cached W tokens |
| history | none before t = 0: decision k has min(k + 1, 4) keyframes; pose time = exposure end (k x 0.5 s + 17 ms) | **used as given**; the missing slots follow the rule the model was trained with (chosen: `backwarp`, below) | decision type m = 1 / 2 / 3 / 4 drawn 0.1 / 0.1 / 0.1 / 0.7 per row (a rollout's own mix), slot tokens of m < 4 recomputed with the serving functions |
| ego poses | local -> rig trajectory, one pose per decision | **used**: 4 history poses in the rig frame of t0; missing ones = the oldest state run backwards at constant body velocity and yaw rate | same rule on the logged poses |
| linear velocity | k = 0: the recorded nuPlan (vx, vy), delivered rotated by -yaw (0.004 m/s from the log after rotating back; 5.2 m/s before); k >= 1: vehicle-model state, vy = 0 (0.026 at k = 1, 0.004 after) | **used as delivered**; start states rotated back (pose-difference test; below 1 m/s the source's rule) | m = 1: recorded (vx, vy); m >= 2: vy = 0 |
| linear acceleration | k = 0: recorded (0.027 m/s^2 from the log); k = 1: ax = d speed / dt of the track (0.055), ay = vx x yaw rate (0.042); k >= 2: ax = d vx / dt of the vehicle model (r 0.97 with the central difference of the fed vx), ay = 0 (0.007) | **used as delivered** | ay: recorded / vx x yaw rate / 0 for m = 1 / 2 / >= 3. **Substituted: ax = the recorded one** at every m (0.11 m/s^2 from the track's d speed / dt; in closed loop ax is the consequence of the model's own previous plan and cannot be built from logs) |
| angular velocity | yaw rate; at k = 0 / 1 within 0.003 / 0.001 rad/s of the log | **used** only to back-extrapolate the missing history poses | spline derivative of the logged yaw |
| angular acceleration | vehicle model | not used (no such input in the adapter) | - |
| route | built once per rollout from the scene's own 5.5 s recorded track projected onto lane centres, extended past the recording along the lane graph (successor with the closest heading); per decision 20 waypoints 4.21 m apart from the ego's projection, the first 40 m dropped: always 10 valid waypoints (40-80 m) + 10 NaN. **Beyond the recording it is the lane's continuation, not where the log went** | **used** through the shipped samples' 4-way rule (first waypoint >= 5 m away: y > 2 m left, < -2 m right). The waypoints themselves as adapter features (arm AR) were tested and not adopted | AlpaSim's own `RouteGeneratorMap` run on the token's track from the NAVSIM logs and the same trajdata map; the token as decision k of the scene that starts k frames before it |
| ego box, rig -> camera | per rollout | camera position: lever arm of the output only | - |
| `time_now` / `time_query` | 17 ms after the frame, + 0.5 s | timestamps of the returned 10 Hz trajectory (the shipped LTF sample's functions) | - |
| map, agents, traffic lights, scene id, recording ground truth | not sent (`send_recording_ground_truth: false`; the scene id is debug info) | none used (scene id only in the driver log) | none |

Route rebuild against the real thing (`ap2_route.py check`, 48 scenes x 10 messages): at the tapped pose the rebuilt waypoints are 0.3 mm from the tapped ones
on average (max 1.4 cm), same number of waypoints and same command in 480 / 480; from the log pose of decisions 0 and 1 (what a training row uses) 5 cm (max 0.42 m),
command 96 / 96. From decision 2 on the car is where the driver took it: the command at the log pose agrees with the tapped one in 83-96 % (LTF's own drift).
The route rule is not NAVSIM's `driving_command`: on navtest they agree on 69.8 % of tokens (8 411 / 12 043; 3 394 of the 8 025 NAVSIM-straight tokens are left or
right by the route rule), on navtrain 73 %; the route could be rebuilt for 98 % of navtrain rows (AlpaSim itself fails 10 of 1 485 public scenes in this step).

## Design choices and what decided them (pilot: navtrain shards 2-4, 25 k rows, 3 000 steps x 64, seed 0 = the SHP-F-s0 recipe)

| arm | training inputs | what it answers |
|:--|:--|:--|
| SHP-F-s0 (stored), SH30-F-s0 (stored, full scale) | NAVSIM-aligned; read as the SH30 driver feeds them: back-warped history when m < 4 | the baseline "served with patches" |
| N0 | the SHP recipe through `ap2_train.py --std navsim` | trainer check: dev ADE 0.683 / 0.649 / 0.632 at 1 000 / 2 000 / 3 000 steps, identical to pp_train's SHP-F-s0 log; navtest plans equal SHP's (ADE difference 0.000) |
| A | AlpaSim inputs, cold-start rule `zero` (only the real slots, the rest invalid) | aligned inputs with no fabricated frames |
| AR | A + the 20 route waypoints as 60 adapter ego features | is a richer route encoding worth it |
| AB | AlpaSim inputs, cold-start rule `backwarp` (the missing slots = the first keyframe re-projected to back-extrapolated poses), trained | which cold-start rule |

Offline read on navtest (training never sees navtest): every token planned as decision k = m - 1 of an AlpaSim rollout with the inputs AlpaSim would deliver.
ADE of the 8 poses to the logged future on 11 972 tokens (those with a logged future and a rebuilt route for every m); EPDMS on a 2 035-token subset through
`python -m jevdrive.bench score-poses --traffic non_reactive` (no EC). Paired differences vs SHP at the same m, bootstrap over 136 logs.
Full table: [ap2/offline_pilot.md](ap2/offline_pilot.md).

| arm | m = 4 ADE (m) | m = 3 | m = 2 | m = 1 | m = 4 EPDMS | m = 4 EPDMS vs SHP |
|:--|--:|--:|--:|--:|--:|:--|
| SHP as served (AlpaSim inputs) | 0.634 | 0.655 | 0.738 | 1.024 | 89.26 | |
| SHP, NAVSIM inputs (its training standard) | 0.630 | | | | 89.25 | |
| N0 = SHP under rule `zero`, untrained for it | 0.634 | 0.658 | 0.874 | 5.010 | | |
| A | 0.649 (+0.016 [+0.011, +0.020]) | 0.649 | 0.687 (-0.051 [-0.067, -0.034]) | 0.943 (-0.080 [-0.130, -0.035]) | 89.49 | +0.23 [-0.37, +0.79] |
| AR | 0.663 (+0.029 [+0.022, +0.036]) | 0.664 | 0.717 | 1.094 (+0.071 [+0.028, +0.110]) | 89.60 | +0.34 [-0.42, +1.01] |
| **AB** | **0.634 (+0.001 [-0.003, +0.004])** | **0.640 (-0.015 [-0.020, -0.010])** | **0.662 (-0.076 [-0.086, -0.065])** | **0.749 (-0.274 [-0.306, -0.247])** | 89.41 | +0.15 [-0.36, +0.65] |

Pre-registered rules, applied:

- **G1, the aligned inputs are learned: pass.** A vs SHP at m = 1 -0.080 m (CI excludes 0), m = 2 -0.051, m = 3 -0.006.
- **G2, nothing regresses at m = 4: pass.** A +0.016 m ADE (line +0.02), EPDMS +0.23 (line -0.3); AB +0.001 m, +0.15.
- **Route encoding: the 4-way command stays.** AR is worse than A in ADE at every m (m = 4 +0.013 [+0.008, +0.018] vs A) and its EPDMS is within noise of A's
  (+0.11, both CIs vs SHP include 0); the rule asked for +0.3 with the CI above 0. The waypoints 40-80 m ahead are mostly the lane's continuation.
- **Cold-start rule: `backwarp`, trained (AB).** AB vs A at m = 1 -0.194 m [-0.219, -0.167] (line -0.05), and better at m = 2 / 3 / 4 as well
  (-0.025 / -0.009 / -0.015). One real frame with seven invalid slots is far from anything the frozen temporal policy has seen (untrained: 5.0 m), and 10 % of
  the rows do not teach it as well as fabricated-but-familiar slots do; the back-warped frames carry no information the ego state lacks, only its format.
- What the alignment changes and what it does not: with full history the AlpaSim command and ego definitions cost the NAVSIM-trained model little
  (SHP 0.630 -> 0.634 m, EPDMS 89.25 -> 89.26; SH30 0.570 -> 0.585 m, 90.43 -> 90.14), although the route rule disagrees with NAVSIM's command on 30 % of
  tokens. The measurable gap is the cold start (decisions 0-2 of 10): SHP as served plans 0.80 m short at 4 s with one keyframe (SH30: 1.10 m), AB 0.38 m.

@@FULL@@

@@LOOP@@
