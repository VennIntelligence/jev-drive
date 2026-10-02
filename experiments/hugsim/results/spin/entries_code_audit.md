# HUGSIM entries: model output -> executed control, and camera handling

Code-reading audit, 2026-10-03. No GPU, no simulator runs. Every statement is tagged **[V]** (verified: file:line
read at the pinned commit) or **[I]** (inferred: from a paper, from an absence in code, or by arithmetic).
The clones were local scratch copies (not committed); the pinned commits below identify what was read.

## Pinned commits

| Alias | Repository | Commit | Date |
|---|---|---|---|
| HUGSIM | github.com/hyzhou404/HUGSIM (main) | `62c690d39fd90020e68a196bd8bcc1c4d4191f2e` | 2025-11-08 |
| UNIAD | github.com/hyzhou404/UniAD_SIM | `5fb279e39912a5ac7f58e00d56b065cadcd0a749` | 2024-12-19 |
| VAD | github.com/hyzhou404/VAD_SIM | `0c749670c4b0883d3cf5763ee3917703c33d9aa4` | 2024-12-19 |
| LTF | github.com/hyzhou404/NAVSIM | `ca0ca7e4368646d8f7b86fb1fdaa1862c946176f` | 2025-05-29 |
| WAJ | github.com/AFARI-Research/WA-JEPA | `bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad` | 2026-09-04 |
| GARAGE | github.com/valeoai/HUGSIM_garage (= yuan-yin/HUGSIM, same HEAD) main | `f3b65adcc662bd6dd7a91d9037e758b2221bfe5f` | 2026-10-02 |
| GARAGE@drivor | same repo, commit "Code used for DrivoR" | `a998d63` | 2026-05-12 |
| GARAGE@toad | same repo, commit "refactor + run TOAD" | `f54c53e` | 2026-08-26 |
| PR57 | hyzhou404/HUGSIM `refs/pull/57/head` | `ead17f2ad97f71fd21fa6f66237a7c05364ed98e` | 2025-11-12 |
| CHSRV | github.com/hyzhou404/HUGSIM_Local_Server (RealADSim ICCV 2025 local test server) | `afaeb1bff76ca24af9728fb3797df6cb3e659d30` | 2025-06-29 |
| CHDEMO | huggingface.co/XDimLab/ICCV2025-RealADSim-ClosedLoop-SubmissionDemo | `efb8a82215bab318004e7834695566e9a1af36eb` | 2025-08-26 |

Repos cloned and searched (`grep -rli "hugsim\|plan_pipe"`), **no HUGSIM client found**: valeoai/DrivoR `fc6e5aa`,
valeoai/TOAD `cfa88e0`, valeoai/Pictura `4f0aa82` (README only), XiaomiAutoL3/DriveZero `2495954` (README + DriveRL
only; HUGSIM appears only as an acknowledgement link, README.md:44), boschresearch/MemoryDrivoR `f63c2ff`,
NVlabs/GTRS `92a740d`, valeoai/VideoActionModel (VaVAM) `738050e`.

## 0. What the benchmark itself does (all entries inherit this)

- Loop: one plan requested per simulator step; `traj2control` -> `env.step` once; 400-step cap. HUGSIM
  `closed_loop.py:58-69` **[V]**. Step is `dt: 0.25` (`configs/sim/kinematic.yaml`) **[V]**. So every client replans at
  4 Hz and only the first iLQR input is executed (`sim/ilqr/lqr.py:34-36`) **[V]**.
- Plan format: `(N, 2)` array, column 0 = right, column 1 = forward ("lidar coordinates, x to right, y to forward"),
  `sim/utils/sim_utils.py:53-57` **[V]**. Spacing is assumed 0.5 s: iLQR `discretization_time=0.5`
  (`sim/ilqr/lqr.py:5`) and the scorer is told `'timestep': 0.5` (`closed_loop.py:85`) **[V]**. N is free; no resampling.
- Reference built by `traj2control` (`sim/utils/sim_utils.py:58-71`) **[V]**: row 0 = zeros (current pose, heading 0),
  rows 1..N = `(forward, right)`; heading per row = `arctan2(b - prev_b, a - prev_a)` with `(a, b) = (right, forward)`,
  i.e. `arctan2(d_forward, d_right)` (the transposed defect), folded into [-pi/2, pi/2] (lines 62-64). No velocity or
  steering reference (columns 3, 4 stay 0); state cost is `[1, 1, 10, 0, 0]` so heading weighs 10x position and speed
  is not tracked directly (`sim/ilqr/lqr.py:6`) **[V]**. Current state is `[0, 0, 0, ego_velo, ego_steer]` (line 67-69).
- iLQR model: `x += v cos(h) dt; y += v sin(h) dt; h += v tan(steer)/L dt`, wheelbase 2.7
  (`sim/ilqr/lqr_solver.py:505-510`, `lqr.py:17`); inputs clipped to accel +-3.0, steering rate +-0.4, steer +-pi/3
  (`lqr.py:13-15`, `lqr_solver.py:383`) **[V]**.
- Plant: `velo += acc*dt; steer += steer_rate*dt; pos += velo*(sin,cos)(theta)*dt; theta += velo*tan(steer)/L*dt`
  with `L = Lr + Lf = 2.7` (`sim/hugsim_env/envs/hug_sim.py:282-289`) **[V]**. No clipping of acc, steer or speed in
  `step`; `min_acc/max_acc/max_speed` from `kinematic.yaml` are only used for the gym `action_space` declaration
  (`hug_sim.py:100-105`); negative speed is possible **[V by reading step(); no clip call in the file]**.
- Origin: the ego pose *is* the front-camera pose. Cameras are rendered at `ego @ v2front @ inv(v2c) @ cam_rect`
  (`hug_sim.py:215-219`), so `CAM_FRONT` sits at the ego origin (plus `cam_rect`) and the bicycle model's pivot is the
  camera, not a rear axle **[V]**. The nuScenes camera config has `CAM_FRONT v2c_trans z = -1.73` (camera 1.73 m ahead
  of the vehicle frame, `configs/sim/nuscenes_camera.yaml`) **[V]**; no client compensates for it (see rows).
- Cameras rendered: six nuScenes-layout cameras at 800x450 for every dataset; on Waymo / KITTI-360 the three BACK
  cameras are returned as all-zero images (`hug_sim.py:226-229`) **[V]**. Per-dataset rig differences that no client
  reads: front `fovx` 65.1 deg (nuScenes, KITTI-360) vs 58.1 deg (Waymo, PandaSet); `cam_rect` = +0.3 m (nuScenes,
  Waymo), +0.7 m (PandaSet), 5 deg pitch rotation (KITTI-360) (`configs/sim/*_camera.yaml:96-101,152-160`) **[V]**.
- Command: `info['command']` = command of the nearest recorded pose (`hug_sim.py:243-246,255`); encoding
  0 right / 1 left / 2 straight (UNIAD `tools/closeloop/dataparser.py:105` comment) **[V]**.
- Failure: client sends `None` -> episode ends (`closed_loop.py:71-72`), but line 74 then indexes `plan_traj[:, [1, 0]]`
  on `None` and raises, so no `eval.json` is written for that scenario **[V by reading; crash itself inferred]**.
- Paper (arXiv 2412.01718, Sec. 4 "Controller", Sec. 6.2, App. E.5 / Table 13): the simulator "expects feedback in the
  form of either planned waypoints or a sequence of control commands"; waypoints are converted with "a Linear
  Quadratic Regulator (LQR) control"; ego follows a discrete kinematic bicycle model (Eq. 18); baselines are evaluation
  APIs that "send back the planned future waypoints". LTF uses the front three cameras, UniAD / VAD six. The paper
  gives no waypoint spacing, origin, or heading convention **[V in paper text]**.
- Controller history **[V via `git show` in GARAGE clone]**: init `93e0e06` (2024-12) used
  `rot = np.arctan((b - prev_b)/(a - prev_a))` with `prev_a, prev_b` never updated (heading = atan(forward/right) of
  the chord from the origin). PR #56 (merged 2025-11-08, = `62c690d`) switched to `arctan2` with updated `prev`, and
  changed the scorer `timestep` back from 0.25 to 0.5. PR #57 (`ead17f2`, open) swaps the unpacking to
  `for i, (b, a) in enumerate(plan_traj)`, giving `arctan2(d_right, d_forward)`. Table 13 of the paper predates all of
  this, so it was produced with the `arctan`-from-origin version **[I]**.

## 1. Table

Columns: (a) frame / conversion, (b) waypoints / horizon, (c) smoothing, (d) heading / controller patch, (e) speed,
(f) replanning / history, (g) fallback, (h) cameras / preprocessing, (i) command.

| Entry | (a) frame and conversion | (b) waypoints, horizon | (c) smoothing / filtering | (d) heading, traj2control patch | (e) speed handling | (f) replan rate, history | (g) fallback | (h) cameras, preprocessing | (i) command to model |
|---|---|---|---|---|---|---|---|---|---|
| **HUGSIM mainline** (benchmark glue) | Expects (right, forward) at ego = front-camera origin; swaps to (forward, right) for iLQR. HUGSIM `sim_utils.py:58-59` **[V]** | Any N at assumed 0.5 s; `lqr.py:5`, `closed_loop.py:85` **[V]** | None **[V]** | `arctan2(d_fwd, d_right)` (transposed) + fold to +-pi/2, `sim_utils.py:61-66` **[V]**; unpatched | No speed reference; speed only implied by waypoint spacing; iLQR accel clip +-3, `lqr.py:13` **[V]**; plant does not clip, `hug_sim.py:283` **[V]** | Plan every 0.25 s step, first input only, `closed_loop.py:58-67`, `lqr.py:34-36` **[V]** | `None` ends episode then crashes at `closed_loop.py:74` **[V]** | Renders 6 cams 800x450; BACK cams zeroed on Waymo/KITTI-360, `hug_sim.py:226-229` **[V]** | Provides `info['command']`, `hug_sim.py:246,255` **[V]** |
| **UniAD** (UniAD_SIM) | Sends `sdc_traj[0]` untouched, UNIAD `tools/closeloop/e2e.py:271-273` **[V]**; nuScenes lidar frame = x right, y forward **[I]**; no origin shift **[V, absence]** | 6 waypoints (`planning_steps = 6`, `projects/configs/stage2_e2e/base_e2e.py:56`) = 3 s at 0.5 s **[V/I spacing]** | In-model only: occupancy-based collision optimization at test time (`use_col_optim = True`, `base_e2e.py:57`; `planning_head.py:193-195`) **[V]**; nothing in the client | None in client; relies on upstream `traj2control` **[V, absence]** | None in client. Ego speed fed via `can_bus[13:16] = [v, 0, 0]`, steer rate in `can_bus[10:13]`, accel slots left zero, `dataparser.py:70-82` **[V]** | Every step (4 Hz). Temporal state = model's own prev-BEV / track queries fed at 4 Hz with fixed `scene_token '062'`, `dataparser.py:92` **[V]**; model was trained on 2 Hz keyframes **[I]** | `except RuntimeError -> results = None`, but `results[0]` is indexed at `e2e.py:255` before the `None` check at 270, so the client crashes instead of sending `None` **[V]** | 6 cams `e2e.py:210-211`; plain `cv2.resize` 800x450 -> 1600x900, mean-subtract, no crop, no pad; `dataparser.py:36-37` **[V]**. Intrinsics rebuilt from fov and scaled x2, `lidar2img = K*l2c`, `dataparser.py:42-50` **[V]**. No yaw/pitch/height compensation; zeroed BACK cams on Waymo/KITTI-360 go in as black **[V, absence + env]** | Yes, raw HUGSIM value, `dataparser.py:105` -> `navi_embed[command]`, `planning_head.py:166` **[V]** |
| **VAD** (VAD_SIM) | `ego_fut_preds[cmd]` are per-step deltas, `np.cumsum` -> positions, sent untouched, VAD `tools/closeloop/e2e.py:252-255,263-264` **[V]**; lidar frame x right, y forward **[I]**; no origin shift | 6 waypoints (`valid_fut_ts=6`, `projects/configs/VAD/VAD_base_e2e.py:80`), 3 s at 0.5 s **[V/I spacing]** | None **[V]** | None in client **[V, absence]** | None in client; speed in `can_bus[13:16]`, `dataparser.py:86-88` **[V]**; config has `ego_his_encoder=None`, `ego_lcf_feat_idx=None` (`VAD_base_e2e.py:78-79`) so ego state does not enter the planner head **[V]** | Every step (4 Hz); `ego_his_trajs` from a 3-pose queue of consecutive 0.25 s frames, `dataparser.py:67-76` **[V]** (unused because `ego_his_encoder=None`); prev-BEV at 4 Hz, `scene_token '062'` `dataparser.py:97` **[V]** | Same bug class: `results[0]` indexed at `e2e.py:252` before the `None` check at 262 **[V]** | 6 cams `e2e.py:209-210`; plain resize to 1600x900, `dataparser.py:39`; K x2, `dataparser.py:44-52` **[V]**. Config's test pipeline scales by 0.8 (`VAD_base_e2e.py:348`) but the client bypasses the pipeline, so the model sees 1600x900 instead of 1280x720 **[V config / I effect]** | Yes: one-hot at `info['command']`, `dataparser.py:104-105,110`; mode picked by argmax of that one-hot, `e2e.py:253-254` **[V]** |
| **LTF** (hyzhou404/NAVSIM) | Model emits NAVSIM poses (x fwd, y left, heading) at the rear axle; client does `[y, x]`, negates col 0 -> (right, forward); LTF `ltf_e2e.py:66-68,73` **[V]**. No rear-axle -> camera shift **[V, absence]** | 8 poses, 4 s at 0.5 s (`TrajectorySampling(time_horizon=4, interval_length=0.5)`, `navsim/agents/transfuser/transfuser_config.py:14`) **[V]** | None **[V]** | Model heading (col 2) is dropped (`way_points[:, :2]`, `ltf_e2e.py:73`); heading recomputed by upstream `traj2control`; no patch **[V]** | None on output. Input velocity / accel are decomposed with `yaw = -info['ego_steer']` (steering angle, not heading; the `ego_rot` line is commented out), `hugsim/dataparser.py:43-50` **[V]** | Every step (4 Hz); single frame, no history (`AgentInput([ego_status],[cameras],[lidar])`, `dataparser.py:63`) **[V]** | `RuntimeError -> traj=None -> send None, exit`, `ltf_e2e.py:59-61,75-80` **[V]**; then upstream crash at `closed_loop.py:74` | 3 cams FRONT / FRONT_RIGHT / FRONT_LEFT -> f0 / r0 / l0, `dataparser.py:10-11,54-55`; each resized 800x450 -> 1920x1080 (`:32`); then NAVSIM stitch: side cams cropped `[28:-28, 416:-416]`, front `[28:-28]`, concat l0-f0-r0, resize to 1024x256, `transfuser_features.py:65-71` **[V]**. So side-camera pixels do enter the single forward-view input, by design. No intrinsics / FOV / pitch / height handling **[V, absence]**; LiDAR = zeros (`dataparser.py:61`) | **No.** `info['command']` read is commented out; `command[1] = 1` always (NAVSIM index 1 = straight), `dataparser.py:36-40` **[V]**; that constant goes into the status feature, `transfuser_features.py:45-51` **[V]** |
| **RealADSim challenge demo** (LTF over HTTP) | Identical conversion, CHDEMO `ltf_e2e.py:58-62` **[V]**; `hugsim/dataparser.py` and `transfuser_features.py` byte-identical to LTF (diffed) **[V]** | 8 x 0.5 s | None | Server-side controller is the **old** `arctan((b-prev_b)/(a-prev_a))` with `prev` never updated, CHSRV `code/sim/utils/sim_utils.py:53-68`; iLQR file identical to mainline **[V]** | as LTF | Every step, CHSRV `code/web_server.py:216-222` **[V]** | `traj is None -> return` (client stops), CHDEMO `ltf_e2e.py:53-55` **[V]** | as LTF | No (same fixed straight) **[V]** |
| **WA-JEPA** | Model emits (x fwd, y left, yaw); client sends `(-y, x)`, WAJ `close_loop/hugsim_planner.py:273-276` **[V]**. History poses taken from `info['ego_box'][0,1,6]` = camera-origin pose, `hugsim_planner.py:197-200` **[V]**; no rear-axle shift **[V, absence]** | 8 waypoints at 2 Hz = 4 s (`trajectory_horizon: 8`, `trajectory_frame_rate_hz: 2.0`, `configs/wa_jepa_hugsim.yaml:61,70`) **[V]** | None; raw model output **[V]** | **Patches** `traj2control` in-process: `rot = np.arctan2(a - prev_a, b - prev_b)` with `(a, b) = (right, forward)`, `close_loop/run_fixed_controller.py:77-92`; default entry of the benchmark script, `scripts/evaluation/run_hugsim_benchmark.sh:125,135-139` **[V]**. Numerically equivalent to PR #57 per its own docstring (`run_fixed_controller.py:30-46`) **[V text / I equivalence]**. iLQR 0.5 s discretization untouched **[V, absence]** | None on output. Input: `vx = ego_velo`, `ax = info['accelerate']` (the last commanded accel), `vy = ay = 0`, `hugsim_planner.py:215-220` **[V]** | Every step (4 Hz). 4 history frames strided by 2 sim steps = 0.5 s spacing, 1.5 s history; warm-up repeats the oldest frame and counts it; `hugsim_planner.py:145,170,230-232,253-262`, `wa_jepa_hugsim.yaml:41,87` **[V]** | On any exception / wrong shape / NaN: send all-zero `(8, 2)` plan ("stay put" -> braking), count it in `planner_stats.json`, `close_loop/hugsim_client.py:193-207` **[V]**; after a fatal model-load error, zeros until the episode ends, `hugsim_client.py:78-95,157-164` **[V]**. Never sends `None` | 4 cams: FRONT_LEFT, FRONT, FRONT_RIGHT, BACK -> l0, f0, r0, b0, `wa_jepa_hugsim.yaml:35,79-83` **[V]**. Each `cv2.resize` (INTER_AREA) 800x450 -> 512x256, scale to [-1, 1], no crop, `hugsim_planner.py:192-194`, `wa_jepa_hugsim.yaml:36` **[V]**. Views are separate tokens with a per-slot camera embedding, no stitching, so no side content in the front view **[V config comment :75-77]**. No intrinsics / FOV / pitch / height handling; `b0` is a black image on Waymo / KITTI-360 **[V, absence + env]** | Yes, remapped HUGSIM (0 R, 1 L, 2 S) -> NAVSIM (0 L, 1 S, 2 R) via `command_map: [2, 0, 1]`, `wa_jepa_hugsim.yaml:93`, `hugsim_planner.py:208-216` **[V]** |
| **DrivoR** (paper Table 2, 345 scenarios) | Client (`dynamo_e2e.sh` in a NAVSIM checkout) is **not public**; only the launcher path is visible, GARAGE@drivor `configs/sim/nuscenes_base.yaml` (`dynamo_path`) and `closed_loop_new.py:155-156` **[V]**. Conversion **unverified** | NAVSIM-style 8 x 0.5 s **[I]** | None in simulator side **[V]**; client unknown | Simulator = PR #57 head plus logging/config only: `git diff ead17f2 a998d63` touches no file under `sim/utils/sim_utils.py`, `sim/ilqr/`, `hug_sim.py` **[V]**. So heading = `arctan2(d_right, d_fwd)`, iLQR still 0.5 s. Paper: "incorrect heading computation ... After applying the necessary fixes, we reproduced all scores" (arXiv 2601.05083 Sec. 4) **[V paper]** | unknown | Every step **[V, same loop]**; history unknown | unknown | unknown in code; WA-JEPA App. A says non-LTF methods use four cameras **[I]** | unknown |
| **TOAD / GTRS / ZTRS / Pictura on HUGSIM_garage** (valeoai) | Clients not public (launchers `HUGSIM_AD_*`, GARAGE `configs/sim/nuscenes_base.yaml`) **[V]**. New optional path: client may send a dict with `pose_delta / velo / steer / acc` and bypass iLQR entirely, GARAGE `closed_loop.py:76-82,105-112` **[V]** | Plan spacing constant `PLAN_TIMESTEP = 0.5`, `closed_loop.py:22` **[V]** | **Resampling added in the simulator**: reference linearly interpolated (`np.interp`) from 0.5 s onto the 0.25 s sim step, same horizon, GARAGE `sim/utils/sim_utils.py:71-82` **[V]** (introduced at `f54c53e`) | Heading from consecutive resampled points, `arctan2(right - prev_right, fwd - prev_fwd)`, `sim_utils.py:86-93`; iLQR solver rebuilt with `discretization_time = sim_dt = 0.25`, `sim_utils.py:97`, `closed_loop.py:114-116` **[V]** | none | Every step | Plan `None` handled before scoring (`closed_loop.py:84-130`) **[V]** | unknown (clients private) | unknown |
| **DriveZero / DriveZero-Scale** (arXiv 2609.06055, 46.6 HD) | No HUGSIM code in XiaomiAutoL3/DriveZero `2495954` **[V]**; nothing verifiable | - | - | Paper Sec. 3.2 / Table 6 does not name a controller, commit or scenario count **[V paper text]** | - | - | - | - | - |
| **Latent-WAM** (arXiv 2603.24581, 28.9 HD) | No code link in paper; no official repo found **[V search]** | 4 s horizon (paper Sec. 3.4) **[V paper]** | - | Not stated | - | - | - | - | - |
| **MM-Future** (arXiv 2609.20377, 32.3 HD) | No code link found **[V search]** | - | - | Not stated | - | - | - | - | - |
| **ECO + VaVAM** (arXiv 2609.31383; RealADSim 2025 Track 2 winner "UT/NV") | Code "coming soon" on project page **[V page]**. Method is itself a plan post-processor between policy and controller | N waypoints at policy dt **[V paper Sec. III]** | **Yes (paper)**: optimization with smoothness / sharp-turn / proximity terms, K = 2 past poses anchored, last waypoint pinned, first waypoint free (paper Fig. 2, Table I) **[V paper, not code]** | "HUGSIM ... default controllers" (paper Sec. V-A); challenge server controller was the old `arctan` version (CHSRV) **[I link]** | Not stated | "Applies the first command ... One step, then re-query" (Fig. 2) **[V paper]** | - | - | "Image + command tokens" (Fig. 2) **[V paper]** |
| **MemoryDrivoR** (arXiv 2608.31029, HUGSIM-nuScenes only) | Paper says "test protocol used by DrivoR", "HUGSIM requests a new plan every 0.25 seconds" (App. A.3) **[V paper]**; no HUGSIM code in boschresearch/MemoryDrivoR `f63c2ff` **[V]** | - | - | Not stated | - | 0.25 s | - | - | - |

## 2. Per-entry notes

**HUGSIM mainline.** Three controller generations exist and published numbers mix them: (1) `arctan` of the chord
from the origin (Dec 2024 - Nov 2025; also the challenge server), (2) `62c690d` `arctan2(d_fwd, d_right)` on
consecutive points, (3) PR #57 `arctan2(d_right, d_fwd)`. A fourth, the garage fork, adds 0.25 s resampling. The
iLQR weights heading 10x over position and tracks no speed, so heading errors dominate the steering command. The ego
origin is the front camera and the bicycle pivots there with L = 2.7; none of the clients shifts a rear-axle plan
forward. The plant does not clip acceleration or speed; only the iLQR input clip (+-3 m/s^2, +-0.4 rad/s) bounds it.
`info['accelerate']` is the last commanded (clipped) accel, which several clients feed back as ego acceleration.

**UniAD_SIM.** Pure pass-through of `sdc_traj` (6 x 0.5 s). The only trajectory shaping is UniAD's own
occupancy-based collision optimizer inside the planning head. The client runs the model at 4 Hz with one fixed
scene token, so BEV temporal fusion sees 0.25 s steps while the net was trained on 0.5 s keyframes (inferred, not
measured). Six cameras with black BACK images on Waymo / KITTI-360. The exception path is broken (`results[0]` on
`None`), so an inference error kills the client rather than sending a stop. The WA-JEPA paper says UniAD used four
cameras in its re-run; this fork uses six, so that re-run configuration is not this code.

**VAD_SIM.** Same structure; cumsum of per-step deltas for the commanded mode. Images go in at 1600x900 although the
shipped config's test pipeline uses scale 0.8; `lidar2img` is consistent with 1600x900, so geometry is right but the
resolution differs from training (inferred effect). Ego history is computed but unused (`ego_his_encoder=None`).

**LTF (hyzhou404/NAVSIM).** The reference NAVSIM-family client and the template other NAVSIM models copy. Three
things in it are not what a careful adapter would do: the driving command is hard-coded to "straight"; velocity and
acceleration are rotated by the *steering angle* instead of being expressed in the ego frame; and nuScenes-layout
800x450 renders are stretched to 1920x1080 and pushed through the nuPlan crop/stitch with no FOV or extrinsic
matching (HUGSIM front fovx is 65.1 or 58.1 deg depending on the dataset, side cameras are yawed ~55 deg). The stitched
1024x256 input therefore contains side-camera content by design. Model heading is discarded; HUGSIM recomputes it.

**RealADSim 2025 challenge.** The official submission demo is the same LTF client over HTTP. The local test server
that participants were given ships the oldest controller (`arctan`, `prev` never updated). Under that controller
`a - prev_a = right` sits in the denominator, so a near-straight plan gets a heading near +-90 deg and an all-zero
plan evaluates 0/0 (arithmetic, not run). Whether the hosted leaderboard backend used exactly this file is not
checkable (backend not public).

**WA-JEPA.** The most carefully engineered public client. It keeps the model output raw (no smoothing, no speed
clamp, no resampling) and instead fixes the simulator side: heading patch equivalent to PR #57, plus two crash
patches for background-traffic code (`run_upstream_compat.py:131-168`). History is correctly strided to 0.5 s.
Command is remapped, not passed raw. Failure policy is a zero plan (brake) with per-scenario accounting. Four
separate views at 512x256, plain resize; no calibration handling of any kind, and the back view is black on two of
four datasets. Note the docstring of `hugsim_planner.py:5-7` says `traj2control` "stay[s] byte-identical", which
contradicts the shipped default entry `run_fixed_controller.py`; the benchmark script is authoritative.

**DrivoR and the valeoai garage.** DrivoR's published 345-scenario numbers were run on `a998d63`, which differs from
PR #57 only in logging, config paths and launchers, so they used the heading fix and the unmodified 0.5 s iLQR. The
later `f54c53e` (TOAD era) adds reference resampling to the sim step and a 0.25 s solver; garage `main` adds a scorer
change (ego box height, upstream PR #76) and an escape hatch that lets a client move the ego with its own vehicle
model. Scores from `a998d63`, `f54c53e` and garage `main` are therefore three different protocols. None of the
valeoai AD-side clients (DrivoR "dynamo", GTRS, ZTRS, Pictura, TOAD) is public.

**Papers without code (DriveZero, Latent-WAM, MM-Future, ECO, MemoryDrivoR).** None states the HUGSIM commit or
controller. DriveZero, Latent-WAM and MM-Future tables copy or re-run baselines without saying which. Latent-WAM
calls a 436-scenario set "pre-challenge" while DrivoR's pre-challenge set is 345. ECO is the only entry whose
contribution is the plan-to-controller interface itself (endpoint-pinned smoothing), described in the paper only.

## 3. Cross-entry findings relevant to our own adapter

1. Nobody with public code smooths, resamples or speed-clamps the plan on the client side. All trajectory-side fixes
   that exist are on the simulator side (PR #57 heading; garage resampling to 0.25 s). **[V across rows]**
2. Nobody shifts between rear-axle and front-camera origin. **[V, absence in UNIAD / VAD / LTF / WAJ clients]**
3. Nobody handles intrinsics, FOV, camera pitch (KITTI-360 `cam_rect` 5 deg) or camera height (`cam_rect` 0.3 / 0.7 m)
   for image-only models; images are plain-resized. Only UniAD / VAD use calibration, through `lidar2img`. **[V]**
4. Fallbacks: baselines effectively have none (send `None` -> scenario without `eval.json`, or crash). WA-JEPA is the
   only one with a braking zero-plan. No entry has a low-speed or stuck heuristic. **[V]**
5. Command: UniAD, VAD, WA-JEPA use it; the LTF reference client never does. **[V]**
6. Replanning is always 4 Hz with only the first iLQR input used; only WA-JEPA strides history to the model's 2 Hz.
   **[V]**

## 4. Claims not verifiable in code

- DrivoR's HUGSIM client: frame conversion, cameras, command use, fallback. Client lives in a private NAVSIM checkout
  (`dynamo_e2e.sh`). Only the simulator commit is verifiable.
- TOAD, GTRS, ZTRS, Pictura clients on HUGSIM_garage: same, launchers only.
- DriveZero, Latent-WAM, MM-Future, GigaPixel: which controller version (mainline, PR #57, garage) and which scenario
  set produced their HUGSIM numbers; whether baseline rows were re-run or copied.
- ECO / VaVAM: all implementation details (paper only); which controller the challenge leaderboard backend ran
  (inferred to be the `arctan` version from HUGSIM_Local_Server, backend not public).
- RealADSim Track 2 2nd place (NVIDIA/FDU) and 3rd (BranchOut): no report or code located.
- WA-JEPA paper App. A: "LTF uses three front-view cameras, while the others use four" and "HUGSIM at commit ead17f2".
  The repo patches mainline in-process rather than checking out `ead17f2`; equivalence to PR #57 is asserted in a
  docstring, not re-derived here. The four-camera UniAD / VAD configuration is not in UniAD_SIM / VAD_SIM (six).
- WA-JEPA's reported 0.4462: whether any scenario in it used the zero-plan fallback (no per-scenario logs published).
- HUGSIM paper Table 13: produced with the Dec 2024 `arctan` controller (inferred from repo history; paper is silent).
- That UniAD / VAD outputs are in (x right, y forward): inferred from nuScenes lidar convention and from
  `traj2control`'s docstring, not traced through the model code.
- Effect sizes quoted in upstream PR #57 comments and issue #75 (heading fix worth ~8.5 HD on 258 scenarios;
  resampling worth +4.35 HD on nuScenes) are third-party measurements posted on GitHub, not checked here.
