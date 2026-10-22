# P2 results: HUGSIM capability profile of third-party drivers (PD4, scope PD7)

HUGSIM 64 scenarios (45 scenes; nuScenes / KITTI-360 / PandaSet / Waymo), preset `exam` (PR #57 controller tree), bench commit
950003af+, read 2026-10-22/23. Box outputs: `$DATA_DIR/runs/wm_surplus/p2_match/` and `$DATA_DIR/runs/bench/hugsim/*_exam-r{p2,p2b}`.
Pre-registration: [../../plans/2026-10-22-p2-match-prereg.md](../../plans/2026-10-22-p2-match-prereg.md) (its cross-model read-out
lines were superseded by PD7 before any of them was used: P2 became a capability profile; our checkpoints are not a method test here).

## Files

| file | what |
|:--|:--|
| [profile/hugsim_exam_profile.md](profile/hugsim_exam_profile.md) | **the profile**: WA-JEPA and shipped openpilot Cinque (`P0`, a non-foundation third-party reference) per manoeuvre class: HD, end shares with CIs, turn passage, example scenario ids |
| [profile/hugsim_exam_arms.md](profile/hugsim_exam_arms.md) | the same two drivers, whole board (HD and parts, end counts) |
| [repeat/](repeat/) | WA-JEPA read twice (`p2`, `p2b`): run-to-run noise and the profile of both repeats |
| [profile/simscale/](profile/simscale/) | SimScale LTF and SimScale DiffusionDrive (camera-only non-foundation references, PD2) beside WA-JEPA and `P0`: profile, whole board, match tables (repeat `p2`); [profile/simscale_p2b/](profile/simscale_p2b/) the second read |
| [simscale_ltf/](simscale_ltf/), [simscale_dd/](simscale_dd/) | per-scenario units of each SimScale model, both repeats (`units_p2.csv`, `units_p2b.csv`) |
| [reference_ours/](reference_ours/) | our checkpoints' reads, recorded as reference only: **input-mismatched, not a method test** (PD7; empty route memory, command only, a rig and renderer we never trained on). Not to be quoted as a comparison |

## What the profile says (WA-JEPA, repeat p2; p2b in brackets where it differs)

How classes are made: the scenario's recorded route only (HUGSIM scenes carry no lane map): turn = route heading range >= 30 deg
(left / right by the net heading change; sharpness 30-60 / 60-100 / > 100), bend 10-30 deg, lateral shift = < 10 deg with the end
>= 2 m to the side (the lane-change proxy; with no lane geometry a shift is not proven to be a lane change), straight otherwise.
CIs: 95 %, bootstrap clustered by scene. "Turn passage" = where the run's progress (RC x route length) ended relative to the arc
holding 10-90 % of the turn's heading change.

| class | scenarios / scenes | HD [CI] | complete | fg collision | off route | turn passage before / inside / through |
|:--|--:|--:|--:|--:|--:|--:|
| straight | 18 / 11 | 0.617 [0.423, 0.815] | 11 (0.61) | 7 (0.39) | 0 | |
| lateral shift (lane-change proxy) | 12 / 9 | 0.388 [0.227, 0.594] | 7 (0.58) | 4 (0.33) | 0 | |
| bend 10-30 deg | 11 / 8 | 0.366 [0.124, 0.670] | 4 (0.36) | 6 (0.55) | 0 | |
| turn left | 16 / 12 | 0.560 [0.369, 0.755] (0.505) | 8 (0.50) | 4 (0.25) | 1 | 3 / 3 / 10 (4 / 3 / 9) |
| turn right | 7 / 5 | 0.316 [0.113, 0.637] | 1 (0.14) | 3 (0.43) | 3 (0.43) | 1 / 4 / 2 |
| turn > 100 deg | 8 / 5 | 0.367 [0.168, 0.620] | 2 (0.25) | 2 (0.25) | 3 (0.38) | 3 / 3 / 2 |
| all | 64 / 45 | 0.484 [0.382, 0.591] (0.458) | 31 (0.48) | 24 (0.38) | 4 | 4 / 7 / 12 |

- WA-JEPA, a grafted foundation world model with the published HUGSIM client, completes under half of the scenarios. Foreground
  collisions (with other vehicles) are the most common end in every class (24 / 64). Right turns and sharp turns are where it
  fails to drive the route: 1 of 7 right-turn scenarios complete, 3 leave the route; of 8 turns over 100 deg only 2 get through the
  turn. Left turns get through in 10 of 16. Classes rest on 5-12 scenes each, so the CIs are wide.
  A side-by-side of turn passage against straight-line driving is the C1 evidence; it is a profile, not a ranking.
- The non-foundation reference (shipped Cinque, no adapter, same harness) completes 8 / 64 (0 of 23 turns; 4 of 23 through the
  turn) and is stuck or spinning in 23. It is a far weaker driver here (HD 0.271).
- Run-to-run noise: WA-JEPA p2 vs p2b +0.026 [+0.003, +0.056] HD on the same 64 scenarios (the simulator is not bit-deterministic);
  the per-class profile is stable to one or two scenarios per cell.
- Example scenario ids per class x end (for a later video page) are in the `examples` column of the profile table.

## Camera-only non-foundation references: SimScale LTF and DiffusionDrive (2026-10-23)

**Adapter** (`experiments/hugsim/scripts/simscale_hugsim_client.py`, launcher `simscale_e2e.sh`, bench drivers `SimScale-LTF` /
`SimScale-DD`, env `envs/alpasim-ltf`). The SimScale models (AlpaSim e2e_challenge samples: `ltf_sim_navtest.ckpt`,
`diffusiondrive_sim_navhard.ckpt`) run through their shipped `policy.py` (`LtfPolicy` / `DiffusionDrivePolicy.predict_batch`, fp16
autocast, warm-up), no model code touched. The client follows HUGSIM's own LTF client (`hugsim_deps/NAVSIM/ltf_e2e.py` +
`hugsim/dataparser.py`):

- Cameras: HUGSIM's CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT -> CAM_L0 / F0 / R0 (the shipped dataparser's map), each 800 x 450 RGB
  resized to 1920 x 1080 (bilinear, as shipped); the policy's own preprocessing then crops L0 / R0 to their central 1088 columns and
  28 rows top / bottom and stitches a 2048 x 512 panorama. HUGSIM's front cameras are ~65 deg wide (nuScenes, KITTI-360) or ~58 deg
  (PandaSet, Waymo) with the side cameras at about +-55 deg, close to the nuPlan rig; current frame only (both models take no history).
- Command (the fix): HUGSIM's per-step `info["command"]` (0 = R / 1 = L / 2 = S) -> NAVSIM one-hot (0 = L / 1 = S / 2 = R) through
  [2, 0, 1], WA-JEPA's map. The shipped dataparser hard-codes straight.
- Ego status: velocity (v, 0) and acceleration (a, 0) in the ego frame, as WA-JEPA's client and AlpaSim's driver. Deviation from
  the shipped dataparser, which rotates both by the steering angle as if it were a yaw.
- DiffusionDrive noise: seed 0, noise index = step (AlpaSim driver pattern), so each scenario's noise is fixed.
- Output (x, y)_lidar = (-y, x)_ego, 8 poses at 0.5 s; a failed step sends a braking plan and is counted (0 failures in all 256
  episodes). Smoke (`scene-0041-medium-00`, a left turn): the command reaches the model (L steps move the plan left), 57 / 85 ms per
  step, 0.4 GB torch peak per client.

**Read**: HUGSIM 64, preset `exam`, two repeats (`p2`, `p2b`, as WA-JEPA), same bench path and controller as WA-JEPA and `P0`.

| class | scenarios | WA-JEPA HD / complete | SimScale-LTF HD [CI] | LTF complete / fg coll / stuck + spin | SimScale-DD HD [CI] (p2b) | DD complete / fg coll / stuck + spin (p2b) |
|:--|--:|--:|--:|--:|--:|--:|
| straight | 18 | 0.617 / 11 | 0.479 [0.332, 0.632] | 7 / 8 / 0 | 0.436 [0.298, 0.576] (0.499) | 6 / 6 / 5 (9 / 6 / 3) |
| lateral shift | 12 | 0.388 / 7 | 0.354 [0.160, 0.625] | 5 / 6 / 0 | 0.375 [0.231, 0.574] (0.361) | 6 / 4 / 1 (6 / 4 / 2) |
| bend 10-30 deg | 11 | 0.366 / 4 | 0.199 [0.017, 0.503] | 2 / 9 / 0 | 0.126 [0.021, 0.318] (0.120) | 1 / 8 / 2 |
| turn left | 16 | 0.560 / 8 | 0.287 [0.124, 0.491] | 2 / 7 / 3 | 0.263 [0.122, 0.423] (0.217) | 1 / 6 / 7 (0 / 7 / 9) |
| turn right | 7 | 0.316 / 1 | 0.319 [0.074, 0.637] | 1 / 3 / 2 | 0.242 [0.058, 0.555] (0.232) | 1 / 4 / 2 |
| turn > 100 deg | 8 | 0.367 / 2 | 0.186 [0.020, 0.381] | 0 / 4 / 3 | 0.183 [0.094, 0.293] (0.195) | 0 / 3 / 4 (0 / 3 / 5) |
| all | 64 | 0.484 [0.382, 0.591] / 31 | 0.342 [0.251, 0.441] | 17 / 33 / 5 | 0.307 [0.228, 0.390] (0.308) | 15 / 28 / 17 (17 / 29 / 18) |

Turn passage (23 turns, before / inside / through): WA-JEPA 4 / 7 / 12, LTF 6 / 9 / 8, DD 8 / 7 / 8 (p2b 9 / 8 / 6), P0 8 / 11 / 4.
Ends over all 64 (complete / fg / bg / off route / stuck / spin): LTF 17 / 33 / 7 / 2 / 4 / 1; DD 15 / 28 / 4 / 0 / 10 / 7 (p2b
17 / 29 / 0 / 0 / 13 / 5); WA-JEPA 31 / 24 / 3 / 4 / 0 / 2. CIs 95 %, bootstrap clustered by scene. Full tables with every end share,
sharpness rows and example ids: [profile/simscale/hugsim_exam_profile.md](profile/simscale/hugsim_exam_profile.md).

- Both camera-only non-foundation NAVSIM models complete about a quarter of the board (17 and 15 / 64), between shipped Cinque
  (8) and WA-JEPA (31). Foreground collisions are their main end (33, 28); DiffusionDrive also stands still or spins in 17
  (stuck = the 400-step end; in the smoke scenario it stopped after ~50 steps and never moved again).
- Turns are where they lose most: left turns complete 2 / 16 and 1 / 16 (WA-JEPA 8 / 16), turns over 100 deg 0 / 8 for both, and
  8 of 23 turns get through for each (WA-JEPA 12). Right turns are poor for all three (1 / 7 complete each), so WA-JEPA's right-turn
  failure is not specific to it. Straight and lateral-shift scenarios are closer (LTF 7 / 18 and 5 / 12; DD 6-9 / 18 and 6 / 12).
- Recorded pairs (match.md, kept for later, not read as a ranking under PD7): HD SimScale-LTF - WA-JEPA -0.142 [-0.215, -0.070],
  SimScale-DD - WA-JEPA -0.177 [-0.249, -0.111] (percentile bootstrap over scenarios, repeat p2). Inputs differ: both SimScale
  models see one frame of three cameras; WA-JEPA four cameras x four frames and pose history.
- Run-to-run noise (p2b - p2, scene-clustered): LTF -0.002 [-0.005, +0.002], 64 / 64 same end (fp16, single frame: nearly
  deterministic); DD +0.001 [-0.025, +0.033], 54 / 64 same end (its noise is seeded, so the spread is the simulator's
  non-bit-determinism, which this model's closed loop amplifies more than LTF's); DD class cells move by up to 4 scenarios between repeats (turn left stuck + spin 7 -> 9, straight complete 6 -> 9).

## What HUGSIM cannot measure

- **Traffic lights.** HUGSIM's score has NC, DAC, TTC, COM and route completion only (`sim/utils/score_calculator.py`); scenes carry
  no light state and no rule check ("traffic light" exists only as a semantic class of the Gaussian reconstruction). Red-light
  behaviour is not measurable here. Where it can be: AlpaSim (nuPlan and PAI renders keep lamps; the red-line stop windows of
  decisions 380 / 393 in the main log) and NAVSIM navtest / navhard (traffic-light compliance enters EPDMS).
- **Lane changes and junctions** as such: no lane map in HUGSIM scenes (`load_HD_map: false`), so only the route-shape proxies above.

## HUGSIM feasibility per camera-only subject (PD2)

HUGSIM gives every external driver the same thing through its FIFO client protocol (`sim/utils/launch_ad.py`): six nuScenes-rig
cameras (FRONT, FRONT_LEFT / RIGHT, BACK, BACK_LEFT / RIGHT) at 800 x 450 for all four datasets, ego velocity / acceleration /
steer / pose, and a per-step route command (0 = right, 1 = left, 2 = straight; checked on 7 turning and straight scenes: it takes the turn's side around each turn
and straight elsewhere), at 4 Hz.

| subject | model input | what a HUGSIM client needs | verdict |
|:--|:--|:--|:--|
| WA-JEPA | L0 / F0 / R0 / B0 x 4 frames at 2 Hz, ego, command | shipped by its authors (`close_loop/hugsim_planner.py`; command map, 4 Hz -> 2 Hz stride, B0 = rendered back) | runs (this read) |
| SimScale LTF, SimScale DiffusionDrive (camera-only) | L0 / F0 / R0 stitched, command, velocity, acceleration | HUGSIM ships an LTF client for the NAVSIM LTF agent (`third_party/hugsim_deps/NAVSIM/ltf_e2e.py`, env `hugsim-ltf` deleted in a cleanup, recipe `experiments/hugsim/archive/install_ltf.sh`). The SimScale models are not NAVSIM agent classes (standalone `policy.py`), so the client's `agent.compute_trajectory` call becomes `predict_batch`; and the shipped dataparser **hard-codes the command to straight** (`command[1] = 1`, the real one commented out), which a turning profile cannot use: pass `info["command"]` with the 0 = R / 1 = L / 2 = S -> NAVSIM map | **runs** (this read; adapter above, one client file) |
| ReCogDrive-2B-RL | front camera (`cam_type single`), ego status, command | a NAVSIM `AgentInput` built per step (HUGSIM's dataparser already builds one) and a camera-only sensor config; VLM 2B + DiT per step | feasible once its env is built (setup.md: rebuilding on py3.10); small-to-medium adapter |
| AutoVLA | multi-frame camera prompts built by its own nuPlan preprocessing (JSON per token) | the prompt / JSON builder must run online per HUGSIM step | medium; check whether its agent can build the prompt from an `AgentInput` before committing |
| DriveVLA-W0 | Emu3 VQ tokens of the camera history (released pre-tokenized for navtest only) | online VQ tokenisation of HUGSIM frames + its inference loop as a server; 8B, 18 GB weights | not small; not tonight |

## Cost

SimScale LTF + DD 2.07 worker-job hours (24 jobs, two repeats each; peak 36.5 GB per 6-slot job, so 8.0 GB per slot is
declared); WA-JEPA 1.55 worker-job hours (two repeats), third-party anchors (P0, SH30) 1.8, our checkpoints 4.88 (all finished before PD7);
a worker job is one card share (often co-located with the P1 sweep). New disk on the box 5.6 GB after deleting the 38.8 GB of
per-episode `.ply` scene copies (videos kept, ~6 MB per episode); SimScale reads 1.7 GB after the same `.ply` drop.
