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
| SimScale LTF, SimScale DiffusionDrive (camera-only) | L0 / F0 / R0 stitched, command, velocity, acceleration | HUGSIM ships an LTF client for the NAVSIM LTF agent (`third_party/hugsim_deps/NAVSIM/ltf_e2e.py`, env `hugsim-ltf` deleted in a cleanup, recipe `experiments/hugsim/archive/install_ltf.sh`). The SimScale models are not NAVSIM agent classes (standalone `policy.py`), so the client's `agent.compute_trajectory` call becomes `predict_batch`; and the shipped dataparser **hard-codes the command to straight** (`command[1] = 1`, the real one commented out), which a turning profile cannot use: pass `info["command"]` with the 0 = R / 1 = L / 2 = S -> NAVSIM map | feasible, small adapter (~one file), env = `envs/alpasim-ltf` |
| ReCogDrive-2B-RL | front camera (`cam_type single`), ego status, command | a NAVSIM `AgentInput` built per step (HUGSIM's dataparser already builds one) and a camera-only sensor config; VLM 2B + DiT per step | feasible once its env is built (setup.md: rebuilding on py3.10); small-to-medium adapter |
| AutoVLA | multi-frame camera prompts built by its own nuPlan preprocessing (JSON per token) | the prompt / JSON builder must run online per HUGSIM step | medium; check whether its agent can build the prompt from an `AgentInput` before committing |
| DriveVLA-W0 | Emu3 VQ tokens of the camera history (released pre-tokenized for navtest only) | online VQ tokenisation of HUGSIM frames + its inference loop as a server; 8B, 18 GB weights | not small; not tonight |

## Cost

WA-JEPA 1.55 worker-job hours (two repeats), third-party anchors (P0, SH30) 1.8, our checkpoints 4.88 (all finished before PD7);
a worker job is one card share (often co-located with the P1 sweep). New disk on the box 5.6 GB after deleting the 38.8 GB of
per-episode `.ply` scene copies (videos kept, ~6 MB per episode).
