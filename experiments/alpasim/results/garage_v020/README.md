# GARAGE020: the host's py123d_garage v0.2.0 baseline on our PAI 281 exam, Harmonizer off

Decision 454. Lane script `experiments/alpasim/scripts/garage020.sh` (job bodies `garage020.py`), env / checkpoint
`setup_garage020.sh`, driver `drivers/garage020.sh`. Box outputs: `$DATA_DIR/runs/alpasim/garage_v020/` (reads with every
`rollout.asl` kept, 20 GB per read; rescoring mirrors; logs).

**Question.** How does the host's new reference baseline (released 2026-10-20, README: PAI 76.0 conditional / 63.9 unconditional,
441 scenes, Harmonizer on) drive on our PAI 281 exam, paired scene by scene with our ACC + GAP reads?

## Numbers (fixed corridor scorer first: decision 453 made it the reference for local PAI 281 numbers)

| read (Harmonizer off) | scenes | fixed scorer mean | zeros | old scorer mean | zeros |
|:--|--:|--:|--:|--:|--:|
| garage v0.2.0 conditional (posttrain seed 0, recommended) | 277 | **0.772** | 41 | 0.750 | 47 |
| garage v0.2.0 unconditional (midtrain seed 0) | 280 | **0.664** | 74 | 0.645 | 79 |
| NTV0-L18-s0, ACC + GAP (same 277 scenes) | 277 | 0.817 | | 0.745 | 58 |

Paired, conditional garage minus our read (95 % paired bootstrap CI, scenes as units; `pai281_garage_c0.md`, `extras_c0.md`):

| ref (ACC + GAP, off) | fixed scorer | old scorer | zeros old (new / fixed) |
|:--|--:|--:|--:|
| NTV0-L18-s0 | -0.046 [-0.100, +0.009] | +0.005 [-0.055, +0.064] | -11 (27 / 38) |
| R6DAG-L18-s0 / s1 / s2 | -0.070 [-0.124, -0.019] / -0.107 [-0.158, -0.057] / -0.086 [-0.139, -0.035] | -0.023 / -0.064 [-0.121, -0.007] / -0.040 | -6 / +4 / -4 |
| R8K-L18-s0 / s1 / s2 | -0.084 [-0.137, -0.032] / -0.093 [-0.146, -0.040] / -0.100 [-0.155, -0.047] | -0.041 / -0.050 / -0.053 (CIs cross 0) | -3 / 0 / 0 |

Unconditional minus NTV0: fixed -0.149 [-0.208, -0.092], old -0.099 [-0.162, -0.037]; minus R6DAG / R8K: -0.18 to -0.22 (fixed),
every CI below 0 (`pai281_garage_u0.md`, `extras_u0.md`).

Where the scorers differ: the fixed scorer adds only +0.022 [+0.007, +0.040] to the garage (6 scenes change) but about +0.07 to
our reads. Our zeros are mostly end-of-log overshoot (NTV0 23, R6DAG / R8K 19-20), which the fix removes; the garage has 6.

Failure profile (old scorer classes, conditional; ours in brackets: NTV0 / R6DAG / R8K):
- zeros: AF-front collisions 20 (3 / 3-4 / 5-6), offroad 18 (11 / 12-20 / 14-17), overshoot 6 (19-23), route-turn 1 (15 / 2-3)
- partials: rear-hit 30 (26 / 18-23 / 18-21), slow 23 (11 / 3-6 / 3-4); launches 29 / 37 (33 / 35 / 35)
- fixed-scorer failure reasons of the garage zeros: at-fault collision 21, offroad 18, corridor 2 (unconditional: corridor 39,
  collision 20, offroad 15: without target points it leaves the route at forks, as their docs say)

New zeros on the 184 scenes NTV0-ag scores >= 0.98 (`perfect_ntv0_*.csv`): conditional 23 zeros + 31 partials (AF-front 13,
offroad 8), unconditional 39 + 29; R6DAG s0-s2 5-12 + 5-9, R8K s0-s2 9-11 + 5-7.

Against the host's own per-scene results (`results/nurec/seed_<n>/results-summary.json` from their checkpoint repo: Harmonizer
ON, AlpaSim e121e59 = the fixed scorer, their 2 Hz camera), on the same 277 scenes: their seed 0 0.771, ours 0.772,
+0.000 [-0.030, +0.032] (seeds 1 / 2: -0.001 / +0.013); same-scene correlation 0.76, zero on both 32, ours only 9, theirs only 11.
Their seeds average 0.768 / 0.752 / 0.761 over all 441 scenes (README 76.0). Our harness therefore reproduces their number; the
Harmonizer does not move this model much on these scenes. (The u0 rows of that table compare the unconditional read with their
conditional results, there being no published unconditional per-scene file.)

## Setting: their driver vs our serve

- Their driver as shipped: `python -m py123d_garage.evaluation.alpasim.evaluate` (wrapped only by the close-record observer
  `alpasim_proxy/scripts/garage_driver.py`), replans every 500 ms and serves the 10 Hz calls from the stored plan (0.203 plans per
  drive call, both reads), no post-processing, no route memory / extrapolation, no ego hood, deterministic flow matching; nurec
  rig next to the checkpoint, as their `scripts/evaluation/alpasim_pai.sh` picks. Conditional = target points 40 / 60 / 80 m
  (the posttrain default); unconditional = their midtrain stage (trained without target points; their figure labels the stage
  "Unconditional"). Our ACC / GAP / VCONT / LEAD are not applied. 0 inference errors, 0 input errors in 562 sessions.
- Harness: the one the NTV0-L18-s0 ACC + GAP read used (`pai_box.sh`: AlpaSim 0bb4c4b, `+e2e_challenge=dev`, Harmonizer off,
  ALPASIM_FAST=1, CONC 4, steering weight 10, NTV0's chunk table so the scenes pair), plus one setting from their driver config:
  `skip_driver_during_force_gt=true` (no drive calls during the 1.7 s force-GT; the ec2 leaderboard preset sets the same through
  alpamayo1's driver config; our stored reads run the dev default, false). Not applied: their 2 Hz camera (`first_frame_offset_us`,
  which needs a patch); the driver gets the preset's 10 Hz frames and plans at 1.7, 2.2, ... s on fresh frames, as on the board.
- Our reads are a different serve: our models + ACC + GAP + VCONT + LEAD, driver called during force-GT.
- Scenes lost to RPC errors: 4 rollouts of the conditional read ended on the renderer's `Expected all tensors to be on the same
  device` (NRE sensorsim render call, traced in the runtime log; not the driver), the unconditional read lost none; tables use the
  scenes every compared read scored (277 / 280, the rest are RPC errors of the reference reads).

## Their three AlpaSim patches (lib/alpasim/tools/patches, applied by their setup_alpasim.sh to e121e59)

None was applied anywhere; the live checkout `third_party/alpasim` (0bb4c4b) is untouched.
- `camera_timing.patch` (runtime): adds `first_frame_offset_us` to camera configs and skips the sensor-sync assert on steps that skip
  the driver, so their one camera can render at 2 Hz from 0.2 s and still land a frame on each plan. Compute saver; changes neither
  the driver contract nor scoring. Not needed with 10 Hz cameras.
- `casadi_pin.patch`: pins `casadi<3.8` in the controller's pyproject (a fresh resolve breaks do_mpc). Dependency only.
- `mtgs_speedup.patch`: nuPlan MTGS renderer only (render only requested cameras, GPU resize, map reuse). Not PAI.
- e121e59 differs from our 0bb4c4b in runtime / eval / wizard configs / grpc only by the corridor-scorer fix (1b024f0), which the
  fixed-scorer column applies.

## Runs, env, cost

- Staged: 1-scene smoke (0.816, 37 plans in 182 drive calls) -> c000 (36 scenes, 0.637 vs NTV0 0.692) -> the other 7 chunks; the
  unconditional read went in as 8 chunks at once on idle cards (same driver code, only the checkpoint differs).
- Conditional c005 / c006 hung at 17:50 (`can't start new thread`: the box pid cap was full); cancelled after 57 min and rerun
  from scratch (18:47). Their 2 x 57 min are in the GPU total.
- Driver thread env: every chunk of both reads started before 8d38075b (19:05:46, which caps the driver's numeric / OpenCV pools at
  8), so all ran with the uncapped env: 935 threads per driver process (counted on the 8 unconditional drivers at 19:06). Scores
  are unaffected (inference on the GPU).
- GPU: 6.1 card-hours of chunk jobs (16 x 13-20 min + smoke + the 2 hung 57 min); rescoring 4 x 8.5 min CPU.
- Checkpoints (`$DATA_DIR/checkpoints/py123d_garage/v0.2.0/ltf-base-joint64-68M`, revision 56edbe90): posttrain / midtrain
  `model_0.pth` + config + nurec rig, README, the per-scene nurec results; sha256 of both checkpoints match HF. The other v0.2.0
  checkpoints were not downloaded. Source: `third_party/py123d_garage-v0.2.0` (tag v0.2.0, c5db115); env `envs/garage020`.

## Files

- `pai281_garage_{c0,u0}.md / .json`: r2.py r281 tables (old scorer classes, paired CIs)
- `extras_{c0,u0}.md`: NTV0-perfect scenes, fixed vs old scorer, paired fixed-scorer table, the host's per-scene results
- `per_scene_{c0,u0}.csv`: per scene: garage old / rescored old / fixed score and failure, NTV0, the host's three seeds
- `perfect_ntv0_{c0,u0}.csv`: new zeros / partials on NTV0-perfect scenes per read
