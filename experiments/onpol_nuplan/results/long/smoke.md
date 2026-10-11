# ONPOL-NUPLAN round 2, item A: long-horizon smoke (2026-10-12)

Pre-registration: [plans/2026-10-12-onpol2-prereg.md](../../plans/2026-10-12-onpol2-prereg.md), section A. Lane ONPOL-NUPLAN, main GPU box,
AlpaSim at 0bb4c4b run as shipped. Code: `scripts/on_long.sh` (the chain), `scripts/on_long_smoke.py` (gate A1), `scripts/on_long_probe.py`
(CPU probes). Numbers: [smoke.json](smoke.json), [span.csv](span.csv), [span.json](span.json),
[window_after160_shipped_cache.log](window_after160_shipped_cache.log). Box: `$DATA_DIR/runs/onpol_nuplan/long/{smoke,probe}`.

**Result: no scene runs beyond 10 decisions on this box, so item A stops here as registered. No held scene was rolled.** The window
override is accepted and changes nothing: the scene length comes from the organisers' prebuilt trajdata cache, which holds 111 frames
(5.5 s) per scene, and a rebuild needs the raw nuPlan logs, which are not on the box. Gate A1 passes trivially (the three runs are
identical to the last digit), which says the override is inert, not that a long window works.

## How the window is set, and what bounds it

- **Override path** (hydra, the key exists in the shipped config, no `+`), appended to `run.sh`'s wizard overrides like every other one:
  `runtime.scene_provider.trajdata.dataset.extra_params.num_timesteps_after=<N>`. It reaches `generated-user-config-0.yaml` and both
  consumers (the runtime's scene loader and the MTGS renderer, which builds its own `UnifiedDataset` from the same file).
- **Units: 20 Hz, not 10 Hz.** `num_timesteps_before / after` index nuPlan `lidar_pc` rows (`trajdata/dataset_specific/nuplan/nuplan_utils.py`,
  `_load_scenes_from_central_tokens`: `start_idx = central_idx - before`, `end_idx = central_idx + after + 1`). 30 + 80 + 1 = 111 rows at
  0.05 s = 1.5 s before and 4.0 s after the token = 5.5 s; `desired_dt: 0.1` resamples that to 56 steps. So `after 160` means 9.5 s
  (18 decisions), not 19 s, and 20 decisions need `after 180` (10.5 s). `n_sim_steps: 200` counts 0.5 s control steps (a 100 s cap),
  and the runtime clips it to the recording: `Clipping n_sim_steps from 200 to 10 because only 10 complete policy steps fit between
  first_policy_timestamp_us=17000 and scene end=5500000` (every rollout of every arm below).
- **The bound is the prebuilt cache.** `$DATA_DIR/datasets/alpasim_nuplan/nuplan_test/` (4.2 G, files dated 2026-08-25, downloaded with
  the challenge data) has one directory per scene with the agent tables already cut to the window: all 1491 public scenes have 111
  frames (`scenes_list.dill`). trajdata keys the cache by dataset name only; with `scenes_list.dill` present and `rebuild_cache: false`
  the raw dataset object is never built and the window parameters are never read. The class's own docstring says so ("pass
  rebuild_cache=True ... to avoid stale cached data being silently reused").
- **A rebuild cannot run here.** Three things are missing, in the order the loader hits them:
  1. the nuPlan devkit in AlpaSim's venv (`No module named 'nuplan'`, raised from `cache_maps`; the loader warns "rebuilding from raw
     NuPlan data requires the NuPlan devkit");
  2. the raw nuPlan test-split sqlite logs, expected at `<data_dir>/test/<log>.db`: `Warning: Database file
     .../alpasim_nuplan/test/2021.10.06.07.26.10_veh-52_01245_02064.db not found, skipping...`, 12 146 times, one per central token. No
     `2021.*.db` exists anywhere on the data disk; the NAVSIM logs on the box are 2 Hz pickles, a different format;
  3. the raw nuPlan maps for the map cache (the gpkg files are on the box under `datasets/navsim/maps`, not wired to this loader; the
     four prebuilt map files of the shipped cache could be reused instead).
- **What does not bound it** (measured on the files, CPU, no rollout; `on_long_probe.py span`, all 1491 scenes):

| Bound, seconds after the scene token | min | p10 | median | p90 | scenes with >= 8.0 s (the `after 160` window) |
|:--|--:|--:|--:|--:|:--|
| Shipped cache | 4.0 | 4.0 | 4.0 | 4.0 | 0 |
| nuPlan log continues (NAVSIM test log, 2 Hz) | 5.0 | 29.0 | 139.0 | 540.0 | 1475 / 1491; held 457 / 460 |
| MTGS reconstruction's own video continues (`assets/<scene>/video_scene_dict.pkl`, the camera frames the Gaussians were fitted to) | 5.3 | 8.2 | 10.2 | 34.9 | 1426 / 1491; held 439 / 460 |
| Logged ego stays inside the reconstructed road block + 10 m buffer (2 Hz resolution) | 5.0 | 9.5 | 12.5 | 47.0 | - |

  In decisions (scene start = token - 1.5 s, first decision at 0.017 s, one per 0.5 s), by the reconstruction video: at least 14 for
  99.8 % of the 1491, at least 18 for 99.3 % (held 457 / 460), at least 20 for 84.4 % (held 381 / 460 = 82.8 %), at least 30 for 27.9 %,
  at least 40 for 19.1 %. So the recordings and the reconstructions would carry about 18 decisions on nearly every scene and 20 on
  five in six; only the cache stops at 10. The 136 logs behind the 1491 scenes (39 behind the held 460) are what a rebuild needs.

## The smoke

Ten train scenes ([smoke10_train.txt](smoke10_train.txt): the first 10 of the `default_rng(0)` permutation of `train.txt`, ten different
logs, no held scene), `P2H10S-F-s0`, unchanged driver path (`run.sh sh30`, `c0b_chain.OVERRIDES`), one stack at a time through the pool
(jobs `1012-002038-b45f`, `1012-002329-2f35`, `1012-002600-bc71`, `1012-002841-ddb1`; 440 s of stack time in total = 0.12 card-hour).

| Arm | Window override | Pool | Rollouts | Decisions per scene | Driver errors | Runtime wall, s per scene | Runtime RSS, GiB | Renderer VRAM, GiB | Disk before pruning, MB per scene | Mean score (train scenes, not a read) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `ship` | none | done | 10 / 10 | 10 | 0 | 9.3 | 10.3 | 11.7 | 27 | 0.7766 |
| `a80` | `num_timesteps_after=80` | done | 10 / 10 | 10 | 0 | 6.5 | 10.3 | 11.3 | 27 | 0.7766 |
| `b160` | `num_timesteps_after=160` | done | 10 / 10 | 10 | 0 | 9.2 | 10.3 | 11.8 | 27 | 0.7766 |
| `c160new` | `num_timesteps_after=160` + `runtime.scene_provider.trajdata.cache_location=<new dir>` | error after 45 s | 0 / 10 | - | 0 | - | - | - | - | - |

- `b160`: every one of the 10 scenes has 10 decisions and the runtime logs the same clipping line with `scene end=5500000`. **No scene
  extends.** `on_long_probe.py window --after 160` shows the same without a rollout: 56 frames at 0.1 s, range 5.500 s, 10 decisions
  for all ten.
- `c160new` (the rebuild into a new, empty cache directory; the shipped cache was not written): the renderer exits before it listens,
  `ERROR: Failed to create get_scene function: No module named 'nuplan'` after 12 146 `Database file ... not found, skipping...`
  warnings; `run_native.py` ends the job with `renderer-0 exited with 0 before listening on 23146`. The new cache directory stayed empty.
- Per scene, all three finished arms give the same result: 7 scenes at 1.0, one at 0.7657, two zeros (`left_corridor_laterally` in
  `2021.09.16.19.27.01_veh-45_01749_03230-89e02236312d5038`, `offroad` in `2021.10.06.07.26.10_veh-52_01245_02064-1c0c94dbcd5359fc`).
- Wall per scene at 10 scenes is start-up dominated (8 concurrent rollouts, 10 scenes); the lane's 460-scene reads ran at 3.3 s per
  scene. Memory and disk are the known 10-decision figures.

**Gate A1.**

| Part | Requirement | Measured | |
|:--|:--|:--|:--|
| Shipped window through the override path (`a80` against `ship`) | no zero flip, scores within 0.01 | 0 flips, largest score difference 0.0; all 100 scene x decision plans identical (0.0 m) | pass |
| Wider window (`b160` against `a80`) | every rollout ends, no driver error, plan points within 0.03 m on 99 % of decisions 0..9 | 10 / 10 rollouts, 0 inference and 0 input errors, 100 / 100 decisions identical (0.0 m) | pass |

The pass carries no information about a long window: `b160` is the shipped scene, served bit for bit. It does show that this stack
reproduces exactly on an identical scene list (three launches, 300 plans, 0.0 m).

## What a long read would take (not done, not registered)

1. **The shipped path:** the nuPlan v1.1 test-split sqlite logs for the 136 logs (39 for the held scenes) under
   `<a new data dir>/test/`, the nuPlan devkit in a venv that can import AlpaSim's trajdata, and a rebuild into a new cache directory
   (`cache_location=<new>`, with `data_dir` and `asset_base_path` as they are, the four prebuilt map files copied in or
   `NUPLAN_MAPS_ROOT` set). The download and the licence are the user's decision; its size was not checked. `asset_base_path`,
   `config_dir` and the renderer's asset lookup are independent of `cache_location`, so the shipped cache is not touched.
2. **Without the raw logs:** a cache written by our own code from the NAVSIM 2 Hz logs (ego pose and boxes with track tokens are
   there). That is a new loader, not AlpaSim as shipped: actors and the logged ego path interpolated from 2 Hz beyond the shipped 4.0 s.
   It needs its own registration.

Either way, these differences from the shipped 10-decision setting would move scores by themselves and must be read beside any
long-horizon number (from the source; none was exercised):

- **Corridor.** `left_corridor_laterally` is a hard zero when the lateral distance to the logged path reaches 4.0 m at any step; the
  round-1 drift read has the distance growing convexly with the decision index, so zeros rise with length for every checkpoint. The
  "first 10 decisions of the long rollout" cut of the prereg is the control for that.
- **Progress.** `progress_rel_to_total` is normalised by the logged distance reachable inside the simulated window, and the 1.0 rule
  for logs under 5 m (`min_gt_distance_for_full_score_m`) applies to the whole window: fewer scenes get the automatic 1.0, and
  stop-and-go behaviour after 4 s enters the score.
- **Route.** `RouteGeneratorMap` takes the recorded path of the window and extends it along map lanes only when less than 80 m lie
  ahead. A longer window means more of the 40-80 m the driver is sent is the recorded path instead of the lane-following extension,
  already from decision 0: the first 10 decisions are then not served the same route as in the shipped run, which is what gate A1's
  second part is for. The set of scenes that fail the route fold-back check today may change.
- **Log replay.** `force_gt_duration_us` stays 0.5 s. Other vehicles replay the log over the longer window with no reaction to the ego
  (`physics_update_mode: NONE`, traffic sim skipped), so rear-end and cut-in collisions with replayed actors grow with length and with
  the ego's deviation; `collision_at_fault` is the scorer's judgement of that.
- **Rendering.** 65 of the 1491 scenes (21 held) leave the reconstruction's own camera coverage before +8.0 s, so the last decisions
  are rendered from extrapolated views. The renderer draws only actors the asset has a model for (`Skipping inactive asset actor`);
  whether vehicles that enter after +4.0 s have one was not checked, and an undrawn actor still counts for collisions.
- **Cost.** One decision costs one 8-camera render: about 1.8x the 10-decision wall at 18 decisions. Extrapolated, not measured:
  6 checkpoints x 460 scenes, about 6 s per scene and stack = 45 min per job, 4.6 stack-hours = 1.5-2.3 card-hours at two to three
  stacks per card, 1.6 h wall at 3 stacks; `rollout.asl` about 45 MB per scene (21 GB per job until pruned); runtime memory unknown
  beyond "11 GB + 0.1 GB per scene at 10 decisions" (declare 100 GB per job until measured).
- **What assumes 10 decisions in our code:** `on_read_loop.py --check` (`N_DRIVE = 10`), the drift scripts' `k = 4..10` slope, and
  the stored reads. `sh30_driver.py` has no session-length constant and `ot2_loop.py` counts rollouts, not decisions; neither was run
  beyond 10.

## Not verified

- Anything about a rollout longer than 10 decisions: none exists. The route, scorer and renderer statements above are from reading
  the source at 0bb4c4b.
- That the raw nuPlan test logs reproduce the shipped cache at the shipped window (the precondition of gate A1 on a rebuilt cache).
- That the official private suite uses a longer window. `aud1_private_gap.md` has the hypothesis; nothing here tests it.
- The size of the nuPlan test-split download and whether single logs can be fetched.
