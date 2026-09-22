Read this when you need to run, inspect, or reproduce the fixed trajectory controllers in Bench2Drive.

The current CLI default is **`carla`**. `tcp` is a reference and `pursuit` is a geometric candidate; no replacement default is qualified yet. G2 `development3` passed the four no-background routes for CARLA and pursuit; TCP failed the turn case 26966. Official smoke route 2390 completed 100% in 216 ticks. These diagnostics do not establish a leaderboard score improvement. See the [shared working folder](../todos/2026-09-22-b2d-controller/README.md), [acceptance plan](../todos/2026-09-22-b2d-controller/plan.md), and [experiment/article notes](../todos/2026-09-22-b2d-controller/article-notes.md) for final evidence and outstanding gates.

**API and timing.** [b2d_controller.py](../scripts/b2d_controller.py) uses only NumPy and the standard library, and supports Python 3.8:

```python
from scripts.b2d_controller import Controller

controller = Controller(preset="carla", speed_window="near")
controller.update(trajectory_xy, t_frame)  # queues a copy; returns bool
throttle, steer, brake = controller.step(t_now, speed_mps, yaw_rate_rps)
diagnostics = controller.diagnostics      # detached JSON-compatible snapshot
controller.reset()                       # required between routes
```

| Boundary | Contract |
|---|---|
| Trajectory | Shape `(20, 2)`, meters, rear-axle origin, x forward / y left; samples at +0.25 through +5.0 s |
| Motion | Simulation seconds, speed m/s, yaw rate rad/s positive left; `step` every 0.05 s |
| Output | CARLA steer positive right; throttle ≤0.75, brake ≤1, absolute steer ≤0.8, steer rate ≤2/s; throttle/brake mutually exclusive |
| History | Timestamped rigid-motion reprojection; 2 s history and 0.5 s stale timeout by default; rejected/invalid/stale inputs have explicit diagnostics |
| Agent | Exact current GPS/IMU/SPEED every tick; `--decimate 4` gives synchronous 5 Hz route replanning and 20 Hz control |
| Cameras | Decimated separately; only coherent same-frame camera sets are exposed, optional for `policy=none` |

The origin is prepended as an auxiliary timed trajectory point. Its bridge to the first future point includes position/tracking offset, so trajectory-derived `target_speed_mps` and `reference_speed_mps` can differ from configured `cruise_mps`, especially at startup and parking. Command-speed error is not interchangeable with error against the independent configured route speed profile. Near speed reads `[age, age+0.25]`; the reference-window option reads `[age+0.25, age+1.0]`.

**Vehicle and input boundary.** The frozen [configuration](../todos/2026-09-22-b2d-controller/results/controller_config.json) comes from stock CARLA 0.9.15 `vehicle.lincoln.mkz_2020`: wheelbase 2.860471491 m, actor-relative rear axle x=-1.388633220 m, front maximum steer approximately 70°. `steering_curve` uses km/h: `(0,1), (20,.9), (60,.8), (120,.7)`. These measurements do not establish an exact bicycle model of the plant. The locked evaluator is `/data/third_party/Bench2Drive`, version 0.0.4, commit `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`.

`--drive controller` currently requires `--policy none`. The adapter obtains its dense path and Mercator transform from paired route GPS/world coordinates: this is a **route-oracle control diagnostic**, without obstacle avoidance, traffic-light policy, or a real planner. Pose uses GNSS/compass plus speed/gyro prediction; the GNSS mount is x=-1.4 m, with gains .05 for position and .1 for heading. Tiny signed standstill speed below .01 m/s is zeroed, while `raw_speed_mps` remains logged. Meaningful reverse remains a controller fault.

Truth is isolated in the logger and does not influence controls. It uses matching-frame snapshots and the rear axle including pitch projection. `truth_cross_track_m` measures route tracking; estimated `route_cross_track_m` and controller-local `cross_track_m` are different metrics. The route extends 3 m beyond its official endpoint and latches parking near that endpoint. Its reference still updates on the four-tick schedule, so parking intent can precede the zero-speed reference by up to three ticks.

**Reproduce.** Run from the repository root in the configured Tokyo shell (`DATA_DIR=/data`, physical GPU 1 exposed by `CUDA_VISIBLE_DEVICES=1`; local renderer rank 0). Use a fresh output directory, one worker, an unused server index, and the tmux workflow in [long-runs.md](long-runs.md). A single-route smoke is:

```bash
DATA_DIR=/data CUDA_VISIBLE_DEVICES=1 /data/envs/carla/bin/python scripts/b2d_run.py \
  --routes /data/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml \
  --route-ids 2390 --workers 1 --server-index 90 --gpu-rank 0 --windowed \
  --rig front3 --width 800 --height 450 --decimate 4 --zero-copy --no-spectator \
  --policy none --drive controller --controller-preset carla --cruise-mps 8 --tm-seed 0 \
  --controller-config todos/2026-09-22-b2d-controller/results/controller_config.json \
  --out /data/runs/b2d/controller/smoke-v2
```

A paired campaign can append the frozen holdout XML and a separate slope brake-hold check:

```bash
DATA_DIR=/data CUDA_VISIBLE_DEVICES=1 /data/envs/carla/bin/python scripts/b2d_controller_campaign.py \
  --routes /data/third_party/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml \
  --presets carla,tcp,pursuit --seeds 0,1 --server-index 92 \
  --controller-config todos/2026-09-22-b2d-controller/results/controller_config.json \
  --holdout-routes todos/2026-09-22-b2d-controller/results/holdout.xml --holdout-seed 0 \
  --slope-check --out /data/runs/b2d/controller/campaign-v2
```

The campaign owns one server across groups, resets worlds through the evaluator, snapshots source/config bytes, and generates a report for each group. `--slope-check` measures stationary full-brake holding separately; it is not an additional closed-loop route score. `finished` means the evaluator returned, not that driving completed. Read official completion/status from `results.json`. Default `--max-ticks 0` adds no artificial cap, but the pinned evaluator still has its built-in `tick_count > 4000` TickRuntime limit. Keep that failure distinct from a user-specified smoke cap.

**Preserve and plot.** Raw attempts contain `control.jsonl`, `trajectories.jsonl`, `route_reference.json`, official results, events, and timing logs. Keep `/data/runs/b2d/controller/development*`, campaign groups, failed attempts, and all figure editions. Capture source bytes before a standalone run; inventory only after the run is stopped:

```bash
/data/envs/carla/bin/python scripts/b2d_controller_archive.py snapshot \
  --out /data/runs/b2d/controller/source-v2 \
  --input todos/2026-09-22-b2d-controller/results/controller_config.json \
  --input todos/2026-09-22-b2d-controller/results/holdout.xml
/data/envs/carla/bin/python scripts/b2d_controller_archive.py inventory \
  --root /data/runs/b2d/controller/campaign-v2 \
  --out /data/runs/b2d/controller/campaign-v2-inventory.json
/data/envs/carla/bin/python scripts/b2d_controller_plot.py \
  --development /data/runs/b2d/controller/development3 \
  --campaign /data/runs/b2d/controller/campaign-v2 --label Dev10 \
  --out /data/runs/b2d/controller/figures-v3
```

Snapshots include source/config hashes rather than relying on HEAD alone. Plotting requires NumPy and Matplotlib and exports English PNG/PDF, exact CSV, source snapshots, and a SHA256 manifest. Nonempty plot output directories are rejected. A campaign directory includes only direct `*-seed*/controller-report.json` groups; nested holdout reports are excluded. Plot holdout separately with `--campaign /data/runs/b2d/controller/campaign-v2/holdout --label Holdout` and a new output directory. A report file selects one group. Failures/retries remain visible, missing metrics stay missing, and collision-free prefixes require recorded event frames.

Last verified: 2026-09-22
