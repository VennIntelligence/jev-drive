# AlpaSim E2E Closed Loop Challenge 2026

Read this when you work on the `alpasim` lane (experiments/alpasim): the challenge's rules, the driver container
contract, the scoring, and how AlpaSim runs on our box. Everything below was read on 2026-10-08 from the challenge
page (https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/), its Gradio config and public leaderboard API, the
report template zip, and `NVlabs/alpasim` branch `e2e_challenge` (HEAD 2023534; the server deploys 0bb4c4b). Re-read
the sources before acting on a date or a limit. Registration and submission are the user's actions, never an agent's.

## Dates and quota

| What | Value |
|---|---|
| New team registration closes | 2026-10-18 |
| Public leaderboard closes; one final valid container per track selected; technical reports due | 2026-10-31 |
| Final results (target) | 2026-11-15; results shared during NeurIPS 2026 |
| Rules / metric / submission format frozen | 2026-09-15 ("except for critical fixes") |
| Official submissions | 3 per team per UTC calendar month, shared across PAI and nuPlan, no carry-over; "Accepted submissions count even if evaluation fails" |
| Warm-up (`nuplan-warmup`, `pai-warmup`) | 3 per warm-up track per rolling 30 days; 32 private scenes, 45 min limit, unscored, never on a leaderboard; status only |
| Final runs | at least the top 3 teams per track; nuPlan finalists get 2 extra runs; Harmonizer off on nuPlan throughout |

## Registration and identity (quoted)

- "Registration is limited to teams affiliated with academic or industry organizations. [...] the team captain's
  verified Hugging Face primary email must be an organization address." Form note: "Applications using common personal
  email services cannot be submitted."
- Form fields (Gradio config): Team Display Name, Requested Team ID ("Lowercase letters, digits, hyphens. 3-40
  characters."), Captain Full Name, Additional Submitter Hugging Face Usernames, Credited Contributors, Organization /
  Affiliation (required; "Used for review and duplicate-organization checks."), Country / Region, Official / Local
  Organization Name (optional), Group / Lab / Department (optional), Brief Project Description (optional), Notes To
  Organizers. Four checkboxes, among them "I understand that the submitter roster is frozen after approval except by
  administrator exception." and "I agree to follow the AlpaSim Challenge rules and organizer instructions."
- Registrations are reviewed and approved by organisers ("Registrations are reviewed to prevent duplicate or
  non-genuine accounts"). "Each participant may belong to only one registered team."
- The public leaderboard API returns per entry: team display name, team id, image tag, submission id, scores, status,
  submitted-at. No organisation, captain or member field. Display names such as "Anonymous (host)" and "Unofficial" are
  on the board; no rule read here restricts the display name.
- Terms and conditions exist ("All registered teams may submit, but some teams may be ineligible for prizes") and are
  shown only after CLI authentication (`terms show`); every submitter and the captain must type `ACCEPT <version>`.
  Not read: they need an approved account.
- Technical report: "Award eligibility requires a final valid submission and a concise technical report", four pages,
  template required. The template README says: "Reports are not anonymous." Its author block is team name,
  organisation "(if applicable)", contact e-mail. Required content: method summary, policy description, datasets /
  annotations / pretrained weights with public source and licence (nuPlan), training and inference setup, limitations.
  No source read here says whether reports are published. "Organizers may request code, logs, or method details for
  award candidates, disputed results, or suspected rule violations". "Winners and selected participants present
  results" at NeurIPS.
- Resources: "Award-eligible nuPlan submissions may use only publicly available datasets, annotations, and model
  weights." PAI-AV may use private resources with a high-level disclosure.

## Driver contract (both tracks)

A submission is one Docker image serving `egodriver.EgodriverService` (gRPC, `src/grpc/alpasim_grpc/v0/egodriver.proto`):
`start_session` (rollout spec: camera calibrations `available_cameras`, ego bounding box and its pose in the rig
frame), `submit_image_observation` (one JPEG per camera frame with `logical_id`, start / end timestamps),
`submit_egomotion_observation` (estimated local->rig poses and rig-frame `DynamicState` velocity / acceleration),
`submit_route` (waypoints in the rig frame), `drive(time_now_us, time_query_us)` -> a trajectory of local->rig poses
with timestamps, `close_session`. No map, no agents, no traffic-light state, no scene id in benchmark runs.
`submit_recording_ground_truth` exists but `send_recording_ground_truth: false`.

- Execution: the official nonlinear MPC tracks the returned trajectory on the simulator's vehicle model. Tunable per
  submission (`--controller-gains`): `long_position_weight`, `lat_position_weight`, `heading_weight`,
  `acceleration_weight`, `rel_front_steering_angle_weight`, `rel_acceleration_weight` (0-10; defaults 2, 1, 1, 0.1, 5,
  1) and `idx_start_penalty` (integer 0-19, default 10). Nothing else. The controller is "not a general trajectory
  validator"; unrealistically harsh braking references have produced zig-zag steering (CONTROLLER_TUNING.md).
- Limits: image <= 40 GiB; <= 16 GiB VRAM per driver instance; no outbound network; read-only root; scratch `/tmp`
  2 GiB, `/run` 64 MiB; 16 replicas of the image on 4 GPUs, 2 concurrent rollouts per replica; "target 0.1 seconds or
  less of model work per `Drive` call", enforced as a per-track throughput budget whose value is shown only in the
  submission status (not read).

## nuPlan track (config `e2e_challenge_nuplan_common/base.yaml`)

| Item | Value |
|---|---|
| Renderer | MTGS reconstructions of nuPlan logs; traffic is vehicles only, no pedestrians or cyclists |
| Cameras | 8 nuPlan cameras (CAM_F0, L0-L2, R0-R2, B0), 1920 x 1080 JPEG, **2 Hz** (`frame_interval_us: 500_000`) |
| Control step | 0.5 s (`control_timestep_us: 500_000`); log replay for the first 0.5 s (`force_gt_duration_us`) |
| Scene | trajdata window of 30 steps before and 80 after at 0.1 s; `n_sim_steps` 200 capped by the recording |
| Route | `route_generator_type: MAP`, waypoints roughly 40-80 m ahead of the ego (`route_start_offset_m: 40`) |
| Physics / traffic sim | skipped (`physics_update_mode: NONE`); other vehicles replay the log |
| Public suite | 1 485 navtest scenes (`+e2e_challenge_nuplan=full`); 10 fail in MAP route generation before the driver runs |
| Official suite | private; no nuPlan reference bundle for the local Drive-IRT tool yet |

Shipped NAVSIM-style samples (LTF, DiffusionDrive, GTRS-Dense) read CAM_L0 / F0 / R0, derive a 4-way command from the
route (first waypoint >= 5 m away: y > 2 m left, < -2 m right) and take velocity / acceleration from `DynamicState`.

## Scoring

- Scene score (`src/eval/src/eval/aggregation/scene_score.py`): 0 if `collision_at_fault`, `offroad` or
  `left_corridor_laterally` is non-zero; otherwise `min(progress_clipped_rel / 0.8, 1)` (1 when the log travelled
  under 5 m). `max_dist_to_gt_trajectory` 4.0 m in the base config.
- Off-road on nuPlan: it was disabled by a config error; fixed in 0bb4c4b (PR 181, 2026-09-14, with a 1 mm seam closure
  for issue 180), which is the deployed commit. So off-road is now a hard failure. "Organizers will rerun only the
  evaluation stage for existing nuPlan submissions"; whether every row on the board is post-fix was not verified.
- Leaderboard columns: Policy Capability Score (Drive-IRT `zoib` fit over all submissions, with a 95 % interval and a
  rank interval; https://ln2697.github.io/navhard-leaderboard-2/docs/), Average Scene Score, Average Distance Between
  At-Fault Incidents (distance / count of `offroad_or_collision_at_fault`).

## Leaderboard, nuPlan track, 2026-10-08 (best entry per team, 30 teams, 96 entries)

| Team | PCS | Avg scene score | At-fault distance |
|---|--:|--:|--:|
| py123d-garage (host) | 1802 [1716, 1888], rank 1-7 | 0.938 | 0.981 |
| SymPhi | 1797 | 0.940 | 0.814 |
| Lucifer AI | 1554 | 0.879 | 0.407 |
| YAX_NAM | 1484 | 0.848 | 0.228 |
| RDY Mobility | 1471 | 0.875 | 0.425 |
| Host: `go_straight_with_delay_v2` | 1295 | 0.799 | 0.258 |
| OpenDriveLab-Org (host): SimScale DiffusionDrive release | 1103 | 0.772 | 0.267 |
| Ulsan Alps: `gtrs-resnet-release` tag | 1031 | 0.761 | 0.411 |
| last of 30 | 629 | 0.294 | 0.046 |

A go-straight host baseline sits at rank 6 of 30, above the released DiffusionDrive and GTRS samples: the private
suite rewards not failing (hard zeros) far more than progress. PAI track: 27 teams, top 2237, host 2000.

## Running it on our box (measured 2026-10-08, decision 184)

The box has no Docker and no root, so AlpaSim runs natively: the wizard runs as shipped with `wizard.run_method=NONE`
(it writes the compose file and every service config), and `experiments/alpasim/scripts/run_native.py` starts each
compose service command as a host process on the card the pool hands out, container mount paths replaced by host
paths; the run ends when the runtime exits. Their source is untouched, at the deployed commit 0bb4c4b
(`~/data/third_party/alpasim`).

```bash
scripts/tmux_run.sh alpasim-env   experiments/alpasim/scripts/setup_env.sh   # uv env, rust, gsplat toolchain
scripts/tmux_run.sh alpasim-fetch experiments/alpasim/scripts/fetch_data.sh  # nuplan_test + configs + part001
scripts/tmux_run.sh alpasim-ltf   experiments/alpasim/scripts/setup_ltf.sh   # shipped LTF sample, own venv
R=$DATA_DIR/runs/alpasim; D=$R/<name>/<ts>
.venv/bin/python -m jevdrive.cl submit --name alpasim-<name> --vram 40 --cpu 16 --log-dir $D/pool -- \
  bash experiments/alpasim/scripts/run.sh $D ltf --scene-list $R/scenes_navtest_full_part001_48.txt \
  +e2e_challenge_nuplan=full runtime.nr_workers=2 runtime.endpoints.renderer.n_concurrent_rollouts=8 \
  runtime.endpoints.driver.n_concurrent_rollouts=8 runtime.endpoints.controller.n_concurrent_rollouts=8 \
  defines.nre_cache_size=9
```

`run.sh <dir> starter|ltf [--tap] [--scene-list F] <wizard overrides>`; `--tap` puts `driver_tap.py` (a logging gRPC
proxy) between runtime and driver. Each run dir has `native_summary.json`, `usage.jsonl` (1 Hz), `native-logs/`,
`aggregate/results-summary.json`.

Deviations from the shipped environment: no containers (no read-only root, no cpu / memory limits); workspace members
installed editable instead of `uv sync --extra all --extra mtgs` (GitHub sources of the in-repo drivers timed out, so
those drivers and physics are not installed; `exclude-newer` not applied); torch 2.9.1+cu128 (their image: CUDA 12.4,
which has no sm_120), gsplat 1.5.3 JIT-built with their `TORCH_CUDA_ARCH_LIST` (196 s once, cached); LTF venv with
torch 2.8.0 instead of the pinned 2.6.0; ffmpeg from imageio-ffmpeg. Full list: `~/data/runs/alpasim/setup/freeze.txt`.

| Measured (LTF sample, part001 scenes, one card) | Value |
|---|---|
| Sequential | ~20 s per scene (session ~10 s for 5 sim s; rest eval + video) |
| 8 concurrent rollouts | 2.96 s per scene; 16 concurrent 2.93 (one renderer process is the bottleneck) |
| VRAM | renderer 5 GB (1 scene), 12.5 GB (8 concurrent), 17.6 GB (16); LTF driver 1 GB |
| CPU / RAM at 8 concurrent | ~4.5 cores; runtime 14 GB RSS |
| Disk | env 8.9 G, LTF env 6.6 G; trajdata cache 4.4 G; assets 0.38 G per scene (part001: 100 scenes, 38 G); output 27 MB per scene |
| LTF `Drive` call | 37-41 ms |
| LTF score | 48 scenes: mean scene score 0.8735, 43 pass; zeros: 4 `left_corridor_laterally`, 1 `offroad` |

Full public suite (1 485 scenes), extrapolated, not run: one stack per card at 8 concurrent, ~25 min wall on 3 cards,
~1.25 card-hours, ~40 GB output. The cost is the data: 458 GiB of tarballs, ~565 G extracted, 20 h or more to
download through the mirror (it throttles long connections; `fetch_data.sh` reconnects every 90 s), against 434 G
free on 2026-10-08.

What a nuPlan-track driver actually receives (tap logs, `dev_ltf/20261008-121717/driver_tap.jsonl`):

- A scene is **5.5 s: 10 `drive` calls** at 2 Hz (`time_now` 0.017, 0.517, ... 4.517 s; query = now + 0.5 s), one
  `start_session`, 10 egomotion, 10 route, 88 images (8 cameras x 11 instants). **No history before t = 0**: the
  first `drive` has one frame per camera and two poses (t = 0 and 0.017 s).
- Within a step the 8 images, the egomotion and the route arrive concurrently in no fixed order, then `drive`; align
  by timestamp. No RPC deadline.
- CAM_F0: fx 1573.5, fy 1496.3, cx 960, cy 560; `rig_to_camera` translation (1.786, -0.026, 1.523) m. Ego box
  5.176 x 2.297 x 1.777 m.
- Egomotion: local->rig poses (local origin = first pose) with rig-frame velocity / acceleration; the first sample's
  velocity looked unrotated ((0.14, 11.47) m/s), guard against it.
- Route: 20 waypoints in the rig frame, ~4.2 m apart, 40-80 m ahead; about half are NaN padding.
- Response: the samples return 10 Hz poses in the local frame with absolute timestamps (starter 51 poses / 5 s; LTF
  resamples its 8 x 0.5 s plan).

### Killed workers and concurrency (2026-10-09)

A runtime worker was SIGKILLed in four stacks: C0 SH30 (10-08 23:38:40, 3 stacks running), C0b `wajepa-c0` and
`ot0-c0` (10-09 09:23:48 and 09:27:18, 9 stacks), M1 stage 2 (10-09 10:36, 9 stacks). Each time one process of about
10 GB went (`Worker 1 died with exit code -9`, `Result pump failed`), the runtime stayed alive and never finished
another rollout (2.6 h in C0).

- It is the box, not AlpaSim: the same single-process SIGKILL hit openpilot servers on 2026-09-25 and six WOD eval
  jobs on 10-08 05:38 (rc 137). What is known about it is in [remote-box.md](remote-box.md), "Host memory".
  Not the kernel OOM killer (`memory.events` max 0, oom 0, oom_kill 0 since the container started on 10-08 12:34,
  before all four kills; `/proc/vmstat` oom_kill 0), not `/dev/shm` (138 G, empty), not pids (`pids.events` max 0),
  not the pool (no `stop` event for those jobs), not the runtime (its only signal is `terminate()`).
- The box sampler that would show the seconds before each kill (`scripts/boxwatch.sh`) had died with the container
  restart on 10-08 12:34 and was not running for any of the four. The dispatcher now keeps it alive.
- `run_native.py` now ends the run with rc 1 as soon as the runtime logs a dead worker, so a kill costs the pool job
  seconds instead of the 20-minute stall watchdog; the chains resubmit as for any failed job.
- One stack (2 workers, 8 concurrent rollouts): host memory `11 GB + ~0.1 GB per scene` at the end of the run
  (233 scenes 26-32 GB, ~350 scenes 40 GB, 400 scenes 52 GB RSS; proportional memory is 85 % of RSS), growing
  ~2 GB / min; 5-7 cores measured (the pool charges that after 5 min, whatever `--cpu` says); VRAM peak 23-32 GB
  (declare 32). Nine stacks grew 5.5 GB / min together and held 219 GB RSS at the first C0b kill.
- Concurrency: VRAM allows two to three stacks per card, the 105-core budget carries about 15,
  and neither was the limit in any kill. No memory figure separates the kills from the clean runs (C0 died with
  three stacks and ~90 GB), so the working cap stays 3 stacks per lane until boxwatch has recorded a kill; the pool
  does not enforce a stack count.

Not verified: numerical agreement with the official Docker environment (no nuPlan reference scores exist), several
renderer replicas per card, three stacks at once, scenes outside part001, the `ec2` preset, controller gain overrides.

Last verified: 2026-10-08
