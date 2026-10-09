# AlpaSim E2E Closed Loop Challenge 2026

Read this when you work on the `alpasim` lane (experiments/alpasim): the challenge's rules, the driver container
contract, the scoring, and how AlpaSim runs on our box. Everything below was read on 2026-10-08 from the challenge
page (https://nvidia-alpasime2eclosedloopchallenge2026.hf.space/), its Gradio config and public leaderboard API, the
report template zip, and `NVlabs/alpasim` branch `e2e_challenge` (HEAD 2023534; the server deploys 0bb4c4b). Re-read
the sources before acting on a date or a limit. Registration and submission are the user's actions, never an agent's.
The submission path (image conventions, registry, CLI, service images) was read from the same branch on 2026-10-09:
"Submission image" below.

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

- Our team: display name `Lao Siji`, team id `lao-siji`, registered by the user on 2026-10-09; status "Registration pending
  review". The form sits in the Registration tab of the challenge page and appears only after a Hugging Face sign-in.
  Next after approval: CLI authentication, `terms show`, `ACCEPT <version>`. The challenge is not in the official NeurIPS
  2026 Competition Track list (16 competitions); its page says "NeurIPS 2026 workshop", workshop and venue not named.

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

Not verified: several renderer replicas per card, three stacks at once, scenes outside part001, the `ec2` preset,
controller gain overrides. Numerical agreement with the official Docker environment: measured on 2026-10-09, next
section (to the third decimal on 47 of 48 scenes, not bit-identical).

## Submission image (measured 2026-10-09, decision 215)

Read this when you build, test or hand over the driver image. Results, the parity analysis and the component / licence
list: [experiments/alpasim/results/tokyo_image_smoke.md](../experiments/alpasim/results/tokyo_image_smoke.md). Pushing,
logging in to the challenge CLI and submitting are the user's actions.

**What the source says about a driver image** (branch `e2e_challenge` at 0bb4c4b: `e2e_challenge/README.md`, the starter
kit and the three SimScale samples, `competitor_cli/alpasim_challenge.py`, the wizard; the warm-up tracks from HEAD 2023534):

| Question | Answer |
|---|---|
| Entry point | The image's own `CMD`; no arguments are passed. Each replica gets `ALPASIM_DRIVER_HOST`, `ALPASIM_DRIVER_PORT`, `ALPASIM_CONTESTANT_REPLICA_INDEX`, `ALPASIM_CONTESTANT_REPLICAS`; the samples default to `0.0.0.0:6789`, `EXPOSE 6789`, `STOPSIGNAL SIGTERM` |
| Health check | None in any sample. The runtime calls `get_version` on every driver address with `wait_for_ready` and `startup_timeout_s` = 240 s, and asserts that all replicas return the same `version_id` / `git_hash` (`src/runtime/alpasim_runtime/validation.py`) |
| Base image, user | Samples: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (starter kit: `python:3.12-slim`), uid / gid 10001, every cache variable pointed at `/tmp`, logs under `/run/alpasim-driver` |
| What the runner mounts | "no host volumes, Docker socket, scene data, or cloud credentials are exposed"; GPU access is provided. The production runner ("direct p5 runner") is not in the repo; the local `run_local_container.sh` uses `--init --cap-drop ALL --security-opt no-new-privileges:true --read-only --pids-limit 1024 --memory 32g --cpus 8 --tmpfs /tmp:size=2g --tmpfs /run:size=64m --gpus all`, and the README calls these "the official container restrictions except outbound network blocking" |
| Registry and tags | `docker tag <local> 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>` and `docker push`, after the CLI's `ecr-login` (a `docker login` with credentials from the API; refused until the terms are accepted). `latest` is rejected; `submit` first runs `docker manifest inspect` on the URI. The image tag is public: the leaderboard API returns it |
| Warm-up | `submit --track nuplan-warmup <uri>`: only in the CLI from commit 7611632 on (the deployed commit's CLI knows `pai` / `nuplan` only). One wave of 32 private scenes (16 replicas x 2), 45 min, unscored, status only; "validates only the image/driver form factor" |
| `--controller-gains` | A JSON object, at most 8 KiB, any non-empty subset of `long_position_weight`, `lat_position_weight`, `heading_weight`, `acceleration_weight`, `rel_front_steering_angle_weight`, `rel_acceleration_weight` (0-10) and `idx_start_penalty` (integer 0-19); example `starter_kit/controller_gains.example.json` = the defaults 2, 1, 1, 0.1, 5, 1, 10. Locally: `controller.mpc_implementation=nonlinear controller.gains.<name>=<v>` |
| Official service images | Not pulled: the wizard's compose file builds `alpasim-base:<repo version>` (0.89.0) from the root `Dockerfile` (`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, `uv sync --extra all --extra mtgs`, no lock file, `exclude-newer = "3 days"`, so package versions float with the build date). On the nuPlan track the renderer, controller and runtime all run in it; `nvcr.io/nvidia/nre/nre-ga:26.04` is the PAI renderer only. Built on Tokyo 2026-10-09: 19.7 GB, torch 2.8.0 (CUDA 12.8 wheels), gsplat 1.5.3, numpy 2.5.3 (`results/tokyo_image/alpasim_base_freeze.txt`) |

**The image** (`experiments/alpasim/docker/`). `python:3.11-slim-bookworm` + the box's serving packages from
`requirements.lock` (uv, hashes required; torch 2.14.0 as the `+cu126` build, the box has `+cu130`) + `alpasim_grpc` built
from their `src/grpc` + `cinque.ort.onnx` + the 18 repo files of `closure.txt` + one `ckpt-final.pt` per tag. Layers go
from stable to volatile, so a new checkpoint or default costs one thin layer. `serve.py` is the `CMD`: `JEV_TAG` picks
the checkpoint (`tagA+tagB` = ensemble) and `JEV_DRIVER=auto` the family recorded at build time (AlpaSim-standard
checkpoints -> `ap2_driver.py`, the others -> `sh30_driver.py`). A bare `docker run` serves the build's `TAG`. Logs go to
`/tmp/alpasim-driver` (0.8 MiB per 48 scenes), nothing is written elsewhere.

```bash
# On the Tokyo box (DATA_DIR=/data). Checkpoint and its native reference first, from the Mac (docs/tokyo-box.md):
#   ssh -A ujs@100.108.238.8 'bash ~/mycode/jev-drive/experiments/alpasim/scripts/tokyo_pull.sh ckpt <tag>'
scripts/tmux_run.sh img-<tag> env DATA_DIR=/data WHEELS=/data/cache/jev_wheels \
  experiments/alpasim/docker/new_tag.sh <tag> [native results-summary.json]
# -> /data/runs/alpasim/tokyo_image/<tag>-<ts>/{STATUS, DONE | ERROR, report.md, IMAGE}; 355 s for AP2-AB-s0 (2026-10-09)
```

`new_tag.sh` = `build.sh` (stages a 0.9 GB context, tags `jev-alpasim:<tag>-<git hash>`) -> `test.sh` (the limits above
plus `--network none`, 2 concurrent synthetic rollouts) -> `smoke.sh` (the 48 scenes in the official containerised stack,
compared with a native run) -> `report.md`. One step alone: `TAG=AP2-AB-s0 build.sh` switches the default without a new
layer above the checkpoints. After a change of `requirements.in`: `wheels.py` (its docstring) and `uv pip compile`.
The trusted image once per host: `docker build -t alpasim-base:0.89.0 /data/third_party/alpasim` (81 min on Tokyo, almost
all download), then copy the generated `*_pb2*.py` out of it into the checkout's `src/grpc/alpasim_grpc/v0/` (the
compose file mounts the checkout's `src/` over the image's).

| Measured on Tokyo, one RTX 3090 per driver | P2H10-F-s0 (`sh30`) | AP2-AB-s0 (`ap2`) |
|---|--:|--:|
| Image: 7.48 GiB (limit 40); read-only root, no network, `/tmp` 0.8 MiB of 2 GiB, `/run` 0 | pass | pass |
| GPU memory, 2 concurrent rollouts (limit 16 GiB) | 3.04 GiB | 3.04 GiB |
| Cold start | 19.4 s | 19.8 s |
| `drive`, 2 concurrent rollouts, p50 / p90 / max | 86 / 97 / 124 ms | 87 / 95 / 121 ms |
| 48 scenes, official containerised stack | 0.9839, 0 zeros | 0.9521, 2 zeros |
| 48 scenes, native on the GPU box | 0.9840, 0 zeros | 0.9320, 3 zeros |

Containerised and native agree to the third decimal on 47 of 48 scenes for each driver and are not bit-identical: the
renderer's first frame is byte-identical, the driver's first plan differs by up to 1.6 cm (fp16 on another GPU / CUDA
build, sampling maps on another CPU), and the closed loop amplifies that; one AP2 at-fault collision became 0.97.

Deviations of the Tokyo smoke from the official run: the gsplat kernel arch list is overridden to the render card's own
(`TORCH_CUDA_ARCH_LIST=8.6`; the shipped `8.9;9.0+PTX` has no kernel for an RTX 3090); the 1-GPU local topology with 8
concurrent rollouts per driver instead of `ec2` (16 replicas x 2); the wizard runs inside the trusted image with
`run_method=NONE` and `PYTHONPATH` on the checkout, then `docker compose up` by hand; the trusted image is today's
resolution of floating versions. Driver change for the image: every gRPC worker thread runs one inference before the
port opens (1.35 s per thread otherwise, inside the first rollouts).

**What the user runs** (none of it was run by an agent; quota: 3 warm-ups per 30 days, 3 official per UTC month):

```bash
# On the Tokyo box. The CLI of the branch head has the warm-up tracks; it is one stdlib-only file.
git clone --depth 1 --branch e2e_challenge https://github.com/NVlabs/alpasim /data/third_party/alpasim-cli
cd /data/third_party/alpasim-cli; C="python3 e2e_challenge/competitor_cli/alpasim_challenge.py"
$C auth-url          # open the URL, sign in with Hugging Face, copy the token (valid 12 h)
$C configure-token   # paste it; stored in ~/.alpasim/challenge.json, mode 600
$C me; $C terms show; $C terms accept   # type: ACCEPT <version>; the captain and every submitter, each on their own account
$C limits
$C ecr-login         # prints the team repository URI
IMG=$(cat /data/runs/alpasim/tokyo_image/<tag>-<ts>/IMAGE)              # e.g. jev-alpasim:p2h10-f-s0-6f3d05d1
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/lao-siji:v1      # the tag is public; not `latest`
docker tag "$IMG" "$URI"; until docker push "$URI"; do sleep 10; done   # 4.9 GB compressed, resumes per layer
$C submit --track nuplan-warmup "$URI"      # warm-up: 32 private scenes, status only
$C status <submission_id>; $C submissions --track nuplan-warmup
$C submit --track nuplan "$URI"             # official; add --controller-gains gains.json to change the MPC gains
```

## Serving switches: the speed profile of the returned trajectory (2026-10-10, decision FIX1_DECISION)

Read this before you change what the drivers return, or build an image with a switch on. Code: `experiments/alpasim/lib/serve_fix.py`
(its docstring is the specification), hooked into `sh30_driver.py`, `ap2_driver.py` and `pai_driver.py`. Results:
[experiments/alpasim/results/fix1_serving.md](../experiments/alpasim/results/fix1_serving.md).

**What the controller reads** (`src/controller/alpasim_controller`, 0bb4c4b, run as shipped). The nonlinear MPC has a 20 x 0.1 s horizon,
is re-solved every 0.1 s on the same reference, and costs the reference only from horizon index `idx_start_penalty` = 10: positions and
heading 1.0-2.0 s ahead of the vehicle's clock, weights 2 (longitudinal) / 1 (lateral) / 1 (heading) against 0.1 on acceleration and 5 / 1
on command changes; acceleration command in [-9, 6] m/s^2, steering in +-0.785 rad. Nothing in the first second of the returned trajectory
is tracked, and no speed is: a reference whose positions at 1-2 s are not reachable at the current speed is met with whatever
acceleration closes the position gap.

| Switch | Default | What it does |
|---|---|---|
| `JEV_VCONT=<s>` | off (0) | The plan's path, re-timed: the speed profile starts at the ego's speed and blends linearly into the plan's own 0.5 s segment speeds over `<s>` seconds. `1.0` = the controller's untracked second. Without it the trajectory starts at the plan's mean speed over 0.5 s whatever the ego's speed is |
| `JEV_LEAD=1` | off | openpilot's lead path (ec95db3f: radard's probability filter and vision lead, the lead MPC of `long_mpc.py`, the planner's min of candidates) on the checkpoint's own `lead` / `lead_prob` outputs of the same forward pass; the served speed profile is the pointwise minimum of the profile above and the MPC's speed solution. Only removes speed, no latch. Needs scipy (in the image) |

Both are read once at driver start; `get_version` carries them (`-vc1-lead`), so all replicas of an image must agree. With both off the
drivers return what they returned before, bit for bit (checked against COL1's replay: 2 600 decisions, difference 0.0 m). Every `drive`
record in `drive.jsonl` then has `fix` (ego speed, plan first-segment speed, lead probability / distance / speed, the MPC's and the
plan's acceleration, metres removed in the first 2 s, serving ms) and `poses_model` (the plan before re-timing).

- nuPlan runs: `env JEV_VCONT=1.0 JEV_LEAD=1 bash experiments/alpasim/scripts/run.sh ...` (the environment reaches the driver), or
  `ENV=VALUE` in an `ot2_loop.py` job spec. PAI: `DRV_ENV="-e JEV_VCONT=1.0 -e JEV_LEAD=1" pai_run.sh ...` (it mounts `serve_fix.py`).
- Image: `VCONT=1.0 LEAD=1 experiments/alpasim/docker/new_tag.sh <tag>` bakes them in as the image's defaults (tag suffix `-vc1.0-lead`).
- AlpaSim-only: the switches act on the trajectory handed to AlpaSim's tracker. `jevdrive.bench` (navtest, navhard, HUGSIM) never
  goes through these drivers.

## PAI track on the Tokyo box (measured 2026-10-09)

Read this when you run the Physical AI AV track locally. Its renderer (NRE) ships only as a container; this section is the
containerised stack on the Tokyo box ([tokyo-box.md](tokyo-box.md)). The GPU box runs the same renderer without Docker, see
"PAI on the GPU box" below, and agrees with it on 39 of 40 scenes. Results:
[experiments/alpasim/results/pai_smoke.md](../experiments/alpasim/results/pai_smoke.md).

- Renderer image `nvcr.io/nvidia/nre/nre-ga:26.04`: pulled anonymously (no NGC login), 13.3 GB compressed, 27.8 GB on disk.
- Scenes: `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (gated; the token on the box reads revisions `26.01` and `26.04`),
  one `.usdz` of 1.5-2.0 GB per scene, into `/data/datasets/nurec/all-usdzs/<uuid>.usdz` (the wizard's cache layout);
  5-8 MB/s through Clash at 4 streams.
- The organisers' reference runs on the 441-scene curated validation split (`e2e_challenge/local_evaluation/data/pai`,
  8 subjects x 3 rollouts) are git-lfs pointers in the checkout; the summaries are in `/data/runs/alpasim/pai1/ref/<subject>.json`
  (from `media.githubusercontent.com/media/NVlabs/alpasim/0bb4c4b/<path>`, sha256 checked against the pointers).
- The dev preset's renderer runs with `--enable-harmonizer` and fetches `nvidia/DiffusionHarmonizer` at start, which fails
  inside the container. The public leaderboard evaluates the unharmonised renderer (`e2e_challenge/ec2.yaml`), so `pai_run.sh`
  removes the flag by default; `HARMONIZER=1` keeps it and mounts the weights from `/data/datasets/nurec/harmonizer` (the
  reference runs were rendered that way).

```bash
# on the Tokyo box, tmux `jev`, windows named pai-*; S = experiments/alpasim/scripts (a copy is in /data/runs/alpasim/pai1/code)
P=/data/runs/alpasim/pai1
python3 $S/pai_scenes.py --src /data/third_party/alpasim --ref $P/ref --n 40 --first 10 --out $P/pai_scenes_40.tsv
bash $S/pai_fetch.sh $P/pai_scenes_40.tsv 1                      # stage 1 = the nested 10; 2 = all 40
CONC=4 DRV_ENV="-e PAI_DUMP=10" bash $S/pai_run.sh $P/runs/<name> $P/pai_scenes_40.tsv
python3 $S/pai_report.py table --runs <label>=$P/runs/<name> --ref $P/ref
/data/envs/tfv6/bin/python $S/pai_report.py frames --run $P/runs/<name> --out $P/runs/<name>/fig     # needs matplotlib
```

`pai_run.sh` starts the driver (the nuPlan submission image with `lib/pai_core.py` and `lib/pai_driver.py` mounted), has the
wizard write the compose file inside `alpasim-base` (`+e2e_challenge=dev`, `run_method=NONE`) and runs it under its own
compose project (`pai-<name>`), network (`pai-nonet`) and ports (6400+). The nuPlan image smoke (`docker/smoke.sh`) uses
compose project `run` with `--remove-orphans`: a PAI stack started from a directory named `run` would be taken down by it.

| Measured, one RTX 3090 (24 GB), unharmonised renderer | Value |
|---|---|
| One scene, sequential | 147 s wall: about 45 s start-up, about 100 s for the 20 s rollout (200 steps, 6 cameras), eval and video |
| 10 scenes, 4 concurrent rollouts | 621 s and 569 s: about 60 s per scene |
| VRAM | renderer 4.6 GiB (1 scene), 18.1-18.7 GiB peak (4 concurrent); physics 0.3-1.1 GiB; our driver 3.0 GiB |
| Host RAM, 4 concurrent | runtime 9.3 GiB, renderer 6.0 GiB, driver 2.0 GiB |
| Output | 0.42-0.50 GB per scene (video and `rollout.asl`) |

Four concurrent rollouts with the driver on the same card use about 23 of 24 GiB; more was not tried.
With `HARMONIZER=1` the same 10 scenes took 4 436 s (7.1x slower); the score did not move (0.1539 vs 0.1538).

Last verified: 2026-10-09

## PAI on the GPU box (measured 2026-10-10, lane PAI2)

Read this to run PAI-track scenes through the pool. Parity and baseline tables:
[experiments/alpasim/results/pai2_box.md](../experiments/alpasim/results/pai2_box.md). No Docker, no root: the renderer is the
`nvcr.io/nvidia/nre/nre-ga:26.04` image unpacked into a directory and started by `run_native.py` like every other service; the wizard
writes the same `+e2e_challenge=dev` preset as on Tokyo (harmonizer flag dropped, as `pai_run.sh` does). Physics (`physics_server`) is the
shipped one, installed into the shared AlpaSim venv with `uv pip install --no-deps warp-lang -e src/physics`.

One-time setup (done on the box, `$DATA_DIR/tools/nre-rootfs` 27 GB, scenes in `$DATA_DIR/datasets/nurec`):

```bash
scripts/tmux_run.sh pai2-nre env DATA_DIR=$DATA_DIR bash experiments/alpasim/scripts/pai_nre_pull.sh          # image -> tools/nre-rootfs
scripts/tmux_run.sh pai2-scenes env HFROOT=https://hf-mirror.com bash experiments/alpasim/scripts/pai_fetch.sh <scenes.tsv> <max stage> 4
```

- `nvcr.io` through Clash runs at 0.1 MB/s; DaoCloud's mirror `nvcr.m.daocloud.io` (anonymous, direct) gave 1.4-3 MB/s, so
  `pai_nre_pull.sh` uses it (13.6 GB compressed, 100 min). Scenes come from `hf-mirror.com` (needs the HF token in
  `~/.cache/huggingface/token`, direct, 8 ranged streams per scene, 4 scenes at a time: 4-13 MB/s, 1.6-2.0 GB per scene). Both go through
  `ranged_get.sh` (resume, one writer per file, bytes appended only on HTTP 206). The 10 first scenes have the same sha256 as on Tokyo.
- GitHub can be unreachable from the box (name resolution); `git fetch` retries, or copy the file from the Mac.

Run a scene list (the card is whatever the pool gives; the run sees it as device 0):

```bash
R=$DATA_DIR/runs/alpasim/pai2/<name>; mkdir -p $R/pool
.venv/bin/python -m jevdrive.cl submit --no-check --name pai-<tag> --vram 24 --cpu 6 --ram 40 --log-dir $R/pool -- \
  env CONC=4 SH30_TAG=<checkpoint tag> bash experiments/alpasim/scripts/pai_native.sh $R <scenes.tsv (STAGE=n) | file of scene ids>
python3 experiments/alpasim/scripts/pai2_report.py baseline --runs s0=$R [--ref <ref dir>] --out table.md   # or: parity --ref tokyo=<run>
```

`pai_native.sh` = `run.sh <dir> pai` (driver `scripts/drivers/pai.sh`: `pai_driver.py`, so `JEV_VCONT`, `JEV_LEAD`, `PAI_*`, `DRV_PY` pass
through as in `pai_run.sh`) + `run_native.py --sub` (rewrite `/app` to the unpacked image, drop `--enable-harmonizer`, per-run renderer
cache) + `--rewrite-configs` (the generated runtime config names `/mnt/nre-data/...`). The result is `$R/aggregate/results-summary.json`.
A scene list is one simulator launch: the simulator reproduces only for an identical list, so reuse the chunk files in
`experiments/alpasim/results/pai/chunks/` (`a10`, `b1`, `b2a`, `b2b` = Tokyo's chunks of the 40; `e1`..`e6` = the extension).

Scenes: the curated validation split has 441 scenes (the challenge's local evaluation set; `sim_suites_curated.csv`, the train split has 1 761
more, 2 202 in the public catalogue). `pai_scenes.py` takes N scenes, one per quantile bin of the organisers' reference difficulty (mean of
8 subjects x 3 rollouts), seed 0: `pai_scenes_40.tsv` is `--n 40`; `pai_scenes_ext120.tsv` is `--n 120 --exclude pai_scenes_40.tsv --shuffle`
over the other 401 (stage 1 = a nested 60, shuffled, so any prefix is a random sample of the difficulty range).

| Measured, one stack (CONC 4, 10 scenes per launch, 12 runs) | Value |
|---|---|
| VRAM | renderer 16-21 GiB, driver 1.9, physics 1.3-1.8: 20-25 GiB per stack (pool: `--vram 24`) |
| Host RAM (sum of process RSS peaks) | 23-28 GiB (runtime 11, renderer 12, driver 2.5, physics 1, controller 0.8; pool: `--ram 40`) |
| Cores | 2.5-2.8 on average (pool: `--cpu 6`), box load 30-50 from other lanes at the time |
| Wall | 64-78 s per scene (Tokyo's 3090: 57-62); 10 scenes in 643-801 s plus about 25 s start-up; one scene alone 219 s |
| Output | about 0.45 GB per scene (video, `rollout.asl`) |

Agreement with Tokyo (P2H10-F-s0, the 40 scenes, Tokyo's chunk lists): mean 0.1505 against 0.1694, 37 of 40 scene classes equal, 39 of 40
scores equal to 0.01; the first 10 scenes are identical (0.1539 against 0.1538), twice on the box. The one real flip (b988494a, 0.752 on
Tokyo, 0 here) is the cm-level numerics of decision 215. Baseline of P2H10-F-s0 / -s1 on 60 scenes and the organisers' reference rows:
[pai2_box.md](../experiments/alpasim/results/pai2_box.md). `run_native.py` ends a run with rc 1 only when no rollout produced metrics, a scored
zero is a finished rollout.

Last verified: 2026-10-10

