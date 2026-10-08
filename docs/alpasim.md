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

## Running it on our box

Pending (experiments/alpasim). Known on 2026-10-08: the box has no Docker and we have no root, and AlpaSim ships a
Docker Compose deployment; the public navtest MTGS assets are 15 shards, 458 GiB compressed (plus the 2 GiB trajdata
cache), against 206 GB free on the data disk.

Last verified: 2026-10-08
