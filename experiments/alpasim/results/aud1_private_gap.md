# AUD1: did the private nuPlan suite change the driver's behaviour, or punish behaviour we already have?

2026-10-11. Descriptive audit, nothing promoted, nothing submitted, driver unchanged. Question: our official nuPlan submission
(`APY10m10-AB-s0`, image tag `apy10m10-ab-s0-v1`, submission 94d2e447) scored on the private 1000-rollout suite 0.640 average scene
score, 0.132 "Average Distance Between At-Fault Incidents", 2.288 "Distance to GT Trajectory". Locally the same checkpoint scores
0.935. Numbers: [aud1/](aud1/). Code (uncommitted): `scripts/aud1_collect.py` (box, reads stored runs), `scripts/aud1_report.py`,
`scripts/aud1_calib_arms.py`, `scripts/aud1_calib_compare.py`. Box outputs: `$DATA_DIR/runs/alpasim/aud1/`.

**Answer: reading (b).** Locally the driver stays 1.13 m from the logged path (mean of per-scene maxima). That is half the private
2.288 m, and the 95 % interval [1.03, 1.24] is far from it. At-fault incidents come 6.5 times as often per km on the private suite as
on the public one. No public stratum is anywhere near as bad as 0.64. The WA-JEPA reference degrades much less from local to
leaderboard. On the private scenes the driver leaves the logged path and has at-fault events far more often than on any part of the
public suite. Whatever triggers this is not visible in the public scenes.

## 1. The two leaderboard metrics on the 1491 public scenes

**Definitions** (AlpaSim `src/eval` at the deployed commit 0bb4c4b; leaderboard columns from `e2e_challenge/local_evaluation/evaluate.py`
on the `e2e_challenge` branch, HEAD 2023534, and `docs/OPERATIONS.md`):

- Per rollout, `dist_to_gt_trajectory` (m) = MAX over the simulated timesteps of the distance from the ego polygon's centroid to the whole
  logged ego path (`scorers/ground_truth.py`, shapely `LineString.distance`; the first two steps are set to 0). Before the time
  aggregation, every timestep after the first one with `dist_to_gt_trajectory >= 4 m` is dropped (`aggregation/main.py`,
  `RemoveTimestepsAfterEvent`, `max_dist_to_gt_trajectory = 4.0`). A rollout that leaves the path therefore scores just above 4 m
  (5.1-5.2 m on average here), never its true maximum. The leaderboard's "Distance to GT Trajectory" is the run-level mean of this
  over clips (`aggregate_over_clips`: plain mean over the clips of a run).
- Per rollout, `dist_traveled_m` = arc length of the ego's own path at the last kept step (after the same truncation), and
  `offroad_or_collision_at_fault` = MAX of the per-step flag (0 / 1).
- `avg_dist_between_incidents_at_fault` (**km**) = sum(`dist_traveled_m`) / sum(`offroad_or_collision_at_fault`) / 1000 over all clips of
  a run (`aggregation/processing.py`, `get_avg_dist_between_incidents`). `evaluate.py` reads this from `metrics_results` in
  `results-summary.json`. Average scene score = mean of per-scene scores, and scenes that fail before the driver runs count as 0.
- Rollouts that fail in route generation (12 of 1491 here, the same for every driver) have no metrics and do not enter either metric.
- Check: recomputing these from the per-rollout records reproduces a chunk's own `metrics_results` exactly (CF1 Fresh-A run of
  APY10m10-AB-s0: 0.56487 km, 1.16570 m). The values below pool all 1491 scenes as one run. They come from the stored rollouts of CF1 / OT3
  (APY10m10-AB-s0, P2H10-F-s0) and C0b / C0c (WA-JEPA reference, fp32), with no rerun.

| driver | avg scene score [95 % CI, logs] | at-fault distance km [CI] | GT distance m [CI] | GT median m | clips >= 4 m | at-fault events / clips | mean travelled m |
|:--|--:|--:|--:|--:|--:|--:|--:|
| APY10m10-AB-s0 (submitted) | 0.935 [0.919, 0.952] | 0.863 [0.639, 1.355] | 1.130 [1.026, 1.236] | 0.43 | 7.7 % | 48 / 1479 | 28.0 |
| P2H10-F-s0 | 0.919 [0.901, 0.939] | 0.722 [0.512, 1.233] | 1.020 [0.934, 1.106] | 0.49 | 5.5 % | 55 / 1479 | 26.9 |
| WA-JEPA reference | 0.895 [0.866, 0.922] | 0.531 [0.370, 0.849] | 1.193 [1.050, 1.323] | 0.54 | 9.9 % | 75 / 1479 | 26.9 |

The CIs come from a log-clustered bootstrap (44 logs, 10 000 draws). Tables: `aud1/task1_runlevel.csv`, `aud1/runlevel.json` (also every
contributing run dir with its own `metrics_results`).

**Against the private suite and the leaderboard:**

| | score | at-fault km | GT m |
|:--|--:|--:|--:|
| APY10m10-AB-s0 local -> private | 0.935 -> **0.640** | 0.863 -> **0.132** (x 0.15) | 1.130 -> **2.288** (x 2.0) |
| WA-JEPA local -> WA-JEPA-class leaderboard entries | 0.895 -> 0.76 | 0.531 -> 0.63 (x 1.2) | 1.193 -> 1.85 (x 1.55) |
| host go-straight baseline, private | 0.799 | 0.258 | 1.15 |

What the private values imply, using the local per-clip structure (`aud1/task1_by_outcome.csv`). Clips below 4 m average 0.79 m and
clips at or above 4 m average 5.23 m, so a mean of 2.288 m needs about **34 % of private clips to leave the path by 4 m**, against 7.7 %
locally. At the local 28 m travelled per clip, 0.132 km per incident means about **21 % of private clips with an at-fault offroad or
collision**, against 3.2 % locally. Both figures fit a score of 0.64. They are approximate: the private scene lengths are unknown, but
the shared `e2e_challenge_nuplan_common/base.yaml` window, 30 + 80 steps at 0.1 s, is the one the private presets inherit, so 10
decisions per scene is likely there too. Every public scene has exactly 10.

So local GT distance is far below 2.3 m: **reading (b)**. On the private scenes the driver leaves the logged path about 4x as often
and causes at-fault incidents about 6x as often. The WA-JEPA-class entries get 1.55x the GT distance and *fewer* incidents per km than
our local WA-JEPA. The private suite is harder for everyone on the path metric, but our driver falls off much more. Caveat: the
leaderboard WA-JEPA-class entries are not proven to be our local reference configuration (precision, cold start).

## 2. Calibration diversity

The CAM_F0 calibration the driver received at `start_session` (K, R, t, distortion as `sh30_driver.Session` reads it) was taken from the
APY10m10-AB-s0 driver logs for every scene that started a session (1479; the 12 route-fold scenes never do). `aud1/task2_calib_by_vehicle.csv`:

- **Every scene has its own calibration key**: 1479 distinct `navsim_zs.calib_key` values (R, K, D at 4 decimals) in 1479 scenes. The
  intrinsics change per scene, even within one log. fx runs 1435-1623, and fy 755-1638 (fy / fx 0.53-1.07; 30 scenes below 0.90). cx,
  cy = 960, 560 everywhere except veh-08 (970.5, 578.4). Distortion is zero everywhere: MTGS renders an undistorted pinhole camera. (Corrected 2026-10-12, FAMILY: the request carries no distortion, but MTGS renders WITH the asset's colmap distortion, k1 about -0.34; the driver is never told. Measured harmless: telling it the distortion moves the score +0.003; ../../alpasim_family/results/dependence.md.)
- The mounting rotation is a per-vehicle / per-log property: 132 distinct rotations at 3 decimals across 16 vehicles and 44 logs. Yaw
  -1.78 to +2.24 deg (veh-48 about -1.7, veh-45 about +2.0, the rest within +-0.5), pitch 0.19-2.41 deg, translation x 1.57-1.81 m,
  height 1.49-1.58 m.
- So the warp has been exercised on 1479 distinct calibrations from 16 vehicles. Rotations span about 4 deg of yaw and 2 deg of
  pitch. Scores show no rig effect: `fy/fx` strata 0.909-0.956, and the vehicles that differ (veh-45, veh-48, rotated +-2 deg) are not
  the weak ones (task 3). A private suite from other nuPlan vehicles would have to differ a lot more to matter, if its reported
  calibration matches its rendering.

**Substitution run** (3 pool jobs on 4 veh-35 scenes, about 0.1 card-hour; `$DATA_DIR/runs/alpasim/aud1/calib/{control,veh48,veh45}/`,
`aud1/calib_compare.json`). The runtime's own camera override (`+runtime.extra_cameras=[{logical_id: CAM_F0, rig_to_camera, intrinsics}]`,
`camera_catalog.py`) replaced CAM_F0's K, R and t with a veh-48 scene's (yaw -1.76 deg, pitch 2.12 deg, fy 1261) or a veh-45 scene's
(yaw +1.91, pitch 2.30, fy 1600). Control: veh-35's own, yaw 0.15, pitch 1.44, fy 1475-1570. **Finding: the MTGS renderer ignores the
override.** The rendered CAM_F0 JPEGs are byte-identical to the control's in all 8 cases, but the driver is sent the donor calibration.
This run therefore measures a calibration **mismatch** (right images, wrong K / R / t), not a different rig rendered consistently.
That test is not possible on this renderer.

| scene (veh-35) | control score / GT m | veh-48 calib: score / GT m / max lateral change of plans 0-3 (m) | veh-45 calib: score / GT m / lateral change (m) |
|:--|--:|--:|--:|
| ..0c318d79 (straight, 8.9 m/s) | 1.00 / 0.29 | 0.79 / 2.26 / 0.73, 0.56, 0.19, 0.54 | 1.00 / 0.16 / 0.49, 0.28, 0.16, 0.05 |
| ..45e8b10f (straight, 12.6 m/s) | 1.00 / 0.18 | 0.74 / 0.48 / 1.26, 1.15, 0.37, 0.46 | 1.00 / 0.38 / 1.02, 0.56, 0.50, 0.32 |
| ..98017c16 (right) | 1.00 / 2.67 | 0.95 / 1.94 / 1.46, 0.72, 0.27, 0.40 | 1.00 / 3.09 / 1.36, 0.17, 0.09, 0.14 |
| ..fe800ded (left) | 1.00 / 0.38 | 0.92 / 0.65 / 0.20, 0.56, 0.38, 0.44 | 1.00 / 2.85 / 0.17, 0.88, 1.68, 1.20 |

The control reproduces the stored CF1 rollouts (first plans identical to the cm). Under a 2 deg yaw / 0.7-1 deg pitch mismatch the
first plans stay sane: no swerve, endpoints within 0.2-1.5 m laterally of the control. They shift by 0.5-1.5 m at 4 s, as about 2 deg
over 30 m predicts. The veh-48 calibration also shortens them (endpoint x 2-15 m shorter by plan 3, so lower progress: scores
0.74-0.95, no zero). The model frames differ from the control's by 15-24 grey levels on average. The driver is sensitive to the
calibration it is given, but nothing breaks. This matters only if a private rig's reported calibration does not match its rendering,
which this run cannot check.

## 3. Public score by start condition (`aud1/task3_strata.csv`)

Metadata per scene: initial ego speed |v| and longitudinal acceleration from WA-JEPA's first `drive` (the logged state at t = 0.017 s,
before policy control); the 4-way command and the first route waypoint (rig frame) from the ap2 driver's first decision; the logged
path length (`gt_dist_traveled_m`); the CAM_F0 calibration. Scene length is not a stratum: all 1479 scored scenes have 10 decisions.
"none" = the 12 route-fold scenes (0 for every driver).

| stratum | level | n | APY10m10-AB-s0 mean / zeros | P2H10-F-s0 mean / zeros | WA-JEPA mean / zeros |
|:--|:--|--:|--:|--:|--:|
| initial speed m/s | <1 | 241 | **0.925 / 16** | 0.915 / 9 | 0.940 / 9 |
| | 1-5 | 334 | 0.920 / 24 | 0.910 / 24 | 0.871 / 41 |
| | 5-10 | 719 | 0.951 / 27 | 0.928 / 39 | 0.896 / 72 |
| | >10 | 185 | 0.976 / 4 | 0.966 / 5 | 0.932 / 12 |
| command | left | 475 | 0.914 / 36 | 0.903 / 37 | 0.863 / 62 |
| | straight | 665 | 0.973 / 12 | 0.955 / 18 | 0.946 / 29 |
| | right | 339 | 0.923 / 23 | 0.905 / 22 | 0.871 / 43 |
| route lateral offset (first waypoint, abs y) | <2 m | 660 | 0.973 / 12 | 0.954 / 18 | 0.945 / 29 |
| | 2-10 m | 354 | 0.938 / 18 | 0.922 / 20 | 0.918 / 27 |
| | >10 m | 460 | 0.902 / 41 | 0.890 / 39 | 0.826 / 78 |
| first waypoint x (ahead) | behind (<0) | 4 | 0.500 / 2 | 0.485 / 2 | 0.250 / 3 |
| | beside / near (0-20 m) | 111 | **0.835 / 18** | 0.850 / 14 | 0.746 / 27 |
| | 20-35 m | 238 | 0.906 / 20 | 0.890 / 21 | 0.826 / 41 |
| | >=35 m (normal) | 1121 | 0.962 / 31 | 0.943 / 40 | 0.935 / 63 |
| route waypoints at t0 | <10 | 53 | 0.895 / 5 | 0.916 / 3 | 0.810 / 9 |
| | 10 | 1426 | 0.944 / 66 | 0.927 / 74 | 0.905 / 125 |
| logged path length | <5 m | 29 | **0.759 / 7** | 0.931 / 2 | 0.966 / 1 |
| | 5-20 m | 376 | 0.921 / 27 | 0.906 / 23 | 0.892 / 34 |
| | 20-40 m | 582 | 0.936 / 28 | 0.912 / 38 | 0.894 / 59 |
| | 40-60 m | 408 | 0.974 / 9 | 0.955 / 13 | 0.915 / 33 |
| | >60 m | 84 | 0.997 / 0 | 0.985 / 1 | 0.917 / 7 |
| initial accel m/s^2 | braking < -1 | 80 | **0.885 / 8** | 0.919 / 5 | 0.820 / 14 |
| | -1..1 | 1303 | 0.944 / 61 | 0.924 / 70 | 0.908 / 109 |
| | accel > 1 | 96 | 0.973 / 2 | 0.967 / 2 | 0.884 / 11 |

The per-vehicle and fy/fx strata are in the CSV. Weakest vehicles for APY: veh-45 0.892, veh-52 0.903, veh-48 0.909, veh-40 / veh-42
0.918. APY is at or above WA-JEPA on every vehicle except veh-28 (-0.017; veh-25 and veh-26 tie).

Weak strata for the submitted driver:

- **Scenes where the logged car hardly moves (logged path < 5 m, 29 scenes): 0.759, 7 zeros (5 at-fault collisions, 2 corridor exits)**,
  against 2 for P2H10-F-s0 and 1 for WA-JEPA. APY drives 7.2 m on average where the log drives 3.7 m (P2H10 6.1, WA-JEPA 4.8). It moves
  off into stopped traffic. The same shows in starts from standstill (v0 < 1 m/s: 16 zeros against 9 / 9) and in braking starts (8
  against 5). This is the only place where APY is worse than both other drivers. A private suite richer in queue / stop-line scenes
  would hit exactly this, and at-fault collisions are what the private at-fault distance shows.
- Route first waypoint beside / near the ego (0-20 m) 0.835 and large lateral offsets / turns (> 10 m) 0.902 are the hardest strata for
  every driver. APY is better than WA-JEPA there.
- None of these is large enough to produce 0.64 by composition. The worst stratum with n >= 20 is 0.759 (the < 5 m one). Even a private
  suite made only of turns (0.90-0.92), near routes (0.84) and standstill starts (0.93) would land around 0.85-0.90. The private gap
  needs behaviour the public scenes do not show (reading b). The strata point to where to look first: standstill / queue starts and
  collisions with stopped traffic.

## What was run

- Read only: stored `aggregate/results-summary.json` and `driver-logs/drive.jsonl` of the CF1 / OT3 runs (APY10m10-AB-s0, P2H10-F-s0;
  manifests `ot3/a`, `cf1/a`) and of C0b / C0c (WA-JEPA, `c0c/manifest.json`). No closed loop was rerun for tasks 1 and 3.
- Task 2: 3 pool jobs (`alpasim-aud1-{control,veh48,veh45}`, `run.sh <dir> ap2`, `AP2_TAG=APY10m10-AB-s0 SH30_DUMP=4`, the CF1
  overrides, 4 scenes each; about 100 s each). Plus 2 jobs that failed at once on a malformed override (the runtime requires 6
  radial coefficients). Under 0.15 card-hour in total.
