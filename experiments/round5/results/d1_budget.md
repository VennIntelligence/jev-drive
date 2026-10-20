# ROUND5 D1: where the remaining PAI points are, and which a training-data change can reach (2026-10-21)

Diagnosis, no training, no GPU: stored reads only. Inputs: PAI 281 ACC + GAP reads of NTV0-L18-s0 (the stored d366 read A0), R2C-L18-s0 and
R2J-L18-s0 (round 2), with their rf_tax extracts and pai_fix facts; the PAI-FIX native (VC + LEAD) reads of NTV0 / R2C / R2J / R3JW / R3JNF; the
21 stored 281 reads of pai_fix calib (`calib_scores.csv.gz`); every inference decision of the eight reads' driver logs (model plan `poses_model`,
served plan, ACC / lead fields), extracted on the box by `scripts/d1.py extract` (pool CPU job). Map coverage of the curated-train scenes:
`scripts/d1.py mapcov` (pool CPU job, alpasim env). Read-outs: `scripts/d1_mac.py all` (Mac). 1 point = 1 scene; partials count 1 - score.
Classes are pai_fix.py `fclass`; `zero-other` here is mostly `cor_drift` (corridor left laterally on a straight road) and `cor_lane_change`.

Per-scene files: [d1_scenes_281.csv](d1_scenes_281.csv) (scores, classes, best-of), [d1_lateral.csv](d1_lateral.csv),
[d1_launch.csv](d1_launch.csv), [d1_overshoot.csv](d1_overshoot.csv), [d1_budget_281.csv](d1_budget_281.csv), [d1_budget_fix.csv](d1_budget_fix.csv),
[d1_mapcov_train.csv](d1_mapcov_train.csv) (NuRec map coverage of the 853 curated-train scenes on the box).

## 1. Point budget
### 1a. PAI 281 under ACC + GAP: points lost per class (1 point = 1 scene; partials count 1 - score)

280 scenes all three reads score without an RPC error. `reach any` = points that the per-scene best of the stored reads (median 24 reads a scene: calib's 21 stored 281 reads incl. 6 native / rerun rows, R2C / R2J 281 AG, PAI-FIX native on the T281 scenes) gets above this read; `reach AG` = the same over ACC + GAP reads only (the submission serve); `never` = lost - reach any (no stored read gets it).

| class | NTV0 281 AG scenes / lost / reach any / reach AG / never | R2C 281 AG scenes / lost / reach any / reach AG / never | R2J 281 AG scenes / lost / reach any / reach AG / never |
|---|--:|--:|--:|
| overshoot | 23 / 23.0 / 8.0 / 3.0 / 15.0 | 22 / 22.0 / 7.0 / 3.0 / 15.0 | 21 / 21.0 / 6.0 / 2.0 / 15.0 |
| route-turn | 15 / 15.0 / 12.0 / 10.1 / 3.0 | 15 / 15.0 / 11.9 / 8.9 / 3.1 | 2 / 2.0 / 0.0 / 0.0 / 2.0 |
| offroad | 12 / 12.0 / 6.9 / 5.9 / 5.1 | 11 / 11.0 / 6.1 / 5.0 / 4.9 | 20 / 20.0 / 15.0 / 12.0 / 5.0 |
| AF-front | 3 / 3.0 / 2.0 / 2.0 / 1.0 | 4 / 4.0 / 3.0 / 3.0 / 1.0 | 8 / 8.0 / 6.0 / 6.0 / 2.0 |
| side | 0 / 0.0 / 0.0 / 0.0 / 0.0 | 1 / 1.0 / 1.0 / 1.0 / 0.0 | 0 / 0.0 / 0.0 / 0.0 / 0.0 |
| zero-other | 6 / 6.0 / 4.2 / 3.0 / 1.8 | 7 / 7.0 / 4.8 / 3.5 / 2.2 | 13 / 13.0 / 10.8 / 9.5 / 2.2 |
| rear-hit | 26 / 10.4 / 6.1 / 6.0 / 4.3 | 25 / 10.8 / 6.4 / 6.4 / 4.4 | 33 / 19.1 / 14.4 / 13.4 / 4.6 |
| slow | 11 / 2.2 / 1.4 / 1.2 / 0.8 | 10 / 1.7 / 1.4 / 1.2 / 0.3 | 12 / 4.1 / 4.0 / 3.8 / 0.1 |
| hit | 0 / 0.0 / 0.0 / 0.0 / 0.0 | 0 / 0.0 / 0.0 / 0.0 / 0.0 | 0 / 0.0 / 0.0 / 0.0 / 0.0 |
| **all** | 96 / 71.6 / 40.7 / 31.2 / 30.9 | 95 / 72.5 / 41.6 / 32.1 / 30.9 | 109 / 87.2 / 56.2 / 46.7 / 30.9 |

Ceiling: of 280 scenes, some stored read scores >= 0.98 on 237, only partially on 20, and **no stored read scores at all on 23**; per-scene best-of mean 0.890 (ACC + GAP reads only 0.856: solved 227, zero in every read 34). The human log scores 1.000 on 277 / 277 (d381).

### 1b. NTV0-L18-s0 + ACC + GAP (the PAI candidate), classes ranked by points lost: reachable by some stored checkpoint vs never

| rank | class | lost | reachable (any read) | reachable (AG reads) | never got by any read | scenes where some read scores >= 0.98 |
|--:|---|--:|--:|--:|--:|--:|
| 1 | overshoot | 23.0 | 8.0 | 3.0 | 15.0 | 8 / 23 |
| 2 | route-turn | 15.0 | 12.0 | 10.1 | 3.0 | 12 / 15 |
| 3 | offroad | 12.0 | 6.9 | 5.9 | 5.1 | 5 / 12 |
| 4 | rear-hit | 10.4 | 6.1 | 6.0 | 4.3 | 16 / 26 |
| 5 | zero-other | 6.0 | 4.2 | 3.0 | 1.8 | 3 / 6 |
| 6 | AF-front | 3.0 | 2.0 | 2.0 | 1.0 | 2 / 3 |
| 7 | slow | 2.2 | 1.4 | 1.2 | 0.8 | 7 / 11 |
| 8 | side | 0.0 | 0.0 | 0.0 | 0.0 | 0 / 0 |
| 9 | hit | 0.0 | 0.0 | 0.0 | 0.0 | 0 / 0 |

### 1c. PAI-FIX native (VC + LEAD) reads: points lost per class and subset

| read | subset | n | mean | lost | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 fix VL | TARGET | 60 | 0.497 | 30.2 | 12.0 | 4.0 | 5.0 | 4.0 | 1.0 | 0.0 | 2.5 | 1.6 | 0.0 |
| NTV0 fix VL | G-LAUNCH | 6 | 0.856 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | 0.2 | 0.0 |
| NTV0 fix VL | G-LEAD | 6 | 0.748 | 1.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.5 | 0.0 | 0.0 |
| NTV0 fix VL | T281 | 46 | 0.144 | 39.4 | 16.0 | 13.0 | 9.0 | 0.0 | 0.0 | 0.0 | 0.7 | 0.7 | 0.0 |
| R2C fix VL | TARGET | 60 | 0.523 | 28.6 | 13.0 | 5.0 | 3.0 | 4.0 | 0.0 | 0.0 | 2.1 | 1.5 | 0.0 |
| R2C fix VL | G-LAUNCH | 6 | 0.865 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6 | 0.2 | 0.0 |
| R2C fix VL | G-LEAD | 6 | 0.765 | 1.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.4 | 0.0 | 0.0 |
| R2C fix VL | T281 | 46 | 0.127 | 40.1 | 18.0 | 12.0 | 9.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |
| R2J fix VL | TARGET | 60 | 0.511 | 29.4 | 9.0 | 0.0 | 6.0 | 4.0 | 1.0 | 2.0 | 3.5 | 3.9 | 0.0 |
| R2J fix VL | G-LAUNCH | 6 | 0.535 | 2.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.8 | 0.0 | 0.0 |
| R2J fix VL | G-LEAD | 6 | 0.843 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | 0.3 | 0.0 |
| R2J fix VL | T281 | 46 | 0.279 | 33.1 | 13.0 | 3.0 | 8.0 | 4.0 | 0.0 | 0.0 | 2.7 | 2.5 | 0.0 |
| R3JW fix VL | TARGET | 60 | 0.474 | 31.5 | 12.0 | 1.0 | 6.0 | 2.0 | 1.0 | 2.0 | 4.4 | 3.2 | 0.0 |
| R3JW fix VL | G-LAUNCH | 6 | 0.517 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.0 | 0.0 |
| R3JW fix VL | G-LEAD | 6 | 0.882 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.2 | 0.0 |
| R3JW fix VL | T281 | 46 | 0.301 | 32.1 | 13.0 | 1.0 | 9.0 | 4.0 | 1.0 | 0.0 | 2.6 | 1.6 | 0.0 |
| R3JNF fix VL | TARGET | 60 | 0.527 | 28.4 | 8.0 | 2.0 | 6.0 | 2.0 | 1.0 | 1.0 | 4.5 | 3.8 | 0.0 |
| R3JNF fix VL | G-LAUNCH | 6 | 0.508 | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.9 | 0.0 | 0.0 |
| R3JNF fix VL | G-LEAD | 6 | 0.797 | 1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 | 0.4 | 0.0 |
| R3JNF fix VL | T281 | 46 | 0.276 | 33.3 | 13.0 | 3.0 | 9.0 | 4.0 | 0.0 | 0.0 | 2.7 | 1.6 | 0.0 |

### 1d. What R2J changes against NTV0 on PAI 281 (points, per scene, ACC + GAP)

| | points | scenes | by class (points) |
|---|--:|--:|---|
| R2J better than NTV0 (by NTV0's class) | +22.6 | 42 | route-turn 9.1 (11), offroad 3.9 (4), rear-hit 3.6 (16), overshoot 2.0, zero-other 2.0, AF-front 1.0, slow 1.0 |
| R2J worse than NTV0 (by R2J's class) | -38.1 | 60 | rear-hit 10.3 (24), offroad 8.6 (9), zero-other 8.6 (9: drift / lane change), AF-front 5.0 (5), slow 3.6 (10), overshoot 2.0 |
| per-scene max(NTV0, R2J) | mean 0.825 | | NTV0 0.744, R2J 0.689; max(NTV0, R2C, R2J) 0.833 |

## 2. Offroad and lateral anatomy (PAI 281 ACC + GAP)

Every offroad zero plus every lateral corridor zero (`cor_drift`, `cor_lane_change`). Event = first offroad sample (else first corridor exit).
Geometry from the rollout against the human log (station / lateral on the log polyline) and the extract's map lanes. Turn classes: the log's
heading change over the 30 m around the event station >= 25 deg (junction / curve) or 10-25 deg (bend); wide = the ego is outside the turn,
cut = inside. Straight (< 10 deg): `edge` = the ego is under 3 m off the log (a road-edge / shoulder exit); at |lat| >= 3 m (one lane over)
the lane distance tells who crossed a lane line in the last 10 s (max distance to the nearest aligned lane centre > 1.2 m). Almost none follow a
stop, a rear-hit or a late arrival: in `d1_lateral.csv` `ctx` is empty for 17 / 18 (NTV0) and 32 / 33 (R2J); the exceptions are one late
arrival (NTV0 `a68ad579`, +3 s) and one start from standing (R2J `ea468790`).

### 2a. Lateral zeros by geometry (PAI 281 ACC + GAP)

| lateral zero (offroad + cor_drift + lane change), geometry class | NTV0 | R2C | R2J |
|---|--:|--:|--:|
| bend cut | 2 | 3 | 5 |
| bend wide | 0 | 0 | 1 |
| curve turn cut | 0 | 0 | 1 |
| junction turn cut | 2 | 1 | 2 |
| junction turn wide | 3 | 4 | 2 |
| lane change | 1 | 2 | 2 |
| straight: both crossed | 1 | 0 | 4 |
| straight: edge (|lat| < 3 m) | 5 | 3 | 8 |
| straight: ego left its lane | 0 | 1 | 5 |
| straight: human changed lane, ego kept | 4 | 4 | 3 |
| all | 18 | 18 | 33 |

### 2b. R2J's 21 lateral zeros that NTV0 does not have (NTV0 / R2C class on the same scene; plan at the decision 2.5 s before the event, lateral vs the log; lane distance = max over the last 10 s to the nearest aligned lane centre)

| R2J new lateral zero | R2J_class | NTV0 | R2C | v_ego | lat_ev | lat_grow_s | plan_lat4 | plan_dlat | ego_lane_max | hum_lane_max |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4e4a2a51 | bend cut | overshoot | overshoot | 4.24 | -1.14 | 12.50 | -2.97 | -4.52 | 1.84 | 1.35 |
| baabc57d | bend cut | ok | ok | 5.11 | -1.26 | 8.60 | -1.54 | -1.13 | 1.59 | 2.23 |
| e1c6ae9f | bend cut | ok | ok | 10.01 | 2.65 | 8.40 | 0.24 | -2.08 | 3.48 | 0.57 |
| abd45a30 | bend wide | route-turn | route-turn | 5.18 | -0.71 | 0.30 | 0.32 | 0.35 | 0.42 | 0.33 |
| ea468790 | curve turn cut | ok | slow | 6.17 | -1.53 | 0.50 | -0.65 | -1.04 | 3.10 | 0.42 |
| b680a689 | junction turn cut | ok | ok | 4.57 | 2.55 | 1.20 | -2.78 | -2.54 | 2.62 | 1.70 |
| 98f88a4e | lane change | rear-hit | zero-other | 13.02 | 3.64 | 1.10 | -0.15 | 0.29 | 2.31 | 0.92 |
| 13d646af | straight: both crossed | ok | ok | 16.65 | 3.89 | 6.60 | 3.43 | 4.14 | 3.63 | 2.34 |
| 596af0c7 | straight: both crossed | rear-hit | rear-hit | 18.11 | -3.38 | 2.00 | -1.61 | -1.46 | 1.60 | 1.60 |
| 8a143cb2 | straight: both crossed | ok | ok | 2.58 | -3.74 | 8.90 | -2.59 | -0.41 | 2.78 | 2.95 |
| cdf95fd1 | straight: both crossed | ok | ok | 8.32 | -3.97 | 4.90 | -4.00 | -1.34 | 2.20 | 1.66 |
| 0f4b8753 | straight: edge (|lat| < 3 m) | ok | ok | 4.37 | -1.60 | 14.00 | -0.96 | -1.33 | 3.25 | 2.03 |
| 6e7b42a3 | straight: edge (|lat| < 3 m) | ok | ok | 36.87 | -0.31 | 8.40 | -0.49 | -0.91 | 0.77 | 0.24 |
| 7750265d | straight: edge (|lat| < 3 m) | overshoot | overshoot | 11.71 | 0.40 | 0.00 | -1.05 | -1.21 | 1.35 | 1.33 |
| a2b415d5 | straight: edge (|lat| < 3 m) | ok | ok | 16.63 | -0.90 | 6.70 | -0.78 | -0.13 | 0.63 | 0.39 |
| db8693db | straight: edge (|lat| < 3 m) | ok | ok | 7.66 | 0.05 | 11.10 | 0.10 | -0.21 | 0.62 | 0.46 |
| 1e31b2ad | straight: ego left its lane | ok | ok | 16.28 | -3.96 | 8.90 | -2.91 | -3.52 | 1.64 | 0.31 |
| 4a863a41 | straight: ego left its lane | ok | ok | 25.29 | 3.99 | 6.60 | 3.06 | 0.51 | 1.76 | 0.59 |
| 4cb05d65 | straight: ego left its lane | ok | ok | 38.53 | 4.01 | 8.20 | 3.13 | -0.01 | 1.77 | 0.36 |
| 8120803d | straight: ego left its lane | ok | ok | 28.48 | -3.89 | 2.30 | -3.08 | -2.69 | 1.52 | 0.49 |
| b27b3375 | straight: ego left its lane | ok | ok | 22.21 | 4.00 | 3.20 | 3.99 | 2.72 | 1.85 | 0.43 |

### 2c. NuRec vector maps of the curated-train scenes (the evaluator's offroad surface)

- scenes with a vector map: 853 / 853; lanes per scene median 241 (min 32, max 1514)
- human footprint inside the lane union over the clip (moving scenes, 846): min margin >= -0.3 m in 85.3%, >= -0.5 m in 90.0%; share of poses inside median 0.963
- cost: map load 0.08 s, lane union 0.027 s per scene; a 128 x 96 signed-distance raster (shapely, one core) 0.28 s median (86 timed scenes, p90 0.49 s)

## 3. Launch anatomy

t_L = the human log is 0.5 m past its handoff pose (round 3's definition); launch scenes = handoff speed < 1 m/s and the log moves > 5 m. From
t_L: `ego_move_s` = when the ego is 0.5 m past its handoff pose (negative = before the human); `plan_1s / 2s / 4s` = median arc of the model plan at
1 / 2 / 4 s over the decisions in [t_L, t_L + 4 s]; `deferred` = share of those decisions standing (< 0.3 m/s) with a plan that covers < 0.3 m
in 1 s but >= 2 m by 4 s ("go later", re-planned every 0.1 s); `ego_4s` / `human_4s` = distance along the log after 4 s; `stall` = the ego moves
< 1 m in 4 s while the human moves > 3 m; `fol_gap0` = bumper gap to the nearest logged vehicle behind the ego at t_L (replayed follower);
`rear_after_s` = rear collision time after t_L; `v_at_rear`, `ego_d_rear` / `human_d_rear` = ego speed and distances at that moment.

### 3. PAI 281 ACC + GAP, all 37 launch scenes

| from the human's launch t_L | NTV0 | R2C | R2J |
|---|--:|--:|--:|
| n | 37.00 | 37.00 | 37.00 |
| launched | 33.00 | 33.00 | 21.00 |
| ego_move_s | -1.08 | -1.08 | 0.01 |
| plan_1s | 3.30 | 3.29 | 1.36 |
| plan_2s | 7.46 | 7.44 | 3.43 |
| plan_4s | 15.99 | 18.24 | 8.78 |
| deferred | 0.08 | 0.08 | 0.24 |
| ego_4s | 12.15 | 11.41 | 5.60 |
| human_4s | 12.90 | 12.90 | 12.90 |
| ego_v2 | 2.52 | 2.48 | 1.47 |
| human_v2 | 3.64 | 3.64 | 3.64 |
| stall | 0.00 | 0.00 | 5.00 |
| rear_hit | 11.00 | 13.00 | 19.00 |
| fol_gap0 | 6.04 | 5.87 | 5.33 |
| rear_after_s | 8.30 | 7.37 | 6.14 |
| v_at_rear | 2.02 | 1.95 | 0.27 |
| ego_d_rear | 39.09 | 22.40 | 5.75 |
| human_d_rear | 30.17 | 24.36 | 11.61 |

### 3. PAI 281 ACC + GAP, the 13 launches R2C makes and R2J loses

| from the human's launch t_L | NTV0 | R2C | R2J |
|---|--:|--:|--:|
| n | 13.00 | 13.00 | 13.00 |
| launched | 13.00 | 13.00 | 0.00 |
| ego_move_s | -1.19 | -1.19 | 0.04 |
| plan_1s | 2.34 | 2.34 | 0.10 |
| plan_2s | 5.32 | 5.30 | 0.32 |
| plan_4s | 11.30 | 11.28 | 3.58 |
| deferred | 0.21 | 0.22 | 0.61 |
| ego_4s | 8.53 | 8.50 | 1.54 |
| human_4s | 8.22 | 8.22 | 8.22 |
| ego_v2 | 1.30 | 1.31 | 0.03 |
| human_v2 | 2.91 | 2.91 | 2.91 |
| stall | 0.00 | 0.00 | 5.00 |
| rear_hit | 1.00 | 2.00 | 9.00 |
| fol_gap0 | 5.72 | 4.92 | 4.11 |
| rear_after_s | 5.42 | 6.45 | 5.87 |
| v_at_rear | 0.92 | 0.13 | 0.02 |
| ego_d_rear | 80.31 | 42.67 | 1.90 |
| human_d_rear | 23.65 | 17.93 | 9.72 |

### 3. PAI-FIX G-LAUNCH (6 scenes, native VC + LEAD)

| from the human's launch t_L | NTV0 | R2C | R2J | R3JW | R3JNF |
|---|--:|--:|--:|--:|--:|
| n | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| launched | 5.00 | 5.00 | 3.00 | 3.00 | 3.00 |
| ego_move_s | -0.02 | -0.02 | 0.33 | 0.38 | 0.33 |
| plan_1s | 2.19 | 2.19 | 1.26 | 0.88 | 1.02 |
| plan_2s | 5.79 | 5.75 | 3.36 | 2.47 | 2.79 |
| plan_4s | 16.11 | 15.71 | 9.82 | 8.10 | 8.55 |
| deferred | 0.09 | 0.07 | 0.18 | 0.21 | 0.18 |
| ego_4s | 6.44 | 6.47 | 4.41 | 3.41 | 3.87 |
| human_4s | 13.16 | 13.16 | 13.16 | 13.16 | 13.16 |
| ego_v2 | 1.39 | 1.40 | 0.94 | 0.68 | 0.80 |
| human_v2 | 3.08 | 3.08 | 3.08 | 3.08 | 3.08 |
| stall | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| rear_hit | 1.00 | 1.00 | 4.00 | 4.00 | 4.00 |
| fol_gap0 | 3.69 | 3.69 | 3.71 | 3.69 | 3.70 |
| rear_after_s | 7.31 | 7.61 | 10.31 | 9.36 | 9.76 |
| v_at_rear | 6.31 | 6.67 | 0.23 | 0.47 | 0.23 |
| ego_d_rear | 19.73 | 21.89 | 6.90 | 3.36 | 4.79 |
| human_d_rear | 39.88 | 42.79 | 12.64 | 10.94 | 12.84 |

### 3d. The lost launches per scene

| scene, read | fclass | ego_move | win_pm_a1 | win_pm_a2 | win_pm_a4 | win_defer | e_d4 | h_d4 | fol_gap0 | rear_t | v_at_rear |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 22fa5c92 R2C | ok | -5.17 | 4.34 | 8.88 | 18.24 | 0.00 | 40.73 | 10.85 | 4.11 | nan | nan |
| 22fa5c92 R2J | slow | -4.17 | 0.16 | 0.32 | 3.19 | 1.00 | 0.87 | 10.85 | nan | nan | nan |
| 52bd89ce R2C | AF-front | 0.04 | 0.55 | 1.43 | 5.02 | 0.00 | 2.29 | 2.16 | 3.14 | nan | nan |
| 52bd89ce R2J | rear-hit | 0.04 | 0.45 | 1.12 | 3.89 | 0.05 | 2.22 | 2.16 | 3.14 | 6.14 | 0.38 |
| 68e53160 R2C | ok | -7.18 | 8.15 | 15.02 | 26.76 | 0.00 | 79.09 | 15.59 | nan | 5.52 | 0.19 |
| 68e53160 R2J | rear-hit | 4.72 | 0.09 | 0.32 | 4.32 | 1.00 | 0.29 | 15.59 | 3.08 | 2.72 | 0.02 |
| 76f7e025 R2C | ok | -5.84 | 14.62 | 29.31 | 58.42 | 0.00 | 86.77 | 27.11 | nan | nan | nan |
| 76f7e025 R2J | rear-hit | 6.56 | 0.10 | 0.22 | 3.42 | 1.00 | 0.20 | 27.11 | 2.12 | 2.16 | 0.00 |
| 7c6529f6 R2C | rear-hit | -9.83 | 0.11 | 0.79 | 4.58 | 0.93 | 3.32 | 6.00 | 6.05 | 7.37 | 0.07 |
| 7c6529f6 R2J | rear-hit | -9.83 | 0.06 | 0.29 | 2.00 | 0.50 | 2.19 | 6.00 | 5.07 | 5.87 | 0.03 |
| 9fa1d6eb R2C | ok | -0.68 | 2.34 | 5.30 | 11.28 | 0.00 | 8.50 | 8.22 | 3.15 | nan | nan |
| 9fa1d6eb R2J | rear-hit | -0.68 | 0.33 | 0.48 | 3.58 | 0.33 | 4.57 | 8.22 | 3.15 | 4.52 | 0.02 |
| a1b166cc R2C | ok | -5.71 | 5.60 | 11.78 | 24.69 | 0.00 | 48.59 | 14.69 | nan | nan | nan |
| a1b166cc R2J | rear-hit | 5.29 | 0.10 | 0.23 | 3.37 | 1.00 | 0.19 | 14.69 | 5.72 | 2.08 | 0.00 |
| c8cab401 R2C | ok | -0.19 | 1.67 | 3.35 | 7.29 | 0.00 | 5.28 | 4.75 | 5.72 | nan | nan |
| c8cab401 R2J | rear-hit | 0.01 | 1.36 | 2.98 | 6.90 | 0.00 | 5.08 | 4.75 | 5.61 | 8.61 | 0.04 |
| ca173de1 R2C | ok | -1.58 | 8.92 | 18.61 | 39.37 | 0.00 | 34.62 | 18.97 | nan | nan | nan |
| ca173de1 R2J | rear-hit | 14.22 | 0.08 | 0.25 | 4.59 | 1.00 | 0.09 | 18.97 | nan | 6.22 | 0.00 |
| e37aa159 R2C | route-turn | -0.07 | 0.08 | 0.41 | 4.88 | 0.75 | 1.55 | 1.01 | 8.66 | nan | nan |
| e37aa159 R2J | rear-hit | -0.07 | 0.09 | 0.24 | 3.05 | 0.72 | 1.49 | 1.01 | 8.66 | 9.73 | 0.00 |
| ea468790 R2C | slow | 0.04 | 0.27 | 0.88 | 4.94 | 0.40 | 2.08 | 1.06 | nan | nan | nan |
| ea468790 R2J | offroad | 0.04 | 0.30 | 0.81 | 4.41 | 0.38 | 2.15 | 1.06 | nan | nan | nan |
| f1069db5 R2C | ok | -1.19 | 4.90 | 11.13 | 26.71 | 0.00 | 17.59 | 21.20 | nan | nan | nan |
| f1069db5 R2J | slow | 1.31 | 0.36 | 0.88 | 4.65 | 0.25 | 1.64 | 21.20 | nan | nan | nan |
| fda39805 R2C | ok | 0.05 | 0.08 | 0.41 | 4.72 | 0.75 | 1.54 | 0.99 | nan | nan | nan |
| fda39805 R2J | slow | 0.05 | 0.06 | 0.20 | 3.15 | 0.75 | 1.54 | 0.99 | nan | nan | nan |

### 3e. Training labels at standstill and at launch (`scripts/d1.py labels`, box; label = the row's 4 s future)

| pool | standstill rows (v0 < 0.3) | stay (4 s arc < 0.5 m) | stand then go (1 s < 0.3 m, 4 s >= 2 m) | go now (1 s >= 0.3 m) | launch rows (v0 < 1, 4 s >= 2 m): label arc 1 / 2 / 4 s (m) |
|---|--:|--:|--:|--:|---|
| on1_PAIH (plain) | 16 946 | 10 434 (61.6 %) | 3 722 (22.0 %) | 950 (5.6 %) | 0.31 / 1.34 / 6.34 |
| on1_PAIH05 (f = 0.5) | 17 013 | 12 926 (76.0 %) | 1 251 (7.4 %) | 214 (1.3 %) | 0.58 / 1.45 / 3.62 |
| on1_PAIH15 (f = 1.5) | 10 410 | 4 238 (40.7 %) | 4 069 (39.1 %) | 1 361 (13.1 %) | 0.20 / 1.64 / 11.25 |
| navtrain (navtrain_full, 103 288 rows) | 6 693 | 22 (0.3 %) | 3 452 (51.6 %) | 3 216 (48.0 %) | 0.71 / 2.33 / 8.77 |

Lane-change-shaped moving labels (v0 > 5 m/s, 4 s point |y| > 2.5 m with |yaw| < 12 deg): on1_PAIH 10 710 of 90 109 (11.9 %), navtrain 3 549 of
44 512 (8.0 %); of the PAIH ones with a route label (10 703), the route polyline passes within 1.2 m of the label's 4 s point in 56 %.

## 4. Overshoot anatomy (J family vs NTV0 / R2C)

Overshoot zeros (cor_route_end: the human ends still, the ego leaves the corridor past the log's end; with the human braking over >= 15 m, which
drops 1-3 creep / standstill rests per read). Human rest station = first log time below 0.3 m/s that stays below 0.5; human brake onset = start of
the final deceleration (1 s smoothed speed, deceleration > 0.4 m/s^2). At 40 / 20 / 10 m before the human's rest station: human speed `h_v`,
ego speed `e_v`, the model plan's 4 s end against the rest station `pl_end` (+ = past it) and its end speed `pl_v4`. `plan_has_stop` = scenes with
any decision in the last 60 m whose model plan ends below 0.5 m/s; `first_stop_plan_m` = distance to the rest station at the first such plan;
`stop_plan_end_m` = where those plans stop (median, + = past the human); `served_over_model` = served / model 4 s arc in the last 40 m.

### 4. PAI 281 ACC + GAP

| overshoot zeros (human brakes >= 15 m) | NTV0 | R2C | R2J |
|---|--:|--:|--:|
| n | 21.00 | 20.00 | 20.00 |
| plan_has_stop | 1.00 | 1.00 | 9.00 |
| human_brake_m | 42.31 | 43.18 | 42.51 |
| human_v_brake | 10.73 | 10.85 | 10.78 |
| h_v@40 | 10.51 | 10.55 | 10.55 |
| e_v@40 | 10.67 | 10.84 | 10.79 |
| pl_end@40 | -1.45 | -0.93 | -3.64 |
| pl_v4@40 | 7.49 | 7.42 | 6.53 |
| h_v@20 | 8.06 | 8.13 | 8.03 |
| e_v@20 | 10.53 | 10.63 | 9.63 |
| pl_end@20 | 18.81 | 19.53 | 5.23 |
| pl_v4@20 | 8.83 | 8.19 | 3.28 |
| h_v@10 | 5.67 | 5.68 | 5.66 |
| e_v@10 | 10.80 | 10.87 | 8.08 |
| pl_end@10 | 33.91 | 35.90 | 9.79 |
| pl_v4@10 | 9.76 | 10.68 | 2.92 |
| plan_brake_m | 35.58 | 34.89 | 38.21 |
| first_stop_plan_m | -3.34 | -2.40 | 15.89 |
| stop_plan_end_m | 11.41 | 9.61 | 2.34 |
| ego_v_at_rest | 10.93 | 11.31 | 6.91 |
| served_over_model | 1.00 | 0.99 | 1.00 |

### 4. PAI-FIX native (T281-STOP + STOP + STOP-L)

| overshoot zeros (human brakes >= 15 m) | NTV0 | R2C | R2J | R3JW | R3JNF |
|---|--:|--:|--:|--:|--:|
| n | 26.00 | 28.00 | 21.00 | 23.00 | 19.00 |
| plan_has_stop | 0.00 | 2.00 | 9.00 | 10.00 | 8.00 |
| human_brake_m | 43.93 | 43.93 | 44.05 | 44.05 | 44.05 |
| human_v_brake | 10.77 | 10.77 | 10.73 | 10.80 | 10.73 |
| h_v@40 | 10.48 | 10.48 | 10.51 | 10.51 | 10.56 |
| e_v@40 | 12.12 | 12.03 | 11.68 | 11.52 | 11.91 |
| pl_end@40 | 4.40 | 3.37 | -3.38 | -4.32 | -2.70 |
| pl_v4@40 | 7.49 | 7.68 | 6.08 | 6.50 | 6.40 |
| h_v@20 | 7.92 | 7.92 | 7.96 | 7.88 | 7.99 |
| e_v@20 | 12.52 | 12.42 | 9.73 | 9.79 | 9.74 |
| pl_end@20 | 28.21 | 28.43 | 2.46 | 6.14 | 2.88 |
| pl_v4@20 | 10.77 | 10.56 | 3.32 | 4.30 | 4.28 |
| h_v@10 | 5.50 | 5.50 | 5.50 | 5.49 | 5.50 |
| e_v@10 | 12.70 | 12.85 | 7.50 | 7.95 | 8.17 |
| pl_end@10 | 41.83 | 41.87 | 11.18 | 10.63 | 12.27 |
| pl_v4@10 | 12.57 | 12.38 | 3.78 | 3.04 | 4.87 |
| plan_brake_m | 38.01 | 38.94 | 43.55 | 42.73 | 43.55 |
| first_stop_plan_m | nan | 6.59 | 13.99 | 13.62 | 12.97 |
| stop_plan_end_m | nan | 2.10 | 4.33 | 4.87 | 5.07 |
| ego_v_at_rest | 13.10 | 13.09 | 6.34 | 5.78 | 6.25 |
| served_over_model | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |

## Reading

1. **Budget.** NTV0 + ACC + GAP loses 71.6 points on 280 scenes. Some stored read gets 40.7 of them (31.2 under ACC + GAP alone); 30.9 no stored
   read ever gets (23 scenes score 0 in all reads, 15 of them stop-line overshoots). By points reachable from our own checkpoints: route-turn 12.0
   (R2J already takes 13 of 15), overshoot 8.0 (but only 3.0 under ACC + GAP), offroad 6.9, rear-hit 6.1, zero-other 4.2, AF-front 2.0, slow 1.4.
   By points nobody gets: overshoot 15.0, offroad 5.1, rear-hit 4.3, route-turn 3.0. R2J's turn and stop gains are real (+22.6 points, max(NTV0,
   R2J) = 0.825) but it pays 38.1 points elsewhere: rear-hit / slow 13.9, offroad + drift / lane 17.2, AF-front 5.0.
2. **Offroad.** J does not lose junction turns: its lateral zeros grow on straight roads (straight classes 10 -> 20, all lateral 18 -> 33) and
   on bends cut on the inside (2 -> 5). 14 of R2J's 21 new lateral zeros are on straight roads (12 of them in scenes NTV0 scores 1.0), at speed
   (median 16.5 m/s, up to 38): the ego leaves its lane where the human stayed (5, lateral grows over 2-9 s), both cross (4), or a slow edge drift (5).
   2.5 s before the event the model plan already ends at the event's lateral (plan_lat4 ~ lat_ev), so it is the plan, not the controller.
   The PAI human labels carry more lane-change-shaped 4 s labels (|y| > 2.5 m, |yaw| < 12 deg, > 5 m/s) than navtrain: 11.9 % vs 8.0 % of moving
   rows, and in 44 % of those the route polyline does not go where the label goes: J imitates lane changes whose cue the input lacks (hypothesis,
   not tested). A road hinge would not stop a lane change into a legal lane; it would act on the 8 edge exits and the inside cuts.
   **Map raster for PAI rows: feasible.** Every one of the 853 curated-train scenes has a NuRec vector map (lanes 32-1514, road edges); the human
   footprint stays within 0.3 m of the lane union in 85 % of moving scenes (0.5 m: 90 %), so a nuPlan-style fallback (keep the row unhinged or use a
   road-edge layer when the human itself is outside; the evaluator itself falls back to road-edge distance) is needed for the rest. Cost: 0.28 s
   a 128 x 96 raster on one core (unvectorised shapely); about 187 k distinct poses (PAIH 135.8 k + CPAIX 50.9 k; the stretch rows reuse the plain
   row's pose) = about 15 core-hours, roughly 10-15 min as one pool CPU job on 90 cores, about 5 GB fp16 for the plain pools.
3. **Launch: deferred start plus slow acceleration, not a late plan.** On the 13 PAI 281 launches R2C makes and R2J loses (9.6 points), R2J's
   plan at the human's launch moves 0.10 m in 1 s, 0.32 m in 2 s and 3.6 m in 4 s (R2C 2.3 / 5.3 / 11.3), and 61 % of its decisions in the 4 s after
   t_L are standing "go later" plans (R2C 22 %). In 6 of them the ego never moves (stall; R2C scores 1.0 in all 6): the plan keeps promising to go
   in about 2 s and every re-plan defers it again. In 4 of the 6 the replayed follower (standing 2-6 m behind) leaves with the human's queue and
   hits us 2-6 s after t_L at 0.0 m/s; the other 2 end as slow partials. The others are slow: R2J moves on time (G-LAUNCH ego_move +0.33 s vs -0.02 s) but covers 4.4 m in 4 s vs R2C 6.5 (human 13.2), and
   is hit at 0.2-0.5 m/s, 6-8 m behind the human's position. J-W / J-NF look the same (deferred 0.18-0.21, ego 3.4-3.9 m). The rows explain
   it: PAI human standstill rows are 62 % "stay 4 s", 22 % "stand then go" and 5.6 % "go now" (navtrain 0.3 / 52 / 48 %; J-W removed only the
   first group); PAI launch labels are slower than navtrain (4 s arc 6.3 vs 8.8 m) and the f = 0.5 stretch rows slower still (3.6 m, by
   construction 0.25 x the acceleration).
4. **Overshoot: J brakes, too late and too softly.** J's plan contains a stop in 9 of 20 overshoot scenes (NTV0 1 of 21), and its 4 s end
   at 20 m before the human's rest point is 5 m past it (NTV0 19 m), at 10 m 10 m past (NTV0 34 m). The human brakes from about 10.8 m/s over
   about 42 m (8.0 m/s at 20 m, 5.7 at 10 m); R2J is at 9.6 / 8.1 m/s there and crosses the rest point at 6.9 m/s (NTV0 10.9). Stop plans appear
   only about 15 m out and stop 2-5 m past the human. The serve is not the cause (served / model 1.00). The native PAI-FIX reads and J-W / J-NF show
   the same: J's speed profile runs about 10 m (about 1 s) behind the human's, so about 1 s of braking onset is missing.

**For the next recipe.** The human PAI rows are the only thing that has fixed route turns (15 -> 2) and started stops, so keep them, but filter
them by state. (a) At standstill keep only rows whose cue is in view: drop "stay" and "stand then go later" rows unless a lead or a light gives
the timing, and drop the f = 0.5 stretch rows on launch rows; the target is the navtrain balance (about half "go now"). That removes the
deferred-start stall that costs most of the 9.6 launch points. (b) Drop or down-weight moving rows whose 4 s label leaves the route lane (12 % of
moving PAI rows, 44 % of them off the route), which targets J's straight-road lane departures (about 8-9 points). (c) Build the PAI road raster
from the NuRec maps (about 10-15 min CPU) so the PAI rows carry the road hinge, aimed at the 8 edge exits and the inside cuts. (d) Stops are half
learned: the remaining gap is the onset, about 10 m (1 s) too late, and 15 of 23 overshoot scenes are scored by no checkpoint, so imitation alone has
not reached them; the stop needs a signal on the model's own approach states, constrained against rear-hit (d402).

Limits: one seed per checkpoint; geometry classes come from the extract (0.5 s grid, map lanes near the rollout) and the class thresholds
are mine, not registered; the lane-change and standstill row shares are label statistics, not tested causes; the best-of ceiling mixes serves
(native and ACC + GAP) and uses 21-24 reads a scene for 270 scenes but only 3 for 10 of the 280.
