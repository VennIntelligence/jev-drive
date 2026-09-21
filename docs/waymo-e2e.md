# Waymo Open Dataset E2E (WOD-E2E v1.0.0)

Read this when you need the Waymo end-to-end driving data on the box, or need to download or re-fetch it.

## Where it lives

`$DATA_DIR/datasets/waymo_e2e/`:

| Path | Content |
|---|---|
| `front3/<shard name>` | slim shards, same file names and `E2EDFrame` TFRecord format as the bucket, but only the FRONT, FRONT_LEFT and FRONT_RIGHT images (original JPEG bytes). Every other field is unchanged. Also the three small json/txt files, as-is |
| `manifest.csv` | one row per finished file: raw vs slim bytes, frames, download and slim seconds |
| `raw/` | transient: full shards waiting to be slimmed, deleted once the slim copy is verified |

We keep only the front three cameras because the disk cannot hold the raw 1.65 TB next to NAVSIM.
The full data (all 8 cameras) can always be re-fetched from the bucket.

## Bucket and sizes

`gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/`, flat, 1.65 TB (needs the licensed Google account).

| Split | Shards | Raw size | Notes |
|---|---|---|---|
| `val_202504211843.tfrecord-*` | 93 | 242 GB (2.6 GB/shard) | futures and (for ~0.1% of frames) 3 rated trajectories |
| `training_202504031202_202504151040.tfrecord-*` | 263 | 941 GB (3.6 GB/shard) | futures; no rated trajectories in the frames checked |
| `test_202504211836-202504220845.tfrecord-*` | 266 | 464 GB (1.7 GB/shard) | future states empty (hidden) |
| `test_sequence_frames_for_submission.json`, `..._as_list.txt`, `val_sequence_name_to_scenario_cluster.json` | | small | |

Slim/raw ratio: 0.44, so front3 is ~0.73 TB in total (val shard 0: 2.62 GB -> 1.15 GB, 1150 frames). See `manifest.csv` for all shards.

## Data facts (checked on val, training and test shards)

- Each record is one `E2EDFrame` (`waymo_open_dataset/protos/end_to_end_driving_data.proto`). A shard holds
  independent frames of many sequences in shuffled order (val shard 0: 1150 frames from 425 sequences).
  `frame.context.name` is `<sequence id>-<frame index>` (val 0..238, test 8..149), and the shards are sparse
  samples of it, not video. Every timing field is zeroed -- `timestamp_micros`, `pose_timestamp`,
  `camera_trigger_time`, `camera_readout_done_time` -- so the index interval has to be measured, not assumed;
  it is **0.1000 s (10.00 Hz)**, see "How long is one frame index?" below.
- 8 cameras per frame, JPEG, in every split. FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT: 972x1079
  (portrait); REAR_LEFT, REAR_RIGHT: 972x587; REAR: 972x551. ~330 KB per front JPEG.
  `context.camera_calibrations` has intrinsics, extrinsics, width, height, rolling shutter direction.
- `past_states`: 16 steps (4 s at 4 Hz) of pos x/y, vel x/y, accel x/y (no z). Ego frame at t=0:
  +x forward, +y left, origin at the rear axle middle.
- `future_states`: 20 steps (5 s at 4 Hz) of pos x/y/z only; empty in test.
- `intent`: UNKNOWN / GO_STRAIGHT / GO_LEFT / GO_RIGHT (val shard 0: 84% straight, 7% left, 9% right).
- `preference_trajectories`: 3 rated 21-point trajectories with scores 0-10, on **exactly one frame per val
  sequence** (at the 12 s mark); every other frame carries 3 placeholders scored -1. This is what the official
  Rater Feedback Score is computed against -- see "Rater Feedback Score" below. Train and test have none.

Inspect a shard: `scripts/download_waymo_e2e.sh inspect <tfrecord>`.

## Download

```bash
scripts/tmux_run.sh waymo scripts/download_waymo_e2e.sh            # small val train test, in that order
scripts/tmux_run.sh waymo scripts/download_waymo_e2e.sh val --route proxy --streams 16
```

- Idempotent and resumable: files in `manifest.csv` whose slim file exists are skipped, and complete raw shards
  left in `raw/` are slimmed without re-downloading. Just rerun it.
- Pipeline (`scripts/waymo_e2e.py`): 32-64 range-GET streams into a sparse `.part` file per shard, then one process
  per shard checks md5 and TFRecord CRCs, drops the other five cameras, re-reads and verifies the slim copy
  (same frames, same kept cameras), then deletes the raw shard. Slimming runs at ~420 MB/s per process,
  so the network is always the bottleneck.
- Disk guard: before each shard it checks free space on the data disk minus what in-flight shards still need.
  If the next shard would leave less than 200 GB (`--min-free-gb`), it finishes in-flight shards and exits,
  logging why. Free space, then rerun.
- Logs: `$DATA_DIR/runs/waymo_e2e/download/<time>/log.txt` and `events.jsonl`
  (`shard_downloaded`, `shard_slimmed`, `status` every 10 min, `stop`, `end`).
- Tooling: its own venv `$DATA_DIR/envs/waymo` (protobuf with the upb backend, google-crc32c, tqdm; no TensorFlow),
  protos compiled from the official repo at a pinned commit into `$DATA_DIR/envs/waymo/gen`. The script builds both.

## Network route

- The box's total download bandwidth is capped at ~18 MB/s, shared by all jobs (see network-proxy.md).
- Google OAuth (`oauth2.googleapis.com`) is unreachable directly; `storage.googleapis.com` is reachable,
  so the token always comes through Clash and the data route is a free choice (`--route direct|proxy`).
- The direct path to GCS is lossy and loses badly. Measure before a bulk transfer; measured 2026-09-21
  on a quiet link, 100 s steady-state samples: direct 32 streams 6.1 MB/s, direct 64 streams 7.2 MB/s,
  Clash (Tokyo-01, 0.1x) 32 streams 14.3 MB/s. So the training split runs with `--route proxy --streams 32`
  at ~15 MB/s. On a 0.1x node the whole 1.65 TB costs only ~165 GB of subscription quota.
- 2026-09-20, under contention from another bulk download: direct 32 streams 2.0 MB/s, direct 64 5.0,
  Clash 16 7.6. Alone that day both routes reached ~16 MB/s, so direct is only competitive on an idle link.
- A full run (1.65 TB) takes ~30 h at 15 MB/s; the training split alone (941 GB) ~17.5 h.
- **The download's own `status` line reports a cumulative average, so it lags a decaying link badly.** Its
  MB/s is total bytes over total elapsed, and its ETA follows: with a fast first few hours it read 9.0 MB/s
  and "17.3 h to go" at a moment when the link had actually been doing 4.3 MB/s for half an hour, an ETA
  understated by more than a day. For the rate that matters, difference two `status` events in
  `events.jsonl`: `gb` over `t`, half an hour apart. The train split ran at 13.5 MB/s in the early evening,
  6.3 at 19:19, 5.4 at 21:00 and 4.3 by 22:50 on the same node, so the instantaneous rate is worth
  re-measuring before quoting any finish time.

## gcloud login

- gcloud CLI: `~/data/tools/google-cloud-sdk` (not on PATH). To use it:
  `export CLOUDSDK_PYTHON=$HOME/data/tools/google-cloud-sdk/platform/bundledpythonunix/bin/python3 PATH=$HOME/data/tools/google-cloud-sdk/bin:$PATH`
- Credentials live in `~/.config/gcloud` (the licensed Google account). Never copy them or put them in the repo.
- Login only works through Clash. To redo it: open a tmux window and run `bash -l ~/data/tools/gauth.sh`,
  then open the printed URL in a local browser and paste the code back.
- Delete `~/.config/gcloud` before saving an instance image.

## Prepared data path

`jevdrive/waymo.py`, driven by `scripts/waymo_prepare.sh`, turns the slim shards into the frame index, the
training targets and the frozen features, and writes challenge submissions. Rerun it whenever new shards have
landed: every step is incremental.

```bash
scripts/tmux_run.sh waymo-prep scripts/waymo_prepare.sh                # index -> check -> report
scripts/waymo_prepare.sh index --workers 8                             # only shards that have no cache
scripts/waymo_prepare.sh features --cams front3 --batch-size 4         # frozen Qwen3-VL features
scripts/waymo_prepare.sh bench --limit 256                             # cost of each input choice
```

It needs the submission proto, which the download script does not compile; the first run fetches the proto
sources once (2 MB) and compiles both into the same `$DATA_DIR/envs/waymo/gen`. Everything else runs in the
project venv: `jevdrive.waymo` only needs protobuf, and only for the index and the submission.

Outputs, under `$DATA_DIR/processed/waymo_e2e/`:

| Path | Content |
|---|---|
| `shards/<shard>.parquet` | per-shard scan cache. A shard that has one is never read again |
| `index.parquet` | one row per frame, sorted by (split, sequence, frame) |
| `past.npy`, `future.npy` | `(n, 16, 6)` and `(n, 20, 3)` float32, row-aligned with the index |
| `rater.parquet` | the rater-scored trajectories: one row per (frame, trajectory), with the index row it belongs to |
| `report/*.csv` | the tables below, as of the last `report`: sequences, history, baselines, rater, onset_sweep, subsets |
| `features/<set>/` | `<name>.npy` float16 + `index.parquet` + `meta.json`, same layout as `processed/nuscenes/<version>/features/` |
| `submissions/` | `E2EDChallengeSubmission` tar.gz files |

### Index

One process per shard walks the TFRecord framing, parses each `E2EDFrame` once and writes a row:

| Column | Content |
|---|---|
| `sequence`, `frame` | `context.name` split at the last `-`. One index step is 0.1000 s, measured (below), not assumed |
| `split`, `shard` | |
| `rec_off`, `rec_len` | byte span of the record payload: `f.seek(rec_off); E2EDFrame.FromString(f.read(rec_len))` |
| `front_off/len`, `front_left_*`, `front_right_*` | byte span of each JPEG **inside the shard file**, so one camera is one `pread` plus `Image.open`, with no protobuf and no full-record read. Found at index time by searching the record for the parsed JPEG bytes and verifying the whole slice |
| `intent` | 0 UNKNOWN, 1 GO_STRAIGHT, 2 GO_LEFT, 3 GO_RIGHT |
| `has_future` | 20 future steps present (false for the whole test split) |
| `n_pref` | rated trajectories with a score >= 0: 3 on one frame per val sequence, 0 everywhere else |
| `cluster` | val scenario cluster, from `val_sequence_name_to_scenario_cluster.json` |

The ego states go to `past.npy` / `future.npy` instead of into the index, which keeps the index around 30 bytes
a row and lets training memory-map the trajectories.

Speed on the box with 6 worker processes, while three downloads were running: **14.5 GB / 14 497 frames in
6.8 s** = 2.1 GB/s, 2 100 frames/s. The whole slim dataset (~0.73 TB, ~720 k frames) is therefore about
**6 minutes**, and a rerun costs only the new shards. Rerunning is always safe: the merge is rebuilt from the
per-shard caches.

Sequences, on the shards downloaded so far:

| split | shards | of | frames | sequences | frames/seq | frame min | frame max | gap = 1 | gap median | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | **93** | 93 | 106 360 | 479 | 222.0 | 0 | 239 | **1.000** | 1 | **1.000** |
| train | 89 | 263 | 139 895 | 2 037 | 68.7 | 8 | 238 | 0.339 | 2 | 0.339 |
| test | 1 | 266 | 738 | 580 | 1.3 | 8 | 149 | 0.006 | 35.5 | 0.099 |

**val is complete**: 93 of 93 shards, and `gap = 1` on 100% of consecutive pairs, so every frame of every val
clip is on disk with no interior holes. That settles the prediction this table used to carry: coverage does
track `shards / of`, and a finished split is dense. Numbers below are at **93/93 val, 89/263 train**; rerun
`scripts/waymo_prepare.sh` to refresh them.

`coverage` is the share of the source 10 Hz grid that the downloaded shards hold, and `gap` is the step between
consecutive indexed frames of one sequence. Coverage tracks `shards / of` almost exactly, which says that
**a split's frames are spread over its shards at random and that a complete split holds essentially every frame
of every clip**: val runs from index 0 to 238 (~24 s) and test stops at 149, the 12 s mark after which the test
future is hidden. Sequences are dense in the dataset; they are sparse only in what is on disk right now.

### How long is one frame index? 0.1000 s, measured

Every timing field in WOD-E2E is zeroed -- `timestamp_micros`, `pose_timestamp`, `camera_trigger_time`,
`camera_readout_done_time` are all 0 (only `shutter`, the 10 ms exposure, survives). So the interval between
consecutive `context.name` indices cannot be read off a frame and **must be measured**. Do not assume it.

`scripts/waymo_prepare.sh` does not need it to build the index, but every window expressed in seconds does, so
it was measured from the ego trajectories: arc length along the ego path is frame-independent, and for two
frames A and B of one sequence with index gap `d`, the distance A covers in the 3.75 s ending at `d * dt` must
equal the distance B's own `past_states` cover over their whole 3.75 s window. A's path is known over
[-3.75, +5] s from `past_states` + `future_states`, so solving that equation gives `d * dt` directly.

Over **22 464 frame pairs** on val, restricted to frames whose speed actually changes (a constant speed dates
nothing): **dt = 0.1000 s, i.e. 10.00 Hz**, with an interquartile range of ~0 and the same answer at every gap
size from 2 to 40 indices. `FRAME_DT = 0.1` is therefore measured, not nominal.

One thing this does **not** reconcile: at 10 Hz the val index range 0-238 is 23.8 s of frames and the test
range 8-149 is 14.1 s, neither of which matches the challenge page's "20 s clips, first 12 s visible". The test
submission frames sit at index 148-150, i.e. 14.8-15.0 s after the clip's first indexed frame, which is
exactly 5 s -- one prediction horizon -- before the end of a 20 s clip. Treat the measured 10.00 Hz as solid
and the mapping from index to wall-clock position in the clip as unresolved; nothing we compute depends on it.

### Image history

`history_rows(df, n_back, stride)` returns, for each target frame, the rows of the target plus `n_back` earlier
frames of the same sequence, `stride` frame indices apart, the real `dt` of each slot, and which slots were
exact.

**Waymo cannot drift out of phase.** `frame` is an integer index and `stride` counts indices, so
`frame - k * stride` is exact arithmetic. This is the one place where a time-based request against irregular
timestamps goes wrong -- the requested interval and the native interval are never quite equal, the phase walks,
and a half-step tolerance eventually cannot hold, which is what bit the nuScenes V-JEPA clips. Here a slot
simply exists or does not.

**What did need fixing is the definition of complete.** A window is complete only when **every slot is an exact
hit**; `tol` now defaults to 0. Two fallbacks remain so the returned array stays rectangular, and neither is
history:

1. with `tol > 0`, the nearest indexed frame of the same sequence within +/- `tol` indices, never at or after
   the target, older preferred on a tie;
2. otherwise the previous (newer) slot is repeated -- last-frame padding, which for slot 1 repeats the target.

`exact` says which slots are real and `history_set(df, n_back, stride)` is the mask of strictly complete
windows. Never credit a model with history it was not shown.

Completeness on the val shards present, and what the shortfall is made of:

Reported **per split**, because the splits are downloaded at different rates and a pooled figure is just their
mixing ratio. On the complete val split:

| n_back | stride | span | **complete** | span inside clip | missing shards | predicted from 29 shards |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 s | **0.9955** | 0.9955 | 0.0000 | 0.9999 |
| 1 | 5 | 0.5 s | **0.9774** | 0.9775 | 0.0001 | 0.9993 |
| 3 | 5 | 1.5 s | **0.9321** | 0.9325 | 0.0003 | 0.9766 |
| 3 | 10 | 3.0 s | **0.8644** | 0.8651 | 0.0006 | 0.9211 |
| 7 | 5 | 3.5 s | **0.8418** | 0.8426 | 0.0008 | 0.9009 |
| 3 | 20 | 6.0 s | **0.7292** | 0.7302 | 0.0010 | 0.7859 |

With val complete, `complete` and `span inside clip` agree to a thousandth: **every remaining incomplete window
is one whose span runs off the front of its own clip**, and nothing is waiting on a shard. Train, at 89 of 263
shards, is the other regime -- 0.334 complete at a single slot against a 0.9998 ceiling.

**The last column is a correction.** This doc used to project 0.977 / 0.921 / 0.901 / 0.786 for the 1.5 s, 3 s,
3.5 s and 6 s windows at full val; the measured answers are **0.932 / 0.864 / 0.842 / 0.729**, three to six
points lower. The projection used the *split's* lowest frame index as the clip start, but val sequences are
not the same length (199 frames at the 10th percentile, 229 at the 90th), so short sequences' early frames were
counted as reachable. `span_inside` now bounds per sequence, which is what made the decomposition above exact.

**Which is which** depends on where the split is. While one is downloading the shortfall is almost entirely
missing shards -- train, at 89 of 263, is complete on 0.334 of frames against a 0.9998 ceiling. Once it
finishes the shortfall is entirely clip geometry: on val `missing shards` is 0.000-0.001, and what is left is
target frames sitting too near the start of their own clip for the span to fit. Only long spans bite.

**The tolerance rule would still be crediting padding here.** On complete val, "nearest frame within
`stride // 2`" reports 0.9864 / 0.9412 / 0.8871 / 0.7747 for the 0.5 s, 1.5 s, 3 s and 6 s windows against
strict answers of 0.9774 / 0.9321 / 0.8644 / 0.7292. The gap is smaller than it was on a sparse split -- where
it reached 33x -- but it never goes to zero, because the rule is happy to substitute a neighbouring frame for
one that genuinely does not exist. Those numbers are not completeness at any download level.

**Clip-set discipline.** When any row of a comparison table consumes image history, the **whole** table --
ego-only rows included -- must be restricted to the strictly complete set, because a padded window did not see
the history charged to it and rows scored on different frame sets are not comparable. `eval_set(df, split,
clip=(n_back, stride))` is that restriction and `baseline_table` / `subset_table` take a `clip` argument.
On complete val it leaves **103 953** frames for a 1x5 window, **99 141** for 3x5 and **91 941** for 3x10, so
multi-frame experiments are no longer sample-limited at all. The ego-only tables in this doc use no clip
restriction, because no row in them consumes history.

### Targets and inputs

- **Target.** `future_states.pos_x/pos_y` already is 20 waypoints in the current rear-axle ego frame, +x
  forward, +y left, first point at t+0.25 s -- exactly the submission convention. `future_xy()` only drops z;
  there is no transform to get wrong.
- **Ego state.** `ego_state()` flattens `past.npy` to 96 numbers: 16 steps at 4 Hz of pos x/y, vel x/y,
  accel x/y, oldest first. The position at t=0 is the origin on every frame (checked).
- **Intent.** `intent_onehot()` gives a 4-way one-hot. UNKNOWN does not occur in the shards seen so far
  (val: 85% GO_STRAIGHT, 7.6% GO_LEFT, 7.4% GO_RIGHT).
- Two things about `past_states` are worth knowing: the last `vel_*`/`accel_*` entry repeats the previous one,
  so there are only 15 independent velocity samples; and in turns the velocity direction disagrees with the
  direction of the position differences by a few degrees. `past_kinematics()` therefore estimates speed and
  yaw rate from the positions, and returns the raw vectors as well.

### Ego-only baselines

`baselines()` builds five trajectories from the past states alone, in the submission frame. ADE/FDE against the
logged future, in metres, on the complete val split (106 360 frames):

ADE@5s is also broken out per subset (`subsets()`), defined in the spirit of `jevdrive/labels.py` so the
nuScenes and Waymo tables line up. Sizes: straight 90 681, left 7 968, right 7 711, turn by intent 15 679,
already turning 11 060, straight 46 580, **pre-onset 1 510**.

| baseline | ADE@3s | FDE@3s | ADE@5s | FDE@5s | straight | left | right | turn (intent) | already turning | straight | **pre-onset** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero (stand still) | 8.866 | 16.333 | 14.278 | 27.097 | 15.55 | 5.29 | 8.57 | 6.90 | 12.46 | 20.83 | 11.49 |
| cv_vel (given velocity vector) | 1.155 | 2.936 | 2.805 | 7.323 | 2.60 | 3.01 | 4.96 | 3.97 | 5.41 | 2.66 | 4.25 |
| cv (speed along the heading) | 1.158 | 2.942 | 2.810 | 7.335 | 2.61 | 3.00 | 4.93 | 3.95 | 5.37 | 2.67 | 4.25 |
| ca (+ longitudinal acceleration) | 0.908 | 2.507 | 2.611 | 7.682 | 2.35 | 3.19 | 5.09 | 4.12 | 5.49 | **2.08** | 4.00 |
| ctrv (constant turn rate) | 1.065 | 2.762 | 2.688 | **7.225** | 2.65 | **2.09** | 3.71 | **2.89** | **4.30** | 2.68 | 4.22 |
| ctra (turn rate + acceleration) | **0.807** | **2.319** | **2.488** | 7.594 | 2.42 | 2.16 | **3.63** | 2.88 | 4.34 | 2.09 | **3.97** |

**This ADE is against the logged future and is not the leaderboard's ADE** -- see the next section, where the
same baselines are scored against the top-rated rater trajectory, which is how the official ADE is defined.
Against the log, the ego state alone gets to 2.53 m at 5 s; against the rater trajectory it gets 3.5-3.9 m
while the log itself gets 2.63 m. Do not compare the numbers in this table with a leaderboard row.

Two places where the baselines are visibly weak, and where a visual model should be measured:

- **Turns, and especially turns that have not started.** See the next section -- this is the sharpest cell in
  the table, and the direct analogue of the nuScenes hard-subset result.
- **The long horizon.** Extrapolating acceleration wins at 3 s (0.83 m) and loses at 5 s (FDE 7.71 m against
  CTRV's 7.23 m), because nothing in the ego state says when the car will stop. That is exactly what the
  cameras are for.

### The pre-maneuver-onset subset

`jevdrive/labels.py` calls a nuScenes frame **hard** when the car is not turning yet (`|current yaw rate| <
1 deg/s`) but turns within the horizon. On that subset the ego-state probe's turn recall collapses to 0.010
while mid-layer Qwen features reach 0.246 ([research/qwen-latent-driving.md](../research/qwen-latent-driving.md)).
`subsets()["pre_onset"]` is the Waymo version of it, with the same structure and thresholds picked from Waymo:

| | |
|---|---|
| Not turning yet | `\|yaw rate\| < 1.0 deg/s` over the past window -- the same cut as nuScenes, and it lands in the same place: it keeps 60% of usable Waymo val frames against nuScenes' 64% |
| Turns later | `\|bearing\| > 5 deg` at 3 s, the bearing of the chord from the origin to the waypoint at the horizon |
| Guards | the car must have moved >= 1 m in the last second and >= 3 m over the horizon |

**Why the bearing and not a heading change.** Waymo stores no future yaw, only positions, so the heading has to
be derived. Differencing consecutive future waypoints -- the obvious way, and what `labels.py` can afford with
20 Hz nuScenes poses -- is unusable here: at 4 Hz with a car that is often nearly stopped, its 99th percentile
is **358 deg of "heading change" over 2 s** on val, pure noise from near-zero displacements. The chord bearing
is one `atan2` on the endpoint, equals half the heading change for a constant-curvature arc, and degrades
gracefully. It validates against intent: the pre-onset subset is **56% GO_LEFT/GO_RIGHT** against **0.3%** on
the straight subset, and `check` asserts that separation.

Thresholds, and what each leaves to score (`onset_sweep()`):

| yaw rate < | bearing > | horizon | frames | share of val | rater-scored | turn intent |
|---:|---:|---:|---:|---:|---:|---:|
| **1.0** | **5** | **3 s** | **1 510** | 1.4% | **6** | 0.60 |
| 1.0 | 8 | 3 s | 907 | 0.9% | 2 | 0.83 |
| 1.0 | 5 | 5 s | 3 111 | 2.9% | 14 | 0.34 |
| 1.0 | 12 | 5 s | 1 421 | 1.3% | 9 | 0.75 |
| 2.0 | 5 | 3 s | 2 464 | 2.3% | 9 | 0.53 |
| 3.0 | 5 | 5 s | 6 353 | 6.0% | 29 | 0.29 |

**RFS cannot be reported on this subset. That is now measured, not projected.** On the complete val split,
**6** of the 479 rater-scored frames fall in it at the default thresholds, and 29 at the loosest setting worth
using -- because rater labels exist on exactly one frame per sequence and that frame is not chosen to be a
pre-onset moment. The projection made at 29 shards said 6, and 6 is what the full split holds. So the
pre-onset argument is an **ADE** argument, on **1 510 frames** (the projection said ~1 570). RFS stays the
metric for the split as a whole.

The ADE version says the same thing the nuScenes recall number says:

| baseline | all | already turning | **pre-onset** | straight |
|---|---:|---:|---:|---:|
| frames | 106 360 | 11 060 | 1 510 | 46 580 |
| cv (no yaw rate) | 2.810 | 5.368 | 4.251 | 2.670 |
| ctrv (+ yaw rate) | 2.688 | 4.302 | 4.216 | 2.677 |
| **what the yaw rate buys** | -0.122 | **-1.066** | **-0.035** | +0.007 |
| ca (no yaw rate) | 2.611 | 5.489 | 4.004 | 2.080 |
| ctra (+ yaw rate) | 2.488 | 4.342 | 3.967 | 2.087 |
| **what the yaw rate buys** | -0.123 | **-1.147** | **-0.037** | +0.007 |

Knowing the current yaw rate is worth **1.07-1.15 m of ADE@5s while the car is already turning, and 0.04 m
before the turn starts** -- a 30x difference, on the complete split. The ego state carries the turn only once the turn is underway;
at the moment before onset it is blind, exactly as on nuScenes. That is the cell a camera has to win, and the
one to put in the paper. Per-subset numbers with RFS next to ADE are in
`processed/waymo_e2e/report/subsets.csv`.

### Rater Feedback Score: computable locally, on one frame per val sequence

**RFS -- the metric the leaderboard actually ranks on -- can be computed on val, and we compute it.** The
earlier reading that val's `preference_trajectories` are "almost all invalid" was right about the count and
wrong about the conclusion: they are rare **by design**, not missing.

| | |
|---|---|
| Rater-scored frames | exactly **one per val sequence**, with exactly 3 rated trajectories, scores 0-10 |
| Where | frame index **147-150**, i.e. the 12 s mark -- the same point in the clip as the 1 505 test submission frames |
| On the complete val split | **479** -- exactly one per sequence, as predicted. The count was projected at 68 frames (12 shards) and again at 163 (29 shards) by scaling with coverage; both projections said 479 and 479 is what arrived |
| Waypoints per rated trajectory | 21 for the large majority, 7-20 for a handful; the metric truncates to 20 and pads short ones by repeating the last waypoint |
| Clusters present | 10 of the 11; val has no `Spotlight` sequences |

This is not a proxy. It is the real metric, on the same protocol shape as the test set (one frame per clip),
on a split with published labels. The limitation is sample size: 479 frames, full stop -- one per sequence is
all there is -- and the thinnest scenario clusters hold only a handful of them.

`rater_feedback_score()` is a port of
`waymo_open_dataset/metrics/python/rater_feedback_utils.py` and is **bit-identical to it** when both are given
float64 (the official code inherits the caller's dtype, so fed raw proto float32 it differs from ours in the
7th decimal; we promote, which is the more accurate of the two). Constants, verbatim from that file: trust
region checked at **3 s and 5 s**, base thresholds **1.0 m / 1.8 m**, multiplied by **1.0 lateral and 4.0
longitudinal**, scaled by `clip(0.5 + 0.5 (v - 1.4) / 9.6, 0.5, 1)` on the speed at t=0, decay **0.1** per
threshold of overshoot, floor **4.0** for a candidate not fully inside any single rater's region at both
horizons. Per frame: the best rater at each horizon, then the mean of the two horizons. The leaderboard number
is the mean per scenario cluster, then an unweighted mean over clusters (`E2EDMetrics.average_score`).

On all 479 rater-scored val frames:

| trajectory | RFS (cluster mean) | RFS (frame mean) | in trust region | ADE@3s | ADE@5s |
|---|---:|---:|---:|---:|---:|
| top-rated rater trajectory | 9.59 | 9.60 | 1.00 | 0 | 0 |
| **logged future** | **8.13** | **8.18** | 0.77 | 1.34 | **2.70** |
| worst-rated rater trajectory | 7.71 | 7.67 | 1.00 | 1.27 | 3.50 |
| cv_vel | **7.12** | **7.06** | 0.56 | 1.48 | 3.33 |
| cv | 7.10 | 7.05 | 0.55 | 1.49 | **3.35** |
| ctrv | 7.02 | 7.02 | 0.52 | 1.47 | 3.36 |
| ca | 6.84 | 6.80 | 0.49 | 1.48 | 3.87 |
| ctra | 6.80 | 6.78 | 0.47 | 1.46 | 3.93 |
| zero (stand still) | 5.38 | 5.39 | 0.28 | 7.25 | 12.32 |

ADE here is against the **top-rated rater trajectory**, which is the official definition
(`E2EDMetrics.ade_at_three_sec`: "we compute per frame ADE using the ground truth trajectory with the highest
rater score"). Three things follow, and they change how we should read the leaderboard:

- **The logged future scores 2.70 m ADE@5s, and published official-test ADEs are 2.65 (RAP) to 2.94
  (Poutine-Base).** On the official ADE the field is already at "predicts the log perfectly" level -- RAP is
  fractionally *past* it -- so ADE has almost no headroom and is a poor thing to optimise. RFS has headroom:
  the logged future is at 8.13-8.18 and the best public test RFS is 8.043.
- **Ego-only is at RFS 6.8-7.1 and ADE 3.3-3.9 m.** Against the rater trajectory the ego-only baselines are
  0.6-1.2 m worse than both SOTA and the log, which is the gap vision has to close -- unlike the log-ADE table
  above, where ego-only looked deceptively competitive. Published RFS for comparison: RAP 8.043,
  Poutine 7.986, AutoVLA 7.556, OpenEMMA 5.158.
- **Roughly half of every ego-only prediction falls outside every rater's trust region** and is floored at 4.0.
  That is where the score is lost, and it is a much sharper training signal than a mean displacement.

The sample is now final at 479 frames, so these numbers will not move again. Across 68 -> 163 -> 479 frames
every row moved by less than 0.5 RFS and the ordering held. One thing the full split did change: the constant
turn rate models, which trailed on the cluster mean at 163 frames, come out level with the constant velocity
ones (7.02 against 7.10), so the earlier gap there was sampling noise, not signal. The thinnest scenario
clusters still hold only a handful of frames, which is why the frame mean sits beside the cluster mean.

### Frozen features

The Waymo path reuses `jevdrive/features.py`: same model, same prompt-free chat template, same layers and
poolings, same float16 `.npy` + `index.parquet` + `meta.json` layout. Only the reader differs -- a Dataset that
`pread`s the JPEG spans out of the shard and hands the decoded images to the same transform, with items ordered
by (shard, offset) so every worker walks one shard forwards. `index.parquet` carries `frame_name`
(`<sequence>-<frame>`, which is the submission id), the row in the global index, and the cameras.

Three images in one forward is the same computation as three single-image forwards where it can be: the joint
pass's `vit_mean` and `vis_mean` equal the mean of the per-camera passes to bf16 precision. What it adds is
cross-camera attention in the LLM layers -- and that is free, because the three images share one prompt.

The front cameras are portrait 972x1079, so the size knob is `--long-side` (the longer side), not width.
Measured on val, 256 frames, 8 DataLoader workers, with three downloads running:

| set | cameras | input | tokens/frame | ms/frame (3 runs) | peak VRAM | bytes/frame |
|---|---|---|---:|---:|---:|---:|
| `qwen_front3` **(default)** | 3, one forward | native 972x1079 | 3060 | 127 - 216 | 9.8 GB | 47 KB |
| `qwen_front3_l800` | 3, one forward | long side 800 | 1725 | 75 - 134 | 9.2 GB | 47 KB |
| `qwen_front3sep` | 3, one forward each | native | 3 x 1020 | 170 - 246 | 8.8 GB | 141 KB |
| `qwen_front3sep_l800` | 3, one forward each | long side 800 | 3 x 575 | 104 - 138 | 8.6 GB | 141 KB |
| `qwen_front` | FRONT only | native | 1020 | 69 - 82 | 8.8 GB | 47 KB |
| `qwen_front_l800` | FRONT only | long side 800 | 575 | 31 - 44 | 8.6 GB | 47 KB |

**Read the ranges, not the numbers.** The card is shared -- another agent's `jevdrive.planner_v0` held ~7 GB of
it for part of this -- and the same six configurations, run three times, moved by up to 1.7x. Token counts,
VRAM and bytes are exact; ms/frame is "about this, on a busy box". Benchmark again on a quiet card before
quoting a latency anywhere.

Batch size does not change throughput at all -- one frame at 3060 tokens already fills the GPU. Measured back
to back in one run, batches of 1, 2, 4 and 8 gave 216, 219, 216 and 211 ms/frame. Only memory moves: peak VRAM
8.7 / 9.1 / 9.7 / 11.2 GB, and **batch 16 was killed by the host OOM killer**, because 8 DataLoader workers
prefetching 4 batches each hold ~40 GB of pixel values (one native front3 frame is ~76 MB of them).

**Batch size does change the features, so it is not a free knob: leave it at 4.** Re-extracting 128 frames of
an already-built val shard and comparing with what is on disk, batch 4 reproduces all ten arrays bit for bit,
while batch 8 reproduces none of them -- a different batch shape picks different GEMM tiling, and the error
compounds through 36 decoder layers to a mean relative deviation of 2.0e-2 on `L36_last`, far above float16
storage noise. Two feature sets built at different batch sizes cannot be compared, and it buys nothing anyway
(125.0 vs 125.7 ms/frame). Everything under `qwen_front3` -- val, train and test -- is built at batch 4.

**Default: `front3`, three cameras in one forward, native pixels.** The choice is not about speed. Joint and
per-camera forwards are within ~25% of each other in both directions across runs, for a clear reason: the joint
pass does 3x the self-attention work of the three separate passes together (3060 tokens instead of 3 x 1020),
and the separate passes do 3x the kernel launches, so which wins depends on how loaded the card is. What is not
close is that per-camera **triples the stored features** (141 KB vs 47 KB a frame) and throws away the
cross-camera attention, which is the only reason to hand the model the side views at all. Dropping to the front
camera alone is 2-3x cheaper, but that is an ablation, not a default: the side views are what the turning
frames need, and turning frames are where the ego-only baselines above are weakest. `--long-side 800` is the
cheap fallback -- 0.6x the time for exactly the same stored bytes -- if the full extraction turns out not to fit.

Extrapolated to the whole dataset -- **~726 k frames** (val 107 k, train ~414 k, test ~205 k, from the 2.27 MB
of raw bytes per frame that both the val and test shards show), at 47 KB of float16 features per frame and
127-216 ms a frame:

| Part | frames | GPU time at the default | features |
|---|---:|---:|---:|
| val, every frame | 107 k | 3.8 - 6.4 h | 5.0 GB |
| train, every frame | 414 k | 14.6 - 24.8 h (14.5 h measured) | 19.4 GB |
| test, every frame | 205 k | 7.2 - 12.3 h | 9.6 GB |
| test, only the 1 505 submission frames | 1.5 k | ~4 min | 0.07 GB |
| **whole dataset** | **726 k** | **26 - 44 h** | **34 GB** |

So the feature cache is not a storage problem at all, and one pass over everything is one to two days of a
shared GPU. Two obvious savings if that is too much: extract test only for the submission frames and their
history, and subsample train (every 5th index is 2 Hz and cuts it to 3 - 5 h).

### Extracting as shards land

`scripts/tmux_run.sh wfeat-train env INTERVAL=300 scripts/waymo_features_watch.sh train` extracts a split
while it is still downloading. The script only sets the environment; the loop is
`jevdrive.waymo features_inc --watch`, and each pass **re-runs the index before deciding what to extract**.
That order is the whole point: a loop that is incremental in extraction but not in arrival asks the index what
exists, gets the answer the index was built with, and sleeps while new shards pile up unseen. Each pass logs
what it saw -- shards on disk, shards indexed, shards built, out of the split's total -- not only what it did,
and warns when shards sit unbuilt while nothing moves. A shard that raises is logged and skipped so it cannot
block the rest; one that fails twice is set aside loudly.

A shard is the unit of work: its own arrays, its own `index.parquet`, and a `meta.json` written only when it
is finished, which is also its done-marker. So a crash costs one shard, and `load_features` merges whatever is
finished, keyed on `frame_name` (never on `row`, which is a position into an index that grows).

The model is loaded and compiled once for the whole run rather than once per pass, because a pass is often a
single shard and a reload each time would be pure overhead.

**Expect to be download-bound, and set `INTERVAL` to roughly the arrival cadence.** Extraction is a fixed
~3.4 min per 1 574-frame shard; the download is the variable side, and it decayed badly over one night --
13.5 MB/s and a shard every 4.5 min at the start, 4.3 MB/s and a shard every 11 - 13 min by midnight. So the
card sits idle most of the time and the run finishes when the download does. A short `INTERVAL` buys nothing
once that is true and costs something real: every pass rebuilds the four index files under any other reader
(above), so `INTERVAL=900` against a ~13 min cadence rather than `INTERVAL=300` cuts that churn threefold for
the same wall time.

That also means **an idle card is the normal state and not a stall**. The watcher's alarm is therefore a
multiple of the shard cadence the run has actually measured, not a fixed hour, and it separates the two
failures worth waking someone for: shards arriving and not being built (extraction is stuck) versus nothing
arriving at all (the download died). In a download-bound run the second one otherwise looks exactly like
healthy idling, which is why "shards on disk are unbuilt" alone is not enough of a test.

**Two invisible bugs once combined into a third, worse one, which is why both matter.** `feature_status`
globbed `f"{split}_*.tfrecord-*"`, which matches `val_` and `test_` but not the train shards, named
`training_...`; so `of` was 0 for train and the stop condition `built >= of` could never be satisfied. On
its own that is a watcher that does not know when to stop. Combined with an alarm shaped so that idling
looked healthy, it is a watcher that, **after all 263 shards were built and everyone had stopped watching,
would have gone on rebuilding the four index files every interval forever** -- firing the hazard above
continuously, unattended, with nothing in any log saying so. Neither bug alone would have been noticed. When
this loop is given a new split, check that `feature_status` reports a non-zero `of` for it before trusting
the run to end by itself.

**What this does to everything else on the box.** The re-index at the top of every pass rewrites
`index.parquet`, `past.npy`, `future.npy` and `rater.parquet` together, so while the watcher runs those four
files change every few minutes and the index grows (261 497 to 272 458 rows over one afternoon). Two
consequences for anything else reading the processed tree at the same time:

- The four are built aside and renamed in back to back, so no reader ever sees a half-written file and the
  inconsistent window is microseconds rather than the whole length of a rebuild. **That is a smaller window,
  not a guarantee**: four renames are not one atomic act, and a reader can still land between two of them.
- `rater.parquet`'s `row` column is a **position** into the index as it stood when that rebuild happened. Mix
  vintages and the rater trajectories sit beside the wrong frames. This is the bug the per-shard feature
  memmaps had before they were keyed on `frame_name`, in a different file, and it fails quietly:
  `waymo_stage_a.rfs_rows` drops out-of-range rows without complaining, so a corrupted RFS column can be
  produced with no error at all.

So **pin a snapshot before reading the processed tree alongside the watcher**:
`DATA_DIR=$(scripts/snapshot_processed.sh <name>) python -m jevdrive.waymo_l0 ...` copies the four files,
checks they agree with each other, and symlinks everything else (the per-shard feature directories are
append-only and immutable once written, so they need no copy). Reading the feature shards alone is always
safe; it is the four index files that move.

### Where the extraction time goes

Profiled on a quiet card while the train download was running (2026-09-21,
[todos/2026-09-21-waymo-train-features.md](../todos/2026-09-21-waymo-train-features.md)). Per frame, meaning
three cameras in one forward:

| Stage | ms | Where it runs |
|---|---:|---|
| `pread` the three JPEG spans | 0.1 | DataLoader worker |
| JPEG decode, 3 x 972x1079 | 17.8 | DataLoader worker |
| Qwen processor (smart_resize to 960x1088, normalise, patchify) | 38.6 | DataLoader worker |
| **one item, decode + processor** | **63.1** | one core |
| GPU forward, compiled, batch 4 | **132.0** | GPU |
| GPU forward, uncompiled, batch 4 | 183.4 | GPU (so `torch.compile` is worth 28%) |

**This path is GPU-bound, not preprocessing-bound.** The GPU wants an item every 132 ms and one core produces
one every 63 ms, so two workers already keep it fed; end-to-end throughput (125 - 132 ms/frame) equals the
forward alone, i.e. I/O and preprocessing are fully hidden. `loader_workers()` therefore defaults to
`max(2, min(6, n_cpus() // 4))` = 6 here, a 3x margin that leaves the other 19 cores to whatever is
downloading the shards; the old `n_cpus() // 2` = 12 bought nothing.

Inside the forward, at batch 8: ViT 44.5 ms, the 36 decoder layers 72.8 ms, merger + DeepStack + pooling +
H2D 10.3 ms. The decoder part is ~3.0 B non-embedding parameters over 3 074 tokens, i.e. ~18.4 TFLOP in
72.8 ms = **253 TFLOPS bf16, at this card's dense roofline**. There is nothing left to win in the pipeline:
the only ways to go faster are fewer tokens (lower resolution) or fewer layers (a real early exit), and both
change the stored features.

One lead that did **not** pay off, recorded so it is not chased again: Pillow can have libjpeg decode at 1/2,
1/4 or 1/8 scale via `img.draft("RGB", (w, h))`, which cut a Qwen policy's latency sharply elsewhere in this
project. It does not apply here. `smart_resize` maps the native 972x1079 front camera to 960x1088, so there is
no downscale for `draft` to exploit -- the 1/2 draft is 486x540 and would have to be interpolated back up,
changing every feature -- and it would only save 3-5 ms of CPU that is already hidden behind the GPU.

### Submission

`write_submission(frame_names, trajectories, path, meta)` writes an `E2EDChallengeSubmission` tar.gz per
`end_to_end_driving_submission.proto`: `(n, 20, 2)` metres in the current rear-axle ego frame, one final
trajectory per frame (no K candidates, no confidences), `submission_type = E2ED_SUBMISSION`, and the identity
and pretraining fields `account_name`, `unique_method_name`, `authors`, `affiliation`, `description`,
`method_link`, `uses_public_model_pretraining`, `public_model_names`, `num_model_parameters`. It refuses a
non-finite or wrong-shaped trajectory and duplicate frame names, requires the three fields the rules mark
required, and warns about any frame missing from `test_sequence_frames_for_submission.json` (1 505 frames, one
per test clip, at the 12 s mark). `read_submission()` parses one back, and `check` asserts that a full
1 505-frame submission round-trips with every field intact.

The test quota is 6 submissions per 30 days, so nothing here ever uploads: it only writes the file.

### Checks

`scripts/waymo_prepare.sh check` runs on whatever shards are on disk and asserts that records re-read from
their byte spans have the expected `context.name`, that the stored JPEG spans are byte-identical to the parsed
images and decode at 972x1079, that the past position at t=0 is the origin, that test futures are hidden and
train/val futures are not, that history windows stay inside the sequence and in the past with `dt` matching the
frame indices, that the baselines are ordered as they should be, that the rater trajectories line up with the
index and the raw records and that RFS gives a rater trajectory its own label back, exactly 4.0 to anything far
away, and more to the logged future than to standing still, and that a submission round-trips.


Last verified: 2026-09-21
