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
| val | 29 | 93 | 33 208 | 479 | 69.3 | 0 | 238 | 0.317 | 2 | 0.317 |
| test | 1 | 266 | 738 | 580 | 1.3 | 8 | 149 | 0.006 | 35.5 | 0.099 |

Every number below that depends on how much is downloaded is stamped **at 29 of 93 val shards**; rerun
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

| n_back | stride | span | complete @ 12 shards | **complete @ 29 shards** | span inside clip | missing shards | at full val | (old rule: within tol) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 s | 0.1248 | **0.3054** | 0.9999 | 0.6945 | 0.9999 | 0.3054 |
| 1 | 5 | 0.5 s | 0.1140 | **0.2946** | 0.9993 | 0.7047 | 0.9993 | 0.8106 |
| 3 | 5 | 1.5 s | 0.0020 | **0.0288** | 0.9766 | 0.9478 | 0.9766 | 0.5540 |
| 3 | 10 | 3.0 s | 0.0017 | **0.0253** | 0.9211 | 0.8959 | 0.9211 | 0.8175 |
| 3 | 20 | 6.0 s | 0.0007 | **0.0227** | 0.7859 | 0.7632 | 0.7859 | 0.7490 |
| 7 | 5 | 3.5 s | 0.0000 | **0.0003** | 0.8991 | 0.8988 | 0.8991 | 0.2581 |

The two `complete` columns are the attribution checking itself: 17 val shards landed between them, coverage
went 0.138 -> 0.317, and single-slot completeness went 0.125 -> 0.305, tracking coverage almost exactly, while
`span inside clip` did not move at all (0.9998 -> 0.9999). Completeness is a download property; the ceiling is
geometry.

**Which is which: essentially all of it is missing shards, none of it is clip geometry.** `span inside clip`
-- the requested span lies within the index range the split is known to reach -- is 78-100%, and
`missing shards` accounts for virtually the whole gap down to `complete`. So `at full val` equals
`span inside clip`: once the 93 val shards are down, a 1.5 s window is complete on 98% of frames, a 3 s window
on 92%, a 3.5 s seven-frame window on 90%, and a 6 s window on 78%. The clip only ever bites at long spans,
where the target frame is too near the start of its clip.

**The last column is the correction.** The earlier version of this table quoted "within tol" as completeness.
It was crediting padding: at a 6 s span it now claims 0.749 where the strict answer is 0.0227, a 33x
overstatement, and at 0.5 s 0.811 against 0.295. Those numbers should never have been reported as completeness
and are kept only to show the size of the error.

**Clip-set discipline.** When any row of a comparison table consumes image history, the **whole** table --
ego-only rows included -- must be restricted to the strictly complete set, because a padded window did not see
the history charged to it and rows scored on different frame sets are not comparable. `eval_set(df, split,
clip=(n_back, stride))` is that restriction and `baseline_table` / `subset_table` take a `clip` argument.
At 29 shards it leaves **9 997** val frames for a 1x5 window, **978** for 3x5 and **858** for 3x10 -- enough
for a first multi-frame run, and growing with every shard. The ego-only tables in this doc use no clip
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
logged future, in metres, on the 13 759 val frames downloaded so far:

ADE@5s is also broken out per subset (`subsets()`), defined in the spirit of `jevdrive/labels.py` so the
nuScenes and Waymo tables line up. Sizes: straight 28 333, left 2 449, right 2 426, turn by intent 4 875,
already turning 3 434, straight 14 543, **pre-onset 489**.

| baseline | ADE@3s | FDE@3s | ADE@5s | FDE@5s | straight | left | right | turn (intent) | already turning | straight | **pre-onset** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero (stand still) | 8.899 | 16.393 | 14.328 | 27.185 | 15.58 | 5.53 | 8.63 | 7.07 | 12.49 | 20.91 | 11.50 |
| cv_vel (given velocity vector) | 1.159 | 2.941 | 2.809 | 7.330 | 2.59 | 3.19 | 5.01 | 4.09 | 5.47 | 2.66 | 4.25 |
| cv (speed along the heading) | 1.162 | 2.948 | 2.815 | 7.342 | 2.60 | 3.17 | 4.99 | 4.08 | 5.43 | 2.67 | 4.25 |
| ca (+ longitudinal acceleration) | 0.919 | 2.537 | 2.641 | 7.761 | 2.36 | 3.37 | 5.19 | 4.28 | 5.62 | **2.11** | 3.96 |
| ctrv (constant turn rate) | 1.064 | 2.756 | 2.683 | **7.208** | 2.64 | **2.19** | 3.73 | **2.96** | **4.28** | 2.68 | 4.22 |
| ctra (turn rate + acceleration) | **0.813** | **2.336** | **2.505** | 7.642 | 2.43 | 2.25 | **3.68** | 2.96 | 4.36 | 2.11 | **3.93** |

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

| yaw rate < | bearing > | horizon | frames | share of val | rater-scored | at full val | turn intent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.0** | **5** | **3 s** | **489** | 1.5% | **2** | **6** | 0.57 |
| 1.0 | 8 | 3 s | 279 | 0.8% | 1 | 3 | 0.81 |
| 1.0 | 5 | 5 s | 1 006 | 3.0% | 4 | 12 | 0.33 |
| 1.0 | 12 | 5 s | 448 | 1.4% | 2 | 6 | 0.74 |
| 2.0 | 5 | 3 s | 800 | 2.4% | 3 | 9 | 0.52 |
| 2.0 | 8 | 3 s | 453 | 1.4% | 1 | 3 | 0.78 |

**RFS cannot be reported on this subset, now or realistically ever.** Only 2 of the 163 rater-scored frames we
hold fall in it, and scaling to a complete val split gives **6 frames at the default thresholds, 12 at the
loosest** -- because rater labels exist on exactly one frame per sequence and that frame is not chosen to be a
pre-onset moment. Going from 68 to 163 rater-scored frames did not add a single pre-onset one, which sharpened
the projection from 14 to 6 rather than improving it. So the pre-onset argument has to be made on **ADE**,
where the subset is 489 frames today and about **1 570 at full val** (1.5% of 106 671). RFS stays the metric
for the split as a whole.

The ADE version says the same thing the nuScenes recall number says:

| baseline | all | already turning | **pre-onset** | straight |
|---|---:|---:|---:|---:|
| frames | 33 208 | 3 434 | 489 | 14 543 |
| cv (no yaw rate) | 2.815 | 5.432 | 4.252 | 2.670 |
| ctrv (+ yaw rate) | 2.683 | 4.278 | 4.221 | 2.678 |
| **what the yaw rate buys** | -0.132 | **-1.154** | **-0.031** | +0.008 |
| ca (no yaw rate) | 2.641 | 5.616 | 3.962 | 2.106 |
| ctra (+ yaw rate) | 2.505 | 4.359 | 3.926 | 2.114 |
| **what the yaw rate buys** | -0.136 | **-1.257** | **-0.036** | +0.008 |

Knowing the current yaw rate is worth **1.15-1.26 m of ADE@5s while the car is already turning, and 0.03 m
before the turn starts** -- a 37x difference. The ego state carries the turn only once the turn is underway;
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
| On disk now | **163** frames, from 163 distinct sequences, spread over all 29 downloaded shards |
| When val is complete | **479** -- one per sequence. 479 x 31.7% coverage = 152 expected, 163 observed (and at 12 shards it was 66 expected, 68 observed) |
| Waypoints per rated trajectory | 21 for the large majority, 7-20 for a handful; the metric truncates to 20 and pads short ones by repeating the last waypoint |
| Clusters present | 10 of the 11; val has no `Spotlight` sequences |

This is not a proxy. It is the real metric, on the same protocol shape as the test set (one frame per clip),
on a split with published labels. The only limitation is sample size: 479 frames when val finishes, 163 today,
and some scenario clusters will hold only a handful of frames.

`rater_feedback_score()` is a port of
`waymo_open_dataset/metrics/python/rater_feedback_utils.py` and is **bit-identical to it** when both are given
float64 (the official code inherits the caller's dtype, so fed raw proto float32 it differs from ours in the
7th decimal; we promote, which is the more accurate of the two). Constants, verbatim from that file: trust
region checked at **3 s and 5 s**, base thresholds **1.0 m / 1.8 m**, multiplied by **1.0 lateral and 4.0
longitudinal**, scaled by `clip(0.5 + 0.5 (v - 1.4) / 9.6, 0.5, 1)` on the speed at t=0, decay **0.1** per
threshold of overshoot, floor **4.0** for a candidate not fully inside any single rater's region at both
horizons. Per frame: the best rater at each horizon, then the mean of the two horizons. The leaderboard number
is the mean per scenario cluster, then an unweighted mean over clusters (`E2EDMetrics.average_score`).

On the 163 rater-scored val frames we have:

| trajectory | RFS (cluster mean) | RFS (frame mean) | in trust region | ADE@3s | ADE@5s |
|---|---:|---:|---:|---:|---:|
| top-rated rater trajectory | 9.48 | 9.45 | 1.00 | 0 | 0 |
| **logged future** | **8.13** | **8.04** | 0.76 | 1.25 | **2.45** |
| worst-rated rater trajectory | 7.69 | 7.46 | 1.00 | 1.19 | 3.34 |
| cv_vel | **6.89** | 6.78 | 0.52 | 1.50 | 3.38 |
| cv | 6.87 | **6.78** | 0.52 | 1.52 | 3.41 |
| ca | 6.74 | 6.59 | 0.47 | 1.38 | 3.63 |
| ctrv | 6.69 | 6.72 | 0.50 | 1.50 | 3.45 |
| ctra | 6.63 | 6.54 | 0.45 | 1.38 | 3.72 |
| zero (stand still) | 5.34 | 5.38 | 0.29 | 6.48 | 11.07 |

ADE here is against the **top-rated rater trajectory**, which is the official definition
(`E2EDMetrics.ade_at_three_sec`: "we compute per frame ADE using the ground truth trajectory with the highest
rater score"). Three things follow, and they change how we should read the leaderboard:

- **The logged future scores 2.45 m ADE@5s, and published official-test ADEs are 2.65 (RAP) to 2.94
  (Poutine-Base).** On the official ADE the field is already at "predicts the log perfectly" level, so ADE has
  almost no headroom left and is a poor thing to optimise. RFS has headroom: the logged future is at 8.04-8.13
  and the best public test RFS is 8.043.
- **Ego-only is at RFS ~6.6-6.9 and ADE ~3.4-3.7 m.** Against the rater trajectory the ego-only baselines are
  ~1 m worse than both SOTA and the log, which is the gap vision has to close -- unlike the log-ADE table
  above, where ego-only looked deceptively competitive. Published RFS for comparison: RAP 8.043,
  Poutine 7.986, AutoVLA 7.556, OpenEMMA 5.158.
- **About half of every ego-only prediction falls outside every rater's trust region** and is floored at 4.0.
  That is where the score is lost, and it is a much sharper training signal than a mean displacement.

Caveats: the cluster mean is still noisy because the thinnest clusters hold only a handful of frames, which is
why the frame mean sits next to it. Between 68 and 163 frames every row moved by less than 0.4 RFS and the
ordering held, so the picture is stable, but treat per-cluster values as indicative until val is complete.

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

The model is loaded and compiled once for the whole run rather than once per pass. Extraction takes ~3.4 min
per 1 574-frame shard against a ~4.5 min download cadence, so once it has caught up a pass is a single shard,
and a reload each time would be pure overhead. With `INTERVAL=300` the steady state lags the download by about
four shards.

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
