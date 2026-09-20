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
- Pipeline (`scripts/waymo_e2e.py`): 64 range-GET streams into a sparse `.part` file per shard, then one process
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

- The box's total download bandwidth is capped at ~18 MB/s, shared by all jobs (measured 2026-09-20).
- Google OAuth (`oauth2.googleapis.com`) is unreachable directly; `storage.googleapis.com` is reachable.
  So the token always comes through Clash, and the data goes direct (default `--route direct`, 64 streams).
  Bulk data also stays off Clash so it does not burn the proxy subscription's traffic.
- Measured alone: direct 2.7 MB/s per stream, 15.8 MB/s with 32 streams; Clash 12.9 MB/s single stream,
  16.4 MB/s with 16 streams; AutoDL turbo 0.07 MB/s (unusable). Direct and Clash both reach the cap.
  With another download running (NAVSIM from hf-mirror, domestic), each job gets a share of the ~18 MB/s,
  and the lossy direct route gets a small one: direct 32 streams 2.0 MB/s, direct 64 streams 5.0 MB/s,
  Clash 16 streams 7.6 MB/s. `--route proxy` gets more of the cap but only takes it from the other job.
- A full run (1.65 TB) takes ~26 h at the cap.

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
| val | 12 | 93 | 13 759 | 479 | 28.7 | 0 | 238 | 0.136 | 5 | 0.138 |
| test | 1 | 266 | 738 | 580 | 1.3 | 8 | 149 | 0.006 | 35.5 | 0.099 |

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

| n_back | stride | span | **complete** | span inside clip | missing shards | at full val | (old rule: within tol) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 s | 0.1248 | 0.9998 | 0.8750 | 0.9998 | 0.1248 |
| 1 | 5 | 0.5 s | 0.1140 | 0.9984 | 0.8845 | 0.9984 | 0.4582 |
| 3 | 5 | 1.5 s | 0.0020 | 0.9761 | 0.9741 | 0.9761 | 0.1031 |
| 3 | 10 | 3.0 s | 0.0017 | 0.9216 | 0.9200 | 0.9216 | 0.4023 |
| 3 | 20 | 6.0 s | 0.0007 | 0.7813 | 0.7806 | 0.7813 | 0.6025 |
| 7 | 5 | 3.5 s | 0.0000 | 0.8982 | 0.8982 | 0.8982 | 0.0047 |

**Which is which: essentially all of it is missing shards, none of it is clip geometry.** `span inside clip`
-- the requested span lies within the index range the split is known to reach -- is 78-100%, and
`missing shards` accounts for virtually the whole gap down to `complete`. So `at full val` equals
`span inside clip`: once the 93 val shards are down, a 1.5 s window is complete on 98% of frames, a 3 s window
on 92%, a 3.5 s seven-frame window on 90%, and a 6 s window on 78%. The clip only ever bites at long spans,
where the target frame is too near the start of its clip.

**The last column is the correction.** The earlier version of this table quoted "within tol" as completeness.
It was crediting padding: at a 6 s span it claimed 0.603 where the strict answer is 0.0007, an 860x
overstatement, and at 0.5 s it claimed 0.458 against 0.114. Those numbers should not have been reported as
completeness and are kept here only to show the size of the error.

**Clip-set discipline.** When any row of a comparison table consumes image history, the **whole** table --
ego-only rows included -- must be restricted to the strictly complete set, because a padded window did not see
the history charged to it and rows scored on different frame sets are not comparable. `eval_set(df, split,
clip=(n_back, stride))` is that restriction and `baseline_table` / `subset_table` take a `clip` argument.
Today it leaves **1 649** val frames for a 1x5 window, **29** for 3x5 and **24** for 3x10, so
**history-consuming experiments have to wait for more val shards**; the ego-only tables in this doc use no
clip restriction and say so.

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
nuScenes and Waymo tables line up. Sizes: straight 11 693, left 1 046, right 1 020, turn by intent 2 066,
already turning 1 436, straight 5 979, **pre-onset 226**.

| baseline | ADE@3s | FDE@3s | ADE@5s | FDE@5s | straight | left | right | turn (intent) | already turning | straight | **pre-onset** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero (stand still) | 8.890 | 16.380 | 14.315 | 27.156 | 15.61 | 5.52 | 8.44 | 6.96 | 12.47 | 20.96 | 11.48 |
| cv_vel (given velocity vector) | 1.156 | 2.932 | 2.801 | 7.316 | 2.59 | 3.16 | 4.87 | 4.00 | 5.34 | 2.66 | 4.21 |
| cv (speed along the heading) | 1.159 | 2.939 | 2.807 | 7.326 | 2.60 | 3.14 | 4.84 | 3.98 | 5.30 | 2.67 | 4.22 |
| ca (+ longitudinal acceleration) | 0.923 | 2.545 | 2.645 | 7.762 | 2.38 | 3.36 | 4.99 | 4.17 | 5.44 | **2.12** | 3.98 |
| ctrv (constant turn rate) | 1.065 | 2.758 | 2.687 | **7.226** | 2.65 | **2.22** | 3.63 | **2.92** | **4.28** | 2.67 | 4.19 |
| ctra (turn rate + acceleration) | **0.825** | **2.366** | **2.533** | 7.711 | 2.46 | 2.30 | **3.61** | 2.95 | 4.41 | 2.13 | **3.95** |

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
| **1.0** | **5** | **3 s** | **226** | 1.6% | **2** | **14** | 0.56 |
| 1.0 | 8 | 3 s | 125 | 0.9% | 1 | 7 | 0.82 |
| 1.0 | 5 | 5 s | 446 | 3.2% | 3 | 21 | 0.34 |
| 2.0 | 5 | 3 s | 358 | 2.6% | 2 | 14 | 0.53 |
| 3.0 | 5 | 3 s | 461 | 3.4% | 3 | 21 | 0.50 |
| 3.0 | 5 | 5 s | 861 | 6.3% | 5 | 35 | 0.30 |

**RFS cannot be reported on this subset, now or realistically ever.** Only 2 of the 68 rater-scored frames we
hold fall in it, and scaling to a complete val split gives **14 frames at the default thresholds, 35 at the
loosest** -- because rater labels exist on exactly one frame per sequence and that frame is not chosen to be a
pre-onset moment. So the pre-onset argument has to be made on **ADE**, where the subset is 226 frames today
and about **1 750 at full val** (1.6% of 106 671). RFS stays the metric for the split as a whole.

The ADE version says the same thing the nuScenes recall number says:

| baseline | all | already turning | **pre-onset** | straight |
|---|---:|---:|---:|---:|
| cv (no yaw rate) | 2.807 | 5.303 | 4.216 | 2.667 |
| ctrv (+ yaw rate) | 2.687 | 4.277 | 4.192 | 2.674 |
| **what the yaw rate buys** | -0.120 | **-1.026** | **-0.024** | +0.007 |
| ca (no yaw rate) | 2.645 | 5.443 | 3.975 | 2.121 |
| ctra (+ yaw rate) | 2.533 | 4.411 | 3.946 | 2.127 |
| **what the yaw rate buys** | -0.112 | **-1.032** | **-0.029** | +0.006 |

Knowing the current yaw rate is worth **1.03 m of ADE@5s while the car is already turning, and 0.02 m before
the turn starts** -- a 40x difference. The ego state carries the turn only once the turn is underway; at the
moment before onset it is blind, exactly as on nuScenes. That is the cell a camera has to win, and the one to
put in the paper. Per-subset numbers with RFS next to ADE are in `processed/waymo_e2e/report/subsets.csv`.

### Rater Feedback Score: computable locally, on one frame per val sequence

**RFS -- the metric the leaderboard actually ranks on -- can be computed on val, and we compute it.** The
earlier reading that val's `preference_trajectories` are "almost all invalid" was right about the count and
wrong about the conclusion: they are rare **by design**, not missing.

| | |
|---|---|
| Rater-scored frames | exactly **one per val sequence**, with exactly 3 rated trajectories, scores 0-10 |
| Where | frame index **147-150**, i.e. the 12 s mark -- the same point in the clip as the 1 505 test submission frames |
| On disk now | 68 frames, from 68 distinct sequences, spread over all 12 downloaded shards |
| When val is complete | **479** -- one per sequence. 479 x 13.8% coverage = 66 expected, 68 observed |
| Waypoints per rated trajectory | 21 for 197 of the 204, and 7-20 for the rest; the metric truncates to 20 and pads short ones by repeating the last waypoint |
| Clusters present | 10 of the 11; val has no `Spotlight` sequences |

This is not a proxy. It is the real metric, on the same protocol shape as the test set (one frame per clip at
12 s), on a split with published labels. The only limitation is sample size: 479 frames when val finishes,
68 today, and some scenario clusters will hold only a handful of frames.

`rater_feedback_score()` is a port of
`waymo_open_dataset/metrics/python/rater_feedback_utils.py` and is **bit-identical to it** when both are given
float64 (the official code inherits the caller's dtype, so fed raw proto float32 it differs from ours in the
7th decimal; we promote, which is the more accurate of the two). Constants, verbatim from that file: trust
region checked at **3 s and 5 s**, base thresholds **1.0 m / 1.8 m**, multiplied by **1.0 lateral and 4.0
longitudinal**, scaled by `clip(0.5 + 0.5 (v - 1.4) / 9.6, 0.5, 1)` on the speed at t=0, decay **0.1** per
threshold of overshoot, floor **4.0** for a candidate not fully inside any single rater's region at both
horizons. Per frame: the best rater at each horizon, then the mean of the two horizons. The leaderboard number
is the mean per scenario cluster, then an unweighted mean over clusters (`E2EDMetrics.average_score`).

On the 68 rater-scored val frames we have:

| trajectory | RFS (cluster mean) | RFS (frame mean) | in trust region | ADE@3s | ADE@5s |
|---|---:|---:|---:|---:|---:|
| top-rated rater trajectory | 9.53 | 9.50 | 1.00 | 0 | 0 |
| **logged future** | **8.08** | **8.21** | 0.78 | 1.44 | **2.63** |
| worst-rated rater trajectory | 7.74 | 7.52 | 1.00 | 1.23 | 3.29 |
| ca | **7.23** | 6.97 | 0.52 | 1.52 | 3.66 |
| cv | 7.19 | **7.00** | 0.53 | 1.63 | 3.51 |
| cv_vel | 7.08 | 6.97 | 0.54 | 1.62 | 3.50 |
| ctra | 6.66 | 6.82 | 0.49 | 1.55 | 3.89 |
| ctrv | 6.64 | 6.97 | 0.53 | 1.63 | 3.61 |
| zero (stand still) | 5.00 | 5.21 | 0.29 | 7.49 | 12.73 |

ADE here is against the **top-rated rater trajectory**, which is the official definition
(`E2EDMetrics.ade_at_three_sec`: "we compute per frame ADE using the ground truth trajectory with the highest
rater score"). Three things follow, and they change how we should read the leaderboard:

- **The logged future scores 2.63 m ADE@5s, and published official-test ADEs are 2.65 (RAP) to 2.94
  (Poutine-Base).** On the official ADE the field is already at "predicts the log perfectly" level, so ADE has
  almost no headroom left and is a poor thing to optimise. RFS has headroom: the logged future is at 8.08-8.21
  and the best public test RFS is 8.043.
- **Ego-only is at RFS ~7.1-7.2 and ADE ~3.5 m.** Against the rater trajectory the ego-only baselines are
  ~0.9 m worse than both SOTA and the log, which is the gap vision has to close -- unlike the log-ADE table
  above, where ego-only looked deceptively competitive. Published RFS for comparison: RAP 8.043,
  Poutine 7.986, AutoVLA 7.556, OpenEMMA 5.158.
- **Half of every ego-only prediction falls outside every rater's trust region** and is floored at 4.0. That
  is where the score is lost, and it is a much sharper training signal than a mean displacement.

Caveats on the 68-frame number: the cluster mean is noisy because some clusters hold 1-2 frames (`Cut_ins` 1,
`Construction` 2, `Others` 2), which is why the frame mean is reported next to it and why `ctrv`/`ctra` rank
below `cv`/`ca` on the cluster mean but tie on the frame mean. Treat per-cluster values as indicative until
val is fully downloaded, and prefer the frame mean while n is small.

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
prefetching 4 batches each hold ~40 GB of pixel values (one native front3 frame is ~76 MB of them). Batch 4 is
the default; raise the batch only together with fewer workers or a smaller prefetch.

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
| train, every frame | 414 k | 14.6 - 24.8 h | 19.4 GB |
| test, every frame | 205 k | 7.2 - 12.3 h | 9.6 GB |
| test, only the 1 505 submission frames | 1.5 k | ~4 min | 0.07 GB |
| **whole dataset** | **726 k** | **26 - 44 h** | **34 GB** |

So the feature cache is not a storage problem at all, and one pass over everything is one to two days of a
shared GPU. Two obvious savings if that is too much: extract test only for the submission frames and their
history, and subsample train (every 5th index is 2 Hz and cuts it to 3 - 5 h).

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


Last verified: 2026-09-20
