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
  `frame.context.name` is `<sequence id>-<frame index>` (index 8..237), so the source runs at ~10 Hz over ~20 s,
  but the shards are sparse samples, not video. `timestamp_micros` and image `pose_timestamp` are zeroed.
- 8 cameras per frame, JPEG, in every split. FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT: 972x1079
  (portrait); REAR_LEFT, REAR_RIGHT: 972x587; REAR: 972x551. ~330 KB per front JPEG.
  `context.camera_calibrations` has intrinsics, extrinsics, width, height, rolling shutter direction.
- `past_states`: 16 steps (4 s at 4 Hz) of pos x/y, vel x/y, accel x/y (no z). Ego frame at t=0:
  +x forward, +y left, origin at the rear axle middle.
- `future_states`: 20 steps (5 s at 4 Hz) of pos x/y/z only; empty in test.
- `intent`: UNKNOWN / GO_STRAIGHT / GO_LEFT / GO_RIGHT (val shard 0: 84% straight, 7% left, 9% right).
- `preference_trajectories`: up to 3 rated 21-point trajectories, score 0-10, invalid ones score -1.

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
| `report/*.csv` | the three tables below, as of the last `report` |
| `features/<set>/` | `<name>.npy` float16 + `index.parquet` + `meta.json`, same layout as `processed/nuscenes/<version>/features/` |
| `submissions/` | `E2EDChallengeSubmission` tar.gz files |

### Index

One process per shard walks the TFRecord framing, parses each `E2EDFrame` once and writes a row:

| Column | Content |
|---|---|
| `sequence`, `frame` | `context.name` split at the last `-`. The frame index is at the source ~10 Hz, so a step of 1 is 0.1 s |
| `split`, `shard` | |
| `rec_off`, `rec_len` | byte span of the record payload: `f.seek(rec_off); E2EDFrame.FromString(f.read(rec_len))` |
| `front_off/len`, `front_left_*`, `front_right_*` | byte span of each JPEG **inside the shard file**, so one camera is one `pread` plus `Image.open`, with no protobuf and no full-record read. Found at index time by searching the record for the parsed JPEG bytes and verifying the whole slice |
| `intent` | 0 UNKNOWN, 1 GO_STRAIGHT, 2 GO_LEFT, 3 GO_RIGHT |
| `has_future` | 20 future steps present (false for the whole test split) |
| `n_pref` | preference trajectories with a score >= 0 |
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

### Image history

`history_rows(df, n_back, stride)` returns, for each target frame, the rows of the target plus `n_back` earlier
frames of the same sequence, `stride` source indices (0.1 s each) apart, the real `dt` of each slot, and which
slots were exact. A requested index can be missing because its shard is not downloaded, so:

1. take the exact index if it is indexed;
2. else the nearest indexed frame of the same sequence within +/- `tol` indices (default `stride // 2`), never
   at or after the target, older preferred on a tie;
3. else repeat the previous (newer) slot -- ordinary last-frame padding, which for slot 1 repeats the target.

`dt` is the real offset, so a model can use it (or drop the sample) instead of trusting the nominal stride, and
`exact` says which slots came from rule 1.

Completeness on the val shards present, as the share of target frames whose whole window resolves:

| n_back | stride | span | exact | within tol | slots exact | slots within tol |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 s | 0.125 | 0.125 | 0.125 | 0.125 |
| 1 | 5 | 0.5 s | 0.114 | 0.458 | 0.114 | 0.458 |
| 3 | 5 | 1.5 s | 0.002 | 0.103 | 0.114 | 0.450 |
| 3 | 10 | 3.0 s | 0.002 | 0.402 | 0.110 | 0.681 |
| 3 | 20 | 6.0 s | 0.001 | 0.603 | 0.100 | 0.757 |
| 7 | 5 | 3.5 s | 0.000 | 0.005 | 0.110 | 0.429 |

These numbers are a property of **the download, not of the dataset**: `exact` is close to `coverage ** n_back`,
so it climbs to ~1 for every window once val's 93 shards are on disk. Nothing in the plan has to work around
missing history; the tolerance and padding rules exist so that experiments can start before the download ends.

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

| baseline | ADE@3s | FDE@3s | ADE@5s | FDE@5s | ADE@5s straight | left | right |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero (stand still) | 8.890 | 16.380 | 14.315 | 27.156 | 15.61 | 5.52 | 8.44 |
| cv_vel (given velocity vector) | 1.156 | 2.932 | 2.801 | 7.316 | 2.59 | 3.16 | 4.87 |
| cv (speed along the heading) | 1.159 | 2.939 | 2.807 | 7.326 | 2.60 | 3.14 | 4.84 |
| ca (+ longitudinal acceleration) | **0.923** | **2.545** | 2.645 | 7.762 | 2.38 | 3.36 | 4.99 |
| ctrv (constant turn rate) | 1.065 | 2.758 | 2.687 | 7.226 | 2.65 | 2.22 | 3.63 |
| ctra (turn rate + acceleration) | 1.059 | 2.744 | **2.678** | **7.217** | 2.64 | **2.22** | **3.62** |

This is our development metric and **is not the leaderboard's ADE**: the official secondary metric scores
against the highest-rated rater trajectory on the official test split, and val's `preference_trajectories` are
almost all invalid (score -1), so RFS cannot be computed locally at all.

Read the table as the bar a visual model has to clear. CTRA at **2.68 m ADE@5s** and plain constant velocity at
**2.81 m** are close enough to the published official-test ADEs (RAP 2.65, Poutine 2.74) that a mean ADE will
not show a visual gain, whatever the protocol difference. The gap lives in the turns: constant velocity costs
4.84 m on GO_RIGHT against 2.60 m on GO_STRAIGHT, and a constant turn rate brings GO_RIGHT down to 3.62 m.
Report ADE per intent and per scenario cluster, never the mean alone.

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

PLACEHOLDER_BENCH

PLACEHOLDER_BENCH_TEXT

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
frame indices, that the baselines are ordered as they should be, and that a submission round-trips.


Last verified: 2026-09-20
