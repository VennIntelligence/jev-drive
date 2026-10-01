**Summary.** WOD-E2E v1.0.0 lives in `$DATA_DIR/datasets/waymo_e2e/` as front3-only slim shards (FRONT, FRONT_LEFT,
FRONT_RIGHT; raw 1.65 TB bucket `gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/`, slim ~0.73 TB). Splits: val 93
shards, training 263, test 266 (test futures hidden); frame index step is 0.1000 s (measured, timing fields are zeroed).
Download: `scripts/tmux_run.sh waymo scripts/download_waymo_e2e.sh`; idempotent, resumable, default
`--route proxy --streams 32` at ~15 MB/s (box link ~18 MB/s; full run ~30 h, train alone ~17.5 h); the `status` ETA is a
cumulative average and lags. Prepared data (index, targets, features, RFS, submission) is
`scripts/waymo_prepare.sh` / `jevdrive/waymo.py`, outputs under `$DATA_DIR/processed/waymo_e2e/`. RFS is computable
locally on one frame per val sequence (12 s mark).

**Sections.**
- Where it lives - dataset directory layout and why only front3
- Bucket and sizes - splits, shard counts, raw/slim sizes
- Data facts (checked on val, training and test shards) - E2EDFrame fields, cameras, states, intent, rater data
- Download - script, resume, disk guard, logs, tooling
- Network route - direct vs Clash throughput measurements and defaults
- gcloud login - gcloud CLI path, credentials, Clash-only login
- Prepared data path - waymo_prepare.sh commands and output files
### Index / How long is one frame index? / Image history / Targets and inputs - per-frame index, 0.1 s, history, targets
### Ego-only baselines / The pre-maneuver-onset subset - ADE/FDE baselines on val, pre-onset subset
### Rater Feedback Score ... / Frozen features / Extracting as shards land - RFS, features, incremental
### Where the extraction time goes / Submission / Checks - profile, challenge tar.gz writer, sanity checks
