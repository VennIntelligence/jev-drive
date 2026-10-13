# Round 2: where things stand (drained for the box reboot, 2026-10-13 ~22:50 JST)

Prereg: [plans/2026-10-13-round2-prereg.md](../../plans/2026-10-13-round2-prereg.md) (pushed before any rollout). Chain: [scripts/rf_round2.sh](../../scripts/rf_round2.sh).

## Done
- **Rollouts: all 6 of 6 chunks.** RSGbothSB-L18-s1 on onpol-train (1031 scenes, 18 decisions, ol_chain.sh roll1 unedited, cards 1-5): 172 / 172 / ... rollouts per
  chunk, 0 failed, ~3060 dumps per chunk. Box: `$DATA_DIR/runs/onpol_long/roll/RSGbothSB-L18-s1/c{0..5}/run` (onpol_sim.pkl + dumps; asl of scored rollouts
  already pruned, 117 asl / 5 GB left). Dumps are not stripped yet (~35 GB; `strip` after tok cuts them to ~6 GB).
- **Rows (on_rows geo) built:** 17 924 rows of 18 352 decisions -> `$DATA_DIR/runs/op_parity/cache/on1_L18SB1`, `runs/onpol_nuplan/rows/L18SB1`.
  on_rows' strict R2 prints FAIL for one excluded rollout (2021.09.09.17.18.51_veh-48_00657_00876-798842f0b1c253b4, frame_err 2.1 m, t0_rel -2.0), the same
  case as round 1 (one rollout whose time does not match the log yields no rows); the gate of record is ol_labels.py gate (R1 / R2 as amended), run in `tok`.
- **ol_labels build (labels K / C, part of the geo job)**: was at 303 / 325 lane fields when drained; job `1013-220254-9dd9` (r2-geo) left running.
  It either finished before the reboot (pool log `$DATA_DIR/runs/route_follow/round2/pool/r2-geo/log.txt` ends without ERROR and STATUS says done) or was killed.

## Not done (all queued jobs were cancelled; nothing of these exists)
tok (tokens + R1 + ol_labels gates) -> strip, rm_build pool -> pool gate -> side tags on1_L18SB1s -> merge on1_L18U2s -> smoke + train R2SB-L18-s{0,1} /
R2SBU-L18-s{0,1} -> held 460 reads + public 10 chunks -> report. No checkpoint was trained, nothing was read.

## Continue after the reboot (box, repo root; pool dispatcher up in tmux jev:pool)
    cd ~/data/jev-drive && git pull && export DATA_DIR=/root/autodl-tmp/ujs
    cat $DATA_DIR/runs/route_follow/round2/pool/r2-geo/STATUS     # geo done?
    # geo done:
    bash experiments/route_follow/scripts/rf_round2.sh resume tok
    # geo killed mid-way (labels unfinished): rows from scratch (on_rows geo overwrites on1_L18SB1)
    bash experiments/route_follow/scripts/rf_round2.sh resume geo
Either queues the whole remaining chain with --after (about 30 min rows / gates, then 4 trainings of ~7 min, 4 x (6 held + 10 public) read jobs,
the report job `r2-report`; outputs in `$DATA_DIR/runs/route_follow/round2/`). After it: copy held_raw.txt, held18/, guard/, follow.{md,json}, inside.json,
side/, sidetag_r1.json, weights_*.json, poolgate json into this directory, run `rf_sidetag.py read` on the Mac against results/sidebias + rsgboth_sb CSVs,
write summary.md with the four lines of the prereg.
Disk: the pool step needs ~15 GB more (front.npy 4.4 GB, union front ~9 GB); strip frees ~29 GB.
