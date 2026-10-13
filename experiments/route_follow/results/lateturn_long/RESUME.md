# lateturn_long: state at the pre-reboot drain (2026-10-13 ~22:30 JST)

## Feasibility (done)
- `num_timesteps_after` counts 20 Hz steps: 380 = token + 19 s = 40 decisions (190 gave only 21 decisions; that smoke is in `smoke_after190_9.5s` on the box).
- Cache `$DATA_DIR/datasets/alpasim_nuplan_long/after380_lt` (86 scenes, built with `build_cache.py --scenes`, config `runs/route_follow/lateturn_long/long_cfg.yaml`).
- Smoke (RSG-L18-s0, 3 scenes) and the full reads ran without sim errors: 82 of 86 scenes run 40 decisions; 4 end at the log end (22, 29, 31, 33 decisions) with a clean clip ("Clipping n_sim_steps"), no freeze, normal score.
- Limit is the MTGS reconstruction: video covers token + 12.9 s median on these 86 (p10 9.4 s); only ~23 / 86 cover 20 s. Past coverage the renderer draws outside the reconstruction (no failure, but a rendering confound). Report covered vs uncovered.

## Done on the box (`$DATA_DIR/runs/route_follow/lateturn_long`)
- Reads: RSG-L18-s0, RSGboth-L18-s0, RB0-F-s0 x all 86 scenes (`read/<model>/c0..c2`), 9/9 chunks complete, 258 rollouts. rollout.asl of scored non-failed rollouts pruned.
- `scan.csv`: class scan of the 86 on the long window (RF_K_END=39).

## Pending
- Pool job `lt-long-post` (1013-221904-aafc) was queued, not started: rf_follow + rf_tax + shape + scores + report -> `report/summary.md`, `rollouts.csv`. If the reboot lost it, resubmit.

## Continue after reboot
    ssh autodl 'cd ~/data/jev-drive && git pull && export DATA_DIR=/root/autodl-tmp/ujs && bash experiments/route_follow/scripts/rf_lateturn_long.sh submit-post'
Then copy `report/{summary.md,summary.json,rollouts.csv}`, `scan.csv`, `shape.csv` to this folder, add reach-share (scan.csv: turn found in window) and covered/uncovered split (experiments/onpol_nuplan/results/long/span.csv, video_after_s), and append the dated addendum to research/route-follow/late-turns.md. Then delete `read/*/c*/rollouts` asl files.
