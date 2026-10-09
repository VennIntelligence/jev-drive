# CF1: confirmation read of the AlpaSim candidate recipes on scenes no recipe choice has touched (pre-registration, 2026-10-09)

Lane CF1. Follows decisions 212 / 213 (OT3). Written and pushed before any CF1 closed-loop score is read. Local closed loop only; nothing is
registered with or submitted to AlpaSim. No WA-JEPA weights or features in any driver.

## Question

OT3 selected on the first 700 public scenes and read YR10m10-F (lambda-10 hinge + 10 % yaw-rate rows) against P2H10-F on the 400 later scenes of
part004 / 006 / 010 / 011: +0.0153, log-clustered CI [+0.0043, +0.0267]. Do the same recipes separate on scenes that no recipe of this family has been
read on? APY10m10-AB (AlpaSim input standard + the same rows) is the exploratory arm.

## Scenes

- **Fresh-A**: the 391 scenes of part012, 013, 014, 015 (labels as in `c0c/shards.tsv`), list `c0b/lists/cf1fresh.txt` (sorted scene ids). 6 nuPlan logs
  (10 / 25 / 49 / 87 / 106 / 114 scenes). Never run for P2H10, YR10m10 or APY10m10; no recipe choice used them (the C0c drivers SH30 / AP2 / OT30 /
  WA-JEPA were scored on them, which selected nothing in this family).
- **Fresh-B**: OT3's 400 later scenes (`c0b/lists/ot3new.txt`, 17 logs, one shared with Fresh-A). P2H10 and YR10m10 exist there (OT3); APY10m10 is run here.
- **Fresh** = Fresh-A + Fresh-B = 791 scenes, 22 logs. **All** = the 700 selection scenes + Fresh = 1491 scenes, 44 logs.
- Scenes whose route fails AlpaSim's sanity check ("route folds back", 12 known in the 1491) stay in every denominator; they are listed in the report.

## Drivers (all closed loop, same list and same chunking for all)

P2H10-F-s0, P2H10-F-s1, YR10m10-F-s0, YR10m10-F-s1 (NAVSIM input standard, `sh30`), APY10m10-AB-s0, APY10m10-AB-s1 (AlpaSim input standard, `ap2`).
Checkpoints `$DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt`, launched as OT3 launched them. Fresh-A: one pool job of 391 scenes per checkpoint
(6 jobs). Fresh-B: APY10m10-AB s0 / s1 as one job of the 400 scenes each, the list and chunking OT3 used for P2H10 / YR10m10 there. A recipe is the per-scene
mean of its two seeds.

Driver code: every run of this lane uses the box checkout at launch (commit recorded in `STATUS` and in the report), i.e. LAT1's driver: slot warp on the
card (pixel-identical to the CPU path, documented in results/lat1_frame_synthesis.md), compiled model passes (`SH30_COMPILE=1`, default; tolerance-level,
not bit-identical: P2H10-F-s0 on the 700 scenes 0.9481 vs 0.9484, same 25 zeros, no scene moves by more than 0.02, 0 scenes by more than 0.05), nvJPEG
off. OT3's runs on the first 700 and part of its 400-scene runs predate the compiled default. That is the documented check of "old vs current code";
it covers P2H10-F-s0 on all 700 scenes, so no chunk is rerun for it. The OT3 YR10m10 / P2H10 runs on Fresh-B are used as they are (no rerun).

## Statistics (all pre-set)

Per-scene recipe score = mean of the two seeds. Difference = paired per-scene difference of recipe means. 95 % bootstrap CI, 10 000 draws, seed 0:
log-clustered (resampling whole nuPlan logs, `date_vehicle`) and scene-resampled; the log-clustered one decides. Code: `scripts/ot3_report.py`'s `ci2`
(same bootstrap as OT3), wrapped by `scripts/cf1_report.py`.

## Lines (as given)

- **Primary**: YR10m10 (mean of 2 seeds, per scene) minus P2H10 (mean of 2 seeds) on the part012-015 scenes (Fresh-A); **confirmed** if point estimate
  >= +0.005 and log-clustered 95 % CI lower bound > 0; **refuted** if point estimate <= 0; else **inconclusive**. Also report the pooled 791 fresh scenes
  and all 1491 where runs exist.
- **Secondary (exploratory arm)**: APY10m10 minus P2H10 and APY10m10 minus YR10m10 on the 791 fresh scenes, same statistics; APY becomes a candidate
  only if APY minus YR10m10 >= +0.005 with log-clustered lower bound > 0.
- **Report per recipe**: mean, zeros split (at-fault collision / offroad / left corridor), slow count (0 < score < 1), at-fault events, per-seed means,
  scene-resampled and log-clustered CIs. Route-fold scenes stay in the denominator and are listed.

## Known limits, stated before the data

- Fresh-A has 6 logs. A log-clustered bootstrap over 6 clusters is coarse (a lower bound > 0 needs the sign to hold across almost all resamples of 6 logs);
  the verdict follows the registered rule regardless, the scene-resampled CI and the pooled 791 / 1491 reads are reported next to it, not instead of it.
- Two seeds per recipe, one simulation per checkpoint; seed spread inside a recipe was about 0.002-0.007 on 700 scenes.
- Shard labels of the new scenes are inferred (contiguous 100-scene blocks).

## Execution

One chained script `scripts/cf1_chain.sh` in tmux `jev` (STATUS / DONE / ERROR under `$DATA_DIR/runs/alpasim/cf1/`): waits until the simulator stacks of
other lanes leave room (this lane at most 3, box total <= 6), runs `scripts/ot2_loop.py` with `OT_LANE=cf1` (GPU pool, watchdog, fill-in of missing
scenes), then the report. Declared VRAM 32 GB per stack (`cl vram` measured 29-32 GB). A worker death (runner exits rc 1) -> the boxwatch rows of the two
minutes before it are saved to the results folder, the chunk is rerun, and the report says so.

## Amendment 1 (2026-10-09, written and pushed before any AP2H10 score on the fresh scenes is read)

Question: is APY10m10's gain the AlpaSim input standard alone, or do the yaw-rate rows add anything on top of it? This is the last read: **after it no untouched
scenes remain** (every one of the 1491 public scenes will have been scored for the recipes of this family; any later read is a repeat).

Runs: AP2H10-AB-s0 and AP2H10-AB-s1 (lambda-10 hinge, AlpaSim input standard, no yaw-rate rows; OT3 checkpoints; OT3 already has them on the first 700 scenes) on the 791
fresh scenes with the lists and chunking used there for the other recipes: `cf1fresh` (391, one job per checkpoint) and `ot3new` (400, one job per checkpoint). Same driver checkout
as the CF1 runs (c9d2e07b; `git diff c9d2e07b HEAD` over `experiments/alpasim/lib`, `scripts` and `jevdrive` is empty at launch). Box checkout hash is recorded per run (`checkout.txt` in
each run dir, plus the chain log). Pool jobs, at most 3 of this lane's stacks, nothing submitted to AlpaSim.

Lines (as given):
- (a) APY10m10 minus AP2H10 (2-seed means, 791 fresh scenes, log-clustered 95 % CI): the rows add on top of the input standard if >= +0.005 with lower bound > 0; no added
  value if the point estimate is < +0.005 or the CI includes 0.
- (b) AP2H10 minus P2H10 on the 791 and on all 1491: the input standard helps if >= +0.005 with lower bound > 0.

Report: the same per-recipe table (mean, zeros split, slow, at-fault events, per-seed), the 4-recipe table on all 1491 scenes with all pairwise differences, and the statement that no
untouched scenes remain. Outcome goes into decision 216 as an amendment (no new decision unless it flips 216).
