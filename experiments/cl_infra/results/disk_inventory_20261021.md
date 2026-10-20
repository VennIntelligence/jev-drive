# Box disk inventory 2026-10-21 (GPU box, `$DATA_DIR` = /root/autodl-tmp/ujs)

Inventory only: nothing was deleted, moved or modified. This is a ranked list of space that could be freed, for the user to approve
item by item. The rules are the same as in the earlier passes ([cleanup/](../cleanup/), [docs/storage.md](../../../docs/storage.md)):
inventory and deletion never happen in one dispatch, and one `rollout.asl` per `rollouts` dir is kept as a sample.

**Units.** GB = 10^9 bytes of apparent file size, each inode counted once. `df` reports GiB-ish "G", so 1 000 GB here is about 931 G there.

**Scan.** 2026-10-20 23:44-23:45 CST (box clock). One `find -printf` (size, mtime, nlink, inode, path) per depth-1 subtree, 24 in parallel
under `nice -n 19 ionice -c3`, covering 7.68 M files and 6 641 GB unique (matches `df`). Aggregates were done with pandas on the box.
Scripts and outputs are in `$DATA_DIR/tmp/inv_1021/` on the data disk: `scan.sh`, `agg.py`, `cls.py`, `files.parquet`, `dirs.tsv`, `cls.tsv`,
`ext.tsv`, `asl.tsv`, `asl_reads.tsv`, `ckpt_runs.tsv`. The checkpoint citation grep is in the Mac `tmp/inv_1021/` (gitignored).

## State during the scan

| | |
|:--|:--|
| `df` at the scan | 6.7T, 558 G free (92 %) |
| `df` 10-21 00:00 | **864 G free (88 %)**. A user-approved `zc-swap` pool job (`/tmp/zc_swap.py`, 23:45-23:55) replaced the copied `front.npy` of five merged pools with zero-copy `front.parts.json` indexes: on1_JTPH 138.6, JTCP 66.5, JTX 55.4, JT 44.5, PPM160 11.3 = **316.3 GB freed**, all five `SWAPPED`, bitwise verified. These are not candidates here. `runs/op_parity/cache` went from 800 GB to about 484 GB. |
| Pool (`cl top` / `cl queue`) | At 23:40: 8 running (`pai-r2j-l18-s0-c000`..`c007`, ROUND2 R2J + ACC + GAP PAI 281 -> `runs/round2/pai281/R2J-L18-s0-ag`) and 2 queued (`fx-tax-R2J-L18-s0-281ag`, `r2-s2gate`). By 00:00 all were done; `r2-s2gate` wrote `runs/round2/S2STOP`. The pool was idle after that. |
| Outside the pool | NuRec fetch, tmux `jev`, started 23:54: `nurec_fetch.py --list experiments/pai_rows/results/download_queue_v2.csv` (64 scenes, 111 GB, floor 600 G). It writes into `datasets/nurec/_dl` and `all-usdzs`. |
| Wait loops still alive | `runs/pai_av` CHAIN_EGO_DONE (23 h), `runs/pai_rows` pr-probe-H DONE (4.7 h), `runs/alpasim_proxy/arena/report/cand` DONE (3 days). All of them poll files outside every candidate below. The 10-20 `lat_diag` loop that held `native_gap/dump` is gone. |
| Open files (`/proc/*/fd`) | The R2J read dir, `third_party/alpasim/.venv`, `tools/nre-rootfs`, the dispatcher / boxwatch locks, `clash`, and during the swap the source pools. Nothing else on the data disk was open. |

**Merged-pool refs after the swap** (`python -m jevdrive.tokens refs runs/op_parity/cache`; before the swap there were no
manifests at all). These source pools can no longer be deleted while the merged pools live:

| source `front.npy` | indexed by |
|:--|:--|
| `on1_L18RB0@warp` | on1_JT, on1_JTCP, on1_JTPH, on1_JTX, on1_PPM160 |
| `on1_L18RTB1@warp`, `on1_L18YS@warp`, `on1_L18Y@warp`, `on1_NR1TV@warp` | on1_JT, on1_JTCP, on1_JTPH, on1_JTX |
| `on1_CPAI570@warp` | on1_JTCP |
| `on1_PAIH@warp`, `on1_PAIH05@warp`, `on1_PAIH15@warp` | on1_JTPH |
| `on1_PN0@warp` .. `on1_PN5@warp` | on1_JTX |
| `on1_PPAI160@warp` | on1_PPM160 |

**Flag for the main session.** The brief lists R2J-L18-s0 as a current candidate. Decision 412's status column already reads "不采用 R2J"
(R2J not adopted), and tonight's in-queue gate stopped stage 2 (`S2STOP`). Everything R2J is held below (class "ask") until the main
session settles which is right.

**Lesson from 10-20.** Decision 402 records that the 10-20c pass deleted `op_probe/labels`, which CURATED-PAI still needed. The labels
were rebuilt (3.9 GB, 10-20 11:23) and are kept. Re-check readers right before any deletion, not only here.

## Where the disk is (scan time, GB)

| top level | GB | biggest parts |
|:--|--:|:--|
| `datasets` | 4 057 | `nurec/all-usdzs` 2 344, `navsim` 644 (`sensor_blobs` 595), `alpasim_nuplan/navtest` 595, `waymo_e2e/front1` 235, `hugsim` 121, `physical_ai_av` 38, `nuscenes` 35 |
| `runs` | 2 224 | `op_parity` 960 (`cache` 800 -> about 484 after the swap, `runs` 86, `hugsim` 46), `defect_fix` 179, `alpasim` 170 (`ap2` 129), `pai_rows` 138, `route_follow` 86, `practice_pai` 83, `pai_fix` 68, `alpasim_proxy` 59, `route_mem` 52 |
| `envs` | 105 | 23 envs, 2.6-10 GB each (uv cannot hardlink on this disk, so every env is a full copy) |
| `third_party` | 57 | `carla` 31, `alpasim` 10, `alpamayo1.5` 9 |
| `models` | 49 | `openpilot` 25 (`trt_cache` 18), `gemma-4-E2B-it` 10.5, `vjepa2_1` 5.2, `sam3.1` 3.5 |
| `tools` | 41 | `nre-rootfs` 40 (the AlpaSim runtime rootfs; the box has no docker) |
| `cache` | 40 | `huggingface` 23.5, `uv` 14.0, `cuda` 1.1 |
| `processed` | 39 | `op_adapt/wodval` 14.0, `waymo_e2e/features` 12.5, `drive_backbones` 7.6 |
| `tmp` | 20 | 19 git worktrees `df_old_*` / `nx_old_*` 17.9 |
| rest | 10 | `jev-drive` 8.4 (`.venv` 6.5), `checkpoints`, `snapshots`, `ckpt`, `uv` |

rollout.asl across `runs/`: 7 100 files, 359 GB. The sample rule (keep one per `rollouts` dir) would free 216 GB; the samples left would be
about 143 GB across 751 reads.

## Totals by tier

| tier | grade | items | GB | what |
|:--|:--|--:|--:|:--|
| 1 | safe | 4 | **35** | git worktrees of old revs, the uv cache, NuRec split leftovers of complete scenes, one failed read |
| 2 | probably safe | 24 | **289** | asl of closed reads (lineage, SIMP, NU-FLAT F / FR, CURATED-PAI, closed RTB1 / NR1 arms), dumps of finished diagnostics, row-building temporaries of a stopped line, smoke / identity and uncited checkpoints, concluded-line caches and envs |
| 3 | ask | 20 | **975** (+ 1.5 on the system disk) | the AP2 token cache, HUGSIM outputs, the live `pai_rows` build parts, paused CARLA / B2D material, reference and fresh R2J reads, cited checkpoints, kept-on-purpose rolls, models, small old run dirs |
| | | | **1 299** | data disk, no overlaps between items, GB at scan time |

Not counted: the 316 GB the zero-copy swap already freed, and everything in "Kept" below.

## The 15 largest candidates

| # | path | GB | grade | one line |
|--:|:--|--:|:--|:--|
| 1 | `runs/alpasim/ap2/cache` | 128.7 | ask | AP2 / OT token cache of the 10-08 AlpaSim line; only `ap2_*` / `ot3_rows` read it; rebuild = GPU encode of 12 navtrain shards |
| 2 | `runs/pai_rows/rows` | 62.3 | ask | PAI-ROWS (d410) per-scene row files; `pr.sh` rebuilds the pools from them when scenes are appended, and the fetch for an append is running now |
| 3 | lineage asl (`defect_fix/lineage/read/{P0-VL,P0-AG,P0-NAT,NTV0-NAT,RB0-VL}`) | 49.3 | probably safe | zero-score rollout.asl of the LINEAGE-PAI281 reads (d405 成立), 8 samples per read stay |
| 4 | `runs/op_parity/hugsim` | 46.4 | ask | HUGSIM onnx (33.7) and videos (11.3) of the op_parity HUGSIM line; the user excluded HUGSIM on 10-20c |
| 5 | `runs/pai_rows/cache` | 44.6 | ask | PAI-ROWS per-scene token parts (H 33.3, S 11.3); same append dependency as #2 |
| 6 | `runs/hugsim-exam` | 43.4 | ask | HUGSIM exam ply / videos; kept since 10-09 for spin attribution; user-excluded on 10-20c |
| 7 | `runs/bench/hugsim` | 37.4 | ask | HUGSIM bench videos (20.1) and `_onnx` (16.1); user-excluded on 10-20c |
| 8 | pai_fix SIMP asl (`pai_fix/reads/SIMP{A,B,C,F,N}1K-L18-s0-*`) | 34.1 | probably safe | zero-score asl of the six SIMP exam reads (d409 不采用 SIMP); facts csv stay |
| 9 | `third_party/carla` | 31.2 | ask | CARLA 0.9.15, deliberately kept for the paused B2D lines |
| 10 | R2J asl (`round2/pai281/R2J-L18-s0-ag`, `pai_fix/reads/{R2J,R2JH,R2C}-L18-s0-vl`) | 30.7 | ask | tonight's ROUND2 reads; held, see the flag above |
| 11 | `runs/pai_rows/stretch` | 29.8 | ask | PAI-ROWS 0.5x / 1.5x stretch parts; same append dependency as #2 |
| 12 | `op_parity/runs` arms cited only in READMEs / results (not VT) | 24.5 | ask | 309 closed-arm checkpoints, 0.08-0.16 GB each |
| 13 | `runs/practice_nuplan/roll/JTN-L18-s0` | 24.0 | ask | PRACTICE-NUPLAN rolls (driver logs 22); row source of on1_PN0..6 (now pinned by on1_JTX); held on 10-20c for STAND-R |
| 14 | `cache/huggingface` | 23.5 | ask | HF models: Qwen3-VL-4B 8.9, siglip2 4.6, unidepth 2.8, simlingo 2.6, vjepa2 1.3, segformer 1.0, ... |
| 15 | `runs/practice_pai/curated/cache` | 22.3 | probably safe | CURATED-PAI (d402, stopped by registration) per-scene token parts, already concatenated into `on1_CPAI570@warp/front.npy` |

## Tier 1: safe

| path | GB | what | last write | why it looks safe | lost / regenerate | deps |
|:--|--:|:--|:--|:--|:--|:--|
| `tmp/df_old_*` (18) + `tmp/nx_old_936ac891` | 17.9 | detached git worktrees at old revs, for the bitwise identity checks of jt / r2 / nu_flat / simp / lon | 10-17 .. 10-20 | every identity check is done; the scripts recreate a missing one (`[[ -d $WT ]] \|\| git worktree add`, seconds) | nothing | remove with `git worktree remove` (+ `prune`), not `rm`; the pool is idle |
| `cache/uv` | 14.0 | uv download / build cache | 10-19 | docs/remote-box.md: envs are full copies, `uv cache clean` breaks none | the next `uv` install downloads again | none |
| `datasets/nurec/_dl/{b5eb1918,ca2b3416}-*.usdz.{0..7}` | 2.2 | split download pieces from 10-13 | 10-13 | both scenes are complete in `all-usdzs` | nothing | a fetch is running in `_dl`: delete these exact names only |
| `runs/alpasim_proxy/wait_set/failed/APY10m10-AB-s0-tfirst` asl | 0.9 | 312 asl of a failed wait-set attempt | 10-17 | marked failed; the good read is `wait_set/reads/APY10m10-AB-s0` | nothing | none |

## Tier 2: probably safe

Every line here is closed or stopped by its decision row, and no running or queued job uses it. Summaries, scores, csv, parquet and
`aggregate/` stay in every read; only `rollout.asl` beyond the per-`rollouts` sample goes.

| path | GB | what | last write | lane / decision | lost / regenerate | deps / note |
|:--|--:|:--|:--|:--|:--|:--|
| `defect_fix/lineage/read/{P0-VL,P0-AG,P0-NAT,NTV0-NAT,RB0-VL}` asl | 49.3 | zero-score asl of five PAI 281 reads (P0 VC+LEAD / ACC+GAP / native, NTV0 native, RB0-F VC+LEAD) | 10-20 17:40 | LINEAGE-PAI281 d405 成立 | replay of those zeros; re-read 0.7-0.9 card-h each | 8 samples per read stay; RB0-NAT is in tier 3 (lead_break source) |
| `pai_fix/reads/SIMP{C,N,F,B,A}1K-L18-s0-{vl,ag}` asl | 34.1 | six PAI-FIX exam reads of the SIMP arms and their control SIMPN1K | 10-20 19:30 | SIMP d409 不采用 | re-read 0.44 card-h each | facts in `pai_fix/facts/*.csv` stay; a re-tax needs the asl |
| `practice_pai/curated/cache` | 22.3 | per-scene token parts of the d402 rolls | 10-20 10:30 | CURATED-PAI d402 按登记停 | rebuilding on1_CPAI570 from scratch needs a re-encode (GPU); the pool itself stays | `cp.sh pool` / `r1` read it, but the line is stopped; `curated/rows` is kept (round2 `lon_anchor.py` reads it) |
| `defect_fix/native_gap/dump` | 17.0 | nuPlan driver-log dumps (drive.jsonl 16.5) of NATIVE-GAP | 10-19 11:20 | d372 成立 (diagnostic); held on 10-20c only for a lat_diag loop that is gone | a re-dump costs one arena read | `lamp_pixels.py` (d373 done) and `lat_diag.py` (PN closed) read it; `dec.parquet` / `set_census.parquet` beside it stay |
| `runs/body1` | 18.1 | g2 dumps 7.6, labels 3.2, road labels 2.9, cl reads | 10-11 | body1 concluded | rebuild from the g2 / label scripts (GPU, hours) | readers are body1's own scripts only |
| `op_parity/runs`: 195 arms cited nowhere (not in decisions, decision files, READMEs, results or code) | 16.4 | mostly the 10-09 Q*-F sweep (QS / QT / QN / QL / QC / QF / QV / QX ...) | 10-09 | closed | retrain about 0.3-1 card-h each | list in `ckpt_runs.tsv` + Mac `tmp/inv_1021/ckpt_cls.tsv` (class `none`, sub `arm`); VTCP2-s1 moved to tier 3 |
| `processed/op_adapt/wodval` | 14.0 | WOD val features of the op_adapt line | 09-30 | op_adapt concluded (d67-81); 10-14 ask list | GPU re-extract | `op_parity/wod_parity.py` names it |
| `processed/waymo_e2e/features` | 12.5 | WOD-E2E frozen features | 10-08 | elicitation / real_transfer concluded | GPU re-extract | `jevdrive/elicit_e1.py`, `real_g0/g1.py`, `nq3_q6.py` (concluded) |
| `runs/lowboard_diag` | 10.6 | PAI pkl caches (`pai*/x`) and small reads | 10-10 | concluded | CPU rebuild | none live |
| `op_parity/runs` smoke / identity runs (106: `*GG-old/new`, `*-IDENT-*`, `smoke-*`, `*smk*`, `*SM0*`, ...) | 9.2 | checkpoints of 60-step identity checks and smokes | 10-10 .. 10-20 | identity results are recorded in the prereg supplements | a rerun is minutes each | pattern list; check each name against the table before deleting |
| `practice_pai/curated/read/{CPAIC1K,CPAIP1K}-L18-s0` asl | 8.6 | CURATED-PAI exam reads | 10-20 12:20 | d402 按登记停 | re-read about 0.9 card-h each | |
| `defect_fix/leadstart_probe` + `op_parity/cache/idx_leadstart_tv@warp` | 8.6 | LEADSTART-PROBE features (4.8) and its pool (3.4) | 10-19 | d383 diagnostic | rebuild (GPU encode) | `leadstart_probe.{py,sh}` only; split def `leadstart_tv.v1.json` stays |
| `processed/drive_backbones` | 7.6 | frozen-backbone features | 09-26 | driving_backbones d40 concluded | GPU re-extract | concluded-line modules in `jevdrive/` |
| `runs/navsim/metric_cache/v1_e6sub` | 7.0 | PDM v1 metric cache, navtrain subset | 09-26 | op_parity turn_ceiling / elicit_e6 (concluded) | CPU hours | navtest caches (`v1/v2_navtest`) stay for the bench |
| `runs/long_contact` | 6.2 | LONG-PROBE reads / features | 10-11 | concluded | rebuild | none live |
| NU-FLAT F / FR asl: `defect_fix/jt/arena/{B2F,B2FR}-L18-s1@off`, `route_mem/read/{B2F,B2FR}-L18-s1` | 6.0 | arena held + held 460 reads of the two NU-FLAT arms | 10-20 23:04 | d411 不采用 | re-read about 1 card-h each | the CTRL B2C reads are in tier 3 |
| `defect_fix/launch_diag/dump` | 5.7 | driver dumps of LAUNCH-DIAG | 10-18 | d363 diagnostic; STAND line stopped (d400) | a re-dump is one wait-set read | `stand.py` / `free_m.py` "dump" / "late" jobs read it (STAND / FREE-M closed) |
| `datasets/nurec/_dl/*.usdz.part` (9 of the 10-20 fetch) | 5.6 | partial downloads of 9 tranche-2 eligible scenes (rest / bend) | 10-20 09:20-18:55 | not on disk, not in `download_queue_v2.csv` | a later fetch of these scenes downloads 5.6 GB again | do not touch the 3 parts the running fetch writes (`08c25bed`, `1e3402b7`, `a3a76c29` at 00:01) |
| `defect_fix/redline_probe/dump` | 4.8 | driver dumps of REDLINE-PROBE | 10-19 | d373 diagnostic | a re-dump is one read | `lamp_pixels` / `warp_loop` scripts (done) |
| `route_mem/read/{RTB1N-L18-s0,RTB1O-L18-s0,RTB1O-L18-s1,RTB1OB1-L18-s1}` asl | 4.7 | held 460 reads of closed RTB1 arms | 10-17 | RTB1 family closed | re-read about 1 card-h each | RTB1ARO-L18-s0 is in tier 3 (`rm_align.py` opens it) |
| `envs/{sam3,ultralytics}` + `models/{sam3.1,ultralytics}` | 15.6 | SAM 3 / YOLO envs and weights | 09-25 .. 10-12 | fastperc d45, elicitation, cosmos: all concluded; 10-14 tier 2 | re-create the env (about 10 min) and re-download | readers are concluded-line modules only |
| NR1* wait-set reads asl (`alpasim_proxy/wait_set/reads/NR1*`) | 2.3 | wait set v2 reads of the closed NR1 arms | 10-17 | NR1 line closed (d346) | re-read minutes each | the anchor reads (APY / YR / RSG / G0014) are in tier 3 |
| `alpasim_proxy/pai_pcs/smoke/{GS,G0014r2,G0014r3}` asl | 2.1 | three 2-scene smoke reads | 10-19 | pai_pcs smoke (d392 done) | minutes | |
| `op_parity/cache/on1_PN6` + `@warp` | 1.2 | PN6 pool | 10-19 | practice_nuplan closed (d385); not in on1_JTX's manifest | rebuild from the PN rolls | `lat_diag.py` reads PN0 only |

## Tier 3: ask

| path | GB | what | last write | why ask | lost / regenerate | deps |
|:--|--:|:--|:--|:--|:--|:--|
| `runs/alpasim/ap2/cache` | 128.7 | AP2 token cache (12 navtrain shards of `bw` 6.8 GB + `cold` 2.8 GB); no duplicates (sample hashes differ) | 10-08 | kept on purpose 10-14 / 10-20b for `rf_cf.py`, which now reads `ap2/route` only; readers left: `ap2_train.py`, `ot3_rows.py`, `ap2_offline.py`, `ap2_prep.py` (AP2 / OT2 / OT3, 10-08 .. 10-10) | rebuild with `ap2_prep.py` (GPU, several card-h); AP2-AB-s0 is decision-cited | single biggest item |
| HUGSIM outputs: `op_parity/hugsim` 46.4, `hugsim-exam` 43.4, `bench/hugsim` 37.4, `opctrl/closed` + `opctrl_long/closed` 15.8 (scene ply 14.2), `hugsim-wajepa` 4.5, `hugsim-{derot,lowsel,lowspeed,launchlong}` 3.1 | 150.6 | onnx exports, per-episode ply, videos | 09-25 .. 10-11 | the user excluded HUGSIM on 10-20c; `hugsim-exam` ply feed spin attribution / `export_review_cases.py` | re-export onnx (minutes), re-run episodes (hours) | `bench/hugsim/_onnx` is read by the bench when a HUGSIM run starts |
| `runs/pai_rows/{rows,cache,stretch}` | 136.7 | PAI-ROWS per-scene rows 62.3, token parts 44.6, stretch parts 29.8 | 10-20 22:05 | live: `pr.sh` rebuilds `on1_PAIH*` / `on1_CPAIX` from them when scenes are appended (`scenes.tsv` + `pr.sh chain H\|S`), and the 64-scene fetch for that is running | rebuild 5.5 card-h | the pools themselves are pinned by on1_JTPH |
| paused CARLA / B2D: `envs/{scout-tfv6,bridgedrive,garage,simlingo,blue,b2d-tcp}` 49.3, `third_party/carla` 31.2, `runs/op_route_ft` 17.0, `runs/op_wide_ft` 7.6 | 105.1 | leaderboard-model envs, CARLA, route adapter weights | 09-23 .. 10-13 | deliberately kept since 10-10 (docs/storage.md) | env rebuild about 10 min each; CARLA re-extract | b2d_monitor / phone_wheel use Tokyo, not these |
| reference reads asl (sample rule): PAI-FIX `pai_fix/reads/NTV0-L18-s0-{ag,vl}` + `NTV0-L18-s1-ag` 13.6, lineage `RB0-NAT` 10.6, `onpol_long/read/YS*` 7.7, arena `hold/reads/{leash3,lead,full,stop3}` 6.7, `alpasim_long/held/h18` 6.5, `route_follow/acc_gap/read/s2` 5.5, NU-FLAT CTRL `B2C-L18-s1` (jt/arena + route_mem) 3.4, `wait_cap/next/{cont/wide/pub/RTB1*,zvid2}` 3.0, `alpasim/hold_image/native48_jtn_s1_*` 2.7, `route_mem/read/{RSG-L18-s0,RSG-L18-s1,RTB1ARO-L18-s0}` 3.0, wait-set anchors (APY / YR / RSG / G0014 ...) 2.1 | 64.8 | zero-score asl of reference, anchor and control reads | 10-12 .. 10-20 | reference rows of PAI-FIX; RB0-NAT is lead_break's row source (d407 "本探针作离线闸"); `ev.py`, `cap_audit/ca.py`, `easy_items.py`, `rm_align.py`, `nx_*.py`, `stoppt.sh` / `jt.sh` open these by path; acc_gap s2 = private anchor; hold_image = submitted JTN-s1 image | re-read 0.4-1 card-h each | |
| ROUND2 R2J reads asl: `round2/pai281/R2J-L18-s0-ag` 17.4, `pai_fix/reads/{R2J,R2JH,R2C}-L18-s0-vl` 13.3 | 30.7 | tonight's reads | 10-20 23:44 | R2J is named a candidate in the brief; d412 says not adopted; `S2STOP` written 00:00 | re-read about 0.9 card-h (281) / 0.44 (PAI-FIX) | hold until the main session decides |
| `op_parity/runs` checkpoints of closed arms cited in READMEs / results (not VT; 309 runs) 24.5, in decision rows / decision files (90 runs) 8.0 | 32.5 | closed-arm checkpoints | 10-06 .. 10-20 | a doc or decision row reads them; reads can be repeated only while they exist | retrain 0.3-1 card-h each | candidates / submitted / controls (18 runs, 1.6 GB) are under "Kept" |
| `op_parity/runs/{VT-*,VTCP2-*}` (12) | 20.3 | vis_train checkpoints, 1.5-1.7 GB each | 10-11 | kept 10-11b "so every read in reads.md can be repeated" | retrain (VT line, hours each) | VTCP2-s1 is cited nowhere, the rest are |
| `runs/alpasim` except `ap2` (`fix1` 7.7, `pai2` 7.0, `lat1` 4.1, `ot3` 3.0, `tr1` 2.9, ...) | 37.7 | early AlpaSim reads, already cut to one asl per `rollouts` dir | 10-08 .. 10-19 | what is left are the samples; `bd1_diag.py` reads `tr1`; `fix1_pai_box.sh` names `fix1` / `pai2` | lose the last replayable sample | |
| route_follow working dirs: `recovery_envelope` 9.2, `p4` 7.4, `guard` 7.2, `stop_probe` 6.6, `pait` 6.2, `rsgacc` 3.6, `rewind*` 4.5 | 44.7 | rolls, rows, features, videos | 10-13 .. 10-18 | route_follow is live; readers: `pai_pcs_local.py`, `rf_dagger.py`, `turn_gallery.py`, `rm_align.py`, `nx_*.py`, `rf_stopprobe.sh`, `pp.sh` (rsgacc chunks) | rebuild from rolls / reads | |
| small run dirs of closed or paused lines (40 dirs, 0.4-5.8 GB each; list below) | 70.5 | results, small rolls, features | 09-23 .. 10-13 | per-dir value differs (road_probe = d275 paper evidence, eyes_vs_head = `ev.py`, steer_long / alpasim_family = live READMEs, junction_eyes `step0` kept on purpose) | varies | |
| `runs/practice_nuplan/roll/JTN-L18-s0` | 24.0 | PN rolls (driver logs 22.0) | 10-19 01:43 | row source of on1_PN0..6 (pinned by on1_JTX); held on 10-20c for STAND-R relabels; PN closed (d385), STAND stopped (d400) | re-roll about 4 card-h | |
| `cache/huggingface` | 23.5 | Qwen3-VL-4B 8.9, siglip2 4.6, unidepth 2.8, simlingo 2.6, vjepa2 1.3, segformer 1.0, dinov2 0.7 | 09-19 .. 10-08 | unidepth / segformer readers are concluded or paused (factor_wm, nq2); Qwen3-VL-4B / siglip2 still back feature scripts | re-download (hf-mirror slow, docs/storage.md) | |
| `op_parity/cache` pools no manifest pins: `on1_JTCP` (tab, road, drivable, agent; 15.2 + teacher 1.6), `on1_L18RB0J` 9.4, `on1_L18P` 6.3, `on1_L18RSGallheld` 2.9 | 35.4 | JTCP = CURATED-PAI merged pool (now a manifest over CPAI570 + the L18 sources); L18RB0J = route_follow `rf_v2` pool; L18P = onpol_long P arm (dropped from the JT recipe, "amendment B"); RSGallheld = RSG held rolls | 10-12 .. 10-20 | JTCP: d402 / d408 / d409 closed, but `lon_anchor.sh` / `simp.sh` / `cp.sh` name it. L18P: `je0.py` / `rp.py` (d275 paper evidence) and its source rolls were deleted 10-20c, so it cannot be rebuilt cheaply. RSGallheld: `critic_pilot`, `awr_onpol` (prepared, not launched), `rf_dagger` | rebuild from rows where the rows still exist | deleting on1_JTCP removes a manifest, not CPAI570 |
| `runs/op_parity/{wod_slot,wod,turn_selwod,b2d_plan_smooth}` | 16.6 | WOD slot onnx, WOD features, TRT engines, B2D gifs | 10-07 .. 10-11 | op_parity README is live; `wod_*` scripts | rebuild | |
| `runs/{vlm_thin,vlm_cmp}` | 16.8 | VLM thin-head features 9.4, comparison features 7.4 | 10-02 .. 10-03 | vlm_arb paused, its scripts read these; on the ask list since 10-08 | GPU re-extract | |
| `runs/onpol_long/{roll,rows}` | 14.7 | RB0-F-s0 rolls 6.5 (source of on1_L18RB0, the base of every recipe), other rolls, rows 3.5 | 10-12 .. 10-18 | relabelling on1_L18RB0 needs the RB0-F rolls; `df_zero.py`, `rf_rewind.sh`, `rf_entryspeed.sh`, `wc_cl.sh` read `onpol_long/roll` | re-roll | |
| `models/{gemma-4-E2B-it,vjepa2_1}` | 15.7 | small-VLM first look (d182, kept on purpose), V-JEPA 2.1 | 10-07 .. 10-08 | kept by decision | re-download (ModelScope script about 10 min) | |
| `runs/op_dagger` | 6.0 | reprojection rollout clips | 10-05 | op_dagger paused, not concluded | rebuild | |
| `/tmp` strays (system disk, not the data disk) | 1.5 | 602 entries listed by `python3 -m jevdrive.cl.disk prune --dry`; largest `/tmp/tmpm03cqvjh` 1.07 GB, `/tmp/pilm` 56 MB, about 15 x 50 MB `tmp*` | various | the dispatcher never deletes strays; the owners are unknown | | system disk 7.3 / 30 G used |

The 40 small dirs, GB and last write: `op_common_cause` 5.8 (10-03, paused), `op_adapt` 5.6 (09-30), `steer_long` 4.6 (10-13, live),
`vlm_arb_stoppos` 4.1 (10-03), `skill_pack` 3.6 (10-03), `alpasim_family` 3.4 (10-12, live), `op_guard` 3.1 (10-06), `road_probe` 3.1
(10-12, d275), `cosmos_full` 2.9 (09-30, kept samples), `op_img_cmd` 2.9 (10-03, paused), `junction_eyes` 2.7 (10-12, `step0` kept),
`eyes_vs_head` 2.3 (10-12), `b2d` 1.9 (09-23), `vis_train` 1.8 (10-12), `night2` 1.6, `op_layer` 1.6, `route_only` 1.5, `corridor` 1.4,
`pai_op` 1.2, `elicitation` 1.2, `cosmos` 1.2 (kept samples), `op_drive_record_seed0` 1.0, `real-data-transfer` 1.0, `alpamayo_turns` 0.9,
`op_l_b2d` 0.9, `p5v1` 0.8, `leaderboard_audit` 0.8, `drive_backbones` 0.8, `real_vs_render` 0.8, `zeroshot-exam` 0.7, `sched` 0.6,
`lbx_b2d` 0.6, `top10_t2` 0.6, `cn_pair` 0.6, `vlm_arb_vmerge3` 0.6, `fusion_diag` 0.5, `p6` 0.5, `op_drive` 0.5, `waymo_ladder` 0.5,
`wm_vs_reproj` 0.5 (all under `runs/`).

Beyond the sample rule: the asl samples left after all of the above are about 143 GB across 751 reads. Deleting those removes the last
replayable rollout of a read and was never done before; ask per read.

## Kept (not candidates)

| what | GB | why |
|:--|--:|:--|
| `datasets/nurec/all-usdzs` | 2 344 | 1 301 scenes = 860 training-eligible (`pai_av/results/nurec_select.csv`) + 160 val160 + 281 test281. None is in `nurec_train_excluded.csv`, and there are no duplicate names or same-size copies (one 10 MB `.rsync-partial` stub only). The running fetch adds 64 more. |
| other `datasets/` (navsim 644, alpasim_nuplan 599, waymo_e2e 235, hugsim 121, physical_ai_av 38, nuscenes 35, ...) | 1 713 | kept on request; disk-wide duplicate check (same name and size, partial md5): **no real duplicates on the disk**. The same-size groups in `ap2/cache`, `navtrain_full*` and checkpoints are distinct shards. |
| `runs/op_parity/cache` after the swap | about 484 | every pool is either named by a live recipe (`navtrain_full*` / `yr1_*` / `idx_waitcap_*` / `idx_selfiter_*` / `on1_WAIT` = the base trainval of the JT / NTV0 / STAND / R2 commands; `lb_*` = bench) or pinned by the refs table above; the four unpinned exceptions are in tier 3 |
| checkpoints of candidates / submitted / controls / references | 1.6 | NTV0-L18-s0 / s1, R2J-L18-s0, R2C-L18-s0, STANDFMXH-L18-s1 / s0, JTN-L18-s1 / s0, RSG-L18-s0, JTX1K-L18-s0, JTNS1K-L18-s0, B2C-L18-s1, RB0-F-s0, YS-L18-s0 / s1, SH30-F-s0, P2H10-F-s0, P2-F-s0 |
| live inputs | | `defect_fix/lat/L18NR1v2` (12.0, the raster of every JT / STAND / R2 command), `defect_fix/pai_accgap/NTV0-L18-s0` (REF read: `rf_tax`, `pai_gap.py`, `r2.sh tax281`, `stoppt.sh`), `route_follow/pai_input_loop/{N0,tax}`, `practice_pai/curated/rows` (round2 `lon_anchor.py`), `practice_pai/{rows,cache}/PPAI160` and `defect_fix/{inertia,lead_break}` (the inertia / lead-break offline gates in `r2.sh`), `op_probe/labels` (rebuilt for d402), `route_mem/feat`, `pai_fix/{facts,tax}`, `round2/` |
| infrastructure | | `tools/nre-rootfs` 39.7, `third_party/alpasim` 10.1, `jev-drive` 8.4, `models/openpilot` 25.1 and `runs/op_interp/trt_cache` 13.5 (engines of kept models), `runs/op_adapt_H/{runs,onnx}` 15.0 (bench `adapt_h`), `runs/navsim/metric_cache/{v1,v2}_navtest` 7.0 (bench), navsim / openpilot / wajepa / alpasim-ltf / hugsim / jevdrive / op-train envs |
| `datasets/nurec/_dl` live parts and lock files | small | the running fetch resumes them |

## Docker and build caches

The GPU box has no docker; AlpaSim runs from `tools/nre-rootfs` (39.7 GB, kept). For reference, Tokyo (`ssh ujs@100.108.238.8`,
`docker system df`, 10-21 00:05; not pruned, not in scope): 31 images, 70.5 GB (docker counts 68.4 GB reclaimable, 4 in use; the
`jev-alpasim:*` and ECR submission tags share layers, 8.0-8.1 GB each), build cache 19.4 GB (17 entries, all reclaimable), volumes 26.1 GB.
Tokyo `/` has 1.3T free and `/data` 1.8T free, so there is no pressure there.
