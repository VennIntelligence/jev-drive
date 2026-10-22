# P2 swap: WA-JEPA trained with the NTV0 recipe, read on PAI 281 — feasibility and resource estimate (2026-10-23 01:00 CST)

Question: freeze WA-JEPA's V-JEPA 2.1 encoder, train what plays the role of our trained parts with NTV0's batch composition, losses
and step budget, and read it on PAI 281 (ACC + GAP) beside NTV0-L18-s0 and as-shipped WA-JEPA. **Outcome of Phase 0: not buildable
tonight; no training or read was started.** CPU-only inspection, 0 card-hours, 0 bytes of new disk. All facts below were read on the
box on 10-23 00:00–01:00; the estimates are marked as estimates.

## What NTV0 is (the target recipe)

`rm_train.py train --tag NTV0-L18-s0 --init RB0-F-s0` (meta.json on the box, `runs/route_mem/train-NTV0-L18-s0`): 9 500 steps,
batch 128 = 83 navtrain rows (25 % of them anchor rows that only distil Cinque's non-plan heads) + 13 yaw-rate rows (`yr1_navtrain`)
+ 32 S rows (pool `on1_JT`, weights `JT00S.npz`: the 87 012 rows of L18RB0 / L18RTB1 / L18YS / L18Y, 5 100 nuPlan AlpaSim rollouts;
the 82 594 `NR1TV` trainval rows have weight 0; 15 % of logs held out). Losses: plan imitation, navtrain road / agent hinges, S-row
agent + road hinges and lateral terms (lat_onc 0.5, lat_soft 0.1, lat_side 0.5), `nr1_lat` longitudinal term, shape-only hinge
gradient, ho_w 3, lr-scale 0.1, warmup 500. Init RB0-F-s0 (navtrain-trained). Cinque's vision encoder is frozen; every row is a
cached token tensor (8 slots x 32 x 512 fp16 = 256 KB).

## Phase 0 findings

| | Finding | Consequence |
|:--|:--|:--|
| (a) navtrain rows | WA-JEPA needs CAM_L0 / F0 / R0 / B0 x 4 history frames (2 Hz) at 256 x 512. The navtrain sensor blobs were **deleted 10-21**; on disk: 76 logs (2.5 GB, archives `navtrain_{current,history}_3` re-fetched for P1) of 1 310. Our navtrain rows exist only as Cinque tokens. | Re-download all 64 OpenScene navtrain archives (~282 GB compressed; current ~5.7 GB, history ~3.1 GB each). Measured rate: 18 h for the set in Sept (~4.3 MB/s), 66 min for 8.8 GB tonight (~2.2 MB/s): **18–35 h wall**. |
| (a) yaw-rate rows | `ot3_rows.py`: piecewise-linear heading error injected into the history (rotation only), rendered by warping Cinque's slots. | Rebuildable from navtrain raw frames by a per-camera rotation homography over the 4 history frames of 4 cameras (approximation: the camera is not at the rear axle; same approximation as ours). CPU, at load time. |
| (a) S rows | Pool `on1_JT` stores Cinque tokens (`front.npy`) and labels only. Driver dumps (`dump/*.npz`: one JPEG of Cinque's input camera) survive only for L18RB0 (17 429 rows); none for L18RTB1 / L18YS / L18Y. Rollout dirs keep `metrics.parquet`, `drive.jsonl` (ego poses / egomotion per decision), `onpol_sim.pkl` (simulator actor boxes); **no `rollout.asl`, no 4-camera images**. Row metadata has pose_map, the 4 fed history poses, speed / acc / cmd, t_us and t_log_us. | Raw 4-camera frames are **not recoverable**; they must be **re-rendered** from stored states: per row the NuRec nuPlan scene (it has all 8 nuPlan cameras) at the stored ego poses of t0 and the 3 previous decisions (decisions are 0.5 s apart = WA-JEPA's 2 Hz history), actors from the log replay / `onpol_sim.pkl`. A direct gRPC render path exists (`experiments/alpasim_human/scripts/render_probe.py`), but it takes the actor list from an ASL request template, which these rollouts do not keep. Labels (agent / road / lat files) are in the rear-axle row frame and reusable as they are. |
| (b) planner interface | Trainable part = `predictor` (flow transformer, 174.1 M; trajectory flow + scene-latent flow) + `scene_projector` (2.1 M). Frozen: `encoder` 304.7 M (and `target_encoder`, only for its scene loss). Output: 8 poses (x, y, yaw) at 0.5 .. 4 s, rear-axle frame, from 4 Euler steps of a flow from seeded noise. | Horizon, step and frame match our hinges (4 s swept footprint). Differences: the plan is a **sample** of a generative head (hinges must back-propagate through the 4-step sampler with fixed per-row noise); imitation is WA-JEPA's flow-matching loss, not our plan regression; the scene-latent future loss needs 8 future frames x 4 cameras, which S rows do not have (drop it or keep it on navtrain only); Cinque's anchor-row distillation has no analogue (those 25 % become imitation rows); shape-only gradient needs a path / timing split of the 8 poses. |
| (c) AlpaSim driver | WA-JEPA runs on AlpaSim **nuPlan** only (`experiments/alpasim/lib/wajepa_{core,driver}.py`, `run.sh <dir> wajepa`; 48-scene smoke 0.978, held 460 `fam/wajepa0-3`: 115 rollouts per stack in 66 min, driver 6 GB). **No PAI read of WA-JEPA exists.** PAI gives 6 f-theta cameras (front wide 120, front tele 30, cross left / right 120, rear left / right 70), **no rear-centre camera**. | A PAI WA-JEPA driver needs f-theta -> pinhole synthesis of L0 / F0 / R0 (from front wide and the cross cameras) and of B0 (stitched from the two rear-side 70° cameras, partly uncovered), the cold start without history, and ACC + GAP ported from `pai_driver.py`. The as-shipped read depends on the same driver. |
| (d) feature cache | Encoder output `context_scene` is 4 096 x 512 = 4 MB fp16 per sample (P1). Distinct samples: ~103 k navtrain + ~123 k yaw-rate draws + 87 k S rows ≈ 313 k -> **~1.25 TB**, over the 150 GB cap and the box's 1.1 TB free. | Store raw 256 x 512 JPEGs (≈ 400 KB per sample; navtrain ~41 GB, S ~10 GB) and run the frozen encoder **online** in training. |

## Resource estimate for doing it properly

Box: 6 cards, 132 cores; tonight all 6 cards are booked by P1 (sweep + kd-feats) with 13 P2 jobs queued behind them.
Encoder throughput used: P1 fit 1 075 navtrain tokens in 5 x 56 s pool jobs incl. start-up (≈ 6 samples / s per job, fp32, measured);
bf16 batched estimated 15–30 / s (not measured). Render throughput: nuPlan on-policy rollouts 181 k images in 45 min on 8 jobs
(≈ 8 images / s per job incl. driver, measured), 2–3 renderers per card.

| Step | What | GPU (card-h) | CPU (core-h) | Disk peak / final | Engineering | Wall on 6 free cards |
|:--|:--|--:|--:|:--|:--|:--|
| 1 | navtrain raw frames: 64 archives, extract L0 / F0 / R0 / B0, resize to 256 x 512, delete archive | 0 | ~30 (pigz, resize) | ~47 / ~41 GB | 1 h (P1's `get.sh` generalised) | **18–35 h** (proxy-bound; starts at once, overlaps 2–4) |
| 2 | S-row re-render: ~102 k instants (87 k t0 + 3 history per 5 100 rollouts) x 4 cameras ≈ 408 k images at 512 x 256 | 4–8 | ~20 | ~10 / 10 GB | 1–1.5 days: actor list from log replay / `onpol_sim.pkl`, render client from `render_probe.py`, gate = re-render vs stored L18RB0 dumps (front camera), a second gate on hinge labels vs re-rendered view geometry | 1–2 h |
| 3 | yaw-rate rows: rotation homography on 4 cameras x 4 frames | 0 (in the loader) | in training | 0 | 0.5 day + gate vs Cinque's yr rows' injected angles | — |
| 4 | trainer: WA-JEPA flow loss + our hinges (navtrain agent / road, S agent / road / lat_onc / lat_soft / lat_side, nr1_lat) on a differentiable 4-step sample, shape-only split, NTV0 batch 96 navtrain-or-yr / 32 S, 9 500 steps, lr x 0.1, encoder online bf16 | 15–60 (encoder 1.2 M sample-passes = 11–56; planner fwd+bwd through 4 sampler steps +30–50 %) | ~100 (JPEG decode, 128 x 16 images per step) | ckpts ~2 GB | 1.5–2 days incl. gates: hinge parity (same poses -> same value as our lib), zero gradient when inactive, smoke 300 steps (loss falls, no NaN), throughput profile (> 3 h job) | 3–10 h (DDP on 6) |
| 5 | PAI driver for WA-JEPA: f-theta -> 4 pinhole views (B0 stitched), cold start, ACC + GAP | smoke 0.5 | — | — | 1 day + offline view check + one-scene smoke | — |
| 6 | PAI 281 reads (ACC + GAP): as-shipped, trained s0 (s1 for a claim) | 2–6 per read (Cinque 281 ≈ 1 card-h; WA-JEPA heavier inference) | — | logs only | `pai_fix.sh` with a new driver kind, 2 h | ~0.5–1 h per read in parallel chunks |
| **Total** | | **~25–80 card-h** (one seed), +20–65 for a second seed | ~150 | **peak ~60 GB, final ~52 GB** (raw kept for a second seed; ~12 GB if navtrain blobs deleted) | **~5 agent-days** | **~2–3 calendar days** (download-bound; ~1 day of cards) |

## Main risks

1. **B0 does not exist on PAI.** WA-JEPA reads 4 cameras by list position; a stitched rear view is out of distribution for both the
   as-shipped and the trained model. The read then measures camera synthesis as much as the recipe. Check on the as-shipped read first
   (if it is far below its nuPlan level, stop).
2. **Re-rendered S rows differ from what NTV0's S rows were.** Ours were the frames the rolling policy actually saw; re-renders depend
   on reconstructing the actor list without the ASL. The gate on L18RB0 dumps bounds this for one of four sources only.
3. **Hinges through a generative sampler.** Back-propagating through 4 Euler steps with fixed noise trains one noise draw per row; the
   served plan uses seed 1. The hinge may move the sample without moving the flow field everywhere; seed-sweep the hinge value at eval.
4. **Init.** NTV0 fine-tunes RB0 (our navtrain-trained base); the analogue is the shipped WA-JEPA weights, whose navtrain training used
   their losses (incl. the scene-latent loss), not ours. "Same recipe" holds for the NTV0 stage only.
5. **Compute contention.** P1 / P2 hold all cards tonight; the encoder-online trainer is 3–10 h wall on all 6 cards.
6. **Download rate** (2.2–4.3 MB/s through the proxy) sets the calendar; nothing else can start the trainer earlier.

## Unavoidable differences from NTV0 (if built)

Imitation loss = WA-JEPA flow matching (not our plan regression); no anchor-row distillation (Cinque-only heads); scene-latent loss
dropped on S rows (no future frames); S-row images re-rendered (not the rollout's own frames) at 4 cameras; yaw-rate rows warped on 4
pinhole cameras; init = shipped WA-JEPA, not RB0; PAI cameras synthesised into the nuPlan rig (no B0).

## References for the read

NTV0-L18-s0 + ACC + GAP on PAI 281: 0.744, 59 zeros (总表第 366 条; s1 0.738, 62 zeros). As-shipped WA-JEPA on PAI: none.
