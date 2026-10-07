# P2 input parity trained on WOD train: WOD val RFS 8.111 vs shipped 8.005 (+0.11 [-0.06, +0.28], n.s.), ADE@3s -0.47 m; the gain is the WOD inputs, not the fine-tune

Written 2026-10-07. Pre-registration: [plans/2026-10-07-wod-parity-prereg.md](../plans/2026-10-07-wod-parity-prereg.md) (committed before any WOD-trained arm was
scored). Code: `scripts/wod_parity.py` (prep / check / report), `scripts/wod_parity_chain.sh` (the lane), `scripts/pp_train.py` (recipe, unchanged except
sequence-unit splits), `scripts/pp_wod.py bias` + `scripts/wod_zeroshot_openpilot.py --bias` (serving, the decision 155 harness). Tables:
[wod_parity/](wod_parity/) (`full_arms.csv`, `full_paired.csv`, `full_seeds_*`, `pilot_*`, `*_meta.json`).

## Answer

WP2 = the op_parity P2 recipe (Cinque + command / ego state / 4-pose history through the zero-initialised adapter bias, frozen vision, plan pathway
trainable, 0.25 anchor rows distilled to shipped) trained on WOD r2-train with WOD's own inputs (the decision 155 mapping, unchanged). WP1 = the same rows,
order and steps with the adapter absent (fine-tune only). Full recipe: 179 405 rows, 10 000 steps x 128, 2 seeds. WOD val, the decision 155 harness (real
10 Hz frames fed twice, 10 s warm-up), paired bootstrap over sequences (B 4 000).

| arm (seed mean) | RFS (479 rater frames) | d RFS vs shipped [95% CI] | ADE@3s (m, 1 437 frames) | d ADE@3s | ADE@5s (m) | d ADE@5s |
|:--|--:|:--|--:|:--|--:|:--|
| shipped | 8.005 | ref | 1.041 | ref | 2.117 | ref |
| **WP2** (WOD-trained, WOD inputs) | **8.111** | **+0.106 [-0.060, +0.279]** | **0.574** | **-0.466 [-0.534, -0.404]** | **1.457** | **-0.660 [-0.763, -0.564]** |
| WP1 (WOD-trained, inputs zeroed) | 8.012 | +0.007 [-0.022, +0.032] | 0.990 | -0.051 [-0.056, -0.046] | 2.064 | -0.052 [-0.059, -0.046] |
| P2H (navtrain-trained, decision 155) | 7.708 | -0.297 [-0.501, -0.094] | 1.324 | +0.284 [+0.227, +0.341] | 2.668 | +0.551 [+0.449, +0.655] |

| contrast | d RFS [95% CI] | d ADE@3s [95% CI] | d ADE@5s [95% CI] |
|:--|:--|:--|:--|
| WP2 - WP1 (the inputs, on WOD) | +0.099 [-0.066, +0.270] | -0.416 [-0.479, -0.358] | -0.608 [-0.707, -0.515] |
| WP2 - P2H (WOD vs navtrain training; also no hinge) | **+0.403 [+0.210, +0.593]** | -0.750 [-0.830, -0.673] | -1.212 [-1.344, -1.081] |
| WP2 s0 - s1 (seed noise) | +0.005 [-0.042, +0.052] | | |
| WP1 s0 - s1 | -0.009 [-0.024, +0.004] | | |

Per seed: WP2 s0 / s1 RFS 8.114 / 8.109 (+0.109 / +0.104 vs shipped, each CI [-0.06, +0.28]), ADE@3s 0.574 / 0.574; WP1 s0 / s1 8.008 / 8.016.

**Verdict against the pre-registered rules.** WP2 does not *beat* shipped by the rule (CI lower bound -0.060 is not > 0); it is not *equivalent* either (|d|
= 0.106 is not < 0.10); it is not *worse* (CI upper bound +0.279 > 0). The point estimate is +0.11, two RFS noise units, reproduced by both seeds to 0.005,
and the CI is the frame-sampling uncertainty of 479 rater frames, not seed noise. WP2 is distinguishably better than P2H (+0.40, CI excludes 0): training on
WOD removes the decision 155 loss and turns it into a gain. ADE improves strongly and with tight CIs (-45 % at 3 s, -31 % at 5 s).

**Inputs vs fine-tune.** WP1 is shipped to within noise on RFS (+0.007) and gains only 0.05 m ADE: fine-tuning the plan pathway on WOD frames does
almost nothing by itself. Everything WP2 gains comes through the WOD input channels (WP2 - WP1 ADE@3s -0.42 m, CI tight; RFS +0.10, CI contains 0), which is
the mirror image of decision 155 (navtrain-trained inputs carried the loss). Pre-registered rule "inputs useful on WOD" (WP2 - WP1 RFS CI lower bound > 0):
not met on RFS; met on ADE.

**Ceiling.** The logged future itself scores RFS 8.13 on these 479 frames (`docs/waymo-e2e.md`, cluster mean). WP2 imitates the log (targets = logged
future), so its RFS is bounded near that number: WP2 at 8.111 has closed ~85 % of the 0.125 between shipped and a perfect copy of the log. A larger RFS gain
from this recipe is not available by construction; ADE is where imitation shows.

## Night / day (luma labels of `leaderboard_audit/results/night_gap/seq_lum.csv`; rater frames 133 night / 325 day, all frames 397 / 977)

| | RFS night | RFS day | gap (day - night) | ADE@3s night / day | ADE@5s night / day |
|:--|--:|--:|--:|:--|:--|
| shipped | 7.610 | 8.100 | 0.489 | 1.308 / 0.921 | 2.459 / 1.952 |
| WP2 | 7.915 | 8.139 | 0.225 | **0.496 / 0.598** | **1.293 / 1.507** |
| WP1 | 7.642 | 8.111 | 0.469 | 1.230 / 0.882 | 2.375 / 1.913 |
| P2H | 7.404 | 7.740 | 0.336 | 1.312 / 1.338 | 2.635 / 2.694 |

WP2 - shipped: night d RFS +0.305 [-0.077, +0.731], day +0.040 [-0.138, +0.224]; night-gap change dd = d_night - d_day **+0.265 [-0.161, +0.739]**;
ADE@3s night -0.812 [-1.002, -0.639], day -0.324 [-0.380, -0.272]; ADE@5s night -1.165 [-1.451, -0.904], day -0.445 [-0.531, -0.363].

By the pre-registered rule (dd CI lower bound > 0) **no night claim on RFS**: the point estimate halves shipped's 0.49 night gap (decision 131), but 133
night rater frames leave the CI wide. On ADE the night effect is clear: shipped is 0.39 m worse at night than by day at 3 s; WP2 is 0.10 m *better* at night
(not broken down further). WP2 - WP1 night RFS +0.272 [-0.119, +0.702], ADE@3s -0.734.

## Pilot and gates

- **G0 (token path equivalence)**: shipped's plan from the cached val tokens (port fp16) vs the stored harness run `preds/op_cinque` on the 729 rater + extra
  frames at slot >= 9 of a val stream: plan xy mean 0.022 m (gate < 0.05 m: pass), p99 0.19 m, max 2.4 m (one frame), 0.024 m at 4 s.
- **Pilot** (4 874 r2-train / 526 r2-dev rows from 1 889 sequences, 600 steps x 64, seed 0): WP2-pilot RFS 8.032 (+0.027 [-0.080, +0.133]), ADE@3s 0.773
  (-0.268 [-0.315, -0.221]); WP1-pilot 8.019 (+0.014 [-0.005, +0.032]), ADE@3s 0.992. **G1** (WP2-pilot - shipped d RFS >= -0.10): +0.027, **pass** -> full.
- Pilot -> full: RFS +0.027 -> +0.106, ADE@3s -0.27 -> -0.47 m: more WOD data helps the inputs arm; WP1 does not move (+0.014 -> +0.007).
- Training dev (r2-dev, 8 poses 0.5-4 s, ADE to the log): WP2 0.875 (pilot) / 0.628 and 0.630 (full s0 / s1); WP1 1.313 / 1.319 / 1.323; drift of WP2
  with inputs off to shipped 0.042 m (the anchor holds).
- Plan 5 s displacement / logged displacement (median, frames with > 2 m logged): shipped 0.960, WP1 0.961, WP2 0.992 / 0.994, P2H 0.916. WP2 removes
  shipped's 4 % slow bias.

## Data and mapping

- **Frames / tokens**: `processed/op_adapt/wodtrain` (op_adapt round 1 cache): shipped Cinque's stage-3 output of the harness renderer's road + wide frames
  (front three cameras, true camera height), image pair (f - 2, f), at every even frame of the 2 037 WOD train sequences (2 103 contiguous streams, 207 678
  slots). `wod_parity.py prep` runs stage 4 to the `view_39` tokens (1.7 min on one card) and the shipped teacher on the same slots. No new rendering.
- **Protocol**: 9 policy slots at 0.2 s (f - 16 .. f), all real frames, as the harness serves after its 10 s warm-up (the navtrain recipe had 8 + a zero
  slot: deviation, stated in the prereg).
- **Rows**: slot >= 9 of a stream, logged 5 s future present, sequence in `wod/r2-train` (179 405 rows) or `wod/r2-dev` (9 478); intent straight 159 250 /
  left 15 812 / right 13 821 / unknown 0. Splits from `jevdrive.data.splits` by sequence; `check_disjoint(r2-train, r2-dev, wod/val)` passes.
- **Inputs**: `pp_wod.wod_ego` for train and eval alike (intent -> one-hot; past_states 9 / 11 / 13 / 15 as the 4 poses with chord heading; vx from
  positions, vy 0; given accel_x / accel_y). The train rows' ego statistics match val's (ax/3 std 0.070 vs 0.089; vx/10 mean 0.59 vs 0.50, std 0.53 vs 0.50).
- **Targets**: x, y at 0.5 .. 4 s = future_states steps 2, 4, .., 16 (exact on the 4 Hz lattice); yaw = chord heading of the positions 0.25 s either side
  (held from the previous key when the chord is < 0.1 m; 0 at t0). Camera position = FRONT extrinsic translation per sequence.

## Deviations

- No hinge (P2, not P2H; WOD has no drivable labels): WP2 - P2H mixes the training data with the hinge term.
- 9 real policy slots (navtrain recipe: 8 + zero), anchors on WOD frames (navtrain recipe: navtrain frames).
- Serving used the harness as it stood on the box (`51c5540e`: several biases share one warm-up, the last step re-run from a state snapshot); that path
  reproduces the stored decision 155 runs to 0.010 m mean plan xy (`wod_p2h_diag.md`, checks). Shipped and P2H rows are the stored runs.
- Rows come from 2 036 of the 2 037 train sequences (one has no frame at slot >= 9 with a future).

## Cost

Done here at full scale: prep 1.7 min (one card, from the existing trunk cache), check 0.3 min; training 4 x 40-47 min wall at 3.4-4.3 it/s on cards shared
with another lane (alone ~22 min each at 7.6 it/s, the P2H-F rate); serving 6 x 11-25 min (the harness, 1 437 targets). The pilot (2 x 1.4 min training +
2 x 11-24 min serving) was the cheap step; the full run added ~3 card-hours. A larger run (odd frames too, i.e. 2x rows, or r2-dev folded in) would need the
odd-frame stage-3 cache: rendering ~208 k frames (three JPEG decodes each) plus the vision trunk, about 1-2 h on the box; the RFS ceiling above makes it
unlikely to move RFS.

## Caveats

- Open loop. RFS against 479 rater frames: the RFS CI (+/-0.17) is the frame sample, far wider than the seed noise (0.005), so a +0.1 effect cannot be
  separated from 0 on this val set however many seeds are run.
- ADE is against the logged future (the training target), not the top-rated rater trajectory; the ADE gain is the expected effect of imitating the log
  with the ego state known, and the RFS ceiling (log 8.13) bounds what it buys on RFS.
- WOD train rows are 0.2 s apart within a sequence; the effective sample size is far below 179 k.
- Night / day is the luma split of night_gap (dusk excluded); night RFS rests on 133 frames.
