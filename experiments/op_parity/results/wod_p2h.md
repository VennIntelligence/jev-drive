# P2H (P2 + drivable hinge) on WOD-E2E val: RFS -0.30 and ADE +0.28 m (3 s) against shipped Cinque; the loss comes from the ego-state bias, not the fine-tuned weights

Written 2026-10-07. Measurement only (no training, shipped and P2H weights untouched). Code: `scripts/pp_wod.py` (`bias`: WOD inputs to the adapter bias; `report`:
tables), `scripts/wod_zeroshot_openpilot.py --bias` (the existing serving harness with the `intent_bias` input fed per target). Outputs: `wod_p2h.csv` (all rows with CIs),
`wod_p2h_frames.csv` (per frame), `wod_p2h_meta.json`.

## Answer

Shipped Cinque (decision 34 harness, RFS 8.005 reproduced), P2H10-F-s0 / s1 served through the same harness, WOD val, unified interface (true roof height, real 10 Hz frames fed
twice, 10 s warm-up). Paired differences, bootstrap over sequences (B 4 000).

| | RFS (479 rater frames) | d RFS vs shipped [95% CI] | ADE@3s (m, 1 437 frames) | d ADE@3s | ADE@5s (m) | d ADE@5s |
|:--|--:|:--|--:|:--|--:|:--|
| shipped | 8.005 | ref | 1.041 | ref | 2.117 | ref |
| P2H10-F-s0 | 7.718 | -0.287 [-0.496, -0.085] | 1.312 | +0.271 [+0.215, +0.327] | 2.653 | +0.536 [+0.436, +0.637] |
| P2H10-F-s1 | 7.699 | -0.306 [-0.523, -0.097] | 1.337 | +0.296 [+0.238, +0.354] | 2.683 | +0.567 [+0.462, +0.672] |
| **seed mean** | **7.708** | **-0.297 [-0.501, -0.094]** | 1.324 | +0.284 [+0.227, +0.341] | 2.668 | +0.551 [+0.449, +0.655] |

Seed vs seed: s0 - s1 RFS +0.019 [-0.047, +0.088], so the seed noise is small against the effect. **P2H is distinguishably worse than shipped on RFS** (CI excludes 0 for each seed
and the mean; the difference is about 6 times the ~0.05 RFS noise) and worse on ADE at both horizons.

Night / day (luma labels of `night_gap/seq_lum.csv`: night < 50, day >= 120; rater frames 133 night / 325 day, all frames 397 / 977; dusk excluded here), seed mean:

| split | RFS shipped | RFS P2H | d RFS [95% CI] | ADE@3s shipped / P2H | d ADE@3s | ADE@5s shipped / P2H | d ADE@5s |
|:--|--:|--:|:--|--:|:--|--:|:--|
| night | 7.610 | 7.404 | -0.207 [-0.717, +0.325] | 1.308 / 1.312 | +0.004 [-0.102, +0.106] | 2.459 / 2.635 | +0.177 [-0.004, +0.352] |
| day | 8.100 | 7.740 | -0.360 [-0.608, -0.118] | 0.921 / 1.338 | +0.417 [+0.351, +0.486] | 1.952 / 2.694 | +0.742 [+0.618, +0.874] |

Per seed: night d RFS -0.220 / -0.193, day -0.345 / -0.375. The night RFS difference is not distinguishable from 0 (133 frames; CI half-width 0.5), the day one is. The night ADE@3s is
unchanged while day ADE@3s is +0.42 m: the damage is on day frames, where shipped is most accurate (0.92 m); night / day do not separate P2H from shipped on RFS (the two
differences, -0.21 and -0.36, are within each other's CI).

## Diagnostic arms (seed 0 weights, single run each; the other inputs as in the main arm)

| arm | RFS | d RFS [95% CI] | ADE@3s | d ADE@3s | ADE@5s | d ADE@5s |
|:--|--:|:--|--:|:--|--:|:--|
| P2H10-F-s0 as run above | 7.718 | -0.287 [-0.496, -0.085] | 1.312 | +0.271 | 2.653 | +0.536 |
| bias = 0 (P2H weights, inputs zeroed) | 7.984 | -0.021 [-0.056, +0.014] | 1.098 | +0.058 [+0.049, +0.066] | 2.186 | +0.069 [+0.054, +0.084] |
| command zeroed (UNKNOWN) | 7.625 | -0.380 [-0.605, -0.164] | 1.300 | +0.259 | 2.646 | +0.529 |
| ax = ay = 0 | 7.731 | -0.274 [-0.486, -0.070] | 1.425 | +0.385 | 2.849 | +0.733 |
| ax from the position-derived acceleration, ay = 0 | 7.732 | -0.273 [-0.469, -0.086] | 1.171 | +0.131 [+0.077, +0.185] | 2.407 | +0.291 [+0.193, +0.388] |

- The fine-tuned plan pathway itself is neutral on WOD: with the bias zeroed RFS is -0.02 (CI contains 0), ADE +0.06 m. The loss is what the ego-state bias does.
- The command is not the cause (zeroing it changes nothing beyond noise; RFS is if anything lower without it, -0.09 vs the main arm, CI not computed).
- The acceleration input moves ADE a lot (+0.27 / +0.39 / +0.13 m at 3 s for given / zero / position-derived acceleration) but not RFS (-0.27 to -0.29 in all three): the RFS loss
  is not fixed by a better acceleration mapping. These are post-hoc probes of one input mapping, not a tuned arm; the headline row is the mapping fixed before any WOD score was read.
- Median plan 5 s displacement / logged displacement (frames with logged displacement > 2 m): shipped 0.959, bias 0 0.944, P2H s0 / s1 0.918 / 0.914. With the bias the plan is
  4 % slower.

## Setup

- Serving: `wod_zeroshot_openpilot.py --onnx pp-P2H10-F-s{0,1}.onnx --bias bias-<tag>.npz`, the ONNX of the op_parity HUGSIM path (`pp_hugsim.py onnx`; shipped Cinque with the
  arm's trained plan initializers and an `intent_bias` (1, 32, 512) input added to the current frame's tokens and the 8 past policy slots). The bias is computed once per target
  frame by the arm's adapter (ParityAdapter, ego tokens only for P2 / P2H) from the 20 ego features and held for every step of that target's 100-frame warm-up (the bias does not re-enter
  the ONNX state, op_l_onnx.py, so only the last step's value matters). Targets: 479 rater + 958 extra frames, the sets of night_gap.
- Equivalence check: shipped weights through the same bias-input ONNX with zero bias, 60 rater frames, vs the stored shipped run: plan xy mean 0.010 m, p99 0.058 m, max 0.090 m
  (TensorRT fp16 level, as `hugsim_harness.md` test 1).
- Frame protocol: real WOD 10 Hz frames, each fed twice at 20 Hz, i.e. real frames with 0.2 s context pairs and no synthesis, clock factor 1. P2H was trained on protocol W
  (warp-synthesised frames, 0.2 s pairs, 4 slots at 0.2 s on navtrain). The gap and the slot spacing of the model are identical (0.2 s); only the source differs (real here, warp in
  training). That is the closest available to W: Stage A (`stageA.md`) has the gap as the dominant factor, and WOD gives true 0.2 s pairs; warp-synthesising WOD frames would need the
  per-frame motion inputs of pp_prep and would add the synthesis error (W - real, -2.8 EPDMS for shipped on NAVSIM) instead of removing it. Not run: a warp-synthesised WOD arm.

## Input mapping (WOD to the NAVSIM AgentInput features that `parity_adapter.ego_features` / pp_prep use)

| P2 input | WOD source |
|:--|:--|
| command one-hot [left, straight, right] | `intent`: GO_LEFT -> left, GO_STRAIGHT -> straight, GO_RIGHT -> right, UNKNOWN -> all zero (no UNKNOWN in val: 1 258 straight / 90 left / 89 right among the 1 437 frames) |
| 4 poses, 2 Hz, x / 10, y / 10, yaw (rad), current rear-axle frame, oldest first | `past_states` steps 9 / 11 / 13 / 15 (-1.5, -1.0, -0.5, 0 s on the 4 Hz lattice; the WOD frame is rear axle, +x forward, +y left, as NAVSIM); yaw of a key = chord heading of the positions one step either side (WOD stores no yaw), held from the next key when the chord < 0.1 m; yaw at t0 = 0 |
| vx / 10, vy / 10 | speed at t0 from the positions (`waymo.past_kinematics` "v"), vy = 0 |
| ax / 3, ay / 3 | the given accel_x, accel_y of the last past state (which repeats the previous one, `docs/waymo-e2e.md`) |
| present | 1 |

Feature statistics against the NAVSIM navtest rows (`ego_stats.json`; mean / std in the normalised units): speed and poses agree (vx/10 0.50 / 0.50 vs 0.52 / 0.33; x at -1.5 s -0.76 / 0.75
vs -0.76 / 0.49), the WOD acceleration is much smaller in spread (ax/3 std 0.089 vs 0.253, ay/3 0.034 vs 0.225; NAVSIM's are noisier), the straight command is more common (0.875 vs
0.66), vy/10 std 0.00 vs 0.011. The adapter output has the same size on both (bias rms 1.13 on WOD vs 1.16 on navtest rows).

## Caveats

- RFS and ADE are open loop. The ADE of shipped vs P2H differs mainly on day frames at moderate-to-high speed (not broken down further here).
- A distribution shift of the ego features (acceleration spread, yaw from the chord heading, intent mix) is a candidate cause of the ADE loss and the acceleration probe shows the ADE
  is sensitive to it; the RFS loss is not removed by it, so the cause of the RFS loss is not isolated. The adapter was trained only on NAVSIM ego states and warp frames; nothing here
  tests which of the two (ego-state mapping, frame source) matters.
- Night / day: one night RFS CI is wide (133 frames), so no night-specific claim on RFS. The day / night difference of the effect is not significant.
- Diagnostic arms are seed 0 only, one run each; their CIs are against shipped, not against the main arm.
- Extra frames (958) are the sets of night_gap; ADE CIs are bootstraps over the 479 + (sequences of extra frames) sequences, RFS over the 479 rater sequences.
