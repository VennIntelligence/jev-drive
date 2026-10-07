# One driver on WOD-E2E val: shipped Cinque vs it_dw3-s0, with and without the selector

2026-10-04, lane OD2. Pre-registration: [plans/2026-10-04-od2-prereg.md](../../plans/2026-10-04-od2-prereg.md) section 2. Raw numbers:
[wod.json](wod.json). The WOD column for [one_driver.md](../one_driver.md).

Board: the 479 WOD-E2E val rater frames (478 segments), official RFS port (`jevdrive.waymo.rater_feedback_score`, leaderboard
aggregation = mean over scenario clusters), ADE = mean L2 over the 20 waypoints (0.25-5 s) against the logged future, WOD rear axle.
Deltas: paired bootstrap over segments, 2000 draws, 95% CI. Single seed, val only (no test submission).

## Main rows (no trick)

| driver | RFS | delta vs shipped | ADE (m) | delta ADE vs shipped | selector picks |
|:--|--:|:--|--:|:--|:--|
| 1 shipped Cinque | 8.004 | - | 2.464 | - | - |
| 2 shipped + selector r0.6 | 8.004 | +0.000 [+0.000, +0.000] | 2.465 | +0.001 [+0.000, +0.002] | 1 / 479 (0.2%) |
| 3 it_dw3-s0 | 7.979 | -0.026 [-0.118, +0.062] | 2.487 | +0.023 [-0.010, +0.059] | - |
| 4 it_dw3-s0 + selector r0.6 | 7.979 | -0.026 [-0.118, +0.062] | 2.487 | +0.023 [-0.010, +0.059] | 0 / 479 (0%) |

The navtrain refit of the ratio for it_dw3 (od2_ratio chain) returned 0.6, so there is no extra ratio row.

## Per-board trick: longitudinal x1.06 (labelled, separate)

The plan's WOD longitudinal coordinate times 1.06 (the val two-fold cross-fitted factor, experiments/skill_pack/README.md; same
factor as the 7.921 test submission).

| driver + x1.06 | RFS | delta vs shipped (no trick) | delta vs shipped x1.06 | ADE (m) |
|:--|--:|:--|:--|--:|
| 1 shipped x1.06 | 8.120 | +0.116 [+0.023, +0.209] | - | 2.480 |
| 2 shipped + selector r0.6 x1.06 | 8.120 | +0.116 [+0.023, +0.209] | +0.000 | 2.481 |
| 3 it_dw3-s0 x1.06 | 8.080 | +0.075 [-0.048, +0.195] | -0.040 [-0.124, +0.042] | 2.512 |
| 4 it_dw3-s0 + selector r0.6 x1.06 | 8.080 | +0.075 [-0.048, +0.195] | -0.040 [-0.124, +0.042] | 2.512 |

x1.06 on it_dw3 alone: +0.101 [+0.012, +0.189] over it_dw3 without it. x1.06 was fitted on these frames for the shipped model
(cross-fit), so its shipped gain is in-sample-adjacent; for it_dw3 it was not refitted.

## Selector on WOD

- Rule as on NAVSIM (decision 94): second rollout with the history frames re-projected in place to the t0 heading (rot0); its
  plan is kept iff its summed 0-4 s lateral plan std < 0.6 x the native one.
- It almost never fires on WOD: the rot0 / native std ratio has median 1.08 (shipped) / 1.07 (it_dw3), 5th percentile 0.99 / 0.98,
  minimum 0.59 / 0.61. Below 0.8: 0.6% of frames for both; below 1.0: 9.0% / 10.4%. Rows 2 and 4 equal rows 1 and 3.
- rot0 on every frame (no selector, diagnostic only): shipped 7.768, -0.236 [-0.372, -0.121]; it_dw3 7.784, -0.220 [-0.367, -0.090];
  ADE +0.14 / +0.29 m. As on navtest / navtrain (decision 94), removing the real history rotation hurts on real driving; the std
  test correctly keeps the native plan.

## Procedure and what differs from navtest / navhard

- Path: op_adapt_l / op_adapt_H's training port (stage 1-3 frozen + stage 4 + policy) on 10 frames on the 5 Hz lattice
  (f-18 .. f, pairs 0.2 s apart = the 9 policy context slots), from a zero state; the same path as decision 78's and the H chain's
  WOD readouts. A clip gap keeps the contiguous tail with a zero previous image at its start (r2 rater-cache rule; 1 frame).
  Shipped through this path reads 8.004 (op_adapt_l readout O: 8.004); it_dw3 7.979 (readout H-it_dw3-s0: 7.978).
- Serving-path check: the WOD exam runner (`scripts/wod_zeroshot_openpilot.py`, 10 s of WOD frames at 20 Hz into the TensorRT
  model; new `--onnx / --tag` flag) with it_dw3-s0's serving ONNX: shipped 8.005, it_dw3 7.978, -0.027 [-0.118, +0.059]; x1.06
  8.119 / 8.080. Port and serving agree to 0.001, unlike navhard (0.23 gap). The selector was not run on the serving path (the 10 s
  warm-up has no "9 history frames" to rotate, and it does not fire on the port path).
- navtest / navhard ran op_lb (serving ONNX, Cinque without pre-roll, 1.5 s of history: 4 real 2 Hz keys + GIMM-interpolated
  frames). On WOD every history frame is a real 10 Hz camera frame; the port's 1.8 s context is the counterpart.
- rot0 headings: NAVSIM gives ego poses with heading; WOD-E2E gives only 4 Hz past positions and velocity vectors (t0 ego frame).
  The history heading is the past velocity direction (rear axle) minus its t0 value, held at the nearest sample with |v| >= 1 m/s,
  no rotation when the whole past is slower (14.6% of frames), interpolated to the image times. The warp is the same pure rotation
  about the vehicle origin (`op_adapt_h.warp`, = `op_interp.warp_frame` with the position kept). Sign check before scoring
  (`h_wod.py check`, 25 frames with |heading at -1.8 s| > 5 deg): median horizontal phase-correlation shift of the oldest frame
  against t0 drops from 137 px (native) to 25 px (rot0).
- it_dw3-s0 was trained on WOD train (r2-train) frames among its pools; WOD val is held out.

## Reading

On WOD the adapted model is not better than shipped: -0.026 [-0.118, +0.062] RFS, +0.02 m ADE, the same paired delta through port
and serving. The selector is inert on WOD (0-1 of 479 frames), so row 4 = row 3. With the x1.06 trick on both sides it_dw3 is
-0.040 [-0.124, +0.042]. Row 4 does not lose significantly on WOD, but it does not gain either; the d101 "best on every board"
reading holds on WOD only in the "not worse" sense, and the point estimate is negative. Per category (op_adapt_l readout,
same plans): Cut_ins -0.44 [-0.99, -0.03] (n 20), Multi-Lane Maneuvers -0.23 [-0.53, -0.01] (n 42), the other 8 categories between -0.06 and +0.10 with CIs across 0 (frame bootstrap, uncorrected for 10 categories).

## Files

- Code: `experiments/op_adapt_h/scripts/h_wod.py` (run / check / score), chain `h_od2_wod.sh`, serving run
  `scripts/wod_zeroshot_openpilot.py --set rater --onnx $DATA_DIR/runs/op_adapt_H/onnx/it_dw3-s0.onnx --tag Oit_dw3-s0`.
- Box: plans `$DATA_DIR/runs/op_adapt_H/wod/plans.npz` (mean + std, 2 models x native / rot0), chain files
  `$DATA_DIR/runs/op_adapt_H/chain/od2_wod/`, serving predictions `$DATA_DIR/processed/wod_zeroshot/preds/op_cinque_Oit_dw3-s0/`.
