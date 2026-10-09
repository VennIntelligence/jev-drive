# WOD1 pre-registration: do the 2026-10-09 AlpaSim-motivated recipe changes move zero-shot open-loop quality on WOD-E2E val?

2026-10-09. Committed before any score of the new arms is read. Lane WOD1. Results: `results/wod1_recipes.md`, tables `results/wod1_recipes/`.

## Question

The recipes P2-F (no hinge), SH30-F (lambda 30), OT10a05-F (lambda 10 + 10% off-track rows) and YR10m10-F / YR10m25-F (lambda 10 + yaw-rate rows 10% / 25%) were
chosen for AlpaSim closed-loop stability. All train on nuPlan navtrain only with the frozen openpilot encoder. Does any of them change the open-loop quality on
another domain (WOD-E2E val)? Prior: no change (closed-loop stability fixes); one possible exception: yaw-rate rows decouple the injected yaw rate from the heading bias
and reduce reliance on ego-motion history, which might transfer better.

## Method (existing path, nothing new)

Exactly the decision 155 harness as used by decisions 162 / 172 / 176 / mixed-domain: `pp_hugsim.py onnx` (serving ONNX with the arm's plan weights and an `intent_bias`
input) + `pp_wod.py bias` (WOD past states + intent -> adapter bias) + `scripts/wod_zeroshot_openpilot.py --set rater extra` (real 10 Hz frames fed twice, 10 s warm-up,
true roof height), served through the GPU pool, then `scripts/wod_slot.py report` (`mixed_domain.Wod`, `wod_launch_report.Ctx`). Arms (2 seeds each, s0 and s1,
`runs/op_parity/runs/<tag>/ckpt-final.pt`): P2-F, OT10a05-F, YR10m10-F, YR10m25-F served new; P2H10-F and SH30-F use the stored predictions of decisions 155 / 172.
An arm is the per-frame mean of its two seeds' scores.

AP2H10-AB is skipped: its input standard (AlpaSim driver, `ap2_driver.py`) has no mapping from WOD in this harness (`pp_wod.wod_ego` builds the NAVSIM standard ego
features); feeding it would need new input-mapping code, which this lane does not write.

## Metrics

- Primary: WOD-E2E val Rater Feedback Score, 479 rater frames, cluster-mean RFS (`jevdrive.waymo.rater_feedback_score` / `rfs_by_cluster`), as every earlier WOD decision.
- Secondary: ADE@3s and ADE@5s over the 1 437 frames with futures (the same set as decisions 155 / 172).
- Contrast: arm minus P2H10 (2-seed mean arm vs 2-seed mean P2H10), paired; 95% CI = percentile bootstrap over sequences (the clustering unit of decisions 155 / 172 /
  176, B 4000, `Ctx.ci`; ADE over sequences with the same draws). Strata (standstill, moving, turn intent, night, day) are reported but carry no label.

## Reading lines (per arm vs P2H10, on the primary metric)

- "transfers": d RFS > 0 with the 95% CI excluding 0.
- "hurts": d RFS < 0 with the 95% CI excluding 0.
- "no measurable change": otherwise. Detectable size: earlier P2H10-vs-SH30 contrast CI half-width was about 0.05 RFS, so changes below roughly 0.05 are not detectable.

Five contrasts are made against the same reference (P2, SH30, OT10a05, YR10m10, YR10m25 vs P2H10; AP2H10-AB skipped, so six arms including the reference), with no multiple-comparison
correction. One CI excluding 0 among five is expected to occur by chance about a quarter of the time, so a single marginal arm is read as "weak candidate", not as an effect; a
claim needs a CI that excludes 0 by a clear margin or two arms of the same family (YR10m10 and YR10m25) agreeing in sign. The yaw-rate prediction is checked by YR10m10 and YR10m25 together.

## Checks before reading (G0)

Stored P2H10 (7.708) and SH30 (7.734) scores reproduce within 0.002 through the report script; shipped 8.005. Otherwise stop and report. Each new arm's serving must cover
479 / 479 rater and all extra frames. Any job exiting 137: save the last 2 minutes of `$DATA_DIR/runs/boxwatch/<date>.tsv` and report at once. At most 3 WOD eval jobs at a time.

## Out of scope

No training, no new runner, no WA-JEPA, no change to the serving harness. Does not touch AlpaSim or closed-loop numbers.

## Amendment 1 (2026-10-09, before any new-arm score was read)

Scope trim from the user (over-search concern). No RFS / ADE of any new arm (P2-F, OT10a05-F, YR10m10-F, YR10m25-F) had been read when this was written; the only
comparison done so far is a bit-identity check of the re-served P2H10-F-s0 predictions against the stored ones (max abs diff 0.0).
- Primary comparisons are exactly two: YR10m10 vs P2H10 and SH30 vs P2H10 (read by the lines above, 2 contrasts). Every other arm (P2, OT10a05, YR10m25) is a secondary row and gets no label.
- No further eval is launched for P2-F or YR10m25-F beyond what had already been launched: P2-F s0 / s1 had finished and YR10m25-F s0 / s1 were already running when the trim arrived; they finish and are reported as secondary rows only.
- The "yaw-rate prediction" check by YR10m10 and YR10m25 together in the section above is demoted: YR10m25 is a secondary row, not a second primary.
- Concurrency lifted by the coordinator to up to 8 eval jobs (lane cores <= 110); wave 2 and 3 ran 6 jobs together. Rule kept: on any rc 137, save the boxwatch rows and drop to 3.
- AP2H10-AB stays skipped (needs new input-mapping code).
