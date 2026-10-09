# The 2026-10-09 AlpaSim-motivated recipe changes do not measurably change zero-shot WOD-E2E val quality (RFS and ADE); the two primary contrasts are null

Written 2026-10-09. Pre-registration [plans/2026-10-09-wod1-recipes-prereg.md](../plans/2026-10-09-wod1-recipes-prereg.md) (+ amendment 1), committed before any new-arm score was read.
Measurement only: existing checkpoints, no training. Code: the existing decision 155 harness (`pp_hugsim.py onnx`, `pp_wod.py bias`, `scripts/wod_zeroshot_openpilot.py`) via
`scripts/wod1_recipes_chain.sh` (pool jobs), report `scripts/wod_slot.py report --out` (`mixed_domain.Wod`). Tables: [wod1_recipes/](wod1_recipes/) (`arms`, `contrasts`, `ade`, `ade_contrasts`, `plan_shift`).

## Answer

No change. Arm = per-frame mean of seeds s0 and s1; WOD val, 479 rater frames cluster-mean RFS; paired percentile bootstrap over sequences (B 4000); ADE over the 1 437 frames.

| arm | RFS | d RFS vs P2H10 [95% CI] | seeds (s0 / s1) | d ADE@3s (m) vs P2H10 | d ADE@5s (m) | label |
|:--|--:|:--|:--|:--|:--|:--|
| shipped | 8.005 | (P2H10 - shipped = -0.297 [-0.501, -0.094]) | | | | |
| P2H10-F (reference) | 7.708 | ref | 7.718 / 7.699 | ref | ref | |
| **SH30-F** (primary) | 7.734 | +0.026 [-0.027, +0.078] | +0.023 / +0.028 | +0.035 [+0.030, +0.041] | +0.047 [+0.035, +0.060] | no measurable change |
| **YR10m10-F** (primary) | 7.733 | +0.025 [-0.052, +0.102] | +0.016 / +0.034 | +0.005 [-0.003, +0.013] | -0.007 [-0.023, +0.010] | no measurable change |
| P2-F (secondary) | 7.680 | -0.028 [-0.063, +0.006] | +0.006 / -0.061 | +0.024 [+0.020, +0.028] | +0.031 [+0.023, +0.040] | secondary, no label |
| OT10a05-F (secondary) | 7.661 | -0.047 [-0.106, +0.004] | -0.083 / -0.011 | +0.006 [-0.001, +0.012] | +0.010 [-0.003, +0.022] | secondary, no label |
| YR10m25-F (secondary) | 7.738 | +0.030 [-0.064, +0.124] | +0.027 / +0.033 | -0.021 [-0.038, -0.006] | -0.056 [-0.085, -0.028] | secondary, no label |

G0 passed: the stored P2H10 (7.708) and SH30 (7.734) reproduce; shipped 8.005. SH30 - P2H10 reproduces decision 172 exactly (+0.026 [-0.027, +0.078]).
All five recipes sit within -0.05 .. +0.03 RFS of P2H10, i.e. all are 0.26 - 0.30 below shipped; none moves the WOD loss of decisions 155 / 162 (the navtrain adapter bias constant).

## Reading

- Both primary contrasts are "no measurable change". Detectable size is about 0.05 (SH30) to 0.08 (YR10m10; CI half-width 0.077) RFS, so a transfer gain larger than ~0.08 is excluded, smaller ones are not.
- Yaw-rate rows (the one possible exception in the prior): YR10m10 and YR10m25 are both positive in point estimate (+0.025, +0.030) with both seeds positive, concentrated at standstill (YR10m10 +0.144 [-0.006, +0.305]; SS stratum +0.327 [+0.114, +0.719]) and ADE@5s at 25% -0.056 m. All of that is stratum-level, exploratory and uncorrected; with six arms and about nine strata each, one CI excluding 0 is expected by chance. Not a transfer claim.
- ADE: the small ADE differences have tight CIs; they are 0.02 - 0.05 m against a 0.28 m P2H10 - shipped gap, i.e. negligible. Hinge strength (SH30) and no-hinge (P2) both worsen ADE slightly (+0.035, +0.024 m).
- SH30 turn-intent stratum +0.199 [+0.004, +0.411] (n 52) repeats the decision 172 observation, exploratory.
- Strata with a CI excluding 0 in the secondary rows are not labelled.

## Setup and deviations

- AP2H10-AB not run: its AlpaSim input standard has no mapping from WOD in this harness (`pp_wod.wod_ego` builds the NAVSIM standard ego features); it would need new input-mapping code.
- `processed/wod_zeroshot/sets.json` (the harness spans file) was missing on the box; the first batch of jobs failed with FileNotFoundError. It was rebuilt from the index with the `build_sets` code path (rater + extra + check sequences; not the test spans; sets.npz untouched). Check: re-serving the stored P2H10-F-s0 under another tag reproduced the stored predictions bit for bit (1 437 frames, max abs diff 0.0). The turn-selector lane had rebuilt a private copy earlier for the same reason (commit 3c7a9574).
- Serving: 8 + zero slot not used; 9 real slots as in decision 155. Concurrency 3, then up to 6 jobs at once after the coordinator lifted the cap; no job was killed (no rc 137).
- Amendment 1 (scope trim): primary comparisons exactly YR10m10 and SH30 vs P2H10; P2-F, OT10a05-F, YR10m25-F secondary rows. P2-F had finished and YR10m25-F had already been launched when the trim arrived.
- Limits: open loop, 479 frames, 2 seeds, one domain, only the harness mapping of decision 155; sub-0.08 effects not excluded.
