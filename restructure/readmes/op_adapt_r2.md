# op_adapt_r2: detection tokens, S_jev rule scorer, staged lane

status: superseded-by op_adapt_l
decisions: 67
headline: Stage 1 failed: drift 0.397 m (line 0.10) and +11.7 pp slow rate; replaced by real-log imitation, method not refuted

**Question.** Can CARLA pedestrian pairs plus rule-scored perturbations of the native plan (S_jev scorer, optional detection tokens and tele view) adapt Cinque, including the navhard deficit (DAC)?

**Conclusion.** Only the pre-registration v3 was logged as a decision (adds navhard EPDMS description, DDC term, off-start slot set; decisions 67); the off-start slot set then failed check V6 (`rej` feasible on only 0.471 of dev slots, line 0.90) and was removed in v5. The staged run stopped at stage 1 (2 000 steps, arm A seed 0): checklist failed, drift median 0.397 m (line 0.10 m), null-frame slow rate 26.4% vs 14.7% for the original (+11.7 pp), i.e. a globally more conservative model; no full batch ran. Replaced by real-log imitation (decisions 77, 78), which keeps drift at 0.06 m; decisions 78 notes the two are not a like-for-like comparison, so r2's method is not refuted.

**Read more.** research/navhard-deficit-breakdown.md, research/feature-adapter-domain-shift.md; pre-registration and stage-1 log: `git show bcbdde4:todos/2026-09-29-op-adapt-r2-prereg.md`, hand-off: `git show bcbdde4:tmp/2026-09-30-op-adapt-r2-state3.md`

<!-- files:begin -->
<!-- files:end -->
