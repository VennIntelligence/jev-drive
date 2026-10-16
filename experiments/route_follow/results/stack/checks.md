# STACK cheap checks (prereg `plans/2026-10-16-stack-prereg.md` section 1)

## C1: can the Cinque plan head be sampled?

- No. `net.slices["plan"]` = outputs 917 .. 1907, 990 numbers = ONE MDN component: 495 means (33 x 15, `jevdrive.op_adapt.plan_index`) + 495 spreads.
  No hypotheses, no hypothesis probabilities (the 5 x 991 plan with a probability column in `jevdrive/openpilot/model.py` is the old supercombo).
- The spread is never trained by our recipe: the distillation index keeps 495 positions inside the plan slice, all of them the means
  (`op_adapt.distill_index` takes the first half of every MDN head), and the imitation term reads the means only. Decision 220 read it once
  (2 s lateral std, AUC 0.786 for collisions): a weak uncertainty signal, not a source of candidates.
- Candidates are therefore structured perturbations of the model's own plan, `lib/stack_cand.py` (K = 8): lateral offset d in
  {-1.0, -0.5, +0.5, +1.0} m along each pose's normal, ramped 0 -> d over 0-2 s; curvature gain g in {0.8, 1.2} and arc scale s in {0.75, 0.9}
  (`perturb.branch_poses`, copied as `branch_np`). Box selftest: torch vs numpy max |diff| 3.2e-14 m over 300 plans x 8 candidates.
