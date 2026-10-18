# self_model: self forward model offline gate

status: live
decisions: pending
index: SELF-FWD offline gate: plan + ego state -> executed 4 s (prereg written)
key: experiments/self_model/plans/2026-10-19-fwd-gate-prereg.md, experiments/self_model/scripts/sm_gate.py, lib/agent_hinge.py, lib/drivable_hinge.py, research/next-round/practice-plan.md

**Question.** On AlpaSim nuPlan default-serve rollouts, does a small network map (served plan + ego state at the decision) to the 4 s the
car actually executed (controller + the next seven replans) well enough (median 4 s end-point error < 0.2 m), and does judging offroad /
contact on that predicted sweep beat judging it on the plan (AUC gain >= 0.03, log-clustered CI > 0)?

**Conclusion.** Pending.

**Next.** Run the gate (`sm_gate.py`, one pool job).

**Read more.** Prereg (Chinese): [plans/2026-10-19-fwd-gate-prereg.md](plans/2026-10-19-fwd-gate-prereg.md); plan item 4 in
[research/next-round/practice-plan.md](../../research/next-round/practice-plan.md).

<!-- files:begin -->
<!-- files:end -->
