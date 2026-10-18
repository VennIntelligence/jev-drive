# self_model: self forward model offline gate

status: live
decisions: 364
index: SELF-FWD: (a) fails 1.14 m vs 0.2; (b) passes AUC +0.080 (contact-led)

**Question.** On AlpaSim nuPlan default-serve rollouts, does a small network map (served plan + ego state at the decision) to the 4 s the
car actually executed (controller + the next seven replans) well enough (median 4 s end-point error < 0.2 m), and does judging offroad /
contact on that predicted sweep beat judging it on the plan (AUC gain >= 0.03, log-clustered CI > 0)?

**Conclusion.** (a) not met: 4 s end-point median 1.14 m (RSG held) / 1.34 m (JTN held) against plan 1.71 m, line 0.20 m; the gap is
longitudinal replanning, not the controller (plan = execution to 2-4 cm for 1 s). (b) met: offroad-or-contact AUC 0.859 -> 0.939, +0.080
[+0.052, +0.121], contact-led (+0.100; offroad +0.022). Prereg branch: room for the mechanism, model not accurate enough; no training design (decision 364).

**Next.** None registered. If reopened: a longitudinal-only forward term (speed profile), conditioned on the policy being trained.

**Read more.** Readout [results/gate.md](results/gate.md). Prereg (Chinese): [plans/2026-10-19-fwd-gate-prereg.md](plans/2026-10-19-fwd-gate-prereg.md); plan item 4 in
[research/next-round/practice-plan.md](../../research/next-round/practice-plan.md).

<!-- files:begin -->
## Files

- `gate.md` (results): readout
- `2026-10-19-fwd-gate-prereg.md` (plans): SELF-FWD …
- `sm_gate.py` (scripts): a small forward model -> the 4 s the car
- `agent_hinge.py` (lib): the plan's footprint must stay out of …
- `drivable_hinge.py` (lib): Footprint drivable-area hinge on a plan
- `practice-plan.md` (research/next-round): 练习计划：按「可归因的试次」改训练管线（2026-10-19，工作方案）

[results/](results/) 2 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 1 entry points
<!-- files:end -->
