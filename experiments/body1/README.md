# body1: teach the student its own body (swept-footprint contact)

status: live
decisions: 223, 224 (inputs: 220, 219, 218, 212, 205, 204, 203, 200, 198, 197, 192, 170, 166, 160, 158)
index: Contact head on frozen tokens: AUC 0.905 hold logs, 0.885 on d220 decisions
key: experiments/body1/plans/2026-10-10-body1-prereg.md, experiments/body1/results/taxonomy.md, experiments/body1/results/s0_gate.md, experiments/body1/results/stop_closed_loop.md, experiments/body1/results/replan_closed_loop.md, experiments/body1/lib/serve_body.py, experiments/body1/lib/contact_head.py, experiments/body1/scripts/bd1_train.py, experiments/body1/scripts/bd1_gate.py, experiments/body1/lib/sweep.py, experiments/body1/scripts/bd1_rows.py, research/next-round/self-model.md, experiments/alpasim/results/swerve_clearance.md, experiments/alpasim/scripts/swv1_lib.py, experiments/op_parity/lib/agent_hinge.py, experiments/op_parity/lib/drivable_hinge.py, experiments/op_parity/scripts/ot_rows.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/serve_fix.py

**Question.** Does a learned prediction of "the swept footprint of my own plan will touch an object or leave the road", read by a
stop / re-plan in the driver, lower the zero-score scenes of P2H10 on the AlpaSim nuPlan track without costing progress?

**Conclusion.** None yet in closed loop. Offline, the contact predictor on frozen Cinque tokens (arm S0) passes both gates: own-plan
agent contact AUC 0.905 [0.868, 0.934] and boundary 0.978 [0.969, 0.985] on hold logs, 0.885 [0.774, 0.969] on decision 220's AlpaSim
decisions (one of the two seeds alone misses the lower bound): [results/s0_gate.md](results/s0_gate.md).

**Next.** Both serving arms stopped at their one-chunk checks. Stop (4.1, Amendment 2): 2 of 4 at-fault collisions removed on 234 scenes,
but seven false stops cost more (-0.0067 [-0.0159, +0.0011], slow 43 -> 50): [results/stop_closed_loop.md](results/stop_closed_loop.md).
Re-plan (4.2, Amendment 3): the hold-log gate passed (3 contacts created on 49 232 clean decisions against 548 resolved), but on 233 scenes
of one seed the lateral shift removed 1 zero and created 2 (taught-class zeros 10 -> 11, -0.0043 [-0.0172, +0.0112]); one shifted decision
redirects every later plan, and 5 of the 10 baseline zeros were never flagged: [results/replan_closed_loop.md](results/replan_closed_loop.md).
Open, for the main session: the loss arm 4.3 (the lesson in the plan itself, needs its own amendment and the navtest guardrail), or a
serving action with a target that persists across decisions; the 700 x 2 reads and the PAI reads were not run for either arm.

**Read more.** [plans/2026-10-10-body1-prereg.md](plans/2026-10-10-body1-prereg.md); taxonomy and row set in [results/taxonomy.md](results/taxonomy.md); concept in
[research/next-round/self-model.md](../../research/next-round/self-model.md).

<!-- files:begin -->
<!-- files:end -->
