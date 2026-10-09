# body1: teach the student its own body (swept-footprint contact)

status: live
decisions: (inputs: 220, 219, 218, 212, 205, 204, 203, 200, 198, 197, 192, 170, 166, 160, 158)
index: Row set 5.8 M rows; G1 has >= 56 hold positives per class
key: experiments/body1/plans/2026-10-10-body1-prereg.md, experiments/body1/results/taxonomy.md, experiments/body1/lib/sweep.py, experiments/body1/scripts/bd1_rows.py, research/next-round/self-model.md, experiments/alpasim/results/swerve_clearance.md, experiments/alpasim/scripts/swv1_lib.py, experiments/op_parity/lib/agent_hinge.py, experiments/op_parity/lib/drivable_hinge.py, experiments/op_parity/scripts/ot_rows.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/serve_fix.py

**Question.** Does a learned prediction of "the swept footprint of my own plan will touch an object or leave the road", read by a
stop / re-plan in the driver, lower the zero-score scenes of P2H10 on the AlpaSim nuPlan track without costing progress?

**Conclusion.** None yet.

**Next.** Deliverable 1 is done (taxonomy, 5.8 M contact rows, no gate read yet): [results/taxonomy.md](results/taxonomy.md). Contact predictor arms S0 (frozen tokens), S1 (LoRA), S2 (light branch)
against the offline gate (AUC >= 0.80 on held-out navtrain logs and on decision 220's AlpaSim decisions); then the stop arm in
closed loop on the 700 scenes.

**Read more.** [plans/2026-10-10-body1-prereg.md](plans/2026-10-10-body1-prereg.md); taxonomy and row set in [results/taxonomy.md](results/taxonomy.md); concept in
[research/next-round/self-model.md](../../research/next-round/self-model.md).

<!-- files:begin -->
<!-- files:end -->
