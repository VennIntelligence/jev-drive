# body1: teach the student its own body (swept-footprint contact)

status: live
decisions: (inputs: 220, 219, 218, 212, 205, 204, 203, 200, 198, 197, 192, 170, 166, 160, 158)
index: S0 passes offline gates: G1 agent 0.905 / boundary 0.978, G2 0.885
key: experiments/body1/plans/2026-10-10-body1-prereg.md, experiments/body1/results/taxonomy.md, experiments/body1/results/s0_gate.md, experiments/body1/lib/contact_head.py, experiments/body1/scripts/bd1_train.py, experiments/body1/scripts/bd1_gate.py, experiments/body1/lib/sweep.py, experiments/body1/scripts/bd1_rows.py, research/next-round/self-model.md, experiments/alpasim/results/swerve_clearance.md, experiments/alpasim/scripts/swv1_lib.py, experiments/op_parity/lib/agent_hinge.py, experiments/op_parity/lib/drivable_hinge.py, experiments/op_parity/scripts/ot_rows.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/serve_fix.py

**Question.** Does a learned prediction of "the swept footprint of my own plan will touch an object or leave the road", read by a
stop / re-plan in the driver, lower the zero-score scenes of P2H10 on the AlpaSim nuPlan track without costing progress?

**Conclusion.** None yet in closed loop. Offline, the contact predictor on frozen Cinque tokens (arm S0) passes both gates: own-plan
agent contact AUC 0.905 [0.868, 0.934] and boundary 0.978 [0.969, 0.985] on hold logs, 0.885 [0.774, 0.969] on decision 220's AlpaSim
decisions (one of the two seeds alone misses the lower bound): [results/s0_gate.md](results/s0_gate.md).

**Next.** The stop arm (prereg 4.1) with the S0 predictor in the driver hook, closed loop on the 700 scenes; operating point at 2 % flags on
clean hold decisions: recall 0.51, first-contact arc error median 1.8 m / p90 7.4 m. Whether S1 (LoRA) / S2 (light branch) are still needed
is open: the agent half is label-limited and weakest in classes 1, 2 and > 45 deg (0.82 to 0.84).

**Read more.** [plans/2026-10-10-body1-prereg.md](plans/2026-10-10-body1-prereg.md); taxonomy and row set in [results/taxonomy.md](results/taxonomy.md); concept in
[research/next-round/self-model.md](../../research/next-round/self-model.md).

<!-- files:begin -->
<!-- files:end -->
