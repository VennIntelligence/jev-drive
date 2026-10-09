# body1: teach the student its own body (swept-footprint contact)

status: live
decisions: 223 (inputs: 220, 219, 218, 212, 205, 204, 203, 200, 198, 197, 192, 170, 166, 160, 158)
index: Contact head on frozen tokens: AUC 0.905 hold logs, 0.885 on d220 decisions
key: experiments/body1/plans/2026-10-10-body1-prereg.md, experiments/body1/results/taxonomy.md, experiments/body1/results/s0_gate.md, experiments/body1/results/stop_closed_loop.md, experiments/body1/lib/serve_body.py, experiments/body1/lib/contact_head.py, experiments/body1/scripts/bd1_train.py, experiments/body1/scripts/bd1_gate.py, experiments/body1/lib/sweep.py, experiments/body1/scripts/bd1_rows.py, research/next-round/self-model.md, experiments/alpasim/results/swerve_clearance.md, experiments/alpasim/scripts/swv1_lib.py, experiments/op_parity/lib/agent_hinge.py, experiments/op_parity/lib/drivable_hinge.py, experiments/op_parity/scripts/ot_rows.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/serve_fix.py

**Question.** Does a learned prediction of "the swept footprint of my own plan will touch an object or leave the road", read by a
stop / re-plan in the driver, lower the zero-score scenes of P2H10 on the AlpaSim nuPlan track without costing progress?

**Conclusion.** None yet in closed loop. Offline, the contact predictor on frozen Cinque tokens (arm S0) passes both gates: own-plan
agent contact AUC 0.905 [0.868, 0.934] and boundary 0.978 [0.969, 0.985] on hold logs, 0.885 [0.774, 0.969] on decision 220's AlpaSim
decisions (one of the two seeds alone misses the lower bound): [results/s0_gate.md](results/s0_gate.md).

**Next.** The stop arm (prereg 4.1, Amendment 2) stopped at its one-chunk check: on 234 scenes of one seed the stop removed 2 of 4 at-fault
collisions with no new zero, but seven false stops cost more (-0.0067 [-0.0159, +0.0011], slow 43 -> 50), and two checklist items failed (ego
slower after a flag 73 % < 80 %; one acceleration command of -8.53 m/s^2): [results/stop_closed_loop.md](results/stop_closed_loop.md). Open,
for the main session: a new amendment before any rerun (persistence of the flag, cold-start and low-speed decisions, the braking reference),
or go to the re-plan arm 4.2; the full 700 x 2 read, m = 1 and the PAI read (Amendment 2 item 9) were not run.

**Read more.** [plans/2026-10-10-body1-prereg.md](plans/2026-10-10-body1-prereg.md); taxonomy and row set in [results/taxonomy.md](results/taxonomy.md); concept in
[research/next-round/self-model.md](../../research/next-round/self-model.md).

<!-- files:begin -->
<!-- files:end -->
