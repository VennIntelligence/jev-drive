# body1: teach the student its own body (swept-footprint contact)

status: live
decisions: 223, 224, 225, 227, 228 (inputs: 220, 219, 218, 212, 205, 204, 203, 200, 198, 197, 192, 170, 166, 160, 158)
index: Contact head passes offline (AUC 0.905 / 0.885); stop and lateral re-plan halted at one-chunk checks; loss arm 4.3 missed its first pilot gate, second attempt (Amendment 5, w = 3) met its pilot gate, full run in progress
key: experiments/body1/plans/2026-10-10-body1-prereg.md, experiments/body1/results/taxonomy.md, experiments/body1/results/s0_gate.md, experiments/body1/results/stop_closed_loop.md, experiments/body1/results/replan_closed_loop.md, experiments/body1/results/zeros_diagnosis.md, experiments/body1/results/loss_pilot.md, experiments/body1/scripts/bd4_train.py, experiments/body1/lib/loss43.py, experiments/body1/scripts/bd1_diag.py, experiments/body1/lib/serve_body.py, experiments/body1/lib/contact_head.py, experiments/body1/scripts/bd1_train.py, experiments/body1/scripts/bd1_gate.py, experiments/body1/lib/sweep.py, experiments/body1/scripts/bd1_rows.py, research/next-round/self-model.md, experiments/alpasim/results/swerve_clearance.md, experiments/alpasim/scripts/swv1_lib.py, experiments/op_parity/lib/agent_hinge.py, experiments/op_parity/lib/drivable_hinge.py, experiments/op_parity/scripts/ot_rows.py, experiments/alpasim/lib/sh30_driver.py, experiments/alpasim/lib/serve_fix.py

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
Diagnosis of all 49 baseline zeros (2 seeds): every one is in a served plan before it happens, the head flags 29 of them a median 2.5 s
ahead but a clear ramp exists at only a third of the flagged decisions; 15 are route failures and 7 a drivable-label gap:
[results/zeros_diagnosis.md](results/zeros_diagnosis.md). The loss arm 4.3 (Amendment 4: agent hinge on the own plan, hinge-only off-track rows incl. the new `bd4`
family, scorer-layer road label) stopped at its pilot gate: the agent hinge on own-plan positives fell 9.5 % against the registered 30 %,
while the hold-log own-plan contact rates did fall (agent -26 %, boundary -17 % against a same-scale switch-off pilot); no full run and no
closed loop: [results/loss_pilot.md](results/loss_pilot.md). **Amendment 5, a disclosed second attempt made after that one pilot read of hold
logs** (one change: the weight w of the hinge terms on hinge-only rows, selected on a validation part of the train logs): w = 3 selected
(w = 10 fails the dev ADE limit), and its pilot gate on hold logs is met: own-plan agent contact -47.7 % [CI excluding 0], boundary -37.4 %,
dev ADE 0.590 against 0.591 m, continuation slope 0.93 against 1.08 (same file, sections A5.1 to A5.3). Found on the way: item C's raster is
not the scorer's road area (the scorer counts car parks and generic drivable areas as road); it stays as trained and is reported as a
road-and-lane raster. The driver stays P2H10-F until the full run (2 seeds, `P2H10B-F-s{0,1}`, training), G3 at full scale and the staged
closed loop of Amendment 4 item 6 have been read; the 700 x 2 reads and the PAI reads exist for no arm yet.

**Read more.** [plans/2026-10-10-body1-prereg.md](plans/2026-10-10-body1-prereg.md); taxonomy and row set in [results/taxonomy.md](results/taxonomy.md); concept in
[research/next-round/self-model.md](../../research/next-round/self-model.md).

<!-- files:begin -->
<!-- files:end -->
