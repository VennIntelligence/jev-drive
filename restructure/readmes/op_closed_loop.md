# op_closed_loop: openpilot in B2D, arbitration and op-drive

status: concluded
decisions: 57, 74
headline: openpilot native plan never starts in B2D (6/6); as arbitration over a route base +9.7 DS [-5.5, +27.0]

**Question.** Can openpilot drive Bench2Drive routes, and how should its plan be arbitrated with a route-following base (start, turn, red light, resume policy)?

**Conclusion.** Native never starts (6/6; static prior, plan covers 0.97 m in 5 s) and turn desire does not change the plan before junctions. On 10 dev routes the no-perception base scores DS 56.6; best arbitration e2e 66.2, +9.7 [-5.5, +27.0] over base, but a same-mean-speed slow base gets 67.7, so the gain is explained by driving slowly; lateral control by openpilot gives -30 to -48 (decisions 57, design shelved by user). op-drive follow-up: resume policy R1 adopted but never triggered in tuning; dev fails S2 (drive minus pacing control -6.4 [-16.9, +0.2]), held-out not run; privileged red-light stop R3a removes red-light runs (6 -> 0) but deadlocks cancel it (DS 62.6 vs 63.2) (decisions 74). Later work on this line is op_adapt_l and b2d_privileged.

**Read more.** research/openpilot-closedloop-integration.md, research/openpilot-seed0-video-diagnosis.md, docs/closed-loop-runbook.md; pre-registrations: `git show bcbdde4:todos/2026-09-28-op-closedloop.md`, `git show bcbdde4:todos/2026-09-29-op-drive.md`

<!-- files:begin -->
<!-- files:end -->
