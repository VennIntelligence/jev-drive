# b2d_tcp: Real TCP inference with native lateral control

status: concluded
decisions: 28, 30
headline: TCP 3 routes: PI cuts jerk p95 84.5 to 45.1 m/s^3, both arms 2/3 complete; supports longitudinal claims only

**Question.** With the actual TCP network, does the PI longitudinal controller help versus native control, and does a shorter lateral lookahead (.375 s vs .5 s) improve turns.

**Conclusion.** Real TCP, 3 routes x native/PI: both arms 2/3 complete, whole-route speed RMS 2.692664 to 2.458681 m/s, jerk p95 84.47 to 45.05 m/s^3, but PI collided earlier on 1773; kept as an option, not a safety or leaderboard claim (decisions 28). The short lookahead failed per-turn acceptance (4 of 126 required conditions failed), default unchanged. The "native lateral" arm was later found to chase the target point on 96% of ticks, so the pairing supports longitudinal claims only (decisions 30).

**Read more.** docs/b2d-tcp-controller.md, todos/2026-09-23-tcp-controller/final-report.md, todos/2026-09-23-controller-next/tcp-trajectory-contract.md.

<!-- files:begin -->
<!-- files:end -->
