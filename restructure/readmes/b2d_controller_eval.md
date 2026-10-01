# b2d_controller_eval: Frozen L1/L2/L3 controller evaluation

status: concluded
decisions: 41
headline: Our PI/pursuit controllers win on L1 (ramp 0.92 vs 2.38) but not closed loop: DS 86.0 vs 94.1-95.0 for TFv6 executors

**Question.** When DS cannot separate execution layers, how do we judge that a controller is better or easier to use: L1 (no model, same feasible trajectory in and scored), L2 (per plan), L3 (DS non-inferiority guard).

**Conclusion.** On L1 (40 held-out routes x 3 perturbation seeds) our C/D (pursuit + PI) beat the TFv6 author executors A/B (primary ramp 0.92/0.90 vs 2.38/2.10), and the plant-inversion candidate P reached 0.51 with extra jerk 1.64. The advantage does not carry to closed loop: TFv6 mean DS A 94.1, B 95.0, C 86.5, D 86.0, P 77.2, P2 87.8, and no controller was judged better under the frozen rules (decisions 41). P5-P7 follow-ups (expert-replay acceptance failed longitudinally; P7 acceleration cap lost completion, 29/32 vs 31/32) are in the same entry. Status in the entry is still pending (TCP only 8 routes).

**Read more.** docs/b2d-controller.md, research/trajectory-to-control.md, todos/2026-09-23-tfv6-controller/controller-eval-rules.md, todos/2026-09-23-tfv6-controller/controller-scorecard.md, todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers-p7.md.

<!-- files:begin -->
<!-- files:end -->
