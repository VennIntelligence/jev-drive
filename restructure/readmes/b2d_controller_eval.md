# b2d_controller_eval: Frozen L1/L2/L3 controller evaluation

status: concluded
decisions: 41
index: Ours win L1 (ramp 0.92 vs 2.38), not closed loop (DS 86 vs 95)
key: scripts/b2d_controller_eval_campaign.py, scripts/b2d_controller_eval_prepare_v2.py, scripts/b2d_controller_eval_l1_v2.py, scripts/b2d_controller_eval_l1_v3_score.py, scripts/b2d_controller_eval_l23_score.py, scripts/b2d_controller_eval_closed_loop.sh, scripts/b2d_controller_eval_v3_l1.sh, scripts/b2d_controller_eval_refs.py, scripts/b2d_controller_eval_tune_next.py, todos/2026-09-23-tfv6-controller/controller-scorecard.md

**Question.** How to judge a controller when DS cannot: L1, L2, L3 layers?

**Conclusion.** L1 ramp C/D 0.92/0.90 vs TFv6 A/B 2.38/2.10; closed-loop DS A 94.1, B 95.0, C 86.5, D 86.0: none better (decisions 41, pending).

**Read more.** research/trajectory-to-control.md, docs/b2d-controller.md

<!-- files:begin -->
<!-- files:end -->
