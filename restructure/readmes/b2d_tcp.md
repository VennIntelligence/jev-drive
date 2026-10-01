# b2d_tcp: Real TCP inference with native lateral control

status: concluded
decisions: 28, 30
index: PI cuts jerk p95 84.5 to 45.1 m/s^3 on 3 routes
key: scripts/b2d_tcp_campaign.py, scripts/b2d_tcp_control.py, scripts/b2d_tcp_eval_agent.py, scripts/b2d_tcp_comparison_agent.py, scripts/b2d_tcp_visual_agent.py, scripts/b2d_tcp_preprocess.py, scripts/test_b2d_tcp_control.py

**Question.** With real TCP, does PI longitudinal control or shorter lookahead help?

**Conclusion.** Both arms 2/3 complete; jerk p95 84.47 -> 45.05 m/s^3 (decisions 28). Short lookahead failed; native lateral chased target 96% of ticks, longitudinal claims only (30).

**Read more.** docs/b2d-tcp-controller.md, todos/2026-09-23-tcp-controller/final-report.md

<!-- files:begin -->
<!-- files:end -->
