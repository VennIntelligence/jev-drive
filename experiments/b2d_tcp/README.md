# b2d_tcp: Real TCP inference with native lateral control

status: concluded
decisions: 28, 30
index: PI cuts jerk p95 84.5 to 45.1 m/s^3 on 3 routes

**Question.** With real TCP, does PI longitudinal control or shorter lookahead help?

**Conclusion.** Both arms 2/3 complete; jerk p95 84.47 -> 45.05 m/s^3 (decisions 28). Short lookahead failed; native lateral chased target 96% of ticks, longitudinal claims only (30).

**Read more.** docs/b2d-tcp-controller.md, experiments/b2d_tcp/results/tcp-controller/final-report.md

<!-- files:begin -->
## Files

- `b2d_tcp_campaign.py` (archive): Six paired actual-TCP routes, one owned …
- `b2d_tcp_control.py` (archive): Pure numeric native-TCP longitudinal …
- `b2d_tcp_eval_agent.py` (archive): shared 4×0.5 s model plan, native plus …
- `b2d_tcp_comparison_agent.py` (archive): Actual TCP inference with native …
- `b2d_tcp_visual_agent.py` (archive): Official TCP policy with a …
- `b2d_tcp_preprocess.py` (lib): Parallelize independent official TCP …
- `test_b2d_tcp_control.py` (archive): Numerical contract tests for the …

[archive/](archive/) 7 one-off code · [results/](results/) 282 result files · [lib/](lib/) 1 library
<!-- files:end -->
