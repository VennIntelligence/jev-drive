# b2d_privileged: Privileged-rule ceilings on B2D routes

status: live
decisions: 82
index: Red light + green release DS 75.0 to 95.0
key: scripts/b2d_privileged_chain.py, scripts/b2d_privileged_checks.py, scripts/b2d_privileged_focus.py, scripts/b2d_privileged_geometry.py, scripts/b2d_privileged_report.py, scripts/b2d_privileged_plots.py

**Question.** How much DS returns if op-drive gets CARLA ground truth on 7 low routes?

**Conclusion.** Diagnostic only: red light + green release DS +20.0 [+15, +30]; all rules together collisions 5 -> 17, DS -1.6 (decisions 82).

**Next.** Validate red-light rule on all dev routes, then skills one scene type at a time.

**Read more.** research/openpilot-closedloop-integration.md, plans/2026-10-01-b2d-privileged-ceiling.md

<!-- files:begin -->
<!-- files:end -->
