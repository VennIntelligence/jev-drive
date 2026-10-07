# b2d_privileged: Privileged-rule ceilings on B2D routes

status: live
decisions: 82
index: Red light + green release DS 75.0 to 95.0

**Question.** How much DS returns if op-drive gets CARLA ground truth on 7 low routes?

**Conclusion.** Diagnostic only: red light + green release DS +20.0 [+15, +30]; all rules together collisions 5 -> 17, DS -1.6 (decisions 82).

**Next.** Validate red-light rule on all dev routes, then skills one scene type at a time.

**Read more.** research/openpilot-diagnosis/index.html, plans/2026-10-01-b2d-privileged-ceiling.md

<!-- files:begin -->
## Files

- `b2d_privileged_chain.py` (scripts): Staged, resumable privileged-ceiling …
- `b2d_privileged_checks.py` (scripts): Independent synthetic geometry checks …
- `b2d_privileged_focus.py` (scripts): Run only previously low-scoring dev …
- `b2d_privileged_geometry.py` (lib): Current-state privileged arbitration …
- `b2d_privileged_report.py` (scripts): Current-state event readouts and …
- `b2d_privileged_plots.py` (scripts): Paper-style figures for the registered …

[results/](results/) 40 result files · [plans/](plans/) 1 live plans · [lib/](lib/) 1 library · [scripts/](scripts/) 5 entry points
<!-- files:end -->
