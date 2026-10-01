# b2d_controller: Fixed-trajectory controller for Bench2Drive

status: concluded
decisions: 26, 27, 29, 30
index: No controller qualified; PI DS 59.1 vs 53.8, lateral +10.8%

**Question.** Can a fixed-trajectory controller beat the CARLA/TCP default in B2D closed loop?

**Conclusion.** None qualified (decisions 26); PI DS 59.147 vs 53.811, completion 16/20 vs 17/20 (27). Turn CTE .392 -> .089 m but guards failed (29, 30).

**Read more.** research/trajectory-to-control.md, docs/b2d-controller.md, docs/b2d-controller-lateral.md

<!-- files:begin -->
## Files

- `b2d_controller_campaign.py` (lib): Run controller comparison groups …
- `b2d_controller_archive.py` (lib): Preserve controller experiment inputs …
- `b2d_controller_compare.py` (archive): Compare controller campaign groups …
- `b2d_controller_validate.py` (archive): Sensor-driven map-route validation …
- `b2d_controller_g4.py` (archive): Audit frozen three-preset G4 groups
- `b2d_calibrate.py` (archive): Record stock MKZ physics and bounded …
- `b2d_controller_pi_selftest.py` (archive): Fixed PI candidate synthetic G1 evidence
- `b2d_controller_turn_selftest.py` (archive): Supplementary synthetic G1 for the …
- `test_b2d_controller.py` (archive): Deterministic contract tests
- `b2d_controller_plot.py` (archive): Reproducible publication figures for …

[archive/](archive/) 36 one-off code · [results/](results/) 709 result files · [figs/](figs/) 2 figures · [lib/](lib/) 2 library
<!-- files:end -->
