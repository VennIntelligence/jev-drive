# b2d_controller: Fixed-trajectory controller for Bench2Drive

status: concluded
decisions: 26, 27, 29, 30
index: No controller qualified; PI DS 59.1 vs 53.8, lateral +10.8%
key: scripts/b2d_controller_campaign.py, scripts/b2d_controller_archive.py, scripts/b2d_controller_compare.py, scripts/b2d_controller_validate.py, scripts/b2d_controller_g4.py, scripts/b2d_calibrate.py, scripts/b2d_controller_pi_selftest.py, scripts/b2d_controller_turn_selftest.py, scripts/test_b2d_controller.py, scripts/b2d_controller_plot.py

**Question.** Can a fixed-trajectory controller beat the CARLA/TCP default in B2D closed loop?

**Conclusion.** None qualified (decisions 26); PI DS 59.147 vs 53.811, completion 16/20 vs 17/20 (27). Turn CTE .392 -> .089 m but guards failed (29, 30).

**Read more.** research/trajectory-to-control.md, docs/b2d-controller.md, docs/b2d-controller-lateral.md

<!-- files:begin -->
<!-- files:end -->
