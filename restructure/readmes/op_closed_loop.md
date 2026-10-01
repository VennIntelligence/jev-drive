# op_closed_loop: openpilot in B2D: arbitration, op-drive

status: concluded
decisions: 57, 74
index: Native never starts (6/6); arbitration +9.7 DS is slowness
key: scripts/op_arb_agent.py, scripts/op_arb_server.py, scripts/op_arb.sh, jevdrive/op_arb_report.py, jevdrive/op_arb_figs.py, scripts/op_drive_dev.sh, scripts/op_drive_tune.sh, scripts/op_drive_ds_loss.py, scripts/op_drive_collisions.py, scripts/check_op_calibration.py

**Question.** Can openpilot drive B2D routes, and how to arbitrate its plan with a route base?

**Conclusion.** Native never starts (6/6); arbitration DS 66.2 vs 56.6 (+9.7 [-5.5, +27.0]) but a slow base gets 67.7 (decisions 57). op-drive fails S2 (-6.4); red-light stop 62.6 vs 63.2 (decisions 74).

**Read more.** research/openpilot-closedloop-integration.md, research/openpilot-seed0-video-diagnosis.md

<!-- files:begin -->
<!-- files:end -->
