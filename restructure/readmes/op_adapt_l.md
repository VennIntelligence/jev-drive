# op_adapt_l: Cinque adapted on real-log human futures

status: live
decisions: 77, 78, 79, 80, 81
index: Stop capture 0.252 to 0.559 open loop; B2D no gain
key: jevdrive/op_adapt_l.py, jevdrive/op_adapt_l_arms.py, scripts/op_adapt_l_chain.py, scripts/op_adapt_l_train.py, scripts/op_adapt_l_readout.py, scripts/op_adapt_l_report.py, scripts/op_l_b2d_chain.py, jevdrive/op_l_b2d_report.py, scripts/op_adapt_l_gate_curve_chain.py, scripts/make_op_adapt_l_figs.py

**Question.** Can adapted Cinque learn start/stop/turn from real-log futures?

**Conclusion.** WOD val: start 0.531 -> 0.632, stop 0.252 -> 0.559, turn 0.726 -> 0.843 (decisions 78-80). B2D closed loop: paired diff -4.1 / -7.8 / +9.8, no gain (decisions 81).

**Next.** HUGSIM closed loop; CARLA-frame capture (decisions 81).

**Read more.** research/openpilot-closedloop-integration.md, plans/2026-10-01-op-adapt-L-prereg.md

<!-- files:begin -->
<!-- files:end -->
