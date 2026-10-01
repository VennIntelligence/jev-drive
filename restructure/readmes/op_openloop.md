# op_openloop: openpilot on open-loop boards

status: concluded
decisions: 34, 36, 37, 39, 66, 73
index: NAVSIM score is input protocol: interpolation 52.1 to 84.2
key: scripts/op_interp.py, jevdrive/openloop_standing.py, scripts/op_interp_full.sh, scripts/op_interp_gimm.sh, scripts/op_interp_nav.sh, scripts/op_interp_by_command.py, scripts/navhard_deficit.py, scripts/make_op_interp_figs.py, scripts/make_openloop_standing_figs.py, scripts/op_lb_select.py

**Question.** How does openpilot score on open-loop boards; how much is the 2 Hz contract?

**Conclusion.** WOD RFS 8.005 vs cv 7.10 (decisions 34). Interpolation lifts navtest PDMS 52.1 -> 84.2, navhard 9.3 -> 33.3 (decisions 36, 37); laneChange +0.72 (66, 73).

**Read more.** research/openpilot-openloop-standing.md, research/openpilot-openloop-integration.md, research/navhard-deficit-breakdown.md

<!-- files:begin -->
<!-- files:end -->
