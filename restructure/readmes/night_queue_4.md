# night_queue_4: Ghost test, recipe ladder, mode head

status: concluded
decisions: 35, 44, 58
index: No position memory in TFv6/BridgeDrive/BLUE/SimLingo (ghost 5-13%)
key: scripts/nq4_g_lane.py, jevdrive/nq4_g.py, jevdrive/nq4_k_readouts.py, jevdrive/nq4_k_followup.py, jevdrive/nq4_x.py, jevdrive/nq4_opl_report.py, scripts/nq4_gk.sh, scripts/nq4_k.sh, scripts/nq4_opl.sh, scripts/nq4_x_controller.py

**Question.** Do leaderboard families memorize routes (G); does a recipe ladder help (K)?

**Conclusion.** G: ghost rate 5.0-13.3%, no shift/swap collapse; BridgeDrive 0.2% is replay shift (decisions 58). K: P5 pedestrian flips 0.49-0.99% across the ladder, pack criterion fails (35).

**Read more.** research/leaderboard-vs-ability.md, research/results/nq4/g/final/g.md, `git show bcbdde4:todos/2026-09-26-night-queue-4.md`

<!-- files:begin -->
<!-- files:end -->
