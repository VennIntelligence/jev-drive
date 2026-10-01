# zeroshot_b2d: Zero-shot Bench2Drive, Alpamayo 1.5 and openpilot

status: concluded
decisions: 33
index: n=5 smoke: Alpamayo DS 60.8, SR 2/5; openpilot DS 2.7 voided
key: scripts/zeroshot_b2d_alp.sh, scripts/zeroshot_b2d_op.sh, scripts/b2d_zoo_planner.py, scripts/zeroshot_b2d_alp_speed.py, scripts/zeroshot_b2d_alp_stall.py, scripts/zeroshot_b2d_report.py, scripts/zeroshot_b2d_summary.py, scripts/test_b2d_zoo_pid_wrap.py

**Question.** Can Alpamayo 1.5 and openpilot drive Bench2Drive zero-shot?

**Conclusion.** n=5: Alpamayo DS 60.8, RC 70.1, SR 2/5; openpilot DS 2.7 voided as adapter bug (decisions 33). Alpamayo stalls = Zoo PID reverse-plan bug; F1 fix run paused at 17/220.

**Read more.** research/openpilot-and-open-driving-models.md, `git show bcbdde4:todos/2026-09-24-zeroshot-exam/bench2drive.md`

<!-- files:begin -->
<!-- files:end -->
