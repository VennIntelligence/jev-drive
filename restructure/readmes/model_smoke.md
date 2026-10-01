# model_smoke: Alpamayo and openpilot smoke runs, rig robustness

status: concluded
decisions: 33, 36
index: smoke: openpilot 1-3 ms/step; 2 deg yaw gives 4.6x lateral error
key: scripts/openpilot_bench.py, scripts/openpilot_rig_study.py, scripts/openpilot_replay.py, scripts/alpamayo_eval.py, scripts/openpilot_migration_figs.py, scripts/openpilot_desire_probe.py, scripts/fetch_comma1m.py

**Question.** Do Alpamayo 1.5 and openpilot run sanely here, and how much does a changed camera rig hurt?

**Conclusion.** Smoke only: openpilot 0.99-3.33 ms/step; Alpamayo minADE_6 0.74 m; 2 deg yaw lateral x4.6, 2 Hz axis x9 (decisions 36). B2D n=5: Alpamayo DS 60.8, openpilot 2.7 void (decisions 33).

**Read more.** research/openpilot-and-open-driving-models.md, docs/zeroshot-adapters.md

<!-- files:begin -->
<!-- files:end -->
