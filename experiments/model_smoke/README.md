# model_smoke: Alpamayo and openpilot smoke runs, rig robustness

status: concluded
decisions: 33, 36
index: smoke: openpilot 1-3 ms/step; 2 deg yaw gives 4.6x lateral error

**Question.** Do Alpamayo 1.5 and openpilot run sanely here, and how much does a changed camera rig hurt?

**Conclusion.** Smoke only: openpilot 0.99-3.33 ms/step; Alpamayo minADE_6 0.74 m; 2 deg yaw lateral x4.6, 2 Hz axis x9 (decisions 36). B2D n=5: Alpamayo DS 60.8, openpilot 2.7 void (decisions 33).

**Read more.** research/openpilot-diagnosis/index.html, docs/zeroshot-adapters.md

<!-- files:begin -->
## Files

- `openpilot_bench.py` (archive): Numerics and latency of openpilot …
- `openpilot_rig_study.py` (archive): openpilot camera-rig robustness on real …
- `openpilot_replay.py` (lib): Replay comma1M segments through …
- `alpamayo_eval.py` (archive): fetch clips, sweep, open-loop quality …
- `openpilot_migration_figs.py` (archive): Figures for fc65452
- `openpilot_desire_probe.py` (archive): Does an openpilot model act on turn / …
- `fetch_comma1m.py` (archive): Pick and fetch a few comma1M segments …

[archive/](archive/) 10 one-off code · [results/](results/) 13 result files · [figs/](figs/) 15 figures · [lib/](lib/) 2 library
<!-- files:end -->
