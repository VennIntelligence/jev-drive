# tfv6_rules: TFv6 rules x interface, B2D by hazard family

status: concluded
decisions: 38, 31
index: public B2D noise: single-eval DS SD 0.80; rules x interface not run

**Question.** Do TFv6 rules and control interface change hazard reaction, and how noisy is the public B2D ranking?

**Conclusion.** Only public per-route data (209 routes) has results: DS SD 0.80, top-5 within noise (decisions 38). Rules x interface runs stopped 2026-09-25 with no results.

**Read more.** research/benchmarks/index.html, `git show bcbdde4:todos/2026-09-25-tfv6-rules-interface/README.md`

<!-- files:begin -->
## Files

- `tfv6_rules.py` (lib): TFv6 rules x interface factorial and …
- `tfv6_rules_agent.py` (archive): TFv6 rules x interface agent
- `tfv6_rules_run.sh` (archive): TFv6 rules x interface runs
- `tfv6_rules_batch.sh` (archive): The TFv6 rules x interface batch, in …
- `make_tfv6_rules_figs.py` (archive): Figures for fc65452

[archive/](archive/) 4 one-off code · [results/](results/) 4 result files · [lib/](lib/) 1 library
<!-- files:end -->
