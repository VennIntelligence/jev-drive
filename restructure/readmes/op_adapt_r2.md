# op_adapt_r2: Detection tokens and S_jev scorer lane

status: superseded-by op_adapt_l
decisions: 67
index: Stage 1 failed: drift 0.397 m (line 0.10)
key: jevdrive/op_adapt_r2.py, scripts/op_adapt_r2_train.py, scripts/op_adapt_r2_readout.py, scripts/op_adapt_r2_chain.sh, scripts/op_adapt_r2_checks.py, jevdrive/op_adapt_score.py, jevdrive/op_adapt_det.py, scripts/op_adapt_r2_cache.py, scripts/op_adapt_r2_lane.py, tests/test_op_adapt_score.py

**Question.** Can CARLA pedestrian pairs plus rule-scored perturbations adapt Cinque?

**Conclusion.** Stage 1 stopped: drift 0.397 m, slow rate +11.7 pp, no full batch (decisions 67). Superseded by op_adapt_l; method not refuted.

**Read more.** research/navhard-deficit-breakdown.md, `git show bcbdde4:todos/2026-09-29-op-adapt-r2-prereg.md`

<!-- files:begin -->
<!-- files:end -->
