# op_adapt_r2: Detection tokens and S_jev scorer lane

status: superseded-by op_adapt_l
decisions: 67
index: Stage 1 failed: drift 0.397 m (line 0.10)

**Question.** Can CARLA pedestrian pairs plus rule-scored perturbations adapt Cinque?

**Conclusion.** Stage 1 stopped: drift 0.397 m, slow rate +11.7 pp, no full batch (decisions 67). Superseded by op_adapt_l; method not refuted.

**Read more.** research/openpilot-diagnosis/index.html, `git show bcbdde4:todos/2026-09-29-op-adapt-r2-prereg.md`

<!-- files:begin -->
## Files

- `op_adapt_r2.py` (lib): arms, data, sampler, losses, model
- `op_adapt_r2_train.py` (lib): op-adapt round 2 trainer
- `op_adapt_r2_readout.py` (lib): O and the adapted model are read
- `op_adapt_r2_chain.sh` (archive): op-adapt r2 whole staged chain, one …
- `op_adapt_r2_checks.py` (archive): op-adapt round 2, package C numerical …
- `op_adapt_score.py` (archive): op-adapt round 2, package S
- `op_adapt_det.py` (lib): the detection-token plug-in (fc65452
- `op_adapt_r2_cache.py` (archive): op-adapt round 2, package C GPU passes
- `op_adapt_r2_lane.py` (archive): one self-advancing lane per stage with
- `test_op_adapt_score.py` (archive): Unit checks of the S_jev scorer

[archive/](archive/) 23 one-off code · [results/](results/) 37 result files · [lib/](lib/) 4 library
<!-- files:end -->
