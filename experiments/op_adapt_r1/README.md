# op_adapt_r1: PyTorch Cinque port, stage-4 unfreeze

status: concluded
decisions: 55
index: Exact fp32 port; pedestrian AUC gain +0.076 nuScenes, +0.009 CARLA

**Question.** Can openpilot vision be changed cheaply, and does real pedestrian supervision help on CARLA?

**Conclusion.** Exact fp32 port (error 2e-4 m); drift 0.058 / 0.061 m. Pedestrian AUC gain +0.076 nuScenes (line +0.10), +0.009 CARLA P5 (line 0.60): both fail (decisions 55).

**Read more.** experiments/feature_adapter/README.md, `git show bcbdde4:todos/2026-09-28-op-adapt.md`

<!-- files:begin -->
## Files

- `op_torch.py` (jevdrive): openpilot driving models as trainable …
- `op_adapt_train.py` (archive): the B trial
- `op_adapt_eval.py` (archive): op-adapt step 3 readouts, feature pass
- `op_adapt_equiv.py` (archive): numerical equivalence of the PyTorch …
- `op_adapt_ref.py` (archive): onnxruntime reference outputs for the …
- `op_adapt_bench.py` (archive): training throughput of the Cinque port …
- `op_adapt_cache.py` (lib): per-stream caches of Cinque's frozen …
- `op_adapt_readout.py` (lib): op-adapt step 3 readouts on the feature …
- `op_adapt_next_cache.sh` (archive): op-adapt, next B round, real-data half

[archive/](archive/) 6 one-off code · [results/](results/) 21 result files · [lib/](lib/) 2 library
<!-- files:end -->
