# op_adapt_r1: PyTorch Cinque port, stage-4 unfreeze

status: concluded
decisions: 55
index: Exact fp32 port; pedestrian AUC gain +0.076 nuScenes, +0.009 CARLA
key: jevdrive/op_torch.py, scripts/op_adapt_train.py, scripts/op_adapt_eval.py, scripts/op_adapt_equiv.py, scripts/op_adapt_ref.py, scripts/op_adapt_bench.py, scripts/op_adapt_cache.py, scripts/op_adapt_readout.py, scripts/op_adapt_next_cache.sh

**Question.** Can openpilot vision be changed cheaply, and does real pedestrian supervision help on CARLA?

**Conclusion.** Exact fp32 port (error 2e-4 m); drift 0.058 / 0.061 m. Pedestrian AUC gain +0.076 nuScenes (line +0.10), +0.009 CARLA P5 (line 0.60): both fail (decisions 55).

**Read more.** research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-28-op-adapt.md`

<!-- files:begin -->
<!-- files:end -->
