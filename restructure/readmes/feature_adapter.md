# feature_adapter: Where pedestrian information lives in openpilot

status: concluded
decisions: 62, 63
index: CARLA P5 pedestrian AUC 0.51-0.53 vs 0.83 nuScenes
key: scripts/op_layer_probe.py, scripts/op_cosmos_probe.py

**Question.** Can a feature adapter fix CARLA pedestrians (E0 layer probe, E1 Cosmos)?

**Conclusion.** P5 AUC 0.51-0.53 at every layer vs 0.83: adapter infeasible (decisions 62). Cosmos pairs stage 3 AUC 0.815; size-driven, so narrowed to P5 (decisions 63).

**Read more.** research/feature-adapter-domain-shift.md, `git show bcbdde4:todos/2026-09-29-e0-layer-probe.md`

<!-- files:begin -->
<!-- files:end -->
