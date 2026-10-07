# feature_adapter: Where pedestrian information lives in openpilot

status: concluded
decisions: 62, 63
index: CARLA P5 pedestrian AUC 0.51-0.53 vs 0.83 nuScenes

**Question.** Can a feature adapter fix CARLA pedestrians (E0 layer probe, E1 Cosmos)?

**Conclusion.** P5 AUC 0.51-0.53 at every layer vs 0.83: adapter infeasible (decisions 62). Cosmos pairs stage 3 AUC 0.815; size-driven, so narrowed to P5 (decisions 63).

**Read more.** `git show bcbdde4:todos/2026-09-29-e0-layer-probe.md`

<!-- files:begin -->
## Files

- `op_layer_probe.py` (archive): at which openpilot layer does CARLA …
- `op_cosmos_probe.py` (lib): 2 x 2 {CARLA, Cosmos} x {x+, x-} …

[archive/](archive/) 1 one-off code · [results/](results/) 12 result files · [lib/](lib/) 1 library
<!-- files:end -->
