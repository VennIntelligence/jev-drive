# feature_adapter: where pedestrian information lives in openpilot

status: concluded
decisions: 62, 63
headline: CARLA P5 pedestrian AUC is 0.51-0.53 at every layer vs 0.83 on real nuScenes; adapter infeasible there

**Question.** Is there a layer where a feature adapter can fix CARLA pedestrians (E0 layer probe), and does Cosmos re-rendering restore readability (E1)?

**Conclusion.** On the P5 CARLA batch pedestrian AUC is 0.51-0.53 at every layer (stage 1-4, vision, temporal) while real nuScenes stage 3 reaches 0.83; judged G-none, adapter infeasible for that batch (decisions 62). On 75 Cosmos pairs CARLA pedestrians are readable (stage 3 AUC 0.815, Cosmos cell 0.746); readability is set by pedestrian size (below 500 px both cells 0.57), so decisions 62's G-none was narrowed to the P5 batch (decisions 63). E2 (adapter) not opened.

**Read more.** research/feature-adapter-domain-shift.md; pre-registrations: `git show bcbdde4:todos/2026-09-29-e0-layer-probe.md`, `git show bcbdde4:todos/2026-09-29-e1-cosmos-probe.md`

<!-- files:begin -->
<!-- files:end -->
