# world_model: latent world model loop on openpilot + V-JEPA 2 features

status: concluded
decisions: 54, 60, 61, 65, 76
headline: W failed from action-scene confounding, not the latent; WL-2 held-out fails (C3a H = 0.38): no-go

**Question.** Can an action-conditioned world model on frozen latents (openpilot `temporal` + V-JEPA 2 `mean`) roll out hazard consequences and support a critic and action selection for the JEPA + openpilot main line?

**Conclusion.** W failed because action-scene confounding in expert logs, not because the latent lacks hazards; the mainline stays and data is redone as CARLA action forks (decisions 54, corrected in place from "shelve"). WL: brake > hold reads correctly 90.5% (W original 10.3%) and the critic reaches AUC 0.909, but lateral shift consistency 61.7%, offline selection (x+ unsafe down only 30%) and pedestrian flips 0% fail (decisions 61). WL-2 on fresh held-out forks: C1a 88.7%, C2 AUC 0.860, but C3a H = 0.38 [0.12, 0.60] and C1c AUC 0.674 fail, so no-go by the registered reading (decisions 76). Side findings: CARLA night dimming fixed by `B2D_KEEP_STREET_LIGHTS=1` (decisions 60); in-place rewind is not equivalent to from-scratch generation (decisions 65).

**Read more.** research/wl2-results.md, research/carla-rewind-branching.md, research/midterm-gaps.md, `git show bcbdde4:todos/2026-09-28-wm-loop.md`, `git show bcbdde4:todos/2026-09-29-wl2-prereg.md`, `git show bcbdde4:tmp/2026-09-27-wm-loop-diagnosis.md`

<!-- files:begin -->
<!-- files:end -->
