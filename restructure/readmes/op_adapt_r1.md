# op_adapt_r1: PyTorch port of Cinque, stage-4 unfreeze, pedestrian aux heads

status: concluded
decisions: 55
headline: Exact fp32 openpilot port; pedestrian readability gain only +0.076 AUC on nuScenes, +0.009 on CARLA P5; both fail

**Question.** Can openpilot's vision layers be changed cheaply without breaking it, and does real-data pedestrian supervision (nuScenes GT corridor pedestrians + distillation) make CARLA pedestrians readable?

**Conclusion.** Exact fp32 PyTorch port (plan max error 2e-4 m); stage-4 unfreeze + distillation keeps normal-frame plan drift at 0.058 / 0.061 m median (nuScenes / WOD val) and WOD ADE +0.36%, at 0.07 GPU-h per 100k samples (decisions 55). Pedestrian readability on nuScenes val only +0.076 AUC (line +0.10, fail) and +0.009 on CARLA P5 (line 0.60, fail), same verdicts over 3 seeds (decisions 55). The original model already reads near pedestrians on real data (temporal AUC 0.83 within 10 m), so 'no pedestrian info in vision' was restricted to CARLA (corrected in place, decisions 55; further narrowed by decisions 62, 63).

**Read more.** research/feature-adapter-domain-shift.md, research/openpilot-and-open-driving-models.md; pre-registration: `git show bcbdde4:todos/2026-09-28-op-adapt.md`

<!-- files:begin -->
<!-- files:end -->
