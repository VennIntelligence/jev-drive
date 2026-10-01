# probe_planner_v0: frozen Qwen3-VL probe and planner v0, Waymo stage A

status: concluded
decisions: 1, 2, 3, 3b, 3c, 4, 5, 8, 9, 10, 12, 13, 14
headline: Vision-over-ego-prior at pre-onset falsified (CI [-0.062, +0.030]); late fusion gives ADE -0.029 m; vocabulary K >= 1024

**Question.** With Qwen3-VL features frozen, how much does vision add over an ego-state prior in a thin planner head (nuScenes probe v0 and planner v0, then Waymo stage A dry run and train features), and which head, K, layer, resolution and fusion defaults follow?

**Conclusion.** The framing "vision is an increment over the ego prior, largest at pre-maneuver onset" (decisions 1) was falsified on the Waymo train split: pre-onset vision-minus-ego delta is a true null (CI upper +0.030, lower -0.062) while DiD is +0.111 (decisions 3d, full numbers in prediag). Defaults that survived: metric is RFS with ADE alongside (decisions 2), 800 px input (decisions 4), middle layers with image-token mean pooling (decisions 5), fixed-vocabulary classification with K >= 1024 because K = 64 leaves 16.2% of nuScenes frames outside every trust region (decisions 8, 9), late fusion of ego and vision logits (ADE -0.029 m [-0.040, -0.019], vision on ego worth only 3.8%; decisions 10). Waymo data handling: DINOv3 unusable so DINOv2 + V-JEPA 2 + SigLIP2 (decisions 12), exact-hit history window (decisions 13), frame interval measured at 10.00 Hz (decisions 14).

**Read more.** research/qwen-latent-driving.md, research/frozen-vlm-planner.md, docs/waymo-e2e.md, `git show bcbdde4:todos/2026-09-19-probe-v0.md`, `git show bcbdde4:todos/2026-09-20-planner-v0.md`, `git show bcbdde4:todos/2026-09-20-waymo-stage-a-dryrun.md`, `git show bcbdde4:todos/2026-09-21-waymo-train-features.md`

<!-- files:begin -->
<!-- files:end -->
