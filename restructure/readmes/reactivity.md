# reactivity: P5 CARLA counterfactual pair exams and the M-C reaction head

status: concluded
decisions: 32, 42
headline: Dual-stream paired-difference head flips pedestrians 43.3% [35.0, 50.7] vs 0-3% for reweighting; ridge_late flips 0%

**Question.** Can a frozen-feature thin head react to a hazard, measured on CARLA counterfactual pairs (x+ with the hazard, x- without; expert re-run on both sides)? If not, is the limit the representation or the readout?

**Conclusion.** P5 v0 exam is valid (96% of pairs identical in ego until the factor is visible; null p95 0.04 m/s), but our CARLA-trained `ridge_late` head flips 0% and the public TFv6 waypoint channel 39.4% [28.5, 50.1] (decisions 32). The v0 pedestrian zero was a data-size problem: on P5 v1 (101 routes) the dual-stream paired-difference head (Qwen + openpilot, prior = openpilot `ridge_late`) flips pedestrians 43.3% [35.0, 50.7] while hard-example reweighting stays 0-3%; openpilot's vision layer carries no pedestrian signal in CARLA (AUC 0.515-0.519), but decisions 55 later limited that to CARLA frames (decisions 42, corrected in place).

**Read more.** research/prediag-2026-09/README.md, research/openpilot-openloop-integration.md, `git show bcbdde4:todos/2026-09-25-reactivity-program.md`, `git show bcbdde4:todos/2026-09-24-p5-carla-pairs-v0.md`, `git show bcbdde4:todos/2026-09-24-p5-v1-e-layer.md`, `git show bcbdde4:todos/2026-09-23-p5-vlm-metaaction-proto.md`

<!-- files:begin -->
<!-- files:end -->
