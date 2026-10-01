# reactivity: P5 CARLA pair exams and M-C reaction head

status: concluded
decisions: 32, 42
index: Dual-stream paired-diff head flips pedestrians 43.3%; ridge_late 0%
key: jevdrive/p5_pairs.py, jevdrive/p5_qwen.py, jevdrive/reactivity_mc.py, jevdrive/p5v1.py, jevdrive/reactivity_figs.py, jevdrive/op_route.py, scripts/reactivity.sh, scripts/reactivity_v1_arm.sh, scripts/p5v1_gen.sh, scripts/p5_gen.sh

**Question.** Can a frozen-feature thin head react to a hazard on CARLA counterfactual pairs?

**Conclusion.** P5 v0 exam is valid, but ridge_late flips 0%, TFv6 39.4% (decisions 32). On P5 v1 the dual-stream paired-difference head flips pedestrians 43.3% [35.0, 50.7] vs 0-3% for reweighting (42).

**Read more.** research/results/reactivity/, research/prediag-2026-09/README.md, `git show bcbdde4:todos/2026-09-25-reactivity-program.md`

<!-- files:begin -->
<!-- files:end -->
