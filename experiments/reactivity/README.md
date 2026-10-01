# reactivity: P5 CARLA pair exams and M-C reaction head

status: concluded
decisions: 32, 42
index: Dual-stream paired-diff head flips pedestrians 43.3%; ridge_late 0%

**Question.** Can a frozen-feature thin head react to a hazard on CARLA counterfactual pairs?

**Conclusion.** P5 v0 exam is valid, but ridge_late flips 0%, TFv6 39.4% (decisions 32). On P5 v1 the dual-stream paired-difference head flips pedestrians 43.3% [35.0, 50.7] vs 0-3% for reweighting (42).

**Read more.** experiments/reactivity/results/reactivity/, research/prediag-2026-09/README.md, `git show bcbdde4:todos/2026-09-25-reactivity-program.md`

<!-- files:begin -->
## Files

- `p5_pairs.py` (jevdrive): counterfactual scenario pairs in CARLA …
- `p5_qwen.py` (jevdrive): Qwen `L18_*` features for a P5 frame …
- `reactivity_mc.py` (jevdrive): dual-stream reaction head on the P5 …
- `p5v1.py` (lib): the CARLA counterfactual pair exam with …
- `reactivity_figs.py` (archive): Figures for the reactivity program …
- `op_route.py` (jevdrive): Route into the head / into the backbone
- `reactivity.sh` (archive): on the shared box
- `reactivity_v1_arm.sh` (archive): Arm the P5 v1 re-check chain as gated …
- `p5v1_gen.sh` (archive): Run it as slot p5v1-gen
- `p5_gen.sh` (archive): drive route variants

[archive/](archive/) 14 one-off code · [results/](results/) 66 result files · [figs/](figs/) 12 figures · [lib/](lib/) 1 library
<!-- files:end -->
