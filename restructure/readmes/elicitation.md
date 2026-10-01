# elicitation: carrying the CARLA-elicited reaction to real data (E1-E6, I3)

status: concluded
decisions: 42, 44
headline: Zero-shot elicitation is harmful: WOD RFS -1.02 [-1.21, -0.82], NAVSIM PDMS -8.2; log twin and inpainting pairs fail

**Question.** The paired-difference reaction head works in CARLA; does it transfer to real data zero-shot, via log twin pairs, or via real-frame edit pairs?

**Conclusion.** Zero-shot is harmful: WOD RFS -1.02 [-1.21, -0.82] (Cinque) / -1.52 (Lebowski), NAVSIM PDMS -8.2 [-8.9, -7.5]; the Qwen-free 20 Hz student (E5: 52.7% pedestrian flips in CARLA) is harmful by an order of magnitude less (decisions 42, 44). Log twin pairs (E3) and inpainting edit pairs (E2) do not work; later G0-G3 rounds also failed (see real_transfer). Seed check: M-C pedestrian flips 43.3 / 43.6 / 42.4% (Cinque).

**Read more.** research/results/elicitation/, `git show bcbdde4:todos/2026-09-26-elicitation-program.md`, `git show bcbdde4:todos/2026-09-26-overnight-queue.md`

<!-- files:begin -->
<!-- files:end -->
