# best_of_n: multi-policy best-of-N self-imitation judged by the scorer

status: concluded
decisions: 379
index: best-of-4 self-imitation fails: arena native -0.020 vs control, static -0.059

**Question.** If four of our own checkpoints (JTN-L18-s1, NTV0-L18-s0, RSG-L18-s0, NR1FB10-L18-s0) each drive the same 1 200 practice-start windows natively and the best-scoring rollout's own states with its own served plan become imitation rows (+ S hinges), does a 2 000-step JTN-L18-s1 continuation with these rows beat the same continuation without them on the arena held split native read by >= +0.01 (CI lower bound > 0), with held-460 zeros <= control + 2?

**Conclusion.** Gate failed: 248 informative windows of 1 193 (JTN-s1 wins 192, 161 of 164 static; NTV0 wins 25 of 57 launch), 2 443 rows (1 774 drawable); arena native BNX2K - BNC2K -0.0197 [-0.0349, -0.0063] (static -0.059, launch +0.027), held-460 zeros 36 vs 33. The control alone (JTN-s1 + 2 000 steps of its recipe) reads +0.0143 [+0.0002, +0.0345] over JTN-s1 natively (0.905) (decision 379).

**Read more.** [plans/2026-10-19-bn-prereg.md](plans/2026-10-19-bn-prereg.md) (prereg, Chinese working note).

<!-- files:begin -->
## Files


[results/](results/) 7 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
