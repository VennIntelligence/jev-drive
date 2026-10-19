# best_of_n: multi-policy best-of-N self-imitation judged by the scorer

status: live
decisions: (pending)
index: 4 own models roll 1 200 practice starts; best rollout's own plan as target; JTN-s1 2k-step pilot

**Question.** If four of our own checkpoints (JTN-L18-s1, NTV0-L18-s0, RSG-L18-s0, NR1FB10-L18-s0) each drive the same 1 200 practice-start windows natively and the best-scoring rollout's own states with its own served plan become imitation rows (+ S hinges), does a 2 000-step JTN-L18-s1 continuation with these rows beat the same continuation without them on the arena held split native read by >= +0.01 (CI lower bound > 0), with held-460 zeros <= control + 2?

**Conclusion.** (pending)

**Next.** Smoke (10 windows x 4 models) -> staged launch -> gate (plans/2026-10-19-bn-prereg.md sections 7-8).

**Read more.** [plans/2026-10-19-bn-prereg.md](plans/2026-10-19-bn-prereg.md) (prereg, Chinese working note).

<!-- files:begin -->
## Files


[plans/](plans/) 1 live plans · [scripts/](scripts/) 2 entry points
<!-- files:end -->
