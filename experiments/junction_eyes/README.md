# junction_eyes: teach Cinque's own eyes to see junctions

status: live
decisions: (pending)
index: step 0 (new rulers R-A / R-B / R-C, frozen-eye Z-0 line) running; step 1 LoRA eye arms gated on it
key: research/next-round/junction-eyes-plan.md, research/next-round/junction-eyes.md, experiments/junction_eyes/plans/, experiments/junction_eyes/results/step0.md, experiments/road_probe/results/rp_real.md, experiments/corridor/scripts/head1_train.py, jevdrive/op_torch.py, jevdrive/op_adapt.py

**Question.** Does the frozen Cinque vision trunk already see the near corridor (own lane ahead, road edge, oncoming
lane) at junctions on a fair ruler, and if not, can LoRA on the trunk (map labels as targets, no plan loss) teach it
without forgetting the native camera?

**Conclusion.** Pending.

**Next.** Step 0 (rulers, frozen probe, Z-0 line, R-C field-of-view check) -> pre-registration B -> step 1 arms
(L-F0, L-F0-nav, U-F0, plus L-W90 / L-3V if R-C triggers them) -> G1 / G2 / G3 -> step 2 policy only if an arm passes.

**Read more.** [research/next-round/junction-eyes-plan.md](../../research/next-round/junction-eyes-plan.md) (the
execution plan, Chinese), [research/next-round/junction-eyes.md](../../research/next-round/junction-eyes.md)
(evidence background), `plans/` (pre-registrations A / B, amendments).

<!-- files:begin -->
<!-- files:end -->
