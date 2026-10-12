# junction_eyes: teach Cinque's own eyes to see junctions

status: live
decisions: 283
index: frozen eye misses junction corridor, R-A 0.99 m vs 0.5; step 1 runs

**Question.** Does the frozen Cinque vision trunk already see the near corridor (own lane ahead, road edge, oncoming
lane) at junctions on a fair ruler, and if not, can LoRA on the trunk (map labels as targets, no plan loss) teach it
without forgetting the native camera?

**Conclusion.** Step 0: on the fair route-conditioned ruler the frozen eye does not see the near corridor at junctions (R-A at 20 m near
junctions 0.99 m real / 1.52 m reconstructed against a 0.5 m bar; blind 2.13 / 3.78; 62 % of the 20 m label points lie outside the
wide slot), so step 1 runs L-F0, L-F0-nav, U-F0, L-W90 and L-3V against a 0.10 m G1 bar (decision 283).

**Next.** Step 1 (pre-registration B): L-F0, L-F0-nav, U-F0, L-W90, L-3V, 2 seeds each -> read protocol with a
pipeline-matched Z' -> G1 / G2 / G3 -> step 2 policy only if an arm passes.

**Read more.** [research/next-round/junction-eyes-plan.md](../../research/next-round/junction-eyes-plan.md) (the
execution plan, Chinese), [research/next-round/junction-eyes.md](../../research/next-round/junction-eyes.md)
(evidence background), `plans/` (pre-registrations A / B, amendments), [results/step0.md](results/step0.md) (step-0 read), [results/rows.md](results/rows.md) (row tables).

<!-- files:begin -->
## Files

- `junction-eyes-plan.md` (research/next-round): 路口眼：执行稿（交协调者与三个执行者）
- `junction-eyes.md` (research/next-round): 路口眼：在 nuPlan 学校里用地图教眼睛（方案稿，供审核）
- `step0.md` (results): the frozen eye on the new rulers
- `rp_real.md` (experiments/road_probe/results): is the junction blur in the frozen …
- `head1_train.py` (experiments/corridor/scripts): a head on the FROZEN Cinque vision …
- `op_torch.py` (jevdrive): openpilot driving models as trainable …
- `op_adapt.py` (jevdrive): adapting openpilot's vision layer

[results/](results/) 18 result files · [figs/](figs/) 5 figures · [plans/](plans/) 2 live plans · [lib/](lib/) 3 library · [scripts/](scripts/) 13 entry points
<!-- files:end -->
