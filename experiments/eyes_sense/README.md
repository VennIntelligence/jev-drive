# eyes_sense: same eyes, better car sense (2 x 2)

status: concluded
decisions: 255 (inputs: 136, 232, 233, 236, 243, 244, 245, 248, 254)
index: side views stay closed (> 45 deg cut-in +1.02 with S); body lesson navhard +2.2

**Question.** In one joint policy training from the shipped weights (lambda-10 recipe), do two more frozen views (EYES: the untrained 3-view token
bank of decision 248's RW0) lower the > 45 deg cut-inside and off-road failure rates without a < 5 deg, NC or EC cost, and how do they compare
with the body lesson (SENSE: `P2H10S`) read in the same tables?

**Conclusion.** Side views do not reopen (decision 255): on top of the body lesson they raise > 45 deg cut-inside by +1.02 [+0.19, +1.91] pp and > 45 deg off-road by +1.32 [+0.32, +2.34] pp with EPDMS < 5 deg and EC down (stays closed); on the plain base the same rates rise (+0.92, +1.19) but the bank is read too lightly for the registered check (`:mshuf` +0.18 < 0.2: not measured). The body lesson moves the boards with or without them: (c) - (a) navtest +0.38 [+0.20, +0.57], > 45 deg cut-inside -0.73 [-1.39, -0.19], navhard +2.22 [+0.25, +4.36]; (d) - (b) navhard +1.98 [+0.60, +3.43]; nuPlan 700 (c) - (a) mean +0.0003, collision + offroad zeros -2.0, corridor +2.0 (CIs include 0). Results [results/contrast.md](results/contrast.md).

**Next.** Nothing registered. Open: the arm (d) reads better with its bank masked (+0.50), a consequence of training the body lesson's hinge-only rows without memory; rendering side views for perturbed states would be the test.

**Read more.** [plans/2026-10-11-eyes-sense-prereg.md](plans/2026-10-11-eyes-sense-prereg.md); decision 248 and experiments/vis_train/results/reads.md.

<!-- files:begin -->
## Files


[results/](results/) 12 result files · [plans/](plans/) 1 live plans · [scripts/](scripts/) 3 entry points
<!-- files:end -->
