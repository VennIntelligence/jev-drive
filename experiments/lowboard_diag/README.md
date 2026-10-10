# lowboard_diag: why the two low closed-loop boards lose points

status: live
decisions: (draft entry handed to main; inputs: 138, 153, 170, 218, 219, 220, 226, 227, 235)
index: PAI: longitudinal 43% of lost score, route 23%; corridor zeros are not missing route info; HUGSIM: three-way tie, route 1%
key: experiments/lowboard_diag/plans/2026-10-10-lowdiag-prereg.md, experiments/lowboard_diag/results/pai.md, experiments/lowboard_diag/results/hugsim.md, experiments/lowboard_diag/results/pai/units.csv, experiments/lowboard_diag/results/hugsim/tables.md, experiments/lowboard_diag/scripts/lbd_pai.py, experiments/lowboard_diag/scripts/lbd_hugsim.py

**Question.** On AlpaSim PAI (base P2H10-F served, 60 scenes) and HUGSIM 64 (SH30), which share of the lost score is longitudinal,
route, clearance or other, was each failure already in the served plan, and was the shipped base model's own output right there?

**Conclusion.** One seed per board, stored rollouts. PAI: longitudinal 42.8% of the lost score (zeros 28.6% + progress loss 14.2%),
route 23.4%, clearance 18.2%, other 15.6%; of the 18 corridor zeros 5 are overruns of the logged stop, 2 spins, 5 drifts, 5 turns or
forks where the correct command does not move the plan, 1 late turn, 0 with missing route information; the shipped weights' plan clears
the event in 6 of 29. HUGSIM: other 31.1%, longitudinal 30.9%, clearance 30.3% (not separable), route 1.0%.

**Next.** Main decides the next core from the draft decision entry; open items are listed at the end of each result doc.

**Read more.** [results/pai.md](results/pai.md), [results/hugsim.md](results/hugsim.md),
[plans/2026-10-10-lowdiag-prereg.md](plans/2026-10-10-lowdiag-prereg.md) (class definitions, priority, "base right", two amendments).

<!-- files:begin -->
<!-- files:end -->
