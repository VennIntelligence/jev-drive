# lowboard_diag: why the two low closed-loop boards lose points

status: live
decisions: 237, 239, 241 (inputs: 138, 153, 170, 218, 219, 220, 226, 227, 235)
index: PAI: longitudinal 43% of lost score, route 23%; corridor zeros are not missing route info; HUGSIM: three-way tie, route 1%
key: experiments/lowboard_diag/results/seeds.md, experiments/lowboard_diag/plans/2026-10-10-lowdiag-prereg.md, experiments/lowboard_diag/results/pai.md, experiments/lowboard_diag/results/hugsim.md, experiments/lowboard_diag/results/pai/units.csv, experiments/lowboard_diag/results/hugsim/tables.md, experiments/lowboard_diag/scripts/lbd_pai.py, experiments/lowboard_diag/scripts/lbd_hugsim.py

**Question.** On AlpaSim PAI (base P2H10-F served, 60 scenes) and HUGSIM 64 (SH30), which share of the lost score is longitudinal,
route, clearance or other, was each failure already in the served plan, and was the shipped base model's own output right there?

**Conclusion.** One seed per board, stored rollouts. PAI: longitudinal 42.8% of the lost score (zeros 28.6% + progress loss 14.2%),
route 23.4%, clearance 18.2%, other 15.6%; of the 18 corridor zeros 5 are overruns of the logged stop, 2 spins, 5 drifts, 5 turns or
forks where the correct command does not move the plan, 1 late turn, 0 with missing route information; the shipped weights' plan clears
the event in 6 of 29. HUGSIM: other 31.1%, longitudinal 30.9%, clearance 30.3% (not separable), route 1.0%.

**Next.** Main decides the next core from the draft decision entry; open items are listed at the end of each result doc.

**HLEAD (2026-10-10), the first fix tried on a class of this table.** openpilot's own lead path (the AlpaSim switch of decision 226,
now `jevdrive/openpilot/lead_long.py`, HUGSIM agent option `op_lead`) in the HUGSIM loop: none of the 9 L1 units ends on its lead any
more (1 completes, 4 stand behind the parked car until max_steps, 2 end on background later, 2 are struck by another, oncoming actor),
L1 HD +0.155 [+0.036, +0.304]; HUGSIM 64 HD does not move, -0.005 [-0.068, +0.049], because 3 completed scenarios are lost to the slower
launch. One seed. [results/hlead.md](results/hlead.md), [plans/2026-10-10-hlead-prereg.md](plans/2026-10-10-hlead-prereg.md),
`scripts/lbd_hlead.py`, tables in `results/hlead/`.

**LOWDIAG2 (2026-10-10), seed spread of all of the above** (decision 241; prereg amendment 3, scripts and definitions frozen). PAI: four base seeds
(`P2H10-F-s0..s3` served; s2 / s3 re-run, scores differ from decision 235 by 0.017 / 0.010): longitudinal 34-45% of the lost score
stays the largest class, corridor zeros 17-18, overran the logged stop 4-5, straight-road drift 5 in every seed, missing route information 0 in every
seed; seed 0 was an extreme on fast turn entry (4 against 2-4), L1 contact (2 against 0), longitudinal zeros (28.6% against 15-21%, the rest sits in progress
loss) and inert command (5 against 3-6). 25 of 60 scenes are zeros in all four seeds, 24 in none. HUGSIM `SH30-F-s1`: three-way tie and route 1-2% hold.
HLEAD repeat (s1, s0 again): 0 / 10 and 0 / 9 L1 units end on their lead, HD flat (-0.015 [-0.078, +0.037], -0.005 [-0.068, +0.050]), the three
lost scenarios of decision 239 are lost again. [results/seeds.md](results/seeds.md), [results/seeds_pai.md](results/seeds_pai.md).

**Read more.** [results/pai.md](results/pai.md), [results/hugsim.md](results/hugsim.md),
[plans/2026-10-10-lowdiag-prereg.md](plans/2026-10-10-lowdiag-prereg.md) (class definitions, priority, "base right", two amendments).

<!-- files:begin -->
<!-- files:end -->
