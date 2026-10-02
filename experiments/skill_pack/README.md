# skill_pack: NAVSIM skill pack N0-N4

status: concluded
decisions: 64, 68, 69, 70, 71, 72, 73, 75, 88, 94
index: Navtest PDMS 84.2 to 91.59 (N3), flat at N4

**Question.** Can a scorer head over native plan plus extra slots raise NAVSIM PDMS, weights unchanged?

**Conclusion.** PDMS 84.2 -> N0 84.94 -> N1 87.32 -> N2 90.60 -> N3 91.59 -> N4 91.43 (-0.17 [-0.42, +0.09]); metric alignment, navhard only N4 above native (decisions 64, 68-72, 75).

**navhard off-road diagnosis (2026-10-03).** [results/navhard-offroad/report.md](results/navhard-offroad/report.md), decision 88.

**History alignment as an input rule (2026-10-03 night).** Removing the history rotation everywhere is rejected (navhard 32.67 / 31.99 vs 33.33, navtest -8 / -10 PDMS: stage 2 up, real scenes down); a navtrain-calibrated confidence selector between the shipped and the aligned rollout (post hoc) gives navhard 34.31 (+0.98 [+0.03, +1.97]) at navtest -0.09 [-0.18, -0.01]. [results/history-align/report.md](results/history-align/report.md), plan [plans/2026-10-04-history-align-plan.md](plans/2026-10-04-history-align-plan.md).

**Read more.** research/leaderboard-skill-pack.md, research/navhard-deficit-breakdown.md

<!-- files:begin -->
## Files

- `skill_pack_n0.py` (archive): a switch between openpilot
- `skill_pack_n1.py` (archive): retrain E6's Hydra-style
- `navsim_heads.py` (jevdrive): fit on navtrain, predict navtest and
- `navsim_raise.py` (archive): NAVSIM score raising on top of skill …
- `navsim_qwen.py` (jevdrive): Qwen `L18_*` features on NAVSIM tokens
- `skill_pack_n0.sh` (archive): prep -> score T -> select -> score …
- `skill_pack_n1.sh` (archive): GIMM frames -> Cinque `temporal` …
- `navsim_raise_n3.sh` (archive): N2's configuration on 19 968 + 40 000 …
- `navsim_raise_n4.sh` (archive): N3's configuration on every navtrain row
- `skill_pack_nav_decomp.py` (archive): NAVSIM navtest per-token PDMS loss …

[archive/](archive/) 19 one-off code · [results/](results/) 80 result files · [plans/](plans/) 2 live plans · [scripts/](scripts/) 23 entry points
<!-- files:end -->
