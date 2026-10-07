# zeroshot_openloop: Zero-shot open-loop exams, Alpamayo and openpilot

status: concluded
decisions: 34, 37, 39
index: WOD RFS Cinque 8.005, Alpamayo 8.034, above cv 7.103

**Question.** How do untrained Alpamayo 1.5 and openpilot score on WOD-E2E, NAVSIM, nuScenes, PhysicalAI-AV?

**Conclusion.** WOD RFS: Cinque 8.005, Alpamayo 8.034, Lebowski 7.886, cv 7.103 (decisions 34); NAVSIM PDMS 44-52 vs cv 20.7, navhard EPDMS 9-11 (decisions 37); nuScenes L2 worse than cv (decisions 39).

**Read more.** research/openpilot-diagnosis/index.html, docs/navsim.md

<!-- files:begin -->
## Files

- `wod_zeroshot.py` (archive): WOD-E2E zero-shot exam, project-venv …
- `wod_zeroshot_alpamayo.py` (lib): Alpamayo 1.5 runner
- `navsim_zs_chain.sh` (archive): the remaining Alpamayo phases for one …
- `navsim_zs_alpamayo.py` (lib): Alpamayo 1.5 zero-shot on NAVSIM. Runs
- `navsim_zs_report.py` (archive): collect the devkit score CSVs into …
- `nusc_zs.py` (archive): nuScenes open-loop zero-shot exam …
- `nusc_zs_alpamayo.py` (lib): Alpamayo 1.5 zero-shot on nuScenes …
- `pai_openpilot.py` (archive): openpilot on PhysicalAI-AV, Alpamayo's …
- `navsim_rig.py` (lib): Virtual NAVSIM cameras rendered from …
- `navsim_agent.py` (archive): A NAVSIM agent that replays …

[archive/](archive/) 15 one-off code · [results/](results/) 21 result files · [figs/](figs/) 15 figures · [lib/](lib/) 4 library · [scripts/](scripts/) 1 entry points
<!-- files:end -->
