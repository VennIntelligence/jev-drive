# probe_planner_v0: Frozen Qwen3-VL probe and planner v0

status: concluded
decisions: 1, 2, 3, 3b, 3c, 4, 5, 8, 9, 10, 12, 13, 14
index: pre-onset vision delta null (CI [-0.062, +0.030]); K >= 1024

**Question.** With Qwen3-VL frozen, how much does vision add over an ego prior in a thin planner head?

**Conclusion.** Vision-over-ego at pre-onset falsified on Waymo train (CI [-0.062, +0.030], decisions 3d). Defaults: 800 px, K >= 1024, late fusion ADE -0.029 m (decisions 4-10).

**Read more.** research/decisions/001.md, docs/waymo-e2e.md

<!-- files:begin -->
## Files

- `probe_v0.py` (archive): index -> labels -> features -> probes …
- `planner_v0.py` (archive): trajectory targets -> vocabulary -> …
- `planner.py` (jevdrive): a fixed trajectory vocabulary scored by …
- `waymo_stage_a.py` (jevdrive): the half-val dress rehearsal
- `labels.py` (jevdrive): future-turn labels and past ego-state …
- `probe_v0.sh` (archive): Rerun probe v0 end to end on the box
- `planner_v0.sh` (archive): Rerun planner v0 end to end on the box
- `waymo_stage_a.sh` (archive): Stage A on Waymo E2E, half-val dress …

[archive/](archive/) 8 one-off code · [figs/](figs/) 3 figures
<!-- files:end -->
