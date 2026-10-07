# night_queue_4: Ghost test, recipe ladder, mode head

status: concluded
decisions: 35, 44, 58
index: No position memory in TFv6/BridgeDrive/BLUE/SimLingo (ghost 5-13%)

**Question.** Do leaderboard families memorize routes (G); does a recipe ladder help (K)?

**Conclusion.** G: ghost rate 5.0-13.3%, no shift/swap collapse; BridgeDrive 0.2% is replay shift (decisions 58). K: P5 pedestrian flips 0.49-0.99% across the ladder, pack criterion fails (35).

**Read more.** research/benchmarks/index.html, experiments/night_queue_4/results/g/final/g.md, `git show bcbdde4:todos/2026-09-26-night-queue-4.md`

<!-- files:begin -->
## Files

- `nq4_g_lane.py` (archive): the ghost / perturbation matrix for the …
- `nq4_g.py` (archive): Night queue 4, G, plus the closed-loop …
- `nq4_k_readouts.py` (archive): the capability readouts of the recipe …
- `nq4_k_followup.py` (archive): Night queue 4, K follow-up
- `nq4_x.py` (archive): Night queue 4, X and the cross-fitted …
- `nq4_opl_report.py` (archive): tables of the op_native_launch arms
- `nq4_gk.sh` (archive): the G + K + X closed-loop chain
- `nq4_k.sh` (archive): Night queue 4, K-prep chain
- `nq4_opl.sh` (archive): openpilot's native plan in Bench2Drive …
- `nq4_x_controller.py` (archive): X-only independent longitudinal target

[archive/](archive/) 38 one-off code · [results/](results/) 175 result files
<!-- files:end -->
