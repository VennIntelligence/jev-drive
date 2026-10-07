# op_openloop: openpilot on open-loop boards

status: concluded
decisions: 34, 36, 37, 39, 66, 73
index: NAVSIM score is input protocol: interpolation 52.1 to 84.2

**Question.** How does openpilot score on open-loop boards; how much is the 2 Hz contract?

**Conclusion.** WOD RFS 8.005 vs cv 7.10 (decisions 34). Interpolation lifts navtest PDMS 52.1 -> 84.2, navhard 9.3 -> 33.3 (decisions 36, 37); laneChange +0.72 (66, 73).

**Read more.** research/openpilot-diagnosis/index.html

<!-- files:begin -->
## Files

- `op_interp.py` (lib): openpilot on a 2 Hz benchmark history
- `openloop_standing.py` (jevdrive): openpilot's open-loop standing on …
- `op_interp_full.sh` (archive): Full-benchmark run of the default …
- `op_interp_gimm.sh` (archive): GIMM-VFI on the context-rate grid only
- `op_interp_nav.sh` (archive): NAVSIM stage, trimmed after the WOD …
- `op_interp_by_command.py` (archive): Straight / left / right / start-frame …
- `navhard_deficit.py` (archive): Navhard two-stage EPDMS deficit …
- `make_op_interp_figs.py` (archive): Figures
- `make_openloop_standing_figs.py` (archive): Figures
- `op_lb_select.py` (archive): Pre-registered arm selection for the …

[archive/](archive/) 13 one-off code · [results/](results/) 48 result files · [figs/](figs/) 10 figures · [lib/](lib/) 1 library
<!-- files:end -->
