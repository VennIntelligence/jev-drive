# hugsim: HUGSIM install, controller, zero-shot exam, I3

status: concluded
decisions: 19, 44
index: fixed2 controller passes acceptance; 4 Hz clock +25-38% lateral

**Question.** Can HUGSIM be the real-appearance closed-loop column, and how do Alpamayo and openpilot score on it?

**Conclusion.** Upstream trackers fail acceptance, `fixed2` passes; openpilot 4 Hz clock +25-38% lateral error; 64 scenarios scored (openpilot Cinque HD-Score 0.033 with the official controller, 0.278 with PR #57; decision 90); no leaderboard submission, the public RealADSim Space is down (decision 19). I3 pairs: no head beats the prior (decisions 44).

**Read more.** [results/history_derotate.md](results/history_derotate.md) (2026-10-03 night: re-rendering the history at the current heading below 3 m/s removes 9 of 10 PR #57 spins, unrotated replay 8 / 10 still spin; the spins are the model extrapolating its own history yaw; HD on non-spin flat, de-spun cars often stop behind the obstacle; selector arm), [results/controller_spin.md](results/controller_spin.md) (why the car spins after the controller fix: official-controller spins are the transposed heading, PR #57 spins are the openpilot plan itself, fixed2 does not help; what leading entries do at the interface), [results/attack_passability.md](results/attack_passability.md) (oncoming attacker scenarios, passability), docs/hugsim.md, `git show bcbdde4:todos/2026-09-25-hugsim-exam/README.md`

<!-- files:begin -->
## Files

- `install.sh` (archive): Build the HUGSIM simulation env on a …
- `zs_exam.sh` (archive): HUGSIM zero-shot exam driver
- `zs_run.py` (archive): Batch runner of HUGSIM's official …
- `zs_agent.py` (archive): HUGSIM agent process for the zero-shot …
- `preset_eval.py` (archive): Score the HUGSIM controller acceptance …
- `pairs_run.sh` (archive): N shard workers of pairs_render.py share
- `hugsim_zs.py` (jevdrive): HUGSIM zero-shot adapters for Alpamayo …
- `hugsim_pairs.py` (jevdrive): real-appearance counterfactual pairs …
- `hugsim_preset.py` (archive): The scene's logged ego trajectory as a …

[archive/](archive/) 31 one-off code · [results/](results/) 30 result files · [figs/](figs/) 15 figures · [plans/](plans/) 1 live plans · [scripts/](scripts/) 20 entry points
<!-- files:end -->
