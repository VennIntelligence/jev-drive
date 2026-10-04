# op_closed_loop: openpilot in B2D: arbitration, op-drive

status: concluded
decisions: 57, 74, 118, 120, 121, 122
index: Native never starts (6/6); arbitration +9.7 DS is slowness

**Question.** Can openpilot drive B2D routes, and how to arbitrate its plan with a route base?

**Conclusion.** Native never starts (6/6); arbitration DS 66.2 vs 56.6 (+9.7 [-5.5, +27.0]) but a slow base gets 67.7 (decisions 57). op-drive fails S2 (-6.4); red-light stop 62.6 vs 63.2 (decisions 74).

**Junction turns, action head alone (camera rig).** [results/junction_rig122.md](results/junction_rig122.md): at the open-loop-aligned camera (1.59 m, 1.86 m) zones off takes 0/13 choice, 1/12 forced turns (1.433 m: 2/13, 4/12); 1.22 m is the only height where the head reaches the exit (5/10, 3/9) but over-steers. Earlier: [results/junction_closed_loop.md](results/junction_closed_loop.md), [results/junction_forced_and_arrow.md](results/junction_forced_and_arrow.md).

**Read more.** research/openpilot-closedloop-integration.md, research/openpilot-seed0-video-diagnosis.md

<!-- files:begin -->
## Files

- `op_arb_agent.py` (lib): Bench2Drive agent for the openpilot …
- `op_arb_server.py` (archive): openpilot policy server for the …
- `op_arb.sh` (archive): openpilot closed-loop integration study
- `op_arb_report.py` (lib): openpilot closed-loop diagnosis and the …
- `op_arb_figs.py` (archive): Figures Reads a copy (plans.jsonl and
- `op_drive_dev.sh` (archive): tag f on the 10 dev routes, one seed on …
- `op_drive_tune.sh` (archive): op-drive resume-policy tuning
- `op_drive_ds_loss.py` (archive): how many DS points do red lights and …
- `op_drive_collisions.py` (archive): classify every collision and every …
- `check_op_calibration.py` (archive): CPU-only check that the CARLA openpilot …

[archive/](archive/) 15 one-off code · [results/](results/) 40 result files · [figs/](figs/) 4 figures · [lib/](lib/) 2 library
<!-- files:end -->
