# op_adapt_l: Cinque adapted on real-log human futures

status: live
decisions: 77, 78, 79, 80, 81
index: Stop capture 0.252 to 0.559 open loop; B2D no gain

**Question.** Can adapted Cinque learn start/stop/turn from real-log futures?

**Conclusion.** WOD val: start 0.531 -> 0.632, stop 0.252 -> 0.559, turn 0.726 -> 0.843 (decisions 78-80). B2D closed loop: paired diff -4.1 / -7.8 / +9.8, no gain (decisions 81).

**Next.** HUGSIM closed loop; CARLA-frame capture (decisions 81).

**Read more.** research/openpilot-diagnosis/index.html, plans/2026-10-01-op-adapt-L-prereg.md

<!-- files:begin -->
## Files

- `op_adapt_l.py` (lib): log-imitation adaptation of openpilot …
- `op_adapt_l_arms.py` (lib): the named run configurations
- `op_adapt_l_chain.py` (scripts): the whole queue as ONE self-advancing …
- `op_adapt_l_train.py` (scripts): op-adapt L trainer
- `op_adapt_l_readout.py` (scripts): Every adapted model is read against the …
- `op_adapt_l_report.py` (scripts): every model's readout -> the registered …
- `op_l_b2d_chain.py` (scripts): one self-advancing chain
- `op_l_b2d_report.py` (lib): per-run readouts, paired judgement and …
- `op_adapt_l_gate_curve_chain.py` (scripts): Staged tmux supervisor for gate and …
- `make_op_adapt_l_figs.py` (scripts): Figures

[results/](results/) 53 result files · [figs/](figs/) 11 figures · [plans/](plans/) 5 live plans · [lib/](lib/) 3 library · [scripts/](scripts/) 21 entry points
<!-- files:end -->
