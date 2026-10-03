# HUGSIM sel3: is the 25-step replay window faithful? (2026-10-04, OD2)

Question: the B2D selector port needed a 132-step replay window and a fix for a spurious desire rising edge at replay start. HUGSIM sel3
(`derot_below` / `derot_sel` in `experiments/hugsim/archive/zs_agent.py`) replays 25 simulator steps (4 model steps each, first frame held
for the 80-step warm-up) and does not handle the desire edge. Does the replay reproduce the native plan when nothing is rotated?

Control (partial: no rerun of the 64 scenes was made): it_dw3-s0, PR #57 controller, 6 scenarios (`scripts/sel3_check6.txt`, the first six of the
spin 10), `derot_sel` 0.6 with `derot_rotate false`, the native plan is always kept (dynamics equal the native run) and
`derot.dpos` logs the max |replay - native| plan position over the horizon. Chain `scripts/sel3_window_chain.sh`, report `scripts/sel3_window_report.py`.

| arm | window | desire at replay start | replays (6 scenes) | dpos median / p95 / max | share > 0.05 m |
|:--|:--|:--|--:|:--|:--|
| A (as in sel3) | 25 steps (buffer 26) | prev desire zero | 260 | 0.0000 / 0.000 / 0.000 m | 0 |
| B (B2D-style) | 33 steps (buffer 34, 136 model steps >= 132) | desire of the step before the window | 261 | 0.0000 / 0.000 / 0.000 m | 0 |

**Result: identical.** Both replays equal the native plan exactly, so the HUGSIM selector needs no window or desire fix, and the HUGSIM sel3 column
of `one_driver.md` is unchanged. Why it differs from B2D: HUGSIM already holds the first buffered frame for the 80-step warm-up and steps 4 model
steps per frame, so a 25-step replay covers 176 model steps, more than the 132-step queues; B2D had one model step per frame (100 frames was < 132).

Limits: 6 scenes only (4 of them have fewer than 51 steps, so the window is rarely truncated; one 220-step Waymo scene supplies 216 of the 260 replays),
unrotated control only (it tests replay fidelity, not the rotated plan's quality), spin scenes only. The 64-scene rerun was not run (stopped by the coordinator).
Code change: `derot_prev_desire` and `derot_ctx` options are available but unused by sel3.
Raw: box `$DATA_DIR/runs/op_adapt_H/hugsim/it_dw3-s0_chk{A,B}/`.
