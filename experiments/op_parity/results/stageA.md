# Stage A: which part of the frame protocol matters to shipped Cinque (no training)

Written 2026-10-06. Design: [../plans/2026-10-06-parity-prereg.md](../plans/2026-10-06-parity-prereg.md), addendum 1 (written before any
pilot score was read). Script: `scripts/pp_stageA.py` (`run` plans, `report` tables). Tables: [stageA_cells.md](stageA_cells.md) (every
cell, every subscore), [stageA_paired.md](stageA_paired.md) (paired vs R8, bootstrap over 78 navtest logs, B 10 000).

Tokens: the 1 499 navtest tokens of decision 116 (`lb_hq_navtestX`) that carry the real nuPlan 10 Hz CAM_F0 frames on the 0.2 s lattice; no
token had a missing frame. Model: shipped Cinque, port fp16, zero state, plan at t0, native NAVSIM rig. Frames before -1.5 s are a zero image
in every cell. Reference R8 = real frames, 0.2 s pairs, 8 slots at 0.2 s (the protocol the model was trained on, with true frames).

## Answer

| cell | queue (valid slots) | pair gap | source | EPDMS | diff vs R8 [95% CI] | EP | DAC | speed ratio (plan / log, 4 s) | t0 token cos dist | plan ADE vs R8 (m) |
|:--|:--|--:|:--|--:|:--|--:|--:|--:|--:|--:|
| R8 | 8 @ 0.2 s | 0.2 | real | 84.09 | ref | 83.5 | 93.9 | 0.892 | 0 | 0 |
| G8 | 8 @ 0.2 s | 0.2 | GIMM | 83.71 | -0.38 [-1.73, +0.96] | 78.9 | 95.5 | 0.815 | 0.037 | 1.13 |
| W8 | 8 @ 0.2 s | 0.2 | warp | 81.27 | -2.82 [-4.25, -1.45] | 82.5 | 93.3 | 0.966 | 0.083 | 1.60 |
| R8g4 | 8 @ 0.2 s | 0.4 | real | 63.55 | -20.54 [-23.79, -17.35] | 94.5 | 83.9 | 1.504 | 0.158 | 7.36 |
| S8 | 8 @ 0.2 s | 0 (static) | real | 63.38 | -20.71 [-23.33, -18.01] | 38.9 | 96.2 | 0.007 | 0.334 | 9.67 |
| R4f | 4 @ 0.2 s | 0.2 | real | 84.19 | +0.10 [-1.13, +1.38] | 80.3 | 95.1 | 0.827 | 0 | 0.80 |
| W4f | 4 @ 0.2 s | 0.2 | warp | 82.49 | -1.60 [-3.38, +0.15] | 79.3 | 95.3 | 0.933 | 0.083 | 1.48 |
| N4 | 4 @ 0.5 s (2 Hz keys) | 0.5 | real | 53.44 | -30.65 [-33.97, -27.34] | 95.0 | 81.0 | 1.725 | 0.237 | 10.04 |
| W4d | 4 @ 0.5 s (2 Hz keys) | 0.2 | warp | 78.05 | -6.04 [-7.91, -4.22] | 75.0 | 92.6 | 0.877 | 0.083 | 1.41 |
| S4d | 4 @ 0.5 s (2 Hz keys) | 0 (static) | real | 63.31 | -20.78 [-23.39, -18.07] | 39.4 | 95.6 | 0.016 | 0.334 | 9.61 |

Contrasts (all paired, same tokens):

1. **Synthesis error is small and is a speed bias.** GIMM - real -0.38 EPDMS (CI includes 0) with plans 6 % slower (speed ratio -0.059),
   EP -4.6, DAC +1.6: decision 116's "GIMM reads ego motion as slower" again. Warp - real -2.82 with plans 5 % faster.
2. **The pair gap is the dominant factor, and it is a calibration shift, not missing information.** With real frames, 0.4 s pairs instead of
   0.2 s make the plan 1.69x as fast (speed ratio 0.892 -> 1.504; EPDMS -20.5, EP +11, DAC -10); native 2 Hz pairs (0.5 s, N4) 1.93x
   (1.725; -30.7). Removing the motion cue altogether (static pair) stops the car (speed ratio 0.007; -20.7). The frozen encoder turns pair
   displacement into ego speed assuming a 0.2 s gap: a longer gap carries the same motion information, read at the wrong scale.
3. **Fewer slots cost nothing.** 4 slots at 0.2 s vs 8 at 0.2 s: +0.10 [-1.13, +1.38] (plans 5 % slower).
4. **The slot spacing costs a few points.** Same source and gap (warp, 0.2 s), 4 slots at 0.5 s vs 4 at 0.2 s: W4d - W4f = -4.4 EPDMS
   (W4d -6.04 vs W4f -1.60 against R8), plans 6 % slower, EP -4.3 against W4f.
5. **Native 2 Hz for the shipped model** (N4) loses 30.7 EPDMS, of which the gap explains most (the speed ratio 1.73), the spacing the rest.

## What this means for Stage B

- The frame protocol enters shipped Cinque mainly as the ego speed read from the image pair. That is exactly the quantity P2 provides
  explicitly, and the plan pathway is trained; so whether N (no synthesis) is viable is a training question (Stage B), and the estimand is the
  protocol x (P2 - P1) interaction: P1 under N has to re-learn the speed scale from the 0.5 s pair, P2 can read it from `vx`.
- Because every protocol shift shows up as a speed shift with EP and DAC moving in opposite directions, Stage B must keep the speed-ratio
  and EP / DAC guards next to EPDMS (as pre-registered); a protocol can look better on EPDMS by being slower (G8 here).
- GIMM is not "better information" than real frames for the shipped model (-0.38, CI includes 0); its effect is a 6 % slowdown.

## Caveats

- One model (shipped), one set of 1 499 tokens (78 logs, includes the 504 decision 116 looked at first). No training here, so no seed noise.
- 0.5 s pairs on the 0.2 s queue cannot be built from the real frames (only the lattice times exist); R8g4 (0.4 s) is the proxy. N4 / W4d differ in
  source (real vs warp) as well as gap; the warp error (W8 - R8, -2.8) bounds that part.
- The oldest pair of every queue has a zero image as its earlier frame (frames before -1.5 s), as in the op_lb protocol.
