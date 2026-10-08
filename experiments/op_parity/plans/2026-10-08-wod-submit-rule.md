# wod-submit: how the two WLG seeds become one test trajectory (written before any test-side prediction, bias or statistic exists)

Package: WOD-E2E test submission of WLG (WP2 recipe + adapter gated off below 0.5 m/s; decision 169, val RFS 8.187 = mean of the two seeds' per-frame scores).

Rule, fixed now:
1. Submitted trajectory = the per-waypoint mean of the seed-0 and seed-1 trajectories (`WLG-full-s0`, `WLG-full-s1`), same frame, same 20 x/y waypoints.
   Reason: one trajectory per frame is required; the mean of two draws of one recipe has the lower variance and is not tied to a seed label.
2. Check on the 479 val rater frames (cluster-mean RFS, the convention of wod_launch_report.py): accept the mean trajectory if
   |RFS(mean trajectory) - 8.187| <= max(0.05, |RFS(s0) - RFS(s1)|) (the larger of a fixed floor and the seed spread).
3. If the check fails, submit seed 0 alone (fixed by index, not by val score) and report the failure. No other fallback, no test-side look decides.
4. Serving is identical to val: same ONNX (`pp-WLG-full-s*.onnx`), real frames (`--frames real`), 9 real slots, bias from the adapter on `wod_ego(past, intent)`,
   zeroed where the fed speed (ego[:, 4] * 10) < the checkpoint's `stop_gate` (0.5 m/s). Test `past` and `intent` come from the WOD index rows of the 1 505 frames.
