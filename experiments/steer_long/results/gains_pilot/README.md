# Gains pilot (steer-long section 5 (1)): `rel_front_steering_angle_weight` 20 on PAI

Written before any pilot rollout. Run on the Tokyo box 2026-10-13 night (route-night section 9 step 1).

Arm: `YS-L18-s0` and `YS-L18-s1`, serving `JEV_VCONT=1.0 JEV_LEAD=1`, wizard override `controller.gains.rel_front_steering_angle_weight=20`
(`WIZ` in pai_eval.sh); everything else as the stored pai281 runs. Baseline: the stored pai281 `ys0` / `ys1` rollouts of the same scenes (default 5;
a rerun of an unchanged configuration is identical per scene, docs/tokyo-box.md).

Scene groups (`scripts/sl_pilot_lists.py` from the stored pai281 rows, `tmp/steer_long/rows.parquet`). B registered 34 / 34 / 34; the stored
data give fewer, so all qualifying scenes are taken:
- `pilot_g1.txt` 14 scenes: a handoff spin (|vx wz| > 6 m/s^2 within 1.5 s after t = 1.7 s) in any of ybase / ys0 / ys1. The "34" in section 5 is
  the 34 spin rollouts of the three labels, which are 14 scenes.
- `pilot_g2.txt` 32 scenes: handoff speed >= 20 m/s, no spin in any label (all such scenes).
- `pilot_g3.txt` 34 scenes: low-speed turn = heading change >= 45 deg within 4 s with v < 5 m/s throughout (the registered low segment), after the
  handoff, no spin, handoff < 20 m/s; 13 clean (score > 0 in ys0 and ys1, all of them) + 21 with a zero in ys0 or ys1.

Baseline (ys0 + ys1): g1 spins 22, zeros 21; g2 zeros 8; g3 zeros 40.
Early-stop lines (section 5 (1), made numeric): PASS iff spins in g1 (both seeds) <= 7 (= 22 / 3) AND zeros in g2 + g3 (both seeds) <= 50 (= 48 + 2).
Spin and zero are measured exactly as for the baseline (controller CSV vx, wz; scenario score == 0).
