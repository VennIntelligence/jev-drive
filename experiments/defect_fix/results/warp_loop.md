# WARP-LOOP: does the speed ratchet run through the synthesised slots? (2026-10-19)

Diagnosis, no training. Prereg [plans/2026-10-19-warp-loop-prereg.md](../plans/2026-10-19-warp-loop-prereg.md). Code
`scripts/warp_loop.py` (cf, report, s2x, s2report) and `scripts/warp_loop.sh` (cf, sdump, ident, s2). Raw tables:
[warp_loop/report_tables.txt](warp_loop/report_tables.txt), [warp_loop/report.json](warp_loop/report.json),
[warp_loop/s2_report.json](warp_loop/s2_report.json). Box outputs: `$DATA_DIR/runs/defect_fix/warp_loop/`.

**Short answer.** The frames carry the ego's speed far more than the ego row does. Open loop, the plan's 4 s arc follows the speed of the
warp track with gain 0.69 (JTN-L18-s1; RSG 0.57, NTV0 0.66). It follows the history-row speed with gain only 0.07 (RSG 0.20, NTV0 0.07).
Both together give 0.77 (1 = the plan continues the speed exactly). On the human-speed swap, frames alone recover 43 % of the plan's
shortfall, the row alone 33 % (decision 372's 29-44 % number), and both together 88 %. So the "rest" in decision 372 is in the frames.
But the loop is a speed **level** follower, not a trend follower. Holding the warp at the current speed with no history trend changes the
open-loop plan by +0.01 m. In closed loop (`SH30_WARP=const`), the ratchet barely moves: the speed ratio at k = 9 goes 0.912 to 0.922
(rule: >= +0.02). Launches get worse (launch stratum 0.856 to 0.790, zeros 11 to 22), and the mean drops 0.8909 to 0.8818, -0.009
[-0.016, -0.002]. On static windows the standing warp does not remove the creep amplification: the zero windows keep 83-95 % of their arc
growth. There the ego row's speed history is the bigger channel: a standing row keeps about 40 %.

## Stage 1: open loop on the stored native dumps

**Inputs.** Native arena held dumps (`SH30_DUMP`) of 270 moving windows for RSG-L18-s0, JTN-L18-s1 (native_gap) and NTV0-L18-s0 (new, same
command). Static windows: redline_probe's dumps kept only k = 0 / 3, so JTN-L18-s1's native loop was re-dumped on the same 983 windows
(`warp_loop.sh sdump`; 878 static + 105 G / N3). Decisions are exactly 0.5 s apart, and keyframe j of decision k is slot 7 of decision
k - 3 + j (k = 0..2: the cold start, 1..3 keys). The history poses come from the fed row, and each key's body velocity from that key's own
decision.
**Gate.** I rebuilt the 8 slots along the native track with the serve's GPU lattice (`SH30_SYNTH=gpu`). 26-73 % are byte-equal; the rest
differ by a handful of bytes. The plan from the rebuilt frames matches the plan from the dumped frames within p99 0.044 m. 54 of 17 770
decisions exceed the preregistered 0.05 m, by at most 0.0625 m. All 54 have plan arcs >= 23.7 m, so the excess is fp16 rounding of the plan
output. The dumped frames against the dumped plan agree within max 0.125 / p99 0.06 m (native_gap's gate). `R_hum` reproduces native_gap
cf's H within 0.016 m.

**Variants.** Keyframes are fixed. Only the six synthesised slots are re-warped (W), or only the row's speed history is rewritten (R:
vx vy, ax, 3 past poses), or both (B). Tracks: hum (the human's speed at each key's station), const (the current speed, no trend), stand
(all poses at t0), s08 / s12 (the ego's own arcs x 0.8 / 1.2). The gain G = Σ ΔA(s12 - s08) / Σ 4 s x Δū, where ū = the track's 1.5 s
mean speed.

### Gain of the 4 s plan arc against track speed (moving decisions, k >= 2, ego >= 2 m/s; window bootstrap 95 % CI)

| model | stratum | n | G_W (frames) | G_R (row) | G_B (both) | G_W - G_R |
|---|---|--:|--:|--:|--:|--:|
| JTN-L18-s1 | all | 1 942 | **0.691** [0.667, 0.713] | 0.067 [0.060, 0.074] | 0.765 | +0.624 [+0.601, +0.645] |
| JTN-L18-s1 | free | 1 668 | 0.687 | 0.061 | 0.755 | +0.625 |
| JTN-L18-s1 | lead <= 30 m | 206 | 0.705 | 0.070 | 0.785 | +0.634 |
| JTN-L18-s1 | after stop | 68 | 0.832 | 0.346 | 1.178 | +0.486 [+0.349, +0.590] |
| JTN-L18-s1 | win cruise / braking / turn | 292 / 526 / 298 | 0.718 / 0.530 / 0.808 | 0.029 / 0.028 / 0.131 | | |
| RSG-L18-s0 | all | 1 950 | 0.566 [0.547, 0.584] | 0.197 [0.182, 0.213] | 0.776 | +0.368 [+0.347, +0.388] |
| RSG-L18-s0 | after stop | 77 | 0.557 | 0.654 | 1.237 | -0.097 [-0.253, +0.031] |
| NTV0-L18-s0 | all | 1 951 | 0.660 [0.636, 0.681] | 0.065 [0.058, 0.074] | 0.734 | +0.594 [+0.571, +0.615] |
| NTV0-L18-s0 | after stop | 77 | 0.838 | 0.391 | 1.226 | +0.447 |

The JT / NTV stage cut the row channel to a third of RSG's (0.20 to 0.07) and raised the frame channel (0.57 to 0.66-0.69). That is why
history-only fixes (d320, d343, JEV_VHIST) could not move the slowness. The self-gain (G_B) is the same 0.73-0.78 in all three models.

### Human-speed swap (decision 372's counterfactual, split; human >= 2 m/s at the station)

| model | n | ego / human | plan F / human | recovery W | recovery R | recovery B | gain W / R / B (vp - vp_F) / (vh - v) |
|---|--:|--:|--:|--:|--:|--:|--:|
| JTN-L18-s1 | 1 920 | 0.959 | 0.926 | 0.43 | 0.33 | **0.88** | 0.60 / 0.45 / 1.18 |
| RSG-L18-s0 | 1 923 | 0.940 | 0.882 | 0.42 | 0.40 | 0.88 | 0.58 / 0.58 / 1.22 |
| NTV0-L18-s0 | 1 951 | 0.948 | 0.908 | 0.45 | 0.32 | 0.87 | 0.61 / 0.44 / 1.17 |

### Trend against level (median ΔA against F, m; JTN-L18-s1 moving)

| track | W | R | B |
|---|--:|--:|--:|
| const (current speed, no trend) | **+0.01** | +0.68 | +0.57 |
| stand | -6.27 | -4.72 | -11.18 |

Removing the deceleration trend from the frames does nothing (+0.01 m on a 22.7 m median arc; braking windows -0.08). The row's trend does
matter (ax and pose spacing, +0.68 m; braking windows +3.25 m). The frames encode how fast the ego goes, not whether it is slowing.

**Reading (prereg).** Frames are the main channel: G_W >= G_R, and the CI lower bound of G_W - G_R is > 0, in all three models. On JTN1:
"frames carry the larger part". The stage-2 condition (G_W >= G_R and G_W >= 0.10) holds.

### Static windows (JTN-L18-s1, re-dumped, 878 static; zero = JTN1 native zero)

Creep growth = (mean over k = 5..9 of the median arc) - (arc at k = 0), m:

| class | windows (zeros) | F | W_stand | W_stand / F | R_stand | B_stand | W_const |
|---|--:|--:|--:|--:|--:|--:|--:|
| P red line no lead, zeros | 87 | 13.22 | 10.99 | **0.83** | 5.48 | 5.39 | 13.15 |
| U unknown light, zeros | 29 | 11.41 | 9.86 | 0.86 | 4.32 | 4.67 | 11.08 |
| F far lead, zeros | 12 | 12.24 | 11.67 | 0.95 | 4.99 | 4.64 | 11.87 |
| N1 close lead, zeros | 5 | 9.80 | 8.42 | 0.86 | 4.20 | 4.29 | 9.23 |
| P red line, successes | 152 | 1.48 | 1.67 | 1.13 | 1.03 | 1.24 | 1.48 |

P zeros by k (median arc, m), F / W_stand / R_stand / B_stand: k0 3.52 / 3.52 / 3.52 / 3.50; k3 6.13 / 6.25 / 4.96 / 5.28; k5 11.6 / 11.2 /
6.0 / 7.1; k9 21.4 / 17.0 / 12.8 / 10.7.
Zero-window decisions k >= 2 with arc < 2.5 m: 0.9 % F, 0.9 % W_stand, 1.5 % R_stand, 1.1 % B_stand.

**Reading (prereg).** "Not through the warp": W_stand keeps 83-95 % of the growth on every zero class, and k = 0 is unchanged (the human,
and so the ego, stands). The standing row cuts the growth to about 40 %. Even standing frames plus a standing row keep about 40 %, and no
decision drops below the 2.5 m creep threshold. The rest is in the two real keyframes, which see the car creeping, and in the scene. At
creep speeds (< 3 m/s) a 0.2 s warp moves the image by very little, so the warp channel is weak exactly where the static creep lives.

## Stage 2: closed loop, `SH30_WARP=const` (JTN-L18-s1, arena held, native, 18 chunks)

Switch: `sh30_core.warp_track` (env `SH30_WARP`, default off). Off, it makes exactly the same calls as before. Identity: 12 chunk-0 windows
with the switch off reproduce the stored native read's plans 120 / 120, max difference 0.0. With const on, 119 / 120 differ. const = the
history poses re-spaced along the ego's own polyline at the current measured speed, standing below 0.5 m/s. The ego row is untouched.

| | native (stored) | SH30_WARP=const |
|---|--:|--:|
| mean (2 228) | 0.8909 | 0.8818 (-0.0091 [-0.0162, -0.0024]) |
| zeros | 210 | 228 (22 fixed, 40 broken) |
| ego / human speed at the same station, k = 2 / 5 / 9 | 0.987 / 0.943 / 0.912 | 0.992 / 0.954 / 0.922 |
| ratio by decision stratum free / lead / turn / after stop | 0.956 / 0.956 / 0.984 / 1.000 | 0.969 / 0.957 / 0.993 / 0.965 |
| launch stratum (200) mean, zeros | 0.856, 11 | **0.790, 22** |
| static (878) mean, zeros | 0.847, 134 | 0.852, 130 (P 87 to 83, U 29 to 30, F 12 to 10) |
| turn (213) / braking (372) / other (449) | 0.898 / 0.922 / 0.938 | 0.875 / 0.917 / 0.928 |
| static median arc k0 / k9 (m) | 2.14 / 2.49 | 2.07 / 2.49 |

**Reading (prereg).** The ratchet is not weakened: the k = 9 ratio rises only +0.010 (< +0.02), and the drop from k = 2 to k = 9 is -0.075
native against -0.070 here. The creep loop is not broken: 4 fewer static zeros (< 20). The switch costs the after-stop launch: a warp
without the acceleration trend makes the plan launch later, so the launch stratum loses 0.066 and has twice the zeros. This is a
diagnostic, not a submission.

## What it means

- **Decision 372's open question is answered.** The 56-71 % of the plan shortfall that the row does not recover runs through the
  synthesised frames. Frames and row together recover 87-88 % (close to d285's 83 % on PAI).
- **The loop is a level loop with self-gain about 0.75, not a trend loop.** The plan follows the speed its frames show (gain 0.6-0.7)
  plus a small row term. The per-step bias (the plan asks 2-4 % less than the current speed, d372 A2) gets locked in because the next
  frames show the slower speed. A warp held at the current speed keeps the level, so it keeps the loop closed. That is why const changed
  nothing open loop (+0.01 m) and little closed loop. Opening the loop would need a warp speed independent of the ego's own. The only such
  speed at serve time is a prior (route / map speed), and that would also cut the frames' real information. Feeding back the human's
  speed is not available at serve time.
- **The fix is in the bias, not in the carrier.** The carrier (frames that encode their own speed) is what NAVSIM training rewarded,
  because along the human's track it predicts the future. The 2-4 % per-step shortfall is a plan bias. A training-side correction of that
  bias (on-policy rows where the frames come from the policy's own slower track and the label keeps the human's speed) attacks the cause.
  Serve-side warp manipulation does not: it hurts launches (-0.066) and turns (-0.023).
- **Static creep is a different path.** There the row's speed history is the larger channel (it keeps ~40 % of the growth when standing),
  and the real keyframes carry the rest. The warp gives little at creep speeds. The red-line hold remains the training-signal problem of
  d373, not a frame-synthesis problem.

## Limits

- Open loop on stored closed-loop inputs: one decision's counterfactual, not its consequences (stage 2 gives those for const only).
  One seed per model. The gain is a linear slope over 0.8x-1.2x; the stand and hum rows are larger moves. A few "after stop" and "turn"
  human-recovery CIs are wide, because the denominators are near zero there.
- Static analysis is JTN-L18-s1 only (its native loop re-dumped). RSG / NTV0 static creep was not re-dumped.
- Stage 2 tested one decorrelated track (const, standing < 0.5 m/s). The standing-below-0.5 part did not move the static zeros.
