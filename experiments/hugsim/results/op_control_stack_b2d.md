# openpilot's own lateral path on B2D `drive`: DS -3.66 [-9.02, +0.34], line FAILS (point < -2); the shipped arm already steers from the action curvature

Written 2026-10-04. Diagnostic on main's call after the HUGSIM line failed ([op_control_stack.md](op_control_stack.md), decision 118). Pre-registration (committed before any CARLA run): [../plans/2026-10-04-op-control-stack-b2d-prereg.md](../plans/2026-10-04-op-control-stack-b2d-prereg.md).
Code: `lib/op_ctrl.py` (`OpLateral`, new `sync`), hook in `scripts/b2d_zeroshot_agent.py` (env `OP_CTRL`, inert otherwise; `_curvature_steer`, `_steer_curvature`, tick-log field `opc`),
`experiments/hugsim/scripts/opctrl_b2d_{lane,report,extra}.py`. Tables: `opctrl_b2d/` (`b2d.md`, `runs.csv`, `paired.csv`, `extra.md`). Card 0 under lease `opc-b2d` (released). 19 routes x seeds 2 / 3, 38 runs per arm, no missing runs.
Baseline = vmerge2's `v2-drive-s2/s3` (same code, env inert).

## Premise correction (read from code and the box's cfg, before running)

The shipped `drive` arm does not track plan positions on lane-follow segments. `op_arb.sh` gives `drive` `lat_exec = curv` by default (confirmed in `lsc_b2d/jobs/*/op/cfg/drive.json`):
`_curvature_steer` already converts the Cinque **action-head curvature** (action[0] / max(1, v)^2) to CARLA steer through the bicycle geometry at 20 Hz, limited only by the controller's steer_rate.
Route geometry through P7 pursuit steers in command zones (junctions) and on `div` (openpilot's path and the route disagree). So the HUGSIM contrast "plan tracking vs action" does not exist on B2D;
the `opc` arm adds only what openpilot has on the car and the shipped path lacks: modeld's v <= 0.3 hold and controlsd's latActive, `clip_curvature` (jerk 5 / max(v,1)^2, 3 m/s^2, |kappa| <= 0.2), and a 0.2 s pure lateralDelay (seconds, CARLA runs real time: no 1.25 dilation).

**Geometry (unchanged from shipped, documented).** realised curvature kappa (right-positive) -> angle = atan(wheelbase * kappa) -> normalised steer = angle / (max_steer_deg * speed scale of the P7 steering curve), then the steer_rate limit (2.0 / s) and max_steer clip.
Wheelbase 2.8605 m, max steer 70 deg from the P7 config. No Ackermann correction (the shipped path has none). When another owner steers (zone / div / warm-up) OpLateral follows the curvature of the current steer (controlsd inactive: desired := measured), so a hand-back starts from the wheel's position.
Longitudinal, the arbitration and the path given to the controller are unchanged (the arm's env is the only difference).

## Result [E]

| arm | runs | DS | RC | red light | stop infr. | collisions | vehicle | layout | outside lanes | blocked |
|---|---|---|---|---|---|---|---|---|---|---|
| drive (shipped) | 38 | 67.82 | 84.71 | 12 | 2 | 10 | 6 | 4 | 3 | 1 |
| opc | 38 | 64.16 | 82.16 | 13 | 2 | 11 | 6 | 5 | 3 | 1 |

Paired opc - drive, mean over 19 routes [95% route-cluster CI]: **DS -3.66 [-9.02, +0.34]**, RC -2.55 [-8.73, +2.63], red light +0.03, collisions +0.03 [0.00, +0.08].
By set: obstacle (4) +0.69 [0.00, +1.70]; light (6) -6.45 [-19.34, 0.00]; stop sign (3) 0; other (6) -5.60 [-14.53, +1.82].
**Line (DS point >= -2 and CI upper > 0): FAIL** (point -3.66; the CI upper bound +0.34 passes, so zero is not excluded). Repeat noise of the shipped arm itself is about 3.4 DS (decision 107), the same size.

Per-route changes (DS, seed 2 / seed 3): 15 of 19 routes identical or within 1 DS in both arms. Changes: 37969 58.2 / 58.8 -> **11.3** / 57.6 (seed 2: layout collision, RC 19%); 15612 70 / 100 -> 70 / **22.6** (seed 3: red light, blocked); 27870 70 / 100 -> 70 / **70** (seed 3: a red light the shipped run did not have); 9196 22.9 / 40.5 -> 32.3 / 42.0 (gain, seed 2 RC 54 -> 100); 19324 28.8 -> 33.4 (gain).
Reading: 15612 and 27870 swap 100 <-> 70 across seeds already in the shipped arm (the red-light stop-and-release timing; 15612 seed 2 vs 3 is 70 vs 100), so they are the repeat noise, not steering. 37969 seed 2 is the route that lost most in decisions 107 and 113 as well; the collision tick is **zone-owned** (route geometry steers, `lat_why = zone`, command 5; the opc path was not steering), so the path changed the run only by shifting earlier state.
Neither a larger turning radius nor an understeer signature is visible in the failures.

**Realised low-speed c** (`lat == op`, non-warm, non-zone steps, heading change over 5 ticks per degree of the plan's 1 s direction; run-cluster bootstrap, magnitude; `extra.md`):

| v (m/s) | drive | opc |
|---|---|---|
| 0-1 | 0.014 [0.002, 0.043] | 0.013 [0.013, 0.028] |
| 1-2 | 0.059 [0.017, 0.097] | 0.086 [0.033, 0.201] |
| 2-3 | 0.157 [0.081, 0.230] | 0.063 [0.030, 0.235] |
| 0-3 | 0.071 [0.032, 0.121] | 0.059 [0.027, 0.146] |

c is unchanged within the CIs (as expected: the shipped path was already the action head; the 0.2 s delay shifts, not shrinks, the response). B2D has no launch spins in either arm, so there is nothing to remove; HUGSIM's c 0.013 comes from the same action head, not from the added clip / delay.

## Who steers and what the added path does

| arm | owner | ticks | heading change delivered (sum abs dyaw) |
|---|---|---|---|
| drive | openpilot (lat == op) / zone / div | 41.5% / 21.6% / 37.0% | 5.9% / 82.7% / 11.3% |
| opc | openpilot / zone / div | 34.9% / 27.7% / 37.4% | 6.0% / 82.7% / 11.3% |

- The openpilot path steers about a third of the ticks but delivers 6% of the heading change; 83% of the turning is in command zones, where the route geometry steers in both arms. The added path therefore cannot change junction behaviour directly.
- On openpilot-owned ticks (13 143 traced): 43% are v <= 0.3 (hold / inactive: waiting at lights and stops); `clip_curvature` binds on 0.07% of moving ticks, the 3 m/s^2 cap on 0.000%; the commanded |kappa| is 0.0011 / 0.0052 / 0.0140 (median / p90 / p99), max 0.070 (R 14 m).
  In bends with |kappa_cmd| > 0.01 (233 ticks) realised / commanded slope is 1.145: the delayed command is not attenuated. So the effective change is the 0.2 s delay.

**Turning at junctions (counterfactual, logged `act_k` on zone-owned ticks, v > 1 m/s)**: against the curvature the car actually realised under the route geometry, the action curvature has the same sign 93-94% of the time on turn ticks (R < 33 m, ~490 ticks per arm, mean 3.4-3.5 m/s) but only **0.43 of the needed curvature (regression slope) and 0.55-0.56 summed**. The action head alone would turn about half as much as a junction needs: it would understeer at junctions. That is why `drive` hands junctions to the route geometry, and why replacing the whole lateral with the action curvature (as the HUGSIM arm did, where exam routes have almost no bends) is not tested here: B2D's zone ownership already covers the case the head cannot do.

## Verdict and limits

- Line FAIL on the point estimate (-3.66 vs >= -2), CI [-9.02, +0.34] contains 0. Same shape as decision 113's low-pass on B2D (-4.11 [-10.69, +1.05]) and the same route (37969 seed 2) carrying most of it; but the mechanism here is not damping: the arm changes almost nothing about how the car is steered (c unchanged, clip binds 0.07%).
  Expect that the DS difference is mostly chaotic divergence of the closed loop from a 0.2 s shift, i.e. repeat-noise scale. Not adoptable as a score trick; there is no B2D launch-spin problem for it to fix.
- Limits: single run per route / seed, 2 seeds; seeds differ by up to 21-34 DS on a route (opc vs drive: 21.2 / 33.5 max per-route sd), so the CI is wide and the line (point >= -2) cannot be passed or failed reliably by a ~3 DS effect; the plant is openpilot's pure delay (no EPS dynamics; the steer_rate limit of the shipped controller stays); junction turns are zone-owned, so openpilot's path was not tested at junctions (only counterfactually through `act_k`).
