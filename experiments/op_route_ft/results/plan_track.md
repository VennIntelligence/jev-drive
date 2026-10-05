# Plan tracking vs action curvature on the B2D 25 junction turns

Question: openpilot is graded on its PLAN on WOD-E2E, but B2D closed loop executes the ACTION head's curvature (`lat_exec: curv`), which turns only about 0.3-0.5x of what a junction needs (d120);
the plan-derived curvature is 1.2-1.35x. Does executing the plan path (`lat_exec: p7`, P7 pursuit tracking the model's plan positions; longitudinal unchanged) take the junction turns, and does the
route-command fine-tune's plan (reads the command open loop, CARLA exit 0.58 vs 0.29, d128) carry the command into closed loop when the plan is what gets executed?

Setup: the guard `b2d_turns` unit set (decision 127): 25 turns (13 choice, 12 forced) on 20 val routes, B2D `spec` preset (aligned camera 1.59 m / 1.86 m), zones off (`"zones": false, "div_m": 1e9`),
seed 2, route turn desire OFF, one run per cell. Arms shipped, rc-ctl-s0, rc-bear-s0, rc-poly-s0 with `LAT_EXEC=p7`; hybrid `hyb` for shipped (action curvature below 3 m/s, plan tracking above, hysteresis 0.5 m/s;
new `lat_exec: hyb` in `lib/op_arb_agent.py`, declared as trick `hybrid_plan_above_3mps`). Reference = the existing curv units with desire off (`results/desire_off.md`; rc-poly-s0 has no desire-off curv run, its
curv row is the guard's desire-ON unit, so its pairing mixes desire; shipped curv lacks route 26365, 24 turns paired). Code: `scripts/plan_track_lane.py`, `plan_track_report.py`, `plan_track_gif_lane.py`;
units `$DATA_DIR/runs/op_route_ft/plan_track/<arm>/b2d/turns-s2-k*`; per-turn json `plan_track.json`. 35 jobs ran on three cards (6 CARLA workers each) in about 85 min.

## Result

Executing the plan path does not take the turns: it takes none (shipped 0/25, rc-ctl 0/25, rc-poly 0/25, rc-bear 1/25, hybrid 0/25; curv 3-7/25), and the cause is not the junction. With plan tracking
the car is lost in the first 20 s of the route: it ends more than 3 m off the route within 20 s in 16/20 (shipped), 13/20 (rc-ctl), 14/20 (rc-bear), 12/20 (rc-poly) runs, against 2-5/20 under curv, and then sits or crawls
(median window speed 0.0 m/s, stopped 61-73% of the window, route RC 11-34 against 52-63, route DS paired p7 - curv -26.0 [-33.3, -18.7] shipped, -29.6 [-42.0, -18.0] rc-ctl, -16.6 [-29.6, -5.3] rc-bear, -27.9 [-39.8, -17.3] rc-poly).
Only 5-15 of the 25 turns are entered at all (curv 21-22). The one rc-bear p7 "took" (5423 turn 0) is a spin that happened to pass within 5 m of the exit (BEV panel, top left), not a turn.

Mechanism (logs, not a counterfactual): on shipped 27994 and hybrid 27994 the car launches to 4 m/s at t = 6-8 s, the plan path it is tracking swerves (the car ends 5-15 m from where the route run starts within about 6 s; yaw of the truth log flips sign), and from then on the base
speed governor reads 1.0 m/s from the rejoin path while the plan stops (plan v(5 s) about 0.15 m/s); the car creeps in small loops (panels: black / red solid curves loop around the route). This is d118's low-speed plan-tracking
instability (HUGSIM spin) appearing on B2D as a start-of-route loss. The hybrid does not cure it: it only uses the action head below 3 m/s and the launch passes 3 m/s, so it hands over to the plan right when the plan is least reliable
(hybrid 10/20 routes off route > 3 m in 20 s, DS 13.1 against 30.4).

Does the command matter under plan execution? rc-bear - rc-ctl took +0.04 [+0.00, +0.13] (all), +0.08 [+0.00, +0.23] (choice), forced 0; entered +0.16 [+0.00, +0.35]. Not measurable: no arm gets to the junction in a state where a
choice could show. Under curv the same contrast is +0.04 [-0.17, +0.26] (d128), i.e. no difference there either. "Turn-in" under p7 in the table is the action head's own logged curvature (the plan is what steers), median -2.5 to -3.5 m = the
car is not at the junction yet, meaningless as a timing readout.

## Reading

1. The plan-versus-action gap (d120: plan-derived curvature 1.2-1.35x vs action 0.3-0.5x) cannot be exploited through this executor: tracking the plan in the B2D harness fails before any junction.
2. The d128 open-loop plan readout of the command (0.58 vs 0.29) is therefore untested in closed loop, not refuted: the plan is not executable at launch speed in this harness.
3. A fair test needs the plan executor only where it is stable (for example plan tracking from a steady 4-6 m/s approach, or only inside the junction zone), with the launch handled by the action head or the route.
   The hybrid at 3 m/s is the wrong switch point.

Caveats: single seed, 20 routes x 1 run, desire off; the start-of-route loss may depend on the harness (tracker P7 gains, the 4 m/s launch, rejoin governor) as much as on the plan; the base-governor link in the mechanism is read from logs of two runs.

## Tables

### Per arm and executor (25 turns: 13 choice, 12 forced; desire off)

took = within 5 m of the dense exit; leaves lane = peak cross-track > 1.75 m, share of entered turns; peak = peak cross-track in the turn window, median over entered turns (m); collisions = window collisions over the 25 turns; turn-in = arc past the turn start at the first plan step with act_k >= 0.5 / R_min towards the commanded side (median over turns that steer; the action head's own curvature, also logged under p7); v med = median speed in the 15 s window after entry, stop = share of time < 0.3 m/s; route DS = mean over the routes finished.

| arm | executor | took choice (13) | took forced (12) | took all | entered | leaves lane | peak m, median | window collisions | turn-in m, median (n) | v med m/s | stop share | route DS (n) | route RC | off route > 3 m in first 20 s |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| shipped | curv | 1 / 13 | 2 / 11 | 3 / 24 | 21 | 17 / 21 | 8.49 | 7 | 3.0 (10) | 0.9 | 0.44 | 30.4 (19) | 51.9 | 2 / 19 |
| shipped | p7 | 0 / 13 | 0 / 12 | 0 / 25 | 5 | 4 / 5 | 7.80 | 2 | -2.5 (5) | 0.0 | 0.61 | 4.1 (20) | 10.8 | 16 / 20 |
| shipped | hyb | 0 / 13 | 0 / 12 | 0 / 25 | 10 | 9 / 10 | 7.30 | 1 | 0.5 (8) | 0.1 | 0.54 | 13.1 (20) | 24.1 | 10 / 20 |
| rc-ctl-s0 | curv | 3 / 13 | 2 / 12 | 5 / 25 | 21 | 18 / 21 | 5.32 | 3 | 2.9 (20) | 1.1 | 0.40 | 38.5 (20) | 62.1 | 4 / 20 |
| rc-ctl-s0 | p7 | 0 / 13 | 0 / 12 | 0 / 25 | 11 | 9 / 11 | 5.41 | 6 | -3.5 (9) | 0.0 | 0.66 | 8.9 (20) | 19.4 | 13 / 20 |
| rc-bear-s0 | curv | 1 / 13 | 5 / 12 | 6 / 25 | 22 | 18 / 22 | 5.71 | 11 | 4.5 (22) | 1.6 | 0.38 | 30.5 (20) | 63.1 | 5 / 20 |
| rc-bear-s0 | p7 | 1 / 13 | 0 / 12 | 1 / 25 | 15 | 14 / 15 | 5.97 | 4 | -2.5 (15) | 0.0 | 0.73 | 13.9 (20) | 34.3 | 14 / 20 |
| rc-poly-s0 | curv (desire ON) | 2 / 13 | 5 / 12 | 7 / 25 | 22 | 18 / 22 | 8.28 | 6 | 4.2 (15) | 2.0 | 0.33 | 39.0 (20) | 61.9 | 4 / 20 |
| rc-poly-s0 | p7 | 0 / 13 | 0 / 12 | 0 / 25 | 12 | 11 / 12 | 6.64 | 4 | -2.5 (10) | 0.0 | 0.67 | 11.1 (20) | 23.3 | 12 / 20 |

### Paired differences (turns paired, route-cluster bootstrap, 95%)

| contrast | took, all | took, choice | took, forced | entered |
|---|---|---|---|---|
| shipped: p7 - curv | -0.12 [-0.26, +0.00] (n 24) | -0.08 [-0.23, +0.00] (n 13) | -0.18 [-0.40, +0.00] (n 11) | -0.67 [-0.87, -0.46] (n 24) |
| rc-ctl-s0: p7 - curv | -0.20 [-0.36, -0.05] (n 25) | -0.23 [-0.46, +0.00] (n 13) | -0.17 [-0.38, +0.00] (n 12) | -0.40 [-0.60, -0.21] (n 25) |
| rc-bear-s0: p7 - curv | -0.20 [-0.41, +0.00] (n 25) | +0.00 [-0.23, +0.23] (n 13) | -0.42 [-0.73, -0.17] (n 12) | -0.28 [-0.48, -0.09] (n 25) |
| rc-poly-s0: p7 - curv | -0.28 [-0.46, -0.10] (n 25) | -0.15 [-0.38, +0.00] (n 13) | -0.42 [-0.73, -0.15] (n 12) | -0.40 [-0.60, -0.21] (n 25) |
| shipped: hyb - curv | -0.12 [-0.26, +0.00] (n 24) | -0.08 [-0.23, +0.00] (n 13) | -0.18 [-0.40, +0.00] (n 11) | -0.46 [-0.70, -0.25] (n 24) |
| shipped: hyb - p7 | +0.00 [+0.00, +0.00] (n 25) | +0.00 [+0.00, +0.00] (n 13) | +0.00 [+0.00, +0.00] (n 12) | +0.20 [+0.00, +0.41] (n 25) |
| p7: rc-bear - rc-ctl (does the command matter?) | +0.04 [+0.00, +0.13] (n 25) | +0.08 [+0.00, +0.23] (n 13) | +0.00 [+0.00, +0.00] (n 12) | +0.16 [+0.00, +0.35] (n 25) |
| curv: rc-bear - rc-ctl | +0.04 [-0.17, +0.26] (n 25) | -0.15 [-0.46, +0.15] (n 13) | +0.25 [+0.00, +0.55] (n 12) | +0.04 [+0.00, +0.12] (n 25) |
| p7: rc-ctl-s0 - shipped | +0.00 [+0.00, +0.00] (n 25) | +0.00 [+0.00, +0.00] (n 13) | +0.00 [+0.00, +0.00] (n 12) | +0.24 [+0.04, +0.46] (n 25) |
| p7: rc-bear-s0 - shipped | +0.04 [+0.00, +0.13] (n 25) | +0.08 [+0.00, +0.23] (n 13) | +0.00 [+0.00, +0.00] (n 12) | +0.40 [+0.21, +0.62] (n 25) |
| p7: rc-poly-s0 - shipped | +0.00 [+0.00, +0.00] (n 25) | +0.00 [+0.00, +0.00] (n 13) | +0.00 [+0.00, +0.00] (n 12) | +0.28 [+0.08, +0.50] (n 25) |

### Route DS paired (p7 - curv, routes with both)

| arm | mean diff DS | n |
|---|--:|--:|
| shipped | -26.0 [-33.3, -18.7] | 19 |
| rc-ctl-s0 | -29.6 [-42.0, -18.0] | 20 |
| rc-bear-s0 | -16.6 [-29.6, -5.3] | 20 |
| rc-poly-s0 | -27.9 [-39.8, -17.3] | 20 |

### Per turn (Y/n took, peak cross-track m, turn-in m)

| route | turn | forced | kind | side | R_min | shipped curv | shipped p7 | shipped hyb | rc-ctl-s0 curv | rc-ctl-s0 p7 | rc-bear-s0 curv | rc-bear-s0 p7 | rc-poly-s0 curv | rc-poly-s0 p7 |
|---|--:|--:|---|---|--:|---|---|---|---|---|---|---|---|---|
| 10255 | 0 | 0 | junction | R | 6.4 | n 25.1 None | not entered | not entered | n 17.6 9.8 | n 0.8 -5.5 | n 25.0 3.8 | n 1.9 -4.0 | n 25.0 None | n 1.1 -3.5 |
| 15102 | 0 | 0 | junction | L | 11.5 | n 25.0 None | not entered | n 6.3 -2.8 | n 25.0 17.2 | n 5.1 None | n 5.7 8.5 | n 6.3 -2.5 | n 25.0 13.8 | n 5.2 None |
| 24416 | 0 | 1 | curve | R | 36.6 | n 1.4 -2.5 | n 6.0 -2.5 | n 1.4 -2.5 | Y 1.4 -2.5 | n 5.8 -2.5 | Y 1.4 -2.5 | n 6.0 -2.5 | Y 1.5 -2.5 | n 7.5 -2.5 |
| 24758 | 0 | 1 | curve | L | 33.3 | n 1.7 None | n 24.9 -3.2 | n 25.1 9.5 | n 1.7 30.2 | n 16.2 -3.5 | Y 1.7 31.2 | n 10.5 -2.2 | Y 1.7 32.0 | n 8.1 -1.8 |
| 24758 | 1 | 0 | curve | R | 31.2 | not entered | not entered | not entered | not entered | not entered | Y 1.4 6.2 | not entered | Y 1.6 5.8 | not entered |
| 24944 | 0 | 1 | curve | L | 10.7 | n 8.5 None | not entered | n 10.7 None | n 4.7 1.5 | not entered | n 6.9 1.2 | n 4.6 -7.5 | n 7.0 2.2 | n 2.9 -2.2 |
| 24944 | 1 | 0 | junction | L | 8.7 | not entered | not entered | not entered | not entered | not entered | not entered | not entered | not entered | not entered |
| 25051 | 0 | 1 | junction | R | 6.2 | n 21.9 None | not entered | not entered | n 4.9 -0.2 | n 14.2 -5.2 | n 4.1 4.8 | n 1.8 -4.2 | n 5.4 5.0 | n 10.9 -2.8 |
| 26153 | 0 | 1 | curve | R | 41.0 | Y 5.7 -6.5 | not entered | not entered | Y 2.8 -3.8 | not entered | Y 2.8 -4.2 | not entered | Y 3.1 0.5 | not entered |
| 26153 | 1 | 1 | curve | L | 15.4 | n 3.7 -2.2 | not entered | not entered | n 19.6 2.5 | not entered | n 2.8 -4.0 | not entered | n 3.1 None | not entered |
| 26365 | 0 | 1 | curve | R | 42.5 | - | not entered | not entered | not entered | not entered | not entered | not entered | not entered | not entered |
| 26723 | 0 | 1 | curve | R | 12.0 | n 12.5 3.8 | n 1.4 0.2 | n 11.9 3.5 | n 5.3 1.2 | n 1.4 0.0 | n 5.8 3.2 | n 5.4 0.0 | n 14.3 4.2 | n 2.1 0.8 |
| 26723 | 1 | 1 | curve | R | 11.3 | not entered | not entered | not entered | not entered | not entered | not entered | not entered | not entered | not entered |
| 26872 | 0 | 0 | junction | L | 12.6 | n 25.0 None | not entered | n 5.9 10.0 | n 25.0 3.2 | n 4.9 -3.5 | n 25.1 10.5 | not entered | n 25.1 None | not entered |
| 27297 | 0 | 0 | junction | R | 6.6 | n 25.0 None | not entered | not entered | Y 1.5 -2.0 | not entered | n 17.4 2.8 | n 7.7 -3.8 | Y 1.8 -0.8 | n 16.4 -11.5 |
| 27994 | 0 | 1 | junction | L | 7.0 | n 4.8 None | not entered | not entered | n 25.0 None | not entered | Y 1.2 5.8 | not entered | n 9.6 None | not entered |
| 28008 | 0 | 1 | curve | L | 33.7 | Y 3.7 -2.5 | n 7.8 -2.5 | n 8.3 -2.5 | n 5.4 -2.5 | n 8.0 -2.5 | Y 4.6 -2.5 | n 8.3 -2.5 | Y 4.5 -2.5 | n 7.8 -2.5 |
| 28008 | 1 | 0 | junction | L | 8.1 | n 25.1 None | not entered | not entered | Y 4.5 5.5 | not entered | n 8.5 4.5 | not entered | n 25.0 None | not entered |
| 28147 | 0 | 0 | junction | L | 15.2 | n 25.0 None | not entered | n 25.0 None | n 25.0 4.5 | not entered | n 14.2 20.5 | n 8.3 1.2 | n 25.0 None | not entered |
| 28180 | 0 | 1 | junction | R | 4.7 | n 25.1 2.5 | not entered | not entered | n 24.6 -1.0 | not entered | n 2.6 2.0 | n 5.0 -2.8 | Y 6.4 11.5 | not entered |
| 334 | 0 | 0 | junction | L | 7.4 | n 2.4 4.2 | not entered | not entered | n 3.0 -2.2 | not entered | n 2.1 1.0 | not entered | n 25.0 10.8 | not entered |
| 34183 | 0 | 0 | junction | L | 13.4 | n 12.6 12.5 | not entered | n 6.3 6.2 | n 4.4 3.8 | n 5.4 -8.5 | n 6.2 4.8 | n 7.0 -2.2 | n 0.2 None | n 10.5 0.8 |
| 5423 | 0 | 0 | junction | R | 6.2 | Y 1.4 3.5 | n 13.1 -8.5 | not entered | Y 2.9 3.2 | not entered | n 15.7 5.8 | Y 2.2 -1.2 | n 11.6 -3.0 | not entered |
| 6999 | 0 | 0 | junction | L | 6.1 | n 7.4 None | not entered | not entered | n 25.0 7.0 | n 2.4 None | n 21.3 5.2 | n 5.1 -6.0 | n 23.3 5.0 | n 2.4 None |
| 9196 | 0 | 0 | junction | L | 8.1 | n 1.7 6.8 | not entered | n 5.8 -4.2 | n 7.8 3.2 | n 6.5 0.8 | n 7.6 4.5 | n 25.0 -0.8 | n 25.1 3.2 | n 5.8 -5.8 |
