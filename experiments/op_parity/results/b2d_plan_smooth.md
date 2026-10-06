# B2D: steering from the smoothed plan curvature (`spec_plan_smooth`) does not make openpilot turn at junctions; the cause moves from "not chosen / late" to "stopped in the junction"

Written 2026-10-07. Question (decision 149 follow-up): on HUGSIM, `spec_plan_smooth` (lateral curvature = `(psi(1.5) - psi(0.5)) / (s(1.5) - s(0.5))` of the model's own
plan, then openpilot's clip + lateral delay) beat the action head on turning routes. Does it make the car take B2D junction turns, where every earlier attempt failed
(decisions 121, 127, 128, 133, 134, 137)? Small read first, as asked: 3 routes per arm, stop if plan_smooth also turns 0 junctions.

## Setup

- Harness = decision 127's `olnz` arm, unchanged except for the lateral source: `op_arb.sh arm spec` (action) vs `arm spec_plan_smooth` (plan_smooth), zones off
  (`"zones": false, "div_m": 1e9`), open-loop-aligned camera (1.59, 0, 1.86), route turn desire on, tm seed 2, clip + 0.2 s delay (B2D `spec`; HUGSIM uses 0.25 s).
  Everything but the curvature source is identical between the two columns.
- Routes: 28008, 5423, 10255 of decision 127's 25 turns = 4 labelled turns: 28008:0 (forced, -53 deg, R 34 m, need 0.030 1/m), 28008:1 (choice, -90 deg, R 8.1 m),
  5423:0 and 10255:0 (choice, +90 deg, R 6.2 / 6.4 m).
- Arms: P2-F-s0 and P2H10-F-s0 (op_parity checkpoints: the arm's `pp-<tag>.onnx` with `intent_bias` plus its bias server `pp_hugsim.py serve`), shipped Cinque.
  Shipped `action` = decision 127's cached `olnz` run (same routes, same config); shipped `plan_smooth` run fresh.
- Code: preset `spec_plan_smooth` and the agent key `arb.curv_src` in `lib/op_arb_agent.py` (the server returns `curvature_smooth`, `scripts/zeroshot_policy_server.py`;
  every plan record now logs `k_sm` next to `act_k`); P2 parity inputs on B2D (`OpArbAgent.parity_ego` -> `op_arb_server._parity_bias` -> bias server; `op_arb.sh` env
  `PARITY_TAG`); runner and report `scripts/b2d_plan_smooth.py`. Tables: [b2d_plan_smooth/b2d_plan_smooth_tables.md](b2d_plan_smooth/b2d_plan_smooth_tables.md)
  (+ per-turn / per-route CSVs).

## Result (4 turns per arm; one seed)

| arm | exec | turned | went straight | off-route | collision | never entered | requested / needed curvature, median | route DS mean (3) |
|---|---|---|---|---|---|---|---|---|
| P2-F-s0 | action | 1 | 0 | 1 | 2 | 0 | 1.61 | 23.4 |
| P2-F-s0 | plan_smooth | 1 | 1 | 1 | 1 | 0 | 1.44 | 33.5 |
| P2H10-F-s0 | action | 1 | 0 | 2 | 1 | 0 | 3.12 | 23.0 |
| P2H10-F-s0 | plan_smooth | 1 | 0 | 2 | 1 | 0 | 1.63 | 31.9 |
| Cinque (shipped) | action (cache) | 1 | 0 | 2 | 1 | 0 | 0.59 | 31.0 |
| Cinque (shipped) | plan_smooth | 1 | 1 | 1 | 0 | 1 | 1.30 | 28.6 |

The one turn taken in every row is 28008:0, the gentle forced curve (the shipped action arm takes it too). **Junction turns (the three choice turns): 0 / 3 in all six rows.**
By the stop rule the 25-turn run was not started.

Why the cause changes (steering split of `rft_split`, signed requested curvature; per-turn table in the tables file):

- Under `action`, 8 of 9 junction-turn cells are "not chosen" or "late" (shipped requests 0.04-0.06 x the needed curvature on the two "not chosen" turns; P2 arms 0.03 on 28008:1, 1.0-3.7 x on the late ones).
- Under `plan_smooth`, 8 of 8 entered junction cells request 0.86-1.96 x the needed curvature with turn-in before the turn start (-0.8 to -14 m): "in time". The car is
  nevertheless stopped for 0.3-0.9 of the turn window (median speed 0-1.4 m/s) and never drives the turn. On 10255 the P2 plan_smooth run reaches the junction at
  6 m/s, slows to 1.5 m/s with `k_sm` 0.19 (needed 0.157), and stands at < 0.1 m/s for the remaining 60 s until "Agent got blocked"; the plan's 3 s speed is 0.7-1.1 m/s
  throughout (the plan itself asks for a near stop, decision 128 item 7). Steering at standstill does nothing.
- So the HUGSIM gain (smooth plan curvature fixes the lateral request) does carry over (the lateral request is right and on time), but it is not the binding constraint on
  B2D junctions: the longitudinal side (plan wants to stop in the junction) is.
- Route DS (3 routes, not a test): P2 +10.1 and P2H10 +8.9 with plan_smooth, shipped -2.4 (vs the cached action run). All are blocked / timed-out failures; the DS rise
  comes from fewer collisions (P2 hits 5 -> 2 summed over the 3 routes), not from progress.

## Caveats and deviations

- n = 4 turns, 3 junction turns, one seed; the closed loop is not bitwise reproducible (the GIF re-run of 10255 reproduced the plan_smooth outcome and DS 26.9 / RC 41.4,
  the action re-run differed: RC 61.7 vs 60.3, DS 24.6 vs 16.0).
- P2 on B2D is my port of the HUGSIM parity inputs, not validated against the NAVSIM training distribution: ego features from the agent's pose history on the real clock
  (4 rear-axle poses at -1.5 / -1 / -0.5 / 0 s, vx = speed, ax = speed change over 0.5 s), command one-hot from the route turn desire (20 m before a LEFT / RIGHT command,
  else straight). The bias is not zero (P2 action requests 1.6-3.1 x the needed curvature, shipped 0.6), but the ego inputs may still be off-distribution.
- The B2D `spec` delay is 0.2 s, not HUGSIM's 0.25 s (kept so the two columns differ only in the curvature source).
- The GPU pool could not place the units for 3 h (a 1000-route `b2dc-all` collection held 6 CARLA servers on each card); no job was preempted.
- Artifacts: Chinese page and GIFs in `tmp/b2d-plan-smooth/` (gitignored): `p2_plan_smooth_10255.gif`, `p2_action_10255.gif` (chase view + model input frames).

## `jevdrive.bench` on real CARLA (first run)

`bench.b2d` ran for real: one route (24211) with the default agent through `b2d.stages` -> pool -> `scripts/b2d_run.py` -> `collect` produced `units.csv` / `summary.json`
(Completed, DS 50.0). Found and fixed / noted:

1. `b2d.submit`, named in the module docstring, did not exist: added (`jevdrive/bench/b2d.py`, test `TestB2D.test_submit_dry`).
2. `bench.run(model, "b2d")` still raises ("unknown bench"; `plan()` does not route to `b2d`): not wired, use `b2d.submit`.
3. `B2DAgent` cannot start the openpilot servers (op server, bias server) that `lib/op_arb_agent.py` needs; those arms go through `op_arb.sh` units (`cllib.b2d_unit`,
   which gained `arm=` / `env=`) as every B2D openpilot result does. The agent runs in `envs/carla` (python 3.8, no `onnx`): `jevdrive.openpilot.model` is not importable
   there, hence the smoothed curvature is computed in the policy server, not in the agent.
