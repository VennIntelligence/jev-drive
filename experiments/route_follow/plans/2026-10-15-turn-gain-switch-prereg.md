# Pre-registration: TURN-GAIN serve-side switch on PAI (diagnostic / exam technique, NOT adopted), 2026-10-15

**Label.** A diagnostic of the policy, run at serving time. It is not a submission switch and not adopted without the user, whatever it
reads. No row is built from these rollouts.

**Question.** With the policy on its own states (no log handover, no branch search), is turn magnitude the bottleneck? Decisions 296 / 298:
a lateral gain x1.25 on the policy's own plan completes many turns when it starts at a handover (training scenes: any cell 20 / 27), but the
approach start (D40) completes only 4 / 26 exam turns. Here the gain is not started at a chosen handover; it switches itself on whenever the
policy's own 2 s plan under-takes a turn its own route memory shows.

Written and committed before any rollout with the switch on is run. Amendments are appended, never edited in place.

## Switch (`JEV_TURN_GAIN=<g>`, experiments/alpasim/lib/turn_gain.py, pai_driver.py; unset = bit-identical serving)

At every inference, from the route memory the driver fed (rmem; RSG-L18-s0 = the old -6 .. +72 m projection layout) and the model's plan
(after serve_fix, before the controller):
- h_ref = route heading at the +0 m slot when valid, else the ego's heading; dr40 = the signed heading change (vs h_ref) of the valid slot in
  0 .. +40 m with the largest |change|; L2 = the plan's arc from the ego through its 0.5 .. 2.0 s poses; dplan = the plan's yaw at 2.0 s;
  dr_arc = the route's heading change at arc L2 (slot headings unwrapped, linear in the offset).
- On when |dr40| >= 45 deg, dr_arc has dr40's sign and sign(dr_arc) x dplan < 0.6 |dr_arc|. Then the served poses = branch_poses(plan, g):
  every pose yaw and segment course multiplied by g at the same segment lengths (onp_log_driver.branch_poses with s = 1, the curvature
  push of decisions 290 / 296 / 298; turn_gain.selftest checks the copy). Off: the plan unchanged.
- Logged per inference: `tg` {on, dr40, dr_arc, dplan, L2, g}; `poses_branch` (served) while on; `poses` stays the plan.

## Run

- Checkpoint RSG-L18-s0; serving = the reference `/data/runs/route_night/r281/rsg` (pai_eval.sh, wizard
  `controller.gains.rel_front_steering_angle_weight=20`, `JEV_VCONT=1.0 JEV_LEAD=1`, CONC 4, image `jev-alpasim:p2h10-f-s0-6f3d05d1`) plus
  `JEV_TURN_GAIN=1.25`. Tokyo, both cards, 2 chunks (`rf_v2pai.sh tg 1.25`), out `/data/runs/route_night/turn_gain/g1.25`.
- Scenes (47, `results/turn_gain/scenes47.txt`): the 36 strict turns of RSG-L18-s0's PAI 281 read (`results/pai281/strict36.txt` =
  `results/pai_follow/scenes.csv` label `rsg` with a turn = plan_strict36.json + 68e53160) and decision 298's exam turn scenes
  (`experiments/alpasim/results/pai_exam/exam_turn26.txt`), deduplicated (15 overlap).
- Identity = the stored r281/rsg sessions (no re-run; the extract `/data/runs/route_night/pai_follow/r281.pkl`, label `rsg`). Reproduced by
  the read code before this prereg: strict 5 / 36 completed (decision 292), exam turn scenes 2 / 26 (decision 298), 31 zeros on the 47.
- Smoke first (`rf_v2pai.sh tgsmoke 1.25`, 2 scenes, card 0): `tgcheck` — on / off recomputed from the logged rmem and plan (near-threshold
  flips from the 5 / 4-decimal logs listed, not failures), served = branch_poses(plan, 1.25) on every on-decision, and every plan up to a
  session's first on-decision equal to r281/rsg at the same k (PAI reruns are bit-identical, decision 292 / 296). A crash or a pre-switch
  difference stops the run.
- g = 1.5 only if 1.25 completes >= 3 more strict turns than the identity and the stop rule does not fire.

## Measures (rf_v2pai.py turns / table, per scene, paired with the identity)

- Completion = rf_pai_follow's rule on each label's own AlpaSim route (strict turn >= 60 deg within 30 m; completed = the ego crosses
  exit + 10 m before the first zero event and within 2.5 m of the route); outcome otherwise event / off / short_rear / short_slow / no turn.
- Where it failed: zero kind, the ego's route station at the zero event relative to the entry.
- Shape gate (decision 293, rf_p4.gate_of): completed and peak kappa, dkappa/ds within the human class p95, a_lat <= 4.42 m/s^2, speed at
  the entry <= class v_e + 1 m/s; track on [entry - 40, exit] before the zero event (rf_shape operators).
- Switch decisions per scene (count, first on-decision's route station vs the entry).
- Zeros and mean score over the 47 (paired, bootstrap over scenes); new zeros on the non-turn part = g zero, identity not, event before
  the entry or the scene has no strict turn; offroad among them.

## Lines

- **Turn magnitude is a bottleneck (supported):** strict-36 completion >= 5 + 6 = 11 / 36 with no new zero on the non-turn part of the 47.
- **Not the bottleneck (refuted):** < 8 / 36 (fewer than 3 more than the identity).
- In between, or >= 11 with new non-turn zeros: inconclusive.
- **Stop rule:** 1.25 adds >= 3 new offroad zeros before the entry -> no 1.5 run.
- Reported regardless: exam turn scene completions (of 26), the shape gate count, per-scene table.

Evidence ceiling: one checkpoint, one seed, 47 scenes; a serving-side diagnostic of where the turn fails, not a recipe.

## Amendment 1 (2026-10-15 ~03:00 JST, after the smoke, before the 47-scene run)

Smoke `/data/runs/route_night/turn_gain/g1.25-smoke` (09528162, 132ae6cb; commit e2b30aa5): no crash, `tg` on every inference, served =
branch_poses(plan, 1.25) to the 4-decimal log (max 1.1e-4). Two defects of the registered rule, fixed before any read:
- It fired on a standing car (132ae6cb k 0-15: L2 0-0.6 m, dr_arc 0-1 deg; 6 recomputed flips were rounding of those near-zero values) and
  so made "every plan before the first on-decision equals the identity" vacuous (first on at k 0 in both sessions).
- It scaled plans that turn AGAINST the route (09528162 k 40-80: dplan +21 to +55 deg against a route turning -20 to -87 deg): the gain then
  amplifies a wrong-way turn, which is not the magnitude question.
Change (turn_gain.py; selftest extended): on additionally needs L2 >= 1 m, |dr_arc| >= 2 deg and sign(dr_arc) x dplan >= -1 deg. Everything
else, the scenes, lines and stop rule are unchanged. The smoke is rerun with the fixed rule (same 2 scenes, card 0) together with an
identity re-run of RSG-L18-s0 without the switch on the same 2 scenes (card 1, `RUNPFX=id-`) checked bit-identical to r281/rsg (rf_p4.py s1):
the driver edits must not change unswitched serving.
