# HUGSIM: is the "oncoming vehicle drives straight at the ego" case passable?

Question 2 of the open-loop review follow-up, HUGSIM half (Bench2Drive half:
[vlm_arb/results/opposite_vehicle_passability.md](../../vlm_arb/results/opposite_vehicle_passability.md)).
Status of every claim: **V** = verified (file:line or log), **I** = inferred.
HUGSIM checkout: `62c690d` plus our patches 0001-0003 (`$DATA_DIR/third_party/HUGSIM`); line numbers are that checkout.

## 1. Which case

The user's description is "an obstructing / leading vehicle turns left, then an oncoming vehicle drives straight at the
ego without slowing; the ways out are reversing fast or accelerating through". No HUGSIM case on the page has a
vehicle that turns left on a script (see section 2), so the match is by behaviour, not literal.

| page case | scenario | what is scripted | fits the description? |
|---|---|---|---|
| hugsim-07 | nuScenes scene-0138 extreme-00 | static car 20 m ahead in the ego lane + `AttackPlanner` vehicle starting 60 m ahead, heading the opposite way | best fit: blocked lane, head-on vehicle that never yields. Nothing turns left on a script; the attacker's heading drifts by about 7 deg (V, section 3) |
| hugsim-06 | Waymo 100613054308 hard-00 | one `ConstantPlanner` car, 2 m/s, from 38 m ahead / 21 m aside | oncoming, never yields; but it is a plain crossing, trivially passable (V, section 4) |
| hugsim-08 | Waymo 144248042870 extreme-00 | one `AttackPlanner` car starting 10 m ahead / 18 m aside, heading -1.9 rad | attacker cuts in from the side and its heading swings from -107 to -96 deg while closing (V); this is the only case where a vehicle visibly turns toward the ego |
| hugsim-03/04/05 | static inserted car or empty road | nothing moves | no |

If the user meant 07 or 08, the answer below applies to both; I cannot tell which one from the page alone.

## 2. How the benchmark scripts the other vehicle

| item | value | source |
|---|---|---|
| actor kinds in the 436 released scenarios | AttackPlanner 111, ConstantPlanner 315, IDM 4, none other | `grep` over `$DATA_DIR/datasets/hugsim/scenarios/*/*.yaml` (V) |
| IDM (the only planner that follows a map route and can turn) | only 4 scenarios (nuScenes 0013-hard-00, 0138-/0166-/0411-medium-01), all `load_HD_map: true`, not set up on our box | same grep (V) |
| ConstantPlanner | straight line at the yaml speed and yaw, never reacts | `sim/utils/agent_controller.py:304-312` (V) |
| AttackPlanner | spline sampler (accel grid -2..5 m/s^2, bound -6..5) picks the trajectory that minimises the distance to the **constant-velocity extrapolation of the ego over 5 s**; collision term only vs the 2nd..3rd nearest neighbours; no braking, no yielding | `agent_controller.py:210-300`, `plan.py:92-110` (V) |
| replan rate | `ATTACK_FREQ` = 10 steps = 2.5 s in extreme yamls; between replans the plan is executed blind | `plan.py:57,110`, yaml (V) |
| easy / medium / hard / extreme | none / static or slow ConstantPlanner / ConstantPlanner or IDM, some AttackPlanner / AttackPlanner (93 of 165 extreme yamls) | yaml counts: hard has 18 Attack, extreme 93 Attack + 72 Constant (V) |
| ego dynamics | kinematic bicycle, 0.25 s step, wheelbase 2.7 m, declared acc range +-2 m/s^2 (`configs/sim/kinematic.yaml`); iLQR tracker bound 3.0 m/s^2 (our tracker-v2 patch) | `hug_sim.py:277-290`, `lqr-tracker-v2.patch` (V) |
| collision | box intersection in the ego frame; "fg_collision" does not say who hit whom | `hug_sim.py:21-30` (V) |
| expert / reference driver | none exists in HUGSIM; the paper text was not retrievable here (abstract only), so the intended behaviour is inferred from the code: the attacker aims at where the ego would be if it kept its current velocity | I |

Consequence of the last two rows (I, supported by the counterfactuals below): the attacker is a homing vehicle that
re-aims every 2.5 s at the ego's constant-velocity extrapolation. Standing still is therefore the worst reply
(the extrapolated target is the ego itself), while a change of motion **after** the last replan makes it miss.

## 3. Recorded timeline of hugsim-07 (Cinque, PR#57 controller)

From `infos.pkl` of `runs/hugsim-exam/scored-op/cinque-fixed/zs/scene-0138_extreme_00`, 0.25 s steps
(`scripts/attack_tracks.py`). All V.

| t (s) | ego speed (m/s) | ego x (m) | attacker distance (m) | attacker speed (m/s) | note |
|---|---|---|---|---|---|
| 0.0 | 1.0 | 0 | 59.3 | 3.0 (yaml) | static car 20.0 m ahead, lateral -0.3 |
| 2.5 | 2.2 | 3.7 | 48.7 | 3.8 | replan 1 (heading unchanged) |
| 5.0 | 2.4 | 10.1 | 30.3 | 5.6 | replan 2: attacker heading starts to change (step 21) |
| 7.5 | 0.3 | 12.6 | 11.6 | 7.1 | replan 3, last one; ego already nearly stopped 7.4 m behind the static car |
| 8.75 | 0.1 | 12.7 | about 4 | 7.4 | overlap at step 35 |

Ego speed peaks at 2.7 m/s (step 16) and the ego brakes from t = 4.5 s on, i.e. it reacts to the static car, not to
the attacker (no cause for it can be read from these logs; the model's reasoning is not visible). Time from the attacker
becoming a possible conflict (it is on screen from t = 0 at 59 m) to contact: 8.75 s; time from the last replan: 1.25 s.

## 4. Could anything have avoided contact

Method (`scripts/attack_counterfactual.py`, CPU, hugsim env): rebuild `hug_sim.step()` + `planner.plan_traj()` with the
upstream `AttackPlanner` code and the simulator's own `create_rectangle` collision polygons, with two details found in
the harness work: `closed_loop.py:35,52` calls `env.reset()` twice, so actors step twice at t = 0; the box yaw of
actors is `stat[3]` while the ego box uses `-theta`.
**Validation**: replaying the recorded ego actions reproduces every actor position with 0.000 m error in all 19 attack
scenarios, and the first overlap step equals the recorded termination step in 17 of 19 (the other two are 1 step off).
Then 892 scripted ego manoeuvres per scenario were run: brake-and-hold (38), reverse (38), lane-change swerve with
speed profile (816). "free" = no overlap with any actor within 14 s; "ground" = additionally all four corners stay on
reconstructed ground points (a DAC proxy; ground points are sparse, so it is conservative). Manoeuvres are not model
outputs and need full knowledge of the simulator; they show what the rules allow.

| scenario (Cinque outcome) | brake-hold free / ground | reverse free / ground | swerve free / ground (of 816) |
|---|---|---|---|
| hugsim-07 nuScenes 0138 extreme (fg_collision) | 0 / 0 of 38 | 7 / 0 | 45 / 4 |
| hugsim-08 Waymo 144248042870 extreme (fg_collision) | 2 / 0 | 18 / 0 | 160 / 0 |
| hugsim-06 Waymo 100613054308 hard, ConstantPlanner (fg_collision) | 25 / 25 | 33 / 3 | 682 / 64 |

All 19 attack scenarios of the 64-scenario sample are in `results/attack_counterfactual.csv`: at least one free
manoeuvre exists in 19 of 19, a free manoeuvre that also stays on ground points in 10 of 19. Brake-and-hold avoids
contact in only 6 of the 19 scenarios (hard-01 x3 with 2 variants each, extreme-01 095 and 124 with 32-33, and hugsim-08 with 2); in 13 scenarios, including hugsim-07, none of its 38 variants works. Reversing is free only when it leaves the reconstructed ground (behind the start there is none).
The few on-ground solutions in hugsim-07 all begin the swerve between t = 4.5 and 6.0 s, accelerate at +3 m/s^2 and
shift 3 m sideways, i.e. they need a lateral move of about one lane width past the static car within the window between
the second and third replan. Window length about 1 s (swerve starts 4.5-6.0 s in the grid, 0.5 s resolution). I: this is
a grid over one manoeuvre family, not a search for the optimum, so "none" entries are weaker evidence than the "exists" entries.

## 5. Do models pass

Our 64-scenario sample, PR#57 controller, the 19 attack scenarios (`scored_op.csv`, `scored_base.csv`; V):

| agent | complete | fg_collision | other |
|---|---|---|---|
| openpilot Cinque | 3 | 16 | 0 |
| openpilot Lebowski | 1 | 15 | 2 off_route, 1 max_steps |
| LTF (official client) | 1 | 15 | 3 bg_collision |
| constant velocity | 0 | 18 | 1 bg_collision |

Per scenario: `attack_counterfactual.csv`, last four columns.

Published (HUGSIM paper Table 13, four-dataset mean, `published_table13.csv`; V as a copy of the table, hard and extreme
mix attackers with ConstantPlanner cars): HD-Score by difficulty, easy / medium / hard / extreme: UniAD 0.487 / 0.295 / 0.273 /
0.143, VAD 0.243 / 0.099 / 0.104 / 0.083, LTF 0.528 / 0.246 / 0.198 / 0.081; mean NC at extreme 0.54 / 0.36 / 0.28. No
per-scenario results, no leaderboard rows and no per-scenario public result of any top entry were obtainable (the
leaderboard Space was down on 2026-09-24 and 2026-10-03; WA-JEPA reports only an overall 0.4462). So "how often is it
passed" is known only at difficulty level: at extreme every published agent has NC below 0.55.

## 6. Verdict (HUGSIM)

- Passable in the simulator's rules: yes, in 19 of 19 attack scenarios there is a contact-free manoeuvre; in 10 of 19 it
  also stays on mapped ground. It is not passable by the two replies the user names: stopping or braking is the worst
  reply (0 of 38 brake-hold variants in hugsim-07), and reversing only "works" by leaving the reconstructed road.
  The workable reply is a late lateral swerve with acceleration, timed between replans, which no tested model produced (V that no
  model avoided hugsim-07; I that none could be expected to).
- What it tests: the extreme level measures whether the planner can leave its lane toward free space in about one second
  against a vehicle that homes in on the extrapolated ego. It is not a yielding test; HD-Score then multiplies by route
  completion, so even a successful swerve has to rejoin the route to score. The user's "nit-picking" reading is fair for
  the cases where the ego is boxed in by a static car (07); it is not fair for hugsim-06, where waiting or driving on both
  pass (LTF scores 0.967 there).
- Caveat: the matching of the user's description to hugsim-07/08 is by behaviour; no scripted left turn exists.

## Files

- `experiments/hugsim/scripts/attack_tracks.py`, `attack_counterfactual.py`, `attack_counterfactual_table.py`
- `experiments/hugsim/results/attack_counterfactual.csv`
