# Pre-registration: INTENT HOLD serve-side switch on PAI (diagnostic / exam technique, NOT adopted), 2026-10-15

**Label.** A diagnostic of the policy, run at serving time. It is not a submission switch and not adopted without the user, whatever it
reads. No row is built from these rollouts. Written and committed before any rollout with the switch on is run; amendments are appended.

**Question.** Decision 308 (plan creep): in failed turns the plans are smooth and consistent and the controller tracks each to 2-3 cm, but the
car turns ~2.5 deg less than each plan asks over 2 s, and every new plan re-anchors at the car's actual pose and accepts the shortfall
(RSG-L18-s0 on PAI: 0.063 m outward per metre travelled in failed turns vs 0.015 in completed ones; 2-4 m over the approach; over-turn zeros
creep inward). If the served path is held to the policy's own earlier intent instead of re-anchoring, do the turns complete? This is the
serve-side test of mechanism 3 of research/next-round/turn-night.md section 9 ("a new plan should return to its own previous path").
Decision 301 (a plain lateral gain x1.25 on the plan, 6 / 36) is the magnitude control: this switch changes where the plan is anchored, not
how much it turns.

## Switch (`JEV_INTENT_HOLD=1`, `JEV_IH_A=<a>`; experiments/alpasim/lib/intent_hold.py, pai_driver.py; unset = not imported, serving unchanged)

At every inference, from the anchor p0 (the rear axle's pose at t0 in the session's local frame), the plan P (8 rear-axle poses at 0.5 s,
after serve_fix, as turn_gain), the fed speed v, the fed route memory and the command; per session a world-frame intent path I:
- W = P in the world frame with the rear axle as its first point.
- Reset (serve P unchanged, I = W) when: v < 1 m/s; P's arc < 1 m; no turn = the route memory's largest heading change in 0 .. 40 m ahead
  (turn_gain's dr40 on the same 40 slots) < 15 deg and the command is not LEFT / RIGHT; I is empty; or P's 2 s point lies more than 1.5 m
  from I (the world changed: obstacle, new route).
- Hold otherwise: s = arc length from the current rear axle; I_r = I from the rear axle's projection onto I; new intent
  B(s) = (1 - a) I_r(s) + a W(s) on the common arc, beyond I_r's end W(s) shifted by (1 - a) times the junction gap (continuous).
  Served path Q(s) = B(s) + w(s) (rear axle - B(0)) with w a cosine ramp 1 -> 0 over L1 = max(P's arc at 1.0 s, 1 m): it starts at the
  rear axle and is on B after ~1 s. Guard: the correction Q - W clipped to 1.0 m at every point (0.25 m arc grid), and every served pose
  pulled back to within 1.0 m of P's pose. Served poses = the clipped path at P's own arc length per time step (P's speed profile); yaw = P's
  yaw + (served chord heading - P's chord heading) at that arc (chord +- 0.5 m). I = B.
- Logged per inference: `ih` {on, why (slow / short / noturn / empty / far / "" = hold), d2, e0 (signed offset of the rear axle from I),
  cmax (max |served - P| over the poses), clip, dr40, v, a}; `poses_ih` (served) while holding; `poses` stays the plan.
- Variants: a = 0.3 (registered) and a = 0.15.

Open-loop replay on the identity's stored decisions before this prereg (rf_intent.py replay, a 0.3, 49 scenes; the car there followed P, so
corrections are larger than in closed loop): hold at 7 526 / 9 735 decisions (noturn 1 391, slow 672, far 100, short 24, empty 22);
max |served - plan| p50 0.14 m, p90 0.83 m, at the 1 m clip 469; |e0| p50 0.007 m.

## Run

- Checkpoint RSG-L18-s0; serving = the reference `/data/runs/route_night/r281/rsg` (pai_eval.sh, wizard
  `controller.gains.rel_front_steering_angle_weight=20`, `JEV_VCONT=1.0 JEV_LEAD=1`, CONC 4) plus the switch. Tokyo, one card, one stack
  (`rf_intent.sh chain`): the card is taken only when it has had no compute process, < 1 GB and no live pai_eval.sh naming it for 2 min.
- Scenes (49, `results/intent_hold/scenes49.txt`): the 47 of the turn-gain diagnostic (`results/turn_gain/scenes47.txt` = the 36 strict
  turns, which contain the 22 runway turns of decision 303 and the 5 completed strict turns, plus decision 298's exam turn scenes) and the
  PAI over-turn zeros of decision 299 not yet in them (300465f7, d8bfaaf2; 2466266e is already in). Routine deviation from the brief: of the
  11 over-turn zeros of d299 / d306 only 2 + 1 are PAI scenes (2466266e, 300465f7 with a near strict turn; d8bfaaf2 without one); the other
  9 are held nuPlan rollouts and cannot be run on PAI.
- Identity = the stored r281/rsg sessions (no re-run). Reproduced by the read code before this prereg: strict 5 / 36 (132ae6cb, 382a7247,
  3c2822d9, 8a143cb2, d90fe123), runway 3 / 22 (132ae6cb, 382a7247, d90fe123), zeros 33 / 49, spins 0 / 49, shape gate 1.
- Bit identity first, on the first 2 scenes (09528162, 132ae6cb), same card: (i) the new driver without the switch, checked bit-identical to
  r281/rsg (rf_p4.py s1: plans, tracks, scores; the driver edit must not change unswitched serving); (ii) smoke with a 0.3, `ihcheck`:
  every decision recomputed in sequence from the logged anchor / plan / speed / memory / command (branch flips from the 4-decimal logs
  listed), served = the recomputation, and every plan up to a session's first hold equal to r281/rsg at the same k. A crash, an identity
  difference or a pre-hold difference stops the chain (STOP).
- Then a 0.3 on the 49 scenes and a 0.15 on the 49 scenes, back to back on the same card (no idle gap for another lane to fall into; the
  chain is cheaper than a second wait). a 0.15 is READ only if a 0.3 fails a line below or fires the stop rule; otherwise it is reported as
  exploratory.

## Measures

- Completion (rf_v2pai.py turns / table: rf_pai_follow's rule on each label's own AlpaSim route; strict turn >= 60 deg within 30 m;
  completed = exit + 10 m before the first zero and within 2.5 m of the route), outcome and zero kind, where it failed; decision 293 shape
  gate; hold decisions per scene (count, first hold's station vs the entry).
- Zeros on the 49 (paired with the identity); new zeros outside the turns = zero under the switch, not under the identity, the event before
  the entry or no strict turn on the route; offroad ones among them.
- Spins: rf_pai281's rule on the 49 scenes.
- MECHANISM: plan creep = rf_switching.py's Amendment 1a measure (window entry -/+ 20 m before the first zero; j = the 2 s point of each
  plan against the previous plan's path, per metre of ego travel, median per turn; summed j), via rf_receding's extract on each run.
  Primary object: the SERVED plan (`poses_ih` where holding, else `poses`), which plays the role `poses` played in decision 308 (there the
  served plan was the plan). Also reported on the model plan (`poses`): if the creep was the plan accepting the car's shortfall, holding the
  car on its intent lowers the model plan's creep too; if the plan widens by itself, the model plan keeps creeping and only the served one
  drops (to ~a of it). The identity's numbers come from the same code on r281/rsg.

## Lines (each reported separately; per variant read)

- strict-36 completion >= 10 (identity 5; RV2b 7; APP1X 11).
- runway-22 completion >= 6 (identity 3).
- none of the identity's 5 completed strict turns lost.
- new zeros outside the turns <= 2.
- spins <= 2 (identity 0 on the 49).
- reported, no line: the d293 shape gate count; zeros and the paired mean score on the 49; exam turn scenes.
- MECHANISM: |median creep per metre| of the served plan over the turns that complete <= 0.03 m / m (the completed-turn level of d308),
  reported for all turns, failed and completed, on the served and the model plan.
- **Stop rule:** a 0.3 produces >= 3 new offroad zeros before the entry -> a 0.15 is read instead and the line stops there.

Reading: all five completion lines pass and the mechanism passes -> "holding the policy's own intent completes the turns: the gap is the
re-anchoring loop, not the plan" (a serve-side diagnostic; a training version would follow, not this switch). Completion lines fail with the
mechanism passing -> the creep is removed but is not what stops the turns. Mechanism failing -> the hold does not remove the creep in closed
loop; nothing is concluded about re-anchoring.

Evidence ceiling: one checkpoint, one seed, 49 scenes; a serving-side diagnostic of where the turn fails, not a recipe.
