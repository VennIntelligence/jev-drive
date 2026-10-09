# SWV1: does the plan know where the ego body will be? Swept-footprint clearance on the collision rollouts (2026-10-10)

Offline read of existing AlpaSim rollouts (nuPlan public scenes: P2H10-F s0 / s1 and APY10m10-AB s0 / s1; PAI: P2H10-F-s0 baseline and
COL1's re-timed arm `v1`), no closed-loop run, no training. Pre-registration with amendment 1:
[plans/2026-10-10-swv1-prereg.md](../plans/2026-10-10-swv1-prereg.md) (pushed before any number). Tables: [swerve/](swerve/)
(`classes.md`, `classes.csv`, `decisions_<dom>.csv`, `bc_nuplan.md`, `bc_pai.md`, `cf_<dom>.csv`, `signals_<dom>.csv`). Figures:
`../figs/swerve/`. Code: `scripts/swv1_lib.py`, `swv1_a.py`, `swv1_bc.py`, `swv1_report.py`, `swv1_replay.py` (on COL1's extractions and
replay inputs). Simulator boxes, the logged path and recorded object motion are labels and oracle inputs; arms that use them are ceilings.
Decision 220. Builds on COL1 ([collisions.md](collisions.md), decision 219).

## 1. Result in short

1. **Every collision is already in the plan.** In all 66 at-fault collision rollouts read (22 + 24 nuPlan, 12 + 8 PAI) some served plan
   in the last 3 s has a swept ego box that intersects the struck object's box at the same time; the first such plan is issued a median
   of 3.0 s (nuPlan, 5 s scenes) and 3.3 s (PAI) before impact. Class (iii) "the planned sweep clears, the executed one does not" is
   empty. The simulator's controller delivers the plan: lateral tracking error p95 has a median of 0.02-0.04 m per rollout (largest
   0.17 m), delivered yaw rate is 1.0-1.08 x the demanded one (three PAI rollouts excepted). Execution is not the problem.
2. **Class shares** (figure `fig1_classes.png`): nuPlan P2H10-F: straight into an in-path vehicle (i) 4 of 22, a turn or go-around whose
   own sweep intersects (ii) 6, drift without a manoeuvre (iv, decision 205) 10, other (v) 2. PAI baseline: (i) 6 of 12, (ii) 5, (iv) 1.
   The user's item 2 is 27 % of the nuPlan collisions and 42 % of the PAI ones; item 1 is 18 % and 50 %; on nuPlan the largest class is
   still drift (45 %), and its plans also run through the object.
3. **Not a lateral-acceleration problem.** The plans are slow: the largest planned v^2 kappa in any collision rollout is 2.8 m/s^2
   (nuPlan) and 3.0-3.3 m/s^2 (PAI), the median of the per-rollout maxima 0.4 and 1.4 m/s^2. A speed cap from a lateral-acceleration
   limit avoids **0 of 22 and 0 of 12** collisions at every limit from 4.0 down to 1.0 m/s^2, openpilot's 3.0 included. The plan falls
   short laterally: the smallest rigid shift that would clear is a median of 0.85 m (nuPlan P2H10-F; 1.0-2.6 m in the turns) and 1.6 m (PAI).
4. **The information that is missing has a large ceiling.** A stop before the first planned intersection, with the simulator's boxes
   (privileged), avoids 11 of 22 nuPlan and 5 of 12 PAI collisions with a 0.5 m margin and 20 of 22 / 8 of 12 with a 2 m margin, while
   1.0 % of 198 clean nuPlan rollouts and 0 of 21 clean PAI rollouts lose more than 10 % of progress.
5. **No head the driver discards carries it on nuPlan.** Against "the planned sweep intersects an object": the lead head 0.49, road
   edges 0.72 [0.64, 0.82], lane lines 0.46, meta 0.36-0.70, desire 0.66, all below the line. One registered signal passes, the served
   plan's own lateral spread at 2 s (0.786 [0.738, 0.865]); it drops to 0.65 on the all-decision read and to 0.70 on APY10m10, flags 10
   of 22 rollouts 1.5 s ahead, and says nothing about where the contact is. On PAI the lead head passes (0.876 [0.735, 0.975], 9 of 12
   flagged 1.5 s ahead, median 6.1 s), but its lateral position adds nothing over its probability (0.850) and the footprint rule built
   on it avoids 1 of 12: this is the in-path half COL1 / FIX1 already use.
6. **Registered lines.** Line B (a serving-side switch avoids >= 30 % of class (ii) + (iii) at <= 5 % clean cost): not met by either
   non-privileged arm (0 of 6 and 0 of 5; the class has fewer than 8 rollouts per domain, so descriptive; over all collisions 0 of 22
   and 0-1 of 12). Line C (AUC >= 0.75, lower bound > 0.65): nuPlan only `plan_std2_FT`; PAI the lead signals.
7. **Reading.** Item 2 is a clearance-estimation failure of the plan (ii), on top of drift (iv); neither is execution or speed in turns.
   There is no serving-side switch to forward to lane FIX1 from this lane. The fix is training-side: supervise the consequence of the
   adapter's own plan (section 6).

## 2. Method

- Rollouts and records: COL1's extractions (`cases.pkl`, `lead/ctrl_logs.pkl` on the GPU box; `x/*/logs.pkl` and replays on the Tokyo
  box), nothing re-simulated. Collision rollouts C and clean rollouts N as registered (N: 198 nuPlan control rollouts of P2H10-F-s0, 21
  PAI rollouts without any collision flag). New: one pool job (`swv1_replay.py`, 5 min, 23 MB) replays the nuPlan driver inputs again
  keeping every output head of FT (served checkpoint) and P0 (shipped policy weights on the same tokens).
- Sweep: the served trajectory of a decision (41 poses at 0.1 s over 4 s), the simulator's ego box (nuPlan 5.176 x 2.297 m; PAI from the
  rollout), against every object box at the same time. Objects continue at their last velocity beyond their record (amendment 1).
  `short` = the smallest rigid lateral shift of the plan after which its sweep clears the struck object (0.1 m steps, up to 3 m).
- Counterfactuals: the ego keeps its executed path and is re-timed (never ahead of the original); contact = ego box against the struck
  object's recorded box; avoided = no contact to the end of the rollout and not merely deferred.

Class rules as registered, with three refinements made after the first table and before the shares were written (all in `swv1_a.py`):
1. A turn is a turn of the **log** (>= 20 deg within 8 s either side of the impact; the 5 s scene on nuPlan). A plan that turns >= 20 deg
   on a straight log is drift (decision 205: the plan continues the yaw it sees), flagged `plan_turn_on_straight` (6 of 22 nuPlan).
2. A go-around needs something that would be reached within 4 s at the current speeds (a lead 40 m ahead at the ego's speed is not it).
3. "Tight pass": the ego is within 0.5 m of the logged path and clips a standing object beside it (the log passes it with room) = a
   manoeuvre, class (ii) / (iii) by the sweep (1 PAI rollout, `p04`).
The logged-path offset on nuPlan is COL1's published one; the logged path is continued 40 m straight (COL1's rule).

## 3. A. Classes

| set | n | (i) straight into an in-path object | (ii) manoeuvre, planned sweep intersects | (iii) manoeuvre, planned sweep clears | (iv) drift | (v) other | planned sweep intersects the struck object in the last 3 s | first intersecting plan, median s before impact | median shortfall m | largest planned a_lat m/s^2 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nuPlan, P2H10-F s0 + s1 | 22 | 4 (18 %) | 6 (27 %) | 0 | 10 (45 %) | 2 (9 %) | 22 / 22 | 3.0 | 0.85 | 2.78 |
| nuPlan, APY10m10-AB s0 + s1 | 24 | 9 (38 %) | 8 (33 %) | 0 | 6 (25 %) | 1 (4 %) | 24 / 24 | 3.0 | 2.00 | 2.77 |
| PAI, P2H10-F-s0 baseline | 12 | 6 (50 %) | 5 (42 %) | 0 | 1 (8 %) | 0 | 12 / 12 | 3.3 | 1.60 | 3.02 |
| PAI, re-timed arm `v1` | 8 | 5 (62 %) | 2 (25 %) | 0 | 1 (12 %) | 0 | 8 / 8 | 3.5 | 1.70 | 3.26 |

Clips (COL1's ids, `figs/collisions/`), to check against what was seen:

| domain | (i) | (ii) | (iv) | (v) |
|:--|:--|:--|:--|:--|
| nuPlan P2H10-F | n01, n02 | n05, n07, n11 (turns: 30, 62, 22 deg, 5 rollouts in 3 scenes), n04 (go-around, one seed) | n03, n06, n08, n09, n12 (and the other seed of n04) | n10 |
| PAI baseline | p01, p03, p05, p09, p10, p11 | p02 (go-around), p04 (tight pass behind a parked car), p07, p08, p12 (turns) | p06 | |

- **(ii) on nuPlan is the wide turn.** n07 (figure `ex_n07_turn_ii.png`): from 3 s before impact every plan's sweep runs through the
  vehicle standing at the far side, short by 2.8 -> 1.9 m, at a planned lateral acceleration of 1.3-1.5 m/s^2 and 3-5 m/s. n05: short
  by 1.0 m at 0.45 m/s^2. n11: a launch into a left turn across a 12 m/s vehicle (2.8 m/s^2, the largest in the set).
- **(ii) on PAI**: p07 and p02 leave the lane at 13-14 m/s into the adjacent lane's vehicle (short by 1.8 and 1.3 m, planned contact
  2.5-2.9 s ahead; `ex_p07_goaround_ii.png`, `ex_p02_goaround_ii.png`); p04 pulls out behind a parked car, short by 0.3 m.
- **(iv)** plans intersect too (first 1.5-5 s before impact, short by 0.4-1.3 m): the drifting plan has no clearance term either
  (`ex_n06_drift_iv.png`, `ex_n03_drift_iv.png`).
- **(i)**: the plan drives through the lead at every decision (`ex_n01_lead_i.png`); lanes FIX1 / TR1 / DIAG1.
- Execution: lateral tracking p95 per rollout: median 0.02 m (nuPlan), 0.04 m (PAI), largest 0.17 m. Yaw rate delivered / demanded
  (rms over the window, median) 1.04 / 1.08 / 1.00. Three PAI rollouts (9ea70552, 605bf77a, 0ee89cab) deliver 3-12 x the demanded yaw
  rate in the window (the controller's braking zig-zag of COL1 section 4.3); their plans intersect anyway.
- Per rollout: [swerve/classes.md](swerve/classes.md).

## 4. B. Counterfactuals (kinematic; `swerve/bc_nuplan.md`, `bc_pai.md`; figure `fig2_ceiling_signals.png`)

| arm | nuPlan P2H10-F avoided / 22 | class (ii) | nuPlan APY10m10-AB / 24 | PAI baseline / 12 | class (ii) | clean rollouts losing > 10 % progress (nuPlan 198 / PAI 21) |
|:--|--:|--:|--:|--:|--:|:--|
| lateral-acceleration cap 3.0 m/s^2 (**primary**: openpilot `MAX_LATERAL_ACCEL_NO_ROLL`) | 0 | 0 / 6 | 0 | 0 | 0 / 5 | 0 % / 0 % (never binds on nuPlan; binds in 12 of 21 clean PAI rollouts without a loss) |
| cap 1.7 m/s^2 (openpilot's total-acceleration budget in turns) | 0 | 0 / 6 | 0 | 0 | 0 / 5 | 0 % / 0 % |
| cap 1.0 m/s^2 | 0 | 0 / 6 | 0 (2 deferred) | 0 | 0 / 5 | 6.1 % / 9.5 % |
| lead-sweep stop, P0 lead head (**registered arm**) | 0 | 0 / 6 | 0 | 1 | 0 / 5 | 0 % / 0 % |
| lead-sweep stop, 2 m margin (sensitivity) | 0 | 0 / 6 | 0 | 2 | 0 / 5 | 0 % / 0 % |
| footprint-aware stop, 0.5 m margin (**privileged ceiling**) | 11 | 1 / 6 | 13 | 5 | 1 / 5 | 1.0 % / 0 % |
| footprint-aware stop, 2 m margin (privileged ceiling, sensitivity) | 20 | 6 / 6 | 21 | 8 | 3 / 5 | 1.0 % / 0 % |
| shipped weights' plan on the same tokens: no P0 plan in the last 3 s intersects | 2 | 0 / 6 | 1 | 1 | 0 / 5 | open loop, no cost read |
| logged human path at the driven speed (COL1's column, by this lane's class) | 20 | 4 / 6 | | 8 | 3 / 5 | privileged |

- **openpilot's own limits** (source at `tmp/opsrc`, commit ec95db3): `MAX_LATERAL_ACCEL_NO_ROLL = 3.0 m/s^2` and `MAX_LATERAL_JERK =
  5.0 m/s^3` in `selfdrive/controls/lib/drive_helpers.py::clip_curvature`: a clip on the commanded curvature, not a speed limiter (it
  would under-steer, not slow). The only curve-related longitudinal term is `_A_TOTAL_MAX_V = [1.7, 3.2]` at 20 / 40 m/s in the
  longitudinal planner, a cap on acceleration in turns. Speed in a turn comes from the model's plan in both systems.
- The cap arms change nothing because the collisions happen at 2-7 m/s (nuPlan median 2.5 m/s) and, on a path that goes through a
  standing object, slowing only defers. The lead-sweep stop fails for a different reason: near contact the lead head keeps the lead 2-10 m
  away (decision 157's near-range over-read; in n01 it reads 4-10 m at a true gap of 1.5-0.3 m), so the stop point recedes with the ego.
- The privileged stop with 0.5 m is half as good as with 2 m because the contacts are grazes: the executed path differs from each plan
  by decimetres, and a stop 0.5 m of arc before the planned first contact still touches beside the object.
- The shipped weights' plan is no better laterally: it intersects the struck object at some decision in 20 of 22 (nuPlan) and 11 of 12
  (PAI) rollouts, at a lower share of decisions (median 0.80 / 0.57 against 0.90 / 0.78 on nuPlan, 0.56 against 0.94 on PAI) because it
  is shorter (decision 219), not because it steers clear (green lines in the example figures).
- The logged human path clears 4 of the 6 nuPlan class (ii) rollouts (not n11) and all 10 of class (iv).

## 5. C. Non-privileged signals (`swerve/signals_<dom>.csv`)

Label: the served plan's sweep intersects an object box. Primary read: window decisions of the collision rollouts with the sweep on the
struck object, against clean decisions without any intersection, speed-matched; AUC with a bootstrap over logs.

| signal | nuPlan P2H10-F (122 decisions, 22 rollouts, 9 logs / 1827 clean) | nuPlan APY10m10-AB (P2H10 controls, flagged) | PAI baseline (322 decisions, 12 scenes / 1857 clean) |
|:--|:--|:--|:--|
| `lead_sweep` P0 (plan footprint against the lead head's box) | 0.489 [0.365, 0.644] | 0.628 [0.493, 0.780] | **0.876 [0.735, 0.975]** usable |
| `lead_need` P0 (COL1's 1-D rule, reference) | 0.488 [0.353, 0.644] | 0.630 | 0.846 [0.714, 0.962] usable |
| `lead_p` P0 | 0.397 [0.297, 0.509] | 0.494 | 0.850 [0.720, 0.953] usable |
| `edge_margin` P0 (ego box against the model's road edges) | 0.718 [0.643, 0.822] | 0.615 | not stored |
| `lane_exc` P0 (ego box beyond the inner lane lines) | 0.456 [0.380, 0.534] | 0.474 | not stored |
| `meta_brake` / `meta_dis` / `meta_steer` P0 | 0.484 / 0.382 / 0.701 [0.656, 0.810] | 0.493 / 0.393 / 0.625 | not stored |
| `desire_lc` P0 | 0.659 [0.548, 0.789] | 0.617 | not stored |
| `plan_std2_FT` / `plan_std4_FT` (served plan's lateral spread) | **0.786 [0.738, 0.865]** usable / 0.766 [0.705, 0.855] candidate | 0.697 [0.589, 0.826] / 0.668 | not stored |
| `plan_std2_P0` | 0.745 [0.689, 0.835] | 0.618 | not stored |
| `base_gap` / `base_len` (served plan against the P0 plan) | 0.581 / 0.428 | 0.684 / 0.493 | 0.709 / 0.741 |
| plan-only baselines: speed / `alat` / 4 s heading change | 0.485 / 0.665 / 0.677 | 0.499 / 0.627 / 0.626 | 0.469 / 0.496 / 0.532 |
| `combo` (out-of-fold logistic regression on all of the above) | 0.707 [0.557, 0.870] | 0.639 | 0.868 [0.645, 0.976] |
| privileged reference: the same sweep against the simulator's boxes frozen at the decision time | 0.728 [0.624, 0.846] | 0.763 | 0.869 [0.733, 0.967] |

- **nuPlan**: the geometric heads do not separate. `plan_std2_FT` passes the registered line on P2H10-F and nowhere else: 0.697 on
  APY10m10, 0.651 [0.586, 0.745] on the secondary read (every decision whose sweep intersects against every one that does not), rollouts
  flagged >= 1.5 s before impact at 10 % false positives 10 of 22 (median lead 0.5 s). It marks the collision rollouts (turns, drifting
  plans: `alat` and heading change alone give 0.67-0.68) more than the intersection. One pass among 26 non-privileged reads, no multiplicity correction.
- **PAI**: every lead-derived signal passes and they are the same signal (the 10 % threshold of `lead_sweep` is the probability gate
  itself). 9 of 12 rollouts are flagged >= 1.5 s ahead (median 6.1 s): COL1's result, read against a different label. The lateral
  position of the lead adds 0.03 AUC over the probability and nothing in the loop arm (section 4). Lane lines, road edges, meta and the
  plan spread were not stored by COL1's PAI replay and both Tokyo cards were in use by lane FIX1: not read on PAI.
- How often the label fires without a collision: at 7.7 % of clean nuPlan decisions (48 of 198 rollouts) and 7.1 % of clean PAI
  decisions (12 of 21 rollouts) the planned sweep intersects an object under constant-velocity continuation, and the next plans move off
  it. With objects frozen at the decision time it is 35 % / 17 %; on nuPlan with the sweep truncated at the 5 s record 1.2 %.
- Difference to decisions 158 / 179 / 203: those read pooled features or one head against 4 s non-reactive simulator outcomes of single
  open-loop tokens. Here the label is a geometric property of the served plan at closed-loop decisions, and the heads are placed
  against the same swept box. The answer on nuPlan is the same (edges 0.72 here, 0.64 there against DAC); the new facts are that the
  plan's own sweep is the failure in every collision, and that the plan's spread is the only output that moves with it.

## 6. D. Conclusions

1. **Which failure.** Item 2 is a clearance-estimation failure of the plan, class (ii): 6 of 22 on nuPlan (wide turns), 5 of 12 on PAI
   (lane departures into the adjacent lane, wide turns, a tight pass). Class (iii) is empty; the controller does what the plan says.
   Speed is not the lever: no lateral-acceleration cap avoids anything. Drift (iv) stays the largest nuPlan class (10 of 22) and shares
   the same property: the plan runs its own box through a standing vehicle 1.5-5 s ahead without changing course or stopping.
2. **Cheapest serving-side switch with a real ceiling: none found.** The ceiling exists (footprint stop: 20 of 22 and 8 of 12 at 1 % /
   0 % clean cost) but needs object boxes; the model's lead head supplies them only for the in-path vehicle, where FIX1's registered lead
   cap is the better use (it holds at standstill; the footprint rule on the same head avoids 0 of 22 and 1 of 12). Nothing is forwarded
   to FIX1 as a registered arm. For completeness, the one signal that passed line C, as an exact definition should a diagnostic arm be
   wanted: `s = exp(plan[495 + 15 i + 1])` interpolated at 2 s on `T_IDXS` (the served checkpoint's lateral position spread; no extra
   pass); trigger when s > 0.46 m (10 % false positives on the clean nuPlan decisions). It has no stop point, and slowing on the same
   path avoided nothing in section 4, so its expected closed-loop effect is progress loss in turns; not recommended.
3. **What a training-side fix has to supervise.** The consequence label of the adapter's own plan, as in research/next-round/plan.md's
   consequence-supervised branch: for each training row, the served-geometry sweep (ego box along the predicted 4 s trajectory) against
   logged agent boxes at the same time and against the road boundary; targets = intersects (0 / 1), time and arc of the first contact,
   lateral shortfall. From this lane: (a) the label is well-posed and early (first intersecting plan 3 s before impact, shortfall
   0.3-2.6 m, so the head needs sub-0.5 m lateral precision at 5-15 m range); (b) on-policy rows are required: the collisions sit at
   states the log never visits (drift 0.5-2.4 m off the path, 13-39 m ahead of the log on PAI), and at 7 % of clean decisions the label
   fires transiently, so open-loop navtrain rows alone give mostly the easy negatives; (c) it is not learnable from the outputs the
   frozen model already has: road edges 0.72, lane lines 0.46, lead 0.49 on nuPlan, combination 0.71, and even the simulator's current
   boxes without their future reach only 0.73 there; the lead head is enough only for the in-path case (0.88 on PAI). So the branch
   needs a near-range object signal that the frozen heads do not provide (decision 203 reached this from the open-loop side), and its
   supervision should include object motion, not only current occupancy. Decision 158's plan-head hinge on frozen features stalled at
   pilot scale with the same label family; the difference to carry forward is predicting the consequence as an output (contact, arc)
   that a stop / re-plan can read, rather than only shaping the plan.

## 7. Limits

- Counterfactuals are kinematic and open loop: the executed path with logged decisions; the model never sees its changed state; traffic
  is replayed; a stopped ego can be hit from behind. "Avoided" for the privileged stop is an upper bound.
- nuPlan scenes last 5 s (about 10 decisions, objects at 2 Hz): the sweep beyond the record uses constant-velocity continuation, the
  lead time is capped at 5 s, and class counts are small and paired (22 rollouts = 12 scenes, 9 logs; the (ii) class is 4 scenes).
  PAI: 12 collisions, one seed. Line B is descriptive in both domains (fewer than 8 class (ii) + (iii) rollouts).
- Class rules use the logged path as lane reference and were refined three times after the first table (section 2); "manoeuvre intent"
  is inferred from the log's turn and from an object ahead, not from the model.
- APY10m10-AB positives are read against P2H10-F controls. PAI has only the lead head and the plans stored.
- `plan_std` is read as exp of the plan slice's second half (lateral column); its unit was not calibrated against errors.
- The lead box uses selection 0 at t = 0 moved at the reported lead speed, not the head's 2 s / 4 s entries (their frame is ambiguous);
  stated before the arm was read, a deviation from the registered "0, 2, 4 s interpolated".
- Example figures and class assignments were checked against COL1's clip sentences, not by re-watching all 31 clips.

## 8. Cost

GPU box: one pool job, 5 min on one card (0.08 card-hours), 23 MB under `$DATA_DIR/runs/alpasim/swv1/`; nothing re-simulated, COL1's
deleted control logs were not re-created (its `ctrl_logs.pkl` and `lead/msgs` were enough). Tokyo box: read only (29 MB of
frame-stripped replays written to `/tmp/swv1_rep`, removed). Analysis on the Mac from `tmp/swv1/`.
