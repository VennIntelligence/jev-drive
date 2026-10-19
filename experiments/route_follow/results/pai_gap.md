# PAI-GAP: where NTV0-L18-s0 + ACC + GAP loses points on PAI 281 (2026-10-19)

Diagnosis, no training. This is the PAI counterpart of decision 372. Prereg [../plans/2026-10-19-pai-gap-prereg.md](../plans/2026-10-19-pai-gap-prereg.md)
(4dd56fe0). Its amendment 1 (edf308c4) was written after the A0-only dry run and before H had any rollout. Code: `scripts/pai_gap.sh`
(pool chain) and `scripts/pai_gap.py` (read-out). Generated tables: [pai_gap/tables.md](pai_gap/tables.md) and
[pai_gap/pai_gap.json](pai_gap/pai_gap.json). The box root `$DATA_DIR/runs/route_follow/pai_gap/` holds the H read, `tax/`, and the per-scene
`scenes.csv`, `attr.csv`, `cuts.csv.gz` and `onset.csv`. One seed throughout.

**Inputs.**
- **A0** is the stored read the candidate was chosen on (d366): 280 scenes, mean 0.7443, 71.6 points lost (1 point = one scene).
- **N0** is the same model without ACC + GAP (d376).
- **Six other stored ACC + GAP reads** (NTV0-s1, NTV0-hl, JTN-s0/s1, NTVX, RSG) feed best-of-reads.
- **H** is new: a log-replay read on A0's chunk lists. `onp_log_driver.py` makes the MPC track the human's log, while the model runs at
  the human's states under the A0 serve.

**Gates.**
- H c000: 6 362 / 6 362 decisions after the force-GT window return the log, and there are 0 RPC errors.
- H's ego stays on the log: median 0.05 m, p90 0.12 m off.
- A0 and H share the clock (0 us difference over 277 scenes).
- The logged traffic equals the rollout.asl actors on 58 A0 rollouts (98 648 actor-steps, 100 % within 0.05 m). PAI traffic is pure log
  replay.
- **Deviation:** the first H submit failed on `sim_fast` (the log driver had no module-level `CAM`; fixed in 5512a6d1). A resubmit then
  queued all 8 chunks ungated. c001..c007 were cancelled before they started and resubmitted behind the gate (`pai_gap.sh resume`).

## 1. Where the points go

**The human scores 1.000 on all 277 H scenes.** No A0 loss is unattainable (attribution iii = 0 points), and the ceiling is 1.0. On
nuPlan the same ceiling was 0.990 (d372).

| loss (A0, 280 scenes) | scenes | points | stratum where it sits | best-of-7 reads recovers | attribution (points): (i) / (i-cl) / (ii) / mixed |
|---|--:|--:|---|--:|---|
| **corridor: passes the human's rest point** (cor_route_end; the human ends still, 4 m disc) | 23 | **23.0** | stop/lead 22 | 2 | 4 / **18** / 1 / 0 |
| **turn / junction** (corridor straight-at-turn 11, turn under 3 / over 1; offroad at a junction or curve 8) | 23 | **23.0** | turn + stop/lead | 1 | **15** / 8 / 0 / 0 |
| **rear-hit partials** (logged follower hits a slower ego; progress truncated) | 26 | **10.4** | launch 4.5, stop/lead 2.6 | 2.9 | 2.5 / - / **3.8** / 4.2 |
| drift / lane / straight offroad (cor_drift 5, lane change 1, off_straight 4) | 10 | 10.0 | mixed | 3 | 3 / 7 / 0 / 0 |
| AF-front | 3 | 3.0 | stop/lead, turn | 1 | 0 / 0 / 0 / 3 |
| slow partials | 11 | 2.2 | spread | 0.6 | 1.0 / - / 0.7 / 0.5 |
| **all** | 96 | **71.6** | | 11.0 (best-of 0.783) | |

How to read the attribution columns:
- **(i)**: the model plan already departs from the human on the human's own state (H, 4 s before the event station). For lateral losses
  that is a lateral error >= 1 m at 4 s. For cor_route_end it is a 4 s overshoot >= 4 m. For partials it is a 2 s arc below 0.9 x the
  human's.
- **(i-cl)**: the plan is right on the human's state and goes wrong on the model's own state.
- **(ii)**: the serve side. For lateral zeros: N0 scores and ACC serves faster than both the model and the human. For partials: >= 50 % of
  the window's served speed is set by the lead cut or the curve limit.

**Arithmetic to +0.05 (14 points).**
- Either of the two top rows alone is worth +0.082 (23 points).
- The rear-hit row is worth at most +0.037. Its serve-side part (3.8) is lead cuts on real leads (item 2).
- Static windows, d372's main loss on nuPlan, are 2 scenes here (1 point).

**Row 1, passing the rest point (23).** Twenty of the 23 have no logged vehicle within 15 m of the human's rest point. 17 have a map wait
line within 12 m (STOP 10, STOP + UNKNOWN 6, UNKNOWN 1). So these are stops at stop lines or signs without a lead, the PAI analogue of
d372's red line.
- **On the human's state the plan stops nearly where the human does.** The median 4 s overshoot is 2.0 m, and 18 / 23 are under 4 m.
- **In A0 the car reaches those stations at 1.41 x the human's speed.** The served speed equals the model's own plan (served / model 1.00),
  and the term that sets it is mostly ACC `set`. ACC's v_set = max(handoff speed, the plan's far speed).
- **The cause is not ACC.** N0, the same model without ACC, also passes 20 / 23.
- Exploratory onset read: take the first H decisions where the human (>= 3 m/s) brakes to a stop within 4 s. There the plan covers 1.17 x
  the human's 4 s arc in **both** the 20 scenes A0 fails and the 20 where A0 stops and scores. On the human's state the plan cannot tell
  the two groups apart.
- Reading: the model brakes when its own state is already braking, which is the trend follower of d375 / d376. It does not start the stop
  on its own state from the line ahead.

**Row 2, turns and junctions (23).** 15 points are (i). On the human's own state, at the turn, the plan already ends 1.8 m (median; 2.1 m
for the straight-at-turn ones) off the human's 4 s point. This is the plan not taking the route's turn, not closed-loop drift. Served /
model = 1.00, so ACC does not change the path speed here. The curve limit sets the speed on about 40 % of the decisions (median scene, straight-at-turn).

## 2. JEV_LEAD cuts: real leads, not phantoms

A0 has 50 474 inference decisions at >= 1 m/s, and 28 485 of them carry a lead cut > 0.

| subset | decisions | real lead in path | adjacent lane | nothing | no lead reported (MPC transient) | phantom share, decisions / cut metres |
|---|--:|--:|--:|--:|--:|--:|
| cut > 0 | 28 485 | 14 215 | 429 | 190 | 13 651 | 0.50 / **0.085** |
| cut >= 0.5 m | 7 800 | 7 195 | 252 | 135 | 218 | 0.078 / 0.078 |
| ACC won = lead | 9 526 | 8 798 | 316 | 157 | 255 | 0.076 / 0.078 |
| H (the human's own states) | 29 162 | 12 909 | 135 | 57 | 16 061 | 0.56 / 0.036 |

- When a lead is reported, it is a real logged vehicle in the plan's path 96 % of the time (14 215 of 14 834). The reported distance has
  median error 0.0 m (IQR -4.3 .. +1.6 m).
- Half of the cut decisions report no lead. These are the MPC's own transients, at 0.1 m per decision.
- **Phantoms are 8.5 % of the cut metres, 6.4 % before the first event.**
- Ceiling worth of removing them: 0.8 points pooled (+0.003). The strict per-scene bound is 5.5 points in 16 scenes (+0.020), and that
  bound does not count the AF-front risk.
- The real-lead cuts are more conservative than the human. During them the served speed is median 0.96 x the human's speed at the same
  station, and below 0.9 x on 38 % of decisions. The lead cut's remaining cost is therefore the gap calibration on a real lead (a serve
  question), not phantoms.

## 3. What each top loss is, and the model-side lever

1. **Passing the human's rest point at a stop line: the model, on its own state (i-cl, 18 of 23 points).** It is not a sim artefact (H
   = 1.0). It is not ACC: N0 fails the same scenes, and served = model. It is not a missing lead (20 / 23 have none). Labels on the
   human's state are already met, so more imitation at the human's state will not fix it.
   - **Lever:** a one-sided consequence row on the model's own approach states. Use PAI / nuPlan on-policy S rows where the human ends
     still within the plan's reach with no lead, and penalise only the plan's 4 s arc beyond the human's rest station (minus a margin).
     This is no speed target and no launch term, so it stays off the waiting / launch axis of d352 / d353 / d365. It acts only on moving
     rows with a rest point ahead.
   - **Precondition (d362):** "the human will stop" is only weakly readable in PAI tokens, except stop lines within about 3 s (AUC
     0.912). Run the stop-line probe on these 17 wait-line scenes first. If the line is not in the tokens, the problem is the input.
2. **Turns and junctions: the plan itself (i, 15 of 23 points).** On the human's own state at the turn, the plan already ends about 2 m
   off the human, mostly going straight where the route turns.
   - **Lever:** route-conditioned turn rows at the human's states on PAI scenes (the practice-pool idea, d367, at junctions only) plus a
     check that the route memory points into the turn.
   - Closed lines warn against expert-label DAgger (d321). This lever is plain imitation at on-log states, which those lines did not
     test.
3. **Rear-hit partials (10.4 points):** split between serve (3.8, real-lead cuts that are more cautious than the human) and model (2.5,
   under-planned launch, arc 0.79-0.88 x the human's). It is too small to be the next lever.

## Limits

- One seed and one H read. The MPC tracks the log within 0.12 m (p90).
- Attribution windows are 4 s before the event station. The (i) thresholds (1 m lateral, 4 m overshoot, 0.9 arc) are the registered ones.
- The onset read is exploratory.
- Item 2 classes come from logged boxes along the model's plan path, so a lead in the human's lane that the plan does not enter counts as
  adjacent.
- Points are local PAI 281 points, not PCS.
