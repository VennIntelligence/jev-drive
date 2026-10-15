# Pre-registration: RULE EXPERT, a map-based rule planner driven in closed loop as a teacher check (no training), 2026-10-15

**Why.** research/next-round/turn-night.md sections 4-6: every hinge on the policy's own states failed (d296 / d298 / d301 / d303; the
corridor labels of d305 fail their own human gate). The next line is DAgger with a rule expert that gives a complete, consistent target
trajectory from any state. Step 1 (this file): can the expert itself drive the turns, in the same AlpaSim loop, controller and scoring as
the model? Inputs that shape the design: d293 (human entry speed: median 4.6 m/s at the entry, deceleration from about 26 m before it),
d296 / d298 (a log handover to the entry completes 11 / 26 exam turns; the policy's own approach fails), d287 (the route message starts
40 m ahead; the served route as a path ties RSG), d305 (route-lane chains from the map: rf_corridor.py resolves them for nuPlan and PAI;
humans cut corners, so the chain is a path to track, not a hard boundary), d289 (a human driving by hand scored 0.275 on PAI).
Written and committed before any expert rollout. Amendments are appended, never edited in place.

## Expert (`experiments/alpasim/lib/rule_expert.py`, numpy only; packs from `scripts/rf_expert.py pack`)

PRIVILEGED: reads the map, the human log's speeds and the replayed agents by scene id (`debug_info.scene_id`, local runs only). A teacher /
probe, never a submission. Served through the existing drivers (`JEV_EXPERT=<pack dir>`): sh30_driver.py (nuPlan held) and pai_driver.py
(PAI). The checkpoint RSG-L18-s0 still runs in the driver so every drive.jsonl field the readers use is there (`ego`, `rmem`, `cmd`); its plan
is logged as `poses_model`; `poses` is the expert's served plan; `expert` logs the expert's state (station, lateral offset, speed target,
stop point, contacts, next turn). Every other serving switch is off (asserted: JEV_VCONT, JEV_LEAD, JEV_TRACK, JEV_ROUTE, JEV_TURN_GAIN).
Runtime, controller (official MPC; PAI with `controller.gains.rel_front_steering_angle_weight=20` as the reference reads), route messages
and scoring are unchanged.

**Pack (per scene, offline, the scene loader of the run's own config).**
1. Route = AlpaSim's own MAP route (`RouteGenerator.create`, 40 m) generated along the log's 0.5 s poses (the route the run serves when the
   car follows the log), with the log before the route start prefixed (rf_corridor's way; this is the "lane the ego is on, continued along
   the route's direction" for the first 40 m where the route message is absent).
2. Chain = `rf_corridor.SceneMap.match` on that polyline (d305's rules: candidates within 1 m and 60 deg, score heading + 0.5 x polygon
   distance, A-B-A flicker removed, one missing connector filled from the successor graph, transitions successor / change / break);
   ambiguous candidates (two distinct overlapping lanes, frequent on PAI) resolved toward the **route points**.
3. Path = the pieces' centre-line samples in route-station order, joined at the midpoint of consecutive station ranges; a lane change is
   blended (smoothstep) over [mid - 15, mid + 15] m inside both pieces' overlap; resampled at 0.5 m; 30 m straight before the start and
   150 m after the end. No piece -> the route polyline itself (`fallback`, counted).
4. Turns on the path: runs of stations a with |h(a + 30 m) - h(a)| >= 45 deg (h = chord heading over +-5 m), entry / exit where 10 % /
   90 % of the change over [a0, a1 + 30] is reached.
5. Human speed by station: the log's speed when it first passed the station (5-sample box, then +-2.5 m), NaN where it never passed.
6. Replayed agents: `ds.traffic_objects` (what the simulator replays, physics NONE), within 150 m of the log path.

**Per decision** (rear-axle anchor pose and measured speed of the driver's decision; nuPlan 2 Hz, PAI 10 Hz):
- Speed target by station: cruise = max(human speed, 3 m/s) where known, else 12 m/s (the trajdata map carries no speed limit; the
  12 m/s cap of the brief). From 26 m before every turn entry, linear in distance down to the entry speed V_E (`--expert-entry-v`,
  default 4.6 m/s = d293's all-turn median; the class medians need topology labels that only the 36 strict PAI turns have), V_E through the
  turn, then up at 1.5 m/s^2 after the exit; never above the cruise target; capped by v^2 |kappa| <= 4.0 m/s^2 on the path's curvature
  (d293: human a_lat median 2.4, p95 4.4); a braking envelope at 3 m/s^2 makes the profile reachable (the car may brake up to 6 m/s^2 when
  above it).
- Yield: the plan's ego box (the rig box, sides + 0.25 m) against every replayed box whose centre is ahead of the ego box centre, at
  0.1 s steps over 3 s, using the logged boxes at the same times; at the first contact the target becomes 0 from 1.5 m before the
  contact pose (up to 3 passes for later contacts). Boxes already touching the ego at t0 are ignored (nothing to yield to).
- Stop lines and traffic lights: **ignored** for this check.
- Path: pure pursuit from the rear axle on the path, lookahead max(6 m, LA x v) (`--expert-lookahead` LA, default 1.2 s), curvature
  within 0.2 1/m, kinematic bicycle at 0.1 s for 4 s; out = the 8 rear-axle poses at 0.5 .. 4.0 s in the anchor frame (the model's plan
  format; heading = the integrated course, so the poses and their yaw are consistent).

Default variant: LA 1.2 s, V_E 4.6 m/s, label `XP-L12-E46`. Every other constant above is fixed for this check.

## Reads

| read | where | scenes | serving | reference |
|:--|:--|:--|:--|:--|
| held | box, pool, 6 chunk jobs (`rf_expert.sh box-read` = `rm_chain.sh read RSG-L18-s0 i 6 <label>`) | held 460 / 18 decisions (452 scored) | sh30 driver + JEV_EXPERT (pack_held) | stored `route_mem/read/RSG-L18-s{0,1}` |
| PAI | Tokyo, `pai_eval.sh` with the reference chunk lists, CONC 4, one stack per card (`rf_expert.sh tokyo-run`) | 281 | pai driver + JEV_EXPERT (pack_pai), gains 20 | stored `/data/runs/route_night/r281/rsg` (RSG-L18-s0, gains 20, VCONT 1.0, LEAD 1) |

Readers unchanged: held.py, ol_confirm.py (paired vs RSG-L18-s0 and -s1), rf_follow.py (turn completion), rf_appheld.py (entry state);
rf_pai281.py (all 281 / exam 72 / control 75 vs rsg), rf_pai_follow extract + rf_v2pai.py turns / table (strict 36, d293 shape gate,
spins), rf_appread.py (the 22 runway turns, results/approach/pai_runway22.txt).

## Statistics and lines

- **Held turns**: the 24 turn scenes of RSG-L18-s0's rf_follow read (`report/follow_a3`, RSG-L18-s0 16 / 24, RSG-L18-s1 14 / 24);
  the expert's outcome is its own rf_follow row on the same scene (a scene where the expert's row is not a turn counts as not completed and
  is listed). **Line E1: completed >= 20 / 24 (>= 80 %).** Reported with it: zeros, at-fault zeros, mean score on the held 452 (ol_confirm,
  paired vs RSG-L18-s0 / -s1).
- **PAI strict turns**: the 36 of `results/pai281/strict36.txt` (RSG 5 / 36, RV2b 7 / 36), outcome by rf_v2pai turns on the expert's own
  route. **Line E2: completed >= 29 / 36 (>= 80 %).** Reported with it: the 22 runway turns (RSG 3), exam-72 zeros, the d293 shape gate,
  spins, all-281 mean.
- **E3 (sanity)**: the expert's mean score on the PAI control-75 scenes and on the held straight scenes (held scenes whose expert path has
  no turn before the log's end station) is not below RSG-L18-s0's on the same scenes by more than 0.05 (the expert may drive slower;
  progress saturates at 0.8, d285).
- **Verdict rule**: E1 and E2 both pass -> the expert qualifies as a teacher. One passes -> report which turns fail and why and stop.
  Neither -> the line stops. The per-turn failure list (both sets) gives each failed turn one class, in this order: **wrong lane chain**
  (the pack's chain check: no chain turn of the route turn's sign within 15 m of its entry, or the route lies > 2.0 m from the chain
  somewhere in [entry - 20, exit + 10] m) / **collision** (zero event = collision) / **corridor** (zero event = offroad or
  left_corridor_laterally, or outcome `off`) / **speed** (outcome `short`: no zero, the turn exit + 10 m not reached in time); with the
  expert's yield stops (decisions with a contact in the turn window) noted.

## Order, smoke and cost

1. Packs: held 460 on the box (pool CPU job), PAI 281 on Tokyo (alpasim-base, CPU). Pack coverage (fallbacks, breaks, ambiguous samples,
   chain-check failures) is reported before any rollout statistic.
2. **Smoke**: 3 held turn scenes (`results/rule_expert/smoke_held.txt`: e14711ef RSG event, 77155a60 R, e9cfc98f L short) and 3 PAI strict
   turns (`smoke_pai.txt`: 22fa5c92 L, 3232fdb7 R, 9f393c8a R; all RSG events). BEV figure of the expert's chain vs the human path vs
   the driven paths (`figs/rule_expert_smoke.png`). Obvious chain or tracking errors are fixed and recorded as **Amendment 1** before the
   full reads; the smoke scenes stay in the full sets and are re-driven there.
3. Full reads with the Amendment-1 variant. Cost: held 6 chunk jobs, about 10 min each (APP1X's chunk 0: 598 s), about 1.0 card-h, about
   15 min wall on the 3 cards; PAI 281 on Tokyo about 2.3 h on two cards (RV2b's read; the expert adds about 2 ms per 10 Hz call).

## Caveats (stated in advance)

- Privileged and log-informed: the human's speeds set the cruise target and the replayed agents are known 3 s ahead exactly (the
  simulator replays them). A pass says the rule layer can drive the turns given that knowledge, not that a camera policy can.
- Stop lines / red lights are ignored; a red-light run with crossing traffic is caught only by the yield rule.
- Held rollouts are 18 decisions (9 s): an expert slower than the human through a turn can end `short` (the human finished the turn by
  construction of the turn set). This is counted as a speed failure, not excused.
- One run per set (the simulator is deterministic per chunk list; d215's cm-level numerics remain).
