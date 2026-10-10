# BODY1 stage 2: CARLA rows with our student in the loop, by scene class (design and cost; nothing generated, nothing trained)

2026-10-10. Design document for section 5 of [the prereg](2026-10-10-body1-prereg.md) and Amendment 4 item 9, written after decisions 227
and 229. It is not a pre-registration: the lines proposed in section 4 become binding only when the main session writes them into an
amendment before the first read. One measurement was run for the cost table (section 6, M1); no row, cache or checkpoint was built.

## 0. Cost and recommendation

### Cost per stage (before running)

Rates behind the table are in section 6 (M = measured in this repo, E = estimated). One CARLA "worker" = one server + one route client +
the student's two policy servers. Episode = one short route at one site, about 700 ticks at 20 Hz (E).

| Stage | What | Episodes / ticks | CARLA servers | Card-hours (low - high) | Wall | CPU cores | Disk kept | Basis |
|---|---|--:|--:|--:|--:|--:|--:|---|
| P0 | zero-generation domain pre-check on the existing b2dc cache (section 4) | 0 | 0 | 0.3 (0.2 - 0.5) | 1.5 h | 24 for 15 min | 1 GB | M5, E |
| code | policy path in the collector, segment scheduler, site generator, labels, row builder (section 8) | - | - | 0 | 1.5 - 2 agent-days | - | 0 | E |
| S0 | 8 - 10 cases per class, viewed (BEV + model view) | 24 - 30 / 20 k | 6 (one job) | 0.25 (0.2 - 0.4) | 0.5 h | 21 | 2.5 GB (frames kept) | M1, M3 |
| S1 | pilot, about 1 %, gate | 36 - 40 / 28 k | 6 | 0.6 (0.5 - 0.8) incl. reads | 1 h | 21 | 4 GB (frames kept) | M1, M3, M5 |
| S2a | first tranche, about 10 %, class 2 first; pilot-scale training gate D3 | 360 / 250 k | up to 30 | 3.5 (2.8 - 4.5) incl. 2 pilot trainings and reads | 2 h | up to 105 | 7 GB | M1, M3, E |
| S2b | the rest | 3 240 / 2.27 M | up to 30 | 25 (19.5 - 33.5) incl. tokens | 4.9 h on 5 cards (3.8 - 6.6), 8.2 h on 3 | up to 105 | 47 - 66 GB | M1, M3, M4 |
| total | generation + its reads | 3 600 / 2.5 M | | **30 (23 - 40)** | about 3 days elapsed with the code | | **52 - 73 GB**, peak <= 95 GB | |
| after | the training read if D3 passed: 2 seeds, G3-style offline gate, 700 x 2 closed loop | - | - | about 5 (E, from the loss arm: G3 1.3 M, closed loop 1.7 M) | 4 h | | 2 GB | |

- Box: 6 cards; the pool places CARLA jobs on cards 0, 1, 3, 4, 5 (card 2 is held outside the pool for this lane and takes the token and
  pilot-training jobs). 6 servers per card is the pool's cap and the GPU knee (runbook), so at most 30 workers when the other lanes are idle.
- PIDs: 30 workers x about 300 threads (E: server 154 M + route client 13 - 69 M + two policy servers) = 9 k of the 16 384 plan cap.
- RAM: 5.8 GB RSS per worker measured (M1) + PDM-Lite and the recorder, so `--ram 45` per 6-worker job; the pool spaces the starts
  (decision 222's kill line). Disk: 352 GB free now; the peak leaves >= 255 GB, above the 150 GB floor. Frames are transient (113 KB per
  tick, M3): each job turns its clips into tokens and labels and deletes the frames except a 5 % review sample; kept in full they would be
  285 GB.
- The stage sizes can be halved without changing the design by building class 2 only (1 800 episodes, about 14 card-hours).

### Recommendation

1. **Run P0 first, before any generator code.** It costs 0.3 card-hours and reads two things on rows we already hold: whether P2H10's own
   plan from CARLA frames looks like its plan on real frames, and whether the S0 contact head, trained on navtrain only, reads CARLA road
   edges and objects zero-shot. A boundary AUC under 0.70 ends stage 2 for classes 2 and 3 with nothing built.
2. **"Our student drives the route" does not work in CARLA with the driver as it ships, and the design does not rely on it.** In the cost
   probe (6 junction routes, 17 604 ticks, `P2H10-F-s0`, `spec_plan_smooth`) the car stood still for 66 to 96 % of the ticks of every
   route and covered 29 to 96 m in 75 to 200 s, with its own 1 s plan speed at 0.1 to 0.3 m/s while standing (decisions 128 and 151 saw the
   same on fewer routes). On the AlpaSim nuPlan track the same checkpoint keeps 0.94 progress. So the states this student reaches by itself
   in CARLA are mostly standing states that it does not reach on the board. The loop is therefore **expert-carried with short student
   segments** (section 2): PDM-Lite carries the car to a set point at speed, the student takes over for 2 to 6 s (steering always;
   speed from the student or held by the expert, decided at S0 by yield), PDM-Lite takes back and its realized recovery is the target.
3. **Go only as far as S1 on the current evidence** (P0 + code + S0 + S1: about 1.2 card-hours and 2 agent-days). S2a and S2b are
   released by the gates of sections 4 and 5, not by this document. My prior that the full set raises the AlpaSim score is low to
   moderate: no settled decision shows CARLA-rendered rows improving a real-image board for this student, three show a lesson that stays in
   CARLA or harms (44, 53, 134), and the probe above is a fourth warning. The reason to spend the first 1.2 card-hours anyway: the 15
   corridor zeros and the > 45 deg bucket have no other candidate source of rows (plane reprojection of navtrain does not reach 1 to 4 m
   and 5 to 30 deg), and the gate is cheap and decisive.
4. **Build class 2 (turn / route) first.** It is the only class a navtrain body lesson cannot reach (decision 227 item 5), it is the least
   appearance-bound (command + ego + road layout), and decision 229 says what is missing is a target, which the expert recovery supplies.

## 1. Scene classes to CARLA scenario recipes

Sites come from the collector's generator ([b2dc_routes.py](../../b2d_collect/scripts/b2dc_routes.py), CPU, offline `carla.Map`): after
the positional hold-out from `b2d/bench2drive220` and `bench2drive-0.0.4-val` it has 10 180 candidates (5 380 junction connectors over the
twelve towns, 6 415 Leaderboard 2.0 scenario instances in Town12 / Town13, 146 lane-change segments); b2dc-train@v2 used 1 000 of them.
New code: curve segments without a junction, and two custom static-actor layouts (below). A **site** is one junction connector, one
scenario trigger or one curve segment; an **episode** is one pass of a site with its own weather, actor seed, student seed and set points.

| Class (user's) | Sub-class | Sites (full) | Recipe: where | Actors | Set points of the student segments |
|---|---|--:|---|---|---|
| 1 obstacle ahead | 1a scrape while going around | 200 | LB types `ParkedObstacle(TwoWays)`, `Accident(TwoWays)`, `ConstructionObstacle(TwoWays)`, `VehicleOpensDoorTwoWays`, `HazardAtSideLane(TwoWays)` (cyclists at the edge), `StaticCutIn`, `ParkingCutIn`; plus the custom **parked row** on roads with a `Parking` or `Shoulder` lane in the ten small towns | as the scenario ships; parked row: 3 to 8 static vehicles (random blueprint, 0.2 to 0.6 m from the lane edge, one of them 0.3 to 0.8 m into the lane in half of the episodes) | 25 to 5 m before the first object, and beside it |
| | 1b drive into it | 100 | custom **standing lead at launch** on any straight lane and before junctions; `BlockedIntersection`, `HardBreakRoute`, `ParkingExit` | a vehicle standing 2 to 12 m ahead of the spawn; it leaves after 0 to 6 s in half of the episodes, stays in the other half (then the expert goes around where the map allows, else holds) | tick 20 (launch from the spawn pose, section 2) |
| 2 turn | 2a inside / outside scrape | 150 | junction connectors with an object placed 0.3 to 1.0 m from the swept path at the inside or outside of the exit (custom static vehicle or prop), `InvadingTurn`, `VehicleTurningRoute(Pedestrian)` | as stated | 30 to 5 m before the junction entry, and at the entry |
| | 2b does not make the turn / cuts inside | 300, at least 200 over 45 deg | connectors by cell: turn angle 20 - 45 / 45 - 75 / 75 - 110 deg x minimum radius < 10 / 10 - 20 / > 20 m x left / right x signalised (green on arrival) / unsignalised / T | background traffic of the scenario type (`Vanilla*`, `T_Junction`, `*JunctionLeftTurn`, `*RightTurn`), light forced green on arrival in 2 of 3 episodes | as 2a, plus one in the junction |
| | 2c takes another branch | 150 | junctions with at least two exits, route straight or the gentler exit (the 10 corridor zeros of decision 227 that are not inside cuts) | none or light | 30 to 5 m before the entry |
| 3 leaving the road | 3a curve without a junction | 150 | road segments with a heading change of 15 to 90 deg over 80 m outside junctions (Town07 rural, Town04 / 06 ramps, Town12 / 13 arcs), new generator | none | curve entry and mid-curve |
| | 3b drift on a straight | 150 | straight segments, narrow lanes and kerbs preferred, half with a parked row (overlaps 1a by design: decision 220's class iv) | none or parked row | random |

- **Weather and time of day**: SimLingo's random ranges as b2dc used them (29.8 % night in v2), drawn per episode, so a site's three
  episodes differ; `B2D_KEEP_STREET_LIGHTS=1` (decision 60). Reported as a 3 x 3 table (day / dusk / night x clear / wet / fog) per class.
- **Episodes per site**: 3 (different weather, set points, student seed s0 / s1, perturbation draw). 1 200 sites x 3 = 3 600 episodes.
- **Hold-out**: the collector's rule against the Bench2Drive evaluation routes stays (no shared junction, no trigger within 50 m).
- **How diversity is measured** (reported at S1 and after every tranche, per class; a class that fails is topped up, not padded):
  1. distinct sites and towns; no site holds more than 0.5 % of its class's rows (cap by down-weighting, not by dropping);
  2. the effective number of sites, exp(entropy of the row mass over sites), at least 0.6 x the site count;
  3. occupancy of the cell grid of the class (class 2: angle x radius x side x control = 54 cells, each non-empty cell at least 5 sites);
  4. the state envelope against decision 227's failing decisions: the joint histogram of lateral offset (0 - 0.5 / 0.5 - 1 / 1 - 2 / 2 - 4 m),
     heading error (0 - 2 / 2 - 5 / 5 - 15 / 15 - 30 deg) and speed (< 1 / 1 - 3 / 3 - 8 / > 8 m/s) at hand-over, every cell that holds a
     failing decision of `results/diag/states.csv` covered by at least 100 student segments;
  5. independent units are counted as student segments and sites, never as ticks (about 9 000 segments at 1 200 sites in the full set; for
     scale: navtrain has 12 994 class-2 tokens, 10 773 over 45 deg, 2 313 go-around).

## 2. The loop

**Agent.** One leaderboard agent: the collector [b2dc_agent.py](../../b2d_collect/scripts/b2dc_agent.py) (PDM-Lite as shipped, run every
tick on the true state; the recorder) with its `_drive()` hook implemented. The hook exists and raises `NotImplementedError` today.

**The student, as it ships.** [lib/op_arb_agent.py](../../../lib/op_arb_agent.py), preset `spec_plan_smooth` (lateral = the smoothed
curvature of the model's own plan through openpilot's clip and 0.2 s delay; longitudinal = the harness's openpilot scheduler), zones off
(`"zones": false, "div_m": 1e9`), with `scripts/zeroshot_policy_server.py` and the parity bias server (`PARITY_TAG=P2H10-F-s{0,1}`, ONNX
`runs/op_parity/hugsim/onnx/pp-P2H10-F-s{0,1}.onnx`), exactly the stack of decision 151 and of the probe. Plan tracking (`p7`) is not used:
it loses the car at launch (decision 133). Two of our own input ports must be fixed first, both stated limits of decision 151: the fed
acceleration and lateral velocity are zero in `OpArbAgent.parity_ego` (P2H10's fed acceleration is a score-giving ingredient), and the
command comes from the route desire at 20 m while the labels use `b2dc_labels.route_command`; the agent and the labels must call the same
function. Check: ego features of the agent equal `b2dc_labels.tick_labels` on the same ticks.

**Camera and image path.** The repo's settled rig: `jevdrive.openpilot.interface.B2D_SPEC_MOUNT` (1.59 m ahead of the rear axle, 1.86 m
high, level; decision 127), openpilot road + wide sensors from `zeroshot_rigs.openpilot_sensor_specs(0.05, mount)`, packed by
`b2dc_frames.Packer` (bit-identical to the policy server's packing) and stored lossless. Ego is the Lincoln MKZ of Bench2Drive.

**Command.** NAVSIM one-hot from the route geometry (`route_command`, `Route.from_geometry`); the raw `turn_next` / `turn_dist` are
stored so the lookahead can be re-cut (`b2d_prep.py recmd`). Open: no single lookahead reproduces navtrain's command statistics (precision
/ recall 0.73 / 0.95 on navtrain against 0.56 / 0.81 at 20 m and 0.74 / 0.63 at 10 m on b2dc). Proposed rule, fixed before S0: the
distance-to-turn distribution at command onset measured on navtrain's 11 310 junction-approach frames (decision 93), sampled per episode.

**Episode schedule (expert-carried, student segments).**
1. PDM-Lite drives from the spawn to the first set point (it reaches the site at traffic speed; the student alone does not, see 0).
2. **Student segment**, length U(2, 6) s, in one of three modes, the mix fixed at S0 by yield:
   - F, full student: steering and speed from the student (the state the student itself reaches, including its slowing);
   - L, student steering with the expert's speed (throttle / brake from PDM-Lite): keeps the segment at the board's speeds, where the
     route failures live; it is a hybrid and is tagged as such;
   - I, injected: on top of F or L in half of the segments, a curvature bias pulse of 0.5 to 1.5 s sized for a lateral offset of 0.5 to
     4 m and a heading error of 5 to 30 deg at its end, or a speed scale of 0.7 to 1.3. Injected through the applied control, not by
     teleport, so pose history and frames stay physically continuous (the student reads 1.5 s of pose history and frame pairs).
3. **Cut** of a student segment: its length is reached; box centre more than 4.5 m from the route (just past the scorer's 4 m corridor);
   footprint more than 1.0 m off the wide road raster; heading error over 45 deg; standing for 5 s; or a real contact. In 30 % of the
   segments contact is allowed to happen (the episode ends there; rows up to the contact are kept, the collision event is the label); in
   the others the segment is cut when the privileged sweep of the current motion predicts contact within 0.5 s (a near miss with a
   recovery after it).
4. **Recovery**: PDM-Lite takes the controls for at least 6 s and until the car is within 0.3 m and 3 deg of its path, then the next set
   point. Its realized trajectory is the target of every tick from the hand-over on (E2 below). Episode cap 60 s, 2 to 3 segments.
5. **Launch states** (1b): the only use of a set pose: the hero is placed before the first tick with a lateral offset of U(-1, 1) m and a
   heading offset of U(-12, 12) deg behind the standing object; the history is the static pose, as at a real launch. The student gets the
   controls from tick 20.

**How the target is made (DAgger-style relabel).** Two sources, both from PDM-Lite on the true state:
- **E2, realized**: the recorded 4 s future of the expert-driven ticks (dynamics, reactive traffic, PDM-Lite's own hazard logic and lane
  shifts around obstacles). Exists for every tick of a recovery, the first of which is the state the student reached.
- **E1, forecast**: `AutoPilot.forecast_ego_agent(transform, speed, num_future_frames, target_speed, route_points)`, PDM-Lite's own
  kinematic-bicycle roll-out of its steering and speed controllers along the route. It is a pure query (it saves and restores the
  controller state) and PDM-Lite already calls it every tick with 2.0 s; called with 80 frames every fourth tick it gives a 4 s target
  from every student-driven state. It assumes no hazard (constant target speed, no braking), so it is a **path** target. No other planner
  in the repo or in Bench2Drive can be queried from an arbitrary state: PDM-Lite's control is one step, CARLA cannot fork a world, and
  `carla_rewind` is not equivalent (decision 65).
- Student-driven ticks get E1 only if E1 passes its check; otherwise they stay hinge-only rows, a row kind the lane's trainer already has
  (`lib/loss43.py`).

**Quality checks of the expert (S0 by eye, S1 as numbers, per class).** After a hand-over: contact within 6 s <= 2 %; more than 0.2 m off
the wide raster <= 2 %; back within 0.5 m of its path inside 6 s >= 90 %; progress over the recovery >= 0.85 x the same site driven by the
expert alone; share of recoveries whose speed falls by more than half (the expert's answer is a stop, not a rejoin) reported; E1 against
E2 at hand-over ticks: lateral error at 2 s median <= 0.3 m and p95 <= 1.0 m. PDM-Lite's own record on b2dc: DS 95.6, 33 of 998 clips with
a collision.

## 3. Labels and row format

Rows at 5 Hz (every fourth tick; AlpaSim decides at 2 Hz, navtrain is 2 Hz). Raw per-tick logs as the collector writes them (`ego.npz`,
`actors.npz` with `kinds.json`, `route.npz`, `lights`, `scen.jsonl`), plus `driver` (expert / F / L), the segment id, the pulse and the
low-rate plan log of the policy server.

| Label | How |
|---|---|
| Queries `q` (n, Q, 8, 3) | `own0` / `own1` = P2H10-F-s0 / -s1 forwarded offline on the row's tokens and ego features (as `bd1_plans.py`; the plan the trainer will see), `exp` = the target (E2, else E1), `route` = the route centre line at the student's arc length, then the lane's perturbation families on the own plans (lat, head, gain, arc, stop), Q = 24 as in `bd1_rows.py` |
| Agent contact `a_hit`, `a_t`, `a_s`, `a_cls`, `a_obj`, `a_rear`, `a_t0`, `a_clr`, `a_lat` | [lib/sweep.py](../lib/sweep.py) `agent_labels` on boxes (N, 9, K = 32, 5) built from `actors.npz` at matching times (20 Hz, so no size or pose interpolation between annotations), four classes + static props (cones and barriers are labels here; they are not on navtrain) |
| Boundary `b_hit`, `b_t`, `b_s`, `b_margin`, `b_cov`, `b_t0` | `sweep.boundary_labels` on the 128 x 96 grid of `drivable_hinge.py`, two rasters (below) |
| Corridor (new) `c_lat` (n, Q, 8), `c_t`, `c_branch` | lateral distance of each query pose to the route centre line; first time beyond 4 m (the scorer's `left_corridor_laterally`); which exit of the junction the query takes |
| Target `tgt` (n, 8, 3), `tgt_src`, `tgt_ok` | E2 / E1; `tgt_ok` false inside 5 s before a contact of the expert (b2d_prep's collision rule) |
| Real events | collision sensor and leaderboard infractions with tick; contact truth of the executed motion |
| Tags | class bits and primary (Amendment 1 membership via `contact_head.classes`), sub-class, manoeuvre, hazard, turn angle (signed), minimum radius, > 45 deg flag, speed, `off` (dy, dpsi against the route), segment mode, time since hand-over, site, town, weather cell, student seed |

- **Footprint.** `sweep.py` hard-codes the nuPlan Pacifica (5.18 x 2.30 m); the hero is the MKZ (4.9 x 2.1 m). The sweep gets a footprint
  argument, labels are made with the MKZ (`drivable_hinge.Hinge(..., footprint="mkz")` exists) and the clearance and lateral gap are
  stored, so a Pacifica-margin label can be derived.
- **Road area.** AlpaSim's offroad scorer accepts the lanes and the RoadArea = road segments + intersections + generic drivable areas +
  car parks (Amendment 5 note b). CARLA equivalents from the OpenDRIVE map: lanes of type `Driving` and `Bidirectional`; the filled
  junction polygon for intersections (the union of connector lanes leaves holes inside wide junctions, nuPlan intersections have none);
  `Parking` lanes and `Shoulder` for the generic and car-park surfaces; `Stop`, `Entry`, `Exit`, `OnRamp`, `OffRamp`. Two rasters are
  stored: `strict` (Driving + Bidirectional + junction fill) and `wide` (+ Parking, Shoulder, Stop, ramps); lines are read on `wide`. b2dc
  used Driving + Parking + Bidirectional without the junction fill. Car parks that are not lanes in the OpenDRIVE file have no label
  (b2dc: `ParkingExit` footprints 0.47 inside); rows that start off the raster are flagged, as the lane does (Amendment 4 note x).
- **Format.** One file per (family, shard) with the keys of `rows/<cache dir>.npz` (`names`, `log` = site id, the cluster unit, `hold`,
  `gi`, `off`, `q`, `qok`, `par`, `qfam`, `qbase`, `qfam_names`, `a_*`, `b_*`) plus the new keys above; families `c2f` / `c2l` / `c2x`
  (student F / L, expert recovery) registered in `b1.FAMS`. Tokens: [b2d_prep.py](../../op_parity/scripts/b2d_prep.py) with the source
  directory as an argument: frozen-encoder `view_39` tokens of the pair (t - 4, t) in one `ticks.npy`, `front_idx.npy` for the 8 context
  slots, `tab.npz`, `teacher.npz`, `hinge_labels.npz`. `pp_train.py --host` already reads a navtrain shard and a cache of this layout in
  one batch (`b2d_trainer_check.py`).
- **Counterfactual limit.** Actors reacted to the motion that was executed, not to the queried trajectory. It is milder than navtrain's
  replayed agents and is the same kind of limit.

## 4. The domain question

**What the settled decisions say.**
- CARLA-elicited lessons did not carry to real frames: a reaction head elicited in CARLA is harmful on WOD and NAVSIM (44: RFS -1.02,
  three transfer routes failed), stacked on the best NAVSIM head it costs 5 to 8 PDMS (53), and CARLA pedestrians cannot be read from
  any pooled Cinque feature (AUC 0.51 to 0.53 against 0.83 on nuScenes, 62).
- CARLA rows in a mixed fine-tune were learned in CARLA and not used in the loop: exit pairs gave 0.58 open-loop exits and 4 to 7 of 25
  junctions (128); near, slow turn-entry rows fit open loop, kept the real-row gain and turned 0 of 9 (134, not added).
- The frozen encoder is sensitive to rendering even inside nuPlan: the S0 boundary AUC falls from 0.978 on real hold logs to 0.89 on
  AlpaSim's rendered frames, and the loss is in the input (227 item 2).
- What did carry across domains is carried by ego history, not appearance: the plan follows a fake history yaw with the same sign in real
  data and CARLA (92).
- This document's probe and decisions 128 / 151: the student's own speed plan in CARLA frames is a near stop. Whether that comes from the
  frames or from our ego-input port (zero acceleration, command source) is not known.

**What is different here.** (1) The encoder stays frozen and no appearance head is trained; CARLA rows reach only the plan pathway and
the adapter. (2) The terms are about the student's own plan (hinges, a rejoin path), and the largest target class is a route failure that
is decided by command, ego state and coarse road layout. (3) The lane's own evidence that imperfect frames suffice for own-plan terms:
hinge-only rows on plane-reprojected frames halved own-plan contact on real hold logs (G3, decision 229). **What is not different**: the
plan pathway reads the same frozen tokens, so it can learn a CARLA-only branch of the lesson, which is what 128 and 134 look like.

**Pre-checks, cheapest first (lines proposed here; binding once written into an amendment before the read).**

| Check | Material | Read | Line | Miss |
|---|---|---|---|---|
| P0-D1 student plan in CARLA frames | b2dc cache `b2d_v2` (383 089 rows, never trained on by P2H10; expert-line states), P2H10-F-s0 / -s1 forwarded as `bd1_plans.py`, with the acceleration fed | on moving (> 3 m/s), non-junction rows: own-plan boundary rate (margin < -0.20 m, MKZ) and agent rate; 4 s arc length against the expert's; on junction-turn rows the turn-direction agreement | boundary rate <= 3 x the navtrain on-log rate of the same reader; arc-length ratio >= 0.7; turn direction >= 0.80 (P2-F-s0 measured 0.86) | the student's plan in CARLA is not its real-board plan: fix the input port once and re-read, else stop |
| P0-D2 contact head zero-shot | the same rows, own plans + lat-perturbed queries, truth by `sweep.py` on b2dc's SDF and on boxes built from `actors.npz`; S0 head (two seeds, mean logit), navtrain-trained | boundary AUC and agent AUC, clustered by route | boundary >= 0.80 (hold 0.978, AlpaSim 0.89); agent >= 0.75 (hold 0.905, AlpaSim 0.882; reference with the simulator's current boxes 0.73) | boundary < 0.70: stage 2 ends for classes 2 and 3. Agent < 0.65: class 1 is dropped. In between: go to S1 and read again there |
| S1 the same two reads on the pilot's rows | about 6 k rows at student-reached states | as D1 / D2, per class and for > 45 deg, plus own-plan contact rates of P2H10 against navtrain's (log 0.8 % / 2.1 %, ot1 1.7 % / 4.9 %, yr1 3.6 % / 7.7 % agent / boundary after t0) | as above; the rates within a factor 3 of the navtrain family with the nearest state offsets | as above |
| D3 carry-over to real rows (S2a) | the 10 % tranche mixed into the lane's pilot recipe (2 navtrain shards, 3 000 steps, 1 seed; CARLA rows at the hinge-only share 13 / 128) against the switch-off pilot `P2H10-P-s0`; read on REAL rows only: navtrain hold logs, on-log + `bd4` states at junction approaches | own-plan corridor-exit rate (centre >= 4 m from the logged path inside 4 s; base rate to be measured first, with the lateral deviation at 4 s as the continuous form if there are under 30 positives), > 45 deg separately; dev ADE; decision 205's continuation slope; own-plan 4 s arc length | corridor read falls >= 20 % relative with the log-clustered interval excluding 0; dev ADE <= switch-off + 0.01 m; slope <= switch-off + 0.05; arc length >= 0.98 x switch-off (the progress guard decision 229 lacked) | stage 2 ends; the tranche stays as a probe set |

**What kills stage 2 early**, in order of cost: P0-D2 boundary under 0.70; P0-D1 missed twice; at S0, more than half of the F and L
segments ending in a stand-still or the hand-over envelope not reaching beyond 0.5 m / 5 deg without the pulse (then the rows are
injected states only and "student in the loop" is not what they are); the expert's quality lines of section 2; the S1 reads; D3.

## 5. Staging and splits

| Stage | Size | Viewed / read | Gate |
|---|---|---|---|
| P0 | no generation | D1, D2 on the b2dc cache | section 4 |
| S0 | 8 to 10 episodes per class, one per sub-class at least, three towns | per episode a strip of BEV panels (road raster, actors, route, executed path coloured by driver, the student's plan, the target, first contact) over the model's road and wide input frames at the same ticks (`loss_bev.py` / `b2dc_gif.py`); frames all kept | by eye and by count: hand-over works, labels sit where the picture says, the expert recovers, segment-mode mix chosen; kill lines of section 4 |
| S1 | about 1 %: 36 to 40 episodes, 12 or more per class, both student seeds | diversity counts, the expert's quality lines, D1 / D2 on student-reached states, contact and corridor rates per class, yield (student segments per episode, share cut by each rule), ticks per second, GB per tick against section 6 | all of: expert lines; D1 / D2; at least 1.5 usable student segments per episode; cost within 1.5 x the table |
| S2a | about 10 %, class 2 first (300 of the 360 episodes) | diversity table; D3 | D3 |
| S2b | the rest, staged 1 job -> one card -> all cards (long-runs.md) | per-tranche diversity and quality tables | none; the read is the training arm's (section 7) |

Splits through `jevdrive.data.splits` (`run.use_split` in every script): `carla/s2-sites@v1` (unit route; one member per episode with
its site id), `carla/s2-train` and `carla/s2-hold` **by town**: hold = Town05 and Town13 entire (an urban grid and the Leaderboard
validation town), train = the other ten; inside the train towns a validation part by `sha256(site) % 10 == 1` (the lane's Amendment 5
rule) for any weight selection. A site never crosses parts. The Bench2Drive evaluation routes stay excluded by position.

## 6. Rates behind the cost table

| # | Quantity | Value | Status | Source |
|---|---|---|---|---|
| M1 | student in the loop, 6 workers on one otherwise idle card | 85 to 192 ms per tick per worker, tick-weighted 135 ms; route overhead 12 to 55 s (large maps the high end); 6.8 ticks/s per busy worker including overhead on routes of 1 500 to 4 000 ticks; GPU util 91 %; VRAM peak 51.1 GB (8.5 per worker); RSS peak 34.8 GB (5.8 per worker); 13 cores peak (2.2 per worker); 0 server restarts of 6 | measured 2026-10-10, pool job `1010-080900-acb5` (0.26 card-h), `$DATA_DIR/runs/body1/s2p/` (29 MB) | this document |
| M2 | the same stack, 3 workers on a shared card | 121 to 242 ms per tick | measured 2026-10-07 | decision 151's runs |
| M3 | collector on top of a driver | PDM-Lite 21 ms + packing 7.5 + logging 5 + H.264 0.8 ms per tick; frames 113 KB per tick lossless; 6.3 GB VRAM and 2.3 cores per worker; 902 k ticks in 3.3 h on 3 cards x 6 workers (25 ticks/s per card), 48 server restarts | measured | b2d_collect `results/full.md`, `stages.md` |
| M4 | tokens and labels | 32.9 KB per pair token (12.6 GB / 383 089); 300 to 400 pairs/s per job; SDF 24.5 KB per row uncompressed (9.4 GB / 383 089) | measured | op_parity `results/b2d_cache.md` |
| M5 | plan forward | 203 s for 242 432 states x 2 models | measured | `results/taxonomy.md` section 6 |
| E1 | stage-2 tick | 135 + 21 + 13 + 3 (the 4 s forecast at 5 Hz) = 172 ms in route (120 to 230) | estimated from M1 + M3 | |
| E2 | episode | 700 ticks (b2dc: median 381, mean 904; here capped at 60 s with 2 to 3 segments), 35 s overhead: 155 s, so 4.5 ticks/s per worker (3.3 to 5.8) and 27 per card at 6 workers (20 to 35) | estimated | |
| E3 | servers per card | 6 (pool cap; knee measured for camera agents) at 9 GB each | measured cap, VRAM from M1 | runbook |
| E4 | disk per tick kept | tokens at 5 Hz 8.2 KB + two SDF rasters at 5 Hz 4 to 12 KB (compressed, ratio 3 x assumed, to uncompressed) + logs and rows 2 KB = 14 to 22 KB; 5 % review frames 5.7 KB | M4 + estimate | |
| E5 | cores per worker | 3.5 (student 2.2 M + PDM-Lite and recorder share of 2.3 M) | estimated | |
| E6 | restarts and reruns | 5 % of ticks (b2dc: 48 restarts in 1 000 routes) | estimated | |

Not measured: PDM-Lite and the student in one agent process (E1 adds them); the 4 s forecast's cost; episode length under the segment
schedule; the compression ratio of the SDF; the thread count of a combined worker. S1 replaces every E with a measured value before S2.

## 7. How the rows would be used, and the lines

Ranked candidate uses (each is one arm of the lane's trainer on top of P2H10, every ingredient kept; none is run by this document):

1. **Route / rejoin path imitation at junctions (class 2 with 2c, incl. > 45 deg).** Expert-recovery rows (`c2x`) and student rows with
   a checked E1 as imitation rows whose target is the expert's **path** re-timed to the student's own arc length, so the speed profile is
   untouched by construction (decision 229's open question; decision 221's arc-length arms changed speed on purpose and lost navtest).
   Read: corridor zeros (15 of 49) and the inside cuts over 45 deg; navtest inside-cut % and cannot-make-turn % (2.47 % in the task book, 2.60 % for P2H10-F at G3's checkout).
2. **Targets for hinge-only states (classes 1a, 2a, 3).** The lane's hinge-only rows say "not there" and cost progress (229: slow scenes
   217 -> 277). CARLA student rows carry both hinges and a target that passes the object or rejoins at speed. Read: L3 and L2 at
   unchanged or lower collision + offroad zeros.
3. **Launch rows behind a standing object (1b).** Hold / creep / go-around targets, which navtrain's standing rows do not contain
   (decision 221 item 5: they are launches by selection). Read: the 4 launch collisions of 14, and PAI if the nuPlan lines pass. Lowest:
   4 zeros in 2 scenes, the low-speed cell is partly covered by `bd4`, and it touches the speed prior that decides L3.

Lines (unchanged from Amendment 4 item 6, the stricter of readings A and B deciding; a regression check): L1a collision + offroad +
corridor zeros down in the two-seed total and in neither seed up; L1b collision + offroad not up; for use 1 additionally **L1c: corridor
zeros down in the two-seed total and in neither seed up**; L2 mean difference >= 0 with the log-clustered lower bound > -0.005; L3 slow
scenes <= 1.1 x base; the > 45 deg bucket separately at every read; navtest >= base - 0.3 through `jevdrive.bench`; `dev_drift_off` <=
0.30. Offline gate before any closed loop: D3's reads at full scale on both seeds.

**What stays unreachable.** The 4 collision zeros whose object is outside the camera; the 7 offroad zeros where the NAVSIM raster shows a
margin (cause undetermined, decision 229); the rendered-frame loss of the boundary signal on AlpaSim (0.978 -> 0.89), which CARLA frames do
not address; the board's progress-against-the-log scoring (the trade of decisions 221, 224, 226, 229); the PAI track. And by
construction nothing here teaches the student to keep moving in CARLA.

## 8. Open risks and what could not be found out

1. **The student's standing in CARLA** (section 0). Cause unknown (frames, or the ego-input port with zero acceleration). If the fixed
   port does not change it, mode F yields standing rows and the set is modes L and I: the student's steering at the expert's speed. That is
   weaker than the task book's "the states the student itself reaches", and P0-D1 measures the same thing open loop first.
2. **Executor mismatch.** On AlpaSim a nonlinear MPC tracks the 4 s plan; in CARLA the plan's smoothed curvature is steered at 20 Hz.
   The states reached differ for the same plan. The injected pulses are sized from decision 227's envelope for that reason.
3. **The expert as a target.** PDM-Lite brakes on inflated forecast boxes; its recoveries may be stops, and its speed may be below the
   nuPlan log drivers'. Use 1 takes only its path; uses 2 and 3 depend on the share of stop recoveries measured at S1. From 4 m and 30 deg
   off its route PDM-Lite's steering PID is outside the range it was tuned in; not tested by anyone.
4. **E1 at 4 s.** PDM-Lite uses its forecast for 2.0 s; whether the bicycle roll-out stays close to what the car then does over 4 s
   from off-route states is unknown until S0 compares E1 with E2.
5. **Command.** No lookahead reproduces navtrain's command statistics; a mis-timed command would itself teach a late or early turn.
6. **Vehicle and view.** MKZ against Pacifica (0.3 m shorter, 0.2 m narrower, another rear-axle offset); the 1.86 m view is the open-loop
   boards' and is not AlpaSim's camera. Decision 137: training at that view cost native-camera ADE; the encoder is frozen here and the
   anchor rows stay, the `dev_drift_off` guard is read.
7. **CARLA object readability** (decision 62 for pedestrians; vehicles and static props not measured). P0-D2's agent half reads it.
8. **Engineering.** PDM-Lite runs in `envs/simlingo`, the op_arb agent in `envs/carla` (Python 3.8) with the model behind sockets; the
   merged agent has to drive the policy servers from the collector's process. The cost table assumes that works without a second sensor
   set. Custom static actors inside a leaderboard route (parked rows, standing lead) are new code against the scenario runner.
9. **Closed loop is not reproducible bit for bit** and RenderThread deaths are retried (runbook); a tranche is complete when its site
   list is, read from `summary.json`.
10. Not found out: the base rate of own-plan corridor exits on navtrain hold logs (D3's read); how many junction connectors fall in each
    class-2 cell after the hold-out (the 5 380 are not broken down by angle and radius in `summary.json`); whether car parks exist as
    surfaces outside the lane graph in the small towns; the traffic-light share of the standing time in the probe (on two of the six routes the logged light state
    is 0 for most of the standing ticks, red if the log uses CARLA's coding; on the other four it is not).

## 9. Probe record

`$DATA_DIR/runs/body1/s2p/` on the box: `submit_probe.py` (one pool job, owner `body1`, `cllib.b2d_unit` with 6 routes of decision
127's set: 10255, 5423, 28008, 15102, 28147, 334; `--ram 70`, 18 cores, 6 CARLA servers), `arm/` (b2d_run outputs, `done/<route>.json`
profiles). Wall 922 s, 0.26 card-hours, 29 MB. Per route: ticks 1 588 / 4 000 / 4 000 / 1 505 / 2 511 / 4 000; ms per tick 129 / 192 /
125 / 162 / 131 / 85; share of ticks below 0.1 m/s 0.85 / 0.96 / 0.83 / 0.66 / 0.76 / 0.94; distance 30 / 29 / 96 / 61 / 67 / 33 m. It
is a cost probe: one seed, no baseline arm, nothing scored.
