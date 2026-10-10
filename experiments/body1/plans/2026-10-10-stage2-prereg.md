# BODY1 stage 2, P0: pre-registration of the zero-generation pre-check

2026-10-10. Binds stage P0 of [the stage 2 design](2026-10-10-stage2-carla-design.md) (section 4, rows P0-D1 and P0-D2) and one extra
probe, P0-E. Pushed before any P0 number exists. Lines here are the design's proposals, tightened where stated; they are never loosened
afterwards. No row is generated and nothing is trained: the student (P2H10-F-s0 / -s1), the S0 contact head (two full-scale seeds,
`runs/body1/s0/full-step/20261010-024906`, `-024909`) and the blind head (`full-blind/20261010-024909`) are frozen, and no threshold,
weight or checkpoint is chosen on the rows read here.

## 1. Material

- **Cache read**: `$DATA_DIR/runs/op_parity/cache/b2d_v2/` (meta version `b2d1`, 383 089 rows of 998 clips, built 2026-10-07 from our
  PDM-Lite collection b2dc-train@v2, decision 152): `ticks.npy` + `front_idx.npy` (frozen `view_39` tokens of the 8 slots), `tab.npz` (the
  20-dim ego vector with the recorded body-frame velocity and acceleration and the route command at the collected 30 m lookahead, logged
  4 s future, speed), `hinge_labels.npz` (drivable SDF, 128 x 96, Driving + Parking + Bidirectional lanes, no junction fill), `extra.npz`
  (route, town, scenario type, tick).
- **What the cache does not hold and is recomputed**: agent boxes. They are built from the collector's raw per-tick logs
  (`runs/b2d_collect/data/all/attempts/<route>/<n>/clip/actors.npz` + `kinds.json`, 20 Hz, so boxes at t0 + 0, 0.5 .. 4 s are read at
  matching ticks without interpolation between annotations) into the layout of the navtrain agent labels: (N, 9, K = 32, 5) nearest
  first, in the row's rear-axle frame. Kinds are mapped to the classes of `runs/body1/labels/classes.txt`; kinds without a navtrain
  counterpart are dropped from the truth and counted in the report.
- **Subsample, fixed here**: rows with `tick % 10 == 0` (2 Hz, navtrain's rate), all 998 clips, both of the cache's split parts (the
  split is irrelevant: nothing is fitted). If the forward does not fit the budget, clips are dropped by `sha256(route) % 2 == 1`, not rows.
- **Why reading this cache is not building on it** (the task book forbids training on b2dc-train@v2): the rows are a test set only. No
  gradient step, no model selection, no threshold and no row of a later training set comes from them; their states are all on the
  expert's line, which is exactly why they cannot be the stage 2 rows and can only answer "does the frozen stack read CARLA frames".
- **Navtrain reference, same reader**: on-log states of `navsim/body1-hold-logs` (121 logs), plans from `runs/body1/plans/`, labels from
  `runs/body1/rows/`.

Definitions used below. Speed = fed ego speed at t0. Moving = speed > 3 m/s; standing = speed < 0.5 m/s. Expert arc = 4 s arc length of
the logged future (PDM-Lite in CARLA, the human log on navtrain). Own arc / heading = the same on the student's plan (`bd1_plans.py`
conversion, 8 rear-axle poses). Turn row = moving and |logged 4 s heading change| > 20 deg; > 45 deg bucket = the same above 45 deg;
straight row = moving and below 5 deg. Both student seeds are pooled (rows x 2) unless stated. Intervals: percentile cluster bootstrap by
route (CARLA) or log (navtrain), 2 000 resamples, seed 0; AUC intervals by `contact_head.auc_boot`. A second interval clustered by town
(12 clusters) is reported for the two D2 AUCs.

## 2. P0-D1: is the student's plan on CARLA frames like its plan on real frames

Read once with the cache's ego vector as it is (acceleration fed). Four statistics, each with the navtrain value from the same code:

| # | Statistic | Line (all four must hold for "like") |
|---|---|---|
| a | arc ratio on moving rows with expert arc >= 4 m: median of own arc / expert arc; and the collapse share = rows with own arc < 0.5 x expert arc | median >= 0.80 (design: 0.70) and >= 0.85 x the navtrain median; collapse share <= navtrain's + 10 points |
| b | launch: standing rows whose expert arc is >= 8 m (the expert is about to go): share with own arc >= 0.5 x expert arc; and on moving rows the median of (own displacement in the first 1 s) / speed | launch share >= 0.6 x navtrain's; 1 s ratio >= 0.85 and within 0.10 of navtrain's |
| c | turn direction on turn rows: sign of the own 4 s heading change equals the logged sign; the same in the > 45 deg bucket; median of own / logged heading change reported | >= 0.80 on turn rows, >= 0.75 in the > 45 deg bucket |
| d | own-plan boundary contact rate on straight rows (margin < -0.20 m after t = 0, MKZ footprint in CARLA, Pacifica on navtrain; rows already outside at t = 0 excluded) ; agent contact rate reported | <= 3 x the navtrain on-log rate of the same reader |

"Unlike" in numbers = any line missed. Reported per statistic by class proxy (straight / turn 20-45 / > 45 deg / standing; scenario type
groups of `extra.npz` for obstacle scenarios) and by town. A miss of (c) alone may be re-read once on `b2d_v2L20` (the same rows with the
command re-cut at 20 m), disclosed as a second read; no other re-read.

## 3. P0-D2: the S0 contact head, zero-shot on CARLA rows

- Score: mean of the two seeds' logits (as G1 / G2), queries own0 / own1 (the student's two plans) pooled. Secondary queries, reported
  without a line: lateral ramps of own0 at +-0.5, +-1.0, +-1.5 m (linear in time, heading unchanged).
- Truth by `lib/sweep.py`: boundary positive = margin < -0.20 m with first contact after t = 0; rows in [-0.20, 0), outside at t = 0 or
  without raster coverage carry no label (`contact_head.targets`). Agent positive = `a_hit` (rear-end by a faster object is not a contact).
- Footprint: the head was trained on Pacifica labels, the CARLA hero is the MKZ. Both truths are computed (`sweep.py` constants swapped
  for the MKZ: front 3.829, rear -1.064, half width 0.918). **The lower of the two AUCs is the one compared with every line.**
- Primary subset: rows with speed >= 0.5 m/s (48 % of the cache stands still and is trivially negative). All rows, moving rows, turn
  rows and the > 45 deg bucket are reported separately; per town; per weather where the collector logged it.
- If the own-plan positives of a contact type are fewer than 100 or sit in fewer than 20 routes, that line is read on own + lateral-ramp
  queries pooled and flagged as such.

| Read | Pass | Weak | Fail |
|---|---|---|---|
| boundary AUC | >= 0.80 **and** route-clustered lower bound >= 0.75 **and** S0 - blind head >= 0.05 on the same rows (paired cluster-bootstrap lower bound > 0) | anything between | < 0.70 |
| agent AUC | >= 0.75 **and** lower bound >= 0.70 | anything between | < 0.65 |
| > 45 deg bucket, boundary and agent | reported; boundary < 0.70 there is flagged as "over-45 rows unreadable" and carried into the class 2 decision below | | |

The two extra conditions on a pass (lower bound, margin over the blind head) are tightenings of the design's point lines: a pass that
the blind head reaches too is a read of ego state and plan shape, not of the frame, and counts as weak.

## 4. P0-E: the standing student and the two input ports

A probe of the loop, not a benchmark: no driving score, route completion or infraction count is reported from it.

- **Fix 1 (fed acceleration, lateral velocity)**: `OpArbAgent.parity_ego` feeds vy = ay = 0 and ax from a 0.5 s speed difference. Behind a
  default-off switch it feeds the body-frame velocity and acceleration of the hero the way `b2dc_labels.tick_labels` computes them from
  the simulator state (the navtrain convention: body frame at the rear axle, x forward, y left).
- **Fix 2 (command)**: behind a second default-off switch the command is `b2dc_labels.route_command(Route.from_geometry(...), s, 20.0)`
  on the agent's own route, the function the labels use, at a 20 m lookahead (the distance the desire rule uses today, so the change is
  the function, not the distance). No lookahead sweep.
- With both switches off every existing run is bit-identical (the ego vector of a recorded tick stream is compared before / after).
- **Correctness, on recorded values, before any CARLA job**: (i) the recorded ego stream of >= 20 b2dc clips (>= 3 towns, >= 5 with a
  junction turn) replayed through the fixed `parity_ego`: all 20 ego dims within 1e-3 of the cache's `tab.ego` at the same ticks except
  the command at the three one-hot dims, which is compared with `b2d_v2L20` (agreement >= 0.99 of ticks; every disagreement explained);
  (ii) convention against navtrain: sign and scale of fed ax, ay, vy on moving rows: regression slope of ay on vx x yaw rate in
  [0.8, 1.2] on both navtrain and the replayed stream, ax quantiles (p5, p50, p95) of the two printed side by side.
- **Arms** on the six routes of the design's probe (10255, 5423, 28008, 15102, 28147, 334), `P2H10-F-s0`, `spec_plan_smooth`, zones
  off, one seed, one pool job of 6 CARLA workers each: A0 as shipped, A1 both fixes. Both log the hero's traffic-light state and the
  nearest actor ahead every tick (a logging-only addition, default off). A2 (fix 1 only) and A3 (fix 2 only) are run only if A1 passes
  or lowers the unexcused standing share by >= 10 points, and only inside the budget.
- **Reads per route and pooled**: share of ticks with speed < 0.1 m/s; metres driven; the student's own-plan speed at 1 s while
  standing; the share of standing ticks that are *excused*: the hero's light is red or yellow, or an actor box lies within 12 m ahead of
  the front bumper inside +-1.75 m of the heading line.
- **Line**: after the fix the student moves = unexcused standing ticks / all ticks < 20 % pooled over the six routes **and** < 30 % on
  at least five of the six. Otherwise the design's fallback holds.
- Limits known in advance: six routes, one seed, closed loop is not bit-reproducible (the A0 rerun gives the spread against the
  design's own probe of the same arm); "excused" is a geometric rule.

## 5. What each outcome means for stage 2

| Outcome | Stage 2 |
|---|---|
| D2 boundary fail (< 0.70) | **Stop** for classes 2 and 3; nothing is built. Class 1 alone is not started by this lane agent; its agent AUC is reported |
| D1 lines (c) or (d) missed (after the one allowed re-read of c) | **Stop**: the student's lateral plan on CARLA frames is not its real-board plan, so rows of its steering are not rows of the board's student |
| D2 boundary pass, D1 (c), (d) hold, P0-E line met | **Go** to code + S0 + S1 of the design with mode F included |
| D2 boundary pass, D1 (c), (d) hold, P0-E line missed or D1 (a) / (b) missed | **Go** to code + S0 + S1 with the fallback: expert-carried, modes L and I only (the student's steering at the expert's speed); "the states the student itself reaches" is then not what the rows are, and the report says so |
| D2 boundary weak | **Class 2 only** to S1, where D2 is read again on student-reached states and must reach 0.80 or stage 2 ends |
| D2 agent fail (< 0.65) | class 1 dropped; agent weak: class 1 is not built before the S1 re-read |

Nothing beyond S1 is released by P0 in any outcome. The report gives an updated cost line if the measured tick rates differ from the
design's M1.

## 6. Budget and conduct

1.5 card-hours, about 4 h wall, 3 GB on the box. D1 / D2: one GPU job through the pool. P0-E: at most 6 CARLA workers at once, pool
only, `--ram` declared. Code outside `experiments/body1/` is touched only by default-off switches (`lib/op_arb_agent.py`), listed in the
report. Deviations from this file are listed in the report with their reason.
