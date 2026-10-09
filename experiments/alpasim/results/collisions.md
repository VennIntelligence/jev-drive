# COL1: at-fault collisions of the openpilot-based driver in AlpaSim (2026-10-09)

Research on the collisions, not a fix: what is hit and why on the 1491 public nuPlan scenes (GPU box) and on 40 PAI scenes (Tokyo box),
what part of openpilot's longitudinal safety path is in our loop, and whether the frozen model's own lead outputs saw the object.
Pre-registration (lead-output read, with two amendments for a PAI diagnostic arm): [plans/2026-10-09-col1-lead-prereg.md](../plans/2026-10-09-col1-lead-prereg.md).
Tables: [collisions/](collisions/) (`nuplan_tables.md`, `nuplan_cases.csv`, `nuplan_lead.md`, `nuplan_lead.csv`, `nuplan_clip_cases.csv`,
`pai_tables.md`, `pai_cases.csv`, `pai_plumbing.md`, `pai_arm_v1.md`, `pai_clip_cases.csv`). Clips: `../figs/collisions/`.
Code: `scripts/col1_nuplan.py`, `col1_lead.py`, `col1_clip.py`, `col1_ctrl.sh` (nuPlan); `col1_pai_extract.py`, `col1_pai_replay.py`, `col1_pai.py`,
`col1_pai_plumb.py`, `col1_pai_arm.py`, `col1_pai_clip.py`, `col1_pai_chain.sh`, `col1_pai_chain2.sh`, `col1_lib.py`, `lib/col1_pai_driver.py` (PAI).
Nothing was trained, registered or submitted. Simulator state (object boxes, map, logged path) is a label and an oracle only. WA-JEPA and the
organisers' PAI reference models appear as reference rows only. Decision 219.

## 1. Result in short

1. **None of openpilot's longitudinal safety path is in our loop.** The model computes its full output vector at every decision; the
   driver reads the plan's positions and yaw and nothing else. `lead` / `lead_prob` are computed and discarded; there is no longitudinal
   planner, no follow distance, no stop-and-go logic (section 2).
2. **nuPlan public scenes: the collisions are lateral.** P2H10-F has 22 at-fault collision rollouts in 12 scenes (12 + 10 per seed of
   1491). 16 are at least 0.5 m off the logged path at impact, 20 would be clear on the logged path at the driven speed, median speed
   2.4 m/s. A stopped lead is 4 of 22 (2 scenes, launches from rest). WA-JEPA (reference) passes 17 of the 22 (section 3).
3. **PAI: the collisions are longitudinal.** 40 scenes, mean scene score 0.1694, 30 zeros, 12 at-fault collisions: 11 front contacts, the
   ego more than 2 m ahead of the log in 9, median speed 6.7 m/s; the struck object is straight ahead of the ego for at least 2 s in 9
   (sections 4, 5).
4. **The frozen model's lead head saw the vehicle ahead in time on PAI.** All 3 class-A rollouts are reported 6.2-7.7 s before impact;
   by the ego's own corridor the head is on the struck vehicle for 74-92 % of the approach in 5 of 12. The plan does not slow: the served
   plan runs through the object in 10 of 12, the shipped policy's plan in 7. Registered line L2 (guard ceiling): passes on PAI (7 / 12
   turned, 0 / 21 false-alarm scenes), fails on nuPlan (6 / 22, 14 % false-alarm scenes). L1 is descriptive in both (C_A of 4 and 3)
   (section 6).
5. **One serving-side defect on PAI.** The returned trajectory's first segment has the plan's mean speed, not the ego's. Above 23 m/s
   it is 5-11 % short in 4 of 5 scenes and all 4 spin at the hand-over. The diagnostic arm that starts the trajectory at the ego's speed:
   0.1694 -> 0.2921 on the 40 scenes (+0.1227 [-0.0055, +0.2531]), zeros 30 -> 25, 2 of the 4 spins gone, collisions 12 -> 8 (section 4.3).
6. **The command split is odd and harmless**; speed, acceleration, latency, slots and camera geometry check out (sections 4.1, 4.2).
7. Cheapest in-loop use of the lead outputs: a lead-conditioned deceleration cap on the plan, from outputs already computed. Ceiling
   from the logs: at most 7 of 12 PAI collision zeros at no false alarm in 21 control scenes; a net loss on nuPlan (section 7).

## 2. What of openpilot's longitudinal path is in the loop

Read from `lib/sh30_core.py`, `lib/sh30_driver.py`, `lib/pai_core.py`, `lib/pai_driver.py` and `experiments/op_parity/scripts/pp_train.py` (`PModel`).

| Piece of shipped openpilot | In our AlpaSim driver |
|---|---|
| Vision encoder (frozen Cinque) | yes, unchanged: 8 policy slots -> (8, 32, 512) tokens |
| Policy network | yes, with the trained plan pathway and the adapter bias added to the tokens (`PModel.forward`) |
| Output vector (18 452 values): `plan` 990, `lead` 144, `lead_prob` 3, `meta` 55, `action` 4, `desire_state` 8, lane lines, road edges, pose ... | computed in full at every decision |
| What the driver reads from it | `out[0, self.pi]`: the 33 x 15 plan means only; of those, position (columns 0-2) and yaw (column 11) |
| Plan velocity / acceleration columns | not read |
| `lead` / `lead_prob` (the vision lead that openpilot's radard turns into lead tracks) | not read: computed and discarded |
| `action` (the model's own desired acceleration / curvature), `meta` (brake / disengage predictions) | not read |
| Longitudinal planner (`LongitudinalMpc`: cruise and lead constraints, follow distance, stop-and-go), FCW, desired-speed logic | absent |
| Lateral control | absent; the simulator's MPC tracks the returned positions |
| What goes to the simulator | 8 rear-axle poses at 0.5 s, linearly resampled to a 10 Hz position trajectory (the shipped LTF sample's `build_trajectory_from_plan`) |

So nothing of openpilot's lead-based longitudinal path is in the loop. The only longitudinal signal is the spacing of the plan's
positions, produced by a plan pathway that was fine-tuned on NAVSIM imitation with an ego-state adapter, and tracked as positions by the
simulator's controller. Shipped openpilot in its default mode takes speed from the lead-aware longitudinal planner, not from the plan.

## 3. nuPlan public scenes (1491 scenes, 44 logs)

Inventory (all at-fault-collision rollouts of the six runs still have their `rollout.asl`; every case is an original rollout):

| driver | scenes | zeros | at-fault collision rollouts |
|:--|--:|--:|--:|
| **P2H10-F-s0** | 1491 | 89 | **12** |
| **P2H10-F-s1** | 1491 | 85 | **10** |
| APY10m10-AB-s0 / s1 | 1491 | 83 / 89 | 10 / 14 |
| AP2H10-AB-s0 / s1 | 700 (the 791 later scenes were still running in lane CF1 and were not read) | 28 / 31 | 7 / 9 |
| WA-JEPA (reference) | 1491 | 146 | 5 |
| SH30-F-s0, AP2-AB-s0, OT30-F-s0 / s1, YR10m10-F-s0 / s1 | 1491 | 139, 132, 125 / 106, 80 / 77 | 32, 32, 19 / 19, 7 / 7 |

P2H10-F: 22 collision rollouts in 12 distinct scenes of 9 logs; 10 of the 12 scenes collide in both seeds, so this is 12 situations, not 22.

| struck object | P2H10-F rollouts (scenes) | share | off the logged path >= 0.5 m | clear on the logged path at the driven speed | WA-JEPA (reference) passes | median ego m/s | APY10m10-AB (n = 24) | AP2H10-AB (n = 16, 700 scenes) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| lead stopped (in lane, ahead) | 4 (2) | 18% | 0 | 4 | 4 / 4 | 2.0 | 7 | 4 |
| slow lead | 0 | 0% | | | | | 3 | 3 |
| cut-in (a faster vehicle from the next lane or from behind) | 4 (2) | 18% | 2 | 4 | 4 / 4 | 3.8 | 3 | 4 |
| crossing | 1 (1) | 5% | 1 | 1 | 0 / 1 | 2.1 | 3 | 1 |
| parked / static beside the path | 5 (3) | 23% | 5 | 5 | 5 / 5 | 1.6 | 3 | 0 |
| standing neighbour beside the path | 6 (3) | 27% | 6 | 6 | 4 / 6 | 3.9 | 3 | 4 |
| side contact while turning | 2 (1) | 9% | 2 | 0 | 0 / 2 | 4.3 | 0 | 0 |
| side contact, moving neighbour | 0 | | | | | | 2 | 0 |
| oncoming, other | 0 | | | | | | 0 | 0 |
| **all** | **22 (12)** | | **16 (73%)** | **20 (91%)** | **17 (77%)** | **2.4** | 24 | 16 |

Reading (P2H10-F, n = 22; full list of circumstances in `collisions/nuplan_tables.md`):

- **Lateral, not longitudinal.** 16 of 22 are at least 0.5 m off the logged path at impact (median 0.94 m), 15 of them toward the struck
  object's side; 20 would be clear had the ego stayed on the logged path at the driven speed; 11 would be clear at the log's position
  along the path with the driven lateral offset. Only 2 (one scene, a launch into a left turn across a 12 m/s vehicle) are cleared by
  neither. This is decision 205's closed-loop heading drift ending against whatever stands beside the lane: parked and standing
  vehicles are 11 of 22.
- **Low speed.** Median 2.4 m/s at impact, 14 below 3 m/s, 6 start from rest; the log's heading change is under 5 deg in 13.
- **The class-A share is small and specific.** 4 rollouts, 2 scenes: both are launches from rest into a car stopped 1.8-3.3 m ahead,
  on the path (offset < 0.25 m), with plans of 1.2-2.7 x the current speed. Every one of the six runs hits these two scenes.
  The AlpaSim-input recipes have more of it (APY10m10 10 of 24, AP2H10 7 of 16 class A) and more starts from rest (14 / 24, 7 / 16).
- **Seen, and the human also came close.** The object is in the front camera's field within 60 m for at least 2 s before impact in 16
  (never in 4: vehicles arriving from behind); the logged human passed within 1 m of the same object in 10.
- **Other drivers.** WA-JEPA (reference) passes 17 of the 22 (all stopped-lead, cut-in and parked cases; not the turning scene nor the
  crossing one); SH30-F-s0 passes 3; the other P2H10 seed passes 2.

## 4. PAI scenes: plumbing against driving (40 scenes, Tokyo box)

Scenes: the 40 of pai1's `pai_scenes_40.tsv` (curated validation split, spread over the references' difficulty). pai1 ran the first 10
(`a10_c4`); COL1 ran the other 30 with the same runner and settings (`pai_run.sh`, 4 concurrent rollouts, one plan per call, Harmonizer
off: `b1`, `b2a`, `b2b`). **P2H10-F-s0: mean scene score 0.1694, 30 zeros: 12 at-fault collisions, 9 offroad, 9 left corridor.** On the same
40 scenes the organisers' Alpamayo 1 reference averages 0.505 and the eight references together 0.196 (their 3 rollouts each, Harmonizer on).
pai1's own write-up is [pai_smoke.md](pai_smoke.md); its reading "the frames are geometrically right, the failures are behaviour" holds
for the frames and is corrected below for one serving-side defect it mentioned as a possible cause and did not isolate.

### 4.1 What was checked and is sound (`collisions/pai_plumbing.md`, all 40 scenes)

| Check | Reading |
|---|---|
| Speed and acceleration fed to the adapter against the simulated ego | median difference 0.00 m/s in every scene; the differences in the tails are the spinning rollouts |
| Frame latency (`drive` time minus the newest frame used) | 0 ms in every scene |
| Slots | all 8 are rendered frames from 1.4 s on; the 14 earlier decisions per scene lie inside the 1.7 s of forced log replay |
| Offline replay of the logged messages through the driver class | reproduces the run's plan at 100.0 % of the 7 947 decisions (largest 4 s end-point difference 0.000 m) |
| Tracking 0.5 s after a decision, scenes without a spin | lateral p95: median 0.04 m over the 34 scenes, at most 0.3 m in 29, largest 0.84 m: the vehicle goes where the plan points |
| Camera geometry | as pai1 found: horizon, lanes and projected plans line up in every frame looked at (16 clips here) |
| First egomotion sample | velocity (0, 0) at any speed; only the first decision uses it, inside the forced replay |

### 4.2 The command (the 922 / 408 / 656 split) is odd and does not matter

The route is map-generated (`route_generator_type: MAP`, first waypoint 36-55 m ahead), not the logged path: its first waypoint is more
than 1 m beside the logged path (median over the rollout) in 14 of 40 scenes, about one lane in several (also because the ego is past the
log's end in some). The shipped rule (first waypoint beyond 5 m, |y| > 2 m) then gives left / straight / right = 2879 / 2539 / 2529
over the 40 scenes; the same rule applied to the logged path from the logged pose gives 912 / 3561 / 703. So 68 % of the calls carry a
turn command against 31 % on the log. The plan hardly reads it: with the command forced to straight on the same tokens the 4 s end
point moves by a median of 0.02 m (per-scene medians 0.00-1.48 m; p95 above 1 m in 8 scenes, at junctions) and the 4 s arc length by a
factor of 0.97-1.04. Not a cause of the zeros.

### 4.3 The serving-side defect: the returned trajectory does not start at the ego's speed

The driver returns the plan's 0.5 s points, linearly resampled to 10 Hz (the shipped LTF sample's function). The first segment of that
trajectory therefore has the plan's mean speed over its first 0.5 s, whatever the ego's speed is, and the simulator's nonlinear MPC
tracks positions. Just before the hand-over (decisions at 1.45-1.75 s, where the state is still the log's; `collisions/pai_tables.md`,
last section; figure `research/alpasim-collisions/figs/fig2_handover.png`):

| ego speed just before the hand-over | scenes | served plan, first 0.5 s against the ego's speed (median) | shipped policy on the same tokens | 4 s mean, served / shipped |
|---|--:|--:|--:|--:|
| 0-2 m/s | 7 | (standing) | (standing) | +65 % / -86 % |
| 2-5 m/s | 3 | +3 % | -15 % | +3 % / -4 % |
| 5-13 m/s | 19 | **+8 %** | -1 % | +3 % / -2 % |
| 13-23 m/s | 6 | -6 % | -4 % | -10 % / -10 % |
| above 23 m/s | 5 | **-6 %** | -6 % | -6 % / -4 % |

- **Above 23 m/s: spin.** In 4 of the 5 scenes (4f779a92 23.9 m/s, 05f35348 29.7, dfb0277e 33.1, b45734f4 34.8) the first segment is
  5-11 % below the ego's speed, and the controller's commands in the first 1.2 s after the hand-over are saturated braking (-9.0 m/s^2)
  with 0.65-0.73 rad of steering; all four spin (2 scored offroad, 2 left corridor). The plans there are straight (0.5 s lateral under
  0.1 m). The fifth (1fcacc17, 29.8 m/s) has -1 %, a largest steering command of 0.004 rad and no spin at the hand-over. CONTROLLER_TUNING.md
  names harsh braking references as a source of zig-zag steering.
- **At 5-13 m/s: acceleration.** The served first segment is 8 % above the ego's speed (shipped: -1 %), the MPC chases the position, the
  next decision is re-anchored at the new speed and asks for more again. In 15 of the 19 scenes the largest acceleration in the first 1.2 s after the hand-over is 1.4-3.8 m/s^2; at
  12 s it is above 1.2 x the log's speed in 15 of the 31 rollouts that are not offroad (median 1.17 x).
- **At rest: launch.** The served plan asks for 0.6-1.7 m/s over 4 s in the four scenes that start at rest (shipped: at most
  0.1 m/s). This is the adapter's speed prior of decisions 162 / 167, on a new domain. Decision 218 (lane DIAG1, read in parallel on these
  rollouts) attributes the 5-13 m/s excess to the adapter continuing the fed acceleration, and the launches to the bias constant.

**Diagnostic arm `v1`** (pre-registration amendments 1 and 2; `lib/col1_pai_driver.py`: the plan's own path, re-timed to start at the
ego's speed and to join the plan's speed profile after 1 s; nothing else changed; not a candidate, no line):

| 40 scenes | mean scene score | score 1 | zeros | at-fault collision | offroad | left corridor |
|---|--:|--:|--:|--:|--:|--:|
| baseline | 0.1694 | 4 | 30 | 12 | 9 | 9 |
| `v1` | 0.2921 | 8 | 25 | 8 | 5 | 12 |

Paired difference +0.1227 [-0.0055, +0.2531] (scene bootstrap; one run per arm, the simulator's own spread is not in it); 10 scenes up,
2 down, 28 unchanged. On the first 10 scenes alone, read before the extension: 0.1538 -> 0.3332. Full table: `collisions/pai_arm_v1.md`.

- **Hand-over spins: 2 of 4 gone.** 4f779a92 (23.9 m/s; largest steering command 0.728 -> 0.004 rad, braking -9.0 -> -2.2 m/s^2) and
  dfb0277e (33.1 m/s; 0.725 -> 0.002 rad, score 0 -> 1.00) no longer spin. 05f35348 (29.7 m/s, plan 11 % short) and b45734f4 (34.8 m/s,
  6 % short) still do (0.65 / 0.77 rad, -9.0 m/s^2): with a 1 s ramp their reference still asks for 2-3 m/s^2 of braking, and the MPC does
  not hold that at 30-35 m/s. Amendment 1 said in advance that both rollouts above 23 m/s known then would stop spinning; one did, one did not.
- **Collisions 12 -> 8.** Of the 12, 2 pass (071d15c4, 0ee89cab: both launches from rest behind traffic), 3 turn into another zero
  (2 left corridor, 1 offroad), 7 collide again; one former offroad scene now collides. The approach speeds barely move: at 12 s the ego
  is above 1.2 x the log's speed in 18 of 35 rollouts that are not offroad (baseline 15 of 31).
- Zero flags, baseline -> `v1`: offroad -> none 4, collision -> none 2, corridor -> none 1; none -> corridor 2.

Reading: the defect is real and serving-side, it is the trigger of the hand-over spins, and a 1 s speed ramp is a partial repair
(controller gains are a submission parameter and were not varied). It is not why the driver collides.

### 4.4 Split of the PAI zeros

| 30 zeros of the baseline | n | plumbing or driving |
|---|--:|---|
| spin at the hand-over (2 offroad, 2 left corridor) | 4 | **plumbing**: trajectory speed step against the nonlinear MPC above 23 m/s (in `v1`: 1 passes, 1 becomes left corridor, 2 still spin) |
| at-fault collisions | 12 | **driving**: speed above the log's (adapter speed prior) with no reaction to the vehicle ahead, 7 of them after leaving the logged lane; the position chasing of section 4.3 adds to the launches (2 pass in `v1`) |
| other offroad (7) and left corridor (7) | 14 | driving, not taken apart here (pai1: wide turns, speed held into slow manoeuvres); 4 of them become non-zero in `v1`; 2 non-zero scenes become zeros |

Nothing in the inputs is wrong: camera, speed, timing and slots check out, and the odd command split has no effect on the plan.

## 5. PAI taxonomy of the 12 at-fault collisions (`collisions/pai_tables.md`, `pai_cases.csv`)

| struck object | n | share | scenes | ego off the logged path > 1 m | ego ahead of the log > 2 m | Alpamayo 1 (reference) passes |
|:--|--:|--:|:--|--:|--:|--:|
| adjacent-lane vehicle (the ego left the logged lane; 2 moving, 2 standing or nearly so) | 4 | 33% | 13fb89b9, 24a50fcc, 0ee89cab, 8ecf02e3 | 4 | 4 | 3 / 4 |
| parked / static | 3 | 25% | a28b6685, 605bf77a, c6c01d25 | 2 | 1 | 2 / 3 |
| lead stopped | 2 | 17% | 9ea70552, 0e3771c0 | 0 | 2 | 1 / 2 |
| slow lead | 1 | 8% | 071d15c4 | 0 | 1 | 1 / 1 |
| crossing | 1 | 8% | 8a365ac2 | 0 | 1 | 1 / 1 |
| pedestrian | 1 | 8% | 8f3902f9 | 1 | 0 | 1 / 1 |
| **all** | **12** | | | **7** | **9** | **9 / 12** |

Class rules (on the simulator's boxes at the first at-fault step): heading within 45 deg of the ego's and centre within 1.5 m of the
logged path = lead (stopped below 0.5 m/s); the same but 1.5 m or more beside the logged path = adjacent-lane vehicle (parked / static if it
never moves); moving at 45-135 deg = crossing; non-vehicle labels by label. All 12 were looked at as clips.

- **Longitudinal first.** 11 of 12 are front contacts; the ego is more than 2 m ahead of the log in 9 (13-39 m in 6); median ego speed at
  impact 6.7 m/s (2 below 3 m/s); 9 would be clear at the log's position along the path with the driven lateral offset, 8 on the logged
  path at the driven speed. On nuPlan the same counterfactuals read 11 and 20 of 22: there the collisions are lateral, here the ego
  outruns the log into replayed traffic, and in 7 of 12 it has also left the logged lane by more than 1 m.
- **What is straight ahead of the ego.** By the logged lane 3 of 12 are class A. By the ego's own straight-ahead corridor the struck
  object is ahead for at least 2 s of the last 8 in 9 of 12, and the shipped lead head is on it (prob >= 0.5, distance within max(2 m,
  30 %) of the gap) for 74-92 % of that time in 5 (071d15c4, 9ea70552, 0e3771c0, 24a50fcc, 8ecf02e3), 41-44 % in 2 (605bf77a,
  0ee89cab) and 0-7 % in 2 (the parked car at the kerb, the pedestrian).
- **The plans go through the object.** In the last 4 s the served plan reaches the object within its 4 s (object at constant speed) at
  half or more of the decisions in 10 of 12 rollouts; the shipped policy's plan on the same tokens in 7 of 12, with a 4 s arc that is
  0.50-0.87 of the served one. The shipped plan is slower, and it is not what stops shipped openpilot either.
- **Start.** 5 of 12 start at 1 m/s or less (071d15c4, a28b6685, 0ee89cab, 8a365ac2, 0e3771c0). pai1's "collision after 3 m" is
  a28b6685: pulling out behind a car parked 2 m ahead, which the log passes at 0.36 m; the first plans are 2.5 x the current speed.
- **Visibility.** In the front-wide field within 60 m for at least 2 s before impact in 11 of 12 (not the crossing vehicle).
- **Reference.** Alpamayo 1 scores above zero on 9 of the 12 scenes (mean of its 3 rollouts).

## 6. Did the frozen model's lead outputs see it (pre-registered read)

Signals: P0 = the shipped policy weights without the adapter bias on the same vision tokens (primary); FT = the served checkpoint's own
output vector. They agree closely everywhere (same line results), so the heads are intact in the served checkpoint and cost nothing.

| | nuPlan public scenes (`collisions/nuplan_lead.md`) | PAI (`collisions/pai_tables.md`) |
|---|---|---|
| C (at-fault-collision rollouts of P2H10-F) | 22 (s0 + s1) | 12 (s0) |
| C_A (class A) | 4 rollouts, 2 scenes | 3 |
| Controls N | 198 rollouts, 1980 decisions | 21 rollouts, 2192 decisions |
| **L1: S_time (P0)**, share of class-A rollouts seen in time | 2 / 4 = 0.50 [0.00, 1.00], lead times 4.0, 3.5 s | 3 / 3 = 1.00, lead times 7.7, 6.2, 6.5 s |
| L1 reading | descriptive only (fewer than 8) | descriptive only (fewer than 8) |
| Other recipes, reported separately | APY10m10-AB 8 / 10 = 0.80 [0.50, 1.00] (above the 0.60 line); AP2H10-AB 4 / 7 = 0.57 | |
| **L2 at the primary point** (p* 0.5, a* 1.5): turned / C | 6 / 22 = 0.27 [0.09, 0.45] | 7 / 12 = 0.58 [0.33, 0.83] |
| false-alarm scenes / N | 28 / 198 = 0.14 [0.10, 0.19] | 0 / 21 |
| false-alarm decisions | 132 / 1980 = 6.7 % | 15 / 2192 = 0.7 % |
| L2 reading (turned >= 0.40 and false-alarm scenes <= 0.15) | **not as a rule** | **passes: an in-loop test is worth a lane** |
| Turned, by class | 4 / 4 lead stopped; 2 standing-neighbour rollouts from one cold-start trigger; 0 / 12 cut-in, crossing, parked, turning | 2 / 3 class A; 3 / 4 adjacent-lane; 2 / 3 parked; 0 / 2 crossing, pedestrian |
| Post hoc: turned with the lead head on the struck object at the trigger | | 6 / 12 |

Lead head against the simulator's gap:

| gap | nuPlan controls with an object in the corridor: prob >= 0.5 / median distance error | PAI, all labelled decisions: prob >= 0.5 / median error |
|---|--:|--:|
| 0-5 m | 93 % / -0.7 m (class-A collision decisions: 93 % / +1.2 m) | 71 % / +3.4 m |
| 5-10 m | 90 % / -1.1 m | 63 % / +0.5 m |
| 10-20 m | 88 % / -1.4 m | 60 % / -0.4 m |
| 20-40 m | 77 % / -1.8 m | 62 % / -1.0 m |
| above 40 m | 57 % / -3.6 m | 38 % / -15.1 m |
| nothing in the corridor | | prob >= 0.5 at 17 % of decisions |

Reading:

- **nuPlan: the lead outputs cannot be the fix.** Only 4 of 22 collisions have a lead to see; the rule turns those and nothing else,
  and it false-alarms in 14 % of clean 5 s scenes (stopped queues the log is also waiting in). The collisions there are a lateral problem
  (section 3). In one of the two class-A scenes the head reads 5-10 m where the gap is under 2 m (the near-range over-read of
  decision 157).
- **PAI: yes, in time, for the frontal half.** All 3 class-A rollouts are seen 6-8 s ahead, and counting by the ego's own corridor
  the head is on the struck vehicle in 5 of 12 for most of the approach (figure `fig3_lead.png`). The registered line L2 passes
  (7 / 12 turned, 0 / 21 false-alarm scenes). Two cautions on the 7: the registered "turned" needs only a trigger while a stop is still
  possible, so for objects beside the ego it can be a trigger on another lead (the post hoc count with the head on the struck object is
  6 / 12); and the rule did not turn 0e3771c0 (creeping in a queue at 0.6-0.9 m/s: above the standstill threshold and too slow for the deceleration branch).
- The PAI controls trigger falsely at 0.7 % of decisions and in no scene for 1 s; nuPlan's 14 % against PAI's 0 % is the difference
  between 5 s scenes that start inside a standing queue and 20 s scenes that mostly do not.

## 7. The cheapest in-loop use, and its ceiling from the logs (not implemented here)

Two serving-side pieces, in order of cost; both use only what the forward pass already computes.

1. **Return a trajectory that starts at the ego's speed** (section 4.3). Not a lead mechanism; it removes the spins and the position
   chasing. Measured, not estimated: the arm `v1` above.
2. **A lead-conditioned speed cap on the plan**: when lead prob >= 0.5 and the constant deceleration needed to match the lead's speed 4 m
   short exceeds 1.5 m/s^2, or the ego stands within 6 m of a lead, re-time the plan along its own path to that deceleration (hold at
   standstill). No extra model pass with the served checkpoint's own lead outputs; one policy pass more (13 ms) with the shipped weights.
   Ceiling from the logs, an upper bound in both tracks (a stop is assumed to cause no other zero; traffic does not react):

| | zeros it could turn | cost seen in the controls |
|---|---|---|
| PAI, 40 scenes | at most 7 of the 12 collision zeros (6 with the head on the struck object), i.e. up to 7 / 40 = +0.18 mean scene score if each became a full score, which the progress term will not grant | 0 / 21 control scenes with a false trigger held for 1 s; supported triggers in 3 |
| nuPlan, 1491 scenes | at most 6 of 22 collision rollouts (3 of 11 per seed, about 0.2 % of scenes, +0.002) | 14 % of clean scenes slowed for at least 1 s (28 / 198), each losing progress in a 5 s scene |

   So the rule is worth an in-loop test on PAI and is a net loss on the nuPlan track. How it differs from what is settled: decision 157
   subtracted a fixed 2 m from stored plans on HUGSIM / navtest; decision 140 gated launches on the lead head in HUGSIM; decisions 179 / 203
   read token-level AUCs against open-loop, non-reactive 4 s labels on navtest. None is a deceleration cap from the lead outputs in a
   20 s closed loop behind replayed queues, and decision 203's finding (the native lead outputs are the most informative unprivileged
   signal for moving-lead failures) points the same way.
3. What neither piece touches: the 9 PAI collisions' other ingredient, speed above the log's (the adapter's prior), the adjacent-lane
   cases that begin with the ego leaving its lane, and everything on nuPlan that is drift.

## 8. Clips (31; `figs/collisions/`)

nuPlan (15; `collisions/nuplan_clip_cases.csv` has one sentence per clip): `n01`-`n13` are original rollouts (classes in proportion:
2 lead stopped, 3 parked, 3 standing neighbour, 2 cut-in, 1 turning, 1 crossing, 1 slow lead of another recipe), `c01` / `c02` are
contrasts from the control re-run (the same scene as `n13` passed by P2H10-F-s0; a stopped lead approached at 6.6 m/s and handled).
Left: bird's-eye view with the map; right: the road and wide model frames of that decision; bottom: speed, gap and lead-head distance, lead prob.

PAI (16; `collisions/pai_clip_cases.csv`): `pai_p01`-`pai_p12` are all 12 collisions (original rollouts: 4 from pai1's `a10_c4`, 8 from
COL1's `b1` / `b2a` / `b2b`); `pai_c1` a slowing queue handled; `pai_c2` the hand-over spin; `pai_c3` / `pai_c4` the scenes of `p01` and
`c2` in the diagnostic arm. Same layout without a map (none is delivered on PAI); the model frames are those of the offline replay,
which reproduces the run's plans exactly. In `pai_c3` the contact is gone but the ego still follows at 3-5 m while accelerating to
11 m/s: the arm removes the position chasing, not the speed prior.

## 9. Limits

- PAI: one checkpoint, one seed, one rollout per scene, 40 scenes; 12 collisions, 3 of class A. Line L1 is descriptive in both tracks.
  The simulator's run-to-run spread on PAI is not measured, so the arm's interval is a scene bootstrap only.
- "Turned" is an upper bound by construction (section 6). The guard was not run in the loop; a stop behind replayed traffic can be hit
  from behind (not at fault, but the rollout ends) and loses progress.
- Visibility is "in the camera's field within 60 m", without an occlusion test. Class rules use the logged lane, which mislabels objects
  ahead of an ego that has left it (section 5 gives both counts).
- nuPlan: AP2H10 on the 791 later scenes was still running in lane CF1 and is not included; the controls are the part001 + part002 scenes
  of a re-run of the three original 700-scene chunks (all 700 scores identical to the original run), not a re-run of two shards alone.
  Six of the eight slots per decision are synthesised frames there.
- The command finding is an open-loop replay with one command forced at a time.
- nuPlan clip sentences for `n02`-`n05`, `n11`-`n13`, `c01` were written from the tables; the others and all PAI clips were looked at.

## 10. Cost

GPU box: about 0.8 card-hours (three control stacks of 12-14 min, three replay jobs); at most 5 simulator stacks ran at once (2 of this
lane with CF1's 3), no worker was killed, peak host anon memory in that window 167.5 GiB; the raised cap of 6 stacks was never reached
because only one chunk was left when it arrived. `$DATA_DIR/runs/alpasim/col1/` holds 19 GB (the control runs with all logs).
Tokyo box: 28 scenes downloaded, about 50 GB (stated here because it exceeds 5 GB), `/data/datasets/nurec` now 40 scenes; 34 GB under `/data/runs/alpasim/col1` (runs with logs, extractions, replays). GPU time there: 30 baseline scenes (10 + 11 + 43 min, one card each), 40 diagnostic scenes (10 + 11 + 11 + 42 min), about 50 min of offline replay; both cards were idle before and after. One batch of 10 scenes ran 4 x slower than the others on card 0 (cause not looked for).

