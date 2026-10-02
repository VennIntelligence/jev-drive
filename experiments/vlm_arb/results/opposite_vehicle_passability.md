# Bench2Drive: the emergency vehicle that takes priority (route 9196) and similar scenarios

Question 2 of the open-loop review follow-up, Bench2Drive half (HUGSIM half:
[hugsim/results/attack_passability.md](../../hugsim/results/attack_passability.md)).
**V** = verified (file:line or log), **I** = inferred.

## 1. Which case

The page's Bench2Drive section has no case where a leading vehicle turns left. Candidates that contain "an oncoming vehicle
that drives at the ego without slowing":

| page case | route | scenario type (`bench2drive_0.0.4_val.xml`) | what happens | fits |
|---|---|---|---|---|
| b2d-12b | 9196 (Town12) | `OppositeVehicleTakingPriority_1`, direction right | ego turns left at a junction, a firetruck at 15-17 m/s comes from the right side and hits it | best fit for "oncoming vehicle, no slowing"; no leading vehicle exists in the scene logs (V: only the ego and the firetruck are vehicles in `scene.jsonl`) |
| nvcl-04 | 2390 | `VanillaNonSignalizedTurn_1` | Alpamayo + Zoo PID controller reads a reversing plan as throttle and drives into a crossing car | no: that is a controller defect, and the scenario has no scripted adversary |
| b2d-12a | 27043 | right-turn merge | ego merges into traffic, hit from behind | no |

Only 9196 is analysed below as the main case; if the user meant a HUGSIM case, see the HUGSIM file.

## 2. How the benchmark scripts it

`third_party/Bench2Drive/scenario_runner/srunner/scenarios/opposite_vehicle_taking_priority.py` (Bench2Drive 7ec25d1c), all V:

| item | value | line |
|---|---|---|
| actor | emergency vehicle (`special_type: emergency`; the firetruck in 9196), spawned 30 m before the junction on the entry road, physics off until triggered | 53, 123, 136 |
| trigger | the first of: ego time-to-arrival at the collision point <= 2.2 s, or ego within 12 m of it | 65-66, 279-283 |
| behaviour | `ConstantVelocityAgentBehavior` at 60 km/h (16.7 m/s) to the sink, `ignore_vehicles` and `ignore_traffic_lights` true; it never brakes or yields | 55, 294 |
| lights | `TakingPriority` leaves the junction's lights alone (the red-light twin freezes them) | 256-317 |
| intended behaviour | ego "has to handle this situation correctly", i.e. not enter while the adversary crosses and then continue (scenario_runner docs, `list_of_scenarios.md:27-32`, for the red-light twin) | |
| reference driver | PDM-Lite is a privileged rule-based expert; its code (carla_garage) is not on the box and was not read | I |

Geometry of 9196 in our logs (V, `v2-gifr2-drive-s0-d/attempts/9196/1`, `scene.jsonl`): the ego starts at (500, 4511) heading
west, 18 m before a red light 1176, turns left onto the road x = 463 heading south; the firetruck waits at (463, 4474)
(36 m north of the junction, present from the first frame, stationary), starts at the trigger and drives south on x = 463,
which is the lane the ego's left turn exits into. So the ego merges into the truck's lane from the side.

## 3. Timeline (our run, ego "drive" arm, DS 14.9)

Clocks: `scene.jsonl` / `ticks.jsonl` use t_scene; the GIF overlay, `contacts.jsonl` and the review text use
t_gif = t_scene + 0.95 s (V: contact frame 624 is t = 30.70 in `contacts.jsonl` and 29.75 in `scene.jsonl`; the summary's
"30.7 s" is the GIF clock). Table in t_gif.

| t_gif (s) | event | source |
|---|---|---|
| 0 | scenario starts (trigger point is the ego start); firetruck exists, parked 36 m north of the junction | route xml, `scene.jsonl` (V) |
| 7.8 | firetruck first appears in the actor list (parked) | `opposite_vehicle_timeline.py` (V). Whether a camera sees it is not measurable: `frames/` of this run is empty |
| 0 - 25.2 | ego creeps to x = 483 and stands still (from t_gif 20.2) 20 m before the junction, light 1176 red | `ticks.jsonl` (V) |
| 25.2 | ego starts moving on red (throttle 0.66) | `ticks.jsonl` (V) |
| 28.6 | **trigger**: truck starts, ego at x = 474.6 (11.6 m from the junction centre line), 4.7 m/s | `scene.jsonl` (V) |
| 28.6 - 30.7 | truck covers 36 m at 16.6 m/s; ego never brakes (throttle 0.44 -> 0.31, speed 4.7 -> 4.0 from the curve) | (V) |
| 30.7 | contact, ego 4.0 m/s, truck 15.4 m/s; infraction record places it at (466.2, 4513.9) | `results.json`, `contacts.jsonl` (V) |

Time from trigger to contact: 2.1 s. The truck is visible to cameras at best 2 s before contact (I: it comes out from the
north cross street at 16 m/s; before the trigger it is parked and says nothing about its intent).

### Was avoidance available

`opposite_vehicle_avoidance.py` takes the ego path as driven and asks, at each tick, whether constant braking from that
instant would stop the ego short of the contact point (front half-length 2.45 m clearance). Result (V on the log; the
deceleration values are assumptions, the observed ego decel in the same run was up to 7.2 m/s^2 over 0.5 s):

| t_gif | ego speed | path left to contact (after front clearance) | stop with 3 / 5 / 7 m/s^2 |
|---|---|---|---|
| 28.2 | 4.4 | 8.6 m | yes / yes / yes |
| 28.7 | 4.7 | 6.3 m | yes / yes / yes |
| 29.2 | 4.7 | 3.9 m | yes / yes / yes |
| 29.7 | 4.5 | 1.6 m | no / no / yes |
| 30.2 | 4.0 | -0.5 m | no / no / no |

Braking was available from the trigger until about t_gif 29.2 s with ordinary deceleration and until 29.7 s with hard
braking: a window of 0.6-1.1 s after the trigger. "Accelerate through" is not an option here (I): the exit of the turn is
the truck's own lane, so a faster ego would end up in front of a vehicle doing 16.7 m/s and be hit from behind. "Reverse"
is also unnecessary: stopping short, or arriving later, is enough. The scenario is designed to fit this: the trigger is
set so that an ego at normal junction speed has about 2.2 s to react.

Two other attempts at the same route have the same trigger and did not collide (V, `opposite_vehicle_9196_encounters.csv`):

| arm | ego speed at trigger | closest approach | ego speed there | result |
|---|---|---|---|---|
| v2-gif-drive-s0-b | 4.05 m/s | 4.7 m | 2.0 m/s | passed behind the truck, DS 70 (red light only) |
| v2-gifr1-drive-s0-d | 3.45 m/s | 6.2 m | 1.3 m/s | passed behind the truck, DS 70 |
| v2-gifr2-drive-s0-d (the case) | 4.69 m/s | 0 (contact) | 4.0 m/s | collision |

So 1 m/s of entry speed (and the same throttle policy) decides the outcome; it is a timing, not an impossibility.

## 4. Do models pass it

Exact route 9196 is not in the public per-route data: `bench2drive_0.0.4_val.xml` (our evaluation set, 220 routes) shares
only 2 route ids with the old `bench2drive220.xml` that all public per-route files cover (V, id sets compared on the box).
Public results therefore exist only per scenario type (5 routes per type, old set). Tables from
`scripts/opposite_vehicle_public.py` over `leaderboard_audit/results/b2d-family/public_routes.csv`
(success = official `success` flag, i.e. no infraction; DS = mean driving score; runs = published runs of that model,
BLUE has 6 seeds, DriveMoE and Orion-Lite 2). All V.

**OppositeVehicleTakingPriority** (public routes 2127, 2129, 2143, 2913, 3697):

| model | success | mean DS |
|---|---|---|
| TF++, ORION | 5/5 | 100 |
| TCP-traj, SimLingo, TFv6, R2SE, Orion-Lite (2 runs: 8/10) | 4/5 | 86-96 |
| PDM-Lite (privileged expert) | 4/5 (route 2913: DS 65) | 93 |
| BLUE (6 seeds) | 22/30 | 88.9 |
| UniDriveVLA | 3/5 | 87.9 |
| SparseDriveV2 | 2/5 | 66.9 |
| MindDrive, MindDrive-3B, Hydra-NeXt, DriveMoE (2 runs), Drive-pi0, VAD, UniAD-Base, UniAD-Tiny | 0 | 20-39 |
| pooled over all 130 route-runs | 65 (50%) | |

9 of the 18 listed models pass at least 60% of the five routes; 8 pass none. Even the expert fails one of five.

**OppositeVehicleRunningRedLight** (routes 2082, 26944, 26950, 2844, 2847; same truck behaviour, lights frozen for the ego green): pooled 94 of 130 (72%);
8 entries 100%, PDM-Lite 5/5, SimLingo 2/5, DriveMoE 4/10, UniAD-Base 0/5. **YieldToEmergencyVehicle** (a different scenario: siren from behind): pooled 5 of
129, only PDM-Lite passes (BLUE 0/30), unrelated to the case. **VanillaNonSignalizedTurn** (nvcl-04's type): pooled 90 of 130 (69%);
PDM-Lite, TF++, SimLingo, R2SE, BLUE (30/30) all pass. Full per-model and per-route tables:
`results/opposite_vehicle_public.csv` and the script output.

How the passing models do it: not established. Papers and code of these models were not read here (I: the
privileged expert reads actor boxes; camera models can only react after the trigger, so they must enter the junction
slowly or brake in about one second).

Our own runs (`opposite_vehicle_own_runs.csv`, 36 attempts over all arms, V):

| route | scenario | attempts | completed without vehicle collision | note |
|---|---|---|---|---|
| 9196 | OppositeVehicleTakingPriority_1 | 27 | 4 (DS 70, all with a red-light penalty); 0 reach DS 100 | 23 attempts record a vehicle collision; 19 a red light |
| 8859, 9102, 9218 | OppositeVehicleTakingPriority_1 | 9 | 9 | DS 80, no collision, no red light |

So the same scenario type is passed on three of our four routes; 9196 is the hard one because the ego is released from a
red light and is still accelerating into the conflict area when the 12 m trigger fires (V for the numbers, I for the cause).

## 5. Verdict

- Passable: yes. The scripted vehicle never yields, but it is triggered by the ego and needs 2.1 s to arrive; braking or
  entering slower than about 4 m/s avoids it (two of our own attempts did; the privileged expert passes 4 of 5 routes of the
  type, 8 of 18 public models pass none).
- How common: about half of all public route-runs of the type (65 of 130), 100% for the best models, 0% for eight entries; our agent
  passes 4 of 27 attempts on this route and 9 of 9 on the other three routes of the type.
- What it tests: yielding to a priority violator whose appearance can only be anticipated by approaching a junction
  slowly, plus a reaction within about one second; it is not a stop-line test (the stop line is irrelevant once the
  ego is already moving) and "reverse fast" or "accelerate through" are not the intended answers. The user's "nit-picking"
  reading does not hold for this route: braking from 4.7 m/s was physically available for about a second after the trigger.
- Not established: whether the cameras could see the truck before the trigger (no frames saved), the code-level
  behaviour of the passing models, and any result for exactly route 9196 by other models.

## Files

- scripts: `experiments/vlm_arb/scripts/opposite_vehicle_public.py`, `opposite_vehicle_timeline.py`,
  `opposite_vehicle_encounters.py`, `opposite_vehicle_avoidance.py`, `opposite_vehicle_own_runs.py`
- results: `opposite_vehicle_public.csv`, `opposite_vehicle_own_runs.csv`, `opposite_vehicle_9196_encounters.csv`,
  `opposite_vehicle_9196_avoidance.csv` (empty encounter csvs for 8859/9102/9218 were removed: no scene logs)
