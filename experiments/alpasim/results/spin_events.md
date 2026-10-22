# Spin / loss-of-control events in PAI model reads (2026-10-22, CPU diagnostic, no GPU)

Question (user, from driving PAI by hand): hard braking at speed with a straight plan makes the official MPC spin the car
(`/data/runs/alpasim_human/exam_pai_1022-154753` on Tokyo). Do the "inexplicable" left rolls / spins at speed in our model reads come from
the same mechanism, and what do they cost?

**Answer: not in the current reads.** The mechanism is real, and it is the one decisions 285 / 291 / 309 found. The trajectory demands a
deceleration jump the MPC cannot follow (its accel command goes to -9 m/s^2), so the MPC burns distance by steering (0.3-0.9 rad), and the
linear bicycle model yaws as that steering implies. It cost points in the path-0 reads before 2026-10-15. Since decision 309, every PAI read serves
`JEV_TRACK=gate`: the reference is re-timed from the ego's speed with deceleration bounded at -5 m/s^2 and longitudinal jerk at 10 m/s^3.
Across the 16 current PAI 281 reads (NTV0, R6DAG, R6DAG2, R8K; ACC + GAP, Harmonizer off / on) **no spin precedes any scored failure; 0
points**. What looks like a car rolling over in those videos is the simulator pose fault: z / pitch jump first, the MPC reacts after. It
appears in 29 rollouts, all in scenes that score 1.0 or after the failure already happened, so it also costs 0 points. The hand mapping has
no gate, which is why the user meets the effect when driving by hand.

Code `scripts/spin_events.py` (commit 32a0066d); outputs `results/spin_events/` (`events.csv` per event, `events_rollouts.csv` every rollout
with an event, an sf_detect flag or roll > 5 deg, `summary.md`, `*_p0*` the path-0 calibration, `reads.txt` the roots on the box).

## Vehicle model and the mechanism

`alpasim_controller/vehicle_model.py` (as shipped): above 5 m/s a linear dynamic bicycle model (Ford Fusion, Caf 93.5 kN/rad, Car
176 kN/rad, understeering). Lateral and longitudinal dynamics are decoupled, so braking by itself cannot yaw the car; the yaw comes from
steering. The user's events, controller CSV (`stacks/*/sim/controller`):

| scene | t s | v m/s | accel command | MPC steering command | peak wz rad/s | sideslip vy m/s |
|:--|--:|--:|:--|--:|--:|--:|
| 49597f01 | 13.4 | 34.6 | +1.2 -> -9.0 in 0.5 s | -0.36 rad | -1.65 | 1.76 |
| 9ea70552 | 4.7 | 17.9 | +3.1 -> -8.6 in 0.5 s | -0.48 rad | -1.64 | -1.46 |
| 071d15c4 | 17.8 | 18.3 | +2.5 -> -9.0 in 0.6 s | +0.24 rad | +0.73 | 0.70 |

The human's own delta stays at 0.001 rad. The MPC steers by itself while the reference brakes harder than its accel bound and accel time
constant allow; once the accel command sits at -9, the steering falls back and the yaw dies out within about 0.5 s.

## Detector

On the controller rows (0.1 s) joined to the decision records (`drive.jsonl`; the trajectory actually handed to the controller,
`poses_track` if the gate built one): yaw excess = |wz - v kappa_plan| (kappa_plan = the served path's mean curvature over its first 1 s). An event is
v >= 8 m/s and either yaw excess >= 0.3 rad/s for >= 0.2 s or sideslip |vy| / vx >= 0.15. The rule of decisions 291 / 309 (`sf_detect`) is
reported alongside. Calibration on the path-0 read `RSG-L18-s0` (281, `steer_long/spinfix/v8/pai/rsg`, d309's identity):
sf_detect 26 (d309: 26). The plan-relative rule gives 17 events in 15 scenes. In every one the MPC brakes to -4.4 ... -9 m/s^2 with
0.6-0.9 rad of steering; 11 of the 17 go left, 4 are at >= 15 m/s (4cb05d65 38.4 m/s and 32be03e4 37.4 m/s at the 1.8-1.9 s handoff, then
again at 5 s; 9d9052bd 29.1 m/s at 17.4 s; 61e73e86 21.3 m/s at 13.3 s), and 10 are on zero scenes. So the user's memory of fast cars
spinning left is right for the reads before the gate.

## Current reads (gate path, PAI 281, ACC + GAP)

| read (Harmonizer off / on) | mean | zeros | plan-relative events | of them before a zero's failure | sf_detect flags | sf flags before a zero's failure | pose-fault rollouts |
|:--|--:|--:|:--|:--|:--|:--|:--|
| NTV0-s0 | 0.744 / 0.747 | 59 / 58 | 0 / 3 | 0 / 0 | 5 / 2 | 0 / 0 | 1 / 0 |
| R6DAG-s0 | 0.777 / 0.793 | 53 / 46 | 1 / 1 | 1 / 1 (4f7cb072) | 10 / 3 | 0 / 0 | 3 / 1 |
| R6DAG-s1 | 0.817 / 0.781 | 43 / 49 | 1 / 1 | 1 / 1 (4f7cb072) | 8 / 6 | 0 / 0 | 2 / 2 |
| R6DAG-s2 | 0.792 / 0.782 | 51 / 48 | 1 / 2 | 1 / 1 (4f7cb072) | 9 / 6 | 0 / 0 | 1 / 2 |
| R8K-s0 | 0.794 / 0.775 | 50 / 53 | 3 / 1 | 1 / 1 (4f7cb072) | 10 / 7 | 0 / 0 | 3 / 1 |
| R8K-s1 | 0.802 / 0.770 | 47 / 54 | 0 / 4 | 0 / 1 (4f7cb072) | 6 / 11 | 0 / 0 | 3 / 3 |
| R8K-s2 | 0.806 / 0.795 | 47 / 47 | 1 / 2 | 1 / 1 (4f7cb072) | 3 / 6 | 0 / 0 | 2 / 1 |
| R6DAG2-s0 / s1 (off only) | 0.790 / 0.796 | 50 / 50 | 1 / 1 | 0 / 1 (4f7cb072) | 3 / 11 | 0 / 0 | 1 / 3 |
| DAgger training rolls s0 / s1 (716 / 718 train scenes, no facts) | 0.714 / 0.732 | 170 / 162 | 4 / 3 | 1 (83135623, pose fault at 17.6 s) / 0 | 14 / 16 | 0 / 0 | 2 / 4 |

Every exam zero with a plan-relative event before its failure (12 reads) is one scene, 4f7cb072, and it is not a spin. The rule fires
at 2.4-2.6 s with 0.3-0.4 rad/s of excess; the pai_fix facts put the offroad at 2.7 s. Its route bends left (route waypoint at 42 m is +5.6 m left) and the served plan follows it (y at 2 s +3 ... +5 m). The MPC
steering stays at <= 0.08 rad, and the yaw is the plan's yaw under ACC braking at -4 ... -5. The car leaves the drivable area on the left
at 22 m/s (fclass `off_straight`, every model including RB0-F and R5LC; NTV0 survives to 12.4 s). The 14 deg body roll in the video comes
after the offroad, from the terrain. The other plan-relative events are mild (0.3-0.6 rad/s, MPC steering <= 0.18 rad) or come after the
failure:
- 7c346fd5, 5 reads, 16-19 s, 20 m/s: braking -5 ... -6 on a straight plan with 0.3 rad/s of yaw. These are rear-hit partials, the loss
  is the rear hit, and the scene still scores 0.73-0.84.
- 444fca69, 2466266e, 9b6c68b4: at 8-9 m/s, 6-7 s after an overshoot zero.
- 6ee9ee78, 1a89339b, 15c75c83: on 1.0 scenes.
All 44 sf_detect flags on zero rollouts start after the scored failure.

**Pose fault** (d291's d8bfaaf2 kind): 29 rollouts in 6 scenes. In each the z / pitch jumps first, with a benign plan and the accel command
at -3 ... +1, and the MPC reacts after. Rows:
- 6e7b42a3 R8K-s0 at 18.7 s, 34.8 m/s: z +2.4 m in 0.1 s, pitch -66 deg, then roll -66 deg, MPC -9 m/s^2 and 1.2 rad.
- 0cc7fa67 R8K-s1-agH at 17.4 s, 16.9 m/s: pitch -51 deg.
- 51400458 in 13 reads at 15-17 s, 5-6 m/s.
- b680a689 in 10 reads at 17-19 s, after the offroad at 10 s in 7 of them.
- 23666aa2 (3 reads) and b4a3c975, at 15-20 s, 5-7 m/s.
All of them are in scenes scoring 1.0 or after the failure. These are the "rolls to the left" a video shows at the end of a rollout. They
cost nothing, but they are not driving behaviour. The 160-scene reads exist only for older checkpoints (RSG, RB0-F) and were not scanned.

## What this means for the driver and the human mapping

- Model serve: nothing to fix. The gate (`experiments/alpasim/lib/track_ref.py`, AMIN -5, JLON 10) already caps the deceleration and jerk
  of the reference. The only brake-related lateral motion left is a 0.3 rad/s yaw on 7c346fd5 at -5 ... -6 (ACC braking for the vehicle
  ahead), with no score consequence.
- Hand mapping: the same remedy applies. Feed the human's trajectory through the gate's longitudinal re-timing (start from the ego's speed
  and accel; deceleration >= -5, or at most the MPC's -9 with jerk <= 10 m/s^3), so the reference never asks for more than the accel
  time constant (0.1 s) allows. Test as a small pilot with an early-stop gate:
  1. Offline replay of the three logged events above through `track_ref` (CPU). Pass if the reference's first 1 s demand stays within
     -9 m/s^2 and 10 m/s^3.
  2. Re-drive the same 10 scenes by hand on Tokyo (by the other agent / card 1). Stop at the first rollout where the MPC steering command
     exceeds 0.1 rad on a straight plan (|kappa| < 0.005) while braking at >= 15 m/s. Pass if none of the 10 does, and stopping distance
     at full brake from 30 m/s stays within 10 % of v^2 / 18 (no lost braking).
- Videos: a rollout that rolls over at 15-19 s is the simulator, not the model. `events_rollouts.csv` (`roll_t`, `roll_pitch`) marks them.
