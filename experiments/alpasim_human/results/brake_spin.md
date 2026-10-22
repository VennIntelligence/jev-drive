# Brake spin of the hand mapping, and the track_ref gate as the fix (2026-10-22)

**Question.** In the user's drives a straight full brake at speed spun the car (34.6 m/s: yaw rate -1.6 rad/s with a planned
road-wheel angle of 0.001 rad). Which decelerations and onsets spin, why, and what should the mapping do?

**Answer.** The official nonlinear MPC steers when the plan asks for a deceleration step it cannot follow: its acceleration
command saturates at its -9 m/s^2 bound and the only remaining way to shed the longitudinal position error is heading. The
direction is random (tiny asymmetries pick it), so it is not a bias. The hand plan (8 m/s^2 + drag, step onset) is far inside
the spin region at every speed >= 15 m/s. Passing the hand plan through the model drivers' gate (`experiments/alpasim/lib/track_ref.py`,
mode gate: -5 m/s^2, jerk 10 m/s^3, decision 309) removes it: MPC steering <= 0.002 rad, no spin, no drift, at 15-33 m/s.
Confirmed in the full stack on card 1 (30 m/s step full brake: off spins, 0.81 rad MPC steer; on: straight, < 0.001 rad). The price is braking: a full brake from 30 m/s stops in 81 m offline / 91 m in the stack instead of the mapping's 50 m. Now the default
(`HUMAN_TRACK_GATE=1`, lib/human_driver.py).

## Evidence in the user's drives

`exam_pai_1022-154753` and the two 10-13 sessions, braking events (brake > 0.3, start > 8 m/s), heading change minus the plan's:

| Scene | v at onset m/s | Brake max | Plan accel min | Excess heading rad | Lateral at end m | Peak excess yaw rate |
|---|--:|--:|--:|--:|--:|--:|
| 49597f01 | 34.6 | 1.00 | -9.17 | -1.19 | -13.6 | -1.61 |
| 9ea70552 | 17.9 | 1.00 | -8.50 | -0.99 | -7.9 | -1.59 |
| 071d15c4 | 18.8 | 1.00 | -8.49 | +0.44 | +3.7 | +0.79 |
| 4f779a92 (10-13) | 26.6 | 0.78 | -6.89 | -0.01 | -0.6 | -0.04 |
| 49597f01 (10-13) | 29.3 | 0.69 | -5.95 | +0.04 | +2.8 | +0.06 |
| 6 more (brake 0.35-0.91) | 8.9-23.4 | <= 0.91 | >= -7.52 | within +-0.16 | | within +-0.35 |

The controller's own log (`sim/controller/*.csv`) for 49597f01: steering command -0.14, -0.30, -0.33, -0.36 rad in the 0.4 s after
the brake, while the acceleration command went from +1.2 to -9.0 (the bound) and stayed there. Signs differ between events.

## Offline probe: the controller's own code, no renderer

`scripts/brake_probe.py` runs `alpasim_controller.system.System` (the controller service's class; nonlinear MPC, the stack's
config: horizon 20 x 0.1 s, gains long 2 / lat 1 / heading 1 / accel 0.1, accel bound [-9, 6]) in a closed loop against the human
driver's own plan (`rollout`, 41 poses at 0.1 s, re-planned every 0.1 s from the current pose), as the runtime calls it. The fed-back
pose gets a small perturbation (yaw N(0, 1 mrad), lateral N(0, 1 cm) per step) standing in for the physics ground correction: with an
exactly symmetric problem the MPC never steers. Each case cruises 3 s, then brakes with brake input b(t) = min(1, t / ramp),
plan accel = -cap x b - drag(v). 720 cases (5 speeds x 6 caps x 4 ramps x 3 plan curvatures x 2 seeds); Tokyo
`/data/runs/alpasim_human/test/brake_probe/final/cases.jsonl`.

Peak |yaw rate - plan yaw rate| (rad/s), max over 6 runs; S = spin (> 0.5), K = kink (0.1-0.5):

| v m/s | onset | cap 2 | cap 3 | cap 4 | cap 5 | cap 6 | cap 8 (current) |
|--:|---|--:|--:|--:|--:|--:|--:|
| 10 | step | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | K 0.35 |
| 15 | step | 0.00 | 0.01 | 0.01 | 0.01 | K 0.27 | S 2.02 |
| 15 | 0.6 s | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | K 0.15 |
| 20 | step | 0.01 | 0.02 | 0.03 | 0.08 | S 0.93 | S 2.16 |
| 20 | 0.6 s | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | S 0.98 |
| 20 | 1.0 s | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | K 0.37 |
| 25 | step | 0.02 | 0.04 | K 0.16 | S 0.63 | S 1.18 | S 2.19 |
| 25 | 0.3 s | 0.01 | 0.01 | 0.02 | 0.03 | 0.07 | S 1.57 |
| 25 | 1.0 s | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | S 0.87 |
| 33 | step | 0.05 | K 0.17 | S 0.59 | S 0.95 | S 1.50 | S 2.50 |
| 33 | 0.3 s | 0.02 | 0.03 | 0.05 | 0.08 | K 0.39 | S 1.84 |
| 33 | 0.6 s | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | S 1.47 |
| 33 | 1.0 s | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | S 1.25 |

- With a step onset the safe cap falls with speed (6 at 10 m/s, 4 at 20-25, 2-3 at 33). A 0.6 s onset makes caps up to 6
  clean at every speed. Cap 8 spins from 15 m/s whatever the onset, because 8 + drag + the catch-up for the 0.1 s actuator lag
  puts the MPC at its -9 bound (MPC accel command: cap 6 / 0.6 s at 33 m/s -8.3; cap 8 always -9.0).
- Drift ("kink") and spin are the same instability: a short yaw-rate pulse at the onset leaves a heading error that stays,
  because the next plan starts from the car's new heading. Below the spin threshold it shows as a heading kink of 0.04-0.25 rad
  and metres of lateral drift (e.g. 33 m/s, cap 4, step: 0.14 rad, 7.6 m).
- Direction: random. Same cell, different noise seed: opposite signs (25 m/s, cap 8, step: heading error +0.91 / -0.75 rad,
  drift +18.2 / -14.6 m). Without noise, a plan steering of +-0.0005 rad decides it, mirror-symmetric (+0.0005 -> -9.7 m,
  -0.0005 -> +9.7 m at 25 m/s, cap 8). No systematic bias; with no braking the drift is below 0.6 m at any speed.
- Not the solver budget: with IPOPT max_iter 300 instead of 30 the same cases spin (the steering is the optimum). Gains:
  acceleration_weight 0 or rel_acceleration_weight 0.1 remove the cap-6 spins at 33 m/s, lowering long_position_weight does not;
  the spin goes with the acceleration command reaching the bound.

## The gate (HUMAN_TRACK_GATE=1)

`lib/human_driver.py` hands its 41-pose rollout (8 points at 0.5 s, rig frame) to `track_ref.build(..., "gate", a0)` with the
measured acceleration a0 (DynamicState), and serves the rebuilt reference: deceleration >= -5 m/s^2 reached with jerk <= 10 m/s^3
from a0, path unchanged. Step full brake (cap 8 + drag), 3 seeds x 2 plan curvatures (0, 0.0005 rad):

| Gate | v m/s | Spin | Max MPC steer rad | Peak excess yaw rate | Drift at stop m (straight) | Stop distance m | Mapping's own (v^2 / 2 (8.25 + 0.0008 v^2)) |
|---|--:|--:|--:|--:|--:|--:|--:|
| off | 15 | 6/6 | 0.77 | 2.02 | -2.4 .. +2.3 | 14.8 | 13.3 |
| off | 25 | 6/6 | 0.68 | 2.19 | -18.0 .. +18.2 | 38.0 | 35.7 |
| off | 30 | 6/6 | 0.74 | 2.39 | -33.4 .. +33.6 | 53.8 | 50.2 |
| off | 33 | 6/6 | 0.77 | 2.50 | -42.4 .. +42.6 | 64.6 | 59.7 |
| -5 (default) | 15 | 0/6 | 0.001 | 0.00 | -0.0 .. +0.2 | 22.5 | 13.3 |
| -5 (default) | 25 | 0/6 | 0.001 | 0.00 | -0.1 .. +0.6 | 57.7 | 35.7 |
| -5 (default) | 30 | 0/6 | 0.002 | 0.00 | -0.2 .. +0.8 | 81.3 | 50.2 |
| -5 (default) | 33 | 0/6 | 0.002 | 0.01 | -0.2 .. +0.9 | 97.3 | 59.7 |
| -7 | 30 | 0/6 | 0.002 | 0.00 | -0.1 .. +0.6 | 63.4 | 50.2 |
| -8 | 30 | 0/6 | 0.002 | 0.00 | -0.1 .. +0.6 | 58.7 | 50.2 |
| -8 | 33 | 0/6 | 0.002 | 0.00 | -0.2 .. +0.7 | 70.0 | 59.7 |
| -8.5 | 33 | 0/6 | 0.024 | 0.10 | -0.2 .. +0.7 | 69.4 | 59.7 |
| -8, jerk 20 | 25-33 | 10/18 | 0.29 | 0.88 | | | |
| -7, jerk 40 | 33 | 3/6 | 0.18 | 0.55 | | | |

- The gate passes the "MPC steer <= 0.1 rad, no spin" criterion at 15-33 m/s with amin down to -8.
- Stopping distance is not within 10 % of the mapping's: the MPC settles at -6.15 m/s^2 under the -5 reference (the reference is
  re-anchored to the car every 0.1 s), and the 10 m/s^3 onset costs ~0.5-0.8 s. At 30 m/s: 81 m (+62 %) with the default, 59 m
  (+17 %) with amin -8. The jerk limit is what prevents the spin: 20 m/s^3 or more spins again.
- Option, not applied: `JEV_TRACK_CFG=amin=-8` (passed into the driver container) for 0.8 g braking at +17 % distance.

## Full stack on card 1

Scene 49597f01 (highway), the real stack (`human_stack.sh`, renderer and physics on card 1, `HUMAN_PREFIX=probe`, real time),
`fake_wheel.py --brake-at 30`: full throttle, then a step to full brake (1.0) once the driver reports 30 m/s, held to a stop.
Same scene and script, gate off and on (Tokyo `/data/runs/alpasim_human/test/brake_stack/g0`, `g1`):

| Gate | v at onset | Peak yaw rate | Heading change | Lateral at stop | Max MPC steer | Min MPC accel | Stop distance | Mapping's own |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| off | 30.0 m/s | 2.48 rad/s | 2.15 rad | 43.3 m | 0.81 rad | -9.0 | 62.0 m | 50.3 m |
| on (default) | 30.1 m/s | 0.00 rad/s | 0.00 rad | -0.01 m | < 0.001 rad | -6.16 | 90.7 m | 50.4 m |

The step brake spins in the stack as in the user's drives and offline; with the gate the car stops straight. The stack's stop
distance (90.7 m, from the last driver call before the brake) is longer than offline (81.3 m) by about the driver's input
latency at 30 m/s; it is 80 % longer than the mapping promises, the open trade-off above.

## Files

- `scripts/brake_probe.py`: offline probe (controller code, no renderer). `scripts/fake_wheel.py --brake-at V`: scripted full
  brake at a speed. `scripts/human_stack.sh`: `HUMAN_ALLOW_CARD1`, `HUMAN_PREFIX`, `HUMAN_JIT_DIR` for probes beside the user's
  stack, `HUMAN_TRACK_GATE` (default 1).
