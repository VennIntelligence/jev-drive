# Is the p7 "spin / small circles" a sign or frame bug in the plan-position executor?

Question: decision 133 found that executing openpilot's plan positions (`lat_exec: p7`) on the B2D 25 junction turns takes 0-1 / 25 turns
and loses the car within 20 s of the route start in small loops ([plan_track.md](plan_track.md)). A GIF reading suggested "the plan bends left
while the car turns right". CARLA is left-handed (y right, yaw clockwise seen from above), so a sign flip between openpilot's frame and the
controller is the first suspect. The action-curvature executor (`curv`) does not go through this chain and works (3-7 / 25).

**Verdict: no sign or frame bug.** The executor turns the car the way the plan bends in 96-97% of moving frames on every p7 arm, the same as the
curv control; a synthetic left arc turns the CARLA car left, a right arc right, a straight line straight. The loops are the plan's own
curvature at crawl speed (median 0.21 1/m, R about 5 m, between 0.3 and 1.5 m/s), tracked faithfully. One real defect exists but is not a sign error
and does not produce the loops: at standstill the plan is shorter than 0.5 m in 31% of p7 frames and `place()` extends its last chord to the
governor's arc, so the pre-launch wheel angle follows noise, not the model's intent (below). Decision 133 need not be reopened on the sign question.

Code: [`scripts/p7_sign_check.py`](../scripts/p7_sign_check.py) (runs on the box over the d133 units; numbers in [`p7_sign_check.json`](p7_sign_check.json)),
unit test [`tests/test_p7_sign.py`](../../../tests/test_p7_sign.py) (`python -m unittest tests.test_p7_sign`, 5 tests, CARLA stubbed).

## Frame chain (plan -> CARLA), where signs flip

| step | code | frame in | frame out | sign handling |
|---|---|---|---|---|
| 1 | `zeroshot_rigs.openpilot_plan_to_rig` | openpilot calib frame at the camera: x fwd, **y right**, yaw **right-positive** | rig, rear axle: x fwd, **y left** | `q = (x, -y)`, `psi = -yaw`; `rear = d + q - R(psi) d` (camera at `d` = (1.59, 0)) |
| 2 | `b2d_zeroshot_agent.resample` to `TIMES` | rig | rig | none |
| 3 | `op_arb_agent.place([0,0] + op_path, s_fin)` (`lib/op_arb_agent.py:574-575`) | rig | rig | none (re-times the geometry by the arbitrated arc profile; extrapolates the last chord past the plan's end) |
| 4 | `Controller.update` / `_accept_pending` | rig at the source pose | controller odometry (y left, yaw left-positive) | rotation by the source pose only |
| 5 | `Controller.step(now, v, -world_gyro)` | IMU gyro z = +d(CARLA yaw)/dt (clockwise; calibrated r = 0.99995, `b2d_agent.py:328`) | yaw rate **left-positive** | **flip 1**: `-world_gyro` |
| 6 | pursuit: `curvature = 2 y / |aim|^2` (left-positive), `raw_steer = -angle / (max_steer * scale)` | left-positive curvature | CARLA steer **right-positive** | **flip 2**: `-angle` |
| (route) | `world_to_local(points, xy, yaw)` | CARLA world | rig | second axis `dx sin(yaw) - dy cos(yaw)` = projection on CARLA's left `(sin yaw, -cos yaw)`: **y left** |
| (curv) | `_curvature_steer`: `atan(L kappa)` with server curvature right-positive | right-positive | CARLA steer | none (`act_k` logged right-positive) |

Three flips (openpilot y, gyro, steer), each matched by its consumer. The controller + pose chain is the one that drove the route oracle to DS 81.9
in the P7 acceptance test (`experiments/b2d_tfv6/results/tfv6-controller/controller-scorecard.md`), so steps 4-6 were already checked in CARLA; step 1 is
the only part p7 adds, and the unit test checks it.

## (b) Unit test: synthetic arcs through the real chain

`tests/test_p7_sign.py` builds openpilot plans of a car whose rear axle drives a constant-curvature arc (camera track and yaw in openpilot's
y-right / right-positive frame), runs them through `openpilot_plan_to_rig` -> `resample` -> `place` -> the P7 `Controller` (P7.json, as the agent
loads it), and closes the loop on a kinematic plant in CARLA's left-handed world (positive steer = right, yaw clockwise, gyro = +d yaw / dt,
Ackermann inner-wheel command converted back to the bicycle centre angle).

| plan | rig y at 5 s | first CARLA steer | CARLA yaw over 6 s | end point in the start frame | measured / plan curvature |
|---|---|---|---|---|---|
| left arc, 0.05 1/m, 5 m/s | > 0 (left) | < 0 (left) | decreasing (= left in CARLA) | y > 2 m (left) | 1.24 |
| right arc, -0.05 1/m | < 0 | > 0 | increasing | y < -2 m | 1.24 |
| straight | 0 | 0 | |dyaw| < 1e-3 | |y| < 0.05 m | - |

The rear-axle chord curvature after step 1 equals the plan's curvature exactly (the `R(psi) d` term is right). The 24% overshoot is the P7
`rear_slip` correction, which adds the PhysX tyre slip that the kinematic plant does not have; sign and order are right.

## (a) Logs: plan side vs measured turn side

Per plan record (20 Hz) on the d133 units where openpilot owns lateral and is executed (`lat == "op"`, not warm; hyb only above its 3 / 2.5 m/s
switch; curv = the same records under the action-curvature executor, plan only logged). `y1` = plan lateral at 1 s (rig, y left);
yaw rate over the next 1 s from the truth pose, left-positive = -d(CARLA yaw)/dt; steer = mean CARLA steer over the next 0.5 s, sign flipped
to left-positive. Agreement = share of equal signs; frames with v > 0.5 m/s and |y1| > 0.05 m for yaw rate.

| arm / executor | n | sign(y1) = sign(yaw rate) | Pearson | sign(y1) = sign(steer) (n) |
|---|--:|--:|--:|--:|
| shipped p7 | 9 788 | 0.973 | 0.83 | 0.965 (20 153) |
| rc-ctl-s0 p7 | 9 271 | 0.973 | 0.84 | 0.982 (16 793) |
| rc-bear-s0 p7 | 6 728 | 0.958 | 0.81 | 0.958 (10 438) |
| rc-poly-s0 p7 | 6 279 | 0.961 | 0.84 | 0.975 (10 041) |
| shipped hyb (plan phase) | 879 | 0.935 | 0.87 | 0.952 (879) |
| **pooled p7** | **32 066** | **0.967** | **0.83** | 0.970 (57 425) |
| shipped curv | 2 982 | 0.908 | 0.78 | 0.879 (3 479) |
| rc-ctl-s0 curv | 4 624 | 0.982 | 0.83 | 0.979 (4 907) |
| rc-bear-s0 curv | 4 928 | 0.981 | 0.90 | 0.952 (5 378) |
| **pooled curv** | **12 534** | **0.964** | **0.85** | 0.943 (13 764) |

A sign bug in the p7 path would give agreement near 0 under p7 and near 1 under curv; both are 0.96-0.97. The path actually handed to P7
(`place(plan, s_fin)` at the 2 s arc) agrees with the yaw rate 0.965 (n 34 463, Pearson 0.88). The plan also agrees in sign with openpilot's
own action curvature (`act_k`) in 0.91 of p7 frames (0.93 curv), so step 1 does not flip the plan relative to the model's other head.

## What makes the loops

1. **The plan itself is tightly curved at crawl speed under p7, and the car follows it.** Plan curvature (chord to the 3 s point) and measured
   curvature (yaw rate / v), medians, pooled:

   | speed | p7 plan | p7 measured | p7 share R < 10 m (measured) | curv plan | curv measured |
   |---|--:|--:|--:|--:|--:|
   | 0.3-1.5 m/s (n 40 576 p7 / 13 039 curv) | 0.214 | 0.188 | 0.79 | 0.031 | 0.008 |
   | 1.5-3 m/s | 0.133 | 0.197 | 0.64 | 0.014 | 0.004 |
   | > 3 m/s | 0.079 | 0.123 | 0.55 | 0.004 | 0.002 |

   91% of p7 frames are below 1 m/s (curv 80%). Under p7 the creeping car drives the R of about 5 m the plan draws, sign-consistent in 95% of frames,
   and keeps doing so because the next plan is again curved: a loop. Under curv the plan at the same speed is 7x straighter (the car is still in
   its lane) and the action head is 5x weaker than under p7 (0.008 vs 0.038), so nothing loops. This is decision 133's mechanism (low-speed plan
   instability, as decision 118 on HUGSIM), now with the executor ruled out. Above 1.5 m/s the measured curvature is 1.5x the 3 s chord
   curvature; part is the chord metric (it averages a tightening plan), part the P7 rear-slip gain; it does not change the sign.

2. **Standstill plans steer by noise (a real executor defect, not a sign error).** At standstill the governor still asks to move (base profile
   about 1 m/s) while openpilot's plan is a sub-metre blob: plan arc at 5 s < 0.5 m in 31% of p7 frames, < 2 m in 48%. `place()`
   (`lib/op_arb_agent.py:139-151`, used at line 575) extends the last chord of that blob straight to the governor's arc, and P7 aims 3 m along it.
   The aim then has no relation to the model's intent: sign(aim) vs sign(action curvature) Pearson 0.03 for plans < 0.5 m (pooled n 57 509; the
   per-arm agreement swings from 0.19 on rc-poly to 0.94 on rc-ctl, i.e. set by a few long standstills), against 0.49 for plans >= 5 m. The
   blob's direction includes the rotation-about-the-camera term of step 1: a standstill plan whose yaw drifts 0.03 rad to the right moves the rear
   axle 5 cm to the LEFT, and its extrapolation points left across the road (`StandstillPlan` test). This sets the wheel angle before launch
   (steer beyond 0.7 of full lock in about 15% of the < 0.5 m standstill frames, local subset of the logs), but once the car rolls only 3% of frames have a plan under 2 m, so it is not what
   keeps the loops going.

3. **"Plan left, car right".** The published GIF (`figs/plan_track_rc-bear-s0_5423_turn0.gif`) draws no plan, only the chase view and the model
   input frames, so the picture cannot be read from it directly. In the logs the candidates are: the 3-4% of moving frames where plan and yaw rate
   disagree (steer-rate limit 2 /s while the plan reverses side between 0.2 s updates), the standstill extrapolation in point 2 (the camera-frame
   plan heads one way, the executed chord the other), and the plan vs the ROUTE: under p7 the plan's side agrees with the route's side at 10 m in
   only 0.56 of frames (curv 0.80; hyb 0.28, Pearson -0.46), i.e. once the car is off its lane the plan bends towards the road it now sees, not
   back to the route. None of these is a sign flip.

## Proposed fix (not applied; main decides)

Not needed for signs. For the standstill defect, a minimal change in `lib/op_arb_agent.py` `_plan` at line 574: execute the plan geometry only
when the plan is long enough to carry a direction, e.g.

```python
plan_ok = arc(np.r_[[[0.0, 0.0]], op_path])[-1] >= 2.0          # a sub-2 m standstill plan has no heading
geom = np.r_[[[0.0, 0.0]], op_path] if lat_src == "op" and plan_exec and plan_ok else bpath
```

(or straight ahead instead of `bpath` if route geometry at launch is not wanted). It would be a declared executor change, and it would not remove
the crawl-speed loops of point 1; it is not worth a 25-turn rerun on its own.

## Bearing on decision 133

Its result stands and the sign question is closed: p7 executes the plan with the right sign. Its stated mechanism ("low-speed plan tracking is
unstable") can be sharpened: at crawl speed the plan itself has a 5 m radius under p7 and the executor reproduces it; the start-of-route loss is
the model's plan, plus a pre-launch wheel angle set by extrapolated standstill noise. Caveats: correlations pool autocorrelated 20 Hz frames
(n overstates the independent evidence; per-arm results agree); the plan curvature uses the logged 1/2/3/5 s points only; the plant in the unit
test is kinematic (no tyre slip, no steering lag).
