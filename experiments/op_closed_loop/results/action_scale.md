# Is the action head's ~0.45 curvature gain real? Decode, target and controller checked against the openpilot source and real logs

Written 2026-10-05. Question (user: "maybe our controller is the problem"): decoded as `action[0] / max(1, v)^2`, the shipped Cinque action head gives
~0.45 of the curvature the car drove (rft.py `ACT_ALPHA`, fitted on the op_adapt_H teachers against the 1 s pure-pursuit curvature: nav -0.51,
wod -0.39), and decision 120 read a flat ~0.5 at every junction radius in B2D. Three candidates: (a) our decode is off, (b) the comparison target
is wrong, (c) the car-side controller makes the car follow more than the desired curvature.
Code: `experiments/op_closed_loop/scripts/action_scale_{replay,report}.py`. Data on the box: `$DATA_DIR/runs/op_closed_loop/action_scale/`
(windows, per-window replays, `bank.npz`, `comma.npz`, `action_scale_summary.json`). Figure `../figs/action_scale.png`. CPU only (all three cards
were leased to `rft-pt`): Cinque on onnxruntime CPU, 3 processes x 10 threads.

![action scale](../figs/action_scale.png)

What to look at: left, comma1M frames (native comma rig, 32 windows, v > 3 m/s): the points lie on the dashed gain-1 line up to |k| ~0.05 1/m
(R > 20 m) and fall below it only in the slow sharp turns (|k| > 0.1, junctions at 3-8 m/s), which pull the pooled fit to 0.63; right, the
pooled slope and r against the time offset of the measured curvature (r peaks at d 0.25-0.3 s, the model's own action_t 0.275 s).

## Answer

**Neither (a) nor (c). The ~0.45 is mostly (b), a wrong comparison, plus one real property of the head in slow sharp turns.**

- (a) The decode is exactly modeld's (same output positions, units, time, sign; section 1-2). On the native comma rig the decoded curvature
  matches the curvature the car drove 0.25-0.45 s later with gain **0.97-1.04 above 8 m/s** (curves and sharp turns alike, r 0.97-0.99). A 2x
  decode error would show there; it does not. **No decode correction.**
- (c) No openpilot lateral controller has a gain between desired and achieved curvature: torque / angle / curvature controllers all
  drive the yaw-rate-calibrated curvature to the desired one after `lateralDelay` (unity, integrator or exact inverse model). `lib/op_ctrl.py`
  (clip + pure delay, gain 1) is the right model of the car side; no change.
- (b) The 0.45 combines three things that are not the head: the 1 s pure-pursuit target (WOD g_cal 0.46 -> 0.71 against the
  instantaneous curvature at t + 0.275; comma1M pooled 0.37 -> 0.69), the regression direction (decoded on measured shrinks a noisy
  predictor), and the camera-height scale of the open-loop boards: the model sees its own speed at 0.92 (WOD) / 0.81 (navtrain) of the true
  speed (decision 104), `action[0]` is a lateral acceleration in that perceived speed, and dividing by the true v^2 shrinks the curvature by
  0.84 / 0.66. Decoded with the model's own plan speed the WOD / navtrain heads are calibrated (g_cal 0.96 [0.90, 1.02] / 1.10 [1.06, 1.14]).
- The real part: in **slow sharp turns on the native rig (v 3-8 m/s, |k| > 0.02 1/m, 463 frames, 7 windows: junction turns a human drove)
  the head gives 0.65 [0.59, 0.87] of the driven curvature**, while the plan-derived curvature gives 0.98 [0.92, 1.10]. No time shift fixes it
  (0.66 at t + 0 to 0.89 at t + 1.5 s). This is the same under-turn as decision 120's B2D ~0.5 (B2D adds its own speed scale, 0.95^2 at 1.433 m,
  0.87^2 = 0.76 at the `spec` 1.86 m camera), so the junction under-turn is a property of the action head in slow sharp turns, present on real
  data with openpilot's own camera, and every closed-loop junction result (d118, d121, d122, d127-129) executed it faithfully.

## 1. openpilot source (master ec95db3f, opendbc 35f7e081; sparse clone `tmp/opsrc/`)

| step | what the source does | file:line |
|---|---|---|
| output slicing | `slice_outputs`: `model_output[np.newaxis, output_slices[k]]`, slices from the compiled model's metadata `output_slices` | selfdrive/modeld/modeld.py:150, 188-189, 212 |
| `action` parsing | **none**: `Parser.parse_outputs` parses plan / lead / lanes etc. as MDN, `action` is passed through raw | selfdrive/modeld/parse_model_outputs.py:95-120 |
| desired curvature | `desired_curvature = model_output['action'][0,0] / max(1.0, v_ego)**2`, `desired_accel = action[0,1]`; `smooth_value(.., LAT_SMOOTH_SECONDS = 0.0)` (no smoothing) above `MIN_LAT_CONTROL_SPEED` 0.3 m/s, held below | modeld.py:46-48, 52-75 |
| time | the model is told the time: `action_t = [lat_action_t, long_action_t]`, `lat_action_t = lateralDelay + DT_MDL (frame delay) + DT_MDL/2` = 0.275 s at the lagd default 0.2 s; `v_ego = carState.vEgo` (wheel speed) | modeld.py:365-366, 396-403 |
| controlsd | `new_desired_curvature = modelV2.action.desiredCurvature` when latActive (else the measured curvature), then `clip_curvature` (jerk 5 m/s^3 / max(v,1)^2, lateral accel 3 m/s^2, abs 0.2); no gain | selfdrive/controls/controlsd.py:120-133; controls/lib/drive_helpers.py:28-42 |
| torque cars | PID in lateral-accel space: setpoint = desired_curvature v^2 delayed by `lateralDelay` (request buffer), measurement = `-VM.calc_curvature(steeringAngle - offset, v, roll) * v^2`; feed-forward = desired lateral accel (minus roll, latAccelOffset, plus friction); integrator (frozen below 5 m/s); output converted to torque with the online `latAccelFactor` | controls/lib/latcontrol_torque.py:59-108 |
| angle cars | `angle = VM.get_steer_from_curvature(-desired_curvature, v, roll) + angleOffset`: open-loop inverse of the vehicle model | controls/lib/latcontrol_angle.py:16-25 |
| curvature cars | PID on `desired - actual`, feed-forward `kf * desired` (kf 1 without a PID) | controls/lib/latcontrol_curvature.py:33-55 |
| calibration of the measurement | paramsd fits steerRatio / stiffness / angle offset so that the steering-angle curvature matches the yaw rate; torqued fits `latAccelFactor` from (torque, `v * yaw_rate - roll g`) | selfdrive/locationd/paramsd.py:29-79; locationd/torqued.py:177-207 |
| lagd | correlates `desiredCurvature * v^2` with `yaw_rate * v` (same sign, same units), keeps only points with `abs(desired - actual) <= 0.6 m/s^2`, delay clipped to [0.15, 0.65] s | selfdrive/locationd/lagd.py:24-34, 283-290 |

Reading: every controller drives the car's curvature (steering-angle curvature calibrated to yaw rate / v) to the desired curvature after
`lateralDelay`, with unit steady-state gain (integrator or exact inverse model, both calibrated online to the yaw rate). Nothing on the car side
multiplies the desired curvature; lagd's own validity gate (desired and actual lateral acceleration within 0.6 m/s^2) assumes they agree.
Sign: openpilot curvature + = right (calibrated frame, z down), consistent with `yaw_rate` in lagd.

## 2. Our decode against the model's metadata

Cinque (`cinque.onnx`, checkpoint `b9facbcc-.../12864`) `output_slices`: `action` = [2062, 2066) (4 values), `plan` [917, 1907),
`hidden_state` [2066, 18450). openpilot reads `action[0,0] = out[2062]`, `action[0,1] = out[2063]`; our `decode()` takes `mdn_mu(out[2062:2066])`
= `out[2062:2064]`, the same two numbers. Units, slice, time and sign of `decode()` match modeld exactly; `lib/op_ctrl.py` matches controlsd
(VERBATIM clip_curvature) and models the car as the pure `lateralDelay`, which is what the source's controllers aim at. **(a) is ruled out on
the code side.**

## 3. Measured curvature vs decoded

Gains are reported two ways. `g_reg` = slope of decoded on measured (through the origin), what rft.py's 0.45 and decision 120's binned ratio
measure; `g_cal` = 1 / slope of measured on decoded, the gain of a predictor in the calibration sense (a calibrated, noisy predictor has
g_cal = 1 and g_reg < 1: regressing a prediction on the outcome shrinks by the unexplained variance). The action head predicts curvature
0.275 s ahead from images, so its g_reg is below its true gain by construction; g_cal is the number to compare with 1. 95% CIs: cluster
bootstrap (comma1M: window; WOD / nav: scene cluster; B2D: run). v > 3 m/s everywhere.

### 3a. comma1M: native comma 3/3X rig, real 20 Hz frames (the only local real data with openpilot's camera)

32 windows of 10 s (6 s warm-up before each) from 32 segments with both cameras at 1928x1208, picked by peak yaw rate (8 straight, 12 curve,
12 junction-like); 6 348 frames with v > 3 m/s. Measured curvature = localizer yaw rate in the calibrated frame (smoothed over 0.25 s) / speed.
Model speed (plan velocity at t = 0) / localizer speed: median 1.004 (no scale error on the native rig).

| target / subset | n (windows) | g_reg [95% CI] | g_cal [95% CI] | r |
|---|---|---|---|---|
| pooled, yaw rate / v at t + 0 | 6348 (32) | 0.61 [0.56, 0.85] | 0.68 [0.61, 0.92] | 0.94 |
| pooled, at t + 0.2 | 6348 (32) | 0.63 [0.57, 0.90] | 0.69 [0.62, 0.93] | 0.96 |
| pooled, at t + 0.3 (max r) | 6348 (32) | 0.63 [0.58, 0.91] | 0.69 [0.63, 0.94] | 0.96 |
| pooled, at t + 1.0 | 6348 (32) | 0.60 [0.52, 0.90] | 0.77 [0.69, 1.11] | 0.88 |
| pooled, 1 s pure pursuit (rft.py's target) | 6347 (32) | **0.32** [0.30, 0.51] | 0.37 [0.33, 0.58] | 0.93 |
| pooled, plan-derived curvature vs t + 0.2 | 6348 (32) | 0.97 [0.91, 1.06] | 0.98 [0.92, 1.08] | 0.99 |
| pooled, action / max(1, v_plan)^2 vs t + 0.2 | 6348 (32) | 0.75 [0.57, 0.96] | 0.86 [0.66, 1.01] | 0.93 |
| v > 8, \|k\| 0.003-0.02 (curves) | 1354 (21) | 0.98 [0.87, 1.05] | **1.04** [0.95, 1.09] | 0.97 |
| v > 8, \|k\| > 0.02 (sharp) | 299 (7) | 0.96 [0.93, 1.01] | **0.97** [0.94, 1.04] | 0.99 |
| v 3-8, \|k\| 0.003-0.02 | 239 (9) | 0.87 [0.58, 1.13] | 1.11 [0.69, 1.28] | 0.89 |
| v 3-8, \|k\| > 0.02 (junction turns) | 463 (7) | 0.61 [0.55, 0.85] | **0.65** [0.59, 0.87] | 0.96 |
| straight, \|k\| < 0.003 | 3993 (32) | 1.07 [0.93, 1.21] | 1.75 [1.41, 2.24] | 0.78 |

(plan-derived g_cal in the four speed x curvature cells: 1.00 / 1.01 / 1.01 / 0.98. On straights g_cal is dominated by lane-centring noise
around 0 and is not a gain. All rows: `action_scale_report.py report`.)

### 3b. op_adapt_H teachers (the data ACT_ALPHA was fitted on): WOD (camera 1.81 m) and navtrain (1.87 m above the road)

Logged future at 0.25 s steps (rear axle); instantaneous curvature from a cubic fit through 0-1.5 s.

| board / target | n (clusters) | g_reg [95% CI] | g_cal [95% CI] | r |
|---|---|---|---|---|
| WOD, 1 s pure pursuit (rft.py's target) | 1057 (780) | **0.39** [0.35, 0.42] | 0.46 [0.43, 0.48] | 0.92 |
| WOD, instantaneous at t + 0 | 1060 (782) | 0.54 [0.50, 0.58] | 0.63 [0.60, 0.68] | 0.92 |
| WOD, instantaneous at t + 0.275 | 1060 (782) | 0.60 [0.55, 0.65] | **0.71** [0.67, 0.76] | 0.92 |
| WOD, instantaneous at t + 0.5 | 1060 (782) | 0.63 [0.57, 0.69] | 0.78 [0.72, 0.85] | 0.90 |
| WOD, decoded with the model's own speed, action / max(1, v_plan)^2, vs t + 0.275 | 1060 (782) | 0.82 [0.76, 0.89] | **0.96** [0.90, 1.02] | 0.93 |
| WOD, plan-derived curvature (get_curvature_from_plan) vs t + 0.275 | 1060 (782) | 0.94 [0.89, 0.99] | 1.00 [0.95, 1.05] | 0.97 |
| nav, 1 s pure pursuit (rft.py's target) | 899 (571) | **0.51** [0.49, 0.54] | 0.60 [0.58, 0.63] | 0.92 |
| nav, instantaneous at t + 0.275 | 900 (572) | 0.51 [0.49, 0.54] | **0.60** [0.58, 0.63] | 0.92 |
| nav, decoded with v_plan, vs t + 0.275 | 900 (572) | 0.97 [0.92, 1.01] | **1.10** [1.06, 1.14] | 0.94 |
| nav, plan-derived vs t + 0.275 | 900 (572) | 0.89 [0.87, 0.91] | 0.96 [0.93, 0.98] | 0.96 |

Model speed (plan velocity at t = 0) / logged speed: WOD median 0.917, nav 0.814 (decision 104: the camera height makes the model see the
world at ~0.7 scale on NAVSIM). `action[0]` is a lateral acceleration in the model's own perceived speed: a = kappa * v_model^2; dividing by
the true v^2 shrinks the curvature by (v_model / v)^2 = 0.84 (WOD) / 0.66 (nav). With the model's own speed the head is calibrated (g_cal 0.96 /
1.10), and the plan-derived curvature is calibrated too (1.00 / 0.96).

### 3c. B2D (CARLA, `drive` arms eval-drive-s0*, camera 1.433 m), ticks where the route geometry steers

`act_k` (logged decoded curvature) vs the car's curvature from the CARLA ground-truth yaw at t + 0.2 s, only on ticks with `lat == route`
(the car's motion does not come from the action head; mostly command zones, i.e. decision 118/120's counterfactual).

| subset | n (runs) | g_reg [95% CI] | g_cal [95% CI] | r |
|---|---|---|---|---|
| all | 17665 (150) | 0.53 [0.50, 0.57] | 0.70 [0.68, 0.73] | 0.87 |
| command zone | 13983 (96) | 0.54 [0.51, 0.57] | 0.68 [0.65, 0.71] | 0.89 |
| divergence fallback | 3682 (105) | 0.53 [0.20, 0.94] | 1.08 [0.92, 1.37] | 0.68 |
| R < 50 m | 5608 (95) | 0.54 [0.50, 0.57] | 0.69 [0.66, 0.71] | 0.88 |

Model speed / true speed on B2D: 0.945 (`drive`, 1.433 m), 0.96-0.97 (1.22 / 1.433 m arms of `rig122`), **0.87** at the open-loop-aligned `spec`
camera (1.86 m, `rig122` `ol*` arms). In command zones the head is reading a turn it was not asked for (decision 121: alone it goes straight
2/38 times right), so g_cal 0.68-0.70 there mixes "doesn't know the exit" with scale; outside zones (divergence fallback) it is 1.08 [0.92, 1.37].

## 4. Which of (a) / (b) / (c)

| candidate | verdict | numbers |
|---|---|---|
| (a) decode / slice / units / time / sign | **no** | same output positions as modeld (`action` [2062, 2066), first two values, no MDN parsing, no scale); native rig v > 8 m/s g_cal 0.97-1.04, r 0.97-0.99, best time offset 0.3-0.45 s; correction size **0** |
| (b) comparison target | **yes, most of the 0.45** | 1 s pure pursuit vs instantaneous curvature at t + 0.275: WOD 0.46 -> 0.71, comma1M 0.37 -> 0.69; board camera scale (v_model / v)^2: WOD 0.84, nav 0.66, B2D 0.89 (1.433 m) / 0.76 (1.86 m); model-speed decode on WOD / nav: 0.96 / 1.10 |
| (c) car-side controller gain | **no** | torque PID with integrator on yaw-rate-calibrated curvature, angle = exact inverse model, lagd / torqued / paramsd all calibrate to yaw rate / v; effective gain 1 after `lateralDelay` |
| head's own under-turn | **yes, slow sharp turns only** | native rig v 3-8 m/s, \|k\| > 0.02: 0.65 [0.59, 0.87] (plan 0.98); B2D junction readings (d120 ~0.5) are this times the board's speed scale |

Consequences.
1. Closed loop: `decode()` and `lib/op_ctrl.py` stay as they are; the junction results executed what openpilot would execute. On boards whose
   camera height shrinks the model's perceived speed, the executed curvature is additionally (v_model / v)^2 smaller than on a car (B2D `spec`
   1.86 m: 0.76); that is a property of the board's rig (decision 104), labelled, not a controller fix.
2. rft.py `ACT_ALPHA = 0.45` against the 1 s pure-pursuit curvature is not "the shipped head's own convention": against the instantaneous
   curvature at t + action_t the head's convention is 1.0 on the native rig. With 0.45 x pure pursuit as the action target the fine-tune
   asks for roughly half the curvature the shipped head already gives at speed, and keeps the junction under-turn as a target. The consistent
   target is action[0] = kappa_inst(t + 0.275) * max(1, v_model)^2 (v_model = the model's own plan speed, = v on a correctly mounted camera),
   gain 1, right-positive.

## Caveats

- comma1M drives are by humans (possibly with openpilot engaged, model and car unknown); the head is compared with what the car did, not with a
  logged command. No rlog with `modelV2.action`, `carControl` or `lateralDelay` exists locally, so the car-side gain is from the source only.
- WOD / nav targets come from 4 Hz / 2 Hz logged poses (curvature by a cubic fit); the op_adapt_H teacher runs the model on the H trunk bank
  (10 images per sample), not a 20 Hz replay.
- B2D readout is restricted to route-steered ticks; it inherits decision 120's counterfactual in command zones.
