# openpilot interface: openpilot as on the car, per-board deviations

Read this when you run openpilot (shipped Cinque or a fine-tune) on any board, write a harness for a new board, or quote
an openpilot number. One spec, `jevdrive/openpilot/interface.py`, says what openpilot does on a comma device in a car;
every board harness resolves what it actually runs against it and writes `interface.json` (resolved values + the
declared deviations) into every run dir. A deviation that is not declared for that board stops the run. Results:
[experiments/leaderboard_audit/results/unified_interface.md](../experiments/leaderboard_audit/results/unified_interface.md).
Tests: `python -m unittest tests.test_openpilot_interface -v`.

## The spec (openpilot as on the car)

| Key | Spec value | Source |
|---|---|---|
| `rig.height_m` | 1.22 m, behind the windshield | openpilot training rig; decision 104 (height drives the 0.67-0.70 scale error), 108 |
| `rig.level` | calibrated frame = road frame, pitch 0 | road f 910 px, horizon row 47.6; wide f 455 px, 58.7 deg, horizon row 151.8 (`jevdrive/openpilot/frames.py`) |
| `rig.wide` | one wide camera (120 deg fisheye) warped to 58.7 deg | comma 3X; road and wide offset not emulated (user's car ran one camera cropped twice) |
| `history.frames`, `history.rate_hz`, `history.clock` | real frames at 20 Hz, model clock = wall clock | modeld |
| `history.warmup` | real frames (the car stands with openpilot on) | - |
| `lateral.source` | action head: desired curvature = action[0] / max(1, v)^2, held below 0.3 m/s | modeld, openpilot ec95db3f (decision 118) |
| `lateral.exec` | `op-path`: controlsd latActive (v > 0.3), `clip_curvature` (jerk 5 m/s^3 / max(v,1)^2, lateral accel 3 m/s^2, abs curvature <= 0.2), then the car realises it after `lateralDelay` | `lib/op_ctrl.py` (VERBATIM functions), decision 118 |
| `lateral.delay_s` | 0.2 s (model time) | lagd initial value; HUGSIM 0.25 simulator s = 0.2 model s under the 1.25 clock dilation |
| `lon.source` | action[1] -> LongControl | decision 119; **no board runs it** (it fails launches), see deviations |
| `command.channel` | none: nothing tells the model where to go; desire only for a driver-initiated lane change. **desire is not a choice signal** (decisions 92, 121) | - |
| `command.route_geometry` | none | - |
| `light.source` | none: the model sees the light as pixels; a VLM reading pixels is a declared deviation | - |
| forbidden | `"resume": "nored"` (a ground-truth light holds the stop latch), on every board | ledger 2026-10-05 section 8 |

## Per-board resolution (default presets)

| Board | Harness | Default preset | Deviations (privilege) |
|---|---|---|---|
| NAVSIM navtest / navhard | `scripts/op_lb.py` (+ `op_interp` export) | true CAM_F0 height | height 1.87 m (LB); wide cut from CAM_F0 (LB); history = GIMM synthesis of 2 Hz keyframes (LB, decision 116); plan submitted, scorer LQR drives (LB); command none in the headline (desire schedules only in named arms); lever / retime adapters, selector when named (semi) |
| WOD-E2E | `scripts/wod_zeroshot_openpilot.py` | true roof height | height 1.81 m (LB); wide stitched from 3 cameras (LB); real 10 Hz frames fed twice, 10 s real warm-up (LB); open loop (LB); x1.06 longitudinal stretch only on the test submission (semi, decision 107) |
| HUGSIM | `experiments/hugsim/lib/zs_agent.py` via `experiments/hugsim/archive/zs_run.py --preset` | `spec` | dataset camera heights 1.2-1.8 m, KITTI-360 pitch (LB); wide stitched (LB); 4 Hz renders on a dilated clock (LB); 5 s static warm-up on the first frame, no real history exists (LB); iLQR tracks the plan's speed (LB, decision 119); simulator command -> turn desire (LB); forward_only / straight_stop / initial speed 1.0 m/s (semi) |
| B2D / CARLA | `lib/op_arb_agent.py` via `experiments/op_closed_loop/archive/op_arb.sh` arm `spec` | `spec` | camera at the open-loop boards' viewpoint, x 1.59 m, z 1.86 m (LB-aligned, see below); our longitudinal scheduler (set speed 8 m/s + curvature cap, lead-head IDM, plan while rolling, stop latch, timer resume) (semi); route turn desire (semi); **`DRIVE_ZONES` dense-route geometry steers in command zones and on divergence (semi fallback, not an openpilot capability)**; coast_v 2.5 (semi) |

B2D reporting rule: every B2D row carries the action-only reading next to it (`"zones": false, "div_m": 1e9`, arm suffix
`nz`); decision 121: with the zones off, 2 of 38 choice turns go the right way.

Camera on B2D (2026-10-05, user decision): the `spec` camera is the viewpoint of the open-loop boards, not openpilot's 1.22 m and not a B2D special case. The
real-data rigs (NAVSIM, WOD) cannot be moved, so B2D is aligned to them and the fine-tune is to teach the model that viewpoint on every board. Extrinsics in the
rear-axle frame, level after openpilot's calibration: NAVSIM CAM_F0 x 1.665 m, y -0.019, z 1.862 m above the road (40 frames; ego origin 0.35 m above the road + 1.512;
real pitch -1.3 deg); WOD front x 1.519 m, y +0.026, z ~1.86 m (1984 segments; z 1.8065 above the ground origin; pitch -0.2 deg). `spec` = the mean, **(1.59, 0, 1.86)**
(`interface.B2D_MOUNTS["openloop"]`); the mean so that neither board is privileged. The camera sits ~0.4 m above the MKZ roof and 2.2 m behind the bumper, no body in
frame. Named presets: `spec_bumper122` (1.22 m at the front bumper line, x 3.8 m: decision 125's `spec`, a declared trick), `spec_windshield143` (1.433 m, 1.779 m
behind: the legacy `drive` camera). Decision 125 (6 routes): 1.22 m beat 1.433 m with the zones off (5/7 turns vs 0/7); the junction rerun at the aligned camera is
`experiments/op_closed_loop/results/junction_rig122.md`. Privileged / vmerge arms still assume x 1.779 m and keep the `drive` preset. NAVSIM and WOD keep their true
heights; a virtual 1.22 m camera costs -1.97 PDMS (navtrain) and -0.49 RFS (WOD rater frames) open loop.

HUGSIM warm-up and clock: `spec` keeps the 5 s static warm-up and the dilate clock (= decision 118's arm). Without the warm-up the car does
not launch (11 scenes: mean HD 0.358 vs 0.544, 3 scenes -0.75 to -0.94); the hold clock is worse (HD 0.049, 8 of 11 stuck). Both stay as diagnostic
presets.

## Presets and reproducibility

| Board | Preset | What it is |
|---|---|---|
| HUGSIM | `spec` | tree `opctrl`, `OP_CTRL {"delay": 0.25}`, opts `{"op_ctrl": true}` (5 s static warm-up), dilate clock = decision 118's arm (default for cinque / lebowski; alias `opctrl_d118`) |
| HUGSIM | `spec_cold`, `spec_hold` | diagnostics: no static warm-up; + hold clock with `OP_CTRL {"delay": 0.2}` (neither launches) |
| HUGSIM | `exam` | legacy: `--controller` and `--opts` literally; every HUGSIM number before 2026-10-05; all older chain scripts pass it |
| B2D | `spec` | `drive` + `op_ctrl {"delay": 0.2}` + rig `B2D_SPEC_MOUNT` = (1.59, 0, 1.86) (open-loop-aligned), resume timer; `spec_bumper122`, `spec_windshield143` = same with the earlier cameras; default when a config names neither a preset nor an arb mode |
| B2D | `drive` | the `drive` arbitration of every B2D result up to 2026-10-05 (moved verbatim from `op_arb.sh`): raw action curvature through the bicycle model, no clip / delay, 1.433 m |
| B2D | explicit `"arb": {"mode": ...}` | the older arms (native, acc, e2e, switch, base, ...) unchanged |

`nored` cannot be reproduced any more: the agent refuses it. Results that used it (decisions 74, 81, 82, 102) are listed
in the results note for relabelling. `op_camera_tick` defaults to 0.05 s (every scored run set 0.05 explicitly; 5 Hz is an
undeclared deviation now).

## Writing a harness

```python
from jevdrive.openpilot import interface as IF
values = IF.resolve_hugsim(opts, controller, dataset, op_ctrl_rule)   # or resolve_navsim / resolve_wod / b2d_values
IF.write(run_dir, IF.record("hugsim", preset, values, config=opts))   # raises on an undeclared or forbidden deviation
```

A new board adds a `DECLARED[board]` entry (privilege `real-car` / `LB` / `semi` / `priv`, why, decision) and a
resolver; a new trick on an existing board extends its `DECLARED` entry, in the same commit as the code.

Out of scope here (deferred with training): route-polyline input into the ONNX (`experiments/op_route_cmd/`) and a merged
fine-tune with a common guard set.

Last verified: 2026-10-05
