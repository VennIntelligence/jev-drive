# Fixed-window lateral controller diagnostics

Read this when you need the isolated turn experiments, optional pose propagation, or their evidence boundaries.

The campaign uses complete routes 24240 (left, 8 m/s),26966 (sharp right, 8 m/s), and17563 (two separate S windows, 6 m/s). It runs a 20-point/5-second route oracle without background traffic. It does not run TCP inference or establish leaderboard improvement. Protocols, all attempts, source snapshots and figures are indexed in the [working folder](../todos/2026-09-23-lateral-followup/README.md).

## Completed interpolation test

`Controller(..., aim_interpolation="hermite")` optionally replaces linear pursuit aim sampling with local chord-length cubic Hermite interpolation. It preserves rotation/mirror behavior and falls back to linear for unsupported local geometry. Default interpolation remains `linear`; speed sampling and control limits are unchanged.

The ideal circular-motion test reduced sampling ripple, but the actual six-case experiment failed four of 147 required conditions. Sharp-right CTE RMS/P95 exceeded allowed regressions, and both left/right emitted steering-rate P95 increased instead of falling 20%. All six basic G2 checks passed, which did not qualify the candidate. See the [complete report](../todos/2026-09-23-lateral-followup/results/aim-g2-v2-report-v1/README.md).

## Optional rear-axle propagation

Controller-agent configuration accepts `pose_lateral_coefficient_s2_per_m`, also inside the existing `adapter` object. It maps to `PoseFilter(..., lateral_coefficient_s2_per_m=0.)`. The default is zero; values must be finite and nonnegative. No vehicle-specific coefficient is enabled automatically.

For an evaluated interval, the optional displacement is `-k * mean_speed**2 * mean_world_gyro * dt * right(midpoint_yaw)`, inserted before the unchanged GNSS correction. Speed/gyro means use the previous and current sensor frames; the midpoint uses previous estimated yaw plus half the current gyro increment. Reset clears gyro history. Invalid compensation fails before updating pose/history. The existing speed deadband, reverse guard and bounded compass dropout remain in force.

The measured development coefficient is 0.010659832 s²/m. It is an empirical same-MKZ 6/8 m/s term, not an identified tire parameter or a cross-vehicle default. In frozen physical-trajectory replay, sharp-right localization improved but left-turn localization worsened. The subsequent six-case [closed-loop protocol](../todos/2026-09-23-lateral-followup/pose-followup/protocol.md) reduced CTE RMS in all four windows: 23.21% right, 27.25% left, 6.87% S1, 4.17% S2. It still failed one of 156 required conditions: sharp-right body-heading P95 increased 1.178 degrees against a 1-degree allowance. No confirmation run or default promotion followed. The [full report](../todos/2026-09-23-lateral-followup/final-report.md) distinguishes tracking gains, localization costs, physical jerk and the remaining directional error.

`control.jsonl` retains the inputs, actual controls and pose. Its `pose_status` adds coefficient, interval validity/reason, mean speed/gyro, midpoint yaw, dt and pre-GNSS displacement. `no_interval` has null computation fields, zero-coefficient valid intervals report `disabled`, and positive-coefficient intervals report `applied` even when the formula evaluates to zero. Faults retain explicit reasons.

The 125 controller tests passed. Production replay exactly reproduced the original zero-coefficient pose and the independently written fixed-coefficient shadow across 2761 frames. This establishes implementation identity, not performance qualification.

## Rear-slip pursuit and Ackermann inverse (lateral v2)

Two more opt-ins, both pursuit-only, off by default and bit-identical when off (the recorded GPU-box pose-g2 replay, 2758 frames, reproduces every emitted control):

- `pursuit_frame="rear_slip"` with `rear_slip_c_per_rad` (11.0 for the stock MKZ) rotates the aim point into the rear-axle velocity frame before `kappa = 2y/d²`. The sideslip is `beta = atan((v + 1 m/s) * yaw_rate / (c * 9.81))`, the PhysX linear-tyre form derived in [lateral-physics.md](../todos/2026-09-23-controller-next/lateral-physics.md), computed from SPEED and gyro only.
- `steer_inverse="ackermann"` with `track_width_m` (1.5929 m, front wheel spacing in `calibration-physics.json`) commands the inner-wheel angle `atan2(L|kappa|, 1 - w|kappa|/2)`, because PhysX (Ackermann accuracy 1) applies the nominal angle to the inner wheel.

A switch without its parameter, or a parameter without its switch, is rejected. `update(traj, t, trajectory_dt=dt)` also accepts N points at an explicit spacing (horizon N·dt, never extrapolated), for real TCP's 4 points at 0.5 s; the default call still takes exactly 20 points at 0.25 s.

For experiments only: `diagnostic_truth_pose_ceiling: true` in the controller config feeds the simulator's rear-axle pose to the route adapter. It works only together with `b2d_controller_validate.py --allow-truth-pose-diagnostic`, labels every telemetry row `pose_source: truth_diagnostic_ceiling`, and raises instead of falling back when truth is missing. `b2d_controller_validate.py` also takes pre-registered run perturbations (`--perturbations`, `--perturbation-ids`: GNSS/IMU noise seeds and spawn lateral/yaw offsets; the leaderboard previously dropped `noise_seed`, so every run used seed 0), `--chase-camera` for a separate off-screen render pass, and `--no-rendering` / `--quality` for sensor-only runs on big maps.

The pre-registered closed-loop test is [lateral-v2-protocol.md](../todos/2026-09-23-controller-next/lateral-v2-protocol.md); the results are in [lateral-v2-report.md](../todos/2026-09-23-controller-next/lateral-v2-report.md). With the fixed pose coefficient, both corrections together cut the 26966 window CTE RMS median from .392 to .089 m and reduced CTE in every development and held-out window. The candidate still failed the frozen action guards: emitted steer-rate P95 rose by more than 20% in 8 windows, and lateral-acceleration P95 by more than 10% in 4 windows, the truth-pose arms included. It is not a default. Town13 servers crash on the shared GPU under load, which left one held-out route incomplete.

## Local environment recovery

**Tokyo box only.** Never source this on the GPU box, which has its own working driver.

An unattended NVIDIA userspace upgrade broke CARLA startup before any driving case. Matching 580.159.03 vendor libraries were extracted to a private directory, with the official archive checksum verified. CARLA's actual process maps and GPU UUID confirmed successful isolated recovery. No host driver installation or reboot was performed.

The isolated 580.159.03 userspace workaround is historical and must not be sourced by current
launches. The running 580.173.02 kernel module matches userspace, and the single RTX 3090 is CUDA
index 0; use `DISPLAY=:0` and CARLA graphics adapter 0. Recheck driver/library versions and the GPU
UUID after a reboot. The [recovery record](../todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/README.md)
preserves the prior failed start and successful retry.

Last verified: 2026-09-23
