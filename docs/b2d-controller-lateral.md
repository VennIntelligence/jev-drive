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

## Local environment recovery

An unattended NVIDIA userspace upgrade broke CARLA startup before any driving case. Matching 580.159.03 vendor libraries were extracted to a private directory, with the official archive checksum verified. CARLA's actual process maps and GPU UUID confirmed successful isolated recovery. No host driver installation or reboot was performed.

While the loaded kernel driver is still 580.159.03, launch local experiments from a shell that sources:

```bash
source /data/tools/nvidia-userspace-580.159.03/isolated/env.sh
```

Then use `CUDA_VISIBLE_DEVICES=1`, `DISPLAY=:0` and CARLA graphics adapter 0, as documented for the [Tokyo box](tokyo-box.md). Recheck driver/library versions after any reboot. The private environment does not repair globally installed libraries. [Recovery evidence and commands](../todos/2026-09-23-lateral-followup/diagnostics/driver-recovery/README.md) preserve the failed start and successful retry separately.

Last verified: 2026-09-23
