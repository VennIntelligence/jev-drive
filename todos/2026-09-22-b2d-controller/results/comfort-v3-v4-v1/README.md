# Matched G2 physical comfort diagnostics: v3 versus v4

Across all 18 matched cases, reducing the PI proportional gain from 1.0 to 0.5 is associated with lower whole-episode longitudinal acceleration and jerk. The integral gain remains 0.25. This is a diagnostic of actual recorded vehicle motion under the route oracle (`policy none`), not a new acceptance gate or an official Driving Smoothness / Driving Score calculation.

| Raw 20 Hz metric | v3 | v4 | Change in mean RMS | Cases with lower RMS |
|---|---:|---:|---:|---:|
| Longitudinal acceleration RMS, m/s² | 2.2968 | 1.8024 | −21.5% | 17/18 |
| Lateral acceleration RMS, m/s² | 1.0624 | 1.0833 | +2.0% | 9/18 |
| Longitudinal jerk RMS, m/s³ | 24.7352 | 18.3398 | −25.9% | 16/18 |
| Lateral jerk RMS, m/s³ | 8.4276 | 7.9592 | −5.6% | 13/18 |

Values above are arithmetic means of the 18 individual episode RMS values, giving every case equal weight. The mean of the individual episode 95th percentiles of absolute longitudinal jerk decreases from 39.2917 to 16.6178 m/s³. This is **not** the 95th percentile of the pooled samples; both are separately preserved in the data.

The largest longitudinal changes occur in the six 6 m/s cases (routes 17563 and 1773-speed6): mean jerk RMS 34.6934 → 18.4501 m/s³. The twelve 8 m/s cases change from 19.7561 → 18.2846 m/s³. Improvement is not universal: raw longitudinal jerk RMS increases on 24240/pi/carla (17.287 → 19.134) and 26966/pi-max/pursuit (19.372 → 23.409). The latter also has higher longitudinal acceleration RMS. Lateral acceleration is slightly higher in the aggregate, so these results do not establish universal comfort improvement.

## Data and inclusion

Inputs are `/data/runs/b2d/controller/development-v3` and `development-v4`, matched by the identical route/variant/preset directory key. Every matching case is retained, regardless of its existing gate result. There are 9,239 v3 samples and 9,106 v4 samples across 36 episodes. Every recorded tick is included: startup, maneuvering, deceleration and parking. No outlier clipping, warmup removal, failure exclusion or matched-duration truncation is applied. First/last samples are retained.

The helper checks finite complete acceleration/axis/angular-rate vectors, unit forward/right axes, consecutive frame numbers, and simulation-time differences of 0.05 s within 1 μs. All 36 episodes passed. Actual timestamps are used for differentiation; typical differences are 0.050000000745 s. Episode durations differ slightly, and parking occupies part of each record. These whole-episode comparisons therefore include the effects of changed speed histories and phase durations, rather than identifying a gain effect at an identical physical state.

The recorded `plant_kinematics.acceleration_mps2` supplies the actual world-space translational acceleration. `forward_vector` and `right_vector` supply the full three-dimensional vehicle axes, including pitch and roll. The acceleration is not inferred from throttle commands. This is not seat-level vibration, gravity-inclusive accelerometer specific force, or a passenger comfort model.

## Exact protocol

For world acceleration **a**, unit vehicle forward **f**, right **r**, and recorded simulation timestamp *t*:

- Longitudinal acceleration = **a** · **f**, m/s². Positive means forward.
- Lateral acceleration = **a** · **r**, m/s². Positive means vehicle-right.
- World jerk **j** = d**a**/dt, using `numpy.gradient(acceleration, times, axis=0, edge_order=1)`: centered three-point differences in the interior, first-order one-sided differences at the two endpoints.
- Longitudinal/lateral jerk = **j** · **f** / **j** · **r**, m/s³.

These jerk components are projections of the derivative of the inertial acceleration vector. They are **not** the derivatives of the body-projected acceleration scalars: d(**a**·**f**)/dt also includes the changing-basis term **a**·d**f**/dt. This distinction matters during turns. The code's analytic checks cover linear acceleration on irregular timestamps, constant world acceleration under a rotating body basis, and constant-value filter preservation.

Primary results use the unfiltered acceleration vectors at 20 Hz. Differentiation amplifies simulator sample-scale fluctuations; the raw peaks and distributions are retained rather than suppressed. A separately named `box5` sensitivity first replaces each world acceleration vector with a centered five-sample arithmetic mean, then applies the same derivative and projections. Interior support spans t−0.10 to t+0.10 s; the first/last two windows are truncated to available samples and renormalized. This is an offline, noncausal filter, with no padding, decimation or excluded boundary samples.

Under `box5`, mean longitudinal acceleration RMS decreases 1.6954 → 1.3811 m/s² (−18.5%) and mean longitudinal jerk RMS 9.5434 → 6.3809 m/s³ (−33.1%). Lateral acceleration RMS increases 3.0%; lateral jerk RMS changes −0.7%. The longitudinal direction of change survives this fixed smoothing sensitivity, while the lateral jerk result is much smaller after smoothing. The two protocols must not be mixed in one comparison.

Per-case statistics include signed minimum/maximum, RMS, and absolute-value median/p95/p99/maximum. Percentiles use NumPy's default linear interpolation. Aggregates explicitly distinguish equal-case mean RMS/p95 from pooled, sample-weighted statistics. ECDF figures give each case total probability mass 1/18, so longer episodes do not dominate those curves. Neither confidence intervals nor significance claims are provided for this deterministic, small diagnostic matrix.

## Reproduce and inspect

Dependencies used: Python 3.8, NumPy 1.23.5, Matplotlib 3.7.5, available in `/data/envs/carla`. No CARLA process, git command, network access or runtime source change is required.

From the isolated worktree root:

```bash
/data/envs/carla/bin/python \
  todos/2026-09-22-b2d-controller/results/comfort-v3-v4-v1/analyze.py \
  --v3 /data/runs/b2d/controller/development-v3 \
  --v4 /data/runs/b2d/controller/development-v4 \
  --out /data/runs/b2d/controller/comfort-v3-v4-reproduction-v1
```

The output path must not exist. Do not rerun over the preserved `artifacts` edition.

- [Exact inputs and derived samples](artifacts/samples.csv): all 18,345 samples; timestamps, frames, original world acceleration, forward/right axes and angular rates, plus both derived protocols. Original angular rates retain explicit deg/s columns; a separately converted yaw rad/s column is included but unused in the acceleration/jerk conclusions.
- [Per-case statistics](artifacts/per-case.csv): all 288 version/case/protocol/metric combinations, including configuration gains and existing status/gate fields.
- [Aggregate results](artifacts/summary.json) and [source/output SHA256 manifest](artifacts/manifest.json): original raw trace, validation result, archived config and summary paths/hashes, exact helper snapshot, Python/library versions, generated artifact hashes.
- [Raw paired figure](artifacts/paired-rms-raw.png), [PDF](artifacts/paired-rms-raw.pdf); [smoothed sensitivity](artifacts/paired-rms-box5.png), [PDF](artifacts/paired-rms-box5.pdf); [equal-case raw distributions](artifacts/distributions-raw.png), [PDF](artifacts/distributions-raw.pdf).

## Relationship to scoring

This answers the user's separate question about abrupt actions with direct physical evidence. It does not replace the pinned benchmark's definitions. The [preserved scoring-source observations](../scoring-source-observations/README.md) document that the pinned Driving Score formula uses route completion and its listed penalties, while Driving Smoothness is calculated separately. They also document that its yaw-acceleration path omits a derivative and its angular-rate path retains degree-valued inputs while labeling radian thresholds. The unmodified official script was **not executed** here, and no physically corrected version is presented as an official score.

No v2 vendor comparison is manufactured: that earlier matrix lacks the full raw kinematics required by this protocol. No Dev10 or full 220-route Driving Score result is inferred from these G2 comfort measurements. All plots label the diagnostic route oracle and retain the cases with regressions.
