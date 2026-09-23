# Independent pre-run analysis review

Read-only review of `analysis/pose_contract.py`, `analysis/analyze_pose.py`, frozen `protocol.md`, and both actual configs. No runtime or analysis source edits by this reviewer.

The numeric gates match the protocol: 26966 window RMS≤0.85×baseline, P95 and post10m RMS no increase; other-window RMS +.03 m; every-window P95 +.05 m, peak +.10 m, heading +1°, speed-error RMS +.10 m/s, mean-speed floor baseline−.20 m/s, lateral-acceleration P95≤1.10× and emitted-rate P95≤1.20×. All original G2 gates remain. Jerk/recovery are diagnostic, without inherited Hermite benefit gates or a newly invented window-pose threshold. Main metrics use all frames; moving samples are auxiliary. Both current configs match linear/max .5/PI .5/.25 and differ only in the pose coefficient.

The independent contract evaluator checks raw-motion frame/time alignment, k, prior/current speed/gyro means, midpoint yaw, pre-GNSS displacement/fusion order and telemetry null/reason semantics. Its second complete raw-GPS replay propagates its own state, which is important: it does not merely trust the previous logged pose. Initialization, bounded compass loss, timestamp faults and reset after invalid_pose are explicitly represented. The legacy-fixture switch marks live qualification false and bypasses selected new-field checks only for explicitly labelled historical fixtures; it must not be used for the live run.

One concrete pre-freeze blocker was sent to the analysis owner: `complete_frames` compared control `sim_time` and validation-trace `sim_time` absolutely. GameTime is relative while the world snapshot has an arbitrary origin (old same-frame example: trace 9.576274677 s versus control 5.050000075 s). The check must allow a constant per-case offset, then verify its invariance and adjacent dt/frame alignment. Motion and control share a clock and still require direct equality. This is an analysis correction, not a change to runtime or performance thresholds.

Also recommended enforcing the complete common configuration contract, beyond only k/linear and GNSS gains, and clearly labelling `pose_left_error` as body-heading-left: the earlier propagation decomposition used reference-left. Absolute position errors remain directly comparable, but those signed components should not be silently merged.

This note records the initial review while the analysis owner is finishing fixes. Final freeze should record the corrected source SHA and verification results. It is not a six-case outcome verdict.

## Final freeze review

**PASS for analysis readiness.** The clock-origin blocker is closed: per-case offsets must be finite and constant within 1e−8 s, with frame alignment and independent raw-motion interval validation retained. Common linear/max .5/PI .5/.25/near/steer limits, route cruise 8/6 m/s, and pair-only-coefficient differences are enforced. All performance thresholds still match the frozen protocol; no additional jerk/recovery or window-pose gate was introduced. Legacy-fixture mode remains explicitly ineligible for live qualification.

Reviewed frozen source SHA256:

- `analyze_pose.py`: `364c5be63c63326e99c8951a46602e83839404b85f0668d7c2e66b28f84b160d`
- `pose_contract.py`: `3cfd8767d6a34aa8da46d9abeaaf703c6819b02adaa14e4dc3e219fb24047c58`

Inspected the analysis owner's production-contract evidence: six generated production-schema replay groups, 2760 frames, independent sensor-replay maximum difference zero, and omission of a single required lateral field rejected for each group. These are analysis verification fixtures, not live driving results. No further blocker found; no tests or CARLA runs repeated by this reviewer.
