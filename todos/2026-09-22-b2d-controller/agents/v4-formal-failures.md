# Formal v4 failure audit (complete)

Owner: `/root/agent`. Runtime/configuration remain frozen and untouched. Formal run: `/data/runs/b2d/controller/formal-v4`, manifest commit `2cca3a9660d989de0d09a7cb1d4e51610c814417`. Expected matrix: Dev10 × three presets × seeds 0/1 = 60 requested route cases. This note and the audit evidence preserve failures rather than tune them away.

Final evidence: [selected 60-case attribution CSV](../results/v4-formal-failure-audit/final-v1/selected-60-attribution.csv), [all 63 attempts](../results/v4-formal-failure-audit/final-v1/all-attempt-attribution.csv), [verification](../results/v4-formal-failure-audit/final-v1/verification.json), and [relocation mapping](../results/v4-formal-failure-audit/relocation-manifest.json). Canonical full raw editions are `/data/runs/b2d/controller/formal-v4-failure-audit/snapshots`; final edition is **034**. The ignored original `snapshots` symlink preserves historical path references. Earlier sections below are retained chronological observations, not alternative final counts.

## Final outcome

All 60 requested route cases are accounted for, with 63 total attempts: three `server_died_rc139` attempts lacked control telemetry and were retried successfully. Slope check passed in the parent campaign. Route completion and strict subset SR remain separate:

| Group | Reached 100% | Strict subset success | Longitudinal low-speed interval ≥5 s |
|---|---:|---:|---:|
| CARLA seed 0 | 9/10 | 1/10 | 5/10 |
| CARLA seed 1 | 8/10 | 1/10 | 5/10 |
| TCP seed 0 | 8/10 | 2/10 | 4/10 |
| TCP seed 1 | 8/10 | 2/10 | 4/10 |
| Pursuit seed 0 | 8/10 | 2/10 | 5/10 |
| Pursuit seed 1 | 8/10 | 2/10 | 5/10 |

Totals: **49/60 reached 100%, 10/60 strict subset success**, 38 cases had recorded collisions, and 28 cases had descriptive longitudinal low-speed intervals of at least five seconds. These are this Dev10 diagnostic subset, not full-220 scores. The cause labels remain unknown rather than converting contact association into responsibility.

All **11 unfinished cases end with the official 4000-tick TickRuntime** (the configured artificial tick cap is zero), despite **zero official vehicle_blocked and route_dev events**:

- Six runs of 25424: all groups/seeds finish at 47.26% after static traffic-warning contact, followed by approximately 190.85–191.00 s of low forward speed. At contact, true rear-axle centerline error is +0.049 m for CARLA, +0.055 m for pursuit and +0.131 m for TCP, with normal tracking. This is strong evidence of contact while near the supplied route; it does not prove the obstacle's complete geometry or blame assignment.
- Two TCP runs of 3514: both finish at 4.6% after Nissan Patrol contact during parking rejoin, followed by approximately 197.85 s of low speed.
- Three runs of 2091: CARLA seed 1 finishes at 60.64%; pursuit seeds 0/1 both at 79.01%, after repeated vehicle contacts and approximately 178.50–179.25 s of low speed. Pursuit has resumed normal motion at the final time limit.

Each of these 11 cases has **zero recorded `invalid_motion` or invalid-pose ticks before its first contact**. Contact-associated stall is established as an observed sequence; the controller/external causal dominance and collision responsibility remain **unknown** without counterpart trajectories, contact impulses or a controlled counterfactual. These records do not establish a new pre-collision lateral-control failure.

Across **68,142 available control rows**, recorded pose degradation, invalid-pose, stale-trajectory and invalid-actuation counts are all zero. There are **8,593 `invalid_motion` safe-controller ticks** on meaningful signed reverse readings; 8,581 are at/after first contact, while 12 occur before contact or in cases with no recorded contact. They are not invalid actuator outputs and must not be conflated with localization faults. The three infrastructure attempts have missing telemetry, so they are excluded from the observed-row rate denominator rather than declared fault-free.

All 166 raw-edition files (218,331,316 bytes) were moved to the separate `/data` archive after the writer stopped; every before/after SHA256 matched. All 549 source files referenced by final edition 034 (188,939,685 bytes), plus helper and archived report hashes, were independently reverified. Old manifests and incomplete edition 012 were retained unchanged. Compact source, final CSVs and manifests remain in the worktree; no simulator, configuration or runtime code was modified by this audit.

## Definitions

`driving_completed` means official `score_route >= 100`. `subset_diagnostic_sr` follows the pinned B2D merge script: status Completed/Perfect and every infraction list empty except `min_speed_infractions`. The denominator for a final group is its ten requested routes, not only successful runs. Harness `finished` and subprocess return code zero do not imply either definition. All attempts remain in evidence; the report owner selects the first harness-finished attempt per route without selecting the best retry.

Collision type and counterpart ID identify a contact, not which party caused it. Ego telemetry has no counterpart pose/velocity history or contact impulse. Collision responsibility stays `unknown` unless independent evidence resolves it. Small centerline error at contact supports accurate route tracking; it does not prove legal behavior, clearance or safe interaction. Large starting error in ParkingExit can be intended parking geometry and is not by itself a tracking failure.

The official `vehicle_blocked` / `route_dev` counters are reported separately from unfinished routes, TickRuntime and physical stalls. A descriptive low-speed segment is consecutive `abs(speed_mps) < 0.5` for at least five seconds; it is not a new acceptance threshold. The separate high-target stall detector can be fragmented when safe-controller ticks have a null target, so both series are retained. All episodes include startup and terminal stopping; event context must distinguish them.

## First eight carla-seed0 attempts

As of edition 003: seven reached 100%, one has strict subset success, zero official blocked/deviation events, zero compass degradation, zero invalid-pose, zero invalid-control and zero stale ticks. These zeros do not mean there were no failures.

- **25424 / ConstructionObstacleTwoWays:** official `Failed - TickRuntime`, 47.26% completion, exactly 4000 ticks / 200 s, official blocked=0 and deviation=0. Static traffic-warning collision at frame 3554, t=8.0 s; speed 7.943 m/s, target approximately 8, brake 0, true centerline error +0.049 m. At t=10.05 s speed is 0.022 m/s; progress remains approximately 50 m through the end. Continuous low-speed interval is 190.85 s. Later negative signed readings cause 646 `invalid_motion` safe-controller ticks, after the collision; no localization dropout. Evidence supports an obstacle-interaction-associated stall while following the diagnostic reference. Which mechanism dominates continued blockage is unresolved; this is not established pre-collision lateral control failure. Original context: `carla-seed0/attempts/25424/1/{criterion_events.json,control.jsonl,results.json}`.
- **3514 / ParkingExit:** completion 100%, strict SR false due to vehicle contact. Initial true centerline offset −3.146 m; collision frame 379 at t=1.85 s, true offset −2.736 m, speed 4.718 m/s, throttle 0.75, brake 0. Contact occurs during the route rejoin. A subsequent low-speed interval spans approximately 14.15 s, then the route completes. Counterpart type Lincoln MKZ, ID 4710. The record cannot establish collision responsibility or clearance around the parked vehicles.
- **3255 / ParkingCrossingPedestrian:** completion 100%, strict SR false. Pedestrian collision frame 1263 at speed 7.971 m/s and true CTE −0.112 m; vehicle contact frames 1374 and 1715. A subsequent low-speed interval lasts 15.50 s. Accurate reference tracking is compatible with unsafe interaction because this route oracle has no avoidance policy.
- **25381 / HazardAtSideLane:** completion 100%, strict SR false. Bicycle-class vehicle contact (`vehicle.gazelle.omafiets`) at frame 2622, speed 8.043 m/s, true CTE −0.039 m. The small CTE does not assign contact responsibility.
- **2091 / NonSignalizedJunctionLeftTurn:** completion 100% after 3882 ticks, strict SR false. Stop-sign infraction frame 7838, first vehicle collision frame 7853; later collisions frames 11265 and 11530. A 179.25 s low-speed interval spans frames 7881–11466, followed by renewed motion and completion. There are 1762 `invalid_motion` ticks on signed reverse readings, predominantly in the post-contact interval. Route-lane excursion is separately penalized; official blocked/deviation remain zero. Completion alone hides this severe stall.
- **27494 / BlockedIntersection:** completion 100%, strict SR false. Vehicle contact frame 11852 at speed 7.952 m/s and true CTE +0.252 m, followed by a 9.10 s low-speed interval.
- **25378 / YieldToEmergencyVehicle:** completion 100%, strict SR false due to failure-to-yield infraction; no collision. No claim of controller failure follows from this behavioral requirement.
- **26405 / StaticCutIn:** completion 100%, strict SR true in this initial group.

The first critical actionable finding is a reporting boundary: inspect TickRuntime and physical low-progress intervals even when official blocked/deviation counters are zero. A second boundary is causal: post-contact signed-reverse safe braking must not be presented as the cause of the earlier collision. These observations warrant transparent qualification, not changes to this frozen run.

## TCP seed 0 parking failure and scene-matching limit

TCP/3514 ends at 4.6% completion with official TickRuntime and a 197.85 s low-speed interval after vehicle contact (frame 13283, t=2.05 s, speed 5.153 m/s, true CTE −2.779 m, steer −0.1403, brake 0). At 200 s the controller still requests throttle 0.75; rejoin curvature concern 0.308 m⁻¹ exceeds its descriptive 0.2 bound. That end-state concern is post-contact evidence, not proof of the initial collision cause.

The counterpart in TCP is a Nissan Patrol (ID 24729), whereas the CARLA seed-0 run contacted a Lincoln MKZ (ID 4710). A shared Traffic Manager seed does not establish identical actor realization across these groups. Thus the 4.6% versus 100% outcomes must not be treated as a controlled identical-scene causal comparison. Collision responsibility and continued-stall causal dominance remain unknown.

## Infrastructure retry and audit preservation

TCP seed 0 / 2091 attempt 1 ends `server_died_rc139` after approximately 19 wall seconds, with no control telemetry. Group `events.jsonl` lines 14–19 preserve route start, failed route end, restart and successful attempt 2. Attempt 2 finishes after 288 ticks. This is recorded separately as an infrastructure failure; the cause of the simulator crash is not assigned to the controller. Final route selection uses attempt 2 as the first harness-finished attempt, while attempt 1 remains in all-attempt cost/reliability records.

The audit helper itself initially assumed a missing telemetry summary contained `stale_ticks`, and stopped while indexing this failed attempt. Partial edition 012 remains with `FAILED.md`; exact earlier helper source is preserved as `audit-v2.py`. The guarded reader resumed at edition 013, now explicitly retaining `telemetry_state=missing`. Missing telemetry is not evidence of zero control faults. Runtime source was untouched throughout.

## Pursuit seed 0: time limit after recovery

Pursuit/2091 ends TickRuntime at 79.01%, after a 178.50 s low-speed interval. First contact is frame 9844, t=5.70 s, speed 7.757 m/s, true CTE −0.072 m; repeated contact with the same Audi TT is frame 12694, t=148.20 s. At the final t=200 s it has recovered to 7.642 m/s and true CTE −0.053 m, progress 60.31 m, with finite normal tracking. Thus the final time limit occurs after renewed motion; it is not evidence that the terminal controller remains stationary. The preceding stop-sign infraction is frame 9829. Event sequencing supports prolonged traffic interaction consuming the time budget, while causal dominance and contact responsibility remain unknown.

Pursuit/25424 also ends at 47.26% after static traffic-warning contact and a 190.85 s low-speed interval, matching the broad obstacle-associated pattern in the reference groups. No pose degradation or invalid-pose event has been observed through 29 completed attempts (including the infrastructure attempt with explicitly missing telemetry).
