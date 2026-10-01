# Formal v4: final G4 audit

**Original G4 necessary conditions are not met. No replacement default is qualified by this comparison.**

All 60 requested routes have selected final official records; all 63 attempts are retained. There are 49 driving-completed routes, 10 successes under the strict subset diagnostic SR rule, and 11 official TickRuntime failures.

| Group | Driving completed | Mean completion % | Strict subset SR | Official DS subset mean | Full CTE RMS m | Before-collision CTE RMS m |
|---|---:|---:|---:|---:|---:|---:|
| carla-seed0 | 9/10 | 94.726 | 10% | 54.292620 | 0.340578 | 0.439765 |
| tcp-seed0 | 8/10 | 85.186 | 20% | 55.512235 | 0.587369 | 0.623041 |
| pursuit-seed0 | 8/10 | 92.627 | 20% | 59.147388 | 0.402602 | 0.413208 |
| carla-seed1 | 8/10 | 90.790 | 10% | 53.329848 | 0.385877 | 0.439765 |
| tcp-seed1 | 8/10 | 85.186 | 20% | 55.512235 | 0.587364 | 0.623041 |
| pursuit-seed1 | 8/10 | 92.627 | 20% | 59.147388 | 0.402597 | 0.413208 |

## Original necessary conditions

- Seed0 candidate completion92.627% is below strongest reference CARLA94.726% by2.099percentage points. Incomplete routes increase from1 to2. Seed1 candidate92.627% exceeds CARLA90.790%; both have2 incomplete routes. TCP85.186% is also checked in each seed.
- Equally weighted full-route CTE RMS is 0.402599745m for candidate versus 0.363227293m for frozen CARLA reference: 10.840% higher, failing the required20% reduction.
- Trajectory-reference speed RMS difference is +0.011194941m/s, within+.1. This is not independent fixed-cruise error or official comfort.
- Official blocked/deviation counts are zero, but28 routes have at least one recorded low-speed interval of5s or longer, including11 unfinished TickRuntime routes. Causal dominance remains unknown; no zero-event shortcut establishes no controller-caused stalls.

## Interpretation and provenance

The frozen lateral reference uses CARLA lateral PID with the same PI(.5,.25), additive lookahead. TCP uses vendor longitudinal control; the candidate is pursuit max with PI(.5,.25). All six archived configuration hashes match the declared mapping. Runtime source commit is recorded in g4.json.

Driving completed means completion>=100. Strict subset SR requires official status Completed/Perfect and no nonempty infraction list except min_speed_infractions; denominator10 per group, not the full220 script denominator. Reported DS values are means of official serialized score_composed. These route-oracle results do not establish planner/full220 or comfort improvements.

Full-route and before-collision metrics remain separate. Long stalls can make full-route CTE small. Every unfinished route has recorded contact before prolonged low speed, but ego-only logs do not identify collision responsibility or isolate controller, obstacle and contact dynamics. Same TrafficManager seed does not ensure identical counterpart realization; near-identical outputs across seeds are not independent random replications.

Three rc139 infrastructure attempts are retained separately from60 route results. Pose faults and invalid actuation are distinct from logged signed-reverse invalid_motion after contact; see the failure audit for exact event/frame context.

[Exact six-group CSV](six-group-table.csv), [manual original-plan checks](manual-qualification.json), [frozen helper output](g4.json), [selected60 evidence](selected-60-attribution.csv), and [all63 attempts](all-attempt-attribution.csv). Existing helper README/output were preserved; this final wrapper adds strongest-reference completion, failure-count and real-stall scrutiny.

The candidate slope-hold check is separately archived by the campaign owner. It cannot overturn the failed Dev10 conditions.
