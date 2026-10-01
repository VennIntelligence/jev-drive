# Frozen G4 audit

g4_condition_failed

| Group | Final | Completion mean | Driving completed | Subset diagnostic SR |
|---|---|---:|---:|---:|
| carla-seed0 | True | 94.726 | 9/10 | 0.1 |
| carla-seed1 | True | 90.78999999999999 | 8/10 | 0.1 |
| tcp-seed0 | True | 85.186 | 8/10 | 0.2 |
| tcp-seed1 | True | 85.186 | 8/10 | 0.2 |
| pursuit-seed0 | True | 92.627 | 8/10 | 0.2 |
| pursuit-seed1 | True | 92.627 | 8/10 | 0.2 |

Full seed conditions and all failed attempts are in g4.json. No new default is declared.

- Subset diagnostic SR uses official record status/infraction semantics with requested-route denominator; not official full220 SR.
- Driving completed (completion>=100), strict SR, G2 gates and harness finished are distinct.
- Official TickRuntime failures remain denominator; user harness caps prevent readiness.
- First harness-finished attempt selection inherited from report; all retries and failures retained, never best attempt.
- Speed comparison is trajectory-derivative reference tracking, not an independent fixed-cruise or official comfort metric.
- Full-route CTE is used for the numerical 20% condition; before-collision metrics shown separately and never substituted.
- All-reference-G2-failed exemption does not apply: frozen CARLA+PI reference passed G2.
- Passing listed G4 conditions alone does not replace G1-G3, repeat variability review, or planned confirmation.
