# Frozen G4 audit

incomplete_comparison

| Group | Final | Completion mean | Driving completed | Subset diagnostic SR |
|---|---|---:|---:|---:|
| pursuit-seed0 | False | None | 0/10 | None |

Full seed conditions and all failed attempts are in g4.json. No new default is declared.

- Subset diagnostic SR uses official record status/infraction semantics with requested-route denominator; not official full220 SR.
- Driving completed (completion>=100), strict SR, G2 gates and harness finished are distinct.
- Official TickRuntime failures remain denominator; user harness caps prevent readiness.
- First harness-finished attempt selection inherited from report; all retries and failures retained, never best attempt.
- Speed comparison is trajectory-derivative reference tracking, not an independent fixed-cruise or official comfort metric.
- Full-route CTE is used for the numerical 20% condition; before-collision metrics shown separately and never substituted.
- All-reference-G2-failed exemption does not apply: frozen CARLA+PI reference passed G2.
- Passing listed G4 conditions alone does not replace G1-G3, repeat variability review, or planned confirmation.
