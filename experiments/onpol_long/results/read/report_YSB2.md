# ONPOL-LONG Amendment 14: arm B' (separate oncoming hinge channel): stopped at the label gate, no training

Label build `ol_dirraster.py onc` on pool `on1_L18YS` (34 788 rows, 0 errors; `onc_stats_L18YS.json`), job 1012-180644-4535, 1 min on 64 cores:

| registered label gate (before any training) | value | line | met |
|:--|--:|:--|:--|
| oncoming hinge active on the served plan (any footprint corner sample inside the oncoming region) | 0.207 | <= 0.10 | **no** |
| road hinge active on the served plan (pool raster, margin 0.3 m) | 0.122 (0.12185) | = 0.122 | yes |

By log turn: 0.187 on straight rows (< 15 deg), 0.245 at 15-45 deg, 0.335 at > 45 deg. The oncoming region is non-empty on every row (a two-way road
or an opposite roadblock is within the 80 m query almost everywhere). By the registered rule the four trainings (`YSO25` / `YSO50`, seeds 0 / 1) were
not started; their jobs and the 24 read chunks were cancelled / failed on the dependency.

Diagnostic after the gate (read-only, not a re-tuning of the line):

| | share of rows |
|:--|--:|
| minimum corner SDF on the served plan < 0 / < -0.25 / < -0.5 / < -1 / < -2 m | 0.207 / 0.156 / 0.119 / 0.072 / 0.018 |
| ego footprint at t0 already inside the oncoming region | 0.099 (0.48 of the active rows) |

The first entry is at the start of the plan for most active rows (median 0.2 s), and the active rows are only slightly further from the logged path
(|e0| median 0.29 m against 0.13 m). So about half of the activation is the car's own position at t0, close to the log: the region as defined
(baseline heading at the lane point nearest the row, whole roadblocks whose lanes all point away, minus intersections and connectors) marks part of
where the human drove as oncoming. zeros_mech's lane classification (the evaluator's lane polygons, centre-line yaw at the projection; d268 G-dir)
puts the log's own poses in an oncoming lane 0.4 % of the time, so the excess is in this label's geometry (most likely the roadblock-level removal
or lanes of the other direction overlapping the ego's lane polygons near junctions), not in the driving. The same geometry was arm B's removal set,
so arm B's raster (decision 274) carried it too. A rebuild that classifies by the evaluator's own lane polygons and centre-line yaw at the projection
(zeros_mech's rule) and drops the roadblock layer would be the fix; it needs a new amendment.

The arm B collect directory was moved out of `runs/zeros_mech/collect/` (see report_YSB.md); arm B' would have written its collect under
`runs/onpol_long/zm/` through `scripts/ol_zm.py`.
