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


## Amendment 15: the oncoming region by zeros_mech's lane rule, arm B' and arm B redone: stopped again at the label gates, no training

`ol_zmonc.py build` (job 1012-181242-8bc4, 0 errors; `onc_stats_L18YS_zm.json`): every 0.5 m cell classified by `zm_read.Lanes`'s rule (evaluator lane polygons,
centre-line yaw at the projection, same < 60 deg, oncoming > 120 deg, no roadblock layer) at the row's t0 heading.

| gate | value | line | met |
|:--|--:|:--|:--|
| human: the log's own t0 rear axle in an oncoming lane (Lanes.cls itself) | 0.0048 (in a lane: 0.0048; same direction 0.9946) | <= 0.01 | yes |
| arm B': oncoming hinge active on the served plan | 0.227 | <= 0.10 | **no** |
| road hinge active (pool raster) | 0.122 | = 0.122 | yes |
| arm B redone: road hinge active with road minus oncoming cells | 0.374 | <= 0.16 | **no** |

All six trainings (YSO25 / YSO50 / YSBz x 2 seeds), the 36 read chunks and both report jobs were cancelled before starting.

Diagnostic after the gates (read-only): the human's own footprint, not its rear axle, enters the corrected region often. Its box at t0 has a corner in
the oncoming region on 0.128 of rows (0.064 by more than 0.5 m), and its logged 4 s path does somewhere on 0.252 (0.164 by more than 0.5 m). By log turn
the served-plan activity is 0.19 straight, 0.30 at 15-45 deg, 0.47 on turns. The rule is right for a point (the human's rear axle is oncoming 0.5 % of
the time, as in d268), but a car-sized footprint driven normally overlaps oncoming lane polygons at lane edges and in junctions, so a footprint hinge on
this region would penalise the human's own driving on a fifth of the rows. A usable oncoming label would have to be point-based (rear axle or box centre,
as the zeros_mech reading itself is) or tolerate an overlap depth; that is a new design, not a gate tweak.

For decision 274 (the void arm B): 84 % of the cells arm B removed are oncoming by this rule (37 982 212 removed cells), and the corrected directional
raster is *more* active on the served plans than arm B's (0.374 against 0.285). So arm B's high road-hinge activity, which went with its worse zeros
and GT, was not mainly mask error: removing oncoming lanes from a footprint-checked road raster fires on normal driving under either geometry.


## Amendment 16: point-based oncoming hinge variants (label study, no training)

`scripts/ol_onclab.py` (job 1012-181903-051c, CPU, < 2 min; `onclab_L18YS.json`), on the zeros_mech-rule region of Amendment 15, 34 788 rows. Active = the variant enters
the oncoming region at any of the 41 dense steps (0-4 s); the human path = the log's 9 poses at 0-4 s, linear to the same steps; heading of the region = the row's t0 heading.

| variant | served plan | human path | served straight / turn (>= 45 deg) | human straight / turn | usable (served <= 0.10, human <= 0.03) |
|:--|--:|--:|:--|:--|:--|
| (a) rear-axle point | 0.072 | 0.021 | 0.042 / 0.262 | 0.009 / 0.120 | yes |
| (b) box centre | 0.075 | 0.021 | 0.045 / 0.266 | 0.009 / 0.127 | yes |
| (c) footprint, 0.5 m allowed depth | 0.130 | 0.054 | 0.096 / 0.337 | 0.027 / 0.222 | no |
| (d) footprint, 1.0 m allowed depth | 0.083 | 0.023 | 0.055 / 0.253 | 0.011 / 0.120 | yes |

d268's across-traffic early-inside zero events of YS-L18-s1 (11 with a decision two before the event; region computed on the held scene with the same rule):
the served plan two decisions before the event is active in 9 / 11 for (a), 8 / 11 for (b) and (d) (c687799d differs: rear axle only). The two never active:
330166de (corridor) and 8c9e3fd9 (offroad).

Caveat for the usable variants: on turning rows the human's own path is active on 0.12 (straight 0.01), so on turns the hinge would still charge some
human-like plans (the region is classified at the t0 heading for the whole 4 s). No training is queued; the variant is the user's choice.
