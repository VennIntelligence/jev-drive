# ONPOL-NUPLAN, self-consequence L: gates (2026-10-11)

Code: `lib/on_lon.py` (loss, `timing_only`, label builder), `scripts/on_ctx.py` (scene context), `scripts/on_lon_pack.py` (pack + pool gate),
`scripts/on_lon_gates.py` (gates i - iii). Definitions and every deviation from the prereg / design.md: the docstring of `lib/on_lon.py`.
All numbers below are in the json files of this directory; nothing here is a training result.

## Context (`lon_ctx.json`)

1491 of 1491 scenes labelled from 136 NAVSIM test logs. Route origin 2 s before the scene token, 261 m grid, 24 light keyframes, 8 line
slots (no scene uses all 8). Logged keyframes against the route: lateral max 0.18 m, arc p99.9 0.31 m; two scenes with a folded route
(a log that stands still: 0.47 m and 2.55 m), dropped by the packer. 12 scenes have a 1 s frame gap inside the window, so rows are aligned
by timestamp. Largest reach needed (arc at the last decision + 4 s at the logged speed + 40 m) 212 m. Lines: 843 traffic-light lines,
384 eligible, 254 of them red somewhere in 11.5 s; 297 stop-sign lines (17 eligible; masked in B, they only close P). Turns of >= 45 deg:
586 scenes, 583 eligible.

## (i) `timing_only` (`lon_grad.json`): pass

4000 synthetic rows with every item active on 1283 to 1897 of them, float64. Values equal the unprojected ones bit for bit (also float32).
Cross-heading gradient at most 7e-18 per item against an along-heading gradient of 0.12 to 0.27 (the unprojected gradient has a
cross-heading part of 0.03 to 0.12); yaw gradient exactly 0; the kept gradient equals the along-heading part of the unprojected one
(3e-17) and a central finite difference along the heading (300 active rows per item, all within 1e-4, max 3e-10). Float32 as the trainer
runs it: cross-heading 7e-9 against 0.46, yaw 0, finite; a standing plan (all poses at the origin) gives a finite gradient.

## (ii) Operators against `lc.py` / `sweep.py` (`lon_kat.json`): pass

On identical inputs (2000 random routes of 131 and 261 vertices, 764 000 points each) the torch projection equals `lc.project` exactly
(arc and lateral offset, 0.0); dense interpolation and front point 3e-5 m, speed 1.5e-4 m/s, plan arc 1.7e-4 m (float32 rounding).
End to end (torch dense -> torch projection against numpy dense -> numpy projection) 0.03 to 0.06 % of points differ by more than 1 mm,
up to 0.05 m in arc: the projection is discontinuous where a point is equidistant from two route segments, and the inputs differ by
rounding. Within 3.5 m of the route the largest difference is 0.7 mm.

## (iii) Label agreement with LONG-PROBE on navtrain (`lon_navtrain.json`): pass

Shards 2 and 3, log / ot1 / yr1 states (40 440 states, 80 880 rows with the stored `P2H10-F` s0 / s1 own plans), every state's route and
boxes re-expressed in the state's own frame (the path a pool row takes).

| label = 1 | rows | hinge > 0 | share | line |
|:--|--:|--:|--:|--:|
| A (`a_lab`), A hinge at margin 0 | 267 | 259 | 0.970 | 0.95 |
| B red light (`b_lab`, kind 2), B hinge at the line itself | 81 | 81 | 1.000 | 0.95 |
| C (`c_lab`), C hinge as trained | 154 | 154 | 1.000 | 0.95 |

Descriptive, the hinges as trained (on-policy margins) on the same rows:

| item | active, all rows | active, label negative | mean when active | the logged driver's own future as the plan (17 216 log states): active / mean |
|:--|--:|--:|--:|:--|
| A | 1.52 % | 1.17 % | 0.26 m | 0.52 % / 0.10 m |
| B | 0.20 % | 0.08 % | 0.20 m | 0.20 % / 0.10 m |
| C | 0.80 % | 0.34 % | 0.13 m/s | 0.03 % / 0.31 m/s |

The fixed on-policy margins are not log-consistent: the logged driver itself closes to under 2 m (or under its current gap - 0.5 m) behind
a lead in 0.5 % of states and stops within the last metre before a red line in 0.2 %, by about 0.1 m each. On navtrain log states B is
active as often on the logged future as on the model's own plan.

## (iv) Activation on the pool (`lon_pool-train.json`; smoke: `lon_pool-smoke10.json`)

Pool `train`: 20 386 rows of 1025 scenes (`P2H10S-F-s0` and `-s1`), 20 352 with labels (34 rows of the two folded-route scenes get none).
Pack: `$DATA_DIR/runs/onpol_nuplan/rows/train/lon.npz`, items A B C P, P on strictly open rows against the amended D_k.
Join checks: the log's travel against `on_rows.py`'s `D_reg` median 0.1 mm, p99 6 cm, max 0.34 m; the arc ahead of the log median 0.2 mm,
p99 7 cm, max 0.65 m; the row pose against the logged pose at decisions 0 / 1 at most 1.5 / 1.8 cm; the row's log time by frame index
against `t_log_us` at most 2 ms; light state known at every dense step. Rows: 0.6 % more than 5 m ahead of the log, 2.6 % more than 5 m
behind; lateral offset from the route median 0.04 m, p99 1.98 m.

| item | rows with context | active on the served plan (rows, scenes) | mean / p90 / max when active | active on the imitation target | the log's own future at decisions 0, 1 (4092 rows) | 0.2 % rule |
|:--|:--|--:|:--|--:|--:|:--|
| A | 47.5 % have an eligible lead | 1.37 % (279, 41) | 0.43 / 1.15 / 3.76 m | 0.39 % | 0.20 %, mean 0.05 m | in |
| B | 23.7 % a light line ahead, 12.0 % red in the window, 9.1 % red now | 0.43 % (88, 11) | 0.59 / 2.09 / 6.92 m (median 0.08 m) | 0.25 % | 0.39 %, mean 0.01 m | in |
| C | 36.0 % of rows are in a scene with an eligible turn | 1.35 % (275, 41) | 0.14 / 0.33 / 1.34 m/s | 0.52 %, mean 0.04 m/s | 0 | in |
| P | 11.5 % open (34.5 % by design.md's rule; 22.3 % closed by an actor within 4.5 m of the route, 4.5 % by a line) | 5.43 % (1107, 110) | 1.38 / 3.68 / 9.22 m | 0 | 0.29 %, mean 0.19 m | in |

P variants for the record: against the log's own travel 5.74 %, mean 2.01 m; with design.md's open rule 16.6 %, mean 1.68 m.
Activation of A and C grows with the decision index (A 0.7 % at k = 0 to 2.2 % at k = 9; C 0.5 % to 2.4 % at k = 7); B is flat (0.3 to 0.5 %).
The item masses are concentrated: the 20 most active rows come from 7 (A), 4 (B), 5 (C) and 12 (P) scenes; B is active in 11 scenes in all.
The most active rows per item with scene, seed and k: `items.<item>.top20` in the json.

![preview](../../figs/lon_preview-train.png)

What to look at: each panel is one row in its own frame (x forward): the 3 most active rows of each item from different scenes. Grey band
= the route, blue = the served plan from the ego box, green dashed = the imitation target, red dots = the dense steps the item charges
(P: red circles on the charged poses), boxes = actors at the row's time with their 4 s tracks (red = A-eligible), ticks across the route =
stop lines (red = must stop now, orange = red later in the window, green = usable light line not red, black = not usable), orange cross =
the turn peak. A: the plan runs through a lead (panels 1, 2), and in panel 3 it passes a stopped lead while 2 m beside the route (the
kind of row where a purely along-route gap is debatable). B: a row standing at a red line whose plan creeps 15 m through it (panel 4),
a plan through a lead and a red line at 8.8 m/s (5, the same row as A's panel 2), a plan that reaches a line that turns red (6).
C: plans that go wide and fast into the turn. P: empty road, the plan much shorter than the target's travel.

Smoke (`smoke10`, 100 rows, before the full pool): A, B, C 0 of 100, P 3 of 100; preview `figs/lon_preview-smoke10.png`.

## Audit

An independent read of the three files (Opus, no access to these results) found no sign, frame, shape or sentinel error and no gradient
path around `timing_only`. Its findings and what was done: B charged a plan for every red step after a crossing on green (fixed: nothing
after a crossing on a non-red step; known-answer test green-then-red / red / red-then-green = 0 / 0.68 / 0); the row time was rebuilt
from the frame index (now `t_log_us`); P's "open" did not see crossing actors or unusable lines (strict rule, P gated on the pose's
lateral offset); C's cap inherited a standstill of the log before the turn (floored at the log's peak speed) and was 1 m wider than
written (now the two vertices around s); A had no gate on the plan's own lateral offset (added); routes folded by standstill jitter (the
two scenes dropped); NaN on rows without context (the pack is finite everywhere). Not changed: only the first >= 45 deg turn of a route
carries C; `on_rows.py` gives every actor class 0, so every simulator actor counts as a vehicle.
After the fixes gates (i) - (iii) were rerun (the numbers above); the fixes themselves were not audited a second time.
