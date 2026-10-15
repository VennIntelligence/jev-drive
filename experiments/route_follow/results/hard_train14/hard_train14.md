# HT14: RSG-L18-s0 vs RBSG2-L18-s0 on the strict PAI turns of hard_train (2026-10-15)

Inventory read before a recovery lesson. No GPU, nothing rolled out: both numbers come from **stored** PAI 281 reads on Tokyo,
`/data/runs/route_night/r281/rsg` (2026-10-13 04:16-07:03) and `/data/runs/route_night/r281/rbsg2` (2026-10-13 09:14-12:05). Same serving:
gains `rel_front_steering_angle_weight` 20, `JEV_VCONT=1.0 JEV_LEAD=1` (r281 README), **JEV_TRACK path 0** (the plan as is; both reads predate
`track_ref.py`, added 10:41 that day, and the `gate` default of decision 309), same launcher, 8 chunks, 4 per card. They pair with each other
and with other path-0 reads only, not with `v2pai/*-gate` (docs/alpasim.md). Extract: rf_pai_follow's `r281.pkl` (md5 e242bd10, the one
behind `results/pai_follow/` and `results/p3_branch/shape.md`). Script `scripts/rf_ht14.py`; rows `hard_train14.csv`, rates `summary.json`.

## Turn set

- `alpasim/results/pai_exam/hard_train.txt` ∩ `pai281/strict36.txt` = 15 scenes. 68e53160 is dropped (in `turn_gallery/plan_strict36.json`
  `bad` / P4's `bad`: completion target, exit + 10 m, outside the log), leaving **14**: the same 14 hard_train scenes as P4's N = 27.
- `trainable.txt` ∩ strict36 = **3** (132ae6cb, 6fd8cf30, 8a143cb2). hard_train and trainable are disjoint, so 14 + 3 = 17.

## Pass rules

- **(a) completed**: rf_pai_follow's strict rule (`results/pai_follow/scenes.csv`, outcome `completed`: the ego crosses exit + 10 m before
  any zero event, <= 2.5 m from the route). Entry / exit asserted equal to plan_strict36.
- **(b) completed and shape gate** (decision 293; rf_p4.gate_of without its entry-speed term): on the label's own controller track,
  window [entry - 40 m, exit], cut at the first zero event, rf_shape's operators: peak kappa <= the topo class's human p95, peak dkappa/ds <=
  class p95 (all-turn p95 for classes with < 5 human turns, here no_straight_exit), peak a_lat <= 4.42 m/s^2 (all-turn human p95).
  The extra column `v0 <= v_e + 1` is P4's entry-speed term, reported but not part of (b).

## Rates (Wilson 95 % CI)

| set | checkpoint | (a) completed | (b) completed + shape gate | (b) + entry speed |
|:--|:--|:--|:--|--:|
| hard_train 14 | RSG-L18-s0 | 0 / 14 = 0 % [0, 22] | 0 / 14 = 0 % [0, 22] | 0 |
| hard_train 14 | RBSG2-L18-s0 | 4 / 14 = 29 % [12, 55] | 4 / 14 = 29 % [12, 55] | 4 |
| trainable 3 | RSG-L18-s0 | 2 / 3 [21, 94] | 2 / 3 [21, 94] | 1 |
| trainable 3 | RBSG2-L18-s0 | 2 / 3 [21, 94] | 2 / 3 [21, 94] | 2 |
| all 17 | RSG-L18-s0 | 2 / 17 = 12 % [3, 34] | 2 / 17 = 12 % [3, 34] | 1 |
| all 17 | RBSG2-L18-s0 | 6 / 17 = 35 % [17, 59] | 6 / 17 = 35 % [17, 59] | 6 |

Every completed turn of either checkpoint passes the shape gate, so (a) = (b) here; the gate bites only on 33c8976b (kappa and dkappa
over the class p95 for both checkpoints), which neither completes. Paired on the 14: 4 discordant scenes, all RBSG2 completes / RSG does not
(exact McNemar p = 0.125). RBSG2's 4 hard_train completions are all right turns (268949d2, b40f4729, ddc3e8df, ea468790); RSG left the route
(> 2.5 m, `off`) on three of them and zeroed (corridor) on the fourth. Neither completes any of the 9 others; 5 of those (403d558a,
e5e47fd2, eaba3f6a, f9e361aa, 1e8e939e) end with almost no curvature at all (kmax <= 0.014, entry speed 11-14 m/s or zeroed before the
entry): the car never turns.

## Per turn

kmax (1/m), dkappa (1/m^2), a_lat (m/s^2), v0 = speed at the entry (m/s, blank: zeroed before it). Gate = shape gate pass.

| set | scene | side / deg | topo | RSG outcome (score) | RBSG2 outcome (score) | RBSG2 kmax / dk / a_lat / v0 | RSG pass a / b | RBSG2 pass a / b |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| hard_train | 1e8e939e | R -49 | curve_single | event collision (0.00) | event collision (0.00) | 0.014 / 0.002 / 2.57 / - | n / n | n / n |
| hard_train | 268949d2 | R -88 | no_straight_exit | off (1.00) | completed (0.91) | 0.100 / 0.021 / 2.68 / 2.59 | n / n | y / y |
| hard_train | 3232fdb7 | R -87 | straight_exit | event corridor (0.00) | short_slow (0.94) | 0.064 / 0.009 / 1.02 / 3.37 | n / n | n / n |
| hard_train | 33c8976b | L +107 | curve_single | event offroad (0.00) | event offroad (0.00) | 0.181 / 0.046 / 0.57 / 2.58 (gate fails k, dk) | n / n | n / n |
| hard_train | 37d660b1 | R -85 | straight_exit | event corridor (0.00) | event corridor (0.00) | 0.066 / 0.009 / 3.39 / 5.49 | n / n | n / n |
| hard_train | 403d558a | R -93 | straight_exit | event corridor (0.00) | event corridor (0.00) | 0.001 / 0.001 / 0.21 / 11.6 | n / n | n / n |
| hard_train | a174faed | R -90 | straight_exit | event corridor (0.00) | event corridor (0.00) | 0.067 / 0.012 / 1.93 / 5.76 | n / n | n / n |
| hard_train | a4d2f7cf | L +110 | straight_exit | short_slow (0.98) | short_slow (1.00) | 0.063 / 0.009 / 1.04 / 4.33 | n / n | n / n |
| hard_train | b40f4729 | R -89 | straight_exit | off (1.00) | completed (1.00) | 0.095 / 0.018 / 3.13 / 2.96 | n / n | y / y |
| hard_train | ddc3e8df | R -66 | curve_single | event corridor (0.00) | completed (0.68) | 0.082 / 0.015 / 1.34 / 2.57 | n / n | y / y |
| hard_train | e5e47fd2 | L +95 | straight_road | event corridor (0.00) | event corridor (0.00) | 0.002 / 0.001 / 0.27 / 13.7 | n / n | n / n |
| hard_train | ea468790 | R -83 | curve_single | off (0.55) | completed (0.68) | 0.079 / 0.012 / 1.55 / 2.94 | n / n | y / y |
| hard_train | eaba3f6a | R -90 | straight_exit | event corridor (0.00) | event corridor (0.00) | 0.004 / 0.001 / 0.73 / 12.6 | n / n | n / n |
| hard_train | f9e361aa | R -81 | straight_exit | event corridor (0.00) | event corridor (0.00) | 0.002 / 0.001 / 0.37 / 11.5 | n / n | n / n |
| trainable | 132ae6cb | L +86 | no_straight_exit | completed (0.04) | completed (0.05) | 0.121 / 0.022 / 1.52 / 5.33 | y / y | y / y |
| trainable | 6fd8cf30 | R -88 | straight_exit | short_rear (0.30) | short_rear (0.32) | 0.056 / 0.009 / 0.55 / 3.22 | n / n | n / n |
| trainable | 8a143cb2 | R -77 | curve_single | completed (1.00) | completed (1.00) | 0.084 / 0.023 / 2.14 / 1.80 | y / y | y / y |

RSG's shape numbers are in `hard_train14.csv`. Outcomes: `event` = zero event before exit + 10 m; `off` = crossed > 2.5 m off the route;
`short_slow` / `short_rear` = still on the route when the rollout ended (slow / truncated by a rear contact). 132ae6cb "completes" with a
score of 0.04-0.05 (the turn is done, the rest of the rollout scores almost nothing). n = 14 is small; the CIs above are the honest size.
