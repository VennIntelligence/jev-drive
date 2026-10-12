# LOCAL-MAP: would a carried local map have saved the frozen probe at the road-exit zeros?

Pre-registration: [../plans/2026-10-12-local-map-prereg.md](../plans/2026-10-12-local-map-prereg.md) (pushed before any read; section 7 amended G-carry before the read). Estimates made d decisions earlier (d = 2 / 4 / 6 / 8 = 1 / 2 / 3 / 4 s) are carried into the current decision's rear-axle frame with the egomotion the driver receives and read at x = camera + D. **L1** = decision 275's probe (two-seed mean) ego-lane boundaries (three points per side at 10 / 20 / 40 m, joined, no extrapolation); **L2** = the route waypoints the driver received (lane-centre points ~4.2 m apart starting ~40 m ahead), error against the map lane centre (L + R) / 2; **L3** = L2's centre with L1's width (same d, else the current probe). d = 0 is the current decision (L1 d = 0 = decision 275's probe). Labels are decision 275's, unchanged: the lane the logged route occupies at D. Note that the route and the label both follow the logged path's lane, so L2 measures how well the *received* route carries that lane into the zero window's frame, not an independent map.

## Gates

| gate | reading |  |
|:--|--:|--:|
| G_frame | decisions 1764, anchor_share 1.0000, anchor_max 0.0000, sim_share 1.0000, sim_median 0.0000, sim_max 0.0000 | pass |
| G_carry | points 16047, median 0.0000, p95 0.0000, mirror_median 3.9183, chord_median 0.0006, chord_p95 0.0956 | pass |
| G_sign_mirror | ratio 1797896.0874 | pass |
| G_route | zero_points 16998, zero_share 0.9965, held_decisions 8129, held_share 0.9990, held_median 0.0000 | pass |
| G_sign | L20 1.4253, R20 -1.9826 | pass |
| G_d0 | r3 1.2551, r3_eval 45, j010 1.1492, j1020 1.6833 | pass |

## Z (registered): the 98 road-exit zero windows, D = 20 m

**Verdict: local map works: L2 newest covering 0.351 m (28 windows).** Line: a fixed rule (d in 2-8 or newest covering) with window median < 0.5 m on >= 23 covered windows; every covered fixed rule > 1.0 m = never seen clearly. Window error as decision 275's R3 (per-side window medians, mean of the sides; L2: the centre error); CI = window bootstrap (B 2000).

| line | rule | window median (m) | p90 (m) | share <= 0.5 m | covered windows |
|:--|--:|--:|--:|--:|--:|
| L1 carried probe | d = 0 (now) | 1.255 [1.033, 2.023] | 6.114 | 0.111 | 45 |
| L1 carried probe | d = 2 (1 s) | 2.020 [1.369, 3.901] | 8.027 | 0.103 | 39 |
| L1 carried probe | d = 4 (2 s) | 2.959 [1.437, 10.277] | 21.765 | 0.118 | 34 |
| L1 carried probe | d = 6 (3 s) | 2.423 [0.786, 5.709] | 25.273 | 0.105 | 21 |
| L1 carried probe | d = 8 (4 s) | 3.864 [0.726, 8.170] | 10.687 | 0.083 | 14 |
| L1 carried probe | oracle: best d (2-8) per window, not a rule | 1.653 [0.854, 3.901] | 8.027 | 0.154 | 39 |
| L2 carried route (centre) | d = 0 (now) | 38.164 [23.814, 41.951] | 41.951 | 0.000 | 6 |
| L2 carried route (centre) | d = 2 (1 s) | 32.516 [16.098, 42.886] | 42.138 | 0.000 | 5 |
| L2 carried route (centre) | d = 4 (2 s) | 36.766 [0.443, 42.886] | 42.325 | 0.250 | 4 |
| L2 carried route (centre) | d = 6 (3 s) | 0.294 [0.098, 32.516] | 41.016 | 0.636 | 11 |
| L2 carried route (centre) | d = 8 (4 s) | 0.203 [0.087, 0.418] | 34.216 | 0.789 | 19 |
| L2 carried route (centre) | newest covering | 0.351 [0.108, 3.653] | 38.164 | 0.607 | 28 |
| L2 carried route (centre) | oracle: best d (2-8) per window, not a rule | 0.215 [0.087, 0.431] | 33.366 | 0.750 | 20 |
| L3 route centre + probe width | d = 0 (now) | 38.164 [23.814, 41.951] | 41.951 | 0.000 | 6 |
| L3 route centre + probe width | d = 2 (1 s) | 32.516 [16.098, 42.886] | 42.138 | 0.000 | 5 |
| L3 route centre + probe width | d = 4 (2 s) | 36.766 [0.610, 42.886] | 42.325 | 0.000 | 4 |
| L3 route centre + probe width | d = 6 (3 s) | 0.412 [0.203, 32.516] | 41.016 | 0.455 | 11 |
| L3 route centre + probe width | d = 8 (4 s) | 0.331 [0.280, 0.715] | 34.216 | 0.474 | 19 |
| L3 route centre + probe width | newest covering | 0.604 [0.295, 3.636] | 38.164 | 0.357 | 28 |
| L3 route centre + probe width | oracle: best d (2-8) per window, not a rule | 0.358 [0.270, 0.618] | 33.366 | 0.550 | 20 |

Other distances (window median m / covered windows):

| line | rule | D = 10 m | D = 40 m |
|:--|--:|--:|--:|
| L1 | d = 0 (now) | 1.391 / 65 | 1.175 / 25 |
| L1 | d = 2 (1 s) | 2.089 / 59 | 15.916 / 1 |
| L1 | d = 4 (2 s) | 3.223 / 58 | - / 0 |
| L1 | d = 6 (3 s) | 3.850 / 53 | - / 0 |
| L1 | d = 8 (4 s) | 4.557 / 43 | - / 0 |
| L1 | oracle | 1.743 / 59 | 15.916 / 1 |
| L2 | d = 0 (now) | 53.047 / 4 | 0.076 / 18 |
| L2 | d = 2 (1 s) | 34.366 / 1 | 0.145 / 19 |
| L2 | d = 4 (2 s) | 33.361 / 1 | 0.137 / 18 |
| L2 | d = 6 (3 s) | 30.653 / 1 | 0.091 / 15 |
| L2 | d = 8 (4 s) | 0.273 / 6 | 0.106 / 12 |
| L2 | newest covering | 0.373 / 21 | 0.131 / 21 |
| L2 | oracle | 0.282 / 7 | 0.143 / 19 |
| L3 | d = 0 (now) | 53.047 / 4 | 0.705 / 13 |
| L3 | d = 2 (1 s) | 34.366 / 1 | 0.579 / 17 |
| L3 | d = 4 (2 s) | 33.361 / 1 | 0.580 / 15 |
| L3 | d = 6 (3 s) | 30.653 / 1 | 0.309 / 13 |
| L3 | d = 8 (4 s) | 0.302 / 3 | 0.416 / 10 |
| L3 | newest covering | 1.738 / 14 | 0.575 / 17 |
| L3 | oracle | 0.303 / 4 | 0.562 / 17 |

L2 secondary (distance from the map centre point at 20 m to the carried route polyline): d = 0 (now) 19.025 m (45 windows), d = 2 (1 s) 14.578 m (39 windows), d = 4 (2 s) 9.712 m (36 windows), d = 6 (3 s) 5.746 m (34 windows), d = 8 (4 s) 1.562 m (32 windows). Age of the newest route covering 20 m on window decisions: median 5.0 decisions, none on 0.405 of them.

## H (registered): held turn decisions (route turn >= 20 deg) 0-20 m before a junction, D = 20 m

785 decisions; per decision both sides pooled (L2: centre error); scene-cluster bootstrap (B 2000).

| line | rule | median (m) | p90 (m) | n (points) |
|:--|--:|--:|--:|--:|
| L1 carried probe | d = 0 (now) | 1.377 [0.975, 1.863] | 4.466 [3.256, 6.119] | 380 |
| L1 carried probe | d = 2 (1 s) | 2.181 [1.423, 3.066] | 6.935 [5.210, 9.285] | 302 |
| L1 carried probe | d = 4 (2 s) | 3.022 [2.113, 5.495] | 14.062 [8.801, 21.031] | 239 |
| L1 carried probe | d = 6 (3 s) | 4.141 [2.245, 6.546] | 11.110 [8.261, 22.330] | 185 |
| L1 carried probe | d = 8 (4 s) | 5.878 [3.656, 8.148] | 12.320 [8.619, 21.869] | 112 |
| L2 carried route (centre) | d = 0 (now) | 47.189 [31.635, 47.392] | 49.957 [31.635, 50.285] | 14 |
| L2 carried route (centre) | d = 2 (1 s) | 46.985 [31.635, 47.189] | 48.980 [31.635, 49.162] | 13 |
| L2 carried route (centre) | d = 4 (2 s) | 47.020 [31.635, 47.243] | 48.979 [31.635, 49.158] | 13 |
| L2 carried route (centre) | d = 6 (3 s) | 43.203 [0.218, 47.404] | 48.806 [2.282, 49.157] | 15 |
| L2 carried route (centre) | d = 8 (4 s) | 0.568 [0.418, 3.085] | 35.059 [0.701, 49.006] | 24 |
| L2 carried route (centre) | newest covering | 1.285 [0.411, 42.750] | 47.664 [1.323, 49.190] | 48 |
| L3 route centre + probe width | d = 0 (now) | 47.173 [31.635, 47.359] | 50.139 [31.676, 50.577] | 28 |
| L3 route centre + probe width | d = 2 (1 s) | 46.947 [31.635, 47.195] | 49.162 [31.756, 49.301] | 26 |
| L3 route centre + probe width | d = 4 (2 s) | 47.020 [31.635, 47.243] | 49.158 [31.717, 49.465] | 26 |
| L3 route centre + probe width | d = 6 (3 s) | 43.203 [0.272, 47.425] | 48.973 [3.399, 49.387] | 30 |
| L3 route centre + probe width | d = 8 (4 s) | 0.661 [0.254, 3.113] | 47.770 [1.118, 49.021] | 48 |
| L3 route centre + probe width | newest covering | 1.264 [0.316, 43.146] | 47.730 [1.717, 49.279] | 96 |

Descriptive, other held subsets (median m / points; CI B 300):

| line | rule | turns in junction | turns 20-60 m before | straight |
|:--|--:|--:|--:|--:|
| L1 | d = 0 (now) | 1.114 / 278 | 0.442 / 258 | 0.253 / 8166 |
| L1 | d = 2 (1 s) | 3.290 / 242 | 1.228 / 214 | 0.292 / 7420 |
| L1 | d = 4 (2 s) | 9.062 / 190 | 1.353 / 160 | 0.375 / 5762 |
| L1 | d = 6 (3 s) | 8.838 / 104 | 3.802 / 62 | 0.410 / 3513 |
| L1 | d = 8 (4 s) | 6.659 / 46 | 12.811 / 32 | 0.379 / 2121 |
| L2 | d = 0 (now) | - / 0 | 28.434 / 27 | 35.393 / 24 |
| L2 | d = 2 (1 s) | - / 0 | 19.510 / 21 | 33.437 / 19 |
| L2 | d = 4 (2 s) | - / 0 | 19.262 / 12 | 0.028 / 452 |
| L2 | d = 6 (3 s) | 0.126 / 19 | 0.062 / 34 | 0.034 / 1143 |
| L2 | d = 8 (4 s) | 0.143 / 34 | 0.081 / 38 | 0.041 / 1382 |
| L2 | newest covering | 0.142 / 61 | 0.248 / 71 | 0.042 / 2445 |
| L3 | d = 0 (now) | - / 0 | 28.412 / 54 | 35.409 / 48 |
| L3 | d = 2 (1 s) | - / 0 | 20.552 / 42 | 33.437 / 38 |
| L3 | d = 4 (2 s) | - / 0 | 19.215 / 24 | 0.061 / 904 |
| L3 | d = 6 (3 s) | 0.339 / 38 | 0.247 / 68 | 0.111 / 2286 |
| L3 | d = 8 (4 s) | 0.316 / 68 | 0.388 / 76 | 0.133 / 2764 |
| L3 | newest covering | 0.282 / 122 | 0.538 / 142 | 0.127 / 4890 |

## Plan check (registered): would a lane-keeping rule on the carried map have disagreed with the served plan before the zero?

Per window: the served plan's rear-axle path at 10 or 20 m outside the carried lane [R, L] on any window decision. 'map says out' uses the true lane labels on the same decisions (windows where both are evaluable); lead = decisions from the first disagreement to k_ev.

| line | rule | evaluable windows | carried map says out | map says out (both eval.) | both say out | carried out, map in | median lead (decisions) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| L1 carried probe | d = 0 (now) | 80 | 57 | 37 | 30 | 3 | 3.0 |
| L1 carried probe | d = 2 (1 s) | 75 | 64 | 36 | 33 | 3 | 4.0 |
| L1 carried probe | d = 4 (2 s) | 75 | 64 | 37 | 34 | 3 | 4.0 |
| L1 carried probe | d = 6 (3 s) | 72 | 65 | 37 | 34 | 3 | 4.0 |
| L1 carried probe | d = 8 (4 s) | 60 | 57 | 33 | 32 | 1 | 3.0 |
| L3 route centre + probe width | d = 0 (now) | 15 | 15 | 2 | 2 | 3 | 2.0 |
| L3 route centre + probe width | d = 2 (1 s) | 16 | 16 | 3 | 3 | 1 | 2.0 |
| L3 route centre + probe width | d = 4 (2 s) | 21 | 21 | 6 | 6 | 1 | 2.0 |
| L3 route centre + probe width | d = 6 (3 s) | 31 | 29 | 10 | 10 | 1 | 2.0 |
| L3 route centre + probe width | d = 8 (4 s) | 29 | 25 | 11 | 9 | 0 | 2.0 |
| L3 route centre + probe width | newest covering | 45 | 40 | 21 | 18 | 3 | 2.0 |

Reference: on the true map the served plan leaves the route lane at 10 or 20 m in 37 of the 53 windows where the map lane is defined.

## Exploratory (decided after the read, not registered)

- **Wrong-leg crossings.** The registered first-crossing rule also takes a route that starts beyond the 20 m line and bends back across it (a route received before a turn crosses x = camera + 20 m on the far leg, tens of metres to the side): that is the 30-50 m tail of L2 / L3 at d = 0-4 and on held turns. A label-free fix, a route counts only if its carried start lies at or behind the line:

| line | rule | Z window median (m) | Z p90 | Z windows | H pre-junction median (m) | H points |
|:--|--:|--:|--:|--:|--:|--:|
| L2 carried route (centre) | newest covering | 0.351 [0.108, 3.653] | 38.164 | 28 | 1.285 [0.411, 42.750] | 48 |
| L2 carried route (centre) | newest covering, route starts at or behind the line | 0.226 [0.098, 0.443] | 10.873 | 25 | 0.583 [0.332, 1.351] | 34 |
| L3 route centre + probe width | newest covering | 0.604 [0.295, 3.636] | 38.164 | 28 | 1.264 [0.316, 43.146] | 96 |
| L3 route centre + probe width | newest covering, route starts at or behind the line | 0.412 [0.260, 0.715] | 10.873 | 25 | 0.618 [0.289, 1.259] | 68 |

- **Why coverage is low.** In the window decisions the car covers a median 7.1 m in 2 s and 13.7 m in 4 s (slow, junction speeds); a route received 4 s earlier starts a median 17.6 m ahead of the camera, so it reaches the 20 m line only when the car has moved ~20 m since. 0.422 of the window decisions have no received route (of any age) covering 20 m by the start-behind rule (and 0.267 of all window decisions fall in the first 8 decisions of a rollout). The probe's own memory (L1) reaches 20 m only through its 40 m point, which is the probe's weakest output on turns, and gets worse with d.

- **Base rate of the plan check on ordinary driving.** Share of decisions where the rule says the served plan's rear axle is outside the lane at 10 or 20 m (evaluable decisions in brackets). On ordinary held driving the rule fires on 7-25 % of decisions, and the true map itself says 'out' on 16 % (the served plan crosses the route lane's lines in normal driving, and the label lane is the logged route's lane): a disagreement in the window is informative only against that base rate, and the carried-probe versions fire more often than the current probe.

| lane source | zero-window decisions | held, all decisions | held straight |
|:--|--:|--:|--:|
| probe now (L1 d = 0) | 0.533 (229) | 0.117 (4651) | 0.067 (2682) |
| probe carried 2 s (L1 d = 4) | 0.833 (305) | 0.250 (4360) | 0.114 (2548) |
| L3 newest, start-behind | 0.854 (89) | 0.160 (2038) | 0.115 (1472) |
| map labels | 0.664 (149) | 0.163 (5083) | 0.121 (3024) |

