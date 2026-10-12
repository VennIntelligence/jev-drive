# EYES-VS-HEAD: did openpilot's own perception heads read the scene when our driver scored zero?

Registered in [plans/2026-10-12-eyes-vs-head-prereg.md](../plans/2026-10-12-eyes-vs-head-prereg.md) (pushed before any read, `afac87a7`); script `scripts/ev.py` (msgs / replay / read) and `scripts/ev_report.py`; box `$DATA_DIR/runs/eyes_vs_head/` (msgs, replay pickles, read run dir `20261012-165552`). Heads: **P0** = the shipped openpilot policy weights without our adapter on the same vision tokens (registered); **FT** = the served checkpoint's own heads (descriptive). Lane-line probability = sigmoid(lane_lines_prob)[1::2], openpilot's `laneLineProbs`.

## Gates

| gate | line | read | verdict |
|:--|--:|--:|--:|
| G-msg | >= 99% of the selected zero rollouts with asl and messages | 316 / 316 rollouts, 0 errors (every label also has 8 held scenes scored 0 with no rollout at all, the route-fold scenes zeros_mech skips) | pass |
| G-rep | replayed served plan within 0.10 m of the stored one at 4 s on >= 95% | 1.0000 of 5685 (median 1.2 cm) | pass |
| G-frame | decision-0 anchor within 0.10 m of the logged rear axle on >= 99% | 1.0000 of 516 (max 8.5e-09 m) | pass |
| G-sign | ctrl straight, ego frame left +: edge 0 > line 1 > 0 > line 2 > edge 1 (medians at 10 m); edge error < mirrored | +6.32 > +1.33 > 0 > -1.31 > -4.15; 2.24 < 3.35 m | pass |

The prereg's second half of G-sign (lane-line error against the mirrored line) is uninformative for lines that sit symmetric about the ego, so the mirror test is run on the road edges (asymmetric); stated here, not changed after a read.

## R1 eyes quality overall

Samples: **ctrl** = col1's control re-run (P2H10-F-s0, 200 public scenes, every rollout's asl kept, 10 decisions; no selection), **held** = every decision of the zero-score rollouts of YR10m10-F-s0/s1 and YS-L18-s0/s1 (selected on failure), by position relative to the event. straight / turn = the route command at the decision (STRAIGHT vs LEFT / RIGHT). both = lines 1 and 2 (the ego lane) both with prob > 0.5; p0..p3 = presence of each line. Errors (m) = distance from the line's point at d m ahead of the camera to the nearest map lane boundary on the cross-section, lines with prob > 0.5 only (n20 = such line points at 20 m).

| sample | head | n | both | p0 | p1 | p2 | p3 | med10 | p90_10 | med20 | p90_20 | n20 | med40 | p90_40 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl straight | P0 | 1135 | 0.240 | 0.029 | 0.323 | 0.340 | 0.011 | 0.495 | 0.771 | 0.468 | 0.762 | 753 | 0.441 | 0.792 |
| ctrl turn | P0 | 865 | 0.003 | 0.000 | 0.020 | 0.018 | 0.000 | 0.622 | 1.019 | 0.639 | 0.905 | 33 | 0.659 | 1.544 |
| ctrl all | P0 | 2000 | 0.138 | 0.017 | 0.192 | 0.201 | 0.006 | 0.500 | 0.779 | 0.480 | 0.773 | 786 | 0.444 | 0.808 |
| held pre-window straight | P0 | 295 | 0.142 | 0.027 | 0.237 | 0.275 | 0.010 | 0.493 | 1.174 | 0.418 | 1.148 | 151 | 0.342 | 1.275 |
| held pre-window turn | P0 | 1566 | 0.006 | 0.000 | 0.010 | 0.019 | 0.001 | 0.677 | 1.025 | 0.662 | 1.406 | 46 | 0.831 | 1.820 |
| held window (2 s before event) | P0 | 1099 | 0.005 | 0.000 | 0.012 | 0.028 | 0.001 | 0.497 | 1.119 | 0.568 | 1.116 | 44 | 1.091 | 1.538 |
| held after event | P0 | 1231 | 0.000 | 0.000 | 0.019 | 0.038 | 0.000 | 0.628 | 1.020 | 0.591 | 1.147 | 70 | 0.471 | 1.880 |

FT (the served checkpoint's own heads; descriptive):

| sample | head | n | both | p0 | p1 | p2 | p3 | med10 | p90_10 | med20 | p90_20 | n20 | med40 | p90_40 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl straight | FT | 1135 | 0.241 | 0.028 | 0.323 | 0.342 | 0.011 | 0.496 | 0.770 | 0.466 | 0.762 | 755 | 0.443 | 0.791 |
| ctrl turn | FT | 865 | 0.003 | 0.000 | 0.021 | 0.018 | 0.000 | 0.618 | 1.012 | 0.600 | 0.894 | 34 | 0.640 | 1.505 |
| ctrl all | FT | 2000 | 0.138 | 0.016 | 0.193 | 0.202 | 0.006 | 0.500 | 0.779 | 0.478 | 0.771 | 789 | 0.445 | 0.806 |
| held pre-window straight | FT | 295 | 0.142 | 0.027 | 0.241 | 0.271 | 0.010 | 0.493 | 1.210 | 0.427 | 1.156 | 151 | 0.349 | 1.281 |
| held pre-window turn | FT | 1566 | 0.006 | 0.000 | 0.010 | 0.020 | 0.001 | 0.665 | 1.023 | 0.662 | 1.406 | 47 | 0.839 | 1.900 |
| held window (2 s before event) | FT | 1099 | 0.005 | 0.000 | 0.013 | 0.029 | 0.001 | 0.501 | 1.119 | 0.572 | 1.143 | 46 | 1.092 | 1.548 |
| held after event | FT | 1231 | 0.000 | 0.000 | 0.019 | 0.039 | 0.000 | 0.631 | 1.015 | 0.594 | 1.158 | 71 | 0.474 | 1.936 |

Not registered, for reading the line: errors of lines 1 and 2 at every decision regardless of probability (medall / p90all), the signed error at 20 m (s1 = line 1, positive = further left than the nearest boundary; s2 = line 2), and the road-edge heads against the road-area boundary (re_*):

| sample | medall10 | p90all10 | medall20 | p90all20 | medall40 | s1_20 | s2_20 | re_med10 | re_p90_10 | re_med20 | re_p90_20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl straight | 0.434 | 0.772 | 0.408 | 0.830 | 0.418 | -0.345 | 0.401 | 2.185 | 10.996 | 2.341 | 11.203 |
| ctrl turn | 0.590 | 1.732 | 0.819 | 2.925 | 1.135 | -0.402 | -0.075 | 2.357 | 9.170 | 2.554 | 9.796 |
| ctrl all | 0.473 | 1.240 | 0.511 | 1.631 | 0.539 | -0.351 | 0.313 | 2.244 | 10.252 | 2.419 | 10.683 |
| held pre-window straight | 0.492 | 1.432 | 0.489 | 1.460 | 0.526 | -0.234 | 0.228 | 1.568 | 7.050 | 1.923 | 8.633 |
| held pre-window turn | 0.584 | 2.126 | 0.865 | 4.259 | 1.379 | -0.075 | 0.227 | 2.352 | 10.699 | 3.150 | 11.638 |
| held window (2 s before event) | 0.994 | 4.530 | 1.455 | 9.089 | 1.605 | -0.205 | -0.081 | 2.742 | 9.393 | 2.993 | 11.311 |
| held after event | 1.068 | 5.632 | 1.178 | 5.772 | 1.223 | -0.190 | 0.075 | 1.785 | 7.082 | 1.713 | 7.375 |

Exploratory, not registered: P0 with decision 104's scale undone (lateral offsets from the camera divided by 0.689, the lane-width ratio openpilot reads on real nuPlan frames; fixed before this read, not fitted here), errors at every decision regardless of probability:

| sample | medall10 | p90all10 | medall20 | p90all20 | medall40 | s1_20 | s2_20 | re_med20 | re_p90_20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl straight | 0.211 | 0.774 | 0.246 | 0.859 | 0.377 | 0.198 | -0.111 | 1.165 | 13.786 |
| ctrl turn | 0.487 | 1.677 | 0.746 | 2.968 | 1.143 | -0.053 | -0.300 | 2.099 | 12.710 |
| ctrl all | 0.290 | 1.245 | 0.367 | 1.621 | 0.538 | 0.156 | -0.151 | 1.527 | 13.416 |
| held pre-window straight | 0.360 | 1.309 | 0.413 | 1.438 | 0.612 | 0.126 | -0.143 | 1.052 | 7.108 |
| held pre-window turn | 0.563 | 2.065 | 0.787 | 4.168 | 1.243 | 0.079 | 0.030 | 2.686 | 14.627 |
| held window (2 s before event) | 0.926 | 4.504 | 1.410 | 9.222 | 1.637 | 0.063 | -0.183 | 2.793 | 13.177 |
| held after event | 0.869 | 5.888 | 1.034 | 5.449 | 1.134 | 0.077 | 0.087 | 1.562 | 8.215 |

**Line (ctrl straight, P0).** (i) both ego lines present 0.240 (line >= 0.80): NOT met; (ii) median error at 20 m of present ego lines 0.468 m (line < 0.30): NOT met; (iii) p90 0.762 m (line < 0.75): NOT met -> **eyes not usable by the registered line**.

Splits (P0): junction = rear axle in two lane polygons > 30 deg apart; city = the scene's map.

| sample | n | both | p1 | p2 | med20 | p90_20 | n20 | medall20 | s1_20 | s2_20 | re_med20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ctrl straight, junction | 46 | 0.022 | 0.043 | 0.043 | 0.421 | 0.502 | 4 | 0.379 | -0.386 | 0.231 | 3.737 |
| ctrl straight, no junction | 1089 | 0.249 | 0.335 | 0.353 | 0.469 | 0.764 | 749 | 0.409 | -0.345 | 0.407 | 2.267 |
| ctrl turn, junction | 72 | 0.000 | 0.000 | 0.000 | - | - | 0 | 1.034 | -0.244 | 0.076 | 4.523 |
| ctrl turn, no junction | 793 | 0.004 | 0.021 | 0.020 | 0.639 | 0.905 | 33 | 0.808 | -0.415 | -0.084 | 2.470 |
| ctrl, las_vegas | 2000 | 0.138 | 0.192 | 0.201 | 0.480 | 0.773 | 786 | 0.511 | -0.351 | 0.313 | 2.419 |
| held pre, boston | 267 | 0.094 | 0.116 | 0.127 | 0.941 | 1.485 | 65 | 0.847 | -0.505 | 0.069 | 2.597 |
| held pre, las_vegas | 408 | 0.056 | 0.098 | 0.098 | 0.408 | 0.589 | 80 | 0.594 | -0.154 | 0.130 | 2.391 |
| held pre, pittsburgh | 589 | 0.000 | 0.020 | 0.007 | 0.587 | 0.959 | 16 | 0.599 | -0.160 | 0.261 | 3.314 |
| held pre, singapore | 597 | 0.005 | 0.005 | 0.055 | 0.322 | 1.166 | 36 | 1.223 | 0.224 | 0.390 | 3.014 |

## R2 the zeros: eyes right / head wrong (a), eyes wrong (b), ambiguous (c)

Window = decisions k_ev - 4 .. k_ev (2 s before the event). Per decision: eyes right = lines 1 and 2 present (prob > 0.5), both within 0.5 m of a map boundary at 10 m and 20 m, and the two matched boundaries adjacent (the ego lane framed); eyes wrong = line 1 or 2 missing (prob < 0.5) or > 1.0 m off at 10 or 20 m; crosses = the served plan's rear axle beyond the predicted line 1 / 2 or road edge within 0-40 m ahead of the camera. (a) some window decision with eyes right and a crossing on the failure side (collisions: any side); (b) not (a) and eyes wrong on at least half of the window's evaluable decisions; (c) the rest: c1 eyes right but the plan stays inside the seen lane, c2 errors 0.5-1 m or not evaluable, c3 crossing on the other side.

P0 heads (registered):

| group | type | n | a | b | c | c1 | c2 | c3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | corridor | 31 | 0 | 31 | 0 | 0 | 0 | 0 |
| YR10m10-F-s0 | offroad | 33 | 0 | 33 | 0 | 0 | 0 | 0 |
| YR10m10-F-s0 | collision | 10 | 0 | 9 | 1 | 0 | 1 | 0 |
| YR10m10-F-s1 | corridor | 30 | 0 | 30 | 0 | 0 | 0 | 0 |
| YR10m10-F-s1 | offroad | 37 | 0 | 37 | 0 | 0 | 0 | 0 |
| YR10m10-F-s1 | collision | 4 | 0 | 4 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | corridor | 22 | 0 | 22 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | offroad | 14 | 0 | 14 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | collision | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | corridor | 19 | 0 | 19 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | offroad | 15 | 0 | 15 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | collision | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | corridor | 19 | 0 | 19 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | offroad | 17 | 0 | 17 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | collision | 7 | 0 | 7 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | corridor | 20 | 0 | 20 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | offroad | 13 | 0 | 13 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | collision | 7 | 0 | 7 | 0 | 0 | 0 | 0 |

**Line.** YS-L18-s1 + YR10m10-F-s0 corridor + offroad zeros: (a) 0 / 98 = 0.000 (line >= 0.60), (b) 98 / 98 = 1.000 (line >= 0.40) -> **the eyes are not good enough on these scenes**. Per driver: YR10m10-F-s0 (a) 0.000 / (b) 1.000 of 64; YS-L18-s1 (a) 0.000 / (b) 1.000 of 34.

FT heads (descriptive):

| group | type | n | a | b | c | c1 | c2 | c3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | corridor | 31 | 0 | 31 | 0 | 0 | 0 | 0 |
| YR10m10-F-s0 | offroad | 33 | 0 | 33 | 0 | 0 | 0 | 0 |
| YR10m10-F-s0 | collision | 10 | 0 | 9 | 1 | 0 | 1 | 0 |
| YR10m10-F-s1 | corridor | 30 | 0 | 30 | 0 | 0 | 0 | 0 |
| YR10m10-F-s1 | offroad | 37 | 0 | 37 | 0 | 0 | 0 | 0 |
| YR10m10-F-s1 | collision | 4 | 0 | 4 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | corridor | 22 | 0 | 22 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | offroad | 14 | 0 | 14 | 0 | 0 | 0 | 0 |
| YS-L18-s0 | collision | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | corridor | 19 | 0 | 19 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | offroad | 15 | 0 | 15 | 0 | 0 | 0 | 0 |
| YS-L18-s1 | collision | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | corridor | 19 | 0 | 19 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | offroad | 17 | 0 | 17 | 0 | 0 | 0 | 0 |
| YS2G-L18-s0 | collision | 7 | 0 | 7 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | corridor | 20 | 0 | 20 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | offroad | 13 | 0 | 13 | 0 | 0 | 0 | 0 |
| YS2G-L18-s1 | collision | 7 | 0 | 7 | 0 | 0 | 0 | 0 |

Exploratory, not registered (decided after seeing that the probabilities are low everywhere, R1): the same rule with the probability condition dropped (the ego-lane adjacency check is also dropped in this variant):

| group | type | n | a | b | c | c1 | c2 | c3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | corridor | 31 | 1 | 28 | 2 | 0 | 0 | 2 |
| YR10m10-F-s0 | offroad | 33 | 1 | 30 | 2 | 1 | 1 | 0 |
| YR10m10-F-s0 | collision | 10 | 1 | 6 | 3 | 1 | 2 | 0 |
| YR10m10-F-s1 | corridor | 30 | 1 | 27 | 2 | 0 | 0 | 2 |
| YR10m10-F-s1 | offroad | 37 | 0 | 34 | 3 | 1 | 2 | 0 |
| YR10m10-F-s1 | collision | 4 | 0 | 2 | 2 | 1 | 1 | 0 |
| YS-L18-s0 | corridor | 22 | 1 | 19 | 2 | 1 | 1 | 0 |
| YS-L18-s0 | offroad | 14 | 1 | 12 | 1 | 0 | 1 | 0 |
| YS-L18-s0 | collision | 9 | 0 | 6 | 3 | 1 | 2 | 0 |
| YS-L18-s1 | corridor | 19 | 1 | 17 | 1 | 0 | 0 | 1 |
| YS-L18-s1 | offroad | 15 | 1 | 13 | 1 | 0 | 1 | 0 |
| YS-L18-s1 | collision | 9 | 0 | 8 | 1 | 1 | 0 | 0 |
| YS2G-L18-s0 | corridor | 19 | 0 | 18 | 1 | 0 | 0 | 1 |
| YS2G-L18-s0 | offroad | 17 | 0 | 16 | 1 | 0 | 1 | 0 |
| YS2G-L18-s0 | collision | 7 | 1 | 4 | 2 | 1 | 1 | 0 |
| YS2G-L18-s1 | corridor | 20 | 1 | 18 | 1 | 0 | 0 | 1 |
| YS2G-L18-s1 | offroad | 13 | 0 | 12 | 1 | 0 | 1 | 0 |
| YS2G-L18-s1 | collision | 7 | 1 | 4 | 2 | 1 | 1 | 0 |

Exploratory, not registered: the geometry-only rule on the scale-corrected P0 heads (above):

| group | type | n | a | b | c | c1 | c2 | c3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | corridor | 31 | 0 | 29 | 2 | 0 | 0 | 2 |
| YR10m10-F-s0 | offroad | 33 | 0 | 27 | 6 | 5 | 1 | 0 |
| YR10m10-F-s0 | collision | 10 | 0 | 7 | 3 | 3 | 0 | 0 |
| YR10m10-F-s1 | corridor | 30 | 0 | 28 | 2 | 1 | 0 | 1 |
| YR10m10-F-s1 | offroad | 37 | 0 | 30 | 7 | 4 | 2 | 1 |
| YR10m10-F-s1 | collision | 4 | 0 | 2 | 2 | 1 | 1 | 0 |
| YS-L18-s0 | corridor | 22 | 1 | 17 | 4 | 1 | 2 | 1 |
| YS-L18-s0 | offroad | 14 | 0 | 11 | 3 | 2 | 1 | 0 |
| YS-L18-s0 | collision | 9 | 1 | 7 | 1 | 1 | 0 | 0 |
| YS-L18-s1 | corridor | 19 | 1 | 16 | 2 | 1 | 1 | 0 |
| YS-L18-s1 | offroad | 15 | 0 | 12 | 3 | 3 | 0 | 0 |
| YS-L18-s1 | collision | 9 | 0 | 8 | 1 | 1 | 0 | 0 |
| YS2G-L18-s0 | corridor | 19 | 0 | 17 | 2 | 1 | 0 | 1 |
| YS2G-L18-s0 | offroad | 17 | 0 | 14 | 3 | 2 | 1 | 0 |
| YS2G-L18-s0 | collision | 7 | 0 | 3 | 4 | 2 | 2 | 0 |
| YS2G-L18-s1 | corridor | 20 | 0 | 18 | 2 | 2 | 0 | 0 |
| YS2G-L18-s1 | offroad | 13 | 0 | 10 | 3 | 2 | 1 | 0 |
| YS2G-L18-s1 | collision | 7 | 0 | 4 | 3 | 3 | 0 | 0 |

YR10m10-F-s0 + YS-L18-s1 by zero type:

| rule | corridor_a | corridor_b | corridor_c | offroad_a | offroad_b | offroad_c | collision_a | collision_b | collision_c |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| registered (P0) | 0 | 50 | 0 | 0 | 48 | 0 | 0 | 18 | 1 |
| geometry only (P0, exploratory) | 2 | 45 | 3 | 2 | 43 | 3 | 1 | 14 | 4 |
| geometry only, scale-corrected (P0, exploratory) | 1 | 45 | 4 | 0 | 39 | 9 | 0 | 15 | 4 |
| FT | 0 | 50 | 0 | 0 | 48 | 0 | 0 | 18 | 1 |

## R3 the lead head at the at-fault collisions

All rollouts of the six labels with an at-fault contact (k_c >= 0; zeros_mech). Window k_c - 4 .. k_c. reported = lead_prob > 0.5 with the lead within 2 m of the struck object's box point nearest the camera and its speed within 2 m/s of the object's speed along the ego heading; other = lead_prob > 0.5 but not matching; none = lead_prob <= 0.5 throughout. Object class by its heading relative to the ego at k_c - 2.

| hit | n | reported | other | none |
|:--|--:|--:|--:|--:|
| same | 28 | 0 | 19 | 9 |
| crossing | 8 | 0 | 0 | 8 |
| oncoming | 15 | 0 | 3 | 12 |
| - | 0 | 0 | 0 | 0 |

Same-direction collisions: reported 0 / 28. The lead head is forward-only and trained on in-path leads, so crossing and oncoming objects are listed separately and are not a failure of the head.

| group | sid | type | k_c | hit | dpsi | res | lp_max | min_dist | hit_x | hit_y | lead_x |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | daf38cec | collision | 10 | crossing | 111.960 | none | 0.177 | - | 3.842 | 0.330 | 17.937 |
| YS-L18-s0 | daf38cec | collision | 11 | crossing | 91.422 | none | 0.203 | - | 3.169 | 0.518 | 13.796 |
| YS-L18-s0 | 84b38cda | collision | 14 | crossing | -92.716 | none | 0.237 | - | 3.539 | -0.320 | 13.711 |
| YS-L18-s1 | daf38cec | collision | 11 | crossing | 90.672 | none | 0.202 | - | 3.390 | 0.449 | 10.507 |
| YS-L18-s1 | 84b38cda | collision | 14 | crossing | -81.756 | none | 0.329 | - | 3.406 | -0.154 | 11.312 |
| YS2G-L18-s0 | 84b38cda | collision | 14 | crossing | -79.217 | none | 0.094 | - | 3.433 | -0.252 | 14.101 |
| YS2G-L18-s1 | daf38cec | collision | 11 | crossing | 92.080 | none | 0.334 | - | 2.807 | 1.239 | 13.476 |
| YS2G-L18-s1 | 84b38cda | collision | 15 | crossing | -87.430 | none | 0.125 | - | 2.593 | -0.804 | 25.234 |
| YR10m10-F-s0 | bf00c76f | collision | 15 | oncoming | -154.593 | other | 0.626 | 3.233 | 3.897 | 0.282 | 10.637 |
| YR10m10-F-s0 | 67077701 | offroad | 16 | oncoming | 148.988 | none | 0.132 | - | 0.928 | -1.015 | 11.793 |
| YR10m10-F-s0 | e14711ef | corridor | 12 | oncoming | -127.987 | none | 0.037 | - | 1.569 | 0.194 | 18.933 |
| YR10m10-F-s0 | 8551466c | collision | 16 | oncoming | -148.065 | none | 0.454 | - | 2.451 | -0.778 | 11.975 |
| YR10m10-F-s1 | bf00c76f | collision | 16 | oncoming | -168.901 | other | 0.899 | 1.513 | 2.614 | -0.062 | 11.036 |
| YR10m10-F-s1 | 67077701 | corridor | 15 | oncoming | -178.422 | none | 0.138 | - | 1.604 | -0.105 | 19.856 |
| YR10m10-F-s1 | e14711ef | corridor | 12 | oncoming | -126.688 | none | 0.052 | - | 2.827 | -1.481 | 15.284 |
| YS-L18-s0 | 47e0ff3b | collision | 16 | oncoming | -169.235 | none | 0.342 | - | 2.291 | 0.158 | 18.656 |
| YS-L18-s0 | e14711ef | corridor | 14 | oncoming | -129.768 | none | 0.030 | - | 2.226 | -0.870 | 16.605 |
| YS-L18-s1 | bf00c76f | collision | 17 | oncoming | -179.891 | other | 0.974 | 0.627 | 2.037 | -0.047 | 12.481 |
| YS-L18-s1 | 47e0ff3b | collision | 16 | oncoming | -170.946 | none | 0.098 | - | 1.742 | -0.098 | 18.609 |
| YS-L18-s1 | bd70654a | collision | 14 | oncoming | 176.754 | none | 0.216 | - | 1.608 | 0.998 | 18.200 |
| YS2G-L18-s0 | e6056b57 | corridor | 10 | oncoming | 149.082 | none | 0.320 | - | 2.606 | -0.818 | 15.512 |
| YS2G-L18-s0 | 67077701 | collision | 15 | oncoming | 177.459 | none | 0.021 | - | 3.073 | -0.855 | 32.762 |
| YS2G-L18-s1 | e6056b57 | corridor | 10 | oncoming | 148.813 | none | 0.251 | - | 2.477 | -0.764 | 15.411 |
| YR10m10-F-s0 | cecd27d3 | collision | 16 | same | 4.058 | none | 0.494 | - | 1.935 | -1.244 | 16.227 |
| YR10m10-F-s0 | 0c3c4cd5 | collision | 17 | same | 0.993 | none | 0.247 | - | 1.686 | 0.849 | 19.592 |
| YR10m10-F-s0 | 9d18c2e4 | collision | 16 | same | -6.619 | other | 0.995 | 5.235 | 2.069 | 1.213 | 10.858 |
| YR10m10-F-s0 | 0722adec | collision | 11 | same | -4.937 | other | 0.976 | 27.639 | 2.378 | 1.171 | 31.867 |
| YR10m10-F-s0 | e14f1ffb | collision | 17 | same | -1.358 | none | 0.366 | - | 1.756 | 1.124 | 8.845 |
| YR10m10-F-s0 | 16661524 | collision | 16 | same | 20.346 | other | 0.669 | 3.141 | 2.967 | -1.433 | 6.060 |
| YR10m10-F-s0 | 700237df | collision | 12 | same | 8.325 | none | 0.362 | - | 1.953 | -1.187 | 15.035 |
| YR10m10-F-s1 | 0722adec | collision | 11 | same | -4.557 | other | 0.967 | 21.661 | 1.960 | 1.129 | 23.617 |
| YR10m10-F-s1 | cebd7d74 | offroad | 16 | same | -13.722 | none | 0.048 | - | 2.696 | 1.443 | 7.792 |
| YR10m10-F-s1 | 700237df | collision | 15 | same | 9.368 | none | 0.383 | - | 0.968 | -1.002 | 16.980 |
| YS-L18-s0 | 8051c60f | collision | 12 | same | -5.606 | other | 1.000 | 9.190 | 1.048 | -1.092 | 10.194 |
| YS-L18-s0 | 7bbb6cfc | collision | 17 | same | -5.723 | other | 0.994 | 12.695 | 1.822 | 1.239 | 14.377 |
| YS-L18-s0 | b49e6399 | collision | 9 | same | -1.660 | other | 0.942 | 23.483 | 2.380 | 1.123 | 14.632 |
| YS-L18-s0 | 0722adec | collision | 10 | same | -5.302 | other | 0.933 | 21.368 | 1.859 | 1.215 | 12.992 |
| YS-L18-s0 | e14f1ffb | collision | 16 | same | -1.967 | other | 0.550 | 18.887 | 1.830 | 0.937 | 17.310 |
| YS-L18-s1 | 8051c60f | collision | 9 | same | -5.401 | other | 0.999 | 10.050 | 1.350 | 1.028 | 11.319 |
| YS-L18-s1 | 7bbb6cfc | collision | 17 | same | -4.827 | none | 0.315 | - | 1.981 | 1.126 | 14.978 |
| YS-L18-s1 | 0722adec | collision | 10 | same | -4.645 | other | 0.940 | 23.707 | 1.920 | 1.267 | 25.570 |
| YS2G-L18-s0 | 8051c60f | collision | 12 | same | -4.980 | other | 0.992 | 9.609 | 1.534 | -1.173 | 11.123 |
| YS2G-L18-s0 | 7bbb6cfc | collision | 17 | same | -2.416 | none | 0.315 | - | 1.917 | 0.889 | 14.432 |
| YS2G-L18-s0 | 9a89dccc | collision | 6 | same | 7.402 | other | 0.975 | 10.010 | 1.855 | -1.331 | 14.194 |
| YS2G-L18-s0 | b49e6399 | collision | 9 | same | 0.729 | other | 0.933 | 12.905 | 2.157 | 1.475 | 15.022 |
| YS2G-L18-s0 | e14f1ffb | collision | 16 | same | -4.166 | none | 0.491 | - | 1.977 | 0.696 | 9.865 |
| YS2G-L18-s1 | 8051c60f | collision | 10 | same | -5.323 | other | 0.999 | 9.295 | 0.981 | 1.056 | 10.201 |
| YS2G-L18-s1 | b49e6399 | collision | 9 | same | -0.053 | other | 0.869 | 13.356 | 2.230 | 1.399 | 15.538 |
| YS2G-L18-s1 | b87c15c9 | collision | 12 | same | -7.117 | other | 0.891 | 12.223 | 1.843 | 0.451 | 16.147 |
| YS2G-L18-s1 | 0722adec | collision | 11 | same | -5.225 | other | 0.944 | 18.938 | 2.686 | 1.127 | 21.570 |
| YS2G-L18-s1 | e14f1ffb | collision | 16 | same | -2.909 | other | 0.679 | 18.817 | 1.993 | 0.880 | 11.092 |

## R4 (exploratory) the human log against the predicted ego lane

The logged rear-axle path where it crosses d m ahead of the camera, against the P0 ego lane (lines 1, 2): inside the lane, and more than 0.5 m from its centre; on decisions with both lines present, and on all decisions.

| sample | d | n_both | inside_both | off_both | n_all | inside_all | off_all |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ctrl | 10 | 275 | 1.000 | 0.004 | 1998 | 0.807 | 0.347 |
| ctrl | 20 | 275 | 1.000 | 0.004 | 1952 | 0.719 | 0.438 |
| held pre | 10 | 51 | 1.000 | 0.000 | 1697 | 0.509 | 0.703 |
| held pre | 20 | 51 | 1.000 | 0.118 | 1410 | 0.374 | 0.757 |

## Per-event table (YS-L18-s1 and YR10m10-F-s0)

cls = registered class (P0), cls_geo = the exploratory geometry-only class, cls_geoK = the same on scale-corrected heads, cls_FT = FT heads; j = the decision the class rests on; both / right / wrong / cross = window decisions with both ego lines present / eyes right / eyes wrong / plan crossing a predicted line or edge; p1, p2 = median ego-line probabilities in the window; e20 = median ego-line error at 20 m regardless of probability.

| group | sid | type | k_ev | side | cls | sub | cls_geo | sub_geo | cls_geoK | cls_FT | j | n_win | both | right | wrong | cross | p1 | p2 | e20 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| YR10m10-F-s0 | 0722adec | collision | 11 | R | b |  | c | c1 | c | b | 7 | 5 | 0 | 0 | 5 | 0 | 0.253 | 0.273 | 0.284 |
| YR10m10-F-s0 | 0c3c4cd5 | collision | 17 | L | c | c2 | c | c2 | c | c | 17 | 5 | 3 | 0 | 2 | 1 | 0.523 | 0.615 | 0.428 |
| YR10m10-F-s0 | 16661524 | collision | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 4 | 0.022 | 0.037 | 6.064 |
| YR10m10-F-s0 | 700237df | collision | 12 | R | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 0 | 0.034 | 0.032 | 0.491 |
| YR10m10-F-s0 | 8551466c | collision | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.034 | 0.022 | 3.401 |
| YR10m10-F-s0 | 9d18c2e4 | collision | 16 | R | b |  | c | c2 | b | b | 12 | 5 | 0 | 0 | 5 | 3 | 0.032 | 0.014 | 0.532 |
| YR10m10-F-s0 | bf00c76f | collision | 15 | L | b |  | a |  | c | b | 11 | 5 | 0 | 0 | 5 | 1 | 0.028 | 0.019 | 0.302 |
| YR10m10-F-s0 | cecd27d3 | collision | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 0 | 0.149 | 0.064 | 0.681 |
| YR10m10-F-s0 | daf38cec | collision | 10 | R | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.023 | 0.012 | 9.839 |
| YR10m10-F-s0 | e14f1ffb | collision | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.017 | 0.013 | 1.412 |
| YR10m10-F-s0 | 0393e11b | corridor | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.021 | 0.004 | 4.050 |
| YR10m10-F-s0 | 0fa603b5 | corridor | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.023 | 0.011 | 2.188 |
| YR10m10-F-s0 | 1081728e | corridor | 17 | R | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 3 | 0.015 | 0.008 | 6.869 |
| YR10m10-F-s0 | 13b9787d | corridor | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.011 | 0.019 | 5.024 |
| YR10m10-F-s0 | 1b183271 | corridor | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.016 | 0.010 | 1.119 |
| YR10m10-F-s0 | 20728d3e | corridor | 9 | R | b |  | b |  | b | b | 5 | 5 | 0 | 0 | 5 | 3 | 0.060 | 0.009 | 1.329 |
| YR10m10-F-s0 | 2959d5db | corridor | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.010 | 0.996 |
| YR10m10-F-s0 | 330166de | corridor | 15 | L | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 5 | 0.032 | 0.012 | 1.337 |
| YR10m10-F-s0 | 47da19c4 | corridor | 12 | L | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.012 | 0.012 | 7.744 |
| YR10m10-F-s0 | 4faa4706 | corridor | 13 | R | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.018 | 0.016 | 7.363 |
| YR10m10-F-s0 | 5b7d7e1a | corridor | 9 | L | b |  | b |  | b | b | 5 | 5 | 0 | 0 | 5 | 5 | 0.006 | 0.004 | 10.397 |
| YR10m10-F-s0 | 5ddfd1fe | corridor | 15 | L | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.010 | 0.878 |
| YR10m10-F-s0 | 69dec595 | corridor | 11 | L | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.018 | 0.017 | 2.814 |
| YR10m10-F-s0 | 77155a60 | corridor | 11 | L | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 3 | 0.006 | 0.004 | 1.247 |
| YR10m10-F-s0 | 83e895f7 | corridor | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.012 | - |
| YR10m10-F-s0 | 84b38cda | corridor | 13 | L | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.009 | 0.011 | 11.067 |
| YR10m10-F-s0 | a3c4e4bf | corridor | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.012 | 2.342 |
| YR10m10-F-s0 | a3dd6d53 | corridor | 12 | L | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.032 | 0.044 | 0.805 |
| YR10m10-F-s0 | ae878998 | corridor | 12 | R | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.029 | 0.015 | 2.111 |
| YR10m10-F-s0 | aec02ec2 | corridor | 8 | R | b |  | b |  | c | b | 4 | 5 | 0 | 0 | 5 | 4 | 0.167 | 0.044 | 0.651 |
| YR10m10-F-s0 | b628faa4 | corridor | 12 | R | b |  | c | c3 | c | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.058 | 0.045 | 0.363 |
| YR10m10-F-s0 | b7c35b12 | corridor | 16 | R | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.006 | 0.004 | 2.211 |
| YR10m10-F-s0 | b814c5b7 | corridor | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.018 | 0.008 | 2.456 |
| YR10m10-F-s0 | bddb70b4 | corridor | 12 | L | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 4 | 0.019 | 0.006 | 2.003 |
| YR10m10-F-s0 | caa3170f | corridor | 12 | R | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.071 | 0.023 | 1.334 |
| YR10m10-F-s0 | d7b1349f | corridor | 10 | R | b |  | c | c3 | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.012 | 0.023 | 0.555 |
| YR10m10-F-s0 | e072351f | corridor | 13 | L | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 3 | 0.029 | 0.008 | 2.060 |
| YR10m10-F-s0 | e14711ef | corridor | 8 | R | b |  | b |  | b | b | 4 | 5 | 0 | 0 | 5 | 5 | 0.011 | 0.009 | 11.239 |
| YR10m10-F-s0 | e6056b57 | corridor | 13 | L | b |  | a |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.018 | 0.017 | 2.785 |
| YR10m10-F-s0 | f3512ae1 | corridor | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.017 | 0.007 | 1.174 |
| YR10m10-F-s0 | fe4e5fdf | corridor | 13 | R | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.048 | 0.015 | 8.354 |
| YR10m10-F-s0 | 02015675 | offroad | 7 | R | b |  | b |  | b | b | 3 | 5 | 0 | 0 | 5 | 5 | 0.035 | 0.007 | 3.152 |
| YR10m10-F-s0 | 078bc102 | offroad | 10 | L | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 4 | 0.012 | 0.006 | 11.409 |
| YR10m10-F-s0 | 089e3eac | offroad | 15 | L | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 5 | 0.030 | 0.022 | 7.905 |
| YR10m10-F-s0 | 152679f2 | offroad | 15 | R | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 0 | 0.354 | 0.507 | 1.033 |
| YR10m10-F-s0 | 1f544304 | offroad | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 4 | 0.007 | 0.009 | 1.560 |
| YR10m10-F-s0 | 3502b309 | offroad | 11 | L | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.017 | 0.021 | 12.101 |
| YR10m10-F-s0 | 365c0937 | offroad | 10 | R | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.009 | 0.008 | 8.666 |
| YR10m10-F-s0 | 3b4031de | offroad | 10 | L | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.008 | 0.004 | - |
| YR10m10-F-s0 | 4e3f39fa | offroad | 12 | R | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 4 | 0.028 | 0.012 | 1.579 |
| YR10m10-F-s0 | 53f8bc40 | offroad | 17 | R | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.008 | 0.015 | 5.903 |
| YR10m10-F-s0 | 5fe06174 | offroad | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 5 | 0.008 | 0.021 | 3.821 |
| YR10m10-F-s0 | 64a7186a | offroad | 0 | L | b |  | c | c2 | c | b | 0 | 1 | 0 | 0 | 1 | 0 | 0.012 | 0.466 | 0.406 |
| YR10m10-F-s0 | 661ce644 | offroad | 7 | R | b |  | b |  | b | b | 3 | 5 | 0 | 0 | 5 | 5 | 0.018 | 0.006 | 1.936 |
| YR10m10-F-s0 | 67077701 | offroad | 10 | R | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.009 | 0.004 | 8.764 |
| YR10m10-F-s0 | 78c26c7e | offroad | 14 | R | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 5 | 0.012 | 0.010 | 13.640 |
| YR10m10-F-s0 | 8c9e3fd9 | offroad | 14 | L | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 3 | 0.005 | 0.004 | 12.881 |
| YR10m10-F-s0 | 9a377868 | offroad | 0 | L | b |  | b |  | b | b | 0 | 1 | 0 | 0 | 1 | 1 | 0.008 | 0.013 | 1.789 |
| YR10m10-F-s0 | 9b0a5826 | offroad | 8 | L | b |  | b |  | c | b | 4 | 5 | 0 | 0 | 5 | 0 | 0.061 | 0.200 | 1.137 |
| YR10m10-F-s0 | 9d1d720d | offroad | 3 | R | b |  | b |  | c | b | 0 | 4 | 0 | 0 | 4 | 0 | 0.352 | 0.099 | 1.294 |
| YR10m10-F-s0 | a09e4156 | offroad | 6 | R | b |  | a |  | c | b | 2 | 5 | 0 | 0 | 5 | 4 | 0.088 | 0.023 | 0.528 |
| YR10m10-F-s0 | b9c124bd | offroad | 14 | R | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 5 | 0.015 | 0.009 | 0.972 |
| YR10m10-F-s0 | bd70654a | offroad | 12 | R | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.013 | 7.920 |
| YR10m10-F-s0 | c687799d | offroad | 13 | L | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.015 | 0.017 | 4.917 |
| YR10m10-F-s0 | ccfe1da3 | offroad | 10 | L | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 0 | 0.020 | 0.032 | 1.125 |
| YR10m10-F-s0 | da6d3b68 | offroad | 0 | L | b |  | b |  | c | b | 0 | 1 | 0 | 0 | 1 | 1 | 0.110 | 0.688 | 0.641 |
| YR10m10-F-s0 | dad96574 | offroad | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.016 | 0.015 | 1.737 |
| YR10m10-F-s0 | ddc1271e | offroad | 4 | L | b |  | b |  | b | b | 0 | 5 | 0 | 0 | 5 | 4 | 0.005 | 0.002 | 5.486 |
| YR10m10-F-s0 | e511376f | offroad | 14 | R | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 1 | 0.043 | 0.023 | 1.369 |
| YR10m10-F-s0 | f082cf21 | offroad | 11 | L | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.011 | 0.009 | 12.403 |
| YR10m10-F-s0 | f1b802f6 | offroad | 0 | R | b |  | b |  | b | b | 0 | 1 | 0 | 0 | 1 | 1 | 0.023 | 0.008 | 1.196 |
| YR10m10-F-s0 | f4495a91 | offroad | 4 | L | b |  | c | c1 | c | b | 0 | 5 | 0 | 0 | 5 | 0 | 0.388 | 0.342 | 0.361 |
| YR10m10-F-s0 | f9ed38d9 | offroad | 9 | R | b |  | b |  | b | b | 5 | 5 | 0 | 0 | 5 | 1 | 0.014 | 0.007 | 4.063 |
| YR10m10-F-s0 | fde7abb6 | offroad | 15 | R | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 5 | 0.012 | 0.009 | 8.058 |
| YS-L18-s1 | 0722adec | collision | 10 | R | b |  | c | c1 | c | b | 6 | 5 | 0 | 0 | 5 | 0 | 0.189 | 0.413 | 0.315 |
| YS-L18-s1 | 47e0ff3b | collision | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 3 | 0.008 | 0.005 | 1.477 |
| YS-L18-s1 | 7bbb6cfc | collision | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 4 | 0.057 | 0.028 | 1.150 |
| YS-L18-s1 | 8051c60f | collision | 9 | L | b |  | b |  | b | b | 5 | 5 | 0 | 0 | 5 | 0 | 0.070 | 0.044 | 1.097 |
| YS-L18-s1 | 84b38cda | collision | 14 | R | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 4 | 0.033 | 0.073 | 1.170 |
| YS-L18-s1 | b24575ef | collision | 17 | R | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 4 | 0.035 | 0.017 | 0.873 |
| YS-L18-s1 | bd70654a | collision | 14 | L | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 2 | 0.020 | 0.006 | 2.540 |
| YS-L18-s1 | bf00c76f | collision | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 0 | 0.016 | 0.011 | 0.670 |
| YS-L18-s1 | daf38cec | collision | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.023 | 0.012 | 7.153 |
| YS-L18-s1 | 02b68b9c | corridor | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 4 | 0.024 | 0.017 | 2.341 |
| YS-L18-s1 | 0fa603b5 | corridor | 17 | R | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.006 | 0.008 | 2.060 |
| YS-L18-s1 | 330166de | corridor | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 0 | 0.024 | 0.018 | 0.590 |
| YS-L18-s1 | 47da19c4 | corridor | 11 | L | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.019 | 0.013 | 1.261 |
| YS-L18-s1 | 4faa4706 | corridor | 14 | R | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.019 | 6.980 |
| YS-L18-s1 | 5ddfd1fe | corridor | 15 | L | b |  | b |  | a | b | 11 | 5 | 0 | 0 | 5 | 5 | 0.012 | 0.008 | 0.737 |
| YS-L18-s1 | 8eefb96f | corridor | 10 | L | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 1 | 0.042 | 0.062 | 1.036 |
| YS-L18-s1 | 94652fa0 | corridor | 15 | L | b |  | b |  | b | b | 11 | 5 | 0 | 0 | 5 | 0 | 0.176 | 0.044 | 1.130 |
| YS-L18-s1 | a3882e6a | corridor | 10 | L | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.014 | 0.023 | 1.841 |
| YS-L18-s1 | a3dd6d53 | corridor | 13 | L | b |  | b |  | b | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.032 | 0.027 | 1.644 |
| YS-L18-s1 | aec02ec2 | corridor | 8 | R | b |  | b |  | c | b | 4 | 5 | 0 | 0 | 5 | 0 | 0.174 | 0.047 | 0.720 |
| YS-L18-s1 | b628faa4 | corridor | 13 | R | b |  | c | c3 | c | b | 9 | 5 | 0 | 0 | 5 | 5 | 0.081 | 0.047 | 0.163 |
| YS-L18-s1 | b7c35b12 | corridor | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 4 | 0.008 | 0.004 | 3.050 |
| YS-L18-s1 | d7b1349f | corridor | 11 | R | b |  | b |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.016 | 0.012 | 1.168 |
| YS-L18-s1 | ded4ce0e | corridor | 14 | L | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 4 | 0.015 | 0.014 | 0.547 |
| YS-L18-s1 | e14711ef | corridor | 8 | R | b |  | b |  | b | b | 4 | 5 | 0 | 0 | 5 | 5 | 0.017 | 0.012 | 7.588 |
| YS-L18-s1 | e6056b57 | corridor | 9 | L | b |  | a |  | b | b | 5 | 5 | 0 | 0 | 5 | 5 | 0.020 | 0.016 | 0.682 |
| YS-L18-s1 | f3512ae1 | corridor | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 5 | 0.015 | 0.010 | 1.976 |
| YS-L18-s1 | f42258b6 | corridor | 16 | L | b |  | b |  | b | b | 12 | 5 | 0 | 0 | 5 | 0 | 0.008 | 0.009 | 1.223 |
| YS-L18-s1 | 02015675 | offroad | 7 | R | b |  | b |  | b | b | 3 | 5 | 0 | 0 | 5 | 5 | 0.029 | 0.004 | 4.883 |
| YS-L18-s1 | 078bc102 | offroad | 9 | L | b |  | b |  | b | b | 5 | 5 | 0 | 0 | 5 | 4 | 0.013 | 0.003 | 10.886 |
| YS-L18-s1 | 152679f2 | offroad | 16 | L | b |  | b |  | c | b | 12 | 5 | 1 | 0 | 5 | 0 | 0.410 | 0.753 | 1.122 |
| YS-L18-s1 | 53f8bc40 | offroad | 12 | L | b |  | b |  | b | b | 8 | 5 | 0 | 0 | 5 | 5 | 0.016 | 0.016 | 2.629 |
| YS-L18-s1 | 64a7186a | offroad | 0 | L | b |  | c | c2 | c | b | 0 | 1 | 0 | 0 | 1 | 0 | 0.012 | 0.466 | 0.406 |
| YS-L18-s1 | 667ea4f7 | offroad | 17 | L | b |  | b |  | b | b | 13 | 5 | 0 | 0 | 5 | 0 | 0.018 | 0.009 | 5.481 |
| YS-L18-s1 | 8c9e3fd9 | offroad | 10 | R | b |  | b |  | b | b | 6 | 5 | 0 | 0 | 5 | 5 | 0.025 | 0.005 | 2.326 |
| YS-L18-s1 | 9a377868 | offroad | 0 | L | b |  | b |  | b | b | 0 | 1 | 0 | 0 | 1 | 1 | 0.008 | 0.013 | 1.789 |
| YS-L18-s1 | be6f90f0 | offroad | 5 | L | b |  | b |  | b | b | 1 | 5 | 0 | 0 | 5 | 3 | 0.013 | 0.012 | 4.692 |
| YS-L18-s1 | c687799d | offroad | 14 | L | b |  | b |  | b | b | 10 | 5 | 0 | 0 | 5 | 5 | 0.009 | 0.008 | 5.559 |
| YS-L18-s1 | da6d3b68 | offroad | 0 | L | b |  | b |  | c | b | 0 | 1 | 0 | 0 | 1 | 0 | 0.110 | 0.688 | 0.641 |
| YS-L18-s1 | ddc1271e | offroad | 7 | R | b |  | b |  | b | b | 3 | 5 | 0 | 0 | 5 | 0 | 0.009 | 0.006 | - |
| YS-L18-s1 | f1b802f6 | offroad | 0 | R | b |  | b |  | b | b | 0 | 1 | 0 | 0 | 1 | 1 | 0.023 | 0.008 | 1.196 |
| YS-L18-s1 | f9ed38d9 | offroad | 7 | L | b |  | b |  | b | b | 3 | 5 | 0 | 0 | 5 | 4 | 0.019 | 0.014 | 2.231 |
| YS-L18-s1 | fde7abb6 | offroad | 11 | L | b |  | a |  | b | b | 7 | 5 | 0 | 0 | 5 | 5 | 0.013 | 0.004 | 4.674 |
