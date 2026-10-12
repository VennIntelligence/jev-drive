# ROAD-PROBE: is city road geometry in the frozen Cinque vision tokens?

Pre-registration: [../plans/2026-10-12-road-probe-prereg.md](../plans/2026-10-12-road-probe-prereg.md) (pushed before any read). Probe = decision 243's HEAD1 head on the frozen tokens (8 slots x 32 x 512) + the 20-dim ego (command included), trained on 69 419 on-policy rows of 1020 train scenes (28 log units); read on a fresh held run of `YS-L18-s1` (every decision of the held scenes) and on decision 269's 98 road-exit zero windows (served tokens by replay). Errors are |prediction - map label| in metres at x = camera + d in the rear-axle frame (ev.py's frame); the ego-lane label is the lane the logged route occupies at that distance. Medians with a scene-cluster bootstrap CI (B 2000).

Held: 8129 decisions on 452 scenes, 4498 straight (route turn < 5 deg over 40 m), 2109 turns (>= 20 deg); zero group: 1764 decisions, 98 registered windows.

## Gates

| gate | reading |  |
|:--|--:|--:|
| G_roll | rollouts 460, mean 0.826, stored_rollouts 460, stored_mean 0.826 | pass |
| G_tok | n 8129, share 1.000, median 0.016, max 0.094 | pass |
| G_rep | n 1764, missing 0, share 1.000, median 0.012 | pass |
| G_frame_train | scenes 1020, share 1.000, max 0.000 | pass |
| G_frame_held | scenes 452, share 1.000, max 0.000 | pass |
| G_frame_zero | scenes 72, share 1.000, max 0.000 | pass |
| G_lab | straight 4498, share 0.908 | pass |
| G_sign | L20 1.509, R20 -1.883 | pass |

## R3 (registered verdict): the 98 road-exit zero windows, 20 m ego-lane boundary error

**Verdict: frozen eyes do not see city roads.** Probe (2-seed mean) window median 1.255 [1.033, 2.023] m (line < 0.3 repairable, > 1.0 do not see), share of windows with both boundaries within 0.5 m 0.111 [0.022, 0.200] (line >= 0.6); 45 windows evaluable on at least one side, 45 on both.

| predictor | window median (m) | share both <= 0.5 m | windows (one side / both) |
|:--|--:|--:|--:|
| probe (2 seeds, mean) | 1.255 [1.033, 2.023] | 0.111 [0.022, 0.200] | 45 / 45 |
| probe seed 0 | 1.280 [0.768, 1.973] | 0.133 [0.044, 0.222] | 45 / 45 |
| probe seed 1 | 1.411 [0.919, 2.394] | 0.089 [0.022, 0.178] | 45 / 45 |
| linear floor | 2.463 [1.408, 3.781] | 0.044 [0.000, 0.111] | 45 / 45 |
| blind floor (ego + command) | 2.969 [2.455, 3.760] | 0.044 [0.000, 0.111] | 45 / 45 |
| shipped heads | 5.609 [3.222, 8.280] | 0.022 [0.000, 0.067] | 45 / 45 |
| shipped heads, 0.689 undone | 5.547 [3.459, 8.282] | 0.067 [0.000, 0.156] | 45 / 45 |

## R1: held straight decisions, 20 m

| predictor | left median (m) | right median (m) | both, median (m) | both, p90 (m) | n (boundary points) |
|:--|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.236 [0.214, 0.261] | 0.279 [0.249, 0.312] | 0.253 [0.234, 0.279] | 1.181 [1.046, 1.350] | 8166 |
| probe seed 0 | 0.248 [0.228, 0.276] | 0.289 [0.264, 0.324] | 0.269 [0.245, 0.295] | 1.228 [1.074, 1.416] | 8166 |
| probe seed 1 | 0.275 [0.254, 0.303] | 0.320 [0.296, 0.353] | 0.298 [0.276, 0.322] | 1.298 [1.157, 1.459] | 8166 |
| linear floor | 0.575 [0.548, 0.612] | 0.626 [0.591, 0.670] | 0.600 [0.572, 0.636] | 1.653 [1.559, 1.793] | 8166 |
| blind floor (ego + command) | 0.434 [0.396, 0.476] | 0.493 [0.445, 0.554] | 0.460 [0.423, 0.511] | 2.072 [1.795, 2.391] | 8166 |
| shipped heads | 0.542 [0.504, 0.589] | 0.650 [0.612, 0.684] | 0.603 [0.573, 0.634] | 2.537 [2.255, 2.815] | 8166 |
| shipped heads, 0.689 undone | 0.447 [0.403, 0.490] | 0.337 [0.304, 0.388] | 0.393 [0.357, 0.435] | 2.468 [2.050, 2.858] | 8166 |

## R2: held turn decisions (route turn >= 20 deg), 20 m

| predictor | left median (m) | right median (m) | both, median (m) | both, p90 (m) | n (boundary points) |
|:--|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.847 [0.727, 0.987] | 0.899 [0.767, 1.044] | 0.877 [0.756, 1.001] | 3.567 [2.756, 4.884] | 1718 |
| probe seed 0 | 0.891 [0.765, 1.036] | 0.908 [0.749, 1.071] | 0.900 [0.765, 1.036] | 3.746 [2.865, 4.973] | 1718 |
| probe seed 1 | 0.970 [0.837, 1.160] | 1.028 [0.884, 1.198] | 0.992 [0.863, 1.170] | 3.701 [2.969, 4.861] | 1718 |
| linear floor | 1.432 [1.213, 1.638] | 1.317 [1.130, 1.550] | 1.361 [1.183, 1.594] | 4.360 [3.768, 5.010] | 1718 |
| blind floor (ego + command) | 2.085 [1.782, 2.535] | 1.971 [1.754, 2.277] | 2.006 [1.766, 2.401] | 5.551 [4.967, 6.266] | 1718 |
| shipped heads | 3.867 [3.133, 4.945] | 4.005 [3.293, 4.650] | 3.940 [3.216, 4.749] | 9.014 [8.420, 9.705] | 1718 |
| shipped heads, 0.689 undone | 3.312 [2.650, 4.404] | 3.796 [2.888, 4.540] | 3.650 [2.767, 4.468] | 8.546 [7.949, 9.288] | 1718 |

Secondary line (decision 269's 'eyes usable' (ii)-(iii), straight, both boundaries pooled): probe median 0.253 (< 0.30) and p90 1.181 (< 0.75): **not met**. Shipped heads: both ego lines present (prob > 0.5) on 0.188 of straight and 0.003 of turn decisions of this held run.

### Decision 269's measure (distance to the nearest map lane boundary, 20 m, left and right predictions pooled)

| subset | predictor | median (m) | p90 (m) | n |
|:--|--:|--:|--:|--:|
| straight | probe (2 seeds, mean) | 0.246 [0.227, 0.268] | 1.029 | 8968 |
| straight | probe seed 0 | 0.257 [0.236, 0.281] | 1.017 | 8968 |
| straight | probe seed 1 | 0.284 [0.265, 0.306] | 1.120 | 8968 |
| straight | blind floor (ego + command) | 0.412 [0.384, 0.448] | 1.388 | 8968 |
| straight | linear floor | 0.547 [0.523, 0.569] | 1.393 | 8968 |
| straight | shipped heads | 0.541 [0.520, 0.566] | 1.404 | 8968 |
| straight | shipped heads, 0.689 undone | 0.335 [0.312, 0.361] | 1.265 | 8968 |
| turn | probe (2 seeds, mean) | 0.891 [0.806, 0.993] | 4.164 | 3944 |
| turn | probe seed 0 | 0.879 [0.783, 0.981] | 4.123 | 3944 |
| turn | probe seed 1 | 0.961 [0.864, 1.079] | 4.313 | 3944 |
| turn | blind floor (ego + command) | 1.105 [1.000, 1.223] | 4.746 | 3944 |
| turn | linear floor | 0.962 [0.876, 1.055] | 3.482 | 3944 |
| turn | shipped heads | 1.131 [1.017, 1.264] | 5.920 | 3944 |
| turn | shipped heads, 0.689 undone | 1.067 [0.950, 1.180] | 5.780 | 3944 |

### Descriptive: other distances, road edges, lane heading (medians; heading in degrees)

straight:

| predictor | LR10 | LR20 | LR40 | E10 | E20 | E40 | h20_deg |
|:--|--:|--:|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.220 | 0.253 | 0.387 | 0.636 | 0.659 | 0.831 | 0.827 |
| probe seed 0 | 0.233 | 0.269 | 0.420 | 0.665 | 0.683 | 0.887 | 0.933 |
| probe seed 1 | 0.251 | 0.298 | 0.441 | 0.727 | 0.757 | 0.953 | 1.129 |
| linear floor | 0.527 | 0.600 | 0.834 | 1.039 | 1.353 | 1.597 | 3.950 |
| blind floor (ego + command) | 0.368 | 0.460 | 0.733 | 1.835 | 1.893 | 2.086 | 1.648 |
| shipped heads | 0.581 | 0.603 | 0.651 | 1.446 | 1.507 | 1.603 | 0.619 |
| shipped heads, 0.689 undone | 0.326 | 0.393 | 0.669 | 1.126 | 1.166 | 1.463 | 0.843 |

gentle:

| predictor | LR10 | LR20 | LR40 | E10 | E20 | E40 | h20_deg |
|:--|--:|--:|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.369 | 0.487 | 0.775 | 0.577 | 0.661 | 0.871 | 1.386 |
| probe seed 0 | 0.365 | 0.494 | 0.841 | 0.678 | 0.730 | 0.961 | 1.598 |
| probe seed 1 | 0.433 | 0.546 | 0.809 | 0.630 | 0.763 | 0.998 | 1.820 |
| linear floor | 0.670 | 0.821 | 1.142 | 0.975 | 1.413 | 1.728 | 4.834 |
| blind floor (ego + command) | 0.638 | 0.863 | 1.578 | 1.848 | 1.995 | 2.478 | 3.619 |
| shipped heads | 0.795 | 1.007 | 1.512 | 1.114 | 1.207 | 1.555 | 2.140 |
| shipped heads, 0.689 undone | 0.519 | 0.806 | 1.664 | 1.078 | 1.149 | 2.187 | 2.397 |

turn:

| predictor | LR10 | LR20 | LR40 | E10 | E20 | E40 | h20_deg |
|:--|--:|--:|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.564 | 0.877 | 1.423 | 1.085 | 1.454 | 2.492 | 4.823 |
| probe seed 0 | 0.624 | 0.900 | 1.427 | 1.192 | 1.579 | 2.599 | 5.081 |
| probe seed 1 | 0.612 | 0.992 | 1.459 | 1.222 | 1.515 | 2.687 | 6.267 |
| linear floor | 0.957 | 1.361 | 1.649 | 1.626 | 2.231 | 3.505 | 10.945 |
| blind floor (ego + command) | 0.843 | 2.006 | 3.872 | 2.714 | 3.583 | 5.945 | 15.181 |
| shipped heads | 1.329 | 3.940 | 5.545 | 2.158 | 3.869 | 7.447 | 33.395 |
| shipped heads, 0.689 undone | 1.285 | 3.650 | 4.260 | 2.194 | 3.384 | 7.325 | 31.458 |

## Learning curve (probe seed 0, nested shares of the fit log units)

| share | log units | scenes | rows | held straight 20 m median | held turn 20 m median | R3 window median | R3 share |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0.125 | 4 | 200 | 13512 | 0.410 | 1.167 | 1.942 | 0.067 |
| 0.250 | 7 | 260 | 17563 | 0.356 | 1.245 | 1.709 | 0.156 |
| 0.500 | 14 | 594 | 40479 | 0.318 | 1.103 | 1.306 | 0.022 |
| 1.000 | 27 | 1010 | 68728 | 0.269 | 0.900 | 1.280 | 0.133 |

- turn: slope -0.104 m per doubling of log units [-0.163, -0.044] -> **more data still helps** (registered rule: CI below 0).
- r3: slope -0.257 m per doubling of log units [-0.535, 0.010] -> **no reliable gain from more data** (registered rule: CI below 0).

## Turn decisions by distance to the junction (held; ego-lane boundary error at 20 m, both sides pooled, median m)

turn:

| predictor | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m |
|:--|--:|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 1.114 (278) | 1.149 (218) | 1.683 (162) | 0.467 (144) | 0.419 (114) | 0.851 (802) |
| linear floor | 1.837 (278) | 1.782 (218) | 2.947 (162) | 0.852 (144) | 0.621 (114) | 1.220 (802) |
| blind floor (ego + command) | 2.096 (278) | 1.962 (218) | 3.465 (162) | 1.418 (144) | 1.106 (114) | 2.014 (802) |
| shipped heads | 4.729 (278) | 4.493 (218) | 5.196 (162) | 1.389 (144) | 3.013 (114) | 3.191 (802) |
| shipped heads, 0.689 undone | 4.164 (278) | 4.319 (218) | 5.321 (162) | 1.249 (144) | 2.154 (114) | 2.830 (802) |

straight:

| predictor | in junction | 0-10 m | 10-20 m | 20-40 m | 40-60 m | none within 60 m |
|:--|--:|--:|--:|--:|--:|--:|
| probe (2 seeds, mean) | 0.422 (410) | 0.441 (432) | 0.390 (484) | 0.273 (1136) | 0.256 (1022) | 0.225 (4682) |
| linear floor | 0.669 (410) | 0.819 (432) | 0.651 (484) | 0.602 (1136) | 0.609 (1022) | 0.569 (4682) |
| blind floor (ego + command) | 0.700 (410) | 0.707 (432) | 0.582 (484) | 0.494 (1136) | 0.456 (1022) | 0.409 (4682) |
| shipped heads | 0.649 (410) | 0.745 (432) | 0.594 (484) | 0.587 (1136) | 0.595 (1022) | 0.596 (4682) |
| shipped heads, 0.689 undone | 0.621 (410) | 0.675 (432) | 0.627 (484) | 0.375 (1136) | 0.357 (1022) | 0.356 (4682) |

## Exploratory: is the ego in an oncoming lane (decision 268 H3), AUC

| predictor | held AUC (pos / neg) | zero rollouts AUC (pos / neg) |
|:--|--:|--:|
| probe (2 seeds, mean) | 0.921 (206 / 7882) | 0.824 (165 / 1518) |
| probe seed 0 | 0.911 (206 / 7882) | 0.818 (165 / 1518) |
| linear floor | 0.861 (206 / 7882) | 0.776 (165 / 1518) |
| blind floor (ego + command) | 0.828 (206 / 7882) | 0.761 (165 / 1518) |

## Exploratory (decided after the read, not registered)

- **Why 53 of 98 windows have no ego-lane label.** In the 457 window decisions the 20 m line crosses few lane edges (median 3; on the decisions without a label the bracketing gap around the route is usually wider than 6 m or one side is missing), the route crossing is missing on 0.105; the car is 1.34 m off the logged route (median), > 2 m on 0.344. The registered R3 therefore reads the 45 windows where the route lane is well defined at 20 m.

- **R3 under decision 272's measure** (distance of each predicted ego line at 20 m to the nearest map lane boundary; defined wherever the line crosses any boundary, 95 windows):

| predictor | decision median (m) | window median (m) | windows | share of windows <= 0.5 m |
|:--|--:|--:|--:|--:|
| probe (2 seeds, mean) | 1.420 | 1.531 | 95 | 0.071 |
| linear floor | 1.424 | 1.508 | 95 | 0.051 |
| blind floor (ego + command) | 1.631 | 1.785 | 95 | 0.102 |
| shipped heads | 1.795 | 2.003 | 95 | 0.051 |
| shipped heads, 0.689 undone | 1.782 | 1.919 | 95 | 0.102 |

- **Error against the car's offset from the route** (probe, ego-lane measure, 20 m, mean of the two sides): the error is ~0.9 m already while the car is still on the route, so the R3 failure is not just the car having left the lane the label describes.

| decisions | abs offset from route (m) | n | median error (m) |
|:--|--:|--:|--:|
| all window-rollout decisions | 0-0.5 | 386 | 0.934 |
| all window-rollout decisions | 0.5-1 | 71 | 0.721 |
| all window-rollout decisions | 1-2 | 65 | 0.954 |
| all window-rollout decisions | 2-inf | 132 | 1.892 |
| held turns | 0-0.5 | 693 | 0.906 |
| held turns | 0.5-1 | 77 | 0.767 |
| held turns | 1-2 | 59 | 0.673 |
| held turns | 2-inf | 30 | 1.216 |

