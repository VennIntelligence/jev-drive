# NATIVE-GAP tables (native_gap_report.py)

Gate: common held windows 2228; means RSG-L18-s0 0.7857, NTV0-L18-s0 0.7699, JTN-L18-s0 0.8744, JTN-L18-s1 0.8909; decisions per window 10; ego on the human track at k = 0: lateral median 0.000 m, p99 0.000 m; |v - vh_now| at k = 0 median 0.049 m/s.

## A. The speed ratchet on moving windows

Ego speed / human speed at the same station (decisions k >= 2, valid, human >= 2 m/s there), median by decision index:

| k | RSG | NTV0 | JTN0 | JTN1 |
|---|--:|--:|--:|--:|
| 2 | 0.989 | 0.987 | 0.987 | 0.987 |
| 3 | 0.968 | 0.966 | 0.966 | 0.968 |
| 4 | 0.944 | 0.949 | 0.950 | 0.954 |
| 5 | 0.916 | 0.933 | 0.936 | 0.943 |
| 6 | 0.893 | 0.914 | 0.923 | 0.934 |
| 7 | 0.875 | 0.902 | 0.914 | 0.922 |
| 8 | 0.858 | 0.892 | 0.901 | 0.918 |
| 9 | 0.831 | 0.880 | 0.891 | 0.912 |

By decision stratum (median ratio):

| decision stratum | RSG | NTV0 | JTN0 | JTN1 | decisions (RSG) |
|---|--:|--:|--:|--:|--:|
| after-stop | 1.030 | 0.991 | 1.005 | 1.000 | 486.000 |
| free | 0.930 | 0.945 | 0.948 | 0.956 | 5969.000 |
| lead | 0.958 | 0.951 | 0.958 | 0.956 | 555.000 |
| turn | 0.967 | 0.976 | 0.979 | 0.984 | 1463.000 |

Decisions where the ego stands (< 0.3 m/s) at a station the human passed at >= 2 m/s (the launch lag; excluded from the next table): RSG 32, NTV0 18, JTN0 27, JTN1 18.

After a stop, the ego moving again (>= 0.3 m/s), by time since its last stop (median ratio):

| time since the ego's last stop | RSG | NTV0 | JTN0 | JTN1 |
|---|--:|--:|--:|--:|
| 0-1 s | 0.405 | 0.126 | 0.374 | - |
| 1-2 s | 0.926 | 0.896 | 0.894 | 0.893 |
| 2-3 s | 1.068 | 1.015 | 1.043 | 1.026 |
| > 3 s | 1.122 | 1.067 | 1.094 | 1.114 |

Step decomposition (k -> k+1, k = 2..8, both valid; human >= 2 m/s at both stations, ego >= 1 m/s). Change of log(ego / human speed at the same station) = **plan term** log(vp(0.5 s) / v) - log(vh[k+1] / vh[k]) + **execution term** log(v[k+1] / vp(0.5 s)). Sums over all steps (= the summed drop of the log ratio), share of the total, and the same in m/s:

| model | stratum | steps | per_step_total | per_step_plan | per_step_exec | plan_share | plan_ms | exec_ms | lag05_m | lag10_m | lag20_m |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG | all | 7193 | -0.0447 | -0.0457 | +0.0010 | +1.0228 | -0.1749 | +0.0059 | -0.0164 | +0.0046 | +0.1681 |
| RSG | free | 4921 | -0.0586 | -0.0562 | -0.0023 | +0.9605 | -0.1991 | -0.0079 | -0.0151 | +0.0140 | +0.2085 |
| RSG | lead | 548 | -0.0416 | -0.0470 | +0.0054 | +1.1303 | -0.2688 | +0.0217 | -0.0326 | -0.0064 | +0.2581 |
| RSG | turn | 1370 | -0.0216 | -0.0257 | +0.0041 | +1.1910 | -0.1115 | +0.0227 | -0.0133 | -0.0105 | +0.0607 |
| RSG | after-stop | 354 | +0.0536 | +0.0250 | +0.0286 | +0.4670 | +0.0623 | +0.1073 | -0.0355 | -0.0784 | -0.2454 |
| NTV0 | all | 7183 | -0.0350 | -0.0346 | -0.0005 | +0.9867 | -0.1290 | -0.0010 | -0.0102 | +0.0101 | +0.1150 |
| NTV0 | free | 4933 | -0.0470 | -0.0436 | -0.0034 | +0.9286 | -0.1565 | -0.0117 | -0.0098 | +0.0174 | +0.1780 |
| NTV0 | lead | 546 | -0.0379 | -0.0422 | +0.0043 | +1.1133 | -0.2448 | +0.0154 | -0.0262 | -0.0032 | +0.2227 |
| NTV0 | turn | 1348 | -0.0107 | -0.0112 | +0.0004 | +1.0382 | -0.0202 | +0.0050 | +0.0024 | +0.0025 | -0.0915 |
| NTV0 | after-stop | 356 | +0.0429 | +0.0140 | +0.0289 | +0.3262 | +0.0173 | +0.0986 | -0.0392 | -0.0853 | -0.2415 |
| JTN0 | all | 7070 | -0.0364 | -0.0336 | -0.0027 | +0.9245 | -0.1100 | -0.0093 | -0.0036 | +0.0183 | +0.0967 |
| JTN0 | free | 4916 | -0.0495 | -0.0439 | -0.0056 | +0.8865 | -0.1402 | -0.0204 | -0.0026 | +0.0282 | +0.1804 |
| JTN0 | lead | 547 | -0.0310 | -0.0349 | +0.0039 | +1.1251 | -0.2013 | +0.0147 | -0.0229 | -0.0055 | +0.1659 |
| JTN0 | turn | 1339 | -0.0077 | -0.0071 | -0.0006 | +0.9213 | +0.0048 | +0.0022 | +0.0061 | +0.0044 | -0.1088 |
| JTN0 | after-stop | 268 | +0.0501 | +0.0244 | +0.0258 | +0.4862 | +0.0571 | +0.0894 | -0.0243 | -0.0722 | -0.2664 |
| JTN1 | all | 7056 | -0.0292 | -0.0264 | -0.0029 | +0.9015 | -0.0829 | -0.0104 | -0.0002 | +0.0186 | +0.0652 |
| JTN1 | free | 4914 | -0.0413 | -0.0354 | -0.0058 | +0.8584 | -0.1076 | -0.0219 | +0.0016 | +0.0278 | +0.1416 |
| JTN1 | lead | 546 | -0.0299 | -0.0340 | +0.0042 | +1.1392 | -0.1950 | +0.0160 | -0.0200 | -0.0058 | +0.1380 |
| JTN1 | turn | 1340 | +0.0006 | +0.0011 | -0.0006 | +2.0188 | +0.0302 | +0.0015 | +0.0092 | +0.0063 | -0.1276 |
| JTN1 | after-stop | 256 | +0.0475 | +0.0205 | +0.0270 | +0.4314 | +0.0379 | +0.0920 | -0.0294 | -0.0687 | -0.3015 |

Losing steps only (shares of their summed loss):

| losing steps (total < 0) | steps | share_of_steps | plan | exec |
|---|--:|--:|--:|--:|
| RSG | 5262.000 | 0.732 | 0.994 | 0.006 |
| NTV0 | 4858.000 | 0.676 | 0.971 | 0.029 |
| JTN0 | 4588.000 | 0.649 | 0.933 | 0.067 |
| JTN1 | 4464.000 | 0.633 | 0.925 | 0.075 |

Window ratio at k = 2 / k = 9 (median, windows valid to the end): RSG 0.989 / 0.791, NTV0 0.987 / 0.853, JTN0 0.987 / 0.870, JTN1 0.987 / 0.894

## A2. Open-loop counterfactual: the same frames, the human's speed history

RSG: gate (replayed F plan vs the dumped plan, max |dxy| over the 8 poses) max 0.0905 m, p99 0.0624 m; 2700 decisions, 2046 with k >= 2 and the human >= 2 m/s
JTN1: gate (replayed F plan vs the dumped plan, max |dxy| over the 8 poses) max 0.1250 m, p99 0.0632 m; 2700 decisions, 2033 with k >= 2 and the human >= 2 m/s

Plan speed over 0-2 s (arc2 / 2 s) / the human's speed at the ego's station; F = as served, H = speed history = the human's; recovery = (H - F) / (1 - F) of the medians; gain = median (vp_H - vp_F) / (vh - v) per decision (|vh - v| > 0.5 m/s; 1 = the plan moves one for one with the fed speed, 0 = it ignores it); plan 0.5 s = the central speed at 0.5 s:

| model | decisions | n | ego_over_human | plan_F | plan_H | recovery | gain | plan_F_05 | plan_H_05 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG | all | 2046 | 0.930 | 0.863 | 0.923 | 0.436 | 0.477 | 0.889 | 0.927 |
| RSG | ego < 0.9 x human | 877 | 0.751 | 0.614 | 0.748 | 0.347 | 0.457 | 0.668 | 0.762 |
| RSG | ego < 0.7 x human | 354 | 0.545 | 0.402 | 0.581 | 0.299 | 0.367 | 0.441 | 0.577 |
| JTN1 | all | 2033 | 0.951 | 0.912 | 0.938 | 0.293 | 0.354 | 0.925 | 0.943 |
| JTN1 | ego < 0.9 x human | 729 | 0.768 | 0.616 | 0.725 | 0.285 | 0.353 | 0.684 | 0.752 |
| JTN1 | ego < 0.7 x human | 266 | 0.527 | 0.355 | 0.524 | 0.262 | 0.303 | 0.426 | 0.545 |

## B. Static windows natively

Per cue type (cue: the arena set's lead at the window start, then the census light labels; arc4 = the plan's 4 s arc, m, median; roll k = first decision k >= 2 with the ego above 0.2 m/s, median over the windows that roll; rolled = share that roll; disp end = the ego's distance from its start at k = 9):

| model | cue | windows | zeros | corridor | mean | arc4_k0 | arc4_k2 | arc4_max_k3 | roll_k | rolled | disp_end |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG | all | 878 | 385 | 359 | 0.562 | 3.968 | 4.642 | 7.556 | 3.000 | 0.908 | 1.938 |
| RSG | lead 8-30 m | 49 | 26 | 26 | 0.469 | 4.094 | 4.902 | 14.366 | 3.000 | 1.000 | 4.743 |
| RSG | lead <= 8 m | 477 | 54 | 32 | 0.887 | 3.359 | 3.753 | 3.644 | 4.000 | 0.830 | 0.813 |
| RSG | light of unknown state, no lead | 75 | 50 | 48 | 0.333 | 5.009 | 6.943 | 18.356 | 3.000 | 1.000 | 6.408 |
| RSG | no light, no lead | 38 | 27 | 27 | 0.289 | 4.561 | 5.469 | 11.477 | 3.000 | 1.000 | 3.606 |
| RSG | red light at the line, no lead | 239 | 228 | 226 | 0.046 | 5.619 | 7.493 | 26.575 | 3.000 | 1.000 | 7.312 |
| NTV0 | all | 878 | 403 | 364 | 0.541 | 4.069 | 4.832 | 7.348 | 3.000 | 0.960 | 2.418 |
| NTV0 | lead 8-30 m | 49 | 28 | 28 | 0.429 | 4.069 | 5.143 | 15.890 | 3.000 | 1.000 | 4.136 |
| NTV0 | lead <= 8 m | 477 | 67 | 32 | 0.860 | 3.356 | 3.803 | 3.727 | 4.000 | 0.966 | 1.087 |
| NTV0 | light of unknown state, no lead | 75 | 54 | 52 | 0.280 | 5.329 | 6.993 | 20.484 | 3.000 | 0.867 | 6.566 |
| NTV0 | no light, no lead | 38 | 26 | 26 | 0.316 | 4.400 | 5.155 | 11.830 | 3.000 | 0.763 | 3.581 |
| NTV0 | red light at the line, no lead | 239 | 228 | 226 | 0.046 | 5.692 | 7.423 | 26.047 | 3.000 | 1.000 | 7.373 |
| JTN0 | all | 878 | 172 | 166 | 0.804 | 2.136 | 2.280 | 2.849 | 5.000 | 0.432 | 0.182 |
| JTN0 | lead 8-30 m | 49 | 13 | 13 | 0.735 | 2.211 | 2.320 | 5.985 | 5.000 | 0.694 | 0.800 |
| JTN0 | lead <= 8 m | 477 | 4 | 0 | 0.992 | 1.770 | 1.874 | 2.175 | 5.000 | 0.249 | 0.083 |
| JTN0 | light of unknown state, no lead | 75 | 42 | 42 | 0.440 | 3.049 | 3.592 | 19.106 | 4.000 | 0.733 | 3.399 |
| JTN0 | no light, no lead | 38 | 2 | 2 | 0.947 | 2.477 | 2.504 | 3.500 | 2.000 | 0.211 | 0.001 |
| JTN0 | red light at the line, no lead | 239 | 111 | 109 | 0.536 | 3.221 | 3.650 | 16.844 | 5.000 | 0.682 | 2.470 |
| JTN1 | all | 878 | 134 | 126 | 0.847 | 2.142 | 2.274 | 2.669 | 5.000 | 0.399 | 0.130 |
| JTN1 | lead 8-30 m | 49 | 12 | 12 | 0.755 | 2.284 | 2.361 | 4.410 | 5.000 | 0.592 | 0.621 |
| JTN1 | lead <= 8 m | 477 | 5 | 0 | 0.990 | 1.762 | 1.883 | 2.175 | 5.000 | 0.233 | 0.060 |
| JTN1 | light of unknown state, no lead | 75 | 29 | 29 | 0.613 | 2.969 | 3.390 | 15.579 | 4.000 | 0.733 | 2.422 |
| JTN1 | no light, no lead | 38 | 1 | 1 | 0.974 | 2.463 | 2.495 | 2.841 | 2.000 | 0.158 | 0.001 |
| JTN1 | red light at the line, no lead | 239 | 87 | 84 | 0.636 | 3.227 | 3.634 | 14.857 | 5.000 | 0.623 | 1.559 |

JTN1 static successes (744) vs zeros (134): arc4 at k0 1.97 / 3.49 m, k2 2.11 / 4.36, k3 2.14 / 5.74, max k>=3 2.49 / 20.90; v at k3 0.01 / 0.10 m/s, k5 0.00 / 0.61; rolled 0.29 / 0.99; zero roll k {2.0: 26, 3.0: 10, 4.0: 54, 5.0: 42, 6.0: 1}
  AUC of a0 (plan 4 s arc at that decision) for the window's zero: 0.862
  AUC of a2 (plan 4 s arc at that decision) for the window's zero: 0.907
  AUC of a3 (plan 4 s arc at that decision) for the window's zero: 0.949

RSG static successes (493) vs zeros (385): arc4 at k0 3.34 / 5.15 m, k2 3.75 / 6.97, k3 3.38 / 8.97, max k>=3 3.65 / 24.53; v at k3 0.13 / 0.30 m/s, k5 0.25 / 1.31; rolled 0.84 / 1.00; zero roll k {2.0: 42, 3.0: 314, 4.0: 29}
  AUC of a0 (plan 4 s arc at that decision) for the window's zero: 0.885
  AUC of a2 (plan 4 s arc at that decision) for the window's zero: 0.937
  AUC of a3 (plan 4 s arc at that decision) for the window's zero: 0.955

Zero rate of the static windows by the plan's 4 s arc at decision 0 (all four models; k = 0 is under force-GT, so this is the plan on the human's own standing input):

| plan 4 s arc at k = 0 (m) | RSG | NTV0 | JTN0 | JTN1 | windows (all models) |
|---|--:|--:|--:|--:|--:|
| (0.0, 1.5] | 0.000 | 0.000 | 0.000 | 0.000 | 270.000 |
| (1.5, 2.0] | 0.000 | 0.000 | 0.029 | 0.019 | 544.000 |
| (2.0, 2.5] | 0.000 | 0.000 | 0.094 | 0.023 | 398.000 |
| (2.5, 3.0] | 0.033 | 0.038 | 0.388 | 0.333 | 393.000 |
| (3.0, 4.0] | 0.227 | 0.186 | 0.523 | 0.451 | 797.000 |
| (4.0, 6.0] | 0.706 | 0.697 | 0.441 | 0.353 | 860.000 |
| (6.0, 99.0] | 0.833 | 0.911 | 0.714 | 0.833 | 250.000 |

JTN1 by decision (zeros | successes), median plan arc4 m / ego v m/s / displacement m:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| arc4 zeros | 3.49 | 3.52 | 4.36 | 5.74 | 7.99 | 10.94 | 13.48 | 16.56 | 18.89 | 20.90 |
| arc4 ok | 1.97 | 2.03 | 2.11 | 2.14 | 2.16 | 2.18 | 2.18 | 2.18 | 2.22 | 2.23 |
| v zeros | 0.00 | 0.00 | 0.03 | 0.10 | 0.28 | 0.61 | 1.09 | 1.76 | 2.52 | 3.29 |
| v ok | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| disp zeros | 0.00 | 0.00 | 0.00 | 0.03 | 0.14 | 0.36 | 0.76 | 1.53 | 2.53 | 4.07 |
| disp ok | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 |

RSG by decision (zeros | successes), median plan arc4 m / ego v m/s / displacement m:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| arc4 zeros | 5.15 | 5.20 | 6.97 | 8.97 | 11.79 | 14.82 | 17.97 | 20.84 | 22.84 | 24.53 |
| arc4 ok | 3.34 | 3.41 | 3.75 | 3.38 | 2.85 | 2.41 | 2.26 | 2.42 | 2.62 | 2.99 |
| v zeros | 0.00 | 0.00 | 0.08 | 0.30 | 0.71 | 1.31 | 2.10 | 2.92 | 3.75 | 4.48 |
| v ok | 0.00 | 0.00 | 0.04 | 0.13 | 0.21 | 0.25 | 0.26 | 0.27 | 0.24 | 0.21 |
| disp zeros | 0.00 | 0.00 | 0.01 | 0.11 | 0.36 | 0.87 | 1.69 | 2.98 | 4.73 | 6.77 |
| disp ok | 0.00 | 0.00 | 0.01 | 0.05 | 0.14 | 0.27 | 0.40 | 0.54 | 0.67 | 0.80 |

Static windows, JTN1 vs RSG: both zero 134, JTN1 only 0, RSG only 251.

## C. Points to 0.93

2228 common held windows; mean: RSG 0.7857, NTV0 0.7699, JTN0 0.8744, JTN1 0.8909; best-of-4 per window 0.9127; 0.93 needs 2072 points, JTN1 has 1985.0 (gap 87.1).

| loss class (points = summed 1 - score) | RSG | NTV0 | JTN0 | JTN1 | JTN1 windows | best-of-4 on JTN1's windows |
|---|--:|--:|--:|--:|--:|--:|
| static zero: red light at the line, no lead | 226.0 | 226.0 | 109.0 | 85.0 | 85.0 | 84.0 |
| static zero: light of unknown state, no lead | 50.0 | 54.0 | 42.0 | 29.0 | 29.0 | 29.0 |
| human-track offroad (ceiling) | 26.0 | 24.0 | 23.0 | 23.0 | 23.0 | 23.0 |
| offroad zero: other strata | 10.0 | 19.0 | 20.0 | 21.0 | 21.0 | 6.1 |
| launch partial (late / slow launch) | 3.6 | 3.6 | 15.5 | 17.7 | 47.0 | 2.6 |
| corridor zero: braking | 16.0 | 18.0 | 13.0 | 13.0 | 13.0 | 12.0 |
| offroad zero: turn stratum | 9.0 | 16.0 | 11.0 | 13.0 | 13.0 | 6.0 |
| static zero: lead 8-30 m | 26.0 | 28.0 | 13.0 | 12.0 | 12.0 | 12.0 |
| slow partial: other | 7.0 | 7.1 | 7.6 | 7.7 | 57.0 | 4.7 |
| corridor zero: other | 9.0 | 9.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| static zero: lead <= 8 m | 53.0 | 66.0 | 3.0 | 4.0 | 4.0 | 3.0 |
| AF-front zero | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 | 1.0 |
| slow partial: turn | 2.3 | 2.1 | 2.8 | 2.2 | 15.0 | 1.5 |
| rear-hit partial | 1.8 | 2.1 | 2.2 | 2.1 | 8.0 | 1.7 |
| slow partial: braking | 2.9 | 2.0 | 3.0 | 2.0 | 28.0 | 1.1 |
| slow partial: cruise | 1.9 | 1.7 | 1.8 | 1.3 | 16.0 | 1.0 |
| corridor zero: launch | 1.0 | 3.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| static zero: no light, no lead | 27.0 | 26.0 | 2.0 | 1.0 | 1.0 | 1.0 |
| AF-side zero | 1.0 | 0.0 | 0.0 | 0.0 | - | - |
| corridor zero: turn | 2.0 | 2.0 | 2.0 | 0.0 | - | - |
| all | 477.4 | 512.6 | 279.9 | 243.0 | 381.0 | 194.6 |

- stands at red, no lead, static: 239 windows; JTN1 152.0 points, stand-still read (starter) 237.0, RSG 11.0
- launches at green, no lead: 40 windows; JTN1 31.3 points, stand-still read (starter) 18.3, RSG 38.4
- launches at green, behind a lead: 34 windows; JTN1 30.8 points, stand-still read (starter) 20.3, RSG 32.4
- 'stay at a red line without a lead' (perfect on the first group, never launching on the second) against JTN1: +85.0 -13.0 = +72.0 points

Ceiling: 26 human-track offroad windows -> 0.9883.

