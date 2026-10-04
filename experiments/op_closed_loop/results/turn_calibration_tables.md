B2D rows: 552577 total; primary arms (drive / opc / lsc, seeds 2-3): 43342 rows, 19 routes; held-out routes (other arms, routes not in the primary 19): 77728 rows, 50 routes

## 1. Ratio (model / required) by required |kappa| (bins of the reference; reference = dense route chord curvature over max(4, min(v, 20)) m on B2D; logged human future chord curvature at 10 m on WOD / NAVSIM)

| |kappa_req| bin | B2D action head (primary 19 routes) | B2D action head (held-out routes) | B2D plan, 1 s point (primary) | B2D plan, 1 s point (held-out) | WOD plan (real 10 Hz history) | NAVSIM plan (2 Hz history interpolated) | NAVSIM plan (native 2 Hz exam protocol) |
|---|---|---|---|---|---|---|---|
| 0.003-0.006 (R 167-333 m) | 1.30 [0.57, 2.29] (n 548, 14 cl) | 1.22 [0.79, 1.68] (n 2125, 27 cl) | 2.10 [1.15, 3.08] (n 533, 14 cl) | 1.75 [1.07, 2.19] (n 2147, 27 cl) | 0.91 [0.72, 1.08] (n 51, 51 cl) | 1.44 [1.32, 1.55] (n 1305, 108 cl) | 1.24 [1.07, 1.40] (n 1307, 108 cl) |
| 0.006-0.012 (R 83-167 m) | 0.81 [0.27, 1.07] (n 1450, 14 cl) | 0.82 [0.13, 1.01] (n 1240, 26 cl) | 1.22 [0.58, 2.10] (n 1476, 14 cl) | 1.12 [0.42, 1.39] (n 1127, 26 cl) | 0.94 [0.52, 1.46] (n 27, 27 cl) | 1.49 [1.41, 1.57] (n 958, 107 cl) | 1.77 [1.59, 1.95] (n 961, 107 cl) |
| 0.012-0.025 (R 40-83 m) | 0.45 [0.24, 0.93] (n 1302, 14 cl) | 0.54 [0.30, 0.76] (n 855, 26 cl) | 1.45 [0.77, 1.84] (n 1289, 14 cl) | 1.13 [0.75, 1.50] (n 796, 26 cl) | 0.97 [0.74, 1.18] (n 25, 25 cl) | 1.48 [1.38, 1.57] (n 782, 101 cl) | 1.95 [1.83, 2.06] (n 789, 101 cl) |
| 0.025-0.050 (R 20-40 m) | 0.48 [0.35, 0.60] (n 1634, 14 cl) | 0.27 [0.16, 0.40] (n 1179, 26 cl) | 1.22 [0.85, 1.44] (n 1577, 14 cl) | 1.55 [1.04, 1.99] (n 1171, 26 cl) | 0.91 [0.56, 1.28] (n 15, 15 cl) | 1.47 [1.40, 1.52] (n 1314, 101 cl) | 1.93 [1.83, 2.01] (n 1324, 101 cl) |
| 0.050-0.100 (R 10-20 m) | 0.63 [0.49, 0.74] (n 4001, 14 cl) | 0.55 [0.49, 0.60] (n 3494, 26 cl) | 1.27 [1.04, 1.45] (n 3905, 14 cl) | 1.29 [1.15, 1.42] (n 3452, 26 cl) | 0.65 [0.24, 0.99] (n 18, 18 cl) | 1.40 [1.34, 1.46] (n 1106, 87 cl) | 1.68 [1.63, 1.74] (n 1115, 87 cl) |
| 0.100-0.250 (R 4-10 m) | 0.51 [0.43, 0.60] (n 3633, 11 cl) | 0.54 [0.44, 0.65] (n 3277, 18 cl) | 1.35 [1.15, 1.50] (n 3630, 11 cl) | 1.25 [1.15, 1.36] (n 3191, 18 cl) | 1.06 [0.87, 1.22] (n 7, 7 cl) | 1.23 [1.16, 1.29] (n 163, 40 cl) | 1.21 [1.07, 1.31] (n 165, 40 cl) |

Same-sign share per bin (action head primary / plan primary / NAVSIM plan): 0.003-0.006 (R 167-333 m) 64% / 66% / 95%; 0.006-0.012 (R 83-167 m) 80% / 77% / 97%; 0.012-0.025 (R 40-83 m) 63% / 78% / 96%; 0.025-0.050 (R 20-40 m) 81% / 88% / 97%; 0.050-0.100 (R 10-20 m) 91% / 97% / 98%; 0.100-0.250 (R 4-10 m) 95% / 98% / 99%

## 1b. Sensitivity of the open-loop ratios to the arc of the chord (a = 10 / 15 / 25 m), required |kappa| 0.012-0.1

| source | a=10 | a=15 | a=25 |
|---|---|---|---|
| WOD plan | 0.77 [0.52, 0.98] (n 58, 58 cl) | 1.05 [0.91, 1.17] (n 39, 39 cl) | 1.08 [1.02, 1.15] (n 17, 17 cl) |
| NAVSIM plan (interp.) | 1.43 [1.37, 1.49] (n 3202, 108 cl) | 1.30 [1.27, 1.34] (n 2473, 105 cl) | 1.26 [1.22, 1.29] (n 733, 77 cl) |
| NAVSIM plan (native) | 1.79 [1.72, 1.86] (n 3228, 108 cl) | 1.53 [1.48, 1.57] (n 2477, 105 cl) | 1.44 [1.39, 1.50] (n 732, 77 cl) |

B2D action head, required over A = 5 m / 10 m / max(4, min(v, 20)) m, |kappa_req| >= 0.04 (primary):
req5 0.51 [0.39, 0.63] (n 8450, 14 cl); req10 0.50 [0.36, 0.62] (n 8926, 14 cl); req_v 0.51 [0.38, 0.63] (n 8434, 14 cl)

## 2. B2D action head: what else the ratio depends on (primary arms; turn ticks |kappa_req| >= 0.04 unless stated)

| split | level | action head (turn ticks >= 0.04) | action head (0.012-0.04) | plan 1 s (>= 0.04) |
|---|---|---|---|---|
| speed (m/s) | 1-2.5 | 0.37 [0.24, 0.55] (n 2461, 9 cl) | 0.39 [0.20, 0.93] (n 816, 9 cl) | 1.53 [1.27, 1.80] (n 2362, 9 cl) |
| speed (m/s) | 2.5-4 | 0.55 [0.41, 0.68] (n 3405, 13 cl) | 0.67 [0.38, 1.41] (n 590, 10 cl) | 1.16 [0.98, 1.32] (n 3339, 13 cl) |
| speed (m/s) | 4-6 | 0.62 [0.42, 0.78] (n 2231, 12 cl) | 0.37 [0.20, 0.59] (n 699, 13 cl) | 1.03 [0.91, 1.17] (n 2222, 12 cl) |
| speed (m/s) | 6-9 | 0.63 [0.26, 0.94] (n 330, 6 cl) | 0.53 [0.18, 0.91] (n 171, 5 cl) | 0.90 [0.50, 1.12] (n 330, 6 cl) |
| speed (m/s) | 9-30 | 0.16 [n/a] (n 7, 1 cl) | n/a (n 0) | 0.12 [n/a] (n 7, 1 cl) |
| side | left (req < 0) | 0.64 [0.52, 0.79] (n 5133, 8 cl) | 0.34 [0.23, 0.46] (n 1569, 8 cl) | 1.43 [1.19, 1.57] (n 5071, 8 cl) |
| side | right (req > 0) | 0.39 [0.21, 0.51] (n 3301, 8 cl) | 0.75 [0.40, 1.23] (n 707, 8 cl) | 1.07 [0.74, 1.36] (n 3189, 8 cl) |
| owner | op (action head steers) | 0.79 [n/a] (n 98, 2 cl) | 0.60 [n/a] (n 31, 1 cl) | 0.79 [n/a] (n 104, 2 cl) |
| owner | route zone (route steers) | 0.48 [0.35, 0.61] (n 6965, 14 cl) | 0.47 [0.33, 0.76] (n 1981, 14 cl) | 1.23 [1.07, 1.39] (n 6810, 14 cl) |
| owner | div (route steers) | 0.74 [-0.02, 1.05] (n 1371, 3 cl) | 0.44 [-0.04, 1.16] (n 264, 3 cl) | 1.43 [-0.04, 1.61] (n 1346, 3 cl) |
| arm | drive | 0.56 [0.44, 0.67] (n 2746, 14 cl) | 0.59 [0.40, 1.01] (n 795, 14 cl) | 1.26 [1.10, 1.42] (n 2697, 14 cl) |
| arm | opc | 0.52 [0.39, 0.63] (n 2832, 14 cl) | 0.44 [0.31, 0.72] (n 764, 14 cl) | 1.25 [1.07, 1.45] (n 2774, 14 cl) |
| arm | lsc | 0.46 [0.31, 0.59] (n 2856, 14 cl) | 0.35 [0.21, 0.48] (n 717, 14 cl) | 1.23 [1.01, 1.46] (n 2789, 14 cl) |
| route desire input | desire = 0 | 0.57 [0.10, 1.01] (n 1727, 3 cl) | 0.83 [0.47, 1.52] (n 466, 12 cl) | 1.28 [0.98, 1.61] (n 1709, 3 cl) |
| route desire input | desire != 0 (route turn desire given) | 0.50 [0.39, 0.61] (n 6707, 14 cl) | 0.37 [0.26, 0.56] (n 1810, 13 cl) | 1.24 [1.07, 1.41] (n 6551, 14 cl) |

maneuver radius (route minimum radius of the turn the tick belongs to, smoothed 1.5 m), action head:

| min radius | action head | plan 1 s |
|---|---|---|
| 0-6 m | 0.41 [n/a] (n 345, 1 cl) | 0.89 [n/a] (n 316, 1 cl) |
| 6-9 m | 0.53 [0.46, 0.60] (n 4382, 9 cl) | 1.35 [1.13, 1.50] (n 4326, 9 cl) |
| 9-13 m | 0.78 [0.66, 0.94] (n 2455, 4 cl) | 1.31 [1.03, 1.54] (n 2425, 4 cl) |
| 13-40 m | 0.59 [n/a] (n 513, 1 cl) | 1.32 [n/a] (n 466, 1 cl) |

B2D turn types present (route maneuvers, dense route): ticks by type (0 straight, 1 bend 8-25 deg, 2 turn 25-60, 3 turn 60-120, 4 tight > 120): {0: 31787, 1: 0, 2: 0, 3: 11555, 4: 0}
img command given (non-'None') ticks in these arms: 0

## 2b. Speed x curvature, action head, all B2D routes pooled (primary + held-out; cells with >= 100 ticks)

| speed (m/s) | |k| 0.012-0.04 | |k| 0.04-0.1 | |k| 0.1-0.3 |
|---|---|---|---|
| 1-2.5 | 0.40 [0.28, 0.68] (n 8328, 27 cl) | 0.46 [0.30, 0.51] (n 12414, 32 cl) | 0.55 [0.49, 0.63] (n 10614, 23 cl) |
| 2.5-4 | 0.67 [0.43, 1.17] (n 6345, 34 cl) | 0.63 [0.49, 0.71] (n 12829, 39 cl) | 0.57 [0.49, 0.66] (n 18189, 28 cl) |
| 4-6 | 0.40 [0.26, 0.57] (n 7565, 37 cl) | 0.72 [0.51, 0.81] (n 21350, 38 cl) | 0.51 [0.43, 0.59] (n 6671, 26 cl) |
| 6-9 | 0.53 [0.17, 0.88] (n 1193, 19 cl) | 0.69 [0.23, 1.02] (n 1763, 15 cl) | 0.23 [n/a] (n 219, 2 cl) |

Joint fit on all routes, log(ratio of a cell) = a + b ln|k| + c ln v (cells of the table above weighted by ticks; cluster bootstrap over routes):

| quantity | b (curvature exponent - 1) | c (speed exponent) |
|---|---|---|
| action head | +0.07 [-0.07, +0.22] | +0.26 [-0.13, +0.55] |
| plan 1 s | -0.01 [-0.12, +0.16] | -0.64 [-0.79, -0.43] |

## 2c. Integrated heading change over each junction turn (sum of kappa * v * dt over the turn's ticks / the route's turn angle), runs that cover the turn

| model curvature | ratio over (run, turn) groups [route-cluster CI] | groups | routes |
|---|---|---|---|
| action head | 0.61 [0.54, 0.67] | 874 | 36 |
| plan 1 s point | 1.18 [1.07, 1.29] | 874 | 36 |

## 3. Open-loop plan ratio by turn angle of the logged future, speed and side (|kappa_req(10 m)| >= 0.006)

| source | split | level | ratio |
|---|---|---|---|
| WOD plan | turn angle (deg) | <10 | 0.92 [0.72, 1.15] (n 44, 44 cl) |
| WOD plan | turn angle (deg) | 10-25 | 0.97 [0.74, 1.22] (n 16, 16 cl) |
| WOD plan | turn angle (deg) | 25-60 | 0.84 [0.45, 1.14] (n 22, 22 cl) |
| WOD plan | turn angle (deg) | >=60 | 0.79 [0.44, 1.04] (n 10, 10 cl) |
| WOD plan | speed (m/s) | <5 | 0.79 [0.58, 0.98] (n 67, 67 cl) |
| WOD plan | speed (m/s) | 5-9 | 1.16 [1.05, 1.27] (n 22, 22 cl) |
| WOD plan | speed (m/s) | 9-14 | n/a (n 3) |
| WOD plan | speed (m/s) | >=14 | n/a (n 0) |
| WOD plan | side (sign of required) | left | 0.56 [0.12, 0.92] (n 37, 37 cl) |
| WOD plan | side (sign of required) | right | 1.01 [0.86, 1.14] (n 55, 55 cl) |
| NAVSIM plan (interp.) | turn angle (deg) | <10 | 1.35 [1.25, 1.45] (n 880, 105 cl) |
| NAVSIM plan (interp.) | turn angle (deg) | 10-25 | 1.50 [1.43, 1.55] (n 1148, 103 cl) |
| NAVSIM plan (interp.) | turn angle (deg) | 25-60 | 1.44 [1.37, 1.50] (n 1830, 105 cl) |
| NAVSIM plan (interp.) | turn angle (deg) | >=60 | 1.30 [1.25, 1.36] (n 465, 72 cl) |
| NAVSIM plan (interp.) | speed (m/s) | <5 | 1.36 [1.29, 1.44] (n 2268, 104 cl) |
| NAVSIM plan (interp.) | speed (m/s) | 5-9 | 1.52 [1.49, 1.55] (n 1843, 108 cl) |
| NAVSIM plan (interp.) | speed (m/s) | 9-14 | 1.41 [1.31, 1.50] (n 206, 42 cl) |
| NAVSIM plan (interp.) | speed (m/s) | >=14 | 0.69 [n/a] (n 6, 1 cl) |
| NAVSIM plan (interp.) | side (sign of required) | left | 1.42 [1.36, 1.47] (n 2650, 108 cl) |
| NAVSIM plan (interp.) | side (sign of required) | right | 1.40 [1.31, 1.48] (n 1673, 100 cl) |
| NAVSIM plan (native) | turn angle (deg) | <10 | 2.10 [1.91, 2.28] (n 889, 105 cl) |
| NAVSIM plan (native) | turn angle (deg) | 10-25 | 1.96 [1.88, 2.05] (n 1158, 103 cl) |
| NAVSIM plan (native) | turn angle (deg) | 25-60 | 1.73 [1.65, 1.81] (n 1842, 105 cl) |
| NAVSIM plan (native) | turn angle (deg) | >=60 | 1.45 [1.36, 1.54] (n 465, 72 cl) |
| NAVSIM plan (native) | speed (m/s) | <5 | 1.61 [1.52, 1.69] (n 2299, 104 cl) |
| NAVSIM plan (native) | speed (m/s) | 5-9 | 1.99 [1.94, 2.04] (n 1843, 108 cl) |
| NAVSIM plan (native) | speed (m/s) | 9-14 | 1.88 [1.48, 2.26] (n 206, 42 cl) |
| NAVSIM plan (native) | speed (m/s) | >=14 | 0.07 [n/a] (n 6, 1 cl) |
| NAVSIM plan (native) | side (sign of required) | left | 1.85 [1.78, 1.93] (n 2675, 108 cl) |
| NAVSIM plan (native) | side (sign of required) | right | 1.51 [1.40, 1.61] (n 1679, 100 cl) |

## 4. Linear or nonlinear: slope of the binned ratio on log|kappa_req| (bins >= 0.006), cluster-bootstrapped

| source | slope per ln-unit of |kappa| | 95% CI | reading |
|---|---|---|---|
| B2D action head (primary 19 routes) | -0.059 | [-0.221, +0.104] | flat within CI (linear) |
| B2D action head (held-out routes) | -0.072 | [-0.147, +0.126] | flat within CI (linear) |
| B2D plan, 1 s point (primary) | +0.011 | [-0.229, +0.209] | flat within CI (linear) |
| B2D plan, 1 s point (held-out) | +0.056 | [-0.052, +0.292] | flat within CI (linear) |
| WOD plan (real 10 Hz history) | -0.009 | [-0.178, +0.124] | flat within CI (linear) |
| NAVSIM plan (2 Hz history interpolated) | -0.085 | [-0.113, -0.053] | ratio falls with sharpness |
| NAVSIM plan (native 2 Hz exam protocol) | -0.195 | [-0.257, -0.137] | ratio falls with sharpness |

## 5. Candidate calibration map (not adopted): fitted on the primary 19 routes, checked on held-out routes

- linear gain: model = 0.510 x required (ratio of sums over |kappa_req| >= 0.01); the calibration is kappa_cal = kappa_act / 0.510
- power law: ratio(|k|) = 0.570 x (|k| / 0.05)^-0.072; kappa_cal solves ratio(|k|) |k| = |kappa_act|
- speed-dependent gain (turn ticks >= 0.04, speed bins [(1, 2.5), (2.5, 4), (4, 9)], mean speeds [1.84, 3.32, 4.97]): [0.373, 0.551, 0.622]

| map | set | ratio of sums (turn ticks >= 0.012) | CI | RMSE of kappa_cal vs required (1/m) | ratio in 0.012-0.04 | in 0.04-0.1 | in >= 0.1 |
|---|---|---|---|---|---|---|---|
| identity | fit (primary) | 0.51 | [0.39, 0.61] | 0.0745 | 0.47 | 0.61 | 0.50 |
| linear gain | fit (primary) | 1.00 | [0.77, 1.21] | 0.0852 | 0.91 | 1.20 | 0.99 |
| power law | fit (primary) | 0.95 | [0.74, 1.16] | 0.0848 | 0.83 | 1.14 | 0.97 |
| piecewise-linear gain | fit (primary) | 0.94 | [0.72, 1.14] | 0.0843 | 0.88 | 1.10 | 0.96 |
| speed-dependent gain | fit (primary) | 0.99 | [0.77, 1.18] | 0.0881 | 0.92 | 1.14 | 1.03 |
| identity | held-out routes | 0.51 | [0.44, 0.59] | 0.0764 | 0.39 | 0.53 | 0.54 |
| linear gain | held-out routes | 1.01 | [0.87, 1.16] | 0.1051 | 0.77 | 1.03 | 1.06 |
| power law | held-out routes | 0.94 | [0.82, 1.05] | 0.0779 | 0.70 | 0.97 | 0.98 |
| piecewise-linear gain | held-out routes | 0.93 | [0.81, 1.04] | 0.0772 | 0.72 | 0.95 | 0.97 |
| speed-dependent gain | held-out routes | 1.06 | [0.89, 1.22] | 0.1332 | 0.71 | 1.00 | 1.17 |

Plan-derived curvature (NAVSIM, interpolated protocol): linear gain fitted on half of the logs, checked on the other half / WOD / B2D (|kappa_req| >= 0.012):

gain fitted: 1.390 (n 1742). Residual ratio after dividing the plan curvature by it:

| set | before | after dividing by the gain |
|---|---|---|
| NAVSIM other half | 1.42 [1.37, 1.47] (n 1623, 49 cl) | 1.02 [0.98, 1.06] (n 1623, 49 cl) |
| WOD (real 10 Hz) | 0.85 [0.65, 1.02] (n 65, 65 cl) | 0.61 [0.47, 0.72] (n 65, 65 cl) |
| B2D plan 1 s (primary) | 1.25 [1.07, 1.43] (n 10542, 14 cl) | 0.90 [0.77, 1.02] (n 10542, 14 cl) |
| B2D plan 1 s (held-out) | 1.26 [1.16, 1.36] (n 8682, 26 cl) | 0.90 [0.84, 0.98] (n 8682, 26 cl) |

## 6. Dense vs sparse route: kinematic pure pursuit through the 38 route turns (>= 25 deg) of 36 routes

| path given | speed (m/s) | peak cross-track vs dense centreline, median [p90] (m) | runs with peak > 1 m | > 1.75 m (lane half-width) | peak commanded |kappa| (1/m), median | heading change delivered / route's |
|---|---|---|---|---|---|---|
| dense route (shipped zones) | 3 | 0.28 [0.38] | 0% | 0% | 0.119 | 1.01 |
| leaderboard downsample (50 m) | 3 | 3.24 [4.94] | 100% | 82% | 0.127 | 1.01 |
| road polyline, 10 m decimation, no noise | 3 | 1.14 [1.60] | 63% | 5% | 0.135 | 1.01 |
| road polyline, 5 m decimation, no noise | 3 | 0.42 [0.60] | 0% | 0% | 0.121 | 1.01 |
| road polyline 10 m, noise sd 1 m | 3 | 1.60 [2.53] | 87% | 40% | 0.140 | 1.01 |
| ... sd 2 m | 3 | 2.58 [4.23] | 98% | 79% | 0.146 | 1.00 |
| ... sd 4 m | 3 | 4.82 [8.00] | 100% | 98% | 0.173 | 1.01 |
| dense route (shipped zones) | 5 | 0.52 [0.66] | 0% | 0% | 0.114 | 1.02 |
| leaderboard downsample (50 m) | 5 | 3.06 [4.90] | 100% | 97% | 0.091 | 1.03 |
| road polyline, 10 m decimation, no noise | 5 | 1.24 [1.70] | 66% | 5% | 0.105 | 1.02 |
| road polyline, 5 m decimation, no noise | 5 | 0.65 [0.89] | 5% | 0% | 0.112 | 1.02 |
| road polyline 10 m, noise sd 1 m | 5 | 1.67 [2.70] | 88% | 45% | 0.109 | 1.02 |
| ... sd 2 m | 5 | 2.59 [4.29] | 98% | 79% | 0.113 | 1.02 |
| ... sd 4 m | 5 | 4.74 [8.00] | 100% | 96% | 0.130 | 1.02 |
| dense route (shipped zones) | 8 | 0.98 [1.30] | 47% | 0% | 0.099 | 1.05 |
| leaderboard downsample (50 m) | 8 | 3.02 [5.01] | 100% | 100% | 0.074 | 1.04 |
| road polyline, 10 m decimation, no noise | 8 | 1.61 [2.15] | 97% | 39% | 0.091 | 1.04 |
| road polyline, 5 m decimation, no noise | 8 | 1.10 [1.50] | 66% | 5% | 0.095 | 1.05 |
| road polyline 10 m, noise sd 1 m | 8 | 2.00 [3.03] | 96% | 66% | 0.089 | 1.04 |
| ... sd 2 m | 8 | 2.73 [4.16] | 98% | 83% | 0.090 | 1.04 |
| ... sd 4 m | 8 | 4.56 [7.76] | 100% | 95% | 0.096 | 1.03 |

By turn direction and radius, leaderboard downsample at 5 m/s (peak cross-track median, m; peak |kappa| sparse / dense):

| group | n turns | peak cross-track (lb50) | peak cross-track (dense) | peak |kappa| lb50 / dense |
|---|---|---|---|---|
| left turns | 24 | 3.30 | 0.44 | 0.91 |
| right turns | 14 | 1.85 | 0.64 | 0.78 |
| R_min < 6 m | 2 | 3.15 | 0.86 | 0.73 |
| R_min 6-9 m | 21 | 2.56 | 0.64 | 0.78 |
| R_min 9-13 m | 13 | 3.10 | 0.43 | 0.93 |
| R_min >= 13 m | 2 | 6.12 | 0.31 | 1.42 |
