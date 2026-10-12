# ZEROS-MECH: what the S-family zeros come from (held 452 scenes, 39 logs, 8 labels, 18 decisions)

Registered in [plans/2026-10-12-zeros-mech-prereg.md](../plans/2026-10-12-zeros-mech-prereg.md) (pushed before any read); script `scripts/zm_read.py`; box run dirs `$DATA_DIR/runs/zeros_mech/collect/20261012-155215` (collect) and `$DATA_DIR/runs/zeros_mech/rows/20261012-154945` (rows). Groups: base = `YR10m10-F-s0` (YS's start), YS = `YS-L18-s0/s1`, YS2G = `YS2G-L18-s0/s1`, YNONE = `YNONE-L18-s0/s1`, YR-s1 = `YR10m10-F-s1`. CIs: log-clustered bootstrap (39 logs, 10 000 draws, one draw for all statistics).

## Gates

| gate | line | read | verdict |
|:--|--:|--:|--:|
| G-cov | >= 99% of rollouts with 18 decisions; scene errors <= 1% | 0.9934 of 3616 rollouts; 0 scene errors | pass |
| G-act | asl actor poses vs replayed log <= 0.05 m on >= 99%; EGO vs anchor | 158200 / 158200 pairs (556 rollouts, max 1.6e-05 m), missing 0; EGO 10002 / 10002 (max 3.9e-06 m) | pass |
| G-dir | logged poses in a lane: same >= 95%, oncoming <= 1% | same 0.9957, oncoming 0.0039 of 8111 (25 in no lane) | pass |
| G-row | recomputed agent hinge = active.npz (1e-4) on >= 99.9% of rows | 1.0000 of 17306 rows (max diff 1.2e-07); 1446 active (0.084) | pass |
| G-zero | YS-L18-s1 zeros = zeros.csv | 43 / 43, 0 mismatches | pass |

The first full collect (`20261012-154957`) failed G-cov (215 scene errors: 64 workers each started BLAS / polars / rayon thread pools sized to the box and hit the thread limit; 8 more were the route-fold scenes without rollouts) and was voided; the code was fixed (one thread per worker, those 8 scenes skipped) and the whole collect rerun once. The asl stores EGO at its box centre (1.461 m ahead of the rear axle); G-act compares it with the anchor moved there.

## Zeros by label (context)

| label | zeros | corridor | offroad | collision | other | at-fault collisions |
|:--|--:|--:|--:|--:|--:|--:|
| YNONE-L18-s0 | 124 | 43 | 56 | 25 | 0 | 25 |
| YNONE-L18-s1 | 116 | 38 | 56 | 22 | 0 | 22 |
| YR10m10-F-s0 | 74 | 31 | 33 | 10 | 0 | 10 |
| YR10m10-F-s1 | 71 | 30 | 37 | 4 | 0 | 4 |
| YS-L18-s0 | 45 | 22 | 14 | 9 | 0 | 9 |
| YS-L18-s1 | 43 | 19 | 15 | 9 | 0 | 9 |
| YS2G-L18-s0 | 43 | 19 | 17 | 7 | 0 | 7 |
| YS2G-L18-s1 | 40 | 20 | 13 | 7 | 0 | 7 |

## R1 turn timing and command (H1)

Turn scenes: the log turns >= 45 deg over the 18 decisions (125 scenes); straight: < 15 deg (248); bend: the rest. delta = t_ego - t_log at half the log's heading change (s; never reached = +inf, `censored`); ds = the same in station along the logged path (m; + = turns further along = late).

| group | turn rollouts | censored | delta median [IQR] (s) | ds median [IQR] (m) | t_log - t_cmd (s, median) | t_ego - t_cmd | command side = turn side |
|:--|--:|--:|--:|--:|--:|--:|--:|
| base | 125 | 0.120 | +0.10 [-0.35, +0.95] | +1.1 [-0.7, +3.1] | 4.59 | 4.70 | 0.970 |
| YS | 250 | 0.112 | +0.12 [-0.31, +0.74] | +0.4 [-1.1, +1.8] | 4.59 | 4.43 | 0.957 |
| YS2G | 250 | 0.108 | +0.09 [-0.26, +0.75] | +0.3 [-0.9, +1.9] | 4.59 | 4.39 | 0.971 |
| YNONE | 250 | 0.124 | +0.15 [-0.21, +1.07] | +0.9 [-0.5, +3.0] | 4.59 | 4.66 | 0.966 |
| YR-s1 | 125 | 0.144 | +0.16 [-0.27, +1.04] | +1.1 [-0.5, +3.0] | 4.59 | 4.52 | 0.970 |

YS - base, median delta: +0.02 [-0.15, +0.14] s; median ds: -0.69 [-1.13, -0.00] m.

Corridor zeros on turn scenes, by where the ego reaches half the log's heading change along the road (ds) and its side of the log's arc at the event:

| group | events | early-inside | late-outside | early-outside | late-inside | same by the sign of delta (EI / LO / EO / LI) |
|:--|--:|--:|--:|--:|--:|--:|
| base | 23 | 5 | 18 | 0 | 0 | 5 / 14 / 4 / 0 |
| YS | 25 | 10 | 14 | 1 | 0 | 10 / 11 / 4 / 0 |
| YS2G | 26 | 6 | 20 | 0 | 0 | 6 / 12 / 8 / 0 |
| YNONE | 40 | 4 | 36 | 0 | 0 | 4 / 31 / 5 / 0 |
| YR-s1 | 24 | 2 | 20 | 2 | 0 | 2 / 14 / 8 / 0 |

Straight scenes (< 15 deg): share of in-play decisions with a LEFT / RIGHT command, and the corridor zeros' command around the drift onset (k_d = first decision with |e| > 1 m after which the offset keeps its sign and grows; window k_d - 3 .. k_d):

| group | non-straight command share of decisions | corridor zeros | command in window | share | flipped in window | non-straight from the start | command toward the drift side | route turns within 60 m | base rate (clean straight rollouts, any 4-decision window) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 0.346 [0.284, 0.410] | 7 | 7 | 1.00 | 3 | 1 | 2 | 4 | 0.41 |
| YS | 0.366 [0.307, 0.426] | 12 | 12 | 1.00 | 2 | 4 | 8 | 9 | 0.48 |
| YS2G | 0.343 [0.281, 0.405] | 12 | 12 | 1.00 | 3 | 5 | 7 | 12 | 0.44 |
| YNONE | 0.433 [0.376, 0.493] | 35 | 34 | 0.97 | 13 | 8 | 7 | 15 | 0.53 |
| YR-s1 | 0.331 [0.265, 0.398] | 4 | 4 | 1.00 | 2 | 1 | 1 | 2 | 0.40 |

YS - base, non-straight command share on straight scenes: +0.020 [-0.001, +0.041].

Peak heading rate on turn scenes: shortfall = 1 - sum(ego peak) / sum(log peak) over the label's turn rollouts (yaw rate over 0.5 s; curvature = heading change over 1 s / max(arc, 2 m), which takes speed out):

| label | turn rollouts | yaw-rate shortfall | curvature shortfall |
|:--|--:|--:|--:|
| YNONE-L18-s0 | 125 | 0.131 [0.082, 0.175] | 0.019 [-0.124, 0.132] |
| YNONE-L18-s1 | 125 | 0.143 [0.098, 0.191] | 0.037 [-0.070, 0.134] |
| YR10m10-F-s0 | 125 | 0.123 [0.080, 0.168] | 0.051 [-0.041, 0.129] |
| YR10m10-F-s1 | 125 | 0.137 [0.085, 0.190] | 0.066 [-0.060, 0.173] |
| YS-L18-s0 | 125 | 0.021 [-0.024, 0.074] | -0.065 [-0.169, 0.029] |
| YS-L18-s1 | 125 | 0.008 [-0.033, 0.055] | -0.068 [-0.151, 0.015] |
| YS2G-L18-s0 | 125 | 0.073 [0.031, 0.116] | 0.035 [-0.033, 0.100] |
| YS2G-L18-s1 | 125 | 0.075 [0.032, 0.122] | 0.031 [-0.048, 0.103] |

YS - base: yaw rate -0.108 [-0.142, -0.070], curvature -0.117 [-0.183, -0.047]. Descriptive: the served plan's 4 s heading change over the log's from the same moment, on decisions where the log turns >= 30 deg in the next 4 s: base 0.862, YS 0.861, YS2G 0.863, YNONE 0.860.

**Lines (YS pooled).** H1-command: (a) straight-scene corridor zeros with a non-straight command at or within 1.5 s before the drift onset: 12 / 12 = 1.00 (line >= 0.5; same side as the drift 0.67; base rate 0.48); (b) turn-scene corridor zeros early-inside 10 vs late-outside 14 (line EI >= LO) -> **H1-command partial: straight-road part only ((b) fails)**. H1-smooth: yaw-rate shortfall >= 0.20 with CI above 0 in all 8 labels: **not supported**; YS - base -0.108 [-0.142, -0.070]: **not the S lesson**; curvature version all >= 0.20: False.

## R2 cut-around the lead (H2)

In-play decisions. lead = a vehicle 0-15 m ahead of the front bumper, centre |y| < 1.5 m, heading within 30 deg, slower than the ego by > 1 m/s (or stopped while the ego moves); no-lead = no vehicle in that box; follow = in the box but not slower. d(t) = the served plan's lateral offset from the logged path at t minus the ego's own offset (left +); s_onc d(3) = toward the oncoming side (+).

| group | condition | decisions | mean |d(3)| (m) | mean s_onc d(3) (m) | |d(2)| / |d(4)| | |y_h(3)| | |y_log(3)| | share |d(3)| >= 0.5 m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| base | lead | 239 | 0.333 [0.242, 0.424] | +0.029 [-0.108, +0.145] | 0.219 / 0.435 | 0.466 | 0.491 | 0.180 |
| base | no-lead | 6660 | 0.705 [0.598, 0.813] | +0.095 [-0.004, +0.197] | 0.445 / 1.012 | 1.442 | 0.932 | 0.328 |
| base | follow | 679 | 0.243 [0.180, 0.317] | +0.034 [-0.026, +0.104] | 0.157 / 0.347 | 0.252 | 0.314 | 0.108 |
| YS | lead | 454 | 0.412 [0.269, 0.599] | +0.199 [+0.056, +0.405] | 0.301 / 0.525 | 0.548 | 0.583 | 0.233 |
| YS | no-lead | 13726 | 0.581 [0.504, 0.655] | +0.188 [+0.118, +0.259] | 0.380 / 0.825 | 1.376 | 0.800 | 0.304 |
| YS | follow | 1276 | 0.217 [0.168, 0.288] | +0.068 [+0.009, +0.148] | 0.141 / 0.306 | 0.236 | 0.258 | 0.099 |
| YS2G | lead | 435 | 0.387 [0.230, 0.557] | +0.135 [-0.005, +0.322] | 0.262 / 0.504 | 0.468 | 0.552 | 0.221 |
| YS2G | no-lead | 13714 | 0.545 [0.469, 0.619] | +0.144 [+0.074, +0.216] | 0.355 / 0.778 | 1.371 | 0.759 | 0.288 |
| YS2G | follow | 1316 | 0.210 [0.159, 0.277] | +0.039 [-0.025, +0.120] | 0.137 / 0.303 | 0.216 | 0.253 | 0.084 |
| YNONE | lead | 422 | 0.668 [0.487, 0.904] | +0.163 [-0.053, +0.417] | 0.477 / 0.840 | 0.763 | 0.836 | 0.351 |
| YNONE | no-lead | 13158 | 0.847 [0.742, 0.946] | +0.115 [+0.016, +0.209] | 0.557 / 1.184 | 1.549 | 1.118 | 0.396 |
| YNONE | follow | 1194 | 0.305 [0.243, 0.387] | +0.093 [+0.011, +0.193] | 0.192 / 0.439 | 0.312 | 0.351 | 0.141 |
| YR-s1 | lead | 268 | 0.352 [0.212, 0.504] | +0.088 [-0.107, +0.272] | 0.229 / 0.476 | 0.396 | 0.537 | 0.160 |
| YR-s1 | no-lead | 6688 | 0.666 [0.568, 0.760] | +0.060 [-0.035, +0.158] | 0.422 / 0.950 | 1.441 | 0.882 | 0.320 |
| YR-s1 | follow | 697 | 0.205 [0.168, 0.253] | +0.016 [-0.028, +0.073] | 0.128 / 0.299 | 0.207 | 0.249 | 0.093 |

Differences (the line uses YS - base, |d(3)|):

| a - b | lead: |d(3)| | lead: s_onc d(3) | no-lead: |d(3)| | no-lead: s_onc d(3) |
|:--|--:|--:|--:|--:|
| YS-base | +0.079 [-0.069, +0.260] | +0.170 [-0.027, +0.424] | -0.124 [-0.178, -0.071] | +0.093 [+0.031, +0.158] |
| YS-YNONE | -0.256 [-0.415, -0.136] | +0.036 [-0.084, +0.192] | -0.266 [-0.328, -0.206] | +0.073 [+0.017, +0.139] |
| YS2G-base | +0.053 [-0.092, +0.209] | +0.106 [-0.064, +0.333] | -0.161 [-0.211, -0.111] | +0.049 [-0.003, +0.104] |
| YNONE-base | +0.334 [+0.197, +0.529] | +0.134 [-0.082, +0.403] | +0.142 [+0.084, +0.204] | +0.020 [-0.039, +0.075] |

Difference in differences of |d(3)| ((lead - no-lead) - same for base): YS +0.202 [+0.047, +0.397], YS2G +0.214 [+0.072, +0.374].

Mean |d(3)| under the lead condition by gap (0-5 / 5-10 / 10-15 m): base 0.663 (n 14) / 0.328 (n 95) / 0.302 (n 130); YS 0.393 (n 35) / 0.386 (n 170) / 0.433 (n 249); YS2G 0.267 (n 28) / 0.397 (n 149) / 0.394 (n 258); YNONE 0.774 (n 46) / 0.521 (n 174) / 0.770 (n 202).

At-fault collisions (window: contact decision k_c and the 6 before = 3 s; lead = the lead condition at any of them; deflection = |d(3)| >= 0.5 m at any of them; object hit classified where it was 1 s before contact, in the ego frame):

| group | collisions | cut-around | straight-in | swerve, no lead | drift | cut-around share | object hit | hit a static object | hit = the lead |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 10 | 2 | 0 | 7 | 1 | 0.20 | adjacent 4, other 3, oncoming 2, crossing 1 | 2 | 0 |
| YS | 18 | 5 | 0 | 9 | 4 | 0.28 | adjacent 6, crossing 4, oncoming 4, other 2, - 2 | 0 | 0 |
| YS2G | 14 | 6 | 0 | 6 | 2 | 0.43 | other 5, adjacent 5, crossing 3, oncoming 1 | 0 | 0 |
| YNONE | 47 | 18 | 0 | 26 | 3 | 0.38 | other 17, adjacent 14, - 7, oncoming 7, in-lane lead 1, crossing 1 | 6 | 0 |
| YR-s1 | 4 | 0 | 0 | 3 | 1 | 0.00 | adjacent 2, - 1, oncoming 1 | 1 | 0 |

Row side (pool `on1_L18Y`, 17306 rows; agent hinge active on 1446): the object that binds the hinge on the served plan, classified by its position at t0 in the row's own frame (descriptive, not in the line):

| rows | n | in-lane same-direction lead | adjacent (1.5-3 m) | crossing (60-120 deg) | oncoming (> 120 deg) | other | full lead condition (closing) | binding step inside the ego's corridor (0.3 m margin) | object static (< 0.5 m/s) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all active | 1446 | 0.124 | 0.152 | 0.073 | 0.140 | 0.511 | 0.051 | 1.000 | 0.250 |
| top decile | 145 | 0.228 | 0.214 | 0.041 | 0.090 | 0.428 | 0.131 | 1.000 | 0.434 |

**Lines (YS pooled).** (i) lead: YS - base |d(3)| +0.079 [-0.069, +0.260] m (line >= +0.20 with CI above 0) and no-lead -0.124 [-0.178, -0.071] m (line < +0.10): not met; (ii) cut-around share of YS's at-fault collisions 0.28 of 18 vs base 0.20 of 10 (line >= 0.5 and above base): not met. Refutation line (lead CI includes 0 and cut-around < 1/4): not met -> **H2 not established**.

## R3 lane direction (H3)

Lane class of the rear axle (primary) and of the front bumper centre against the scene's lanes (evaluator's lane polygons and centre-line direction): same if any containing lane is within 60 deg of the ego heading, else oncoming if any is beyond 120 deg, else crossing; none = in no lane. Shares of in-play decisions; `wrong_lane` = the evaluator's own flag (unwrapped angle, known false positives).

| group | decisions | rear axle in an oncoming lane | front in an oncoming lane | rear crossing | rear in no lane | evaluator wrong_lane |
|:--|--:|--:|--:|--:|--:|--:|
| base | 7578 | 0.0103 [0.0051, 0.0158] | 0.0170 [0.0101, 0.0243] | 0.0009 | 0.0033 | 0.0581 [0.0427, 0.0744] |
| YS | 15456 | 0.0162 [0.0077, 0.0260] | 0.0199 [0.0107, 0.0305] | 0.0008 | 0.0028 | 0.0633 [0.0468, 0.0799] |
| YS2G | 15465 | 0.0155 [0.0076, 0.0241] | 0.0167 [0.0085, 0.0264] | 0.0005 | 0.0031 | 0.0615 [0.0456, 0.0777] |
| YNONE | 14774 | 0.0154 [0.0080, 0.0236] | 0.0223 [0.0133, 0.0320] | 0.0012 | 0.0034 | 0.0663 [0.0491, 0.0844] |
| YR-s1 | 7653 | 0.0101 [0.0047, 0.0161] | 0.0157 [0.0080, 0.0242] | 0.0007 | 0.0027 | 0.0568 [0.0425, 0.0717] |

Differences: YS - base rear +0.0059 [+0.0001, +0.0125], front +0.0029 [-0.0029, +0.0095]; YS2G - base rear +0.0052 [+0.0004, +0.0103], front -0.0003 [-0.0061, +0.0064]; YNONE - base rear +0.0051 [+0.0004, +0.0111], front +0.0053 [-0.0002, +0.0122].

Zeros with the rear axle in an oncoming lane at any decision in the 2 s before the event (k_ev - 4 .. k_ev):

| group | corridor | collision | offroad | corridor + collision share |
|:--|--:|--:|--:|--:|
| base | 5 / 31 | 2 / 10 | 3 / 33 | 0.17 |
| YS | 12 / 41 | 6 / 18 | 5 / 29 | 0.31 |
| YS2G | 11 / 39 | 1 / 14 | 4 / 30 | 0.23 |
| YNONE | 15 / 81 | 7 / 47 | 9 / 112 | 0.17 |
| YR-s1 | 6 / 30 | 1 / 4 | 3 / 37 | 0.21 |

Lead-conditioned deflections (lead and |d(3)| >= 0.5 m) whose plan end point (4 s) lies in an oncoming lane: base 0.02 of 43; YS 0.05 of 106; YS2G 0.01 of 96; YNONE 0.00 of 148; YR-s1 0.00 of 43.

**Lines (YS pooled).** Corridor + collision zeros with oncoming occupancy in the 2 s before the event: 18 / 59 = 0.31 (line >= 0.25) -> **H3 gets an arm**; oncoming decision share YS - base +0.0059 [+0.0001, +0.0125] (line: CI above 0) -> **the on-policy lesson made it worse**.

## YS-L18-s1: the 43 zeros, one line each

Columns: scene token (first 8 characters), type, event decision, scene class (log heading change over the rollout, deg), ego offset from the logged path at the event (m, left +), R1 reading, R2 reading (3 s before the event; collisions: class and object hit), R3 (oncoming in the 2 s before / rear-axle lane class at the event), and the rule-based fit (H1-cmd: straight / bend with a non-straight command in the onset window, or turn early-inside; H1-wide: turn late-outside; H2: lead and deflection; H3: oncoming; start: offroad from decision 0).

| scene | type | k_ev | scene class | e at event | R1 | R2 | R3 | fits |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 8051c60f | collision | 9 | straight +1 | +2.2 | onset k7, cmd yes (flip), away from drift | lead, deflect; cut-around, hit adjacent | - / same | H1-cmd, H2 |
| 0722adec | collision | 10 | straight -2 | -0.1 | onset k10, cmd no | no lead; drift, hit adjacent | - / same | none (drift) |
| daf38cec | collision | 11 | turn L +87 | -2.4 | ds +1.6 m, delta -0.80 s, late-outside | no lead, deflect; swerve, no lead, hit crossing | oncoming / cross | H1-wide, H3 |
| bd70654a | collision | 14 | turn L +71 | +1.7 | ds -0.7 m, delta -1.15 s, early-inside | no lead, deflect; swerve, no lead, hit oncoming | oncoming / onc | H1-cmd, H3 |
| 84b38cda | collision | 14 | straight -6 | -0.0 | onset k14, cmd yes, toward drift | no lead, deflect; swerve, no lead, hit crossing | - / same | H1-cmd |
| 47e0ff3b | collision | 16 | turn L +68 | +1.9 | ds -1.2 m, delta -2.12 s, early-inside | no lead, deflect; swerve, no lead, hit oncoming | oncoming / onc | H1-cmd, H3 |
| 7bbb6cfc | collision | 17 | straight -1 | +0.3 | onset k17, cmd yes (flip), away from drift | no lead, deflect; swerve, no lead, hit adjacent | - / same | H1-cmd |
| b24575ef | collision | 17 | straight +0 | -1.3 | onset k16, cmd yes (flip), away from drift | lead, deflect; cut-around, hit - | - / same | H1-cmd, H2 |
| bf00c76f | collision | 17 | straight -6 | +2.7 | onset k9, cmd yes, away from drift | no lead, deflect; swerve, no lead, hit oncoming | oncoming / onc | H1-cmd, H3 |
| aec02ec2 | corridor | 8 | turn L +77 | -3.6 | never half-turned, late-outside | no lead, deflect | - / same | H1-wide |
| e14711ef | corridor | 8 | turn L +104 | -3.9 | ds +4.8 m, delta +1.26 s, late-outside | no lead, deflect | - / same | H1-wide |
| e6056b57 | corridor | 9 | turn L +66 | +4.1 | ds -13.7 m, delta -3.15 s, early-inside | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| 8eefb96f | corridor | 10 | bend R -18 | +3.8 | onset k5, cmd yes (from start), away from drift | no lead, deflect | - / same | H1-cmd |
| a3882e6a | corridor | 10 | turn L +87 | +4.3 | ds -6.1 m, delta -2.00 s, early-inside | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| 47da19c4 | corridor | 11 | turn R -90 | +4.2 | ds +2.8 m, delta -0.04 s, late-outside | no lead, deflect | - / same | H1-wide |
| d7b1349f | corridor | 11 | turn L +65 | -4.5 | ds +6.6 m, delta +0.29 s, late-outside | no lead, deflect | - / same | H1-wide |
| 02b68b9c | corridor | 11 | bend R -28 | -3.0 | onset k10, cmd yes (flip), away from drift | no lead, deflect | - / same | H1-cmd |
| a3dd6d53 | corridor | 13 | turn L +53 | +3.7 | ds -5.0 m, delta -0.58 s, early-inside | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| b628faa4 | corridor | 13 | turn L +58 | -4.0 | never half-turned, late-outside | no lead, deflect | - / same | H1-wide |
| ded4ce0e | corridor | 14 | straight -0 | +3.6 | onset k10, cmd yes, toward drift | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| 4faa4706 | corridor | 14 | turn L +80 | -4.0 | ds +3.3 m, delta +1.39 s, late-outside | no lead, deflect | - / same | H1-wide |
| 94652fa0 | corridor | 15 | straight -11 | +4.4 | onset k7, cmd yes (flip), away from drift | no lead, deflect | - / same | H1-cmd |
| 5ddfd1fe | corridor | 15 | straight +0 | +2.4 | onset k14, cmd yes (from start), toward drift | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| f42258b6 | corridor | 16 | straight +4 | +2.6 | onset k13, cmd yes (from start), toward drift | lead, deflect | - / same | H1-cmd, H2 |
| b7c35b12 | corridor | 16 | straight -5 | +0.5 | onset k16, cmd yes, toward drift | no lead, deflect | - / same | H1-cmd |
| 330166de | corridor | 17 | turn L +51 | +2.9 | ds -0.8 m, delta -0.22 s, early-inside | lead, deflect | - / same | H1-cmd, H2 |
| f3512ae1 | corridor | 17 | turn L +72 | +3.6 | ds -5.0 m, delta -0.13 s, early-inside | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| 0fa603b5 | corridor | 17 | turn L +80 | -3.8 | ds +3.1 m, delta +0.74 s, late-outside | no lead, deflect | oncoming / onc | H1-wide, H3 |
| f1b802f6 | offroad | 0 | turn L +73 | -0.0 | ds +2.7 m, delta +0.95 s, late-outside | no lead | - / same | H1-wide, start |
| da6d3b68 | offroad | 0 | bend R -38 | +0.0 | onset k0, cmd yes (from start), away from drift | no lead | - / same | H1-cmd, start |
| 64a7186a | offroad | 0 | straight +7 | +0.0 | onset k0, cmd no | no lead | - / same | start |
| 9a377868 | offroad | 0 | bend L +29 | +0.0 | onset k0, cmd yes (from start), toward drift | no lead | - / same | H1-cmd, start |
| be6f90f0 | offroad | 5 | bend L +36 | +0.2 | onset k5, cmd yes (from start), toward drift | no lead, deflect | - / same | H1-cmd |
| 02015675 | offroad | 7 | turn R -95 | -0.9 | ds -1.6 m, delta -0.45 s, early-inside | no lead, deflect | - / same | H1-cmd |
| ddc1271e | offroad | 7 | straight +1 | -0.6 | onset k7, cmd yes (from start), toward drift | no lead, deflect | - / same | H1-cmd |
| f9ed38d9 | offroad | 7 | straight +8 | +0.0 | onset k7, cmd yes, away from drift | no lead, deflect | - / same | H1-cmd |
| 078bc102 | offroad | 9 | turn R -128 | +0.5 | ds -0.6 m, delta -0.48 s, early-outside | no lead, deflect | - / same | none (drift) |
| 8c9e3fd9 | offroad | 10 | turn R -105 | -0.6 | ds -0.6 m, delta +0.40 s, early-inside | no lead, deflect | - / same | H1-cmd |
| fde7abb6 | offroad | 11 | turn L +62 | +2.8 | ds -4.8 m, delta -2.17 s, early-inside | no lead, deflect | oncoming / onc | H1-cmd, H3 |
| 53f8bc40 | offroad | 12 | turn L +88 | +3.6 | ds -4.7 m, delta -0.89 s, early-inside | lead, deflect | - / same | H1-cmd, H2 |
| c687799d | offroad | 14 | turn L +70 | +1.8 | ds -1.1 m, delta -2.38 s, early-inside | no lead, deflect | oncoming / same | H1-cmd, H3 |
| 152679f2 | offroad | 16 | straight -1 | +0.2 | onset k16, cmd yes, toward drift | no lead | - / same | H1-cmd |
| 667ea4f7 | offroad | 17 | bend R -41 | +1.1 | onset k17, cmd yes (from start), away from drift | no lead, deflect | - / same | H1-cmd |

Fit counts: H1-cmd 31, H1-wide 9, H2 5, H3 13, start 4, none 2 (an event can fit several).

## Exploratory (after the registered read; not in the verdicts)

1. Events after the ego has passed the end of the human's logged path (its station beyond the log's own station at t_17); the evaluator then measures the corridor against the straight extension of the log's last heading:

| group | straight-scene corridor zeros past the log's end | turn-scene corridor zeros | all zeros |
|:--|--:|--:|--:|
| base | 2 / 7 | 2 / 23 | 8 / 74 |
| YS | 9 / 12 | 4 / 25 | 17 / 88 |
| YS2G | 9 / 12 | 4 / 26 | 16 / 83 |
| YNONE | 11 / 35 | 4 / 40 | 19 / 240 |

2. The plan's own lateral motion y_h(3) (ego heading frame) on decisions where the log turns < 10 deg in the next 4 s (curvature small), lead vs no-lead: d(3) is measured against the logged path, which itself moves sideways when the human passes a slow lead or bears off (8051c60f: the plan goes straight, |y_h(3)| <= 0.1 m, while the log bears right, d(3) 1.1-2.5 m):

| condition | decisions YS / base | YS mean |y_h(3)| | base | YS - base |y_h(3)| | YS - base s_onc y_h(3) |
|:--|--:|--:|--:|--:|--:|
| lead | 176 / 90 | 0.153 [0.100, 0.215] | 0.149 [0.101, 0.204] | +0.004 [-0.061, +0.054] | +0.055 [-0.049, +0.141] |
| no-lead | 5401 / 2666 | 0.555 [0.470, 0.638] | 0.568 [0.482, 0.652] | -0.013 [-0.048, +0.017] | -0.008 [-0.040, +0.027] |

3. YS's registered cut-around collisions: max |d(3)|, max |y_h(3)| (the plan's own sideways motion; large on turns by curvature) and max |y_log(3) - y_h(3)| (how far the logged path is from the ego's heading line at the plan's 3 s point) in the window:

| label | scene | scene class | k_c | max |d(3)| | max |y_h(3)| | max |y_log - y_h| | object hit |
|:--|--:|--:|--:|--:|--:|--:|--:|
| YS-L18-s0 | 8051c60f | straight | 12 | 1.77 | 0.06 | 3.30 | other |
| YS-L18-s1 | 8051c60f | straight | 9 | 2.46 | 0.19 | 3.45 | adjacent |
| YS-L18-s0 | b49e6399 | straight | 9 | 1.79 | 2.93 | 4.12 | other |
| YS-L18-s1 | b24575ef | straight | 17 | 1.74 | 2.98 | 4.69 | - |
| YS-L18-s0 | 47e0ff3b | turn | 16 | 1.45 | 9.16 | 7.44 | oncoming |

4. Row side, the 739 binding objects classed `other`: 0.32 at |y| >= 3 m at t0 (reached by a curving plan), 0.83 not ahead of the ego box centre, 0.12 at 30-60 deg.
5. Oncoming decisions inside a junction overlap (two containing lanes > 30 deg apart): base 0.18, YS 0.14, YS2G 0.18, YNONE 0.19.
6. Turn zeros by turn side (across = turning across the oncoming lanes: left under right-hand traffic, right in Singapore, 113 of the 452 scenes; kerb = the other way); early-inside / late-outside counts, corridor zeros and all zero types on turn scenes, and ds over all turn rollouts:

| group | corridor, across: EI / LO | corridor, kerb: EI / LO | all zeros, across: EI / LO | all zeros, kerb: EI / LO | ds median [IQR], across (m) | ds, kerb (m) |
|:--|--:|--:|--:|--:|--:|--:|
| base | 5 / 8 | 0 / 10 | 6 / 19 | 2 / 17 | +0.04 [-1.22, +1.61] | +1.78 [+0.77, +4.59] |
| YS | 10 / 3 | 0 / 11 | 21 / 8 | 1 / 12 | -0.31 [-1.69, +1.15] | +1.21 [+0.30, +3.31] |
| YS2G | 6 / 8 | 0 / 12 | 11 / 17 | 2 / 15 | -0.22 [-1.43, +1.25] | +1.08 [+0.14, +3.05] |
| YNONE | 4 / 18 | 0 / 18 | 9 / 39 | 4 / 27 | +0.09 [-1.11, +2.06] | +1.71 [+0.56, +4.19] |

7. Share of corridor + collision zeros with oncoming occupancy, difference to base (event-level, log-clustered): YS - base +0.13 [-0.02, +0.31]; YS2G - base +0.06 [-0.08, +0.18]; YNONE - base +0.00 [-0.10, +0.09].
8. Corridor + collision zeros with oncoming occupancy per seed (counts, not shares): base 7.0 of 41.0, YS 9.0 of 29.5, YS2G 6.0 of 26.5, YNONE 11.0 of 64.0, YR-s1 7.0 of 34.0; the decision-level rise against base is shared by the no-row continuation (YNONE - base above).

## Verdicts by the registered lines

| hypothesis | verdict (registered line) | what qualifies it (numbers above) | arm (prereg section 8) |
|:--|--:|--:|--:|
| H1-command | partial: straight-road part only ((b) fails) | (a) is met in every group (base 7 / 7); YS's 12 events come from 7 scenes; 9 happen after the ego passed the end of the logged path, the route turns within 60 m in 9 (both in 7) | C (route waypoints into the adapter): straight-road half only |
| H1-smooth | not supported; not the S lesson | S makes turns sharper, not smoother: yaw-rate shortfall YS - base -0.108 [-0.142, -0.070]; plans of every label turn about 0.86 of the log's next 4 s | - |
| H2 | not established | plan's own |y_h(3)| under a closing lead on straight road: YS - base +0.004 [-0.061, +0.054] m; no collision hits the lead; in-lane leads bind 12% of hinge-active rows | A (in-lane same-direction objects out of the lateral agent hinge): no |
| H3 | gets an arm; lesson made it worse | 'worse' is marginal (CI lower +0.0001) and shared by YNONE (+0.0051 [+0.0004, +0.0111]); per seed the oncoming zeros stay (7 base vs 9.0 YS) while other zeros fall | B (direction-aware road raster): yes |

