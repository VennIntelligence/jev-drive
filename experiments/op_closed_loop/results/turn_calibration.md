# Can openpilot turn through junctions by itself? Turn-magnitude calibration of the action head and the plan (candidate per-board trick, not adopted)

Written 2026-10-04. CPU-only analysis of existing logs, no new runs. Follows decision 118 point 5 (action curvature has the right sign at junctions but ~0.5 of the needed magnitude).
Code: `experiments/op_closed_loop/scripts/turn_calibration_{lib,b2d_extract,openloop_extract,sparse,report}.py`. Full tables (all splits, CIs): [turn_calibration_tables.md](turn_calibration_tables.md). Intermediate arrays on the box: `$DATA_DIR/runs/op_closed_loop/turn_calibration/`.
Figure: `../figs/turn_calibration.png` (A ratio vs required curvature per source with CIs; B ratio vs speed; C pure pursuit through the route turns on sparse routes).

![turn calibration](../figs/turn_calibration.png)

What to look at: in A the blue action-head curves sit at ~0.5 from R 80 m down to R 4 m (flat), orange / green plan-derived curves sit at 1.2-1.5; in C the leaderboard-downsampled route cuts the corner by ~3 m.

## Method (short)

- Statistic: ratio = sum(model_k * sign(req_k)) / sum(|req_k|) per bin (wrong-sign ticks count negative), 95% cluster-bootstrap CI (B2D: route; WOD: scene; NAVSIM: log). Curvature right-positive.
- B2D: 1 293 runs, 552 k non-warm ticks with v >= 1 m/s. Required = chord curvature of the dense route over max(4, min(v, 20)) m from the ego's projection; model = logged `act_k` (action head, regardless of who steered: in junction zones the route steered, so this is the counterfactual of decision 118) and the plan-derived chord curvature at the 1 s plan point (`op_xy`). Primary set = drive / opc / lsc, seeds 2-3 (19 routes); held-out = the 50 other routes of the other v2 arms.
- WOD (479 rater frames, real 10 Hz history) and NAVSIM navtest (12 146 tokens, 2 Hz history interpolated, plus the native 2 Hz exam protocol): plan chord curvature at 10 m arc vs the logged human future's. The open-loop files carry no action head (not saved; a replay would need a GPU card), so open loop is plan-derived only.
- B2D routes contain only ~90 degree junction turns (R_min 4-15 m) and almost no bends; gentle turns / bends come from NAVSIM / WOD.

## Results

1. **Action head, B2D: constant gain ~0.5, not a sharpness effect.** Ratio 0.45 [0.24, 0.93] / 0.48 [0.35, 0.60] / 0.63 [0.49, 0.74] / 0.51 [0.43, 0.60] for |k| 0.012-0.025 / 0.025-0.05 / 0.05-0.1 / 0.1-0.25 (R 83 -> 4 m). Held-out routes: 0.54 / 0.27 / 0.55 / 0.54. Slope of the binned ratio on ln|k|: -0.06 [-0.22, +0.12] (primary), -0.07 [-0.15, +0.14] (held-out): flat, so the response is linear with gain ~0.5, not "the sharper the more it under-turns". Over a whole junction turn the integrated heading change of the action head is 0.61 [0.54, 0.67] of the route's turn angle (874 run-turns, 36 routes); the plan-derived curvature gives 1.18 [1.07, 1.29].
   Below |k| 0.012 (R > 83 m) the ratio is noise: same-sign only 64-80%; the head there carries lane-centering corrections, not path curvature. Sign agreement is 91-95% for R < 20 m.
2. **Speed**: ratio 0.37 [0.24, 0.55] at 1-2.5 m/s, 0.55 [0.41, 0.68] at 2.5-4, 0.62 [0.42, 0.78] at 4-6, 0.63 [0.26, 0.94] at 6-9 (turn ticks). Joint log fit: curvature exponent b +0.07 [-0.07, +0.22], speed exponent c +0.26 [-0.13, +0.55]: speed has at most a weak effect (the slow creep inside a turn is under-turned more). The plan-derived curvature shows the opposite speed trend (1.53 at 1-2.5 m/s to 0.90 at 6-9; c -0.64 [-0.79, -0.43]).
3. **Left vs right**: turn ticks 0.64 [0.52, 0.79] left vs 0.39 [0.21, 0.51] right, but 0.34 vs 0.75 in the 0.012-0.04 band and 8 routes per side: no consistent asymmetry. Arms identical within CI (drive 0.56, opc 0.52, lsc 0.46). Radius: 6-9 m 0.53 [0.46, 0.60] (9 routes), 9-13 m 0.78 [0.66, 0.94] (4 routes), the others 1 route.
4. **Plan-derived curvature does not under-turn**: B2D 1.2-1.35 for |k| >= 0.012 (primary and held-out), NAVSIM interpolated 1.4-1.5 (R > 20 m) falling to 1.23 [1.16, 1.29] at R < 10 m (slope -0.085 [-0.113, -0.055] per ln|k|: a small sharpness effect, in the direction of the intuition but the ratio stays above 1), NAVSIM native 2 Hz protocol 1.9 falling to 1.2 (slope -0.195 [-0.25, -0.14]; input protocol, decision 36), WOD 0.9-1.0 (n 7-51 per bin, no firm reading; 0.77 [0.52, 0.98] at a = 10 m, 1.05-1.08 at 15-25 m). So the plan is the right magnitude (over by 20-40% on two boards) and the action head is the part that is half-size, consistent with decision 118 ("action 幅度是计划推曲率的 1/2-1/5").
5. **Desire**: not separable. Route-turn desire is on in 80% of turn ticks; desire = 0 turn ticks come from 3 routes: 0.57 [0.10, 1.01] vs 0.50 [0.39, 0.61] with desire. No image command in these logs (0 ticks). Consistent with decision 92 (desire does not steer), but this log cannot test it.

## Dense vs sparse route (kinematic pure pursuit, 38 route turns >= 25 degrees of 36 routes, start 25 m before, constant speed 3 / 5 / 8 m/s, Ld = max(3, v + 1.5) m)

| path given | peak cross-track vs dense centreline at 5 m/s, median [p90] | > 1.75 m (leaves lane) | heading delivered / route's |
|---|---|---|---|
| dense route (shipped zones) | 0.52 [0.66] m | 0% | 1.02 |
| road polyline, 5 m decimation, no noise | 0.65 [0.89] m | 0% | 1.02 |
| road polyline, 10 m decimation, no noise | 1.24 [1.70] m | 5% | 1.02 |
| 10 m polyline + 1 m / 2 m noise | 1.67 / 2.59 m | 45% / 79% | 1.02 |
| leaderboard downsample (50 m, junction entry / exit points only) | 3.06 [4.90] m | 97% | 1.03 |

The sparse route still gets the heading change right (the target after the turn lies along the exit direction), so the commanded turn magnitude is not the problem (peak |k| lb50 / dense 0.73-0.93 for R < 13 m); the path is: the polyline is a chord across the junction and the car cuts the inside corner by ~3 m (left turns 3.3 m, right 1.9 m, larger radius worse). Noise variants are dominated by the injected noise itself (error passes through ~1:1), not by the junction. Reading: "steer by sparse route" is not a viable non-privileged replacement for the dense-route zones with the leaderboard's 50 m route or a noisy road polyline; it is viable only with a map-grade centreline (5 m decimation, sub-metre accuracy), i.e. with road geometry that a real navigation stack does not give; the alternative is to let perception supply the geometry. Limits: ideal pose on the centreline at the start, no steering dynamics, speeds fixed.

## Candidate calibration map (per-board trick, labelled, not adopted)

Action head, fitted on the primary 19 routes, checked on the 50 held-out routes (turn ticks |k_req| >= 0.012; ratio of calibrated command to required):

| map | fit set | held-out | held-out by band 0.012-0.04 / 0.04-0.1 / >= 0.1 | held-out RMSE (1/m) |
|---|---|---|---|---|
| identity | 0.51 | 0.51 [0.44, 0.59] | 0.39 / 0.53 / 0.54 | 0.076 |
| linear gain k / 0.51 | 1.00 | 1.01 [0.85, 1.16] | 0.77 / 1.03 / 1.06 | 0.105 |
| power law (exponent -0.07, i.e. ~linear) | 0.95 | 0.94 [0.82, 1.05] | 0.70 / 0.97 / 0.98 | 0.078 |
| speed-dependent gain (0.37 / 0.55 / 0.62 at 1.8 / 3.3 / 5.0 m/s) | 0.99 | 1.06 [0.89, 1.22] | 0.71 / 1.00 / 1.17 | 0.133 |

A single gain of ~1.9-2.0 on the action curvature restores the integrated turn magnitude on held-out routes (1.01 [0.85, 1.16]) and the power law / piecewise fits add nothing (the response is linear). The tick-level RMSE does not improve (it rises with the gain, since the head's noise is amplified), so the map fixes the bias in the turn, not the per-tick match; and the curvature is the head's own, so the gain must be applied at the junction only, not on lane-follow where the head carries small corrections. Plan-derived curvature: gain 1.39 fitted on half of the NAVSIM logs gives 1.02 [0.98, 1.06] on the other half and 0.90 [0.78, 1.03] / 0.90 [0.84, 0.98] on B2D primary / held-out, but 0.61 [0.47, 0.74] on WOD (n 65): it does not transfer to WOD.

## Verdict and limits

- Linear, not nonlinear: the action head's gain is ~0.5 from R 80 m to R 4 m (slope on ln|k| not different from 0), speed matters at most weakly. The user's intuition (sharper turn, more under-turn) is not seen in the action head; it is weakly visible in the plan-derived curvature on NAVSIM (1.5 to 1.2) which still over-turns.
- openpilot cannot turn a junction with the action head alone on B2D (about 0.5-0.6 of the heading); a fixed x2 gain would be the calibration, to be tested in closed loop (not done here).
- Limits: B2D CIs rest on 11-14 routes per bin (primary) and the junction ticks are the route-steered counterfactual (op-owned turn ticks: 98); required curvature is the route's, not what a good driver would drive (cutting corners would shrink the needed curvature); no action head in the open-loop files (WOD / NAVSIM are plan-derived only, WOD is small); the sparse-route result is kinematic.

## Draft decision paragraph (中文)

**转弯幅度标定（action 头在路口只出约一半曲率，是线性的增益，不是越急越欠转）**：2026-10-04，接第 118 条第 5 点。CPU 分析，没跑新实验，结果 [experiments/op_closed_loop/results/turn_calibration.md](../../experiments/op_closed_loop/results/turn_calibration.md)，图 `experiments/op_closed_loop/figs/turn_calibration.png`。1. B2D 1 293 个 run 约 55 万 tick，对照稠密路线的弦曲率：action 头与所需曲率之比在 R 83 → 4 m 各档为 0.45 / 0.48 / 0.63 / 0.51（19 条主路线），留出的 50 条路线 0.54 / 0.27 / 0.55 / 0.54；比值对 ln|κ| 的斜率 −0.06 [−0.22, +0.12]，平的，所以是线性增益约 0.5，没有「越急越欠转」。整个路口转弯的积分转角是路线转角的 0.61 [0.54, 0.67]。速度只有弱影响（1–2.5 m/s 0.37，4–6 m/s 0.62），左右、arm 之间无一致差异。R > 83 m 的低曲率段同号只有 64–80%，那里 action 头做的是车道内修正，比值没有意义。2. 计划推出的曲率不欠转：B2D 1.2–1.35，NAVSIM（插值协议）1.4–1.5，R < 10 m 降到 1.23（斜率 −0.085，方向同用户直觉，但始终大于 1），WOD 0.9–1.0（n 小）；减半的是 action 头，不是计划。3. 稀疏路线（leaderboard 50 m 降采样）上的纯跟踪：转角能转够（1.03），但弦线穿过路口，抄内角 3.1 m 中位数（97% 超出半个车道），带 1–2 m 噪声的 10 m 路网折线 1.7–2.6 m；只有地图级中心线（5 m 间隔、无噪声）0.65 m 与稠密路线 0.52 m 相当。结论：靠稀疏路线转弯不能替代稠密路线区，除非几何来自感知。4. 候选的逐榜单 trick（未采纳）：在路口对 action 曲率乘约 1.95（1/0.51），在留出路线上积分转弯比值 1.01 [0.85, 1.16]，幂律 / 分段不比线性好；逐 tick RMSE 反而升高。desire 影响在这些日志里分不出来（无 desire 的转弯 tick 只来自 3 条路线），也没有图像指令。**状态**：描述性，未闭环验证。限定：路口 tick 是路线接管下的反事实；B2D 只有约 90° 转弯；开环文件没存 action 头，WOD / NAVSIM 只有计划推曲率；稀疏路线是运动学仿真。**会推翻或推进本条的证据**：路口乘 1.95 增益的 B2D 闭环 DS / 路口碰撞；真车 rlog 里 action 与实际曲率的比值。
