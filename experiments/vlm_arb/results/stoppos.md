# Can frozen openpilot (Cinque) tell the junction entrance and the stop line?

Offline probe, no CARLA run, no closed loop, no driving-code change. Plan (pre-registered, committed before any feature existed): [../plans/2026-10-03-stoppos-probe.md](../plans/2026-10-03-stoppos-probe.md). Code: `../scripts/stoppos_*.py` (chain: `stoppos_chain.py`). All tables and per-readout CSVs: [stoppos/](stoppos/) (`tables.md` holds every table in full; `stoppos/hold4/` is the sensitivity run). Split: `b2d/stoppos-cv5-fold{0..4}` (route-grouped, registered).

## Verdict (question 5)

- **Not good enough to replace the map for the stop position.** Linear / MLP probes on all four taps put the bumper-to-stop-line distance within a route-macro MAE of 2.4 to 3.5 m in 0-10 m (CL truth: 2.4 to 4.0 m; pooled 0-5 m best 2.6 m, hidden tap) and 3.5 to 5.5 m beyond 20 m, with strong shrinkage to the mean (bias -3 to -5 m at 20-40 m). The preregistered bar was median |error| <= 1.0 m. The calibrated 80% interval is about 12 m wide.
- **Not a refinement on top of the route's junction point either (as far as this data can show).** On the same frames the route command point gives the entrance to 0.3 m (P4) / 0.9 m (CL) and, with a constant offset, the stop line to 1.2 m (0-5 m), 2.5 m (5-10 m) on the closed-loop truth, versus 2.4-4.0 m for the probes. A leave-one-route-out residual probe on top of the route estimate was worse than the route estimate in every bin (exploratory, 14 routes).
- **What is readable:** whether a stop line / light / sign is within about 40 m. `stop40` AUC 0.915 to 0.950 (speed/time/odometer 0.84), junction has a light 0.91 to 0.94 (clock 0.80), junction has a stop sign 0.96 to 0.98 (clock 0.84); AUC gains over the clock baseline 0.075 to 0.144, route-bootstrap intervals exclude zero. The 4-class action head (free / slow / creep / at the line) is 0.87 to 0.90 accurate (majority 0.75, clock 0.81, route-only 0.66) and calibrated (ECE 0.01 to 0.06). So it is a detector of "a stop target is near", not a ruler.
- **Native outputs:** no stop-line, light or junction head exists. The plan slows toward every junction (also without a light); the brake-press head separates red/yellow from green approaches at the same distance (AUC 0.77 to 0.88 without a detected lead) but carries no usable distance.
- **What would be needed:** keep the route point as the distance anchor (entrance to about 1 m), and replace the metric head by (a) the detector heads above and (b) a stop-line-offset estimate. The offset (command point minus stop line) is 2.5 to 3.4 m on 9 of 12 logged junctions (Town12, 15, 11) but 5.4, 5.5 and 10.2 m on three older-town junctions: that spread is what a probe or detector must resolve and 12 junctions cannot show it. Data alone will not fix the frozen metric head: the curve below flattens at about 2.6 m from 80 routes. Candidate fixes, not tested here: light adaptation of the vision encoder with in-domain approach labels (op_adapt_l route), or a painted-stop-line detector anchored on the route. A small collection run would need about 150 light-approach episodes over at least four town styles with the stop waypoint logged each tick (map query for labels only) and the 5 Hz road / wide frames saved (about 75 min of simulation, no new driving code).
- A probe fitted on CARLA frames is in-domain supervised; everything below is CARLA, supervised, route-held-out.

## Data (counted first)

| | CL (closed-loop logs, 2 Hz road + wide frames) | P4 (5 Hz Waymo-rig frames) |
|:--|--:|--:|
| attempts / routes / frames | 214 / 21 / 33 806 | 152 / 151 (one id twice) / 19 331 |
| towns | 8 | 12 |
| routes with a junction | 15 (151 junction approaches) | 111 |
| routes with a light and a known stop line (light approaches) | 14 (141) | 38 (38) |
| light, line unknown (towns without painted lines) | 0 | 15 |
| stop-sign routes / junction with neither | 0 / 1 | 32 / 26 |
| light-approach routes per bin 0-5 / 5-10 / 10-20 / 20-40 m | 14 / 6 / 6 / 3 | 35 / 31 / 30 / 28 |

52 light approaches with a stop line in total (14 + 38). Enough for a route-grouped CV, thin for the CL bins beyond 5 m (6, 6, 3 routes). Full counts: `stoppos/data_counts.csv`.

Labels (targets only): CL from the agent logs (`junc_dist`, `tl_dist` = CARLA stop waypoint); P4 from route + pose + the OpenDRIVE file (no simulator). Validation on the 13 closed-loop junctions (`stoppos_label_validation.csv`): entrance 8 of 13 within 1 m, 12 of 13 within 2.5 m (mean 0.95 m; the plan text says "8 within 0.8 m", the exact count is 5 within 0.8 and 8 within 1.0); light presence 13 of 13; painted line minus CARLA stop waypoint 3.0 m, sd 0.3 (n = 8). Consequence the plan did not foresee: **on P4 the stop line is the entrance minus 3.0 m by construction, so the route baseline is exact there (circular) and P4 says nothing about route-vs-probe for the stop line; the CL truth does.** On the P4 frames the self-consistency check (first frame whose bumper lies in a junction, `d_junc` there) gave -0.23 +- 0.74 m (n = 95).

## 1. What openpilot outputs natively

Output slices of `cinque.onnx` (ONNX metadata): lane lines (4 x 33 x 2) and probabilities, road edges, meta (engaged, disengage / hard-brake 3-5 / gas / brake-press / blinker probabilities at 2-10 s), desire prediction, pose, road transform, plan (33 x 15, mean and std), lead (3 x 6 x 4) and lead probability, desire state, action, hidden state. **No stop-line, traffic-light or junction head.**

Measured on the 20 Hz logs of the `drive` units (220 attempts, 28 light routes, 11 junction routes without a light, v >= 3 m/s; [stoppos/stoppos_native.png](stoppos/stoppos_native.png)):

- Plan speed at 3 s over current speed rises with distance, within-route Spearman +0.60 [0.43, 0.75] on red/yellow, +0.74 [0.58, 0.87] on green, +0.63 [0.38, 0.86] at junctions without a light: the plan slows toward every junction, not toward a line, and the binned medians are not monotone (red: 0.76, 0.54, 0.93, 1.05, 0.81, 0.80 for 0-5 ... 30-40 m; dip at 5-10 m). Fails the registered monotonic-and-above-spread criterion. Brake-press probability: rho -0.35 / -0.31 / -0.43, again light-agnostic.
- Light state is partly there: red/yellow vs green at the same distance, AUC of the brake-press probability 0.71 / 0.78 / 0.83 / 0.95 (0-5 ... 20-40 m); with frames where no lead is detected 0.82 / 0.77 / 0.84 / 0.87 (routes per bin 7 / 5 / 9 / 1 red; ego speed alone 0.37 to 0.59). Inferred: the head reflects human braking before red lights; the 20-40 m bin has one red and three green routes.

## 2. Probe error by true distance

Route-macro MAE in m with a route-bootstrap 95% interval and signed bias in parentheses (prediction minus truth); out-of-fold, every route predicted by a model that never saw it. Figure: [stoppos/stoppos_mae_by_bin.png](stoppos/stoppos_mae_by_bin.png). Look at: the probe curves are flat and near the speed/time/odometer curve; the route curve is an order of magnitude lower.

Stop line, CL truth (14 routes):

| input | 0-5 m | 5-10 m | 10-20 m | 20-40 m |
|:--|:--|:--|:--|:--|
| temporal, linear | 4.03 [2.27, 6.61] (+3.05) | 5.48 [3.11, 9.16] (+0.53) | 6.83 [4.81, 8.32] (-6.22) | 15.4 [9.7, 25.9] (-15.4) |
| hidden, linear | 3.45 [2.33, 4.70] (+2.84) | 2.35 [1.57, 3.32] (-1.51) | 6.97 [5.90, 8.42] (-6.97) | 18.2 [11.0, 27.1] (-18.2) |
| hidden, MLP | 1.79 [1.27, 2.45] (+0.84) | 3.62 [2.38, 4.98] (-3.38) | 9.98 [8.51, 11.28] (-9.98) | 21.3 [17.4, 28.1] (-21.3) |
| native, linear | 3.87 [2.58, 5.43] (+1.44) | 2.77 [1.51, 4.09] (-1.56) | 6.42 [4.67, 8.08] (-6.19) | 14.6 [7.6, 27.5] (-14.6) |
| route command - offset | 1.24 [0.49, 2.26] (+0.88) | 2.48 [0.68, 5.10] (+0.94) | 2.99 [0.65, 6.71] (+0.13) | 7.2 [0.1, 19.0] (-3.5) |
| speed / time / odometer (GBM) | 4.26 [3.73, 4.79] (+2.89) | 5.36 [3.82, 7.31] (-2.02) | 10.2 [7.7, 13.0] (-10.0) | 21.8 [16.1, 31.3] (-21.8) |

Stop line, pooled (CL + P4, 52 routes; P4 label circular for the route row, so omitted): hidden linear 2.58 / 2.37 / 3.45 / 5.56; temporal linear 3.41 / 2.92 / 3.24 / 4.13; hidden MLP 1.53 / 2.81 / 4.95 / 7.55; clock GBM 2.95 / 2.90 / 3.05 / 5.22 (0-5 ... 20-40 m). Paired against the clock GBM (0-10 m): hidden -0.58 [-1.52, +0.22], temporal -0.13 [-1.07, +0.75]: **not distinguishable from reading speed / time / odometer**; hidden + clock features -0.82 [-1.74, -0.04] (marginal). Against the route baseline the probe is worse by 2.3 [1.8, 2.7] m (hidden, 0-10 m; CL truth only in `tables.md`).

Junction entrance, all routes: hidden linear 3.46 / 3.24 / 3.09 / 4.71; temporal MLP 3.05 / 2.66 / 2.62 / 3.64; route command 0.31 / 0.32 / 0.28 / 0.28; clock GBM 2.92 / 3.95 / 2.82 / 5.14. The probe beats the clock only at 10-40 m (temporal -0.80 [-1.31, -0.29]).

Classifiers (out-of-fold AUC, route-bootstrap interval; frames / routes in `stoppos/clf.csv`): `stop40` hidden 0.950 [0.924, 0.976], temporal 0.915 [0.891, 0.942], native 0.925; `light` 0.91 to 0.94 (vision 0.944 [0.911, 0.970]); `sign` 0.96 to 0.98 (hidden 0.984 [0.967, 0.994]); route command score for `stop40` 0.72, clock GBM 0.84 / 0.80 / 0.84. `light` / `sign` on CL alone are degenerate (14 of 15 junction routes have a light) and not interpretable; P4 carries them (0.91 to 0.97).

Domain transfer (`transfer_*.csv`): P4 -> CL stop-line MAE 2.4 / 4.7 / 10.5 m (temporal, 0-5 / 5-10 / 10-20), `stop40` AUC 0.84 (temporal), 0.86 (native); CL -> P4 fails (AUC 0.5 to 0.6; 21 training routes, different cameras).

## 3. A head that speaks the stack's language

Primary tap by the registered rule (lowest pooled MAE of the stop line in 0-10 m): `hidden`. Figure: [stoppos/stoppos_calibration_replay.png](stoppos/stoppos_calibration_replay.png).

**(a) 4 classes** (free > 20 m or no line / slow 6-20 / creep 1.5-6 / at the line <= 1.5 m): accuracy 0.900 (hidden), route-weighted majority 0.751, clock GBM 0.815, route command rule 0.662. ECE (top label) uncalibrated 0.037, temperature-scaled 0.062 (hidden); other taps 0.012 to 0.055: temperature scaling on a 25% route split did not help reliably (it helps vision / native, hurts temporal / hidden). Row-normalised confusion (temperature-scaled): free 0.985 correct; slow 0.47 (0.51 called free); creep 0.59 (0.18 called at the line, 0.18 free); at the line 0.79 (0.18 called free). So about one in five at-the-line frames carries no "stop" signal.

**(b) Interval** (linear quantile heads, nominal 80%): raw 10-90% covers 0.37 [0.31, 0.44] (hidden; 0.35 to 0.48 per bin): overconfident. The conformalised 80% interval covers 0.88 [0.83, 0.93] overall (0.92, 0.92, 0.84, 0.76 by bin; CL alone 0.79, and 0.05 in its 20-40 m bin, 3 routes) at a mean **width of 12.5 m**: coverage is bought with an interval far too wide to place a stop.

**(c) Stopping rules replayed** on held-out light approaches along the logged trajectories (open loop; 179 approaches, 52 routes; implied stop point x = true distance at the firing frame - v^2 / (2 x 3 m/s^2), x < 0 is past the line). Registered plain rule U(m) (fire when the upper bound <= m, gated by `stop40` > 0.5) at v = 7 m/s cannot brake in time (past the line in 81 to 96% of firings), so I added the speed-aware form (threshold m + v^2/(2a); post-hoc). With m = 4 m, all approaches / CL approaches starting >= 12 m before the line (57, 6 routes) / P4 (38):

| rule | fired | x, line: median [p10, p90] | past the line | notes |
|:--|--:|:--|--:|:--|
| probe upper bound U'(4) | 0.32 / 0.18 / 0.71 | -2.1 [-5.4, +1.1] (all) | 69% | never fires in CL when it matters, overshoots when it does |
| probe lower bound L'(4) | 0.91 / 0.74 / 0.97 | CL start >= 12 m: +14.9 [+8.6, +17.4]; P4 +6.1 [+3.2, +13.0] | 1 to 5% | never overshoots, stops 6 to 15 m short |
| route command - offset R'(4) | 1.00 / 1.00 / 1.00 | CL start >= 12 m: +2.8 [+0.3, +46.6]; P4 +3.3 [+2.5, +3.9] (circular) | 7% (CL) | within the route's own +-1 to 3 m |

Stop relative to the junction entrance (x_entr, median): U' +2.2 m, route R' +6.3 m. False fires on approaches without a light line within 40 m (4 m rule): L' 22 to 30%, U' 0% (CL) to 13% (P4 no-light junctions), 0 to 3% on stop-sign junctions. Full tables: `stoppos/replay_summary.csv`, `replay_false_fire_rates.csv`.

## 4. How much data

Hidden tap, linear + logistic, route-grouped folds, random training subsets (`stoppos/scaling_hidden.csv`):

| training routes (light routes) | MAE stop 0-10 m | MAE stop 10-40 m | `stop40` AUC |
|--:|--:|--:|--:|
| 10 (3) | 4.52 +- 2.20 | 9.54 | 0.72 |
| 20 (6) | 3.94 +- 1.66 | 8.29 | 0.81 |
| 40 (12) | 3.39 +- 1.19 | 6.66 | 0.88 |
| 80 (24) | 2.64 +- 0.56 | 5.56 | 0.93 |
| all, about 138 (42) | 2.63 +- 0.56 | 4.57 | 0.96 |

Detection saturates at about 40 to 80 routes (12 to 24 light approaches), the metric error stops improving at 80 routes at about 2.6 m. This matches decision 80's order of magnitude (about 100 episodes per class) for the detector, not for the ruler.

## Deviations from the plan and caveats

- **CL replay.** Registered: 2 Hz frames held 10 steps. Measured fidelity against the 20 Hz heads logged at the same time ([stoppos/fidelity.csv](stoppos/fidelity.csv)): plan speed correlation 0.60 (bias -0.8 to -1.6 m/s), brake-press 0.24, lane / lead probabilities 0.86 to 0.91. Holding each frame 4 steps (read as if 5 Hz; time-warped, speed biased +1.0 to +1.6 m/s) matches much better (speed 0.93, brake-press 0.87, lane / lead 0.95 / 0.91). Chosen on fidelity only, before any probe number, I ran the whole pipeline a second time on it (`stoppos/hold4/`). Conclusions are the same: CL stop-line MAE 2.4 to 3.5 m in 5-10 m / 0-5 m (temporal, hidden), route 1.2 / 2.5 m; `stop40` AUC on CL 0.86 to 0.91; primary tap hidden again. All numbers above are the registered hold-10 run unless said otherwise.
- Training weights (not in the plan): every route sums to 1, attempts of a route share equally (the evaluation weighting).
- Speed-aware rule variants, the red-vs-green native readout, the no-lead control and the residual refinement (`refine_cl.csv`) are post-hoc / exploratory; the primary tap is chosen on the same out-of-fold predictions it is reported on (mildly optimistic).
- The probe does not beat reading speed / time / odometer for metric distance; it does for presence classification. P4 runs start 27 to 29 m before the entrance, so time and odometer are strong clocks there.
- CL attempts 214 (plan: 224; 10 lost for missing request records), P4 routes 151 (one id has two attempts).

## Verified vs inferred

Verified (computed from logs and replayed features): all tables above, the fidelity numbers, label validation on 13 junctions, the native-head measurements. Inferred: that the brake-press head reflects human braking at red lights; that the P4 stop line is accurate to about 0.3 m in Town13 (assumed from Town12 / 15); that a painted-line detector or adapted encoder would close the gap; the size of the collection run. Not tested: closed-loop behaviour, 20 Hz frames, other models (Lebowski, V-JEPA), junctions in old town styles beyond the 4 logged.
