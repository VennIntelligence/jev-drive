# Turn gain: P2's sharp-turn deficit is a mild proportional shrinkage (slope 0.94), not a cap, and the openpilot control chain does not clip

**Verdict: shrinkage (mild).** On navtest turns > 20 deg P2 predicts 0.94 [0.93, 0.95] of the logged heading change (P0 0.80, WA-JEPA 0.98), 0.92-0.93 above 45 deg, with no ceiling inside the predictions; in HUGSIM spec the lateral chain clips 0.0-0.2% of steps (never within 8 steps of a failure). The gain gap to WA-JEPA is only 3-4%, so a pure turning-gain fix cannot close the sharp-turn failure gap; closed-loop turn failures are not control-limited.

Written 2026-10-06. Script `../scripts/opj_turn_gain.py` (CPU, stored outputs only; `navtest` / `hugsim` collectors run on the box, `report` on the Mac). Figures and tables in [turn-gain/](turn-gain/) (`fig1`-`fig5`, PNG + PDF; `tables/*.csv`; raw per-token / per-step tables in `turn-gain/data/`, gitignored). Pure-Chinese page: [../../../research/turn-gain/index.html](../../../research/turn-gain/index.html).

## Setup

navtest (12 027 tokens that have the logged future and all plan sets; WA-JEPA lacks 119 of the 12 146). Plans: P0 shipped on W frames (`warp-cinque_PPP0`), P2-F s0/s1, P2H10-F s0/s1 (hinge), WA-JEPA (`done_union.pkl`). All are 8 poses at 0.5 s in the t0 rear-axle frame. Per plan: heading change at 2 / 4 s (unwrapped), lateral offset at 2 / 4 s, mean curvature over [0, T] (heading / arc length), peak segment curvature and peak lateral acceleration (segment heading rate x speed). Bins as `opj_build.py` / `opj_figs.py`: logged |heading change at 4 s| < 5 / 5-20 / 20-45 / > 45 deg, left and right by sign. Seeds are stacked (same log cluster). Gain = slope through the origin of predicted on logged (turn-direction folded), and ratio of sums; 95% cluster bootstrap over the 136 logs (B 1000).

## 1. Turn gain on navtest

Ratio of sums, predicted / logged heading change at 4 s ([fig1](turn-gain/fig1_gain_scatter.png), [fig2](turn-gain/fig2_gain_by_bin.png), `tables/gain.csv`):

| stratum | n | P0 (W) | P2 | P2 + hinge | WA-JEPA |
|:--|--:|--:|--:|--:|--:|
| 5-20 left / right | 1446 / 1146 | 1.00 / 1.02 | 1.02 / 1.01 | 1.03 / 1.01 | 0.98 / 1.07 |
| 20-45 left / right | 1057 / 580 | 0.91 / 0.88 | 0.99 / 1.00 | 0.99 / 1.00 | 0.98 / 1.05 |
| > 45 left / right | 918 / 599 | 0.79 / 0.75 | 0.93 / 0.94 | 0.93 / 0.94 | 0.96 / 0.99 |
| > 20 both, slope through origin [CI] | 3154 | 0.80 [0.76, 0.84] | 0.94 [0.93, 0.95] | 0.94 [0.93, 0.95] | 0.98 [0.97, 0.995] |

Other metrics, > 20 deg both sides, ratio of sums: lateral offset at 4 s P2 0.97, hinge 0.97, WA 0.99, P0 1.00; offset at 2 s 0.99 / 0.99 / 0.97 / 1.06; mean curvature over 4 s 0.95 / 0.95 / 0.97 / 0.85. The < 5 deg bin (0.86-0.91 for all models, WA included) is noise in a ratio of tiny numbers.

- Gain is flat at 1.0 up to 45 deg and drops to 0.93 above: a mild decline, same direction in every model (WA 0.96-0.99). Part of any decline at the extreme of a bin variable chosen on the logged value is regression to the mean, so 0.93 is an upper bound on a real saturation.
- Hinge changes nothing in gain (0.94 vs 0.94, same per bin).
- P0 shipped is a real under-turner (0.75-0.79 above 45 deg); P2 fine-tuning recovered most of it.

## 2. Is there a ceiling? ([fig3](turn-gain/fig3_ceiling.png), `tables/quantiles.json`)

| quantity (turn direction) | logged p99 / max | P2 p99 / max | hinge p99 / max | WA p99 / max |
|:--|--:|--:|--:|--:|
| heading change at 4 s (deg) | 79.8 / 103 | 76.0 / 94.7 | 76.2 / 94.7 | 79.1 / 99.4 |
| peak segment curvature (1/m) | 0.155 / 0.252 | 0.129 / 0.22 | 0.129 / 0.21 | 0.147 / 0.305 |
| peak lateral accel v^2 k (m/s^2) | 2.69 / 3.32 | 2.45 / 3.22 | 2.45 / 3.12 | 2.62 / 3.33 |

(seed s0 / s1 pooled by max; the table shows the larger.) The predicted distributions are the logged ones scaled by 0.9-0.95 down the whole tail; P2 itself reaches 3.1-3.2 m/s^2 and 0.21-0.22 1/m, next to the logged and WA maxima. The curves do not flatten at 3 m/s^2 or 0.2 1/m (the openpilot limits are marked in the figure). The logged data themselves top out at 3.3 m/s^2, so a training target cap near 3 would bind only the top 0.1%.

## 3. Speed ([fig5](turn-gain/fig5_speed.png), `tables/speed.csv`), turns > 20 deg

A lateral-acceleration-limited cap would bite at high speed and scale with v^2. It does not: Spearman correlation of the relative under-turn with t0 speed is -0.10 for P2 (-0.12 hinge, +0.07 WA, -0.45 for P0): faster is, if anything, slightly better. P2 gain by speed tercile (median 1.9 / 4.2 / 6.4 m/s): 0.92 / 0.98 / 0.96. By the logged peak lateral acceleration of the turn (< 1.5 / 1.5-2.5 / > 2.5 m/s^2, n 1632 / 1301 / 221): 0.98 / 0.94 / 0.89 (WA 1.00 / 0.98 / 0.94, P0 0.87 / 0.78 / 0.75). The harder the logged turn, the larger the shortfall, in every model; the WA curve is parallel, shifted up by about 0.05.

Residual structure ([resid.csv](turn-gain/tables/resid.csv), heading error at 4 s, turn direction, deg): above 45 deg P2 has bias -4.3, RMS 10.9 (WA -1.5, 9.9); 20-45 deg bias -0.2, RMS 7.6 (WA +0.1, 6.5). Lateral offset RMS at 4 s above 45 deg: P2 1.62 m, WA 1.52 m. Share of tokens with an undershoot of more than 30%: 6.4% (P2) vs 2.2% (WA) above 45 deg. So P2's deficit to WA is mostly dispersion (RMS about 1 deg to 1.1 deg more) plus a 3 deg bias, not a gain collapse.

## 4. Closed loop, HUGSIM 64 ([fig4](turn-gain/fig4_closed_loop.png), `tables/clip.csv`, `ends.csv`, `curve_gain.csv`)

Source: `zs_steps.jsonl` (model requested curvature kappa, speed, pose) and `sim.log` `op_ctrl` lines (act = requested, des = after `clip_curvature`, real, kmean) of the 4 spec arms (P2 and hinge, 2 seeds) x 64 scenarios; the 23 turning scenarios = route yaw range >= 30 deg (17 scenes). The requested / applied trace is logged, no estimate needed. Checked: a CPU replay of `lib/op_ctrl.py` `OpLateral` on the logged kappa and speed reproduces the logged clip steps exactly.

| spec, 23 turning scenarios | P2 s0 | P2 s1 | hinge s0 | hinge s1 |
|:--|--:|--:|--:|--:|
| steps clipped (jerk / lat accel / abs limit) | 3 (0.19%) | 0 | 2 (0.13%) | 1 (0.07%) |
| runs with any clip | 1 | 0 | 1 | 1 |
| max abs requested curvature (1/m; limit 0.2) | 0.065 | 0.051 | 0.061 | 0.090 |
| max requested lateral accel at sim speed (m/s^2; limit 3) | 2.04 | 1.82 | 1.79 | 1.44 |
| max requested / lateral-accel curvature limit | 0.68 | 0.61 | 0.60 | 0.48 |
| clips within the last 8 steps before the run ends | 0 | 0 | 0 | 0 |
| steps over the limit if speed were the model-clock 1.25x | 4 | 0 | 0 | 0 |

All 64: clipped steps 0.08 / 0 / 0.05 / 0.03%, 1 / 0 / 1 / 1 runs. The 6 clipped steps in turning runs fall in 3 bg_collision runs, none within 8 steps of their end. The chain acts as a pass-through: the largest request uses 68% of the lateral-acceleration limit and 45% of the curvature limit. (Caveat, declared: the harness feeds the clip the simulator speed; at the 1.25x model clock the limit would be 1.56x tighter and 4 steps of P2 s0 would exceed it.)

Exam preset (iLQR tracks plan positions, no curvature chain): contrast by replaying the same chain on the logged kappa, 23 turning scenarios: 0.00 / 0.05 / 0.00 / 0.00% of steps (P2 s0, s1, hinge s0, s1), requested curvature up to 0.12-0.16 1/m, lateral accel up to 2.0-2.2 m/s^2. Not a cap there either.

Requested vs route curvature (`curve_gain.csv`; curve steps = |route curvature| > 0.02 1/m, nearest route point within 3 m; 100-110 steps per spec arm in 13-15 routes): slope through the origin of requested kappa on route curvature is 0.09-0.10 (spec, all 4 arms), achieved curvature from pose 0.08; median speed 3.0-3.3 m/s, 13-23% of curve steps below 2 m/s. In the exam preset the slope is noisy (0.28-0.90, 112-256 steps). This is not a gain of the turn mapping: the model gets no route (decisions 92, 121), these are steps near failures, many at 0.5 m/s; it says the closed-loop failures happen with the model asking for little curvature, nowhere near a limit. Treat as descriptive.

## Reading and caveats

- Decision value: turning gain is already 0.94 (P2) vs 0.98 (WA); a training pilot aimed at larger turn amplitudes can buy at most 3-6% of heading change in > 45 deg turns, while the DAC / EPDMS gap to WA on turns (6-9 EPDMS, hinge.md) sits with dispersion and the vision features (decision 147), not with amplitude.
- The control chain is exonerated: in spec the clip fires on 0.0-0.2% of steps and never before a failure; requests stay far below 3 m/s^2 and 0.2 1/m. The exam preset has no chain; the replay shows no clipping either.
- Limits: one logged future per token (a rare strong turn); regression-to-mean bias in binning on the logged value; the closed-loop route-curvature comparison is descriptive; the clip is evaluated at simulator speed as the harness does; WA-JEPA closed-loop traces were not re-read (its sim logs are in another tree and the question is P2 vs hinge).
