# Tracker-lag pre-compensation on NAVSIM / navhard (2026-10-04): rejected

Plan and pre-registration (addendum 1 written before any test-board number): [../../plans/2026-10-04-tracker-precomp-plan.md](../../plans/2026-10-04-tracker-precomp-plan.md).
Gate: navhard delta > 0 with CI lower bound > 0, and navtest delta >= -0.30. Official scorers; the in-process harness matches every official CSV.
This is a per-board adapter trick (user decision 2026-10-04) and is reported as one.

## The tracker (read from the devkit; v1.1 and v2 simulators identical)

One-step LQR every 0.1 s; lateral horizon 1 s, Q = diag(1, 10, 0), R = 1; longitudinal tracks the reference speed 1 s ahead.
Kinematic bicycle with a 0.05 s steering low-pass. **Every token starts at steering angle 0**, which is where the lateral lag comes from.

| navhard, shipped Cinque | 1 s | 2 s | 4 s | mean position error to the plan |
|:--|--:|--:|--:|--:|
| uncompensated | 0.38 | 0.72 | 0.95 | 1.11 m |
| compensated (full) | 0.66 | 0.98 | 0.98 | 0.09 m |

Realised fraction of the planned lateral offset (tokens with |plan y| > 0.5 m).

## Transforms

Reference-free (plan and ego_status only): Levenberg-Marquardt through the devkit's own simulator, submit p + alpha (u* - p), alpha fitted on navtrain only.
`full` (pre-registered) fits time-aligned position and heading; `path` (post hoc) fits the path and keeps the tracker's own speed profile.
navtrain (base 82.12): full alpha 0.25 / 0.5 / 0.75 / 1 = -2.36 / -4.83 / -5.03 / -6.77 (no alpha passes; alpha 1 scored as registered); path = **+0.23 [-0.26, +0.70]** / -1.38 / -3.86 / -6.96 (alpha* 0.25).

## Test boards

| arm | navhard delta [95% CI] | navtest delta [95% CI] | verdict |
|:--|:--|:--|:--|
| shipped full alpha 1 | -5.70 [-8.06, -3.30] | -5.91 [-6.29, -5.52] | reject |
| shipped path alpha 1 (post hoc) | -1.61 [-4.08, +0.75] | -4.31 [-4.72, -3.90] | reject |
| shipped path alpha 0.25 (post hoc) | +0.42 [-1.01, +1.81] | -0.18 [-0.41, +0.05] | ambiguous |
| shipped ideal tracker (diagnostic) | -2.34 [-4.95, +0.23] | - | - |
| N4 full alpha 1 | -10.53 [-13.03, -7.91] | -8.47 [-8.82, -8.12] | reject |
| N4 path alpha 1 | -4.32 [-6.63, -2.07] | -3.75 [-4.08, -3.42] | reject |
| N4 path alpha 0.25 | -0.66 [-1.82, +0.47] | -0.05 [-0.21, +0.10] | reject |

Six arms on navhard, no multiplicity correction.

## Where it wins and loses (shipped full alpha 1, navhard per-token mean delta, points)

| stage | NC | DAC | DDC | EP | TTC | LK | HC | EC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| stage 1 (real) | +0.4 | -2.2 | -1.4 | -2.8 | +0.2 | -7.1 | -64.9 | -30.7 |
| stage 2 (synthetic offset start) | +2.3 | +6.0 | +2.2 | -2.5 | +2.9 | +1.1 | -53.0 | -24.3 |

navtest full alpha 1: DAC -2.41, DDC -1.03, EP -4.79, comfort -18.5.

## Realising the plan, not gaming the metric

Non-comfort changes of the compensated arm track the ideal tracker (plan taken as the rollout): stage 2 DAC +6.0 vs +5.3, NC +2.3 vs +2.1; stage 1 DAC -2.2 vs -4.0; no term exceeds the ideal by more than ~1 point.
Comfort-neutral upper bound (HC / EC restored): shipped full alpha 1 +2.83 [+0.34, +5.43], ideal +1.98, path alpha 0.25 +1.36 [-0.08, +2.75]; N4 -1.5 to +0.1; navtest already loses on DAC / EP before comfort.
So the tracker's lag acts as a useful low-pass on openpilot's near-term lateral plan: realising the plan exactly scores worse on real scenes.

## Difference from the lateral-gain calibration (-3.38, decision 88)

| 2,000 navtest tokens | 1 s | 2 s | 4 s | mean shift of the submitted end point |
|:--|--:|--:|--:|--:|
| uncompensated | 0.38 | 0.75 | 0.97 | - |
| gain | 0.44 | 0.86 | 1.12 | 0.29 m |
| compensation | 0.73 | 0.99 | 0.99 | 0.05 m |

Gain changes intent (overshoots at 4 s, barely touches the 1 s lag) and loses on DAC / NC / TTC / EP; compensation keeps intent, removes the lag and loses on comfort and real-scene DAC / LK.
Same root: the score does not reward realising openpilot's near-term lateral action more strongly or sooner.

## Other boards

navtest v2 EPDMS uses the same tracker (not scored, same direction expected). HUGSIM has its own iLQR with 4 Hz closed-loop replanning and scores the plan; B2D (CARLA PID) and WOD (open-loop plan metrics) have no NAVSIM tracker.

## Not tried

A comfort-constrained compensation; gating compensation by a reference-free "start is off the path" signal (the stage-2 DAC gain is real).

Data: `*.json`, `tracking*.csv` in this folder; raw runs on the box at `$DATA_DIR/runs/skill_pack/trk/`. Code: `experiments/skill_pack/scripts/trk_*.py`, `trk_chain*.sh`.
