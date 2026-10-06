# Joint action head, stage 1 small read: the line stops as registered. The action head learns the plan's gain on NAVSIM, but at 1-3 m/s its command is noise

Written 2026-10-07. Pre-registration: [../plans/2026-10-07-joint-action-prereg.md](../plans/2026-10-07-joint-action-prereg.md), committed (2c3bb724) before any
J training. Code: `scripts/pp_train.py` (`--act-lab plan|log|logwin`, `--ego-lat-drop`), `scripts/pp_joint_probe.py`, `scripts/pp_joint_chain.sh`,
`scripts/pp_joint_report.py`. Tables: [joint_action/stage1.md](joint_action/stage1.md); per-run rows `joint_action/stage1_runs.csv`; gate `joint_action/gate_s0.json`;
offline probes `joint_action/probe_s0.json`, `joint_action/probe_lo_s0.json`.

Setup: HP-F recipe (P2 + hinge 10, pilot scale, W frames, 3 000 steps, seed 0) with the on-policy action pathway trainable, action[0] out of the
distillation and fitted to label x max(1, v0)^2 (speed weight min(1, v0 / 3)), vy / ay zeroed on half the rows. JC = own plan curvature over 0.5-1.5 s
(the `spec_plan_smooth` conversion), JL = logged curvature at t + 0.275 s, JW = logged curvature over 0.5-1.5 s. Base = HP-F-s0 (action distilled to shipped).

## Gate (HUGSIM turn23, `spec` = the action head steers; one run per scenario)

| | HP | JC | JL | JW |
|---|---|---|---|---|
| turn23 HD, `spec` | 0.251 | 0.178 | 0.213 | 0.279 |
| turn23 HD, same checkpoint's plan through `spec_plan_smooth` | 0.328 | 0.314 | 0.322 | 0.270 |
| spec - HP spec (paired, 95% CI) | | -0.073 [-0.161, -0.003] | -0.038 [-0.128, +0.057] | +0.028 [-0.072, +0.129] |
| spec - own smooth | -0.077 | -0.136 [-0.249, -0.043] | -0.110 [-0.206, -0.027] | +0.009 [-0.114, +0.125] |
| completes / spins (turn23, spec) | 1 / 2 | 0 / 0 | 3 / 2 | 5 / 1 |
| heading-rate sign flips / 100 moving steps | 0.00 | 3.89 | 0.29 | 1.23 |
| requested abs kappa / mean step change (1/m) | 0.0071 / 0.0017 | 0.0258 / 0.0143 | 0.0159 / 0.0055 | 0.0212 / 0.0097 |
| verdict | | fail (d1, flips) | fail (d1) | fail (d1 +0.028 < 0.03, flips) |

No arm reaches d1 >= +0.03, so the line stops as registered (no stage 2, no full-scale training). Plan guard holds everywhere: dev ADE 0.59-0.60 m
vs 0.607, drift_off 0.050-0.051 (HP 0.051). spin10 was cut after the 5 scenarios it shares with turn23 (the verdict could not change; box crowded).

## Offline: the head does learn the gain and agrees with the plan above 3 m/s

Dev rows of the training shards, v0 > 3 m/s (n 216): action gain on the logged curvature HP 0.68 -> JC 0.87, JL 0.99, JW 0.88; command on own plan
curvature 0.73 -> 1.03-1.04, correlation 0.92 -> 0.99 (JC / JW); history feedback gain g_hist about 0 for all (no copying of the car's own past curvature);
vy / ay zeroed changes nothing (slope 1.00). In HUGSIM above 4 m/s JC's request equals smooth's (abs kappa 0.0106 vs 0.0109 at 4-8 m/s).

## Why it fails in closed loop: low-speed amplification of the action parametrization

action[0] is a lateral acceleration; the command is action[0] / max(1, v)^2. At 1-3 m/s a small error in action[0] is a large curvature error, and the
registered loss (speed weight min(1, v0 / 3), error measured in action[0] units) barely trains that regime. Measured:

- Offline 1-3 m/s (n 67): abs command curvature JC 0.031, JL 0.028, JW 0.030 vs own plan 0.021; command on plan slope 1.18-1.26; RMS(command - plan) 0.021-0.024
  1/m, as large as the signal. HP is the opposite: 0.009, slope 0.34 (shipped is very conservative at low speed).
- HUGSIM JC `spec` abs kappa by speed: 0-1 m/s 0.041, 1-2 0.040, 2-4 0.029, 4-8 0.011, > 8 0.006; the same checkpoint's plan through smooth: 0.005, 0.015,
  0.014, 0.011, 0.009. The HUGSIM turning routes are driven mostly at 1-4 m/s, so the jitter (3.9 flips / 100 steps, step change 4x smooth) lands where the
  turns are. This is decision 149's single-point failure again (noise amplification), now inside the learned head.

## What it means

The joint training does what it should at speed (gain 0.68 -> ~1, agreement with the plan 0.99), and the closed-loop loss is located: the low-speed
regime of the a_lat / v^2 parametrization, not plan disagreement, copycat feedback or vy / ay leakage. JW is the only arm that is not worse than HP
(+0.028, n.s.) and equals its own smooth plan (+0.009). A follow-up would have to be a new registration: loss in curvature space with the low-speed rows
weighted (or the label taken at max(v, 3)^2), plus a closed-loop jitter penalty or the engine feedback arm; the prereg's feedback arm is not triggered (its
trigger is d1 >= 0.03 failing only on guards).

Caveats: single seed, pilot scale, one HUGSIM run per scenario, 23 turning scenarios from 17 scenes; dev probe n 216 / 67 rows.
